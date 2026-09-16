//! Job detail and result retrieval endpoints.

use std::sync::Arc;

use crate::api::{
    ContentType, DisplayPath, FileProvenance, FileResult, FileStatusKind, JobId, JobInfo,
    JobResultResponse, JobStatus,
};
use axum::Json;
use axum::extract::{Path, State};

use crate::AppState;
use crate::error::ServerError;

/// Return full detail for a single job, including per-file statuses.
///
/// This is the primary polling endpoint for the CLI: the client calls it
/// repeatedly until `status` reaches a terminal value.
#[utoipa::path(
    get,
    path = "/jobs/{job_id}",
    tag = "jobs",
    params(
        ("job_id" = String, Path, description = "Job identifier")
    ),
    responses(
        (status = 200, description = "Job detail", body = JobInfo),
        (status = 404, description = "Job not found", body = crate::openapi::ErrorResponse)
    )
)]
pub(crate) async fn get_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Result<Json<JobInfo>, ServerError> {
    let job_id = JobId::from(job_id);
    state
        .control
        .backend
        .get_job(&job_id)
        .await
        .map(Json)
        .ok_or_else(|| ServerError::JobNotFound(job_id))
}

/// Retrieve all output files for a terminal job.
///
/// Blocks until the job has completed, failed, or been interrupted -- returns
/// 409 for jobs that are still running or queued. The response includes the
/// full staged content of each successful output file. Paths-mode jobs keep
/// the execution-host write, but also preserve a staged copy for download.
/// Cancelled jobs are intentionally excluded because they may have incomplete
/// results.
#[utoipa::path(
    get,
    path = "/jobs/{job_id}/results",
    tag = "jobs",
    params(
        ("job_id" = String, Path, description = "Job identifier")
    ),
    responses(
        (status = 200, description = "Results for terminal jobs", body = JobResultResponse),
        (status = 404, description = "Job not found", body = crate::openapi::ErrorResponse)
    )
)]
pub(crate) async fn get_results(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Result<Json<JobResultResponse>, ServerError> {
    let job_id = JobId::from(job_id);
    let detail = state
        .control
        .backend
        .get_job_detail(&job_id)
        .await
        .ok_or_else(|| ServerError::JobNotFound(job_id.clone()))?;

    // Intentionally excludes Cancelled: cancelled jobs may have incomplete results.
    if !matches!(
        detail.status,
        JobStatus::Completed | JobStatus::Failed | JobStatus::Interrupted
    ) {
        return Err(ServerError::JobNotTerminal(JobId::from(format!(
            "Job {job_id} is still {}",
            detail.status
        ))));
    }

    let mut files = Vec::new();
    for r in &detail.results {
        let content = if r.error.is_none() {
            let path = result_content_path(&detail, &r.filename);
            read_result_content(&path).await?
        } else {
            String::new()
        };
        let provenance = match &r.error {
            None => provenance_of(r.content_type, &content),
            Some(_) => FileProvenance::NotRead,
        };
        files.push(FileResult {
            filename: r.filename.clone(),
            content,
            content_type: r.content_type,
            error: r.error.clone(),
            provenance,
        });
    }

    Ok(Json(JobResultResponse {
        job_id,
        status: detail.status,
        files,
    }))
}

/// Retrieve the output for a single file within a job.
///
/// Waits for the individual file to reach a terminal status, not the whole job,
/// so clients can fetch results incrementally as files finish. Supports
/// command-policy matching (e.g., requesting `sample.wav` finds `sample.cha`,
/// and speaker identification maps `sample.cha` to its evidence JSON).
#[utoipa::path(
    get,
    path = "/jobs/{job_id}/results/{*filename}",
    tag = "jobs",
    params(
        ("job_id" = String, Path, description = "Job identifier"),
        ("filename" = String, Path, description = "Requested output filename")
    ),
    responses(
        (status = 200, description = "One file result", body = FileResult),
        (status = 404, description = "File not found", body = crate::openapi::ErrorResponse)
    )
)]
pub(crate) async fn get_single_result(
    State(state): State<Arc<AppState>>,
    Path((job_id, filename)): Path<(String, String)>,
) -> Result<Json<FileResult>, ServerError> {
    let job_id = JobId::from(job_id);
    let detail = state
        .control
        .backend
        .get_job_detail(&job_id)
        .await
        .ok_or_else(|| ServerError::JobNotFound(job_id.clone()))?;

    // Find file status
    let fs = detail
        .file_statuses
        .iter()
        .find(|fs| *fs.filename == *filename)
        .ok_or_else(|| {
            ServerError::FileNotFound(format!("File {filename} not found in job {job_id}"))
        })?;

    if !fs.status.is_terminal() {
        return Err(ServerError::FileNotReady(format!(
            "File {filename} is still {}",
            fs.status
        )));
    }

    if fs.status == FileStatusKind::Error {
        return Ok(Json(FileResult {
            filename: DisplayPath::from(filename.as_str()),
            content: String::new(),
            content_type: ContentType::Chat,
            error: fs.error.clone().or(Some("Unknown error".into())),
            provenance: FileProvenance::NotRead,
        }));
    }

    let result_entry = primary_result_entry(&detail, &filename).ok_or_else(|| {
        ServerError::FileNotFound(format!("Result for {filename} not found in job {job_id}"))
    })?;

    let content_type = result_entry.content_type;
    let out_filename = result_entry.filename.clone();

    let content = read_result_content(&result_content_path(&detail, &out_filename)).await?;

    let provenance = provenance_of(content_type, &content);
    Ok(Json(FileResult {
        filename: out_filename,
        content,
        content_type,
        error: None,
        provenance,
    }))
}

/// Use the producer's output policy, including suffixes and directory identity.
/// Stem-only matching both missed evidence JSON and confused nested inputs.
fn primary_result_entry<'a>(
    detail: &'a crate::store::JobDetail,
    input: &str,
) -> Option<&'a crate::store::FileResultEntry> {
    let expected =
        crate::recipe_runner::runtime::result_display_path_for_command(detail.command, input);
    detail
        .results
        .iter()
        .find(|r| r.error.is_none() && r.filename == expected)
}

/// Read one successful result file's provenance stamps as a typed state.
///
/// A stamp of ours that does not parse is that file's state, never a failure
/// of the whole request and never a silently shorter history, so the rest of
/// the job's results are still served. Matched on the content type with no
/// catch-all, so a new content type has to state whether it carries stamps.
fn provenance_of(content_type: ContentType, content: &str) -> FileProvenance {
    match content_type {
        ContentType::Chat => match crate::provenance::extract_provenance(content) {
            Ok(entries) => FileProvenance::Parsed { entries },
            Err(unparseable) => FileProvenance::Unparseable {
                reason: unparseable.to_string(),
            },
        },
        ContentType::Csv | ContentType::Text | ContentType::Json => FileProvenance::NotRead,
    }
}

async fn read_result_content(path: &std::path::Path) -> Result<String, ServerError> {
    tokio::fs::read_to_string(path).await.map_err(|error| {
        ServerError::Io(std::io::Error::new(
            error.kind(),
            format!("failed to read result file {}: {error}", path.display()),
        ))
    })
}

fn result_content_path(
    detail: &crate::store::JobDetail,
    out_filename: &DisplayPath,
) -> std::path::PathBuf {
    detail
        .staging_dir
        .join("output")
        .join(out_filename.as_ref())
        .as_path()
        .to_owned()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::api::{ContentType, DisplayPath, FileStatusEntry, FileStatusKind, JobStatus};
    use crate::store::{FileResultEntry, JobDetail};

    use super::result_content_path;

    fn job_detail(paths_mode: bool) -> JobDetail {
        JobDetail {
            command: crate::ReleasedCommand::Align,
            status: JobStatus::Completed,
            paths_mode,
            staging_dir: batchalign_types::paths::ServerPath::new("/tmp/jobs/job-1"),
            results: vec![FileResultEntry {
                filename: DisplayPath::from("nested/sample.cha"),
                content_type: ContentType::Chat,
                error: None,
            }],
            file_statuses: vec![FileStatusEntry {
                filename: DisplayPath::from("nested/sample.cha"),
                status: FileStatusKind::Done,
                error: None,
                error_category: None,
                error_codes: None,
                error_line: None,
                bug_report_id: None,
                stamp: crate::api::FileStampOutcome::Unrecorded,
                started_at: None,
                finished_at: None,
                next_eligible_at: None,
                progress_current: None,
                progress_total: None,
                progress_stage: None,
                progress_label: None,
            }],
        }
    }

    /// An unparseable stamp is one file's state, not a failed request: the
    /// file still reports, and a well-formed file beside it parses.
    #[test]
    fn an_unparseable_stamp_is_that_files_provenance_state() {
        use crate::api::FileProvenance;

        let stamped = "@Comment:\t[fc-ba3 align | fa=wave2vec-fa-v1 ; lang=eng | 2026-09-15T19:15:00-04:00]\n";
        assert!(matches!(
            super::provenance_of(ContentType::Chat, stamped),
            FileProvenance::Parsed { entries } if entries.len() == 1
        ));

        let broken = "@Comment:\t[fc-ba3 align]\n";
        assert!(matches!(
            super::provenance_of(ContentType::Chat, broken),
            FileProvenance::Unparseable { reason } if reason.contains("[fc-ba3 align]")
        ));

        assert_eq!(
            super::provenance_of(ContentType::Json, broken),
            FileProvenance::NotRead
        );
    }

    #[test]
    fn result_content_path_uses_staging_output_for_paths_mode() {
        let detail = job_detail(true);
        assert_eq!(
            result_content_path(&detail, &DisplayPath::from("nested/sample.cha")),
            PathBuf::from("/tmp/jobs/job-1/output/nested/sample.cha")
        );
    }

    #[test]
    fn runtime_regression_speaker_result_preserves_directory_identity() {
        let mut detail = job_detail(true);
        detail.command = crate::ReleasedCommand::SpeakerIdentify;
        detail.results[0].filename = "nested/sample_speaker_identity.json".into();
        detail.results[0].content_type = ContentType::Json;
        let result = super::primary_result_entry(&detail, "nested/sample.cha").unwrap();
        assert_eq!(
            result.filename.as_ref(),
            "nested/sample_speaker_identity.json"
        );
        assert!(super::primary_result_entry(&detail, "other/sample.cha").is_none());
        detail.results[0].error = Some("failed".into());
        assert!(super::primary_result_entry(&detail, "nested/sample.cha").is_none());
    }

    #[test]
    fn runtime_regression_result_preserves_chat_and_media_input_mapping() {
        let mut detail = job_detail(false);
        assert!(super::primary_result_entry(&detail, "nested/sample.cha").is_some());
        detail.command = crate::ReleasedCommand::Transcribe;
        assert!(super::primary_result_entry(&detail, "nested/sample.wav").is_some());
        assert!(super::primary_result_entry(&detail, "other/sample.wav").is_none());
    }
}
