//! Crash recovery: reload persisted jobs from SQLite on startup.

use std::collections::HashMap;

use crate::api::{DisplayPath, FileStatusKind, JobId, JobStatus, NumSpeakers, ReleasedCommand};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use super::super::job::{
    Job, JobDispatchConfig, JobExecutionState, JobFilesystemConfig, JobIdentity, JobRuntimeControl,
    JobScheduleState, JobSourceContext, RecoveryDisposition, StoredSubmitter, Submitter,
};
use super::super::{
    FileFailure, FilePhase, FilePhaseColumns, FileProgress, FileResultEntry, FileStatus,
    JobStatusColumns, JobStore,
};
use crate::error::ServerError;

/// Rows a load reinterpreted, to rewrite once from what was loaded.
#[derive(Debug, Default)]
struct RowRepairs {
    /// File rows, with the phase each was read as.
    files: Vec<(JobId, String, FilePhase)>,
    /// Job rows whose submitter columns held a name with no address.
    submitters: Vec<JobId>,
}

/// Persisted startup-recovery update for one job: the job's status columns
/// and the phases of the files recovery requeued, each written whole.
#[derive(Debug, Clone)]
struct RecoveredJobPersistence {
    /// Job whose persisted status needs canonicalization.
    job_id: JobId,
    /// The job's status after recovery reconciliation, as it is.
    status: JobStatusColumns,
    /// Files recovery requeued, with the phase each now has.
    requeued_files: Vec<(String, FilePhase)>,
}

fn append_recovery_note(existing: Option<String>, note: impl Into<String>) -> Option<String> {
    let note = format!("[recovery] {}", note.into());
    match existing {
        Some(existing) if existing.trim().is_empty() => Some(note),
        Some(existing) => Some(format!("{existing}\n{note}")),
        None => Some(note),
    }
}

/// Decode a stored `diagnostics` column (JSON written by
/// `FileOutputDiagnostics::to_column_json`). Text this build cannot read is
/// not guessed at: it is dropped with a `[recovery]` note, which makes the
/// row foreign, so it is reported and never rewritten.
fn recover_output_diagnostics(
    job_id: &str,
    filename: &str,
    raw: Option<&str>,
) -> (Option<crate::api::FileOutputDiagnostics>, Option<String>) {
    let Some(raw) = raw else {
        return (None, None);
    };
    match serde_json::from_str(raw) {
        Ok(diagnostics) => (Some(diagnostics), None),
        Err(error) => {
            warn!(
                job_id,
                filename,
                %error,
                "Unreadable persisted output diagnostics during crash recovery",
            );
            (
                None,
                Some(format!(
                    "unreadable output diagnostics were dropped: {error}"
                )),
            )
        }
    }
}

fn recover_job_status(job_id: &str, raw_status: &str) -> (JobStatus, Option<String>) {
    match raw_status.parse() {
        Ok(status) => (status, None),
        Err(error) => {
            warn!(
                job_id,
                raw_status,
                %error,
                "Invalid persisted job status during crash recovery",
            );
            (
                JobStatus::Failed,
                Some(format!(
                    "invalid persisted job status '{raw_status}' was coerced to 'failed'"
                )),
            )
        }
    }
}

fn recover_file_status(
    job_id: &str,
    filename: &str,
    raw_status: &str,
) -> (FileStatusKind, Option<String>) {
    match raw_status.parse() {
        Ok(status) => (status, None),
        Err(error) => {
            warn!(
                job_id,
                filename,
                raw_status,
                %error,
                "Invalid persisted file status during crash recovery",
            );
            (
                FileStatusKind::Error,
                Some(format!(
                    "invalid persisted file status '{raw_status}' was coerced to 'error'"
                )),
            )
        }
    }
}

fn recover_failure_category(
    job_id: &str,
    filename: &str,
    raw_category: Option<&str>,
) -> (Option<crate::scheduling::FailureCategory>, Option<String>) {
    let Some(raw_category) = raw_category else {
        return (None, None);
    };

    match raw_category.parse() {
        Ok(category) => (Some(category), None),
        Err(error) => {
            warn!(
                job_id,
                filename,
                raw_category,
                %error,
                "Invalid persisted failure category during crash recovery",
            );
            (
                None,
                Some(format!(
                    "invalid persisted error_category '{raw_category}' was ignored"
                )),
            )
        }
    }
}

/// A stored file row read as the one phase it describes, and what reading it
/// found: whether the row is the phase's own image, held columns the phase
/// does not own, or holds a value this build cannot read.
pub(crate) enum RecoveredFilePhase {
    /// The row is exactly its phase's column image.
    Exact(FilePhase),
    /// The row held columns its phase does not own (`dropped` says which):
    /// it is rewritten once from the phase, so the next read finds nothing to
    /// report.
    Repairable {
        /// The phase the row describes.
        phase: FilePhase,
        /// The columns the phase does not own, for the report.
        dropped: String,
    },
    /// The row holds a status or failure category this build cannot read
    /// (another build's vocabulary, as after a rollback). It is read as an
    /// error with a `[recovery]` note, and never rewritten: a rewrite would
    /// destroy what the other build wrote.
    Foreign(FilePhase),
}

impl RecoveredFilePhase {
    /// The phase the row is read as, by value.
    pub(crate) fn into_phase(self) -> FilePhase {
        match self {
            Self::Exact(phase) | Self::Repairable { phase, .. } | Self::Foreign(phase) => phase,
        }
    }
}

/// A stored `file_statuses` row as the one phase it describes: the database
/// boundary for file rows, shared by startup interruption and job loading.
///
/// An unknown status is coerced to `error` and an unknown failure category
/// dropped, each with a `[recovery]` note appended to the row's error text
/// ([`RecoveredFilePhase::Foreign`]); columns the phase does not own are
/// found by [`FilePhase::from_row`] ([`RecoveredFilePhase::Repairable`]). The
/// caller reports and acts.
pub(crate) fn recover_file_phase(
    job_id: &str,
    row: &crate::db::FileStatusRow,
) -> RecoveredFilePhase {
    let (status, status_note) = recover_file_status(job_id, &row.filename, &row.status);
    let (error_category, category_note) =
        recover_failure_category(job_id, &row.filename, row.error_category.as_deref());
    let (diagnostics, diagnostics_note) =
        recover_output_diagnostics(job_id, &row.filename, row.diagnostics.as_deref());
    let mut error = row.error.clone();
    let mut foreign = false;
    for note in [status_note, category_note, diagnostics_note]
        .into_iter()
        .flatten()
    {
        error = append_recovery_note(error, note);
        foreign = true;
    }
    let (phase, dropped) = FilePhase::from_row(FilePhaseColumns {
        status,
        error: error.as_deref(),
        error_category,
        diagnostics: diagnostics.as_ref(),
        started_at: row.started_at,
        finished_at: row.finished_at,
        next_eligible_at: row.next_eligible_at,
    });
    match (foreign, dropped) {
        (true, _) => RecoveredFilePhase::Foreign(phase),
        (false, Some(dropped)) => RecoveredFilePhase::Repairable { phase, dropped },
        (false, None) => RecoveredFilePhase::Exact(phase),
    }
}

impl JobStore {
    /// Load jobs from DB into memory (crash recovery).
    pub async fn load_from_db(&self) -> Result<usize, ServerError> {
        let db = match &self.db {
            Some(db) => db,
            None => return Ok(0),
        };

        let rows = db.load_all_jobs().await?;
        let ttl_cutoff = self.config.job_ttl_days.cutoff(self.now());
        let (loaded, recovered_updates, repairs) = self
            .registry
            .mutate_all(move |jobs| {
                let mut loaded = 0;
                let mut recovered_updates = Vec::new();
                let mut repairs = RowRepairs::default();

                'rows: for row in rows {
                    if row.submitted_at < ttl_cutoff {
                        continue;
                    }

                    let (status, job_status_note) = recover_job_status(&row.job_id, &row.status);
                    let command = match ReleasedCommand::try_from(row.command.as_str()) {
                        Ok(command) => command,
                        Err(_) => {
                            warn!(job_id = %row.job_id, command = %row.command,
                                "Unknown command in DB, skipping job recovery");
                            continue;
                        }
                    };

                    // Validate the row's command/options pairing before
                    // collecting any persistence repairs, including for jobs
                    // whose files have not yet reached a terminal state.
                    if let Err(error) = crate::command_model::command_spec(command).selected_output_policy(&row.options) {
                        warn!(job_id = %row.job_id, %error, "Inconsistent command options in DB, skipping job recovery");
                        continue;
                    }

                    let mut file_statuses = HashMap::new();
                    let mut results: Vec<FileResultEntry> = Vec::new();
                    for fs_row in &row.file_statuses {
                        // The database boundary: the row's columns become the
                        // one phase they describe.
                        let phase = match recover_file_phase(&row.job_id, fs_row) {
                            RecoveredFilePhase::Exact(phase) => phase,
                            RecoveredFilePhase::Repairable { phase, dropped } => {
                                warn!(
                                    job_id = %row.job_id,
                                    filename = %fs_row.filename,
                                    "Recovered file status: {dropped}; rewriting the row \
                                     from its phase"
                                );
                                repairs.files.push((
                                    JobId::from(row.job_id.clone()),
                                    fs_row.filename.clone(),
                                    phase.clone(),
                                ));
                                phase
                            }
                            RecoveredFilePhase::Foreign(phase) => {
                                warn!(
                                    job_id = %row.job_id,
                                    filename = %fs_row.filename,
                                    "Recovered file row holds a value this build cannot \
                                     read; read as {}, the row is left as stored",
                                    phase.kind()
                                );
                                phase
                            }
                        };
                        let fs_status = phase.kind();
                        let file_error = phase
                            .failure()
                            .and_then(FileFailure::message)
                            .map(str::to_owned);
                        file_statuses.insert(
                            fs_row.filename.clone(),
                            FileStatus {
                                filename: DisplayPath::from(fs_row.filename.clone()),
                                phase,
                                // Not persisted: a restored status never held
                                // a stamp decision, and saying so beats
                                // inventing one.
                                stamp: crate::api::FileStampOutcome::Unrecorded,
                                current_attempt_id: None,
                                progress: FileProgress::default(),
                                worker_waits: crate::store::OpenWorkerWaits::default(),
                            },
                        );

                        if fs_status.is_terminal() {
                            // Persisted file-status names identify inputs, not artifacts.
                            let artifact = match crate::recipe_runner::runtime::primary_output_artifact(
                                command,
                                &row.options,
                                &DisplayPath::from(fs_row.filename.clone()),
                            ) {
                                Ok(artifact) => artifact,
                                Err(error) => {
                                    warn!(job_id = %row.job_id, %error, "Inconsistent command options in DB, skipping job recovery");
                                    continue 'rows;
                                }
                            };
                            results.push(FileResultEntry {
                                // A file that wrote output (clean or
                                // diagnosed) names its artifact; an error
                                // names its input.
                                filename: if fs_status.wrote_output() {
                                    artifact.display_path
                                } else {
                                    DisplayPath::from(fs_row.filename.clone())
                                },
                                content_type: artifact.content_type,
                                error: file_error,
                            });
                        }
                    }

                    let completed_files = file_statuses
                        .values()
                        .filter(|file_status| file_status.status().is_terminal())
                        .count() as i64;

                    let job_id_newtype = JobId::from(row.job_id.clone());
                    let mut job_error = row.error.clone();
                    if let Some(note) = job_status_note {
                        job_error = append_recovery_note(job_error, note);
                    }
                    let mut job = Job {
                        identity: JobIdentity {
                            job_id: job_id_newtype.clone(),
                            correlation_id: if row.correlation_id.is_empty() {
                                row.job_id.clone().into()
                            } else {
                                row.correlation_id.into()
                            },
                        },
                        dispatch: JobDispatchConfig {
                            command,
                            lang: {
                                let (spec, valid) =
                                    crate::api::LanguageSpec::parse_from_db(&row.lang);
                                if !valid {
                                    tracing::warn!(
                                        job_id = %row.job_id,
                                        raw_lang = %row.lang,
                                        "Invalid language code in DB, falling back to eng"
                                    );
                                }
                                spec
                            },
                            num_speakers: NumSpeakers(row.num_speakers),
                            options: row.options,
                            runtime_state: std::collections::BTreeMap::new(),
                            debug_traces: false,
                        },
                        source: JobSourceContext {
                            submitter: match Submitter::from_columns(
                                row.submitted_by,
                                row.submitted_by_name,
                            ) {
                                StoredSubmitter::Recorded(submitter) => Some(submitter),
                                StoredSubmitter::Absent => None,
                                StoredSubmitter::NameWithoutAddress { name } => {
                                    warn!(
                                        job_id = %row.job_id,
                                        submitter_name = %name,
                                        "Recovered job row names a submitter with no \
                                         address; recorded as no submitter, and the row \
                                         is rewritten so"
                                    );
                                    repairs.submitters.push(JobId::from(row.job_id.clone()));
                                    None
                                }
                            },
                            source_dir: row.source_dir.into(),
                        },
                        filesystem: JobFilesystemConfig {
                            filenames: row.filenames.into_iter().map(DisplayPath::from).collect(),
                            has_chat: row.has_chat,
                            staging_dir: batchalign_types::paths::ServerPath::from(row.staging_dir),
                            paths_mode: row.paths_mode,
                            source_paths: row
                                .source_paths
                                .into_iter()
                                .map(batchalign_types::paths::ClientPath::from)
                                .collect(),
                            output_paths: row
                                .output_paths
                                .into_iter()
                                .map(batchalign_types::paths::ClientPath::from)
                                .collect(),
                            before_paths: Vec::new(),
                            media_mapping: batchalign_types::paths::MediaMappingKey::from(
                                row.media_mapping,
                            ),
                            media_subdir: batchalign_types::paths::RepoRelativePath::from(
                                row.media_subdir,
                            ),
                            // source_dir is owned by JobSourceContext; the runner snapshot
                            // assembles RunnerFilesystemConfig.source_dir from there.
                            source_dir: Default::default(),
                        },
                        execution: JobExecutionState {
                            status,
                            file_statuses,
                            results,
                            error: job_error,
                            completed_files,
                        },
                        schedule: JobScheduleState {
                            submitted_at: row.submitted_at,
                            completed_at: row.completed_at,
                            next_eligible_at: row.next_eligible_at,
                            num_workers: row.num_workers.map(|n| n as i64),
                            lease: row.lease,
                            last_cancel: row.last_cancelled_at.map(|at| {
                                crate::store::JobLastCancelInfo {
                                    at,
                                    source: row
                                        .last_cancelled_source
                                        .clone()
                                        .unwrap_or_else(|| "api".to_string()),
                                    host: row.last_cancelled_host.clone(),
                                    reason: row.last_cancelled_reason.clone(),
                                }
                            }),
                        },
                        runtime: JobRuntimeControl {
                            cancel_token: CancellationToken::new(),
                            runner_active: false,
                            run_generation: crate::store::RunGeneration::FIRST,
                        },
                        execution_plan: None,
                    };

                    if status.is_recoverable() {
                        let resumable: Vec<String> = job
                            .execution
                            .file_statuses
                            .iter()
                            .filter(|(_, file_status)| file_status.status().is_resumable())
                            .map(|(filename, _)| filename.clone())
                            .collect();
                        let disposition = job.reconcile_recovered_runtime_state();
                        let requeued_files = match disposition {
                            RecoveryDisposition::Requeued => resumable
                                .into_iter()
                                .filter_map(|filename| {
                                    let phase =
                                        job.execution.file_statuses.get(&filename)?.phase.clone();
                                    Some((filename, phase))
                                })
                                .collect(),
                            RecoveryDisposition::Failed | RecoveryDisposition::Completed => {
                                Vec::new()
                            }
                        };
                        recovered_updates.push(RecoveredJobPersistence {
                            job_id: job_id_newtype.clone(),
                            status: job.status_columns(),
                            requeued_files,
                        });
                    }

                    jobs.insert(job_id_newtype, job);
                    loaded += 1;
                }

                (loaded, recovered_updates, repairs)
            })
            .await;

        // Rows the reader had to reinterpret are rewritten once from what was
        // loaded, so the same report is not made again at every startup.
        // Before the recovery writes below, which may move a repaired file on.
        // A repair that cannot be written is reported, and startup goes on:
        // the row reads the same at the next startup and is repaired then.
        for (job_id, filename, phase) in &repairs.files {
            if let Err(error) = db.update_file_status(job_id, filename, phase, None).await {
                warn!(%job_id, filename, %error, "Could not rewrite a recovered file row");
            }
        }
        for job_id in &repairs.submitters {
            if let Err(error) = db.write_job_submitter(job_id, None).await {
                warn!(%job_id, %error, "Could not rewrite a recovered submitter");
            }
        }

        for update in recovered_updates {
            let job_id: &str = update.job_id.as_ref();
            db.write_job_status(job_id, &update.status).await?;
            db.update_job_lease(job_id, None).await?;

            for (filename, phase) in &update.requeued_files {
                db.update_file_status(job_id, filename, phase, None).await?;
            }
        }

        info!(loaded = loaded, "Loaded jobs from DB");
        Ok(loaded)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::sync::broadcast;

    use super::*;
    use crate::api::ContentType;
    use crate::api::MachineTime;
    use crate::config::ServerConfig;
    use crate::db::{JobDB, NewJobRecord};
    use crate::options::{CommandOptions, CommonOptions, MorphotagOptions};
    use crate::store::JobStore;
    use crate::ws::BROADCAST_CAPACITY;

    /// Build a test insert payload for startup-recovery coverage.
    fn make_job_record(job_id: &str, status: JobStatus, filenames: Vec<String>) -> NewJobRecord {
        let has_chat = filenames.iter().map(|_| true).collect();

        NewJobRecord {
            job_id: job_id.to_string(),
            correlation_id: job_id.to_string(),
            command: "morphotag".to_string(),
            lang: "eng".to_string(),
            num_speakers: 1,
            status,
            staging_dir: "/tmp/staging".to_string(),
            filenames,
            has_chat,
            options: CommandOptions::Morphotag(MorphotagOptions {
                common: CommonOptions::default(),

                ..Default::default()
            }),
            media_mapping: String::new(),
            media_subdir: String::new(),
            source_dir: "/corpus".to_string(),
            submitter: Some(crate::store::Submitter::client(
                std::net::Ipv4Addr::LOCALHOST.into(),
                "localhost".into(),
            )),
            submitted_at: MachineTime::now(),
            paths_mode: false,
            source_paths: Vec::new(),
            output_paths: Vec::new(),
        }
    }

    /// Open an isolated SQLite DB and `JobStore` pair for recovery tests.
    async fn test_store_with_db() -> (JobStore, Arc<JobDB>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(JobDB::open(Some(dir.path())).await.unwrap());
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::new(
            ServerConfig::default(),
            Some(db.clone()),
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );
        (store, db, dir)
    }

    /// Restore evidence through the actual SQLite-to-store boundary, without directories.
    #[tokio::test]
    async fn restart_recovery_preserves_speaker_result_policy() {
        for paths_mode in [false, true] {
            let db = Arc::new(JobDB::in_memory_for_test().await.unwrap());
            let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
            let store = JobStore::new(
                ServerConfig::default(),
                Some(db.clone()),
                tx,
                std::sync::Arc::new(crate::clock::SystemClock),
            );
            let mut record = make_job_record(
                "evidence-job",
                JobStatus::Completed,
                vec!["nested/sample.cha".into(), "nested/failed.cha".into()],
            );
            record.command = "speaker_identify".into();
            record.paths_mode = paths_mode;
            record.options = serde_json::from_value(serde_json::json!({
                "command": "speaker_identify", "enrollments": ["100-500:REF"],
                "threshold": 0.5, "tiers": []
            }))
            .unwrap();
            db.insert_job(&record).await.unwrap();
            db.seed_file_status_row(
                "evidence-job",
                "nested/sample.cha",
                "done",
                None,
                None,
                Some("json"),
                None,
                None,
                None,
            )
            .await
            .unwrap();
            db.seed_file_status_row(
                "evidence-job",
                "nested/failed.cha",
                "error",
                Some("no evidence"),
                None,
                Some("json"),
                None,
                None,
                None,
            )
            .await
            .unwrap();
            store.load_from_db().await.unwrap();
            let detail = store
                .get_job_detail(&JobId::from("evidence-job"))
                .await
                .unwrap();
            let result = detail.results.iter().find(|r| r.error.is_none()).unwrap();
            assert_eq!(
                result.filename.as_ref(),
                "nested/sample_speaker_identity.json"
            );
            assert_eq!(result.content_type, ContentType::Json);
            assert!(
                detail
                    .file_statuses
                    .iter()
                    .any(|s| s.filename == "nested/sample.cha")
            );
            let failed = detail.results.iter().find(|r| r.error.is_some()).unwrap();
            assert_eq!(failed.filename.as_ref(), "nested/failed.cha");
            assert_eq!(failed.error.as_deref(), Some("no evidence"));
        }
    }

    #[tokio::test]
    async fn restart_recovery_preserves_explicit_native_encoding_in_both_io_modes() {
        for paths_mode in [false, true] {
            for (format, content_type) in [("wav", ContentType::Wav), ("mp3", ContentType::Mp3)] {
                let (store, db, _dir) = test_store_with_db().await;
                let mut record = make_job_record(
                    "native-job",
                    JobStatus::Completed,
                    vec!["nested/source.wav".into()],
                );
                record.command = "convert".into();
                record.paths_mode = paths_mode;
                record.options = serde_json::from_value(
                    serde_json::json!({"command":"convert", "format":format}),
                )
                .unwrap();
                db.insert_job(&record).await.unwrap();
                db.seed_file_status_row(
                    "native-job",
                    "nested/source.wav",
                    "done",
                    None,
                    None,
                    Some(format),
                    None,
                    None,
                    None,
                )
                .await
                .unwrap();
                assert_eq!(store.load_from_db().await.unwrap(), 1);
                let detail = store
                    .get_job_detail(&JobId::from("native-job"))
                    .await
                    .unwrap();
                assert_eq!(
                    detail.results[0].filename.as_ref(),
                    format!("nested/source.converted.{format}")
                );
                assert_eq!(detail.results[0].content_type, content_type);
                assert_eq!(detail.options, record.options);
            }
        }
    }

    #[tokio::test]
    async fn recovery_refuses_inconsistent_options_before_repairing_pending_rows() {
        let (store, db, _dir) = test_store_with_db().await;
        let mut record =
            make_job_record("mismatched", JobStatus::Queued, vec!["source.wav".into()]);
        record.command = "convert".into(); // options still name the original command
        db.insert_job(&record).await.unwrap();
        assert_eq!(store.load_from_db().await.unwrap(), 0);
        assert!(store.get(&JobId::from("mismatched")).await.is_none());
        let retained = db.load_all_jobs().await.unwrap();
        assert_eq!(retained.len(), 1);
        assert_eq!(retained[0].options, record.options);
        assert_eq!(retained[0].command, "convert");
    }

    /// Startup recovery re-queues resumable work and persists the queued state.
    #[tokio::test]
    async fn load_from_db_requeues_resumable_interrupted_jobs() {
        let (store, db, _dir) = test_store_with_db().await;
        db.insert_job(&make_job_record(
            "job-1",
            JobStatus::Running,
            vec!["a.cha".into(), "b.cha".into()],
        ))
        .await
        .unwrap();
        db.seed_file_status_row(
            "job-1",
            "a.cha",
            "done",
            None,
            None,
            Some("chat"),
            Some(crate::unix_time(10.0)),
            Some(crate::unix_time(20.0)),
            None,
        )
        .await
        .unwrap();
        db.seed_file_status_row(
            "job-1",
            "b.cha",
            "processing",
            None,
            None,
            None,
            Some(crate::unix_time(15.0)),
            None,
            None,
        )
        .await
        .unwrap();
        db.recover_interrupted(MachineTime::now()).await.unwrap();

        store.load_from_db().await.unwrap();

        let info = store.get(&JobId::from("job-1")).await.unwrap();
        assert_eq!(info.status, JobStatus::Queued);
        let requeued_file = info
            .file_statuses
            .iter()
            .find(|file| file.filename == "b.cha")
            .unwrap();
        assert_eq!(requeued_file.status, FileStatusKind::Queued);
        assert!(requeued_file.started_at.is_none());

        let rows = db.load_all_jobs().await.unwrap();
        assert_eq!(rows[0].status, "queued");
        let persisted_file = rows[0]
            .file_statuses
            .iter()
            .find(|file| file.filename == "b.cha")
            .unwrap();
        assert_eq!(persisted_file.status, "queued");
        assert!(persisted_file.started_at.is_none());
    }

    /// Startup recovery finalizes all-terminal interrupted jobs and clears leases.
    #[tokio::test]
    async fn load_from_db_finalizes_terminal_interrupted_jobs() {
        let (store, db, _dir) = test_store_with_db().await;
        db.insert_job(&make_job_record(
            "job-2",
            JobStatus::Running,
            vec!["a.cha".into(), "b.cha".into()],
        ))
        .await
        .unwrap();
        db.seed_file_status_row(
            "job-2",
            "a.cha",
            "done",
            None,
            None,
            Some("chat"),
            Some(crate::unix_time(10.0)),
            Some(crate::unix_time(20.0)),
            None,
        )
        .await
        .unwrap();
        db.seed_file_status_row(
            "job-2",
            "b.cha",
            "error",
            Some("boom"),
            Some("worker_crash"),
            None,
            Some(crate::unix_time(11.0)),
            Some(crate::unix_time(21.0)),
            None,
        )
        .await
        .unwrap();
        db.update_job_lease(
            "job-2",
            Some(
                &crate::scheduling::LeaseRecord::new(
                    "node-a".into(),
                    crate::unix_time(35.0),
                    crate::unix_time(40.0),
                )
                .expect("an ordered fixture lease"),
            ),
        )
        .await
        .unwrap();
        db.recover_interrupted(MachineTime::now()).await.unwrap();

        store.load_from_db().await.unwrap();

        let info = store.get(&JobId::from("job-2")).await.unwrap();
        assert_eq!(info.status, JobStatus::Completed);
        assert_eq!(info.completed_files, 2);
        assert!(info.active_lease.is_none());

        let rows = db.load_all_jobs().await.unwrap();
        assert_eq!(rows[0].status, "completed");
        assert!(rows[0].lease.is_none());
    }

    /// Recovery preserves invalid persisted job status evidence instead of silently dropping it.
    #[tokio::test]
    async fn load_from_db_preserves_invalid_job_status_evidence() {
        let (store, db, _dir) = test_store_with_db().await;
        db.insert_job(&make_job_record(
            "job-bad-status",
            JobStatus::Queued,
            vec!["a.cha".into()],
        ))
        .await
        .unwrap();
        db.seed_job_status_column("job-bad-status", "mystery_status")
            .await
            .unwrap();

        store.load_from_db().await.unwrap();

        let info = store.get(&JobId::from("job-bad-status")).await.unwrap();
        assert_eq!(info.status, JobStatus::Failed);
        let error = info
            .error
            .expect("recovery should preserve invalid status evidence");
        assert!(error.contains("invalid persisted job status 'mystery_status'"));
    }

    /// Recovery preserves invalid per-file persistence evidence instead of silently normalizing it.
    #[tokio::test]
    async fn load_from_db_preserves_invalid_file_persistence_evidence() {
        let (store, db, _dir) = test_store_with_db().await;
        db.insert_job(&make_job_record(
            "job-bad-file-status",
            JobStatus::Queued,
            vec!["bad.cha".into()],
        ))
        .await
        .unwrap();
        db.seed_file_status_row(
            "job-bad-file-status",
            "bad.cha",
            "mystery_file_status",
            Some("original failure"),
            Some("mystery_category"),
            None,
            Some(crate::unix_time(10.0)),
            Some(crate::unix_time(11.0)),
            None,
        )
        .await
        .unwrap();

        store.load_from_db().await.unwrap();

        let info = store
            .get(&JobId::from("job-bad-file-status"))
            .await
            .unwrap();
        let file = info
            .file_statuses
            .iter()
            .find(|file| file.filename == "bad.cha")
            .expect("recovered file status");
        assert_eq!(file.status, FileStatusKind::Error);
        assert!(file.error_category.is_none());
        let error = file
            .error
            .clone()
            .expect("recovery should preserve invalid file persistence evidence");
        assert!(error.contains("original failure"));
        assert!(error.contains("invalid persisted file status 'mystery_file_status'"));
        assert!(error.contains("invalid persisted error_category 'mystery_category'"));
    }

    /// A row the reader had to reinterpret is rewritten once from what was
    /// loaded, so the next startup reads it without a report: a `done` file
    /// with a stale error loses the error column, and a job naming a
    /// submitter with no address loses the name.
    #[tokio::test]
    async fn reinterpreted_rows_are_repaired_once_on_load() {
        let (store, db, _dir) = test_store_with_db().await;
        let mut record = make_job_record("job-repair", JobStatus::Completed, vec!["a.cha".into()]);
        record.submitter = None;
        db.insert_job(&record).await.unwrap();
        db.seed_file_status_row(
            "job-repair",
            "a.cha",
            "done",
            Some("stale error"),
            None,
            None,
            Some(MachineTime::now()),
            Some(MachineTime::now()),
            None,
        )
        .await
        .unwrap();
        db.seed_job_submitter_columns("job-repair", "", "someone")
            .await
            .unwrap();

        store.load_from_db().await.unwrap();

        let rows = db.load_all_jobs().await.unwrap();
        let file = &rows[0].file_statuses[0];
        assert_eq!(file.status, "done");
        assert_eq!(file.error, None, "the stale error column is rewritten away");
        assert_eq!(
            rows[0].submitted_by_name, "",
            "the stray name is rewritten away"
        );
        assert!(
            matches!(
                recover_file_phase("job-repair", file),
                RecoveredFilePhase::Exact(_)
            ),
            "a repaired row needs no second repair"
        );
    }
    /// A row holding a status or category this build cannot read (another
    /// build's vocabulary, as after a rollback) is read as an error with a
    /// note, and left as stored: rewriting it would destroy what the other
    /// build wrote.
    #[tokio::test]
    async fn a_row_with_another_builds_vocabulary_is_never_rewritten() {
        let (store, db, _dir) = test_store_with_db().await;
        db.insert_job(&make_job_record(
            "job-foreign",
            JobStatus::Completed,
            vec!["new.cha".into()],
        ))
        .await
        .unwrap();
        db.seed_file_status_row(
            "job-foreign",
            "new.cha",
            "a_newer_status",
            Some("original failure"),
            Some("a_newer_category"),
            None,
            Some(crate::unix_time(10.0)),
            Some(crate::unix_time(11.0)),
            None,
        )
        .await
        .unwrap();

        store.load_from_db().await.unwrap();

        let rows = db.load_all_jobs().await.unwrap();
        let file = &rows[0].file_statuses[0];
        assert_eq!(file.status, "a_newer_status");
        assert_eq!(file.error_category.as_deref(), Some("a_newer_category"));
        assert_eq!(file.error.as_deref(), Some("original failure"));
    }

    /// A repair write that fails is reported and startup goes on: the jobs
    /// still load, and the row is repaired at a later startup.
    #[tokio::test]
    async fn a_failed_repair_write_does_not_stop_startup() {
        let (store, db, _dir) = test_store_with_db().await;
        db.insert_job(&make_job_record(
            "job-repair-fails",
            JobStatus::Completed,
            vec!["a.cha".into()],
        ))
        .await
        .unwrap();
        db.seed_file_status_row(
            "job-repair-fails",
            "a.cha",
            "done",
            Some("stale error"),
            None,
            None,
            Some(MachineTime::now()),
            Some(MachineTime::now()),
            None,
        )
        .await
        .unwrap();
        db.fail_file_status_updates().await.unwrap();

        let loaded = store
            .load_from_db()
            .await
            .expect("a failed repair write must not fail startup");
        assert_eq!(loaded, 1);
    }
}
