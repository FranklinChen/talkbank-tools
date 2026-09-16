//! Per-file Rust-owned V2 dispatch for media-analysis commands.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::Semaphore;
use tracing::{error, info, warn};

use crate::api::WorkerLanguage;
use crate::cache::UtteranceCache;
use crate::ensure_wav;
use crate::recipe_runner::runtime::{
    ChatOutputTarget, result_display_path_for_command, write_text_output_artifact,
};
use crate::runner::DispatchHostContext;
use crate::runner::debug_dumper::DebugDumper;
use crate::runner::util::{
    FileRunTracker, FileStage, FileTaskOutcome, RunnerEventSink, classify_server_error,
    classify_worker_error, drain_supervised_file_tasks, is_retryable_worker_failure,
    spawn_supervised_file_task, user_facing_error,
};
use crate::scheduling::{FailureCategory, RetryPolicy, WorkUnitKind};
use crate::store::{PendingJobFile, RunnerJobSnapshot, unix_now};
use crate::transcribe::{
    SpeakerEvidenceRunParams, SpeakerEvidenceSource, resolve_speaker_evidence_for_audio,
};
use crate::types::worker_v2::{AvqiResultV2, OpenSmileResultV2, TaskResultV2};
use crate::worker::artifacts_v2::PreparedArtifactRuntimeV2;
use crate::worker::avqi_request_v2::{
    AvqiBuildInputV2, PreparedAvqiRequestIdsV2, build_avqi_request_v2,
};
use crate::worker::execute_result_v2::require_success_result;
use crate::worker::opensmile_request_v2::{
    OpenSmileBuildInputV2, PreparedOpenSmileRequestIdsV2, build_opensmile_request_v2,
};
use crate::worker::pool::WorkerPool;

use super::diarize_turns::{SpeakerTurnsSource, format_turns_json};

use crate::api::{ContentType, NumWorkers};

use super::MediaAnalysisDispatchPlan;
use super::asr_media::resolve_paths_mode_or_staging_input;

/// Shared runtime dependencies for top-level media-analysis dispatch.
pub(crate) struct MediaAnalysisDispatchRuntime {
    /// Worker pool used for typed V2 media-analysis requests.
    pub pool: Arc<WorkerPool>,
    /// Shared durable evidence cache used by standalone diarization.
    pub cache: Arc<UtteranceCache>,
    /// Maximum number of file tasks to run concurrently for this job.
    pub num_workers: NumWorkers,
}

/// Dispatch per-file media-analysis commands through typed worker protocol V2.
pub(crate) async fn dispatch_media_analysis_v2(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    runtime: MediaAnalysisDispatchRuntime,
    plan: MediaAnalysisDispatchPlan,
) {
    let sink = host.sink().clone();
    let file_parallelism_hint = match &plan {
        MediaAnalysisDispatchPlan::Opensmile { kernel_plan, .. }
        | MediaAnalysisDispatchPlan::Avqi { kernel_plan }
        | MediaAnalysisDispatchPlan::Diarize { kernel_plan, .. } => {
            kernel_plan.file_parallelism_hint
        }
    };
    let file_parallelism = runtime
        .num_workers
        .0
        .max(1)
        .min(file_parallelism_hint.max(1));
    let file_sem = Arc::new(Semaphore::new(file_parallelism));
    let mut tasks = Vec::new();

    for file in &job.pending_files {
        if job.cancel_token.is_cancelled() {
            break;
        }

        let Ok(permit) = file_sem.clone().acquire_owned().await else {
            tracing::warn!("file semaphore closed during shutdown");
            break;
        };
        let sink = sink.clone();
        let pool = runtime.pool.clone();
        let cache = runtime.cache.clone();
        let job = job.clone();
        let file = file.clone();
        let filename = file.filename.clone();
        let plan = plan.clone();

        tasks.push(spawn_supervised_file_task(
            filename,
            "media-analysis V2 file task",
            async move {
                let _permit = permit;
                process_one_media_analysis_file_v2(&job, sink.clone(), &pool, &cache, &file, &plan)
                    .await
            },
        ));
    }

    let abnormal_exits = drain_supervised_file_tasks(
        sink.as_ref(),
        &job.identity.job_id,
        &job.cancel_token,
        tasks,
    )
    .await;
    if abnormal_exits > 0 {
        warn!(
            job_id = %job.identity.job_id,
            abnormal_exits,
            "Supervised media-analysis V2 file tasks exited abnormally"
        );
    }
}

async fn process_one_media_analysis_file_v2(
    job: &RunnerJobSnapshot,
    sink: Arc<dyn RunnerEventSink>,
    pool: &Arc<WorkerPool>,
    cache: &Arc<UtteranceCache>,
    file: &PendingJobFile,
    plan: &MediaAnalysisDispatchPlan,
) -> FileTaskOutcome {
    let job_id = &job.identity.job_id;
    let correlation_id = &*job.identity.correlation_id;
    let file_index = file.file_index;
    let filename = file.filename.as_ref();
    let lifecycle = FileRunTracker::new(sink.as_ref(), job_id, filename);
    let started_at = unix_now();

    lifecycle
        .begin_first_attempt(
            WorkUnitKind::FileInfer,
            started_at,
            FileStage::ResolvingAudio,
        )
        .await;

    let original_audio_path =
        resolve_paths_mode_or_staging_input(&job.filesystem, file_index, filename);

    let retry_policy = RetryPolicy::default();
    for attempt_number in 1..=retry_policy.max_attempts {
        if attempt_number > 1 {
            lifecycle
                .restart_attempt(WorkUnitKind::FileInfer, unix_now(), FileStage::Processing)
                .await;
        } else {
            lifecycle.stage(FileStage::Processing).await;
        }

        match dispatch_one_media_analysis_attempt(
            job,
            pool,
            cache,
            file_index,
            filename,
            &original_audio_path,
            plan,
        )
        .await
        {
            Ok((result_filename, output_text, output_type)) => {
                lifecycle.stage(FileStage::Writing).await;
                let finished_at = unix_now();
                let result_display_path = result_filename.clone().into();
                let target =
                    ChatOutputTarget::new(&job.filesystem, file_index, &result_display_path);
                if let Err(error) = write_text_output_artifact(&target, &output_text).await {
                    let err_msg = format!("Failed to write output for {filename}: {error}");
                    lifecycle
                        .fail(&err_msg, FailureCategory::System, finished_at)
                        .await;
                    return FileTaskOutcome::TerminalStateRecorded;
                }

                lifecycle
                    .complete_with_result(result_filename.clone().into(), output_type, finished_at)
                    .await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
            Err(DispatchFailure::RetryableWorker(error, category)) => {
                let finished_at = unix_now();
                let has_retry_budget = attempt_number < retry_policy.max_attempts;
                if has_retry_budget && is_retryable_worker_failure(category) {
                    let retry_number = attempt_number;
                    let backoff_ms = retry_policy.backoff_for_retry(retry_number);
                    let retry_at =
                        crate::api::UnixTimestamp(finished_at.0 + (backoff_ms.0 as f64 / 1000.0));
                    lifecycle
                        .retry(
                            retry_at,
                            category,
                            &format!("Worker error: {error}; retrying in {backoff_ms} ms"),
                            finished_at,
                        )
                        .await;
                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms.0)).await;
                    continue;
                }

                let raw_msg = format!("Worker error: {error}");
                warn!(
                    job_id = %job_id,
                    filename,
                    category = %category,
                    raw_error = %raw_msg,
                    "Media-analysis error (raw)"
                );
                let user_msg = user_facing_error(category, "Analysis", filename, &raw_msg);
                lifecycle.fail(&user_msg, category, finished_at).await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
            Err(DispatchFailure::Terminal(error, category)) => {
                let finished_at = unix_now();
                error!(
                    job_id = %job_id,
                    correlation_id = %correlation_id,
                    filename = %filename,
                    error = %error,
                    "Media-analysis V2 dispatch failed"
                );
                let user_msg = user_facing_error(category, "Analysis", filename, &error);
                lifecycle.fail(&user_msg, category, finished_at).await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
        }
    }

    FileTaskOutcome::MissingTerminalState
}

enum DispatchFailure {
    RetryableWorker(String, FailureCategory),
    Terminal(String, FailureCategory),
}

async fn dispatch_one_media_analysis_attempt(
    job: &RunnerJobSnapshot,
    pool: &Arc<WorkerPool>,
    cache: &Arc<UtteranceCache>,
    file_index: usize,
    filename: &str,
    original_audio_path: &Path,
    plan: &MediaAnalysisDispatchPlan,
) -> Result<(String, String, ContentType), DispatchFailure> {
    let audio_path = ensure_wav::ensure_wav(original_audio_path, None)
        .await
        .map_err(|error| {
            DispatchFailure::Terminal(
                format!("Media conversion failed for {filename}: {error}"),
                FailureCategory::Validation,
            )
        })?;

    match plan {
        MediaAnalysisDispatchPlan::Opensmile {
            kernel_plan: _,
            feature_set,
        } => {
            dispatch_opensmile_attempt(job, pool, file_index, filename, &audio_path, feature_set)
                .await
        }
        MediaAnalysisDispatchPlan::Avqi { kernel_plan: _ } => {
            dispatch_avqi_attempt(job, pool, file_index, filename, &audio_path).await
        }
        MediaAnalysisDispatchPlan::Diarize {
            kernel_plan: _,
            backend,
            expected_speakers,
            cache_policy,
        } => {
            dispatch_diarize_attempt(
                job,
                pool,
                cache,
                filename,
                SpeakerEvidenceRunParams {
                    audio_path: &audio_path,
                    backend: *backend,
                    expected_speakers: *expected_speakers,
                    cache_policy: *cache_policy,
                },
            )
            .await
        }
    }
}

async fn dispatch_opensmile_attempt(
    job: &RunnerJobSnapshot,
    pool: &Arc<WorkerPool>,
    file_index: usize,
    filename: &str,
    audio_path: &Path,
    feature_set: &str,
) -> Result<(String, String, ContentType), DispatchFailure> {
    let artifacts = PreparedArtifactRuntimeV2::new("opensmile_v2").map_err(|error| {
        DispatchFailure::Terminal(
            format!("failed to create openSMILE V2 artifact runtime: {error}"),
            FailureCategory::Validation,
        )
    })?;

    let request = build_opensmile_request_v2(
        artifacts.store(),
        OpenSmileBuildInputV2 {
            ids: &PreparedOpenSmileRequestIdsV2::new(
                format!("opensmile-v2-request-{file_index}"),
                format!("opensmile-v2-audio-{file_index}"),
            ),
            audio_path,
            feature_set,
            feature_level: "functionals",
        },
    )
    .await
    .map_err(|error| {
        DispatchFailure::Terminal(
            format!("failed to build openSMILE V2 request: {error}"),
            FailureCategory::Validation,
        )
    })?;

    // Media-analysis (opensmile, avqi) is not language-aware, but
    // `dispatch_execute_v2` still needs a concrete worker-pool key. We
    // refuse to invent one: if the job carries `Auto` / `PerFile`,
    // surface a typed error so the user passes `--lang <iso3>`.
    let pool_key = job.dispatch.lang.as_resolved().cloned().ok_or_else(|| {
        DispatchFailure::Terminal(
            format!(
                "media analysis requires `--lang <iso3>`; got '{}'.",
                job.dispatch.lang
            ),
            FailureCategory::Validation,
        )
    })?;
    let response = pool
        .dispatch_execute_v2(&pool_key, &request)
        .await
        .map_err(|error| {
            DispatchFailure::RetryableWorker(error.to_string(), classify_worker_error(&error))
        })?;

    // Outcome and payload are read as one thing. This path used to test the
    // payload first, and since a failed request carries `result: None` by
    // construction, every typed error response came out as "missing a result
    // payload" with the worker's own code and message discarded.
    let result = match require_success_result(&response, "openSMILE").map_err(|failure| {
        DispatchFailure::Terminal(failure.into(), FailureCategory::ProviderTerminal)
    })? {
        TaskResultV2::OpensmileResult(result) => result,
        other => {
            return Err(DispatchFailure::Terminal(
                format!("openSMILE V2 returned unexpected payload: {other:?}"),
                FailureCategory::ProviderTerminal,
            ));
        }
    };
    if !result.success {
        return Err(DispatchFailure::Terminal(
            result
                .error
                .clone()
                .unwrap_or_else(|| "openSMILE V2 runtime failed without detail".into()),
            FailureCategory::ProviderTerminal,
        ));
    }

    Ok((
        opensmile_result_filename(filename),
        format_opensmile_csv(result),
        ContentType::Csv,
    ))
}

async fn dispatch_avqi_attempt(
    job: &RunnerJobSnapshot,
    pool: &Arc<WorkerPool>,
    file_index: usize,
    filename: &str,
    cs_audio_path: &Path,
) -> Result<(String, String, ContentType), DispatchFailure> {
    let sv_audio_path = resolve_avqi_sv_path(cs_audio_path).ok_or_else(|| {
        DispatchFailure::Terminal(
            format!("AVQI input {filename} is missing a paired .sv. audio file name"),
            FailureCategory::Validation,
        )
    })?;
    let sv_audio_path = ensure_wav::ensure_wav(&sv_audio_path, None)
        .await
        .map_err(|error| {
            DispatchFailure::Terminal(
                format!("Media conversion failed for AVQI pair {filename}: {error}"),
                FailureCategory::Validation,
            )
        })?;

    let artifacts =
        PreparedArtifactRuntimeV2::new(format!("avqi_v2_{file_index}")).map_err(|error| {
            DispatchFailure::Terminal(
                format!("failed to create AVQI V2 artifact runtime: {error}"),
                FailureCategory::Validation,
            )
        })?;
    let request = build_avqi_request_v2(
        artifacts.store(),
        AvqiBuildInputV2 {
            ids: &PreparedAvqiRequestIdsV2::new(
                format!("avqi-v2-request-{file_index}"),
                format!("avqi-v2-cs-{file_index}"),
                format!("avqi-v2-sv-{file_index}"),
            ),
            cs_audio_path,
            sv_audio_path: &sv_audio_path,
        },
    )
    .await
    .map_err(|error| {
        DispatchFailure::Terminal(
            format!("failed to build AVQI V2 request: {error}"),
            FailureCategory::Validation,
        )
    })?;

    // Media-analysis (opensmile, avqi) is not language-aware, but
    // `dispatch_execute_v2` still needs a concrete worker-pool key. We
    // refuse to invent one: if the job carries `Auto` / `PerFile`,
    // surface a typed error so the user passes `--lang <iso3>`.
    let pool_key = job.dispatch.lang.as_resolved().cloned().ok_or_else(|| {
        DispatchFailure::Terminal(
            format!(
                "media analysis requires `--lang <iso3>`; got '{}'.",
                job.dispatch.lang
            ),
            FailureCategory::Validation,
        )
    })?;
    let response = pool
        .dispatch_execute_v2(&pool_key, &request)
        .await
        .map_err(|error| {
            DispatchFailure::RetryableWorker(error.to_string(), classify_worker_error(&error))
        })?;

    // Outcome and payload read as one thing, for the reason in the openSMILE
    // path above.
    let result = match require_success_result(&response, "AVQI").map_err(|failure| {
        DispatchFailure::Terminal(failure.into(), FailureCategory::ProviderTerminal)
    })? {
        TaskResultV2::AvqiResult(result) => result,
        other => {
            return Err(DispatchFailure::Terminal(
                format!("AVQI V2 returned unexpected payload: {other:?}"),
                FailureCategory::ProviderTerminal,
            ));
        }
    };
    if !result.success {
        return Err(DispatchFailure::Terminal(
            result
                .error
                .clone()
                .unwrap_or_else(|| "AVQI V2 runtime failed without detail".into()),
            FailureCategory::ProviderTerminal,
        ));
    }

    Ok((
        avqi_result_filename(filename),
        format_avqi_report(result, pool_key.as_ref()),
        ContentType::Text,
    ))
}

/// Serialize an openSMILE result to CSV using BA2's
/// `features-as-rows, single 'value' column` shape:
///
/// ```csv
/// feature,value
/// alphaFeature,1.5
/// betaFeature,2.5
/// ```
///
/// BA2's call chain that produces this shape:
///
/// 1. `batchalign/pipelines/opensmile/engine.py:88-93`: opensmile-python
///    returns `features_df` with shape `(N_segments, N_features)`. BA2
///    transposes once: `results_df = features_df.T`.
/// 2. `batchalign/cli/cli.py:546`: `features_df.to_csv(output_csv,
///    header=['value'], index_label='feature')`. With
///    `feature_level='functionals'` (the only mode BA2 exposes at the
///    CLI), the source frame collapses to `(N_features, 1)`, so the
///    CSV is a two-column file: `feature`, `value`.
///
/// BA3 emits the same shape so BA2-era researcher scripts that parse
/// `opensmile.csv` keep working. Feature order is alphabetical (it
/// comes from `BTreeMap`); BA2's order is opensmile-python's natural
/// feature-set order, which is feature-set-dependent.
fn format_opensmile_csv(result: &OpenSmileResultV2) -> String {
    // BA2's `feature_level='functionals'` invariant means one segment
    // per file. Higher-level callers should not be feeding multi-segment
    // (LLD-mode) results into this serializer, BA2 itself would have
    // crashed in pandas if asked to write multi-column data with a
    // single-element header list. Take the first segment when present
    // and ignore any extras.
    let segment = result.rows.first();
    let mut lines = Vec::with_capacity(result.num_features.saturating_add(1) as usize);
    lines.push("feature,value".to_string());
    if let Some(row) = segment {
        for (feature, value) in row {
            lines.push(format!("{feature},{value}"));
        }
    }
    // BA2's `pandas.to_csv` writes a trailing newline after the last
    // row. Mirror that: downstream `cat`/`wc`/diff tooling treats the
    // file as one record per line including the final one.
    lines.join("\n") + "\n"
}

/// Serialize an AVQI result to text using BA2's prose shape:
///
/// ```text
/// AVQI: 5.123
/// CPPS: 67.890
/// HNR: 12.346
/// Shimmer Local: 0.012
/// Shimmer Local dB: 1.234
/// LTAS Slope: -2.346
/// LTAS Tilt: 0.679
/// CS File: foo.cs.wav
/// SV File: foo.sv.wav
/// Language: eng
/// ```
///
/// BA2's writer at `batchalign/cli/cli.py:499-510` uses uppercase
/// labels with colon-space separator, three-decimal precision for the
/// seven numeric metrics, and trailing newline per line. BA3 emits the
/// same shape so BA2-era parsers of `.avqi.txt` keep working.
fn format_avqi_report(result: &AvqiResultV2, language: &str) -> String {
    let metrics = [
        ("AVQI", result.avqi),
        ("CPPS", result.cpps),
        ("HNR", result.hnr),
        ("Shimmer Local", result.shimmer_local),
        ("Shimmer Local dB", result.shimmer_local_db),
        ("LTAS Slope", result.slope),
        ("LTAS Tilt", result.tilt),
    ];
    let mut lines = Vec::with_capacity(metrics.len() + 3);
    for (label, value) in metrics {
        lines.push(format!("{label}: {value:.3}"));
    }
    lines.push(format!("CS File: {}", result.cs_file));
    lines.push(format!("SV File: {}", result.sv_file));
    lines.push(format!("Language: {language}"));
    lines.join("\n") + "\n"
}

async fn dispatch_diarize_attempt(
    job: &RunnerJobSnapshot,
    pool: &Arc<WorkerPool>,
    cache: &Arc<UtteranceCache>,
    filename: &str,
    params: SpeakerEvidenceRunParams<'_>,
) -> Result<(String, String, ContentType), DispatchFailure> {
    // Diarization is not language-aware, but `dispatch_execute_v2` still
    // needs a concrete worker-pool key (same contract as opensmile/avqi).
    let pool_key = job.dispatch.lang.as_resolved().cloned().ok_or_else(|| {
        DispatchFailure::Terminal(
            format!(
                "media analysis requires `--lang <iso3>`; got '{}'.",
                job.dispatch.lang
            ),
            FailureCategory::Validation,
        )
    })?;
    let backend = params.backend;
    let audio_path = params.audio_path;
    let resolution =
        resolve_speaker_evidence_for_audio(pool, cache, WorkerLanguage::from(pool_key), params)
            .await
            .map_err(|error| {
                let category = classify_server_error(&error);
                if is_retryable_worker_failure(category) {
                    DispatchFailure::RetryableWorker(error.to_string(), category)
                } else {
                    DispatchFailure::Terminal(error.to_string(), category)
                }
            })?;
    let evidence_identity = audio_path.to_string_lossy().into_owned();
    let dumper = DebugDumper::new(job.dispatch.options.common().debug_dir.as_deref());
    dumper
        .dump_speaker_evidence(&evidence_identity, &resolution.trace())
        .map_err(|error| DispatchFailure::Terminal(error.to_string(), FailureCategory::System))?;
    match resolution.source() {
        SpeakerEvidenceSource::ReplayedDerived => info!(
            cache_key = %resolution.cache_key(),
            backend = ?backend,
            "Replaying standalone speaker diarization evidence"
        ),
        SpeakerEvidenceSource::DerivedFromRaw => info!(
            cache_key = %resolution.cache_key(),
            backend = ?backend,
            "Deriving standalone speaker turns from retained raw evidence"
        ),
        SpeakerEvidenceSource::Inferred(reason) => info!(
            cache_key = %resolution.cache_key(),
            backend = ?backend,
            reason = ?reason,
            "Committed fresh standalone speaker diarization evidence"
        ),
    }

    let turns_json = format_turns_json(
        SpeakerTurnsSource::from_backend(backend),
        resolution.segments(),
    )
    .map_err(|error| {
        DispatchFailure::Terminal(
            format!("diarize output for {filename} is defective: {error}"),
            FailureCategory::ProviderTerminal,
        )
    })?;

    Ok((
        diarize_result_filename(filename),
        turns_json,
        ContentType::Json,
    ))
}

fn diarize_result_filename(filename: &str) -> String {
    result_display_path_for_command(crate::api::ReleasedCommand::Diarize, filename).to_string()
}

fn opensmile_result_filename(filename: &str) -> String {
    result_display_path_for_command(crate::api::ReleasedCommand::Opensmile, filename).to_string()
}

fn avqi_result_filename(filename: &str) -> String {
    result_display_path_for_command(crate::api::ReleasedCommand::Avqi, filename).to_string()
}

fn resolve_avqi_sv_path(cs_audio_path: &Path) -> Option<PathBuf> {
    let file_name = cs_audio_path.file_name()?.to_string_lossy();
    let lower = file_name.to_ascii_lowercase();
    let idx = lower.find(".cs.")?;
    let replacement = format!("{}.sv.{}", &file_name[..idx], &file_name[idx + 4..]);
    Some(cs_audio_path.with_file_name(replacement))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avqi_pair_resolution_rewrites_cs_to_sv() {
        let path = Path::new("/tmp/sample.cs.wav");
        assert_eq!(
            resolve_avqi_sv_path(path).expect("pair path"),
            PathBuf::from("/tmp/sample.sv.wav")
        );
    }

    #[test]
    fn avqi_output_filename_strips_cs_marker() {
        assert_eq!(avqi_result_filename("sample.cs.wav"), "sample.avqi.txt");
    }

    #[test]
    fn opensmile_output_filename_replaces_extension() {
        assert_eq!(
            opensmile_result_filename("sample.mp3"),
            "sample.opensmile.csv"
        );
    }

    /// BA2's opensmile CSV (`batchalign/cli/cli.py:546`) writes the
    /// post-transpose DataFrame with `header=['value']` and
    /// `index_label='feature'`. For the only mode BA2 exposes at the
    /// CLI (`feature_level='functionals'`), the source frame has shape
    /// `(N_features, 1)`, so the CSV is:
    ///
    /// ```csv
    /// feature,value
    /// alphaFeature,1.5
    /// betaFeature,2.5
    /// ```
    ///
    /// BA3 must emit the same shape so downstream BA2-era researcher
    /// scripts keep working.
    #[test]
    fn opensmile_csv_matches_ba2_feature_value_shape() {
        let mut row = std::collections::BTreeMap::new();
        row.insert("alphaFeature".to_string(), 1.5);
        row.insert("betaFeature".to_string(), 2.5);
        let result = OpenSmileResultV2 {
            feature_set: "eGeMAPSv02".to_string(),
            feature_level: "functionals".to_string(),
            num_features: 2,
            duration_segments: 1,
            audio_file: "sample.mp3".to_string(),
            rows: vec![row],
            success: true,
            error: None,
        };
        let csv = format_opensmile_csv(&result);
        assert_eq!(csv, "feature,value\nalphaFeature,1.5\nbetaFeature,2.5\n");
    }

    /// Empty result (no segments) should still emit the BA2 header so
    /// downstream parsers don't trip on a zero-byte file.
    #[test]
    fn opensmile_csv_empty_result_emits_header_only() {
        let result = OpenSmileResultV2 {
            feature_set: "eGeMAPSv02".to_string(),
            feature_level: "functionals".to_string(),
            num_features: 0,
            duration_segments: 0,
            audio_file: "sample.mp3".to_string(),
            rows: vec![],
            success: true,
            error: None,
        };
        let csv = format_opensmile_csv(&result);
        assert_eq!(csv, "feature,value\n");
    }

    /// BA2's avqi text report (`batchalign/cli/cli.py:499-510`) writes
    /// the seven metric values plus `CS File`, `SV File`, `Language`
    /// fields with `{Label}: {value:.3f}\n` formatting:
    ///
    /// ```text
    /// AVQI: 5.123
    /// CPPS: 67.890
    /// HNR: 12.345
    /// Shimmer Local: 0.012
    /// Shimmer Local dB: 1.234
    /// LTAS Slope: -2.345
    /// LTAS Tilt: 0.678
    /// CS File: foo.cs.wav
    /// SV File: foo.sv.wav
    /// Language: eng
    /// ```
    ///
    /// BA3 must emit the same shape so BA2-era researcher scripts that
    /// parse `.avqi.txt` keep working.
    #[test]
    fn avqi_report_matches_ba2_text_shape() {
        let result = AvqiResultV2 {
            avqi: 5.1234,
            cpps: 67.8900,
            hnr: 12.3456,
            shimmer_local: 0.0123,
            shimmer_local_db: 1.2345,
            slope: -2.3456,
            tilt: 0.6789,
            cs_file: "foo.cs.wav".to_string(),
            sv_file: "foo.sv.wav".to_string(),
            success: true,
            error: None,
        };
        let report = format_avqi_report(&result, "eng");
        let expected = "AVQI: 5.123\n\
                        CPPS: 67.890\n\
                        HNR: 12.346\n\
                        Shimmer Local: 0.012\n\
                        Shimmer Local dB: 1.234\n\
                        LTAS Slope: -2.346\n\
                        LTAS Tilt: 0.679\n\
                        CS File: foo.cs.wav\n\
                        SV File: foo.sv.wav\n\
                        Language: eng\n";
        assert_eq!(report, expected);
    }
}
