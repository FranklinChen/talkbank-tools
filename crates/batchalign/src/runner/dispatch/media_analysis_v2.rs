//! Per-file Rust-owned V2 dispatch for media-analysis commands.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::Semaphore;
use tracing::{error, info, warn};

use crate::api::WorkerLanguage;
use crate::cache::UtteranceCache;
use crate::ensure_wav;
use crate::runner::DispatchHostContext;
use crate::runner::debug_dumper::DebugDumper;
use crate::runner::util::{
    FileRunTracker, FileStage, FileTaskOutcome, RunnerEventSink, classify_server_error,
    classify_worker_error, drain_supervised_file_tasks, is_retryable_worker_failure,
    spawn_supervised_file_task, user_facing_error,
};
use crate::scheduling::{FailureCategory, RetryPolicy, WorkUnitKind};
use crate::store::{PendingJobFile, RunnerJobSnapshot};
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

use super::audio_output::{
    ChatOutput, FileOutput, MergeAbbreviations, write_primary_output_artifact,
};
use super::diarize_chat::MappedDiarizeSource;
use super::diarize_turns::{SpeakerTurnsSource, format_turns_json};

use crate::api::NumWorkers;

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
        let host = host.clone();

        tasks.push(spawn_supervised_file_task(
            filename,
            "media-analysis V2 file task",
            async move {
                let _permit = permit;
                process_one_media_analysis_file_v2(
                    &job,
                    &host,
                    sink.clone(),
                    &pool,
                    &cache,
                    &file,
                    &plan,
                )
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
    host: &DispatchHostContext,
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

    lifecycle
        .begin_first_attempt(WorkUnitKind::FileInfer, FileStage::ResolvingAudio)
        .await;

    let original_audio_path =
        resolve_paths_mode_or_staging_input(&job.filesystem, file_index, filename);

    if let MediaAnalysisDispatchPlan::Diarize { output_mode, .. } = plan
        && !output_mode.accepts_source_name(filename)
    {
        lifecycle
            .fail(
                "diarize source does not match its admitted mode; use --speaker-map for timed CHAT",
                FailureCategory::Validation,
            )
            .await;
        return FileTaskOutcome::TerminalStateRecorded;
    }

    // Parse once, outside worker retries. Retain the same immutable source and
    // its admitted timing/identity decisions until inference succeeds.
    let mapping = match plan {
        MediaAnalysisDispatchPlan::Diarize {
            output_mode: crate::options::DiarizeOutputMode::MappedChat { mapping },
            ..
        } => Some(mapping),
        _ => None,
    };
    let transcript = if mapping.is_some() {
        lifecycle.stage(FileStage::Reading).await;
        let text = match tokio::fs::read_to_string(&original_audio_path).await {
            Ok(text) => text,
            Err(error) => {
                lifecycle
                    .fail(
                        &format!("Failed to read CHAT input: {error}"),
                        FailureCategory::InputMissing,
                    )
                    .await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
        };
        lifecycle.stage(FileStage::Parsing).await;
        match crate::pipeline::text_infer::admit_retained_text(
            &crate::chat_parser(),
            &text,
            talkbank_model::model::TranscriptName::for_path(&original_audio_path),
        ) {
            Ok(source) => Some(source.into_valid_file()),
            Err(error) => {
                lifecycle
                    .fail(&error.to_string(), classify_server_error(&error))
                    .await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
        }
    } else {
        None
    };
    let mapped_source = match (transcript.as_ref(), mapping) {
        (Some(transcript), Some(mapping)) => {
            match MappedDiarizeSource::admit(transcript, mapping) {
                Ok(source) => Some(source),
                Err(error) => {
                    lifecycle.fail(&error.to_string(), error.category()).await;
                    return FileTaskOutcome::TerminalStateRecorded;
                }
            }
        }
        _ => None,
    };
    let mut attempt = MediaAnalysisAttempt {
        job,
        host,
        pool,
        cache,
        file_index,
        filename,
        original_path: &original_audio_path,
        plan,
        mapped_source,
    };

    let retry_policy = RetryPolicy::default();
    for attempt_number in 1..=retry_policy.max_attempts {
        if attempt_number > 1 {
            lifecycle
                .restart_attempt(WorkUnitKind::FileInfer, FileStage::Processing)
                .await;
        } else {
            lifecycle.stage(FileStage::Processing).await;
        }

        match attempt.run().await {
            Ok(output) => {
                lifecycle.stage(FileStage::Writing).await;
                let written = match write_primary_output_artifact(
                    &job.filesystem,
                    job.dispatch.command,
                    &job.dispatch.options,
                    file_index,
                    filename,
                    output,
                )
                .await
                {
                    Ok(written) => written,
                    Err(error) => {
                        lifecycle
                            .fail(&error.operator_message("Analysis"), error.category())
                            .await;
                        return FileTaskOutcome::TerminalStateRecorded;
                    }
                };
                written.record(&lifecycle).await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
            Err(DispatchFailure::RetryableWorker(error, category)) => {
                let has_retry_budget = attempt_number < retry_policy.max_attempts;
                if has_retry_budget && is_retryable_worker_failure(category) {
                    let retry_number = attempt_number;
                    let backoff_ms = retry_policy.backoff_for_retry(retry_number);
                    lifecycle
                        .retry_after(
                            backoff_ms.duration(),
                            category,
                            &format!("Worker error: {error}; retrying in {backoff_ms} ms"),
                        )
                        .await;
                    tokio::time::sleep(backoff_ms.duration()).await;
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
                lifecycle.fail(&user_msg, category).await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
            Err(DispatchFailure::Terminal(error, category)) => {
                error!(
                    job_id = %job_id,
                    correlation_id = %correlation_id,
                    filename = %filename,
                    error = %error,
                    "Media-analysis V2 dispatch failed"
                );
                let user_msg = user_facing_error(category, "Analysis", filename, &error);
                lifecycle.fail(&user_msg, category).await;
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

struct MediaAnalysisAttempt<'runtime, 'source> {
    job: &'runtime RunnerJobSnapshot,
    host: &'runtime DispatchHostContext,
    pool: &'runtime Arc<WorkerPool>,
    cache: &'runtime Arc<UtteranceCache>,
    file_index: usize,
    filename: &'runtime str,
    original_path: &'runtime Path,
    plan: &'runtime MediaAnalysisDispatchPlan,
    // Consumed only after worker/cache success. A retry retains the same proof.
    mapped_source: Option<MappedDiarizeSource<'source>>,
}

impl MediaAnalysisAttempt<'_, '_> {
    async fn run(&mut self) -> Result<FileOutput, DispatchFailure> {
        let Self {
            job,
            host,
            pool,
            cache,
            file_index,
            filename,
            original_path,
            plan,
            mapped_source,
        } = self;
        let job = *job;
        let host = *host;
        let pool = *pool;
        let cache = *cache;
        let file_index = *file_index;
        let filename = *filename;
        let original_audio_path = *original_path;
        match *plan {
            MediaAnalysisDispatchPlan::Opensmile {
                kernel_plan: _,
                feature_set,
            } => {
                let audio_path = prepare_analysis_audio(original_audio_path).await?;
                dispatch_opensmile_attempt(job, pool, file_index, &audio_path, feature_set)
                    .await
                    .map(|body| FileOutput::Evidence { body })
            }
            MediaAnalysisDispatchPlan::Avqi { kernel_plan: _ } => {
                dispatch_avqi_attempt(job, pool, file_index, original_audio_path)
                    .await
                    .map(|body| FileOutput::Evidence { body })
            }
            MediaAnalysisDispatchPlan::Diarize {
                kernel_plan: _,
                backend,
                expected_speakers,
                cache_policy,
                output_mode,
            } => {
                if let crate::options::DiarizeOutputMode::MappedChat { .. } = output_mode {
                    let source = mapped_source.as_ref().ok_or_else(|| {
                        DispatchFailure::Terminal(
                            "mapped diarization source proof was already consumed".to_owned(),
                            FailureCategory::System,
                        )
                    })?;
                    if !source.needs_inference() {
                        let source = mapped_source.take().ok_or_else(|| {
                            DispatchFailure::Terminal(
                                "mapped diarization source proof was already consumed".to_owned(),
                                FailureCategory::System,
                            )
                        })?;
                        let document = source.apply(&[]).map_err(|error| {
                            DispatchFailure::Terminal(error.to_string(), error.category())
                        })?;
                        return Ok(FileOutput::Chat(ChatOutput {
                            document: document.into(),
                            shortfalls: Vec::new(),
                            merge_abbreviations: MergeAbbreviations::Leave,
                        }));
                    }
                    let recording = super::media_search::resolve_transcript_media(
                        job,
                        host,
                        filename,
                        original_audio_path,
                        || crate::media::DeclaredMedia::from_document(source.document()),
                        None,
                    )
                    .await
                    .map_err(|error| {
                        DispatchFailure::Terminal(error.message, FailureCategory::Validation)
                    })?;
                    let audio_path = prepare_analysis_audio(&recording).await?;
                    let resolution = resolve_diarize_evidence(
                        job,
                        pool,
                        cache,
                        SpeakerEvidenceRunParams {
                            audio_path: &audio_path,
                            backend: *backend,
                            expected_speakers: *expected_speakers,
                            cache_policy: *cache_policy,
                        },
                    )
                    .await?;
                    let source = mapped_source.take().ok_or_else(|| {
                        DispatchFailure::Terminal(
                            "mapped diarization source proof was already consumed".to_owned(),
                            FailureCategory::System,
                        )
                    })?;
                    let document = source.apply(resolution.segments()).map_err(|error| {
                        DispatchFailure::Terminal(error.to_string(), error.category())
                    })?;
                    return Ok(FileOutput::Chat(ChatOutput {
                        document: document.into(),
                        shortfalls: Vec::new(),
                        merge_abbreviations: MergeAbbreviations::Leave,
                    }));
                }
                let audio_path = prepare_analysis_audio(original_audio_path).await?;
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
                .map(|body| FileOutput::Evidence { body })
            }
        }
    }
}

async fn prepare_analysis_audio(source: &Path) -> Result<PathBuf, DispatchFailure> {
    ensure_wav::ensure_wav(source, None).await.map_err(|error| {
        DispatchFailure::Terminal(
            format!("Media conversion failed for {}: {error}", source.display()),
            FailureCategory::Validation,
        )
    })
}

/// Pair the original recordings before conversion can replace their names.
struct AvqiSourcePair {
    cs: PathBuf,
    sv: PathBuf,
}

impl AvqiSourcePair {
    fn resolve(cs: &Path) -> Result<Self, DispatchFailure> {
        let sv = resolve_avqi_sv_path(cs).ok_or_else(|| {
            DispatchFailure::Terminal(
                format!(
                    "AVQI input {} is missing a paired .sv. audio file name",
                    cs.display()
                ),
                FailureCategory::Validation,
            )
        })?;
        Ok(Self {
            cs: cs.to_owned(),
            sv,
        })
    }

    async fn prepare(self) -> Result<PreparedAvqiPair, DispatchFailure> {
        let cs_audio = prepare_analysis_audio(&self.cs).await?;
        let sv_audio = prepare_analysis_audio(&self.sv).await?;
        Ok(PreparedAvqiPair {
            source: self,
            cs_audio,
            sv_audio,
        })
    }
}

/// Converted paths remain bound to their original, name-resolved source pair.
struct PreparedAvqiPair {
    source: AvqiSourcePair,
    cs_audio: PathBuf,
    sv_audio: PathBuf,
}

async fn dispatch_opensmile_attempt(
    job: &RunnerJobSnapshot,
    pool: &Arc<WorkerPool>,
    file_index: usize,
    audio_path: &Path,
    feature_set: &str,
) -> Result<String, DispatchFailure> {
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

    Ok(format_opensmile_csv(result))
}

async fn dispatch_avqi_attempt(
    job: &RunnerJobSnapshot,
    pool: &Arc<WorkerPool>,
    file_index: usize,
    cs_audio_path: &Path,
) -> Result<String, DispatchFailure> {
    let pair = AvqiSourcePair::resolve(cs_audio_path)?.prepare().await?;

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
            cs_audio_path: &pair.cs_audio,
            sv_audio_path: &pair.sv_audio,
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
    let report = AdmittedAvqiReport::admit(result, &pair.source)?;

    Ok(report.format(pool_key.as_ref()))
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
struct AdmittedAvqiReport<'a> {
    result: &'a AvqiResultV2,
    source: &'a AvqiSourcePair,
}

impl<'a> AdmittedAvqiReport<'a> {
    fn admit(
        result: &'a AvqiResultV2,
        source: &'a AvqiSourcePair,
    ) -> Result<Self, DispatchFailure> {
        if !result.success || result.error.is_some() {
            return Err(DispatchFailure::Terminal(
                result
                    .error
                    .clone()
                    .unwrap_or_else(|| "AVQI V2 runtime failed without detail".into()),
                FailureCategory::ProviderTerminal,
            ));
        }
        if [
            result.avqi,
            result.cpps,
            result.hnr,
            result.shimmer_local,
            result.shimmer_local_db,
            result.slope,
            result.tilt,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err(DispatchFailure::Terminal(
                "AVQI V2 returned non-finite metrics".into(),
                FailureCategory::ProviderTerminal,
            ));
        }
        Ok(Self { result, source })
    }

    fn format(&self, language: &str) -> String {
        let result = self.result;
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
        lines.push(format!("CS File: {}", self.source.cs.display()));
        lines.push(format!("SV File: {}", self.source.sv.display()));
        lines.push(format!("Language: {language}"));
        lines.join("\n") + "\n"
    }
}

async fn dispatch_diarize_attempt(
    job: &RunnerJobSnapshot,
    pool: &Arc<WorkerPool>,
    cache: &Arc<UtteranceCache>,
    filename: &str,
    params: SpeakerEvidenceRunParams<'_>,
) -> Result<String, DispatchFailure> {
    let backend = params.backend;
    let resolution = resolve_diarize_evidence(job, pool, cache, params).await?;
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
    Ok(turns_json)
}

/// Both output modes share the same worker/cache admission and retained trace.
async fn resolve_diarize_evidence(
    job: &RunnerJobSnapshot,
    pool: &Arc<WorkerPool>,
    cache: &Arc<UtteranceCache>,
    params: SpeakerEvidenceRunParams<'_>,
) -> Result<crate::transcribe::ResolvedSpeakerEvidence, DispatchFailure> {
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

    Ok(resolution)
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
    fn avqi_source_pair_retains_container_names_before_conversion() {
        let pair = match AvqiSourcePair::resolve(Path::new("recording.cs.mp4")) {
            Ok(pair) => pair,
            Err(_) => panic!("source pair should resolve"),
        };
        assert_eq!(pair.cs, PathBuf::from("recording.cs.mp4"));
        assert_eq!(pair.sv, PathBuf::from("recording.sv.mp4"));
        assert!(AvqiSourcePair::resolve(Path::new("cache-fingerprint.wav")).is_err());
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
            cs_file: "temporary/prepared-cs.pcm".to_string(),
            sv_file: "temporary/prepared-sv.pcm".to_string(),
            success: true,
            error: None,
        };
        let pair = match AvqiSourcePair::resolve(Path::new("foo.cs.wav")) {
            Ok(pair) => pair,
            Err(_) => panic!("source pair should resolve"),
        };
        let admitted = match AdmittedAvqiReport::admit(&result, &pair) {
            Ok(report) => report,
            Err(_) => panic!("successful finite metrics should admit"),
        };
        let report = admitted.format("eng");
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
        let mut refused = result.clone();
        refused.success = false;
        assert!(AdmittedAvqiReport::admit(&refused, &pair).is_err());
        refused.success = true;
        refused.error = Some("analysis failed".into());
        assert!(AdmittedAvqiReport::admit(&refused, &pair).is_err());
        refused.error = None;
        refused.avqi = f64::NAN;
        assert!(AdmittedAvqiReport::admit(&refused, &pair).is_err());
    }
}
