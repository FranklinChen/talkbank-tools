//! Transcription dispatch and per-file transcribe pipeline.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::api::{EngineVersion, NumWorkers};
use crate::cache::UtteranceCache;
use crate::pipeline::PipelineServices;
use crate::runner::DispatchHostContext;
use crate::scheduling::{FailureCategory, WorkUnitKind};
use crate::worker::pool::WorkerPool;
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::store::{RunnerJobSnapshot, unix_now};
use crate::transcribe::TranscribeOptions;

use super::super::util::{
    FileRunTracker, FileStage, FileTaskOutcome, RunnerEventSink, drain_supervised_file_tasks,
    record_file_cancelled_before_dispatch, spawn_supervised_file_task,
};
use super::TranscribeDispatchPlan;
use super::asr_media::{
    PreparedAsrMediaInput, prepare_asr_media_input, preserved_media_name_for_chat,
    resolve_paths_mode_or_staging_input,
};
use super::audio_output::{FileOutput, MergeAbbreviations};
use super::audio_task::{AudioFileTask, AudioTaskReporting, run_audio_file_task};

/// Shared runtime dependencies for top-level transcribe dispatch.
///
/// The runner always passes this bundle together after it chooses the
/// transcribe command family, so keeping it typed here makes the dispatch seam
/// explicit and keeps the function signature narrow.
pub(crate) struct TranscribeDispatchRuntime {
    /// Worker pool used for ASR and optional speaker diarization requests.
    pub pool: Arc<WorkerPool>,
    /// Shared utterance cache used by post-ASR server-side stages.
    pub cache: Arc<UtteranceCache>,
    /// Current engine version string for cache partitioning.
    pub engine_version: EngineVersion,
    /// Maximum number of file tasks to run concurrently for this job.
    pub num_workers: NumWorkers,
}

/// Outcome of waiting for one file's dispatch slot on the per-job
/// file-parallelism semaphore.
///
/// Named apart from a bare `Result<OwnedSemaphorePermit, _>` because there
/// are two distinct reasons the wait can end without a permit
/// (cancellation, a closed semaphore) and callers must react differently to
/// each: a cancellation means the remaining files get an explicit typed
/// `Cancelled` outcome recorded (see `record_file_cancelled_before_dispatch`),
/// a closed semaphore is an unrelated shutdown condition that is only
/// logged.
enum DispatchSlot {
    /// A dispatch slot was acquired; the file may proceed.
    Acquired(tokio::sync::OwnedSemaphorePermit),
    /// The job's cancellation token fired before a slot was acquired.
    Cancelled,
    /// The semaphore was closed (all senders dropped) before a slot was
    /// acquired.
    SemaphoreClosed,
}

/// Wait for one file's dispatch slot, racing the semaphore acquire against
/// the job's cancellation token.
///
/// A plain `.await` on the semaphore alone cannot observe a cancellation
/// while it is pending: the wait can block arbitrarily long (every slot
/// busy on a long-running file), so a cancellation issued during that wait
/// would otherwise be invisible until the semaphore finally admits the
/// waiter. Racing the two futures means a cancellation observed mid-wait
/// takes effect immediately.
async fn acquire_dispatch_slot(
    file_sem: &Arc<tokio::sync::Semaphore>,
    cancel_token: &CancellationToken,
) -> DispatchSlot {
    tokio::select! {
        biased;
        () = cancel_token.cancelled() => DispatchSlot::Cancelled,
        acquired = file_sem.clone().acquire_owned() => match acquired {
            Ok(permit) => DispatchSlot::Acquired(permit),
            Err(_) => DispatchSlot::SemaphoreClosed,
        },
    }
}

/// Dispatch transcribe via the server-side infer path.
///
/// Like FA, transcribe is per-file (each file has its own audio). Files are
/// processed concurrently up to `num_workers` at a time, bounded by a semaphore.
pub(crate) async fn dispatch_transcribe_infer(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    runtime: TranscribeDispatchRuntime,
    plan: TranscribeDispatchPlan,
) {
    let TranscribeDispatchPlan {
        kernel_plan,
        base_options,
        should_merge_abbrev,
    } = plan;
    let job_id = &job.identity.job_id;
    let sink = host.sink().clone();

    // Process files concurrently, bounded by available workers.
    let file_parallelism = runtime
        .num_workers
        .0
        .max(1)
        .min(kernel_plan.file_parallelism_hint.max(1));
    let file_sem = Arc::new(tokio::sync::Semaphore::new(file_parallelism));
    let mut tasks = Vec::new();

    let mut pending_iter = job.pending_files.iter();
    while let Some(file) = pending_iter.next() {
        // The wait for a dispatch slot can block arbitrarily long (every
        // slot busy on a long-running file); a plain `.await` on the
        // semaphore alone means a cancellation issued during that wait is
        // invisible until the waiting file finally acquires a permit and
        // starts anyway. `acquire_dispatch_slot` races the acquire against
        // the cancellation token so a cancellation observed HERE, mid-wait,
        // takes effect immediately instead of after one more file gets
        // dispatched.
        let permit = match acquire_dispatch_slot(&file_sem, &job.cancel_token).await {
            DispatchSlot::Acquired(permit) => permit,
            DispatchSlot::Cancelled => {
                // Every file still waiting, including this one, is
                // recorded as cancelled rather than silently dropped from
                // the run: see `record_file_cancelled_before_dispatch`.
                for skipped in std::iter::once(file).chain(pending_iter.clone()) {
                    record_file_cancelled_before_dispatch(
                        sink.as_ref(),
                        job_id,
                        skipped.filename.as_ref(),
                    )
                    .await;
                }
                break;
            }
            DispatchSlot::SemaphoreClosed => {
                tracing::warn!("file semaphore closed during shutdown");
                break;
            }
        };
        let sink = sink.clone();
        let pool = runtime.pool.clone();
        let cache = runtime.cache.clone();
        let job = job.clone();
        let engine_version = runtime.engine_version.clone();
        let mut opts = base_options.clone();
        let file = file.clone();
        let filename = file.filename.clone();

        tasks.push(spawn_supervised_file_task(
            filename,
            "transcribe file task",
            async move {
                let _permit = permit;
                let services = PipelineServices::new(&pool, &cache, &engine_version);
                process_one_transcribe_file(
                    &job,
                    sink.clone(),
                    services,
                    &file,
                    &mut opts,
                    should_merge_abbrev,
                )
                .await
            },
        ));
    }

    let abnormal_exits =
        drain_supervised_file_tasks(sink.as_ref(), job_id, &job.cancel_token, tasks).await;
    if abnormal_exits > 0 {
        warn!(
            job_id = %job_id,
            abnormal_exits,
            "Supervised transcribe file tasks exited abnormally"
        );
    }
}

async fn prepare_transcribe_media_input(
    filesystem: &crate::store::RunnerFilesystemConfig,
    file_index: usize,
    filename: &str,
) -> Result<PreparedAsrMediaInput, crate::error::ServerError> {
    let original_audio_path = resolve_paths_mode_or_staging_input(filesystem, file_index, filename);
    let media_name = preserved_media_name_for_chat(&original_audio_path, &original_audio_path);
    prepare_asr_media_input(original_audio_path, media_name, filename).await
}

struct TranscribeAudioTask<'a> {
    audio_path: PathBuf,
    services: PipelineServices<'a>,
    opts: TranscribeOptions,
    debug_dir: Option<&'a Path>,
    /// See `AlignAudioTask`: a property of the CHAT this task produces.
    merge_abbreviations: MergeAbbreviations,
}

#[async_trait]
impl AudioFileTask for TranscribeAudioTask<'_> {
    type AttemptOutput = String;

    async fn run_attempt(
        &mut self,
        progress_tx: crate::runner::util::ProgressSender,
    ) -> Result<Self::AttemptOutput, crate::error::ServerError> {
        crate::transcribe::process_transcribe(
            &self.audio_path,
            self.services,
            &self.opts,
            Some(progress_tx),
            self.debug_dir,
        )
        .await
    }

    async fn finalize_success(
        &mut self,
        output: Self::AttemptOutput,
    ) -> Result<FileOutput, crate::error::ServerError> {
        // Transcribe BUILDS its document from audio, so there is no admitted
        // input level to carry forward and this is where its proof is born.
        // See `gate_built_chat_output` for why the bar is L1 here and why
        // align does not come through it.
        let document = crate::runner::dispatch::audio_output::gate_built_chat_output(
            &output,
            crate::api::ReleasedCommand::Transcribe,
        )
        .map_err(|failure| crate::error::ServerError::Validation(failure.to_string()))?;
        Ok(FileOutput::Chat {
            document,
            merge_abbreviations: self.merge_abbreviations,
        })
    }
}

/// Process a single audio file through the transcribe pipeline.
async fn process_one_transcribe_file(
    job: &RunnerJobSnapshot,
    sink: Arc<dyn RunnerEventSink>,
    services: PipelineServices<'_>,
    file: &crate::store::PendingJobFile,
    opts: &mut TranscribeOptions,
    should_merge_abbrev: bool,
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

    let prepared_media =
        match prepare_transcribe_media_input(&job.filesystem, file_index, filename).await {
            Ok(prepared) => prepared,
            Err(error) => {
                let err_msg = error.to_string();
                lifecycle
                    .fail(&err_msg, FailureCategory::Validation, unix_now())
                    .await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
        };
    let PreparedAsrMediaInput {
        original_audio_path: _,
        inference_audio_path: audio_path,
        media_name,
    } = prepared_media;

    opts.media_name = media_name;

    if !audio_path.exists() {
        let err_msg = format!(
            "Resolved transcribe media path does not exist: {}",
            audio_path.display()
        );
        lifecycle
            .fail(&err_msg, FailureCategory::Validation, unix_now())
            .await;
        return FileTaskOutcome::TerminalStateRecorded;
    }

    let audio_path_str = audio_path.to_string_lossy();
    tracing::info!(
        job_id = %job_id,
        correlation_id = %correlation_id,
        filename = %filename,
        audio_path = %audio_path_str,
        "Starting transcribe for file"
    );

    let debug_dir = job.dispatch.options.common().debug_dir.as_deref();
    let shell_sink = sink.clone();
    let mut task = TranscribeAudioTask {
        audio_path,
        services,
        opts: opts.clone(),
        debug_dir,
        merge_abbreviations: MergeAbbreviations::from_option(should_merge_abbrev),
    };

    run_audio_file_task(
        job,
        shell_sink,
        file,
        &lifecycle,
        AudioTaskReporting {
            work_unit_kind: WorkUnitKind::FileInfer,
            running_stage: FileStage::Transcribing,
            command_label: "Transcription",
        },
        &mut task,
    )
    .await
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::path::Path;
    use std::sync::Arc;

    use tokio::sync::broadcast;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::api::{
        DisplayPath, EngineVersion, FileStatusKind, JobId, JobStatus, NumSpeakers, ReleasedCommand,
        UnixTimestamp,
    };
    use crate::api::{LanguageCode3, LanguageSpec};
    use crate::cache::UtteranceCache;
    use crate::db::JobDB;
    use crate::media::tools::MediaTool;
    use crate::options::{
        AsrEngineName, CommandOptions, CommonOptions, TranscribeOptions as TranscribeCommand,
    };
    use crate::runner::util::StoreRunnerEventSink;
    use crate::store::{
        FileStatus, Job, JobDispatchConfig, JobExecutionState, JobFilesystemConfig, JobIdentity,
        JobLeaseState, JobRuntimeControl, JobScheduleState, JobSourceContext, JobStore,
    };
    use crate::transcribe::AsrBackend;
    use crate::ws::BROADCAST_CAPACITY;

    /// Build a minimal transcribe job whose source path points at a missing
    /// media file so setup fails before any model work runs.
    fn make_transcribe_job(job_id: &str, source_path: &Path, output_path: &Path) -> Job {
        let filename = "missing.mp4";
        let mut file_statuses = HashMap::new();
        file_statuses.insert(
            filename.to_string(),
            FileStatus::new(DisplayPath::from(filename)),
        );

        Job {
            identity: JobIdentity {
                job_id: JobId::from(job_id),
                correlation_id: format!("test-{job_id}").into(),
            },
            dispatch: JobDispatchConfig {
                command: ReleasedCommand::Transcribe,
                lang: LanguageSpec::Resolved(LanguageCode3::eng()),
                num_speakers: NumSpeakers(1),
                options: CommandOptions::Transcribe(TranscribeCommand {
                    common: CommonOptions::default(),
                    asr_engine: AsrEngineName::RevAi,
                    diarize: false,
                    wor: false.into(),
                    merge_abbrev: false.into(),
                    batch_size: 8,
                    utseg_fallback: false.into(),
                }),
                runtime_state: BTreeMap::new(),
                debug_traces: false,
            },
            source: JobSourceContext {
                submitted_by: "127.0.0.1".into(),
                submitted_by_name: "localhost".into(),
                source_dir: Default::default(),
            },
            filesystem: JobFilesystemConfig {
                filenames: vec![DisplayPath::from(filename)],
                has_chat: vec![false],
                staging_dir: Default::default(),
                paths_mode: true,
                source_paths: vec![batchalign_types::paths::ClientPath::new(
                    source_path.to_string_lossy().to_string(),
                )],
                output_paths: vec![batchalign_types::paths::ClientPath::new(
                    output_path.to_string_lossy().to_string(),
                )],
                before_paths: Vec::new(),
                media_mapping: Default::default(),
                media_subdir: Default::default(),
                source_dir: Default::default(),
            },
            execution: JobExecutionState {
                status: JobStatus::Queued,
                file_statuses,
                results: Vec::new(),
                error: None,
                completed_files: 0,
            },
            schedule: JobScheduleState {
                submitted_at: UnixTimestamp(1.0),
                completed_at: None,
                next_eligible_at: None,
                num_workers: None,
                lease: JobLeaseState {
                    leased_by_node: None,
                    expires_at: None,
                    heartbeat_at: None,
                },
                last_cancel: None,
            },
            runtime: JobRuntimeControl {
                cancel_token: CancellationToken::new(),
                runner_active: false,
                run_generation: crate::store::RunGeneration::FIRST,
            },
            execution_plan: None,
        }
    }

    /// Media-conversion failure should still record a failed attempt because
    /// the first attempt now begins before audio normalization.
    #[tokio::test]
    async fn missing_media_records_failed_attempt() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let source_path = tempdir.path().join("missing.mp4");
        let output_path = tempdir.path().join("out");
        let cache_dir = tempdir.path().join("cache");
        let db = Arc::new(JobDB::open(Some(tempdir.path())).await.expect("open db"));
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = Arc::new(JobStore::new(
            crate::config::ServerConfig::default(),
            Some(db.clone()),
            tx,
        ));
        store
            .submit(make_transcribe_job(
                "job-transcribe",
                &source_path,
                &output_path,
            ))
            .await
            .expect("submit job");

        let snapshot = store
            .runner_snapshot(&JobId::from("job-transcribe"))
            .await
            .expect("runner snapshot");
        let file = snapshot
            .pending_files
            .first()
            .cloned()
            .expect("pending file");
        let pool = WorkerPool::new(crate::worker::pool::PoolConfig::default());
        let cache = UtteranceCache::sqlite(Some(cache_dir))
            .await
            .expect("open cache");
        let engine_version = EngineVersion::from("test-asr");
        let services = PipelineServices::new(&pool, &cache, &engine_version);
        let sink = StoreRunnerEventSink::wrap(store.clone());
        let mut opts = crate::transcribe::TranscribeOptions {
            backend: AsrBackend::Worker(crate::transcribe::AsrWorkerMode::LocalWhisperV2),
            diarize: false,
            speaker_backend: None,
            lang: LanguageSpec::Resolved(LanguageCode3::eng()),
            num_speakers: 1,
            with_utseg: true,
            with_morphosyntax: false,
            cache_policies: crate::transcribe::TranscribeCachePolicies::uniform(
                crate::params::CachePolicy::UseCache,
            ),
            allow_stanza_fallback_utseg: false,
            write_wor: false,
            media_name: None,
            engine_extras: std::collections::BTreeMap::new(),
        };

        process_one_transcribe_file(&snapshot, sink, services, &file, &mut opts, false).await;

        let attempts = db
            .load_attempts_for_job("job-transcribe")
            .await
            .expect("load attempts");
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].work_unit_kind, WorkUnitKind::FileInfer);
        assert_eq!(
            attempts[0].outcome,
            crate::scheduling::AttemptOutcome::Failed
        );
        assert_eq!(
            attempts[0].failure_category,
            Some(FailureCategory::Validation)
        );

        let detail = store
            .get_job_detail(&JobId::from("job-transcribe"))
            .await
            .expect("job detail");
        assert_eq!(detail.file_statuses.len(), 1);
        assert_eq!(detail.file_statuses[0].status, FileStatusKind::Error);
    }

    #[tokio::test]
    async fn transcribe_media_name_uses_original_basename_after_mp4_conversion() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().join("cache");
        let original_audio_path = dir.path().join("interview.mp4");
        let ffmpeg_out = MediaTool::Ffmpeg
            .async_command()
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono",
                "-t",
                "0.1",
                original_audio_path.to_string_lossy().as_ref(),
            ])
            .output()
            .await;
        if ffmpeg_out.is_err() || !ffmpeg_out.expect("ffmpeg output").status.success() {
            eprintln!("skipping: could not generate test mp4");
            return;
        }

        let converted_audio_path =
            crate::ensure_wav::ensure_wav(&original_audio_path, Some(&cache_dir))
                .await
                .expect("convert mp4 to cached wav");

        assert_ne!(
            converted_audio_path.file_stem(),
            original_audio_path.file_stem(),
            "test requires cached wav basename to differ from original media basename"
        );
        assert_eq!(
            preserved_media_name_for_chat(&original_audio_path, &converted_audio_path).as_deref(),
            Some("interview"),
            "CHAT @Media should preserve the original media basename after conversion"
        );
    }

    /// A cancellation that arrives while a file is genuinely blocked
    /// waiting for a dispatch slot (zero permits available) must win the
    /// race and be reported, not leave the waiter hanging or silently let
    /// it through.
    #[tokio::test]
    async fn acquire_dispatch_slot_reports_cancellation_during_the_wait() {
        let file_sem = Arc::new(tokio::sync::Semaphore::new(0));
        let cancel_token = CancellationToken::new();

        let waiting_sem = file_sem.clone();
        let waiting_token = cancel_token.clone();
        let handle =
            tokio::spawn(async move { acquire_dispatch_slot(&waiting_sem, &waiting_token).await });

        // Hand control to the scheduler so the spawned task actually
        // reaches its pending point (blocked on both the empty semaphore
        // and the not-yet-cancelled token) before cancellation fires.
        // Without this, the cancellation could race ahead of the task
        // ever being polled, which would prove nothing about the WAIT
        // being interrupted.
        tokio::task::yield_now().await;
        assert!(
            !handle.is_finished(),
            "the task must still be genuinely waiting on the semaphore before cancellation"
        );

        cancel_token.cancel();

        let outcome = handle.await.expect("task should not panic");
        assert!(
            matches!(outcome, DispatchSlot::Cancelled),
            "a cancellation observed mid-wait must be reported as Cancelled, not silently \
             dropped or treated as an acquired slot"
        );
    }

    /// The ordinary, uncancelled path: a slot becomes available and is
    /// reported as acquired.
    #[tokio::test]
    async fn acquire_dispatch_slot_reports_acquired_when_a_permit_is_free() {
        let file_sem = Arc::new(tokio::sync::Semaphore::new(1));
        let cancel_token = CancellationToken::new();

        let outcome = acquire_dispatch_slot(&file_sem, &cancel_token).await;
        assert!(matches!(outcome, DispatchSlot::Acquired(_)));
    }
}
