//! Per-file dispatch for `speaker-identify`.
//!
//! The command is `align`'s shape with a different output: a CHAT transcript
//! comes in, its recording is resolved beside it, one model runs over spans of
//! that recording, and one document is written per file. Everything shared with
//! `align` is shared code: the six-rung media search
//! (`media_search::resolve_transcript_media`) and the retry, lifecycle,
//! progress and writeback shell (`audio_task::run_audio_file_task`). What is
//! new here is only the part that is actually new.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use tracing::warn;

use crate::chat_ops::speaker_identity::{
    EmbeddingInferenceFailure, EmbeddingRequest, EmbeddingResponse, RunFacts,
    SpeakerEmbeddingInference, ThresholdPolicy, TierSelection, identify_speakers,
    pinned_embedding_revision, read_utterances,
};
use crate::error::ServerError;
use crate::options::CommandOptions;
use crate::runner::DispatchHostContext;
use crate::scheduling::{FailureCategory, WorkUnitKind};
use crate::store::{RunnerJobSnapshot, unix_now};
use crate::worker::pool::WorkerPool;
use crate::worker::speaker_embedding_request_v2::{
    PreparedRecording, parse_speaker_embedding_response_v2, prepare_recording_for_embedding,
};

use super::super::util::{
    FileRunTracker, FileStage, FileTaskOutcome, RunnerEventSink, drain_supervised_file_tasks,
    spawn_supervised_file_task,
};
use super::audio_output::FileOutput;
use super::audio_task::{AudioFileTask, AudioTaskReporting, run_audio_file_task};
use super::media_search::resolve_transcript_media;

/// The worker-backed embedding capability, as the pipeline sees it.
///
/// The pipeline depends on the TRAIT, so the whole decision path is provable
/// against a fake, and production still cannot ask the worker about a span that
/// never went through `PreparedPcm::locate`: the trait method consumes an
/// `EmbeddingRequest`, which only `identify_speakers` builds.
struct WorkerEmbedding {
    pool: Arc<WorkerPool>,
    pool_key: crate::api::LanguageCode3,
    recording: PreparedRecording,
    // Held so the prepared PCM file outlives every request that references it.
    // Dropping the runtime deletes the artifact, and a request naming a file
    // the worker can no longer open fails in a way that reads like a worker
    // fault rather than a lifetime bug.
    _artifacts: crate::worker::artifacts_v2::PreparedArtifactRuntimeV2,
}

#[async_trait]
impl SpeakerEmbeddingInference for WorkerEmbedding {
    async fn embed(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, EmbeddingInferenceFailure> {
        let response = self
            .pool
            .dispatch_execute_v2(&self.pool_key, &self.recording.request_for(&request))
            .await
            .map_err(|error| EmbeddingInferenceFailure::Dispatch {
                detail: error.to_string(),
            })?;

        parse_speaker_embedding_response_v2(&response, &request).map_err(|error| {
            EmbeddingInferenceFailure::InvalidResponse {
                detail: error.to_string(),
            }
        })
    }
}

/// Per-file task: read the transcript, score it, hand back the evidence.
struct SpeakerIdentityTask {
    filename: String,
    chat_text: String,
    audio_path: PathBuf,
    media_display: String,
    pool: Arc<WorkerPool>,
    pool_key: crate::api::LanguageCode3,
    options: crate::options::SpeakerIdentifyOptions,
}

#[async_trait]
impl AudioFileTask for SpeakerIdentityTask {
    /// The serialized evidence document.
    type AttemptOutput = String;

    async fn run_attempt(
        &mut self,
        _progress_tx: crate::runner::util::ProgressSender,
    ) -> Result<Self::AttemptOutput, ServerError> {
        let parser = crate::chat_parser();
        let chat_file = batchalign_transform::parse::parse_strict(&parser, &self.chat_text)
            .map_err(|errors| {
                ServerError::Validation(format!(
                    "speaker-identify requires a parseable CHAT transcript: {errors}"
                ))
            })?;

        let tiers = TierSelection::from_option(&self.options.tiers);
        let utterances = read_utterances(&chat_file, &tiers);

        let model_revision = pinned_embedding_revision()
            .map_err(|error| ServerError::Validation(error.to_string()))?;

        // Decoded ONCE. Every enrolled span and every utterance indexes into
        // this one decode, which is what makes their vectors comparable: two
        // embeddings from separately decoded files can differ for reasons that
        // have nothing to do with who was speaking.
        let artifacts =
            crate::worker::artifacts_v2::PreparedArtifactRuntimeV2::new("speaker_embedding_v2")
                .map_err(|error| ServerError::Validation(error.to_string()))?;
        let recording = prepare_recording_for_embedding(artifacts.store(), &self.audio_path)
            .await
            .map_err(|error| ServerError::Validation(error.to_string()))?;
        let prepared = recording.prepared;

        let inference = WorkerEmbedding {
            pool: self.pool.clone(),
            pool_key: self.pool_key.clone(),
            recording,
            _artifacts: artifacts,
        };

        let facts = RunFacts {
            transcript: self.filename.clone(),
            media: self.media_display.clone(),
            prepared_sample_rate_hz: prepared.sample_rate_hz(),
            embedding_backend: crate::types::worker_v2::SpeakerEmbeddingBackendV2::Pyannote,
            embedding_model_revision: model_revision,
            tiers: tiers.recorded(),
            produced_by: format!("batchalign3 {}", crate::cli::build_hash()),
        };

        let evidence = identify_speakers(
            facts,
            &self.options.enrollments,
            &utterances,
            prepared,
            &ThresholdPolicy::new(self.options.threshold),
            self.options.permutation,
            &inference,
        )
        .await
        .map_err(|error| ServerError::Validation(error.to_string()))?;

        serde_json::to_string_pretty(&evidence)
            .map(|mut json| {
                json.push('\n');
                json
            })
            .map_err(|error| {
                ServerError::Persistence(format!(
                    "could not serialize speaker-identity evidence for {}: {error}",
                    self.filename
                ))
            })
    }

    async fn finalize_success(
        &mut self,
        output: Self::AttemptOutput,
    ) -> Result<FileOutput, ServerError> {
        // Evidence, not CHAT: the transcript is not rewritten by this command.
        Ok(FileOutput::Evidence { body: output })
    }
}

/// Dispatch `speaker-identify` over every pending file.
pub(crate) async fn dispatch_speaker_identity(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    pool: Arc<WorkerPool>,
) {
    let job_id = &job.identity.job_id;
    let sink = host.sink().clone();

    let CommandOptions::SpeakerIdentify(options) = &job.dispatch.options else {
        let message =
            "speaker-identify job carries options for another command; its row is corrupt"
                .to_owned();
        sink.fail_job(job_id, &message, unix_now()).await;
        return;
    };

    // Speaker embedding is language-independent, but the pool still needs a
    // concrete key. Refused rather than invented, exactly as the other audio
    // commands do.
    let Some(pool_key) = job.dispatch.lang.as_resolved().cloned() else {
        let message = format!(
            "speaker-identify requires `--lang <iso3>`; got '{}'.",
            job.dispatch.lang
        );
        sink.fail_job(job_id, &message, unix_now()).await;
        return;
    };

    let mut tasks = Vec::new();
    for file in &job.pending_files {
        if job.cancel_token.is_cancelled() {
            break;
        }
        let job = job.clone();
        let host = host.clone();
        let sink = sink.clone();
        let pool = pool.clone();
        let pool_key = pool_key.clone();
        let options = options.clone();
        let file = file.clone();
        let filename = file.filename.clone();

        tasks.push(spawn_supervised_file_task(
            filename,
            "speaker-identify file task",
            async move {
                process_one_file(&job, &host, sink, pool, pool_key, options, &file).await
            },
        ));
    }

    let abnormal_exits =
        drain_supervised_file_tasks(sink.as_ref(), job_id, &job.cancel_token, tasks).await;
    if abnormal_exits > 0 {
        warn!(
            job_id = %job_id,
            abnormal_exits,
            "Supervised speaker-identify file tasks exited abnormally"
        );
    }
}

async fn process_one_file(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    sink: Arc<dyn RunnerEventSink>,
    pool: Arc<WorkerPool>,
    pool_key: crate::api::LanguageCode3,
    options: crate::options::SpeakerIdentifyOptions,
    file: &crate::store::PendingJobFile,
) -> FileTaskOutcome {
    let job_id = &job.identity.job_id;
    let filename = file.filename.as_ref();
    let file_index = file.file_index;
    let lifecycle = FileRunTracker::new(sink.as_ref(), job_id, filename);

    lifecycle
        .begin_first_attempt(WorkUnitKind::FileInfer, unix_now(), FileStage::Reading)
        .await;

    let read_path: PathBuf =
        if job.filesystem.paths_mode && file_index < job.filesystem.source_paths.len() {
            job.filesystem.source_paths[file_index]
                .assume_shared_filesystem()
                .as_path()
                .to_owned()
        } else {
            job.filesystem
                .staging_dir
                .join("input")
                .join(filename)
                .as_path()
                .to_owned()
        };

    let chat_text = match tokio::fs::read_to_string(&read_path).await {
        Ok(content) => content,
        Err(error) => {
            lifecycle
                .fail(
                    &format!("Failed to read input: {error}"),
                    FailureCategory::InputMissing,
                    unix_now(),
                )
                .await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
    };

    lifecycle.stage(FileStage::ResolvingAudio).await;
    let original_audio_path =
        match resolve_transcript_media(job, host, filename, read_path.as_path(), &chat_text, None)
            .await
        {
            Ok(path) => path,
            Err(unresolved) => {
                lifecycle
                    .fail(&unresolved.message, FailureCategory::Validation, unix_now())
                    .await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
        };

    let audio_path = match crate::ensure_wav::ensure_wav(&original_audio_path, None).await {
        Ok(path) => path,
        Err(error) => {
            lifecycle
                .fail(
                    &format!("Media conversion failed for {filename}: {error}"),
                    FailureCategory::Validation,
                    unix_now(),
                )
                .await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
    };

    let media_display = original_audio_path.to_string_lossy().to_string();
    let mut task = SpeakerIdentityTask {
        filename: filename.to_owned(),
        chat_text,
        audio_path,
        media_display,
        pool,
        pool_key,
        options,
    };

    run_audio_file_task(
        job,
        sink.clone(),
        file,
        &lifecycle,
        AudioTaskReporting {
            work_unit_kind: WorkUnitKind::FileInfer,
            running_stage: FileStage::Processing,
            command_label: "Speaker identification",
        },
        &mut task,
    )
    .await
}
