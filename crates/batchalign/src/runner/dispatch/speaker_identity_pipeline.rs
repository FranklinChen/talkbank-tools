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
use talkbank_model::model::TranscriptName;
use talkbank_model::validation::ValidChatFile;
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
use crate::store::RunnerJobSnapshot;
use crate::worker::pool::WorkerPool;
use crate::worker::speaker_embedding_request_v2::{
    PreparedRecording, parse_speaker_embedding_response_v2, prepare_recording_for_embedding,
};

use super::super::util::{
    FileRunTracker, FileStage, FileTaskOutcome, RunnerEventSink, classify_server_error,
    drain_supervised_file_tasks, spawn_supervised_file_task,
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
            .map_err(EmbeddingInferenceFailure::from)?;

        parse_speaker_embedding_response_v2(&response, &request)
            .map_err(EmbeddingInferenceFailure::from)
    }
}

/// Per-file task: read the transcript, score it, hand back the evidence.
struct SpeakerIdentityTask {
    filename: String,
    transcript: ValidChatFile,
    audio_path: PathBuf,
    media_display: String,
    pool: Arc<WorkerPool>,
    pool_key: crate::api::LanguageCode3,
    options: crate::options::SpeakerIdentifyOptions,
}

/// Retained transcript content has no generated-tier exemption. The producer
/// admits the named source once; attempts only borrow its immutable proof.
fn admit_speaker_transcript(
    source: &str,
    path: &std::path::Path,
) -> Result<ValidChatFile, ServerError> {
    crate::pipeline::text_infer::admit_retained_text(
        &crate::chat_parser(),
        source,
        TranscriptName::for_path(path),
    )
    .map(|source| source.into_valid_file())
}

#[async_trait]
impl AudioFileTask for SpeakerIdentityTask {
    /// The serialized evidence document.
    type AttemptOutput = String;

    async fn run_attempt(
        &mut self,
        _progress_tx: crate::runner::util::ProgressSender,
    ) -> Result<Self::AttemptOutput, ServerError> {
        let tiers = TierSelection::from_option(&self.options.tiers);
        let utterances = read_utterances(self.transcript.document(), &tiers);

        let model_revision = pinned_embedding_revision()?;

        // Decoded ONCE. Every enrolled span and every utterance indexes into
        // this one decode, which is what makes their vectors comparable: two
        // embeddings from separately decoded files can differ for reasons that
        // have nothing to do with who was speaking.
        let artifacts =
            crate::worker::artifacts_v2::PreparedArtifactRuntimeV2::new("speaker_embedding_v2")?;
        let recording =
            prepare_recording_for_embedding(artifacts.store(), &self.audio_path).await?;
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
            produced_by: format!("batchalign3 {}", crate::build_hash()),
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
        .await?;

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
        sink.fail_job(job_id, &message).await;
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
        sink.fail_job(job_id, &message).await;
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
        .begin_first_attempt(WorkUnitKind::FileInfer, FileStage::Reading)
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
                )
                .await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
    };

    lifecycle.stage(FileStage::Parsing).await;
    let transcript = match admit_speaker_transcript(&chat_text, read_path.as_path()) {
        Ok(transcript) => transcript,
        Err(error) => {
            lifecycle
                .fail(&error.to_string(), classify_server_error(&error))
                .await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
    };

    lifecycle.stage(FileStage::ResolvingAudio).await;
    let original_audio_path = match resolve_transcript_media(
        job,
        host,
        filename,
        read_path.as_path(),
        || crate::media::DeclaredMedia::from_document(transcript.document()),
        None,
    )
    .await
    {
        Ok(path) => path,
        Err(unresolved) => {
            lifecycle
                .fail(&unresolved.message, FailureCategory::Validation)
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
                )
                .await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
    };

    let media_display = original_audio_path.to_string_lossy().to_string();
    let mut task = SpeakerIdentityTask {
        filename: filename.to_owned(),
        transcript,
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

#[cfg(test)]
mod admission_tests {
    use super::*;

    const SOURCE: &str = "@UTF8\n@Begin\n@Languages:\teng\n\
        @Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n\
        @Media:\tsample, audio\n*PAR:\thello . \u{15}0_1000\u{15}\n@End\n";

    #[test]
    fn complete_named_source_is_the_task_input() {
        let admitted = admit_speaker_transcript(SOURCE, std::path::Path::new("sample.cha"))
            .expect("complete source");
        let utterances = read_utterances(admitted.document(), &TierSelection::AllTiers);
        assert_eq!(utterances.len(), 1);
        assert!(matches!(
            utterances[0].timing,
            crate::chat_ops::speaker_identity::UtteranceTiming::Window(_)
        ));
        assert!(matches!(
            crate::media::DeclaredMedia::from_document(admitted.document()),
            crate::media::DeclaredMedia::Expected
        ));
    }

    #[test]
    fn parseability_does_not_admit_missing_required_headers() {
        let source = SOURCE.replace("@Languages:\teng\n", "");
        assert!(batchalign_transform::parse::parse_strict(&crate::chat_parser(), &source).is_ok());
        let error = admit_speaker_transcript(&source, std::path::Path::new("sample.cha"))
            .expect_err("parseable does not mean valid");
        assert!(matches!(error, ServerError::ChatAdmission(_)));
        assert_eq!(classify_server_error(&error), FailureCategory::Validation);
    }

    /// Align may admit a source whose `%wor` is corrupt because it
    /// regenerates that tier; speaker identification regenerates nothing, so
    /// the same corruption refuses the file. The corruption is an unreadable
    /// bullet, invalid under every validity policy. (A reversed bullet was
    /// used until the pinned Chatter made `%wor` intervals lenient, as CHECK
    /// is, which turned this test red for a reason unrelated to its claim.)
    #[test]
    fn retained_dependent_tier_corruption_has_no_regeneration_exemption() {
        let source = SOURCE.replace("@End\n", "%wor:\thello \u{15}invalid\u{15} .\n@End\n");
        let error = admit_speaker_transcript(&source, std::path::Path::new("sample.cha"))
            .expect_err("speaker evidence does not regenerate word timing");
        assert!(matches!(error, ServerError::ChatAdmission(_)));
        assert_eq!(classify_server_error(&error), FailureCategory::Validation);
    }

    #[test]
    fn filename_and_main_interval_are_part_of_complete_admission() {
        for (source, path) in [
            (SOURCE.to_owned(), "other.cha"),
            (SOURCE.replace("0_1000", "1000_0"), "sample.cha"),
        ] {
            let error = admit_speaker_transcript(&source, std::path::Path::new(path))
                .expect_err("named structure must be valid before resolving media");
            assert_eq!(classify_server_error(&error), FailureCategory::Validation);
        }
    }
}
