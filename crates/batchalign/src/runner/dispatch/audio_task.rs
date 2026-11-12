//! Shared outer task shell for audio-backed commands that produce one file.
//!
//! This is intentionally narrower than a full shared audio pipeline. `align`,
//! `transcribe` and `speaker-identify` still own different input preparation
//! and inner execution semantics, but they share the runner-side
//! retry/lifecycle/progress and final writeback shell.
//!
//! # Why the shell is not CHAT-specific
//!
//! It was, until 2026-09-02: `finalize_success` returned a `String` that the
//! shell wrote as CHAT. A command producing a JSON evidence file had nowhere
//! to go, and the honest options were a third copy of this shell or a flag
//! the CHAT path ignored. Returning [`FileOutput`] instead unifies the
//! writeback for all three commands.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::warn;

use crate::error::ServerError;
use crate::runner::util::{
    AfterFailure, FileAttempts, FileRunTracker, FileStage, FileTaskOutcome, RunnerEventSink,
    classify_server_error, spawn_observed_progress_forwarder, user_facing_error,
};
use crate::scheduling::{FailureCategory, WorkUnitKind};
use crate::store::{PendingJobFile, RunnerJobSnapshot};

use super::audio_output::{FileOutput, write_primary_output_artifact};

/// Command-owned inner task for one audio-backed file.
#[async_trait]
pub(crate) trait AudioFileTask {
    type AttemptOutput: Send;

    /// Run one inner attempt with a fresh per-attempt progress channel.
    async fn run_attempt(
        &mut self,
        progress_tx: crate::runner::util::ProgressSender,
    ) -> Result<Self::AttemptOutput, ServerError>;

    /// Convert one successful attempt result into the document to persist.
    ///
    /// The task states the KIND of document, because it is the only thing that
    /// knows. The shell then has one writeback path and no flag to get wrong.
    async fn finalize_success(
        &mut self,
        output: Self::AttemptOutput,
    ) -> Result<FileOutput, ServerError>;

    /// Optional command-owned recovery step before the shell records a retry.
    /// Unavailable analysis is an output shortfall. Cancellation or a typed
    /// internal recovery fault stops the file; provider errors do not.
    async fn on_retryable_worker_failure(
        &mut self,
        _lifecycle: &FileRunTracker<'_>,
        _error: &ServerError,
    ) -> Result<(), crate::provenance::UtrRecoveryStop> {
        Ok(())
    }
}

/// How one command names itself while the shell reports its progress.
///
/// The three travel together because they are one fact in three fields: what
/// this command is called, which work unit it books, and which stage a user
/// sees while it runs. They were three positional arguments of an eight-argument
/// function, where `WorkUnitKind::FileInfer` and `FileStage::Transcribing`
/// could be swapped between two commands without the compiler noticing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AudioTaskReporting {
    /// The work unit this file books.
    pub work_unit_kind: WorkUnitKind,
    /// The stage a user sees while the attempt runs.
    pub running_stage: FileStage,
    /// The command's name in operator-facing messages.
    pub command_label: &'static str,
}

/// Shared runner-owned shell for one audio-backed file task.
pub(crate) async fn run_audio_file_task<Task>(
    job: &RunnerJobSnapshot,
    sink: Arc<dyn RunnerEventSink>,
    file: &PendingJobFile,
    lifecycle: &FileRunTracker<'_>,
    reporting: AudioTaskReporting,
    task: &mut Task,
) -> FileTaskOutcome
where
    Task: AudioFileTask + Send,
{
    let AudioTaskReporting {
        work_unit_kind,
        running_stage,
        command_label,
    } = reporting;
    let job_id = &job.identity.job_id;
    let filename = file.filename.as_ref();
    let file_index = file.file_index;
    let mut attempts = FileAttempts::begin(lifecycle, work_unit_kind, running_stage).await;
    loop {
        // The observer records each checkout of this attempt that waits on a
        // saturated pool (without a deadline); the file shows "waiting for a
        // worker" while one does.
        let (progress_tx, wait_observer, forwarder) =
            spawn_observed_progress_forwarder(sink.clone(), filename.to_string());

        let attempt = crate::worker::pool::checkout_wait::observing_pool_waits(
            wait_observer,
            task.run_attempt(progress_tx),
        )
        .await;
        // Every update the attempt sent is published before its next stage.
        forwarder.finished().await;
        match attempt {
            Ok(output) => {
                let file_output = match task.finalize_success(output).await {
                    Ok(file_output) => file_output,
                    Err(error) => {
                        warn!(
                            job_id = %job_id,
                            correlation_id = %job.identity.correlation_id,
                            filename = %filename,
                            error = %error,
                            "Audio command finalization failed"
                        );
                        return lifecycle
                            .fail(
                                &format!("Failed to finalize {command_label} output: {error}"),
                                FailureCategory::System,
                            )
                            .await
                            .into();
                    }
                };
                lifecycle.stage(FileStage::Writing).await;
                let written = match write_primary_output_artifact(
                    &job.filesystem,
                    job.dispatch.command,
                    &job.dispatch.options,
                    file_index,
                    filename,
                    file_output,
                )
                .await
                {
                    Ok(written) => written,
                    Err(error) => {
                        warn!(
                            job_id = %job_id,
                            correlation_id = %job.identity.correlation_id,
                            filename = %filename,
                            error = %error,
                            "Failed to write audio command output"
                        );
                        return lifecycle
                            .fail(
                                // Both the sentence and the category come off
                                // the typed failure: a document refused by the
                                // output gate is a validity failure that wrote
                                // nothing, not a system-level write error, and
                                // this call site used to hardcode `System` and
                                // the word "Failed to write" for both.
                                &error.operator_message(command_label),
                                error.category(),
                            )
                            .await
                            .into();
                    }
                };

                // Done, or Diagnosed when a generating producer's output was
                // written with its findings: terminal either way, never an
                // error and never retried.
                return written.record(lifecycle).await.into();
            }
            Err(error) => {
                let category = classify_server_error(&error);
                let raw_msg = format!("{command_label} failed: {error}");
                warn!(
                    job_id = %job_id,
                    correlation_id = %job.identity.correlation_id,
                    filename,
                    category = %category,
                    raw_error = %raw_msg,
                    "Audio command error (raw)"
                );
                let message = user_facing_error(category, command_label, filename, &raw_msg);
                attempts = match attempts.after_error(&error, category, message).await {
                    AfterFailure::Retry(grant) => {
                        // Timing recovery runs while the retry is granted and
                        // not yet spent, so a cancelled recovery ends the file.
                        match task.on_retryable_worker_failure(lifecycle, &error).await {
                            Ok(()) => {}
                            Err(crate::provenance::UtrRecoveryStop::Cancelled) => {
                                return grant
                                    .abandon(
                                        "Timing recovery cancelled",
                                        FailureCategory::Cancelled,
                                    )
                                    .await
                                    .into();
                            }
                            Err(crate::provenance::UtrRecoveryStop::Failed(failure)) => {
                                return grant
                                    .abandon(&failure.to_string(), FailureCategory::System)
                                    .await
                                    .into();
                            }
                        }
                        grant.wait().await
                    }
                    AfterFailure::Recorded(recorded) => return recorded.into(),
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    mod utr_tests;
    use std::collections::BTreeMap;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::api::{
        CorrelationId, DisplayPath, JobId, LanguageCode3, LanguageSpec, NumSpeakers,
        ReleasedCommand,
    };
    use crate::options::{
        AsrEngineName, CommandOptions, CommonOptions, TranscribeOptions, WorTierPolicy,
    };
    use crate::runner::dispatch::audio_output::ChatOutput;
    use crate::scheduling::AttemptOutcome;
    use crate::store::{
        PendingJobFile, RunnerDispatchConfig, RunnerFilesystemConfig, RunnerJobIdentity,
    };
    use crate::worker::error::WorkerError;
    use batchalign_types::paths::{ClientPath, ServerPath};

    use crate::runner::util::test_sink::RecordingSink;

    struct FakeAudioTask {
        attempts: Arc<AtomicUsize>,
        recoveries: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl AudioFileTask for FakeAudioTask {
        type AttemptOutput = crate::pipeline::post_validate::PostValidated;

        async fn run_attempt(
            &mut self,
            _progress_tx: crate::runner::util::ProgressSender,
        ) -> Result<Self::AttemptOutput, ServerError> {
            let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
            if attempt == 0 {
                Err(ServerError::Worker(WorkerError::Stalled {
                    waited_for: crate::worker::LoadingWait::Ready,
                    stall: crate::worker::LoadingStall::new(
                        crate::api::PositiveSeconds::literal::<1>(),
                    ),
                }))
            } else {
                Ok(crate::pipeline::post_validate::PostValidated::for_test(
                    "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello .\n@End\n",
                    ReleasedCommand::Transcribe,
                ))
            }
        }

        async fn finalize_success(
            &mut self,
            output: Self::AttemptOutput,
        ) -> Result<FileOutput, ServerError> {
            Ok(FileOutput::Chat(ChatOutput {
                document: output.into(),
                shortfalls: Vec::new(),
                exclusions: Vec::new(),
            }))
        }

        async fn on_retryable_worker_failure(
            &mut self,
            _lifecycle: &FileRunTracker<'_>,
            _error: &ServerError,
        ) -> Result<(), crate::provenance::UtrRecoveryStop> {
            self.recoveries.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn fake_job(tmp: &tempfile::TempDir) -> RunnerJobSnapshot {
        RunnerJobSnapshot {
            identity: RunnerJobIdentity {
                job_id: JobId::from("audio-shell"),
                correlation_id: CorrelationId::from("corr-audio-shell"),
            },
            dispatch: RunnerDispatchConfig {
                command: ReleasedCommand::Transcribe,
                lang: LanguageSpec::Resolved(LanguageCode3::eng()),
                num_speakers: NumSpeakers(1),
                options: CommandOptions::Transcribe(TranscribeOptions {
                    auto_speakers: false,
                    common: CommonOptions::default(),
                    asr_engine: AsrEngineName::Whisper,
                    diarize: false,
                    wor: WorTierPolicy::Omit,
                    merge_abbrev: false.into(),
                    utseg_fallback: false.into(),
                    batch_size: 8,
                }),
                runtime_state: BTreeMap::new(),
                debug_traces: false,
            },
            filesystem: RunnerFilesystemConfig {
                paths_mode: true,
                source_paths: vec![ClientPath::new("/input/test.mp3")],
                output_paths: vec![ClientPath::new(
                    tmp.path()
                        .join("requested/test.cha")
                        .to_string_lossy()
                        .to_string(),
                )],
                before_paths: Vec::new(),
                staging_dir: ServerPath::new(tmp.path().join("staging")),
                media_mapping: Default::default(),
                media_subdir: Default::default(),
                source_dir: ClientPath::new("/input"),
            },
            cancel_token: CancellationToken::new(),
            pending_files: vec![PendingJobFile {
                file_index: 0,
                filename: DisplayPath::from("nested/test.mp3"),
                has_chat: false,
            }],
        }
    }

    #[tokio::test]
    async fn audio_shell_retries_retryable_worker_failures_then_writes_output() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let job = fake_job(&tmp);
        let file = job.pending_files[0].clone();
        let sink_impl = Arc::new(RecordingSink::default());
        let lifecycle = FileRunTracker::new(sink_impl.as_ref(), file.filename.as_ref());
        lifecycle
            .begin_first_attempt(WorkUnitKind::FileInfer, FileStage::ResolvingAudio)
            .await;

        let attempts = Arc::new(AtomicUsize::new(0));
        let recoveries = Arc::new(AtomicUsize::new(0));
        let mut task = FakeAudioTask {
            attempts: attempts.clone(),
            recoveries: recoveries.clone(),
        };
        let outcome = run_audio_file_task(
            &job,
            sink_impl.clone(),
            &file,
            &lifecycle,
            AudioTaskReporting {
                work_unit_kind: WorkUnitKind::FileInfer,
                running_stage: FileStage::Transcribing,
                command_label: "Transcription",
            },
            &mut task,
        )
        .await;

        assert!(matches!(outcome, FileTaskOutcome::TerminalStateRecorded(_)));
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert_eq!(recoveries.load(Ordering::SeqCst), 1);
        assert_eq!(sink_impl.retries().len(), 1);
        assert_eq!(sink_impl.done_files().len(), 1);
        assert!(sink_impl.errors().is_empty());
        assert!(
            std::fs::read_to_string(tmp.path().join("requested/test.cha"))
                .expect("written output")
                .contains("*PAR:\thello .")
        );
    }

    /// A task whose producer generated output that fails admission.
    struct DiagnosedAudioTask {
        attempts: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl AudioFileTask for DiagnosedAudioTask {
        type AttemptOutput = crate::pipeline::post_validate::ProducedOutput;

        async fn run_attempt(
            &mut self,
            _progress_tx: crate::runner::util::ProgressSender,
        ) -> Result<Self::AttemptOutput, ServerError> {
            use talkbank_model::model::{Line, TierContentItems, UtteranceContent, Word};

            self.attempts.fetch_add(1, Ordering::SeqCst);
            let chat = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello .\n@End\n";
            let mut file =
                batchalign_transform::parse::parse_lenient(&crate::chat_parser(), chat).0;
            for line in &mut file.lines {
                if let Line::Utterance(utt) = line {
                    utt.main.content.content = TierContentItems::new(vec![
                        UtteranceContent::Word(Box::new(Word::new(
                            talkbank_model::Location::Unlocated,
                            talkbank_model::WordText::new("hello").expect("nonempty fixture word"),
                        ))),
                        UtteranceContent::Word(Box::new(Word::new(
                            talkbank_model::Location::Unlocated,
                            talkbank_model::WordText::new("b2").expect("nonempty fixture word"),
                        ))),
                    ]);
                }
            }
            Ok(crate::pipeline::post_validate::PostValidated::produced(
                file,
                ReleasedCommand::Transcribe,
            ))
        }

        async fn finalize_success(
            &mut self,
            output: Self::AttemptOutput,
        ) -> Result<FileOutput, ServerError> {
            Ok(FileOutput::Chat(ChatOutput {
                document: output,
                shortfalls: Vec::new(),
                exclusions: Vec::new(),
            }))
        }
    }

    /// RED FIRST (2026-10-06): the runner shell writes a diagnosed transcript
    /// and records the file as diagnosed with its findings: not done, not an
    /// error, and not retried.
    #[tokio::test]
    async fn audio_shell_writes_a_diagnosed_transcript_and_records_it_diagnosed() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let job = fake_job(&tmp);
        let file = job.pending_files[0].clone();
        let sink_impl = Arc::new(RecordingSink::default());
        let lifecycle = FileRunTracker::new(sink_impl.as_ref(), file.filename.as_ref());
        lifecycle
            .begin_first_attempt(WorkUnitKind::FileInfer, FileStage::ResolvingAudio)
            .await;
        let attempts = Arc::new(AtomicUsize::new(0));
        let mut task = DiagnosedAudioTask {
            attempts: attempts.clone(),
        };

        let outcome = run_audio_file_task(
            &job,
            sink_impl.clone(),
            &file,
            &lifecycle,
            AudioTaskReporting {
                work_unit_kind: WorkUnitKind::FileInfer,
                running_stage: FileStage::Transcribing,
                command_label: "Transcription",
            },
            &mut task,
        )
        .await;

        assert!(matches!(outcome, FileTaskOutcome::TerminalStateRecorded(_)));
        assert_eq!(attempts.load(Ordering::SeqCst), 1, "never retried");
        assert!(sink_impl.done_files().is_empty());
        assert!(sink_impl.errors().is_empty());
        assert!(sink_impl.retries().is_empty());
        let diagnosed = sink_impl.diagnosed();
        assert_eq!(diagnosed.len(), 1);
        assert!(
            diagnosed[0]
                .first_findings()
                .iter()
                .any(|finding| finding.code.as_deref() == Some("E220")),
            "{:?}",
            diagnosed[0]
        );
        assert_eq!(
            sink_impl.finished_attempts(),
            vec![AttemptOutcome::Succeeded],
            "the attempt produced and wrote its output"
        );
        let written = std::fs::read_to_string(tmp.path().join("requested/test.cha"))
            .expect("the diagnosed transcript is on disk");
        assert!(written.contains("*PAR:\thello b2 ."), "{written}");
    }
}
