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
//! shell wrote as CHAT, and a `should_merge_abbrev: bool` rode along as the
//! eighth positional argument. A command producing a JSON evidence file had
//! nowhere to go, and the honest options were a third copy of this shell or a
//! flag the CHAT path ignored. Returning [`FileOutput`] instead deletes the
//! argument, unifies the writeback for all three commands, and makes
//! "abbreviation merging on a JSON document" unrepresentable rather than
//! silently ignored.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::warn;

use crate::error::ServerError;
use crate::runner::util::{
    FileRunTracker, FileStage, FileTaskOutcome, RunnerEventSink, classify_server_error,
    is_retryable_worker_failure, spawn_progress_forwarder, user_facing_error,
};
use crate::scheduling::{FailureCategory, RetryPolicy, WorkUnitKind};
use crate::store::{PendingJobFile, RunnerJobSnapshot, unix_now};

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
    async fn on_retryable_worker_failure(
        &mut self,
        _lifecycle: &FileRunTracker<'_>,
        _error: &ServerError,
    ) {
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
///
/// `should_merge_abbrev` used to be the eighth positional argument and no
/// longer is: it is a property of the CHAT document the task returns, so it
/// travels with that document instead.
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
    let retry_policy = RetryPolicy::default();
    for attempt_number in 1..=retry_policy.max_attempts {
        if attempt_number > 1 {
            lifecycle
                .restart_attempt(work_unit_kind, unix_now(), running_stage)
                .await;
        } else {
            lifecycle.stage(running_stage).await;
        }

        let progress_tx =
            spawn_progress_forwarder(sink.clone(), job_id.clone(), filename.to_string());

        match task.run_attempt(progress_tx).await {
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
                        lifecycle
                            .fail(
                                &format!("Failed to finalize {command_label} output: {error}"),
                                FailureCategory::System,
                                unix_now(),
                            )
                            .await;
                        return FileTaskOutcome::TerminalStateRecorded;
                    }
                };
                lifecycle.stage(FileStage::Writing).await;
                let finished_at = unix_now();
                let primary_output = match write_primary_output_artifact(
                    &job.filesystem,
                    job.dispatch.command,
                    file_index,
                    filename,
                    file_output,
                )
                .await
                {
                    Ok(primary_output) => primary_output,
                    Err(error) => {
                        warn!(
                            job_id = %job_id,
                            correlation_id = %job.identity.correlation_id,
                            filename = %filename,
                            error = %error,
                            "Failed to write audio command output"
                        );
                        lifecycle
                            .fail(
                                // Both the sentence and the category come off
                                // the typed failure: a document refused by the
                                // output gate is a validity failure that wrote
                                // nothing, not a system-level write error, and
                                // this call site used to hardcode `System` and
                                // the word "Failed to write" for both.
                                &error.operator_message(command_label),
                                error.category(),
                                finished_at,
                            )
                            .await;
                        return FileTaskOutcome::TerminalStateRecorded;
                    }
                };

                lifecycle
                    .complete_with_result(
                        primary_output.display_path.clone(),
                        primary_output.content_type,
                        finished_at,
                    )
                    .await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
            Err(error) => {
                let finished_at = unix_now();
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
                let err_msg = user_facing_error(category, command_label, filename, &raw_msg);
                let has_retry_budget = attempt_number < retry_policy.max_attempts;

                if matches!(&error, ServerError::Worker(_))
                    && is_retryable_worker_failure(category)
                    && has_retry_budget
                {
                    task.on_retryable_worker_failure(lifecycle, &error).await;
                    let backoff_ms = retry_policy.backoff_for_retry(attempt_number);
                    let retry_at =
                        crate::api::UnixTimestamp(finished_at.0 + (backoff_ms.0 as f64 / 1000.0));
                    lifecycle
                        .retry(
                            retry_at,
                            category,
                            &format!("{err_msg}; retrying in {backoff_ms} ms"),
                            finished_at,
                        )
                        .await;
                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms.0)).await;
                    continue;
                }

                lifecycle.fail(&err_msg, category, finished_at).await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
        }
    }

    FileTaskOutcome::MissingTerminalState
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use tokio_util::sync::CancellationToken;

    use super::super::audio_output::MergeAbbreviations;
    use super::*;
    use crate::api::{
        CorrelationId, DisplayPath, JobId, LanguageCode3, LanguageSpec, NumSpeakers,
        ReleasedCommand, UnixTimestamp,
    };
    use crate::options::{
        AsrEngineName, CommandOptions, CommonOptions, TranscribeOptions, WorTierPolicy,
    };
    use crate::scheduling::{AttemptOutcome, RetryDisposition};
    use crate::store::{
        PendingJobFile, RunnerDispatchConfig, RunnerFilesystemConfig, RunnerJobIdentity,
    };
    use crate::worker::error::WorkerError;
    use batchalign_types::paths::{ClientPath, ServerPath};

    #[derive(Default)]
    struct RecordingState {
        retries: usize,
        done: usize,
        errors: usize,
        started_attempts: usize,
        finished_attempts: Vec<AttemptOutcome>,
    }

    struct RecordingSink {
        state: Arc<Mutex<RecordingState>>,
    }

    impl RecordingSink {
        fn new() -> (Arc<Self>, Arc<Mutex<RecordingState>>) {
            let state = Arc::new(Mutex::new(RecordingState::default()));
            (
                Arc::new(Self {
                    state: state.clone(),
                }),
                state,
            )
        }
    }

    #[async_trait]
    impl RunnerEventSink for RecordingSink {
        async fn mark_file_processing(
            &self,
            _job_id: &JobId,
            _filename: &str,
            _started_at: UnixTimestamp,
        ) {
        }
        async fn mark_file_done(
            &self,
            _job_id: &JobId,
            _filename: &str,
            _finished_at: UnixTimestamp,
            _result: Option<crate::store::CompletedFileOutput>,
        ) {
            self.state.lock().unwrap().done += 1;
        }
        async fn mark_file_error(
            &self,
            _job_id: &JobId,
            _filename: &str,
            _error: &str,
            _category: FailureCategory,
            _finished_at: UnixTimestamp,
        ) {
            self.state.lock().unwrap().errors += 1;
        }
        async fn start_file_attempt(
            &self,
            _job_id: &JobId,
            _filename: &str,
            _work_unit_kind: WorkUnitKind,
            _started_at: UnixTimestamp,
        ) {
            self.state.lock().unwrap().started_attempts += 1;
        }
        async fn finish_file_attempt(
            &self,
            _job_id: &JobId,
            _filename: &str,
            outcome: AttemptOutcome,
            _failure_category: Option<FailureCategory>,
            _disposition: RetryDisposition,
            _finished_at: UnixTimestamp,
        ) {
            self.state.lock().unwrap().finished_attempts.push(outcome);
        }
        async fn mark_file_retry_pending(
            &self,
            _job_id: &JobId,
            _filename: &str,
            _retry_at: UnixTimestamp,
            _category: FailureCategory,
            _message: &str,
            _finished_at: UnixTimestamp,
        ) {
            self.state.lock().unwrap().retries += 1;
        }
        async fn clear_file_retry_state(&self, _job_id: &JobId, _filename: &str) {}
        async fn set_file_progress(
            &self,
            _job_id: &JobId,
            _filename: &str,
            _stage: FileStage,
            _current: Option<i64>,
            _total: Option<i64>,
        ) {
        }
        async fn unfinished_files(&self, _job_id: &JobId) -> Vec<DisplayPath> {
            Vec::new()
        }
        async fn file_status_label(&self, _job_id: &JobId, _filename: &str) -> Option<String> {
            None
        }
        async fn bump_forced_terminal_errors(&self, _count: usize) {}
        async fn fail_job(&self, _job_id: &JobId, _error: &str, _failed_at: UnixTimestamp) {}
        async fn mark_job_running(&self, _job_id: &JobId) {}
        async fn record_job_worker_count(&self, _job_id: &JobId, _worker_count: usize) {}
        async fn requeue_job_after_memory_gate(&self, _job_id: &JobId, _retry_at: UnixTimestamp) {}
        async fn bump_deferred_work_units(&self) {}
        async fn bump_memory_gate_aborts(&self) {}
        async fn finalize_job(
            &self,
            _job_id: &JobId,
            _expected_generation: crate::store::RunGeneration,
            _final_status: crate::api::JobStatus,
            _completed_at: UnixTimestamp,
        ) -> Option<String> {
            None
        }
    }

    struct FakeAudioTask {
        attempts: Arc<AtomicUsize>,
        recoveries: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl AudioFileTask for FakeAudioTask {
        type AttemptOutput = String;

        async fn run_attempt(
            &mut self,
            _progress_tx: crate::runner::util::ProgressSender,
        ) -> Result<Self::AttemptOutput, ServerError> {
            let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
            if attempt == 0 {
                Err(ServerError::Worker(WorkerError::ReadyTimeout {
                    timeout_s: 1,
                }))
            } else {
                Ok("@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\thello .\n@End\n".to_string())
            }
        }

        async fn finalize_success(
            &mut self,
            output: Self::AttemptOutput,
        ) -> Result<FileOutput, ServerError> {
            Ok(FileOutput::Chat {
                document: crate::runner::dispatch::audio_output::gate_built_chat_output(
                    &output,
                    ReleasedCommand::Transcribe,
                )
                .map_err(|failure| ServerError::Validation(failure.to_string()))?,
                merge_abbreviations: MergeAbbreviations::Leave,
            })
        }

        async fn on_retryable_worker_failure(
            &mut self,
            _lifecycle: &FileRunTracker<'_>,
            _error: &ServerError,
        ) {
            self.recoveries.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn fake_job(tmp: &tempfile::TempDir) -> RunnerJobSnapshot {
        RunnerJobSnapshot {
            run_generation: crate::store::RunGeneration::FIRST,
            identity: RunnerJobIdentity {
                job_id: JobId::from("audio-shell"),
                correlation_id: CorrelationId::from("corr-audio-shell"),
            },
            dispatch: RunnerDispatchConfig {
                command: ReleasedCommand::Transcribe,
                lang: LanguageSpec::Resolved(LanguageCode3::eng()),
                num_speakers: NumSpeakers(1),
                options: CommandOptions::Transcribe(TranscribeOptions {
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
        let (sink_impl, state) = RecordingSink::new();
        let lifecycle = FileRunTracker::new(
            sink_impl.as_ref(),
            &job.identity.job_id,
            file.filename.as_ref(),
        );
        lifecycle
            .begin_first_attempt(
                WorkUnitKind::FileInfer,
                unix_now(),
                FileStage::ResolvingAudio,
            )
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

        assert!(matches!(outcome, FileTaskOutcome::TerminalStateRecorded));
        let state = state.lock().unwrap();
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert_eq!(recoveries.load(Ordering::SeqCst), 1);
        assert_eq!(state.retries, 1);
        assert_eq!(state.done, 1);
        assert_eq!(state.errors, 0);
        assert!(
            std::fs::read_to_string(tmp.path().join("requested/test.cha"))
                .expect("written output")
                .contains("*PAR:\thello .")
        );
    }
}
