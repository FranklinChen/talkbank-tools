//! Per-file lifecycle tracker.
//!
//! Dispatch code records a file's lifecycle through [`FileRunTracker`], which
//! stamps every event from the sink's clock. The free helpers that used to
//! forward one sink call each, taking a time from their caller, are gone: a
//! time-taking door beside the tracker was a way around it.

use crate::api::{ContentType, DisplayPath, JobId};
use crate::scheduling::{AttemptOutcome, FailureCategory, RetryDisposition, WorkUnitKind};
use crate::store::{CompletedFileOutput, FileCompletion};

use super::{FileStage, RunnerEventSink};

/// Update ephemeral progress fields on a file and broadcast the update.
///
/// Progress fields are never persisted to SQLite; they are purely for
/// live display in the CLI/TUI/React dashboard. The one free helper left:
/// progress carries no time, so a caller holding the sink cannot misdate it.
pub(crate) async fn set_file_progress(
    sink: &dyn RunnerEventSink,
    job_id: &JobId,
    filename: &str,
    stage: FileStage,
    current: Option<i64>,
    total: Option<i64>,
) {
    sink.set_file_progress(job_id, filename, stage, current, total)
        .await;
}

// ---------------------------------------------------------------------------
// FileTaskOutcome: completion contract for supervised file tasks
// ---------------------------------------------------------------------------

/// Explicit completion contract for one supervised file task.
///
/// A task returns `TerminalStateRecorded` only after it has already written the
/// final file status that the runner should trust. An early return or panic
/// that skips that write path surfaces as `MissingTerminalState` so the
/// supervisor can record a concrete failure. A job cancellation that stopped
/// the task mid-flight is its own outcome, issued only by the supervision
/// wrapper that observed the cancellation.
#[derive(Debug)]
pub(crate) enum FileTaskOutcome {
    /// The task itself recorded success or terminal failure for the file.
    TerminalStateRecorded,
    /// The task exited without recording a terminal file state.
    MissingTerminalState,
    /// The job was cancelled and the supervisor dropped the task's future
    /// wherever it was: in a worker dispatch, a cache wait, or between stages.
    /// Nothing after that point ran, so no output, debug dump or cache write
    /// belongs to a cancelled job.
    StoppedByCancellation,
}

// ---------------------------------------------------------------------------
// FileRunTracker: per-file lifecycle helper
// ---------------------------------------------------------------------------

/// Runner-side helper for one file's lifecycle and attempt bookkeeping.
///
/// Every event is stamped here, from the sink's clock, never by the caller:
/// a pipeline cannot record a file finishing at a time it chose, and an event
/// that writes two records (the file and its attempt) writes one instant to
/// both. Dispatch code should prefer this helper over hand-sequencing raw
/// store mutations. That keeps the per-file state machine explicit:
///
/// - begin the first processing attempt
/// - move between human-readable stages
/// - restart the attempt after retryable failures
/// - finish as success, retry, or terminal error
pub(crate) struct FileRunTracker<'a> {
    sink: &'a dyn RunnerEventSink,
    job_id: &'a JobId,
    filename: &'a str,
}

impl<'a> FileRunTracker<'a> {
    /// Bind the helper to one `(job_id, filename)` pair.
    pub(crate) fn new(sink: &'a dyn RunnerEventSink, job_id: &'a JobId, filename: &'a str) -> Self {
        Self {
            sink,
            job_id,
            filename,
        }
    }

    /// Mark the file as processing, open the first durable attempt, and set the
    /// initial stage label shown to operators.
    pub(crate) async fn begin_first_attempt(&self, work_unit_kind: WorkUnitKind, stage: FileStage) {
        let started_at = self.sink.now();
        self.sink
            .mark_file_processing(self.job_id, self.filename, started_at)
            .await;
        self.sink
            .clear_file_retry_state(self.job_id, self.filename)
            .await;
        self.sink
            .start_file_attempt(self.job_id, self.filename, work_unit_kind, started_at)
            .await;
        self.stage(stage).await;
    }

    /// Open a durable setup attempt that fails before the file ever enters the
    /// normal processing pipeline. The attempt opens and fails at one instant,
    /// since setup is refused before any work runs.
    ///
    /// This is used for preflight rejection paths such as missing or
    /// incompatible media, where attempt history is still wanted but the file
    /// should not be advertised as actively processing.
    pub(crate) async fn record_setup_failure(&self, error: &str, category: FailureCategory) {
        let refused_at = self.sink.now();
        self.sink
            .clear_file_retry_state(self.job_id, self.filename)
            .await;
        self.sink
            .start_file_attempt(
                self.job_id,
                self.filename,
                WorkUnitKind::FileSetup,
                refused_at,
            )
            .await;
        self.sink
            .mark_file_error(self.job_id, self.filename, error, category, refused_at)
            .await;
    }

    /// Clear retry-only state, open the next attempt, and publish the stage
    /// label for the new run.
    pub(crate) async fn restart_attempt(&self, work_unit_kind: WorkUnitKind, stage: FileStage) {
        let started_at = self.sink.now();
        self.sink
            .clear_file_retry_state(self.job_id, self.filename)
            .await;
        self.sink
            .start_file_attempt(self.job_id, self.filename, work_unit_kind, started_at)
            .await;
        self.stage(stage).await;
    }

    /// Update the current human-readable progress stage.
    pub(crate) async fn stage(&self, stage: FileStage) {
        set_file_progress(self.sink, self.job_id, self.filename, stage, None, None).await;
    }

    /// Record a retryable failure now, eligible again after `backoff`.
    pub(crate) async fn retry_after(
        &self,
        backoff: std::time::Duration,
        category: FailureCategory,
        message: &str,
    ) {
        let finished_at = self.sink.now();
        self.sink
            .mark_file_retry_pending(
                self.job_id,
                self.filename,
                finished_at.deadline_after(backoff),
                category,
                message,
                finished_at,
            )
            .await;
    }

    /// Record a terminal file failure.
    pub(crate) async fn fail(&self, error: &str, category: FailureCategory) {
        let finished_at = self.sink.now();
        self.sink
            .mark_file_error(self.job_id, self.filename, error, category, finished_at)
            .await;
    }

    /// Mark the file as done with a downloadable result and close the active
    /// attempt as successful.
    ///
    /// Records no stamp decision: use [`Self::complete_with_stamped_result`]
    /// where the command decided one, so `Unrecorded` means "this command does
    /// not stamp per file" rather than "somebody forgot".
    pub(crate) async fn complete_with_result(
        &self,
        result_filename: DisplayPath,
        content_type: ContentType,
    ) {
        self.complete_with_stamped_result(
            result_filename,
            content_type,
            crate::api::FileStampOutcome::Unrecorded,
        )
        .await;
    }

    /// Mark the file as done, recording what the command decided about
    /// stamping it with provenance.
    pub(crate) async fn complete_with_stamped_result(
        &self,
        result_filename: DisplayPath,
        content_type: ContentType,
        stamp: crate::api::FileStampOutcome,
    ) {
        self.complete(FileCompletion::Clean(CompletedFileOutput {
            filename: result_filename,
            content_type,
            stamp,
            exclusions: Vec::new(),
        }))
        .await;
    }

    /// Mark the file as done: its output passed admission with nothing
    /// requested missing, and is listed with what its producer left out on
    /// purpose (information, never a reason to diagnose it).
    pub(crate) async fn complete_clean(
        &self,
        result_filename: DisplayPath,
        content_type: ContentType,
        exclusions: Vec<crate::api::OutputExclusionRecord>,
    ) {
        self.complete(FileCompletion::Clean(CompletedFileOutput {
            filename: result_filename,
            content_type,
            stamp: crate::api::FileStampOutcome::Unrecorded,
            exclusions,
        }))
        .await;
    }

    /// Mark the file as diagnosed: its output was written, together with what
    /// output admission found in it. Closes the attempt as successful, since
    /// the attempt did produce and write its output; the diagnostics travel
    /// with the file's terminal phase, never as an error and never retried.
    pub(crate) async fn complete_diagnosed(
        &self,
        result_filename: DisplayPath,
        content_type: ContentType,
        diagnostics: crate::api::FileOutputDiagnostics,
        exclusions: Vec<crate::api::OutputExclusionRecord>,
    ) {
        self.complete(FileCompletion::Diagnosed {
            result: CompletedFileOutput {
                filename: result_filename,
                content_type,
                stamp: crate::api::FileStampOutcome::Unrecorded,
                exclusions,
            },
            diagnostics,
        })
        .await;
    }

    /// Mark the file as done without a downloadable artifact and close the
    /// active attempt as successful.
    pub(crate) async fn complete_without_result(&self) {
        self.complete(FileCompletion::WithoutResult).await;
    }

    /// The file finished and its attempt succeeded, both at one instant.
    async fn complete(&self, completion: FileCompletion) {
        let finished_at = self.sink.now();
        self.sink
            .mark_file_done(self.job_id, self.filename, finished_at, completion)
            .await;
        self.sink
            .finish_file_attempt(
                self.job_id,
                self.filename,
                AttemptOutcome::Succeeded,
                None,
                RetryDisposition::Succeed,
                finished_at,
            )
            .await;
    }
}

// ---------------------------------------------------------------------------
// ProgressUpdate: typed channel messages from orchestrators to dispatch
// ---------------------------------------------------------------------------

/// A progress update from an orchestrator to the dispatch layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProgressUpdate {
    /// Typed lifecycle/progress label.
    pub label: FileStage,
    /// Current progress counter (optional).
    pub current: Option<i64>,
    /// Total items for progress (optional).
    pub total: Option<i64>,
}

impl ProgressUpdate {
    /// Construct a typed progress update for the shared file-status channel.
    pub(crate) fn new(label: FileStage, current: Option<i64>, total: Option<i64>) -> Self {
        Self {
            label,
            current,
            total,
        }
    }
}

/// Sender half for progress updates. Orchestrators hold this.
pub(crate) type ProgressSender = tokio::sync::mpsc::UnboundedSender<ProgressUpdate>;
