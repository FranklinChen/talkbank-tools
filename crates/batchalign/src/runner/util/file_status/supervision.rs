//! Spawning and draining supervised file tasks, progress forwarding, and
//! terminal-state fallback cleanup.
//!
//! The supervision layer ensures that once a spawned file task stops running,
//! the runner knows whether the file reached a terminal state. Panics, early
//! returns, and cancellations are caught and converted to explicit failures.

use std::future::Future;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;

use crate::api::{DisplayPath, JobId};
use crate::scheduling::{AttemptOutcome, FailureCategory, RetryDisposition};
use crate::store::unix_now;

use super::tracker::{set_file_error, set_file_progress};
use super::{FileTaskOutcome, RunnerEventSink};

// ---------------------------------------------------------------------------
// SpawnedFileTask: handle to one supervised file task
// ---------------------------------------------------------------------------

/// Handle to one spawned file task whose terminal file-state transition is
/// supervised by the runner rather than inferred later from a job-wide sweep.
pub(crate) struct SpawnedFileTask {
    /// Human-readable task role for diagnostics (`"align file"`, etc.).
    pub role: &'static str,
    /// Logical filename owned by this task.
    pub filename: DisplayPath,
    /// Owned task: losing the supervisor aborts it rather than detaching work
    /// that could write errors after the worker pool or registry shuts down.
    pub handle: AbortOnDropHandle<FileTaskOutcome>,
}

/// Spawn one supervised file task.
///
/// The inner future still owns the real command logic. The supervision layer is
/// responsible only for one invariant: once the task stops running, the runner
/// must know whether the corresponding file already reached a terminal state.
pub(crate) fn spawn_supervised_file_task<F>(
    filename: DisplayPath,
    role: &'static str,
    future: F,
) -> SpawnedFileTask
where
    F: Future<Output = FileTaskOutcome> + Send + 'static,
{
    let handle = AbortOnDropHandle::new(tokio::spawn(future));

    SpawnedFileTask {
        role,
        filename,
        handle,
    }
}

// ---------------------------------------------------------------------------
// drain_supervised_file_tasks: await all tasks and handle abnormal exits
// ---------------------------------------------------------------------------

/// Drain a batch of supervised file tasks and convert abnormal exits into
/// explicit file failures immediately.
///
/// This keeps panics and early returns from being discovered only by the
/// runner's coarse "force unfinished files to terminal state" fallback.
pub(crate) async fn drain_supervised_file_tasks(
    sink: &dyn RunnerEventSink,
    job_id: &JobId,
    cancel_token: &CancellationToken,
    tasks: Vec<SpawnedFileTask>,
) -> usize {
    let mut abnormal_exits = 0usize;

    for task in tasks {
        match task.handle.await {
            Ok(FileTaskOutcome::TerminalStateRecorded) => {}
            Ok(FileTaskOutcome::MissingTerminalState) => {
                abnormal_exits += 1;
                record_abnormal_file_task_exit(
                    sink,
                    job_id,
                    task.filename.as_ref(),
                    task.role,
                    cancel_token.is_cancelled(),
                    None,
                )
                .await;
            }
            Err(join_error) => {
                abnormal_exits += 1;
                record_abnormal_file_task_exit(
                    sink,
                    job_id,
                    task.filename.as_ref(),
                    task.role,
                    cancel_token.is_cancelled(),
                    Some(join_error.to_string()),
                )
                .await;
            }
        }
    }

    abnormal_exits
}

// ---------------------------------------------------------------------------
// spawn_progress_forwarder: bridge progress channel to the event sink
// ---------------------------------------------------------------------------

/// Create a progress channel and spawn a forwarder task that routes updates
/// to the store for a specific `(job_id, filename)`.
///
/// Returns the sender half. The forwarder runs until the sender is dropped.
pub(crate) fn spawn_progress_forwarder(
    sink: Arc<dyn RunnerEventSink>,
    job_id: JobId,
    filename: String,
) -> super::ProgressSender {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<super::ProgressUpdate>();
    tokio::spawn(async move {
        while let Some(update) = rx.recv().await {
            set_file_progress(
                sink.as_ref(),
                &job_id,
                &filename,
                update.label,
                update.current,
                update.total,
            )
            .await;
        }
    });
    tx
}

// ---------------------------------------------------------------------------
// force_terminal_file_states: last-resort cleanup for leaked tasks
// ---------------------------------------------------------------------------

/// Fallback cleanup for any files that still failed to reach a terminal state.
///
/// The supervised file-task boundary should normally make this path a no-op.
/// It remains as a last-resort guard against leaked tasks or other control-
/// plane bugs.
pub(crate) async fn force_terminal_file_states(
    sink: &dyn RunnerEventSink,
    job_id: &JobId,
) -> usize {
    let unfinished: Vec<DisplayPath> = sink.unfinished_files(job_id).await;

    if unfinished.is_empty() {
        return 0;
    }

    let now = unix_now();
    for filename in &unfinished {
        let last_status = sink
            .file_status_label(job_id, filename)
            .await
            .unwrap_or_default();
        let msg = format!("File did not reach terminal status (last status: {last_status})");
        set_file_error(sink, job_id, filename, &msg, FailureCategory::System, now).await;
    }

    sink.bump_forced_terminal_errors(unfinished.len()).await;
    unfinished.len()
}

// ---------------------------------------------------------------------------
// record_file_cancelled_before_dispatch: never-started files on cancellation
// ---------------------------------------------------------------------------

/// Record a file's dispatch as cancelled before its task was ever spawned.
///
/// A cancellation observed while a file is still waiting for a dispatch slot
/// (the per-job file-parallelism semaphore) must not leave the file silently
/// absent from the run's accounting: this gives it the same typed
/// `Cancelled` outcome a task that started and was then cancelled gets from
/// [`record_abnormal_file_task_exit`]'s cancelled branch, so a cancelled
/// job's file list always accounts for every pending file, not just the
/// ones whose task actually started.
pub(crate) async fn record_file_cancelled_before_dispatch(
    sink: &dyn RunnerEventSink,
    job_id: &JobId,
    filename: &str,
) {
    let finished_at = unix_now();
    let message = "dispatch cancelled before this file's task was ever started".to_owned();

    sink.finish_file_attempt(
        job_id,
        filename,
        AttemptOutcome::Cancelled,
        Some(FailureCategory::Cancelled),
        RetryDisposition::TerminalFailure,
        finished_at,
    )
    .await;

    set_file_error(
        sink,
        job_id,
        filename,
        &message,
        FailureCategory::Cancelled,
        finished_at,
    )
    .await;
}

// ---------------------------------------------------------------------------
// record_abnormal_file_task_exit: internal helper
// ---------------------------------------------------------------------------

/// Record a non-standard file-task exit as an explicit terminal failure.
///
/// This path is only for supervision failures: task panic, task cancellation,
/// or a task returning without ever marking its file done/error.
async fn record_abnormal_file_task_exit(
    sink: &dyn RunnerEventSink,
    job_id: &JobId,
    filename: &str,
    role: &str,
    job_cancelled: bool,
    join_error: Option<String>,
) {
    let finished_at = unix_now();
    let (message, category, outcome) = if job_cancelled {
        (
            format!("{role} stopped after job cancellation before recording a terminal file state"),
            FailureCategory::Cancelled,
            AttemptOutcome::Cancelled,
        )
    } else if let Some(join_error) = join_error {
        (
            format!("{role} panicked before recording a terminal file state: {join_error}"),
            FailureCategory::System,
            AttemptOutcome::Failed,
        )
    } else {
        (
            format!("{role} exited without recording a terminal file state"),
            FailureCategory::System,
            AttemptOutcome::Failed,
        )
    };

    sink.finish_file_attempt(
        job_id,
        filename,
        outcome,
        Some(category),
        RetryDisposition::TerminalFailure,
        finished_at,
    )
    .await;

    set_file_error(sink, job_id, filename, &message, category, finished_at).await;
}
