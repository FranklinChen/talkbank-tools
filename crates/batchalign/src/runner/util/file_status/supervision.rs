//! Spawning and draining supervised file tasks, progress forwarding, and
//! terminal-state fallback cleanup.
//!
//! The supervision layer ensures that once a spawned file task stops running,
//! the runner knows whether the file reached a terminal state. Panics, early
//! returns, and cancellations are caught and converted to explicit failures.
//!
//! # A file task belongs to its job, and a cancelled job stops its tasks
//!
//! Every file task runs inside the `FileTaskScope` its runner established
//! around command dispatch. The scope carries two facts a spawned task would
//! otherwise lose, because `tokio::spawn` does not inherit task-locals:
//!
//! - the job id, re-established as the worker pool's `CURRENT_JOB_ID`, so a
//!   worker dispatch registers against the job and a cancel can find it;
//! - the job's cancellation token, which the supervisor races against the
//!   task. On cancel the task's future is DROPPED wherever it is, so no later
//!   stage (inference, forced alignment, output, debug dump) runs for a
//!   cancelled job, and a checked-out worker with a request in flight is
//!   discarded rather than returned to the pool.
//!
//! Before this, neither fact crossed the spawn: a cancelled align job's
//! in-flight files ran to completion,
//! because no dispatch had registered against the job for the cancel to kill
//! and nothing told the tasks to stop.

use std::future::Future;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tracing::Instrument;

use crate::api::{DisplayPath, JobId};
use crate::runner::job_scope::{FileTaskScope, SpawnScope};
use crate::scheduling::{AttemptOutcome, FailureCategory, RetryDisposition};

use super::tracker::set_file_progress;
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
/// responsible for two invariants: once the task stops running, the runner
/// must know whether the corresponding file already reached a terminal state;
/// and a task of a cancelled job stops (see the module docs).
pub(crate) fn spawn_supervised_file_task<F>(
    filename: DisplayPath,
    role: &'static str,
    future: F,
) -> SpawnedFileTask
where
    F: Future<Output = FileTaskOutcome> + Send + 'static,
{
    let scope = FileTaskScope::current();
    // Heap-owned before it enters any wrapper (see `crate::owned_future`).
    let future: crate::owned_future::OwnedFuture<'static, FileTaskOutcome> = Box::pin(future);
    // Every event the task emits carries its file and job (see
    // `SpawnScope::file_span`); `tokio::spawn` would otherwise start it with
    // no span at all.
    let span = scope.file_span(&filename, role);
    let handle = AbortOnDropHandle::new(tokio::spawn(
        async move {
            match scope {
                SpawnScope::Job(scope) => scope.supervise(future).await,
                // Only tests of this module spawn outside a runner. In
                // production it would mean a dispatch path that bypassed
                // `run_hosted_job`: say so, because such a task can be neither
                // cancelled nor found by a cancel's worker kill.
                SpawnScope::Unscoped => {
                    tracing::warn!(
                        role,
                        "supervised file task spawned outside a job scope; \
                     job cancellation cannot stop it"
                    );
                    future.await
                }
            }
        }
        .instrument(span),
    ));

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
        // A task that returned or died without a terminal state is attributed
        // to the cancellation when the job was cancelled: it may have seen the
        // token itself and returned early.
        let exit = match (task.handle.await, cancel_token.is_cancelled()) {
            (Ok(FileTaskOutcome::TerminalStateRecorded), _) => continue,
            // The supervisor stopped it: an expected consequence of the
            // cancel, not an abnormal exit.
            (Ok(FileTaskOutcome::StoppedByCancellation), _) => {
                record_abnormal_file_task_exit(
                    sink,
                    job_id,
                    task.filename.as_ref(),
                    task.role,
                    FileTaskExit::Cancelled,
                )
                .await;
                continue;
            }
            (Ok(FileTaskOutcome::MissingTerminalState) | Err(_), true) => FileTaskExit::Cancelled,
            (Ok(FileTaskOutcome::MissingTerminalState), false) => FileTaskExit::NoTerminalState,
            (Err(join_error), false) => FileTaskExit::Panicked(join_error.to_string()),
        };
        abnormal_exits += 1;
        record_abnormal_file_task_exit(sink, job_id, task.filename.as_ref(), task.role, exit).await;
    }

    abnormal_exits
}

// ---------------------------------------------------------------------------
// spawn_progress_forwarder: bridge progress channel to the event sink
// ---------------------------------------------------------------------------

/// Create a progress channel and spawn a forwarder task that routes updates
/// to the store for a specific `(job_id, filename)`.
///
/// Also returns the observer that records each of the file's checkouts
/// starting and stopping a wait on a saturated pool (install it around the
/// attempt with `worker::pool::checkout_wait::observing_checkout_waits`),
/// and the forwarder itself. The store keeps the reported stage and the open
/// waits as two facts and shows "waiting for a worker" while any wait is
/// open, so the forwarder publishes each as it comes, in order, and nothing
/// here restores a stage.
///
/// The forwarder runs until both the sender and the observer are dropped and
/// it has published everything they sent; await
/// [`FileProgressForwarder::finished`] after the attempt, before the file's
/// next stage is published, so no late update lands after it.
pub(crate) fn spawn_observed_progress_forwarder(
    sink: Arc<dyn RunnerEventSink>,
    job_id: JobId,
    filename: String,
) -> (
    super::ProgressSender,
    Arc<dyn crate::worker::pool::checkout_wait::CheckoutWaitObserver>,
    FileProgressForwarder,
) {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<super::ProgressUpdate>();
    let (wait_tx, mut wait_rx) =
        tokio::sync::mpsc::unbounded_channel::<crate::store::WorkerWaitChange>();
    let handle = tokio::spawn(async move {
        let mut progress_open = true;
        let mut waits_open = true;
        // Until both channels are closed and drained: a closed progress
        // channel must not drop a wait's end still queued, which would leave
        // the file counted as waiting.
        while progress_open || waits_open {
            tokio::select! {
                update = rx.recv(), if progress_open => match update {
                    Some(update) => {
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
                    None => progress_open = false,
                },
                change = wait_rx.recv(), if waits_open => match change {
                    Some(change) => sink.set_file_worker_wait(&job_id, &filename, change).await,
                    None => waits_open = false,
                },
            }
        }
    });
    (
        tx,
        Arc::new(ForwardedWaits { events: wait_tx }),
        FileProgressForwarder { handle },
    )
}

/// A running progress forwarder, to be awaited once its sender and observer
/// are gone.
#[must_use = "await `finished` after the attempt so no late update lands after its next stage"]
pub(crate) struct FileProgressForwarder {
    handle: tokio::task::JoinHandle<()>,
}

impl FileProgressForwarder {
    /// Wait until everything sent to the forwarder is published. Its sender
    /// and observer must already be dropped, or this waits for them.
    pub(crate) async fn finished(self) {
        if let Err(error) = self.handle.await {
            tracing::warn!(%error, "a file progress forwarder ended abnormally");
        }
    }
}

/// Run `work` with every checkout wait inside it recorded on these files, so
/// each shows "waiting for a worker" while any checkout of the work waits.
/// For work that is not an audio-file attempt: a text batch over one file or
/// several, or align's utterance-timing pre-pass. Returns once every wait is
/// recorded as ended.
pub(crate) async fn observing_file_waits<F: std::future::Future>(
    sink: &Arc<dyn RunnerEventSink>,
    job_id: &JobId,
    filenames: impl IntoIterator<Item = String>,
    work: F,
) -> F::Output {
    let (observers, forwarders): (Vec<_>, Vec<_>) = filenames
        .into_iter()
        .map(|filename| {
            // The work reports its own progress; only waits go through here.
            let (_progress, observer, forwarder) =
                spawn_observed_progress_forwarder(sink.clone(), job_id.clone(), filename);
            (observer, forwarder)
        })
        .unzip();
    let output = crate::worker::pool::checkout_wait::observing_checkout_waits(
        Arc::new(FannedOutWaits(observers)),
        work,
    )
    .await;
    for forwarder in forwarders {
        forwarder.finished().await;
    }
    output
}

/// One wait, shown on several files.
struct FannedOutWaits(Vec<Arc<dyn crate::worker::pool::checkout_wait::CheckoutWaitObserver>>);

impl crate::worker::pool::checkout_wait::CheckoutWaitObserver for FannedOutWaits {
    fn waiting(&self, target: &crate::worker::WorkerTarget, lang: &crate::api::WorkerLanguage) {
        for observer in &self.0 {
            observer.waiting(target, lang);
        }
    }

    fn resumed(&self) {
        for observer in &self.0 {
            observer.resumed();
        }
    }
}

/// The observer half of an observed forwarder.
struct ForwardedWaits {
    events: tokio::sync::mpsc::UnboundedSender<crate::store::WorkerWaitChange>,
}

impl crate::worker::pool::checkout_wait::CheckoutWaitObserver for ForwardedWaits {
    fn waiting(&self, _target: &crate::worker::WorkerTarget, _lang: &crate::api::WorkerLanguage) {
        // A send fails only when the forwarder has stopped, which it does
        // only after this observer is dropped: unreachable while it is
        // called. The pool logs the wait regardless.
        let _ = self.events.send(crate::store::WorkerWaitChange::Started);
    }

    fn resumed(&self) {
        // As for `waiting`.
        let _ = self.events.send(crate::store::WorkerWaitChange::Ended);
    }
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

    let now = sink.now();
    for filename in &unfinished {
        let last_status = sink
            .file_status_label(job_id, filename)
            .await
            .unwrap_or_default();
        let msg = format!("File did not reach terminal status (last status: {last_status})");
        sink.mark_file_error(job_id, filename, &msg, FailureCategory::System, now)
            .await;
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
    let finished_at = sink.now();
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

    sink.mark_file_error(
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

/// Why a file task ended without recording its own terminal state.
enum FileTaskExit {
    /// The job was cancelled.
    Cancelled,
    /// The task panicked or was aborted; the join error says which.
    Panicked(String),
    /// The task returned without recording a terminal state.
    NoTerminalState,
}

/// Record a non-standard file-task exit as an explicit terminal failure.
///
/// This path is only for supervision failures: task panic, task cancellation,
/// or a task returning without ever marking its file done/error.
async fn record_abnormal_file_task_exit(
    sink: &dyn RunnerEventSink,
    job_id: &JobId,
    filename: &str,
    role: &str,
    exit: FileTaskExit,
) {
    let finished_at = sink.now();
    let (message, category, outcome) = match exit {
        FileTaskExit::Cancelled => (
            format!("{role} stopped after job cancellation before recording a terminal file state"),
            FailureCategory::Cancelled,
            AttemptOutcome::Cancelled,
        ),
        FileTaskExit::Panicked(join_error) => (
            format!("{role} panicked before recording a terminal file state: {join_error}"),
            FailureCategory::System,
            AttemptOutcome::Failed,
        ),
        FileTaskExit::NoTerminalState => (
            format!("{role} exited without recording a terminal file state"),
            FailureCategory::System,
            AttemptOutcome::Failed,
        ),
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

    sink.mark_file_error(job_id, filename, &message, category, finished_at)
        .await;
}
