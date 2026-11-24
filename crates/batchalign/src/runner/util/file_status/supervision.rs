//! Spawning and draining supervised file tasks, progress forwarding, and
//! terminal-state fallback cleanup.
//!
//! The supervision layer ensures that once a spawned file task stops running,
//! the runner knows whether the file reached a terminal state. Panics, early
//! returns, and cancellations are caught and converted to explicit failures.
//!
//! # A file task belongs to its job, and a cancelled job stops its tasks
//!
//! Every file task runs inside the `RunScope` its runner established
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

use crate::api::DisplayPath;
use crate::runner::job_scope::{RunContext, SpawnScope};
use crate::scheduling::{AttemptOutcome, FailureCategory, RetryDisposition};

use super::tracker::set_file_progress;
use super::{FileTaskOutcome, RunnerEventSink};

// ---------------------------------------------------------------------------
// SpawnedFileTask: handle to one supervised file task
// ---------------------------------------------------------------------------

/// Handle to one spawned file task whose terminal file-state transition is
/// supervised by the runner rather than inferred later from a job-wide sweep.
pub struct SpawnedFileTask {
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
pub fn spawn_supervised_file_task<F>(
    filename: DisplayPath,
    role: &'static str,
    future: F,
) -> SpawnedFileTask
where
    F: Future<Output = FileTaskOutcome> + Send + 'static,
{
    let scope = RunContext::current();
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
pub async fn drain_supervised_file_tasks(
    sink: &dyn RunnerEventSink,
    cancel_token: &CancellationToken,
    tasks: Vec<SpawnedFileTask>,
) -> usize {
    let mut abnormal_exits = 0usize;
    for task in tasks {
        let joined = task.handle.await;
        if let Settled::Abnormal =
            settle(sink, cancel_token, &task.filename, task.role, joined).await
        {
            abnormal_exits += 1;
        }
    }
    abnormal_exits
}

/// How one joined file task left its file.
enum Settled {
    /// The task recorded the file's terminal state, or the supervisor stopped
    /// it for the job's cancellation (recorded here as the expected cancel).
    Expected,
    /// The task died without a terminal state; recorded here as a failure.
    Abnormal,
}

/// The one decision, for every runner, of what a joined file task means for
/// its file. A task that died without a terminal state is attributed to the
/// cancellation when the job was cancelled (an abort is how a cancelled task
/// ends), and to a panic otherwise.
async fn settle(
    sink: &dyn RunnerEventSink,
    cancel_token: &CancellationToken,
    filename: &DisplayPath,
    role: &'static str,
    joined: Result<FileTaskOutcome, tokio::task::JoinError>,
) -> Settled {
    let exit = match (joined, cancel_token.is_cancelled()) {
        (Ok(FileTaskOutcome::TerminalStateRecorded(_)), _) => return Settled::Expected,
        // The supervisor stopped it: an expected consequence of the cancel,
        // not an abnormal exit, but the file still needs its terminal state.
        (Ok(FileTaskOutcome::StoppedByCancellation), _) => {
            record_abnormal_file_task_exit(sink, filename.as_ref(), role, FileTaskExit::Cancelled)
                .await;
            return Settled::Expected;
        }
        (Err(_), true) => FileTaskExit::Cancelled,
        (Err(join_error), false) => FileTaskExit::Panicked(join_error.to_string()),
    };
    record_abnormal_file_task_exit(sink, filename.as_ref(), role, exit).await;
    Settled::Abnormal
}

// ---------------------------------------------------------------------------
// SupervisedFileTasks: a bounded set for runners that read as they spawn
// ---------------------------------------------------------------------------

/// Supervised file tasks joined in completion order, for a runner that keeps
/// at most N files in flight and so must wait for whichever finishes first
/// ([`drain_supervised_file_tasks`] joins in spawn order). Each entry keeps
/// its filename with its handle, so a task that dies is still settled against
/// its own file; dropping the set aborts every task still running.
pub struct SupervisedFileTasks {
    running: futures::stream::FuturesUnordered<
        futures::future::BoxFuture<'static, (SpawnedFileTask, JoinedOutcome)>,
    >,
}

type JoinedOutcome = Result<FileTaskOutcome, tokio::task::JoinError>;

impl SupervisedFileTasks {
    pub(crate) fn new() -> Self {
        Self {
            running: futures::stream::FuturesUnordered::new(),
        }
    }

    /// Tasks still running.
    pub(crate) fn len(&self) -> usize {
        self.running.len()
    }

    pub(crate) fn push(&mut self, mut task: SpawnedFileTask) {
        use futures::FutureExt;
        self.running.push(
            async move {
                // The handle stays in the task while it is awaited, so
                // dropping this future (the set) still aborts the work.
                let joined = (&mut task.handle).await;
                (task, joined)
            }
            .boxed(),
        );
    }

    /// Wait for the next task to end and settle its file (at once when the
    /// set is empty).
    pub(crate) async fn settle_next(
        &mut self,
        sink: &dyn RunnerEventSink,
        cancel_token: &CancellationToken,
    ) {
        use futures::StreamExt;
        if let Some((task, joined)) = self.running.next().await {
            settle(sink, cancel_token, &task.filename, task.role, joined).await;
        }
    }

    /// Wait for every task and settle each file.
    pub(crate) async fn drain(
        mut self,
        sink: &dyn RunnerEventSink,
        cancel_token: &CancellationToken,
    ) {
        use futures::StreamExt;
        while let Some((task, joined)) = self.running.next().await {
            settle(sink, cancel_token, &task.filename, task.role, joined).await;
        }
    }
}

// ---------------------------------------------------------------------------
// spawn_progress_forwarder: bridge progress channel to the event sink
// ---------------------------------------------------------------------------

/// Create a progress channel and spawn a forwarder task that routes updates
/// to the store for a specific `(job_id, filename)`.
///
/// Also returns the observer that records each of the file's pool waits (a
/// checkout on a saturated pool, a load queued for host memory) starting and
/// stopping (install it around the attempt with
/// `worker::pool::checkout_wait::observing_pool_waits`),
/// and the forwarder itself. The store keeps the reported stage and the open
/// waits as two facts and shows "waiting for a worker" while any wait is
/// open, so the forwarder publishes each as it comes, in order, and nothing
/// here restores a stage.
///
/// The forwarder runs until both the sender and the observer are dropped and
/// it has published everything they sent; await
/// [`FileProgressForwarder::finished`] after the attempt, before the file's
/// next stage is published, so no late update lands after it.
pub fn spawn_observed_progress_forwarder(
    sink: Arc<dyn RunnerEventSink>,
    filename: String,
) -> (
    super::ProgressSender,
    Arc<dyn crate::worker::pool::checkout_wait::PoolWaitObserver>,
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
                    Some(change) => sink.set_file_worker_wait(&filename, change).await,
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
pub struct FileProgressForwarder {
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
pub async fn observing_file_waits<F: std::future::Future>(
    sink: &Arc<dyn RunnerEventSink>,
    filenames: impl IntoIterator<Item = String>,
    work: F,
) -> F::Output {
    let (observers, forwarders): (Vec<_>, Vec<_>) = filenames
        .into_iter()
        .map(|filename| {
            // The work reports its own progress; only waits go through here.
            let (_progress, observer, forwarder) =
                spawn_observed_progress_forwarder(sink.clone(), filename);
            (observer, forwarder)
        })
        .unzip();
    let output = crate::worker::pool::checkout_wait::observing_pool_waits(
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
struct FannedOutWaits(Vec<Arc<dyn crate::worker::pool::checkout_wait::PoolWaitObserver>>);

impl crate::worker::pool::checkout_wait::PoolWaitObserver for FannedOutWaits {
    fn model_load(&self, change: crate::store::ModelLoadChange) {
        for observer in &self.0 {
            observer.model_load(change);
        }
    }

    fn waiting(&self, wait: crate::store::PoolWait) {
        for observer in &self.0 {
            observer.waiting(wait);
        }
    }

    fn resumed(&self, wait: crate::store::PoolWait) {
        for observer in &self.0 {
            observer.resumed(wait);
        }
    }
}

/// The observer half of an observed forwarder.
struct ForwardedWaits {
    events: tokio::sync::mpsc::UnboundedSender<crate::store::WorkerWaitChange>,
}

impl crate::worker::pool::checkout_wait::PoolWaitObserver for ForwardedWaits {
    fn model_load(&self, change: crate::store::ModelLoadChange) {
        let _ = self
            .events
            .send(crate::store::WorkerWaitChange::ModelLoad(change));
    }

    fn waiting(&self, wait: crate::store::PoolWait) {
        // A send fails only when the forwarder has stopped, which it does
        // only after this observer is dropped: unreachable while it is
        // called. The pool logs the wait regardless.
        let _ = self
            .events
            .send(crate::store::WorkerWaitChange::Started(wait));
    }

    fn resumed(&self, wait: crate::store::PoolWait) {
        // As for `waiting`.
        let _ = self
            .events
            .send(crate::store::WorkerWaitChange::Ended(wait));
    }
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
pub async fn record_file_cancelled_before_dispatch(sink: &dyn RunnerEventSink, filename: &str) {
    let finished_at = sink.now();
    let message = "dispatch cancelled before this file's task was ever started".to_owned();

    sink.finish_file_attempt(
        filename,
        AttemptOutcome::Cancelled,
        Some(FailureCategory::Cancelled),
        RetryDisposition::TerminalFailure,
        finished_at,
    )
    .await;

    sink.mark_file_error(filename, &message, FailureCategory::Cancelled, finished_at)
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
}

/// Record a non-standard file-task exit as an explicit terminal failure.
///
/// This path is only for supervision failures: a task panic or a task
/// cancellation.
async fn record_abnormal_file_task_exit(
    sink: &dyn RunnerEventSink,
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
    };

    sink.finish_file_attempt(
        filename,
        outcome,
        Some(category),
        RetryDisposition::TerminalFailure,
        finished_at,
    )
    .await;

    sink.mark_file_error(filename, &message, category, finished_at)
        .await;
}
