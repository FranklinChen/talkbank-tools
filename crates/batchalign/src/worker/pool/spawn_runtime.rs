//! The Tokio runtime a pool's worker processes are bound to.
//!
//! # The defect this removes
//!
//! A `tokio::process::Child` registers its pipes with the reactor of the
//! runtime that CREATED it, and it stays bound to that reactor for its whole
//! life. Until now the pool spawned workers on whatever runtime happened to be
//! current at the call site, which is right in production (one runtime per
//! process, alive for the process's life) and wrong wherever a pool outlives
//! the runtime that first used it.
//!
//! The harness is exactly that case. `#[tokio::test]` builds a runtime per
//! test and drops it at the end of the test, while the shared live fixture
//! keeps ONE warmed [`super::WorkerPool`] for the whole binary. A test that
//! triggered the first spawn for a key left its worker registered with its own
//! reactor; when that test ended, the next test inherited a pooled worker whose
//! reactor was gone and failed with
//!
//! ```text
//! A Tokio 1.x context was found, but it is being shutdown.
//! ```
//!
//! which looked order-dependent and intermittent rather than structural. That
//! is `golden_coref_spa_passthrough` failing inside a shared session at
//! `ensure_command_capabilities` while passing alone.
//!
//! # The fix
//!
//! The pool captures the runtime it is CONSTRUCTED on and creates every child
//! process there, whoever calls it. A worker's reactor is then a property of
//! the pool rather than of whichever caller first needed a worker, so no caller
//! can bind one to a runtime shorter-lived than the pool.
//!
//! Only the calls that CREATE an OS resource go through here. Once a worker
//! exists, dispatching to it from another runtime is fine and stays direct: a
//! future awaiting that I/O is woken by the owning reactor, which is precisely
//! why this fix works at all.
//!
//! # Why an unbound pool is a real state, and what it may do
//!
//! A pool can legitimately be built with no runtime current: two unit tests do
//! it today, one to exercise capability admission and one as a sync proptest
//! helper. Neither ever spawns a worker, and neither could: without a reactor
//! there is nowhere for a child's pipes to register.
//!
//! So "no runtime" is not an error at construction, and construction does not
//! panic. It is an error only where it actually bites, at the one boundary that
//! creates a worker, and it surfaces there as a typed [`WorkerError`] naming the
//! cause. The alternatives were both worse: panicking at construction fails a
//! test that was never going to spawn anything, and falling back to an ambient
//! `tokio::spawn` would silently restore the defect this module exists to
//! remove.
//!
//! # Production is unaffected
//!
//! Production builds one pool inside one process-lifetime runtime
//! (`prepare_workers`, an `async fn`, is the only non-test constructor), so the
//! captured handle is always present and is the same runtime the old ambient
//! `tokio::spawn` would have found. The change is a no-op there and
//! load-bearing only where a pool outlives a runtime, which nothing in
//! production does.

use std::future::Future;
use std::sync::Arc;

use tokio::runtime::Handle;

use crate::worker::error::WorkerError;
use crate::worker::handle::{WorkerConfig, WorkerHandle};

use super::shared_gpu::SharedGpuWorker;

/// The runtime every one of this pool's child processes is registered with,
/// or nothing when the pool was built outside a runtime and therefore cannot
/// own a worker at all.
///
/// A newtype rather than a bare [`Handle`] so the pool cannot accidentally be
/// handed some other runtime's handle, and so the four things a pool is allowed
/// to do with its runtime are the only four things it CAN do: create a worker,
/// adopt one as a shared GPU worker, run a background loop, and retire a
/// handle. There is no accessor for the inner handle, so no caller can borrow
/// its way back to an ambient `tokio::spawn` for a worker.
#[derive(Clone)]
pub(super) struct PoolSpawnRuntime(Option<Handle>);

impl PoolSpawnRuntime {
    /// Capture the runtime the pool is being constructed on, if there is one.
    ///
    /// Never panics: see the module note on why an unbound pool is a real
    /// state. A pool that captured nothing refuses at the spawn boundary
    /// instead.
    pub(super) fn captured_at_construction() -> Self {
        Self(Handle::try_current().ok())
    }

    /// The refusal used by every worker-creating method, so all of them name
    /// the same cause in the same words.
    fn unbound(operation: &str) -> WorkerError {
        WorkerError::SpawnFailed(format!(
            "cannot {operation}: this WorkerPool was constructed outside a Tokio runtime, \
             so it has no reactor to register a worker's pipes with"
        ))
    }

    /// Create one worker process on the pool's runtime and wait for it to be
    /// ready.
    ///
    /// The spawn runs as a task ON the captured runtime, so the child's pipes,
    /// and the stderr-drain task `WorkerHandle::spawn` starts for it, both
    /// register with that runtime's reactor rather than the caller's.
    pub(super) async fn spawn_worker(
        &self,
        config: WorkerConfig,
    ) -> Result<WorkerHandle, WorkerError> {
        let handle = self.0.as_ref().ok_or_else(|| Self::unbound("spawn a worker"))?;
        handle
            .spawn(async move { WorkerHandle::spawn(config).await })
            .await
            .map_err(|error| {
                WorkerError::SpawnFailed(format!("worker spawn task did not complete: {error}"))
            })?
    }

    /// Convert one worker into a shared GPU worker on the pool's runtime.
    ///
    /// `SharedGpuWorker::from_handle` starts the background reader task that
    /// owns the worker's stdout for the rest of its life. That task has to live
    /// on the pool's runtime for the same reason the child does: a reader task
    /// on a per-test runtime stops being polled when that test ends, and every
    /// later dispatch to the worker then waits for a response nothing will ever
    /// read.
    pub(super) async fn adopt_shared_gpu_worker(
        &self,
        worker: WorkerHandle,
    ) -> Result<Arc<SharedGpuWorker>, WorkerError> {
        let handle = self
            .0
            .as_ref()
            .ok_or_else(|| Self::unbound("adopt a shared GPU worker"))?;
        handle
            .spawn(async move { Arc::new(SharedGpuWorker::from_handle(worker).await) })
            .await
            .map_err(|error| {
                WorkerError::SpawnFailed(format!(
                    "shared GPU worker adoption task did not complete: {error}"
                ))
            })
    }

    /// Run one long-lived background loop on the pool's runtime.
    ///
    /// `None` when the pool is unbound, which is not a failure: such a pool can
    /// own no workers, so there is nothing for a health-check loop to check.
    /// The caller gets an `Option` rather than a silently dropped task so that
    /// "no loop was started" is a fact it has to handle rather than one it can
    /// assume away.
    pub(super) fn spawn_background<F>(&self, future: F) -> Option<tokio::task::JoinHandle<F::Output>>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.0.as_ref().map(|handle| handle.spawn(future))
    }

    /// Retire one worker handle, whose `Drop` signals the child process.
    ///
    /// Detached onto the pool's runtime when there is one, so the signalling
    /// happens where that child's reactor lives rather than on whichever
    /// saturated checkout triggered the eviction. With no runtime there is no
    /// child to signal, and dropping inline is both correct and what
    /// `WorkerPool::drop` already does outside any runtime.
    ///
    /// The choice lives here rather than at the call site so there is one
    /// answer to "where does a retiring worker get dropped", not one per
    /// caller.
    pub(super) fn retire_worker_handle(&self, worker: WorkerHandle) {
        match self.0.as_ref() {
            Some(handle) => {
                handle.spawn(async move {
                    drop(worker);
                });
            }
            None => drop(worker),
        }
    }
}
