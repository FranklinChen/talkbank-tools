//! Per-key coordination for shared GPU worker creation.
//!
//! # Why this type exists
//!
//! The `gpu_workers` map used to hold `Arc<SharedGpuWorker>` directly, and
//! [`WorkerPool::get_or_create_gpu_worker`](super::WorkerPool::get_or_create_gpu_worker)
//! held the map's mutex across the entire slow path: a process-global spawn
//! semaphore, a cross-process host-memory lease, the Python process spawn, the
//! wait for `{"ready": true}` and a capabilities round trip. Holding the lock was
//! deliberate and prevented a real bug (two callers racing to spawn two worker
//! processes for one key), but it did so by making every OTHER user of the map
//! wait for an unrelated key's model load: dispatches whose worker was already
//! warm, and `/health`, which walks the same map. On a real host that is tens of
//! seconds of unrelated stalling per cold key.
//!
//! A slot moves the coordination from the MAP to the KEY. The map lock is held
//! only long enough to hand out a slot; the spawn runs with the lock released,
//! and duplicate spawns are still impossible because every caller for one key
//! shares one slot.
//!
//! # What this deliberately does NOT change
//!
//! Spawns still serialize globally. `memory_guard::acquire_spawn_permit` takes a
//! process-global semaphore with a single permit, held until the worker signals
//! ready, so that each spawn's memory check sees the previous worker's models
//! already resident. Per-key coordination is orthogonal to that: it stops work
//! that needs NO spawn from queuing behind one. Two cold keys still come up one
//! after the other, by design.
//!
//! # Failure is retryable, and eviction would be a bug
//!
//! A failed or cancelled initializer leaves the slot empty, so the next caller
//! can try again. A dead worker is retired before a new generation is spawned.
//! The former OnceCell retained its first successful worker forever, including
//! after the reader died, poisoning every later FA group. Removing the slot
//! from the map on failure would be actively wrong: another caller may already
//! hold a clone of that slot and be initializing through it, and its worker
//! would then exist with no map entry pointing at it, i.e. an orphaned worker
//! process that shutdown cannot find.

use std::sync::Arc;

use tokio::sync::{Mutex, MutexGuard};

use super::lock_recovered;
use super::shared_gpu::SharedGpuWorker;

#[derive(Default)]
enum Occupancy {
    #[default]
    Empty,
    Transitioning,
    Worker(Arc<SharedGpuWorker>),
}

#[derive(Default)]
struct SlotInner {
    occupancy: std::sync::Mutex<Occupancy>,
    transition: Mutex<()>,
}

/// One stable map entry coordinating all generations of a key's worker.
#[derive(Clone, Default)]
pub(in crate::worker) struct GpuWorkerSlot(Arc<SlotInner>);

/// Own the key's transition until a replacement is published or creation ends.
/// Dropping a cancelled/failed initializer restores Empty before unlocking.
struct SlotTransition<'a> {
    inner: &'a SlotInner,
    _permit: MutexGuard<'a, ()>,
}

impl SlotTransition<'_> {
    fn publish(self, worker: Arc<SharedGpuWorker>) {
        *lock_recovered(&self.inner.occupancy) = Occupancy::Worker(worker);
    }
}

impl Drop for SlotTransition<'_> {
    fn drop(&mut self) {
        let mut occupancy = lock_recovered(&self.inner.occupancy);
        if matches!(*occupancy, Occupancy::Transitioning) {
            *occupancy = Occupancy::Empty;
        }
    }
}

/// What a slot holds at the moment it is read.
///
/// An enum rather than an `Option` so that every reader of the map has to say
/// what an in-flight spawn means for IT. The two answers differ: a status
/// listing wants to report it, and a worker count must not count it.
pub(in crate::worker) enum GpuSlotState {
    /// No successful spawn is retained; the next caller can initialize it.
    Empty,
    /// A caller owns this key's spawn or retirement transition.
    Spawning,
    /// The retained worker currently accepts requests.
    Ready(Arc<SharedGpuWorker>),
    /// The retained worker stopped accepting requests and needs retirement.
    Unavailable,
}

impl GpuWorkerSlot {
    /// A slot with no worker yet. Inserted into the map by the caller that
    /// first asks for a key, before its spawn begins.
    pub(in crate::worker) fn pending() -> Self {
        Self::default()
    }

    /// Read the slot without waiting for an in-flight spawn.
    pub(in crate::worker) fn state(&self) -> GpuSlotState {
        let occupancy = lock_recovered(&self.0.occupancy);
        match &*occupancy {
            Occupancy::Empty => GpuSlotState::Empty,
            Occupancy::Transitioning => GpuSlotState::Spawning,
            Occupancy::Worker(worker) => match worker.check_available() {
                Ok(()) => GpuSlotState::Ready(worker.clone()),
                Err(_) => GpuSlotState::Unavailable,
            },
        }
    }

    /// Retained ownership for pool teardown, even if the process already died.
    /// An in-flight transition owns cleanup and observes pool cancellation.
    pub(in crate::worker) fn retained_worker(&self) -> Option<Arc<SharedGpuWorker>> {
        let occupancy = lock_recovered(&self.0.occupancy);
        match &*occupancy {
            Occupancy::Empty | Occupancy::Transitioning => None,
            Occupancy::Worker(worker) => Some(worker.clone()),
        }
    }

    /// Hand off a live worker, or retire the old generation and initialize one.
    ///
    /// The per-key lock spans retirement and initialization. Failure leaves
    /// Empty, allowing retry; concurrent callers cannot create duplicate
    /// replacement processes. The map lock is never held here.
    pub(in crate::worker) async fn worker_or_init<Init, Fut>(
        &self,
        init: Init,
    ) -> Result<Arc<SharedGpuWorker>, crate::worker::error::WorkerError>
    where
        Init: FnOnce() -> Fut,
        Fut: Future<Output = Result<Arc<SharedGpuWorker>, crate::worker::error::WorkerError>>,
    {
        if let GpuSlotState::Ready(worker) = self.state() {
            return Ok(worker);
        }
        let permit = self.0.transition.lock().await;
        // Another caller may have finished replacement while we waited.
        if let GpuSlotState::Ready(worker) = self.state() {
            return Ok(worker);
        }
        let previous = std::mem::replace(
            &mut *lock_recovered(&self.0.occupancy),
            Occupancy::Transitioning,
        );
        let transition = SlotTransition {
            inner: &self.0,
            _permit: permit,
        };
        if let Occupancy::Worker(worker) = previous {
            worker.shutdown().await;
        }
        let worker = init().await?;
        transition.publish(worker.clone());
        Ok(worker)
    }
}
