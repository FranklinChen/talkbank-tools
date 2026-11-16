//! Capacity belongs to an admitted worker from reservation through retirement.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::WorkerGroup;
mod queue;
use crate::worker::handle::WorkerHandle;
use crate::worker::tcp_handle::TcpWorkerHandle;
pub(super) use queue::IdleQueue;

/// Read-only capacity observations. Only this producer can acquire or refund.
#[derive(Debug, Clone)]
pub(super) struct GlobalCapacity {
    host: crate::host_facts::RuntimeHost,
    permits: Arc<Semaphore>,
    refunded: Arc<tokio::sync::Notify>,
    worker_changes: Arc<tokio::sync::Notify>,
}

impl GlobalCapacity {
    #[cfg(test)]
    pub(super) fn new(limit: usize) -> Self {
        Self::for_host(
            limit,
            crate::worker::handle::WorkerRuntimeConfig::for_tests().host,
        )
    }
    pub(super) fn for_host(limit: usize, host: crate::host_facts::RuntimeHost) -> Self {
        Self {
            host,
            permits: Arc::new(Semaphore::new(limit)),
            refunded: Arc::new(tokio::sync::Notify::new()),
            worker_changes: Arc::new(tokio::sync::Notify::new()),
        }
    }

    pub(super) fn host(&self) -> &crate::host_facts::RuntimeHost {
        &self.host
    }

    pub(super) fn available_permits(&self) -> usize {
        self.permits.available_permits()
    }

    pub(super) fn notifier(&self) -> Arc<tokio::sync::Notify> {
        self.worker_changes.clone()
    }

    pub(super) async fn refund_event(&self) {
        self.refunded.notified().await;
    }

    fn reserve(&self) -> Result<GlobalReservation, PermitRejected> {
        let permit = self
            .permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| PermitRejected)?;
        Ok(GlobalReservation {
            _permit: permit,
            _returned: CapacityReturned(self.refunded.clone()),
        })
    }
}

/// The count can be observed by consumers and changed only by a live slot.
#[derive(Debug, Default, Clone)]
pub(super) struct SlotAccounting(Arc<AtomicUsize>);

impl SlotAccounting {
    pub(super) fn count(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }
}

// The notification belongs to every actual reservation, including one whose
// later per-key admission is refused. Field order refunds before notification.
struct GlobalReservation {
    _permit: OwnedSemaphorePermit,
    _returned: CapacityReturned,
}

/// One original global reservation paired with its admitted group count.
struct WorkerSlot {
    mark: Arc<()>,
    total: SlotAccounting,
    _global: GlobalReservation,
    _changed: CapacityReturned,
}

impl Drop for WorkerSlot {
    fn drop(&mut self) {
        self.total.0.fetch_sub(1, Ordering::Relaxed);
    }
}

// GlobalReservation fields drop in declaration order after its group count
// owner, when present. Notify only after the permit is actually refunded.
struct CapacityReturned(Arc<tokio::sync::Notify>);

impl Drop for CapacityReturned {
    fn drop(&mut self) {
        self.0.notify_one();
    }
}

/// Reservation owns both counts and the pending startup's queue absence.
/// Publishing consumes it. Every other exit drops the original slot.
pub(super) struct ReservedWorkerSlot {
    slot: WorkerSlot,
    group: Arc<WorkerGroup>,
}

impl ReservedWorkerSlot {
    pub(super) fn try_claim(group: &Arc<WorkerGroup>, max: usize) -> Result<Self, ClaimRejected> {
        let permit = group
            .spawn_permits
            .reserve()
            .map_err(|_| ClaimRejected::Global)?;
        let total = &group.total.0;
        loop {
            let current = total.load(Ordering::Relaxed);
            if current >= max {
                return Err(ClaimRejected::PerKey);
            }
            if total
                .compare_exchange(current, current + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return Ok(Self::claimed(group, permit));
            }
        }
    }

    /// Registry adoption and health replacement preserve their global-only
    /// admission policy while retaining a real capacity owner.
    pub(super) fn try_adopt(group: &Arc<WorkerGroup>) -> Result<Self, PermitRejected> {
        let permit = group.spawn_permits.reserve()?;
        group.total.0.fetch_add(1, Ordering::Relaxed);
        Ok(Self::claimed(group, permit))
    }

    fn claimed(group: &Arc<WorkerGroup>, permit: GlobalReservation) -> Self {
        Self {
            slot: WorkerSlot {
                mark: Arc::new(()),
                total: group.total.clone(),
                _global: permit,
                _changed: CapacityReturned(group.worker_returned.clone()),
            },
            group: group.clone(),
        }
    }

    pub(super) fn started(self, handle: WorkerHandle) -> StartedWorker {
        StartedWorker {
            reservation: self,
            handle,
        }
    }

    pub(super) fn publish_stdio(self, handle: WorkerHandle) {
        let Self { slot, group } = self;
        group.idle.publish(PoolWorker {
            handle,
            _slot: slot,
        });
        group.worker_returned.notify_one();
    }

    pub(super) fn publish_tcp(self, handle: TcpWorkerHandle) {
        let Self { slot, group } = self;
        group.tcp_workers.publish(PoolWorker {
            handle,
            _slot: slot,
        });
        group.worker_returned.notify_one();
    }
}

/// The runtime's completed startup retains its actual handle and reservation
/// together until admission either refuses it or consumes publication.
pub(super) struct StartedWorker {
    handle: WorkerHandle,
    reservation: ReservedWorkerSlot,
}

impl StartedWorker {
    pub(super) fn publish(self) {
        let Self {
            reservation,
            handle,
        } = self;
        reservation.publish_stdio(handle);
    }
}

impl Deref for StartedWorker {
    type Target = WorkerHandle;
    fn deref(&self) -> &WorkerHandle {
        &self.handle
    }
}

impl DerefMut for StartedWorker {
    fn deref_mut(&mut self) -> &mut WorkerHandle {
        &mut self.handle
    }
}

/// A bare native handle cannot enter the pool. Its slot travels through every
/// queue, checkout, health check and retirement with that actual handle.
pub(super) struct PoolWorker<T> {
    handle: T,
    _slot: WorkerSlot,
}

/// Detached retirement retains queue-absence evidence on the same runtime
/// task as the actual worker and its original capacity.
pub(super) struct RetiringWorker {
    _worker: DepartedWorker<WorkerHandle>,
}

impl<T> Deref for PoolWorker<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.handle
    }
}

impl<T> DerefMut for PoolWorker<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.handle
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum ClaimRejected {
    Global,
    PerKey,
}

#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("global worker cap reached")]
pub(super) struct PermitRejected;

/// Queue removal carries the same actual worker, home and absence ownership.
/// An unfinished departure drops its native handle and original capacity.
pub(super) struct DepartedWorker<T> {
    worker: PoolWorker<T>,
    home: Arc<WorkerGroup>,
}
impl<T> DepartedWorker<T> {
    fn new(worker: PoolWorker<T>, home: &Arc<WorkerGroup>) -> Self {
        Self {
            worker,
            home: home.clone(),
        }
    }
}
impl<T> Deref for DepartedWorker<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.worker
    }
}
impl<T> DerefMut for DepartedWorker<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.worker
    }
}
impl DepartedWorker<WorkerHandle> {
    pub(super) fn return_idle(self) {
        let Self { worker, home } = self;
        home.idle.publish(worker);
        home.worker_returned.notify_one();
    }
    pub(super) fn retiring(self) -> RetiringWorker {
        RetiringWorker { _worker: self }
    }
}
impl DepartedWorker<TcpWorkerHandle> {
    pub(super) fn return_idle(self) {
        let Self { worker, home } = self;
        home.tcp_workers.publish(worker);
        home.worker_returned.notify_one();
    }
}

/// A sample retains the actual slot allocation, so PID reuse cannot select
/// another worker during the later queue-effect admission.
pub(super) struct ObservedIdleWorker {
    home: Arc<WorkerGroup>,
    mark: Arc<()>,
    pid: crate::worker::WorkerPid,
}
impl ObservedIdleWorker {
    pub(super) fn pid(&self) -> crate::worker::WorkerPid {
        self.pid
    }
    pub(super) fn take(self) -> Option<DepartedWorker<WorkerHandle>> {
        self.home.idle.take_marked(&self.home, &self.mark)
    }
}
impl WorkerGroup {
    pub(super) fn take_stdio(self: &Arc<Self>) -> Option<DepartedWorker<WorkerHandle>> {
        self.idle.take(self)
    }
    pub(super) fn take_tcp(self: &Arc<Self>) -> Option<DepartedWorker<TcpWorkerHandle>> {
        self.tcp_workers.take(self)
    }
    pub(super) fn drain_stdio(self: &Arc<Self>) -> Vec<DepartedWorker<WorkerHandle>> {
        self.idle.drain(self)
    }
    pub(super) fn observe_idle(self: &Arc<Self>) -> Vec<ObservedIdleWorker> {
        self.idle
            .view()
            .iter()
            .map(|worker| ObservedIdleWorker {
                home: self.clone(),
                mark: worker._slot.mark.clone(),
                pid: worker.pid(),
            })
            .collect()
    }
}
