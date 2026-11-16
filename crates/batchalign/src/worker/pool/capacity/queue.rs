//! Queue contents are the sole availability authority.
use super::{DepartedWorker, PoolWorker, WorkerGroup};
use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::{Arc, Mutex, MutexGuard};

pub(in crate::worker::pool) struct IdleQueue<T> {
    workers: Mutex<VecDeque<PoolWorker<T>>>,
    changed: tokio::sync::Notify,
}

/// Read-only views cannot extract workers or manufacture availability.
pub(in crate::worker::pool) struct IdleView<'queue, T>(MutexGuard<'queue, VecDeque<PoolWorker<T>>>);
impl<T> Deref for IdleView<'_, T> {
    type Target = VecDeque<PoolWorker<T>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T> IdleQueue<T> {
    pub(in crate::worker::pool) fn new() -> Self {
        Self {
            workers: Mutex::new(VecDeque::new()),
            changed: tokio::sync::Notify::new(),
        }
    }
    pub(in crate::worker::pool) fn view(&self) -> IdleView<'_, T> {
        IdleView(super::super::lock_recovered(&self.workers))
    }
    pub(in crate::worker::pool) fn changed(&self) -> tokio::sync::futures::Notified<'_> {
        self.changed.notified()
    }
    // Only the capacity producer can insert or remove actual slot owners.
    pub(super) fn publish(&self, worker: PoolWorker<T>) {
        super::super::lock_recovered(&self.workers).push_back(worker);
        self.changed.notify_one();
    }
    pub(super) fn take(&self, home: &Arc<WorkerGroup>) -> Option<DepartedWorker<T>> {
        let mut workers = super::super::lock_recovered(&self.workers);
        workers
            .pop_front()
            .map(|worker| DepartedWorker::new(worker, home))
    }
    pub(super) fn drain(&self, home: &Arc<WorkerGroup>) -> Vec<DepartedWorker<T>> {
        super::super::lock_recovered(&self.workers)
            .drain(..)
            .map(|worker| DepartedWorker::new(worker, home))
            .collect()
    }
    pub(super) fn take_marked(
        &self,
        home: &Arc<WorkerGroup>,
        mark: &Arc<()>,
    ) -> Option<DepartedWorker<T>> {
        let mut workers = super::super::lock_recovered(&self.workers);
        let index = workers
            .iter()
            .position(|worker| Arc::ptr_eq(&worker._slot.mark, mark))?;
        workers
            .remove(index)
            .map(|worker| DepartedWorker::new(worker, home))
    }
}
