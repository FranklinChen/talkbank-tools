//! Invocation-owned registrations retain shared workers until every use ends.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use dashmap::DashMap;

use crate::api::JobId;
use crate::worker::WorkerPid;

/// Identity belongs to a live allocation held by both its map entry and lease.
/// Detach releases the map's reference. Older leases retain their own identity,
/// so releasing one cannot remove a later invocation of the same worker.
#[derive(Debug, Clone)]
struct InvocationIdentity(Arc<()>);

impl PartialEq for InvocationIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for InvocationIdentity {}

impl Hash for InvocationIdentity {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::hash(Arc::as_ptr(&self.0), state);
    }
}

/// The map records invocation ownership. PID deduplication happens only when
/// projecting cancellation targets, so two users of one process stay distinct.
#[derive(Debug, Clone, Default)]
pub(crate) struct JobWorkerTracker {
    by_job: Arc<DashMap<JobId, Mutex<HashMap<InvocationIdentity, WorkerPid>>>>,
}

impl JobWorkerTracker {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn register(&self, job_id: JobId, pid: WorkerPid) -> JobRegistration {
        let identity = InvocationIdentity(Arc::new(()));
        self.by_job
            .entry(job_id.clone())
            .or_default()
            .value()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(identity.clone(), pid);
        JobRegistration {
            tracker: self.clone(),
            job_id,
            identity,
        }
    }

    fn release(&self, job_id: &JobId, identity: &InvocationIdentity) {
        let now_empty = {
            let Some(entry) = self.by_job.get(job_id) else {
                return;
            };
            let mut invocations = entry
                .value()
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            invocations.remove(identity);
            invocations.is_empty()
        };
        if now_empty {
            self.by_job.remove_if(job_id, |_, invocations| {
                invocations
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .is_empty()
            });
        }
    }

    /// Detach current invocations and project each active worker exactly once.
    pub(crate) fn drain(&self, job_id: &JobId) -> Vec<WorkerPid> {
        match self.by_job.remove(job_id) {
            Some((_, invocations)) => distinct_pids(
                invocations
                    .into_inner()
                    .unwrap_or_else(|error| error.into_inner())
                    .into_values(),
            ),
            None => Vec::new(),
        }
    }

    /// Observe worker targets while retaining every invocation's own lease.
    pub(crate) fn snapshot(&self, job_id: &JobId) -> Vec<WorkerPid> {
        match self.by_job.get(job_id) {
            Some(entry) => distinct_pids(
                entry
                    .value()
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .values()
                    .copied(),
            ),
            None => Vec::new(),
        }
    }
}

fn distinct_pids(invocations: impl Iterator<Item = WorkerPid>) -> Vec<WorkerPid> {
    invocations.collect::<HashSet<_>>().into_iter().collect()
}

/// A unique registration can only end through ownership drop or cancellation
/// detach. The tracker and invocation identity stay paired behind this module.
struct JobRegistration {
    tracker: JobWorkerTracker,
    job_id: JobId,
    identity: InvocationIdentity,
}

impl Drop for JobRegistration {
    fn drop(&mut self) {
        self.tracker.release(&self.job_id, &self.identity);
    }
}

/// A worker use inherits its actual task-local job. Unscoped work has no
/// registration payload. Keeping this guard owns registration until completion,
/// cancellation or unwinding, independently of other uses of the same process.
#[must_use = "guard must be held for the dispatch lifetime"]
pub(crate) struct TrackerGuard {
    _registration: Option<JobRegistration>,
}

impl TrackerGuard {
    pub(crate) fn new(tracker: &JobWorkerTracker, pid: WorkerPid) -> Self {
        Self {
            _registration: super::current_job_id().map(|job| tracker.register(job, pid)),
        }
    }
}
