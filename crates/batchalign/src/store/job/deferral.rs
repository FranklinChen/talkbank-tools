//! The result of the store's current-run memory-requeue admission.

use crate::api::MachineTime;

/// Evidence that the registry accepted a retry deadline for this observation.
/// Persistence is attempted by the store before this value leaves it. Later
/// cancellation or supersession still requires the normal store observations.
#[derive(Debug)]
pub(crate) struct DeferredRun {
    retry_at: MachineTime,
}

impl DeferredRun {
    pub(in crate::store) fn admitted(retry_at: MachineTime) -> Self {
        Self { retry_at }
    }

    pub(crate) fn deadline(&self) -> MachineTime {
        self.retry_at
    }

    pub(crate) fn into_deadline(self) -> MachineTime {
        self.retry_at
    }
}

/// A rejected requeue carries no authority to schedule replacement work.
#[derive(Debug)]
#[must_use = "only an admitted requeue can publish a deferred retry"]
pub(crate) enum MemoryRequeueOutcome {
    Requeued(DeferredRun),
    NoLongerRunnable,
}
