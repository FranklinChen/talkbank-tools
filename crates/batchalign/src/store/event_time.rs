//! The instant a job or file event is recorded at.
//!
//! [`EventTime`] is a [`MachineTime`] that only the store's clock can produce
//! ([`JobStore::event_time`](super::JobStore::event_time)). Every runner
//! event (a file starting, finishing, failing or scheduled for retry, a job
//! finishing or failing) and a submission's time take one, so a pipeline
//! cannot record an event at a time it chose: the runner event sink and the
//! store's event methods accept nothing else. A deadline derived from an
//! event, such as a retry's eligibility, is computed from it by
//! [`EventTime::deadline_after`], and is a plain `MachineTime` because it
//! names a future instant, not an event.

use crate::api::MachineTime;

/// An instant read from the store's clock to record an event at.
///
/// No code outside the store can make one from a time it holds:
///
/// ```compile_fail
/// use batchalign::api::MachineTime;
/// use batchalign::store::EventTime;
/// let _ = EventTime(MachineTime::now());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct EventTime(MachineTime);

impl EventTime {
    /// Read by the store from its clock. The one production constructor.
    pub(super) fn from_store_clock(now: MachineTime) -> Self {
        Self(now)
    }

    /// A fixed instant for a test that drives the store or a sink directly.
    #[cfg(test)]
    pub(crate) fn fixed(at: MachineTime) -> Self {
        Self(at)
    }

    /// The instant, for storing or comparing.
    pub fn instant(self) -> MachineTime {
        self.0
    }

    /// The instant `delay` after this event: the one place a deadline is
    /// derived from an event (a file's retry, a job's requeue).
    pub fn deadline_after(self, delay: std::time::Duration) -> MachineTime {
        self.0.plus(delay)
    }
}
