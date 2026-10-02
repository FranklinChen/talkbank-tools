//! Queue lease management for job dispatch.
//!
//! Lease methods control whether a job can be claimed by the local queue
//! dispatcher and manage the heartbeat/expiry lifecycle of active leases.
//! The local queue-claim path is currently test-only but the lease primitives
//! (`clear_lease`, `release_local_dispatch_claim`, `renew_local_dispatch_lease`)
//! are used by the production server during runner teardown and heartbeat
//! renewal.

use crate::api::{MachineTime, NodeId};
use crate::config::LeaseTtl;
use crate::scheduling::LeaseRecord;

use super::Job;

impl Job {
    /// Clear the current queue lease metadata.
    pub(crate) fn clear_lease(&mut self) {
        self.schedule.lease = None;
    }

    /// Return whether a live queue lease currently blocks local dispatch.
    ///
    /// Currently exercised only by the test-only local queue-claim path.
    #[cfg(test)]
    pub(crate) fn lease_blocks_local_dispatch(&self, now: MachineTime) -> bool {
        self.schedule
            .lease
            .as_ref()
            .is_some_and(|lease| lease.is_held_at(now))
    }

    /// Return the earliest time when the job should be reconsidered for dispatch.
    ///
    /// Currently exercised only by the test-only local queue-claim path.
    #[cfg(test)]
    pub(crate) fn next_local_dispatch_wake_at(&self, now: MachineTime) -> Option<MachineTime> {
        let mut wake_at = self
            .schedule
            .next_eligible_at
            .filter(|timestamp| *timestamp > now);
        if let Some(lease) = &self.schedule.lease
            && lease.is_held_at(now)
        {
            wake_at = Some(match wake_at {
                Some(next_eligible_at) => next_eligible_at.min(lease.expires_at()),
                None => lease.expires_at(),
            });
        }
        wake_at
    }

    /// Return whether the job can be claimed by the local queue dispatcher now.
    ///
    /// Currently exercised only by the test-only local queue-claim path.
    #[cfg(test)]
    pub(crate) fn ready_for_local_dispatch(&self, now: MachineTime) -> bool {
        self.execution.status == crate::api::JobStatus::Queued
            && !self.runtime.runner_active
            && !self.lease_blocks_local_dispatch(now)
            && self
                .schedule
                .next_eligible_at
                .is_none_or(|timestamp| timestamp <= now)
    }

    /// Claim the job for local dispatch and return the resulting lease record.
    ///
    /// Currently exercised only by the test-only local queue-claim path.
    #[cfg(test)]
    pub(crate) fn claim_for_local_dispatch(
        &mut self,
        node_id: &NodeId,
        now: MachineTime,
        lease_ttl: LeaseTtl,
    ) -> Option<LeaseRecord> {
        if !self.ready_for_local_dispatch(now) {
            return None;
        }

        self.runtime.runner_active = true;
        self.schedule.lease = Some(LeaseRecord::taken(node_id.clone(), now, lease_ttl));
        self.active_lease()
    }

    /// Release any local dispatch claim and clear the job's live lease.
    pub(crate) fn release_local_dispatch_claim(&mut self) {
        self.runtime.runner_active = false;
        self.clear_lease();
    }

    /// Renew the local dispatch lease when the current node still owns it.
    pub(crate) fn renew_local_dispatch_lease(
        &mut self,
        node_id: &NodeId,
        now: MachineTime,
        lease_ttl: LeaseTtl,
    ) -> Option<LeaseRecord> {
        if !self.runtime.runner_active || self.execution.status.is_terminal() {
            return None;
        }
        match &mut self.schedule.lease {
            Some(lease) if lease.leased_by_node() == node_id => {
                lease.renew(now, lease_ttl);
                Some(lease.clone())
            }
            Some(_) | None => None,
        }
    }

    /// The job's current lease, if a node holds one.
    pub(crate) fn active_lease(&self) -> Option<LeaseRecord> {
        self.schedule.lease.clone()
    }
}
