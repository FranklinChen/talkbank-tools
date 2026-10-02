//! Queue dispatch and lease management.

use crate::api::JobId;
#[cfg(test)]
use crate::store::registry::QueuePoll;

use super::db_helpers::LeaseWrite;

use super::super::JobStore;

/// Result of attempting to renew the local queue lease heartbeat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LeaseRenewalOutcome {
    /// The lease is still active and was renewed.
    Renewed,
    /// The lease should no longer be renewed.
    Stop,
}

impl JobStore {
    /// Return IDs of jobs that are in QUEUED status (for auto-resume on startup).
    pub(crate) async fn queued_job_ids(&self) -> Vec<JobId> {
        self.registry.queued_job_ids().await
    }

    /// Claim queued jobs that are eligible to run now.
    ///
    /// Currently exercised only by the local queue-claim tests.
    #[cfg(test)]
    pub(crate) async fn claim_ready_queued_jobs(&self) -> QueuePoll {
        let now = self.now();
        let claimed = self
            .registry
            .claim_ready_queued_jobs(now, self.node_id(), self.local_lease_ttl())
            .await;

        for claimed in &claimed.claimed_leases {
            self.db_persist_lease(&claimed.job_id, Some(&claimed.lease), LeaseWrite::Claim)
                .await;
        }

        claimed.poll
    }

    /// Claim exclusive runner ownership of one job (see
    /// [`JobRegistry::begin_runner`]). `None` means the job vanished.
    pub(crate) async fn begin_runner(
        &self,
        job_id: &JobId,
    ) -> Option<crate::store::BeginRunnerOutcome> {
        let now = self.now();
        let outcome = self
            .registry
            .begin_runner(job_id, self.node_id(), now, self.local_lease_ttl())
            .await;

        if let Some(crate::store::BeginRunnerOutcome::Started { lease, .. }) = &outcome {
            self.db_persist_lease(job_id, Some(lease), LeaseWrite::RunnerClaim)
                .await;
        }

        outcome
    }

    /// Most recent file activity for one job (stall-alarm input).
    pub(crate) async fn last_file_activity_at(
        &self,
        job_id: &JobId,
    ) -> Option<crate::api::MachineTime> {
        self.registry.last_file_activity_at(job_id).await
    }

    /// Return whether `generation` is still the job's current run generation.
    pub(crate) async fn runner_generation_current(
        &self,
        job_id: &JobId,
        generation: crate::store::RunGeneration,
    ) -> bool {
        self.registry
            .runner_generation_current(job_id, generation)
            .await
    }

    /// Release the runner claim so a queued job may be re-dispatched later.
    pub(crate) async fn release_runner_claim(&self, job_id: &JobId) {
        if !self.registry.release_runner_claim(job_id).await {
            return;
        }

        self.db_persist_lease(job_id, None, LeaseWrite::Release)
            .await;
    }

    /// Renew the local lease for a currently claimed job.
    ///
    /// Returns [`LeaseRenewalOutcome::Renewed`] while the local claim is still
    /// active, or [`LeaseRenewalOutcome::Stop`] when the heartbeat loop should exit.
    pub(crate) async fn renew_job_lease(&self, job_id: &JobId) -> LeaseRenewalOutcome {
        let now = self.now();
        let renewed_lease = self
            .registry
            .renew_job_lease(job_id, self.node_id(), now, self.local_lease_ttl())
            .await;

        match &renewed_lease {
            Some(lease) => {
                self.db_persist_lease(job_id, Some(lease), LeaseWrite::Renew)
                    .await;
                LeaseRenewalOutcome::Renewed
            }
            None => LeaseRenewalOutcome::Stop,
        }
    }
}
