//! Store-bound run capabilities returned after registry and lease admission.

use super::{
    BeginRunnerOutcome, JobStore, RunCompletion, RunnerClaim, RunnerJobSnapshot, StoppedRun,
};
use std::sync::Arc;

/// The actual store and its exclusive registry claim are one non-cloneable owner.
pub struct AdmittedRun {
    store: Arc<JobStore>,
    claim: RunnerClaim,
}

/// Completion-only authority when cancellation preceded execution admission.
pub struct StoppedAdmission {
    store: Arc<JobStore>,
    stopped: StoppedRun,
}

pub enum RunnerAdmission {
    Started(AdmittedRun),
    Stopped(StoppedAdmission),
    RunnerStillLive,
    NotRunnable { status: crate::api::JobStatus },
}

impl JobStore {
    /// Only this store's registry observation can mint a capability bound to it.
    pub(crate) async fn admit_runner(
        self: &Arc<Self>,
        id: &crate::api::JobId,
    ) -> Option<RunnerAdmission> {
        self.begin_runner(id).await.map(|outcome| match outcome {
            BeginRunnerOutcome::Started { claim, .. } => RunnerAdmission::Started(AdmittedRun {
                store: Arc::clone(self),
                claim,
            }),
            BeginRunnerOutcome::Stopped { completion } => {
                RunnerAdmission::Stopped(StoppedAdmission {
                    store: Arc::clone(self),
                    stopped: completion,
                })
            }
            BeginRunnerOutcome::RunnerStillLive => RunnerAdmission::RunnerStillLive,
            BeginRunnerOutcome::NotRunnable { status } => RunnerAdmission::NotRunnable { status },
        })
    }
}

impl AdmittedRun {
    pub(crate) const fn snapshot(&self) -> &Arc<RunnerJobSnapshot> {
        self.claim.snapshot()
    }
    pub(crate) const fn store(&self) -> &Arc<JobStore> {
        &self.store
    }
    pub(super) const fn binding(&self) -> &super::RunBinding {
        self.claim.binding()
    }

    pub(crate) async fn complete(self) -> Result<RunCompletion, crate::error::ServerError> {
        self.store
            .complete_run(self.claim, self.store.event_time())
            .await
    }
    pub(crate) async fn fail(
        self,
        reason: String,
    ) -> Result<RunCompletion, crate::error::ServerError> {
        self.store
            .fail_run(self.claim, reason, self.store.event_time())
            .await
    }
    pub(crate) async fn release(self) {
        self.store.release_run(self.claim).await;
    }
}

impl StoppedAdmission {
    pub(crate) async fn complete(self) -> Result<RunCompletion, crate::error::ServerError> {
        self.store
            .complete_stopped_run(self.stopped, self.store.event_time())
            .await
    }
}
