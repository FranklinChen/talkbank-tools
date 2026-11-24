//! Exclusive run ownership, generations and consuming completion outcomes.
use super::{Job, RunnerJobSnapshot};
use crate::api::{JobId, JobStatus};

/// Monotonic per-job run generation.
///
/// Every restart (`Job::prepare_for_restart`) increments it, so a runner
/// that began under an earlier generation can be recognized as STALE and
/// barred from finalizing the job or force-failing files that now belong
/// to the restarted run. Completion admits this generation and its file
/// cleanup in one registry transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunGeneration(u64);

impl RunGeneration {
    /// Generation of a job's first run, before any restart.
    pub(in crate::store) const FIRST: Self = Self(0);

    /// The next generation (used by restart).
    pub(crate) const fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

impl std::fmt::Display for RunGeneration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "g{}", self.0)
    }
}

/// One run of one job: the identity every write from that run carries.
///
/// A job id alone cannot say whose write it is, because a restart reuses
/// the id under a new [`RunGeneration`]; a generation alone names no job.
/// The pair is minted only from a [`RunnerClaim`] (and, in tests, for a
/// job's first run), so no caller can assemble a binding from a job id and
/// a generation it picked. The store admits a write only while the bound
/// generation is still the job's current one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::store) struct RunBinding {
    job_id: JobId,
    generation: RunGeneration,
}

impl RunBinding {
    /// The job this run belongs to.
    pub(crate) const fn job_id(&self) -> &JobId {
        &self.job_id
    }

    /// Whether this run is still `job`'s current one. The only reading of
    /// the generation: the store asks this, and nothing compares raw
    /// generations.
    pub(in crate::store) fn is_current_for(&self, job: &super::Job) -> bool {
        job.runtime.run_generation == self.generation
    }

    /// The binding of a test job's first run, which has no claim to bind to.
    #[cfg(test)]
    pub(crate) const fn first_run_of(job_id: JobId) -> Self {
        Self {
            job_id,
            generation: RunGeneration::FIRST,
        }
    }
}

impl std::fmt::Display for RunBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.job_id, self.generation)
    }
}

/// Outcome of a runner's attempt to claim exclusive execution of a job.
#[derive(Debug)]
pub(in crate::store) enum BeginRunnerOutcome {
    /// The claim succeeded; the runner owns the job at this generation,
    /// under the lease the claim took (the store persists exactly that one).
    Started {
        /// Linear authority binding this run's generation and snapshot.
        claim: RunnerClaim,
        /// The lease taken with the claim.
        lease: crate::scheduling::LeaseRecord,
    },
    /// Another runner still owns the job (e.g. a restarted job whose
    /// previous runner is mid-teardown). The caller must wait and retry.
    RunnerStillLive,
    /// Cancellation arrived before a runner claimed the job. This owner
    /// can close unfinished files but cannot enter admission or dispatch.
    Stopped { completion: StoppedRun },
    /// A delayed or competing task arrived after this job stopped. It owns
    /// no claim, heartbeat, resources or release authority.
    NotRunnable { status: JobStatus },
}

/// Exclusive authority for one run, minted together with its immutable input.
///
/// Not cloneable: completion consumes it. A restart can supersede the claim,
/// but cannot give its cleanup authority over the restarted files.
#[derive(Debug)]
#[must_use = "a runner claim owns one attempt's completion"]
pub(in crate::store) struct RunnerClaim {
    run: RunBinding,
    snapshot: std::sync::Arc<RunnerJobSnapshot>,
}

/// Completion-only authority for cancellation accepted before dispatch.
#[derive(Debug)]
#[must_use = "stopped files still require terminal bookkeeping"]
pub(in crate::store) struct StoppedRun(RunnerClaim);

impl StoppedRun {
    pub(in crate::store) fn for_job(job: &Job) -> Option<Self> {
        RunnerClaim::for_job(job).map(Self)
    }

    pub(in crate::store) fn into_claim(self) -> RunnerClaim {
        self.0
    }
}

impl RunnerClaim {
    /// The claim for `job`'s current run: `None` when this process has not
    /// admitted the job's inputs, which then cannot run.
    pub(in crate::store) fn for_job(job: &Job) -> Option<Self> {
        Some(Self {
            run: RunBinding {
                job_id: job.identity.job_id.clone(),
                generation: job.runtime.run_generation,
            },
            snapshot: std::sync::Arc::new(job.to_runner_snapshot()?),
        })
    }

    /// Immutable dispatch input bound to the claimed generation.
    pub(crate) const fn snapshot(&self) -> &std::sync::Arc<RunnerJobSnapshot> {
        &self.snapshot
    }

    /// The identity every write of this run carries.
    pub(crate) const fn binding(&self) -> &RunBinding {
        &self.run
    }
}

/// Result of consuming a run's completion authority.
#[derive(Debug)]
pub(crate) enum RunCompletion {
    /// Restart or shutdown superseded this run; no old cleanup was applied.
    Superseded,
    /// Cleanup and final status were admitted together by the registry.
    Finished {
        /// Actual status, including a previously accepted stop.
        status: JobStatus,
        /// Aggregated failure reason for logging.
        error: Option<String>,
    },
}
