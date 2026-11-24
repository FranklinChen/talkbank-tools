//! Claimed inputs become dispatchable only while both resource leases are owned.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::OwnedSemaphorePermit;
use tracing::info;

use crate::host_memory::{BoundedRefusal, JobExecutionPlan, LeaseFailure};
use crate::store::{JobStore, RunnerJobSnapshot};

use super::host::{CapacityRejection, ExecutionHost};
use crate::runner::context::{
    ClaimedDispatch, DispatchEnd, DispatchHostContext, JobDispatchRequest,
};
use crate::runner::preflight::ObservedPreflight;
use crate::runner::util::compute_job_workers;

/// Cancellation belongs to `RunScope`, so execution itself cannot report it.
#[derive(Debug)]
pub(super) enum ExecutionEnd<Deferred> {
    Dispatched,
    Deferred(Deferred),
    NoLongerRunnable,
    Refused(crate::runner::context::JobRefusal),
}

/// Created only alongside the owning `RunScope`, from the same claimed snapshot.
pub(super) struct ExecutionPreparation<'host, H: ExecutionHost> {
    host: &'host H,
    store: Arc<JobStore>,
    run: ClaimedDispatch,
}

/// A successful reservation retains its plan and semaphore until dispatch ends.
/// No caller can construct this value or dispatch the earlier preparation.
struct PreparedExecution<'host, H: ExecutionHost> {
    preparation: ExecutionPreparation<'host, H>,
    slot: OwnedSemaphorePermit,
    plan: JobExecutionPlan,
}

/// A capacity timeout can defer only hosts whose associated type permits it.
enum PreparationOutcome<'host, H: ExecutionHost> {
    Ready(PreparedExecution<'host, H>),
    Deferred(H::Deferred),
    NoLongerRunnable,
}

/// Host-memory reservation failures separated from job execution failures.
enum ExecutionReservationError {
    Capacity {
        requested_workers: crate::api::NumWorkers,
        error: crate::host_memory::WaitExpired,
    },
    Fatal(crate::error::ServerError),
}

impl<'host, H: ExecutionHost> ExecutionPreparation<'host, H> {
    pub(super) fn for_claim(host: &'host H, claim: &crate::store::AdmittedRun) -> Self {
        Self {
            run: ClaimedDispatch::for_run(claim),
            store: Arc::clone(claim.store()),
            host,
        }
    }

    /// The consuming path from the claim's inputs to dispatch, under `RunScope`.
    pub(super) async fn execute(
        self,
    ) -> Result<ExecutionEnd<H::Deferred>, crate::error::ServerError> {
        match self.reserve().await? {
            PreparationOutcome::Ready(prepared) => prepared.dispatch().await,
            PreparationOutcome::Deferred(retry) => Ok(ExecutionEnd::Deferred(retry)),
            PreparationOutcome::NoLongerRunnable => Ok(ExecutionEnd::NoLongerRunnable),
        }
    }

    async fn reserve(self) -> Result<PreparationOutcome<'host, H>, crate::error::ServerError> {
        let slot = self.store.acquire_job_slot().await.map_err(|error| {
            crate::error::ServerError::Persistence(format!("job admission closed: {error}"))
        })?;
        match reserve_job_execution(self.run.host(), self.run.job()).await {
            Ok(plan) => Ok(PreparationOutcome::Ready(PreparedExecution {
                preparation: self,
                slot,
                plan,
            })),
            Err(ExecutionReservationError::Capacity {
                requested_workers,
                error,
            }) => {
                let retry = self
                    .host
                    .capacity_rejected(self.run.host(), self.run.job(), requested_workers, error)
                    .await?;
                Ok(match retry {
                    CapacityRejection::Deferred(retry) => PreparationOutcome::Deferred(retry),
                    CapacityRejection::NoLongerRunnable => PreparationOutcome::NoLongerRunnable,
                })
            }
            Err(ExecutionReservationError::Fatal(error)) => Err(error),
        }
    }
}

impl<H: ExecutionHost> PreparedExecution<'_, H> {
    async fn dispatch(self) -> Result<ExecutionEnd<H::Deferred>, crate::error::ServerError> {
        let Self {
            preparation,
            slot: _slot,
            plan,
        } = self;
        let ExecutionPreparation {
            run,
            host,
            store: _store,
        } = preparation;
        let job = run.job();
        let context = run.host();
        let num_workers = plan.granted_workers;
        let _job_memory_lease = plan.lease;
        let sink = context.sink();

        // Only resource-admitted execution may become running or reach dispatch.
        sink.mark_job_running().await;
        sink.record_job_worker_count(num_workers.0).await;
        info!(
            job_id = %job.identity.job_id,
            correlation_id = %job.identity.correlation_id,
            num_files = job.pending_files.len(),
            requested_workers = plan.requested_workers.0,
            num_workers = num_workers.0,
            reserved_mb = plan.reserved_mb.0,
            command = %job.dispatch.command,
            "Processing files"
        );

        let preflight = ObservedPreflight::observe(run).await.record().await;
        // Worker warming remains plan-specific in routing, after admission.
        let end = host
            .engine()
            .dispatch_job(JobDispatchRequest {
                preflight,
                num_workers,
            })
            .await?;
        Ok(match end {
            DispatchEnd::Dispatched => ExecutionEnd::Dispatched,
            DispatchEnd::Refused(refusal) => ExecutionEnd::Refused(refusal),
        })
    }
}

async fn reserve_job_execution(
    host: &DispatchHostContext,
    job: &RunnerJobSnapshot,
) -> Result<JobExecutionPlan, ExecutionReservationError> {
    let job_id = &job.identity.job_id;
    let command = job.dispatch.command;
    let requested_workers =
        compute_job_workers(command, job.pending_files.len(), host.host_snapshot());
    let coordinator = host.memory_coordinator();
    let job_label = format!(
        "job-execution:{}:{}:{}",
        job_id,
        command,
        job.dispatch.lang.to_worker_language()
    );
    let timeout = host.config().memory_gate_timeout_s.duration();
    let poll_interval = Duration::from_secs(host.config().memory_gate_poll_s.get());
    let plan = coordinator
        .wait_for_job_execution_plan(
            command,
            requested_workers,
            &job_label,
            timeout,
            poll_interval,
        )
        .await;

    match plan {
        Ok(plan) => Ok(plan),
        // Congestion that outlasted the bounded wait: requeue and try later.
        Err(LeaseFailure::Refused(BoundedRefusal::TimedOut(error))) => {
            Err(ExecutionReservationError::Capacity {
                requested_workers,
                error,
            })
        }
        // A plan that can never fit, or a host that cannot be used: waiting
        // would requeue for ever, so the job fails with the reason.
        Err(LeaseFailure::Refused(BoundedRefusal::Final(refusal))) => Err(
            ExecutionReservationError::Fatal(crate::error::ServerError::Persistence(format!(
                "host memory refused job {job_id}: {refusal}"
            ))),
        ),
        Err(LeaseFailure::AttemptDidNotFinish(error)) => Err(ExecutionReservationError::Fatal(
            crate::error::ServerError::Persistence(format!(
                "host-memory planner task failed for job {job_id}: {error}"
            )),
        )),
    }
}
