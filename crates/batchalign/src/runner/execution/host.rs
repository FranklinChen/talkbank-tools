//! Host capabilities determine which memory-admission transitions exist.

use std::convert::Infallible;

use async_trait::async_trait;
use tracing::warn;

use crate::api::NumWorkers;
use crate::error::ServerError;
use crate::host_memory::WaitExpired;
use crate::runner::context::{
    DirectExecutionHost, DispatchHostContext, ExecutionEngine, MemoryGateRejectionDisposition,
    ServerExecutionHost,
};
use crate::store::{DeferredRun, JobStore, RunnerJobSnapshot};

/// Capacity can defer an admitted run or discover it no longer needs work.
pub(super) enum CapacityRejection<Deferred> {
    Deferred(Deferred),
    NoLongerRunnable,
}

/// Internal, statically selected host behavior. Inline execution has no retry.
#[async_trait]
pub(super) trait ExecutionHost: Sync {
    type Deferred: Send;

    fn store(&self) -> &std::sync::Arc<JobStore>;
    fn engine(&self) -> &ExecutionEngine;

    async fn capacity_rejected(
        &self,
        context: &DispatchHostContext,
        job: &RunnerJobSnapshot,
        requested_workers: NumWorkers,
        error: WaitExpired,
    ) -> Result<CapacityRejection<Self::Deferred>, ServerError>;
}

#[async_trait]
impl ExecutionHost for DirectExecutionHost {
    type Deferred = Infallible;

    fn store(&self) -> &std::sync::Arc<JobStore> {
        &self.store
    }
    fn engine(&self) -> &ExecutionEngine {
        &self.engine
    }

    async fn capacity_rejected(
        &self,
        context: &DispatchHostContext,
        _job: &RunnerJobSnapshot,
        _requested_workers: NumWorkers,
        error: WaitExpired,
    ) -> Result<CapacityRejection<Infallible>, ServerError> {
        context.sink().bump_memory_gate_aborts().await;
        Err(ServerError::MemoryPressure(error.to_string()))
    }
}

#[async_trait]
impl ExecutionHost for ServerExecutionHost {
    type Deferred = DeferredRun;

    fn store(&self) -> &std::sync::Arc<JobStore> {
        &self.store
    }
    fn engine(&self) -> &ExecutionEngine {
        &self.engine
    }

    async fn capacity_rejected(
        &self,
        context: &DispatchHostContext,
        job: &RunnerJobSnapshot,
        requested_workers: NumWorkers,
        error: WaitExpired,
    ) -> Result<CapacityRejection<DeferredRun>, ServerError> {
        match self
            .orchestrator
            .handle_memory_gate_rejection(context.sink())
            .await?
        {
            MemoryGateRejectionDisposition::NoLongerRunnable => {
                Ok(CapacityRejection::NoLongerRunnable)
            }
            MemoryGateRejectionDisposition::Requeued(retry) => {
                warn!(
                    job_id = %job.identity.job_id,
                    correlation_id = %job.identity.correlation_id,
                    requested_workers = requested_workers.0,
                    error = %error,
                    retry_at = %retry.deadline(),
                    "Re-queueing job after host-memory capacity rejection"
                );
                Ok(CapacityRejection::Deferred(retry))
            }
        }
    }
}
