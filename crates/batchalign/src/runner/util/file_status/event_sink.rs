//! Runner-owned boundary for publishing file/job lifecycle events.
//!
//! The execution engine reports what happened through the [`RunnerEventSink`]
//! trait instead of reaching into the concrete store implementation directly.
//! [`StoreRunnerEventSink`](crate::store::StoreRunnerEventSink) is the production implementation backed by
//! [`JobStore`](crate::store::JobStore).

use async_trait::async_trait;

use crate::api::MachineTime;
use crate::scheduling::{AttemptOutcome, FailureCategory, RetryDisposition, WorkUnitKind};
use crate::store::FileCompletion;

use super::FileStage;

/// Runner-owned boundary for publishing file/job lifecycle events.
///
/// The execution engine should report what happened through this sink instead
/// of reaching into the concrete store implementation directly.
#[async_trait]
pub(crate) trait RunnerEventSink: Send + Sync {
    /// Now, by the clock this sink records with, as the time of an event.
    /// Every event method below takes an [`EventTime`](crate::store::EventTime),
    /// which only the store's clock produces, so a caller cannot record an
    /// event at a time it chose.
    fn now(&self) -> crate::store::EventTime;
    async fn mark_file_processing(&self, filename: &str, started_at: crate::store::EventTime);
    async fn mark_file_done(
        &self,
        filename: &str,
        finished_at: crate::store::EventTime,
        completion: FileCompletion,
    );
    async fn mark_file_error(
        &self,
        filename: &str,
        error: &str,
        category: FailureCategory,
        finished_at: crate::store::EventTime,
    );
    async fn start_file_attempt(
        &self,
        filename: &str,
        work_unit_kind: WorkUnitKind,
        started_at: crate::store::EventTime,
    );
    async fn finish_file_attempt(
        &self,
        filename: &str,
        outcome: AttemptOutcome,
        failure_category: Option<FailureCategory>,
        disposition: RetryDisposition,
        finished_at: crate::store::EventTime,
    );
    async fn mark_file_retry_pending(
        &self,
        filename: &str,
        retry_at: MachineTime,
        category: FailureCategory,
        message: &str,
        finished_at: crate::store::EventTime,
    );
    async fn clear_file_retry_state(&self, filename: &str);
    async fn set_file_progress(
        &self,
        filename: &str,
        stage: FileStage,
        current: Option<i64>,
        total: Option<i64>,
    );
    /// One of the file's worker checkouts started or stopped waiting on a
    /// saturated pool. A fact of its own, beside the reported stage: the
    /// file shows "waiting for a worker" while any wait is open.
    async fn set_file_worker_wait(&self, filename: &str, change: crate::store::WorkerWaitChange);
    /// The run has its memory and starts processing: the job is running.
    async fn mark_job_running(&self);
    async fn record_job_worker_count(&self, worker_count: usize);
    async fn requeue_job_after_memory_gate(
        &self,
        retry_at: MachineTime,
    ) -> crate::store::MemoryRequeueOutcome;
    async fn bump_memory_gate_aborts(&self);
}
