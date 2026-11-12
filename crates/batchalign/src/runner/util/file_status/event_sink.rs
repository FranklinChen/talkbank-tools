//! Runner-owned boundary for publishing file/job lifecycle events.
//!
//! The execution engine reports what happened through the [`RunnerEventSink`]
//! trait instead of reaching into the concrete store implementation directly.
//! [`StoreRunnerEventSink`] is the production implementation backed by
//! [`JobStore`].

use std::sync::Arc;

use async_trait::async_trait;

use crate::api::MachineTime;
use crate::scheduling::{AttemptOutcome, FailureCategory, RetryDisposition, WorkUnitKind};
use crate::store::{
    AttemptFinishRecord, FileCompletion, FileFailureRecord, FileProgressRecord, FileRetryRecord,
    JobStore,
};

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
    async fn requeue_job_after_memory_gate(&self, retry_at: MachineTime);
    async fn bump_deferred_work_units(&self);
    async fn bump_memory_gate_aborts(&self);
}

/// Store-backed implementation of the runner event sink, bound to one run.
///
/// Every write it makes carries its [`RunBinding`](crate::store::RunBinding),
/// so a caller names neither the job nor the run, and the store
/// refuses a write from a run that is no longer the job's current one. So a
/// task of a cancelled or restarted run, still finishing a step when it was
/// stopped, cannot write onto the files of the run that replaced it.
#[derive(Clone)]
pub(crate) struct StoreRunnerEventSink {
    store: Arc<JobStore>,
    run: crate::store::RunBinding,
}

impl StoreRunnerEventSink {
    /// The sink of the run that `claim` owns: the only production route.
    ///
    /// Not `new`: the return type is the trait object, and clippy's
    /// `new_ret_no_self` rule prefers another name for such factories.
    pub(crate) fn for_run(
        store: Arc<JobStore>,
        claim: &crate::store::RunnerClaim,
    ) -> Arc<dyn RunnerEventSink> {
        Arc::new(Self {
            store,
            run: claim.binding().clone(),
        })
    }

    /// A sink for a test job in its first run, with no claim to bind to.
    #[cfg(test)]
    pub(crate) fn wrap(
        store: Arc<JobStore>,
        job_id: &crate::api::JobId,
    ) -> Arc<dyn RunnerEventSink> {
        Arc::new(Self {
            store,
            run: crate::store::RunBinding::first_run_of(job_id.clone()),
        })
    }
}

#[async_trait]
impl RunnerEventSink for StoreRunnerEventSink {
    fn now(&self) -> crate::store::EventTime {
        self.store.event_time()
    }

    async fn mark_file_processing(&self, filename: &str, started_at: crate::store::EventTime) {
        self.store
            .mark_file_processing(&self.run, filename, started_at)
            .await;
    }

    async fn mark_file_done(
        &self,
        filename: &str,
        finished_at: crate::store::EventTime,
        completion: FileCompletion,
    ) {
        self.store
            .mark_file_done(&self.run, filename, finished_at, completion)
            .await;
    }

    async fn mark_file_error(
        &self,
        filename: &str,
        error: &str,
        category: FailureCategory,
        finished_at: crate::store::EventTime,
    ) {
        self.store
            .mark_file_error(
                &self.run,
                filename,
                &FileFailureRecord {
                    message: error.to_string(),
                    category,
                    finished_at,
                },
            )
            .await;
    }

    async fn start_file_attempt(
        &self,
        filename: &str,
        work_unit_kind: WorkUnitKind,
        started_at: crate::store::EventTime,
    ) {
        self.store
            .start_file_attempt(&self.run, filename, work_unit_kind, started_at)
            .await;
    }

    async fn finish_file_attempt(
        &self,
        filename: &str,
        outcome: AttemptOutcome,
        failure_category: Option<FailureCategory>,
        disposition: RetryDisposition,
        finished_at: crate::store::EventTime,
    ) {
        self.store
            .db_finish_attempt_for_file(
                &self.run,
                AttemptFinishRecord {
                    filename,
                    outcome,
                    failure_category,
                    disposition,
                    finished_at,
                },
            )
            .await;
    }

    async fn mark_file_retry_pending(
        &self,
        filename: &str,
        retry_at: MachineTime,
        category: FailureCategory,
        message: &str,
        finished_at: crate::store::EventTime,
    ) {
        self.store
            .mark_file_retry_pending(
                &self.run,
                filename,
                &FileRetryRecord {
                    message: message.to_string(),
                    category,
                    finished_at,
                    retry_at,
                },
            )
            .await;
    }

    async fn clear_file_retry_state(&self, filename: &str) {
        self.store.clear_file_retry_state(&self.run, filename).await;
    }

    async fn set_file_progress(
        &self,
        filename: &str,
        stage: FileStage,
        current: Option<i64>,
        total: Option<i64>,
    ) {
        self.store
            .set_file_progress(
                &self.run,
                filename,
                &FileProgressRecord {
                    stage: stage.api_stage(),
                    current,
                    total,
                },
            )
            .await;
    }

    async fn set_file_worker_wait(&self, filename: &str, change: crate::store::WorkerWaitChange) {
        self.store
            .set_file_worker_wait(&self.run, filename, change)
            .await;
    }

    async fn mark_job_running(&self) {
        self.store.mark_job_running(&self.run).await;
    }

    async fn record_job_worker_count(&self, worker_count: usize) {
        self.store
            .record_job_worker_count(&self.run, worker_count)
            .await;
    }

    async fn requeue_job_after_memory_gate(&self, retry_at: MachineTime) {
        self.store
            .requeue_job_after_memory_gate(&self.run, retry_at)
            .await;
    }

    async fn bump_deferred_work_units(&self) {
        self.store
            .bump_counter(|c| c.deferred_work_units += 1)
            .await;
    }

    async fn bump_memory_gate_aborts(&self) {
        self.store.bump_counter(|c| c.memory_gate_aborts += 1).await;
    }
}
