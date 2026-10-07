//! Shared recording [`RunnerEventSink`] for tests.
//!
//! One mock, not one per test module: the sink has ~20 methods, and a second
//! hand-written copy is a second thing to forget to update when the trait grows.
//! `set_batch_progress` was added to the trait on 2026-07-29 and the duplicate
//! that existed then had to be patched twice.

use std::sync::Mutex;

use async_trait::async_trait;

use crate::api::{DisplayPath, JobId, JobStatus, MachineTime};
use crate::scheduling::{AttemptOutcome, FailureCategory, RetryDisposition, WorkUnitKind};
use crate::store::FileCompletion;

use super::FileStage;
use super::event_sink::RunnerEventSink;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RecordedProgress {
    pub(crate) job_id: JobId,
    pub(crate) filename: String,
    pub(crate) stage: FileStage,
    pub(crate) current: Option<i64>,
    pub(crate) total: Option<i64>,
}

/// One durable attempt the sink was asked to open.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RecordedAttempt {
    pub(crate) filename: String,
    pub(crate) work_unit_kind: WorkUnitKind,
}

/// One terminal file error the sink was asked to record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RecordedError {
    pub(crate) filename: String,
    pub(crate) error: String,
    pub(crate) category: FailureCategory,
}

#[derive(Default)]
pub(crate) struct RecordingSink {
    progress: Mutex<Vec<RecordedProgress>>,
    worker_waits: Mutex<Vec<(String, crate::store::WorkerWaitChange)>>,
    attempts: Mutex<Vec<RecordedAttempt>>,
    errors: Mutex<Vec<RecordedError>>,
    completed_files: Mutex<Vec<String>>,
}

#[async_trait]
impl RunnerEventSink for RecordingSink {
    fn now(&self) -> crate::store::EventTime {
        crate::store::EventTime::fixed(crate::unix_time(1_700_000_000.0))
    }

    async fn mark_file_processing(
        &self,
        _job_id: &JobId,
        _filename: &str,
        _started_at: crate::store::EventTime,
    ) {
    }

    async fn mark_file_done(
        &self,
        _job_id: &JobId,
        filename: &str,
        _finished_at: crate::store::EventTime,
        _completion: FileCompletion,
    ) {
        self.completed_files
            .lock()
            .expect("completed files lock")
            .push(filename.to_owned());
    }

    async fn mark_file_error(
        &self,
        _job_id: &JobId,
        filename: &str,
        error: &str,
        category: FailureCategory,
        _finished_at: crate::store::EventTime,
    ) {
        self.errors
            .lock()
            .expect("errors lock")
            .push(RecordedError {
                filename: filename.to_string(),
                error: error.to_string(),
                category,
            });
    }

    async fn start_file_attempt(
        &self,
        _job_id: &JobId,
        filename: &str,
        work_unit_kind: WorkUnitKind,
        _started_at: crate::store::EventTime,
    ) {
        self.attempts
            .lock()
            .expect("attempts lock")
            .push(RecordedAttempt {
                filename: filename.to_string(),
                work_unit_kind,
            });
    }

    async fn finish_file_attempt(
        &self,
        _job_id: &JobId,
        _filename: &str,
        _outcome: AttemptOutcome,
        _failure_category: Option<FailureCategory>,
        _disposition: RetryDisposition,
        _finished_at: crate::store::EventTime,
    ) {
    }

    async fn mark_file_retry_pending(
        &self,
        _job_id: &JobId,
        _filename: &str,
        _retry_at: MachineTime,
        _category: FailureCategory,
        _message: &str,
        _finished_at: crate::store::EventTime,
    ) {
    }

    async fn clear_file_retry_state(&self, _job_id: &JobId, _filename: &str) {}

    async fn set_file_progress(
        &self,
        job_id: &JobId,
        filename: &str,
        stage: FileStage,
        current: Option<i64>,
        total: Option<i64>,
    ) {
        self.progress
            .lock()
            .expect("progress lock")
            .push(RecordedProgress {
                job_id: job_id.clone(),
                filename: filename.to_string(),
                stage,
                current,
                total,
            });
    }

    async fn set_file_worker_wait(
        &self,
        _job_id: &JobId,
        filename: &str,
        change: crate::store::WorkerWaitChange,
    ) {
        self.worker_waits
            .lock()
            .expect("worker waits lock")
            .push((filename.to_owned(), change));
    }

    async fn unfinished_files(&self, _job_id: &JobId) -> Vec<DisplayPath> {
        Vec::new()
    }

    async fn file_status_label(&self, _job_id: &JobId, _filename: &str) -> Option<String> {
        None
    }

    async fn bump_forced_terminal_errors(&self, _count: usize) {}

    async fn fail_job(&self, _job_id: &JobId, _error: &str) {}

    async fn mark_job_running(&self, _job_id: &JobId) {}

    async fn record_job_worker_count(&self, _job_id: &JobId, _worker_count: usize) {}

    async fn requeue_job_after_memory_gate(&self, _job_id: &JobId, _retry_at: MachineTime) {}

    async fn bump_deferred_work_units(&self) {}

    async fn bump_memory_gate_aborts(&self) {}

    async fn finalize_job(
        &self,
        _job_id: &JobId,
        _expected_generation: crate::store::RunGeneration,
        _final_status: JobStatus,
        _completed_at: crate::store::EventTime,
    ) -> Option<String> {
        None
    }
}

impl RecordingSink {
    /// Files whose successful completion was recorded, in order.
    pub(crate) fn completed_files(&self) -> Vec<String> {
        self.completed_files
            .lock()
            .expect("completed files lock")
            .clone()
    }

    /// Every file-progress write the sink received, in order.
    pub(crate) fn progress(&self) -> Vec<RecordedProgress> {
        self.progress.lock().expect("progress lock").clone()
    }

    /// Every worker-wait change, in order, with its file.
    pub(crate) fn worker_waits(&self) -> Vec<(String, crate::store::WorkerWaitChange)> {
        self.worker_waits.lock().expect("worker waits lock").clone()
    }

    /// Every durable attempt the sink was asked to open, in order.
    ///
    /// Recorded because "the file failed" and "the file failed with an attempt
    /// behind it" are different facts, and a preflight rejection that records
    /// only the second leaves no attempt history for an operator to read.
    pub(crate) fn attempts(&self) -> Vec<RecordedAttempt> {
        self.attempts.lock().expect("attempts lock").clone()
    }

    /// Every terminal file error the sink was asked to record, in order.
    pub(crate) fn errors(&self) -> Vec<RecordedError> {
        self.errors.lock().expect("errors lock").clone()
    }
}
