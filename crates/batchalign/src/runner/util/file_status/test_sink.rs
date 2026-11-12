//! Shared recording [`RunnerEventSink`] for tests.
//!
//! One mock, not one per test module: the sink has ~20 methods, and a second
//! hand-written copy is a second thing to forget to update when the trait grows.
//! `set_batch_progress` was added to the trait on 2026-07-29 and the duplicate
//! that existed then had to be patched twice.

use std::sync::Mutex;

use async_trait::async_trait;

use crate::api::MachineTime;
use crate::scheduling::{AttemptOutcome, FailureCategory, RetryDisposition, WorkUnitKind};
use crate::store::FileCompletion;

use super::FileStage;
use super::event_sink::RunnerEventSink;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RecordedProgress {
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

/// How one file's completion was recorded.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum RecordedCompletion {
    /// Finished without diagnostics.
    Done { filename: String },
    /// Written with diagnostics.
    Diagnosed {
        filename: String,
        diagnostics: crate::api::FileOutputDiagnostics,
    },
}

impl RecordedCompletion {
    fn filename(&self) -> &str {
        match self {
            Self::Done { filename } | Self::Diagnosed { filename, .. } => filename,
        }
    }
}

/// One terminal file error the sink was asked to record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RecordedError {
    pub(crate) filename: String,
    pub(crate) error: String,
    pub(crate) category: FailureCategory,
}

/// Records what the runner reported. Like the store's sink it stands for
/// one run of one job, so no method names the job.
#[derive(Default)]
pub(crate) struct RecordingSink {
    progress: Mutex<Vec<RecordedProgress>>,
    worker_waits: Mutex<Vec<(String, crate::store::WorkerWaitChange)>>,
    attempts: Mutex<Vec<RecordedAttempt>>,
    errors: Mutex<Vec<RecordedError>>,
    completions: Mutex<Vec<RecordedCompletion>>,
    retries: Mutex<Vec<String>>,
    finished_attempts: Mutex<Vec<AttemptOutcome>>,
}

#[async_trait]
impl RunnerEventSink for RecordingSink {
    fn now(&self) -> crate::store::EventTime {
        crate::store::EventTime::fixed(crate::unix_time(1_700_000_000.0))
    }

    async fn mark_file_processing(&self, _filename: &str, _started_at: crate::store::EventTime) {}

    async fn mark_file_done(
        &self,
        filename: &str,
        _finished_at: crate::store::EventTime,
        completion: FileCompletion,
    ) {
        let filename = filename.to_owned();
        let recorded = match completion {
            FileCompletion::Diagnosed { diagnostics, .. } => RecordedCompletion::Diagnosed {
                filename,
                diagnostics,
            },
            FileCompletion::Clean(_) | FileCompletion::WithoutResult => {
                RecordedCompletion::Done { filename }
            }
        };
        self.completions
            .lock()
            .expect("completions lock")
            .push(recorded);
    }

    async fn mark_file_error(
        &self,
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
        _filename: &str,
        outcome: AttemptOutcome,
        _failure_category: Option<FailureCategory>,
        _disposition: RetryDisposition,
        _finished_at: crate::store::EventTime,
    ) {
        self.finished_attempts
            .lock()
            .expect("finished attempts lock")
            .push(outcome);
    }

    async fn mark_file_retry_pending(
        &self,
        filename: &str,
        _retry_at: MachineTime,
        _category: FailureCategory,
        _message: &str,
        _finished_at: crate::store::EventTime,
    ) {
        self.retries
            .lock()
            .expect("retries lock")
            .push(filename.to_owned());
    }

    async fn clear_file_retry_state(&self, _filename: &str) {}

    async fn set_file_progress(
        &self,
        filename: &str,
        stage: FileStage,
        current: Option<i64>,
        total: Option<i64>,
    ) {
        self.progress
            .lock()
            .expect("progress lock")
            .push(RecordedProgress {
                filename: filename.to_string(),
                stage,
                current,
                total,
            });
    }

    async fn set_file_worker_wait(&self, filename: &str, change: crate::store::WorkerWaitChange) {
        self.worker_waits
            .lock()
            .expect("worker waits lock")
            .push((filename.to_owned(), change));
    }

    async fn mark_job_running(&self) {}

    async fn record_job_worker_count(&self, _worker_count: usize) {}

    async fn requeue_job_after_memory_gate(&self, _retry_at: MachineTime) {}

    async fn bump_deferred_work_units(&self) {}

    async fn bump_memory_gate_aborts(&self) {}
}

impl RecordingSink {
    /// Files whose completion was recorded, done or diagnosed, in order.
    pub(crate) fn completed_files(&self) -> Vec<String> {
        self.completions()
            .iter()
            .map(|completion| completion.filename().to_owned())
            .collect()
    }

    /// Every completion, with how it was recorded, in order.
    pub(crate) fn completions(&self) -> Vec<RecordedCompletion> {
        self.completions.lock().expect("completions lock").clone()
    }

    /// The diagnostics of every file recorded as written with diagnostics.
    pub(crate) fn diagnosed(&self) -> Vec<crate::api::FileOutputDiagnostics> {
        self.completions()
            .into_iter()
            .filter_map(|completion| match completion {
                RecordedCompletion::Diagnosed { diagnostics, .. } => Some(diagnostics),
                RecordedCompletion::Done { .. } => None,
            })
            .collect()
    }

    /// Files recorded as finished without diagnostics, in order.
    pub(crate) fn done_files(&self) -> Vec<String> {
        self.completions()
            .into_iter()
            .filter_map(|completion| match completion {
                RecordedCompletion::Done { filename } => Some(filename),
                RecordedCompletion::Diagnosed { .. } => None,
            })
            .collect()
    }

    /// Files scheduled for a retry, once per retry, in order.
    pub(crate) fn retries(&self) -> Vec<String> {
        self.retries.lock().expect("retries lock").clone()
    }

    /// The outcome of every attempt the sink was asked to finish, in order.
    pub(crate) fn finished_attempts(&self) -> Vec<AttemptOutcome> {
        self.finished_attempts
            .lock()
            .expect("finished attempts lock")
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
