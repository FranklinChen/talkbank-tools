//! File-level job-state mutations on [`JobStore`].

use crate::api::{DisplayPath, JobId};
use crate::scheduling::{AttemptOutcome, RetryDisposition, WorkUnitKind};

use super::super::{
    AttemptFinishRecord, AttemptStartRecord, EventTime, FileCompletion, FileFailureRecord,
    FileProgressRecord, FileRetryRecord, JobStore, PersistedFileUpdate,
};

impl JobStore {
    /// Mark one file as processing and persist the start timestamp.
    pub(crate) async fn mark_file_processing(
        &self,
        job_id: &JobId,
        filename: &str,
        started_at: EventTime,
    ) {
        let Some(update) = self
            .registry
            .mark_file_processing(job_id, filename, started_at.instant())
            .await
        else {
            return;
        };
        self.notify_file_update(&update.job_id, update.file, update.completed_files);

        self.db_update_file(
            job_id,
            PersistedFileUpdate {
                filename,
                phase: &update.phase,
                content_type: None,
            },
        )
        .await;
    }

    /// Mark one file as finished without failing (`Done`, or `Diagnosed`
    /// when its written output carries admission diagnostics) and record its
    /// downloadable result if it has one.
    pub(crate) async fn mark_file_done(
        &self,
        job_id: &JobId,
        filename: &str,
        finished_at: EventTime,
        completion: FileCompletion,
    ) {
        let persisted_content_type: Option<String> = completion
            .result()
            .map(|output| output.content_type.to_string());

        let Some(update) = self
            .registry
            .mark_file_done(job_id, filename, finished_at.instant(), completion)
            .await
        else {
            return;
        };
        self.notify_file_update(&update.job_id, update.file, update.completed_files);

        self.db_update_file(
            job_id,
            PersistedFileUpdate {
                filename,
                phase: &update.phase,
                content_type: persisted_content_type.as_deref(),
            },
        )
        .await;
    }

    /// Mark one file as terminally failed and record the error result.
    pub(crate) async fn mark_file_error(
        &self,
        job_id: &JobId,
        filename: &str,
        failure: &FileFailureRecord,
    ) {
        let Some(update) = self
            .registry
            .mark_file_error(job_id, filename, failure)
            .await
        else {
            return;
        };
        self.notify_file_update(&update.job_id, update.file, update.completed_files);

        self.db_update_file(
            job_id,
            PersistedFileUpdate {
                filename,
                phase: &update.phase,
                content_type: None,
            },
        )
        .await;
        self.db_finish_attempt_for_file(
            job_id,
            AttemptFinishRecord {
                filename,
                outcome: AttemptOutcome::Failed,
                failure_category: Some(failure.category),
                disposition: RetryDisposition::TerminalFailure,
                finished_at: failure.finished_at,
            },
        )
        .await;
    }

    /// Mark one file attempt as started and attach the new attempt record.
    pub(crate) async fn start_file_attempt(
        &self,
        job_id: &JobId,
        filename: &str,
        work_unit_kind: WorkUnitKind,
        started_at: EventTime,
    ) {
        let Some(update) = self
            .registry
            .start_file_attempt(job_id, filename, started_at.instant())
            .await
        else {
            return;
        };
        self.notify_file_update(&update.job_id, update.file, update.completed_files);

        self.db_update_file(
            job_id,
            PersistedFileUpdate {
                filename,
                phase: &update.phase,
                content_type: None,
            },
        )
        .await;
        self.db_start_attempt(
            job_id,
            AttemptStartRecord {
                filename,
                work_unit_kind,
                started_at,
            },
        )
        .await;
        self.bump_counter(|c| c.attempts_started += 1).await;
    }

    /// Mark one file as waiting for a retry after a transient failure.
    pub(crate) async fn mark_file_retry_pending(
        &self,
        job_id: &JobId,
        filename: &str,
        retry: &FileRetryRecord,
    ) {
        let Some(update) = self
            .registry
            .mark_file_retry_pending(job_id, filename, retry)
            .await
        else {
            return;
        };
        self.notify_file_update(&update.job_id, update.file, update.completed_files);

        self.db_update_file(
            job_id,
            PersistedFileUpdate {
                filename,
                phase: &update.phase,
                content_type: None,
            },
        )
        .await;
        self.db_finish_attempt_for_file(
            job_id,
            AttemptFinishRecord {
                filename,
                outcome: AttemptOutcome::RetryableFailure,
                failure_category: Some(retry.category),
                disposition: RetryDisposition::Retry,
                finished_at: retry.finished_at,
            },
        )
        .await;
        self.bump_counter(|c| c.attempts_retried += 1).await;
        self.bump_counter(|c| c.deferred_work_units += 1).await;
    }

    /// Clear transient retry state before a new attempt starts or succeeds.
    pub(crate) async fn clear_file_retry_state(&self, job_id: &JobId, filename: &str) {
        let _ = self.registry.clear_file_retry_state(job_id, filename).await;
    }

    /// Apply a change in one file's open worker waits and notify listeners.
    pub(crate) async fn set_file_worker_wait(
        &self,
        job_id: &JobId,
        filename: &str,
        change: crate::store::WorkerWaitChange,
    ) {
        if let Some(update) = self
            .registry
            .set_file_worker_wait(job_id, filename, change)
            .await
        {
            self.notify_file_update(&update.job_id, update.file, update.completed_files);
        }
    }

    /// Apply an ephemeral progress update to one file and notify listeners.
    pub(crate) async fn set_file_progress(
        &self,
        job_id: &JobId,
        filename: &str,
        progress: &FileProgressRecord,
    ) {
        if let Some(update) = self
            .registry
            .set_file_progress(job_id, filename, progress)
            .await
        {
            self.notify_file_update(&update.job_id, update.file, update.completed_files);
        }
    }

    /// Return the filenames of files that have not yet reached a terminal state.
    pub(crate) async fn unfinished_files(&self, job_id: &JobId) -> Vec<DisplayPath> {
        self.registry.unfinished_files(job_id).await
    }

    /// Return the current file-status label for one file.
    pub(crate) async fn file_status_label(&self, job_id: &JobId, filename: &str) -> Option<String> {
        self.registry.file_status_label(job_id, filename).await
    }
}

#[cfg(test)]
mod tests {
    use tokio::sync::broadcast;

    use crate::api::{ContentType, FileStatusKind, JobId, ReleasedCommand};
    use crate::scheduling::FailureCategory;
    use crate::store::queries::tests::{make_job, test_config};
    use crate::ws::BROADCAST_CAPACITY;

    use super::*;

    /// Completing a file records one result and increments the terminal count.
    #[tokio::test]
    async fn mark_file_done_records_result() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::new(
            test_config(),
            None,
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );
        store
            .submit(make_job(
                "job-1",
                ReleasedCommand::Morphotag,
                vec!["a.cha".into()],
            ))
            .await
            .unwrap();

        store
            .mark_file_done(
                &JobId::from("job-1"),
                "a.cha",
                crate::store::EventTime::fixed(crate::unix_time(10.0)),
                FileCompletion::Clean(crate::store::CompletedFileOutput {
                    filename: DisplayPath::from("a.cha"),
                    content_type: ContentType::Chat,
                    stamp: crate::api::FileStampOutcome::Unrecorded,
                }),
            )
            .await;

        let detail = store.get_job_detail(&JobId::from("job-1")).await.unwrap();
        assert_eq!(detail.results.len(), 1);
        assert_eq!(detail.results[0].error, None);
    }

    /// Retry-pending updates keep the file in processing state with a deadline.
    #[tokio::test]
    async fn mark_file_retry_pending_sets_deadline() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::new(
            test_config(),
            None,
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );
        store
            .submit(make_job(
                "job-1",
                ReleasedCommand::Morphotag,
                vec!["a.cha".into()],
            ))
            .await
            .unwrap();

        store
            .mark_file_retry_pending(
                &JobId::from("job-1"),
                "a.cha",
                &FileRetryRecord {
                    message: "retry later".into(),
                    category: FailureCategory::WorkerTimeout,
                    finished_at: crate::store::EventTime::fixed(crate::unix_time(10.0)),
                    retry_at: crate::unix_time(20.0),
                },
            )
            .await;

        let detail = store.get_job_detail(&JobId::from("job-1")).await.unwrap();
        let file = detail
            .file_statuses
            .into_iter()
            .find(|status| status.filename == "a.cha")
            .unwrap();
        assert_eq!(file.status, FileStatusKind::Processing);
        assert_eq!(file.next_eligible_at, Some(crate::unix_time(20.0)));
    }
}
