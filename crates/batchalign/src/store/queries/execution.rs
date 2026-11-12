//! Runner-owned job execution mutations on [`JobStore`].

use crate::api::MachineTime;

use super::super::JobStore;
use super::super::registry::{CompletionCleanup, RunExit, RunFinishProjection};
use crate::scheduling::{AttemptOutcome, RetryDisposition};
use crate::store::{AttemptFinishRecord, PersistedFileUpdate, RunCompletion, RunnerClaim};

impl JobStore {
    /// Re-queue a job after memory-gate rejection and persist the retry deadline.
    pub(crate) async fn requeue_job_after_memory_gate(
        &self,
        run: &crate::store::RunBinding,
        retry_at: MachineTime,
    ) {
        let Some(job_update) = self.registry.requeue_after_memory_gate(run, retry_at).await else {
            return;
        };
        self.notify_job_item(job_update);

        self.db_persist_job_status(run.job_id()).await;
    }

    /// Mark a job as running and clear any deferred retry deadline.
    pub(crate) async fn mark_job_running(&self, run: &crate::store::RunBinding) {
        let Some(job_update) = self.registry.mark_job_running(run).await else {
            return;
        };
        self.notify_job_item(job_update);

        self.db_persist_job_status(run.job_id()).await;
    }

    /// Record the per-job worker count chosen for this run.
    pub(crate) async fn record_job_worker_count(
        &self,
        run: &crate::store::RunBinding,
        num_workers: usize,
    ) {
        // Recording a worker count is not a status transition; the row is
        // written from the job as it is, whatever status that is.
        if !self
            .registry
            .record_job_worker_count(run, num_workers)
            .await
        {
            return;
        }

        self.db_persist_job_status(run.job_id()).await;
    }

    /// A cancelled-before-claim task has completion authority only.
    pub(crate) async fn complete_stopped_run(
        &self,
        completion: crate::store::StoppedRun,
        finished_at: crate::store::EventTime,
    ) -> Result<RunCompletion, crate::error::ServerError> {
        self.complete_run(completion.into_claim(), finished_at)
            .await
    }

    /// Consume a run claim. Only the registry can admit cleanup, so callers
    /// cannot pass a status inconsistent with the files or clean up a restart.
    pub(crate) async fn complete_run(
        &self,
        claim: RunnerClaim,
        finished_at: crate::store::EventTime,
    ) -> Result<RunCompletion, crate::error::ServerError> {
        self.finish_run(claim, RunExit::Dispatched, finished_at)
            .await
    }

    /// Dispatch failure still consumes the same generation-bound authority.
    pub(crate) async fn fail_run(
        &self,
        claim: RunnerClaim,
        message: String,
        finished_at: crate::store::EventTime,
    ) -> Result<RunCompletion, crate::error::ServerError> {
        self.finish_run(claim, RunExit::Failed(message), finished_at)
            .await
    }

    async fn finish_run(
        &self,
        claim: RunnerClaim,
        exit: RunExit,
        finished_at: crate::store::EventTime,
    ) -> Result<RunCompletion, crate::error::ServerError> {
        let run = claim.binding().clone();
        let job_id = run.job_id().clone();
        let completion = self
            .registry
            .complete_run(claim, exit, finished_at)
            .await
            .ok_or_else(|| crate::error::ServerError::JobNotInLocalStore(job_id.clone()))?;
        let super::super::registry::FinishingRun { claim, projection } = completion;
        let RunFinishProjection::Finished(completion) = projection else {
            self.release_run(claim).await;
            return Ok(RunCompletion::Superseded);
        };
        let (attempt_outcome, forced_errors) = match completion.cleanup {
            CompletionCleanup::Cancelled => (AttemptOutcome::Cancelled, 0),
            CompletionCleanup::SystemFailures(count) => (AttemptOutcome::Failed, count as i64),
        };
        for file in completion.files {
            let update = file.update;
            let filename = update.file.filename.clone();
            self.notify_file_update(&job_id, update.file, update.completed_files);
            self.db_update_file(
                &job_id,
                PersistedFileUpdate {
                    filename: &filename,
                    phase: &update.phase,
                    content_type: None,
                },
            )
            .await;
            self.db_finish_attempt_for_file(
                &run,
                AttemptFinishRecord {
                    filename: &filename,
                    outcome: attempt_outcome,
                    failure_category: Some(file.failure.category),
                    disposition: RetryDisposition::TerminalFailure,
                    finished_at: file.failure.finished_at,
                },
            )
            .await;
        }
        self.bump_counter(|c| c.forced_terminal_errors += forced_errors)
            .await;
        let outcome = RunCompletion::Finished {
            status: completion.job_update.status,
            error: completion.job_update.error.clone(),
        };
        self.notify_job_item(completion.job_update);
        self.db_persist_job_status(&job_id).await;
        self.release_run(claim).await;
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use tokio::sync::broadcast;

    use crate::api::{JobId, JobStatus, ReleasedCommand};
    use crate::store::queries::tests::{claim_runner, make_job, test_config};
    use crate::ws::BROADCAST_CAPACITY;

    use super::*;

    /// Re-queued jobs retain queued status and gain the next retry timestamp.
    #[tokio::test]
    async fn requeue_job_after_memory_gate_sets_retry_deadline() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::for_tests(
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

        let retry_at = crate::unix_time(321.0);
        store
            .requeue_job_after_memory_gate(
                &crate::store::RunBinding::first_run_of(JobId::from("job-1")),
                retry_at,
            )
            .await;

        let info = store.get(&JobId::from("job-1")).await.unwrap();
        assert_eq!(info.status, JobStatus::Queued);
        assert_eq!(info.next_eligible_at, Some(retry_at));
    }

    /// Finalization recomputes completed file counts from terminal file state.
    #[tokio::test]
    async fn run_completion_accounts_for_success_and_unfinished_files() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::for_tests(
            test_config(),
            None,
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );
        let mut job = make_job(
            "job-1",
            ReleasedCommand::Morphotag,
            vec!["a.cha".into(), "b.cha".into()],
        );
        job.execution.file_statuses.get_mut("a.cha").unwrap().phase =
            crate::store::FilePhase::Done {
                exclusions: Vec::new(),
                started_at: None,
                finished_at: Some(crate::unix_time(5.0)),
            };
        store.submit(job).await.unwrap();

        let id = JobId::from("job-1");
        let old_claim = claim_runner(&store, &id).await;
        store
            .cancel(&id, crate::api::CancellationRequest::default())
            .await
            .unwrap();
        store.restart(&id).await.unwrap();
        assert!(matches!(
            store
                .complete_run(
                    old_claim,
                    crate::store::EventTime::fixed(crate::unix_time(499.0))
                )
                .await
                .unwrap(),
            crate::store::RunCompletion::Superseded
        ));
        let restarted = store.get(&id).await.unwrap();
        assert_eq!(restarted.status, JobStatus::Queued);
        assert!(
            restarted
                .file_statuses
                .iter()
                .all(|file| file.error.is_none())
        );
        store.release_runner_claim(&id).await;
        store
            .complete_run(
                claim_runner(&store, &id).await,
                crate::store::EventTime::fixed(crate::unix_time(500.0)),
            )
            .await
            .expect("complete run");
        let info = store.get(&id).await.unwrap();
        assert_eq!(info.status, JobStatus::Failed);
        assert_eq!(info.completed_files, 2);
        assert_eq!(
            info.file_statuses
                .iter()
                .filter(|file| file.status == crate::api::FileStatusKind::Done)
                .count(),
            1
        );
        assert!(
            info.error
                .as_deref()
                .is_some_and(|reason| reason.contains("last status: queued"))
        );
    }

    /// A job that fails because its files failed must surface WHY: the per-file
    /// error messages are aggregated into the job-level error, which is shown
    /// in the in-memory projection (the `/jobs` + dashboard source) AND
    /// persisted to the `jobs.db` `error` column so it survives a reload.
    ///
    /// Regression for the 2026-06 silent-failure bug: a job that failed because
    /// its (Stanza-skew-affected) files failed was recorded with an EMPTY
    /// `error` column and no job-level reason, so the dashboard/jobs.db showed
    /// "failed" with no cause; the only place the real error appeared was the
    /// CLI client response.
    #[tokio::test]
    async fn finalize_failed_job_records_reason_from_file_failures() {
        use std::sync::Arc;

        use crate::db::JobDB;
        use crate::scheduling::FailureCategory;
        use crate::store::FileFailureRecord;

        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(JobDB::open(Some(dir.path())).await.unwrap());
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::for_tests(
            test_config(),
            Some(db.clone()),
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );

        let job_id = JobId::from("job-fail");
        store
            .submit(make_job(
                "job-fail",
                ReleasedCommand::Morphotag,
                vec!["a.cha".into(), "b.cha".into()],
            ))
            .await
            .unwrap();

        // Both files fail with the verbatim worker message, the shape of the
        // Stanza manifest-skew failure that exposed this bug.
        let failure = FileFailureRecord {
            message: "worker bootstrap error: ensure_task failed: md5 mismatch".into(),
            category: FailureCategory::WorkerBootstrap,
            finished_at: crate::store::EventTime::fixed(crate::unix_time(400.0)),
        };
        store
            .mark_file_error(
                &crate::store::RunBinding::first_run_of(job_id.clone()),
                "a.cha",
                &failure,
            )
            .await;
        store
            .mark_file_error(
                &crate::store::RunBinding::first_run_of(job_id.clone()),
                "b.cha",
                &failure,
            )
            .await;

        let completion = store
            .complete_run(
                claim_runner(&store, &job_id).await,
                crate::store::EventTime::fixed(crate::unix_time(500.0)),
            )
            .await
            .expect("complete failed run");
        let crate::store::RunCompletion::Finished { status, error } = completion else {
            panic!("claim must still own this job");
        };
        assert_eq!(status, JobStatus::Failed);
        assert!(
            error
                .as_deref()
                .is_some_and(|reason| reason.contains("md5 mismatch"))
        );

        // (1) In-memory projection (the `/jobs` endpoint + dashboard source)
        //     must carry the reason, not a bare "failed".
        let info = store.get(&job_id).await.unwrap();
        assert_eq!(info.status, JobStatus::Failed);
        let in_memory = info
            .error
            .expect("a failed job must record a job-level error reason (dashboard/jobs)");
        assert!(
            in_memory.contains("md5 mismatch"),
            "in-memory job error should name the file failure, got: {in_memory}"
        );

        // (2) The `jobs.db` `error` column must persist it: a fresh store
        //     reloaded purely from the DB still sees the reason.
        let (tx2, _rx2) = broadcast::channel(BROADCAST_CAPACITY);
        let store2 = JobStore::for_tests(
            test_config(),
            Some(db.clone()),
            tx2,
            std::sync::Arc::new(crate::clock::SystemClock),
        );
        store2.load_from_db().await.unwrap();
        let reloaded = store2.get(&job_id).await.expect("reloaded job present");
        let persisted = reloaded
            .error
            .expect("jobs.db `error` column must be populated for a failed job");
        assert!(
            persisted.contains("md5 mismatch"),
            "persisted job error should name the file failure, got: {persisted}"
        );
    }
}
