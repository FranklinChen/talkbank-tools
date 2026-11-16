//! Runner-owned job execution mutations on [`JobStore`].

use crate::api::MachineTime;

use super::super::JobStore;
use super::super::registry::{CompletionCleanup, RunExit, RunFinishProjection};
use crate::scheduling::{AttemptOutcome, RetryDisposition};
use crate::store::{AttemptFinishRecord, PersistedFileUpdate, RunCompletion, RunnerClaim};

impl JobStore {
    /// Re-queue a job after memory-gate rejection and persist the retry deadline.
    pub(in crate::store) async fn requeue_job_after_memory_gate(
        &self,
        run: &crate::store::RunBinding,
        retry_at: MachineTime,
    ) -> crate::store::MemoryRequeueOutcome {
        let Some(job_update) = self.registry.requeue_after_memory_gate(run, retry_at).await else {
            return crate::store::MemoryRequeueOutcome::NoLongerRunnable;
        };
        self.notify_job_item(job_update);
        self.bump_counter(|c| c.deferred_work_units += 1).await;
        self.db_persist_job_status(run.job_id()).await;
        crate::store::MemoryRequeueOutcome::Requeued(crate::store::DeferredRun::admitted(retry_at))
    }

    /// Mark a job as running and clear any deferred retry deadline.
    pub(in crate::store) async fn mark_job_running(&self, run: &crate::store::RunBinding) {
        let Some(job_update) = self.registry.mark_job_running(run).await else {
            return;
        };
        self.notify_job_item(job_update);

        self.db_persist_job_status(run.job_id()).await;
    }

    /// Record the per-job worker count chosen for this run.
    pub(in crate::store) async fn record_job_worker_count(
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
    pub(in crate::store) async fn complete_stopped_run(
        &self,
        completion: crate::store::StoppedRun,
        finished_at: crate::store::EventTime,
    ) -> Result<RunCompletion, crate::error::ServerError> {
        self.complete_run(completion.into_claim(), finished_at)
            .await
    }

    /// Consume a run claim. Only the registry can admit cleanup, so callers
    /// cannot pass a status inconsistent with the files or clean up a restart.
    pub(in crate::store) async fn complete_run(
        &self,
        claim: RunnerClaim,
        finished_at: crate::store::EventTime,
    ) -> Result<RunCompletion, crate::error::ServerError> {
        self.finish_run(claim, RunExit::Dispatched, finished_at)
            .await
    }

    /// Dispatch failure still consumes the same generation-bound authority.
    pub(in crate::store) async fn fail_run(
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

    use crate::api::{JobId, JobStatus};
    use crate::store::queries::tests::{claim_registry_runner, make_job, test_config};
    use crate::ws::BROADCAST_CAPACITY;

    use super::*;

    /// Guards: accepted memory requeue returns a retry proof for the same
    /// deadline stored on the queued job and records one deferred-work event.
    /// Kind: policy
    /// Not a type: the actor observes mutable job membership and generation.
    /// The proof constructor is private, but its admission must still store
    /// the requested deadline and return the corresponding observation.
    /// State: an isolated in-memory registry performs real claim and requeue
    /// transitions. No persistence claim requires a database in this case.
    /// Concurrency: this case owns the registry and claim until explicit release.
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
                crate::store::queries::tests::FixtureCommand::Morphotag,
                vec!["a.cha".into()],
            ))
            .await
            .unwrap();

        let id = JobId::from("job-1");
        let claim = claim_registry_runner(&store, &id).await;
        let retry_at = crate::unix_time(321.0);
        let outcome = store
            .requeue_job_after_memory_gate(claim.binding(), retry_at)
            .await;
        let crate::store::MemoryRequeueOutcome::Requeued(retry) = outcome else {
            panic!("the current queued run admits its retry");
        };
        assert_eq!(retry.deadline(), retry_at);
        let (_, _, _, deferred, _, _) = store.operational_counters().await;
        assert_eq!(deferred, 1);

        let info = store.get(&JobId::from("job-1")).await.unwrap();
        assert_eq!(info.status, JobStatus::Queued);
        assert_eq!(info.next_eligible_at, Some(retry_at));
        store.release_run(claim).await;
    }

    /// Finalization recomputes completed file counts from terminal file state.
    /// Guards: Restart rejects old completion and current completion counts successful and unfinished files correctly.
    /// Kind: contract
    /// Not a type: A registry claim cannot freeze cancellation or restart observations arriving from other owners after admission.
    /// State: The real in-memory registry actor exercises generation arbitration and terminal projection. Disk and worker processes contribute no effect.
    /// Concurrency: This test owns its registry and broadcast channel. Restart and completion are awaited in the asserted order.
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
            crate::store::queries::tests::FixtureCommand::Morphotag,
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
        let old_claim = claim_registry_runner(&store, &id).await;
        store
            .cancel(&id, crate::api::CancellationRequest::default())
            .await
            .unwrap();
        store
            .restart(&id, &batchalign_types::paths::AllowedRoots::same_user())
            .await
            .unwrap();
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
                claim_registry_runner(&store, &id).await,
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
    /// Guards: File failure reasons reach the terminal job projection and survive database reload.
    /// Kind: contract
    /// Not a type: The admitted binding proves run ownership but cannot establish aggregated error text or persisted SQLite content.
    /// State: One isolated SQLite database is shared with a fresh store to verify real persistence. Synthetic worker error text supplies the external message.
    /// Concurrency: The directory and both registry actors belong to this test. Reload starts only after awaited completion persistence.
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
                crate::store::queries::tests::FixtureCommand::Morphotag,
                vec!["a.cha".into(), "b.cha".into()],
            ))
            .await
            .unwrap();

        let claim = claim_registry_runner(&store, &job_id).await;

        // Both files fail with the verbatim worker message, the shape of the
        // Stanza manifest-skew failure that exposed this bug.
        let failure = FileFailureRecord {
            message: "worker bootstrap error: ensure_task failed: md5 mismatch".into(),
            category: FailureCategory::WorkerBootstrap,
            finished_at: crate::store::EventTime::fixed(crate::unix_time(400.0)),
        };
        store
            .mark_file_error(claim.binding(), "a.cha", &failure)
            .await;
        store
            .mark_file_error(claim.binding(), "b.cha", &failure)
            .await;

        let completion = store
            .complete_run(
                claim,
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
        store2
            .load_from_db(&batchalign_types::paths::AllowedRoots::none())
            .await
            .unwrap();
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
