//! File-level job-state mutations on [`JobStore`].

use crate::scheduling::{AttemptOutcome, RetryDisposition, WorkUnitKind};

use super::super::{
    AttemptFinishRecord, AttemptStartRecord, EventTime, FileCompletion, FileFailureRecord,
    FileProgressRecord, FileRetryRecord, JobStore, PersistedFileUpdate, RunBinding,
};

impl JobStore {
    /// Mark one file as processing and persist the start timestamp.
    pub(in crate::store) async fn mark_file_processing(
        &self,
        run: &RunBinding,
        filename: &str,
        started_at: EventTime,
    ) {
        let update = match self
            .registry
            .mark_file_processing(run, filename, started_at.instant())
            .await
        {
            Ok(update) => update,
            Err(refusal) => {
                refusal.log(run, filename, "mark_file_processing");
                return;
            }
        };
        self.notify_file_update(&update.job_id, update.file, update.completed_files);

        self.db_update_file(
            run.job_id(),
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
    pub(in crate::store) async fn mark_file_done(
        &self,
        run: &RunBinding,
        filename: &str,
        finished_at: EventTime,
        completion: FileCompletion,
    ) {
        let persisted_content_type: Option<String> = completion
            .result()
            .map(|output| output.content_type.to_string());

        let update = match self
            .registry
            .mark_file_done(run, filename, finished_at.instant(), completion)
            .await
        {
            Ok(update) => update,
            Err(refusal) => {
                refusal.log(run, filename, "mark_file_done");
                return;
            }
        };
        self.notify_file_update(&update.job_id, update.file, update.completed_files);

        self.db_update_file(
            run.job_id(),
            PersistedFileUpdate {
                filename,
                phase: &update.phase,
                content_type: persisted_content_type.as_deref(),
            },
        )
        .await;
    }

    /// Mark one file as terminally failed and record the error result.
    pub(in crate::store) async fn mark_file_error(
        &self,
        run: &RunBinding,
        filename: &str,
        failure: &FileFailureRecord,
    ) {
        let update = match self.registry.mark_file_error(run, filename, failure).await {
            Ok(update) => update,
            Err(refusal) => {
                refusal.log(run, filename, "mark_file_error");
                return;
            }
        };
        self.notify_file_update(&update.job_id, update.file, update.completed_files);

        self.db_update_file(
            run.job_id(),
            PersistedFileUpdate {
                filename,
                phase: &update.phase,
                content_type: None,
            },
        )
        .await;
        self.db_finish_attempt_for_file(
            run,
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
    pub(in crate::store) async fn start_file_attempt(
        &self,
        run: &RunBinding,
        filename: &str,
        work_unit_kind: WorkUnitKind,
        started_at: EventTime,
    ) {
        let update = match self
            .registry
            .start_file_attempt(run, filename, started_at.instant())
            .await
        {
            Ok(update) => update,
            Err(refusal) => {
                refusal.log(run, filename, "start_file_attempt");
                return;
            }
        };
        self.notify_file_update(&update.job_id, update.file, update.completed_files);

        self.db_update_file(
            run.job_id(),
            PersistedFileUpdate {
                filename,
                phase: &update.phase,
                content_type: None,
            },
        )
        .await;
        self.db_start_attempt(
            run,
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
    pub(in crate::store) async fn mark_file_retry_pending(
        &self,
        run: &RunBinding,
        filename: &str,
        retry: &FileRetryRecord,
    ) {
        let update = match self
            .registry
            .mark_file_retry_pending(run, filename, retry)
            .await
        {
            Ok(update) => update,
            Err(refusal) => {
                refusal.log(run, filename, "mark_file_retry_pending");
                return;
            }
        };
        self.notify_file_update(&update.job_id, update.file, update.completed_files);

        self.db_update_file(
            run.job_id(),
            PersistedFileUpdate {
                filename,
                phase: &update.phase,
                content_type: None,
            },
        )
        .await;
        self.db_finish_attempt_for_file(
            run,
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
    pub(in crate::store) async fn clear_file_retry_state(&self, run: &RunBinding, filename: &str) {
        if let Err(refusal) = self.registry.clear_file_retry_state(run, filename).await {
            refusal.log(run, filename, "clear_file_retry_state");
        }
    }

    /// Apply a change in one file's open worker waits and notify listeners.
    pub(in crate::store) async fn set_file_worker_wait(
        &self,
        run: &RunBinding,
        filename: &str,
        change: crate::store::WorkerWaitChange,
    ) {
        match self
            .registry
            .set_file_worker_wait(run, filename, change)
            .await
        {
            Ok(update) => {
                self.notify_file_update(&update.job_id, update.file, update.completed_files)
            }
            Err(refusal) => refusal.log(run, filename, "set_file_worker_wait"),
        }
    }

    /// Apply an ephemeral progress update to one file and notify listeners.
    pub(in crate::store) async fn set_file_progress(
        &self,
        run: &RunBinding,
        filename: &str,
        progress: &FileProgressRecord,
    ) {
        match self
            .registry
            .set_file_progress(run, filename, progress)
            .await
        {
            Ok(update) => {
                self.notify_file_update(&update.job_id, update.file, update.completed_files)
            }
            Err(refusal) => refusal.log(run, filename, "set_file_progress"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::api::DisplayPath;
    use tokio::sync::broadcast;

    use crate::api::{ContentType, FileStatusKind, JobId};
    use crate::scheduling::FailureCategory;
    use crate::store::queries::tests::{make_job, test_config};
    use crate::ws::BROADCAST_CAPACITY;

    use super::*;

    fn clean_output(name: &str) -> FileCompletion {
        FileCompletion::Clean(crate::store::CompletedFileOutput {
            exclusions: Vec::new(),
            filename: DisplayPath::from(name),
            content_type: ContentType::Chat,
            stamp: crate::api::FileStampOutcome::Unrecorded,
        })
    }

    fn empty_store() -> JobStore {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        JobStore::for_tests(
            test_config(),
            None,
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        )
    }

    /// A write from a run the job has moved past is refused, so a task of a
    /// cancelled or interrupted run still finishing a step cannot land on the
    /// files of the run that replaced it.
    /// Guards: Shutdown-superseded file writes leave results and file phases untouched.
    /// Kind: contract
    /// Not a type: A retained binding cannot freeze the registry generation changed by a later shutdown request.
    /// State: A private registry actor observes actual transitions and projections. SQLite, files and workers contribute no effect.
    /// Concurrency: The test owns its actor and broadcast channel. Ordered awaited requests specify the interleaving without shared global state.
    #[tokio::test]
    async fn a_superseded_run_cannot_write_a_file() {
        let store = empty_store();
        let job_id = JobId::from("job-1");
        store
            .submit(make_job(
                "job-1",
                crate::store::queries::tests::FixtureCommand::Morphotag,
                vec!["a.cha".into()],
            ))
            .await
            .unwrap();
        // Shutdown supersedes the first run, as a restart would.
        assert_eq!(store.interrupt_all_for_shutdown().await, 1);

        store
            .mark_file_done(
                &crate::store::RunBinding::first_run_of(job_id.clone()),
                "a.cha",
                crate::store::EventTime::fixed(crate::unix_time(10.0)),
                clean_output("a.cha"),
            )
            .await;

        let detail = store.get_job_detail(&job_id).await.unwrap();
        assert!(detail.results.is_empty(), "stale write recorded a result");
        assert!(
            detail
                .file_statuses
                .iter()
                .all(|file| !file.status.is_terminal()),
            "stale write finished a file: {:?}",
            detail.file_statuses
        );
    }

    /// Each refused write names its reason, so the store can tell the
    /// expected refusals (a superseded run, a finished file) from the ones
    /// that mean the runner wrote about something the store never had.
    /// Guards: Unknown files, missing jobs, terminal files and stale runs retain distinct refusal reasons.
    /// Kind: contract
    /// Not a type: Valid bindings cannot prove current registry membership, file names or terminal state after external mutations.
    /// State: A private registry actor observes actual transitions and projections. SQLite, files and workers contribute no effect.
    /// Concurrency: The test owns its actor and broadcast channel. Ordered awaited requests specify the interleaving without shared global state.
    #[tokio::test]
    async fn a_refused_file_write_says_why() {
        use crate::store::registry::RefusedFileWrite;
        use crate::store::{FileWriteRefusal, RunBinding};
        let store = empty_store();
        let job_id = JobId::from("job-1");
        store
            .submit(make_job(
                "job-1",
                crate::store::queries::tests::FixtureCommand::Morphotag,
                vec!["a.cha".into()],
            ))
            .await
            .unwrap();
        let run = RunBinding::first_run_of(job_id.clone());
        let failure = FileFailureRecord {
            message: "stopped".into(),
            category: FailureCategory::Cancelled,
            finished_at: crate::store::EventTime::fixed(crate::unix_time(5.0)),
        };
        let progress = FileProgressRecord {
            stage: crate::api::FileProgressStage::Processing,
            current: None,
            total: None,
        };

        let unknown = store
            .registry
            .set_file_progress(&run, "z.cha", &progress)
            .await
            .map(|_| ());
        let no_job = store
            .registry
            .set_file_progress(
                &RunBinding::first_run_of(JobId::from("job-gone")),
                "a.cha",
                &progress,
            )
            .await
            .map(|_| ());
        store.mark_file_error(&run, "a.cha", &failure).await;
        let finished = store
            .registry
            .set_file_progress(&run, "a.cha", &progress)
            .await
            .map(|_| ());
        assert_eq!(store.interrupt_all_for_shutdown().await, 1);
        let stale = store
            .registry
            .set_file_progress(&run, "a.cha", &progress)
            .await
            .map(|_| ());

        assert_eq!(
            [unknown, no_job, finished, stale],
            [
                Err(RefusedFileWrite::File(FileWriteRefusal::UnknownFile)),
                Err(RefusedFileWrite::NoSuchJob),
                Err(RefusedFileWrite::File(FileWriteRefusal::AlreadyTerminal)),
                Err(RefusedFileWrite::StaleRun),
            ]
        );
    }

    /// Guards: a shutdown-superseded claim cannot mark the job running or
    /// obtain a replacement retry, and cannot alter its eligibility deadline.
    /// Kind: boundary
    /// Not a type: shutdown advances the actor's generation after a claim is
    /// minted. The run proof must be admitted against that changing generation.
    /// State: the private in-memory registry runs real claim and shutdown effects.
    /// No worker or database contributes to the generation-refusal assertion.
    /// Concurrency: shutdown precedes the stale writes in this isolated store.
    #[tokio::test]
    async fn a_superseded_run_cannot_mark_the_job_running_or_requeue_it() {
        let store = empty_store();
        let job_id = JobId::from("job-1");
        store
            .submit(make_job(
                "job-1",
                crate::store::queries::tests::FixtureCommand::Morphotag,
                vec!["a.cha".into()],
            ))
            .await
            .unwrap();
        let claim = crate::store::queries::tests::claim_registry_runner(&store, &job_id).await;
        assert_eq!(store.interrupt_all_for_shutdown().await, 1);
        let before = store.get(&job_id).await.unwrap();

        store.mark_job_running(claim.binding()).await;
        let outcome = store
            .requeue_job_after_memory_gate(claim.binding(), crate::unix_time(321.0))
            .await;
        assert!(matches!(
            outcome,
            crate::store::MemoryRequeueOutcome::NoLongerRunnable
        ));

        let after = store.get(&job_id).await.unwrap();
        assert_eq!(after.status, before.status);
        assert_eq!(after.next_eligible_at, before.next_eligible_at);
        store.release_run(claim).await;
    }

    /// A finished file stays finished within its run: a second terminal write
    /// is refused, so it neither replaces the outcome nor counts twice.
    /// Guards: A second terminal event preserves the first failure and records no extra result.
    /// Kind: contract
    /// Not a type: An admitted claim cannot rule out duplicate asynchronous completion events after a file has become terminal.
    /// State: A private registry actor observes actual transitions and projections. SQLite, files and workers contribute no effect.
    /// Concurrency: The test owns its actor and broadcast channel. Ordered awaited requests specify the interleaving without shared global state.
    #[tokio::test]
    async fn a_finished_file_is_not_finished_again() {
        let store = empty_store();
        let job_id = JobId::from("job-1");
        store
            .submit(make_job(
                "job-1",
                crate::store::queries::tests::FixtureCommand::Morphotag,
                vec!["a.cha".into()],
            ))
            .await
            .unwrap();
        let claim = crate::store::queries::tests::claim_registry_runner(&store, &job_id).await;
        let failure = FileFailureRecord {
            message: "stopped".into(),
            category: FailureCategory::Cancelled,
            finished_at: crate::store::EventTime::fixed(crate::unix_time(5.0)),
        };
        store
            .mark_file_error(claim.binding(), "a.cha", &failure)
            .await;
        store
            .mark_file_done(
                claim.binding(),
                "a.cha",
                crate::store::EventTime::fixed(crate::unix_time(10.0)),
                clean_output("a.cha"),
            )
            .await;

        let info = store.get(&job_id).await.unwrap();
        assert_eq!(info.completed_files, 1);
        let [file] = info.file_statuses.as_slice() else {
            panic!("one file expected");
        };
        assert_eq!(file.status, FileStatusKind::Error);
        let detail = store.get_job_detail(&job_id).await.unwrap();
        assert_eq!(detail.results.len(), 1, "a second result was recorded");
        store.release_run(claim).await;
    }

    /// Retry-pending updates keep the file in processing state with a deadline.
    /// Guards: Retry recording keeps the file processing and exposes the supplied deadline.
    /// Kind: contract
    /// Not a type: A typed retry record cannot establish the real actor projection of status and its eligibility instant.
    /// State: A private registry actor observes actual transitions and projections. SQLite, files and workers contribute no effect.
    /// Concurrency: The test owns its actor and broadcast channel. Ordered awaited requests specify the interleaving without shared global state.
    #[tokio::test]
    async fn mark_file_retry_pending_sets_deadline() {
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

        store
            .mark_file_retry_pending(
                &crate::store::RunBinding::first_run_of(JobId::from("job-1")),
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
