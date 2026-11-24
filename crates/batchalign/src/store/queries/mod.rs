//! Query, mutation, and notification methods on [`JobStore`].

mod db_helpers;
mod dispatch;
mod execution;
pub mod file_state;
mod lifecycle;
mod recovery;
#[cfg(test)]
pub use recovery::RecoveredFilePhase;
pub use recovery::recover_file_phase;
mod runner;

pub use db_helpers::{AttemptFinishRecord, AttemptStartRecord, PersistedFileUpdate};
pub use dispatch::LeaseRenewalOutcome;

use crate::api::{CancellationRequest, JobId, JobInfo, JobListItem, JobStatus, MachineTime};

use tracing::warn;

use super::{JobDetail, JobStore, OperationalCounters};
use crate::error::ServerError;
use crate::ws::WsEvent;

/// Pass through a `Some(&str)` only when the contained string is non-empty.
/// Used at the cancellation-audit boundary so the DB sees genuine NULLs
/// instead of empty strings; lets `string_id!` newtypes (which still
/// permit empties at construction) project cleanly into nullable columns.
fn non_empty(s: Option<&str>) -> Option<&str> {
    s.filter(|v| !v.is_empty())
}

impl JobStore {
    /// How long a locally-dispatched lease lives before it is considered
    /// orphaned (`ServerConfig::local_lease_ttl`).
    const fn local_lease_ttl(&self) -> crate::config::LeaseTtl {
        self.config().local_lease_ttl
    }

    /// Look up a job by ID.
    pub async fn get(&self, job_id: &JobId) -> Option<JobInfo> {
        self.registry.job_info(job_id).await
    }

    /// Return all jobs (newest first).
    pub async fn list_all(&self) -> Vec<JobListItem> {
        self.registry.list_items().await
    }

    /// Signal a job to cancel, recording who/why/when in the
    /// `cancellations` audit table and projecting the most-recent
    /// metadata onto `jobs.last_cancelled_*` columns.
    ///
    /// `provenance` is the caller's self-report from the
    /// `POST /jobs/{id}/cancel` body (TUI fills it; raw curl leaves
    /// it default). Audit row is persisted regardless of whether the
    /// cancel actually changed job state, `accepted=false` records
    /// "user pressed cancel against an already-finished job," which
    /// is itself diagnostic.
    pub async fn cancel(
        &self,
        job_id: &JobId,
        provenance: CancellationRequest,
    ) -> Result<(), ServerError> {
        let now = self.now();
        let registry_outcome = self.registry.request_cancellation(job_id, now).await;
        let accepted = registry_outcome.is_some();

        // Record the audit row regardless of whether the cancel mutated
        // state: `accepted=false` distinguishes "user pressed cancel
        // against an already-finished job" from a state-changing cancel.
        self.record_audit_row(job_id, &provenance, now, accepted)
            .await;

        if !accepted {
            return Err(ServerError::JobNotFound(job_id.clone()));
        }

        // Persist the job's status now, even if the runner is stuck in
        // synchronous code and has not seen the cancellation token yet:
        // otherwise a daemon restart resurrects a cancelled job. A job that
        // was already terminal keeps its own status.
        self.db_persist_job_status(job_id).await;
        Ok(())
    }

    /// Record a cancel-attempt audit row WITHOUT changing job state.
    /// Used by the route handler when a cancel arrives against a job that
    /// is already terminal.
    pub async fn record_terminal_cancel(
        &self,
        job_id: &JobId,
        provenance: CancellationRequest,
    ) -> Result<(), ServerError> {
        let now = self.now();
        self.record_audit_row(job_id, &provenance, now, false).await;
        Ok(())
    }

    /// Persist one cancel attempt to the audit table and project the
    /// most-recent metadata onto the in-memory `Job` so `JobInfo`'s
    /// `last_cancelled_*` fields reflect this cancel without a DB JOIN.
    /// Both `cancel` and `record_terminal_cancel` flow through here.
    ///
    /// Failures to write the audit row are logged at WARN but do not
    /// propagate: the caller's primary cancel work (state change) still
    /// runs even if the audit write fails. Forensic rows are best-effort,
    /// not load-bearing on cancel correctness.
    async fn record_audit_row(
        &self,
        job_id: &JobId,
        provenance: &CancellationRequest,
        requested_at: MachineTime,
        accepted: bool,
    ) {
        let source = provenance.source.unwrap_or(crate::api::CancelSource::Api);
        let source_str = source.to_string();
        let host_str = non_empty(provenance.host.as_ref().map(AsRef::as_ref));
        let reason_str = non_empty(provenance.reason.as_ref().map(AsRef::as_ref));
        let correlation_str = non_empty(provenance.correlation_id.as_ref().map(AsRef::as_ref));
        let in_flight_str = non_empty(provenance.in_flight_filename.as_ref().map(AsRef::as_ref));
        let pid_value = provenance.pid.map(|p| p.0);

        if let Some(db) = &self.db
            && let Err(e) = db
                .insert_cancellation(
                    job_id,
                    requested_at,
                    &source_str,
                    host_str,
                    pid_value,
                    reason_str,
                    correlation_str,
                    in_flight_str,
                    accepted,
                )
                .await
        {
            tracing::warn!(
                job_id = %job_id,
                error = %e,
                "DB insert_cancellation failed"
            );
        }

        let info = crate::store::JobLastCancelInfo {
            at: requested_at,
            source: source_str,
            host: host_str.map(str::to_owned),
            reason: reason_str.map(str::to_owned),
        };
        self.registry.set_last_cancel(job_id, info).await;
    }

    /// Fetch the most recent cancellation audit row for a job, if any.
    ///
    /// Delegates to `JobDB::last_cancellation_for`. Returns `Ok(None)` when
    /// there is no DB (in-memory-only store) or when the job has no audit
    /// rows, so callers can treat `None` uniformly as "no audit information."
    ///
    /// See `JobDB::last_cancellation_for` for the reconciler use-case that
    /// motivates this helper.
    pub async fn last_cancellation_for(
        &self,
        job_id: &JobId,
    ) -> Result<Option<crate::db::CancellationRow>, ServerError> {
        match &self.db {
            Some(db) => db.last_cancellation_for(job_id).await,
            None => Ok(None),
        }
    }

    /// Read every cancel-attempt row for a job. Returns plain rows; the
    /// route layer maps them to wire-format `CancellationRecord`.
    pub async fn list_cancellations(
        &self,
        job_id: &JobId,
    ) -> Result<Vec<crate::db::CancellationRow>, ServerError> {
        match &self.db {
            Some(db) => db.list_cancellations(job_id).await,
            None => Ok(Vec::new()),
        }
    }

    /// Remove a job from the store.
    pub async fn delete(&self, job_id: &JobId) -> Result<(), ServerError> {
        let staging_dir = self
            .registry
            .remove_staging_dir(job_id)
            .await
            .ok_or_else(|| ServerError::JobNotFound(job_id.clone()))?;

        // Clean up staged content after releasing the jobs lock.
        if !staging_dir.as_str().is_empty() {
            let _ = tokio::fs::remove_dir_all(&staging_dir).await;
        }

        if let Some(db) = &self.db
            && let Err(e) = db.delete_job(job_id).await
        {
            warn!(job_id = %job_id, error = %e, "Failed to delete job from DB");
        }

        let _ = self.ws_tx.send(WsEvent::JobDeleted {
            job_id: job_id.clone(),
        });
        Ok(())
    }

    /// Check if a job is running (for delete guard).
    pub async fn is_running(&self, job_id: &JobId) -> Option<bool> {
        self.registry.is_running(job_id).await
    }

    /// Get the status of a specific job.
    pub async fn job_status(&self, job_id: &JobId) -> Option<JobStatus> {
        self.registry.job_status(job_id).await
    }

    /// Set the execution plan for a job (used by staged remote orchestrator).
    pub async fn set_execution_plan(
        &self,
        job_id: &JobId,
        plan: Option<crate::types::execution_plan::ExecutionPlan>,
    ) {
        self.registry
            .update_job(job_id.clone(), move |job| {
                job.execution_plan = plan;
            })
            .await;
    }

    /// Update a job's status and optional error message.
    ///
    /// Used by the staged remote orchestrator to set terminal states
    /// (`Completed`, `Failed`, `WritebackFailed`) after remote execution.
    pub(crate) async fn update_job_status(
        &self,
        job_id: &JobId,
        outcome: crate::store::Outcome,
        error: Option<String>,
    ) {
        let error_clone = error.clone();
        let completed_at = self.now();
        // Goes through the state machine like every other writer: a remote
        // result must not overwrite a status the job already reached, such as
        // a user's cancel landing while the remote leg was still running.
        // `None` (no such job) and `Some(false)` (refused) both mean "persist
        // nothing", so they collapse.
        if self
            .registry
            .update_job(job_id.clone(), move |job| {
                if !job.finalize(outcome, completed_at) {
                    return false;
                }
                if let Some(err) = error_clone {
                    job.execution.error = Some(err);
                }
                true
            })
            .await
            != Some(true)
        {
            return;
        }

        // Persist to SQLite
        self.db_persist_job_status(job_id).await;
    }

    /// Count of currently running jobs.
    pub async fn active_jobs(&self) -> i64 {
        self.registry.active_jobs().await
    }

    /// Approximate number of job slots available.
    pub async fn workers_available(&self) -> i64 {
        let active = self.active_jobs().await;
        (self.host.max_concurrent_jobs() as i64 - active).max(0)
    }

    /// Operational counters for health endpoint.
    pub async fn operational_counters(&self) -> (i64, i64, i64, i64, i64, i64) {
        self.counters
            .inspect(|counters| {
                (
                    counters.worker_crashes,
                    counters.attempts_started,
                    counters.attempts_retried,
                    counters.deferred_work_units,
                    counters.forced_terminal_errors,
                    counters.memory_gate_aborts,
                )
            })
            .await
    }

    pub(crate) async fn bump_counter(&self, f: impl FnOnce(&mut OperationalCounters)) {
        self.counters.mutate(f).await;
    }

    /// Get the staging dir for a job (used by results endpoint).
    pub async fn get_job_detail(&self, job_id: &JobId) -> Option<JobDetail> {
        self.registry.job_detail(job_id).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::api::JobStatus;
    use tokio::sync::broadcast;

    use super::*;
    use crate::api::{DisplayPath, MachineTime};
    use crate::db::JobDB;
    use crate::options::FaEngineName;
    use crate::scheduling::LeaseRecord;
    use crate::store::job::{
        Job, JobDispatchConfig, JobExecutionState, JobFilesystemConfig, JobIdentity,
        JobRuntimeControl, JobScheduleState, JobSourceContext,
    };
    use crate::store::{FileStatus, auto_max_concurrent_from};
    use crate::ws::BROADCAST_CAPACITY;

    pub(super) use crate::store::test_support::claim_registry_runner;

    pub(super) fn test_config() -> crate::config::ServerConfig {
        crate::config::ServerConfig {
            max_concurrent_jobs: Some(2),
            ..Default::default()
        }
    }

    /// Guards: Recovered refused paths retain the recorded output folders independently of misleading source metadata.
    /// Kind: contract
    /// Not a type: Admitted paths cannot prove how serialized history is displayed after a new store refuses those paths under changed authority.
    /// State: An isolated SQLite database exercises persistence and recovery. The owned directory also supplies real staging or admitted path identities required by the stated contract.
    /// Concurrency: The test owns its database directory and stores. Mutations finish before reload, deletion or subsequent projection.
    #[tokio::test]
    async fn output_folders_survive_database_recovery_without_using_source_dir() {
        use batchalign_types::paths::ClientPath;

        let (store, db, directory) = test_store_with_db().await;
        let root = directory.path().canonicalize().unwrap();
        let path = |suffix: &str| ClientPath::new(root.join(suffix).to_str().unwrap());
        let inputs = crate::store::test_support::paths_inputs(
            ["one.cha", "nested/two.cha"]
                .into_iter()
                .map(|name| crate::store::PathsInput {
                    display: name.into(),
                    kind: crate::store::InputKind::Chat,
                    source: crate::store::test_support::admitted(
                        path(&format!("input/{name}")).as_str(),
                    ),
                    output: crate::store::test_support::admitted(
                        path(&format!("results/{name}")).as_str(),
                    ),
                    before: None,
                })
                .collect(),
        );
        let mut job = make_job_with_inputs("persisted-output", FixtureCommand::Morphotag, inputs);
        job.source.source_dir = path("misleading-input");
        let id = JobId::from("persisted-output");
        store.submit(job).await.unwrap();
        let before =
            serde_json::to_value(store.get_job_detail(&id).await.unwrap().output_folders).unwrap();
        drop(store);
        let (tx, _) = broadcast::channel(BROADCAST_CAPACITY);
        let recovered = JobStore::for_tests(
            test_config(),
            Some(db),
            tx,
            Arc::new(crate::clock::SystemClock),
        );
        // A server with no paths-mode folders: the queued job cannot run,
        // so it comes back failed with the reason, never dropped, and its
        // recorded output folders are shown as before.
        assert_eq!(
            recovered
                .load_from_db(&batchalign_types::paths::AllowedRoots::none())
                .await
                .unwrap(),
            1
        );
        let info = recovered.get(&id).await.unwrap();
        assert_eq!(info.status, crate::api::JobStatus::Failed);
        assert!(
            info.error
                .as_deref()
                .is_some_and(|error| error.contains("paths mode is not enabled")),
            "{:?}",
            info.error
        );
        let after =
            serde_json::to_value(recovered.get_job_detail(&id).await.unwrap().output_folders)
                .unwrap();
        assert_eq!(before, after);
        assert_eq!(
            after,
            serde_json::json!({"job_id":"persisted-output", "access": {"state":"recorded", "directories": [root.join("results").to_str().unwrap(), root.join("results/nested").to_str().unwrap()]}})
        );
    }

    /// Create a store backed by a temporary SQLite database.
    /// A refused finalize must not be persisted to SQLite either.
    ///
    /// `Job::finalize` declines to move a job that already reached a terminal
    /// state, but `JobStore::finalize_job` wrote the status it was ASKED for
    /// rather than the one the job actually holds. Registry said `Cancelled`,
    /// `jobs.db` said `completed`.
    ///
    /// That is not cosmetic: startup recovery only revisits rows whose status
    /// is `Interrupted` or `Running`, so a row rewritten to a finished state is
    /// never reconciled and the job is permanently dead. Every other test in
    /// this area asserts only the in-memory side, which is how the hole
    /// survived.
    /// Guards: Late completion preserves cancellation in registry, database and attempt records.
    /// Kind: contract
    /// Not a type: Run admission cannot freeze later cancellation or prove SQLite row content across the completion adapter.
    /// State: An isolated SQLite database verifies cancellation and attempt persistence. A second job checks cleanup-only admission after cancellation.
    /// Concurrency: The test owns its directory, registry actor and broadcast channel. Ordered awaits expose the cancellation-completion interleaving.
    #[tokio::test]
    async fn cancelled_run_completion_persists_cancellation() {
        let (store, db, _dir) = test_store_with_db().await;

        let job_id = JobId::from("cancel-then-finalize");
        store
            .submit(make_job(
                "cancel-then-finalize",
                crate::store::queries::tests::FixtureCommand::Morphotag,
                vec!["a.cha".into()],
            ))
            .await
            .expect("submit");
        let claim = claim_registry_runner(&store, &job_id).await;
        store.mark_job_running(claim.binding()).await;
        store
            .start_file_attempt(
                claim.binding(),
                "a.cha",
                crate::scheduling::WorkUnitKind::FileProcess,
                store.event_time(),
            )
            .await;
        store
            .cancel(&job_id, CancellationRequest::default())
            .await
            .expect("cancel a running job");

        // Completion derives cancellation; callers cannot supply success.
        store
            .complete_run(claim, crate::store::EventTime::fixed(MachineTime::now()))
            .await
            .expect("complete cancelled run");

        assert_eq!(
            store.job_status(&job_id).await,
            Some(JobStatus::Cancelled),
            "in-memory status must stay Cancelled"
        );
        let persisted = db.load_all_jobs().await.expect("load jobs");
        let row = persisted
            .iter()
            .find(|job| job.job_id == job_id.as_ref())
            .expect("the job is in the database");
        assert_eq!(
            row.status, "cancelled",
            "the DATABASE must agree; a row written as a finished state is \
             never revisited by startup recovery"
        );
        let attempts = db.load_attempts_for_job(&job_id).await.expect("attempts");
        assert_eq!(attempts.len(), 1);
        assert_eq!(
            attempts[0].outcome,
            crate::scheduling::AttemptOutcome::Cancelled
        );
        assert_eq!(
            attempts[0].failure_category,
            Some(crate::scheduling::FailureCategory::Cancelled)
        );

        // A cancellation accepted before claim creation grants cleanup only.
        let waiting = JobId::from("cancel-before-claim");
        store
            .submit(make_job(
                waiting.as_ref(),
                crate::store::queries::tests::FixtureCommand::Morphotag,
                vec!["b.cha".into()],
            ))
            .await
            .unwrap();
        store
            .cancel(&waiting, CancellationRequest::default())
            .await
            .unwrap();
        let crate::store::BeginRunnerOutcome::Stopped { completion } =
            store.begin_runner(&waiting).await.unwrap()
        else {
            panic!("a stopped job must not receive a dispatch claim");
        };
        store
            .complete_stopped_run(completion, store.event_time())
            .await
            .unwrap();
        assert!(
            store
                .get(&waiting)
                .await
                .unwrap()
                .file_statuses
                .iter()
                .all(|file| file.status.is_terminal())
        );
        assert!(matches!(
            store.begin_runner(&waiting).await.unwrap(),
            crate::store::BeginRunnerOutcome::NotRunnable {
                status: JobStatus::Cancelled
            }
        ));
    }

    /// A file written with diagnostics is terminal written output end to
    /// end through the store: reported diagnosed with its findings, not a
    /// failure (the job completes), persisted, and restored by a fresh store
    /// from the database as the same phase with the same findings.
    /// Guards: Diagnosed output completes the job and restores the same findings and results after database reload.
    /// Kind: contract
    /// Not a type: Closed completion variants cannot prove diagnostic serialization, result projection or reconstruction from actual SQLite bytes.
    /// State: An isolated SQLite database and fresh store exercise persistence of diagnosed and clean outputs. No CHAT IO or inference is needed.
    /// Concurrency: Both stores and the database directory belong to this test. The restored store observes only completed awaited writes.
    #[tokio::test]
    async fn a_diagnosed_file_completes_its_job_and_round_trips_through_the_database() {
        use crate::store::{CompletedFileOutput, EventTime, FileCompletion};

        let (store, db, dir) = test_store_with_db().await;
        let job_id = JobId::from("diagnosed-job");
        store
            .submit(make_job(
                "diagnosed-job",
                crate::store::queries::tests::FixtureCommand::Morphotag,
                vec!["a.cha".into(), "b.cha".into()],
            ))
            .await
            .expect("submit");
        let claim = claim_registry_runner(&store, &job_id).await;
        store.mark_job_running(claim.binding()).await;
        let diagnostics = crate::api::FileOutputDiagnostics::of_findings(
            vec![crate::api::FileOutputDiagnostics::coded_finding(
                "E220",
                "digits inside a word",
            )],
            vec![crate::api::OutputShortfallRecord::StageSkipped {
                stage: crate::api::OptionalStage::Morphosyntax,
            }],
        );
        for filename in ["a.cha", "b.cha"] {
            store
                .mark_file_processing(
                    claim.binding(),
                    filename,
                    EventTime::fixed(crate::unix_time(1.0)),
                )
                .await;
        }
        store
            .mark_file_done(
                claim.binding(),
                "a.cha",
                EventTime::fixed(crate::unix_time(2.0)),
                FileCompletion::Diagnosed {
                    result: CompletedFileOutput {
                        exclusions: Vec::new(),
                        filename: DisplayPath::from("a.cha"),
                        content_type: crate::api::ContentType::Chat,
                        stamp: crate::api::FileStampOutcome::Unrecorded,
                    },
                    diagnostics: diagnostics.clone(),
                },
            )
            .await;
        store
            .mark_file_done(
                claim.binding(),
                "b.cha",
                EventTime::fixed(crate::unix_time(3.0)),
                FileCompletion::Clean(CompletedFileOutput {
                    exclusions: Vec::new(),
                    filename: DisplayPath::from("b.cha"),
                    content_type: crate::api::ContentType::Chat,
                    stamp: crate::api::FileStampOutcome::Unrecorded,
                }),
            )
            .await;

        store
            .complete_run(claim, EventTime::fixed(crate::unix_time(4.0)))
            .await
            .expect("complete diagnosed output");

        let info = store.get(&job_id).await.expect("job");
        assert_eq!(info.status, JobStatus::Completed);
        assert_eq!(info.completed_files, 2);
        let entry = |info: &crate::api::JobInfo, name: &str| {
            info.file_statuses
                .iter()
                .find(|entry| entry.filename.as_ref() == name)
                .cloned()
                .expect("file entry")
        };
        let diagnosed = entry(&info, "a.cha");
        assert_eq!(diagnosed.status, crate::api::FileStatusKind::Diagnosed);
        assert_eq!(diagnosed.diagnostics.as_ref(), Some(&diagnostics));
        assert_eq!(diagnosed.error, None);
        assert_eq!(diagnosed.finished_at, Some(crate::unix_time(2.0)));
        let detail = store.get_job_detail(&job_id).await.expect("detail");
        assert_eq!(
            detail
                .results
                .iter()
                .filter(|result| result.error.is_none())
                .count(),
            2,
            "the diagnosed output is a downloadable result"
        );

        // A fresh store over the same database restores the same phase.
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let restored = JobStore::for_tests(
            test_config(),
            Some(Arc::new(
                JobDB::open(Some(dir.path())).await.expect("reopen"),
            )),
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );
        drop(db);
        restored
            .load_from_db(&batchalign_types::paths::AllowedRoots::none())
            .await
            .expect("load");
        let info = restored.get(&job_id).await.expect("restored job");
        assert_eq!(info.status, JobStatus::Completed);
        let diagnosed = entry(&info, "a.cha");
        assert_eq!(diagnosed.status, crate::api::FileStatusKind::Diagnosed);
        assert_eq!(diagnosed.diagnostics, Some(diagnostics));
        assert_eq!(
            entry(&info, "b.cha").status,
            crate::api::FileStatusKind::Done
        );
    }

    async fn test_store_with_db() -> (JobStore, Arc<JobDB>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(JobDB::open(Some(dir.path())).await.unwrap());
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::for_tests(
            test_config(),
            Some(db.clone()),
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );
        (store, db, dir)
    }

    /// Commands supported by the lifecycle fixtures, with their actual options.
    pub(super) enum FixtureCommand {
        Align,
        Morphotag,
    }

    impl FixtureCommand {
        fn options(self) -> crate::options::CommandOptions {
            use crate::options::{AlignOptions, CommandOptions, CommonOptions, MorphotagOptions};
            match self {
                Self::Align => CommandOptions::Align(AlignOptions {
                    common: CommonOptions::default(),
                    fa_engine: FaEngineName::Wave2Vec,
                    utr: Default::default(),
                    pauses: false,
                    boundaries: Default::default(),
                    wor: true.into(),
                    media_dir: None,
                    bullet_repair: false,
                    review_level: Default::default(),
                }),
                Self::Morphotag => CommandOptions::Morphotag(MorphotagOptions {
                    common: CommonOptions::default(),

                    ..Default::default()
                }),
            }
        }
    }

    pub(super) fn make_job(id: &str, command: FixtureCommand, filenames: Vec<String>) -> Job {
        let inputs = crate::store::JobInputs::Content(
            filenames
                .into_iter()
                .map(|name| crate::store::ContentInput {
                    display: name.into(),
                    kind: crate::store::InputKind::Chat,
                })
                .collect(),
        );
        make_job_with_inputs(id, command, inputs)
    }

    fn make_job_with_inputs(
        id: &str,
        command: FixtureCommand,
        inputs: crate::store::JobInputs,
    ) -> Job {
        let options = command.options();
        let file_statuses = inputs
            .displays()
            .map(|name| (name.to_string(), FileStatus::new(name.clone())))
            .collect();

        Job {
            identity: JobIdentity {
                job_id: id.into(),
                correlation_id: format!("test-{id}").into(),
            },
            dispatch: JobDispatchConfig {
                command: options.command(),
                lang: crate::api::LanguageSpec::Resolved(crate::api::LanguageCode3::eng()),
                num_speakers: crate::api::NumSpeakers(1),
                options,
                runtime_state: std::collections::BTreeMap::new(),
                debug_traces: false,
            },
            source: JobSourceContext {
                submitter: Some(crate::store::Submitter::client(
                    std::net::Ipv4Addr::LOCALHOST.into(),
                    String::new(),
                )),
                source_dir: Default::default(),
            },
            filesystem: JobFilesystemConfig {
                staging_dir: Default::default(),
                media_mapping: Default::default(),
                media_subdir: Default::default(),
            },
            execution: JobExecutionState {
                lifecycle: crate::store::JobLifecycle::Active {
                    status: crate::api::ActiveStatus::Queued,
                    inputs,
                },
                file_statuses,
                results: Vec::new(),
                error: None,
            },
            schedule: JobScheduleState {
                submitted_at: MachineTime::now(),
                completed_at: None,
                next_eligible_at: None,
                num_workers: None,
                lease: None,
                last_cancel: None,
            },
            runtime: JobRuntimeControl::unclaimed(),
            execution_plan: None,
        }
    }

    /// Guards: A second valid command for an active source receives the precise conflict and original owner.
    /// Kind: contract
    /// Not a type: Closed fixture commands prevent option drift but cannot establish overlap against jobs already in the mutable registry.
    /// State: A private registry actor observes actual transitions and projections. SQLite, files and workers contribute no effect.
    /// Concurrency: The test owns its actor and broadcast channel. Ordered awaited requests specify the interleaving without shared global state.
    #[tokio::test]
    async fn conflict_detection() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::for_tests(
            test_config(),
            None,
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );

        let job1 = make_job(
            "j1",
            crate::store::queries::tests::FixtureCommand::Morphotag,
            vec!["a.cha".into()],
        );
        store.submit(job1).await.unwrap();

        let job2 = make_job(
            "j2",
            crate::store::queries::tests::FixtureCommand::Align,
            vec!["a.cha".into()],
        );
        assert!(matches!(store.submit(job2).await,
            Err(crate::error::ServerError::JobConflict { conflicts, .. })
                if conflicts.len() == 1
                    && conflicts[0].filename.as_ref() == "a.cha"
                    && conflicts[0].job_id.as_ref() == "j1"
                    && conflicts[0].command.as_ref() == "morphotag"
                    && conflicts[0].status == JobStatus::Queued));
    }

    /// Guards: Submission exposes the correct command and persists one queued job row.
    /// Kind: contract
    /// Not a type: Complete input records and closed commands cannot establish actor insertion, API projection or persisted SQLite columns.
    /// State: An isolated SQLite database exercises persistence and recovery. The owned directory also supplies real staging or admitted path identities required by the stated contract.
    /// Concurrency: The test owns its database directory and stores. Mutations finish before reload, deletion or subsequent projection.
    #[tokio::test]
    async fn submit_persists_job_without_relocking_store_state() {
        let (store, db, _dir) = test_store_with_db().await;

        let job = make_job(
            "j1",
            crate::store::queries::tests::FixtureCommand::Morphotag,
            vec!["a.cha".into()],
        );
        store.submit(job).await.unwrap();

        let info = store
            .get(&JobId::from("j1"))
            .await
            .expect("submitted projection");
        assert_eq!(info.command, "morphotag");

        let jobs = db.load_all_jobs().await.unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].job_id, "j1");
        assert_eq!(jobs[0].status, "queued");
    }

    /// Guards: Cancellation agrees in the registry projection and persisted status and completion timestamp.
    /// Kind: contract
    /// Not a type: An admitted fixture cannot prove how a later cancellation serializes its status and completion timestamp to SQLite.
    /// State: An isolated SQLite database exercises persistence and recovery. The owned directory also supplies real staging or admitted path identities required by the stated contract.
    /// Concurrency: The test owns its database directory and stores. Mutations finish before reload, deletion or subsequent projection.
    #[tokio::test]
    async fn cancel_persists_status_after_releasing_store_lock() {
        let (store, db, _dir) = test_store_with_db().await;

        let job = make_job(
            "j1",
            crate::store::queries::tests::FixtureCommand::Morphotag,
            vec!["a.cha".into()],
        );
        store.submit(job).await.unwrap();
        store
            .cancel(&JobId::from("j1"), CancellationRequest::default())
            .await
            .unwrap();

        assert_eq!(
            store
                .get(&JobId::from("j1"))
                .await
                .expect("cancelled projection")
                .status,
            JobStatus::Cancelled
        );

        let jobs = db.load_all_jobs().await.unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].status, "cancelled");
        assert!(jobs[0].completed_at.is_some());
    }

    /// Guards: Deleting a genuinely completed run removes its projection, database row and staged artifacts.
    /// Kind: contract
    /// Not a type: Consuming completion proves the run ended but cannot establish actual row deletion or removal of staged filesystem content.
    /// State: An isolated SQLite database exercises persistence and recovery. The owned directory also supplies real staging or admitted path identities required by the stated contract.
    /// Concurrency: The test owns its database directory and stores. Mutations finish before reload, deletion or subsequent projection.
    #[tokio::test]
    async fn delete_removes_db_row_and_staging_dir_after_unlocking_store() {
        let (store, db, dir) = test_store_with_db().await;
        let staging_dir = dir.path().join("staging-job");
        std::fs::create_dir_all(&staging_dir).unwrap();
        std::fs::write(staging_dir.join("artifact.txt"), "artifact").unwrap();

        let mut job = make_job(
            "j1",
            crate::store::queries::tests::FixtureCommand::Morphotag,
            vec!["a.cha".into()],
        );
        job.filesystem.staging_dir = batchalign_types::paths::ServerPath::from(staging_dir.clone());
        store.submit(job).await.unwrap();

        let id = JobId::from("j1");
        let claim = claim_registry_runner(&store, &id).await;
        store
            .mark_file_done(
                claim.binding(),
                "a.cha",
                store.event_time(),
                crate::store::FileCompletion::Clean(crate::store::CompletedFileOutput {
                    exclusions: Vec::new(),
                    filename: "a.cha".into(),
                    content_type: crate::api::ContentType::Chat,
                    stamp: crate::api::FileStampOutcome::Unrecorded,
                }),
            )
            .await;
        store
            .complete_run(claim, store.event_time())
            .await
            .expect("complete job before deletion");
        store.delete(&id).await.unwrap();
        assert!(store.get(&id).await.is_none());

        let jobs = db.load_all_jobs().await.unwrap();
        assert!(jobs.is_empty());
        assert!(!staging_dir.exists());
    }

    /// Guards: Job listing is newest first while execution admission claims the same jobs oldest first.
    /// Kind: contract
    /// Not a type: Complete input records cannot determine the ordering of observations returned by the live registry queue and listing adapters.
    /// State: A private registry actor observes actual transitions and projections. SQLite, files and workers contribute no effect.
    /// Concurrency: The test owns its actor and broadcast channel. Ordered awaited requests specify the interleaving without shared global state.
    #[tokio::test]
    async fn claim_ready_queued_jobs_orders_by_submission_time() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::for_tests(
            test_config(),
            None,
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );

        let mut early = make_job(
            "j1",
            crate::store::queries::tests::FixtureCommand::Morphotag,
            vec!["a.cha".into()],
        );
        early.schedule.submitted_at = crate::unix_time(100.0);
        store.submit(early).await.unwrap();

        let mut late = make_job(
            "j2",
            crate::store::queries::tests::FixtureCommand::Align,
            vec!["b.cha".into()],
        );
        late.schedule.submitted_at = crate::unix_time(200.0);
        store.submit(late).await.unwrap();

        assert_eq!(
            store
                .list_all()
                .await
                .into_iter()
                .map(|job| job.job_id)
                .collect::<Vec<_>>(),
            [JobId::from("j2"), JobId::from("j1")]
        );

        let poll = store.claim_ready_queued_jobs().await;
        assert_eq!(
            poll.ready_job_ids,
            vec![JobId::from("j1"), JobId::from("j2")]
        );
        assert_eq!(poll.next_wake_at, None);

        assert!(
            store
                .registry
                .runner_claim_active(&JobId::from("j1"))
                .await
                .unwrap()
        );
        assert!(
            store
                .registry
                .runner_claim_active(&JobId::from("j2"))
                .await
                .unwrap()
        );
        let lease = store
            .registry
            .lease_state(&JobId::from("j1"))
            .await
            .unwrap()
            .expect("the claim takes a lease");
        assert_eq!(lease.leased_by_node().as_ref(), store.node_id().as_ref());
    }

    /// Guards: Queue admission claims ready work and retains the exact deferred eligibility deadline.
    /// Kind: contract
    /// Not a type: The owned clock binds instants to this store but cannot determine the queue policy applied to a deferred job.
    /// State: `ManualClockStore` owns the real registry and its exact manual clock. Clock advancement supplies deterministic instants without sleeping.
    /// Concurrency: Each case owns its clock, actor and channel. Requests are awaited around explicit clock advancement or claim release.
    #[tokio::test]
    async fn claim_ready_queued_jobs_skips_deferred_and_reports_next_wake() {
        let clocked = crate::store::test_support::ManualClockStore::at(
            test_config(),
            crate::unix_time(1000.0),
        );
        let store = clocked.store();

        let mut ready = make_job(
            "ready",
            crate::store::queries::tests::FixtureCommand::Morphotag,
            vec!["a.cha".into()],
        );
        ready.schedule.submitted_at = crate::unix_time(100.0);
        store.submit(ready).await.unwrap();

        let deferred_at = clocked.now().plus(std::time::Duration::from_secs(60));
        let mut deferred = make_job(
            "deferred",
            crate::store::queries::tests::FixtureCommand::Align,
            vec!["b.cha".into()],
        );
        deferred.schedule.submitted_at = crate::unix_time(50.0);
        deferred.schedule.next_eligible_at = Some(deferred_at);
        store.submit(deferred).await.unwrap();

        let poll = store.claim_ready_queued_jobs().await;
        assert_eq!(poll.ready_job_ids, vec![JobId::from("ready")]);
        assert_eq!(poll.next_wake_at, Some(deferred_at));

        assert!(
            store
                .registry
                .runner_claim_active(&JobId::from("ready"))
                .await
                .unwrap()
        );
        assert!(
            !store
                .registry
                .runner_claim_active(&JobId::from("deferred"))
                .await
                .unwrap()
        );
        let ready_lease = store
            .registry
            .lease_state(&JobId::from("ready"))
            .await
            .unwrap()
            .expect("the ready job is claimed");
        assert_eq!(
            ready_lease.leased_by_node().as_ref(),
            store.node_id().as_ref()
        );
        assert!(
            store
                .registry
                .lease_state(&JobId::from("deferred"))
                .await
                .unwrap()
                .is_none()
        );
    }

    /// Guards: Queue admission preserves a live foreign lease and reclaims only the expired lease.
    /// Kind: contract
    /// Not a type: Ordered lease values cannot establish the current-clock policy selecting an expired owner over a live owner.
    /// State: `ManualClockStore` owns the real registry and its exact manual clock. Clock advancement supplies deterministic instants without sleeping.
    /// Concurrency: Each case owns its clock, actor and channel. Requests are awaited around explicit clock advancement or claim release.
    #[tokio::test]
    async fn claim_ready_queued_jobs_skips_unexpired_leases_and_reclaims_expired_ones() {
        let clocked = crate::store::test_support::ManualClockStore::at(
            test_config(),
            crate::unix_time(1000.0),
        );
        let store = clocked.store();

        let now = clocked.now();

        let mut leased = make_job(
            "leased",
            crate::store::queries::tests::FixtureCommand::Morphotag,
            vec!["a.cha".into()],
        );
        leased.schedule.lease = Some(
            LeaseRecord::new(
                "other-node".into(),
                now.minus(std::time::Duration::from_secs(10)),
                now.plus(std::time::Duration::from_secs(120)),
            )
            .expect("an ordered fixture lease"),
        );
        store.submit(leased).await.unwrap();

        let mut expired = make_job(
            "expired",
            crate::store::queries::tests::FixtureCommand::Align,
            vec!["b.cha".into()],
        );
        expired.schedule.lease = Some(
            LeaseRecord::new(
                "dead-node".into(),
                now.minus(std::time::Duration::from_secs(600)),
                now.minus(std::time::Duration::from_secs(1)),
            )
            .expect("an ordered fixture lease"),
        );
        store.submit(expired).await.unwrap();

        let poll = store.claim_ready_queued_jobs().await;
        assert_eq!(poll.ready_job_ids, vec![JobId::from("expired")]);
        assert_eq!(
            poll.next_wake_at,
            Some(now.plus(std::time::Duration::from_secs(120)))
        );

        let expired_lease = store
            .registry
            .lease_state(&JobId::from("expired"))
            .await
            .unwrap()
            .expect("the expired lease is reclaimed");
        assert_eq!(
            expired_lease.leased_by_node().as_ref(),
            store.node_id().as_ref()
        );
        let leased_lease = store
            .registry
            .lease_state(&JobId::from("leased"))
            .await
            .unwrap()
            .expect("the live lease stays");
        assert_eq!(leased_lease.leased_by_node().as_ref(), "other-node");
    }

    /// Guards: A live queue claim excludes a second claim and release makes the job eligible again.
    /// Kind: contract
    /// Not a type: A queue observation cannot freeze later claim release or prove that the store clears its retained lease.
    /// State: A private registry actor observes actual transitions and projections. SQLite, files and workers contribute no effect.
    /// Concurrency: The test owns its actor and broadcast channel. Ordered awaited requests specify the interleaving without shared global state.
    #[tokio::test]
    async fn release_runner_claim_makes_job_eligible_again() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::for_tests(
            test_config(),
            None,
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );

        let job = make_job(
            "j1",
            crate::store::queries::tests::FixtureCommand::Morphotag,
            vec!["a.cha".into()],
        );
        store.submit(job).await.unwrap();

        let first_poll = store.claim_ready_queued_jobs().await;
        assert_eq!(first_poll.ready_job_ids, vec![JobId::from("j1")]);

        let second_poll = store.claim_ready_queued_jobs().await;
        assert!(second_poll.ready_job_ids.is_empty());

        store.release_runner_claim(&JobId::from("j1")).await;

        let after_release_poll = store.claim_ready_queued_jobs().await;
        assert_eq!(after_release_poll.ready_job_ids, vec![JobId::from("j1")]);

        store.release_runner_claim(&JobId::from("j1")).await;
        let lease = store
            .registry
            .lease_state(&JobId::from("j1"))
            .await
            .unwrap();
        assert!(lease.is_none());
    }

    /// Guards: Renewal uses the exact advanced store clock and stops after the queue claim is released.
    /// Kind: contract
    /// Not a type: The private clock owner prevents mismatching stores but cannot prove the actual lease mutation or post-release refusal.
    /// State: `ManualClockStore` owns the real registry and its exact manual clock. Clock advancement supplies deterministic instants without sleeping.
    /// Concurrency: Each case owns its clock, actor and channel. Requests are awaited around explicit clock advancement or claim release.
    #[tokio::test]
    async fn renew_job_lease_updates_heartbeat_and_expiry_for_local_claim() {
        let clocked = crate::store::test_support::ManualClockStore::at(
            test_config(),
            crate::unix_time(1000.0),
        );
        let store = clocked.store();

        let job = make_job(
            "j1",
            crate::store::queries::tests::FixtureCommand::Morphotag,
            vec!["a.cha".into()],
        );
        store.submit(job).await.unwrap();
        let _ = store.claim_ready_queued_jobs().await;

        let before = store
            .registry
            .lease_state(&JobId::from("j1"))
            .await
            .unwrap()
            .expect("claimed")
            .heartbeat_at();

        clocked.advance(std::time::Duration::from_secs(5));
        assert_eq!(
            store.renew_job_lease(&JobId::from("j1")).await,
            crate::store::LeaseRenewalOutcome::Renewed
        );

        let lease = store
            .registry
            .lease_state(&JobId::from("j1"))
            .await
            .unwrap()
            .expect("renewed");
        assert_eq!(
            lease.heartbeat_at(),
            before.plus(std::time::Duration::from_secs(5))
        );
        store.release_runner_claim(&JobId::from("j1")).await;
        assert_eq!(
            store.renew_job_lease(&JobId::from("j1")).await,
            crate::store::LeaseRenewalOutcome::Stop
        );
    }

    #[test]
    fn auto_max_concurrent_caps_large_hosts() {
        assert_eq!(auto_max_concurrent_from(28, 8), 8);
    }

    #[test]
    fn auto_max_concurrent_respects_memory_tier() {
        assert_eq!(auto_max_concurrent_from(4, 1), 1);
        assert_eq!(auto_max_concurrent_from(8, 2), 2);
        assert_eq!(auto_max_concurrent_from(4, 8), 4);
    }
}
