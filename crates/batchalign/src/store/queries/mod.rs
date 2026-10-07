//! Query, mutation, and notification methods on [`JobStore`].

mod db_helpers;
mod dispatch;
mod execution;
pub(crate) mod file_state;
mod lifecycle;
mod recovery;
#[cfg(test)]
pub(crate) use recovery::RecoveredFilePhase;
pub(crate) use recovery::recover_file_phase;
mod runner;

pub(crate) use db_helpers::{AttemptFinishRecord, AttemptStartRecord, PersistedFileUpdate};
pub(crate) use dispatch::LeaseRenewalOutcome;

use crate::api::{
    CancellationRequest, JobId, JobInfo, JobListItem, JobStatus, MachineTime, StatusChange,
};

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
    fn local_lease_ttl(&self) -> crate::config::LeaseTtl {
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
    pub async fn update_job_status(
        &self,
        job_id: &JobId,
        status: JobStatus,
        error: Option<String>,
    ) {
        let error_clone = error.clone();
        let completed_at = if status.is_terminal() {
            Some(self.now())
        } else {
            None
        };
        // Goes through the state machine like every other writer: a remote
        // result must not overwrite a status the job already reached, such as
        // a user's cancel landing while the remote leg was still running.
        // `None` (no such job) and `Some(false)` (refused) both mean "persist
        // nothing", so they collapse.
        if self
            .registry
            .update_job(job_id.clone(), move |job| {
                if !job.set_status(status, StatusChange::Finalize) {
                    return false;
                }
                if let Some(err) = error_clone {
                    job.execution.error = Some(err);
                }
                if let Some(ts) = completed_at {
                    job.schedule.completed_at = Some(ts);
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
        (self.max_concurrent as i64 - active).max(0)
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
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::api::{JobStatus, ReleasedCommand};
    use tokio::sync::broadcast;
    use tokio_util::sync::CancellationToken;

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

    pub(super) fn test_config() -> crate::config::ServerConfig {
        crate::config::ServerConfig {
            max_concurrent_jobs: Some(2),
            ..Default::default()
        }
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
    #[tokio::test]
    async fn a_refused_finalize_is_not_persisted() {
        let (store, db, _dir) = test_store_with_db().await;

        let job_id = JobId::from("cancel-then-finalize");
        store
            .submit(make_job(
                "cancel-then-finalize",
                ReleasedCommand::Morphotag,
                vec!["a.cha".into()],
            ))
            .await
            .expect("submit");
        store.mark_job_running(&job_id).await;
        store
            .cancel(&job_id, CancellationRequest::default())
            .await
            .expect("cancel a running job");

        // The runner drains and reports success for work the user stopped.
        store
            .finalize_job(
                &job_id,
                crate::store::RunGeneration::FIRST,
                JobStatus::Completed,
                crate::store::EventTime::fixed(MachineTime::now()),
            )
            .await;

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
    }

    /// A file written with diagnostics is terminal written output end to
    /// end through the store: reported diagnosed with its findings, not a
    /// failure (the job completes), persisted, and restored by a fresh store
    /// from the database as the same phase with the same findings.
    #[tokio::test]
    async fn a_diagnosed_file_completes_its_job_and_round_trips_through_the_database() {
        use crate::store::{CompletedFileOutput, EventTime, FileCompletion};

        let (store, db, dir) = test_store_with_db().await;
        let job_id = JobId::from("diagnosed-job");
        store
            .submit(make_job(
                "diagnosed-job",
                ReleasedCommand::Morphotag,
                vec!["a.cha".into(), "b.cha".into()],
            ))
            .await
            .expect("submit");
        store.mark_job_running(&job_id).await;
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
                .mark_file_processing(&job_id, filename, EventTime::fixed(crate::unix_time(1.0)))
                .await;
        }
        store
            .mark_file_done(
                &job_id,
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
                &job_id,
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

        let completion = store
            .completion_snapshot(&job_id)
            .await
            .expect("completion facts");
        assert!(!completion.any_failed, "a diagnosed file is not a failure");
        assert!(!completion.all_failed);
        store
            .finalize_job(
                &job_id,
                crate::store::RunGeneration::FIRST,
                JobStatus::Completed,
                EventTime::fixed(crate::unix_time(4.0)),
            )
            .await;

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
        let restored = JobStore::new(
            test_config(),
            Some(Arc::new(
                JobDB::open(Some(dir.path())).await.expect("reopen"),
            )),
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );
        drop(db);
        restored.load_from_db().await.expect("load");
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
        let store = JobStore::new(
            test_config(),
            Some(db.clone()),
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );
        (store, db, dir)
    }

    pub(super) fn make_job(
        id: &str,
        command: crate::api::ReleasedCommand,
        filenames: Vec<String>,
    ) -> Job {
        use crate::options::{AlignOptions, CommandOptions, CommonOptions, MorphotagOptions};

        let mut file_statuses = HashMap::new();
        let has_chat: Vec<bool> = filenames.iter().map(|_| true).collect();
        for f in &filenames {
            file_statuses.insert(
                f.clone(),
                FileStatus::new(crate::api::DisplayPath::from(f.as_str())),
            );
        }

        let options = match command {
            crate::api::ReleasedCommand::Align => CommandOptions::Align(AlignOptions {
                common: CommonOptions::default(),
                fa_engine: FaEngineName::Wave2Vec,
                utr: Default::default(),
                pauses: false,
                boundaries: Default::default(),
                wor: true.into(),
                merge_abbrev: false.into(),
                media_dir: None,
                bullet_repair: false,
                review_level: Default::default(),
            }),
            _ => CommandOptions::Morphotag(MorphotagOptions {
                common: CommonOptions::default(),

                ..Default::default()
            }),
        };

        Job {
            identity: JobIdentity {
                job_id: id.into(),
                correlation_id: format!("test-{id}").into(),
            },
            dispatch: JobDispatchConfig {
                command,
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
                filenames: filenames.into_iter().map(DisplayPath::from).collect(),
                has_chat,
                staging_dir: Default::default(),
                paths_mode: false,
                source_paths: Vec::new(),
                output_paths: Vec::new(),
                before_paths: Vec::new(),
                media_mapping: Default::default(),
                media_subdir: Default::default(),
                source_dir: Default::default(),
            },
            execution: JobExecutionState {
                status: JobStatus::Queued,
                file_statuses,
                results: Vec::new(),
                error: None,
                completed_files: 0,
            },
            schedule: JobScheduleState {
                submitted_at: MachineTime::now(),
                completed_at: None,
                next_eligible_at: None,
                num_workers: None,
                lease: None,
                last_cancel: None,
            },
            runtime: JobRuntimeControl {
                cancel_token: CancellationToken::new(),
                runner_active: false,
                run_generation: crate::store::RunGeneration::FIRST,
            },
            execution_plan: None,
        }
    }

    #[tokio::test]
    async fn submit_and_get() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::new(
            test_config(),
            None,
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );

        let job = make_job("j1", ReleasedCommand::Morphotag, vec!["a.cha".into()]);
        store.submit(job).await.unwrap();

        let info = store.get(&JobId::from("j1")).await;
        assert!(info.is_some());
        assert_eq!(info.unwrap().command, "morphotag");
    }

    #[tokio::test]
    async fn conflict_detection() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::new(
            test_config(),
            None,
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );

        let job1 = make_job("j1", ReleasedCommand::Morphotag, vec!["a.cha".into()]);
        store.submit(job1).await.unwrap();

        let job2 = make_job("j2", ReleasedCommand::Align, vec!["a.cha".into()]);
        let result = store.submit(job2).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn cancel_job() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::new(
            test_config(),
            None,
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );

        let job = make_job("j1", ReleasedCommand::Morphotag, vec!["a.cha".into()]);
        store.submit(job).await.unwrap();

        store
            .cancel(&JobId::from("j1"), CancellationRequest::default())
            .await
            .unwrap();
        let info = store.get(&JobId::from("j1")).await.unwrap();
        assert_eq!(info.status, JobStatus::Cancelled);
    }

    #[tokio::test]
    async fn delete_completed_job() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::new(
            test_config(),
            None,
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );

        let mut job = make_job("j1", ReleasedCommand::Morphotag, vec!["a.cha".into()]);
        job.execution.status = JobStatus::Completed;
        store.submit(job).await.unwrap();

        store.delete(&JobId::from("j1")).await.unwrap();
        assert!(store.get(&JobId::from("j1")).await.is_none());
    }

    #[tokio::test]
    async fn submit_persists_job_without_relocking_store_state() {
        let (store, db, _dir) = test_store_with_db().await;

        let job = make_job("j1", ReleasedCommand::Morphotag, vec!["a.cha".into()]);
        store.submit(job).await.unwrap();

        let jobs = db.load_all_jobs().await.unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].job_id, "j1");
        assert_eq!(jobs[0].status, "queued");
    }

    #[tokio::test]
    async fn cancel_persists_status_after_releasing_store_lock() {
        let (store, db, _dir) = test_store_with_db().await;

        let job = make_job("j1", ReleasedCommand::Morphotag, vec!["a.cha".into()]);
        store.submit(job).await.unwrap();
        store
            .cancel(&JobId::from("j1"), CancellationRequest::default())
            .await
            .unwrap();

        let jobs = db.load_all_jobs().await.unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].status, "cancelled");
        assert!(jobs[0].completed_at.is_some());
    }

    #[tokio::test]
    async fn delete_removes_db_row_and_staging_dir_after_unlocking_store() {
        let (store, db, dir) = test_store_with_db().await;
        let staging_dir = dir.path().join("staging-job");
        std::fs::create_dir_all(&staging_dir).unwrap();
        std::fs::write(staging_dir.join("artifact.txt"), "artifact").unwrap();

        let mut job = make_job("j1", ReleasedCommand::Morphotag, vec!["a.cha".into()]);
        job.execution.status = JobStatus::Completed;
        job.filesystem.staging_dir = batchalign_types::paths::ServerPath::from(staging_dir.clone());
        store.submit(job).await.unwrap();

        store.delete(&JobId::from("j1")).await.unwrap();

        let jobs = db.load_all_jobs().await.unwrap();
        assert!(jobs.is_empty());
        assert!(!staging_dir.exists());
    }

    #[tokio::test]
    async fn list_all_ordered() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::new(
            test_config(),
            None,
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );

        let mut j1 = make_job("j1", ReleasedCommand::Morphotag, vec!["a.cha".into()]);
        j1.schedule.submitted_at = crate::unix_time(100.0);
        j1.execution.status = JobStatus::Completed;
        store.submit(j1).await.unwrap();

        let mut j2 = make_job("j2", ReleasedCommand::Align, vec!["b.cha".into()]);
        j2.schedule.submitted_at = crate::unix_time(200.0);
        j2.execution.status = JobStatus::Completed;
        store.submit(j2).await.unwrap();

        let items = store.list_all().await;
        assert_eq!(items.len(), 2);
        // Newest first
        assert_eq!(items[0].job_id, "j2");
        assert_eq!(items[1].job_id, "j1");
    }

    #[tokio::test]
    async fn claim_ready_queued_jobs_orders_by_submission_time() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::new(
            test_config(),
            None,
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );

        let mut early = make_job("j1", ReleasedCommand::Morphotag, vec!["a.cha".into()]);
        early.schedule.submitted_at = crate::unix_time(100.0);
        store.submit(early).await.unwrap();

        let mut late = make_job("j2", ReleasedCommand::Align, vec!["b.cha".into()]);
        late.schedule.submitted_at = crate::unix_time(200.0);
        store.submit(late).await.unwrap();

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

    #[tokio::test]
    async fn claim_ready_queued_jobs_skips_deferred_and_reports_next_wake() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::new(
            test_config(),
            None,
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );

        let mut ready = make_job("ready", ReleasedCommand::Morphotag, vec!["a.cha".into()]);
        ready.schedule.submitted_at = crate::unix_time(100.0);
        store.submit(ready).await.unwrap();

        let deferred_at = MachineTime::now().plus(std::time::Duration::from_secs(60));
        let mut deferred = make_job("deferred", ReleasedCommand::Align, vec!["b.cha".into()]);
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

    #[tokio::test]
    async fn claim_ready_queued_jobs_skips_unexpired_leases_and_reclaims_expired_ones() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::new(
            test_config(),
            None,
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );

        let now = MachineTime::now();

        let mut leased = make_job("leased", ReleasedCommand::Morphotag, vec!["a.cha".into()]);
        leased.schedule.lease = Some(
            LeaseRecord::new(
                "other-node".into(),
                now.minus(std::time::Duration::from_secs(10)),
                now.plus(std::time::Duration::from_secs(120)),
            )
            .expect("an ordered fixture lease"),
        );
        store.submit(leased).await.unwrap();

        let mut expired = make_job("expired", ReleasedCommand::Align, vec!["b.cha".into()]);
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

    #[tokio::test]
    async fn release_runner_claim_makes_job_eligible_again() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::new(
            test_config(),
            None,
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );

        let job = make_job("j1", ReleasedCommand::Morphotag, vec!["a.cha".into()]);
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

    #[tokio::test]
    async fn renew_job_lease_updates_heartbeat_and_expiry_for_local_claim() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::new(
            test_config(),
            None,
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );

        let job = make_job("j1", ReleasedCommand::Morphotag, vec!["a.cha".into()]);
        store.submit(job).await.unwrap();
        let _ = store.claim_ready_queued_jobs().await;

        let before = store
            .registry
            .lease_state(&JobId::from("j1"))
            .await
            .unwrap()
            .expect("claimed")
            .heartbeat_at();

        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
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
        assert!(lease.heartbeat_at() >= before);
    }

    #[tokio::test]
    async fn renew_job_lease_stops_after_claim_is_released() {
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = JobStore::new(
            test_config(),
            None,
            tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        );

        let job = make_job("j1", ReleasedCommand::Morphotag, vec!["a.cha".into()]);
        store.submit(job).await.unwrap();
        let _ = store.claim_ready_queued_jobs().await;
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
