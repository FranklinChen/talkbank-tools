//! SQLite persistence layer: port of `batchalign/serve/job_db.py`.
//!
//! Write-through SQLite database at the runtime-owned `jobs.db` path under the
//! resolved state root. Called by
//! `JobStore` at each state transition for crash recovery. Uses WAL mode
//! with `busy_timeout` for safe concurrent access.
//!
//! All DB operations are natively async via `sqlx::SqlitePool`.

mod insert;
mod query;
mod recovery;
mod schema;
mod update;

pub use insert::NewJobRecord;
pub use query::{CancellationRow, StoredLeaseError};
pub use recovery::PrunedJob;
pub use schema::{AttemptRow, FileStatusRow, JobRow};

use std::path::{Path, PathBuf};

use sqlx::SqlitePool;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous};

use crate::config::RuntimeLayout;
use crate::error::ServerError;

// ---------------------------------------------------------------------------
// JobDB
// ---------------------------------------------------------------------------

/// Write-through SQLite layer for job persistence.
///
/// Thread-safe via `SqlitePool` (connection pooling).
pub struct JobDB {
    pool: SqlitePool,
    db_path: PathBuf,
}

impl JobDB {
    /// Exercise persistence/recovery with real migrations but no filesystem state.
    #[cfg(test)]
    pub(crate) async fn in_memory_for_test() -> Result<Self, ServerError> {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(
                SqliteConnectOptions::new()
                    .in_memory(true)
                    .foreign_keys(true),
            )
            .await?;
        sqlx::migrate!("./migrations").run(&pool).await?;
        Ok(Self {
            pool,
            db_path: PathBuf::from(":memory:"),
        })
    }

    /// Open (or create) the job database and run migrations.
    pub async fn open(db_dir: Option<&Path>) -> Result<Self, ServerError> {
        let layout = RuntimeLayout::from_env();
        Self::open_with_layout(&layout, db_dir).await
    }

    /// Open the job database using an explicit runtime layout for the default
    /// state root.
    pub async fn open_with_layout(
        layout: &RuntimeLayout,
        db_dir: Option<&Path>,
    ) -> Result<Self, ServerError> {
        let db_dir = match db_dir {
            Some(d) => d.to_path_buf(),
            None => layout.state_dir().to_path_buf(),
        };
        std::fs::create_dir_all(&db_dir).map_err(|e| {
            ServerError::Io(std::io::Error::new(
                e.kind(),
                format!("creating db dir {}: {e}", db_dir.display()),
            ))
        })?;
        let db_path = db_dir.join("jobs.db");

        let options = SqliteConnectOptions::new()
            .filename(&db_path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(std::time::Duration::from_millis(10_000))
            .synchronous(SqliteSynchronous::Normal)
            .foreign_keys(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await?;

        sqlx::migrate!("./migrations").run(&pool).await?;

        Ok(Self { pool, db_path })
    }

    /// The path to the database file.
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::MachineTime;
    use crate::error::ServerError;
    use crate::options::{
        AlignOptions, CommandOptions, CommonOptions, FaEngineName, MorphotagOptions,
    };
    use crate::scheduling::{AttemptOutcome, FailureCategory, RetryDisposition, WorkUnitKind};
    use crate::worker::WorkerPid;

    fn morphotag_options() -> CommandOptions {
        CommandOptions::Morphotag(MorphotagOptions {
            common: CommonOptions::default(),

            ..Default::default()
        })
    }

    fn align_options() -> CommandOptions {
        CommandOptions::Align(AlignOptions {
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
        })
    }

    /// Build a new queued job insert payload for DB tests.
    fn make_job_record(
        job_id: &str,
        command: &str,
        options: CommandOptions,
        filenames: Vec<String>,
        has_chat: Vec<bool>,
    ) -> NewJobRecord {
        NewJobRecord {
            job_id: job_id.to_string(),
            correlation_id: job_id.to_string(),
            command: command.to_string(),
            lang: "eng".to_string(),
            num_speakers: 1,
            status: crate::api::JobStatus::Queued,
            staging_dir: "/tmp/staging".to_string(),
            filenames,
            has_chat,
            options,
            media_mapping: String::new(),
            media_subdir: String::new(),
            source_dir: String::new(),
            submitter: Some(crate::store::Submitter::client(
                std::net::Ipv4Addr::LOCALHOST.into(),
                "localhost".into(),
            )),
            submitted_at: crate::unix_time(1700000000.0),
            paths_mode: false,
            source_paths: Vec::new(),
            output_paths: Vec::new(),
        }
    }

    async fn test_db() -> (JobDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db = JobDB::open(Some(dir.path())).await.unwrap();
        (db, dir)
    }

    #[tokio::test]
    async fn schema_creation() {
        let dir = tempfile::tempdir().unwrap();
        let _db = JobDB::open(Some(dir.path())).await.unwrap();
        // Second open should be idempotent (migrations already applied)
        let _db2 = JobDB::open(Some(dir.path())).await.unwrap();
    }

    #[tokio::test]
    async fn insert_and_load_roundtrip() {
        let (db, _dir) = test_db().await;
        let record = make_job_record(
            "job1",
            "morphotag",
            morphotag_options(),
            vec!["test.cha".to_string()],
            vec![true],
        );

        db.insert_job(&record).await.unwrap();

        let jobs = db.load_all_jobs().await.unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].job_id, "job1");
        assert_eq!(jobs[0].correlation_id, "job1");
        assert_eq!(jobs[0].command, "morphotag");
        assert_eq!(jobs[0].filenames, vec!["test.cha"]);
        assert_eq!(jobs[0].file_statuses.len(), 1);
        assert_eq!(jobs[0].file_statuses[0].status, "queued");
    }

    /// A job's status row is written whole: a failed job requeued without an
    /// error is stored without the old one, and the worker count and times
    /// are what the job holds, NULLs included.
    #[tokio::test]
    async fn write_job_status_replaces_every_status_column() {
        use crate::store::JobStatusColumns;
        let (db, _dir) = test_db().await;
        let record = make_job_record(
            "job1",
            "morphotag",
            morphotag_options(),
            vec!["f.cha".into()],
            vec![true],
        );
        db.insert_job(&record).await.unwrap();

        let failed = JobStatusColumns::for_test(
            crate::api::JobStatus::Failed,
            Some("worker crashed".into()),
            Some(crate::unix_time(5.0)),
            Some(4),
            None,
        );
        db.write_job_status("job1", &failed).await.unwrap();
        let jobs = db.load_all_jobs().await.unwrap();
        assert_eq!(jobs[0].status, "failed");
        assert_eq!(jobs[0].error.as_deref(), Some("worker crashed"));
        assert_eq!(jobs[0].num_workers, Some(4));

        let requeued = JobStatusColumns::for_test(
            crate::api::JobStatus::Queued,
            None,
            None,
            None,
            Some(crate::unix_time(9.0)),
        );
        db.write_job_status("job1", &requeued).await.unwrap();
        let jobs = db.load_all_jobs().await.unwrap();
        assert_eq!(jobs[0].status, "queued");
        assert_eq!(jobs[0].error, None, "the old error must not survive");
        assert_eq!(jobs[0].completed_at, None);
        assert_eq!(jobs[0].num_workers, None);
        assert_eq!(jobs[0].next_eligible_at, Some(crate::unix_time(9.0)));
    }

    #[tokio::test]
    async fn update_file_status() {
        let (db, _dir) = test_db().await;
        let mut record = make_job_record(
            "job1",
            "align",
            align_options(),
            vec!["a.cha".into()],
            vec![true],
        );
        record.status = crate::api::JobStatus::Running;
        record.staging_dir = "/tmp".to_string();

        db.insert_job(&record).await.unwrap();

        db.update_file_status(
            "job1",
            "a.cha",
            &crate::store::FilePhase::Done {
                exclusions: Vec::new(),
                started_at: Some(crate::unix_time(1700000001.0)),
                finished_at: Some(crate::unix_time(1700000005.0)),
            },
            Some("chat"),
        )
        .await
        .unwrap();

        let jobs = db.load_all_jobs().await.unwrap();
        assert_eq!(jobs[0].file_statuses[0].status, "done");
        assert_eq!(jobs[0].file_statuses[0].content_type, "chat");
    }

    /// A written file's exclusions survive the database: written as their
    /// own column by the one phase writer and read back, through the
    /// recovery boundary, into exactly the phase that was written. A phase
    /// without any writes NULL, so a later failure leaves none behind.
    #[tokio::test]
    async fn a_done_files_exclusions_round_trip_through_their_column() {
        use crate::api::{ExcludedUtteranceRecord, OffRecordPostcode, OutputExclusionRecord};
        use crate::store::{FileFailure, FilePhase};

        let (db, _dir) = test_db().await;
        let mut record = make_job_record(
            "job1",
            "align",
            align_options(),
            vec!["a.cha".into()],
            vec![true],
        );
        record.status = crate::api::JobStatus::Running;
        db.insert_job(&record).await.unwrap();

        let done = FilePhase::Done {
            started_at: Some(crate::unix_time(1.0)),
            finished_at: Some(crate::unix_time(2.0)),
            exclusions: vec![OutputExclusionRecord::NotInRecording {
                excluded_utterances: 1,
                excluded_words: 4,
                first_excluded: vec![ExcludedUtteranceRecord {
                    utterance: 2,
                    words: 4,
                    postcode: OffRecordPostcode::Diary,
                }],
            }],
        };
        db.update_file_status("job1", "a.cha", &done, Some("chat"))
            .await
            .unwrap();
        let rows = db.load_file_status_rows("job1").await.unwrap();
        assert!(rows[0].exclusions.is_some(), "the column holds them");
        let crate::store::queries::RecoveredFilePhase::Exact(read) =
            crate::store::queries::recover_file_phase("job1", &rows[0])
        else {
            panic!("a row this build wrote reads back exactly");
        };
        assert_eq!(read, done);

        let failed = FilePhase::Error {
            started_at: Some(crate::unix_time(3.0)),
            finished_at: Some(crate::unix_time(4.0)),
            failure: FileFailure::recorded("refused".into(), FailureCategory::Validation),
        };
        db.update_file_status("job1", "a.cha", &failed, None)
            .await
            .unwrap();
        let rows = db.load_file_status_rows("job1").await.unwrap();
        assert_eq!(rows[0].exclusions, None, "an error owns no exclusions");
    }

    /// A file that failed and then succeeded is stored `done` with no error.
    /// The writer used to `COALESCE` the error columns, so the old error
    /// survived the success and every later load reported (and dropped) it.
    /// Every column a phase does not own is now written NULL, and the row
    /// reads back as exactly the phase that was written.
    #[tokio::test]
    async fn a_phase_write_clears_every_column_the_phase_does_not_own() {
        use crate::store::{FileFailure, FilePhase};

        let (db, _dir) = test_db().await;
        let mut record = make_job_record(
            "job1",
            "align",
            align_options(),
            vec!["a.cha".into()],
            vec![true],
        );
        record.status = crate::api::JobStatus::Running;
        db.insert_job(&record).await.unwrap();

        let retry = FilePhase::RetryPending {
            started_at: Some(crate::unix_time(1.0)),
            failed_at: Some(crate::unix_time(2.0)),
            retry_at: crate::unix_time(3.0),
            failure: FileFailure::recorded("worker crashed".into(), FailureCategory::WorkerCrash),
        };
        db.update_file_status("job1", "a.cha", &retry, None)
            .await
            .unwrap();
        let done = FilePhase::Done {
            exclusions: Vec::new(),
            started_at: Some(crate::unix_time(4.0)),
            finished_at: Some(crate::unix_time(5.0)),
        };
        db.update_file_status("job1", "a.cha", &done, Some("chat"))
            .await
            .unwrap();

        let jobs = db.load_all_jobs().await.unwrap();
        let row = &jobs[0].file_statuses[0];
        assert_eq!(row.status, "done");
        assert_eq!(row.error, None);
        assert_eq!(row.error_category, None);
        assert_eq!(row.next_eligible_at, None);
        let (phase, dropped) = FilePhase::from_row(crate::store::FilePhaseColumns {
            exclusions: None,
            status: crate::api::FileStatusKind::Done,
            error: row.error.as_deref(),
            error_category: None,
            diagnostics: None,
            started_at: row.started_at,
            finished_at: row.finished_at,
            next_eligible_at: row.next_eligible_at,
        });
        assert_eq!(phase, done);
        assert_eq!(dropped, None, "nothing stale is left to drop");
    }

    #[tokio::test]
    async fn attempt_roundtrip() {
        let (db, _dir) = test_db().await;
        let mut record = make_job_record(
            "job1",
            "morphotag",
            morphotag_options(),
            vec!["a.cha".into()],
            vec![true],
        );
        record.status = crate::api::JobStatus::Running;
        record.staging_dir = "/tmp".to_string();

        db.insert_job(&record).await.unwrap();

        let (attempt_id, attempt_number) = db
            .insert_attempt_start(
                "job1",
                "a.cha",
                WorkUnitKind::FileProcess,
                crate::unix_time(1700000001.0),
                Some("node-a"),
                Some(4321),
            )
            .await
            .unwrap();
        assert_eq!(attempt_id, "job1:a.cha:1");
        assert_eq!(attempt_number, 1);

        db.finish_attempt(
            &attempt_id,
            AttemptOutcome::Failed,
            Some(FailureCategory::WorkerCrash),
            RetryDisposition::TerminalFailure,
            crate::unix_time(1700000002.5),
        )
        .await
        .unwrap();

        let attempts = db.load_attempts_for_job("job1").await.unwrap();
        assert_eq!(attempts.len(), 1);
        assert_eq!(&*attempts[0].attempt_id, attempt_id);
        assert_eq!(attempts[0].job_id, "job1");
        assert_eq!(attempts[0].work_unit_id, "a.cha");
        assert_eq!(attempts[0].work_unit_kind, WorkUnitKind::FileProcess);
        assert_eq!(attempts[0].attempt_number, 1);
        assert_eq!(attempts[0].outcome, AttemptOutcome::Failed);
        assert_eq!(
            attempts[0].failure_category,
            Some(FailureCategory::WorkerCrash)
        );
        assert_eq!(attempts[0].disposition, RetryDisposition::TerminalFailure);
        assert_eq!(attempts[0].worker_node_id.as_deref(), Some("node-a"));
        assert_eq!(attempts[0].worker_pid, Some(WorkerPid(4321)));
        assert_eq!(
            attempts[0].finished_at,
            Some(crate::unix_time(1700000002.5))
        );
    }

    #[tokio::test]
    async fn cascade_delete() {
        let (db, _dir) = test_db().await;
        let mut record = make_job_record(
            "job1",
            "morphotag",
            morphotag_options(),
            vec!["a.cha".into(), "b.cha".into()],
            vec![true, true],
        );
        record.status = crate::api::JobStatus::Completed;
        record.staging_dir = "/tmp".to_string();

        db.insert_job(&record).await.unwrap();

        db.delete_job(&crate::api::JobId::from("job1"))
            .await
            .unwrap();
        let jobs = db.load_all_jobs().await.unwrap();
        assert!(jobs.is_empty());
    }

    #[tokio::test]
    async fn recover_interrupted() {
        let (db, _dir) = test_db().await;

        // Insert a running job
        let mut running = make_job_record(
            "job1",
            "align",
            align_options(),
            vec!["a.cha".into()],
            vec![true],
        );
        running.status = crate::api::JobStatus::Running;
        running.staging_dir = "/tmp".to_string();

        db.insert_job(&running).await.unwrap();
        db.seed_file_status_row(
            "job1",
            "a.cha",
            "processing",
            None,
            None,
            None,
            Some(crate::unix_time(1700000001.0)),
            None,
            None,
        )
        .await
        .unwrap();

        // Insert a queued job
        let mut queued = make_job_record(
            "job2",
            "morphotag",
            morphotag_options(),
            vec!["b.cha".into()],
            vec![true],
        );
        queued.staging_dir = "/tmp2".to_string();
        queued.submitted_at = crate::unix_time(1700000002.0);

        db.insert_job(&queued).await.unwrap();

        // Insert a completed job (should NOT be interrupted)
        let mut completed = make_job_record(
            "job3",
            "morphotag",
            morphotag_options(),
            vec!["c.cha".into()],
            vec![true],
        );
        completed.status = crate::api::JobStatus::Completed;
        completed.staging_dir = "/tmp3".to_string();
        completed.submitted_at = crate::unix_time(1700000003.0);

        db.insert_job(&completed).await.unwrap();

        let interrupted = db.recover_interrupted(MachineTime::now()).await.unwrap();
        assert_eq!(interrupted.len(), 2);
        assert!(interrupted.contains(&crate::api::JobId::from("job1")));
        assert!(interrupted.contains(&crate::api::JobId::from("job2")));

        let jobs = db.load_all_jobs().await.unwrap();
        for job in &jobs {
            if job.job_id == "job1" || job.job_id == "job2" {
                assert_eq!(job.status, "interrupted");
            }
            if job.job_id == "job3" {
                assert_eq!(job.status, "completed");
            }
        }
    }

    /// A file row holds the columns its phase and output own, and nothing
    /// that is never written (`bug_report_id` was always NULL).
    #[tokio::test]
    async fn file_status_rows_have_no_unwritten_columns() {
        let (db, _dir) = test_db().await;
        let columns: Vec<String> =
            sqlx::query_scalar("SELECT name FROM pragma_table_info('file_statuses')")
                .fetch_all(&db.pool)
                .await
                .unwrap();
        assert!(
            !columns.iter().any(|name| name == "bug_report_id"),
            "{columns:?}"
        );
    }

    /// Startup recovery leaves a job row as a shutdown leaves a live job
    /// (`Job::interrupt_for_shutdown`): interrupted, completed at the
    /// recovery time, no pending retry time; its error and worker count
    /// kept. The recovery UPDATE wrote two columns by hand and kept the retry
    /// time, so the two paths left different rows for one transition.
    #[tokio::test]
    async fn recovery_writes_the_interrupted_status_image() {
        use crate::store::JobStatusColumns;
        let (db, _dir) = test_db().await;
        let record = make_job_record(
            "retrying",
            "morphotag",
            morphotag_options(),
            vec!["a.cha".into()],
            vec![true],
        );
        db.insert_job(&record).await.unwrap();
        db.write_job_status(
            "retrying",
            &JobStatusColumns::for_test(
                crate::api::JobStatus::Queued,
                Some("worker crashed".into()),
                None,
                Some(2),
                Some(crate::unix_time(1700000100.0)),
            ),
        )
        .await
        .unwrap();

        let now = crate::unix_time(1700000050.0);
        db.recover_interrupted(now).await.unwrap();

        let jobs = db.load_all_jobs().await.unwrap();
        let job = &jobs[0];
        assert_eq!(job.status, "interrupted");
        assert_eq!(job.completed_at, Some(now));
        assert_eq!(job.next_eligible_at, None, "no retry time survives");
        assert_eq!(job.error.as_deref(), Some("worker crashed"));
        assert_eq!(job.num_workers, Some(2));
    }

    /// Interrupting a job and its files is one transaction: when a file's
    /// write fails, the job row is not left interrupted with its file still
    /// in flight.
    #[tokio::test]
    async fn interrupting_a_job_and_its_files_is_one_transaction() {
        let (db, _dir) = test_db().await;
        let mut record = make_job_record(
            "in-flight",
            "morphotag",
            morphotag_options(),
            vec!["a.cha".into()],
            vec![true],
        );
        record.status = crate::api::JobStatus::Running;
        db.insert_job(&record).await.unwrap();
        db.fail_file_status_updates().await.unwrap();

        db.recover_interrupted(crate::unix_time(1700000050.0))
            .await
            .expect_err("the file's write fails");

        let jobs = db.load_all_jobs().await.unwrap();
        assert_eq!(jobs[0].status, "running", "the job's write is rolled back");
        assert_eq!(jobs[0].file_statuses[0].status, "queued");
    }

    /// A row holding part of a lease (an owner without an expiry) is refused
    /// where it is read, naming the job and the columns, never repaired.
    #[tokio::test]
    async fn a_row_with_part_of_a_lease_is_refused_on_load() {
        let (db, _dir) = test_db().await;
        let job = make_job_record(
            "half-lease",
            "morphotag",
            morphotag_options(),
            vec!["a.cha".into()],
            vec![true],
        );
        db.insert_job(&job).await.unwrap();
        sqlx::query("UPDATE jobs SET leased_by_node = 'node-a' WHERE job_id = 'half-lease'")
            .execute(&db.pool)
            .await
            .unwrap();

        let error = db.load_all_jobs().await.unwrap_err();
        assert!(
            matches!(
                &error,
                ServerError::StoredLease(StoredLeaseError::Partial { job_id, node, expires_at: None, heartbeat_at: None })
                    if job_id == "half-lease" && node.as_deref() == Some("node-a")
            ),
            "{error}"
        );
    }

    /// A row whose lease expires no later than its heartbeat names a lease
    /// that could not have been held, and is refused through the same
    /// constructor deserialization uses.
    #[tokio::test]
    async fn a_row_with_a_lease_expiring_before_its_heartbeat_is_refused_on_load() {
        let (db, _dir) = test_db().await;
        let job = make_job_record(
            "inverted-lease",
            "morphotag",
            morphotag_options(),
            vec!["a.cha".into()],
            vec![true],
        );
        db.insert_job(&job).await.unwrap();
        sqlx::query(
            "UPDATE jobs SET leased_by_node = 'node-a', lease_heartbeat_at = 50.0, \
             lease_expires_at = 40.0 WHERE job_id = 'inverted-lease'",
        )
        .execute(&db.pool)
        .await
        .unwrap();

        let error = db.load_all_jobs().await.unwrap_err();
        assert!(
            matches!(
                &error,
                ServerError::StoredLease(StoredLeaseError::Unordered { job_id, .. })
                    if job_id == "inverted-lease"
            ),
            "{error}"
        );
    }

    /// A stored time that names no instant is refused where the row is read,
    /// naming its column and value, instead of becoming a job's time.
    #[tokio::test]
    async fn a_stored_time_that_names_no_instant_is_refused_on_load() {
        let (db, _dir) = test_db().await;
        let job = make_job_record(
            "bad-time",
            "morphotag",
            morphotag_options(),
            vec!["a.cha".into()],
            vec![true],
        );
        db.insert_job(&job).await.unwrap();
        sqlx::query("UPDATE jobs SET submitted_at = 1e300 WHERE job_id = 'bad-time'")
            .execute(&db.pool)
            .await
            .unwrap();

        let error = db.load_all_jobs().await.unwrap_err().to_string();
        assert!(error.contains("submitted_at"), "{error}");
        assert!(error.contains("names no instant"), "{error}");
    }

    #[tokio::test]
    async fn prune_expired() {
        let (db, _dir) = test_db().await;

        // Insert an old job (submitted 10 days ago)
        let old_time = MachineTime::now().minus(std::time::Duration::from_secs(10 * 86_400));
        let mut old_job = make_job_record(
            "old",
            "morphotag",
            morphotag_options(),
            vec!["a.cha".into()],
            vec![true],
        );
        old_job.status = crate::api::JobStatus::Completed;
        old_job.staging_dir = "/tmp/old".to_string();
        old_job.submitted_at = old_time;

        db.insert_job(&old_job).await.unwrap();

        // Insert a recent job
        let mut new_job = make_job_record(
            "new",
            "morphotag",
            morphotag_options(),
            vec!["b.cha".into()],
            vec![true],
        );
        new_job.status = crate::api::JobStatus::Completed;
        new_job.staging_dir = "/tmp/new".to_string();
        new_job.submitted_at = MachineTime::now();

        db.insert_job(&new_job).await.unwrap();

        let dirs = db
            .prune_expired(
                crate::config::JobTtlDays::literal::<7>(),
                MachineTime::now(),
            )
            .await
            .unwrap();
        assert_eq!(
            dirs,
            vec![PrunedJob {
                job_id: crate::api::JobId::from("old"),
                staging_dir: batchalign_types::paths::ServerPath::from("/tmp/old"),
            }]
        );

        let jobs = db.load_all_jobs().await.unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].job_id, "new");
    }

    #[tokio::test]
    async fn paths_mode_roundtrip() {
        let (db, _dir) = test_db().await;
        let mut record = make_job_record(
            "job1",
            "morphotag",
            morphotag_options(),
            vec!["a.cha".into()],
            vec![true],
        );
        record.staging_dir = "/tmp".to_string();
        record.source_dir = "/src".to_string();
        record.paths_mode = true;
        record.source_paths = vec!["/src/a.cha".into()];
        record.output_paths = vec!["/out/a.cha".into()];
        record.submitter = Some(crate::store::Submitter::client(
            std::net::Ipv4Addr::LOCALHOST.into(),
            String::new(),
        ));

        db.insert_job(&record).await.unwrap();

        let jobs = db.load_all_jobs().await.unwrap();
        assert!(jobs[0].paths_mode);
        assert_eq!(jobs[0].source_paths, vec!["/src/a.cha"]);
        assert_eq!(jobs[0].output_paths, vec!["/out/a.cha"]);
    }

    // -----------------------------------------------------------------------
    // T097: Recovery evidence preservation tests
    // -----------------------------------------------------------------------

    /// Recovery must NOT erase error messages on files that already failed before
    /// the server crash. A file that was in `"error"` state with a recorded error
    /// message must keep that evidence after `recover_interrupted()`.
    #[tokio::test]
    async fn recovery_preserves_completed_file_errors() {
        let (db, _dir) = test_db().await;

        // Insert a running job with 2 files: one already errored, one still processing.
        let mut job = make_job_record(
            "evidence-job",
            "morphotag",
            morphotag_options(),
            vec!["ok.cha".into(), "broken.cha".into()],
            vec![true, true],
        );
        job.status = crate::api::JobStatus::Running;
        db.insert_job(&job).await.unwrap();

        // File "broken.cha" already failed before the crash.
        db.seed_file_status_row(
            "evidence-job",
            "broken.cha",
            "error",
            Some("Stanza crashed: CUDA OOM"),
            Some("worker_crash"),
            None,
            Some(crate::unix_time(1700000001.0)),
            Some(crate::unix_time(1700000002.0)),
            None,
        )
        .await
        .unwrap();

        // File "ok.cha" was still processing when the server crashed.
        db.seed_file_status_row(
            "evidence-job",
            "ok.cha",
            "processing",
            None,
            None,
            None,
            Some(crate::unix_time(1700000001.0)),
            None,
            None,
        )
        .await
        .unwrap();

        // Simulate server restart: run recovery.
        let interrupted = db.recover_interrupted(MachineTime::now()).await.unwrap();
        assert_eq!(interrupted, vec!["evidence-job"]);

        // Reload and verify evidence preservation.
        let jobs = db.load_all_jobs().await.unwrap();
        let job = &jobs[0];
        assert_eq!(job.status, "interrupted");

        // File "broken.cha" should STILL have its error message, recovery must
        // not overwrite "error" status files.
        let broken = job
            .file_statuses
            .iter()
            .find(|fs| fs.filename == "broken.cha")
            .expect("broken.cha should exist");
        assert_eq!(
            broken.error.as_deref(),
            Some("Stanza crashed: CUDA OOM"),
            "recovery must preserve pre-crash error messages"
        );
        assert_eq!(
            broken.error_category.as_deref(),
            Some("worker_crash"),
            "recovery must preserve error categories"
        );

        // File "ok.cha" should be marked interrupted (was still processing).
        let ok = job
            .file_statuses
            .iter()
            .find(|fs| fs.filename == "ok.cha")
            .expect("ok.cha should exist");
        assert_eq!(
            ok.status, "interrupted",
            "processing file should become interrupted after recovery"
        );
    }

    /// Recovery must preserve timing evidence (started_at, finished_at) on files
    /// that were already done or errored before the crash.
    #[tokio::test]
    async fn recovery_preserves_timing_evidence() {
        let (db, _dir) = test_db().await;

        let mut job = make_job_record(
            "timing-job",
            "align",
            align_options(),
            vec!["timed.cha".into(), "untimed.cha".into()],
            vec![true, true],
        );
        job.status = crate::api::JobStatus::Running;
        db.insert_job(&job).await.unwrap();

        // "timed.cha" completed successfully with timing.
        db.seed_file_status_row(
            "timing-job",
            "timed.cha",
            "done",
            None,
            None,
            None,
            Some(crate::unix_time(1700000010.0)),
            Some(crate::unix_time(1700000020.0)),
            None,
        )
        .await
        .unwrap();

        // "untimed.cha" was still queued.
        // (no update needed: default status is "queued")

        let interrupted = db.recover_interrupted(MachineTime::now()).await.unwrap();
        assert_eq!(interrupted.len(), 1);

        let jobs = db.load_all_jobs().await.unwrap();
        let job = &jobs[0];

        // "timed.cha" should keep its timing and "done" status, recovery only
        // touches queued/processing files.
        let timed = job
            .file_statuses
            .iter()
            .find(|fs| fs.filename == "timed.cha")
            .expect("timed.cha should exist");
        assert_eq!(timed.status, "done", "completed file should stay done");
        assert_eq!(
            timed.started_at,
            Some(crate::unix_time(1700000010.0)),
            "started_at must be preserved"
        );
        assert_eq!(
            timed.finished_at,
            Some(crate::unix_time(1700000020.0)),
            "finished_at must be preserved"
        );

        // "untimed.cha" was queued → should become interrupted.
        let untimed = job
            .file_statuses
            .iter()
            .find(|fs| fs.filename == "untimed.cha")
            .expect("untimed.cha should exist");
        assert_eq!(
            untimed.status, "interrupted",
            "queued file should become interrupted"
        );
    }

    /// A job with mixed file states (done + error + processing + queued) should
    /// have recovery only touch the in-flight files, leaving completed and errored
    /// files exactly as they were.
    #[tokio::test]
    async fn recovery_mixed_file_states() {
        let (db, _dir) = test_db().await;

        let mut job = make_job_record(
            "mixed-job",
            "morphotag",
            morphotag_options(),
            vec![
                "done.cha".into(),
                "error.cha".into(),
                "processing.cha".into(),
                "queued.cha".into(),
            ],
            vec![true, true, true, true],
        );
        job.status = crate::api::JobStatus::Running;
        db.insert_job(&job).await.unwrap();

        db.seed_file_status_row(
            "mixed-job",
            "done.cha",
            "done",
            None,
            None,
            None,
            Some(crate::unix_time(1700000001.0)),
            Some(crate::unix_time(1700000005.0)),
            None,
        )
        .await
        .unwrap();
        db.seed_file_status_row(
            "mixed-job",
            "error.cha",
            "error",
            Some("parse failed: missing @Begin"),
            Some("parse_error"),
            None,
            Some(crate::unix_time(1700000002.0)),
            Some(crate::unix_time(1700000003.0)),
            None,
        )
        .await
        .unwrap();
        db.seed_file_status_row(
            "mixed-job",
            "processing.cha",
            "processing",
            None,
            None,
            None,
            Some(crate::unix_time(1700000004.0)),
            None,
            None,
        )
        .await
        .unwrap();
        // "queued.cha" stays in default queued state.

        let interrupted = db.recover_interrupted(MachineTime::now()).await.unwrap();
        assert_eq!(interrupted, vec!["mixed-job"]);

        let jobs = db.load_all_jobs().await.unwrap();
        let job = &jobs[0];
        assert_eq!(job.status, "interrupted");

        // done.cha: untouched.
        let done = job
            .file_statuses
            .iter()
            .find(|fs| fs.filename == "done.cha")
            .unwrap();
        assert_eq!(done.status, "done");

        // error.cha: untouched, all evidence preserved.
        let errored = job
            .file_statuses
            .iter()
            .find(|fs| fs.filename == "error.cha")
            .unwrap();
        assert_eq!(errored.status, "error");
        assert_eq!(
            errored.error.as_deref(),
            Some("parse failed: missing @Begin")
        );
        assert_eq!(errored.error_category.as_deref(), Some("parse_error"));

        // processing.cha: marked interrupted.
        let processing = job
            .file_statuses
            .iter()
            .find(|fs| fs.filename == "processing.cha")
            .unwrap();
        assert_eq!(processing.status, "interrupted");

        // queued.cha: marked interrupted.
        let queued = job
            .file_statuses
            .iter()
            .find(|fs| fs.filename == "queued.cha")
            .unwrap();
        assert_eq!(queued.status, "interrupted");
    }
}
