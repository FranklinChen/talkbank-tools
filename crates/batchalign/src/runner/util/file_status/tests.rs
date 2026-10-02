//! Tests for file status tracking, supervision, and progress forwarding.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::api::MachineTime;
use crate::api::{
    ContentType, DisplayPath, FileProgressStage, FileStatusKind, JobId, JobStatus, LanguageCode3,
    LanguageSpec, NumSpeakers, ReleasedCommand,
};
use crate::db::JobDB;
use crate::options::{CommandOptions, CommonOptions, MorphotagOptions};
use crate::scheduling::{AttemptOutcome, FailureCategory, RetryDisposition, WorkUnitKind};
use crate::store::{
    FileStatus, Job, JobDispatchConfig, JobExecutionState, JobFilesystemConfig, JobIdentity,
    JobRuntimeControl, JobScheduleState, JobSourceContext,
};
use crate::ws::BROADCAST_CAPACITY;

use super::*;
use crate::store::JobStore;

use super::test_sink::{RecordedProgress, RecordingSink};

#[tokio::test]
async fn dropping_file_supervision_retires_the_child_task() {
    struct Retired(Option<tokio::sync::oneshot::Sender<()>>);
    impl Drop for Retired {
        fn drop(&mut self) {
            if let Some(sender) = self.0.take() {
                let _ = sender.send(());
            }
        }
    }
    let (started, ready) = tokio::sync::oneshot::channel();
    let (retired, done) = tokio::sync::oneshot::channel();
    let task = spawn_supervised_file_task(DisplayPath::from("a.cha"), "test", async move {
        let _guard = Retired(Some(retired));
        let _ = started.send(());
        std::future::pending::<FileTaskOutcome>().await
    });
    ready.await.expect("child started");
    drop(task);
    tokio::time::timeout(Duration::from_secs(1), done)
        .await
        .expect("losing the parent must not detach a running file task")
        .expect("child retired");
}

/// An event emitted inside a file task names its file. `tokio::spawn` starts a
/// future with no span, and the code that emits pipeline decisions does not
/// know its file, so without the span the spawn helper enters, a batch's
/// "needs review" lines could not be traced to a file. Checked on the text the
/// server log's formatter actually writes.
#[tokio::test]
async fn events_inside_a_file_task_name_the_file() {
    use std::sync::Mutex;

    /// Every line the formatter writes, kept for the assertion.
    #[derive(Clone, Default)]
    struct Captured(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for Captured {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .expect("capture lock")
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let captured = Captured::default();
    let writer = captured.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_target(false)
        .with_writer(move || writer.clone())
        .finish();
    // A current-thread test runtime runs the spawned task on this thread, so
    // the thread's default subscriber sees its events.
    let _default = tracing::subscriber::set_default(subscriber);

    let task = spawn_supervised_file_task(DisplayPath::from("lecture.cha"), "test", async {
        tracing::warn!("a decision that needs review");
        FileTaskOutcome::TerminalStateRecorded
    });
    let abnormal = drain_supervised_file_tasks(
        &RecordingSink::default(),
        &JobId::from("job-span"),
        &CancellationToken::new(),
        vec![task],
    )
    .await;
    assert_eq!(abnormal, 0, "the task recorded its own terminal state");

    let text = String::from_utf8(captured.0.lock().expect("capture lock").clone())
        .expect("the log is UTF-8");
    let line = text
        .lines()
        .find(|line| line.contains("a decision that needs review"))
        .expect("the event was logged");
    assert!(
        line.contains("file=lecture.cha"),
        "the event does not name its file: {line}"
    );
}

fn test_config() -> crate::config::ServerConfig {
    crate::config::ServerConfig {
        max_concurrent_jobs: Some(2),
        ..Default::default()
    }
}

fn make_job(id: &str) -> Job {
    let mut file_statuses = HashMap::new();
    file_statuses.insert(
        "a.cha".to_string(),
        FileStatus::new(DisplayPath::from("a.cha")),
    );

    Job {
        identity: JobIdentity {
            job_id: id.into(),
            correlation_id: format!("test-{id}").into(),
        },
        dispatch: JobDispatchConfig {
            command: ReleasedCommand::Morphotag,
            lang: LanguageSpec::Resolved(LanguageCode3::eng()),
            num_speakers: NumSpeakers(1),
            options: CommandOptions::Morphotag(MorphotagOptions {
                common: CommonOptions::default(),

                ..Default::default()
            }),
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
            filenames: vec![DisplayPath::from("a.cha")],
            has_chat: vec![true],
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
async fn progress_forwarder_routes_updates_through_sink_boundary() {
    let sink = Arc::new(RecordingSink::default());
    let job_id = JobId::from("job-progress");
    let tx = spawn_progress_forwarder(sink.clone(), job_id.clone(), "a.cha".to_string());

    tx.send(ProgressUpdate::new(FileStage::Writing, Some(1), Some(3)))
        .expect("send progress update");
    drop(tx);

    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if !sink.progress().is_empty() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("progress forwarder should flush");

    let progress = sink.progress();
    assert_eq!(
        progress.as_slice(),
        &[RecordedProgress {
            job_id,
            filename: "a.cha".to_string(),
            stage: FileStage::Writing,
            current: Some(1),
            total: Some(3),
        }]
    );
}

#[tokio::test]
async fn supervised_task_marks_non_terminal_exit_as_error() {
    let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
    let store = Arc::new(JobStore::new(
        test_config(),
        None,
        tx,
        std::sync::Arc::new(crate::clock::SystemClock),
    ));
    let sink = StoreRunnerEventSink::wrap(store.clone());
    let job_id = JobId::from("job-1");
    store.submit(make_job("job-1")).await.unwrap();

    sink.mark_file_processing(&job_id, "a.cha", sink.now())
        .await;

    let tasks = vec![spawn_supervised_file_task(
        DisplayPath::from("a.cha"),
        "test file task",
        async { FileTaskOutcome::MissingTerminalState },
    )];

    let abnormal =
        drain_supervised_file_tasks(sink.as_ref(), &job_id, &CancellationToken::new(), tasks).await;
    assert_eq!(abnormal, 1);

    let detail = store.get_job_detail(&job_id).await.unwrap();
    let file = detail
        .file_statuses
        .into_iter()
        .find(|entry| entry.filename == "a.cha")
        .unwrap();
    assert_eq!(file.status, FileStatusKind::Error);
    assert!(
        file.error
            .as_deref()
            .is_some_and(|msg| msg.contains("exited without recording a terminal file state"))
    );
}

#[tokio::test]
async fn supervised_task_marks_panic_as_error() {
    let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
    let store = Arc::new(JobStore::new(
        test_config(),
        None,
        tx,
        std::sync::Arc::new(crate::clock::SystemClock),
    ));
    let sink = StoreRunnerEventSink::wrap(store.clone());
    let job_id = JobId::from("job-2");
    store.submit(make_job("job-2")).await.unwrap();

    sink.mark_file_processing(&job_id, "a.cha", sink.now())
        .await;

    let tasks = vec![spawn_supervised_file_task(
        DisplayPath::from("a.cha"),
        "panic file task",
        async {
            panic!("boom");
        },
    )];

    let abnormal =
        drain_supervised_file_tasks(sink.as_ref(), &job_id, &CancellationToken::new(), tasks).await;
    assert_eq!(abnormal, 1);

    let detail = store.get_job_detail(&job_id).await.unwrap();
    let file = detail
        .file_statuses
        .into_iter()
        .find(|entry| entry.filename == "a.cha")
        .unwrap();
    assert_eq!(file.status, FileStatusKind::Error);
    assert!(
        file.error
            .as_deref()
            .is_some_and(|msg| msg.contains("panicked before recording a terminal file state"))
    );
}

/// The tracker stamps every event from the store's clock: the attempt starts
/// when the clock says, a retry deadline is that instant plus the backoff,
/// and the file's duration is the clock's elapsed time, never a time the
/// pipeline chose.
#[tokio::test]
async fn file_run_tracker_retries_then_completes_cleanly() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let db = Arc::new(JobDB::open(Some(tempdir.path())).await.expect("open db"));
    let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
    let start = crate::unix_time(1_700_000_000.0);
    let clock = Arc::new(crate::clock::ManualClock::at(start));
    let store = Arc::new(JobStore::new(
        test_config(),
        Some(db.clone()),
        tx,
        clock.clone(),
    ));
    let sink = StoreRunnerEventSink::wrap(store.clone());
    let job_id = JobId::from("job-tracker");
    store.submit(make_job("job-tracker")).await.unwrap();

    let lifecycle = FileRunTracker::new(sink.as_ref(), &job_id, "a.cha");
    lifecycle
        .begin_first_attempt(WorkUnitKind::FileProcess, FileStage::Reading)
        .await;

    clock.advance(std::time::Duration::from_secs(5));
    lifecycle
        .retry_after(
            std::time::Duration::from_secs(10),
            FailureCategory::ProviderTransient,
            "temporary failure",
        )
        .await;
    let pending = store.get_job_detail(&job_id).await.expect("job detail");
    let pending = pending
        .file_statuses
        .iter()
        .find(|entry| entry.filename == "a.cha")
        .expect("tracked file");
    assert_eq!(
        pending.next_eligible_at,
        Some(crate::unix_time(1_700_000_015.0)),
        "retry deadline = the failure's instant + backoff"
    );

    clock.advance(std::time::Duration::from_secs(10));
    lifecycle
        .restart_attempt(WorkUnitKind::FileProcess, FileStage::Processing)
        .await;

    clock.advance(std::time::Duration::from_secs(2));
    lifecycle
        .complete_with_result(DisplayPath::from("a.ana"), ContentType::Chat)
        .await;

    let detail = store.get_job_detail(&job_id).await.expect("job detail");
    let file = detail
        .file_statuses
        .into_iter()
        .find(|entry| entry.filename == "a.cha")
        .expect("tracked file");
    assert_eq!(file.status, FileStatusKind::Done);
    assert!(file.next_eligible_at.is_none());
    assert!(file.error.is_none());
    assert_eq!(file.finished_at, Some(crate::unix_time(1_700_000_017.0)));

    let attempts = db
        .load_attempts_for_job("job-tracker")
        .await
        .expect("load attempts");
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].outcome, AttemptOutcome::RetryableFailure);
    assert_eq!(attempts[0].disposition, RetryDisposition::Retry);
    assert_eq!(attempts[0].started_at, start);
    assert_eq!(
        attempts[0].finished_at,
        Some(crate::unix_time(1_700_000_005.0))
    );
    assert_eq!(attempts[1].outcome, AttemptOutcome::Succeeded);
    assert_eq!(attempts[1].disposition, RetryDisposition::Succeed);
    assert_eq!(attempts[1].started_at, crate::unix_time(1_700_000_015.0));
    assert_eq!(
        attempts[1].finished_at,
        Some(crate::unix_time(1_700_000_017.0))
    );
}

#[tokio::test]
async fn file_run_tracker_records_setup_failure() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let db = Arc::new(JobDB::open(Some(tempdir.path())).await.expect("open db"));
    let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
    let store = Arc::new(JobStore::new(
        test_config(),
        Some(db.clone()),
        tx,
        std::sync::Arc::new(crate::clock::SystemClock),
    ));
    let sink = StoreRunnerEventSink::wrap(store.clone());
    let job_id = JobId::from("job-setup-failure");
    store.submit(make_job("job-setup-failure")).await.unwrap();

    let lifecycle = FileRunTracker::new(sink.as_ref(), &job_id, "a.cha");
    lifecycle
        .record_setup_failure("media preflight failed", FailureCategory::Validation)
        .await;

    let detail = store.get_job_detail(&job_id).await.expect("job detail");
    let file = detail
        .file_statuses
        .into_iter()
        .find(|entry| entry.filename == "a.cha")
        .expect("tracked file");
    assert_eq!(file.status, FileStatusKind::Error);
    assert_eq!(file.error.as_deref(), Some("media preflight failed"));

    let attempts = db
        .load_attempts_for_job("job-setup-failure")
        .await
        .expect("load attempts");
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].work_unit_kind, WorkUnitKind::FileSetup);
    assert_eq!(attempts[0].outcome, AttemptOutcome::Failed);
    assert_eq!(
        attempts[0].failure_category,
        Some(FailureCategory::Validation)
    );
}

/// The runner-local stage vocabulary must map onto the stable API enum and its
/// labels without drift, since the labels are what users read.
///
/// This test previously also pinned `FileStage::for_batch_command`, a
/// command-to-initial-stage map whose sole caller was the retired batched-text
/// dispatch module. Those assertions went with it: a mapping nothing consumes
/// has no behaviour to pin, and its `_ => Processing` catch-all was the exact
/// shape that makes a new command silently take a wrong default. The
/// recipe-owned execution path sets its stages explicitly per recipe stage
/// instead.
#[test]
fn file_stage_maps_onto_the_stable_api_stage_and_its_labels() {
    assert_eq!(FileStage::Writing.api_stage().label(), "Writing");
    assert_eq!(
        FileStage::CheckingCache.api_stage().label(),
        "Checking cache"
    );
    assert_eq!(
        FileStage::PostProcessing.api_stage().label(),
        "Post-processing"
    );
    assert_eq!(FileStage::Aligning.api_stage(), FileProgressStage::Aligning);
    // `Parsing` had no producer at all until 2026-07-30 (declared in the API
    // enum, referenced only by the TUI colour map). The morphotag per-file task
    // now opens its attempt in it.
    assert_eq!(FileStage::Parsing.api_stage(), FileProgressStage::Parsing);
    assert_eq!(FileStage::Parsing.api_stage().label(), "Parsing");
    assert_eq!(
        FileStage::AnalyzingMorphosyntax.api_stage(),
        FileProgressStage::AnalyzingMorphosyntax
    );
    assert_eq!(
        FileStage::AnalyzingMorphosyntax.api_stage().label(),
        "Analyzing morphosyntax"
    );
}
