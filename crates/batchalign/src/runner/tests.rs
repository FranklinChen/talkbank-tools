use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use tokio::sync::broadcast;

use crate::api::{
    DisplayPath, JobId, JobStatus, LanguageCode3, LanguageSpec, NumSpeakers, ReleasedCommand,
};
use crate::db::JobDB;
use crate::options::{CommandOptions, CommonOptions, OpensmileOptions};
use crate::scheduling::{AttemptOutcome, FailureCategory, WorkUnitKind};
use crate::store::{
    FileStatus, Job, JobDispatchConfig, JobExecutionState, JobFilesystemConfig, JobIdentity,
    JobRuntimeControl, JobScheduleState, JobSourceContext, JobStore,
};
use crate::ws::BROADCAST_CAPACITY;

use super::util::StoreRunnerEventSink;
use super::{command_requires_chat_infer, record_preflight_media_failures};

/// Build a minimal paths-mode media job for prevalidation tests.
fn make_media_job(job_id: &str, source_path: &str) -> Job {
    let filename = "missing.wav";
    let mut file_statuses = HashMap::new();
    file_statuses.insert(
        filename.to_string(),
        FileStatus::new(DisplayPath::from(filename)),
    );

    Job {
        identity: JobIdentity {
            job_id: JobId::from(job_id),
            correlation_id: format!("test-{job_id}").into(),
        },
        dispatch: JobDispatchConfig {
            command: ReleasedCommand::Opensmile,
            lang: LanguageSpec::Resolved(LanguageCode3::eng()),
            num_speakers: NumSpeakers(1),
            options: CommandOptions::Opensmile(OpensmileOptions {
                common: CommonOptions::default(),
                feature_set: "eGeMAPSv02".into(),
            }),
            runtime_state: BTreeMap::new(),
            debug_traces: false,
        },
        source: JobSourceContext {
            submitter: Some(crate::store::Submitter::client(
                std::net::Ipv4Addr::LOCALHOST.into(),
                "localhost".into(),
            )),
            source_dir: Default::default(),
        },
        filesystem: JobFilesystemConfig {
            staging_dir: Default::default(),
            media_mapping: Default::default(),
            media_subdir: Default::default(),
        },
        execution: JobExecutionState {
            lifecycle: crate::store::JobLifecycle::for_test(
                JobStatus::Queued,
                crate::store::test_support::inputs_from_parts(
                    vec![DisplayPath::from(filename)],
                    vec![false],
                    true,
                    vec![batchalign_types::paths::ClientPath::new(source_path)],
                    // A paths-mode file always has somewhere to write.
                    vec![batchalign_types::paths::ClientPath::new(format!(
                        "{source_path}.csv"
                    ))],
                    Vec::new(),
                ),
            ),
            file_statuses,
            results: Vec::new(),
            error: None,
        },
        schedule: JobScheduleState {
            submitted_at: crate::unix_time(1.0),
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

#[test]
fn chat_backed_commands_require_chat_infer() {
    for cmd in [
        ReleasedCommand::Morphotag,
        ReleasedCommand::Utseg,
        ReleasedCommand::Translate,
        ReleasedCommand::Coref,
    ] {
        assert!(command_requires_chat_infer(cmd));
    }
}

#[test]
fn align_requires_chat_infer() {
    assert!(command_requires_chat_infer(ReleasedCommand::Align));
}

#[test]
fn audio_or_composed_commands_do_not_require_chat_infer() {
    assert!(!command_requires_chat_infer(ReleasedCommand::Opensmile));
    assert!(!command_requires_chat_infer(ReleasedCommand::Avqi));
    assert!(!command_requires_chat_infer(ReleasedCommand::Transcribe));
    assert!(!command_requires_chat_infer(ReleasedCommand::Benchmark));
}

/// Preflight media validation should still leave a durable setup attempt so
/// the file's failure appears in the attempt log instead of only in file
/// status.
#[tokio::test]
async fn preflight_media_failure_records_setup_attempt() {
    let tempdir = tempfile::tempdir().expect("tempdir");
    let missing_path = tempdir.path().join("missing.wav");
    let db = Arc::new(JobDB::open(Some(tempdir.path())).await.expect("open db"));
    let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
    let store = Arc::new(JobStore::for_tests(
        crate::config::ServerConfig::default(),
        Some(db.clone()),
        tx,
        std::sync::Arc::new(crate::clock::SystemClock),
    ));
    let job = make_media_job("job-media-preflight", &missing_path.display().to_string());
    // The runner's pending files, as the submitted job projects them.
    let file_list = job
        .pending_files()
        .expect("a submitted job's inputs are admitted");
    store.submit(job).await.expect("submit job");

    let failures = HashMap::from([(0usize, String::from("Media file not found"))]);
    let sink = StoreRunnerEventSink::wrap(store.clone(), &JobId::from("job-media-preflight"));

    let failed_indices =
        record_preflight_media_failures(sink.as_ref(), &file_list, &failures).await;

    assert_eq!(failed_indices, HashSet::from([0usize]));

    let attempts = db
        .load_attempts_for_job("job-media-preflight")
        .await
        .expect("load attempts");
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].work_unit_kind, WorkUnitKind::FileSetup);
    assert_eq!(attempts[0].outcome, AttemptOutcome::Failed);
    assert_eq!(
        attempts[0].failure_category,
        Some(FailureCategory::Validation)
    );

    let detail = store
        .get_job_detail(&JobId::from("job-media-preflight"))
        .await
        .expect("job detail");
    assert_eq!(detail.file_statuses.len(), 1);
    assert_eq!(
        detail.file_statuses[0].status,
        crate::api::FileStatusKind::Error
    );
}
