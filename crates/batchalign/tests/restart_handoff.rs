//! Restart-while-running handoff contract.
//!
//! Field failure (2026-07-10, 345-file transcribe job): an API
//! cancel+restart landed while the runner was mid-file. `restart_job`
//! trusted the job's (already-flipped) status and spawned a second
//! runner while the first was still tearing down. The first runner's
//! force-terminal sweep then mislabeled the freshly re-queued files as
//! "File did not reach terminal status (last status: queued)" and
//! finalized the job `failed`, clobbering the restart, while the second
//! runner processed for another 100 minutes under a `failed` banner and
//! inflated `completed_files` past `total_files`.
//!
//! Contract pinned here: cancel + restart of a RUNNING job hands off to
//! exactly one surviving runner. After the restart is accepted, the job
//! never reports `Failed`, never reports more completed files than it
//! has, and settles `Completed` with every file done.
//!
// Test code: fixtures and polling use unwrap/expect by convention.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::common;

use batchalign::api::{
    FilePayload, JobInfo, JobStatus, JobSubmission, LanguageCode3, LanguageSpec, NumSpeakers,
    ReleasedCommand,
};
use batchalign::config::ServerConfig;
use batchalign::create_test_app;
use batchalign::options::{CommandOptions, CommonOptions, TranscribeOptions};
use batchalign::worker::pool::PoolConfig;
use common::resolve_python;
use common::test_server_fixture::isolate_host_memory_ledger;

/// Per-response artificial worker delay. Long enough that the first
/// runner is reliably mid-file when the cancel+restart lands; short
/// enough to keep the whole test a few seconds.
const ECHO_DELAY_MS: u64 = 400;

/// Enough files that the post-restart runner is busy long enough for a
/// wrong `Failed` finalization from the first runner to be observed.
const NUM_FILES: usize = 8;

fn echo_submission() -> JobSubmission {
    let files = (0..NUM_FILES)
        .map(|i| FilePayload {
            filename: format!("handoff-{i}.cha").into(),
            content: format!("@UTF8\n@Begin\n*CHI:\thandoff {i} .\n@End\n"),
        })
        .collect();
    JobSubmission {
        command: ReleasedCommand::Transcribe,
        lang: LanguageSpec::Resolved(LanguageCode3::eng()),
        num_speakers: NumSpeakers(1),
        files,
        media_files: vec![],
        media_mapping: Default::default(),
        media_subdir: Default::default(),
        source_dir: Default::default(),
        options: CommandOptions::Transcribe(TranscribeOptions {
            auto_speakers: false,
            common: CommonOptions::default(),
            asr_engine: batchalign::options::AsrEngineName::RevAi,
            diarize: false,
            wor: false.into(),
            merge_abbrev: false.into(),
            batch_size: 8,
            utseg_fallback: false.into(),
        }),
        paths_mode: false,
        source_paths: vec![],
        output_paths: vec![],
        display_names: vec![],
        debug_traces: false,
        before_paths: vec![],
    }
}

async fn get_job(client: &reqwest::Client, base_url: &str, job_id: &str) -> JobInfo {
    client
        .get(format!("{base_url}/jobs/{job_id}"))
        .send()
        .await
        .expect("GET /jobs/{id}")
        .json()
        .await
        .expect("parse JobInfo")
}

#[tokio::test]
async fn cancel_restart_while_running_hands_off_to_one_runner() {
    let Some(python_path) = resolve_python() else {
        eprintln!("SKIP: Python 3 with batchalign not available");
        return;
    };
    isolate_host_memory_ledger();

    // Dedicated app (not the shared warm fixture): this test needs a
    // per-response worker delay to hold the race window open.
    let scratch = tempfile::TempDir::new().expect("tempdir");
    let jobs_dir = scratch.path().join("jobs");
    let db_dir = scratch.path().join("db");
    std::fs::create_dir_all(&jobs_dir).expect("mkdir jobs");
    std::fs::create_dir_all(&db_dir).expect("mkdir db");

    let pool_config = PoolConfig {
        python_path,
        test_echo: true,
        test_delay_ms: ECHO_DELAY_MS,
        verbose: 0,
        ..Default::default()
    };
    let (router, _state) = create_test_app(
        ServerConfig::default(),
        pool_config,
        Some(jobs_dir.to_string_lossy().into()),
        Some(db_dir),
        Some("restart-handoff-test".into()),
    )
    .await
    .expect("create test app");

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let port = listener.local_addr().expect("local_addr").port();
    tokio::spawn(async move {
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
        .ok();
    });
    let base_url = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();

    // Submit and wait until the job is genuinely mid-run (>= 1 file done,
    // more pending), so the cancel+restart hits an active runner.
    let info: JobInfo = client
        .post(format!("{base_url}/jobs"))
        .json(&echo_submission())
        .send()
        .await
        .expect("POST /jobs")
        .json()
        .await
        .expect("parse submission");
    let job_id = info.job_id;

    let start = tokio::time::Instant::now();
    loop {
        assert!(
            start.elapsed() < std::time::Duration::from_secs(30),
            "job never started making progress"
        );
        let info = get_job(&client, &base_url, &job_id).await;
        if info.completed_files >= 1 && (info.completed_files as usize) < NUM_FILES {
            break;
        }
        assert!(
            !matches!(
                info.status,
                JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled
            ),
            "job reached terminal state {:?} before the race window; \
             raise NUM_FILES or ECHO_DELAY_MS",
            info.status
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }

    // Cancel, then restart as soon as the server accepts it (cancel is
    // asynchronous, so restart may briefly 409 until the status flips).
    let resp = client
        .post(format!("{base_url}/jobs/{job_id}/cancel"))
        .send()
        .await
        .expect("POST cancel");
    assert_eq!(resp.status(), 200, "cancel accepted");

    let restart_deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        let resp = client
            .post(format!("{base_url}/jobs/{job_id}/restart"))
            .send()
            .await
            .expect("POST restart");
        if resp.status() == 200 {
            break;
        }
        assert!(
            tokio::time::Instant::now() < restart_deadline,
            "restart never accepted, last status {}",
            resp.status()
        );
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }

    // From here on the job belongs to the restarted runner. The old
    // runner's teardown must not clobber it: no Failed status, no
    // counter overflow, and a Completed settle with every file done.
    let settle_deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(60);
    let final_info = loop {
        assert!(
            tokio::time::Instant::now() < settle_deadline,
            "restarted job never settled"
        );
        let info = get_job(&client, &base_url, &job_id).await;
        assert_ne!(
            info.status,
            JobStatus::Failed,
            "restarted job reported Failed while its runner was working \
             (the old runner's sweep clobbered the restart): {:?}",
            info.error
        );
        assert!(
            info.completed_files as usize <= NUM_FILES,
            "completed_files overflowed total ({} > {NUM_FILES}): two runners \
             are counting the same job",
            info.completed_files
        );
        match info.status {
            JobStatus::Completed => break info,
            JobStatus::Cancelled => {
                panic!(
                    "restarted job settled Cancelled; restart was accepted so a runner must own it"
                )
            }
            _ => tokio::time::sleep(std::time::Duration::from_millis(25)).await,
        }
    };

    assert_eq!(
        final_info.completed_files as usize, NUM_FILES,
        "every file must be done after the restarted run"
    );
}
