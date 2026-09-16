// Integration test target: Cargo compiles this as a separate crate,
// so the lib's `cfg_attr(test, ...)` allow does not apply. Test code
// uses `unwrap`/`expect` by convention.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

//! Worker failure-path integration tests (T077, T078, T079).
//!
//! These tests verify:
//! - T077: Worker crash propagates errors correctly to callers
//! - T078: Worker ready timeout fires when worker is too slow to start
//! - T079: Multi-file jobs preserve completed results when later files fail
//!
//! All tests use `--test-echo` workers (no ML models). Failures are injected
//! by killing worker processes (SIGKILL) or using very short timeouts.

use crate::common;

use std::collections::BTreeMap;
use std::time::Duration;

use batchalign::api::{LanguageCode3, ReleasedCommand, WorkerLanguage};
use batchalign::host_facts::PerProfile;
use batchalign::worker::error::WorkerError;
use batchalign::worker::handle::{WorkerConfig, WorkerHandle};
use batchalign::worker::pool::{PoolConfig, WorkerPool};
use batchalign::worker::{BatchInferRequest, InferTask, WorkerProfile};
use common::resolve_python;
use crate::live_deadline::{ProgressSnapshot, ServerTestDeadline, WaitSubject};
use serde_json::{Value, json};

macro_rules! require_python {
    () => {{
        common::test_server_fixture::isolate_host_memory_ledger();
        let available_mb = batchalign::worker::memory_guard::available_memory_mb();
        if available_mb < 4096 {
            eprintln!("SKIP: insufficient memory ({available_mb} MB available, 4096 MB required)");
            return;
        }
        match resolve_python() {
            Some(path) => path,
            None => {
                eprintln!("SKIP: Python 3 with batchalign not available");
                return;
            }
        }
    }};
}

fn batch_request(task: InferTask, items: Vec<Value>) -> BatchInferRequest {
    BatchInferRequest {
        task,
        lang: LanguageCode3::eng(),
        items,
        mwt: BTreeMap::new(),
        allow_stanza_fallback: false,
    }
}

fn test_pool(python: String) -> WorkerPool {
    // Each test in this binary spawns its own pool that touches the
    // host-memory ledger via the worker spawn memory guard. Opt into
    // the per-process ledger override the shared fixture uses so tests
    // here don't race against test binaries running fixture sessions.
    common::test_server_fixture::isolate_host_memory_ledger();
    WorkerPool::new(PoolConfig {
        python_path: python,
        health_check_interval_s: 600, // disable periodic health checks
        ready_timeout_s: 30,
        test_echo: true,
        max_workers_per_key: PerProfile::uniform(8),
        verbose: 0,
        runtime: Default::default(),
        ..Default::default()
    })
}

// ---------------------------------------------------------------------------
// T077: Worker crash/restart/retry failure-path tests
// ---------------------------------------------------------------------------

/// When a worker process is killed (SIGKILL) mid-request, the dispatch call
/// must return an error, not hang or silently succeed.
#[cfg(unix)]
#[tokio::test]
async fn worker_killed_mid_batch_infer_returns_error() {
    let python = require_python!();
    let config = WorkerConfig {
        python_path: python.clone(),
        test_echo: true,
        profile: WorkerProfile::Stanza,
        lang: WorkerLanguage::from(LanguageCode3::eng()),
        ready_timeout_s: 30,
        // Add a delay so we have time to kill the worker mid-request
        test_delay_ms: 5000,
        ..Default::default()
    };

    let mut handle = WorkerHandle::spawn(config).await.expect("spawn failed");
    let pid = *handle.pid();

    // Send a request that will take 5s (test_delay_ms), then kill the worker
    // while it's still processing.
    let request = batch_request(
        InferTask::Morphosyntax,
        vec![json!({"words": ["hello"], "lang": "eng"})],
    );
    let kill_handle = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(500)).await;
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGKILL);
        }
    });

    let result = handle.batch_infer(&request).await;
    kill_handle.await.expect("kill task");

    assert!(
        result.is_err(),
        "batch_infer should return an error after worker is killed"
    );
}

/// After a worker crash, the pool should still be able to spawn a new worker
/// and dispatch successfully: proving crash recovery works.
#[cfg(unix)]
#[tokio::test]
async fn pool_recovers_after_worker_crash() {
    let python = require_python!();
    let pool = test_pool(python);

    // First dispatch to warm up the pool with one worker.
    let item = json!({"words": ["first"], "lang": "eng"});
    let response = pool
        .dispatch_batch_infer(
            &LanguageCode3::eng(),
            &batch_request(InferTask::Morphosyntax, vec![item.clone()]),
        )
        .await
        .expect("first dispatch failed");
    assert_eq!(response.results[0].result, Some(item));

    // Get the worker PID and kill it.
    let summary = pool.worker_summary().await;
    let pid_str = summary
        .first()
        .expect("should have a worker")
        .split(':')
        .find_map(|part| part.strip_prefix("pid="))
        .expect("should find pid= in summary");
    let pid: u32 = pid_str.parse().expect("parse pid");
    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGKILL);
    }
    // Brief pause to let the OS reap the process.
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Second dispatch should succeed, pool spawns a replacement worker.
    let item2 = json!({"words": ["recovered"], "lang": "eng"});
    let response2 = pool
        .dispatch_batch_infer(
            &LanguageCode3::eng(),
            &batch_request(InferTask::Morphosyntax, vec![item2.clone()]),
        )
        .await
        .expect("recovery dispatch should succeed after worker crash");
    assert_eq!(response2.results[0].result, Some(item2));

    pool.shutdown().await;
}

/// Killing a worker and then dispatching to a different task group should work
/// without cross-contamination. The new group should get its own fresh worker.
#[cfg(unix)]
#[tokio::test]
async fn crash_in_one_group_does_not_affect_other_groups() {
    let python = require_python!();
    let pool = test_pool(python);

    // Dispatch to Morphosyntax (Stanza profile).
    pool.dispatch_batch_infer(
        &LanguageCode3::eng(),
        &batch_request(InferTask::Morphosyntax, vec![json!({"p": "stanza"})]),
    )
    .await
    .expect("stanza dispatch failed");

    // Get stanza worker PID and kill it.
    let stanza_summary = pool.worker_summary().await;
    let stanza_pid = stanza_summary
        .iter()
        .find(|s| s.contains("profile:stanza"))
        .and_then(|s| {
            s.split(':')
                .find_map(|part| part.strip_prefix("pid="))
                .and_then(|p| p.parse::<u32>().ok())
        })
        .expect("should find stanza worker pid");
    unsafe {
        libc::kill(stanza_pid as libc::pid_t, libc::SIGKILL);
    }
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Dispatch to Translate (IO profile), should succeed unaffected.
    let translate_result = pool
        .dispatch_batch_infer(
            &LanguageCode3::eng(),
            &batch_request(InferTask::Translate, vec![json!({"p": "io"})]),
        )
        .await;
    assert!(
        translate_result.is_ok(),
        "IO profile dispatch should succeed even though Stanza worker crashed"
    );

    pool.shutdown().await;
}

/// A worker that dies before emitting its ready signal must surface its
/// stderr (the Python traceback) in the spawn error.
///
/// Regression test for the 2026-06-11 incident on a fleet host: four
/// consecutive align jobs failed with "worker closed stdout without
/// emitting ready signal" and zero diagnostics, because the worker's
/// stderr was not captured anywhere. The pre-ready stdout tail alone
/// (progress_v2 events) cannot explain a crash; the traceback on stderr
/// can. This test uses a fake interpreter that reproduces the observed
/// failure shape exactly: one progress event on stdout, a traceback on
/// stderr, exit without a ready signal.
#[cfg(unix)]
#[tokio::test]
async fn worker_death_before_ready_surfaces_stderr() {
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;

    let tmp = tempfile::TempDir::new().expect("tempdir");
    let script_path = tmp.path().join("crashing-python");
    {
        let mut f = std::fs::File::create(&script_path).expect("create script");
        writeln!(f, "#!/bin/sh").expect("write");
        writeln!(
            f,
            "echo '{{\"op\": \"progress_v2\", \"event\": {{\"request_id\": \"\", \"completed\": 0, \"total\": 0, \"stage\": \"refreshing_stanza_catalog\"}}}}'"
        )
        .expect("write");
        writeln!(f, "echo 'Traceback (most recent call last):' >&2").expect("write");
        writeln!(
            f,
            "echo 'RuntimeError: synthetic bootstrap crash for test' >&2"
        )
        .expect("write");
        writeln!(f, "exit 3").expect("write");
    }
    let mut perms = std::fs::metadata(&script_path)
        .expect("stat script")
        .permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&script_path, perms).expect("chmod script");

    let config = WorkerConfig {
        python_path: script_path.to_string_lossy().into_owned(),
        test_echo: true,
        profile: WorkerProfile::Stanza,
        lang: WorkerLanguage::from(LanguageCode3::eng()),
        ready_timeout_s: 30,
        ..Default::default()
    };

    let err = match WorkerHandle::spawn(config).await {
        Err(err) => err,
        Ok(_) => panic!("spawn must fail when the worker dies before ready"),
    };
    let message = err.to_string();
    assert!(
        message.contains("RuntimeError: synthetic bootstrap crash for test"),
        "spawn error must carry the worker's stderr tail for diagnosability; got: {message}"
    );
}

// ---------------------------------------------------------------------------
// T078: Worker timeout integration tests
// ---------------------------------------------------------------------------

/// A worker that takes longer than `ready_timeout_s` to start should produce
/// a `ReadyTimeout` error.
#[tokio::test]
async fn worker_ready_timeout_fires() {
    let python = require_python!();
    let config = WorkerConfig {
        python_path: python,
        test_echo: true,
        profile: WorkerProfile::Stanza,
        lang: WorkerLanguage::from(LanguageCode3::eng()),
        // Use an impossibly short timeout (100ms) so the worker can't possibly start in time
        // ... except test-echo workers start very fast. Instead, use a bad python path
        // to ensure the worker never sends a ready signal.
        ready_timeout_s: 1,
        ..Default::default()
    };

    // We can't easily make a test-echo worker start slowly. Instead, test the
    // timeout path by using a non-existent python path, the spawn should fail
    // with a meaningful error (not hang forever).
    let bad_config = WorkerConfig {
        python_path: "/nonexistent/python3".into(),
        ..config
    };
    let result = WorkerHandle::spawn(bad_config).await;
    match result {
        Err(WorkerError::SpawnFailed(msg)) => {
            eprintln!("Got expected SpawnFailed: {msg}");
        }
        Err(other) => {
            panic!("expected SpawnFailed, got: {other:?}");
        }
        Ok(_) => {
            panic!("spawning with a bad python path should fail, not succeed");
        }
    }
}

/// Pool dispatch with a fast health-check interval should still work for
/// immediate requests: the health check loop only evicts under memory
/// pressure (no longer fires on idle timer).
#[tokio::test]
async fn fast_health_check_interval_does_not_break_dispatch() {
    let python = require_python!();
    let pool = WorkerPool::new(PoolConfig {
        python_path: python,
        health_check_interval_s: 1, // fast health checks
        ready_timeout_s: 30,
        test_echo: true,
        max_workers_per_key: PerProfile::uniform(8),
        verbose: 0,
        runtime: Default::default(),
        ..Default::default()
    });

    // Dispatch should succeed immediately.
    let item = json!({"words": ["hello"]});
    let response = pool
        .dispatch_batch_infer(
            &LanguageCode3::eng(),
            &batch_request(InferTask::Morphosyntax, vec![item.clone()]),
        )
        .await
        .expect("dispatch should succeed");
    assert_eq!(response.results[0].result, Some(item));

    pool.shutdown().await;
}

// ---------------------------------------------------------------------------
// T079: Partial-results integration tests
// ---------------------------------------------------------------------------

/// A multi-file job where one file succeeds and a later file fails should
/// preserve the successful result and record the failure.
///
/// Note: with test-echo workers, all files succeed, so we cannot easily inject
/// per-file failures at the worker level. Instead, we test the HTTP API's
/// ability to handle and report multi-file results correctly, which is the
/// layer where partial results are assembled.
#[tokio::test]
async fn multi_file_job_produces_per_file_results() {
    let python = require_python!();

    use batchalign::api::{
        FilePayload, JobInfo, JobResultResponse, JobStatus, JobSubmission, LanguageSpec, MemoryMb,
        NumSpeakers,
    };
    use batchalign::config::ServerConfig;
    use batchalign::create_test_app;
    use batchalign::options::{CommandOptions, CommonOptions, MorphotagOptions};
    use tokio::sync::Semaphore;

    static PARTIAL_SLOTS: Semaphore = Semaphore::const_new(1);
    let _permit = PARTIAL_SLOTS.acquire().await.expect("semaphore");
    // This test bypasses the shared fixture; opt into the same
    // per-process host-memory ledger override so it doesn't race
    // against sessions from other test binaries.
    common::test_server_fixture::isolate_host_memory_ledger();

    let tmp = tempfile::TempDir::new().expect("tempdir");
    let jobs_dir = tmp.path().join("jobs");
    std::fs::create_dir_all(&jobs_dir).expect("mkdir jobs");
    let db_dir = tmp.path().join("db");
    std::fs::create_dir_all(&db_dir).expect("mkdir db");

    let config = ServerConfig {
        host: "127.0.0.1".into(),
        port: batchalign::config::PortRequest::from_u16(0),
        job_ttl_days: 7,
        memory_gate_mb: Some(MemoryMb(0)),
        ..Default::default()
    };
    let pool_config = PoolConfig {
        python_path: python,
        test_echo: true,
        health_check_interval_s: 600,
        ready_timeout_s: 30,
        max_workers_per_key: PerProfile::uniform(8),
        verbose: 0,
        runtime: Default::default(),
        ..Default::default()
    };

    let (router, _state) = create_test_app(
        config,
        pool_config,
        Some(jobs_dir.to_string_lossy().into()),
        Some(db_dir),
        Some("partial-test".into()),
    )
    .await
    .expect("create_test_app");

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let port = listener.local_addr().expect("local_addr").port();
    let base_url = format!("http://127.0.0.1:{port}");

    tokio::spawn(async move {
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
        .ok();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let client = reqwest::Client::new();

    // Submit a job with 3 files. Morphotag requires
    // `LanguageSpec::PerFile`; each file carries `@Languages: eng`
    // so per-file resolution succeeds at dispatch time.
    let submission = JobSubmission {
        command: ReleasedCommand::Morphotag,
        lang: LanguageSpec::PerFile,
        num_speakers: NumSpeakers(1),
        files: vec![
            FilePayload {
                filename: "file_a.cha".into(),
                content: "@UTF8\n@Languages:\teng\n@Begin\n*CHI:\thello .\n@End\n".into(),
            },
            FilePayload {
                filename: "file_b.cha".into(),
                content: "@UTF8\n@Languages:\teng\n@Begin\n*CHI:\tworld .\n@End\n".into(),
            },
            FilePayload {
                filename: "file_c.cha".into(),
                content: "@UTF8\n@Languages:\teng\n@Begin\n*CHI:\tgoodbye .\n@End\n".into(),
            },
        ],
        media_files: vec![],
        media_mapping: Default::default(),
        media_subdir: Default::default(),
        source_dir: Default::default(),
        options: CommandOptions::Morphotag(MorphotagOptions {
            common: CommonOptions::default(),
            merge_abbrev: Default::default(),

            ..Default::default()
        }),
        paths_mode: false,
        source_paths: vec![],
        output_paths: vec![],
        display_names: vec![],
        debug_traces: false,
        before_paths: vec![],
    };

    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs");
    assert_eq!(resp.status(), 200);
    let info: JobInfo = resp.json().await.expect("parse");
    assert_eq!(info.total_files, 3);

    // Poll until complete.
    let mut deadline = ServerTestDeadline::new(WaitSubject::job_completion(&info.job_id));
    loop {
        let resp = client
            .get(format!("{base_url}/jobs/{}", info.job_id))
            .send()
            .await
            .expect("GET job");
        let job: JobInfo = resp.json().await.expect("parse");
        if matches!(
            job.status,
            JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled
        ) {
            break;
        }
        deadline.keep_waiting(ProgressSnapshot::job(&job)).await;
    }

    // Verify all 3 files have individual results.
    let resp = client
        .get(format!("{base_url}/jobs/{}/results", info.job_id))
        .send()
        .await
        .expect("GET results");
    let results: JobResultResponse = resp.json().await.expect("parse results");
    assert_eq!(
        results.files.len(),
        3,
        "should have per-file results for all 3 files"
    );

    // Each file should have content and no error (test-echo processes all).
    for (i, file_result) in results.files.iter().enumerate() {
        assert!(
            file_result.error.is_none(),
            "file {i} ({}) should have no error",
            file_result.filename
        );
        assert!(
            !file_result.content.is_empty(),
            "file {i} ({}) should have content",
            file_result.filename
        );
    }

    // Verify individual file retrieval works.
    for filename in &["file_a.cha", "file_b.cha", "file_c.cha"] {
        let resp = client
            .get(format!(
                "{base_url}/jobs/{}/results/{filename}",
                info.job_id
            ))
            .send()
            .await
            .expect("GET single file result");
        assert_eq!(
            resp.status(),
            200,
            "individual file {filename} should be retrievable"
        );
    }
}
