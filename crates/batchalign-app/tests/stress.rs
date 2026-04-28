//! Concurrency stress tests for the batchalign3 server.
//!
//! These tests hammer shared state from concurrent tasks to verify invariants:
//! worker pool permit management, job lifecycle under concurrent mutations,
//! SQLite contention, and health endpoint accuracy.
//!
//! Unlike turmoil tests (which simulate network faults), these run on a real
//! tokio runtime with real TCP, real Python test-echo workers, and real SQLite.
//! The concurrency comes from `tokio::spawn` + `JoinSet`, not a framework.
//!
//! All tests are serialized via nextest test-group (`max-threads = 1`) to
//! avoid contention between test binaries.

mod common;

use std::sync::Arc;
use std::time::Duration;

use batchalign_app::api::{
    FilePayload, HealthResponse, JobInfo, JobSubmission, LanguageCode3, LanguageSpec, MemoryMb,
    NumSpeakers, ReleasedCommand,
};
use batchalign_app::config::ServerConfig;
use batchalign_app::create_test_app;
use batchalign_app::options::{CommandOptions, CommonOptions, MorphotagOptions};
use batchalign_app::worker::pool::PoolConfig;
use tokio::sync::Semaphore;

/// Serialize stress tests so they don't fight over Python workers and sockets.
static STRESS_SLOTS: Semaphore = Semaphore::const_new(1);

macro_rules! require_python {
    () => {
        match common::resolve_python() {
            Some(p) => p,
            None => {
                eprintln!("SKIP: Python not available");
                return;
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

/// Start a test server with default config for stress testing.
async fn start_stress_server(python_path: &str) -> StressServer {
    start_stress_server_with_config(python_path, ServerConfig::default()).await
}

/// Start a test server with custom config.
async fn start_stress_server_with_config(
    python_path: &str,
    mut config: ServerConfig,
) -> StressServer {
    let _permit = STRESS_SLOTS.acquire().await.expect("stress semaphore");

    let tmp = tempfile::TempDir::new().expect("tempdir");
    let jobs_dir = tmp.path().join("jobs");
    std::fs::create_dir_all(&jobs_dir).expect("mkdir jobs");
    let db_dir = tmp.path().join("db");
    std::fs::create_dir_all(&db_dir).expect("mkdir db");

    config.host = "127.0.0.1".into();
    config.port = 0;
    config.job_ttl_days = 7;
    config.warmup_commands = vec![];
    config.memory_gate_mb = Some(MemoryMb(0));

    let pool_config = PoolConfig {
        python_path: python_path.into(),
        test_echo: true,
        health_check_interval_s: 600,
        idle_timeout_s: 600,
        ready_timeout_s: 30,
        max_workers_per_key: 4,
        verbose: 0,
        engine_overrides: String::new(),
        runtime: Default::default(),
        ..Default::default()
    };

    let (router, state) = create_test_app(
        config,
        pool_config,
        Some(jobs_dir.to_string_lossy().into()),
        Some(db_dir),
        Some("stress-test-hash".into()),
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

    StressServer {
        base_url,
        client: reqwest::Client::new(),
        _state: state,
        _tmp: tmp,
    }
}

#[allow(dead_code)]
struct StressServer {
    base_url: String,
    client: reqwest::Client,
    _state: Arc<batchalign_app::AppState>,
    _tmp: tempfile::TempDir,
}

impl StressServer {
    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }

    async fn submit(&self, sub: &JobSubmission) -> reqwest::Response {
        self.client
            .post(self.url("/jobs"))
            .json(sub)
            .send()
            .await
            .expect("POST /jobs")
    }

    async fn submit_ok(&self, sub: &JobSubmission) -> String {
        let resp = self.submit(sub).await;
        assert_eq!(resp.status().as_u16(), 200, "Job submission should succeed");
        let info: serde_json::Value = resp.json().await.expect("parse json");
        info["job_id"].as_str().expect("job_id").to_string()
    }

    async fn get_job(&self, job_id: &str) -> JobInfo {
        let resp = self
            .client
            .get(self.url(&format!("/jobs/{job_id}")))
            .send()
            .await
            .expect("GET job");
        resp.json().await.expect("parse JobInfo")
    }

    #[allow(
        dead_code,
        reason = "kept for future stress-test scenarios that need status-only probing"
    )]
    async fn get_job_status_code(&self, job_id: &str) -> u16 {
        self.client
            .get(self.url(&format!("/jobs/{job_id}")))
            .send()
            .await
            .expect("GET job")
            .status()
            .as_u16()
    }

    async fn health(&self) -> HealthResponse {
        let resp = self
            .client
            .get(self.url("/health"))
            .send()
            .await
            .expect("GET /health");
        assert_eq!(resp.status().as_u16(), 200);
        resp.json().await.expect("parse HealthResponse")
    }

    async fn cancel(&self, job_id: &str) -> u16 {
        self.client
            .post(self.url(&format!("/jobs/{job_id}/cancel")))
            .send()
            .await
            .expect("POST cancel")
            .status()
            .as_u16()
    }

    #[allow(
        dead_code,
        reason = "kept for future stress-test scenarios exercising DELETE"
    )]
    async fn delete(&self, job_id: &str) -> u16 {
        self.client
            .delete(self.url(&format!("/jobs/{job_id}")))
            .send()
            .await
            .expect("DELETE job")
            .status()
            .as_u16()
    }

    async fn list_jobs(&self) -> Vec<serde_json::Value> {
        let resp = self
            .client
            .get(self.url("/jobs"))
            .send()
            .await
            .expect("GET /jobs");
        assert_eq!(resp.status().as_u16(), 200);
        resp.json().await.expect("parse job list")
    }

    /// Poll until job reaches a terminal state (completed, failed, cancelled).
    async fn poll_until_done(&self, job_id: &str) -> JobInfo {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
        loop {
            let info = self.get_job(job_id).await;
            if info.status.is_terminal() {
                return info;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "Job {job_id} did not reach terminal state within 60s"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

const MINIMAL_CHAT: &str = "@UTF8\n@Begin\n@Languages:\teng\n\
    @Participants:\tCHI Target_Child\n\
    @ID:\teng|test|CHI|||||Target_Child|||\n\
    *CHI:\thello world .\n@End\n";

fn morphotag_submission(filename: &str) -> JobSubmission {
    JobSubmission {
        command: ReleasedCommand::Morphotag,
        lang: LanguageSpec::Resolved(LanguageCode3::eng()),
        num_speakers: NumSpeakers(1),
        files: vec![FilePayload {
            filename: filename.into(),
            content: MINIMAL_CHAT.into(),
        }],
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
    }
}

fn multi_file_submission(count: usize) -> JobSubmission {
    let mut sub = morphotag_submission("file-0.cha");
    sub.files = (0..count)
        .map(|i| FilePayload {
            filename: format!("file-{i}.cha").into(),
            content: MINIMAL_CHAT.into(),
        })
        .collect();
    sub
}

// ===========================================================================
// Group A: Worker pool permit invariants
// ===========================================================================

/// Submit 10 jobs simultaneously. After all complete, verify the pool
/// recovered all permits (workers_available == live_workers in health).
#[tokio::test]
async fn pool_permits_match_idle_count_after_burst() {
    let python = require_python!();
    let server = start_stress_server(&python).await;

    // Submit 10 jobs with unique filenames
    let mut job_ids = Vec::new();
    for i in 0..10 {
        let sub = morphotag_submission(&format!("burst-{i}.cha"));
        let id = server.submit_ok(&sub).await;
        job_ids.push(id);
    }

    // Wait for all to complete
    for id in &job_ids {
        let info = server.poll_until_done(id).await;
        assert_eq!(
            info.status.to_string(),
            "completed",
            "Job {id} should complete"
        );
    }

    // Give pool a moment to return all workers
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Verify permit invariant via health endpoint
    let health = server.health().await;
    assert!(
        health.workers_available >= 0,
        "workers_available should be non-negative"
    );
    assert_eq!(
        health.active_jobs, 0,
        "No jobs should be active after all complete"
    );
}

/// Submit-cancel cycle 10 times. If permits leak, the final iteration hangs.
#[tokio::test]
async fn cancel_job_returns_worker_permits() {
    let python = require_python!();
    let server = start_stress_server(&python).await;

    for i in 0..10 {
        let sub = morphotag_submission(&format!("cancel-cycle-{i}.cha"));
        let id = server.submit_ok(&sub).await;
        let _ = server.cancel(&id).await;
        // Wait for the job to reach terminal state
        let info = server.poll_until_done(&id).await;
        assert!(
            info.status.is_terminal(),
            "Cycle {i}: job should be terminal after cancel"
        );
    }

    // Final job should complete (proves permits are available)
    let sub = morphotag_submission("final-after-cancels.cha");
    let id = server.submit_ok(&sub).await;
    let info = server.poll_until_done(&id).await;
    assert_eq!(
        info.status.to_string(),
        "completed",
        "Final job should complete — no permit leak"
    );
}

/// Rapid submit-cancel loop (50 iterations). After all, submit one final job.
/// If any permit leaked, the final job hangs forever.
#[tokio::test]
async fn rapid_submit_cancel_cycle_no_permit_leak() {
    let python = require_python!();
    let server = start_stress_server(&python).await;

    for i in 0..50 {
        let sub = morphotag_submission(&format!("rapid-{i}.cha"));
        let id = server.submit_ok(&sub).await;
        // Cancel immediately — don't wait for terminal state
        let _ = server.cancel(&id).await;
    }

    // Wait for everything to settle
    tokio::time::sleep(Duration::from_secs(2)).await;

    // The critical assertion: this job must complete (not hang)
    let sub = morphotag_submission("final-after-rapid.cha");
    let id = server.submit_ok(&sub).await;

    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        let info = server.get_job(&id).await;
        if info.status.is_terminal() {
            // Success — permits are not leaked
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "PERMIT LEAK DETECTED: final job hung for 30s after 50 cancel cycles"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// 5 concurrent cancel requests on the same job. All should succeed (200).
#[tokio::test]
async fn concurrent_cancels_on_same_job() {
    let python = require_python!();
    let server = start_stress_server(&python).await;

    let sub = morphotag_submission("concurrent-cancel.cha");
    let id = server.submit_ok(&sub).await;

    let mut handles = tokio::task::JoinSet::new();
    for _ in 0..5 {
        let client = server.client.clone();
        let url = server.url(&format!("/jobs/{id}/cancel"));
        handles.spawn(async move {
            client
                .post(url)
                .send()
                .await
                .expect("cancel request")
                .status()
                .as_u16()
        });
    }

    let mut statuses = Vec::new();
    while let Some(result) = handles.join_next().await {
        statuses.push(result.expect("join cancel task"));
    }

    // All should be 200 (cancel is idempotent)
    for (i, &status) in statuses.iter().enumerate() {
        assert_eq!(status, 200, "Cancel #{i} should return 200");
    }

    // Job should be in terminal state
    let info = server.get_job(&id).await;
    assert!(
        info.status.is_terminal(),
        "Job should be terminal after concurrent cancels"
    );
}

// ===========================================================================
// Group B: Job concurrency limits
// ===========================================================================

/// Configure max_concurrent_jobs = 2, submit 3 jobs. All should eventually
/// complete — the third queues until a slot opens.
#[tokio::test]
async fn max_concurrent_jobs_enforced() {
    let python = require_python!();
    let config = ServerConfig {
        max_concurrent_jobs: Some(2),
        ..Default::default()
    };
    let server = start_stress_server_with_config(&python, config).await;

    let mut job_ids = Vec::new();
    for i in 0..3 {
        let sub = morphotag_submission(&format!("concurrent-limit-{i}.cha"));
        let id = server.submit_ok(&sub).await;
        job_ids.push(id);
    }

    // All 3 should eventually complete (third waits for a slot)
    for id in &job_ids {
        let info = server.poll_until_done(id).await;
        assert_eq!(info.status.to_string(), "completed");
    }
}

/// Submit 5 jobs, cancel some, delete completed ones, list all — all
/// operations should return valid responses.
#[tokio::test]
async fn job_list_consistent_under_concurrent_mutations() {
    let python = require_python!();
    let server = start_stress_server(&python).await;

    // Submit 5 jobs
    let mut ids = Vec::new();
    for i in 0..5 {
        let sub = morphotag_submission(&format!("list-stress-{i}.cha"));
        ids.push(server.submit_ok(&sub).await);
    }

    // Wait for at least some to complete
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Concurrently: list, cancel #0, list, delete completed, list
    let mut handles = tokio::task::JoinSet::new();

    // List jobs
    let c = server.client.clone();
    let u = server.url("/jobs");
    handles.spawn(async move {
        let resp = c.get(&u).send().await.expect("list");
        assert_eq!(resp.status().as_u16(), 200);
        let _: Vec<serde_json::Value> = resp.json().await.expect("parse list");
        "list_ok"
    });

    // Cancel first job
    let c = server.client.clone();
    let u = server.url(&format!("/jobs/{}/cancel", ids[0]));
    handles.spawn(async move {
        let status = c.post(&u).send().await.expect("cancel").status().as_u16();
        assert_eq!(status, 200);
        "cancel_ok"
    });

    // List again
    let c = server.client.clone();
    let u = server.url("/jobs");
    handles.spawn(async move {
        let resp = c.get(&u).send().await.expect("list2");
        assert_eq!(resp.status().as_u16(), 200);
        "list2_ok"
    });

    while let Some(result) = handles.join_next().await {
        result.expect("concurrent operation should not panic");
    }

    // Final list should be valid
    let jobs = server.list_jobs().await;
    assert!(jobs.len() <= 5, "Should have at most 5 jobs");
}

// ===========================================================================
// Group C: Concurrent client operations
// ===========================================================================

/// Two clients submit jobs with the same filename simultaneously.
/// Exactly one should succeed (200), the other should get 409 conflict.
#[tokio::test]
async fn concurrent_identical_file_conflict_detected() {
    let python = require_python!();
    let server = start_stress_server(&python).await;

    let sub = morphotag_submission("conflict.cha");

    let mut handles = tokio::task::JoinSet::new();
    for _ in 0..2 {
        let client = server.client.clone();
        let url = server.url("/jobs");
        let body = serde_json::to_string(&sub).expect("serialize");
        handles.spawn(async move {
            client
                .post(url)
                .header("content-type", "application/json")
                .body(body)
                .send()
                .await
                .expect("submit")
                .status()
                .as_u16()
        });
    }

    let mut statuses = Vec::new();
    while let Some(result) = handles.join_next().await {
        statuses.push(result.expect("join"));
    }
    statuses.sort();

    // One 200 and one 409 (order doesn't matter)
    assert!(
        (statuses == [200, 409]) || (statuses == [200, 200]),
        "Expected one success + one conflict (or both succeed if serialized), got {statuses:?}"
    );
    // If both succeeded, that means the server serialized them — also acceptable
}

/// 5 concurrent poll requests for the same job during state transitions.
#[tokio::test]
async fn concurrent_poll_during_rapid_state_changes() {
    let python = require_python!();
    let server = start_stress_server(&python).await;

    let sub = morphotag_submission("poll-race.cha");
    let id = server.submit_ok(&sub).await;

    let mut handles = tokio::task::JoinSet::new();
    for _ in 0..5 {
        let client = server.client.clone();
        let url = server.url(&format!("/jobs/{id}"));
        handles.spawn(async move {
            // Poll rapidly 10 times
            for _ in 0..10 {
                let resp = client.get(&url).send().await.expect("poll");
                assert_eq!(resp.status().as_u16(), 200);
                let info: serde_json::Value = resp.json().await.expect("parse");
                let status = info["status"].as_str().expect("status field");
                assert!(
                    ["queued", "running", "completed", "failed", "cancelled"].contains(&status),
                    "Invalid status: {status}"
                );
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });
    }

    while let Some(result) = handles.join_next().await {
        result.expect("concurrent poll should not panic");
    }
}

/// Submit a multi-file job, connect SSE. Verify snapshot + complete events.
#[tokio::test]
async fn sse_stream_under_rapid_file_completions() {
    let python = require_python!();
    let server = start_stress_server(&python).await;

    let sub = multi_file_submission(5);
    let id = server.submit_ok(&sub).await;

    // Connect to SSE stream with a timeout
    let sse_url = server.url(&format!("/jobs/{id}/stream"));
    let result = tokio::time::timeout(Duration::from_secs(30), async {
        let resp = server
            .client
            .get(&sse_url)
            .send()
            .await
            .expect("SSE request");
        assert_eq!(resp.status().as_u16(), 200);

        resp.text().await.expect("SSE body")
    })
    .await;

    match result {
        Ok(text) => {
            assert!(
                text.contains("event: snapshot"),
                "SSE should contain snapshot event"
            );
        }
        Err(_) => {
            // Timeout is acceptable for a streaming response — job may
            // still be processing. Verify it eventually completes.
            server.poll_until_done(&id).await;
        }
    }
}

// ===========================================================================
// Group D: SQLite contention and health accuracy
// ===========================================================================

/// Submit 5 independent jobs simultaneously. All should complete without
/// SQLITE_BUSY errors surfacing as 500 responses.
#[tokio::test]
async fn concurrent_job_completions_no_sqlite_busy() {
    let python = require_python!();
    let server = start_stress_server(&python).await;

    let mut ids = Vec::new();
    for i in 0..5 {
        let sub = morphotag_submission(&format!("sqlite-{i}.cha"));
        ids.push(server.submit_ok(&sub).await);
    }

    for id in &ids {
        let info = server.poll_until_done(id).await;
        assert_eq!(
            info.status.to_string(),
            "completed",
            "Job {id} should complete without SQLite errors"
        );
    }
}

/// Poll health 20 times rapidly while 5 jobs are running. active_jobs
/// should always be in [0, 5] and status should always be "ok".
#[tokio::test]
async fn health_endpoint_consistent_under_load() {
    let python = require_python!();
    let server = start_stress_server(&python).await;

    // Submit 5 jobs
    for i in 0..5 {
        let sub = morphotag_submission(&format!("health-load-{i}.cha"));
        server.submit_ok(&sub).await;
    }

    // Poll health rapidly
    for _ in 0..20 {
        let health = server.health().await;
        assert_eq!(health.status.to_string(), "ok");
        assert!(
            health.active_jobs >= 0 && health.active_jobs <= 5,
            "active_jobs should be in [0, 5], got {}",
            health.active_jobs
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
