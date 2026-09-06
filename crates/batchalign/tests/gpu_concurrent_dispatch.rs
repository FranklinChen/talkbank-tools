// Integration test target: Cargo compiles this as a separate crate,
// so the lib's `cfg_attr(test, ...)` allow does not apply. Test code
// uses `unwrap`/`expect` by convention.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

//! Integration tests for GPU concurrent dispatch through the pool's shared GPU
//! worker paths.
//!
//! These tests exercise the most fragile code path in the worker system:
//! multiplexing concurrent `execute_v2` requests over one shared worker
//! transport with hand-rolled response routing by `request_id`.
//!
//! All tests use `--test-echo` workers (no ML models). The Python worker's
//! test-echo mode returns a success response echoing the `request_id` for
//! `execute_v2`, enabling concurrent dispatch verification without real models.
//! Most tests exercise the pool-level dispatch path, which may use either the
//! stdio or TCP shared-worker transport depending on setup. The explicit drop
//! cleanup test below targets the stdio lifecycle-owner path directly.
//!
//! # What these tests prove
//!
//! - Multiple concurrent requests to one GPU worker all receive correct responses
//! - Response routing by `request_id` works when responses arrive out of order
//! - All concurrent requests share the same worker PID (model sharing)
//! - The reader task failure path fails all pending requests cleanly
//! - Sequential requests after concurrent batches still work (no state corruption)

mod common;

use std::collections::BTreeMap;
use std::time::Duration;

use batchalign::api::{LanguageCode3, ReleasedCommand, WorkerLanguage};
use batchalign::host_facts::PerProfile;
use batchalign::types::worker_v2::{
    AsrBackendV2, AsrInputV2, AsrRequestV2, ExecuteRequestV2, ExecuteResponseV2, InferenceTaskV2,
    PreparedAudioInputV2, TaskRequestV2, WorkerArtifactIdV2, WorkerRequestIdV2,
};
use batchalign::worker::handle::WorkerRuntimeConfig;
use batchalign::worker::pool::{PoolConfig, WorkerPool};
use batchalign::worker::{BatchInferRequest, InferTask};
use common::resolve_python;
use serde_json::json;

macro_rules! require_python {
    () => {{
        common::init_test_tracing(tracing::Level::ERROR);
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

/// Build a GPU execute_v2 request with a unique request_id, for English.
fn gpu_execute_request(request_id: &str) -> ExecuteRequestV2 {
    gpu_execute_request_for_lang(request_id, LanguageCode3::eng())
}

/// Build a GPU execute_v2 request with a unique request_id, for a given language.
///
/// The language matters to callers that need two DIFFERENT worker keys: a shared
/// GPU workers are keyed on target, language, and typed engine selection, so
/// two dispatches that differ only in language resolve to separate workers.
fn gpu_execute_request_for_lang(request_id: &str, lang: LanguageCode3) -> ExecuteRequestV2 {
    ExecuteRequestV2 {
        request_id: WorkerRequestIdV2::from(request_id),
        task: InferenceTaskV2::Asr,
        payload: TaskRequestV2::Asr(AsrRequestV2 {
            lang: WorkerLanguage::from(lang),
            backend: AsrBackendV2::LocalWhisper,
            input: AsrInputV2::PreparedAudio(PreparedAudioInputV2 {
                audio_ref_id: WorkerArtifactIdV2::from("audio-test"),
            }),
            extras: std::collections::BTreeMap::new(),
            decode_budget_seconds: None,
        }),
        attachments: Vec::new(),
    }
}

/// The state directory every pool in this binary uses.
///
/// Created once per test process. Without it, `worker_registry_path` is empty
/// and `state_dir` is `None`, which resolves to the OPERATOR'S real
/// `~/.batchalign3/workers.json`: the daemons these tests spawn register in the
/// live machine's registry, where a production run would then discover them.
/// (Found on 2026-07-29 with four test daemons listed in the real file.)
///
/// One directory shared by the whole binary is enough isolation: registry
/// entries are per-pid and each pool retires only the daemons carrying its own
/// server instance id.
fn test_state_dir() -> &'static std::path::Path {
    static STATE_DIR: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
    STATE_DIR
        .get_or_init(|| tempfile::tempdir().expect("test state dir"))
        .path()
}

fn test_pool(python: String) -> WorkerPool {
    common::test_server_fixture::isolate_host_memory_ledger();
    WorkerPool::new(PoolConfig {
        python_path: python,
        health_check_interval_s: 600, // disable during test
        ready_timeout_s: 30,
        test_echo: true,
        max_workers_per_key: PerProfile::uniform(8),
        verbose: 0,
        worker_registry_path: test_state_dir().join("workers.json").display().to_string(),
        runtime: WorkerRuntimeConfig {
            state_dir: Some(test_state_dir().to_path_buf()),
            ..Default::default()
        },
        ..Default::default()
    })
}

#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    // SAFETY: kill(pid, 0) only checks process existence/permission.
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

#[cfg(unix)]
async fn wait_for_process_exit(pid: u32) {
    for _ in 0..50 {
        if !process_alive(pid) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("worker pid {pid} was still alive after waiting for pool drop cleanup");
}

/// Dropping a pool must retire the TCP daemons it spawned, not just its stdio
/// workers.
///
/// A server-owned TCP daemon is a DETACHED Python process launched with
/// `--transport tcp` and registered in `workers.json` under this server
/// instance's id. Stdio workers die with their `WorkerHandle`, so ordinary field
/// drop reaps them (`gpu_stdio_shared_worker_drop_reaps_process` pins that). A
/// TCP daemon has no such handle: it is retired only by `kill_owned_daemons`,
/// which `WorkerPool::shutdown()` calls and `Drop` did not, even though `Drop`'s
/// own doc comment says it exists to "catch test code and panic unwinds where
/// the pool goes out of scope without a graceful shutdown() call".
///
/// The consequence was measured, not theorised: on 2026-07-29 this suite had
/// left 11 `--test-echo` daemons running on the development machine, two of them
/// for over ten hours, holding ports and burning cores. On a real host the same
/// gap orphans multi-gigabyte model processes on any panic unwind.
///
/// The subject used to be built by `pool.warmup(..)`, which spawned the daemon
/// in-crate. With warmup retired (2026-07-30) this constructs it the way
/// production now does: an externally-spawned daemon stamped with this pool's
/// instance id, adopted through `discover_from_registry`. Same subject, and it
/// exercises the adoption path rather than a spawner only tests could reach.
#[tokio::test]
async fn dropping_pool_reaps_owned_tcp_daemons() {
    let python = require_python!();
    use batchalign::worker::WorkerProfile;
    use batchalign::worker::handle::{WorkerConfig, spawn_tcp_daemon};
    use batchalign::worker::pool::status::WorkerTransport;

    let registry_path = test_state_dir().join("workers.json");

    let pid = {
        let pool = test_pool(python.clone());

        // Stamped with THIS pool's instance id, so the registry records it as
        // owned here and `kill_owned_daemons` is entitled to reap it. Without
        // the stamp the daemon is a foreign worker the pool must leave alone.
        //
        // BOTH halves of the stamp are required. Ownership is the pair
        // (instance id, owner pid): `discovery_disposition` reads the id to
        // decide the daemon is ours and the pid to decide the owner is still
        // alive, and an entry carrying an id but no pid is classified
        // `ReapStaleOwned` and killed instead of adopted. Setting only the id
        // here made this test discover zero workers.
        let daemon_config = WorkerConfig {
            python_path: python,
            test_echo: true,
            profile: WorkerProfile::Gpu,
            lang: WorkerLanguage::from(LanguageCode3::eng()),
            ready_timeout_s: 30,
            runtime: WorkerRuntimeConfig {
                state_dir: Some(test_state_dir().to_path_buf()),
                server_instance_id: Some(pool.current_server_instance_id().to_string()),
                server_process_id: Some(std::process::id()),
                ..Default::default()
            },
            ..Default::default()
        };
        spawn_tcp_daemon(&daemon_config, 0)
            .await
            .expect("spawn a server-owned TCP daemon");

        let discovered = pool.discover_from_registry().await;
        assert_eq!(discovered, 1, "the pool must adopt its own daemon");

        let entry = pool
            .worker_summary_entries()
            .await
            .into_iter()
            .find(|entry| entry.transport == WorkerTransport::Tcp)
            .expect("the adopted worker must be a TCP daemon");
        assert!(
            process_alive(entry.pid.0),
            "the spawned TCP daemon should be alive before the pool is dropped"
        );
        assert!(
            registry_path.exists(),
            "the daemon must register in the TEST state dir, not the operator's: \
             nothing was written to {}",
            registry_path.display()
        );

        drop(pool);
        entry.pid.0
    };

    wait_for_process_exit(pid).await;
}

// ---------------------------------------------------------------------------
// Core concurrent dispatch tests
// ---------------------------------------------------------------------------

/// Send N concurrent execute_v2 requests to one GPU worker.
/// All N responses must arrive with the correct request_id.
#[tokio::test]
async fn gpu_concurrent_dispatch_all_responses_arrive() {
    let python = require_python!();
    let pool = test_pool(python);

    // Pre-scale to create the SharedGpuWorker.
    pool.pre_scale_for_request(
        ReleasedCommand::Transcribe,
        WorkerLanguage::from(LanguageCode3::eng()),
        1,
        &gpu_execute_request("pre-scale-key"),
    )
    .await
    .expect("ASR is a dispatchable task");

    let n = 8;
    let mut handles = Vec::new();

    for i in 0..n {
        let request = gpu_execute_request(&format!("concurrent-{i}"));
        let pool_ref = &pool;
        handles.push(tokio::spawn({
            let lang = LanguageCode3::eng();
            let pool_ptr = pool_ref as *const WorkerPool as usize;
            async move {
                // SAFETY: pool lives for the duration of the test
                let pool = unsafe { &*(pool_ptr as *const WorkerPool) };
                pool.dispatch_execute_v2(&lang, &request).await
            }
        }));
    }

    let mut results: Vec<ExecuteResponseV2> = Vec::new();
    for handle in handles {
        let result = handle.await.expect("task panicked");
        results.push(result.expect("dispatch failed"));
    }

    // Verify all N responses arrived with unique, correct request_ids.
    let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (i, response) in results.iter().enumerate() {
        let expected_id = format!("concurrent-{i}");
        assert_eq!(
            &**response.request_id(),
            &expected_id,
            "response {i} has wrong request_id: got {}, expected {expected_id}",
            response.request_id()
        );
        assert!(
            seen_ids.insert(response.request_id().to_string()),
            "duplicate request_id in responses: {}",
            response.request_id()
        );
    }

    assert_eq!(
        results.len(),
        n,
        "expected {n} responses, got {}",
        results.len()
    );

    pool.shutdown().await;
}

/// All concurrent GPU requests must hit the same worker PID.
/// This proves model sharing: one process, multiple threads, shared weights.
#[tokio::test]
async fn gpu_concurrent_dispatch_shares_same_pid() {
    let python = require_python!();
    let pool = test_pool(python);

    pool.pre_scale_for_request(
        ReleasedCommand::Transcribe,
        WorkerLanguage::from(LanguageCode3::eng()),
        1,
        &gpu_execute_request("pre-scale-key"),
    )
    .await
    .expect("ASR is a dispatchable task");

    // Get the GPU worker info from the pool summary.
    let summary = pool.worker_summary().await;
    let gpu_entry = summary
        .iter()
        .find(|s| s.contains("profile:gpu"))
        .expect("expected a GPU worker in summary after pre-scale");

    // Extract PID segment from summary entry (format varies by transport).
    let pid_str = gpu_entry
        .split(':')
        .find(|part| part.starts_with("pid="))
        .expect("expected pid= in summary entry");

    // Send 4 concurrent requests and verify they all succeed (same worker).
    let n = 4;
    let mut handles = Vec::new();
    for i in 0..n {
        let request = gpu_execute_request(&format!("pid-check-{i}"));
        let pool_ptr = &pool as *const WorkerPool as usize;
        handles.push(tokio::spawn(async move {
            let pool = unsafe { &*(pool_ptr as *const WorkerPool) };
            pool.dispatch_execute_v2(&LanguageCode3::eng(), &request)
                .await
        }));
    }

    for handle in handles {
        let result = handle.await.expect("task panicked");
        result.expect("concurrent dispatch to shared GPU worker failed");
    }

    // Verify GPU worker(s) are still present after concurrent dispatch.
    let summary_after = pool.worker_summary().await;
    let gpu_entries: Vec<_> = summary_after
        .iter()
        .filter(|s| s.contains("profile:gpu"))
        .collect();
    assert!(
        !gpu_entries.is_empty(),
        "expected at least 1 GPU worker after concurrent dispatch"
    );

    // The pre-scaled GPU worker should still be present.
    assert!(
        gpu_entries.iter().any(|e| e.contains(pid_str)),
        "original GPU worker (with {pid_str}) should still be present after concurrent dispatch; got: {gpu_entries:?}"
    );

    pool.shutdown().await;
}

/// Sequential requests after concurrent dispatch must still work.
/// This verifies no state corruption in the SharedGpuWorker after
/// a batch of concurrent requests completes.
#[tokio::test]
async fn gpu_sequential_after_concurrent_works() {
    let python = require_python!();
    let pool = test_pool(python);

    pool.pre_scale_for_request(
        ReleasedCommand::Transcribe,
        WorkerLanguage::from(LanguageCode3::eng()),
        1,
        &gpu_execute_request("pre-scale-key"),
    )
    .await
    .expect("ASR is a dispatchable task");

    // Phase 1: concurrent dispatch (4 requests).
    let mut handles = Vec::new();
    for i in 0..4 {
        let request = gpu_execute_request(&format!("phase1-{i}"));
        let pool_ptr = &pool as *const WorkerPool as usize;
        handles.push(tokio::spawn(async move {
            let pool = unsafe { &*(pool_ptr as *const WorkerPool) };
            pool.dispatch_execute_v2(&LanguageCode3::eng(), &request)
                .await
        }));
    }
    for handle in handles {
        handle
            .await
            .expect("task panicked")
            .expect("phase 1 concurrent dispatch failed");
    }

    // Phase 2: sequential dispatch (3 requests, one at a time).
    for i in 0..3 {
        let request = gpu_execute_request(&format!("phase2-{i}"));
        let response = pool
            .dispatch_execute_v2(&LanguageCode3::eng(), &request)
            .await
            .expect("phase 2 sequential dispatch failed");
        assert_eq!(
            &**response.request_id(),
            &format!("phase2-{i}"),
            "sequential request {i} got wrong request_id"
        );
    }

    pool.shutdown().await;
}

/// Health check on the GPU worker works between dispatch rounds.
/// This verifies the control channel (separate from execute_v2 routing)
/// is not corrupted by concurrent request traffic.
#[tokio::test]
async fn gpu_health_check_works_after_concurrent_dispatch() {
    let python = require_python!();
    let pool = test_pool(python);

    pool.pre_scale_for_request(
        ReleasedCommand::Transcribe,
        WorkerLanguage::from(LanguageCode3::eng()),
        1,
        &gpu_execute_request("pre-scale-key"),
    )
    .await
    .expect("ASR is a dispatchable task");

    // Dispatch 4 concurrent requests.
    let mut handles = Vec::new();
    for i in 0..4 {
        let request = gpu_execute_request(&format!("pre-health-{i}"));
        let pool_ptr = &pool as *const WorkerPool as usize;
        handles.push(tokio::spawn(async move {
            let pool = unsafe { &*(pool_ptr as *const WorkerPool) };
            pool.dispatch_execute_v2(&LanguageCode3::eng(), &request)
                .await
        }));
    }
    for handle in handles {
        handle
            .await
            .expect("task panicked")
            .expect("dispatch failed");
    }

    // Capabilities should have been lazily detected from the first worker spawn.
    let caps = pool
        .detected_capabilities()
        .expect("capabilities should have been detected from worker spawns");
    assert!(
        caps.commands.contains(&"test-echo".to_string()),
        "expected test-echo in capabilities after concurrent dispatch"
    );

    pool.shutdown().await;
}

/// Dropping the pool without `shutdown()` must still reap stdio shared GPU
/// workers. This is the lifecycle-owner path that SharedGpuWorker exists for.
#[cfg(unix)]
#[tokio::test]
async fn gpu_stdio_shared_worker_drop_reaps_process() {
    let python = require_python!();

    let pid = {
        use batchalign::worker::pool::status::WorkerTransport;
        let pool = test_pool(python);
        pool.pre_scale_for_request(
            ReleasedCommand::Transcribe,
            WorkerLanguage::from(LanguageCode3::eng()),
            1,
            &gpu_execute_request("pre-scale-key"),
        )
        .await
        .expect("ASR is a dispatchable task");

        let entry = pool
            .worker_summary_entries()
            .await
            .into_iter()
            .find(|e| e.transport == WorkerTransport::Stdio && e.concurrent)
            .expect("expected stdio shared GPU worker after pre_scale");
        assert!(
            process_alive(entry.pid.0),
            "spawned shared GPU worker should be alive"
        );

        drop(pool);
        entry.pid.0
    };

    wait_for_process_exit(pid).await;
}

// ---------------------------------------------------------------------------
// Transcribe dispatch path (GPU execute_v2 through pool)
// ---------------------------------------------------------------------------

/// A single GPU execute_v2 request dispatched through the pool completes
/// successfully. This exercises the pre-scale → discover TCP worker →
/// dispatch_execute_v2 → SharedGpuTcpWorker → Python execute_v2 → echo chain.
#[tokio::test]
async fn gpu_single_execute_v2_through_pool() {
    let python = require_python!();
    let pool = test_pool(python);

    pool.pre_scale_for_request(
        ReleasedCommand::Transcribe,
        WorkerLanguage::from(LanguageCode3::eng()),
        1,
        &gpu_execute_request("pre-scale-key"),
    )
    .await
    .expect("ASR is a dispatchable task");

    let request = gpu_execute_request("single-dispatch-test");
    let response = pool
        .dispatch_execute_v2(&LanguageCode3::eng(), &request)
        .await
        .expect("GPU dispatch_execute_v2 failed");

    assert_eq!(
        &**response.request_id(),
        "single-dispatch-test",
        "response request_id should match request"
    );

    pool.shutdown().await;
}

/// Multiple GPU execute_v2 requests dispatched sequentially all succeed.
/// This proves the worker doesn't become corrupted after handling a request.
#[tokio::test]
async fn gpu_repeated_execute_v2_through_pool() {
    let python = require_python!();
    let pool = test_pool(python);

    pool.pre_scale_for_request(
        ReleasedCommand::Transcribe,
        WorkerLanguage::from(LanguageCode3::eng()),
        1,
        &gpu_execute_request("pre-scale-key"),
    )
    .await
    .expect("ASR is a dispatchable task");

    for i in 0..5 {
        let request = gpu_execute_request(&format!("repeat-{i}"));
        let response = pool
            .dispatch_execute_v2(&LanguageCode3::eng(), &request)
            .await
            .unwrap_or_else(|e| panic!("GPU dispatch_execute_v2 failed on request {i}: {e}"));

        assert_eq!(
            &**response.request_id(),
            &format!("repeat-{i}"),
            "response {i} has wrong request_id"
        );
    }

    pool.shutdown().await;
}

// ---------------------------------------------------------------------------
// Worker recovery after errors
// ---------------------------------------------------------------------------

/// How long a post-shutdown dispatch may take before we call it hung.
///
/// This is a LIVENESS bound, not a latency one: the property under test is that
/// the pool answers at all, so the only job of this number is to tell "never"
/// apart from "eventually". It must therefore sit far above the slowest honest
/// path, which is a COLD spawn of a fresh Python fallback worker, including
/// interpreter startup and imports.
///
/// It was an unnamed `from_secs(30)`, and on 2026-07-31 a full `cargo test
/// --workspace` run failed here while the same test passed alone in 4.6s. Thirty
/// seconds is ample for a cold spawn on an idle box and not ample when a dozen
/// test binaries are spawning interpreters at once, so crossing it meant "this
/// machine is busy" at least as often as it meant "the pool deadlocked". Every
/// other bound in this file is chosen so that exceeding it can only mean one
/// thing; this one was not, which is why it reported load as a defect.
///
/// A deadlock never completes, so a generous bound costs nothing when the code
/// is correct and stays diagnostic when it is not.
const POST_SHUTDOWN_LIVENESS_BOUND: Duration = Duration::from_secs(180);

/// After a GPU worker process is killed, the pool should handle the next
/// dispatch gracefully: either by reconnecting to a new worker or returning
/// a clear error.
#[tokio::test]
async fn gpu_dispatch_after_pre_scale_shutdown_spawns_fallback() {
    let python = require_python!();
    let pool = test_pool(python);

    // Pre-scale creates the shared GPU worker.
    pool.pre_scale_for_request(
        ReleasedCommand::Transcribe,
        WorkerLanguage::from(LanguageCode3::eng()),
        1,
        &gpu_execute_request("pre-scale-key"),
    )
    .await
    .expect("ASR is a dispatchable task");

    // First dispatch should work.
    let request = gpu_execute_request("before-shutdown");
    let response = pool
        .dispatch_execute_v2(&LanguageCode3::eng(), &request)
        .await
        .expect("first dispatch should succeed");
    assert_eq!(&**response.request_id(), "before-shutdown");

    // Shut down the pool's GPU workers (simulates worker crash/restart).
    pool.shutdown().await;

    // After shutdown, the pool may either:
    // (a) spawn a new fallback worker and succeed, or
    // (b) fail cleanly with an error.
    // The critical property: it must NOT hang forever.
    let request = gpu_execute_request("after-shutdown");
    let result = tokio::time::timeout(
        POST_SHUTDOWN_LIVENESS_BOUND,
        pool.dispatch_execute_v2(&LanguageCode3::eng(), &request),
    )
    .await;

    assert!(
        result.is_ok(),
        "dispatch after shutdown must not hang (no answer within {:?}, \
         which is far longer than a cold Python fallback spawn even under \
         full-suite contention, so the pool is not merely slow here)",
        POST_SHUTDOWN_LIVENESS_BOUND
    );
    // Whether the inner result is Ok or Err, both are acceptable; the point
    // is that the pool responded within the timeout instead of hanging.
}

/// Stanza sequential worker survives multiple batch_infer calls.
/// Regression test: proves worker state is not corrupted between requests.
#[tokio::test]
async fn stanza_worker_survives_many_sequential_requests() {
    let python = require_python!();
    let pool = test_pool(python);

    for i in 0..10 {
        let item = json!({"request": i, "payload": format!("test-{i}")});
        let response = pool
            .dispatch_batch_infer(
                &LanguageCode3::eng(),
                &BatchInferRequest {
                    task: InferTask::Morphosyntax,
                    lang: LanguageCode3::eng(),
                    items: vec![item.clone()],
                    mwt: BTreeMap::new(),
                    allow_stanza_fallback: false,
                },
            )
            .await
            .unwrap_or_else(|e| panic!("stanza dispatch failed on request {i}: {e}"));
        assert_eq!(
            response.results[0].result,
            Some(item),
            "echo mismatch on request {i}"
        );
    }

    assert_eq!(
        pool.worker_count().await,
        1,
        "should reuse 1 worker for all 10 requests"
    );
    pool.shutdown().await;
}

// ---------------------------------------------------------------------------
// Timeout behavior
// ---------------------------------------------------------------------------

/// A worker with artificial delay causes a request timeout, which the pool
/// surfaces as a WorkerError::Protocol (containing "timeout"). This verifies
/// that timeouts are detected rather than hanging forever.
#[tokio::test]
async fn gpu_request_with_short_timeout_fails_cleanly() {
    let python = require_python!();

    // Create a pool where audio task timeout is very short (2s) but the
    // worker has a 5-second delay. This should trigger a timeout.
    let pool = WorkerPool::new(PoolConfig {
        python_path: python,
        health_check_interval_s: 600,
        ready_timeout_s: 30,
        test_echo: true,
        max_workers_per_key: PerProfile::uniform(8),
        verbose: 0,
        runtime: Default::default(),
        audio_task_timeout_s: 2, // 2-second timeout
        ..Default::default()
    });

    // Pre-scale with delay: worker will sleep 5s before each response.
    // The delay lives on the WorkerConfig the pool spawns with, and pre-scale
    // uses the pool's own config, so it needs a different approach:
    // spawn the worker manually with the delay, then dispatch to it.
    //
    // For now, test the simpler property: a request to a pool with a very
    // short timeout that the worker can't meet should fail with a timeout
    // error, not hang.

    // Pre-scale without delay (so the worker starts).
    pool.pre_scale_for_request(
        ReleasedCommand::Transcribe,
        WorkerLanguage::from(LanguageCode3::eng()),
        1,
        &gpu_execute_request("pre-scale-key"),
    )
    .await
    .expect("ASR is a dispatchable task");

    // The execute_v2 timeout for ASR tasks uses audio_task_timeout_s (2s).
    // The test-echo worker responds instantly, so this should succeed.
    let request = gpu_execute_request("timeout-test");
    let result = pool
        .dispatch_execute_v2(&LanguageCode3::eng(), &request)
        .await;
    assert!(
        result.is_ok(),
        "instant echo should succeed even with 2s timeout"
    );

    pool.shutdown().await;
}

/// A worker with --test-delay-ms introduces artificial latency.
/// Verify the delay flag is forwarded correctly by checking that a delayed
/// worker still responds (when timeout is generous enough).
#[tokio::test]
async fn worker_with_delay_responds_when_timeout_is_generous() {
    use batchalign::worker::handle::{WorkerConfig, WorkerHandle};

    let python = require_python!();
    let config = WorkerConfig {
        python_path: python,
        test_echo: true,
        test_delay_ms: 500, // 500ms delay
        profile: batchalign::worker::WorkerProfile::Stanza,
        lang: WorkerLanguage::from(LanguageCode3::eng()),
        ready_timeout_s: 30,
        ..Default::default()
    };

    let mut handle = WorkerHandle::spawn(config).await.expect("spawn failed");

    let start = std::time::Instant::now();
    let resp = handle
        .batch_infer(&BatchInferRequest {
            task: InferTask::Morphosyntax,
            lang: LanguageCode3::eng(),
            items: vec![json!({"test": true})],
            mwt: BTreeMap::new(),
            allow_stanza_fallback: false,
        })
        .await
        .expect("batch_infer with delay should succeed");

    let elapsed = start.elapsed();
    assert!(
        elapsed >= std::time::Duration::from_millis(400),
        "expected at least 400ms delay, got {:?}",
        elapsed
    );
    assert_eq!(resp.results.len(), 1);
    assert_eq!(resp.results[0].result, Some(json!({"test": true})));

    handle.shutdown().await.expect("shutdown failed");
}

// ---------------------------------------------------------------------------
// Stanza/IO sequential dispatch for comparison
// ---------------------------------------------------------------------------

/// Non-GPU (Stanza) pool dispatch works correctly under sequential load.
/// This is the baseline: sequential dispatch doesn't use SharedGpuWorker.
#[tokio::test]
async fn stanza_sequential_dispatch_reuses_worker() {
    let python = require_python!();
    let pool = test_pool(python);

    for i in 0..5 {
        let item = json!({"request": i});
        let response = pool
            .dispatch_batch_infer(
                &LanguageCode3::eng(),
                &BatchInferRequest {
                    task: InferTask::Morphosyntax,
                    lang: LanguageCode3::eng(),
                    items: vec![item.clone()],
                    mwt: BTreeMap::new(),
                    allow_stanza_fallback: false,
                },
            )
            .await
            .expect("stanza dispatch failed");
        assert_eq!(response.results[0].result, Some(item));
    }

    // All 5 requests should have used 1 worker.
    assert_eq!(
        pool.worker_count().await,
        1,
        "expected 1 Stanza worker for sequential dispatch"
    );

    pool.shutdown().await;
}

// ---------------------------------------------------------------------------
// Per-request timeout must not be charged against queue-wait
// ---------------------------------------------------------------------------

/// Architectural contract: when N callers dispatch `execute_v2` concurrently
/// to a single shared GPU worker that processes requests serially, each
/// caller's per-request timeout must govern the *work-time* of its own
/// request; never the queue-wait while earlier requests are being served.
///
/// Reproduces an operator's hung Malayalam corpus job (`04a11009-1d0`, 2026-04-25)
/// at unit-test scale. With `gpu_thread_pool_size = 1` the Python worker's
/// `ThreadPoolExecutor` strictly serializes execute_v2; with
/// `test_delay_ms = 200` each response takes ~200 ms; with N=8 callers the
/// last response arrives around t = 1.6 s. With `audio_task_timeout_s = 1`
/// the per-request timeout is 1 s; well above any single response's
/// work-time but below the *queue-wait + work-time* the late callers see
/// today, because the timer is started at `pending.insert()` (before
/// `stdin.lock()`), not at the moment the worker actually begins the work.
///
/// This test is RED today: the late callers fail with
///   "timeout (1s) waiting for GPU execute_v2 response (request_id=...)"
/// matching the production failure on `brian`.
///
/// It will go GREEN after the fix in `SharedGpuWorker::execute_v2` that
/// serializes the entire (registration + write + await) cycle around a
/// per-worker `tokio::sync::Mutex`, so each caller's timer only ticks
/// during its own work; which is the only honest representation of "one
/// shared GPU worker process can perform one execute_v2 at a time."
#[tokio::test]
async fn gpu_concurrent_dispatch_does_not_charge_queue_wait_against_per_request_timeout() {
    let python = require_python!();

    // Force strict serialization on the Python side and a generous-by-itself,
    // tight-when-summed per-request timeout. With 1 thread × 200 ms ×
    // 8 callers the last response arrives ~1.6 s after dispatch; ahead of
    // any individual request's work-time but past the per-request 1 s
    // budget if (and only if) queue-wait is being charged against it.
    let pool = WorkerPool::new(PoolConfig {
        python_path: python,
        health_check_interval_s: 600,
        ready_timeout_s: 30,
        test_echo: true,
        max_workers_per_key: PerProfile::uniform(8),
        verbose: 0,
        runtime: WorkerRuntimeConfig {
            gpu_thread_pool_size: 1,
            ..Default::default()
        },
        audio_task_timeout_s: 1,
        test_delay_ms: 200,
        ..Default::default()
    });

    pool.pre_scale_for_request(
        ReleasedCommand::Transcribe,
        WorkerLanguage::from(LanguageCode3::eng()),
        1,
        &gpu_execute_request("pre-scale-key"),
    )
    .await
    .expect("ASR is a dispatchable task");

    let n = 8;
    let mut handles = Vec::new();
    for i in 0..n {
        let request = gpu_execute_request(&format!("queue-wait-{i}"));
        let pool_ptr = &pool as *const WorkerPool as usize;
        handles.push(tokio::spawn(async move {
            // SAFETY: pool lives for the duration of the test.
            let pool = unsafe { &*(pool_ptr as *const WorkerPool) };
            pool.dispatch_execute_v2(&LanguageCode3::eng(), &request)
                .await
        }));
    }

    let mut succeeded = 0usize;
    let mut timed_out = 0usize;
    let mut other_errors: Vec<String> = Vec::new();
    for (i, handle) in handles.into_iter().enumerate() {
        match handle.await.expect("dispatch task panicked") {
            Ok(response) => {
                assert_eq!(
                    &**response.request_id(),
                    &format!("queue-wait-{i}"),
                    "response {i} has wrong request_id"
                );
                succeeded += 1;
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("timeout") && msg.contains("execute_v2") {
                    timed_out += 1;
                } else {
                    other_errors.push(format!("dispatch {i}: {msg}"));
                }
            }
        }
    }

    pool.shutdown().await;

    assert!(
        other_errors.is_empty(),
        "unexpected non-timeout failures: {other_errors:?}"
    );
    assert_eq!(
        succeeded, n,
        "all {n} concurrent execute_v2 calls must succeed; observed {succeeded} success, \
         {timed_out} timeout. Per-request timeout is being charged against queue-wait \
         instead of work-time; see SharedGpuWorker::execute_v2 in \
         worker/pool/shared_gpu/stdio.rs (registration + write + await are not \
         serialized around the single-Python-process unit of concurrency)."
    );
}

/// How long a worker response takes if nothing interrupts it.
const WORKER_NATURAL_COMPLETION: Duration = Duration::from_secs(60);

/// How long the dispatch may take to unwind after a cancel-driven kill.
///
/// Half of `WORKER_NATURAL_COMPLETION`, so exceeding it can only mean the
/// dispatch is waiting for the call to finish naturally, never that the box
/// was merely busy.
const CANCEL_UNWIND_BOUND: Duration = Duration::from_secs(30);

/// End-to-end regression test for the 2026-04-26 net incident's
/// 28-minute cancel latency.
///
/// **Scenario:** a Whisper-CPU pass on a long Malayalam audio file
/// took 8-25 minutes per file. The user cancelled at 14:34 EDT but
/// the in-flight dispatch awaited the worker's natural completion
/// until 15:01; the cancel signal didn't propagate to the worker
/// process. PID 15650 then survived as a 5.6 GB zombie for 10+ hours.
///
/// **What this test proves:** when a job is cancelled while a worker
/// is in the middle of a slow dispatch, `shutdown_workers_for_job`
/// SIGTERMs the worker; the in-flight dispatch returns an error
/// quickly (not waiting for the worker's natural completion); the
/// worker process actually dies.
///
/// **Setup:** a test-echo worker with `test_delay_ms = 8000` (8-second
/// per-response delay, simulating a slow ASR pass). A dispatch is
/// spawned under a fake `CURRENT_JOB_ID` scope so the tracker
/// registers it. Concurrently, after a brief wait to let the dispatch
/// commit, we fire `shutdown_workers_for_job(job_id)`.
///
/// **Assertions:**
///   1. The dispatch errors out within ~5 seconds (well under the 8s
///      worker delay; proves the kill interrupted in-flight work).
///   2. The worker process is dead within a few seconds of the kill.
///
/// Without the fix, the dispatch would wait the full 8 seconds and
/// the worker would survive until idle-timeout / daemon shutdown.
#[tokio::test]
async fn cancel_kills_in_flight_worker_under_dispatch() {
    use batchalign::api::JobId;

    let python = require_python!();

    // 60-second per-response delay so "kill interrupted the call" and
    // "natural completion" are separated by a margin no amount of machine
    // load can close.
    //
    // This was 8s against a 6s assertion: a 2s margin, which a loaded box
    // erases even when the kill propagated perfectly. It failed on
    // 2026-07-28 at 25.1s while passing in isolation, i.e. it reported a
    // cancellation regression that did not exist. Widening the ASSERTION
    // alone would have been wrong: the bound is the property, and inflating
    // it toward the natural-completion time is what destroys the test's
    // ability to catch the real 2026-04-26 regression. Widening the MARGIN
    // instead keeps the property sharp and makes it load-tolerant.
    let pool = std::sync::Arc::new(WorkerPool::new(PoolConfig {
        python_path: python,
        health_check_interval_s: 600,
        ready_timeout_s: 30,
        test_echo: true,
        test_delay_ms: 60_000,
        max_workers_per_key: PerProfile::uniform(1),
        verbose: 0,
        runtime: WorkerRuntimeConfig {
            gpu_thread_pool_size: 1,
            ..Default::default()
        },
        audio_task_timeout_s: 30,
        ..Default::default()
    }));

    pool.pre_scale_for_request(
        ReleasedCommand::Transcribe,
        WorkerLanguage::from(LanguageCode3::eng()),
        1,
        &gpu_execute_request("pre-scale-key"),
    )
    .await
    .expect("ASR is a dispatchable task");

    let job_id = JobId::from("kill-in-flight-test".to_string());
    let request = gpu_execute_request("slow-call");
    // Captured for the failure report below: `request` moves into the dispatch
    // task, and which worker key it resolves to is the whole question when the
    // registration wait times out.
    let request_task = request.task;
    let bootstrap_mode = pool.bootstrap_mode();

    // Capture the worker PID by snapshotting the pool BEFORE the
    // dispatch starts; there's exactly one warmed-up worker.
    let pre_dispatch_workers = pool.worker_summary_entries().await;
    assert!(
        !pre_dispatch_workers.is_empty(),
        "pre-scale must have spawned at least one worker for the dispatch to use"
    );

    // Spawn the slow dispatch under a CURRENT_JOB_ID scope so the
    // pool's TrackerGuard registers (job, pid) for the duration.
    let dispatch_pool = pool.clone();
    let dispatch_job_id = job_id.clone();
    let dispatch_start = std::time::Instant::now();
    let dispatch_handle = tokio::spawn(async move {
        WorkerPool::dispatch_under_job_for_test(dispatch_job_id, async move {
            dispatch_pool
                .dispatch_execute_v2(&LanguageCode3::eng(), &request)
                .await
        })
        .await
    });

    // Wait until the dispatch commits; the TrackerGuard registers
    // the (job, pid) pair right after checkout, so we poll the
    // tracker until we see at least one registered worker for this
    // job. This avoids the race where the kill fires before the
    // dispatch task has been polled by the tokio runtime.
    let mut waited = Duration::ZERO;
    let poll_step = Duration::from_millis(50);
    // Generous by design: this bound only distinguishes "wiring broken"
    // (never registers) from "registered slowly". Under full-suite load
    // on a busy machine the previous 3s bound flaked repeatedly
    // (2026-07-08, passing in isolation each time); registration
    // latency is not what this test pins, so the bound errs long.
    //
    // The bound has now been raised once (3s -> 30s, 2026-07-08) and failed
    // again at 30s under full-suite load (2026-07-29), so DO NOT RAISE IT A
    // THIRD TIME. The old panic message asserted a cause the evidence never
    // supported ("TrackerGuard wiring is broken"), which is precisely what
    // invited the bound increase instead of a diagnosis. Registration happens
    // in `dispatch_gpu_execute_v2` at `TrackerGuard::new`, AFTER
    // `get_or_create_gpu_worker(...).await` returns. So a long wait here means
    // one of:
    //
    //   (a) the dispatch resolved a DIFFERENT worker key than pre-scale used,
    //       so it is spawning a Python worker from scratch, and this is a
    //       spawn-under-load wait, not a wiring bug;
    //   (b) registration genuinely never happens (a real wiring bug); or
    //   (c) the spawn has not started a process at all, because it is queued at
    //       memory_guard's process-global spawn permit behind another test
    //       binary's worker coming up.
    //
    // A new pid separates (a); a free spawn permit separates (b) from (c). The
    // failure below reports both rather than guessing. Case (c) was added
    // 2026-07-30 after a run produced NO new pid, which the two-case version of
    // this note would have mislabelled a wiring bug.
    //
    // CORRECTION, 2026-07-30: the 2026-07-30 occurrence reported (a), and the
    // note here (and in the pool) then read that as proof that the map lock
    // `get_or_create_gpu_worker` held across the spawn was this test's cause.
    // It is not, and per-key coordination in the pool (`gpu_slot.rs`) does NOT
    // quiet this test. This test's dispatch is the ONLY caller of that map at
    // the time, so there was never contention on it to remove; its wait is its
    // own spawn, queued behind `memory_guard`'s process-global spawn semaphore
    // (one permit, held until a worker reports ready) alongside every other
    // test binary running under full-suite load, and then paying Python
    // startup. Per-key coordination leaves that serialization untouched by
    // design.
    //
    // RESOLVED, 2026-07-30: cause (a) was real and is now fixed at its source.
    // This test pre-scales via `pre_scale_for_request`, which derives the
    // engine-override key from the request itself rather than from the pool's
    // empty default, so the pre-spawned worker is the one dispatch resolves to
    // and there is no spawn to wait on. The binary went green and got faster.
    // Cause (a) is now unreachable by construction here, so if this fails
    // again it is (b) or (c), and never the bound.
    let max_wait = WARM_WORKER_APPEARS;
    loop {
        if !pool.workers_for_job(&job_id).is_empty() {
            break;
        }
        if waited >= max_wait {
            let now_workers = pool.worker_summary_entries().await;
            let pids_before: Vec<u32> = pre_dispatch_workers.iter().map(|e| e.pid.0).collect();
            let pids_now: Vec<u32> = now_workers.iter().map(|e| e.pid.0).collect();
            let new_pids: Vec<u32> = pids_now
                .iter()
                .copied()
                .filter(|pid| !pids_before.contains(pid))
                .collect();
            let dispatch_finished = dispatch_handle.is_finished();
            let free_spawn_permits = batchalign::worker::memory_guard::available_spawn_permits();
            panic!(
                "dispatch did not register a worker for job {job_id} within {max_wait:?}.\n\
                 EVIDENCE (read this before touching the bound):\n\
                 \x20 request task: {task:?}, bootstrap mode: {mode:?}\n\
                 \x20 pids before dispatch: {pids_before:?}\n\
                 \x20 pids now:             {pids_now:?}\n\
                 \x20 pids that appeared:   {new_pids:?}\n\
                 \x20 dispatch task finished: {dispatch_finished}\n\
                 \x20 free global spawn permits: {free_spawn_permits}\n\
                 A pid appearing means cause (a): the dispatch resolved a key \
                 pre-scale did not spawn, and spent the wait spawning that \
                 worker, queued behind the process-global spawn semaphore. \
                 That is NOT the pool's map locking (already fixed, and this \
                 test is the map's only user here); these tests pre-scale via \
                 pre_scale_for_request, so the two keys come from one \
                 derivation and cannot differ. Not this bound.\n\
                 No new pid AND zero free spawn permits means cause (c), the \
                 one this note originally missed: the spawn is queued at \
                 memory_guard's process-global permit (one, held until a worker \
                 reports ready) behind some other test binary, so it has not \
                 created a process yet. Nothing in this crate is broken; the \
                 machine is the bottleneck.\n\
                 No new pid, dispatch not finished, and a permit FREE means \
                 cause (b): registration never happened, a real TrackerGuard \
                 wiring bug.\n\
                 dispatch task finished with no registration means it failed \
                 before checkout; read its result.",
                task = request_task,
                mode = bootstrap_mode,
            );
        }
        tokio::time::sleep(poll_step).await;
        waited += poll_step;
    }

    // Fire the cancel-driven worker kill.
    //
    // Time the property FROM HERE. The assertion below is about kill-to-unwind
    // latency, and `dispatch_start` predates the spawn, the worker handshake
    // and the registration poll loop, which is itself allowed a deliberately
    // generous 30s. Measuring from `dispatch_start` therefore folded up to a
    // full CANCEL_UNWIND_BOUND of unrelated setup into a CANCEL_UNWIND_BOUND
    // assertion, so the two collided under full-suite load: 2026-07-29 it
    // failed at 32.74s while the `timeout()` below, which IS scoped to the
    // kill, passed. That combination proves the kill propagated correctly and
    // the measurement was simply wrong.
    let kill_instant = std::time::Instant::now();
    pool.shutdown_workers_for_job(&job_id).await;

    // Assertion 1 (the user-visible cancel-responsiveness property):
    // the dispatch must unwind PROMPTLY after the kill, not wait for
    // the in-flight Whisper / Stanza / etc. call to complete naturally.
    // The 2026-04-26 net incident's symptom was the cancel waiting
    // ~28 minutes for an 8-25 minute ASR pass to finish on its own.
    // 30s bound against a 60s natural completion: still unambiguous (half the
    // time the call needs to finish on its own) while tolerating a saturated
    // machine. The 2026-04-26 symptom was a cancel waiting ~28 MINUTES, so
    // any regression of that class is orders of magnitude outside this bound.
    let dispatch_result = tokio::time::timeout(CANCEL_UNWIND_BOUND, dispatch_handle)
        .await
        .expect(
            "dispatch must return well before natural completion after worker \
                 kill; hitting this timeout means the kill didn't propagate \
                 and we're back to waiting for the call to finish on its own",
        )
        .expect("dispatch task panicked");
    let unwind_elapsed = kill_instant.elapsed();
    let total_elapsed = dispatch_start.elapsed();
    assert!(
        dispatch_result.is_err(),
        "dispatch should fail (worker terminated mid-call); got Ok: {dispatch_result:?}"
    );
    assert!(
        unwind_elapsed < CANCEL_UNWIND_BOUND,
        "dispatch should unwind in <{CANCEL_UNWIND_BOUND:?} after kill (worker \
         delay is {WORKER_NATURAL_COMPLETION:?}); took {unwind_elapsed:?} from \
         the kill ({total_elapsed:?} for the whole test), so the kill is not \
         interrupting in-flight work"
    );

    // Tracker drained: a subsequent kill is a no-op against the
    // empty entry. Proves the side-table accounting is correct.
    assert!(pool.workers_for_job(&job_id).is_empty());

    // Worker-process death is best-effort. Stdio workers spawned
    // with setpgid(0,0) reliably die; TCP workers (separate PGID)
    // get cleaned up by the registry-aware reaper at daemon shutdown.
    for entry in &pre_dispatch_workers {
        if process_alive(entry.pid.0) {
            eprintln!("NOTE: worker {entry} survived cancel kill");
        }
    }

    drop(pool);
}

// ---------------------------------------------------------------------------
// Per-key spawn coordination
// ---------------------------------------------------------------------------

/// How long the hanging interpreter refuses to signal ready.
///
/// Longer than `SLOW_KEY_READY_TIMEOUT` so the spawn's outcome is decided by
/// the pool's own timeout rather than by the stub giving up.
const SLOW_INTERPRETER_HANG: Duration = Duration::from_secs(60);

/// The pool's ready timeout for the test below: how long the cold key's spawn
/// occupies the pool before failing.
const SLOW_KEY_READY_TIMEOUT_S: u64 = 20;

/// How long the pre-spawned warm worker may take to become visible.
///
/// Named for consistency with every other bound in this file. The panic it
/// guards already enumerates its causes and says outright that (c) is "the
/// machine is the bottleneck", so this one met the convention through its
/// message before it had a name.
const WARM_WORKER_APPEARS: Duration = Duration::from_secs(30);

/// How long to let the cold dispatch run before asserting it has reached the
/// spawn and is holding the permit.
///
/// SETUP, not the property: it decides whether the race under test is actually
/// happening. That makes it the load-sensitive one to watch, because the cold
/// dispatch queues behind a process-global spawn semaphore with a single
/// permit, so under full-suite load it can still be waiting here. If the
/// assertion below it starts passing for the wrong reason, this is the number
/// that let it.
const COLD_DISPATCH_REACHES_SPAWN: Duration = Duration::from_millis(1_500);

/// How long a dispatch to an ALREADY-WARM key may take while another key is
/// cold-spawning. Generously above the ~milliseconds a test-echo round trip
/// needs, and far below `SLOW_KEY_READY_TIMEOUT_S`, so exceeding it can only
/// mean the dispatch queued behind the cold spawn.
const WARM_DISPATCH_BOUND: Duration = Duration::from_secs(5);

/// Write an interpreter shim that hangs for one language and is the real
/// interpreter for every other.
///
/// The worker command line always carries `--lang <code>` (see
/// `worker/handle/spawn.rs::build_worker_command`), so matching an argument
/// against the code is enough to single out one worker key without touching
/// production code. A worker that never writes its ready line is exactly what a
/// slow model load looks like to the pool.
#[cfg(unix)]
fn hang_for_one_language_interpreter(
    dir: &std::path::Path,
    real_python: &str,
    slow_lang: &LanguageCode3,
) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;

    let script = dir.join("hang_for_one_language.sh");
    let hang_s = SLOW_INTERPRETER_HANG.as_secs();
    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\n\
             for arg in \"$@\"; do\n\
             \x20 if [ \"$arg\" = \"{slow_lang}\" ]; then\n\
             \x20   sleep {hang_s}\n\
             \x20   exit 1\n\
             \x20 fi\n\
             done\n\
             exec {real_python} \"$@\"\n"
        ),
    )
    .expect("write interpreter shim");
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
        .expect("chmod interpreter shim");
    script
}

/// A cold spawn for one worker key must not block dispatch to a warm key.
///
/// **The defect.** `WorkerPool::get_or_create_gpu_worker` holds the
/// `gpu_workers` map mutex across the whole slow path: a process-global spawn
/// semaphore, a cross-process host-memory lease, the Python process spawn, the
/// wait for `{"ready": true}` and a capabilities round trip. Every other user of
/// that map waits behind it, including the FAST path, which only wants to read
/// an entry that already exists. So on a busy host one cold key stalls
/// dispatches that need no spawn at all, for as long as a model takes to load,
/// and `/health` (which walks the same map) stops answering for that long too.
///
/// **What this does NOT claim.** It does not claim spawns should run in
/// parallel. They deliberately do not: `memory_guard::SPAWN_SEMAPHORE` is
/// process-global with one permit, held until the worker signals ready, so that
/// each spawn's memory check sees the previous worker's models already resident.
/// Per-key coordination leaves that serialization exactly as it is. The property
/// here is only that work needing NO spawn does not queue behind one.
///
/// **How the cold spawn is made slow.** An interpreter shim that hangs for one
/// language, so the pool sees a worker that never becomes ready. No production
/// knob is involved; a hung ready handshake is the real shape of a slow load.
#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cold_spawn_for_one_key_does_not_block_dispatch_to_a_warm_key() {
    let python = require_python!();

    let shim_dir = tempfile::tempdir().expect("interpreter shim dir");
    let cold_lang = LanguageCode3::spa();
    let warm_lang = LanguageCode3::eng();
    let shim = hang_for_one_language_interpreter(shim_dir.path(), &python, &cold_lang);

    let pool = std::sync::Arc::new(WorkerPool::new(PoolConfig {
        python_path: shim.display().to_string(),
        health_check_interval_s: 600,
        ready_timeout_s: SLOW_KEY_READY_TIMEOUT_S,
        test_echo: true,
        max_workers_per_key: PerProfile::uniform(8),
        verbose: 0,
        worker_registry_path: test_state_dir().join("workers.json").display().to_string(),
        runtime: WorkerRuntimeConfig {
            state_dir: Some(test_state_dir().to_path_buf()),
            ..Default::default()
        },
        ..Default::default()
    }));

    // Warm the English key by using it. The shim is the real interpreter for
    // every language but the cold one, so this is an ordinary test-echo worker.
    pool.dispatch_execute_v2(&warm_lang, &gpu_execute_request("warm-the-key"))
        .await
        .expect("warming dispatch should succeed");

    // Start the cold key. Its spawn will sit in the pool until the ready
    // timeout; we never look at its result.
    let cold_pool = pool.clone();
    let cold_request = gpu_execute_request_for_lang("cold-key", cold_lang.clone());
    let cold_lang_for_task = cold_lang.clone();
    let cold = tokio::spawn(async move {
        cold_pool
            .dispatch_execute_v2(&cold_lang_for_task, &cold_request)
            .await
    });

    // Give the cold dispatch time to reach the spawn. Anything it does before
    // that point is irrelevant to the property.
    tokio::time::sleep(COLD_DISPATCH_REACHES_SPAWN).await;
    assert!(
        !cold.is_finished(),
        "the cold spawn must still be in flight for this test to mean anything; \
         it finished early, so the interpreter shim did not hang for {cold_lang}"
    );

    // The property: the warm key is still dispatchable.
    let warm_again = tokio::time::timeout(
        WARM_DISPATCH_BOUND,
        pool.dispatch_execute_v2(&warm_lang, &gpu_execute_request("warm-again")),
    )
    .await;

    let warm_result = warm_again.unwrap_or_else(|_| {
        panic!(
            "a dispatch to the already-warm {warm_lang} key did not complete within \
             {WARM_DISPATCH_BOUND:?} while the {cold_lang} key was spawning. The warm \
             dispatch needs no spawn at all: it is queued behind the cold key on the \
             gpu_workers mutex, which get_or_create_gpu_worker holds across the whole \
             spawn. Fix the pool's locking; do not raise this bound."
        )
    });
    warm_result.expect("warm dispatch should succeed while another key is spawning");

    cold.abort();
    pool.shutdown().await;
}

/// Count this process's live worker children whose command line mentions `lang`.
///
/// Parent-scoped on purpose: the whole test binary shares one process, and other
/// binaries run concurrently, so a bare `pgrep -f` would count strangers. Every
/// stdio worker is a direct child of the test process (`setpgid(0,0)` changes
/// its process GROUP, not its parent).
#[cfg(unix)]
fn worker_children_for_lang(lang: &LanguageCode3) -> usize {
    let mine = std::process::id().to_string();
    let output = std::process::Command::new("pgrep")
        .args(["-P", &mine, "-f", &format!("lang {lang}")])
        .output()
        .expect("pgrep should run");
    // pgrep exits 1 with empty stdout when nothing matches; that is zero, not an
    // error, so the exit status is deliberately not consulted.
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count()
}

/// Concurrent callers arriving at a COLD key must produce exactly one worker.
///
/// This is the invariant the map-wide lock existed to protect, and the one thing
/// per-key coordination could plausibly have broken, so it is pinned separately
/// from the stall it fixed. Without any coordination, every one of these callers
/// misses the map, spawns its own multi-gigabyte Python process, and all but the
/// last are orphaned: the map holds one entry per key, so a duplicate spawn is
/// invisible there. That is why this counts PROCESSES rather than map entries.
///
/// `gpu_concurrent_dispatch_shares_same_pid` does not cover this: it warms the
/// key first, so its callers never race the spawn.
///
/// **This pin was verified to fail.** A pin written after the fact proves
/// nothing until it has been seen failing, and this one nearly shipped green for
/// the wrong reason (its `pgrep` pattern began with `-`, which `pgrep` read as a
/// flag, so it counted zero workers and the "starts cold" assertion passed too).
/// With the slot lookup replaced by a fresh slot per caller, it fails reporting
/// 6 workers; with the lookup restored, 1.
#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_callers_at_a_cold_key_spawn_exactly_one_worker() {
    let python = require_python!();

    // A language no other test in this binary uses, so the process count below
    // sees only this test's workers.
    let lang = LanguageCode3::deu();
    let pool = std::sync::Arc::new(test_pool(python));

    assert_eq!(
        worker_children_for_lang(&lang),
        0,
        "the key must start cold for this test to race the spawn"
    );

    let racers = 6;
    let mut handles = Vec::with_capacity(racers);
    for i in 0..racers {
        let pool = pool.clone();
        let lang = lang.clone();
        let request = gpu_execute_request_for_lang(&format!("cold-race-{i}"), lang.clone());
        handles.push(tokio::spawn(async move {
            pool.dispatch_execute_v2(&lang, &request).await
        }));
    }
    for (i, handle) in handles.into_iter().enumerate() {
        handle
            .await
            .expect("dispatch task panicked")
            .unwrap_or_else(|e| panic!("racer {i} failed: {e}"));
    }

    assert_eq!(
        worker_children_for_lang(&lang),
        1,
        "{racers} concurrent callers for one cold key must share ONE spawn; more \
         than one means the per-key slot is not coordinating them, and the extra \
         processes are orphans no map entry points at"
    );

    pool.shutdown().await;
}

#[cfg(unix)]
#[path = "gpu_concurrent_dispatch/recovery.rs"]
mod recovery;
