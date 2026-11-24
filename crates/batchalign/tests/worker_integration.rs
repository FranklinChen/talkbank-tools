// Integration test target: Cargo compiles this as a separate crate,
// so the lib's `cfg_attr(test, ...)` allow does not apply. Test code
// uses `unwrap`/`expect` by convention.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

//! Integration tests for the infer-era Python worker.
//!
//! These tests spawn real Python workers in `--test-echo` mode
//! (no ML models) and verify the Rust side can communicate over the infer-only
//! worker protocol.
//!
//! Requirements: the checkout's Python development environment
//! (`make batchalign-develop`); a test fails, never skips, without it (see
//! `tests/prerequisite.rs`).

use crate::common;

use batchalign::api::{LanguageCode3, NumSpeakers, ReleasedCommand, WorkerLanguage};
use batchalign::host_facts::PerProfile;
use batchalign::worker::error::WorkerError;
use batchalign::worker::handle::{
    WorkerConfig, WorkerHandle, WorkerRuntimeConfig, spawn_tcp_daemon,
};
use batchalign::worker::pool::{PoolConfig, WorkerPool};
use batchalign::worker::registry::{RegistryOwnership, read_registry};
use batchalign::worker::{
    BatchInferRequest, InferRequest, InferTask, WorkerBootstrapMode, WorkerProfile,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::time::Duration;

#[cfg(unix)]
pub fn synthetic_runtime_ready(pid: u32) -> String {
    json!({
        "ready": true,
        "pid": pid,
        "transport": "stdio",
        "runtime": {
            "schema_version": 1,
            "python_version": "3.13.12",
            "python_executable_sha256": "a".repeat(64),
            "batchalign_package_tree_sha256": "b".repeat(64),
            "batchalign_core_extension_sha256": "c".repeat(64),
            "distribution_inventory_sha256": "d".repeat(64),
        },
    })
    .to_string()
}

#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    // Signal 0 probes for existence without delivering anything.
    let pid = i32::try_from(pid).expect("a Unix PID fits pid_t");
    nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_ok()
}

#[cfg(unix)]
fn terminate_pid(pid: u32) {
    let pid = i32::try_from(pid).expect("a Unix PID fits pid_t");
    let _ = nix::sys::signal::kill(
        nix::unistd::Pid::from_raw(pid),
        nix::sys::signal::Signal::SIGTERM,
    );
}

#[cfg(unix)]
async fn wait_for_process_exit(pid: u32) {
    for _ in 0..50 {
        if !process_alive(pid) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("worker pid {pid} was still alive after waiting for exit");
}

#[cfg(unix)]
fn registry_path_for(state_dir: &tempfile::TempDir) -> std::path::PathBuf {
    state_dir.path().join("workers.json")
}

/// A test-echo TCP daemon config pinned to `state_dir`.
///
/// The state dir is threaded through `WorkerRuntimeConfig` so the spawned
/// child writes its registry under this test's own temp directory. It used to
/// be steered by setting `BATCHALIGN_STATE_DIR` on the test process itself,
/// which is process-global and therefore raced the other tests doing the same
/// (2026-07-29). Passing it here keeps these tests independent and parallel.
#[cfg(unix)]
fn test_echo_tcp_config(python_path: String, state_dir: &tempfile::TempDir) -> WorkerConfig {
    WorkerConfig {
        test_echo: true,
        profile: WorkerProfile::Stanza,
        lang: WorkerLanguage::from(LanguageCode3::eng()),
        startup_stall: batchalign::worker::LoadingStall::new(
            batchalign::api::PositiveSeconds::literal::<30>(),
        ),
        runtime: WorkerRuntimeConfig::for_tests().with_state_dir(state_dir.path().to_path_buf()),
        ..WorkerConfig::for_tests_with_python(python_path)
    }
}

fn infer_request(payload: Value) -> InferRequest {
    InferRequest {
        task: InferTask::Morphosyntax,
        lang: LanguageCode3::eng(),
        payload,
    }
}

fn batch_request(task: InferTask, items: Vec<Value>) -> BatchInferRequest {
    BatchInferRequest {
        task,
        lang: LanguageCode3::eng(),
        items,
        mwt: BTreeMap::new(),
    }
}

/// Guards: A real Stanza worker's installed catalog admits Kazakh and English
/// morphology and lacks Quechua.
/// Kind: contract
/// Not a type: Package catalog contents and the Python report are external
/// observations, even after typed language and requirement admission.
/// State: One dedicated per-file Stanza worker reads the installed catalog.
/// Its interpreter must supply Stanza. The probe performs no model inference.
/// Concurrency: The test owns the process and takes the capability response
/// before shutting down its pipe. No shared echo slot is available to it.
#[tokio::test]
#[ignore = "Real Stanza worker and catalog; opt-in via --ignored"]
async fn a_real_workers_catalog_report_admits_kazakh() {
    use batchalign::stanza_registry::StanzaRequirement;
    crate::prerequisite::require_free_memory(4096);
    let python = crate::prerequisite::PythonWorker::require()
        .require_module("stanza")
        .interpreter();
    let config = WorkerConfig {
        test_echo: false,
        profile: WorkerProfile::Stanza,
        task: Some(InferTask::Morphosyntax),
        lang: WorkerLanguage::PerFile,
        startup_stall: batchalign::worker::LoadingStall::new(
            batchalign::api::PositiveSeconds::literal::<120>(),
        ),
        ..WorkerConfig::for_tests_with_python(python)
    };
    let mut worker = WorkerHandle::spawn(
        config,
        std::sync::Arc::new(batchalign::host_memory::NoIdleWorkers),
    )
    .await
    .expect("catalog worker starts");
    let report = worker.capabilities().await;
    worker.shutdown().await.expect("catalog worker shutdown");
    let caps = report.expect("capabilities failed");
    let batchalign::stanza_registry::StanzaCatalogReport::Catalog { languages } =
        caps.stanza_catalog
    else {
        panic!(
            "a real Stanza worker reports its catalog, got {:?}",
            caps.stanza_catalog
        );
    };
    // The server admits a language for morphotag when its entry ships every
    // processor of the requirement; the report is what that reads.
    let serves = |code: &str| {
        languages
            .get(&LanguageCode3::try_new(code).unwrap())
            .is_some_and(|shipped| {
                StanzaRequirement::Morphosyntax
                    .processors()
                    .iter()
                    .all(|required| shipped.contains(required))
            })
    };
    assert!(serves("kaz"));
    assert!(serves("eng"));
    assert!(
        !languages.contains_key(&LanguageCode3::try_new("que").unwrap()),
        "Quechua has no catalog entry"
    );
}

/// Guards: Python's live echo capability report advertises morphology and
/// identifies only FA by engine, with GIL-enabled execution.
/// Kind: contract
/// Not a type: The external Python dispatcher chooses these values when
/// serializing its report. Rust's schema only admits their shape.
/// State: The profile echo worker provides actual capability bytes over stdio.
/// Concurrency: The fixture's exclusive echo lease serializes the exchange.
#[tokio::test]
async fn capabilities_test_echo() {
    let mut lease = common::echo_worker::EchoWorker::Profile
        .checkout()
        .await
        .expect("checkout failed");
    let caps = lease.capabilities().await.expect("capabilities failed");
    assert!(caps.commands.iter().any(|c| c == "test-echo"));
    assert!(caps.commands.iter().any(|c| c == "morphotag"));
    // Test-echo workers intentionally advertise *all* infer tasks. Like a
    // real worker, only forced alignment names an engine (`"test-echo"`) and
    // every other entry is null. The contract is documented in
    // `batchalign/worker/_handlers.py` (`_capabilities` test-echo branch). The
    // pool admits every report it records (`WorkerPool::record_capabilities`)
    // and admission refuses a name for any task but FA, so an echo worker's
    // report must pass that admission like any other. Keep these assertions
    // in sync with the Python handler: when one side changes, the other must
    // match.
    assert!(
        !caps.infer_tasks.is_empty(),
        "test-echo worker must advertise infer tasks so the capability gate passes"
    );
    assert!(
        caps.infer_tasks.contains(&InferTask::Morphosyntax),
        "test-echo capabilities must include Morphosyntax (smoke check on the universal-advertise contract)"
    );
    assert!(
        caps.engine_versions.iter().all(|(task, name)| {
            name.as_ref().map(|name| name.as_str())
                == (*task == InferTask::Fa).then_some("test-echo")
        }),
        "a test-echo worker names only the FA engine, as \"test-echo\"; got {:?}",
        caps.engine_versions
    );
    assert!(!caps.free_threaded);
}

/// Guards: Live single and batch infer replies preserve input payloads and
/// order through Python routing and Rust control completion.
/// Kind: contract
/// Not a type: The peer can return different payloads despite replying in the
/// correct shape. Association and order depend on the actual wire values.
/// State: The profile echo worker supplies both reply forms over one stdio pipe.
/// Concurrency: One exclusive lease retains the pipe for both exchanges.
#[tokio::test]
async fn infer_echo_returns_payload() {
    let mut lease = common::echo_worker::EchoWorker::Profile
        .checkout()
        .await
        .expect("checkout failed");
    let payload = json!({"words": ["hello", "world"], "lang": "eng"});
    let response = lease
        .infer(&infer_request(payload.clone()))
        .await
        .expect("infer failed");
    assert_eq!(response.outcome.result(), Some(&payload));

    let items = vec![
        json!({"words": ["hello"], "lang": "eng"}),
        json!({"words": ["world"], "lang": "eng"}),
    ];
    let response = lease
        .batch_infer(&batch_request(InferTask::Morphosyntax, items.clone()))
        .await
        .expect("batch infer failed");
    let produced: Vec<_> = response
        .results
        .iter()
        .map(|item| item.outcome.result())
        .collect();
    assert_eq!(produced, items.iter().map(Some).collect::<Vec<_>>());
}

/// Guards: Both real echo batches pass through the same idle stdio worker and shutdown empties the pool.
/// Kind: journey
/// Not a type: The native child could be replaced between otherwise successful IPC calls.
/// State: One private pool, real Python worker, two batch payloads and its structured status.
/// Concurrency: The second exchange runs after the first handle returns to its queue.
#[tokio::test]
async fn pool_dispatch_batch_infer_spawns_and_processes() {
    let python = crate::prerequisite::worker_test_python();
    let pool = WorkerPool::new(PoolConfig {
        test_echo: true,
        ..PoolConfig::for_tests_with_python(python)
    });
    let lang = LanguageCode3::eng();

    let mut original = None;
    for marker in ["first", "second"] {
        let item = json!({"words": [marker], "lang": "eng"});
        let response = pool
            .dispatch_batch_infer(
                &lang,
                &batch_request(InferTask::Morphosyntax, vec![item.clone()]),
            )
            .await
            .expect("dispatch failed");
        assert_eq!(response.results[0].outcome.result(), Some(&item));
        assert_eq!(pool.worker_count().await, 1);
        let summary = pool.worker_summary_entries().await;
        let [worker] = summary.as_slice() else {
            panic!("expected one idle echo worker, got {summary:?}");
        };
        assert_eq!(worker.profile, "profile:stanza");
        assert_eq!(worker.lang, lang.as_ref());
        assert_eq!(
            worker.transport,
            batchalign::worker::pool::WorkerTransport::Stdio
        );
        assert_eq!(*original.get_or_insert(worker.pid), worker.pid);
    }
    assert_eq!(pool.observed_worker_runtimes().len(), 1);
    pool.shutdown().await;
    assert_eq!(pool.worker_count().await, 0);
}

/// Guards: Task bootstrap separates two Stanza tasks into correctly labelled workers.
/// Kind: contract
/// Not a type: The valid enum values must still survive argv and pool routing to separate real children.
/// State: A private task-bootstrap pool and two model-free stdio launches.
/// Concurrency: Sequential requests expose reuse of the wrong group if it occurs.
#[tokio::test]
async fn pool_task_bootstrap_separates_same_profile_tasks() {
    let python = crate::prerequisite::worker_test_python();
    let pool = WorkerPool::new(PoolConfig {
        health_check_interval_s: batchalign::api::PositiveSeconds::literal::<60>(),
        startup_stall: batchalign::worker::LoadingStall::new(
            batchalign::api::PositiveSeconds::literal::<30>(),
        ),
        test_echo: true,
        max_workers_per_key: PerProfile::uniform(8),
        verbose: 0,
        runtime: WorkerRuntimeConfig::for_tests().with_bootstrap_mode(WorkerBootstrapMode::Task),
        ..PoolConfig::for_tests_with_python(python)
    });

    let morph_item = json!({"task": "morph"});
    let coref_item = json!({"task": "coref"});
    let r1 = pool
        .dispatch_batch_infer(
            &LanguageCode3::eng(),
            &batch_request(InferTask::Morphosyntax, vec![morph_item.clone()]),
        )
        .await
        .expect("dispatch 1 failed");
    let r2 = pool
        .dispatch_batch_infer(
            &LanguageCode3::eng(),
            &batch_request(InferTask::Coref, vec![coref_item.clone()]),
        )
        .await
        .expect("dispatch 2 failed");
    assert_eq!(r1.results[0].outcome.result().cloned(), Some(morph_item));
    assert_eq!(r2.results[0].outcome.result().cloned(), Some(coref_item));
    assert_eq!(pool.worker_count().await, 2);
    let summary = pool.worker_summary().await;
    assert!(
        summary
            .iter()
            .any(|entry| entry.starts_with("infer:morphosyntax:eng:")),
        "expected infer:morphosyntax worker in summary: {summary:?}"
    );
    assert!(
        summary
            .iter()
            .any(|entry| entry.starts_with("infer:coref:eng:")),
        "expected infer:coref worker in summary: {summary:?}"
    );

    pool.shutdown().await;
}

/// Guards: Pre-scaling morphotag and align publishes their Stanza and GPU worker routes.
/// Kind: contract
/// Not a type: A valid released command can still pre-scale the wrong live route.
/// State: A private echo pool pre-starts the two actual processes.
/// Concurrency: Launches end before inspecting the pool and shutting it down.
#[tokio::test]
async fn pool_pre_scale_uses_infer_targets() {
    let python = crate::prerequisite::worker_test_python();
    let pool = WorkerPool::new(PoolConfig {
        health_check_interval_s: batchalign::api::PositiveSeconds::literal::<60>(),
        startup_stall: batchalign::worker::LoadingStall::new(
            batchalign::api::PositiveSeconds::literal::<30>(),
        ),
        test_echo: true,
        max_workers_per_key: PerProfile::uniform(8),
        verbose: 0,
        ..PoolConfig::for_tests_with_python(python)
    });

    pool.pre_scale(
        ReleasedCommand::Morphotag,
        WorkerLanguage::from(LanguageCode3::eng()),
        1,
    )
    .await;
    pool.pre_scale(
        ReleasedCommand::Align,
        WorkerLanguage::from(LanguageCode3::eng()),
        1,
    )
    .await;

    let summary = pool.worker_summary().await;
    // morphotag → Stanza profile (sequential group), align → GPU profile (SharedGpuWorker)
    assert_eq!(pool.worker_count().await, 2);
    assert!(
        summary
            .iter()
            .any(|entry| entry.starts_with("profile:stanza:eng:")),
        "expected a Stanza profile worker in summary: {summary:?}"
    );
    assert!(
        summary
            .iter()
            .any(|entry| entry.starts_with("profile:gpu:eng:")),
        "expected a GPU profile worker in summary: {summary:?}"
    );

    pool.shutdown().await;
}

/// Guards: Concurrent discovery publishes one worker. Later discovery preserves its original registry entry, and the adopted connection continues serving requests.
/// Kind: journey
/// Not a type: A daemon's real TCP stream, registry file and process lifetime can change independently. Unique claims exclude duplicate preparation, while this native adapter must preserve actual source bytes and communication through repeated discovery.
/// State: One real Python echo daemon, a private registry directory and an admitted pool exercise actual health, capability and batch-infer bytes without models or providers.
/// Concurrency: The fixture owns its loopback daemon, ephemeral port, directory and pool. Cleanup precedes the final failure assertions. Repeated scans and actual request completion establish readiness without inferred sleeps or shared registry changes.
#[cfg(unix)]
#[tokio::test]
async fn discover_from_registry_seeds_capabilities_from_external_tcp_daemon() {
    let python = crate::prerequisite::worker_test_python();
    let state_dir = tempfile::TempDir::new().expect("tempdir");
    // State dir threaded via config, not the environment: see
    // `WorkerRuntimeConfig::state_dir`.
    let registry_path = registry_path_for(&state_dir);

    let external = test_echo_tcp_config(python.clone(), &state_dir);
    let (external_pid, _external_port) =
        spawn_tcp_daemon(&external, 0, &batchalign::host_memory::NoIdleWorkers)
            .await
            .expect("spawn external tcp daemon");

    let pool = WorkerPool::new(PoolConfig {
        health_check_interval_s: batchalign::api::PositiveSeconds::literal::<60>(),
        startup_stall: batchalign::worker::LoadingStall::new(
            batchalign::api::PositiveSeconds::literal::<30>(),
        ),
        test_echo: true,
        max_workers_per_key: PerProfile::uniform(8),
        verbose: 0,
        worker_registry_path: Some(registry_path.clone()),
        ..PoolConfig::for_tests_with_python(python)
    });

    let (first, concurrent) =
        tokio::join!(pool.discover_from_registry(), pool.discover_from_registry());
    let capabilities_observed = pool.detected_capabilities().is_some();

    let entries_after_first = read_registry(&registry_path);
    let second_adoption = pool.discover_from_registry().await;
    let entries_after_second = read_registry(&registry_path);
    let retained_workers = pool.worker_count().await;
    let retained_request = pool
        .dispatch_batch_infer(
            &LanguageCode3::eng(),
            &batch_request(InferTask::Morphosyntax, vec![json!({"retained": true})]),
        )
        .await;
    let daemon_alive = process_alive(external_pid);
    pool.shutdown().await;
    if process_alive(external_pid) {
        terminate_pid(external_pid);
        wait_for_process_exit(external_pid).await;
    }
    assert_eq!(first + concurrent, 1, "one discovered external worker");
    assert!(
        capabilities_observed,
        "the worker seeds a live capability snapshot"
    );
    assert!(
        retained_request.is_ok(),
        "the retained connection still serves its actual request"
    );
    assert!(daemon_alive, "the registered daemon is still alive");
    assert_eq!(
        second_adoption, 0,
        "the already-adopted daemon requires no second adoption"
    );
    assert_eq!(
        retained_workers, 1,
        "one actual daemon must own one pool slot"
    );
    assert_eq!(
        entries_after_first.len(),
        1,
        "first adoption keeps its live registry source"
    );
    assert_eq!(
        entries_after_second.len(),
        1,
        "a repeated scan preserves the live adopted registry entry"
    );
}

/// Guards: Discovery removes a stale server-owned registry row and terminates that daemon.
/// Kind: journey
/// Not a type: An admitted registry row can outlive its actual owner process and retain a live daemon.
/// State: One real server-owned TCP worker and a private registry.
/// Concurrency: The fixture waits for the native exit before inspecting the remaining file.
#[cfg(unix)]
#[tokio::test]
async fn discover_from_registry_reaps_stale_foreign_server_owned_daemon() {
    let python = crate::prerequisite::worker_test_python();
    let state_dir = tempfile::TempDir::new().expect("tempdir");
    // State dir threaded via config, not the environment: see
    // `WorkerRuntimeConfig::state_dir`.
    let registry_path = registry_path_for(&state_dir);

    let mut stale_owned = test_echo_tcp_config(python.clone(), &state_dir);
    stale_owned.runtime.server_instance_id = Some("dead-owner-instance".to_string());
    stale_owned.runtime.server_process_id = Some(4_000_000);
    let (stale_pid, _stale_port) =
        spawn_tcp_daemon(&stale_owned, 0, &batchalign::host_memory::NoIdleWorkers)
            .await
            .expect("spawn stale server-owned tcp daemon");

    let pool = WorkerPool::new(PoolConfig {
        health_check_interval_s: batchalign::api::PositiveSeconds::literal::<60>(),
        startup_stall: batchalign::worker::LoadingStall::new(
            batchalign::api::PositiveSeconds::literal::<30>(),
        ),
        test_echo: true,
        max_workers_per_key: PerProfile::uniform(8),
        verbose: 0,
        runtime: WorkerRuntimeConfig {
            server_instance_id: Some("current-server-instance".to_string()),
            server_process_id: Some(std::process::id()),
            ..WorkerRuntimeConfig::for_tests()
        },
        worker_registry_path: Some(registry_path.clone()),
        ..PoolConfig::for_tests_with_python(python)
    });

    let discovered = pool.discover_from_registry().await;
    assert_eq!(
        discovered, 0,
        "orphaned server-owned daemons should be reaped, not reused"
    );
    wait_for_process_exit(stale_pid).await;

    let entries = read_registry(&registry_path);
    assert!(
        entries.is_empty(),
        "stale server-owned registry entries should be removed: {entries:?}"
    );

    pool.shutdown().await;
}

/// Guards: Pool shutdown retires its server-owned daemon and preserves an external daemon.
/// Kind: journey
/// Not a type: The persisted ownership tags must select the correct real process during shutdown.
/// State: A private registry, two real TCP daemons and their recorded ownership.
/// Concurrency: The test verifies the owned daemon exits, then reaps supplied external work.
#[cfg(unix)]
#[tokio::test]
async fn shutdown_only_kills_current_server_owned_daemons() {
    let python = crate::prerequisite::worker_test_python();
    let state_dir = tempfile::TempDir::new().expect("tempdir");
    // State dir threaded via config, not the environment: see
    // `WorkerRuntimeConfig::state_dir`.
    let registry_path = registry_path_for(&state_dir);

    let external = test_echo_tcp_config(python.clone(), &state_dir);
    let (external_pid, _external_port) =
        spawn_tcp_daemon(&external, 0, &batchalign::host_memory::NoIdleWorkers)
            .await
            .expect("spawn external tcp daemon");

    let mut owned = test_echo_tcp_config(python.clone(), &state_dir);
    owned.runtime.server_instance_id = Some("owning-server-instance".to_string());
    owned.runtime.server_process_id = Some(std::process::id());
    let owned_runtime = owned.runtime.clone();
    let (owned_pid, _owned_port) =
        spawn_tcp_daemon(&owned, 0, &batchalign::host_memory::NoIdleWorkers)
            .await
            .expect("spawn server-owned tcp daemon");

    let pool = WorkerPool::new(PoolConfig {
        health_check_interval_s: batchalign::api::PositiveSeconds::literal::<60>(),
        startup_stall: batchalign::worker::LoadingStall::new(
            batchalign::api::PositiveSeconds::literal::<30>(),
        ),
        test_echo: true,
        max_workers_per_key: PerProfile::uniform(8),
        verbose: 0,
        runtime: owned_runtime,
        worker_registry_path: Some(registry_path.clone()),
        ..PoolConfig::for_tests_with_python(python)
    });

    pool.shutdown().await;
    wait_for_process_exit(owned_pid).await;
    assert!(
        process_alive(external_pid),
        "shutdown should preserve external registry daemons"
    );

    let entries = read_registry(&registry_path);
    assert_eq!(
        entries.len(),
        1,
        "expected only the external daemon to remain"
    );
    assert_eq!(entries[0].pid, external_pid);
    assert_eq!(entries[0].ownership, RegistryOwnership::External);

    terminate_pid(external_pid);
    wait_for_process_exit(external_pid).await;
}

/// Guards: Pre-scaling fills exactly two allowed slots even when asked for four.
/// Kind: contract
/// Not a type: A runtime cap must govern the real workers that complete startup.
/// State: Private Stanza echo route and its worker count.
/// Concurrency: The pool's pre-scale operation finishes before we count or shut down.
#[tokio::test]
async fn pool_pre_scale_respects_max_workers_per_key() {
    let python = crate::prerequisite::worker_test_python();
    let pool = WorkerPool::new(PoolConfig {
        health_check_interval_s: batchalign::api::PositiveSeconds::literal::<60>(),
        startup_stall: batchalign::worker::LoadingStall::new(
            batchalign::api::PositiveSeconds::literal::<30>(),
        ),
        test_echo: true,
        max_workers_per_key: PerProfile::uniform(2),
        verbose: 0,
        ..PoolConfig::for_tests_with_python(python)
    });

    pool.pre_scale(
        ReleasedCommand::Morphotag,
        WorkerLanguage::from(LanguageCode3::eng()),
        4,
    )
    .await;
    let count = pool.worker_count().await;
    assert_eq!(
        count, 2,
        "both permitted slots should be live after pre-scaling"
    );

    pool.shutdown().await;
}

/// Guards: Same-key launch windows never overlap while three caller requests contend.
/// Kind: contract
/// Not a type: Independent child processes can overlap before readiness while typed claims are held.
/// State: A private wrapper holds one directory-backed launch witness until the worker publishes.
/// Concurrency: Three joined callers own unique requests, and the native wrapper latches any overlap.
#[cfg(unix)]
#[tokio::test]
async fn pool_serializes_worker_bootstrap_per_key() {
    let python = crate::prerequisite::worker_test_python();
    let dir = tempfile::TempDir::new().expect("tempdir");
    let wrapped_python = dir.path().join("wrapped-python");
    let overlap = dir.path().join("overlapped");
    std::fs::write(dir.path().join("python"), format!("{python}\n")).expect("interpreter");
    std::fs::write(
        dir.path().join("bootstrap-proxy.py"),
        r#"import sys
from pathlib import Path
from batchalign.worker import _main

starting = Path(__file__).with_name("starting")
original = _main.start_worker
def publish(startup):
    original(startup)
    starting.rmdir()

assert sys.argv[1:3] == ["-P", "-c"], sys.argv[1:3]
sys.argv = ["batchalign-worker", *sys.argv[4:]]
_main.start_worker = publish
_main.main()
"#,
    )
    .expect("write startup observer");
    std::fs::write(
        &wrapped_python,
        r#"#!/bin/sh
dir=$(dirname "$0")
if ! mkdir "$dir/starting"; then
    : > "$dir/overlapped"
    exit 1
fi
IFS= read -r interpreter < "$dir/python"
exec "$interpreter" -P "$dir/bootstrap-proxy.py" "$@"
"#,
    )
    .expect("write wrapped python");
    let mut perms = std::fs::metadata(&wrapped_python)
        .expect("metadata")
        .permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&wrapped_python, perms).expect("chmod wrapped python");

    let pool = WorkerPool::new(PoolConfig {
        health_check_interval_s: batchalign::api::PositiveSeconds::literal::<60>(),
        startup_stall: batchalign::worker::LoadingStall::new(
            batchalign::api::PositiveSeconds::literal::<30>(),
        ),
        test_echo: true,
        max_workers_per_key: PerProfile::uniform(3),
        verbose: 0,
        ..PoolConfig::for_tests_with_python(wrapped_python.to_string_lossy().into_owned())
    });

    let item1 = json!({"request": 1});
    let item2 = json!({"request": 2});
    let item3 = json!({"request": 3});
    let lang = LanguageCode3::eng();
    let request1 = batch_request(InferTask::Morphosyntax, vec![item1.clone()]);
    let request2 = batch_request(InferTask::Morphosyntax, vec![item2.clone()]);
    let request3 = batch_request(InferTask::Morphosyntax, vec![item3.clone()]);
    let (r1, r2, r3) = tokio::join!(
        pool.dispatch_batch_infer(&lang, &request1),
        pool.dispatch_batch_infer(&lang, &request2),
        pool.dispatch_batch_infer(&lang, &request3),
    );
    assert_eq!(
        r1.expect("dispatch 1 failed").results[0]
            .outcome
            .result()
            .cloned(),
        Some(item1)
    );
    assert_eq!(
        r2.expect("dispatch 2 failed").results[0]
            .outcome
            .result()
            .cloned(),
        Some(item2)
    );
    assert_eq!(
        r3.expect("dispatch 3 failed").results[0]
            .outcome
            .result()
            .cloned(),
        Some(item3)
    );
    assert_eq!(pool.worker_count().await, 3);
    assert!(!overlap.exists(), "same-key native bootstrap overlapped");

    pool.shutdown().await;
}

/// Guards: An absent interpreter gives the spawn error variant without reporting ready.
/// Kind: contract
/// Not a type: An executable path value cannot guarantee the operating system can open it.
/// State: A unique nonexistent file in a private directory crosses the real spawn boundary.
/// Concurrency: No worker or ambient interpreter is needed.
#[tokio::test]
async fn spawn_failure_bad_python_path() {
    let dir = tempfile::tempdir().expect("private absent interpreter directory");
    let missing = dir.path().join("missing-python");
    let config = WorkerConfig {
        test_echo: true,
        ..WorkerConfig::for_tests_with_python(missing.to_string_lossy().into_owned())
    };

    let err = match WorkerHandle::spawn(
        config,
        std::sync::Arc::new(batchalign::host_memory::NoIdleWorkers),
    )
    .await
    {
        Err(e) => e,
        Ok(_) => panic!("expected spawn to fail with bad python path"),
    };
    assert!(
        matches!(err, WorkerError::SpawnFailed(_)),
        "expected SpawnFailed, got: {err}"
    );
}

/// Guards: A real non-JSON stdout preamble does not prevent ready admission.
/// Kind: wire
/// Not a type: A child can emit arbitrary text outside an otherwise correct ready protocol.
/// State: A private shell emits noisy stdout, then a valid synthetic ready frame.
/// Concurrency: The test owns the process and the handle retires it on drop.
#[cfg(unix)]
#[tokio::test]
async fn spawn_tolerates_non_json_stdout_preamble_before_ready() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let fake_python = dir.path().join("fake-python");
    let ready = synthetic_runtime_ready(1234);
    std::fs::write(
        &fake_python,
        format!(
            "#!/bin/sh\nprintf 'Downloading: \"https://example.invalid/model.pt\" to /tmp/model.pt\\n'\nprintf '%s\\n' '{ready}'\nsleep 30\n"
        ),
    )
    .expect("write fake python");
    let mut perms = std::fs::metadata(&fake_python)
        .expect("metadata")
        .permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&fake_python, perms).expect("chmod fake python");

    let config = WorkerConfig {
        test_echo: true,
        profile: WorkerProfile::Stanza,
        lang: WorkerLanguage::from(LanguageCode3::eng()),
        num_speakers: NumSpeakers(1),
        startup_stall: batchalign::worker::LoadingStall::new(
            batchalign::api::PositiveSeconds::literal::<30>(),
        ),
        ..WorkerConfig::for_tests_with_python(fake_python.to_string_lossy().into_owned())
    };

    let handle = WorkerHandle::spawn(
        config,
        std::sync::Arc::new(batchalign::host_memory::NoIdleWorkers),
    )
    .await
    .expect("spawn failed");
    assert!(*handle.pid() > 0, "should have a valid pid");
    assert_eq!(handle.transport(), "stdio");
}

/// Guards: Load noise between requests leaves the subsequent health and shutdown replies readable.
/// Kind: wire
/// Not a type: A live worker pipe can interleave arbitrary stdout with valid control frames.
/// State: A private shell emits a ready frame, load noise and health/shutdown responses.
/// Concurrency: One mutable handle owns the sequential request and response stream.
#[cfg(unix)]
#[tokio::test]
async fn health_check_tolerates_non_protocol_stdout_between_requests() {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let fake_python = dir.path().join("fake-python");
    let ready = synthetic_runtime_ready(1234);
    std::fs::write(
        &fake_python,
        format!(
            "#!/bin/sh\nprintf '%s\\n' '{ready}'\nIFS= read -r req || exit 1\nprintf 'torch: loading checkpoint shards\\n'\nprintf '{{\"op\":\"health\",\"response\":{{\"status\":\"ok\",\"command\":\"profile:stanza\",\"lang\":\"eng\",\"pid\":1234,\"uptime_s\":0}}}}\\n'\nIFS= read -r req || exit 0\nprintf '{{\"op\":\"shutdown\"}}\\n'\n"
        ),
    )
    .expect("write fake python");
    let mut perms = std::fs::metadata(&fake_python)
        .expect("metadata")
        .permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&fake_python, perms).expect("chmod fake python");

    let config = WorkerConfig {
        test_echo: true,
        profile: WorkerProfile::Stanza,
        lang: WorkerLanguage::from(LanguageCode3::eng()),
        num_speakers: NumSpeakers(1),
        startup_stall: batchalign::worker::LoadingStall::new(
            batchalign::api::PositiveSeconds::literal::<30>(),
        ),
        ..WorkerConfig::for_tests_with_python(fake_python.to_string_lossy().into_owned())
    };

    let mut handle = WorkerHandle::spawn(
        config,
        std::sync::Arc::new(batchalign::host_memory::NoIdleWorkers),
    )
    .await
    .expect("spawn failed");
    let health = handle.health_check().await.expect("health check failed");
    assert_eq!(health.status, batchalign::worker::WorkerHealthStatus::Ok);
    assert_eq!(health.command, "profile:stanza");
    assert_eq!(health.lang, WorkerLanguage::from(LanguageCode3::eng()));

    handle.shutdown().await.expect("shutdown failed");
}

/// Guards: Morphosyntax and utseg echo payloads are served by the same native Stanza process.
/// Kind: contract
/// Not a type: Two legal tasks in one profile can still be launched on different processes.
/// State: One private echo pool and typed child IDs observed after both requests.
/// Concurrency: Each request returns its worker before the next borrows it.
#[tokio::test]
async fn profile_groups_related_tasks_into_single_worker() {
    let python = crate::prerequisite::worker_test_python();
    let pool = WorkerPool::new(PoolConfig {
        test_echo: true,
        ..PoolConfig::for_tests_with_python(python)
    });
    let lang = LanguageCode3::eng();
    let mut original = None;
    for task in [InferTask::Morphosyntax, InferTask::Utseg] {
        let item = json!({"task": format!("{task:?}")});
        let response = pool
            .dispatch_batch_infer(&lang, &batch_request(task, vec![item.clone()]))
            .await
            .expect("same-profile dispatch");
        assert_eq!(response.results[0].outcome.result(), Some(&item));
        let summary = pool.worker_summary_entries().await;
        let [worker] = summary.as_slice() else {
            panic!("expected one idle same-profile worker: {summary:?}");
        };
        assert_eq!(*original.get_or_insert(worker.pid), worker.pid);
    }
    pool.shutdown().await;
}
