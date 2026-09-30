//! A GPU-profile worker that serves one request at a time must not be the
//! only process for its key.
//!
//! Field failure (2026-09-30): on a CPU-only host (`force_cpu`, one request in
//! flight per process), Whisper UTR requests from two jobs queued on ONE
//! worker process while `max_workers_per_key` allowed several. The Rust pool treated every GPU-profile worker as a shared
//! concurrent process and kept exactly one per key; the Python worker, told
//! `--force-cpu`, served its requests one after another. Two owners decided
//! "is this worker concurrent", and they disagreed.

use super::*;
use std::sync::Arc;

/// The CPU-only host's worker runtime: forced CPU, one request per process.
fn cpu_only_runtime() -> WorkerRuntimeConfig {
    WorkerRuntimeConfig {
        force_cpu: true,
        gpu_thread_pool_size: 1,
        state_dir: Some(test_state_dir().to_path_buf()),
        ..Default::default()
    }
}

fn cpu_only_pool(python: String, per_key: usize) -> WorkerPool {
    common::test_server_fixture::isolate_host_memory_ledger();
    WorkerPool::new(PoolConfig {
        python_path: python,
        health_check_interval_s: 600,
        ready_timeout_s: 60,
        test_echo: true,
        max_workers_per_key: PerProfile::uniform(per_key),
        verbose: 0,
        worker_registry_path: test_state_dir().join("workers.json").display().to_string(),
        runtime: cpu_only_runtime(),
        ..Default::default()
    })
}

/// Pre-scaling a Whisper ASR key to two workers on a CPU-only host yields
/// two worker processes, each serving one request at a time: the configured
/// per-key capacity is used, not collapsed into one shared process.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_request_per_process_gpu_keys_scale_to_the_configured_capacity() {
    let python = require_python!();
    let pool = Arc::new(cpu_only_pool(python, 2));

    pool.pre_scale_for_request(
        ReleasedCommand::Transcribe,
        WorkerLanguage::from(LanguageCode3::eng()),
        2,
        &gpu_execute_request("pre-scale-cpu-only"),
    )
    .await
    .expect("ASR is a dispatchable task");

    let gpu_workers: Vec<_> = pool
        .worker_summary_entries()
        .await
        .into_iter()
        .filter(|entry| entry.profile == "profile:gpu")
        .collect();
    assert_eq!(
        gpu_workers.len(),
        2,
        "a CPU-only GPU key allowed two workers must have two: {gpu_workers:?}"
    );
    assert!(
        gpu_workers.iter().all(|entry| !entry.concurrent),
        "a worker serving one request at a time must not be dispatched to as a \
         shared concurrent process: {gpu_workers:?}"
    );

    // Two requests at once are served by the two processes, each exclusively.
    let lang = LanguageCode3::eng();
    let (request_a, request_b) = (gpu_execute_request("cpu-a"), gpu_execute_request("cpu-b"));
    let (first, second) = tokio::join!(
        pool.dispatch_execute_v2(&lang, &request_a),
        pool.dispatch_execute_v2(&lang, &request_b)
    );
    assert_eq!(&**first.expect("first dispatch").request_id(), "cpu-a");
    assert_eq!(&**second.expect("second dispatch").request_id(), "cpu-b");
}
