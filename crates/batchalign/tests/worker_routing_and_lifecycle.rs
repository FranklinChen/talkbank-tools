// Integration test target: Cargo compiles this as a separate crate,
// so the lib's `cfg_attr(test, ...)` allow does not apply. Test code
// uses `unwrap`/`expect` by convention.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

//! Tests for concurrent worker routing (T080), cross-platform process management
//! (T081), and platform-specific lifecycle/shutdown behavior (T089).
//!
//! T080: Multi-language dispatch creates separate groups; concurrent dispatch
//!       across groups works correctly.
//! T081: Worker spawn, health check, and shutdown work on the current platform.
//! T089: Graceful shutdown sends SIGTERM (Unix); Drop reaps processes; checked-out
//!       workers are warned during shutdown.

use std::collections::BTreeMap;
use std::time::Duration;

use batchalign::api::LanguageCode3;
use batchalign::host_facts::PerProfile;
#[cfg(unix)]
use batchalign::worker::WorkerPid;
use batchalign::worker::pool::{PoolConfig, WorkerPool};
use batchalign::worker::{BatchInferRequest, InferTask};
use serde_json::json;

/// A probe owns its payload and its route until the reply is checked.
struct RoutedEchoBatch(BatchInferRequest);

impl RoutedEchoBatch {
    fn new(task: InferTask, lang: LanguageCode3, marker: &str) -> Self {
        Self(BatchInferRequest {
            task,
            lang,
            items: vec![json!({"probe": marker})],
            mwt: BTreeMap::new(),
        })
    }

    async fn check(self, pool: &WorkerPool) {
        let response = pool
            .dispatch_batch_infer(&self.0.lang, &self.0)
            .await
            .expect("echo dispatch");
        assert_eq!(response.results.len(), self.0.items.len());
        for (actual, item) in response.results.iter().zip(&self.0.items) {
            assert_eq!(actual.outcome.result(), Some(item));
        }
    }
}

fn test_pool(python: String) -> WorkerPool {
    WorkerPool::new(PoolConfig {
        health_check_interval_s: batchalign::api::PositiveSeconds::literal::<600>(),
        startup_stall: batchalign::worker::LoadingStall::new(
            batchalign::api::PositiveSeconds::literal::<30>(),
        ),
        test_echo: true,
        max_workers_per_key: PerProfile::uniform(8),
        verbose: 0,
        ..PoolConfig::for_tests_with_python(python)
    })
}

/// Guards: Different languages get distinct workers, each returning its own request.
/// Kind: contract
/// Not a type: The real worker pool can route valid requests to the same
/// process or return another request's payload across a pipe.
/// State: Two language-specific echo workers run on a private pool.
/// Concurrency: Joined futures borrow the same pool until both replies arrive.
#[tokio::test]
async fn concurrent_dispatch_to_different_languages() {
    let pool = test_pool(crate::prerequisite::worker_test_python());
    let eng = RoutedEchoBatch::new(InferTask::Morphosyntax, LanguageCode3::eng(), "eng");
    let spa = RoutedEchoBatch::new(InferTask::Morphosyntax, LanguageCode3::spa(), "spa");
    tokio::join!(eng.check(&pool), spa.check(&pool));
    assert_eq!(pool.worker_count().await, 2);
    pool.shutdown().await;
}

/// Guards: Concurrent requests reach distinct profile workers that all exit on shutdown.
/// Kind: journey
/// Not a type: Correctly formed requests and a successful shutdown can still
/// misroute the actual payload or leave an idle child running.
/// State: A private pool hosts three echo processes. Unix inspects their native PIDs.
/// Concurrency: Borrowed futures join before the pool is shut down and reaping observed.
#[tokio::test]
async fn concurrent_dispatch_across_task_types() {
    let pool = test_pool(crate::prerequisite::worker_test_python());
    let morph = RoutedEchoBatch::new(InferTask::Morphosyntax, LanguageCode3::eng(), "morph");
    let translate = RoutedEchoBatch::new(InferTask::Translate, LanguageCode3::eng(), "translate");
    let fa = RoutedEchoBatch::new(InferTask::Fa, LanguageCode3::eng(), "fa");
    tokio::join!(morph.check(&pool), translate.check(&pool), fa.check(&pool));
    assert_eq!(pool.worker_count().await, 3);

    #[cfg(unix)]
    let children = ObservedChildren::running(&pool, 3).await;
    pool.shutdown().await;
    assert_eq!(pool.worker_count().await, 0);
    #[cfg(unix)]
    children.wait_for_exit().await;
}

/// Guards: Dropping a pool without shutdown reaps its live worker.
/// Kind: contract
/// Not a type: Rust drops the owner but only the operating system can witness
/// whether the child then exits and is reaped.
/// State: One private real echo worker and its positive native PID.
/// Concurrency: The pool is dropped before its independently observed process is polled.
#[cfg(unix)]
#[tokio::test]
async fn drop_without_shutdown_still_reaps_workers() {
    let pool = test_pool(crate::prerequisite::worker_test_python());
    RoutedEchoBatch::new(InferTask::Morphosyntax, LanguageCode3::eng(), "drop")
        .check(&pool)
        .await;
    let children = ObservedChildren::running(&pool, 1).await;
    drop(pool);
    children.wait_for_exit().await;
}

/// The observed, live children from one private pool, retaining their typed PIDs.
#[cfg(unix)]
struct ObservedChildren(Vec<WorkerPid>);

#[cfg(unix)]
impl ObservedChildren {
    async fn running(pool: &WorkerPool, expected: usize) -> Self {
        let children: Vec<_> = pool
            .worker_summary_entries()
            .await
            .into_iter()
            .map(|entry| entry.pid)
            .collect();
        assert_eq!(children.len(), expected);
        for &pid in &children {
            assert!(process_alive(pid), "worker {pid} was already gone");
        }
        Self(children)
    }

    async fn wait_for_exit(self) {
        for pid in self.0 {
            for _ in 0..50 {
                if !process_alive(pid) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            assert!(!process_alive(pid), "worker {pid} did not exit");
        }
    }
}

#[cfg(unix)]
fn process_alive(pid: WorkerPid) -> bool {
    let native = i32::try_from(*pid).expect("native Unix PID");
    assert!(native > 0, "worker PID must name a process");
    nix::sys::signal::kill(nix::unistd::Pid::from_raw(native), None).is_ok()
}
