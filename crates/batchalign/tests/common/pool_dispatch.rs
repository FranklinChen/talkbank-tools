//! Shared scaffolding for integration tests that drive a [`WorkerPool`]
//! with N concurrent batched-infer dispatches.
//!
//! Two tests in this directory, `worker_pool_thundering_herd` and
//! `pool_per_key_worker_throughput`: share the same shape: build a
//! test-echo pool with a chosen `(max_total, per_key, delay_ms)`,
//! launch CONCURRENT dispatches against it, and observe rejection /
//! completion metrics. This module factors out the parts they were
//! duplicating: the [`PoolConfig`] builder and the dispatch-launch
//! loop. Both call sites previously used an `Arc::as_ptr → &*`
//! reborrow trick to share the pool into spawned tasks; the helper
//! here uses [`Arc::clone`] instead, the atomic clone cost is
//! negligible at the test's scale and the safe form is easier to
//! audit.

use std::sync::Arc;

use batchalign::api::LanguageCode3;
use batchalign::host_facts::PerProfile;
use batchalign::worker::pool::{PoolConfig, WorkerPool};
use batchalign::worker::{BatchInferRequest, BatchInferResponse};

use crate::live_deadline::{ProgressSnapshot, ServerTestDeadline, WaitSubject};

/// Build a `--test-echo` `PoolConfig` with conservative timeouts so a
/// slow Python startup does not trip readiness on a busy host.
///
/// `max_total_workers` and `max_workers_per_key` size the pool's
/// admission caps; `test_delay_ms` lets each echo response wait long
/// enough to make queueing behavior observable.
pub fn echo_pool_config(
    python_path: String,
    max_total_workers: usize,
    max_workers_per_key: PerProfile<usize>,
    test_delay_ms: u64,
) -> PoolConfig {
    PoolConfig {
        python_path,
        max_total_workers,
        max_workers_per_key,
        test_echo: true,
        test_delay_ms,
        health_check_interval_s: batchalign::api::PositiveSeconds::literal::<600>(),
        ready_timeout_s: batchalign::api::PositiveSeconds::literal::<30>(),
        ..PoolConfig::for_tests()
    }
}

/// Outcome of one oversubscribed dispatch: the inner result is the worker
/// pool's own `Result<BatchInferResponse, _>`, the outer is the tokio join.
/// Tests usually flatten with `Ok(Ok(_))` and count successes.
pub type DispatchTaskOutcome = Result<
    Result<BatchInferResponse, batchalign::worker::error::WorkerError>,
    tokio::task::JoinError,
>;

/// Spawn `count` dispatch tasks against `pool`, each using the lang chosen by
/// `lang_for(i)` and the request returned by `build_request(lang, i)`, and
/// wait for all of them under `subject`'s progress deadline. Returns one
/// outcome per task in submission order.
///
/// No dispatch carries its own timeout. A per-dispatch wall clock cannot tell
/// a queue that is draining slowly on a busy machine from one that stopped,
/// and failed the former; the deadline here watches dispatches returning and
/// workers starting, and fails only when neither moves for its idle window
/// (or the burst outlives its ceiling).
pub async fn launch_oversubscribed_dispatches<L, B>(
    pool: &Arc<WorkerPool>,
    count: usize,
    subject: WaitSubject,
    lang_for: L,
    build_request: B,
) -> Vec<DispatchTaskOutcome>
where
    L: Fn(usize) -> LanguageCode3,
    B: Fn(LanguageCode3, usize) -> BatchInferRequest + Send + Sync + 'static,
{
    let build_request = Arc::new(build_request);
    let mut handles = Vec::with_capacity(count);
    for i in 0..count {
        let pool = Arc::clone(pool);
        let lang = lang_for(i);
        let build = Arc::clone(&build_request);
        handles.push(tokio::spawn(async move {
            let request = build(lang.clone(), i);
            pool.dispatch_batch_infer(&lang, &request).await
        }));
    }
    let mut deadline = ServerTestDeadline::new(subject);
    loop {
        let returned = handles.iter().filter(|handle| handle.is_finished()).count();
        if returned == count {
            break;
        }
        deadline
            .keep_waiting(ProgressSnapshot::dispatch_burst(
                returned,
                pool.worker_count().await,
            ))
            .await;
    }
    let mut outcomes = Vec::with_capacity(count);
    for h in handles {
        outcomes.push(h.await);
    }
    outcomes
}

/// Count how many of the dispatch outcomes returned a fully-successful
/// `Ok(Ok(Ok(_)))`. Other shapes (join error, timeout, dispatch error)
/// are not counted; the caller decides whether to log them.
pub fn count_successes(outcomes: &[DispatchTaskOutcome]) -> usize {
    outcomes
        .iter()
        .filter(|outcome| matches!(outcome, Ok(Ok(_))))
        .count()
}
