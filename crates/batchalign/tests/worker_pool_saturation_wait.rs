//! A request that finds the pool saturated waits for a worker; it does not
//! fail.
//!
//! The pool holds one worker in all. The first request (English) takes it and
//! holds it for a few seconds; the second (Spanish) finds no free slot and no
//! idle worker to evict. Its checkout must wait across several report
//! intervals and then succeed once the English worker comes back and can be
//! evicted. Until 2026-10-06 the same configuration failed the second request
//! with "no worker available ... pool saturated" once a fixed checkout
//! deadline passed, which is how a file in a busy batch was lost.

use crate::common;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use batchalign::api::{LanguageCode3, PositiveSeconds};
use batchalign::host_facts::PerProfile;
use batchalign::worker::pool::WorkerPool;
use batchalign::worker::{BatchInferRequest, InferTask};
use common::pool_dispatch::echo_pool_config;
use common::resolve_python;
use serde_json::json;

/// How long the busy worker answers, against a one-second report interval:
/// long enough that the waiting request outlives several intervals.
const BUSY_RESPONSE_MS: u64 = 3_500;

fn request(lang: &LanguageCode3) -> BatchInferRequest {
    BatchInferRequest {
        task: InferTask::Morphosyntax,
        lang: lang.clone(),
        items: vec![json!({"words": ["hello"], "lang": lang.as_ref()})],
        mwt: BTreeMap::new(),
        allow_stanza_fallback: false,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_request_waits_through_a_saturated_pool_instead_of_failing() {
    common::test_server_fixture::isolate_host_memory_ledger();
    let Some(python) = resolve_python() else {
        eprintln!("SKIP: Python 3 with batchalign not available");
        return;
    };

    let mut config = echo_pool_config(python, 1, PerProfile::uniform(1), BUSY_RESPONSE_MS);
    config.checkout_wait_report_interval = Some(PositiveSeconds::literal::<1>());
    // Isolate the global cap from the host's live CPU load.
    config.cpu_gate_threshold_override = Some(f64::INFINITY);
    let pool = Arc::new(WorkerPool::new(config));

    let eng = LanguageCode3::eng();
    let first = {
        let pool = Arc::clone(&pool);
        let eng = eng.clone();
        tokio::spawn(async move { pool.dispatch_batch_infer(&eng, &request(&eng)).await })
    };

    // Let the first request take the only slot before the second arrives.
    let claimed_by = Instant::now() + Duration::from_secs(60);
    while pool.metrics_snapshot().active_workers_total == 0 {
        assert!(
            Instant::now() < claimed_by,
            "the first request never claimed a worker"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let spa = LanguageCode3::spa();
    let waited_from = Instant::now();
    let second = tokio::time::timeout(
        Duration::from_secs(120),
        pool.dispatch_batch_infer(&spa, &request(&spa)),
    )
    .await
    .expect("the waiting request is served once a worker comes back");
    let waited = waited_from.elapsed();

    second.expect("a saturated pool is congestion: the request waits and is served");
    first
        .await
        .expect("join")
        .expect("the request holding the worker completes");
    assert!(
        waited > Duration::from_secs(1),
        "the second request waited past a report interval ({waited:?}), which used to be \
         the deadline that failed it"
    );
    pool.shutdown().await;
}
