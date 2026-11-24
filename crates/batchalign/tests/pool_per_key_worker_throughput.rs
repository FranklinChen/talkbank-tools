// Integration test target. Tests use unwrap/expect by convention; the
// lib's `cfg_attr(test, ...)` allow does not apply here.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]

//! Per-key worker count throughput sweep for the Stanza profile.
//!
//! Sweep the Stanza per-key worker limit and measure a cold-start burst on
//! the current host. Each sweep point admits a positive per-key count and
//! admits a larger dispatch count from the experiment inputs. Hold that
//! workload constant across sweep points and retain the request population.
//!
//! This opt-in measurement loads actual Stanza models. It isn't a CI gate or a
//! substitute for an end-to-end run through the managed server.
//!
//! ## How to run
//!
//! Single-point invocation (set a positive `STANZA_PER_KEY` and a larger `STANZA_DISPATCHES`):
//!
//! ```text
//! STANZA_PER_KEY=7 STANZA_DISPATCHES=16 cargo test -p batchalign \
//!     --test worker_integration_suite pool_per_key_worker_throughput:: -- --ignored
//! ```
//!
//! Each invocation prints exactly one machine-parseable line on
//! stdout:
//!
//! ```text
//! POOL_THROUGHPUT_RESULT,stanza_per_key=K,elapsed_ms=<ms>,\
//!   permit_rejections=<n>,spawn_rejections=<n>,completed=<n>/<total>
//! ```
//!
//! Aggregate those lines across runs to plot the throughput-vs-K
//! curve and locate the knee for a particular host shape.

use crate::common;

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::time::Instant;

use batchalign::api::LanguageCode3;
use batchalign::host_facts::PerProfile;
use batchalign::worker::pool::{PoolConfig, WorkerPool};
use batchalign::worker::{BatchInferRequest, InferTask};
use common::pool_dispatch::launch_oversubscribed_dispatches;
use serde_json::json;

/// One opt-in measurement owns the declared per-key limit and complete workload.
struct SweepPoint {
    per_key: NonZeroUsize,
    dispatches: NonZeroUsize,
}

impl SweepPoint {
    fn from_env() -> Self {
        fn count(name: &str) -> NonZeroUsize {
            std::env::var(name)
                .unwrap_or_else(|_| panic!("set {name} for the sweep"))
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be positive"))
        }
        let per_key = count("STANZA_PER_KEY");
        let dispatches = count("STANZA_DISPATCHES");
        assert!(
            dispatches > per_key,
            "the burst must exceed the per-key limit"
        );
        Self {
            per_key,
            dispatches,
        }
    }
}

/// Items per request. Each item is a synthetic short utterance, words
/// list of fixed length. Stanza tokenizes + tags each item; the
/// per-request work scales linearly with item count.
///
/// Sized so that one request takes long enough (~few hundred ms on
/// loaded workers) to make queueing-vs-parallelism behavior visible
/// without inflating total sweep runtime. Adjust if the test ends up
/// either too noisy (too short) or too slow (too long).
const ITEMS_PER_REQUEST: usize = 50;

/// Build one synthetic batched-infer item, a list of `words` to be
/// tagged by Stanza, paired with a per-item `lang` field as required by
/// the worker protocol. The vocabulary is fixed-but-realistic so each
/// item exercises tokenize + POS + lemma + depparse stages; varying
/// vocabulary across items would just add noise.
fn synthetic_item(lang: LanguageCode3) -> serde_json::Value {
    json!({
        "words": [
            "the", "child", "wanted", "to", "tell", "the", "story",
            "about", "her", "favorite", "trip", ".",
        ],
        "lang": lang.as_ref(),
    })
}

/// Guards: A cold-start Stanza sweep reports observed capacity, time and
/// rejection counts only with a complete request population.
/// Kind: measurement
/// Not a type: Actual model startup, host scheduler and process admission
/// determine runtime and whether all requests finish under measured contention.
/// State: An explicitly admitted sweep point, real Stanza workers and synthetic text.
/// Concurrency: An opt-in pool owns the burst. The progress deadline bounds stalled work.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "Real Stanza models; opt-in via --ignored. Reads STANZA_PER_KEY and STANZA_DISPATCHES."]
async fn measure_stanza_per_key_throughput() {
    let python = crate::prerequisite::PythonWorker::require().interpreter();

    let sweep = SweepPoint::from_env();
    let stanza_k = sweep.per_key.get();
    let dispatches = sweep.dispatches.get();
    eprintln!(
        "Sweep point: stanza_per_key={stanza_k} \
         (dispatches={dispatches}, \
         ITEMS_PER_REQUEST={ITEMS_PER_REQUEST})"
    );

    // Pool sized at the configured per-key value with global capacity
    // generous enough not to be the binding constraint. We are isolating
    // the per-key axis: gpu and io stay at 1 so they cannot absorb
    // headroom and confound the measurement.
    let pool = std::sync::Arc::new(WorkerPool::new(PoolConfig {
        max_total_workers: dispatches,
        max_workers_per_key: PerProfile {
            gpu: 1,
            stanza: stanza_k,
            io: 1,
        },
        test_echo: false,
        // Long timeouts so cold model loads (~30 s) on every spawn
        // don't trip the readiness gate. The observation includes startup.
        health_check_interval_s: batchalign::api::PositiveSeconds::literal::<600>(),
        startup_stall: batchalign::worker::LoadingStall::new(
            batchalign::api::PositiveSeconds::literal::<180>(),
        ),
        ..PoolConfig::for_tests_with_python(python)
    }));

    let lang = LanguageCode3::eng();

    // All dispatches share the same (profile=stanza, lang=eng,
    // no engine_overrides) key, so the per-key cap is the only thing
    // gating concurrency.
    let started = Instant::now();
    let outcomes = launch_oversubscribed_dispatches(
        &pool,
        dispatches,
        crate::live_deadline::WaitSubject::oversubscribed_model_dispatches(dispatches),
        |_| lang.clone(),
        |lang, _| BatchInferRequest {
            task: InferTask::Morphosyntax,
            lang: lang.clone(),
            items: (0..ITEMS_PER_REQUEST)
                .map(|_| synthetic_item(lang.clone()))
                .collect(),
            mwt: BTreeMap::new(),
        },
    )
    .await;

    let completed = outcomes
        .iter()
        .filter(|response| {
            matches!(response, Ok(Ok(batch)) if batch.results.len() == ITEMS_PER_REQUEST
            && batch.results.iter().all(|item| item.outcome.result().is_some()))
        })
        .count();
    for outcome in &outcomes {
        if !matches!(outcome, Ok(Ok(_))) {
            eprintln!("dispatch outcome: {outcome:?}");
        }
    }
    let elapsed = started.elapsed();
    let metrics = pool.metrics_snapshot();

    // The single machine-parseable result line. The shell loop in
    // `scripts/run_bug_030_sweep.sh` greps stdout for this prefix and
    // appends it to a CSV; keep the format stable.
    println!(
        "POOL_THROUGHPUT_RESULT,stanza_per_key={stanza_k},elapsed_ms={},\
         permit_rejections={},spawn_rejections={},completed={}/{}",
        elapsed.as_millis(),
        metrics.permit_rejections_total,
        metrics.spawn_rejections_total,
        completed,
        dispatches
    );

    pool.shutdown().await;

    assert_eq!(
        completed, dispatches,
        "the sweep must account for every response"
    );
}
