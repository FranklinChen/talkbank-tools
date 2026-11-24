use super::*;
use crate::store::JobStore;
use std::sync::Arc;

use tokio::sync::broadcast;

use crate::cache::UtteranceCache;
use crate::config::ServerConfig;
use crate::runner::context::{ExecutionEngine, RunnerExecutionContext};
use crate::store::{ContentInput, InputKind};
use crate::worker::pool::{PoolConfig, WorkerPool};
use crate::ws::BROADCAST_CAPACITY;

fn build_empty_host() -> DirectExecutionHost {
    let (ws_tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
    let store = Arc::new(JobStore::for_tests(
        ServerConfig::default(),
        None,
        ws_tx,
        Arc::new(crate::clock::SystemClock),
    ));
    // These cases stop before any worker effect or cache operation. The pool
    // remains dormant and the cache explicitly has no persistence.
    let pool = Arc::new(WorkerPool::new(PoolConfig {
        test_echo: true,
        ..PoolConfig::for_tests()
    }));
    let engine = ExecutionEngine::new(RunnerExecutionContext::new(
        pool,
        Arc::new(UtteranceCache::noop()),
        true,
    ));
    DirectExecutionHost::new(store, engine)
}

/// Guards: an ID absent from the current registry produces `JobNotInLocalStore`.
/// Kind: boundary
/// Not a type: the consuming claim graph starts only after the actor observes
/// the registry. A lookup can miss or a job can be deleted before that observation.
/// State: one in-memory registry actor, with a dormant pool and no-op cache.
/// The registry lookup is the real effect. No database or worker process is needed.
/// Concurrency: a fresh store isolates the ID namespace for this async lookup.
#[tokio::test]
async fn a_missing_local_job_is_reported_by_the_shared_runner() {
    let host = build_empty_host();
    let foreign_job = JobId::from("foreign-to-this-store");
    let outcome = run_direct_job(&foreign_job, &host).await;
    assert!(matches!(
        outcome,
        Err(crate::error::ServerError::JobNotInLocalStore(id)) if id == foreign_job
    ));
}

/// Guards: a morphotag refusal fails the job and its unfinished file with the
/// dispatch reason, then releases the run's claim.
/// Kind: policy
/// Not a type: ownership fixes the claim and preparation pairing. Choosing the
/// command-specific refusal text and writing it to both store records remain
/// semantic effects of routing and finalization.
/// State: an isolated in-memory registry receives the real completion writes.
/// No SQL persistence or worker is exercised, so the host has neither effect.
/// Concurrency: the fresh store owns this job and its claim and heartbeat.
#[tokio::test]
async fn a_refused_job_fails_its_run_with_the_refusal_reason() {
    let host = build_empty_host();
    let job = crate::store::test_support::queued_morphotag_job_fixture(ContentInput {
        display: "recording.wav".into(),
        kind: InputKind::Media,
    });
    let job_id = job.identity.job_id.clone();
    host.store.submit(job).await.expect("submit the job");

    run_direct_job(&job_id, &host)
        .await
        .expect("a refusal is a handled end, not a server error");

    let info = host
        .store
        .get(&job_id)
        .await
        .expect("job stays in the store");
    let reason = "'morphotag' needs CHAT input for every file, and at least one file has none";
    assert_eq!(info.status, crate::api::JobStatus::Failed);
    assert_eq!(info.error.as_deref(), Some(reason));
    let [file] = info.file_statuses.as_slice() else {
        panic!("one file expected: {:?}", info.file_statuses);
    };
    assert_eq!(file.error.as_deref(), Some(reason));
    assert!(
        info.active_lease.is_none(),
        "the refused run must release its claim: {:?}",
        info.active_lease
    );
}
