//! Job lifecycle: the top-level async tasks that own a single job's
//! execution from start to finish.
//!
//! - `job_task`: spawned background task for server-queued jobs (lease renewal
//!   + execution + cleanup).
//! - `run_server_job_attempt` / `run_direct_job`, entry points for the two
//!   host flavours (queued vs inline).
//! - `run_hosted_job`: shared core: semaphore acquire → memory reservation →
//!   preflight → dispatch → finalize.
//! - `reserve_job_execution`: host-memory coordinator interaction.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use tracing::{error, info, warn};

use crate::api::JobId;
use crate::host_memory::{BoundedRefusal, JobExecutionPlan, LeaseFailure};
use crate::scheduling::FailureCategory;
use crate::store::{JobStore, RunCompletion, RunnerJobSnapshot};

use super::context::{
    DirectExecutionHost, DispatchHostContext, ExecutionReservationError, HostedJobRunOutcome,
    JobDispatchRequest, MemoryGateFailurePolicy, ServerExecutionHost,
};
use super::job_scope::{ClaimedRunEnd, ExecutionEnd};
use super::util::{FileRunTracker, RunnerEventSink, compute_job_workers, preflight_validate_media};

/// Build the future that owns one background job lifecycle.
///
/// Each invocation is a single dispatch attempt. If the memory gate rejects the
/// attempt (`HostedJobRunOutcome::Requeued`), `job_task` spawns a new delayed
/// `job_task` that retries after the backoff deadline. This self-scheduling loop
/// ensures a requeued job always has a future runner, preventing it from staying
/// `Queued` indefinitely and blocking new submissions for the same files.
///
/// # Implementation note: explicit boxed return type
///
/// `job_task` returns `Pin<Box<dyn Future + Send + 'static>>` rather than using
/// `async fn`. The `Requeued` branch needs to recursively call `job_task` inside
/// a `tokio::spawn` closure. With `async fn`, Rust's `Send` inference would see
/// a self-referential opaque `impl Future` and fail to prove `Send`. Making the
/// return type an explicitly-typed boxed future breaks the recursion: the inner
/// call `job_task(...)` has a concrete `Pin<Box<dyn Future + Send>>` type that
/// the compiler can verify is `Send + 'static`.
pub(crate) fn job_task(
    job_id: JobId,
    host: ServerExecutionHost,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'static>> {
    Box::pin(async move {
        info!(job_id = %job_id, "job_task started on main runtime");
        match run_server_job_attempt(&job_id, &host).await {
            Ok(super::context::HostedJobRunOutcome::Completed) => {}
            Ok(super::context::HostedJobRunOutcome::Requeued { retry_at }) => {
                // The completed attempt already consumed and released its
                // claim. Re-spawn after the backoff deadline. Without this,
                // the job stays Queued with next_eligible_at set but no runner
                // permanently blocking new submissions for the same files.
                let delay = retry_at.saturating_duration_since(host.store.now());
                let host_retry = host.clone();
                let job_id_retry = job_id.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(delay).await;
                    job_task(job_id_retry, host_retry).await;
                });
            }
            Err(e) => {
                error!(
                    job_id = %job_id,
                    error = %e,
                    "Job failed with server error"
                );
            }
        }
    })
}

/// Run the server-owned processing lifecycle for one job.
///
/// Scoped under `CURRENT_JOB_ID` so dispatch-site `TrackerGuard`s
/// can register against this job for cancel-driven worker shutdown.
pub(crate) async fn run_server_job_attempt(
    job_id: &JobId,
    host: &ServerExecutionHost,
) -> Result<HostedJobRunOutcome, crate::error::ServerError> {
    crate::worker::pool::job_tracker::CURRENT_JOB_ID
        .scope(job_id.clone(), async {
            run_hosted_job(
                job_id,
                &host.store,
                &host.engine,
                MemoryGateFailurePolicy::Queued {
                    orchestrator: host.orchestrator.clone(),
                },
            )
            .await
        })
        .await
}

/// Run one direct inline job through the shared execution engine.
pub(crate) async fn run_direct_job(
    job_id: &JobId,
    host: &DirectExecutionHost,
) -> Result<(), crate::error::ServerError> {
    let outcome = run_hosted_job(
        job_id,
        &host.store,
        &host.engine,
        MemoryGateFailurePolicy::FailJob,
    )
    .await;
    match outcome? {
        HostedJobRunOutcome::Completed => Ok(()),
        HostedJobRunOutcome::Requeued { retry_at } => Err(crate::error::ServerError::Persistence(
            format!("direct execution unexpectedly requeued job {job_id} until {retry_at}"),
        )),
    }
}

async fn run_hosted_job(
    job_id: &JobId,
    store: &Arc<JobStore>,
    engine: &super::context::ExecutionEngine,
    memory_gate_policy: MemoryGateFailurePolicy,
) -> Result<HostedJobRunOutcome, crate::error::ServerError> {
    // Exclusive-runner claim. Exactly one live runner may own a job: a
    // restarted job's new runner arrives here while the cancelled one is
    // still tearing down its in-flight file, and must WAIT for the release
    // rather than run concurrently (two runners produced the 2026-07-10
    // bogus-failed / double-counted state). The linear claim binds the run
    // generation and snapshot atomically. Dispatch consumes it, then completion
    // admits its generation and file cleanup in one registry transition.
    let claim = {
        const CLAIM_POLL_INTERVAL_MS: u64 = 250;
        const CLAIM_WARN_EVERY_TICKS: u64 = 240; // one warning per minute
        let mut wait_ticks: u64 = 0;
        loop {
            match store.begin_runner(job_id).await {
                None => {
                    return Err(crate::error::ServerError::JobNotInLocalStore(
                        job_id.clone(),
                    ));
                }
                Some(crate::store::BeginRunnerOutcome::Started { claim, .. }) => {
                    break claim;
                }
                Some(crate::store::BeginRunnerOutcome::Stopped { completion }) => {
                    store
                        .complete_stopped_run(completion, store.event_time())
                        .await?;
                    return Ok(HostedJobRunOutcome::Completed);
                }
                Some(crate::store::BeginRunnerOutcome::RunnerStillLive) => {
                    // Warn once a minute; the previous runner releases as
                    // soon as its in-flight file finishes, which for a
                    // long ASR file can legitimately take minutes.
                    if wait_ticks > 0 && wait_ticks.is_multiple_of(CLAIM_WARN_EVERY_TICKS) {
                        warn!(
                            job_id = %job_id,
                            waited_s = wait_ticks * CLAIM_POLL_INTERVAL_MS / 1000,
                            "waiting for the previous runner to release this job"
                        );
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(CLAIM_POLL_INTERVAL_MS))
                        .await;
                    wait_ticks += 1;
                }
                Some(crate::store::BeginRunnerOutcome::NotRunnable { status }) => {
                    info!(job_id = %job_id, %status, "job no longer requires a runner");
                    return Ok(HostedJobRunOutcome::Completed);
                }
            }
        }
    };

    // Dispatch writes only as this run: its sink is minted from the claim.
    let host_context = DispatchHostContext::for_run(store.clone(), &claim);
    let sink = host_context.sink().clone();
    let job = Arc::clone(claim.snapshot());
    let correlation_id = job.identity.correlation_id.clone();
    let ended = super::job_scope::RunScope::for_run(claim)
        .execute(
            store.clone(),
            execute_claimed_job(job, store, engine, memory_gate_policy, &host_context),
        )
        .await;
    let claim = ended.claim;
    // Every end consumes the claim exactly once.
    let completion = match ended.outcome {
        Ok(ClaimedRunEnd::Executed(ExecutionEnd::Requeued { retry_at })) => {
            store.release_run(claim).await;
            return Ok(HostedJobRunOutcome::Requeued { retry_at });
        }
        // The registry reads the job's own cancellation when finishing.
        Ok(ClaimedRunEnd::Executed(ExecutionEnd::Dispatched) | ClaimedRunEnd::Cancelled) => {
            store.complete_run(claim, sink.now()).await?
        }
        // A refusal is the job's answer, not a server fault: the run fails
        // with its reason on the job and on every unfinished file.
        Ok(ClaimedRunEnd::Executed(ExecutionEnd::Refused(refusal))) => {
            store
                .fail_run(claim, refusal.into_reason(), sink.now())
                .await?
        }
        Err(error) => {
            store.fail_run(claim, error.to_string(), sink.now()).await?;
            return Err(error);
        }
    };
    match completion {
        RunCompletion::Superseded => info!(job_id = %job_id, correlation_id = %correlation_id,
            "run superseded by restart or shutdown; leaving resumable files intact"),
        RunCompletion::Finished {
            status,
            error: Some(reason),
        } => error!(
            job_id = %job_id, correlation_id = %correlation_id, %status, %reason, "Job finished"),
        RunCompletion::Finished {
            status,
            error: None,
        } => info!(
            job_id = %job_id, correlation_id = %correlation_id, %status, "Job finished"),
    }
    Ok(HostedJobRunOutcome::Completed)
}

/// This future owns resource admission and dispatch but not completion: only
/// the consuming RunScope can return that authority to its caller.
async fn execute_claimed_job(
    job: Arc<RunnerJobSnapshot>,
    store: &Arc<JobStore>,
    engine: &super::context::ExecutionEngine,
    memory_gate_policy: MemoryGateFailurePolicy,
    host_context: &DispatchHostContext,
) -> Result<ExecutionEnd, crate::error::ServerError> {
    let job_id = &job.identity.job_id;
    let correlation_id = &job.identity.correlation_id;
    let sink = host_context.sink();

    // Acquire semaphore (blocks if too many concurrent jobs).
    // Lifetime invariant: `store` owns the job-slot semaphore for
    // the daemon's lifetime; the only path that closes the
    // semaphore is daemon shutdown, by which point this task is
    // already cancelled or aborted.
    let _permit = store.acquire_job_slot().await.map_err(|error| {
        crate::error::ServerError::Persistence(format!("job admission closed: {error}"))
    })?;

    // Collect file list to process
    let mut file_list = job.pending_files.clone();
    let command = job.dispatch.command;
    let execution_plan = match reserve_job_execution(host_context, job_id, &job).await {
        Ok(plan) => plan,
        Err(ExecutionReservationError::Capacity {
            requested_workers,
            error,
        }) => match &memory_gate_policy {
            MemoryGateFailurePolicy::Queued { orchestrator } => {
                let disposition = orchestrator.handle_memory_gate_rejection(sink).await?;
                match disposition {
                    super::context::MemoryGateRejectionDisposition::Requeued { retry_at } => {
                        warn!(
                            job_id = %job_id,
                            correlation_id = %correlation_id,
                            requested_workers = requested_workers.0,
                            error = %error,
                            retry_at = %retry_at,
                            "Re-queueing job after host-memory capacity rejection"
                        );
                        return Ok(ExecutionEnd::Requeued { retry_at });
                    }
                }
            }
            MemoryGateFailurePolicy::FailJob => {
                let message = error.to_string();
                sink.bump_memory_gate_aborts().await;
                return Err(crate::error::ServerError::MemoryPressure(message));
            }
        },
        Err(ExecutionReservationError::Fatal(error)) => {
            return Err(error);
        }
    };
    let num_workers = execution_plan.granted_workers;
    let _job_memory_lease = execution_plan.lease;

    // Mark as running only after job execution memory has been reserved.
    sink.mark_job_running().await;

    // Record on job and DB
    sink.record_job_worker_count(num_workers.0).await;

    info!(
        job_id = %job_id,
        correlation_id = %correlation_id,
        num_files = file_list.len(),
        requested_workers = execution_plan.requested_workers.0,
        num_workers = num_workers.0,
        reserved_mb = execution_plan.reserved_mb.0,
        command = %command,
        "Processing files"
    );

    // Pre-validate media files (paths_mode only) to fail fast before worker dispatch.
    let media_failures = preflight_validate_media(&file_list).await;

    // Mark invalid files as errors immediately and collect the valid file list.
    file_list = if media_failures.is_empty() {
        file_list
    } else {
        let failed_indices =
            record_preflight_media_failures(sink.as_ref(), &file_list, &media_failures).await;
        file_list
            .into_iter()
            .filter(|file| !failed_indices.contains(&file.file_index()))
            .collect()
    };

    // Pre-scaling happens after the job's dispatch plan exists, in routing for
    // a job-level command and per file language for align: a GPU command's
    // workers are keyed by that plan, and pre-scaling from the options alone
    // warmed a worker no request reached (`runner::dispatch::pre_scale_workers`).

    let end = engine
        .dispatch_job(
            JobDispatchRequest {
                job: job.clone(),
                file_list,
                num_workers,
            },
            host_context,
        )
        .await?;
    Ok(match end {
        super::context::DispatchEnd::Dispatched => ExecutionEnd::Dispatched,
        super::context::DispatchEnd::Refused(refusal) => ExecutionEnd::Refused(refusal),
    })
}

/// Reserve host memory for the job and compute the execution plan (worker
/// count, memory budget).
async fn reserve_job_execution(
    host: &DispatchHostContext,
    job_id: &JobId,
    job: &RunnerJobSnapshot,
) -> Result<JobExecutionPlan, ExecutionReservationError> {
    let command = job.dispatch.command;
    let requested_workers = compute_job_workers(
        command,
        job.pending_files.len(),
        host.effective_config(),
        host.memory_tier(),
    );
    let coordinator = host.memory_coordinator();
    let job_label = format!(
        "job-execution:{}:{}:{}",
        job_id,
        command,
        job.dispatch.lang.to_worker_language()
    );
    let timeout = host.config().memory_gate_timeout_s.duration();
    let poll_interval = Duration::from_secs(host.config().memory_gate_poll_s.get());
    let plan = coordinator
        .wait_for_job_execution_plan(
            command,
            requested_workers,
            &job_label,
            timeout,
            poll_interval,
        )
        .await;

    match plan {
        Ok(plan) => Ok(plan),
        // Congestion that outlasted the bounded wait: requeue and try later.
        Err(LeaseFailure::Refused(BoundedRefusal::TimedOut(error))) => {
            Err(ExecutionReservationError::Capacity {
                requested_workers,
                error,
            })
        }
        // A plan that can never fit, or a host that cannot be used: waiting
        // would requeue for ever, so the job fails with the reason.
        Err(LeaseFailure::Refused(BoundedRefusal::Final(refusal))) => Err(
            ExecutionReservationError::Fatal(crate::error::ServerError::Persistence(format!(
                "host memory refused job {job_id}: {refusal}"
            ))),
        ),
        Err(LeaseFailure::AttemptDidNotFinish(error)) => Err(ExecutionReservationError::Fatal(
            crate::error::ServerError::Persistence(format!(
                "host-memory planner task failed for job {job_id}: {error}"
            )),
        )),
    }
}

/// Record media-prevalidation failures as explicit setup attempts before the
/// job enters any concrete dispatch path.
pub(super) async fn record_preflight_media_failures(
    sink: &dyn RunnerEventSink,
    file_list: &[crate::store::PendingJobFile],
    media_failures: &HashMap<usize, String>,
) -> HashSet<usize> {
    let mut failed_indices = HashSet::with_capacity(media_failures.len());

    for (&idx, err_msg) in media_failures {
        failed_indices.insert(idx);
        if let Some(file) = file_list.iter().find(|file| file.file_index() == idx) {
            FileRunTracker::new(sink, file.filename())
                .record_setup_failure(err_msg, FailureCategory::Validation)
                .await;
        }
    }

    failed_indices
}

#[cfg(test)]
mod tests {
    use super::*;

    use async_trait::async_trait;
    use tokio::sync::broadcast;

    use crate::cache::UtteranceCache;
    use crate::config::{RuntimeLayout, ServerConfig};
    use crate::db::JobDB;
    use crate::runner::context::{
        ExecutionEngine, MemoryGateRejectionDisposition, QueuedJobOrchestrator,
        RunnerExecutionContext, ServerExecutionHost,
    };
    use crate::worker::pool::{PoolConfig, WorkerPool};
    use crate::ws::BROADCAST_CAPACITY;

    struct UnreachableOrchestrator;

    #[async_trait]
    impl QueuedJobOrchestrator for UnreachableOrchestrator {
        async fn handle_memory_gate_rejection(
            &self,
            _sink: &Arc<dyn RunnerEventSink>,
        ) -> Result<MemoryGateRejectionDisposition, crate::error::ServerError> {
            panic!("memory-gate path must not be reached when the job is missing from the store")
        }
    }

    async fn build_empty_host(tempdir: &std::path::Path) -> ServerExecutionHost {
        let layout = RuntimeLayout::from_state_dir(tempdir.join("state"));
        std::fs::create_dir_all(layout.state_dir()).expect("create state dir");
        let db = Arc::new(
            JobDB::open_with_layout(&layout, Some(layout.state_dir()))
                .await
                .expect("open empty job db"),
        );
        let (ws_tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let config = ServerConfig {
            // `None` falls through to the tier-derived headroom.
            memory_gate_mb: None,
            ..ServerConfig::default()
        };
        let store = Arc::new(JobStore::for_tests(
            config,
            Some(db),
            ws_tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        ));
        let pool = Arc::new(WorkerPool::new(PoolConfig {
            test_echo: true,
            ..PoolConfig::for_tests()
        }));
        let _ = pool.start_background_tasks();
        let cache = Arc::new(
            UtteranceCache::sqlite(Some(tempdir.join("cache")))
                .await
                .expect("open utterance cache"),
        );
        let engine = ExecutionEngine::new(RunnerExecutionContext::new(pool, cache, true));
        ServerExecutionHost::new(store, engine, Arc::new(UnreachableOrchestrator))
    }

    /// A missing local job must never produce a silent
    /// `Ok(HostedJobRunOutcome::Completed)` from `run_server_job_attempt`.
    /// The test guards against a regression that hides local store drift.
    #[tokio::test]
    async fn run_server_job_attempt_does_not_silently_complete_foreign_job() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let host = build_empty_host(tempdir.path()).await;

        let foreign_job = JobId::from("foreign-to-this-store");
        let outcome = run_server_job_attempt(&foreign_job, &host).await;

        assert!(
            !matches!(outcome, Ok(HostedJobRunOutcome::Completed)),
            "run_server_job_attempt reported Completed for a job missing from \
             the local store (outcome: {outcome:?}); expected a typed error."
        );
    }

    /// A job-level refusal is the job's answer, carried back through the
    /// claim: the run fails with the refusal's own reason on the job and on
    /// every file it left unfinished, not a generic "did not reach terminal
    /// status". Morphotag needs CHAT, and this file has none, so dispatch
    /// refuses before reaching any worker.
    #[tokio::test]
    async fn a_refused_job_fails_its_run_with_the_refusal_reason() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let host = build_empty_host(tempdir.path()).await;
        let mut job = crate::store::test_support::running_job_fixture();
        job.execution
            .lifecycle
            .force_status_for_test(crate::api::JobStatus::Queued);
        job.runtime.runner_active = false;
        // Its one file becomes a recording: morphotag cannot take it.
        let crate::store::JobInputs::Content(files) =
            job.execution.lifecycle.admitted_mut_for_test()
        else {
            panic!("the fixture is a content-mode job");
        };
        files[0].kind = crate::store::InputKind::Media;
        let job_id = job.identity.job_id.clone();
        host.store.submit(job).await.expect("submit the job");

        let outcome = run_server_job_attempt(&job_id, &host)
            .await
            .expect("a refusal is a handled end, not a server error");
        assert_eq!(outcome, HostedJobRunOutcome::Completed);

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
}
