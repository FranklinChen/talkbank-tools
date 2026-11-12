//! One runner attempt's cancellation scope, including worker preparation.
//!
//! `tokio::spawn` does not inherit task-locals, so a file task spawned by a
//! command's dispatch loses whatever the runner established around it. Two
//! facts must survive that spawn, and this scope carries both:
//!
//! - the job id, re-established inside the task as the worker pool's
//!   `CURRENT_JOB_ID`, so a worker dispatch registers against the job and a
//!   cancel's worker kill can find it;
//! - the job's cancellation token, which the supervisor races against the
//!   task so a cancelled job's task is dropped wherever it is.
//!
//! The runner consumes the scope to supervise the whole dispatch, before
//! capability probing or pre-scaling can start. Child file/background work
//! inherits the same scope. No preparation wait can outlive cancellation just
//! because it occurs before the first file task.
//!
//! See `runner::util::file_status::supervision` for child task supervision.

use std::future::Future;

use tokio_util::sync::CancellationToken;

use crate::api::{DisplayPath, JobId};
use crate::worker::pool::job_tracker::CURRENT_JOB_ID;

use super::util::FileTaskOutcome;

tokio::task_local! {
    /// The job whose file tasks are being spawned. Set once per runner attempt
    /// by `run_hosted_job` around command dispatch.
    static RUN_SCOPE: RunContext;
}

/// Where a file task is being spawned from.
pub(crate) enum SpawnScope {
    /// Inside a runner attempt: the task inherits the job's scope.
    Job(RunContext),
    /// Outside any runner. Only tests of the supervision layer spawn here; in
    /// production it would be a dispatch path that bypassed `run_hosted_job`.
    Unscoped,
}

impl SpawnScope {
    /// The tracing span a file task runs in, naming its file and, inside a
    /// runner, its job.
    ///
    /// `tokio::spawn` starts a future with no span, and the events a file's
    /// work emits (pipeline decisions, engine warnings) come from code that
    /// does not know which file it is working on. Without this span a
    /// 152-file batch interleaves thousands of "needs review" lines that
    /// nothing ties to a file. Built here because the scope is what holds the
    /// job id; the spawn helper, the one route every file task takes, enters it.
    pub(crate) fn file_span(&self, filename: &DisplayPath, role: &'static str) -> tracing::Span {
        match self {
            Self::Job(scope) => {
                tracing::info_span!("file", job_id = %scope.job_id, file = %filename, role)
            }
            Self::Unscoped => tracing::info_span!("file", file = %filename, role),
        }
    }
}

/// Run `work` in the background, under the current job: registered against
/// the job (so a cancel's worker kill finds what it spawns) and dropped the
/// moment the job is cancelled, in the caller's tracing span.
///
/// For work a file task starts but must not wait for, such as warming a
/// language's extra workers after its first probe: the file goes on to its
/// own dispatch while they come up. Nothing joins it; it ends with its work
/// or with the job's cancellation.
pub(crate) fn spawn_background(work: impl Future<Output = ()> + Send + 'static) {
    use tracing::Instrument;

    let span = tracing::Span::current();
    match RunContext::current() {
        SpawnScope::Job(RunContext {
            job_id,
            cancel_token,
        }) => {
            tokio::spawn(
                CURRENT_JOB_ID
                    .scope(job_id, async move {
                        tokio::select! {
                            biased;
                            () = cancel_token.cancelled() => {}
                            () = work => {}
                        }
                    })
                    .instrument(span),
            );
        }
        // Only tests spawn outside a runner; production work spawned here
        // could not be cancelled, which the supervised spawn warns about too.
        SpawnScope::Unscoped => {
            tracing::warn!("background work spawned outside a job scope");
            tokio::spawn(work.instrument(span));
        }
    }
}

/// The span for a file's work spawned from the current task: the one entry
/// every per-file spawn uses (the supervised file tasks, and the morphotag and
/// utseg fan-outs, which run on their own `JoinSet`).
pub(crate) fn current_file_span(filename: &DisplayPath, role: &'static str) -> tracing::Span {
    RunContext::current().file_span(filename, role)
}

/// Shared cancellation context. Children inherit it, but cannot dispatch or
/// finalize the run: only `RunScope` owns that claim.
#[derive(Clone)]
pub(crate) struct RunContext {
    job_id: JobId,
    cancel_token: CancellationToken,
}

/// Linear authority to dispatch exactly once under the claimed input.
pub(super) struct RunScope {
    claim: crate::store::RunnerClaim,
    context: RunContext,
}

/// How execution itself ended. It has no cancelled ending: cancellation is
/// decided by the [`RunScope`] that races execution against the job's
/// cancel token, so execution cannot report one.
#[derive(Debug)]
pub(super) enum ExecutionEnd {
    /// Dispatch ran; files recorded their own terminal states.
    Dispatched,
    /// Host capacity deferred the job; the claim is released for a retry.
    Requeued { retry_at: crate::api::MachineTime },
    /// The whole job was refused; the run fails with this reason.
    Refused(super::context::JobRefusal),
}

/// How a claimed run ended, each way named: the claim's owner turns each one
/// into exactly one store transition.
#[derive(Debug)]
pub(super) enum ClaimedRunEnd {
    /// Execution ran to one of its own endings.
    Executed(ExecutionEnd),
    /// The job's cancellation won before execution ended. Only
    /// [`RunScope::execute`] makes this.
    Cancelled,
}

/// Execution returns the claim with its outcome, never a second run scope.
pub(super) struct EndedDispatch {
    pub claim: crate::store::RunnerClaim,
    pub outcome: Result<ClaimedRunEnd, crate::error::ServerError>,
}

/// The claimed run owns its heartbeat, including abort on dropped execution.
struct RunHeartbeat(tokio::task::JoinHandle<()>);

impl Drop for RunHeartbeat {
    fn drop(&mut self) {
        self.0.abort();
    }
}

impl RunScope {
    /// Bind identity and cancellation from the claimed snapshot, consuming the
    /// only authority that could create another dispatch for this attempt.
    pub(super) fn for_run(claim: crate::store::RunnerClaim) -> Self {
        let job = claim.snapshot();
        let context = RunContext {
            job_id: job.identity.job_id.clone(),
            cancel_token: job.cancel_token.clone(),
        };
        Self { claim, context }
    }

    /// Cancellation wins before polling admission or dispatch. The heartbeat
    /// is created only after the claim exists and ends before claim release.
    pub(super) async fn execute(
        self,
        store: std::sync::Arc<crate::store::JobStore>,
        execution: impl Future<Output = Result<ExecutionEnd, crate::error::ServerError>>,
    ) -> EndedDispatch {
        let Self { claim, context } = self;
        let job_id = context.job_id.clone();
        let cancelled = context.cancel_token.clone();
        let heartbeat_job = job_id.clone();
        let heartbeat = RunHeartbeat(tokio::spawn(async move {
            const STALL_ALARM: std::time::Duration = std::time::Duration::from_secs(30 * 60);
            loop {
                tokio::time::sleep(crate::config::LEASE_HEARTBEAT).await;
                if store.renew_job_lease(&heartbeat_job).await
                    == crate::store::LeaseRenewalOutcome::Stop
                {
                    break;
                }
                if let Some(last_activity) = store.last_file_activity_at(&heartbeat_job).await {
                    let idle = store.now().saturating_duration_since(last_activity);
                    if idle > STALL_ALARM {
                        tracing::error!(job_id = %heartbeat_job, idle_minutes = idle.as_secs() / 60,
                            "job appears STALLED: no file activity while the runner is live");
                    }
                }
            }
        }));
        let outcome = CURRENT_JOB_ID
            .scope(
                job_id,
                RUN_SCOPE.scope(context, async move {
                    tokio::select! {
                        biased;
                        () = cancelled.cancelled() => Ok(ClaimedRunEnd::Cancelled),
                        result = execution => result.map(ClaimedRunEnd::Executed),
                    }
                }),
            )
            .await;
        drop(heartbeat);
        EndedDispatch { claim, outcome }
    }
}

impl RunContext {
    /// The scope the caller is running in.
    pub(crate) fn current() -> SpawnScope {
        match RUN_SCOPE.try_with(Self::clone) {
            Ok(scope) => SpawnScope::Job(scope),
            Err(_) => SpawnScope::Unscoped,
        }
    }

    /// The future a supervised file task actually runs: the command's own
    /// future, registered against the job, and dropped the moment the job is
    /// cancelled.
    pub(crate) async fn supervise(
        self,
        future: crate::owned_future::OwnedFuture<'static, FileTaskOutcome>,
    ) -> FileTaskOutcome {
        let job_id = self.job_id.clone();
        let cancel_token = self.cancel_token.clone();
        CURRENT_JOB_ID
            .scope(
                job_id,
                RUN_SCOPE.scope(self, async move {
                    tokio::select! {
                        // Cancellation first: a task spawned after the cancel
                        // must not take even one step.
                        biased;
                        () = cancel_token.cancelled() => FileTaskOutcome::StoppedByCancellation,
                        outcome = future => outcome,
                    }
                }),
            )
            .await
    }
}
