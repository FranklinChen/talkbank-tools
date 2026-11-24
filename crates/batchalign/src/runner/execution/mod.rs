//! Host entry points for the consuming claim → prepare → dispatch → finish graph.
//!
//! `attempt` owns the claim and completion transitions. `preparation` admits
//! resources and retains their leases through dispatch. `host` fixes which
//! memory-rejection outcomes each execution host can produce.

mod attempt;
mod host;
mod preparation;
#[cfg(test)]
mod tests;

use crate::api::JobId;
use crate::runner::context::{DirectExecutionHost, ServerExecutionHost};
use attempt::AttemptOutcome;
use tracing::{error, info};

/// Build the future that owns one background job lifecycle.
///
/// Each invocation is a single dispatch attempt. If the memory gate rejects the
/// attempt (`AttemptOutcome::Deferred`), `job_task` spawns a new delayed
/// `job_task` that retries after the backoff deadline. This self-scheduling loop
/// ensures a requeued job always has a future runner, preventing it from staying
/// `Queued` indefinitely and blocking new submissions for the same files.
///
/// # Implementation note: explicit boxed return type
///
/// `job_task` returns `Pin<Box<dyn Future + Send + 'static>>` rather than using
/// `async fn`. The `Deferred` branch needs to recursively call `job_task` inside
/// a `tokio::spawn` closure. With `async fn`, Rust's `Send` inference would see
/// a self-referential opaque `impl Future` and fail to prove `Send`. Making the
/// return type an explicitly-typed boxed future breaks the recursion: the inner
/// call `job_task(...)` has a concrete `Pin<Box<dyn Future + Send>>` type that
/// the compiler can verify is `Send + 'static`.
pub fn job_task(
    job_id: JobId,
    host: ServerExecutionHost,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'static>> {
    Box::pin(async move {
        info!(job_id = %job_id, "job_task started on main runtime");
        match run_server_job_attempt(&job_id, &host).await {
            Ok(AttemptOutcome::Completed) => {}
            Ok(AttemptOutcome::Deferred(retry)) => {
                // The completed attempt already consumed and released its
                // claim. Re-spawn after the backoff deadline. Without this,
                // the job stays Queued with next_eligible_at set but no runner
                // permanently blocking new submissions for the same files.
                let delay = retry
                    .into_deadline()
                    .saturating_duration_since(host.store.now());
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

/// Run one queued attempt. The retry capability exists only for this host.
async fn run_server_job_attempt(
    job_id: &JobId,
    host: &ServerExecutionHost,
) -> Result<AttemptOutcome<crate::store::DeferredRun>, crate::error::ServerError> {
    attempt::run_hosted_job(job_id, host).await
}

/// Run one inline job. Its deferred type has no inhabitants.
pub async fn run_direct_job(
    job_id: &JobId,
    host: &DirectExecutionHost,
) -> Result<(), crate::error::ServerError> {
    match attempt::run_hosted_job(job_id, host).await? {
        AttemptOutcome::Completed => Ok(()),
        AttemptOutcome::Deferred(impossible) => match impossible {},
    }
}
