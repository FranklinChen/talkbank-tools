//! The job scope a runner attempt hands to every file task it spawns.
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
//! See `runner::util::file_status::supervision` for how the scope is consumed.

use std::future::Future;

use tokio_util::sync::CancellationToken;

use crate::api::JobId;
use crate::worker::pool::job_tracker::CURRENT_JOB_ID;

use super::util::FileTaskOutcome;

tokio::task_local! {
    /// The job whose file tasks are being spawned. Set once per runner attempt
    /// by `run_hosted_job` around command dispatch.
    static FILE_TASK_SCOPE: FileTaskScope;
}

/// Where a file task is being spawned from.
pub(crate) enum SpawnScope {
    /// Inside a runner attempt: the task inherits the job's scope.
    Job(FileTaskScope),
    /// Outside any runner. Only tests of the supervision layer spawn here; in
    /// production it would be a dispatch path that bypassed `run_hosted_job`.
    Unscoped,
}

/// What every file task of one job attempt inherits from its runner.
///
/// Constructed only by the runner, from the job snapshot it is executing, so
/// the job id and the cancellation token always belong to the same attempt.
#[derive(Clone)]
pub(crate) struct FileTaskScope {
    job_id: JobId,
    cancel_token: CancellationToken,
}

impl FileTaskScope {
    /// The scope of one runner attempt of `job_id`.
    pub(crate) fn new(job_id: JobId, cancel_token: CancellationToken) -> Self {
        Self {
            job_id,
            cancel_token,
        }
    }

    /// The scope the caller is running in.
    pub(crate) fn current() -> SpawnScope {
        match FILE_TASK_SCOPE.try_with(Self::clone) {
            Ok(scope) => SpawnScope::Job(scope),
            Err(_) => SpawnScope::Unscoped,
        }
    }

    /// Run a command's dispatch inside this scope, so every file task it
    /// spawns inherits the job's identity and cancellation.
    pub(crate) async fn enclose<F: Future>(self, dispatch: F) -> F::Output {
        FILE_TASK_SCOPE.scope(self, dispatch).await
    }

    /// The future a supervised file task actually runs: the command's own
    /// future, registered against the job, and dropped the moment the job is
    /// cancelled.
    pub(crate) async fn supervise<F>(self, future: F) -> FileTaskOutcome
    where
        F: Future<Output = FileTaskOutcome>,
    {
        let Self {
            job_id,
            cancel_token,
        } = self;
        CURRENT_JOB_ID
            .scope(job_id, async move {
                tokio::select! {
                    // Cancellation first: a task spawned after the cancel
                    // must not take even one step.
                    biased;
                    () = cancel_token.cancelled() => FileTaskOutcome::StoppedByCancellation,
                    outcome = future => outcome,
                }
            })
            .await
    }
}
