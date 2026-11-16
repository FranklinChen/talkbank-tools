//! One consuming lifecycle from registry claim to a finalized attempt.

use tracing::{error, info, warn};

use crate::api::JobId;
use crate::error::ServerError;
use crate::runner::job_scope::{ClaimedRunEnd, RunScope};
use crate::store::{AdmittedRun, RunCompletion, RunnerAdmission};

use super::host::ExecutionHost;
use super::preparation::{ExecutionEnd, ExecutionPreparation};

/// The host's deferred type carries through every stage of the run.
#[derive(Debug, PartialEq)]
pub(super) enum AttemptOutcome<Deferred> {
    Completed,
    Deferred(Deferred),
}

enum ClaimAdmission<'host, H: ExecutionHost> {
    Execute(ClaimedExecution<'host, H>),
    Settled,
}

/// The host, scope and immutable preparation are bound at the claim edge.
/// There is no constructor accepting an independently supplied snapshot.
struct ClaimedExecution<'host, H: ExecutionHost> {
    scope: RunScope,
    preparation: ExecutionPreparation<'host, H>,
}

/// Only execution can return the claim with its actual supervised outcome.
/// Only finalization can release it or commit a completion.
struct EndedAttempt<Deferred> {
    claim: AdmittedRun,
    outcome: Result<ClaimedRunEnd<ExecutionEnd<Deferred>>, ServerError>,
}

pub(super) async fn run_hosted_job<H: ExecutionHost>(
    job_id: &JobId,
    host: &H,
) -> Result<AttemptOutcome<H::Deferred>, ServerError> {
    match ClaimedExecution::acquire(job_id, host).await? {
        ClaimAdmission::Settled => Ok(AttemptOutcome::Completed),
        ClaimAdmission::Execute(claimed) => claimed.execute().await.finalize().await,
    }
}

impl<'host, H: ExecutionHost> ClaimedExecution<'host, H> {
    async fn acquire(
        job_id: &JobId,
        host: &'host H,
    ) -> Result<ClaimAdmission<'host, H>, ServerError> {
        let store = host.store();
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
                match store.admit_runner(job_id).await {
                    None => {
                        return Err(crate::error::ServerError::JobNotInLocalStore(
                            job_id.clone(),
                        ));
                    }
                    Some(RunnerAdmission::Started(claim)) => {
                        break claim;
                    }
                    Some(RunnerAdmission::Stopped(completion)) => {
                        completion.complete().await?;
                        return Ok(ClaimAdmission::Settled);
                    }
                    Some(RunnerAdmission::RunnerStillLive) => {
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
                        tokio::time::sleep(std::time::Duration::from_millis(
                            CLAIM_POLL_INTERVAL_MS,
                        ))
                        .await;
                        wait_ticks += 1;
                    }
                    Some(RunnerAdmission::NotRunnable { status }) => {
                        info!(job_id = %job_id, %status, "job no longer requires a runner");
                        return Ok(ClaimAdmission::Settled);
                    }
                }
            }
        };

        let preparation = ExecutionPreparation::for_claim(host, &claim);
        Ok(ClaimAdmission::Execute(Self {
            scope: RunScope::for_run(claim),
            preparation,
        }))
    }

    async fn execute(self) -> EndedAttempt<H::Deferred> {
        let Self { scope, preparation } = self;
        let ended = scope.execute(preparation.execute()).await;
        EndedAttempt {
            claim: ended.claim,
            outcome: ended.outcome,
        }
    }
}

impl<Deferred> EndedAttempt<Deferred> {
    async fn finalize(self) -> Result<AttemptOutcome<Deferred>, ServerError> {
        let Self { claim, outcome } = self;
        let job_id = claim.snapshot().identity.job_id.clone();
        let correlation_id = claim.snapshot().identity.correlation_id.clone();
        // Every end consumes the claim exactly once.
        let completion = match outcome {
            Ok(ClaimedRunEnd::Executed(ExecutionEnd::Deferred(retry))) => {
                claim.release().await;
                return Ok(AttemptOutcome::Deferred(retry));
            }
            // The registry reads the job's own cancellation when finishing.
            Ok(
                ClaimedRunEnd::Executed(ExecutionEnd::Dispatched | ExecutionEnd::NoLongerRunnable)
                | ClaimedRunEnd::Cancelled,
            ) => claim.complete().await?,
            // A refusal is the job's answer, not a server fault: the run fails
            // with its reason on the job and on every unfinished file.
            Ok(ClaimedRunEnd::Executed(ExecutionEnd::Refused(refusal))) => {
                claim.fail(refusal.into_reason()).await?
            }
            Err(error) => {
                claim.fail(error.to_string()).await?;
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
        Ok(AttemptOutcome::Completed)
    }
}
