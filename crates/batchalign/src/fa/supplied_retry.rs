//! Shared full/incremental terminal retry under original supplied authority.

use crate::chat_ops::fa::resource_partition::ResourceProbeOutcome;
use crate::chat_ops::fa::supplied_retry::{
    SuppliedRetryAnswer, SuppliedRetrySession, SuppliedRetryStep,
};
use crate::chat_ops::fa::{BulletRepairPolicy, FaApplied, FaFinalized, SuppliedRetrySource};
use crate::error::ServerError;
use crate::runner::util::{FileStage, ProgressSender, ProgressUpdate};
use crate::types::traces::FaFallbackEventTrace;

use super::preparation::{
    FaPreparationContext, PreparationWorkPermit, PreparedFaInput, RetainedPreparation,
    SourceRetryPreparation, materialize_once,
};
use super::transport::{FaWorkerSingle, FaWorkerTransport, OptionalFaInference};
use super::units::{
    CandidateCacheReceipts, CandidateCacheState, DispatchUnit, store_unit_evidence,
};

pub(super) struct FaRetryInputs<'a> {
    pub(super) preparation: FaPreparationContext<'a>,
    pub(super) retry: SourceRetryPreparation,
    pub(super) cached: CandidateCacheReceipts,
}

impl FaRetryInputs<'_> {
    pub(super) async fn finalize(
        mut self,
        applied: FaApplied,
        chat: &mut crate::chat_ops::ChatFile,
        repair: BulletRepairPolicy,
        progress: Option<&ProgressSender>,
    ) -> Result<(FaFinalized, Vec<FaFallbackEventTrace>), ServerError> {
        let imposed = applied
            .then_impose(chat)
            .map_err(super::kept_bullets_failed)?;
        let mut session = SuppliedRetrySession::new(imposed, chat, self.retry.into_scoped());
        let mut fallbacks = Vec::new();
        loop {
            match session.next() {
                SuppliedRetryStep::Complete(imposed) => {
                    return imposed
                        .then_finalize(chat, repair)
                        .map(|finalized| (finalized, fallbacks))
                        .map_err(super::kept_bullets_failed);
                }
                SuppliedRetryStep::Pending(pending) => {
                    if let Some(tx) = progress {
                        let _ = tx.send(ProgressUpdate::new(
                            FileStage::PreparingAlignment,
                            None,
                            None,
                        ));
                    }
                    let (pending, claim) = pending.start();
                    let answer = match resolve_attempt(
                        pending.source(),
                        claim,
                        self.preparation,
                        &mut self.cached,
                        &mut fallbacks,
                        progress,
                    )
                    .await
                    {
                        Ok(answer) => answer,
                        Err(error) => optional_failure(error)?,
                    };
                    session = pending.complete(answer);
                }
            }
        }
    }
}

async fn resolve_attempt(
    source: &SuppliedRetrySource,
    claim: RetainedPreparation<PreparedFaInput>,
    context: FaPreparationContext<'_>,
    cached: &mut CandidateCacheReceipts,
    fallbacks: &mut Vec<FaFallbackEventTrace>,
    progress: Option<&ProgressSender>,
) -> Result<SuppliedRetryAnswer, ServerError> {
    let unit = DispatchUnit::supplied(source, context);
    let unit = &unit;
    let key = unit.cache_key().as_str();
    let available = match claim {
        RetainedPreparation::Settled => {
            return Ok(SuppliedRetryAnswer::Unresolved {
                reason: format!(
                    "identical scope {key} already evaluated; no repeated preparation or execution"
                ),
            });
        }
        RetainedPreparation::Refused(cause) => {
            return Ok(SuppliedRetryAnswer::Unresolved {
                reason: format!("retained exact-scope refusal for {key}: {cause:?}"),
            });
        }
        RetainedPreparation::Ready(PreparedFaInput::Cached(answer)) => {
            return Ok(answer.into_supplied_retry(key));
        }
        RetainedPreparation::Ready(PreparedFaInput::Inference(prepared)) => {
            AvailableRetryInput::Retained(prepared)
        }
        RetainedPreparation::Fresh(permit) => AvailableRetryInput::Fresh(permit),
    };
    let request = source.request();
    cached.lookup(request, context).await;
    if let CandidateCacheState::Admitted(answer) = cached.state(unit) {
        return Ok(answer.clone().into_supplied_retry(key));
    }
    // A missing optional cache entry retains the primary partial output. It
    // cannot acquire a worker-inference permit under RequireCache.
    let authorization =
        match OptionalFaInference::optional_source(context.params.cache_policy, unit) {
            OptionalFaInference::CacheOnly => {
                return Ok(SuppliedRetryAnswer::Unresolved {
                    reason: format!("optional required-cache miss for {key}; inference forbidden"),
                });
            }
            OptionalFaInference::Authorized(authorization) => authorization,
        };
    let prepared = match available {
        AvailableRetryInput::Retained(prepared) => *prepared,
        AvailableRetryInput::Fresh(permit) => {
            match materialize_once(request, context, permit, key).await? {
                ResourceProbeOutcome::Prepared(prepared) => prepared,
                ResourceProbeOutcome::Limited(cause) => {
                    return Ok(SuppliedRetryAnswer::Unresolved {
                        reason: format!("exact-scope resource refusal for {key}: {cause:?}"),
                    });
                }
            }
        }
    };
    if let Some(tx) = progress {
        let _ = tx.send(ProgressUpdate::new(FileStage::Aligning, None, None));
    }
    let projection = FaWorkerTransport::production(context.services)
        .infer_one(FaWorkerSingle {
            authorization,
            prepared,
            worker_lang: context.worker_lang.into(),
            engine: context.params.engine,
        })
        .await?
        .into_projection();
    let source = format!(
        "cache key {key}; {:?}; fallback {:?}",
        projection.source(),
        projection.fallback_event
    );
    if let Some(event) = projection.fallback_event {
        fallbacks.push(event);
    }
    if let Some(raw) = projection.raw_evidence {
        store_unit_evidence(context.services, key, &projection.timings, raw).await;
    }
    Ok(SuppliedRetryAnswer::Timings {
        timings: projection.timings,
        source,
    })
}

enum AvailableRetryInput {
    Retained(Box<super::preparation::PreparedFaRequest>),
    Fresh(PreparationWorkPermit),
}

/// Optional recovery failure preserves primary output; job cancellation still
/// leaves this phase and must reach the runner's cancellation transition.
fn optional_failure(error: ServerError) -> Result<SuppliedRetryAnswer, ServerError> {
    match error {
        ServerError::Cancelled => Err(ServerError::Cancelled),
        error => Ok(SuppliedRetryAnswer::Unresolved {
            reason: format!("optional source retry failed: {error}"),
        }),
    }
}

#[cfg(test)]
mod tests;
