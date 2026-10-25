//! UTR attempt accounting and the single-use permission for fallback recovery.
//!
//! A provider failure is recorded before FA may continue. Successful earlier
//! recovery and failed later attempts are independent evidence; neither erases
//! the other. Only cancellation escapes as a stop rather than a shortfall.

use crate::api::{OutputShortfallRecord, StampJoiner, StampSafeText, UtrPass};
use crate::chat_ops::fa::utr::{UtrCancelled, UtrResult};
use crate::error::ServerError;
use crate::options::UtrEngine;
use crate::scheduling::FailureCategory;

/// One file's report and its fallback permission have one owner. A pending
/// fallback borrows this report, so it cannot be recorded on another file's
/// report or race a second recovery transition.
pub(crate) struct UtrRecovery {
    contribution: UtrContribution,
    fallback: UtrFallback,
}

pub(crate) struct PendingUtrFallback<'a> {
    contribution: &'a mut UtrContribution,
    permit: UtrFallbackPermit,
}

impl UtrRecovery {
    pub(crate) fn not_run() -> Self {
        Self {
            contribution: UtrContribution::NotRun,
            fallback: UtrFallback::Unavailable,
        }
    }

    pub(crate) fn from_initial(
        engine: &UtrEngine,
        result: Result<UtrResult, ServerError>,
    ) -> Result<Self, UtrCancelled> {
        let mut contribution = UtrContribution::NotRun;
        let fallback = contribution.record_initial(engine, result)?;
        Ok(Self {
            contribution,
            fallback,
        })
    }

    pub(crate) fn contribution(&self) -> &UtrContribution {
        &self.contribution
    }

    pub(crate) fn fallback(&mut self) -> Option<PendingUtrFallback<'_>> {
        self.fallback.take().map(|permit| PendingUtrFallback {
            contribution: &mut self.contribution,
            permit,
        })
    }
}

impl PendingUtrFallback<'_> {
    pub(crate) fn engine(&self) -> &UtrEngine {
        self.permit.engine()
    }

    pub(crate) fn record(self, result: Result<UtrResult, ServerError>) -> Result<(), UtrCancelled> {
        self.contribution.record_fallback(self.permit, result)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum UtrContribution {
    NotRun,
    Ran(UtrEngine),
    Unavailable(UtrFailures),
}

/// Non-empty failure history; construction belongs to the observation boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UtrFailures {
    completed: Option<UtrEngine>,
    first: UtrFailure,
    later: Vec<UtrFailure>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UtrFailure {
    engine: UtrEngine,
    pass: UtrPass,
    category: FailureCategory,
    detail: String,
}

impl UtrFailure {
    fn stamp(&self) -> StampSafeText {
        StampSafeText::join(
            &StampSafeText::from_static(self.pass.name()),
            [
                &self.engine.stamp_name(),
                &StampSafeText::from_static(self.category.as_str()),
            ],
            StampJoiner::Colon,
        )
    }
}

/// A failed/incomplete initial pass permits at most one fallback pass.
#[derive(Debug)]
enum UtrFallback {
    Unavailable,
    Available(UtrFallbackPermit),
}

/// Not cloneable or publicly constructible. Recording consumes the permission.
#[derive(Debug)]
struct UtrFallbackPermit {
    engine: UtrEngine,
}

impl UtrFallback {
    fn take(&mut self) -> Option<UtrFallbackPermit> {
        match std::mem::replace(self, Self::Unavailable) {
            Self::Unavailable => None,
            Self::Available(permit) => Some(permit),
        }
    }
}

impl UtrFallbackPermit {
    fn engine(&self) -> &UtrEngine {
        &self.engine
    }
}

enum RecoveryNeed {
    Satisfied,
    Remaining,
    /// The failure is one no request to the same engine can get past, so
    /// this attempt grants no fallback permission.
    Withheld(FallbackWithheld),
}

/// Why a failed UTR pass withholds the same-engine fallback.
///
/// The fallback is the SAME engine over a different request (narrower
/// windows), so it is worth running exactly when the failure belonged to the
/// request. These are the failures that belong to the engine or the machine
/// instead, where retrying cannot succeed and would only spend a second
/// paid call or a second model load to report the same thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FallbackWithheld {
    /// The provider account's quota is exhausted until an account holder
    /// restores it.
    QuotaExhausted,
    /// The provider refuses the account itself (key, permission, credit);
    /// the fallback is sent from the same account.
    AccountRefused,
    /// This machine is refused the engine's pinned model (a gated repository,
    /// a missing token); the fallback asks for the same model.
    ModelAccessDenied,
    /// The engine's worker cannot start: a model that will not load, a
    /// catalog that will not download, a runtime or model identity that
    /// disagrees with the server's. Deterministic across retries, and the
    /// fallback is dispatched to that same worker key.
    WorkerBootstrap,
}

impl std::fmt::Display for FallbackWithheld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::QuotaExhausted => "the provider quota is exhausted",
            Self::AccountRefused => "the provider refuses the account",
            Self::ModelAccessDenied => "this machine is refused the engine's model",
            Self::WorkerBootstrap => "the engine's worker cannot start",
        })
    }
}

/// The fallback policy, stated once as one exhaustive match over the
/// failure's category, so a new category must decide here whether it
/// belongs to the request (fallback permitted) or to the engine (withheld).
fn fallback_after(category: FailureCategory) -> RecoveryNeed {
    match category {
        FailureCategory::ProviderQuotaExhausted => {
            RecoveryNeed::Withheld(FallbackWithheld::QuotaExhausted)
        }
        FailureCategory::ProviderAccountRefused => {
            RecoveryNeed::Withheld(FallbackWithheld::AccountRefused)
        }
        FailureCategory::ModelAccessDenied => {
            RecoveryNeed::Withheld(FallbackWithheld::ModelAccessDenied)
        }
        FailureCategory::WorkerBootstrap => {
            RecoveryNeed::Withheld(FallbackWithheld::WorkerBootstrap)
        }
        // A failure of THIS request says nothing about the fallback's
        // different one: a terminal provider failure (a 4xx, a failed job,
        // an undecodable response, invalid intervals), a refused or
        // malformed exchange, a crash, a timeout, memory pressure (narrower
        // windows need less), a transient fault.
        FailureCategory::ProviderTerminal
        | FailureCategory::ModelOutputInvalid
        | FailureCategory::Validation
        | FailureCategory::ParseError
        | FailureCategory::InputMissing
        | FailureCategory::EvidenceUnavailable
        | FailureCategory::AnalysisUnavailable
        | FailureCategory::WorkerCrash
        | FailureCategory::WorkerTimeout
        | FailureCategory::WorkerProtocol
        | FailureCategory::ProviderTransient
        | FailureCategory::LocalRuntime
        | FailureCategory::MemoryPressure
        | FailureCategory::Cancelled
        | FailureCategory::System => RecoveryNeed::Remaining,
    }
}

impl UtrContribution {
    pub(crate) fn completed_engine(&self) -> Option<&UtrEngine> {
        match self {
            Self::NotRun => None,
            Self::Ran(engine) => Some(engine),
            Self::Unavailable(failures) => failures.completed.as_ref(),
        }
    }

    /// Record the actual attempt before producing any fallback permission.
    fn record_initial(
        &mut self,
        engine: &UtrEngine,
        result: Result<UtrResult, ServerError>,
    ) -> Result<UtrFallback, UtrCancelled> {
        Ok(match self.record(engine, UtrPass::Initial, result)? {
            RecoveryNeed::Satisfied => UtrFallback::Unavailable,
            RecoveryNeed::Withheld(reason) => {
                use crate::options::EngineBackend;
                // The failed pass is already in the shortfalls; this says why
                // no second pass follows it.
                tracing::info!(
                    engine = engine.wire_name(),
                    %reason,
                    "same-engine UTR fallback withheld"
                );
                UtrFallback::Unavailable
            }
            RecoveryNeed::Remaining => UtrFallback::Available(UtrFallbackPermit {
                engine: engine.clone(),
            }),
        })
    }

    fn record_fallback(
        &mut self,
        permit: UtrFallbackPermit,
        result: Result<UtrResult, ServerError>,
    ) -> Result<(), UtrCancelled> {
        self.record(&permit.engine, UtrPass::Fallback, result)?;
        Ok(())
    }

    fn record(
        &mut self,
        engine: &UtrEngine,
        pass: UtrPass,
        result: Result<UtrResult, ServerError>,
    ) -> Result<RecoveryNeed, UtrCancelled> {
        match result {
            Ok(result) => {
                if result.ran() {
                    match self {
                        Self::Unavailable(failures) => failures.completed = Some(engine.clone()),
                        _ => *self = Self::Ran(engine.clone()),
                    }
                }
                Ok(if result.unmatched() > 0 {
                    RecoveryNeed::Remaining
                } else {
                    RecoveryNeed::Satisfied
                })
            }
            Err(ServerError::Cancelled) => Err(UtrCancelled),
            Err(error) => {
                let category = crate::runner::util::classify_server_error(&error);
                let need = fallback_after(category);
                let failure = UtrFailure {
                    engine: engine.clone(),
                    pass,
                    category,
                    detail: error.to_string().chars().take(500).collect(),
                };
                match self {
                    Self::Unavailable(failures) => failures.later.push(failure),
                    _ => {
                        *self = Self::Unavailable(UtrFailures {
                            completed: self.completed_engine().cloned(),
                            first: failure,
                            later: Vec::new(),
                        });
                    }
                }
                Ok(need)
            }
        }
    }

    /// No provider-controlled error text enters CHAT stamp syntax. Engine,
    /// pass and cause are all closed vocabularies, joined by the shared codec.
    pub(super) fn unavailable_stamp(&self) -> Option<StampSafeText> {
        let Self::Unavailable(failures) = self else {
            return None;
        };
        let later: Vec<_> = failures.later.iter().map(UtrFailure::stamp).collect();
        Some(StampSafeText::join(
            &failures.first.stamp(),
            &later,
            StampJoiner::Plus,
        ))
    }

    pub(crate) fn shortfalls(&self) -> Vec<OutputShortfallRecord> {
        use crate::options::EngineBackend;
        let Self::Unavailable(failures) = self else {
            return Vec::new();
        };
        std::iter::once(&failures.first)
            .chain(&failures.later)
            .map(|failure| OutputShortfallRecord::UtrUnavailable {
                engine: failure.engine.wire_name().to_owned(),
                pass: failure.pass,
                category: failure.category,
                detail: failure.detail.clone(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
