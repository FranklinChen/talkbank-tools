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
                let failure = UtrFailure {
                    engine: engine.clone(),
                    pass,
                    category: crate::runner::util::classify_server_error(&error),
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
                Ok(RecoveryNeed::Remaining)
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
