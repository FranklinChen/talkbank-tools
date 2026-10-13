//! Source-bound, independently bounded inspection of one selected optimum.
use super::*;
use std::cell::Cell;

const MAX_WORK: usize = 64_000_000;
const MAX_SCORE_CELLS: usize = 1_048_576;

/// A diagnostic path is complete, or unavailable; partial paths never escape.
pub enum InspectionAdmission {
    /// The producer completed its original tie-broken path.
    Complete(CompletedInspection),
    /// No claim about the matching population, proof or acoustic evidence.
    BudgetExhausted(InspectionBudget),
}

/// A complete path, constructible only by the inspection producer.
pub struct CompletedInspection {
    path: Vec<AlignResult>,
}
impl CompletedInspection {
    /// Read-only diagnostic path, never common-correspondence authority.
    pub fn path(&self) -> &[AlignResult] {
        &self.path
    }
}

/// The independent inspection limit, in its named unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InspectionLimit {
    /// Prepared words, comparison/score cells, split scans and output records.
    Work,
    /// Simultaneously live scalar score cells, not total process memory.
    ScoreStorage,
}

/// An immutable refusal made only by the inspection meter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct InspectionBudget {
    kind: InspectionLimit,
    limit: usize,
    used: usize,
    required_at_least: usize,
}
impl InspectionBudget {
    /// Which operation was denied.
    pub fn kind(self) -> InspectionLimit {
        self.kind
    }
    /// Fixed production limit in that operation's unit.
    pub fn limit(self) -> usize {
        self.limit
    }
    /// Completed/reserved work, or peak score cells admitted so far.
    pub fn used(self) -> usize {
        self.used
    }
    /// Known requirement of the denied operation, saturated on overflow.
    pub fn required_at_least(self) -> usize {
        self.required_at_least
    }
}

enum Stopped<E> {
    Interrupted(E),
    Budget(InspectionBudget),
}

struct Meter<'a, C: WorkControl + ?Sized> {
    control: &'a C,
    work_limit: usize,
    storage_limit: usize,
    work: Cell<usize>,
    peak: Cell<usize>,
}
impl<C: WorkControl + ?Sized> WorkControl for Meter<'_, C> {
    type Error = Stopped<C::Error>;
    fn check(&self) -> Result<(), Self::Error> {
        self.control.check().map_err(Stopped::Interrupted)
    }
    fn alignment_work(&self, units: usize) -> Result<(), Self::Error> {
        self.check()?;
        let required = self.work.get().checked_add(units);
        if required.is_none_or(|required| required > self.work_limit) {
            return Err(Stopped::Budget(InspectionBudget {
                kind: InspectionLimit::Work,
                limit: self.work_limit,
                used: self.work.get(),
                required_at_least: required.unwrap_or(usize::MAX),
            }));
        }
        if let Some(required) = required {
            self.work.set(required);
        }
        Ok(())
    }
    fn alignment_score_storage(&self, cells: usize) -> Result<(), Self::Error> {
        self.check()?;
        if cells > self.storage_limit {
            return Err(Stopped::Budget(InspectionBudget {
                kind: InspectionLimit::ScoreStorage,
                limit: self.storage_limit,
                used: self.peak.get(),
                required_at_least: cells,
            }));
        }
        self.peak.set(self.peak.get().max(cells));
        Ok(())
    }
}

/// Pending state owns the exact input relation and its meter. Finishing takes
/// no replacement source, policy or ledger, and consumes this state.
pub(super) struct PendingInspection<'a, C: WorkControl + ?Sized> {
    payload: &'a [String],
    reference: &'a [String],
    mode: MatchMode,
    meter: Meter<'a, C>,
}
impl<'a, C: WorkControl + ?Sized> PendingInspection<'a, C> {
    pub(super) fn bind(
        payload: &'a [String],
        reference: &'a [String],
        mode: MatchMode,
        control: &'a C,
    ) -> Self {
        Self {
            payload,
            reference,
            mode,
            meter: Meter {
                control,
                work_limit: MAX_WORK,
                storage_limit: MAX_SCORE_CELLS,
                work: Cell::new(0),
                peak: Cell::new(0),
            },
        }
    }
    pub(super) fn finish(self) -> Result<InspectionAdmission, C::Error> {
        match super::super::align_controlled(self.payload, self.reference, self.mode, &self.meter) {
            Ok(path) => Ok(InspectionAdmission::Complete(CompletedInspection { path })),
            Err(Stopped::Budget(refusal)) => Ok(InspectionAdmission::BudgetExhausted(refusal)),
            Err(Stopped::Interrupted(error)) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests;
