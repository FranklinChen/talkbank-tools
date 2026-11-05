//! Certify correspondence independently of an optimal path's tie breaking.
//!
//! Substitution costs two gaps, so optimal matching populations are longest
//! chains of matching (payload, reference) edges. An optimal chain contains one
//! edge at every rank. An edge common to every optimal chain is exactly the sole
//! optimal edge at its rank. Forward/backward Fenwick passes establish ranks
//! without a quadratic-memory DP matrix. Prefix/suffix stripping is deliberately
//! not used here: even a selected identical prefix may be ambiguous.
//! A bounded packed-frontier route can instead prune nonoptimal literal edges
//! before admission; it applies the same sole-optimal-edge-at-rank theorem.

use std::collections::BTreeMap;

use super::comparison::{Alignable, FuzzyComparison, PreparedFuzzyWord};
use super::control::{Uninterrupted, WorkControl, uninterrupted};
use super::packed_literal::key;
use super::{AlignResult, MatchMode};
mod fuzzy;
mod inspection;
mod literal;
mod sparse;
pub use inspection::{CompletedInspection, InspectionAdmission, InspectionBudget, InspectionLimit};

const _: () = assert!(super::COST_SUB == 2 * super::COST_GAP);
const MAX_EDGES: usize = 1_048_576;

/// Independently completed or refused inspection and common correspondence proof.
/// Only the alignment producer can construct this paired observation.
pub struct CorrespondenceAnalysis {
    inspection: InspectionAdmission,
    admission: CorrespondenceAdmission,
}

/// Whether all candidate edges could be examined within the bounded budget.
pub enum CorrespondenceAdmission {
    /// Exact intersection of all optimal lexical matching populations.
    Complete(CommonCorrespondences),
    /// No assertion of ambiguity or uniqueness: proof work was bounded.
    /// Names the bound that was reached, so a refusal can say which one.
    BudgetExhausted(CorrespondenceBudget),
}

impl CorrespondenceAdmission {
    /// Observe only common correspondences under the production proof caps.
    /// This producer never computes an arbitrarily selected inspection path.
    pub fn observe(payload: &[String], reference: &[String], mode: MatchMode) -> Self {
        uninterrupted(Self::observe_controlled(
            payload,
            reference,
            mode,
            &Uninterrupted,
        ))
    }

    /// Interruption cannot produce completed proof or a budget-refusal result.
    pub fn observe_controlled<C: WorkControl + ?Sized>(
        payload: &[String],
        reference: &[String],
        mode: MatchMode,
        control: &C,
    ) -> Result<Self, C::Error> {
        admit_controlled(payload, reference, mode, MAX_EDGES, control)
    }
}

/// The candidate allowance actually denied by this proof's caller policy.
/// Work estimates cannot construct a resource refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorrespondenceBudget {
    edge_limit: usize,
}

impl CorrespondenceBudget {
    fn candidate_edges(edge_limit: usize) -> Self {
        Self { edge_limit }
    }

    /// The original caller's denied allowance, in candidate edges.
    pub fn limit(self) -> usize {
        self.edge_limit
    }
}

/// Matching addresses proven common to every optimal lexical alignment.
/// A selected path or externally assembled address list cannot construct this.
pub struct CommonCorrespondences {
    pairs: Vec<(usize, usize)>,
}

impl CommonCorrespondences {
    /// Addresses in payload order, with strictly increasing reference indices.
    pub fn pairs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.pairs.iter().copied()
    }
}

impl CorrespondenceAnalysis {
    /// Prove common matches and attempt the original diagnostic path separately.
    pub fn observe(payload: &[String], reference: &[String], mode: MatchMode) -> Self {
        uninterrupted(Self::observe_controlled(
            payload,
            reference,
            mode,
            &Uninterrupted,
        ))
    }

    /// Interruption never constructs selected or admitted correspondence evidence.
    pub fn observe_controlled<C: WorkControl + ?Sized>(
        payload: &[String],
        reference: &[String],
        mode: MatchMode,
        control: &C,
    ) -> Result<Self, C::Error> {
        let admission =
            CorrespondenceAdmission::observe_controlled(payload, reference, mode, control)?;
        let inspection =
            inspection::PendingInspection::bind(payload, reference, mode, control).finish()?;
        Ok(Self {
            inspection,
            admission,
        })
    }

    /// Completed inspection, if available; refusal is not an empty path.
    /// Inspection is never authority to use selected pairs as timing evidence.
    pub fn selected(&self) -> Option<&[AlignResult]> {
        match &self.inspection {
            InspectionAdmission::Complete(complete) => Some(complete.path()),
            InspectionAdmission::BudgetExhausted(_) => None,
        }
    }

    /// Inspection refusal is independent of common proof admission.
    pub fn inspection(&self) -> &InspectionAdmission {
        &self.inspection
    }

    /// Read the producer's bounded proof, not a consumer-selected certainty flag.
    pub fn admission(&self) -> &CorrespondenceAdmission {
        &self.admission
    }
}

struct Edge {
    payload: usize,
    reference: usize,
}

enum PreparedCorrespondences<'input> {
    Sparse(sparse::Candidates<'input>),
    Certified(CommonCorrespondences),
}

#[cfg(test)]
fn admit(
    payload: &[String],
    reference: &[String],
    mode: MatchMode,
    edge_limit: usize,
) -> CorrespondenceAdmission {
    uninterrupted(admit_controlled(
        payload,
        reference,
        mode,
        edge_limit,
        &Uninterrupted,
    ))
}

fn admit_controlled<C: WorkControl + ?Sized>(
    payload: &[String],
    reference: &[String],
    mode: MatchMode,
    edge_limit: usize,
    control: &C,
) -> Result<CorrespondenceAdmission, C::Error> {
    match candidate_edges(payload, reference, mode, edge_limit, control)? {
        Ok(PreparedCorrespondences::Sparse(candidates)) => Ok(CorrespondenceAdmission::Complete(
            candidates.certify(control)?,
        )),
        Ok(PreparedCorrespondences::Certified(common)) => {
            Ok(CorrespondenceAdmission::Complete(common))
        }
        Err(budget) => Ok(CorrespondenceAdmission::BudgetExhausted(budget)),
    }
}

fn candidate_edges<'input, C: WorkControl + ?Sized>(
    payload: &'input [String],
    reference: &'input [String],
    mode: MatchMode,
    edge_limit: usize,
    control: &C,
) -> Result<Result<PreparedCorrespondences<'input>, CorrespondenceBudget>, C::Error> {
    control.check()?;
    let mut edges = Vec::new();
    let mut push = |payload, reference| {
        if edges.len() == edge_limit {
            return Err(CorrespondenceBudget::candidate_edges(edge_limit));
        }
        edges.push(Edge { payload, reference });
        Ok(())
    };
    match mode {
        MatchMode::Exact | MatchMode::CaseInsensitive => {
            let literal_mode = match mode {
                MatchMode::Exact => super::LiteralMatchMode::Exact,
                _ => super::LiteralMatchMode::AsciiInsensitive,
            };
            let input =
                literal::LiteralCandidates::observe(payload, reference, literal_mode, control)?;
            if input.raw_edges() > edge_limit {
                match input.certify(edge_limit, control)? {
                    Some(CorrespondenceAdmission::Complete(common)) => {
                        return Ok(Ok(PreparedCorrespondences::Certified(common)));
                    }
                    Some(CorrespondenceAdmission::BudgetExhausted(budget)) => {
                        return Ok(Err(budget));
                    }
                    None => {} // The original bounded sparse producer remains.
                }
            }
            for index in 0..payload.len() {
                control.check()?;
                if let Some(references) = input.references(index) {
                    for reference in references {
                        control.check()?;
                        if let Err(budget) = push(index, *reference) {
                            return Ok(Err(budget));
                        }
                    }
                }
            }
        }
        MatchMode::Fuzzy { threshold } => {
            return fuzzy::FuzzyCandidates::prepare(
                payload,
                reference,
                FuzzyComparison::new(threshold),
                edge_limit,
                control,
            )?
            .finish(control)
            .map(|result| result.map(PreparedCorrespondences::Sparse));
        }
    }
    Ok(Ok(PreparedCorrespondences::Sparse(
        sparse::Candidates::bind(reference, edges),
    )))
}

/// Maximum chain length over reference positions strictly before a position.
struct PrefixMaximum(Vec<usize>);

impl PrefixMaximum {
    fn new(length: usize) -> Self {
        Self(vec![0; length + 1])
    }

    fn before(&self, mut position: usize) -> usize {
        let mut maximum = 0;
        while position > 0 {
            maximum = maximum.max(self.0[position]);
            position &= position - 1;
        }
        maximum
    }

    fn record(&mut self, position: usize, rank: usize) {
        let mut position = position + 1;
        while position < self.0.len() {
            self.0[position] = self.0[position].max(rank);
            position += position.isolate_lowest_one();
        }
    }
}

#[cfg(test)]
mod tests;
