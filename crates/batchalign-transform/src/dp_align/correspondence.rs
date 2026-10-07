//! Certify correspondence independently of an optimal path's tie breaking.
//!
//! Substitution costs two gaps, so optimal matching populations are longest
//! chains of matching (payload, reference) edges. An optimal chain contains one
//! edge at every rank. An edge common to every optimal chain is exactly the sole
//! optimal edge at its rank. Forward/backward Fenwick passes establish ranks
//! without a quadratic-memory DP matrix. Prefix/suffix stripping is deliberately
//! not used here: even a selected identical prefix may be ambiguous.

use std::collections::BTreeMap;

use super::comparison::{Alignable, FuzzyComparison, PreparedFuzzyWord};
use super::{AlignResult, MatchMode};

const _: () = assert!(super::COST_SUB == 2 * super::COST_GAP);
const MAX_EDGES: usize = 1_048_576;
const MAX_FUZZY_COMPARISONS: usize = 64_000_000;

/// Selected alignment and separately admitted common correspondences.
/// Only the alignment producer can construct this paired observation.
pub struct CorrespondenceAnalysis {
    selected: Vec<AlignResult>,
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

/// Which fixed proof budget a correspondence analysis reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrespondenceBudget {
    /// More candidate (payload, reference) match edges than the edge limit.
    CandidateEdges,
    /// A fuzzy relation would need more word comparisons than its limit.
    FuzzyComparisons,
}

impl CorrespondenceBudget {
    /// The production limit this budget enforces, in its own unit
    /// (edges, or word-pair comparisons).
    pub fn limit(self) -> usize {
        match self {
            Self::CandidateEdges => MAX_EDGES,
            Self::FuzzyComparisons => MAX_FUZZY_COMPARISONS,
        }
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
    /// Keep the existing path for inspection; prove common matches separately.
    pub fn observe(payload: &[String], reference: &[String], mode: MatchMode) -> Self {
        Self {
            selected: super::align(payload, reference, mode),
            admission: admit(payload, reference, mode, MAX_EDGES, MAX_FUZZY_COMPARISONS),
        }
    }

    /// Inspection is not authority to use the selected path as timing evidence.
    pub fn selected(&self) -> &[AlignResult] {
        &self.selected
    }

    /// Read the producer's bounded proof, not a consumer-selected certainty flag.
    pub fn admission(&self) -> &CorrespondenceAdmission {
        &self.admission
    }
}

struct Edge {
    payload: usize,
    reference: usize,
    forward: usize,
    backward: usize,
}

fn admit(
    payload: &[String],
    reference: &[String],
    mode: MatchMode,
    edge_limit: usize,
    fuzzy_limit: usize,
) -> CorrespondenceAdmission {
    let mut edges = match candidate_edges(payload, reference, mode, edge_limit, fuzzy_limit) {
        Ok(edges) => edges,
        Err(budget) => return CorrespondenceAdmission::BudgetExhausted(budget),
    };
    let mut ranks = PrefixMaximum::new(reference.len());
    let mut row_start = 0;
    while row_start < edges.len() {
        let row_end = row_start
            + edges[row_start..]
                .iter()
                .take_while(|edge| edge.payload == edges[row_start].payload)
                .count();
        // Delayed updates forbid using two edges from the same payload word.
        for edge in &mut edges[row_start..row_end] {
            edge.forward = ranks.before(edge.reference) + 1;
        }
        for edge in &edges[row_start..row_end] {
            ranks.record(edge.reference, edge.forward);
        }
        row_start = row_end;
    }
    let optimum = ranks.before(reference.len());
    let mut ranks = PrefixMaximum::new(reference.len());
    let mut row_end = edges.len();
    while row_end > 0 {
        let row_start = row_end
            - edges[..row_end]
                .iter()
                .rev()
                .take_while(|edge| edge.payload == edges[row_end - 1].payload)
                .count();
        for edge in &mut edges[row_start..row_end] {
            edge.backward = ranks.before(reference.len() - 1 - edge.reference) + 1;
        }
        for edge in &edges[row_start..row_end] {
            ranks.record(reference.len() - 1 - edge.reference, edge.backward);
        }
        row_end = row_start;
    }
    let mut populations = vec![0usize; optimum + 1];
    for edge in &edges {
        if edge.forward + edge.backward - 1 == optimum {
            populations[edge.forward] += 1;
        }
    }
    let pairs = edges
        .into_iter()
        .filter(|edge| {
            edge.forward + edge.backward - 1 == optimum && populations[edge.forward] == 1
        })
        .map(|edge| (edge.payload, edge.reference))
        .collect();
    CorrespondenceAdmission::Complete(CommonCorrespondences { pairs })
}

fn candidate_edges(
    payload: &[String],
    reference: &[String],
    mode: MatchMode,
    edge_limit: usize,
    fuzzy_limit: usize,
) -> Result<Vec<Edge>, CorrespondenceBudget> {
    let mut edges = Vec::new();
    let mut push = |payload, reference| {
        if edges.len() == edge_limit {
            return Err(CorrespondenceBudget::CandidateEdges);
        }
        edges.push(Edge {
            payload,
            reference,
            forward: 0,
            backward: 0,
        });
        Ok(())
    };
    match mode {
        MatchMode::Exact | MatchMode::CaseInsensitive => {
            let key = |word: &str| match mode {
                MatchMode::Exact => word.to_owned(),
                _ => word.to_ascii_lowercase(),
            };
            let mut postings: BTreeMap<String, Vec<usize>> = BTreeMap::new();
            for (index, word) in reference.iter().enumerate() {
                postings.entry(key(word)).or_default().push(index);
            }
            for (index, word) in payload.iter().enumerate() {
                if let Some(references) = postings.get(&key(word)) {
                    for reference in references {
                        push(index, *reference)?;
                    }
                }
            }
        }
        MatchMode::Fuzzy { threshold } => {
            if payload
                .len()
                .checked_mul(reference.len())
                .is_none_or(|comparisons| comparisons > fuzzy_limit)
            {
                return Err(CorrespondenceBudget::FuzzyComparisons);
            }
            let payload: Vec<_> = payload
                .iter()
                .map(|word| PreparedFuzzyWord::new(word))
                .collect();
            let reference: Vec<_> = reference
                .iter()
                .map(|word| PreparedFuzzyWord::new(word))
                .collect();
            let policy = FuzzyComparison::new(threshold);
            for (p, word) in payload.iter().enumerate() {
                for (r, other) in reference.iter().enumerate() {
                    if word.matches(other, policy) {
                        push(p, r)?;
                    }
                }
            }
        }
    }
    Ok(edges)
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
