//! Exact monotonic proof with bounded packed frontiers, not a selected path.

use super::*;
use crate::dp_align::LiteralMatchMode;
use crate::dp_align::packed_literal::{BIT_WORDS, LiteralReference, PrefixScores, ReferenceOrder};

const MAX_PACKED_WORK: usize = 64_000_000;

pub(super) struct LiteralCandidates<'input> {
    payload: &'input [String],
    reference: &'input [String],
    mode: LiteralMatchMode,
    postings: BTreeMap<String, Vec<usize>>,
    raw_edges: usize,
}

/// Constructed only from this observed input's sizes and raw posting census.
struct PackedGeometry {
    stride: usize,
}

impl PackedGeometry {
    fn admit(input: &LiteralCandidates<'_>, edge_limit: usize) -> Option<Self> {
        let m = input.payload.len();
        let n = input.reference.len();
        if m.saturating_add(n).saturating_mul(6) > MAX_PACKED_WORK {
            return None;
        }
        let blocks = n.div_ceil(64);
        let mut stride = 1usize;
        while stride < m.div_ceil(stride) {
            stride += 1;
        }
        // Three sparse row streams, prefix-index scans and checkpoint copies;
        // two prefix-score queries and one optimal-edge probe per posting.
        let work = m
            .saturating_mul(blocks)
            .saturating_mul(6)
            .saturating_add(input.raw_edges.saturating_mul(3))
            .saturating_add(m.saturating_add(n).saturating_mul(6))
            .saturating_add(2);
        // Retained checkpoint bits, bounded prefix blocks, both row engines,
        // one reverse prefix view and rank/common payload. Include a generous
        // reference-map/header allowance; this is not an RSS/allocator claim.
        let rows = (m / stride + 1)
            .saturating_add(4)
            .saturating_mul(BIT_WORDS)
            .saturating_add((stride + 1).saturating_mul(PrefixScores::BLOCK_WORDS));
        let rank_words =
            std::mem::size_of::<RankPopulation>().div_ceil(std::mem::size_of::<usize>()) + 2;
        let storage = blocks
            .saturating_mul(rows)
            .saturating_add(n.saturating_mul(32))
            .saturating_add(m.saturating_add(1).saturating_mul(rank_words));
        // Never exceed the old four-machine-word Edge payload allowance.
        let edge_words = std::mem::size_of::<Edge>().div_ceil(std::mem::size_of::<usize>());
        (work <= MAX_PACKED_WORK && storage <= edge_limit.saturating_mul(edge_words))
            .then_some(Self { stride })
    }
}

enum RankPopulation {
    Empty,
    Unique { payload: usize, reference: usize },
    Competing,
}

impl RankPopulation {
    fn observe(&mut self, payload: usize, reference: usize) {
        *self = match self {
            Self::Empty => Self::Unique { payload, reference },
            Self::Unique { .. } | Self::Competing => Self::Competing,
        };
    }
}

impl<'input> LiteralCandidates<'input> {
    pub(super) fn observe<C: WorkControl + ?Sized>(
        payload: &'input [String],
        reference: &'input [String],
        mode: LiteralMatchMode,
        control: &C,
    ) -> Result<Self, C::Error> {
        control.check()?;
        let mut postings: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (index, word) in reference.iter().enumerate() {
            if index % 1024 == 0 {
                control.check()?;
            }
            postings.entry(key(word, mode)).or_default().push(index);
        }
        let mut raw_edges = 0usize;
        for (index, word) in payload.iter().enumerate() {
            if index % 1024 == 0 {
                control.check()?;
            }
            if let Some(matches) = postings.get(&key(word, mode)) {
                raw_edges = raw_edges.saturating_add(matches.len());
            }
        }
        Ok(Self {
            payload,
            reference,
            mode,
            postings,
            raw_edges,
        })
    }

    pub(super) fn raw_edges(&self) -> usize {
        self.raw_edges
    }

    pub(super) fn references(&self, payload: usize) -> Option<&[usize]> {
        self.postings
            .get(&key(&self.payload[payload], self.mode))
            .map(Vec::as_slice)
    }

    /// None means this optional workspace cannot be admitted; it grants no
    /// authority and the caller retains the original bounded sparse route.
    pub(super) fn certify<C: WorkControl + ?Sized>(
        &self,
        edge_limit: usize,
        control: &C,
    ) -> Result<Option<CorrespondenceAdmission>, C::Error> {
        let Some(geometry) = PackedGeometry::admit(self, edge_limit) else {
            return Ok(None);
        };
        control.check()?;
        let forward_reference =
            LiteralReference::observe(self.reference, self.mode, ReferenceOrder::Forward, control)?;
        let reverse_reference =
            LiteralReference::observe(self.reference, self.mode, ReferenceOrder::Reverse, control)?;
        let mut forward = forward_reference.start();
        let mut checkpoints = vec![forward.checkpoint()];
        for (p, word) in self.payload.iter().enumerate() {
            forward = forward.extend(word, control)?;
            if (p + 1) % geometry.stride == 0 {
                checkpoints.push(forward.checkpoint());
            }
        }
        let optimum = forward.score();
        drop(forward);
        let mut populations: Vec<_> = (0..=optimum).map(|_| RankPopulation::Empty).collect();
        let mut optimal_edges = 0usize;
        let mut backward = reverse_reference.start();
        let mut end = self.payload.len();
        while end > 0 {
            let start = (end - 1) / geometry.stride * geometry.stride;
            let mut reconstructed = checkpoints[start / geometry.stride].resume();
            let mut prefixes = Vec::with_capacity(end - start);
            for p in start..end {
                prefixes.push(reconstructed.prefixes());
                reconstructed = reconstructed.extend(&self.payload[p], control)?;
            }
            for p in (start..end).rev() {
                control.check()?;
                // This row observes source[p+1..] and the reversed reference.
                let suffix = backward.prefixes();
                if let Some(references) = self.references(p) {
                    for (ordinal, &r) in references.iter().enumerate() {
                        if ordinal % 1024 == 0 {
                            control.check()?;
                        }
                        let rank = prefixes[p - start].before(r) + 1;
                        if rank + suffix.before(self.reference.len() - r - 1) == optimum {
                            if optimal_edges == edge_limit {
                                return Ok(Some(CorrespondenceAdmission::BudgetExhausted(
                                    CorrespondenceBudget::CandidateEdges,
                                )));
                            }
                            optimal_edges += 1;
                            populations[rank].observe(p, r);
                        }
                    }
                }
                backward = backward.extend(&self.payload[p], control)?;
            }
            end = start;
        }
        let pairs = populations
            .into_iter()
            .filter_map(|population| match population {
                RankPopulation::Unique { payload, reference } => Some((payload, reference)),
                RankPopulation::Empty | RankPopulation::Competing => None,
            })
            .collect();
        Ok(Some(CorrespondenceAdmission::Complete(
            CommonCorrespondences { pairs },
        )))
    }
}
