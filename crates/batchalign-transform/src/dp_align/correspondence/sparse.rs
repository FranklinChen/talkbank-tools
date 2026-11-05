//! Consuming score phases retain addresses without uncomputed rank sentinels.

use super::*;

/// A positive chain rank, produced only by an observed prefix maximum.
struct MatchRank(usize);

impl MatchRank {
    fn after(prefix: usize) -> Self {
        Self(prefix + 1)
    }
}

pub(super) struct Candidates<'input> {
    reference: &'input [String],
    edges: Vec<Edge>,
}

struct ForwardRanked<'input> {
    candidates: Candidates<'input>,
    forward: Vec<MatchRank>,
    optimum: usize,
}

struct CompleteRanks<'input> {
    forward: ForwardRanked<'input>,
    /// Backward observations are stored in reverse edge order as computed.
    backward: Vec<MatchRank>,
}

enum Multiplicity {
    Empty,
    Unique,
    Competing,
}

impl Multiplicity {
    fn observe(&mut self) {
        *self = match self {
            Self::Empty => Self::Unique,
            Self::Unique | Self::Competing => Self::Competing,
        };
    }
}

/// Payload allowance of the actual sparse address and two completed ranks.
pub(super) const EDGE_PAYLOAD_BYTES: usize =
    std::mem::size_of::<Edge>() + 2 * std::mem::size_of::<MatchRank>();

impl<'input> Candidates<'input> {
    pub(super) fn bind(reference: &'input [String], edges: Vec<Edge>) -> Self {
        Self { reference, edges }
    }

    pub(super) fn certify<C: WorkControl + ?Sized>(
        self,
        control: &C,
    ) -> Result<CommonCorrespondences, C::Error> {
        if self.edges.is_empty() {
            return Ok(CommonCorrespondences { pairs: Vec::new() });
        }
        self.rank_forward(control)?
            .rank_backward(control)?
            .certify(control)
    }

    fn rank_forward<C: WorkControl + ?Sized>(
        self,
        control: &C,
    ) -> Result<ForwardRanked<'input>, C::Error> {
        let mut ranks = PrefixMaximum::new(self.reference.len());
        let mut forward = Vec::with_capacity(self.edges.len());
        let mut row_start = 0;
        while row_start < self.edges.len() {
            control.check()?;
            let row_end = row_start
                + self.edges[row_start..]
                    .iter()
                    .take_while(|edge| edge.payload == self.edges[row_start].payload)
                    .count();
            // Delayed updates forbid using two edges from one payload word.
            for edge in &self.edges[row_start..row_end] {
                control.check()?;
                forward.push(MatchRank::after(ranks.before(edge.reference)));
            }
            for (edge, rank) in self.edges[row_start..row_end]
                .iter()
                .zip(&forward[row_start..row_end])
            {
                control.check()?;
                ranks.record(edge.reference, rank.0);
            }
            row_start = row_end;
        }
        let optimum = ranks.before(self.reference.len());
        // This phase's Fenwick allocation dies before backward ranking begins.
        Ok(ForwardRanked {
            candidates: self,
            forward,
            optimum,
        })
    }
}

impl<'input> ForwardRanked<'input> {
    fn rank_backward<C: WorkControl + ?Sized>(
        self,
        control: &C,
    ) -> Result<CompleteRanks<'input>, C::Error> {
        let edges = &self.candidates.edges;
        let length = self.candidates.reference.len();
        let mut ranks = PrefixMaximum::new(length);
        let mut backward = Vec::with_capacity(edges.len());
        let mut row_end = edges.len();
        while row_end > 0 {
            control.check()?;
            let row_start = row_end
                - edges[..row_end]
                    .iter()
                    .rev()
                    .take_while(|edge| edge.payload == edges[row_end - 1].payload)
                    .count();
            let rank_start = backward.len();
            for edge in edges[row_start..row_end].iter().rev() {
                control.check()?;
                backward.push(MatchRank::after(ranks.before(length - 1 - edge.reference)));
            }
            for (edge, rank) in edges[row_start..row_end]
                .iter()
                .rev()
                .zip(&backward[rank_start..])
            {
                control.check()?;
                ranks.record(length - 1 - edge.reference, rank.0);
            }
            row_end = row_start;
        }
        Ok(CompleteRanks {
            forward: self,
            backward,
        })
    }
}

impl CompleteRanks<'_> {
    fn certify<C: WorkControl + ?Sized>(
        self,
        control: &C,
    ) -> Result<CommonCorrespondences, C::Error> {
        let optimum = self.forward.optimum;
        let mut populations: Vec<_> = (0..optimum).map(|_| Multiplicity::Empty).collect();
        for (forward, backward) in self.forward.forward.iter().zip(self.backward.iter().rev()) {
            control.check()?;
            if forward.0 + backward.0 - 1 == optimum {
                populations[forward.0 - 1].observe();
            }
        }
        let mut pairs = Vec::new();
        for ((edge, forward), backward) in self
            .forward
            .candidates
            .edges
            .into_iter()
            .zip(self.forward.forward)
            .zip(self.backward.into_iter().rev())
        {
            control.check()?;
            if forward.0 + backward.0 - 1 == optimum
                && matches!(populations[forward.0 - 1], Multiplicity::Unique)
            {
                pairs.push((edge.payload, edge.reference));
            }
        }
        Ok(CommonCorrespondences { pairs })
    }
}
