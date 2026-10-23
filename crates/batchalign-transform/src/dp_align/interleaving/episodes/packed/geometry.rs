//! Conservative checkpoint storage and representation-specific work admission.

use super::*;

#[derive(Clone, Copy)]
pub(super) enum Representation {
    Full,
    Viable { span: usize },
}

pub(super) struct Geometry {
    pub(super) block: usize,
    pub(super) work: usize,
    pub(super) cells: usize,
}

pub(super) fn work_refusal(required_at_least: usize) -> LocalInterleavingBudget {
    LocalInterleavingBudget::WorkBudgetExceeded {
        phase: LocalInterleavingWorkPhase::PackedSearch,
        required_at_least,
    }
}

impl Geometry {
    #[cfg(test)]
    pub(super) fn observe<C: WorkControl + ?Sized>(
        graph: &Graph,
        n: usize,
        preparation: usize,
        control: &C,
    ) -> Result<Result<Self, LocalInterleavingRefusal>, C::Error> {
        Self::for_representation(graph, n, preparation, Representation::Full, control)
    }
    pub(super) fn for_representation<C: WorkControl + ?Sized>(
        graph: &Graph,
        n: usize,
        preparation: usize,
        representation: Representation,
        control: &C,
    ) -> Result<Result<Self, LocalInterleavingRefusal>, C::Error> {
        let count = graph.order.len();
        if count.saturating_add(1) > SCORE_CELLS {
            return Ok(Err(LocalInterleavingRefusal::Budget(
                LocalInterleavingBudget::MemoryBudgetExceeded,
            )));
        }
        let width = match representation {
            Representation::Full => n,
            Representation::Viable { span } => span,
        }
        .div_ceil(64);
        let mut stride = 1;
        while stride < count.div_ceil(stride) {
            stride += 1;
        }
        let strides: Vec<_> = [1usize, 2, 4, 8]
            .into_iter()
            .map(|factor| stride.saturating_mul(factor).min(count))
            .collect();
        let mut delta = vec![0isize; count + 1];
        let mut word_edges = 0usize;
        let mut epsilon_edges = 0usize;
        let mut merges = 0usize;
        for (slot, &node) in graph.order.iter().enumerate() {
            control.check()?;
            let last = graph.edges[node]
                .iter()
                .map(|edge| graph.rank[edge.to])
                .max()
                .unwrap_or(slot);
            if last > slot {
                delta[slot + 1] += 1;
                delta[last + 1] -= 1;
            }
            merges = merges.saturating_add(graph.incoming[node].len().saturating_sub(1));
            for edge in &graph.edges[node] {
                if edge.word.is_some() {
                    word_edges += 1;
                } else {
                    epsilon_edges += 1;
                }
            }
        }
        let mut live = 0isize;
        let mut peak_forward = 0usize;
        let mut retained = [0usize; 4];
        for (slot, &change) in delta[..count].iter().enumerate() {
            if slot % 1024 == 0 {
                control.check()?;
            }
            live += change;
            peak_forward = peak_forward.max(live as usize);
            for (index, &stride) in strides.iter().enumerate() {
                if slot % stride == 0 {
                    retained[index] = retained[index].saturating_add(live as usize);
                }
            }
        }
        delta.fill(0);
        for (slot, &node) in graph.order.iter().enumerate() {
            let first = graph.incoming[node]
                .iter()
                .map(|edge| graph.rank[edge.to])
                .min()
                .unwrap_or(slot);
            if first < slot {
                delta[first] += 1;
                delta[slot] -= 1;
            }
        }
        live = 0;
        let mut peak_backward = 0usize;
        for &change in &delta[..count] {
            live += change;
            peak_backward = peak_backward.max(live as usize);
        }
        let peak = peak_forward.max(peak_backward).saturating_add(1);
        let row = width.saturating_mul(BIT_WORDS);
        let mut best = Self {
            block: strides[0],
            work: usize::MAX,
            cells: usize::MAX,
        };
        for (index, &block) in strides.iter().enumerate() {
            // Retained checkpoints carry node indices and Vec headers.
            // Reserve simultaneous reconstruction roots/rows, reverse rows,
            // prefix summaries, scratch masks and conservative map metadata.
            let cells = retained[index]
                .saturating_mul(row.saturating_add(8))
                .saturating_add(count.div_ceil(block).saturating_mul(3))
                .saturating_add(
                    peak.saturating_mul(6)
                        .saturating_add(block.saturating_mul(2))
                        .saturating_add(8)
                        .saturating_mul(row.saturating_add(width).saturating_add(32)),
                )
                .saturating_add(n.saturating_mul(24));
            // Initial forward, reconstruction for proof and traceback, and
            // backward propagation. Sparse mask advancement is one block pass;
            // each predecessor clone and branch merge is charged as well.
            let passes = word_edges
                .saturating_mul(8)
                .saturating_add(epsilon_edges.saturating_mul(4))
                .saturating_add(merges.saturating_mul(4))
                .saturating_add(word_edges.saturating_mul(3))
                .saturating_add(count.saturating_mul(3))
                .saturating_add(retained[index].saturating_mul(5));
            let work = preparation
                .saturating_add(match representation {
                    Representation::Full => passes.saturating_mul(width),
                    Representation::Viable { .. } => 0,
                })
                // Reserve both geometry surveys if a full-profile attempt
                // precedes viability admission, plus all four graph traversals,
                // expiry scans, edge observation and selected-path probes.
                .saturating_add(count.saturating_mul(24))
                .saturating_add(word_edges.saturating_add(epsilon_edges).saturating_mul(16))
                .saturating_add(n.saturating_mul(6))
                .saturating_add(retained[index].saturating_mul(peak.ilog2() as usize + 2));
            let candidate = Self { block, work, cells };
            let fits = candidate.work <= WORK_CELLS && candidate.cells <= SCORE_CELLS;
            let best_fits = best.work <= WORK_CELLS && best.cells <= SCORE_CELLS;
            if (fits && !best_fits)
                || (fits && best_fits && candidate.cells < best.cells)
                || (!fits && !best_fits && candidate.work < best.work)
            {
                best = candidate;
            }
        }
        // There is always at least one stride for the nonempty graph.
        let geometry = best;
        if geometry.work > WORK_CELLS {
            return Ok(Err(LocalInterleavingRefusal::Budget(work_refusal(
                geometry.work,
            ))));
        }
        if geometry.cells > SCORE_CELLS {
            return Ok(Err(LocalInterleavingRefusal::Budget(
                LocalInterleavingBudget::MemoryBudgetExceeded,
            )));
        }
        Ok(Ok(geometry))
    }
}
