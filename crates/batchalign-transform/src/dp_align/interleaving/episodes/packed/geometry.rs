//! Conservative checkpoint storage and representation-specific work admission.

use super::*;
#[cfg(test)]
use std::num::NonZeroUsize;

/// Geometry preparation consumes the literal input exactly once. Row
/// alternatives retain this owner rather than rebuilding the graph survey.
#[derive(Debug)]
pub(super) struct SurveyedLiteralSearch<'g, 's> {
    prepared: PreparedLiteralSearch<'g, 's>,
    survey: GraphSurvey,
}

impl<'g, 's> SurveyedLiteralSearch<'g, 's> {
    pub(super) fn observe<C: WorkControl + ?Sized>(
        prepared: PreparedLiteralSearch<'g, 's>,
        control: &C,
    ) -> Result<Result<Self, LocalInterleavingRefusal>, C::Error> {
        Ok(
            GraphSurvey::observe(prepared.graph(), control)?
                .map(|survey| Self { prepared, survey }),
        )
    }

    pub(super) fn prepared(&self) -> &PreparedLiteralSearch<'g, 's> {
        &self.prepared
    }

    /// A declared test stride enters before the survey and admission, never
    /// by mutating a plan whose work/storage were computed for another stride.
    #[cfg(test)]
    pub(super) fn testing<C: WorkControl + ?Sized>(
        prepared: PreparedLiteralSearch<'g, 's>,
        block: NonZeroUsize,
        control: &C,
    ) -> Result<Result<Self, LocalInterleavingRefusal>, C::Error> {
        Ok(GraphSurvey::observe_schedule(
            CheckpointSchedule {
                graph: prepared.graph(),
                strides: [block.get(); 4],
            },
            control,
        )?
        .map(|survey| Self { prepared, survey }))
    }

    pub(super) fn project(
        &self,
        representation: Representation,
    ) -> Result<Geometry, LocalInterleavingRefusal> {
        self.survey.project(
            self.prepared.reference().len(),
            self.prepared.preparation(),
            representation,
        )
    }
}

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

/// Representation-independent graph preparation, observed once and retained
/// with its input owner while full and viable rows are considered.
#[derive(Debug)]
struct GraphSurvey {
    count: usize,
    strides: [usize; 4],
    retained: [usize; 4],
    peak: usize,
    word_edges: usize,
    epsilon_edges: usize,
    merges: usize,
}

/// The survey's schedule and graph are one input. Neither projection nor an
/// admitted execution plan can replace this already-observed population.
struct CheckpointSchedule<'graph> {
    graph: &'graph Graph,
    strides: [usize; 4],
}

impl<'graph> CheckpointSchedule<'graph> {
    fn automatic(graph: &'graph Graph) -> Self {
        let count = graph.order.len();
        let mut stride = 1;
        while stride < count.div_ceil(stride) {
            stride += 1;
        }
        Self {
            graph,
            strides: [1usize, 2, 4, 8].map(|factor| stride.saturating_mul(factor).min(count)),
        }
    }
}

impl GraphSurvey {
    fn observe<C: WorkControl + ?Sized>(
        graph: &Graph,
        control: &C,
    ) -> Result<Result<Self, LocalInterleavingRefusal>, C::Error> {
        Self::observe_schedule(CheckpointSchedule::automatic(graph), control)
    }

    fn observe_schedule<C: WorkControl + ?Sized>(
        schedule: CheckpointSchedule<'_>,
        control: &C,
    ) -> Result<Result<Self, LocalInterleavingRefusal>, C::Error> {
        let CheckpointSchedule { graph, strides } = schedule;
        let count = graph.order.len();
        if count.saturating_add(1) > SCORE_CELLS {
            return Ok(Err(LocalInterleavingRefusal::Budget(
                LocalInterleavingBudget::MemoryBudgetExceeded,
            )));
        }
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
        Ok(Ok(Self {
            count,
            strides,
            retained,
            peak: peak_forward.max(peak_backward).saturating_add(1),
            word_edges,
            epsilon_edges,
            merges,
        }))
    }

    /// Pure projection over already-observed graph facts. It cannot revisit
    /// adjacency or allocate another liveness survey.
    fn project(
        &self,
        n: usize,
        preparation: usize,
        representation: Representation,
    ) -> Result<Geometry, LocalInterleavingRefusal> {
        let Self {
            count,
            strides,
            retained,
            peak,
            word_edges,
            epsilon_edges,
            merges,
        } = *self;
        let width = match representation {
            Representation::Full => n,
            Representation::Viable { span } => span,
        }
        .div_ceil(64);
        let row = width.saturating_mul(BIT_WORDS);
        let candidates = std::array::from_fn::<_, 4, _>(|index| {
            let block = strides[index];
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
                // Keep the conservative metadata allowance while admission is
                // migrated. Only one geometry survey is performed; this also
                // covers traversal, expiry, edge and selected-path probes.
                .saturating_add(count.saturating_mul(24))
                .saturating_add(word_edges.saturating_add(epsilon_edges).saturating_mul(16))
                .saturating_add(n.saturating_mul(6))
                .saturating_add(retained[index].saturating_mul(peak.ilog2() as usize + 2));
            Geometry { block, work, cells }
        });
        // Four real candidates make the population nonempty by construction;
        // the initial best is a candidate, never an all-MAX sentinel.
        let [first, rest @ ..] = candidates;
        let geometry = rest.into_iter().fold(first, |best, candidate| {
            let fits = candidate.work <= WORK_CELLS && candidate.cells <= SCORE_CELLS;
            let best_fits = best.work <= WORK_CELLS && best.cells <= SCORE_CELLS;
            if (fits && !best_fits)
                || (fits && best_fits && candidate.cells < best.cells)
                || (!fits && !best_fits && candidate.work < best.work)
            {
                candidate
            } else {
                best
            }
        });
        if geometry.work > WORK_CELLS {
            return Err(LocalInterleavingRefusal::Budget(work_refusal(
                geometry.work,
            )));
        }
        if geometry.cells > SCORE_CELLS {
            return Err(LocalInterleavingRefusal::Budget(
                LocalInterleavingBudget::MemoryBudgetExceeded,
            ));
        }
        Ok(geometry)
    }
}
