//! Exact score-bounded storage for the source DAG. A canonical monotonic path
//! supplies a feasible score L. At reference prefix k and source depth d, any
//! complete path scores at most min(d,k) + min(m-d,n-k). Only states whose
//! upper bound is below L are removed; no optimal correspondence is removed.

use super::*;
use std::num::NonZeroUsize;
use std::ops::Range;

mod witness;
use witness::CanonicalWitness;
mod checkpoints;
pub(super) use checkpoints::ForwardScores;
mod capacity;
use capacity::ReferenceCapacity;

type Score = Option<NonZeroUsize>;

/// None is unreachable, not a zero score. Reachable scores store score + 1.
#[derive(Clone)]
pub(super) struct Row {
    first: usize,
    scores: Vec<Score>,
}

impl Row {
    pub(super) fn get(&self, slot: usize) -> Score {
        slot.checked_sub(self.first)
            .and_then(|index| self.scores.get(index).copied())
            .flatten()
    }

    fn slots(&self) -> Range<usize> {
        self.first..self.first + self.scores.len()
    }

    fn improve(&mut self, slot: usize, score: Score) {
        let cell = &mut self.scores[slot - self.first];
        *cell = (*cell).max(score);
    }
}

fn increment(score: Score) -> Score {
    score.and_then(|score| NonZeroUsize::new(score.get() + 1))
}

fn combined(a: Score, b: Score, matched: usize) -> Score {
    a.zip(b)
        .and_then(|(a, b)| NonZeroUsize::new(a.get() + b.get() - 1 + matched))
}

/// Owns admission and the graph binding; consumers cannot pair rows with a
/// different depth layout or choose an unproven pruning threshold.
pub(super) struct Band<'graph, 'source> {
    graph: &'graph Graph,
    relation: &'graph Relation<'source>,
    source_len: usize,
    reference_len: usize,
    feasible: usize,
    pub(super) block: usize,
    scoring_cells: usize,
    capacity: Option<ReferenceCapacity<'graph, 'source>>,
}

/// One owned literal preparation. Its matching policy cannot describe fuzzy
/// semantics, and ownership of the already-paid witness cannot be duplicated.
pub(super) struct PreparedLiteralSearch<'graph, 'source> {
    graph: &'graph Graph,
    source: &'graph [&'source str],
    reference: &'source [String],
    mode: crate::dp_align::LiteralMatchMode,
    preparation: usize,
    feasible: usize,
}

/// A scalar representation either offers a prepared literal alternative or
/// an actual refusal. Choosing packed execution no longer requires a forged
/// refusal whose policy and phase every consumer has to check again.
pub(super) enum BandAlternative<'graph, 'source> {
    Packed(PreparedLiteralSearch<'graph, 'source>),
    Refused(LocalInterleavingRefusal),
}

impl std::fmt::Debug for BandAlternative<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Packed(_) => f.write_str("prepared literal alternative"),
            Self::Refused(cause) => cause.fmt(f),
        }
    }
}

impl std::fmt::Debug for PreparedLiteralSearch<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedLiteralSearch")
            .field("source_words", &self.source.len())
            .field("reference_words", &self.reference.len())
            .field("feasible", &self.feasible)
            .finish()
    }
}

impl<'graph, 'source> PreparedLiteralSearch<'graph, 'source> {
    pub(super) fn graph(&self) -> &'graph Graph {
        self.graph
    }
    pub(super) fn source(&self) -> &'graph [&'source str] {
        self.source
    }
    pub(super) fn reference(&self) -> &'source [String] {
        self.reference
    }
    pub(super) fn mode(&self) -> crate::dp_align::LiteralMatchMode {
        self.mode
    }
    pub(super) fn matches(&self, source: usize, reference: usize) -> bool {
        self.mode
            .matches_words(self.source[source], &self.reference[reference])
    }
    pub(super) fn preparation(&self) -> usize {
        self.preparation
    }
    pub(super) fn feasible(&self) -> usize {
        self.feasible
    }

    #[cfg(test)]
    pub(super) fn testing_bounds(self) -> impl Iterator<Item = Self> {
        let Self {
            graph,
            source,
            reference,
            mode,
            preparation,
            feasible,
        } = self;
        (0..=feasible).map(move |feasible| Self {
            graph,
            source,
            reference,
            mode,
            preparation,
            feasible,
        })
    }

    #[cfg(test)]
    pub(super) fn testing(graph: &'graph Graph, relation: &'graph Relation<'source>) -> Self {
        let Relation::Literal {
            source,
            reference,
            mode,
        } = relation
        else {
            panic!("a packed-search fixture declares literal matching");
        };
        let witness = uninterrupted(CanonicalWitness::observe(relation, &Uninterrupted))
            .expect("fixture witness must complete; never invent a zero bound on failure");
        Self {
            graph,
            source,
            reference,
            mode: *mode,
            preparation: witness.work(),
            feasible: witness.score(),
        }
    }
}

impl<'graph, 'source> Band<'graph, 'source> {
    pub(super) fn admit<C: WorkControl + ?Sized>(
        graph: &'graph Graph,
        relation: &'graph Relation<'source>,
        source_len: usize,
        reference_len: usize,
        control: &C,
    ) -> Result<Result<Self, BandAlternative<'graph, 'source>>, C::Error> {
        let refused = |cause, preparation, feasible| match (cause, relation) {
            (
                LocalInterleavingRefusal::Budget(LocalInterleavingBudget::WorkBudgetExceeded {
                    phase: LocalInterleavingWorkPhase::BandedSearch,
                    ..
                }),
                Relation::Literal {
                    source,
                    reference,
                    mode,
                },
            ) => BandAlternative::Packed(PreparedLiteralSearch {
                graph,
                source,
                reference,
                mode: *mode,
                preparation,
                feasible,
            }),
            (cause, _) => BandAlternative::Refused(cause),
        };
        // The witness uses this exact relation, including case/fuzzy policy.
        // Prefix/suffix matches are a feasible path even for non-transitive
        // fuzzy relations; we need a lower bound, not a monotonic optimum.
        let mut block = 1;
        while block < reference_len.div_ceil(block) {
            block += 1;
        }
        let dense_work = graph
            .edges
            .len()
            .saturating_mul(reference_len.saturating_add(1));
        let dense_memory = graph
            .edges
            .len()
            .saturating_mul(reference_len / block + block + 6);
        // Do not spend a quadratic witness merely to make a previously
        // affordable dense search exceed its work limit. L=0 is always a
        // feasible path and uses this same scoring implementation unpruned.
        let witness = if dense_work > WORK_CELLS / 2
            && dense_work <= WORK_CELLS
            && dense_memory <= SCORE_CELLS
        {
            CanonicalWitness::zero(relation)
        } else {
            match CanonicalWitness::observe(relation, control)? {
                Ok(witness) => witness,
                Err(refusal) => return Ok(Err(refused(refusal, 0, 0))),
            }
        };
        let mut band = Self {
            graph,
            relation: witness.relation(),
            source_len,
            reference_len,
            feasible: witness.score(),
            block,
            scoring_cells: 0,
            capacity: None,
        };
        let mut preparation = witness.work();
        let (mut work, mut working_cells) = band.geometry(preparation, control)?;
        if work > WORK_CELLS
            && let Some(capacity) = ReferenceCapacity::observe(&witness, control)?
        {
            preparation = preparation.saturating_add(capacity.work());
            band.capacity = Some(capacity);
            (work, working_cells) = band.geometry(preparation, control)?;
        }
        // Admission counts the whole band without allocating/scoring it. Keep
        // the complete required-work diagnostic, not the first over-limit row.
        if work > WORK_CELLS {
            return Ok(Err(refused(
                LocalInterleavingRefusal::Budget(LocalInterleavingBudget::WorkBudgetExceeded {
                    phase: LocalInterleavingWorkPhase::BandedSearch,
                    required_at_least: work,
                }),
                preparation,
                witness.score(),
            )));
        }
        // Capacity payload stays owned by this band through all recomputation.
        band.scoring_cells = working_cells
            .saturating_add(band.capacity.as_ref().map_or(0, ReferenceCapacity::cells));
        if band.scoring_cells > SCORE_CELLS {
            return Ok(Err(refused(
                LocalInterleavingRefusal::Budget(LocalInterleavingBudget::MemoryBudgetExceeded),
                preparation,
                witness.score(),
            )));
        }
        Ok(Ok(band))
    }

    fn geometry<C: WorkControl + ?Sized>(
        &self,
        preparation: usize,
        control: &C,
    ) -> Result<(usize, usize), C::Error> {
        let mut work = preparation;
        let mut peak_row = 0usize;
        let mut block_cells = 0usize;
        let mut peak_block = 0usize;
        for k in 0..=self.reference_len {
            control.check()?;
            let cells = self.slots(k).len();
            work = work.saturating_add(cells);
            peak_row = peak_row.max(cells);
            block_cells = block_cells.saturating_add(cells);
            if k % self.block == 0 {
                peak_block = peak_block.max(block_cells);
                block_cells = cells;
            }
        }
        peak_block = peak_block.max(block_cells);
        // Reserve dense working rows and one inclusive reconstructed block.
        // Retained checkpoint score/index payload is admitted separately,
        // before allocation, under this same machine-word storage limit.
        Ok((work, peak_block.saturating_add(4 * peak_row)))
    }

    fn slots(&self, k: usize) -> Range<usize> {
        let (low, high) = match &self.capacity {
            Some(capacity) => capacity.depths(k),
            None => (
                k.saturating_sub(self.reference_len - self.feasible),
                self.source_len
                    .min(k.saturating_add(self.source_len - self.feasible)),
            ),
        };
        let order = &self.graph.order;
        order.partition_point(|&node| self.graph.depth[node] < low)
            ..order.partition_point(|&node| self.graph.depth[node] <= high)
    }

    fn row(&self, k: usize, fill: Score) -> Row {
        let slots = self.slots(k);
        Row {
            first: slots.start,
            scores: vec![fill; slots.len()],
        }
    }

    pub(super) fn initial(&self) -> Row {
        self.row(0, NonZeroUsize::new(1))
    }

    pub(super) fn terminal(&self) -> Row {
        self.row(self.reference_len, NonZeroUsize::new(1))
    }

    pub(super) fn forward(&self, previous: &Row, k: usize) -> Row {
        let mut row = self.row(k + 1, None);
        for slot in row.slots() {
            row.improve(slot, previous.get(slot));
            // Visit incoming edges: a matched predecessor can be outside this
            // row's band but inside the preceding row's band.
            for edge in &self.graph.incoming[self.graph.order[slot]] {
                let from = self.graph.rank[edge.to];
                row.improve(slot, row.get(from));
                if edge.word.is_some_and(|word| self.relation.matches(word, k)) {
                    row.improve(slot, increment(previous.get(from)));
                }
            }
        }
        row
    }

    pub(super) fn backward(&self, next: &Row, k: usize) -> Row {
        let mut row = self.row(k, None);
        for slot in row.slots().rev() {
            row.improve(slot, next.get(slot));
            for edge in &self.graph.edges[self.graph.order[slot]] {
                let to = self.graph.rank[edge.to];
                row.improve(slot, row.get(to));
                if edge.word.is_some_and(|word| self.relation.matches(word, k)) {
                    row.improve(slot, increment(next.get(to)));
                }
            }
        }
        row
    }

    pub(super) fn observe_edges(
        &self,
        forward: &Row,
        backward: &Row,
        next: Option<(&Row, usize)>,
        optimum: Score,
        words: &mut [WordEvidence],
    ) -> usize {
        let mut added = 0;
        for slot in forward.slots() {
            let Some(prefix) = forward.get(slot) else {
                continue;
            };
            for edge in &self.graph.edges[self.graph.order[slot]] {
                let Some(word) = edge.word else { continue };
                let to = self.graph.rank[edge.to];
                if combined(Some(prefix), backward.get(to), 0)
                    .is_some_and(|score| Some(score) == optimum)
                {
                    words[word].can_be_missing = true;
                }
                if let Some((next, k)) = next
                    && self.relation.matches(word, k)
                    && combined(Some(prefix), next.get(to), 1)
                        .is_some_and(|score| Some(score) == optimum)
                    && words[word].candidates.last() != Some(&k)
                {
                    words[word].candidates.push(k);
                    // The last-index guard counts source/reference pairs once,
                    // even when many episode edges carry the same source word.
                    added += 1;
                }
            }
        }
        added
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_multiplicity_bounds_admit_a_wide_joint_band() {
        let payload: Vec<_> = (0..100)
            .map(|i| vec![if i % 2 == 0 { "a" } else { "b" }.to_owned(); 10])
            .collect();
        let turns: Vec<_> = payload
            .iter()
            .enumerate()
            .map(|(i, words)| SpeakerTurn::new(if i % 2 == 0 { "A" } else { "B" }, words))
            .collect();
        let reference: Vec<_> = std::iter::repeat_n("a".to_owned(), 6000)
            .chain(payload.iter().flatten().cloned())
            .collect();
        let materialized = PreparedGraph::admit(&turns)
            .unwrap()
            .materialize(&Uninterrupted)
            .unwrap();
        let graph = materialized.graph();
        let relation = Relation::new(
            payload.iter().flatten().map(String::as_str).collect(),
            &reference,
            MatchMode::Exact,
        );
        let band = uninterrupted(Band::admit(graph, &relation, 1000, 7000, &Uninterrupted))
            .expect("word-count upper bounds fit the unchanged joint work limit");
        assert_eq!(band.feasible, 1000);
        assert!(band.capacity.is_some());
        assert_eq!(
            band.scoring_cells,
            band.geometry(0, &Uninterrupted).unwrap().1 + band.capacity.as_ref().unwrap().cells()
        );
        assert!((0..=7000).map(|k| band.slots(k).len()).sum::<usize>() < WORK_CELLS);
        let original = Band {
            capacity: None,
            ..band
        };
        assert!(original.geometry(0, &Uninterrupted).unwrap().0 > WORK_CELLS);
    }

    #[test]
    fn capacity_pruning_preserves_all_candidates_and_missingness_on_small_joint_graphs() {
        fn evidence(band: &Band<'_, '_>, words: usize) -> (Score, Vec<(Vec<usize>, bool)>) {
            let mut forward = vec![band.initial()];
            for k in 0..band.reference_len {
                forward.push(band.forward(forward.last().unwrap(), k));
            }
            let optimum = forward.last().unwrap().get(band.graph.rank[band.graph.end]);
            let mut backward = band.terminal();
            let mut words: Vec<_> = (0..words)
                .map(|_| WordEvidence {
                    candidates: Vec::new(),
                    can_be_missing: false,
                })
                .collect();
            band.observe_edges(
                forward.last().unwrap(),
                &backward,
                None,
                optimum,
                &mut words,
            );
            for k in (0..band.reference_len).rev() {
                let previous = band.backward(&backward, k);
                band.observe_edges(
                    &forward[k],
                    &previous,
                    Some((&backward, k)),
                    optimum,
                    &mut words,
                );
                backward = previous;
            }
            (
                optimum,
                words
                    .into_iter()
                    .map(|word| (word.candidates, word.can_be_missing))
                    .collect(),
            )
        }
        for m in 0..=4 {
            for n in 0..=4 {
                for a in 0..1usize << m {
                    let source: Vec<_> = (0..m)
                        .map(|i| if a & (1 << i) == 0 { "a" } else { "b" }.to_owned())
                        .collect();
                    for b in 0..1usize << n {
                        let reference: Vec<_> = (0..n)
                            .map(|i| if b & (1 << i) == 0 { "a" } else { "b" }.to_owned())
                            .collect();
                        for split in 0..=m {
                            let turns = [
                                SpeakerTurn::new("A", &source[..split]),
                                SpeakerTurn::new("B", &source[split..]),
                            ];
                            let materialized = PreparedGraph::admit(&turns)
                                .unwrap()
                                .materialize(&Uninterrupted)
                                .unwrap();
                            let graph = materialized.graph();
                            let relation = Relation::new(
                                source.iter().map(String::as_str).collect(),
                                &reference,
                                MatchMode::Exact,
                            );
                            let mut pruned =
                                uninterrupted(Band::admit(graph, &relation, m, n, &Uninterrupted))
                                    .unwrap();
                            // Exercise the new producer even though small graphs need no rescue.
                            let witness =
                                uninterrupted(CanonicalWitness::observe(&relation, &Uninterrupted))
                                    .unwrap();
                            pruned.capacity =
                                ReferenceCapacity::observe(&witness, &Uninterrupted).unwrap();
                            let dense = Band {
                                graph,
                                relation: &relation,
                                source_len: m,
                                reference_len: n,
                                feasible: 0,
                                block: pruned.block,
                                scoring_cells: 0,
                                capacity: None,
                            };
                            assert_eq!(
                                evidence(&dense, m),
                                evidence(&pruned, m),
                                "m={m} n={n} a={a} b={b} split={split}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_long_near_perfect_canonical_witness_does_not_require_quadratic_work() {
        let payload: Vec<_> = (0..9000).map(|i| format!("word{i}")).collect();
        let mut reference = payload.clone();
        reference[0] = "inserted_first".to_owned();
        reference[8999] = "inserted_last".to_owned();
        let turns = [SpeakerTurn::new("A", &payload)];
        let materialized = PreparedGraph::admit(&turns)
            .unwrap()
            .materialize(&Uninterrupted)
            .unwrap();
        let graph = materialized.graph();
        let relation = Relation::new(
            payload.iter().map(String::as_str).collect(),
            &reference,
            MatchMode::Exact,
        );
        let band =
            uninterrupted(Band::admit(graph, &relation, 9000, 9000, &Uninterrupted)).unwrap();
        assert_eq!(band.feasible, 8998);
        assert!((0..=9000).map(|k| band.slots(k).len()).sum::<usize>() < 50_000);
    }

    #[test]
    fn an_affordable_dense_search_does_not_pay_for_a_quadratic_witness() {
        let payload = vec!["source".to_owned(); 6000];
        let reference = vec!["other".to_owned(); 6000];
        let turns = [SpeakerTurn::new("A", &payload)];
        let materialized = PreparedGraph::admit(&turns)
            .unwrap()
            .materialize(&Uninterrupted)
            .unwrap();
        let graph = materialized.graph();
        let relation = Relation::new(
            payload.iter().map(String::as_str).collect(),
            &reference,
            MatchMode::Exact,
        );
        let band =
            uninterrupted(Band::admit(graph, &relation, 6000, 6000, &Uninterrupted)).unwrap();
        assert_eq!(band.feasible, 0);
        assert!(band.capacity.is_none());
        assert_eq!(band.slots(3000).len(), 6001);
    }

    #[test]
    fn compressible_checkpoints_fit_without_relaxing_work_or_memory_limits() {
        let payload: Vec<_> = (0..17_000).map(|i| format!("word{i}")).collect();
        let other = vec!["unobserved".to_owned()];
        let reference = payload[..400].to_vec();
        let turns = [
            SpeakerTurn::new("A", &payload),
            SpeakerTurn::new("B", &other),
        ];
        let observed = LocalInterleaving::observe(&turns, &reference, MatchMode::Exact)
            .expect("losslessly compressed checkpoints fit the existing score budget");
        assert_eq!(observed.matched_words(), 400);
        for (index, word) in observed.words().enumerate() {
            if index < 400 {
                assert_eq!(word.common().unwrap().matched().reference_index(), index);
                assert!(!word.can_be_missing());
            } else {
                assert!(word.can_be_missing());
                assert!(word.common().is_none());
            }
        }
    }

    #[test]
    fn perfect_long_dialogue_visits_each_depth_state_once() {
        let payload: Vec<Vec<String>> = (0..280)
            .map(|i| (0..10).map(|j| format!("word{}", i * 10 + j)).collect())
            .collect();
        let turns: Vec<_> = payload
            .iter()
            .enumerate()
            .map(|(i, words)| SpeakerTurn::new(if i % 2 == 0 { "A" } else { "B" }, words))
            .collect();
        let reference: Vec<_> = payload.iter().flatten().cloned().collect();
        let materialized = PreparedGraph::admit(&turns)
            .unwrap()
            .materialize(&Uninterrupted)
            .unwrap();
        let graph = materialized.graph();
        let relation = Relation::new(
            payload.iter().flatten().map(String::as_str).collect(),
            &reference,
            MatchMode::Exact,
        );
        let band =
            uninterrupted(Band::admit(graph, &relation, 2800, 2800, &Uninterrupted)).unwrap();
        assert_eq!(band.feasible, 2800);
        assert_eq!(
            (0..=2800).map(|k| band.slots(k).len()).sum::<usize>(),
            36_002
        );
        assert_eq!(graph.edges.len() * 2801, 100_841_602);
        assert_eq!(std::mem::size_of::<Score>(), std::mem::size_of::<usize>());
    }
}
