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

/// A failed admission retains actual input ownership and already-paid work.
/// Only this producer constructs it; an alternative cannot substitute inputs
/// or forget preliminary witness/capacity costs.
#[derive(Clone, Copy)]
pub(super) struct RefusedBand<'graph, 'source> {
    graph: &'graph Graph,
    relation: &'graph Relation<'source>,
    refusal: LocalInterleavingRefusal,
    preparation: usize,
    feasible: usize,
}

impl std::fmt::Debug for RefusedBand<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.refusal.fmt(f)
    }
}

impl<'graph, 'source> RefusedBand<'graph, 'source> {
    pub(super) fn refusal(&self) -> LocalInterleavingRefusal {
        self.refusal
    }
    pub(super) fn graph(&self) -> &'graph Graph {
        self.graph
    }
    pub(super) fn relation(&self) -> &'graph Relation<'source> {
        self.relation
    }
    pub(super) fn preparation(&self) -> usize {
        self.preparation
    }
    pub(super) fn feasible(&self) -> usize {
        self.feasible
    }
    #[cfg(test)]
    pub(super) fn testing_lower_bound(mut self, feasible: usize) -> Self {
        assert!(feasible <= self.feasible);
        self.feasible = feasible;
        self
    }
    #[cfg(test)]
    pub(super) fn testing(graph: &'graph Graph, relation: &'graph Relation<'source>) -> Self {
        Self {
            graph,
            relation,
            preparation: 0,
            feasible: match relation {
                Relation::Literal {
                    source, reference, ..
                } => uninterrupted(CanonicalWitness::observe(
                    relation,
                    source.len(),
                    reference.len(),
                    &Uninterrupted,
                ))
                .map(|witness| witness.score())
                .unwrap_or(0),
                Relation::Fuzzy {
                    source, reference, ..
                } => uninterrupted(CanonicalWitness::observe(
                    relation,
                    source.len(),
                    reference.len(),
                    &Uninterrupted,
                ))
                .map(|witness| witness.score())
                .unwrap_or(0),
            },
            refusal: LocalInterleavingRefusal::WorkBudgetExceeded {
                phase: LocalInterleavingWorkPhase::BandedSearch,
                required_at_least: WORK_CELLS + 1,
            },
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
    ) -> Result<Result<Self, RefusedBand<'graph, 'source>>, C::Error> {
        let refused = |refusal, preparation, feasible| RefusedBand {
            graph,
            relation,
            refusal,
            preparation,
            feasible,
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
            match CanonicalWitness::observe(relation, source_len, reference_len, control)? {
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
                LocalInterleavingRefusal::WorkBudgetExceeded {
                    phase: LocalInterleavingWorkPhase::BandedSearch,
                    required_at_least: work,
                },
                preparation,
                witness.score(),
            )));
        }
        // Capacity payload stays owned by this band through all recomputation.
        band.scoring_cells = working_cells
            .saturating_add(band.capacity.as_ref().map_or(0, ReferenceCapacity::cells));
        if band.scoring_cells > SCORE_CELLS {
            return Ok(Err(refused(
                LocalInterleavingRefusal::MemoryBudgetExceeded,
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

/// Construct a feasible canonical-order path score with linear retained memory.
/// Its only authority is a lower bound, never a correspondence certificate.
fn witness<C: WorkControl + ?Sized>(
    relation: &Relation<'_>,
    m: usize,
    n: usize,
    control: &C,
) -> Result<Result<(usize, usize), LocalInterleavingRefusal>, C::Error> {
    if m.max(n) > WORK_CELLS {
        return Ok(Err(LocalInterleavingRefusal::WorkBudgetExceeded {
            phase: LocalInterleavingWorkPhase::FeasibleWitness,
            required_at_least: m.max(n),
        }));
    }
    let mut prefix = 0;
    while prefix < m.min(n) && relation.matches(prefix, prefix) {
        if prefix % 1024 == 0 {
            control.check()?;
        }
        prefix += 1;
    }
    let mut suffix = 0;
    while suffix < m.min(n) - prefix && relation.matches(m - 1 - suffix, n - 1 - suffix) {
        if suffix % 1024 == 0 {
            control.check()?;
        }
        suffix += 1;
    }
    let a = m - prefix - suffix;
    let b = n - prefix - suffix;
    let work = a.saturating_mul(b).saturating_add(prefix + suffix + 2);
    if work > WORK_CELLS {
        return Ok(Err(LocalInterleavingRefusal::WorkBudgetExceeded {
            phase: LocalInterleavingWorkPhase::FeasibleWitness,
            required_at_least: work,
        }));
    }
    if b.saturating_add(1) > SCORE_CELLS {
        return Ok(Err(LocalInterleavingRefusal::MemoryBudgetExceeded));
    }
    let mut row = vec![0usize; b + 1];
    for i in prefix..m - suffix {
        control.check()?;
        let mut diagonal = 0;
        for j in 0..b {
            if j % 1024 == 0 {
                control.check()?;
            }
            let old = row[j + 1];
            row[j + 1] = old.max(row[j]);
            if relation.matches(i, prefix + j) {
                row[j + 1] = row[j + 1].max(diagonal + 1);
            }
            diagonal = old;
        }
    }
    Ok(Ok((prefix + suffix + row[b], work)))
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
        let (graph, _) = Graph::of(&turns, Budget::nodes(&turns).unwrap());
        let relation = Relation::new(
            payload.iter().flatten().map(String::as_str).collect(),
            &reference,
            MatchMode::Exact,
        );
        let band = uninterrupted(Band::admit(&graph, &relation, 1000, 7000, &Uninterrupted))
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
                            let (graph, _) = Graph::of(&turns, Budget::nodes(&turns).unwrap());
                            let relation = Relation::new(
                                source.iter().map(String::as_str).collect(),
                                &reference,
                                MatchMode::Exact,
                            );
                            let mut pruned =
                                uninterrupted(Band::admit(&graph, &relation, m, n, &Uninterrupted))
                                    .unwrap();
                            // Exercise the new producer even though small graphs need no rescue.
                            let witness = uninterrupted(CanonicalWitness::observe(
                                &relation,
                                m,
                                n,
                                &Uninterrupted,
                            ))
                            .unwrap();
                            pruned.capacity =
                                ReferenceCapacity::observe(&witness, &Uninterrupted).unwrap();
                            let dense = Band {
                                graph: &graph,
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
        let (graph, _) = Graph::of(&turns, Budget::nodes(&turns).unwrap());
        let relation = Relation::new(
            payload.iter().map(String::as_str).collect(),
            &reference,
            MatchMode::Exact,
        );
        let band =
            uninterrupted(Band::admit(&graph, &relation, 9000, 9000, &Uninterrupted)).unwrap();
        assert_eq!(band.feasible, 8998);
        assert!((0..=9000).map(|k| band.slots(k).len()).sum::<usize>() < 50_000);
    }

    #[test]
    fn an_affordable_dense_search_does_not_pay_for_a_quadratic_witness() {
        let payload = vec!["source".to_owned(); 6000];
        let reference = vec!["other".to_owned(); 6000];
        let turns = [SpeakerTurn::new("A", &payload)];
        let (graph, _) = Graph::of(&turns, Budget::nodes(&turns).unwrap());
        let relation = Relation::new(
            payload.iter().map(String::as_str).collect(),
            &reference,
            MatchMode::Exact,
        );
        let band =
            uninterrupted(Band::admit(&graph, &relation, 6000, 6000, &Uninterrupted)).unwrap();
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
        let (graph, _) = Graph::of(&turns, Budget::nodes(&turns).unwrap());
        let relation = Relation::new(
            payload.iter().flatten().map(String::as_str).collect(),
            &reference,
            MatchMode::Exact,
        );
        let band =
            uninterrupted(Band::admit(&graph, &relation, 2800, 2800, &Uninterrupted)).unwrap();
        assert_eq!(band.feasible, 2800);
        assert_eq!(
            (0..=2800).map(|k| band.slots(k).len()).sum::<usize>(),
            36_002
        );
        assert_eq!(graph.edges.len() * 2801, 100_841_602);
        assert_eq!(std::mem::size_of::<Score>(), std::mem::size_of::<usize>());
    }
}
