//! Exact score-bounded storage for the source DAG. A canonical monotonic path
//! supplies a feasible score L. At reference prefix k and source depth d, any
//! complete path scores at most min(d,k) + min(m-d,n-k). Only states whose
//! upper bound is below L are removed; no optimal correspondence is removed.

use super::*;
use std::num::NonZeroUsize;
use std::ops::Range;

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
pub(super) struct Band<'graph> {
    graph: &'graph Graph,
    source_len: usize,
    reference_len: usize,
    feasible: usize,
    pub(super) block: usize,
}

impl<'graph> Band<'graph> {
    pub(super) fn admit<C: WorkControl + ?Sized>(
        graph: &'graph Graph,
        relation: &Relation<'_>,
        source_len: usize,
        reference_len: usize,
        control: &C,
    ) -> Result<Result<Self, LocalInterleavingRefusal>, C::Error> {
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
        let (feasible, preliminary_work) = if dense_work > WORK_CELLS / 2
            && dense_work <= WORK_CELLS
            && dense_memory <= SCORE_CELLS
        {
            (0, 0)
        } else {
            match witness(relation, source_len, reference_len, control)? {
                Ok(witness) => witness,
                Err(refusal) => return Ok(Err(refusal)),
            }
        };
        let band = Self {
            graph,
            source_len,
            reference_len,
            feasible,
            block,
        };
        let mut work = preliminary_work;
        let mut checkpoints = 0usize;
        let mut peak_row = 0usize;
        let mut block_cells = 0usize;
        let mut peak_block = 0usize;
        for k in 0..=reference_len {
            control.check()?;
            let cells = band.slots(k).len();
            work = work.saturating_add(cells);
            if work > WORK_CELLS {
                return Ok(Err(LocalInterleavingRefusal::WorkBudgetExceeded));
            }
            peak_row = peak_row.max(cells);
            block_cells = block_cells.saturating_add(cells);
            if k % block == 0 {
                checkpoints = checkpoints.saturating_add(cells);
                peak_block = peak_block.max(block_cells);
                block_cells = cells;
            }
        }
        peak_block = peak_block.max(block_cells);
        // Retained checkpoints, one reconstructed inclusive block, and spare
        // forward/backward/cloned rows. Row uses one machine word per score.
        if checkpoints
            .saturating_add(peak_block)
            .saturating_add(4 * peak_row)
            > SCORE_CELLS
        {
            return Ok(Err(LocalInterleavingRefusal::MemoryBudgetExceeded));
        }
        Ok(Ok(band))
    }

    fn slots(&self, k: usize) -> Range<usize> {
        let low = k.saturating_sub(self.reference_len - self.feasible);
        let high = self
            .source_len
            .min(k.saturating_add(self.source_len - self.feasible));
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

    pub(super) fn forward(&self, previous: &Row, k: usize, relation: &Relation<'_>) -> Row {
        let mut row = self.row(k + 1, None);
        for slot in row.slots() {
            row.improve(slot, previous.get(slot));
            // Visit incoming edges: a matched predecessor can be outside this
            // row's band but inside the preceding row's band.
            for edge in &self.graph.incoming[self.graph.order[slot]] {
                let from = self.graph.rank[edge.to];
                row.improve(slot, row.get(from));
                if edge.word.is_some_and(|word| relation.matches(word, k)) {
                    row.improve(slot, increment(previous.get(from)));
                }
            }
        }
        row
    }

    pub(super) fn backward(&self, next: &Row, k: usize, relation: &Relation<'_>) -> Row {
        let mut row = self.row(k, None);
        for slot in row.slots().rev() {
            row.improve(slot, next.get(slot));
            for edge in &self.graph.edges[self.graph.order[slot]] {
                let to = self.graph.rank[edge.to];
                row.improve(slot, row.get(to));
                if edge.word.is_some_and(|word| relation.matches(word, k)) {
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
        relation: &Relation<'_>,
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
                    && relation.matches(word, k)
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

    pub(super) fn traceback<C: WorkControl + ?Sized>(
        &self,
        checkpoints: &[Row],
        width: usize,
        reference_len: usize,
        relation: &Relation<'_>,
        control: &C,
    ) -> Result<Vec<Selected>, C::Error> {
        let mut node = self.graph.end;
        let mut k = reference_len;
        let mut selected = Vec::new();
        while k > 0 {
            let start = (k - 1) / width * width;
            let mut rows = vec![checkpoints[start / width].clone()];
            for reference in start..k {
                control.check()?;
                rows.push(self.forward(&rows[reference - start], reference, relation));
            }
            while k > start {
                control.check()?;
                let current = &rows[k - start];
                let previous = &rows[k - start - 1];
                let score = current.get(self.graph.rank[node]);
                debug_assert!(score.is_some());
                if score == previous.get(self.graph.rank[node]) {
                    k -= 1;
                    continue;
                }
                let mut advanced = false;
                for edge in &self.graph.incoming[node] {
                    let from = self.graph.rank[edge.to];
                    if let Some(word) = edge.word
                        && relation.matches(word, k - 1)
                        && increment(previous.get(from)) == score
                    {
                        selected.push(Selected {
                            source: word,
                            reference: k - 1,
                        });
                        node = edge.to;
                        k -= 1;
                        advanced = true;
                        break;
                    }
                    if current.get(from) == score {
                        node = edge.to;
                        advanced = true;
                        break;
                    }
                }
                // The admitted band retains every optimum and its incoming
                // edges. Rows and graph never come from an external producer.
                debug_assert!(advanced);
            }
        }
        selected.reverse();
        Ok(selected)
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
        return Ok(Err(LocalInterleavingRefusal::WorkBudgetExceeded));
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
        return Ok(Err(LocalInterleavingRefusal::WorkBudgetExceeded));
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
        assert_eq!(band.slots(3000).len(), 6001);
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
