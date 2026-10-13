//! Lossless, budget-owned forward checkpoints. Working rows stay dense;
//! retained rows choose dense scores or equal-score runs, counting both run
//! indices and scores against the unchanged machine-word storage budget.

use super::*;

struct Run {
    end: usize,
    score: Score,
}

enum Checkpoint {
    Dense(Row),
    Runs {
        first: usize,
        len: usize,
        runs: Vec<Run>,
    },
}

/// Observe immutable row geometry before allocating any checkpoint payload.
struct StoragePlan<'row> {
    row: &'row Row,
    runs: usize,
    cells: usize,
}

impl<'row> StoragePlan<'row> {
    fn observe(row: &'row Row) -> Self {
        let runs = usize::from(!row.scores.is_empty())
            + row
                .scores
                .windows(2)
                .filter(|pair| pair[0] != pair[1])
                .count();
        let run_cells =
            runs.saturating_mul(std::mem::size_of::<Run>() / std::mem::size_of::<Score>());
        Self {
            row,
            runs,
            cells: run_cells.min(row.scores.len()),
        }
    }

    fn retain(self) -> Checkpoint {
        if self.cells == self.row.scores.len() {
            return Checkpoint::Dense(self.row.clone());
        }
        let mut runs = Vec::with_capacity(self.runs);
        for (index, &score) in self.row.scores.iter().enumerate() {
            match runs.last_mut() {
                Some(Run {
                    end,
                    score: previous,
                }) if *previous == score => *end = index + 1,
                _ => runs.push(Run {
                    end: index + 1,
                    score,
                }),
            }
        }
        Checkpoint::Runs {
            first: self.row.first,
            len: self.row.scores.len(),
            runs,
        }
    }
}

impl Checkpoint {
    fn restore(&self) -> Row {
        match self {
            Self::Dense(row) => row.clone(),
            Self::Runs { first, len, runs } => {
                let mut scores = Vec::with_capacity(*len);
                for run in runs {
                    scores.extend(std::iter::repeat_n(run.score, run.end - scores.len()));
                }
                Row {
                    first: *first,
                    scores,
                }
            }
        }
    }
}

/// The band owns graph/relation/block geometry and the storage reservation.
/// No consumer can supply a replacement checkpoint width or source layout.
pub(in super::super) struct ForwardScores<'band, 'graph, 'source> {
    band: &'band Band<'graph, 'source>,
    checkpoints: Vec<Checkpoint>,
    last: Row,
}

pub(in super::super) struct ScoreBlock {
    start: usize,
    rows: Vec<Row>,
}

impl ScoreBlock {
    pub(in super::super) fn start(&self) -> usize {
        self.start
    }

    pub(in super::super) fn steps_rev(&self) -> impl Iterator<Item = (usize, &Row)> {
        self.rows
            .windows(2)
            .enumerate()
            .rev()
            .map(|(offset, pair)| (self.start + offset, &pair[0]))
    }
}

impl<'band, 'graph, 'source> ForwardScores<'band, 'graph, 'source> {
    pub(in super::super) fn observe<C: WorkControl + ?Sized>(
        band: &'band Band<'graph, 'source>,
        control: &C,
    ) -> Result<Result<Self, LocalInterleavingRefusal>, C::Error> {
        let mut row = band.initial();
        let mut checkpoints = Vec::new();
        let mut retained = 0usize;
        for k in 0..=band.reference_len {
            control.check()?;
            if k % band.block == 0 {
                let plan = StoragePlan::observe(&row);
                let required = retained.saturating_add(plan.cells);
                if required.saturating_add(band.scoring_cells) > SCORE_CELLS {
                    return Ok(Err(LocalInterleavingRefusal::MemoryBudgetExceeded));
                }
                retained = required;
                checkpoints.push(plan.retain());
            }
            if k < band.reference_len {
                row = band.forward(&row, k);
            }
        }
        Ok(Ok(Self {
            band,
            checkpoints,
            last: row,
        }))
    }

    pub(in super::super) fn last(&self) -> &Row {
        &self.last
    }

    pub(in super::super) fn block_ending_at<C: WorkControl + ?Sized>(
        &self,
        end: usize,
        control: &C,
    ) -> Result<ScoreBlock, C::Error> {
        let start = (end - 1) / self.band.block * self.band.block;
        let mut rows = Vec::with_capacity(end - start + 1);
        rows.push(self.checkpoints[start / self.band.block].restore());
        for k in start..end {
            control.check()?;
            rows.push(self.band.forward(&rows[k - start], k));
        }
        Ok(ScoreBlock { start, rows })
    }

    pub(in super::super) fn traceback<C: WorkControl + ?Sized>(
        &self,
        control: &C,
    ) -> Result<Vec<Selected>, C::Error> {
        let mut node = self.band.graph.end;
        let mut k = self.band.reference_len;
        let mut selected = Vec::new();
        while k > 0 {
            let block = self.block_ending_at(k, control)?;
            while k > block.start {
                control.check()?;
                let current = &block.rows[k - block.start];
                let previous = &block.rows[k - block.start - 1];
                let score = current.get(self.band.graph.rank[node]);
                debug_assert!(score.is_some());
                if score == previous.get(self.band.graph.rank[node]) {
                    k -= 1;
                    continue;
                }
                let mut advanced = false;
                for edge in &self.band.graph.incoming[node] {
                    let from = self.band.graph.rank[edge.to];
                    if let Some(word) = edge.word
                        && self.band.relation.matches(word, k - 1)
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
                debug_assert!(advanced);
            }
        }
        selected.reverse();
        Ok(selected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Storage preserves unreachable versus reachable-zero, offsets and every
    /// score. Policy alternatives and encoding are not proved by a type alone.
    #[test]
    fn every_small_score_row_roundtrips_exactly() {
        let alphabet = [None, NonZeroUsize::new(1), NonZeroUsize::new(2)];
        for len in 0..=8u32 {
            for mut code in 0..3usize.pow(len) {
                let mut scores = Vec::new();
                for _ in 0..len {
                    scores.push(alphabet[code % 3]);
                    code /= 3;
                }
                let row = Row { first: 7, scores };
                let plan = StoragePlan::observe(&row);
                assert!(plan.cells <= row.scores.len());
                let restored = plan.retain().restore();
                assert_eq!(restored.first, row.first);
                assert_eq!(restored.scores, row.scores);
            }
        }
    }

    #[test]
    fn incompressible_rows_remain_dense_and_uniform_rows_count_run_indices() {
        let row = Row {
            first: 0,
            scores: vec![None, NonZeroUsize::new(1), None],
        };
        assert!(matches!(
            StoragePlan::observe(&row).retain(),
            Checkpoint::Dense(_)
        ));
        let row = Row {
            first: 0,
            scores: vec![NonZeroUsize::new(1); 1000],
        };
        let plan = StoragePlan::observe(&row);
        assert_eq!(plan.cells, 2);
        assert!(matches!(plan.retain(), Checkpoint::Runs { .. }));
    }
}
