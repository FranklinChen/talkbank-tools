//! One owner for both representations' proof and inspection. Reconstructed
//! blocks are consumed in descending order, never rebuilt just for inspection.

use super::super::profiles::{MatchCuts, Suffix};
use super::cost::{BackwardWork, EdgeWork, TracebackWork};
use super::*;

struct Cursor {
    slot: usize,
    reference: usize,
}

/// Partial inspection cannot escape independently of the complete graph proof.
/// No operation accepts a separately supplied block/cursor pair.
struct PendingProof<'l, 'p, 'g, 's, 'c, C: ?Sized> {
    forward: CompleteForward<'l, 'p, 'g, 's, 'c, C>,
    cursor: Option<Cursor>,
    selected: Vec<Selected>,
}
impl<'l, 'p, 'g, 's, 'c, C: WorkControl + ?Sized> PendingProof<'l, 'p, 'g, 's, 'c, C> {
    fn new(forward: CompleteForward<'l, 'p, 'g, 's, 'c, C>) -> Self {
        let graph = forward.ledger.plan().graph();
        let cursor = Some(Cursor {
            slot: graph.rank[graph.end],
            reference: forward.ledger.plan().length(),
        });
        Self {
            forward,
            cursor,
            selected: Vec::new(),
        }
    }

    fn finish(mut self) -> Result<Result<Observation, LocalInterleavingRefusal>, Stop<C::Error>> {
        let ledger = self.forward.ledger;
        let control = ledger.stage::<EdgeWork>();
        let plan = self.forward.ledger.plan();
        let graph = plan.graph();
        let mut backward = Frontier::<SuffixDirection>::new(plan);
        let mut suffixes: HashMap<usize, Suffix> = HashMap::new();
        let mut candidates = BTreeSet::new();
        let mut missing = vec![false; plan.source().len()];
        for index in (0..self.forward.snapshots.len()).rev() {
            let block = self.forward.reconstruct(index)?;
            for offset in (0..block.rows.len()).rev() {
                let slot = block.start + offset;
                control.check()?;
                if let Row::Viable(prefix) = &block.rows[offset] {
                    for edge in &graph.edges[graph.order[slot]] {
                        let Some(word) = edge.word else { continue };
                        let Some(suffix) = suffixes.get(&graph.rank[edge.to]) else {
                            continue;
                        };
                        let observed =
                            suffix.observe_edge(prefix, self.forward.optimum, &control)?;
                        if observed.can_be_missing {
                            missing[word] = true;
                        }
                        let MatchCuts::Domain {
                            first,
                            bits: matched,
                        } = observed.matches
                        else {
                            continue;
                        };
                        control.charge(matched.len().saturating_mul(3))?;
                        let mask = plan.forward.mask_window(
                            plan.source()[word],
                            first,
                            (matched.len() * 64).min(plan.length() - first),
                        );
                        for (at, (&cuts, &mask)) in matched.iter().zip(&mask).enumerate() {
                            let mut possible = cuts & mask;
                            while possible != 0 {
                                control.candidate()?;
                                let bit = possible.trailing_zeros() as usize;
                                possible &= possible - 1;
                                let addressed = (word, first + at * 64 + bit);
                                if candidates.len() == CANDIDATES
                                    && !candidates.contains(&addressed)
                                {
                                    return Ok(Err(
                                        LocalInterleavingRefusal::CandidateBudgetExceeded,
                                    ));
                                }
                                candidates.insert(addressed);
                            }
                        }
                    }
                }
                let row = backward.compute(slot, &ledger.stage::<BackwardWork>())?;
                if let Row::Viable(profile) = &row {
                    suffixes.insert(
                        slot,
                        Suffix::of(profile, plan.length(), &ledger.stage::<BackwardWork>())?,
                    );
                }
                let expired = backward.commit(slot, row);
                for slot in expired {
                    suffixes.remove(&slot);
                }
            }

            // The same complete block now advances the inspection path. Scalar
            // probes are reserved in preparation; no packed reconstruction is
            // charged or performed here. Incoming-edge and reference tie order
            // is identical to the separate traceback.
            while let Some(cursor) = &mut self.cursor {
                if cursor.slot < block.start {
                    break;
                }
                if cursor.slot == 0 || cursor.reference == 0 {
                    self.cursor = None;
                    break;
                }
                ledger.stage::<TracebackWork>().check()?;
                let score = block.score(cursor.slot, cursor.reference);
                if score == block.score(cursor.slot, cursor.reference - 1) {
                    cursor.reference -= 1;
                    continue;
                }
                let mut advanced = false;
                for edge in &graph.incoming[graph.order[cursor.slot]] {
                    let parent = graph.rank[edge.to];
                    if let Some(word) = edge.word
                        && plan
                            .admitted
                            .refused()
                            .relation()
                            .matches(word, cursor.reference - 1)
                        && block
                            .score(parent, cursor.reference - 1)
                            .is_some_and(|prefix| Some(prefix + 1) == score)
                    {
                        self.selected.push(Selected {
                            source: word,
                            reference: cursor.reference - 1,
                        });
                        cursor.slot = parent;
                        cursor.reference -= 1;
                        advanced = true;
                        break;
                    }
                    if block.score(parent, cursor.reference) == score {
                        cursor.slot = parent;
                        advanced = true;
                        break;
                    }
                }
                debug_assert!(advanced);
                if !advanced {
                    self.cursor = None;
                    break;
                }
            }
        }
        // Only this consuming transition can expose selection, after every
        // all-optimal edge and the candidate cap have completed successfully.
        self.selected.reverse();
        debug_assert_eq!(self.selected.len(), self.forward.optimum);
        let mut words: Vec<_> = missing
            .into_iter()
            .map(|can_be_missing| WordEvidence {
                candidates: Vec::new(),
                can_be_missing,
            })
            .collect();
        for (word, reference) in candidates.into_iter().rev() {
            words[word].candidates.push(reference);
        }
        Ok(Ok(Observation {
            words,
            selected: self.selected,
        }))
    }
}

pub(super) fn observe<C: WorkControl + ?Sized>(
    forward: CompleteForward<'_, '_, '_, '_, '_, C>,
) -> Result<Result<Observation, LocalInterleavingRefusal>, Stop<C::Error>> {
    PendingProof::new(forward).finish()
}
