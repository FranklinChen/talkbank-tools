//! Pending all-optimal domains own their deduplication and candidate allowance.
//! Only the consuming complete proof can publish them as word evidence.

use super::{CANDIDATES, LocalInterleavingBudget, LocalInterleavingRefusal, Plan, WordEvidence};
use std::collections::{BTreeMap, btree_map::Entry, btree_map::VacantEntry};
use std::num::NonZeroU64;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ReferenceBlock(usize);

#[derive(Clone, Copy)]
struct Block {
    index: ReferenceBlock,
    bits: NonZeroU64,
}

/// The common single-block domain allocates no tree. Larger domains store
/// one key per 64 reference slots, never one tree node per addressed pair.
enum Domain {
    Empty,
    Inline(Block),
    #[expect(
        clippy::box_collection,
        reason = "Boxing the rare sparse header keeps every per-word enum slot compact"
    )]
    Sparse(Box<BTreeMap<ReferenceBlock, NonZeroU64>>),
}

enum Missingness {
    NotObserved,
    Witnessed,
}

struct PendingWord {
    domain: Domain,
    cardinality: WordCardinality,
    missing: Missingness,
}

struct WordCardinality(usize);

/// Remaining population allowance is not an individual word's cardinality.
/// Neither can be supplied separately to an insertion or changed by a caller.
struct CandidateAllowance {
    remaining: usize,
}

pub(super) struct PendingCandidates {
    words: Vec<PendingWord>,
    allowance: CandidateAllowance,
}

/// A retained insertion location, not another lookup or a mutation capability
/// a consumer can manufacture. Its unique-bit increment is admitted first.
enum InsertionTarget<'domain> {
    Inline(&'domain mut Domain, Block),
    Merge(&'domain mut NonZeroU64, NonZeroU64),
    Promote(&'domain mut Domain, [Block; 2]),
    Vacant(VacantEntry<'domain, ReferenceBlock, NonZeroU64>, NonZeroU64),
}

struct PreparedInsertion<'domain> {
    target: InsertionTarget<'domain>,
    additional: usize,
    cardinality: &'domain mut WordCardinality,
    allowance: &'domain mut CandidateAllowance,
}

impl<'domain> PreparedInsertion<'domain> {
    fn prepare(census: &'domain mut PendingCandidates, word: usize, block: Block) -> Self {
        let PendingCandidates { words, allowance } = census;
        let PendingWord {
            domain,
            cardinality,
            ..
        } = &mut words[word];
        let merge = |target: &'domain mut NonZeroU64, bits: NonZeroU64| {
            let additional = (bits.get() & !target.get()).count_ones() as usize;
            let merged = *target | bits;
            (InsertionTarget::Merge(target, merged), additional)
        };
        let additional = block.bits.get().count_ones() as usize;
        let (target, additional) = match domain {
            Domain::Empty => (InsertionTarget::Inline(domain, block), additional),
            Domain::Inline(previous) => {
                let previous = *previous;
                if previous.index == block.index {
                    (
                        InsertionTarget::Inline(
                            domain,
                            Block {
                                index: block.index,
                                bits: previous.bits | block.bits,
                            },
                        ),
                        (block.bits.get() & !previous.bits.get()).count_ones() as usize,
                    )
                } else {
                    (
                        InsertionTarget::Promote(domain, [previous, block]),
                        additional,
                    )
                }
            }
            Domain::Sparse(blocks) => match blocks.entry(block.index) {
                Entry::Occupied(entry) => merge(entry.into_mut(), block.bits),
                Entry::Vacant(entry) => (InsertionTarget::Vacant(entry, block.bits), additional),
            },
        };
        Self {
            target,
            additional,
            cardinality,
            allowance,
        }
    }

    fn admit(self) -> Result<(), LocalInterleavingRefusal> {
        if self.additional > self.allowance.remaining {
            return Err(LocalInterleavingRefusal::Budget(
                LocalInterleavingBudget::CandidateBudgetExceeded,
            ));
        }
        self.allowance.remaining -= self.additional;
        self.cardinality.0 += self.additional;
        match self.target {
            InsertionTarget::Inline(target, block) => *target = Domain::Inline(block),
            InsertionTarget::Merge(target, bits) => *target = bits,
            InsertionTarget::Promote(target, blocks) => {
                *target = Domain::Sparse(Box::new(
                    blocks
                        .into_iter()
                        .map(|block| (block.index, block.bits))
                        .collect(),
                ));
            }
            InsertionTarget::Vacant(entry, bits) => {
                entry.insert(bits);
            }
        }
        Ok(())
    }
}

impl PendingCandidates {
    pub(super) fn new(plan: &Plan<'_, '_>) -> Self {
        Self {
            words: plan
                .source()
                .iter()
                .map(|_| PendingWord {
                    domain: Domain::Empty,
                    cardinality: WordCardinality(0),
                    missing: Missingness::NotObserved,
                })
                .collect(),
            allowance: CandidateAllowance {
                remaining: CANDIDATES,
            },
        }
    }

    pub(super) fn observe_missing(&mut self, word: usize) {
        self.words[word].missing = Missingness::Witnessed;
    }

    pub(super) fn observe(
        &mut self,
        word: usize,
        first: usize,
        bits: u64,
    ) -> Result<(), LocalInterleavingRefusal> {
        // Viable profile windows need not begin at a global 64-slot boundary.
        // Normalize their mask into at most two global blocks before union.
        let index = first / 64;
        let shift = first % 64;
        self.insert(word, ReferenceBlock(index), bits << shift)?;
        if shift != 0 {
            self.insert(word, ReferenceBlock(index + 1), bits >> (64 - shift))?;
        }
        Ok(())
    }

    fn insert(
        &mut self,
        word: usize,
        index: ReferenceBlock,
        bits: u64,
    ) -> Result<(), LocalInterleavingRefusal> {
        let Some(bits) = NonZeroU64::new(bits) else {
            return Ok(());
        };
        PreparedInsertion::prepare(self, word, Block { index, bits }).admit()
    }

    pub(super) fn finish(self) -> Vec<WordEvidence> {
        self.words.into_iter().map(PendingWord::finish).collect()
    }
}

impl PendingWord {
    fn finish(self) -> WordEvidence {
        let mut candidates = Vec::with_capacity(self.cardinality.0);
        let mut append = |index: ReferenceBlock, bits: NonZeroU64| {
            let mut bits = bits.get();
            while bits != 0 {
                let bit = 63 - bits.leading_zeros() as usize;
                bits &= !(1u64 << bit);
                candidates.push(index.0 * 64 + bit);
            }
        };
        match self.domain {
            Domain::Empty => {}
            Domain::Inline(block) => append(block.index, block.bits),
            Domain::Sparse(blocks) => {
                for (index, bits) in (*blocks).into_iter().rev() {
                    append(index, bits);
                }
            }
        }
        WordEvidence {
            candidates,
            can_be_missing: match self.missing {
                Missingness::NotObserved => false,
                Missingness::Witnessed => true,
            },
        }
    }
}
