//! Shared Allison-Dix row kernel. Checkpoints retain their exact reference
//! owner; restoring one never accepts replacement masks, policy or width.

use super::{LiteralMatchMode, MatchMode, WorkControl};
use std::collections::HashMap;

mod arithmetic;
pub(super) use arithmetic::merge_shifted;
pub(super) use arithmetic::reverse_bits;
pub(super) use arithmetic::{CutBoundary, observe_edge_cuts};

pub(super) const BIT_WORDS: usize =
    std::mem::size_of::<u64>().div_ceil(std::mem::size_of::<usize>());

#[derive(Clone, Copy)]
pub(super) enum ReferenceOrder {
    Forward,
    Reverse,
}

pub(super) fn policy(mode: MatchMode) -> Option<LiteralMatchMode> {
    match mode {
        MatchMode::Exact => Some(LiteralMatchMode::Exact),
        MatchMode::CaseInsensitive => Some(LiteralMatchMode::AsciiInsensitive),
        MatchMode::Fuzzy { .. } => None,
    }
}

pub(super) fn key(word: &str, mode: LiteralMatchMode) -> String {
    match mode {
        LiteralMatchMode::Exact => word.to_owned(),
        LiteralMatchMode::AsciiInsensitive => word.to_ascii_lowercase(),
    }
}

pub(super) struct LiteralReference<'input> {
    reference: &'input [String],
    mode: LiteralMatchMode,
    masks: HashMap<String, Vec<(usize, u64)>>,
}

impl<'input> LiteralReference<'input> {
    /// Callers admit their complete workspace before constructing masks.
    pub(super) fn observe<C: WorkControl + ?Sized>(
        reference: &'input [String],
        mode: LiteralMatchMode,
        order: ReferenceOrder,
        control: &C,
    ) -> Result<Self, C::Error> {
        control.check()?;
        let mut masks: HashMap<String, Vec<(usize, u64)>> = HashMap::new();
        for j in 0..reference.len() {
            if j % 1024 == 0 {
                control.check()?;
            }
            let original = match order {
                ReferenceOrder::Forward => j,
                ReferenceOrder::Reverse => reference.len() - 1 - j,
            };
            let postings = masks.entry(key(&reference[original], mode)).or_default();
            let block = j / 64;
            let bit = 1u64 << (j % 64);
            if let Some((last, bits)) = postings.last_mut()
                && *last == block
            {
                *bits |= bit;
            } else {
                postings.push((block, bit));
            }
        }
        Ok(Self {
            reference,
            mode,
            masks,
        })
    }

    pub(super) fn start(&self) -> PackedRow<'_, 'input> {
        let blocks = self.reference.len().div_ceil(64);
        PackedRow {
            owner: self,
            bits: vec![0; blocks],
        }
    }

    pub(super) fn postings(&self, word: &str) -> Option<&[(usize, u64)]> {
        self.masks.get(&key(word, self.mode)).map(Vec::as_slice)
    }

    pub(super) fn mask_window(&self, word: &str, first: usize, span: usize) -> Vec<u64> {
        let mut mask = vec![0; span.div_ceil(64)];
        if span == 0 {
            return mask;
        }
        if let Some(postings) = self.postings(word) {
            let begin = postings.partition_point(|&(block, _)| block < first / 64);
            let end = postings.partition_point(|&(block, _)| block < (first + span).div_ceil(64));
            for &(block, bits) in &postings[begin..end] {
                let absolute = block * 64;
                if absolute < first {
                    mask[0] |= bits >> (first - absolute);
                } else {
                    let relative = absolute - first;
                    mask[relative / 64] |= bits << (relative % 64);
                    if !relative.is_multiple_of(64) && relative / 64 + 1 < mask.len() {
                        mask[relative / 64 + 1] |= bits >> (64 - relative % 64);
                    }
                }
            }
        }
        if !span.is_multiple_of(64)
            && let Some(last) = mask.last_mut()
        {
            *last &= (1u64 << (span % 64)) - 1;
        }
        mask
    }

    pub(super) fn advance<C: WorkControl + ?Sized>(
        &self,
        bits: &mut [u64],
        word: &str,
        control: &C,
    ) -> Result<(), C::Error> {
        arithmetic::extend_sparse(bits, self.postings(word).unwrap_or(&[]), control)
    }

    pub(super) fn advance_window<C: WorkControl + ?Sized>(
        &self,
        bits: &mut [u64],
        word: &str,
        first: usize,
        span: usize,
        control: &C,
    ) -> Result<(), C::Error> {
        arithmetic::extend_window_sparse(
            bits,
            self.postings(word).unwrap_or(&[]),
            first,
            span,
            control,
        )
    }
}

pub(super) struct PackedRow<'owner, 'input> {
    owner: &'owner LiteralReference<'input>,
    bits: Vec<u64>,
}

pub(super) struct PackedCheckpoint<'owner, 'input> {
    owner: &'owner LiteralReference<'input>,
    bits: Vec<u64>,
}

impl<'owner, 'input> PackedCheckpoint<'owner, 'input> {
    pub(super) fn resume(&self) -> PackedRow<'owner, 'input> {
        PackedRow {
            owner: self.owner,
            bits: self.bits.clone(),
        }
    }
}

impl<'owner, 'input> PackedRow<'owner, 'input> {
    pub(super) fn checkpoint(&self) -> PackedCheckpoint<'owner, 'input> {
        PackedCheckpoint {
            owner: self.owner,
            bits: self.bits.clone(),
        }
    }

    pub(super) fn extend<C: WorkControl + ?Sized>(
        mut self,
        word: &str,
        control: &C,
    ) -> Result<Self, C::Error> {
        control.check()?;
        self.owner.advance(&mut self.bits, word, control)?;
        Ok(self)
    }

    pub(super) fn score(&self) -> usize {
        self.bits
            .iter()
            .map(|bits| bits.count_ones() as usize)
            .sum()
    }

    pub(super) fn prefixes(&self) -> PrefixScores {
        let mut before = 0;
        let blocks = self
            .bits
            .iter()
            .map(|&bits| {
                let block = PrefixBlock { before, bits };
                before += bits.count_ones() as usize;
                block
            })
            .collect();
        PrefixScores {
            blocks,
            length: self.owner.reference.len(),
            total: before,
        }
    }
}

struct PrefixBlock {
    before: usize,
    bits: u64,
}

pub(super) struct PrefixScores {
    blocks: Vec<PrefixBlock>,
    length: usize,
    total: usize,
}

impl PrefixScores {
    pub(super) const BLOCK_WORDS: usize =
        std::mem::size_of::<PrefixBlock>().div_ceil(std::mem::size_of::<usize>());

    pub(super) fn before(&self, prefix: usize) -> usize {
        debug_assert!(prefix <= self.length);
        if prefix == self.length {
            return self.total;
        }
        let block = &self.blocks[prefix / 64];
        let mask = (1u64 << (prefix % 64)) - 1;
        block.before + (block.bits & mask).count_ones() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dp_align::Uninterrupted;

    #[test]
    fn every_packed_prefix_and_resumed_row_equals_independent_scalar_scores() {
        for n in [0, 1, 63, 64, 65, 127, 128, 129, 257] {
            let reference: Vec<_> = (0..n)
                .map(|i| ["a", "A", "é", "É", "b"][i % 5].to_owned())
                .collect();
            for mode in [LiteralMatchMode::Exact, LiteralMatchMode::AsciiInsensitive] {
                for order in [ReferenceOrder::Forward, ReferenceOrder::Reverse] {
                    let owner =
                        LiteralReference::observe(&reference, mode, order, &Uninterrupted).unwrap();
                    let mut packed = owner.start();
                    let mut scalar = vec![0usize; n + 1];
                    for i in 0..129 {
                        let word = ["A", "b", "é", "a"][i % 4];
                        let checkpoint = packed.checkpoint();
                        let mut resumed = checkpoint.resume();
                        packed = packed.extend(word, &Uninterrupted).unwrap();
                        resumed = resumed.extend(word, &Uninterrupted).unwrap();
                        let mut diagonal = 0;
                        for j in 0..n {
                            let old = scalar[j + 1];
                            let original = match order {
                                ReferenceOrder::Forward => j,
                                ReferenceOrder::Reverse => n - j - 1,
                            };
                            let equal = match mode {
                                LiteralMatchMode::Exact => word == reference[original],
                                LiteralMatchMode::AsciiInsensitive => {
                                    word.eq_ignore_ascii_case(&reference[original])
                                }
                            };
                            scalar[j + 1] = old.max(scalar[j]).max(diagonal + usize::from(equal));
                            diagonal = old;
                        }
                        let scores = packed.prefixes();
                        let restored = resumed.prefixes();
                        for (prefix, &expected) in scalar.iter().enumerate() {
                            assert_eq!(scores.before(prefix), expected);
                            assert_eq!(restored.before(prefix), expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn interrupted_extension_burns_partial_state_and_keeps_the_prior_checkpoint_valid() {
        struct Checks(std::cell::Cell<usize>);
        impl WorkControl for Checks {
            type Error = &'static str;
            fn check(&self) -> Result<(), Self::Error> {
                let next = self.0.get() + 1;
                self.0.set(next);
                if next == 3 { Err("cancelled") } else { Ok(()) }
            }
        }
        // The third check is at block 1024, after preceding blocks changed.
        let reference = vec!["same".to_owned(); 65_537];
        let owner = LiteralReference::observe(
            &reference,
            LiteralMatchMode::Exact,
            ReferenceOrder::Forward,
            &Uninterrupted,
        )
        .unwrap();
        let row = owner.start().extend("same", &Uninterrupted).unwrap();
        let checkpoint = row.checkpoint();
        assert!(matches!(
            row.extend("same", &Checks(std::cell::Cell::new(0))),
            Err("cancelled")
        ));
        // The failed transition returns no row that could be reused.
        assert_eq!(checkpoint.resume().score(), 1);
    }
}
