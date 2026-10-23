//! Profiles for admitted full or score-viable rows. Full rows retain their
//! complete zero-origin domain; viable allowances come from the owned feasible
//! score. Discarded prefixes cannot become viable descendants, and every
//! optimal path survives in both directions.

use super::*;
use crate::dp_align::packed_literal::{CutBoundary, merge_shifted, observe_edge_cuts};
use crate::dp_align::zero_padded::{PackedBits, WindowStart};

#[derive(Clone)]
pub(super) enum Row {
    Unreachable,
    Viable(Profile),
}

#[derive(Clone)]
pub(super) struct Profile {
    first: usize,
    base: usize,
    /// The score increments over `first..last`: its length is the domain's
    /// width, so `last` is stated once, here ([`Self::last`]).
    bits: PackedBits,
    /// Score at `last`, bound to these owned bits by the row producer.
    maximum: usize,
}

impl Profile {
    /// The only root constructor: a constant-zero score function.
    pub(super) fn zero(last: usize) -> Self {
        Self {
            first: 0,
            base: 0,
            bits: PackedBits::zeros(last),
            maximum: 0,
        }
    }
    pub(super) fn first(&self) -> usize {
        self.first
    }
    pub(super) fn last(&self) -> usize {
        self.first + self.bits.length()
    }
    pub(super) fn base(&self) -> usize {
        self.base
    }
    pub(super) fn bits(&self) -> &[u64] {
        self.bits.words()
    }
    pub(super) fn total(&self) -> usize {
        self.maximum
    }
    /// The score at reference position `k`, or `None` outside the row's
    /// domain: the one reader of a forward score, so no caller reads past
    /// the domain on a range test of its own.
    pub(super) fn score(&self, k: usize) -> Option<usize> {
        if k < self.first || k > self.last() {
            return None;
        }
        let at = k - self.first;
        Some(
            self.base
                + self.bits.words()[..at / 64]
                    .iter()
                    .map(|bits| bits.count_ones() as usize)
                    .sum::<usize>()
                + if at.is_multiple_of(64) {
                    0
                } else {
                    (self.bits.words()[at / 64] & ((1u64 << (at % 64)) - 1)).count_ones() as usize
                },
        )
    }

    pub(super) fn advance<C: PackedControl + ?Sized>(
        mut self,
        reference: &LiteralReference<'_>,
        word: &str,
        required: usize,
        slack: usize,
        length: usize,
        control: &C,
    ) -> Result<Row, C::Error> {
        let last = (self.last() + 1).min(length);
        let span = last - self.first;
        // Fused sparse advancement, total-score extraction, and left trimming.
        // Cloning the predecessor is separately charged by the frontier.
        control.charge(3usize.saturating_mul(span.div_ceil(64).saturating_add(1)))?;
        self.bits.resize(span);
        let first = self.first;
        self.bits
            .edit(|words, span| reference.advance_window(words, word, first, span, control))?;
        // This population pass was already required for right trimming. Retain
        // its result so suffix construction never repeats it. Both trims keep
        // this score: right trimming drops a zero increment; left trimming
        // transfers the first increment into the new base.
        self.maximum = self.base
            + self
                .bits
                .words()
                .iter()
                .map(|bits| bits.count_ones() as usize)
                .sum::<usize>();
        // At most the newly appended reference cell can violate the reference
        // gap allowance. It has no score increment when it must be removed.
        let last = match last - self.total() > slack {
            true => last - 1,
            false => last,
        };
        let span = last - self.first;
        self.bits.resize(span);
        if self.base < required {
            // A word adds at most one match, so a row is at most one short.
            if self.base + 1 != required {
                return Err(
                    control.refuse(LocalInterleavingInconsistency::ProfileBaseSkipped {
                        base: self.base,
                        required,
                    }),
                );
            }
            let first = self
                .bits
                .words()
                .iter()
                .enumerate()
                .find(|(_, bits)| **bits != 0)
                .map(|(block, bits)| block * 64 + bits.trailing_zeros() as usize + 1);
            let Some(drop) = first else {
                return Ok(Row::Unreachable);
            };
            self.bits = PackedBits::window(self.bits.padded(), WindowStart::At(drop), span - drop);
            self.first += drop;
            self.base = required;
        }
        Ok(Row::Viable(self))
    }

    pub(super) fn merge<C: PackedControl + ?Sized>(
        self,
        other: Self,
        control: &C,
    ) -> Result<Self, C::Error> {
        if self.base != other.base {
            return Err(
                control.refuse(LocalInterleavingInconsistency::ProfileBasesDiffer {
                    left: self.base,
                    right: other.base,
                }),
            );
        }
        // A pointwise maximum's terminal score is the maximum of both terminal
        // scores, including differently shifted domains with constant tails.
        let maximum = self.maximum.max(other.maximum);
        let first = self.first.min(other.first);
        let last = self.last().max(other.last());
        let width = (last - first).div_ceil(64);
        // Reserve output storage and a fused aligned-input maximum pass.
        control.charge(2usize.saturating_mul(width.saturating_add(1)))?;
        let bits = merge_shifted(
            &self.bits,
            self.first - first,
            &other.bits,
            other.first - first,
            last - first,
            control,
        )?;
        Ok(Self {
            first,
            base: self.base,
            bits,
            maximum,
        })
    }
}

pub(super) struct Suffix {
    first: usize,
    total: usize,
    /// The suffix score declines over `first..last`: its length is the
    /// domain's width ([`Self::last`]).
    bits: PackedBits,
}

/// Complete paired numeric observation. The graph producer still binds the
/// actual source edge and equality relation before recording any candidate.
pub(super) struct EdgeCuts {
    pub(super) can_be_missing: bool,
    pub(super) matches: MatchCuts,
}

pub(super) enum MatchCuts {
    Empty,
    Domain { first: usize, bits: Vec<u64> },
}
impl Suffix {
    pub(super) fn of<C: PackedControl + ?Sized>(
        profile: &Profile,
        length: usize,
        control: &C,
    ) -> Result<Self, C::Error> {
        // Allocation and reversal only. The producing row already owns its
        // maximum; no population scan is performed or charged here.
        control.charge(2usize.saturating_mul(profile.bits.words().len().saturating_add(1)))?;
        Ok(Self {
            first: length - profile.last(),
            total: profile.total(),
            bits: reverse_bits(&profile.bits),
        })
    }
    fn last(&self) -> usize {
        self.first + self.bits.length()
    }

    fn score(&self, k: usize) -> usize {
        let at = k - self.first;
        self.total
            - self.bits.words()[..at / 64]
                .iter()
                .map(|bits| bits.count_ones() as usize)
                .sum::<usize>()
            - if at.is_multiple_of(64) {
                0
            } else {
                (self.bits.words()[at / 64] & ((1u64 << (at % 64)) - 1)).count_ones() as usize
            }
    }
    /// Both edge observations share actual score origins and domain slicing.
    /// The one matching-only cut immediately before the suffix domain is
    /// checked separately; it must never grant deletion/missingness evidence.
    pub(super) fn observe_edge<C: PackedControl + ?Sized>(
        &self,
        forward: &Profile,
        optimum: usize,
        control: &C,
    ) -> Result<EdgeCuts, C::Error> {
        let mut earlier = None;
        if let Some(k) = self.first.checked_sub(1)
            && let Some(score) = forward.score(k)
        {
            control.charge((k - forward.first).div_ceil(64).saturating_add(2))?;
            if score + 1 + self.total == optimum {
                earlier = Some(k);
            }
        }
        let first = forward.first.max(self.first);
        let last = forward.last().min(self.last());
        if first > last {
            return Ok(EdgeCuts {
                can_be_missing: false,
                matches: match earlier {
                    Some(first) => MatchCuts::Domain {
                        first,
                        bits: vec![1],
                    },
                    None => MatchCuts::Empty,
                },
            });
        }
        let span = last - first;
        // Two slices, output allocation and paired numeric scan. Origins are
        // paid outside the window, and the matching-only boundary adds a shift.
        control.charge(
            (if earlier.is_some() { 5usize } else { 4usize })
                .saturating_mul(span.div_ceil(64).saturating_add(1))
                .saturating_add((first - forward.first).div_ceil(64))
                .saturating_add((first - self.first).div_ceil(64)),
        )?;
        let a = PackedBits::window(
            forward.bits.padded(),
            WindowStart::At(first - forward.first),
            span,
        );
        let b = PackedBits::window(
            self.bits.padded(),
            WindowStart::At(first - self.first),
            span + 1,
        );
        let Some(forward_score) = forward.score(first) else {
            return Err(control.refuse(
                LocalInterleavingInconsistency::ProfileScoreOutsideDomain { reference: first },
            ));
        };
        let target = optimum as i64 - forward_score as i64 - self.score(first) as i64;
        let boundary = if last < self.last() {
            CutBoundary::MatchingAllowed
        } else {
            CutBoundary::DeletionOnly
        };
        let observed = observe_edge_cuts(&a, &b, span, target, boundary, control)?;
        let mut matched = observed.matched;
        let first = if let Some(k) = earlier {
            if k + 1 != first {
                return Err(control
                    .refuse(LocalInterleavingInconsistency::EdgeCutGap { earlier: k, first }));
            }
            matched.resize((span + 2).div_ceil(64), 0);
            let mut carry = 0;
            for (block, bits) in matched.iter_mut().enumerate() {
                if block % 1024 == 0 {
                    control.check()?;
                }
                let next = *bits >> 63;
                *bits = (*bits << 1) | carry;
                carry = next;
            }
            matched[0] |= 1;
            k
        } else {
            first
        };
        Ok(EdgeCuts {
            can_be_missing: observed.can_be_missing,
            matches: MatchCuts::Domain {
                first,
                bits: matched,
            },
        })
    }
}

#[cfg(test)]
mod maximum_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn interruption_after_sparse_advancement_mutates_blocks_returns_no_owned_row() {
        struct Cancel(Cell<usize>);
        impl WorkControl for Cancel {
            type Error = &'static str;
            fn check(&self) -> Result<(), Self::Error> {
                self.0.set(self.0.get() + 1);
                if self.0.get() == 3 {
                    Err("cancelled")
                } else {
                    Ok(())
                }
            }
        }
        impl PackedControl for Cancel {
            fn charge(&self, _: usize) -> Result<(), Self::Error> {
                Ok(())
            }
            fn refuse(&self, _: LocalInterleavingInconsistency) -> Self::Error {
                "inconsistent"
            }
        }
        let words = vec!["a".to_owned(); 65_537];
        let reference = LiteralReference::observe(
            &words,
            crate::dp_align::LiteralMatchMode::Exact,
            ReferenceOrder::Forward,
            &Uninterrupted,
        )
        .unwrap();
        let checkpoint = Profile::zero(words.len());
        let result = checkpoint.clone().advance(
            &reference,
            "a",
            0,
            words.len(),
            words.len(),
            &Cancel(Cell::new(0)),
        );
        assert!(matches!(result, Err("cancelled")));
        assert!(checkpoint.bits.words().iter().all(|&bits| bits == 0));
        let Row::Viable(completed) = checkpoint
            .advance(&reference, "a", 0, words.len(), words.len(), &Uninterrupted)
            .unwrap()
        else {
            panic!("reachable advancement");
        };
        assert_eq!(
            completed.score(1),
            Some(1),
            "earlier blocks actually change"
        );
        assert_eq!(completed.score(words.len()), Some(1));
    }
}
