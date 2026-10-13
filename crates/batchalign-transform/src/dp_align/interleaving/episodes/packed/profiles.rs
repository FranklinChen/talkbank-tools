//! Profiles for admitted full or score-viable rows. Full rows retain their
//! complete zero-origin domain; viable allowances come from the owned feasible
//! score. Discarded prefixes cannot become viable descendants, and every
//! optimal path survives in both directions.

use super::*;
use crate::dp_align::packed_literal::{CutBoundary, merge_shifted, observe_edge_cuts};

#[derive(Clone)]
pub(super) enum Row {
    Unreachable,
    Viable(Profile),
}

#[derive(Clone)]
pub(super) struct Profile {
    first: usize,
    last: usize,
    base: usize,
    bits: Vec<u64>,
    /// Score at `last`, bound to these owned bits by the row producer.
    maximum: usize,
}

pub(super) fn slice(bits: &[u64], offset: usize, span: usize) -> Vec<u64> {
    let mut result = Vec::with_capacity(span.div_ceil(64));
    for block in 0..span.div_ceil(64) {
        let at = offset / 64 + block;
        let a = bits.get(at).copied().unwrap_or(0);
        let value = if offset.is_multiple_of(64) {
            a
        } else {
            (a >> (offset % 64)) | (bits.get(at + 1).copied().unwrap_or(0) << (64 - offset % 64))
        };
        result.push(value);
    }
    if !span.is_multiple_of(64)
        && let Some(last) = result.last_mut()
    {
        *last &= (1u64 << (span % 64)) - 1;
    }
    result
}

impl Profile {
    /// The only root constructor: a constant-zero score function.
    pub(super) fn zero(last: usize) -> Self {
        Self {
            first: 0,
            last,
            base: 0,
            bits: vec![0; last.div_ceil(64)],
            maximum: 0,
        }
    }
    pub(super) fn first(&self) -> usize {
        self.first
    }
    pub(super) fn last(&self) -> usize {
        self.last
    }
    pub(super) fn base(&self) -> usize {
        self.base
    }
    pub(super) fn bits(&self) -> &[u64] {
        &self.bits
    }
    pub(super) fn total(&self) -> usize {
        self.maximum
    }
    pub(super) fn score(&self, k: usize) -> Option<usize> {
        if k < self.first || k > self.last {
            return None;
        }
        Some(self.score_inside(k))
    }
    pub(super) fn score_inside(&self, k: usize) -> usize {
        debug_assert!(self.first <= k && k <= self.last);
        let at = k - self.first;
        self.base
            + self.bits[..at / 64]
                .iter()
                .map(|bits| bits.count_ones() as usize)
                .sum::<usize>()
            + if at.is_multiple_of(64) {
                0
            } else {
                (self.bits[at / 64] & ((1u64 << (at % 64)) - 1)).count_ones() as usize
            }
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
        let last = (self.last + 1).min(length);
        let span = last - self.first;
        // Fused sparse advancement, total-score extraction, and left trimming.
        // Cloning the predecessor is separately charged by the frontier.
        control.charge(3usize.saturating_mul(span.div_ceil(64).saturating_add(1)))?;
        self.bits.resize(span.div_ceil(64), 0);
        reference.advance_window(&mut self.bits, word, self.first, span, control)?;
        self.last = last;
        // This population pass was already required for right trimming. Retain
        // its result so suffix construction never repeats it. Both trims keep
        // this score: right trimming drops a zero increment; left trimming
        // transfers the first increment into the new base.
        self.maximum = self.base
            + self
                .bits
                .iter()
                .map(|bits| bits.count_ones() as usize)
                .sum::<usize>();
        // At most the newly appended reference cell can violate the reference
        // gap allowance. It has no score increment when it must be removed.
        if self.last - self.total() > slack {
            self.last -= 1;
        }
        let span = self.last - self.first;
        self.bits.truncate(span.div_ceil(64));
        if !span.is_multiple_of(64)
            && let Some(last) = self.bits.last_mut()
        {
            *last &= (1u64 << (span % 64)) - 1;
        }
        if self.base < required {
            debug_assert_eq!(self.base + 1, required);
            let first = self
                .bits
                .iter()
                .enumerate()
                .find(|(_, bits)| **bits != 0)
                .map(|(block, bits)| block * 64 + bits.trailing_zeros() as usize + 1);
            let Some(drop) = first else {
                return Ok(Row::Unreachable);
            };
            self.bits = slice(&self.bits, drop, span - drop);
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
        debug_assert_eq!(self.base, other.base);
        // A pointwise maximum's terminal score is the maximum of both terminal
        // scores, including differently shifted domains with constant tails.
        let maximum = self.maximum.max(other.maximum);
        let first = self.first.min(other.first);
        let last = self.last.max(other.last);
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
            last,
            base: self.base,
            bits,
            maximum,
        })
    }
}

pub(super) struct Suffix {
    first: usize,
    last: usize,
    total: usize,
    bits: Vec<u64>,
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
        control.charge(2usize.saturating_mul(profile.bits.len().saturating_add(1)))?;
        Ok(Self {
            first: length - profile.last,
            last: length - profile.first,
            total: profile.total(),
            bits: reverse_bits(&profile.bits, profile.last - profile.first),
        })
    }
    fn score(&self, k: usize) -> usize {
        let at = k - self.first;
        self.total
            - self.bits[..at / 64]
                .iter()
                .map(|bits| bits.count_ones() as usize)
                .sum::<usize>()
            - if at.is_multiple_of(64) {
                0
            } else {
                (self.bits[at / 64] & ((1u64 << (at % 64)) - 1)).count_ones() as usize
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
            && forward.first <= k
            && k <= forward.last
        {
            control.charge((k - forward.first).div_ceil(64).saturating_add(2))?;
            if forward.score_inside(k) + 1 + self.total == optimum {
                earlier = Some(k);
            }
        }
        let first = forward.first.max(self.first);
        let last = forward.last.min(self.last);
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
        let a = slice(&forward.bits, first - forward.first, span);
        let b = slice(&self.bits, first - self.first, span + 1);
        let target = optimum as i64 - forward.score_inside(first) as i64 - self.score(first) as i64;
        let boundary = if last < self.last {
            CutBoundary::MatchingAllowed
        } else {
            CutBoundary::DeletionOnly
        };
        let observed = observe_edge_cuts(&a, &b, span, target, boundary, control)?;
        let mut matched = observed.matched;
        let first = if let Some(k) = earlier {
            debug_assert_eq!(k + 1, first);
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
        assert!(checkpoint.bits.iter().all(|&bits| bits == 0));
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
