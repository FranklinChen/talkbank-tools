//! Paired numeric edge observation. No graph or lexical certificate is issued.

use super::{HIGH, LOW, ONES, PACK, RANKS, WorkControl};
use crate::dp_align::zero_padded::{PackedBits, WindowStart};

#[derive(Clone, Copy)]
pub(in crate::dp_align) enum CutBoundary {
    MatchingAllowed,
    DeletionOnly,
}

pub(in crate::dp_align) struct NumericEdgeCuts {
    pub(in crate::dp_align) can_be_missing: bool,
    pub(in crate::dp_align) matched: Vec<u64>,
}

fn equal_byte(a: usize, b: usize, delta: i64) -> u64 {
    if !(-8..=8).contains(&delta) {
        return 0;
    }
    let ranks = RANKS[a] + ONES * 16 - RANKS[b];
    let unequal = ranks ^ (ONES * (16 + delta) as u64);
    let equal = !(((unequal & LOW) + LOW) | unequal | LOW) & HIGH;
    (equal >> 7).wrapping_mul(PACK) >> 56
}

/// F+B=L establishes possible deletion. With suffix decline b(k), matching
/// requires F+B=L when b(k)=1, or F+B=L-1 when b(k)=0. Both share one scan.
pub(in crate::dp_align) fn observe_edge_cuts<C: WorkControl + ?Sized>(
    forward: &PackedBits,
    declines: &PackedBits,
    span: usize,
    mut delta: i64,
    boundary: CutBoundary,
    control: &C,
) -> Result<NumericEdgeCuts, C::Error> {
    control.check()?;
    let declines = declines.padded();
    let mut matched = vec![0u64; (span + 1).div_ceil(64)];
    let mut can_be_missing = delta == 0;
    let first_decline = declines.window_word(WindowStart::At(0), 0) & 1;
    matched[0] = u64::from(delta == 1 - first_decline as i64);
    for (block, &a) in forward.words().iter().enumerate() {
        if block % 1024 == 0 {
            control.check()?;
        }
        let b = declines.window_word(WindowStart::At(0), block);
        // The declines one bit on: b(k + 1) beside each cut k.
        let following = declines.window_word(WindowStart::At(1), block);
        for byte in 0..8 {
            let start = block * 64 + byte * 8;
            if start >= span {
                break;
            }
            let ab = ((a >> (8 * byte)) & 255) as usize;
            let bb = ((b >> (8 * byte)) & 255) as usize;
            let width = (span - start).min(8);
            let valid = (1u64 << width) - 1;
            let deletion = equal_byte(ab, bb, delta) & valid;
            can_be_missing |= deletion != 0;
            let next_declines = (following >> (8 * byte)) & valid;
            let matches = (deletion & next_declines)
                | (equal_byte(ab, bb, delta - 1) & !next_declines & valid);
            let cut = start + 1;
            matched[cut / 64] |= matches << (cut % 64);
            if cut % 64 > 56 && cut / 64 + 1 < matched.len() {
                matched[cut / 64 + 1] |= matches >> (64 - cut % 64);
            }
            delta -= ab.count_ones() as i64 - bb.count_ones() as i64;
        }
    }
    if matches!(boundary, CutBoundary::DeletionOnly) {
        matched[span / 64] &= !(1 << (span % 64));
    }
    Ok(NumericEdgeCuts {
        can_be_missing,
        matched,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dp_align::control::Uninterrupted;

    fn score(bits: &[u64], k: usize) -> i64 {
        (0..k)
            .filter(|&i| bits[i / 64] & (1 << (i % 64)) != 0)
            .count() as i64
    }

    #[test]
    fn paired_edge_observations_equal_every_scalar_cut_and_signed_origin() {
        for span in [0usize, 1, 7, 8, 9, 63, 64, 65, 129] {
            for seed in 0..8 {
                let mut a = vec![0u64; span.div_ceil(64)];
                let mut b = vec![0u64; (span + 1).div_ceil(64)];
                for k in 0..=span {
                    if k < span && (k * 17 + seed) % 19 < 9 {
                        a[k / 64] |= 1 << (k % 64);
                    }
                    if (k * 3 + seed) % 11 < 5 {
                        b[k / 64] |= 1 << (k % 64);
                    }
                }
                for delta in -(span as i64 + 1)..=span as i64 + 1 {
                    for boundary in [CutBoundary::MatchingAllowed, CutBoundary::DeletionOnly] {
                        let observed = observe_edge_cuts(
                            &PackedBits::from_words(a.clone(), span).unwrap(),
                            &PackedBits::from_words(b.clone(), span + 1).unwrap(),
                            span,
                            delta,
                            boundary,
                            &Uninterrupted,
                        )
                        .unwrap();
                        assert_eq!(
                            observed.can_be_missing,
                            (0..=span).any(|k| score(&a, k) - score(&b, k) == delta)
                        );
                        for k in 0..=span {
                            let allowed =
                                k < span || matches!(boundary, CutBoundary::MatchingAllowed);
                            let decline = ((b[k / 64] >> (k % 64)) & 1) as i64;
                            assert_eq!(
                                observed.matched[k / 64] & (1 << (k % 64)) != 0,
                                allowed && score(&a, k) - score(&b, k) + 1 - decline == delta
                            );
                        }
                    }
                }
            }
        }
    }
}
