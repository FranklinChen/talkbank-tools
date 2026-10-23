//! Packed numeric score arithmetic. These functions issue no path or matching
//! certificate; producers own geometry, policy, complete work and graph edges.

use super::WorkControl;
use crate::dp_align::zero_padded::{PackedBits, WindowStart};
mod edge;
pub(in crate::dp_align) use edge::{CutBoundary, observe_edge_cuts};
const ONES: u64 = 0x0101_0101_0101_0101;
const HIGH: u64 = 0x8080_8080_8080_8080;
const LOW: u64 = 0x7f7f_7f7f_7f7f_7f7f;
const PACK: u64 = 0x0102_0408_1020_4080;

const fn ranks() -> [u64; 256] {
    let mut table = [0; 256];
    let mut bits = 0;
    while bits < 256 {
        let mut rank = 0u64;
        let mut byte = 0;
        while byte < 8 {
            rank += ((bits >> byte) & 1) as u64;
            table[bits] |= rank << (8 * byte);
            byte += 1;
        }
        bits += 1;
    }
    table
}
const RANKS: [u64; 256] = ranks();

fn step(old: u64, matches: u64, carry: &mut u64, borrow: &mut bool) -> u64 {
    let x = old | matches;
    let shifted = (old << 1) | *carry;
    *carry = old >> 63;
    let (difference, first) = x.overflowing_sub(shifted);
    let (difference, second) = difference.overflowing_sub(u64::from(*borrow));
    *borrow = first || second;
    x & !difference
}

pub(in crate::dp_align) fn extend_sparse<C: WorkControl + ?Sized>(
    bits: &mut [u64],
    postings: &[(usize, u64)],
    control: &C,
) -> Result<(), C::Error> {
    let (mut carry, mut borrow, mut cursor) = (1, false, 0);
    for (block, state) in bits.iter_mut().enumerate() {
        if block % 1024 == 0 {
            control.check()?;
        }
        let mask = if postings.get(cursor).is_some_and(|&(at, _)| at == block) {
            let mask = postings[cursor].1;
            cursor += 1;
            mask
        } else {
            0
        };
        *state = step(*state, mask, &mut carry, &mut borrow);
    }
    Ok(())
}

/// Maximum of two zero-origin, monotone unit-increment score functions.
fn maximum_word(a: u64, b: u64, difference: &mut i64) -> u64 {
    let mut maximum = 0;
    for byte in 0..8 {
        let ab = ((a >> (8 * byte)) & 255) as usize;
        let bb = ((b >> (8 * byte)) & 255) as usize;
        let increments = if *difference >= 8 {
            ab as u64
        } else if *difference <= -8 {
            bb as u64
        } else {
            let ap = RANKS[ab] + ONES * (16 + *difference) as u64;
            let bp = RANKS[bb] + ONES * 16;
            let choose = ((ap | HIGH) - bp) & HIGH;
            let mask = (choose >> 7) * 255;
            let scores = (ap & mask) | (bp & !mask);
            let previous = (scores << 8) | (16 + (*difference).max(0)) as u64;
            (scores - previous).wrapping_mul(PACK) >> 56
        };
        maximum |= increments << (8 * byte);
        *difference += ab.count_ones() as i64 - bb.count_ones() as i64;
    }
    maximum
}

#[cfg(test)]
pub(in crate::dp_align) fn merge_bits<C: WorkControl + ?Sized>(
    left: &mut [u64],
    right: &[u64],
    control: &C,
) -> Result<(), C::Error> {
    debug_assert_eq!(left.len(), right.len());
    let mut difference = 0i64;
    for (block, (a, &b)) in left.iter_mut().zip(right).enumerate() {
        if block % 1024 == 0 {
            control.check()?;
        }
        *a = maximum_word(*a, b, &mut difference);
    }
    Ok(())
}

/// Shift and merge in one output pass, without materializing padded parents.
/// The producer binds the common score origin and viable union domain:
/// `left` begins `left_offset` bits into the result, `right` `right_offset`.
pub(in crate::dp_align) fn merge_shifted<C: WorkControl + ?Sized>(
    left: &PackedBits,
    left_offset: usize,
    right: &PackedBits,
    right_offset: usize,
    length: usize,
    control: &C,
) -> Result<PackedBits, C::Error> {
    control.check()?;
    let (left, right) = (left.padded(), right.padded());
    let mut difference = 0;
    PackedBits::try_from_fn(length, |block| {
        if block % 1024 == 0 {
            control.check()?;
        }
        Ok(maximum_word(
            left.window_word(WindowStart::Before(left_offset), block),
            right.window_word(WindowStart::Before(right_offset), block),
            &mut difference,
        ))
    })
}

/// Window-aligned sparse equality masks feed the recurrence directly. Each
/// global posting is consumed once; carry and borrow cross local word blocks.
pub(in crate::dp_align) fn extend_window_sparse<C: WorkControl + ?Sized>(
    bits: &mut [u64],
    postings: &[(usize, u64)],
    first: usize,
    span: usize,
    control: &C,
) -> Result<(), C::Error> {
    control.check()?;
    debug_assert_eq!(bits.len(), span.div_ceil(64));
    let mut cursor = postings.partition_point(|&(block, _)| block < first / 64);
    let take = |cursor: &mut usize, block| {
        if let Some(&(at, mask)) = postings.get(*cursor)
            && at == block
        {
            *cursor += 1;
            mask
        } else {
            0
        }
    };
    let shift = first % 64;
    let mut low = take(&mut cursor, first / 64);
    let (mut carry, mut borrow) = (1, false);
    for (block, state) in bits.iter_mut().enumerate() {
        if block % 1024 == 0 {
            control.check()?;
        }
        let high = take(&mut cursor, first / 64 + block + 1);
        let mut mask = if shift == 0 {
            low
        } else {
            (low >> shift) | (high << (64 - shift))
        };
        if (block + 1) * 64 > span {
            mask &= (1 << (span % 64)) - 1;
        }
        *state = step(*state, mask, &mut carry, &mut borrow);
        low = high;
    }
    Ok(())
}

/// Complete reference-cut equality mask. `backward` is a reverse-reference
/// prefix profile; matching uses B(k+1), deletion uses B(k). Numeric cut masks
/// require actual graph-edge and literal-mask binding in their producer.
#[cfg(test)]
fn score_cut_bits<C: WorkControl + ?Sized>(
    forward: &[u64],
    backward: &PackedBits,
    length: usize,
    target: usize,
    matched: bool,
    control: &C,
) -> Result<Vec<u64>, C::Error> {
    control.check()?;
    let cuts = vec![0; (length + 1).div_ceil(64)];
    if target > length || (matched && (target == 0 || length == 0)) {
        return Ok(cuts);
    }
    // The reverse profile built by hand, independently of `reverse_bits`:
    // the stored words reversed, then read from past their padding.
    let words = backward.words();
    let reversed_words = PackedBits::from_fn(words.len() * 64, |block| {
        words[words.len() - 1 - block].reverse_bits()
    });
    let padding = (64 - length % 64) % 64;
    let mut reverse = PackedBits::window(reversed_words.padded(), WindowStart::At(padding), length);
    let mut base = words
        .iter()
        .map(|bits| bits.count_ones() as i64)
        .sum::<i64>();
    let target = if matched {
        base -= (reverse.padded().window_word(WindowStart::At(0), 0) & 1) as i64;
        reverse = PackedBits::window(reverse.padded(), WindowStart::At(1), length);
        target - 1
    } else {
        target
    };
    let mut cuts = score_cuts_aligned(
        forward,
        &reverse,
        base as usize,
        length,
        target,
        CutKind::Deleted,
        control,
    )?;
    if matched {
        cuts[length / 64] &= !(1 << (length % 64));
    }
    Ok(cuts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dp_align::control::Uninterrupted;

    fn packed(words: &[u64], length: usize) -> PackedBits {
        PackedBits::from_words(words.to_vec(), length).unwrap()
    }

    fn score(bits: &[u64], k: usize) -> usize {
        bits[..k / 64]
            .iter()
            .map(|bits| bits.count_ones() as usize)
            .sum::<usize>()
            + if k.is_multiple_of(64) {
                0
            } else {
                (bits[k / 64] & ((1u64 << (k % 64)) - 1)).count_ones() as usize
            }
    }

    #[test]
    fn packed_merges_and_both_edge_cuts_equal_every_short_scalar_profile() {
        for n in 0..=8 {
            for a in 0..1u64 << n {
                for b in 0..1u64 << n {
                    let ab = if n == 0 { Vec::new() } else { vec![a] };
                    let bb = if n == 0 { Vec::new() } else { vec![b] };
                    let mut merged = ab.clone();
                    merge_bits(&mut merged, &bb, &Uninterrupted).unwrap();
                    for k in 0..=n {
                        assert_eq!(score(&merged, k), score(&ab, k).max(score(&bb, k)));
                    }
                    let aligned = reverse_bits(&packed(&bb, n));
                    let total = score(&bb, n);
                    for target in 0..=n + 1 {
                        for kind in [CutKind::Deleted, CutKind::Matched] {
                            let cuts = score_cuts_aligned(
                                &ab,
                                &aligned,
                                total,
                                n,
                                target,
                                kind,
                                &Uninterrupted,
                            )
                            .unwrap();
                            let matched = matches!(kind, CutKind::Matched);
                            let reference = score_cut_bits(
                                &ab,
                                &packed(&bb, n),
                                n,
                                target,
                                matched,
                                &Uninterrupted,
                            )
                            .unwrap();
                            assert_eq!(cuts, reference);
                            for k in 0..=n {
                                let expected = if matched {
                                    k < n && score(&ab, k) + 1 + score(&bb, n - k - 1) == target
                                } else {
                                    score(&ab, k) + score(&bb, n - k) == target
                                };
                                assert_eq!(cuts[k / 64] & (1 << (k % 64)) != 0, expected);
                            }
                        }
                    }
                }
            }
        }
        let mut merged = vec![1];
        merge_bits(&mut merged, &[2], &Uninterrupted).unwrap();
        insta::assert_debug_snapshot!(merged,@r"
        [
            1,
        ]
        ");
    }

    #[test]
    fn aligned_match_shift_padding_and_crossings_equal_scalar_oracles() {
        for n in [0usize, 1, 7, 8, 9, 63, 64, 65, 127, 128, 129, 257] {
            for seed in 0..12 {
                let mut a = vec![0; n.div_ceil(64)];
                let mut b = a.clone();
                for k in 0..n {
                    if (k * 17 + seed) % 19 < 9 {
                        a[k / 64] |= 1 << (k % 64);
                    }
                    if (k * 3 + k / 11 + seed) % 13 < 7 {
                        b[k / 64] |= 1 << (k % 64);
                    }
                }
                let mut merged = a.clone();
                merge_bits(&mut merged, &b, &Uninterrupted).unwrap();
                for k in 0..=n {
                    assert_eq!(score(&merged, k), score(&a, k).max(score(&b, k)));
                }
                let suffix = reverse_bits(&packed(&b, n));
                let total = score(&b, n);
                for target in [0, n / 2, n, n + 1, usize::MAX] {
                    for kind in [CutKind::Deleted, CutKind::Matched] {
                        let cuts =
                            score_cuts_aligned(&a, &suffix, total, n, target, kind, &Uninterrupted)
                                .unwrap();
                        for k in 0..=n {
                            let expected = if matches!(kind, CutKind::Matched) {
                                k < n && score(&a, k) + 1 + score(&b, n - k - 1) == target
                            } else {
                                score(&a, k) + score(&b, n - k) == target
                            };
                            assert_eq!(cuts[k / 64] & (1 << (k % 64)) != 0, expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn signed_origin_cuts_equal_scalar_differences() {
        for n in [0usize, 1, 8, 63, 64, 65, 129] {
            for seed in 0..12 {
                let mut a = vec![0u64; n.div_ceil(64)];
                let mut b = a.clone();
                for k in 0..n {
                    if (k * 7 + seed) % 11 < 6 {
                        a[k / 64] |= 1 << (k % 64);
                    }
                    if (k * 17 + seed) % 13 < 7 {
                        b[k / 64] |= 1 << (k % 64);
                    }
                }
                for target in -(n as i64 + 1)..=n as i64 + 1 {
                    let cuts = score_difference_cuts(&a, &b, n, target, &Uninterrupted).unwrap();
                    for k in 0..=n {
                        assert_eq!(
                            cuts[k / 64] & (1 << (k % 64)) != 0,
                            score(&a, k) as i64 - score(&b, k) as i64 == target
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fused_offset_merges_equal_independent_scalar_maxima() {
        for n in [0usize, 1, 7, 63, 64, 65, 129] {
            for a_offset in [0, 1, 7, 8, 9, 31, 63, 64, 65, 127, 128] {
                for b_offset in [0, 1, 7, 8, 9, 31, 63, 64, 65, 127, 128] {
                    let mut a = vec![0u64; n.div_ceil(64)];
                    let mut b = a.clone();
                    for k in 0..n {
                        if k % 7 < 3 {
                            a[k / 64] |= 1 << (k % 64);
                        }
                        if k % 11 < 6 {
                            b[k / 64] |= 1 << (k % 64);
                        }
                    }
                    let span = n + a_offset.max(b_offset);
                    let result = merge_shifted(
                        &packed(&a, n),
                        a_offset,
                        &packed(&b, n),
                        b_offset,
                        span,
                        &Uninterrupted,
                    )
                    .unwrap();
                    let result = result.words();
                    for k in 0..=span {
                        assert_eq!(
                            score(result, k),
                            score(&a, k.saturating_sub(a_offset).min(n))
                                .max(score(&b, k.saturating_sub(b_offset).min(n)))
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fused_sparse_windows_equal_scalar_advancement_with_padding_and_absence() {
        for span in [0usize, 1, 7, 63, 64, 65, 129] {
            for first in [0, 1, 7, 8, 9, 31, 63, 64, 65, 127, 128] {
                for seed in 0..6 {
                    let global_len = first + span + 73;
                    let mut mask = vec![0u64; global_len.div_ceil(64)];
                    for k in 0..global_len {
                        if seed != 0 && (k * 17 + seed) % 19 < seed * 2 {
                            mask[k / 64] |= 1 << (k % 64);
                        }
                    }
                    let postings: Vec<_> = mask
                        .iter()
                        .copied()
                        .enumerate()
                        .filter(|&(_, bits)| bits != 0)
                        .collect();
                    let mut bits = vec![0; span.div_ceil(64)];
                    for k in 0..span {
                        if (k * 3 + seed) % 11 < 5 {
                            bits[k / 64] |= 1 << (k % 64);
                        }
                    }
                    let mut expected: Vec<_> = (0..=span).map(|k| score(&bits, k)).collect();
                    let mut diagonal = 0;
                    for k in 0..span {
                        let old = expected[k + 1];
                        let equal = mask[(first + k) / 64] & (1 << ((first + k) % 64)) != 0;
                        expected[k + 1] = old.max(expected[k]).max(diagonal + usize::from(equal));
                        diagonal = old;
                    }
                    extend_window_sparse(&mut bits, &postings, first, span, &Uninterrupted)
                        .unwrap();
                    for (k, expected) in expected.into_iter().enumerate() {
                        assert_eq!(score(&bits, k), expected);
                    }
                    if !span.is_multiple_of(64)
                        && let Some(last) = bits.last()
                    {
                        assert_eq!(last >> (span % 64), 0);
                    }
                }
            }
        }
    }
}

/// The vector read from its last bit to its first: bit `k` of the result is
/// bit `length - 1 - k` of `bits`.
pub(in crate::dp_align) fn reverse_bits(bits: &PackedBits) -> PackedBits {
    // Reversing the stored words reverses all `64 * words` stored bits, so
    // the vector's own bits land after `padding` zero bits; the window from
    // there is the reversed vector.
    let words = bits.words();
    let reversed = PackedBits::from_fn(words.len() * 64, |block| {
        words[words.len() - 1 - block].reverse_bits()
    });
    let padding = (64 - bits.length() % 64) % 64;
    PackedBits::window(reversed.padded(), WindowStart::At(padding), bits.length())
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(in crate::dp_align) enum CutKind {
    Deleted,
    Matched,
}

#[cfg(test)]
pub(in crate::dp_align) fn score_cuts_aligned<C: WorkControl + ?Sized>(
    forward: &[u64],
    reverse: &PackedBits,
    total: usize,
    length: usize,
    optimum: usize,
    kind: CutKind,
    control: &C,
) -> Result<Vec<u64>, C::Error> {
    control.check()?;
    let matched = matches!(kind, CutKind::Matched);
    let mut cuts = vec![0; (length + 1).div_ceil(64)];
    if optimum > length || (matched && (optimum == 0 || length == 0)) {
        return Ok(cuts);
    }
    let reverse = reverse.padded();
    // A matched cut reads the reverse profile one bit on: B(k+1).
    let from = match matched {
        true => WindowStart::At(1),
        false => WindowStart::At(0),
    };
    let mut base = total as i64;
    let target = if matched {
        base -= (reverse.window_word(WindowStart::At(0), 0) & 1) as i64;
        optimum - 1
    } else {
        optimum
    };
    cuts[0] = u64::from(base == target as i64);
    for (block, &a) in forward.iter().enumerate() {
        if block % 1024 == 0 {
            control.check()?;
        }
        let b = reverse.window_word(from, block);
        for byte in 0..8 {
            let start = block * 64 + byte * 8;
            if start >= length {
                break;
            }
            let ab = ((a >> (8 * byte)) & 255) as usize;
            let bb = ((b >> (8 * byte)) & 255) as usize;
            let difference = target as i64 - base;
            if (-8..=8).contains(&difference) {
                let ranks = RANKS[ab] + ONES * 16 - RANKS[bb];
                let unequal = ranks ^ (ONES * (16 + difference) as u64);
                let equal = !(((unequal & LOW) + LOW) | unequal | LOW) & HIGH;
                let width = (length - start).min(8);
                let mask = ((equal >> 7).wrapping_mul(PACK) >> 56) & ((1 << width) - 1);
                let cut = start + 1;
                cuts[cut / 64] |= mask << (cut % 64);
                if cut % 64 > 56 && cut / 64 + 1 < cuts.len() {
                    cuts[cut / 64 + 1] |= mask >> (64 - cut % 64);
                }
            }
            base += ab.count_ones() as i64 - bb.count_ones() as i64;
        }
    }
    if matched {
        cuts[length / 64] &= !(1 << (length % 64));
    }
    Ok(cuts)
}

/// Equality of cumulative increment differences, with an independently bound
/// score origin. No fabricated zero-origin LCS row or unsigned bias is needed.
#[cfg(test)]
fn score_difference_cuts<C: WorkControl + ?Sized>(
    a: &[u64],
    b: &[u64],
    length: usize,
    mut delta: i64,
    control: &C,
) -> Result<Vec<u64>, C::Error> {
    control.check()?;
    let mut cuts = vec![0; (length + 1).div_ceil(64)];
    cuts[0] = u64::from(delta == 0);
    for (block, (&a, &b)) in a.iter().zip(b).enumerate() {
        if block % 1024 == 0 {
            control.check()?;
        }
        for byte in 0..8 {
            let start = block * 64 + byte * 8;
            if start >= length {
                break;
            }
            let ab = ((a >> (8 * byte)) & 255) as usize;
            let bb = ((b >> (8 * byte)) & 255) as usize;
            if (-8..=8).contains(&delta) {
                let ranks = RANKS[ab] + ONES * 16 - RANKS[bb];
                let unequal = ranks ^ (ONES * (16 + delta) as u64);
                let equal = !(((unequal & LOW) + LOW) | unequal | LOW) & HIGH;
                let width = (length - start).min(8);
                let mask = ((equal >> 7).wrapping_mul(PACK) >> 56) & ((1 << width) - 1);
                let cut = start + 1;
                cuts[cut / 64] |= mask << (cut % 64);
                if cut % 64 > 56 && cut / 64 + 1 < cuts.len() {
                    cuts[cut / 64 + 1] |= mask >> (64 - cut % 64);
                }
            }
            delta -= ab.count_ones() as i64 - bb.count_ones() as i64;
        }
    }
    Ok(cuts)
}
