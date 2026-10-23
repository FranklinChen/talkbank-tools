//! Generic scalar, shift, trim and footprint controls for producer-owned maxima.
use super::*;
use crate::dp_align::LiteralMatchMode;
use std::cell::Cell;

fn words(length: usize) -> Vec<String> {
    (0..length)
        .map(|index| ["a", "B", "c", "b", "a", "d"][index % 6].to_owned())
        .collect()
}

fn check_maximum(profile: &Profile) {
    let population = profile
        .bits()
        .iter()
        .map(|bits| bits.count_ones() as usize)
        .sum::<usize>();
    assert_eq!(profile.total(), profile.base + population);
    assert_eq!(profile.score(profile.last()), Some(profile.total()));
    let cloned = profile.clone();
    assert_eq!(cloned.total(), profile.total());
    assert_eq!(cloned.bits, profile.bits);
}

#[test]
fn producer_maxima_equal_independent_scalar_lcs_at_every_prefix() {
    for length in [0, 1, 7, 63, 64, 65, 127, 128, 129, 257] {
        let reference = words(length);
        for insensitive in [false, true] {
            let mode = if insensitive {
                LiteralMatchMode::AsciiInsensitive
            } else {
                LiteralMatchMode::Exact
            };
            let literal = LiteralReference::observe(
                &reference,
                mode,
                ReferenceOrder::Forward,
                &Uninterrupted,
            )
            .unwrap();
            let mut profile = Profile::zero(length);
            let mut scores = vec![0usize; length + 1];
            for word in ["a", "b", "unheard", "c", "a", "B", "d", "a"] {
                let mut next = vec![0; length + 1];
                for (index, token) in reference.iter().enumerate() {
                    let matched = if insensitive {
                        word.eq_ignore_ascii_case(token)
                    } else {
                        word == token
                    };
                    next[index + 1] = next[index]
                        .max(scores[index + 1])
                        .max(scores[index] + usize::from(matched));
                }
                scores = next;
                let Row::Viable(next) = profile
                    .advance(&literal, word, 0, length, length, &Uninterrupted)
                    .unwrap()
                else {
                    panic!("full-domain literal row remains reachable")
                };
                profile = next;
                check_maximum(&profile);
                assert_eq!(profile.total(), scores[length]);
                for (prefix, &score) in scores.iter().enumerate() {
                    assert_eq!(
                        profile.score(prefix),
                        Some(score),
                        "length={length} insensitive={insensitive} word={word} prefix={prefix}"
                    );
                }
            }
        }
    }
}

#[test]
fn producer_maxima_survive_both_trims_and_unreachable_rows() {
    let mut left = false;
    let mut right = false;
    let mut unreachable = false;
    for length in [0, 1, 7, 63, 64, 65, 127, 129, 257] {
        let reference = words(length);
        let literal = LiteralReference::observe(
            &reference,
            LiteralMatchMode::Exact,
            ReferenceOrder::Forward,
            &Uninterrupted,
        )
        .unwrap();
        for reference_slack in [0, 1, 4, 63, length] {
            for source_slack in [0, 1, 4, 8] {
                let slack = reference_slack.min(length);
                let mut profile = Profile::zero(slack);
                for (index, word) in ["a", "unheard", "c", "a", "B", "unheard", "b", "a"]
                    .into_iter()
                    .enumerate()
                {
                    let first = profile.first;
                    let last = (profile.last() + 1).min(length);
                    let required = (index + 1).saturating_sub(source_slack);
                    let row = profile
                        .advance(&literal, word, required, slack, length, &Uninterrupted)
                        .unwrap();
                    let Row::Viable(next) = row else {
                        unreachable = true;
                        break;
                    };
                    left |= next.first > first;
                    right |= next.last() < last;
                    check_maximum(&next);
                    profile = next;
                }
            }
        }
    }
    assert!(
        left && right && unreachable,
        "all admission/trimming branches are exercised"
    );
}

#[test]
fn merged_maxima_equal_pointwise_scalar_maxima_across_shifted_domains() {
    let mut profiles = Vec::new();
    for first in [0usize, 1, 63, 64, 65] {
        for span in [0usize, 1, 7, 63, 64, 65, 129] {
            for pattern in 0..3 {
                let mut bits = vec![0u64; span.div_ceil(64)];
                let mut maximum = 7;
                for index in 0..span {
                    if (index + pattern) % 3 == 0 {
                        bits[index / 64] |= 1 << (index % 64);
                        maximum += 1;
                    }
                }
                profiles.push(Profile {
                    first,
                    base: 7,
                    bits: PackedBits::from_words(bits, span).unwrap(),
                    maximum,
                });
            }
        }
    }
    for a in &profiles {
        for b in &profiles {
            let merged = a.clone().merge(b.clone(), &Uninterrupted).unwrap();
            check_maximum(&merged);
            for at in merged.first..=merged.last() {
                let score =
                    |profile: &Profile| profile.score(at.clamp(profile.first, profile.last()));
                assert_eq!(
                    merged.score(at),
                    score(a).max(score(b)),
                    "a=[{},{}] b=[{},{}] at={at}",
                    a.first,
                    a.last(),
                    b.first,
                    b.last()
                );
            }
        }
    }
}

#[test]
fn suffix_preparation_pays_only_allocation_and_reversal_and_preserves_scores() {
    struct Paid(Cell<usize>);
    impl WorkControl for Paid {
        type Error = std::convert::Infallible;
        fn check(&self) -> Result<(), Self::Error> {
            Ok(())
        }
    }
    impl PackedControl for Paid {
        fn charge(&self, units: usize) -> Result<(), Self::Error> {
            self.0.set(self.0.get() + units);
            Ok(())
        }
        fn refuse(&self, inconsistency: LocalInterleavingInconsistency) -> Self::Error {
            panic!("the test search contradicted itself: {inconsistency:?}")
        }
    }
    for length in [0, 1, 63, 64, 65, 127, 128, 129, 257] {
        let reference = words(length);
        let literal = LiteralReference::observe(
            &reference,
            LiteralMatchMode::Exact,
            ReferenceOrder::Forward,
            &Uninterrupted,
        )
        .unwrap();
        let Row::Viable(profile) = Profile::zero(length)
            .advance(&literal, "a", 0, length, length, &Uninterrupted)
            .unwrap()
        else {
            panic!("full row")
        };
        let paid = Paid(Cell::new(0));
        let suffix = Suffix::of(&profile, length, &paid).unwrap();
        assert_eq!(paid.0.get(), 2 * (profile.bits().len() + 1));
        for prefix in suffix.first..=suffix.last() {
            assert_eq!(Some(suffix.score(prefix)), profile.score(length - prefix));
        }
    }
}

#[test]
fn cached_maximum_fits_the_existing_checkpoint_metadata_reservation() {
    let metadata = std::mem::size_of::<(usize, Row)>().div_ceil(std::mem::size_of::<usize>());
    assert!(
        metadata <= 8,
        "checkpoint row metadata needs {metadata} machine words, but reserves eight"
    );
}

#[test]
fn cached_state_snapshot_distinguishes_represented_origin_from_terminal_score() {
    let reference = vec!["a".to_owned(); 65];
    let literal = LiteralReference::observe(
        &reference,
        LiteralMatchMode::Exact,
        ReferenceOrder::Forward,
        &Uninterrupted,
    )
    .unwrap();
    let root = Profile::zero(65);
    let Row::Viable(advanced) = root
        .clone()
        .advance(&literal, "a", 0, 65, 65, &Uninterrupted)
        .unwrap()
    else {
        panic!("advance")
    };
    let Row::Viable(trimmed) = advanced
        .clone()
        .advance(&literal, "unheard", 1, 65, 65, &Uninterrupted)
        .unwrap()
    else {
        panic!("trim")
    };
    let rows = [root, advanced, trimmed];
    for row in &rows {
        check_maximum(row);
    }
    insta::assert_json_snapshot!(rows.iter().map(|row| serde_json::json!({"first":row.first,"last":row.last(),"base":row.base,"maximum":row.total(),"blocks":row.bits().len()})).collect::<Vec<_>>(), @r###"
    [
      {
        "base": 0,
        "blocks": 2,
        "first": 0,
        "last": 65,
        "maximum": 0
      },
      {
        "base": 0,
        "blocks": 2,
        "first": 0,
        "last": 65,
        "maximum": 1
      },
      {
        "base": 1,
        "blocks": 1,
        "first": 1,
        "last": 65,
        "maximum": 1
      }
    ]
    "###);
}

/// RED FIRST: merging two rows of different bases was a debug assertion, so
/// a release build merged them and reported a pointwise maximum over two
/// incomparable score origins. It is refused in every build as the
/// inconsistency it is.
#[test]
fn rows_of_different_bases_are_refused_not_merged() {
    struct Refusing;
    impl WorkControl for Refusing {
        type Error = Option<LocalInterleavingInconsistency>;
        fn check(&self) -> Result<(), Self::Error> {
            Ok(())
        }
    }
    impl PackedControl for Refusing {
        fn charge(&self, _: usize) -> Result<(), Self::Error> {
            Ok(())
        }
        fn refuse(&self, inconsistency: LocalInterleavingInconsistency) -> Self::Error {
            Some(inconsistency)
        }
    }
    let row = |base| Profile {
        first: 0,
        base,
        bits: PackedBits::from_words(vec![0b1], 1).unwrap(),
        maximum: base + 1,
    };
    assert_eq!(
        row(3).merge(row(4), &Refusing).err(),
        Some(Some(LocalInterleavingInconsistency::ProfileBasesDiffer {
            left: 3,
            right: 4
        }))
    );
}
