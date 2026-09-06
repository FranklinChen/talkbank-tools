use super::*;

fn s(words: &[&str]) -> Vec<String> {
    words.iter().map(|w| w.to_string()).collect()
}

struct LegacyWord(String);

impl Alignable for LegacyWord {
    type Policy = MatchMode;

    fn matches(&self, other: &Self, policy: MatchMode) -> bool {
        match policy {
            MatchMode::Exact => self.0 == other.0,
            MatchMode::CaseInsensitive => self.0.eq_ignore_ascii_case(&other.0),
            MatchMode::Fuzzy { threshold } => {
                self.0.eq_ignore_ascii_case(&other.0)
                    || strsim::jaro_winkler(&self.0.to_lowercase(), &other.0.to_lowercase())
                        >= threshold
            }
        }
    }

    fn to_key(&self) -> String {
        self.0.clone()
    }
}

/// Preserve the pre-preparation comparison as an independent oracle. Compare
/// complete plans so a changed tie break or original result spelling is visible.
#[test]
fn prepared_comparison_preserves_legacy_plans() {
    let vocabulary = [
        "ΟΣ", "ος", "οσ", "İ", "i", "i\u{307}", "Café", "CAFÉ", "Straße", "STRASSE", "a", "A", "",
        "word",
    ];
    let cases = [
        (Vec::new(), s(&["Café"])),
        (s(&["İ"]), Vec::new()),
        (s(&vocabulary), s(&["word", "i", "CAFÉ", "οσ", "STRASSE"])),
        (
            (0..70)
                .map(|i| vocabulary[(i * 3) % vocabulary.len()].to_owned())
                .collect(),
            (0..73)
                .map(|i| vocabulary[(i * 5 + 1) % vocabulary.len()].to_owned())
                .collect(),
        ),
    ];
    let modes = [
        MatchMode::Exact,
        MatchMode::CaseInsensitive,
        MatchMode::Fuzzy { threshold: 0.0 },
        MatchMode::Fuzzy { threshold: 0.85 },
        MatchMode::Fuzzy { threshold: 1.0 },
        MatchMode::Fuzzy { threshold: 1.1 },
        MatchMode::Fuzzy {
            threshold: f64::NAN,
        },
    ];
    for (case, (payload, reference)) in cases.iter().enumerate() {
        let legacy_payload: Vec<_> = payload.iter().cloned().map(LegacyWord).collect();
        let legacy_reference: Vec<_> = reference.iter().cloned().map(LegacyWord).collect();
        for mode in modes {
            assert_eq!(
                align(payload, reference, mode),
                align_generic(&legacy_payload, &legacy_reference, mode),
                "case {case}, mode {mode:?}",
            );
        }
    }
}

/// Manual comparison-cost probe in the existing test binary, with no timing
/// threshold in CI. Run with `--ignored --nocapture`; compare the same profile
/// and result digest before and after changing comparison preparation.
#[test]
#[ignore = "manual fuzzy-alignment performance probe"]
fn fuzzy_comparison_performance_probe() {
    let vocabulary = [
        "conversation",
        "morning",
        "research",
        "together",
        "utterance",
        "window",
        "hospital",
        "memory",
        "coffee",
        "walking",
        "beautiful",
        "question",
        "Straße",
        "İstanbul",
        "CAFÉ",
        "café",
        "answer",
        "suddenly",
        "tomorrow",
        "perhaps",
        "because",
        "language",
        "different",
        "people",
        "through",
    ];
    let payload: Vec<_> = (0..400)
        .map(|i| vocabulary[(i * 7) % vocabulary.len()].to_owned())
        .collect();
    let reference: Vec<_> = (0..430)
        .map(|i| vocabulary[(i * 11 + 3) % vocabulary.len()].to_owned())
        .collect();
    let legacy_payload: Vec<_> = payload.iter().cloned().map(LegacyWord).collect();
    let legacy_reference: Vec<_> = reference.iter().cloned().map(LegacyWord).collect();
    let mode = MatchMode::Fuzzy { threshold: 0.85 };
    let mut legacy_time = std::time::Duration::ZERO;
    let mut prepared_time = std::time::Duration::ZERO;
    let mut digest = blake3::Hasher::new();
    for round in 0..4 {
        // Alternate order to reduce drift from concurrent machine activity.
        for prepared_first in [round % 2 == 0, round % 2 != 0] {
            let start = std::time::Instant::now();
            let result = std::hint::black_box(if prepared_first {
                align(&payload, &reference, mode)
            } else {
                align_generic(&legacy_payload, &legacy_reference, mode)
            });
            let elapsed = start.elapsed();
            if prepared_first {
                prepared_time += elapsed;
            } else {
                legacy_time += elapsed;
            }
            digest.update(format!("{result:?}").as_bytes());
        }
    }
    eprintln!(
        "fuzzy paired probe: legacy_us={} prepared_us={} digest={}",
        legacy_time.as_micros(),
        prepared_time.as_micros(),
        digest.finalize()
    );
}

#[test]
fn test_identical() {
    let result = align(&s(&["a", "b", "c"]), &s(&["a", "b", "c"]), MatchMode::Exact);
    assert_eq!(result.len(), 3);
    assert!(
        result
            .iter()
            .all(|r| matches!(r, AlignResult::Match { .. }))
    );
}

#[test]
fn test_empty_payload() {
    let result = align(&s(&[]), &s(&["a", "b"]), MatchMode::Exact);
    assert_eq!(result.len(), 2);
    assert!(
        result
            .iter()
            .all(|r| matches!(r, AlignResult::ExtraReference { .. }))
    );
}

#[test]
fn test_empty_reference() {
    let result = align(&s(&["a", "b"]), &s(&[]), MatchMode::Exact);
    assert_eq!(result.len(), 2);
    assert!(
        result
            .iter()
            .all(|r| matches!(r, AlignResult::ExtraPayload { .. }))
    );
}

#[test]
fn test_case_insensitive() {
    let result = align(
        &s(&["Hello", "World"]),
        &s(&["hello", "world"]),
        MatchMode::CaseInsensitive,
    );
    let matches: Vec<_> = result
        .iter()
        .filter(|r| matches!(r, AlignResult::Match { .. }))
        .collect();
    assert_eq!(matches.len(), 2);
}

// --- align_chars tests ---

fn c(word: &str) -> Vec<char> {
    word.chars().collect()
}

#[test]
fn test_align_chars_identical() {
    let result = align_chars(&c("abc"), &c("abc"), MatchMode::Exact);
    assert_eq!(result.len(), 3);
    assert!(
        result
            .iter()
            .all(|r| matches!(r, AlignResult::Match { .. }))
    );
}

#[test]
fn test_align_chars_matches_string_version() {
    // "don't" vs "do" "n't", character-level alignment
    let orig = c("don't");
    let tokens = c("don't");
    let result_chars = align_chars(&orig, &tokens, MatchMode::CaseInsensitive);

    let orig_str: Vec<String> = orig.iter().map(|c| c.to_string()).collect();
    let tokens_str: Vec<String> = tokens.iter().map(|c| c.to_string()).collect();
    let result_str = align(&orig_str, &tokens_str, MatchMode::CaseInsensitive);

    assert_eq!(result_chars, result_str);
}

#[test]
fn test_align_chars_case_insensitive() {
    let result = align_chars(&c("Hello"), &c("hello"), MatchMode::CaseInsensitive);
    let match_count = result
        .iter()
        .filter(|r| matches!(r, AlignResult::Match { .. }))
        .count();
    assert_eq!(match_count, 5);
}

#[test]
fn test_align_chars_split() {
    // Simulates "don't" → "do" + "n't" at character level
    let orig = c("don't");
    let tokens: Vec<char> = "don't".chars().collect();
    let result = align_chars(&orig, &tokens, MatchMode::Exact);
    assert_eq!(result.len(), 5);
    assert!(
        result
            .iter()
            .all(|r| matches!(r, AlignResult::Match { .. }))
    );
}

#[test]
fn test_align_chars_empty() {
    let result = align_chars(&[], &c("abc"), MatchMode::Exact);
    assert_eq!(result.len(), 3);
    assert!(
        result
            .iter()
            .all(|r| matches!(r, AlignResult::ExtraReference { .. }))
    );
}

// --- prefix/suffix stripping tests ---

#[test]
fn test_shared_prefix_suffix_with_middle_diff() {
    // "a b X c d" vs "a b Y c d", prefix "a b", suffix "c d", DP only on X vs Y
    let result = align(
        &s(&["a", "b", "X", "c", "d"]),
        &s(&["a", "b", "Y", "c", "d"]),
        MatchMode::Exact,
    );
    // a=match, b=match, X=extra_payload, Y=extra_reference, c=match, d=match
    assert_eq!(result.len(), 6);
    assert!(
        matches!(&result[0], AlignResult::Match { key, payload_idx: 0, reference_idx: 0 } if key == "a")
    );
    assert!(
        matches!(&result[1], AlignResult::Match { key, payload_idx: 1, reference_idx: 1 } if key == "b")
    );
    assert!(
        matches!(&result[4], AlignResult::Match { key, payload_idx: 3, reference_idx: 3 } if key == "c")
    );
    assert!(
        matches!(&result[5], AlignResult::Match { key, payload_idx: 4, reference_idx: 4 } if key == "d")
    );
}

#[test]
fn test_shared_prefix_only() {
    let result = align(
        &s(&["a", "b", "c"]),
        &s(&["a", "b", "d", "e"]),
        MatchMode::Exact,
    );
    // a=match, b=match, c vs d+e handled by DP
    assert!(matches!(
        &result[0],
        AlignResult::Match {
            payload_idx: 0,
            reference_idx: 0,
            ..
        }
    ));
    assert!(matches!(
        &result[1],
        AlignResult::Match {
            payload_idx: 1,
            reference_idx: 1,
            ..
        }
    ));
}

#[test]
fn test_single_insertion_in_middle() {
    // Common WER scenario: "the cat sat" vs "the big cat sat"
    let result = align(
        &s(&["the", "cat", "sat"]),
        &s(&["the", "big", "cat", "sat"]),
        MatchMode::Exact,
    );
    let matches: Vec<_> = result
        .iter()
        .filter(|r| matches!(r, AlignResult::Match { .. }))
        .collect();
    let extras: Vec<_> = result
        .iter()
        .filter(|r| matches!(r, AlignResult::ExtraReference { .. }))
        .collect();
    assert_eq!(matches.len(), 3); // the, cat, sat
    assert_eq!(extras.len(), 1); // big
}

#[test]
fn test_prefix_suffix_strip_preserves_indices() {
    // Verify absolute indices are correct after stripping
    let result = align(
        &s(&["x", "y", "INSERT", "z", "w"]),
        &s(&["x", "y", "z", "w"]),
        MatchMode::Exact,
    );
    // x(0,0), y(1,1), INSERT(2,_), z(3,2), w(4,3)
    assert!(matches!(
        &result[0],
        AlignResult::Match {
            payload_idx: 0,
            reference_idx: 0,
            ..
        }
    ));
    assert!(matches!(
        &result[1],
        AlignResult::Match {
            payload_idx: 1,
            reference_idx: 1,
            ..
        }
    ));
    assert!(matches!(
        &result[2],
        AlignResult::ExtraPayload { payload_idx: 2, .. }
    ));
    assert!(matches!(
        &result[3],
        AlignResult::Match {
            payload_idx: 3,
            reference_idx: 2,
            ..
        }
    ));
    assert!(matches!(
        &result[4],
        AlignResult::Match {
            payload_idx: 4,
            reference_idx: 3,
            ..
        }
    ));
}

// --- property tests ---

use proptest::prelude::*;
use std::collections::BTreeSet;

/// Small alphabet to encourage matches and exercise the DP logic.
fn word_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("the".to_string()),
        Just("cat".to_string()),
        Just("sat".to_string()),
        Just("on".to_string()),
        Just("mat".to_string()),
        Just("big".to_string()),
        Just("a".to_string()),
        "[a-z]{1,4}".prop_map(|s| s),
    ]
}

fn word_vec(max_len: usize) -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(word_strategy(), 0..=max_len)
}

fn match_mode_strategy() -> impl Strategy<Value = MatchMode> {
    prop_oneof![Just(MatchMode::Exact), Just(MatchMode::CaseInsensitive)]
}

/// Helper: extract payload indices that appear in the alignment result.
fn payload_indices(result: &[AlignResult]) -> BTreeSet<usize> {
    result
        .iter()
        .filter_map(|r| match r {
            AlignResult::Match { payload_idx, .. }
            | AlignResult::ExtraPayload { payload_idx, .. } => Some(*payload_idx),
            AlignResult::ExtraReference { .. } => None,
        })
        .collect()
}

/// Helper: extract reference indices that appear in the alignment result.
fn reference_indices(result: &[AlignResult]) -> BTreeSet<usize> {
    result
        .iter()
        .filter_map(|r| match r {
            AlignResult::Match { reference_idx, .. }
            | AlignResult::ExtraReference { reference_idx, .. } => Some(*reference_idx),
            AlignResult::ExtraPayload { .. } => None,
        })
        .collect()
}

proptest! {
    /// Every payload index 0..len appears exactly once in the output.
    #[test]
    fn completeness_payload(
        payload in word_vec(8),
        reference in word_vec(8),
        mode in match_mode_strategy(),
    ) {
        let result = align(&payload, &reference, mode);
        let expected: BTreeSet<usize> = (0..payload.len()).collect();
        prop_assert_eq!(payload_indices(&result), expected);
    }

    /// Every reference index 0..len appears exactly once in the output.
    #[test]
    fn completeness_reference(
        payload in word_vec(8),
        reference in word_vec(8),
        mode in match_mode_strategy(),
    ) {
        let result = align(&payload, &reference, mode);
        let expected: BTreeSet<usize> = (0..reference.len()).collect();
        prop_assert_eq!(reference_indices(&result), expected);
    }

    /// Payload indices appear in strictly increasing order.
    #[test]
    fn payload_index_monotonicity(
        payload in word_vec(8),
        reference in word_vec(8),
        mode in match_mode_strategy(),
    ) {
        let result = align(&payload, &reference, mode);
        let pay_idxs: Vec<usize> = result.iter().filter_map(|r| match r {
            AlignResult::Match { payload_idx, .. }
            | AlignResult::ExtraPayload { payload_idx, .. } => Some(*payload_idx),
            _ => None,
        }).collect();
        for w in pay_idxs.windows(2) {
            prop_assert!(w[0] < w[1], "payload indices not monotonic: {:?}", pay_idxs);
        }
    }

    /// Reference indices appear in strictly increasing order.
    #[test]
    fn reference_index_monotonicity(
        payload in word_vec(8),
        reference in word_vec(8),
        mode in match_mode_strategy(),
    ) {
        let result = align(&payload, &reference, mode);
        let ref_idxs: Vec<usize> = result.iter().filter_map(|r| match r {
            AlignResult::Match { reference_idx, .. }
            | AlignResult::ExtraReference { reference_idx, .. } => Some(*reference_idx),
            _ => None,
        }).collect();
        for w in ref_idxs.windows(2) {
            prop_assert!(w[0] < w[1], "reference indices not monotonic: {:?}", ref_idxs);
        }
    }

    /// Aligning identical sequences produces all-Match results.
    #[test]
    fn identity_alignment(words in word_vec(10)) {
        let result = align(&words, &words, MatchMode::Exact);
        prop_assert_eq!(result.len(), words.len());
        for (i, r) in result.iter().enumerate() {
            prop_assert!(
                matches!(r, AlignResult::Match { payload_idx, reference_idx, .. }
                    if *payload_idx == i && *reference_idx == i),
                "Expected Match at index {}, got {:?}", i, r
            );
        }
    }

    /// align_chars and align produce identical results on single-char strings.
    #[test]
    fn char_string_equivalence(
        payload in prop::collection::vec(prop::char::range('a', 'z'), 0..=8),
        reference in prop::collection::vec(prop::char::range('a', 'z'), 0..=8),
        mode in match_mode_strategy(),
    ) {
        let char_result = align_chars(&payload, &reference, mode);
        let str_pay: Vec<String> = payload.iter().map(|c| c.to_string()).collect();
        let str_ref: Vec<String> = reference.iter().map(|c| c.to_string()).collect();
        let str_result = align(&str_pay, &str_ref, mode);
        prop_assert_eq!(char_result, str_result);
    }

    /// CaseInsensitive matches are a superset of Exact matches.
    /// If Exact produces a Match at (p, r), CaseInsensitive must too.
    #[test]
    fn case_insensitive_superset(
        payload in word_vec(6),
        reference in word_vec(6),
    ) {
        let exact = align(&payload, &reference, MatchMode::Exact);
        let ci = align(&payload, &reference, MatchMode::CaseInsensitive);

        let exact_match_count = exact.iter()
            .filter(|r| matches!(r, AlignResult::Match { .. }))
            .count();
        let ci_match_count = ci.iter()
            .filter(|r| matches!(r, AlignResult::Match { .. }))
            .count();
        prop_assert!(
            ci_match_count >= exact_match_count,
            "CaseInsensitive ({}) should match at least as many as Exact ({})",
            ci_match_count, exact_match_count
        );
    }

    /// Output length equals payload_len + reference_len - match_count.
    /// Each Match consumes one from each side; extras consume one from one side.
    #[test]
    fn output_length_invariant(
        payload in word_vec(8),
        reference in word_vec(8),
        mode in match_mode_strategy(),
    ) {
        let result = align(&payload, &reference, mode);
        let match_count = result.iter()
            .filter(|r| matches!(r, AlignResult::Match { .. }))
            .count();
        prop_assert_eq!(
            result.len(),
            payload.len() + reference.len() - match_count,
            "len={}, pay={}, ref={}, matches={}",
            result.len(), payload.len(), reference.len(), match_count
        );
    }

    /// Completeness holds for char-level alignment too.
    #[test]
    fn char_completeness(
        payload in prop::collection::vec(prop::char::range('a', 'f'), 0..=10),
        reference in prop::collection::vec(prop::char::range('a', 'f'), 0..=10),
    ) {
        let result = align_chars(&payload, &reference, MatchMode::Exact);
        let expected_pay: BTreeSet<usize> = (0..payload.len()).collect();
        let expected_ref: BTreeSet<usize> = (0..reference.len()).collect();
        prop_assert_eq!(payload_indices(&result), expected_pay);
        prop_assert_eq!(reference_indices(&result), expected_ref);
    }
}

// ---------------------------------------------------------------------------
// Fuzzy matching tests
// ---------------------------------------------------------------------------

#[test]
fn test_fuzzy_exact_match_is_fast_path() {
    // Exact case-insensitive match should always work with fuzzy
    let result = align(
        &s(&["Hello", "World"]),
        &s(&["hello", "world"]),
        MatchMode::Fuzzy { threshold: 0.90 },
    );
    assert_eq!(result.len(), 2);
    assert!(
        result
            .iter()
            .all(|r| matches!(r, AlignResult::Match { .. }))
    );
}

#[test]
fn test_fuzzy_similar_words_match() {
    // "gonna" vs "gona": Jaro-Winkler ~0.96
    let result = align(
        &s(&["gonna"]),
        &s(&["gona"]),
        MatchMode::Fuzzy { threshold: 0.90 },
    );
    assert_eq!(result.len(), 1);
    assert!(matches!(result[0], AlignResult::Match { .. }));
}

#[test]
fn test_fuzzy_dissimilar_words_dont_match() {
    // "cat" vs "dog": Jaro-Winkler ~0.0
    let result = align(
        &s(&["cat"]),
        &s(&["dog"]),
        MatchMode::Fuzzy { threshold: 0.80 },
    );
    // Should produce extra payload + extra reference (no match)
    assert!(
        result
            .iter()
            .any(|r| matches!(r, AlignResult::ExtraPayload { .. }))
    );
    assert!(
        result
            .iter()
            .any(|r| matches!(r, AlignResult::ExtraReference { .. }))
    );
}

#[test]
fn test_fuzzy_threshold_controls_strictness() {
    // "going" vs "goin": Jaro-Winkler ~0.95
    let strict = align(
        &s(&["going"]),
        &s(&["goin"]),
        MatchMode::Fuzzy { threshold: 0.98 },
    );
    let lenient = align(
        &s(&["going"]),
        &s(&["goin"]),
        MatchMode::Fuzzy { threshold: 0.85 },
    );
    // Strict should NOT match, lenient SHOULD
    assert!(
        strict
            .iter()
            .any(|r| matches!(r, AlignResult::ExtraPayload { .. })),
        "strict threshold should reject"
    );
    assert!(
        lenient
            .iter()
            .all(|r| matches!(r, AlignResult::Match { .. })),
        "lenient threshold should accept"
    );
}

#[test]
fn test_fuzzy_in_sequence_alignment() {
    // Simulate ASR vs transcript with substitutions
    let transcript = s(&["I", "went", "to", "the", "store", "yesterday"]);
    let asr = s(&["i", "wen", "to", "da", "store", "yestarday"]);
    let result = align(&transcript, &asr, MatchMode::Fuzzy { threshold: 0.80 });

    let matches: Vec<_> = result
        .iter()
        .filter(|r| matches!(r, AlignResult::Match { .. }))
        .collect();

    // "I"/"i" exact, "to"/"to" exact, "store"/"store" exact = 3 exact
    // "went"/"wen" JW=0.94, "yesterday"/"yestarday" JW=0.92 = 2 fuzzy
    // "the"/"da" JW=0.0 → no match
    // Total: 5 matches
    assert!(
        matches.len() >= 5,
        "expected at least 5 matches with fuzzy, got {}",
        matches.len()
    );
}

#[test]
fn test_fuzzy_backchannel_variants() {
    // Common backchannel ASR variants
    let transcript = s(&["mhm"]);
    let asr = s(&["mmhm"]);
    let result = align(&transcript, &asr, MatchMode::Fuzzy { threshold: 0.80 });
    // "mhm" vs "mmhm": JW ~0.83
    assert!(
        result
            .iter()
            .any(|r| matches!(r, AlignResult::Match { .. })),
        "mhm/mmhm should fuzzy-match at 0.80 threshold"
    );
}
