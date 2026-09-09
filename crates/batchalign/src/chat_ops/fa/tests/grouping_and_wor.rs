//! Utterance grouping (`group_utterances_*`), `%wor` extraction policy, and reusable-wor detection (`has_reusable_wor_timing_*`, `refresh_existing_alignment_*`).

#![allow(unused_imports, dead_code)]

use super::*;

use talkbank_model::UtteranceIdx;
use talkbank_model::model::{Line, UtteranceContent, WriteChat};
use talkbank_parser::TreeSitterParser;

/// A recording of a stated length, for grouping tests.
///
/// These call sites used to pass `Option<u64>`, and several passed `None`. That
/// meant "the audio length is unknown", a state `Recording` deleted: grouping
/// silently SKIPPED untimed utterances whenever it held, so their words were
/// never aligned. The tests that passed `None` were not asserting that
/// behaviour, they were asserting grouping shape, so they get a recording long
/// enough not to bound anything they check.
#[test]
fn test_group_utterances_single_group() {
    let input = include_str!("../../../../../../test-fixtures/fa_two_timed_utterances.cha");
    let chat = parse_chat(input);
    // The recording is the one the fixture describes: its last bullet ends at
    // 10 s, so there is no trailing gap to extend into and the group ends
    // exactly there.
    let groups = group_utterances(&chat, 20000, &test_recording(10_000)).groups;
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].words.len(), 5); // hello world I want cookie
    assert_eq!(groups[0].audio_start_ms(), 0);
    assert_eq!(groups[0].audio_end_ms(), 10000);
}

#[test]
fn group_window_budget_refuses_one_oversized_utterance_without_losing_the_next() {
    let mut chat = parse_chat(include_str!(
        "../../../../../../test-fixtures/fa_two_timed_utterances.cha"
    ));
    get_test_utterance(&mut chat, 0)
        .main
        .content
        .bullet
        .as_mut()
        .unwrap()
        .timing
        .end_ms = 3_035_820;
    let grouped = group_utterances(&chat, 15_000, &test_recording(4_000_000));
    assert_eq!(
        grouped.refusals.len(),
        1,
        "oversized input needs durable refusal"
    );
    assert_eq!(grouped.groups.len(), 1);
    assert_eq!(
        grouped.groups[0].utterance_indices,
        vec![UtteranceIdx::new(1)]
    );
    assert_eq!(grouped.groups[0].words.len(), 3);
    assert_eq!(
        get_test_utterance(&mut chat, 0)
            .main
            .content
            .bullet
            .as_ref()
            .unwrap()
            .timing
            .end_ms,
        3_035_820,
        "grouping must not rewrite the supplied timing to make it fit"
    );
}

#[test]
fn group_window_budget_includes_trailing_padding() {
    let chat = parse_chat(include_str!(
        "../../../../../../test-fixtures/fa_two_timed_utterances.cha"
    ));
    let grouped = group_utterances(&chat, 10_000, &test_recording(20_000));
    assert_eq!(grouped.groups.len(), 1);
    assert_eq!(grouped.groups[0].audio_start_ms(), 0);
    assert_eq!(grouped.groups[0].audio_end_ms(), 10_000);
    assert_eq!(grouped.groups[0].words.len(), 5);
}

#[test]
fn group_window_budget_preserves_the_extent_of_overlapping_utterances() {
    let mut chat = parse_chat(include_str!(
        "../../../../../../test-fixtures/fa_two_timed_utterances.cha"
    ));
    get_test_utterance(&mut chat, 0)
        .main
        .content
        .bullet
        .as_mut()
        .unwrap()
        .timing
        .end_ms = 10_000;
    get_test_utterance(&mut chat, 1)
        .main
        .content
        .bullet
        .as_mut()
        .unwrap()
        .timing
        .end_ms = 6_000;
    let grouped = group_utterances(&chat, 10_000, &test_recording(10_000));
    assert_eq!(grouped.groups.len(), 1);
    assert_eq!(grouped.groups[0].audio_end_ms(), 10_000);
    assert_eq!(grouped.groups[0].words.len(), 5);
}

#[test]
fn test_wor_policy_fillers_match_between_fa_extraction_and_wor_generation() {
    let main = "&-um there .";

    assert_eq!(collect_proof_fa_words(main), vec!["um", "there"]);
    assert_eq!(generate_proof_wor_words(main), vec!["um", "there"]);
}

#[test]
fn test_wor_policy_replacements_use_original_surface() {
    let main = "what's is dis [: this] ?";

    assert_eq!(collect_proof_fa_words(main), vec!["what's", "is", "dis"]);
    assert_eq!(generate_proof_wor_words(main), vec!["what's", "is", "dis"]);
}

#[test]
fn test_wor_policy_standalone_spoken_tokens_match_between_fa_extraction_and_wor_generation() {
    for (main, expected) in [
        // Fillers (&-) ARE included
        ("&-um play .", &["um", "play"][..]),
        // Fragments (&+) are excluded, BA2 TokenType.ANNOT
        ("&+ss play .", &["play"][..]),
        // Nonwords (&~) are excluded, BA2 TokenType.ANNOT
        ("&~um play .", &["play"][..]),
    ] {
        let expected = words(expected);
        assert_eq!(collect_proof_fa_words(main), expected);
        assert_eq!(generate_proof_wor_words(main), expected);
    }
}

#[test]
fn test_wor_policy_retraced_spoken_tokens_match_between_fa_extraction_and_wor_generation() {
    for (main, expected) in [
        // Fragments excluded even inside retrace
        ("<&+ss> [/] play .", &["play"][..]),
        // Nonwords excluded even inside retrace
        ("<&~um> [/] play .", &["play"][..]),
        // Fillers still included inside retrace
        ("<&-um> [/] play .", &["um", "play"][..]),
        // Untranscribed excluded in all contexts
        ("<xxx> [/] play .", &["play"][..]),
        ("<yyy> [/] play .", &["play"][..]),
        ("<www> [/] play .", &["play"][..]),
    ] {
        let expected = words(expected);
        assert_eq!(collect_proof_fa_words(main), expected);
        assert_eq!(generate_proof_wor_words(main), expected);
    }
}

#[test]
fn test_group_utterances_backwards_bullets() {
    let input = include_str!("../../../../../../test-fixtures/fa_backwards_bullets.cha");
    let chat = parse_chat(input);
    let groups = group_utterances(&chat, 20000, &test_recording(600_000)).groups;
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].words.len(), 1);
    assert_eq!(groups[1].words.len(), 1);
}

#[test]
fn test_group_utterances_splits_on_time() {
    let input = include_str!("../../../../../../test-fixtures/fa_split_on_time.cha");
    let chat = parse_chat(input);
    let groups = group_utterances(&chat, 20000, &test_recording(600_000)).groups;
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].words.len(), 1);
    assert_eq!(groups[1].words.len(), 1);
}

#[test]
fn test_wor_policy_untranscribed_tokens_excluded_from_fa_and_wor() {
    for (main, expected) in [
        // Pure untranscribed utterances produce no FA words and empty %wor
        ("xxx .", &[][..]),
        ("yyy .", &[][..]),
        ("www .", &[][..]),
        // Mixed: real words stay; untranscribed tokens are dropped
        ("xxx play .", &["play"][..]),
        ("yyy play .", &["play"][..]),
        ("www play .", &["play"][..]),
        ("hello xxx world .", &["hello", "world"][..]),
        // Fragments excluded (BA2 TokenType.ANNOT)
        ("&+ss play .", &["play"][..]),
        // Nonwords excluded (BA2 TokenType.ANNOT)
        ("&~um play .", &["play"][..]),
        // Fillers included (BA2 TokenType.FP)
        ("&-um play .", &["um", "play"][..]),
    ] {
        let expected = words(expected);
        assert_eq!(
            collect_proof_fa_words(main),
            expected,
            "FA extraction for: {main}"
        );
        assert_eq!(
            generate_proof_wor_words(main),
            expected,
            "%wor generation for: {main}"
        );
    }
}

#[test]
fn test_group_utterances_splits_on_label_byte_cap() {
    // Two utterances each with 50 five-byte words = 250 bytes per utterance.
    // Combined = 500 bytes > MAX_GROUP_LABEL_BYTES (448).
    // Both fit in a 60-second window, so without the byte limit they'd be one group.
    let fifty_words = vec!["abcde"; 50].join(" ");
    let chat_text = format!(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|test|CHI|||||Child|||\n*CHI:\t{fifty_words} .\x15100_5000\x15\n*CHI:\t{fifty_words} .\x155000_10000\x15\n@End\n"
    );
    let chat = parse_chat(&chat_text);
    let groups = group_utterances(&chat, 60_000, &test_recording(10_000)).groups;
    assert_eq!(
        groups.len(),
        2,
        "expected 2 groups (50+50 words x 5 bytes = 500 > the 448 label-byte cap), got {}",
        groups.len()
    );
    // Every group must stay within the label cap, counted in BYTES, which is
    // the unit the cap is measured in and the only one that bounds the
    // engine's token count from above.
    for (i, group) in groups.iter().enumerate() {
        let bytes: usize = group.words.iter().map(|w| w.text.len()).sum();
        assert!(
            bytes <= MAX_GROUP_LABEL_BYTES,
            "group {i} has {bytes} label bytes, exceeds the {MAX_GROUP_LABEL_BYTES} cap"
        );
    }
}

/// The group label cap counts UTF-8 BYTES, and on non-Latin script that is
/// deliberately conservative: it splits earlier than the engine requires.
///
/// The cap stands in for a limit stated in TOKENS, which grouping cannot
/// measure: it is never told which engine will align a group, and no
/// tokenizer's segmentation is available here. Every token of every one of
/// these tokenizers occupies at least one UTF-8 byte, so a byte count bounds
/// the token count from ABOVE, and staying under the byte cap guarantees
/// staying under the token limit. A CHARACTER count carries no such
/// guarantee: outside ASCII it is smaller than the byte count, so counting
/// characters would let through groups whose token count exceeds the limit,
/// which is the hard `ValueError` this cap exists to prevent.
///
/// This fixture is the worked case. Two utterances of 30 three-character
/// Devanagari words are 180 CHARACTERS together, well under the cap, and 540
/// BYTES together, over it. They do not merge, and that split is the
/// conservative answer: more, smaller FA groups align correctly, where an
/// oversized group is a hard engine failure.
#[test]
fn test_group_utterances_splits_non_latin_conservatively_by_bytes() {
    // U+0928 U+092E U+0938: three codepoints, nine UTF-8 bytes.
    let thirty_words = vec!["\u{928}\u{92e}\u{938}"; 30].join(" ");
    let chat_text = format!(
        "@UTF8\n@Begin\n@Languages:\thin\n@Participants:\tCHI Child\n@ID:\thin|test|CHI|||||Child|||\n*CHI:\t{thirty_words} .\x15100_5000\x15\n*CHI:\t{thirty_words} .\x155000_10000\x15\n@End\n"
    );
    let chat = parse_chat(&chat_text);
    let groups = group_utterances(&chat, 60_000, &test_recording(10_000)).groups;
    let chars: usize = groups
        .iter()
        .flat_map(|group| group.words.iter())
        .map(|word| word.text.chars().count())
        .sum();
    assert_eq!(
        chars, 180,
        "fixture sits UNDER the cap counted in characters"
    );
    assert_eq!(
        groups.len(),
        2,
        "540 bytes is over the 448-byte cap, so the two utterances must not \
         merge: counting characters here would loosen a cap that stands in for \
         a token limit"
    );
    for (i, group) in groups.iter().enumerate() {
        let bytes: usize = group.words.iter().map(|word| word.text.len()).sum();
        assert!(
            bytes <= MAX_GROUP_LABEL_BYTES,
            "group {i} has {bytes} label bytes, over the {MAX_GROUP_LABEL_BYTES} cap"
        );
    }
}

/// The KNOWN LIMIT of the cap, stated as a test rather than only as prose: it
/// bounds a MERGE, not every group.
///
/// `PendingGroup::append` is the only place the cap is consulted, and it is
/// the only place two utterances are joined. One utterance whose own labels
/// exceed the cap becomes its own group and is sent as it stands, because
/// splitting it would mean splitting its audio window and its word list at a
/// position nothing in grouping can justify. So "no group exceeds the cap" is
/// NOT what this code guarantees, and a reader who assumes it will be wrong
/// exactly here.
#[test]
fn test_group_utterances_does_not_split_one_oversized_utterance() {
    // 100 five-byte words plus separators: 500 label bytes in one utterance.
    let hundred_words = vec!["abcde"; 100].join(" ");
    let chat_text = format!(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|test|CHI|||||Child|||\n*CHI:\t{hundred_words} .\x15100_5000\x15\n@End\n"
    );
    let chat = parse_chat(&chat_text);
    let groups = group_utterances(&chat, 60_000, &test_recording(10_000)).groups;
    assert_eq!(
        groups.len(),
        1,
        "one utterance is never split into two groups"
    );
    let bytes: usize = groups[0].words.iter().map(|word| word.text.len()).sum();
    assert!(
        bytes > MAX_GROUP_LABEL_BYTES,
        "the fixture must actually exceed the cap for this limit to be pinned, \
         got {bytes} bytes"
    );
}

/// Untimed utterances are ESTIMATED into the grouping, not dropped from it.
///
/// This test was `test_group_utterances_skips_untimed` and asserted the
/// opposite: that "hello", having no bullet, contributed no word to any FA
/// group and was therefore never aligned. That was not a policy, it was the
/// `Option<u64>` audio length showing through. Grouping skipped untimed
/// utterances whenever the duration was absent, and the duration was absent
/// whenever nobody had probed it, so whether a word got aligned depended on a
/// fact unrelated to the word.
///
/// Requiring a `Recording` deleted that state, so an estimate always exists and
/// both words are grouped. The old assertion is not weakened here; it is gone,
/// because the behaviour it described cannot occur.
#[test]
fn test_group_utterances_estimates_untimed_rather_than_dropping_them() {
    let input = include_str!("../../../../../../test-fixtures/fa_mixed_timed_untimed.cha");
    let chat = parse_chat(input);
    let groups = group_utterances(&chat, 20000, &test_recording(10_000)).groups;
    let grouped_words: usize = groups.iter().map(|g| g.words.len()).sum();
    assert_eq!(grouped_words, 2, "both 'hello' and 'world' reach FA");
}

#[test]
fn test_has_reusable_wor_timing_true_for_complete_wor_roundtrip() {
    let chat = parse_chat(&wor_timed_chat());
    assert!(has_reusable_wor_timing(&chat));
}

#[test]
fn test_has_reusable_wor_timing_false_for_partial_wor_timing() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%wor:\thello \u{15}100_500\u{15} world .\n@End\n".to_string();
    let chat = parse_chat(&input);
    assert!(!has_reusable_wor_timing(&chat));
}

#[test]
fn test_has_reusable_wor_timing_false_for_same_count_lexical_drift() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\tgood night .\n%wor:\tgood \u{15}100_500\u{15} morning \u{15}600_1000\u{15} .\n@End\n";
    let chat = parse_chat(input);
    let utt = get_utterance(&chat, 0);

    assert!(
        !has_reusable_wor_timing_for_utterance(utt),
        "equal word counts cannot make stale timing reusable when the main and %wor words differ"
    );
}

#[test]
fn test_has_reusable_wor_timing_false_when_wor_overruns_next_start() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world . \u{15}1000_1500\u{15}\n%wor:\thello \u{15}1100_1400\u{15} world \u{15}1400_2600\u{15} .\n*CHI:\tmhm . \u{15}2000_2400\u{15}\n%wor:\tmhm \u{15}2000_2400\u{15} .\n@End\n";
    let chat = parse_chat(input);
    assert!(
        !has_reusable_wor_timing(&chat),
        "a %wor span that runs past the next utterance start must not qualify for whole-file reuse"
    );
}

#[test]
fn test_has_reusable_wor_timing_false_when_one_word_dominates_utterance_span() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\talpha beta gamma delta .\n%wor:\talpha \u{15}100_200\u{15} beta \u{15}200_5800\u{15} gamma \u{15}5800_5900\u{15} delta \u{15}5900_6000\u{15} .\n@End\n";
    let chat = parse_chat(input);
    let utt = get_utterance(&chat, 0);
    assert!(
        !has_reusable_wor_timing_for_utterance(utt),
        "a %wor timing distribution with one word consuming most of the utterance span must not qualify for cheap reuse"
    );
}

#[test]
fn test_has_reusable_wor_timing_false_when_one_word_dominates_short_utterance_span() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\tsorry keep going .\n%wor:\tsorry \u{15}100_1401\u{15} keep \u{15}1401_1541\u{15} going \u{15}1541_1668\u{15} .\n@End\n";
    let chat = parse_chat(input);
    let utt = get_utterance(&chat, 0);
    assert!(
        !has_reusable_wor_timing_for_utterance(utt),
        "a short utterance whose first word consumes most of the span must not qualify for cheap reuse"
    );
}

#[test]
fn test_has_reusable_wor_timing_false_when_last_word_collapses_to_near_zero_duration() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\talpha beta gamma delta epsilon .\n%wor:\talpha \u{15}100_500\u{15} beta \u{15}500_900\u{15} gamma \u{15}900_1300\u{15} delta \u{15}1300_1700\u{15} epsilon \u{15}1700_1704\u{15} .\n@End\n";
    let chat = parse_chat(input);
    let utt = get_utterance(&chat, 0);
    assert!(
        !has_reusable_wor_timing_for_utterance(utt),
        "a %wor timing distribution whose final word collapses to near-zero duration must not qualify for cheap reuse"
    );
}

#[test]
fn test_refresh_existing_alignment_rehydrates_main_tier_from_wor() {
    let mut chat = parse_chat(&wor_timed_chat());
    refresh_existing_alignment(&mut chat, true);

    let output = chat.to_chat_string();
    assert!(
        output.contains("hello \u{15}100_500\u{15} world \u{15}600_1000\u{15} ."),
        "Expected refreshed main-tier word timing, got:\n{output}"
    );
    assert!(
        output.contains("%wor:\thello \u{15}100_500\u{15} world \u{15}600_1000\u{15} ."),
        "Expected refreshed %wor tier, got:\n{output}"
    );
}

#[test]
fn rebuild_policy_reconstructs_reused_utterance_bullet_from_wor_hull() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world . \u{15}300_1500\u{15}\n%wor:\thello \u{15}100_500\u{15} world \u{15}600_1000\u{15} .\n@End\n";
    let mut chat = parse_chat(input);

    refresh_existing_alignment_with_boundary_policy(
        &mut chat,
        true,
        ExistingWorBoundaryPolicy::RebuildFromEvidence,
    );

    let bullet = get_utterance(&chat, 0)
        .main
        .content
        .bullet
        .as_ref()
        .expect("reused word evidence should produce a bullet");
    assert_eq!((bullet.timing.start_ms, bullet.timing.end_ms), (100, 1000));
}

#[test]
fn test_group_utterances_includes_untimed_with_interpolation() {
    let input =
        include_str!("../../../../../../test-fixtures/fa_mixed_timed_untimed_interleaved.cha");
    let chat = parse_chat(input);
    let groups = group_utterances(&chat, 20000, &test_recording(50000)).groups;

    // All 6 utterances should be included (none skipped)
    let total_utts: usize = groups.iter().map(|g| g.utterance_indices.len()).sum();
    assert_eq!(total_utts, 6);
}
