//! Utterance grouping (`group_utterances_*`), `%wor` extraction policy, and reusable-wor detection (`has_reusable_wor_timing_*`, `refresh_existing_alignment_*`).

#![allow(unused_imports, dead_code)]

use super::*;

use talkbank_model::UtteranceIdx;
use talkbank_model::model::{Line, UtteranceContent, WriteChat};
use talkbank_parser::TreeSitterParser;

#[test]
fn merged_candidate_retains_each_source_window_and_its_words_before_padding() {
    let chat = parse_chat(include_str!(
        "../../../../../../test-fixtures/fa_two_timed_utterances.cha"
    ));
    let grouped = group_utterances(
        &chat,
        crate::types::engines::FaEngineName::Wave2Vec.aggregation_target(),
        &test_recording(20_000),
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
    );
    assert_eq!(grouped.groups.len(), 1);
    let group = &grouped.groups[0];
    assert_eq!(group.audio_end_ms(), 11_500, "only the candidate is padded");
    let GroupSpan::Single(single) = group.span() else {
        panic!("two source utterances form one candidate");
    };
    let retained: Vec<_> = single
        .source_utterances()
        .map(|source| {
            (
                source.utterance(),
                source.window().audio_start().get(),
                source.window().end().get(),
                source
                    .words()
                    .iter()
                    .map(|word| word.text.as_str())
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    assert_eq!(
        retained,
        vec![
            (UtteranceIdx::new(0), 0, 5_000, vec!["hello", "world"]),
            (
                UtteranceIdx::new(1),
                5_000,
                10_000,
                vec!["I", "want", "cookie"]
            ),
        ]
    );
    let evidence = serde_json::to_value(group.source_windows().unwrap()).unwrap();
    assert_eq!(evidence[0]["localization"]["kind"], "working_bullet");
    assert_eq!(evidence[0]["start_ms"], 0);
    assert_eq!(evidence[0]["end_ms"], 5_000);
    assert_eq!(evidence[1]["localization"]["kind"], "working_bullet");
    assert_eq!(evidence[1]["start_ms"], 5_000);
    assert_eq!(evidence[1]["end_ms"], 10_000);
}

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
    let groups = group_with_declared_window_limit(
        &chat,
        20000,
        &test_recording(10_000),
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
    )
    .groups;
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].word_count(), 5); // hello world I want cookie
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
        .and_then(talkbank_model::BulletState::as_parsed_mut)
        .unwrap()
        .timing
        .end_ms = 3_035_820;
    let refused_start_ms = get_test_utterance(&mut chat, 0)
        .main
        .content
        .bullet
        .as_ref()
        .and_then(talkbank_model::BulletState::as_parsed)
        .unwrap()
        .timing
        .start_ms;
    let grouped = group_with_declared_window_limit(
        &chat,
        15_000,
        &test_recording(4_000_000),
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
    );
    assert_eq!(
        grouped.decisions.len(),
        1,
        "oversized input needs durable refusal"
    );
    // The refusal states the window and cause as data, not only as prose.
    let expected = batchalign_transform::decisions::RefusedWindow::OverBudget {
        start_ms: refused_start_ms,
        end_ms: 3_035_820,
        budget_ms: 15_000,
    };
    assert_eq!(
        grouped.decisions[0].strategy,
        batchalign_transform::decisions::DecisionStrategy::Fa(
            batchalign_transform::decisions::FaStrategy::WindowRefused(expected)
        )
    );
    assert_eq!(grouped.decisions[0].reason, expected.to_string());
    assert!(
        !grouped.decisions[0].reason.contains("msms"),
        "unit must not be doubled: {}",
        grouped.decisions[0].reason
    );
    assert_eq!(grouped.groups.len(), 1);
    assert_eq!(
        grouped.groups[0].utterance_indices(),
        vec![UtteranceIdx::new(1)]
    );
    assert_eq!(grouped.groups[0].word_count(), 3);
    assert_eq!(
        get_test_utterance(&mut chat, 0)
            .main
            .content
            .bullet
            .as_ref()
            .and_then(talkbank_model::BulletState::as_parsed)
            .unwrap()
            .timing
            .end_ms,
        3_035_820,
        "grouping must not rewrite the supplied timing to make it fit"
    );
}

/// A grouping preference must not masquerade as a model input constraint.
/// The complete, supplied window is the request candidate; execution must
/// admit its actual resource demand before it can reach the model.
#[test]
fn supplied_long_utterance_reaches_alignment_as_one_unchanged_request() {
    let chat = parse_chat(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n\
         @ID:\teng|test|CHI|||||Child|||\n\
         *CHI:\twe saw a bird .\x15100_20100\x15\n@End\n",
    );
    let grouped = group_utterances(
        &chat,
        crate::options::FaEngineName::Wav2vecCanto.aggregation_target(),
        &test_recording(20_100),
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
    );
    assert_eq!(
        grouped.groups.len(),
        1,
        "a supplied 20-second utterance must reach resource admission, not be discarded by a grouping target"
    );
    let group = &grouped.groups[0];
    assert_eq!(group.word_count(), 4);
    assert_eq!(group.audio_start_ms(), 100);
    assert_eq!(group.audio_end_ms(), 20_100);
    assert!(grouped.decisions.is_empty());
}

/// A group followed by an utterance with no localization (no bullet, no UTR
/// evidence) is not padded into it: where its speech is is not known. The
/// group used to be finished against the END OF THE RECORDING, and padded up
/// to 1.5 seconds into the next utterance's speech.
#[test]
fn a_group_before_an_unlocalized_utterance_is_not_padded_into_it() {
    let chat = parse_chat(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n\
         @ID:\teng|test|CHI|||||Child|||\n\
         *CHI:\twe saw a bird .\x151000_3000\x15\n\
         *CHI:\tit flew away .\n@End\n",
    );
    let grouped = group_utterances(
        &chat,
        crate::options::FaEngineName::Wav2vecCanto.aggregation_target(),
        &test_recording(60_000),
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
    );
    assert_eq!(
        grouped.groups.len(),
        1,
        "only the bulleted utterance is grouped"
    );
    assert_eq!(
        grouped.groups[0].audio_end_ms(),
        3_000,
        "no padding into the unlocalized utterance that follows"
    );
}

/// A bulleted utterance with no alignable word (`xxx`) still says where
/// that turn's speech is: the group before it ends there, never merged
/// across it or padded into it.
#[test]
fn a_group_is_never_merged_across_or_padded_into_a_bulleted_xxx_turn() {
    let chat = parse_chat(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child, MOT Mother\n\
         @ID:\teng|test|CHI|||||Child|||\n@ID:\teng|test|MOT|||||Mother|||\n\
         *CHI:\twe saw a bird .\x151000_3000\x15\n\
         *MOT:\txxx .\x153200_4000\x15\n\
         *CHI:\tit flew away .\x154200_6000\x15\n@End\n",
    );
    let grouped = group_utterances(
        &chat,
        crate::options::FaEngineName::Wav2vecCanto.aggregation_target(),
        &test_recording(60_000),
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
    );
    assert_eq!(grouped.groups.len(), 2, "the xxx turn separates the groups");
    assert!(
        grouped.groups[0].audio_end_ms() <= 3_200,
        "no padding into the xxx turn's speech: {}",
        grouped.groups[0].audio_end_ms()
    );
}

/// An `xxx` turn whose bullet is not a window inside the recording (here
/// past its end) bounds nothing: the group before it is not padded toward
/// the bullet's claimed start.
#[test]
fn an_xxx_turn_with_an_impossible_bullet_does_not_bound_padding() {
    let chat = parse_chat(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child, MOT Mother\n\
         @ID:\teng|test|CHI|||||Child|||\n@ID:\teng|test|MOT|||||Mother|||\n\
         *CHI:\twe saw a bird .\x151000_3000\x15\n\
         *MOT:\txxx .\x15600000_600500\x15\n@End\n",
    );
    let grouped = group_utterances(
        &chat,
        crate::options::FaEngineName::Wav2vecCanto.aggregation_target(),
        &test_recording(60_000),
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
    );
    assert_eq!(grouped.groups[0].audio_end_ms(), 3_000);
}

/// A following utterance whose window the grouping itself refuses as
/// inverted says nothing about where its speech starts, so the group before
/// it is not padded toward its claimed start.
#[test]
fn a_group_before_a_refused_inverted_window_is_not_padded() {
    let chat = parse_chat(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n\
         @ID:\teng|test|CHI|||||Child|||\n\
         *CHI:\twe saw a bird .\x151000_3000\x15\n\
         *CHI:\tit flew away .\x159000_5000\x15\n@End\n",
    );
    let grouped = group_utterances(
        &chat,
        crate::options::FaEngineName::Wav2vecCanto.aggregation_target(),
        &test_recording(60_000),
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
    );
    assert_eq!(
        grouped.groups[0].audio_end_ms(),
        3_000,
        "no padding toward the refused window's claimed start"
    );
}

/// The last group of a file is still padded toward the end of the recording.
#[test]
fn the_last_group_is_padded_toward_the_end_of_the_recording() {
    let chat = parse_chat(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n\
         @ID:\teng|test|CHI|||||Child|||\n\
         *CHI:\twe saw a bird .\x151000_3000\x15\n@End\n",
    );
    let grouped = group_utterances(
        &chat,
        crate::options::FaEngineName::Wav2vecCanto.aggregation_target(),
        &test_recording(60_000),
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
    );
    assert!(
        grouped.groups[0].audio_end_ms() > 3_000,
        "padded into the trailing silence"
    );
}

#[test]
fn group_window_budget_includes_trailing_padding() {
    let chat = parse_chat(include_str!(
        "../../../../../../test-fixtures/fa_two_timed_utterances.cha"
    ));
    let grouped = group_with_declared_window_limit(
        &chat,
        10_000,
        &test_recording(20_000),
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
    );
    assert_eq!(grouped.groups.len(), 1);
    assert_eq!(grouped.groups[0].audio_start_ms(), 0);
    assert_eq!(grouped.groups[0].audio_end_ms(), 10_000);
    assert_eq!(grouped.groups[0].word_count(), 5);
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
        .and_then(talkbank_model::BulletState::as_parsed_mut)
        .unwrap()
        .timing
        .end_ms = 10_000;
    get_test_utterance(&mut chat, 1)
        .main
        .content
        .bullet
        .as_mut()
        .and_then(talkbank_model::BulletState::as_parsed_mut)
        .unwrap()
        .timing
        .end_ms = 6_000;
    let grouped = group_with_declared_window_limit(
        &chat,
        10_000,
        &test_recording(10_000),
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
    );
    assert_eq!(grouped.groups.len(), 1);
    assert_eq!(grouped.groups[0].audio_end_ms(), 10_000);
    assert_eq!(grouped.groups[0].word_count(), 5);
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
    let groups = group_with_declared_window_limit(
        &chat,
        20000,
        &test_recording(600_000),
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
    )
    .groups;
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].word_count(), 1);
    assert_eq!(groups[1].word_count(), 1);
}

#[test]
fn test_group_utterances_splits_on_time() {
    let input = include_str!("../../../../../../test-fixtures/fa_split_on_time.cha");
    let chat = parse_chat(input);
    let groups = group_with_declared_window_limit(
        &chat,
        20000,
        &test_recording(600_000),
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
    )
    .groups;
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].word_count(), 1);
    assert_eq!(groups[1].word_count(), 1);
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
    let groups = group_with_declared_window_limit(
        &chat,
        60_000,
        &test_recording(10_000),
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
    )
    .groups;
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
        let bytes: usize = group.words().map(|w| w.text.len()).sum();
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
    let groups = group_with_declared_window_limit(
        &chat,
        60_000,
        &test_recording(10_000),
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
    )
    .groups;
    let chars: usize = groups
        .iter()
        .flat_map(|group| group.words())
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
        let bytes: usize = group.words().map(|word| word.text.len()).sum();
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
    let groups = group_with_declared_window_limit(
        &chat,
        60_000,
        &test_recording(10_000),
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
    )
    .groups;
    assert_eq!(
        groups.len(),
        1,
        "one utterance is never split into two groups"
    );
    let bytes: usize = groups[0].words().map(|word| word.text.len()).sum();
    assert!(
        bytes > MAX_GROUP_LABEL_BYTES,
        "the fixture must actually exceed the cap for this limit to be pinned, \
         got {bytes} bytes"
    );
}

/// Supplied timing authorizes a request; an unsupported neighbour stays untimed.
#[test]
fn test_group_utterances_requests_only_localized_words() {
    let input = include_str!("../../../../../../test-fixtures/fa_mixed_timed_untimed.cha");
    let chat = parse_chat(input);
    let groups = group_with_declared_window_limit(
        &chat,
        20000,
        &test_recording(10_000),
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
    )
    .groups;
    let grouped_words: usize = groups.iter().map(|g| g.word_count()).sum();
    assert_eq!(
        grouped_words, 1,
        "only supplied timing authorizes a request"
    );
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
        .and_then(talkbank_model::BulletState::as_parsed)
        .expect("reused word evidence should produce a bullet");
    assert_eq!((bullet.timing.start_ms, bullet.timing.end_ms), (100, 1000));
}

#[test]
fn test_group_utterances_does_not_interpolate_missing_localization() {
    let input =
        include_str!("../../../../../../test-fixtures/fa_mixed_timed_untimed_interleaved.cha");
    let chat = parse_chat(input);
    let groups = group_with_declared_window_limit(
        &chat,
        20000,
        &test_recording(50000),
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
    )
    .groups;

    // The three untimed utterances have no source-bound search evidence.
    let total_utts: usize = groups.iter().map(|g| g.utterance_indices().len()).sum();
    assert_eq!(total_utts, 3);
}
