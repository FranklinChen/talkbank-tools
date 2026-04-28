//! Tests for forced alignment module.

use super::*;
use talkbank_model::model::{Line, UtteranceContent, WriteChat};
use talkbank_parser::TreeSitterParser;

fn parse_chat(text: &str) -> talkbank_model::model::ChatFile {
    let parser = TreeSitterParser::new().unwrap();
    parser.parse_chat_file(text).unwrap()
}

fn get_test_utterance(
    chat: &mut talkbank_model::model::ChatFile,
    idx: usize,
) -> &mut talkbank_model::model::Utterance {
    let mut utt_idx = 0;
    for line in &mut chat.lines {
        if let Line::Utterance(utt) = line {
            if utt_idx == idx {
                return utt;
            }
            utt_idx += 1;
        }
    }
    panic!("Utterance {idx} not found");
}

fn get_utterance(
    chat: &talkbank_model::model::ChatFile,
    idx: usize,
) -> &talkbank_model::model::Utterance {
    let mut utt_idx = 0;
    for line in &chat.lines {
        if let Line::Utterance(utt) = line {
            if utt_idx == idx {
                return utt;
            }
            utt_idx += 1;
        }
    }
    panic!("Utterance {idx} not found");
}

fn wor_timed_chat() -> String {
    "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%wor:\thello \u{15}100_500\u{15} world \u{15}600_1000\u{15} .\n@End\n".to_string()
}

fn proof_chat(main: &str) -> String {
    format!(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\t{main}\n@End\n"
    )
}

fn collect_proof_fa_words(main: &str) -> Vec<String> {
    let chat = parse_chat(&proof_chat(main));
    let utt = get_utterance(&chat, 0);
    let mut out = Vec::new();
    collect_fa_words(&utt.main.content.content, &mut out);
    out
}

fn generate_proof_wor_words(main: &str) -> Vec<String> {
    let chat = parse_chat(&proof_chat(main));
    let utt = get_utterance(&chat, 0);
    utt.main
        .generate_wor_tier()
        .words()
        .map(|word| word.cleaned_text().to_string())
        .collect()
}

fn words(items: &[&str]) -> Vec<String> {
    items.iter().map(|item| (*item).to_string()).collect()
}

#[test]
fn test_group_utterances_single_group() {
    let input = include_str!("../../../../test-fixtures/fa_two_timed_utterances.cha");
    let chat = parse_chat(input);
    let groups = group_utterances(&chat, 20000, None);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].words.len(), 5); // hello world I want cookie
    assert_eq!(groups[0].audio_start_ms(), 0);
    assert_eq!(groups[0].audio_end_ms(), 10000);
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

/// Fillers (`&-`) have stable, alignable phoneme sequences and appear in both
/// FA extraction and `%wor` output. Phonological fragments (`&+`) and
/// nonwords (`&~`) are excluded — they match BA2's `TokenType.ANNOT` policy.
/// Untranscribed tokens (`xxx`, `yyy`, `www`) are also excluded.
#[test]
fn test_wor_policy_standalone_spoken_tokens_match_between_fa_extraction_and_wor_generation() {
    for (main, expected) in [
        // Fillers (&-) ARE included
        ("&-um play .", &["um", "play"][..]),
        // Fragments (&+) are excluded — BA2 TokenType.ANNOT
        ("&+ss play .", &["play"][..]),
        // Nonwords (&~) are excluded — BA2 TokenType.ANNOT
        ("&~um play .", &["play"][..]),
    ] {
        let expected = words(expected);
        assert_eq!(collect_proof_fa_words(main), expected);
        assert_eq!(generate_proof_wor_words(main), expected);
    }
}

/// Retraced content is included in `%wor` (the speaker produced it), but
/// fragments (`&+`) and nonwords (`&~`) inside retraces are still excluded —
/// the exclusion is by word category, not by retrace ancestry.
/// Retraced fillers (`&-`) remain included. Untranscribed tokens (`xxx`,
/// `yyy`, `www`) are excluded regardless of retrace context.
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
    let input = include_str!("../../../../test-fixtures/fa_backwards_bullets.cha");
    let chat = parse_chat(input);
    let groups = group_utterances(&chat, 20000, None);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].words.len(), 1);
    assert_eq!(groups[1].words.len(), 1);
}

#[test]
fn test_group_utterances_splits_on_time() {
    let input = include_str!("../../../../test-fixtures/fa_split_on_time.cha");
    let chat = parse_chat(input);
    let groups = group_utterances(&chat, 20000, None);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].words.len(), 1);
    assert_eq!(groups[1].words.len(), 1);
}

/// `%wor` and FA extraction exclude three categories:
/// 1. Untranscribed (`xxx`/`yyy`/`www`) — no phoneme sequence; cannot align
/// 2. Fragments (`&+`) — BA2 excluded these (TokenType.ANNOT)
/// 3. Nonwords (`&~`) — BA2 excluded these (TokenType.ANNOT)
///
/// Fillers (`&-`) are INCLUDED — BA2 included these (TokenType.FP).
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

/// Regression test: Whisper CTC FA crashes when the total character count in a
/// group exceeds 448 tokens (its hard decoder limit). Groups must be split by
/// char count, not only by time window.
///
/// DiazCollazos/09.cha triggered this with group 0 = 2043 chars in 11 seconds.
#[test]
fn test_group_utterances_splits_on_whisper_token_limit() {
    // Two utterances each with 50 five-character words = 250 chars per utterance.
    // Combined = 500 chars > WHISPER_FA_MAX_LABEL_TOKENS (448).
    // Both fit in a 60-second window, so without the char limit they'd be one group.
    let fifty_words = vec!["abcde"; 50].join(" ");
    let chat_text = format!(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|test|CHI|||||Child|||\n*CHI:\t{fifty_words} .\x15100_5000\x15\n*CHI:\t{fifty_words} .\x155000_10000\x15\n@End\n"
    );
    let chat = parse_chat(&chat_text);
    let groups = group_utterances(&chat, 60_000, Some(10_000));
    assert_eq!(
        groups.len(),
        2,
        "expected 2 groups (50+50 words × 5 chars = 500 > 448 token limit), got {}",
        groups.len()
    );
    // Every group must stay within the Whisper token limit.
    for (i, group) in groups.iter().enumerate() {
        let chars: usize = group.words.iter().map(|w| w.text.len()).sum();
        assert!(
            chars <= WHISPER_FA_MAX_LABEL_TOKENS,
            "group {i} has {chars} chars, exceeds {WHISPER_FA_MAX_LABEL_TOKENS} token limit"
        );
    }
}

#[test]
fn test_group_utterances_skips_untimed() {
    let input = include_str!("../../../../test-fixtures/fa_mixed_timed_untimed.cha");
    let chat = parse_chat(input);
    let groups = group_utterances(&chat, 20000, None);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].words.len(), 1); // only "world"
}

#[test]
fn test_inject_timings_simple() {
    let input = include_str!("../../../../test-fixtures/fa_hello_world_timed.cha");
    let mut chat = parse_chat(input);
    let utt = get_test_utterance(&mut chat, 0);

    let timings = vec![
        Some(WordTiming {
            start_ms: 100,
            end_ms: 500,
        }),
        Some(WordTiming {
            start_ms: 600,
            end_ms: 1000,
        }),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);
    assert_eq!(offset, 2);

    let utt = get_test_utterance(&mut chat, 0);
    let items = &utt.main.content.content;
    match &items[0] {
        UtteranceContent::Word(w) => {
            assert!(
                w.inline_bullet.is_some(),
                "Expected inline_bullet to be set"
            );
        }
        _ => panic!("Expected word"),
    }
}

#[test]
fn test_fa_cache_key() {
    let words = vec!["hello".to_string(), "world".to_string()];
    let key = cache_key(
        &words,
        &AudioIdentity::from_metadata("test.mp3", 1234, 5678),
        0,
        5000,
        FaTimingMode::WithPauses,
        FaEngineType::WhisperFa,
    );
    // Verify it's a valid hex BLAKE3 (64 chars)
    assert_eq!(key.as_str().len(), 64);
    assert!(key.as_str().chars().all(|c| c.is_ascii_hexdigit()));

    // Same inputs -> same key
    let key2 = cache_key(
        &words,
        &AudioIdentity::from_metadata("test.mp3", 1234, 5678),
        0,
        5000,
        FaTimingMode::WithPauses,
        FaEngineType::WhisperFa,
    );
    assert_eq!(key, key2);

    // Different timing mode -> different key
    let key3 = cache_key(
        &words,
        &AudioIdentity::from_metadata("test.mp3", 1234, 5678),
        0,
        5000,
        FaTimingMode::Continuous,
        FaEngineType::WhisperFa,
    );
    assert_ne!(key, key3);
}

#[test]
fn test_apply_fa_results() {
    let input = include_str!("../../../../test-fixtures/fa_hello_world_goodbye_timed.cha");
    let mut chat = parse_chat(input);

    let groups = vec![FaGroup {
        audio_span: TimeSpan::new(0, 10000),
        words: vec![
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(0),
                text: "hello".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(1),
                text: "world".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx(1),
                utterance_word_index: WordIdx(0),
                text: "goodbye".into(),
            },
        ],
        utterance_indices: vec![UtteranceIdx(0), UtteranceIdx(1)],
    }];

    let responses = vec![vec![
        Some(WordTiming {
            start_ms: 100,
            end_ms: 1000,
        }),
        Some(WordTiming {
            start_ms: 1500,
            end_ms: 3000,
        }),
        Some(WordTiming {
            start_ms: 5500,
            end_ms: 8000,
        }),
    ]];

    apply_fa_results(
        &mut chat,
        &groups,
        &responses,
        FaTimingMode::WithPauses,
        true,
    );

    let output = chat.to_chat_string();
    assert!(output.contains("%wor:"), "Output should contain %wor tier");
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
fn test_monotonicity_enforcement() {
    let input = include_str!("../../../../test-fixtures/fa_non_monotonic_bullets.cha");
    let mut chat = parse_chat(input);
    let decisions = enforce_monotonicity(&mut chat);

    // Second utterance (start=2000) is before first (start=5000) -- should be stripped
    let utt = get_test_utterance(&mut chat, 1);
    assert!(
        utt.main.content.bullet.is_none(),
        "Non-monotonic utterance should have timing stripped"
    );

    // Should produce a decision record for the stripped utterance
    assert_eq!(
        decisions.len(),
        1,
        "should have 1 decision for stripped utterance"
    );
    assert_eq!(decisions[0].strategy.strategy_name(), "timing_stripped");
    assert!(decisions[0].needs_review);
}

/// `enforce_monotonicity` clamps end times that overlap the next utterance's
/// start, preventing the systematic ~1000ms overlap from UTR's independent
/// per-utterance token range assignment.
#[test]
fn test_monotonicity_clamps_overlapping_end_times() {
    let input = include_str!("../../../../test-fixtures/fa_overlapping_end_times.cha");
    let mut chat = parse_chat(input);
    let decisions = enforce_monotonicity(&mut chat);

    // Utterance 0: start=1000, original end=5000, next start=4000 → clamped to 4000
    let utt0 = get_test_utterance(&mut chat, 0);
    let b0 = utt0
        .main
        .content
        .bullet
        .as_ref()
        .expect("utt0 should keep timing");
    assert_eq!(
        b0.timing.end_ms, 4000,
        "utt0 end should be clamped to utt1 start"
    );

    // Utterance 1: start=4000, original end=8000, next start=7000 → clamped to 7000
    let utt1 = get_test_utterance(&mut chat, 1);
    let b1 = utt1
        .main
        .content
        .bullet
        .as_ref()
        .expect("utt1 should keep timing");
    assert_eq!(
        b1.timing.end_ms, 7000,
        "utt1 end should be clamped to utt2 start"
    );

    // Utterance 2: last utterance, no successor → end unchanged at 12000
    let utt2 = get_test_utterance(&mut chat, 2);
    let b2 = utt2
        .main
        .content
        .bullet
        .as_ref()
        .expect("utt2 should keep timing");
    assert_eq!(b2.timing.end_ms, 12000, "last utt end should be unchanged");

    // Should produce 2 end_clamped decisions (utt0→utt1, utt1→utt2)
    let clamp_decisions: Vec<_> = decisions
        .iter()
        .filter(|d| d.strategy.strategy_name() == "end_clamped")
        .collect();
    assert_eq!(
        clamp_decisions.len(),
        2,
        "should have 2 end_clamped decisions"
    );
    // `end_clamped` is routine housekeeping — a few-millisecond UTR overlap
    // correction.  It must NOT set needs_review because that writes %xrev: [?],
    // causing CLAN to flag a correctly-aligned utterance for human review.
    // BA2 made these same adjustments silently.  Only `timing_stripped` (where
    // the utterance lost all timing) deserves a human review flag.
    //
    // CURRENTLY RED: needs_review is true, writing %xrev on every trimmed utt.
    // AFTER FIX: needs_review is false; %xalign is still written (audit log)
    // but no %xrev appears.
    assert!(
        !clamp_decisions[0].needs_review,
        "end_clamped must NOT need review — it is routine overlap correction, \
         not an alignment defect requiring human inspection"
    );
}

fn make_fa_words(texts: &[&str]) -> Vec<FaWord> {
    texts
        .iter()
        .enumerate()
        .map(|(i, t)| FaWord {
            utterance_index: UtteranceIdx(0),
            utterance_word_index: WordIdx(i),
            text: t.to_string(),
        })
        .collect()
}

#[test]
fn test_parse_fa_response_token_level() {
    let json = r#"{"tokens": [
            {"text": "hello", "time_s": 0.1},
            {"text": "world", "time_s": 0.6}
        ]}"#;
    let words = make_fa_words(&["hello", "world"]);
    let timings = parse_fa_response(json, &words, 0, FaTimingMode::Continuous).unwrap();
    assert_eq!(timings.len(), 2);
    assert_eq!(
        timings[0],
        Some(WordTiming {
            start_ms: 100,
            end_ms: 100
        })
    );
    assert_eq!(
        timings[1],
        Some(WordTiming {
            start_ms: 600,
            end_ms: 600
        })
    );
}

#[test]
fn test_parse_fa_response_token_level_punctuation_token_is_ignored() {
    let json = r#"{"tokens": [
            {"text": "hello", "time_s": 0.1},
            {"text": ",", "time_s": 0.2},
            {"text": "world", "time_s": 0.6}
        ]}"#;
    let words = make_fa_words(&["hello", "world"]);
    let timings = parse_fa_response(json, &words, 3000, FaTimingMode::Continuous).unwrap();
    assert_eq!(
        timings[0],
        Some(WordTiming {
            start_ms: 3100,
            end_ms: 3100
        })
    );
    assert_eq!(
        timings[1],
        Some(WordTiming {
            start_ms: 3600,
            end_ms: 3600
        })
    );
}

#[test]
fn test_parse_fa_response_token_level_mismatch_does_not_skip_tokens() {
    let json = r#"{"tokens": [
            {"text": "hello", "time_s": 0.1},
            {"text": "there", "time_s": 0.2},
            {"text": "world", "time_s": 0.6}
        ]}"#;
    let words = make_fa_words(&["hello", "world"]);
    let timings = parse_fa_response(json, &words, 0, FaTimingMode::Continuous).unwrap();
    assert_eq!(
        timings[0],
        Some(WordTiming {
            start_ms: 100,
            end_ms: 100
        })
    );
    assert_eq!(timings[1], None);
}

#[test]
fn test_parse_fa_response_indexed_word_level() {
    let json = r#"{"indexed_timings": [
            {"start_ms": 100, "end_ms": 500},
            {"start_ms": 600, "end_ms": 1000}
        ]}"#;
    let words = make_fa_words(&["hello", "world"]);
    let timings = parse_fa_response(json, &words, 5000, FaTimingMode::Continuous).unwrap();
    assert_eq!(timings.len(), 2);
    assert_eq!(timings[0].as_ref().unwrap().start_ms, 5100);
    assert_eq!(timings[0].as_ref().unwrap().end_ms, 5500);
    assert_eq!(timings[1].as_ref().unwrap().start_ms, 5600);
    assert_eq!(timings[1].as_ref().unwrap().end_ms, 6000);
}

#[test]
fn test_parse_fa_response_indexed_length_mismatch_rejected() {
    use crate::fa::alignment::FaAlignmentError;
    let json = r#"{"indexed_timings": [{"start_ms": 100, "end_ms": 500}]}"#;
    let words = make_fa_words(&["hello", "world"]);
    let err = parse_fa_response(json, &words, 0, FaTimingMode::Continuous).unwrap_err();
    // Wave 5 consolidation: typed error replaces the previous stringly
    // "length mismatch" substring check. Assert on the variant shape so
    // a refactor that re-introduces a stringly path fails loudly.
    match err {
        FaAlignmentError::IndexedCountMismatch { expected, actual } => {
            assert_eq!(expected, 2);
            assert_eq!(actual, 1);
        }
        other => panic!("expected IndexedCountMismatch, got {other:?}"),
    }
}

#[test]
fn test_estimate_boundaries_proportional() {
    let input = include_str!("../../../../test-fixtures/fa_two_untimed_with_media.cha");
    let chat = parse_chat(input);
    let estimates = estimate_untimed_boundaries(&chat, 10000);
    assert_eq!(estimates.len(), 2);
    assert_eq!(estimates[0].start_ms, 0);
    assert_eq!(estimates[0].end_ms, 7000);
    assert_eq!(estimates[1].start_ms, 3000);
    assert_eq!(estimates[1].end_ms, 10000);
}

/// Demonstrates the interleaved timed/untimed boundary-estimation bug that
/// caused real alignment failures in hand-edited transcripts.
///
/// When timed and untimed utterances are interleaved, untimed utterances must
/// be estimated by interpolating between neighboring timed utterances — NOT
/// by distributing proportionally across the entire audio duration.
///
/// The old proportional algorithm placed untimed utterance 1 (between timed
/// utts at 10-15s and 20-25s) at ~6-18s based on word-count ratio over the
/// full 50s audio. The correct window is 13-22s (the gap between neighbors
/// plus buffer). The wrong window caused the FA model to search the wrong
/// audio segment, producing missing or collapsed timing.
#[test]
fn test_estimate_boundaries_interpolates_from_neighbors() {
    let input = include_str!("../../../../test-fixtures/fa_mixed_timed_untimed_interleaved.cha");
    let chat = parse_chat(input);
    let estimates = estimate_untimed_boundaries(&chat, 50000);

    // 6 utterances total
    assert_eq!(estimates.len(), 6);

    // utt 0: timed (10000-15000), estimate mirrors real bullet
    assert_eq!(estimates[0], TimeSpan::new(10000, 15000));

    // utt 1: untimed, between timed utt 0 (end=15000) and utt 2 (start=20000)
    // Gap = [15000, 20000], 4 words, only utterance in run
    // raw: 15000-20000, with 2s buffer: 13000-22000
    assert_eq!(estimates[1].start_ms, 13000);
    assert_eq!(estimates[1].end_ms, 22000);

    // utt 2: timed (20000-25000)
    assert_eq!(estimates[2], TimeSpan::new(20000, 25000));

    // utt 3: untimed, in run [3,4] between timed utt 2 (end=25000) and utt 5 (start=40000)
    // Gap = [25000, 40000] = 15000ms, run_words = 4+5 = 9
    // utt 3 (4 words): raw 25000..31666, buffered 23000..33666
    assert_eq!(estimates[3].start_ms, 23000);
    assert_eq!(estimates[3].end_ms, 33666);

    // utt 4 (5 words): raw 31666..40000, buffered 29666..42000
    assert_eq!(estimates[4].start_ms, 29666);
    assert_eq!(estimates[4].end_ms, 42000);

    // utt 5: timed (40000-45000)
    assert_eq!(estimates[5], TimeSpan::new(40000, 45000));
}

/// Ensure ALL utterances (timed and untimed) are included in groups
/// when `total_audio_ms` is provided.
#[test]
fn test_group_utterances_includes_untimed_with_interpolation() {
    let input = include_str!("../../../../test-fixtures/fa_mixed_timed_untimed_interleaved.cha");
    let chat = parse_chat(input);
    let groups = group_utterances(&chat, 20000, Some(50000));

    // All 6 utterances should be included (none skipped)
    let total_utts: usize = groups.iter().map(|g| g.utterance_indices.len()).sum();
    assert_eq!(total_utts, 6);
}

/// Two utterances: first has clean %wor, second has stale %wor (word count mismatch).
/// `find_reusable_utterance_indices` should return only the first.
#[test]
fn test_find_reusable_utterance_indices_mixed_clean_stale() {
    // Utterance 0: "hello world ." with matching %wor (2 words) → reusable
    // Utterance 1: "goodbye my friend ." with stale %wor (1 word) → word count mismatch → stale
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%wor:\thello \u{15}100_500\u{15} world \u{15}600_1000\u{15} .\n*CHI:\tgoodbye my friend .\n%wor:\tgoodbye \u{15}1500_2000\u{15} .\n@End\n";
    let chat = parse_chat(input);

    let reusable = find_reusable_utterance_indices(&chat);
    assert!(reusable.contains(&0), "utterance 0 should be reusable");
    assert!(
        !reusable.contains(&1),
        "utterance 1 should be stale (word count mismatch)"
    );
    assert_eq!(reusable.len(), 1);
}

/// All utterances have clean %wor → all should be reusable.
#[test]
fn test_find_reusable_utterance_indices_all_clean() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%wor:\thello \u{15}100_500\u{15} world \u{15}600_1000\u{15} .\n*CHI:\tgoodbye .\n%wor:\tgoodbye \u{15}1500_2000\u{15} .\n@End\n";
    let chat = parse_chat(input);

    let reusable = find_reusable_utterance_indices(&chat);
    assert_eq!(reusable.len(), 2);
    assert!(reusable.contains(&0));
    assert!(reusable.contains(&1));
}

#[test]
fn test_find_reusable_utterance_indices_excludes_wor_overrun_past_next_start() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world . \u{15}1000_1500\u{15}\n%wor:\thello \u{15}1100_1400\u{15} world \u{15}1400_2600\u{15} .\n*CHI:\tmhm . \u{15}2000_2400\u{15}\n%wor:\tmhm \u{15}2000_2400\u{15} .\n@End\n";
    let chat = parse_chat(input);

    let reusable = find_reusable_utterance_indices(&chat);
    assert!(
        !reusable.contains(&0),
        "utterance 0 should be excluded because its reused %wor span overruns the next start"
    );
    assert!(
        reusable.contains(&1),
        "utterance 1 should remain reusable because it has no following start to overrun"
    );
    assert_eq!(reusable.len(), 1);
}

#[test]
fn test_find_reusable_utterance_indices_excludes_last_word_near_zero_duration() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\talpha beta .\n%wor:\talpha \u{15}100_400\u{15} beta \u{15}400_700\u{15} .\n*CHI:\talpha beta gamma delta epsilon .\n%wor:\talpha \u{15}1000_1400\u{15} beta \u{15}1400_1800\u{15} gamma \u{15}1800_2200\u{15} delta \u{15}2200_2600\u{15} epsilon \u{15}2600_2604\u{15} .\n*CHI:\tzeta eta .\n%wor:\tzeta \u{15}3000_3300\u{15} eta \u{15}3300_3600\u{15} .\n@End\n";
    let chat = parse_chat(input);

    let reusable = find_reusable_utterance_indices(&chat);
    assert!(reusable.contains(&0), "utterance 0 should remain reusable");
    assert!(
        !reusable.contains(&1),
        "utterance 1 should be excluded because its final word collapses to near-zero duration"
    );
    assert!(reusable.contains(&2), "utterance 2 should remain reusable");
    assert_eq!(reusable.len(), 2);
    assert!(
        !has_reusable_wor_timing(&chat),
        "the whole file must not take the all-reusable fast path when one utterance has a collapsed final word"
    );
}

#[test]
fn test_find_reusable_utterance_indices_excludes_short_utterance_last_word_near_zero_duration() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%wor:\thello \u{15}100_400\u{15} world \u{15}400_700\u{15} .\n*CHI:\talright thank you .\n%wor:\talright \u{15}1000_1500\u{15} thank \u{15}1500_1900\u{15} you \u{15}1900_1904\u{15} .\n*CHI:\tgoodbye now .\n%wor:\tgoodbye \u{15}2400_2800\u{15} now \u{15}2800_3200\u{15} .\n@End\n";
    let chat = parse_chat(input);

    let reusable = find_reusable_utterance_indices(&chat);
    assert!(reusable.contains(&0), "utterance 0 should remain reusable");
    assert!(
        !reusable.contains(&1),
        "utterance 1 should be excluded even though it is short because its final word collapses to near-zero duration"
    );
    assert!(reusable.contains(&2), "utterance 2 should remain reusable");
    assert_eq!(reusable.len(), 2);
    assert!(
        !has_reusable_wor_timing(&chat),
        "the whole file must not take the all-reusable fast path when a short utterance has a collapsed final word"
    );
}

#[test]
fn test_find_reusable_utterance_indices_excludes_internal_word_near_zero_duration() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%wor:\thello \u{15}100_400\u{15} world \u{15}400_700\u{15} .\n*CHI:\tI have to unscrew the top .\n%wor:\tI \u{15}1000_1007\u{15} have \u{15}1007_1187\u{15} to \u{15}1187_1327\u{15} unscrew \u{15}1327_1768\u{15} the \u{15}1768_1908\u{15} top \u{15}1908_2149\u{15} .\n*CHI:\tgoodbye now .\n%wor:\tgoodbye \u{15}2400_2800\u{15} now \u{15}2800_3200\u{15} .\n@End\n";
    let chat = parse_chat(input);

    let reusable = find_reusable_utterance_indices(&chat);
    assert!(reusable.contains(&0), "utterance 0 should remain reusable");
    assert!(
        !reusable.contains(&1),
        "utterance 1 should be excluded because an internal word collapses to near-zero duration"
    );
    assert!(reusable.contains(&2), "utterance 2 should remain reusable");
    assert_eq!(reusable.len(), 2);
    assert!(
        !has_reusable_wor_timing(&chat),
        "the whole file must not take the all-reusable fast path when a rerun keeps an internal collapsed word"
    );
}

#[test]
fn test_find_reusable_utterance_indices_excludes_short_utterance_dominance() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%wor:\thello \u{15}100_400\u{15} world \u{15}400_700\u{15} .\n*CHI:\tsorry keep going .\n%wor:\tsorry \u{15}1000_2301\u{15} keep \u{15}2301_2441\u{15} going \u{15}2441_2568\u{15} .\n*CHI:\tgoodbye now .\n%wor:\tgoodbye \u{15}3000_3400\u{15} now \u{15}3400_3800\u{15} .\n@End\n";
    let chat = parse_chat(input);

    let reusable = find_reusable_utterance_indices(&chat);
    assert!(reusable.contains(&0), "utterance 0 should remain reusable");
    assert!(
        !reusable.contains(&1),
        "utterance 1 should be excluded because one word dominates a short utterance span"
    );
    assert!(reusable.contains(&2), "utterance 2 should remain reusable");
    assert_eq!(reusable.len(), 2);
    assert!(
        !has_reusable_wor_timing(&chat),
        "the whole file must not take the all-reusable fast path when a short utterance has a dominant word"
    );
}

/// No utterances have %wor → empty set.
#[test]
fn test_find_reusable_utterance_indices_no_wor() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n*CHI:\tgoodbye .\n@End\n";
    let chat = parse_chat(input);

    let reusable = find_reusable_utterance_indices(&chat);
    assert!(reusable.is_empty());
}

/// `refresh_reusable_utterances` refreshes only the reusable utterances and
/// leaves stale ones untouched.
#[test]
fn test_refresh_reusable_utterances_selective() {
    // Utterance 0: clean %wor, utterance 1: no %wor (stale)
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n%wor:\thello \u{15}100_500\u{15} world \u{15}600_1000\u{15} .\n*CHI:\tgoodbye .\n@End\n";
    let mut chat = parse_chat(input);

    let reusable: std::collections::HashSet<usize> = [0].into_iter().collect();
    orchestrate::refresh_reusable_utterances(&mut chat, &reusable, true);

    let output = chat.to_chat_string();
    // Utterance 0 should have refreshed word timing
    assert!(
        output.contains("hello \u{15}100_500\u{15} world \u{15}600_1000\u{15} ."),
        "Expected refreshed main-tier timing for utt 0, got:\n{output}"
    );
    // Utterance 1 should NOT have timing (it was stale/missing %wor)
    let utt1 = get_test_utterance(&mut chat, 1);
    assert!(
        utt1.main.content.bullet.is_none(),
        "Stale utterance should not get timing from refresh"
    );
}

// ---------------------------------------------------------------------------
// Bug: update_utterance_bullet shrinks pre-existing bullets
// ---------------------------------------------------------------------------
//
// When an utterance already has a hand-linked bullet that covers fillers,
// pauses, gestures, and false starts, FA only produces word timings for
// actual speech. update_utterance_bullet() was unconditionally replacing
// the bullet with min(word_starts)..max(word_ends), shrinking it to just
// the aligned speech and losing the surrounding context.
//
// Reported by a user (2026-03-16) on ACWT corpus: "it's cutting off lots
// of stuff from the already linked lines both at the beginning and ends
// of utterances."

/// Pre-timed utterance with leading filler: bullet start must not advance.
///
/// Input: `*PAR: &-uh I went home . 37397_42983`
/// FA returns timings only for "I", "went", "home" starting at ~42000ms.
/// The original bullet starts at 37397 to cover the "&-uh" filler.
/// update_utterance_bullet must preserve 37397 as the start.
#[test]
fn test_update_utterance_bullet_preserves_start_with_leading_fillers() {
    let input = include_str!("../../../../test-fixtures/fa_pretimed_with_fillers.cha");
    let mut chat = parse_chat(input);
    let utt = get_test_utterance(&mut chat, 0);

    // Verify pre-existing bullet
    let original_bullet = utt.main.content.bullet.clone().unwrap();
    assert_eq!(original_bullet.timing.start_ms, 37397);
    assert_eq!(original_bullet.timing.end_ms, 42983);

    // Simulate FA: only "I", "went", "home" get timed (filler &-uh does not)
    let timings = vec![
        Some(WordTiming::new(42221, 42582)),
        Some(WordTiming::new(42582, 42782)),
        Some(WordTiming::new(42782, 42983)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    postprocess_utterance_timings(utt, FaTimingMode::WithPauses);
    update_utterance_bullet(utt);

    let bullet = utt.main.content.bullet.as_ref().unwrap();
    assert_eq!(
        bullet.timing.start_ms, 37397,
        "Bullet start must be preserved from original (covers leading filler), got {}",
        bullet.timing.start_ms,
    );
    assert_eq!(
        bullet.timing.end_ms, 42983,
        "Bullet end must be preserved from original, got {}",
        bullet.timing.end_ms,
    );
}

/// Pre-timed utterance with trailing gesture: bullet end must not recede.
///
/// Input: `*PAR: and it screwed up &=laughs . 50556_56221`
/// FA returns timings for "and", "it", "screwed", "up" ending at ~55898ms.
/// The original bullet ends at 56221 to cover the "&=laughs" gesture.
/// update_utterance_bullet must preserve 56221 as the end.
#[test]
fn test_update_utterance_bullet_preserves_end_with_trailing_gesture() {
    let input = include_str!("../../../../test-fixtures/fa_pretimed_trailing_gesture.cha");
    let mut chat = parse_chat(input);
    let utt = get_test_utterance(&mut chat, 0);

    // Verify pre-existing bullet
    let original_bullet = utt.main.content.bullet.clone().unwrap();
    assert_eq!(original_bullet.timing.start_ms, 50556);
    assert_eq!(original_bullet.timing.end_ms, 56221);

    // Simulate FA: "and", "it", "screwed", "up" get timed; &=laughs does not
    let timings = vec![
        Some(WordTiming::new(50616, 52596)),
        Some(WordTiming::new(52596, 54637)),
        Some(WordTiming::new(54637, 55718)),
        Some(WordTiming::new(55718, 55898)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    postprocess_utterance_timings(utt, FaTimingMode::WithPauses);
    update_utterance_bullet(utt);

    let bullet = utt.main.content.bullet.as_ref().unwrap();
    assert_eq!(
        bullet.timing.start_ms, 50556,
        "Bullet start must be preserved from original, got {}",
        bullet.timing.start_ms,
    );
    assert_eq!(
        bullet.timing.end_ms, 56221,
        "Bullet end must be preserved from original (covers trailing gesture), got {}",
        bullet.timing.end_ms,
    );
}

/// Untimed utterance (no prior bullet) should still get bullet from word timings.
///
/// This is the existing behavior that must continue to work: when there's
/// no pre-existing bullet, update_utterance_bullet sets it from the word
/// timing span.
#[test]
fn test_update_utterance_bullet_sets_new_bullet_when_none_existed() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n@Media:\ttest, audio\n*CHI:\thello world .\n@End\n";
    let mut chat = parse_chat(input);
    let utt = get_test_utterance(&mut chat, 0);

    // No pre-existing bullet
    assert!(utt.main.content.bullet.is_none());

    let timings = vec![
        Some(WordTiming::new(100, 500)),
        Some(WordTiming::new(600, 1000)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    update_utterance_bullet(utt);

    let bullet = utt.main.content.bullet.as_ref().unwrap();
    assert_eq!(bullet.timing.start_ms, 100);
    assert_eq!(bullet.timing.end_ms, 1000);
}

/// Full apply_fa_results pipeline with a pre-timed utterance containing fillers.
///
/// Verifies the end-to-end pipeline preserves the original bullet when word
/// timings cover only a subset of the utterance duration.
#[test]
fn test_apply_fa_results_preserves_pretimed_bullet() {
    let input = include_str!("../../../../test-fixtures/fa_pretimed_with_fillers.cha");
    let mut chat = parse_chat(input);

    let groups = vec![FaGroup {
        audio_span: TimeSpan::new(37397, 42983),
        words: vec![
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(0),
                text: "I".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(1),
                text: "went".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(2),
                text: "home".into(),
            },
        ],
        utterance_indices: vec![UtteranceIdx(0)],
    }];

    let responses = vec![vec![
        Some(WordTiming::new(42221, 42582)),
        Some(WordTiming::new(42582, 42782)),
        Some(WordTiming::new(42782, 42983)),
    ]];

    apply_fa_results(
        &mut chat,
        &groups,
        &responses,
        FaTimingMode::WithPauses,
        true,
    );

    let utt = get_test_utterance(&mut chat, 0);
    let bullet = utt.main.content.bullet.as_ref().unwrap();
    assert_eq!(
        bullet.timing.start_ms, 37397,
        "Pipeline must preserve original bullet start (covers leading filler), got {}",
        bullet.timing.start_ms,
    );
    assert_eq!(
        bullet.timing.end_ms, 42983,
        "Pipeline must preserve original bullet end, got {}",
        bullet.timing.end_ms,
    );
}

/// Word timings that extend beyond the original bullet should expand it.
///
/// If FA discovers speech starts earlier or ends later than the original
/// bullet, the bullet should grow to accommodate.
#[test]
fn test_update_utterance_bullet_expands_when_words_exceed_original() {
    let input = include_str!("../../../../test-fixtures/fa_pretimed_with_fillers.cha");
    let mut chat = parse_chat(input);
    let utt = get_test_utterance(&mut chat, 0);

    // Original bullet: 37397_42983
    // Simulate FA returning words that start before and end after the bullet
    let timings = vec![
        Some(WordTiming::new(37000, 38000)),
        Some(WordTiming::new(38000, 43500)),
        Some(WordTiming::new(43500, 44000)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    // Skip postprocess (it would clamp to utterance boundary) — test update only
    update_utterance_bullet(utt);

    let bullet = utt.main.content.bullet.as_ref().unwrap();
    assert_eq!(
        bullet.timing.start_ms, 37000,
        "Bullet should expand to earlier word start, got {}",
        bullet.timing.start_ms,
    );
    assert_eq!(
        bullet.timing.end_ms, 44000,
        "Bullet should expand to later word end, got {}",
        bullet.timing.end_ms,
    );
}

/// A large inherited start on a rerun with existing `%wor` must be discarded.
///
/// Re-aligning a previously broken utterance can leave the main-tier bullet
/// starting several seconds before the first timed word, even though the
/// utterance begins directly with spoken words. In that case the earlier start
/// is stale inherited timing, not real leading coverage, and update_utterance_bullet
/// must snap the start back to the FA word span.
#[test]
fn test_update_utterance_bullet_discards_large_stale_start_on_rerun_without_leading_filler() {
    let input = "\
@UTF8\n\
@Begin\n\
@Languages:\teng\n\
@Participants:\tPAR Participant\n\
@ID:\teng|test|PAR|||||Participant|||\n\
@Media:\ttest, audio\n\
*PAR:\thow did this happen ? \u{0015}2000_9970\u{0015}\n\
%wor:\thow did this happen ?\n\
@End\n\
";
    let mut chat = parse_chat(input);
    let utt = get_test_utterance(&mut chat, 0);

    let timings = vec![
        Some(WordTiming::new(9443, 9643)),
        Some(WordTiming::new(9643, 9783)),
        Some(WordTiming::new(9783, 9970)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    update_utterance_bullet(utt);

    let bullet = utt.main.content.bullet.as_ref().unwrap();
    assert_eq!(
        bullet.timing.start_ms, 9443,
        "Large stale start with no untimed leading content must be replaced by first word start, got {}",
        bullet.timing.start_ms,
    );
    assert_eq!(
        bullet.timing.end_ms, 9970,
        "Bullet end should still reflect the last word end, got {}",
        bullet.timing.end_ms,
    );
}

/// A large stale start on a rerun must also be discarded when the leading
/// filler is already timed.
///
/// In this case the filler itself is the first timed `%wor` word, so there is
/// no untimed leading coverage left to preserve. Keeping the earlier
/// authoritative start only preserves stale silence from a previous pass.
#[test]
fn test_update_utterance_bullet_discards_large_stale_start_when_leading_filler_is_timed() {
    let input = "\
@UTF8\n\
@Begin\n\
@Languages:\teng\n\
@Participants:\tPAR Participant\n\
@ID:\teng|test|PAR|||||Participant|||\n\
@Media:\ttest, audio\n\
*PAR:\t&-uh you want me to talk as I'm going ? \u{0015}3480_10590\u{0015}\n\
%wor:\tuh you want me to talk as I'm going ?\n\
@End\n\
";
    let mut chat = parse_chat(input);
    let utt = get_test_utterance(&mut chat, 0);

    let timings = vec![
        Some(WordTiming::new(8473, 8693)),
        Some(WordTiming::new(8693, 8813)),
        Some(WordTiming::new(8813, 8954)),
        Some(WordTiming::new(8954, 9074)),
        Some(WordTiming::new(9074, 9815)),
        Some(WordTiming::new(9815, 10136)),
        Some(WordTiming::new(10136, 10216)),
        Some(WordTiming::new(10216, 10437)),
        Some(WordTiming::new(10437, 10590)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    update_utterance_bullet(utt);

    let bullet = utt.main.content.bullet.as_ref().unwrap();
    assert_eq!(
        bullet.timing.start_ms, 8473,
        "Large stale start must snap to the timed leading filler, got {}",
        bullet.timing.start_ms,
    );
    assert_eq!(
        bullet.timing.end_ms, 10590,
        "Bullet end should still reflect the final word end, got {}",
        bullet.timing.end_ms,
    );
}

#[test]
fn test_postprocess_continuous_does_not_extend_word_across_implausibly_large_gap() {
    let input = "\
@UTF8\n\
@Begin\n\
@Languages:\teng\n\
@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|||||Target_Child|||\n\
@Media:\ttest, audio\n\
*CHI:\talpha beta gamma . \u{0015}0_12000\u{0015}\n\
@End\n\
";
    let mut chat = parse_chat(input);
    let utt = get_test_utterance(&mut chat, 0);

    let timings = vec![
        Some(WordTiming::new(100, 200)),
        Some(WordTiming::new(300, 400)),
        Some(WordTiming::new(10_000, 10_100)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    let dropped = postprocess_utterance_timings(utt, FaTimingMode::Continuous);
    assert_eq!(dropped, 0);

    let mut collected = Vec::new();
    postprocess::collect_word_timings(&utt.main.content.content, &mut collected);

    assert_eq!(
        collected[1],
        Some(WordTiming::new(300, 400)),
        "continuous mode must not stretch beta across a multi-second internal gap"
    );
    assert_eq!(
        collected[2],
        Some(WordTiming::new(10_000, 10_100)),
        "the final word should retain its original non-zero duration"
    );
}

#[test]
fn test_postprocess_continuous_does_not_extend_word_across_one_second_internal_gap() {
    let input = "\
@UTF8\n\
@Begin\n\
@Languages:\teng\n\
@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|||||Target_Child|||\n\
@Media:\ttest, audio\n\
*CHI:\tsorry keep going . \u{0015}4130_5860\u{0015}\n\
@End\n\
";
    let mut chat = parse_chat(input);
    let utt = get_test_utterance(&mut chat, 0);

    let timings = vec![
        Some(WordTiming::new(4265, 4465)),
        Some(WordTiming::new(5548, 5668)),
        Some(WordTiming::new(5688, 5908)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    let dropped = postprocess_utterance_timings(utt, FaTimingMode::Continuous);
    assert_eq!(dropped, 0);

    let mut collected = Vec::new();
    postprocess::collect_word_timings(&utt.main.content.content, &mut collected);

    assert_eq!(
        collected[0],
        Some(WordTiming::new(4265, 4465)),
        "continuous mode must not stretch sorry across a one-second internal gap before keep"
    );
    assert_eq!(
        collected[1],
        Some(WordTiming::new(5548, 5688)),
        "keep should retain its original pre-injection span"
    );
    assert_eq!(
        collected[2],
        Some(WordTiming::new(5688, 5908)),
        "going should retain its original non-zero duration"
    );
}

#[test]
fn test_postprocess_continuous_does_not_extend_compound_filler_across_following_gap() {
    let mut chat = parse_chat(&proof_chat(
        "+\" &-you_know that's not gonna happen . \u{0015}3287_5860\u{0015}",
    ));
    let utt = get_test_utterance(&mut chat, 0);

    // Trace from align-regression-022: FA aligns the compound filler as
    // separate "you" / "know" words, injection merges them back into one CHAT
    // token, and continuous mode must not then stretch that merged filler
    // forward into the next lexical word.
    let timings = vec![
        Some(WordTiming::new(3664, 3744)),
        Some(WordTiming::new(4446, 4646)),
        Some(WordTiming::new(5027, 5247)),
        Some(WordTiming::new(5288, 5428)),
        Some(WordTiming::new(5448, 5588)),
        Some(WordTiming::new(5689, 5949)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    let dropped = postprocess_utterance_timings(utt, FaTimingMode::Continuous);
    assert_eq!(dropped, 0);

    let mut collected = Vec::new();
    postprocess::collect_word_timings(&utt.main.content.content, &mut collected);

    assert_eq!(
        collected[0],
        Some(WordTiming::new(3664, 4646)),
        "continuous mode must not stretch merged compound filler timing into the following word gap"
    );
    assert_eq!(
        collected[1],
        Some(WordTiming::new(5027, 5288)),
        "that's should retain its original pre-injection span"
    );
}

#[test]
fn test_postprocess_continuous_does_not_extend_lexical_word_across_following_filler_gap() {
    let mut chat = parse_chat(&proof_chat("he seems &-um tired ."));
    let utt = get_test_utterance(&mut chat, 0);

    // Trace from align-regression-026: raw FA gives "seems" a normal span, then
    // continuous mode stretches it forward across the gap before a timed filler
    // word, making "seems" dominate the utterance.
    let timings = vec![
        Some(WordTiming::new(3383, 3464)),
        Some(WordTiming::new(3624, 4045)),
        Some(WordTiming::new(4527, 4687)),
        Some(WordTiming::new(4868, 5409)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    let dropped = postprocess_utterance_timings(utt, FaTimingMode::Continuous);
    assert_eq!(dropped, 0);

    let mut collected = Vec::new();
    postprocess::collect_word_timings(&utt.main.content.content, &mut collected);

    assert_eq!(
        collected[1],
        Some(WordTiming::new(3624, 4045)),
        "continuous mode must not stretch a lexical word across the following filler gap"
    );
    assert_eq!(
        collected[2],
        Some(WordTiming::new(4527, 4868)),
        "the filler may still extend to the following lexical word"
    );
}

#[test]
fn test_postprocess_continuous_does_not_extend_filler_across_gap_when_it_would_dominate_utterance()
{
    let mut chat = parse_chat(&proof_chat(
        "and &-um so , she goes to the ball . \u{0015}12990_15730\u{0015}",
    ));
    let utt = get_test_utterance(&mut chat, 0);

    // Trace from align-regression-031: raw FA gives "um" a modest filler span,
    // then continuous mode stretches it across an internal silence up to "so",
    // making the filler dominate the short utterance. Keep the filler's own span
    // instead of smoothing it into a dominant token.
    let timings = vec![
        Some(WordTiming::new(12976, 13496)),
        Some(WordTiming::new(13597, 13997)),
        Some(WordTiming::new(14878, 14978)),
        Some(WordTiming::new(15139, 15219)),
        Some(WordTiming::new(15259, 15419)),
        Some(WordTiming::new(15419, 15479)),
        Some(WordTiming::new(15499, 15579)),
        Some(WordTiming::new(15599, 15759)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    let dropped = postprocess_utterance_timings(utt, FaTimingMode::Continuous);
    assert_eq!(dropped, 0);

    let mut collected = Vec::new();
    postprocess::collect_word_timings(&utt.main.content.content, &mut collected);

    assert_eq!(
        collected[1],
        Some(WordTiming::new(13597, 13997)),
        "continuous mode must not stretch a filler across a gap when the bridged filler span would dominate the utterance"
    );
    assert_eq!(
        collected[2],
        Some(WordTiming::new(14878, 15139)),
        "the following lexical word may still extend to the next lexical boundary"
    );
}

#[test]
fn test_postprocess_continuous_heals_near_zero_lexical_word_before_filler_when_not_dominant() {
    let mut chat = parse_chat(&proof_chat(
        "and I &-um put more butter in to make sure it gets nice and brown . \u{0015}4280_9540\u{0015}",
    ));
    let utt = get_test_utterance(&mut chat, 0);

    // Trace from align-regression-027: raw FA leaves "I" at 20 ms before a
    // timed filler gap. Continuous mode should heal that near-zero lexical word
    // when extending it to the filler start still keeps the word well below the
    // dominance threshold for the utterance.
    let timings = vec![
        Some(WordTiming::new(4281, 4521)),
        Some(WordTiming::new(5022, 5042)),
        Some(WordTiming::new(6383, 6443)),
        Some(WordTiming::new(7123, 7263)),
        Some(WordTiming::new(7283, 7423)),
        Some(WordTiming::new(7444, 7684)),
        Some(WordTiming::new(7744, 7844)),
        Some(WordTiming::new(7904, 7984)),
        Some(WordTiming::new(8064, 8224)),
        Some(WordTiming::new(8244, 8404)),
        Some(WordTiming::new(8604, 8724)),
        Some(WordTiming::new(8925, 9125)),
        Some(WordTiming::new(9165, 9365)),
        Some(WordTiming::new(9385, 9465)),
        Some(WordTiming::new(9485, 9785)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    let dropped = postprocess_utterance_timings(utt, FaTimingMode::Continuous);
    assert_eq!(dropped, 0);

    let mut collected = Vec::new();
    postprocess::collect_word_timings(&utt.main.content.content, &mut collected);

    assert_eq!(
        collected[1],
        Some(WordTiming::new(5022, 6383)),
        "continuous mode should heal a near-zero lexical word by extending it to the following filler start when that bridge is not dominant"
    );
    assert_eq!(
        collected[2],
        Some(WordTiming::new(6383, 7123)),
        "the filler should still extend to the following lexical word"
    );
}

#[test]
fn test_postprocess_continuous_rebalances_near_zero_lexical_word_from_following_filler_span() {
    let mut chat = parse_chat(&proof_chat(
        "I'm not sure if that's a [/] &-um a boat that's &-like flipped up on the top of the car . \u{0015}4360_8520\u{0015}",
    ));
    let utt = get_test_utterance(&mut chat, 0);

    // Trace from align-regression-028: raw FA already lands the first lexical
    // "a" exactly on the filler boundary at only 20 ms, while continuous mode
    // stretches the following filler forward into a much larger span. The
    // lexical word should reclaim enough time from that filler span to avoid a
    // near-zero collapse.
    let timings = vec![
        Some(WordTiming::new(4323, 4403)),
        Some(WordTiming::new(4443, 4563)),
        Some(WordTiming::new(4603, 4764)),
        Some(WordTiming::new(4764, 4804)),
        Some(WordTiming::new(4824, 5004)),
        Some(WordTiming::new(5224, 5244)),
        Some(WordTiming::new(5244, 5284)),
        Some(WordTiming::new(5565, 5585)),
        Some(WordTiming::new(5805, 6105)),
        Some(WordTiming::new(6266, 6506)),
        Some(WordTiming::new(6807, 7027)),
        Some(WordTiming::new(7087, 7347)),
        Some(WordTiming::new(7387, 7488)),
        Some(WordTiming::new(7708, 7788)),
        Some(WordTiming::new(7848, 7908)),
        Some(WordTiming::new(7988, 8189)),
        Some(WordTiming::new(8229, 8289)),
        Some(WordTiming::new(8309, 8389)),
        Some(WordTiming::new(8429, 8689)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    let dropped = postprocess_utterance_timings(utt, FaTimingMode::Continuous);
    assert_eq!(dropped, 0);

    let mut collected = Vec::new();
    postprocess::collect_word_timings(&utt.main.content.content, &mut collected);

    assert_eq!(
        collected[5],
        Some(WordTiming::new(5224, 5264)),
        "continuous mode should let a collapsed lexical word reclaim the minimum 40 ms from the following filler span"
    );
    assert_eq!(
        collected[6],
        Some(WordTiming::new(5264, 5565)),
        "the filler should keep the rest of its expanded span after lending 20 ms back to the lexical word"
    );
}

#[test]
fn test_postprocess_continuous_rebalances_near_zero_lexical_word_from_following_lexical_span() {
    let mut chat = parse_chat(&proof_chat(
        "and then I will put a thin layer of horseradish on one of the pieces , a thin layer of a Thousand Island dressing on the same piece . \u{0015}5310_14570\u{0015}",
    ));
    let utt = get_test_utterance(&mut chat, 0);

    // Trace from align-regression-029: raw FA already leaves the second-clause
    // determiner "a" at 20 ms directly against the following lexical word
    // "Thousand". Continuous mode should rebalance that shared boundary so the
    // near-zero lexical word reaches the minimum duration floor instead of being
    // preserved as an implausible sliver.
    let timings = vec![
        Some(WordTiming::new(5381, 5501)),
        Some(WordTiming::new(5582, 5822)),
        Some(WordTiming::new(6202, 6222)),
        Some(WordTiming::new(6282, 6502)),
        Some(WordTiming::new(6842, 7082)),
        Some(WordTiming::new(7303, 7323)),
        Some(WordTiming::new(7623, 7823)),
        Some(WordTiming::new(7863, 8083)),
        Some(WordTiming::new(8103, 8143)),
        Some(WordTiming::new(8203, 8803)),
        Some(WordTiming::new(9284, 9364)),
        Some(WordTiming::new(9464, 9564)),
        Some(WordTiming::new(9584, 9624)),
        Some(WordTiming::new(9644, 9724)),
        Some(WordTiming::new(9784, 10264)),
        Some(WordTiming::new(10945, 10965)),
        Some(WordTiming::new(11405, 11585)),
        Some(WordTiming::new(11645, 11845)),
        Some(WordTiming::new(11865, 11925)),
        Some(WordTiming::new(11925, 11945)),
        Some(WordTiming::new(11945, 12305)),
        Some(WordTiming::new(12385, 12545)),
        Some(WordTiming::new(12565, 12926)),
        Some(WordTiming::new(13486, 13706)),
        Some(WordTiming::new(13786, 13886)),
        Some(WordTiming::new(13966, 14226)),
        Some(WordTiming::new(14266, 14687)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    let dropped = postprocess_utterance_timings(utt, FaTimingMode::Continuous);
    assert_eq!(dropped, 0);

    let mut collected = Vec::new();
    postprocess::collect_word_timings(&utt.main.content.content, &mut collected);

    assert_eq!(
        collected[19],
        Some(WordTiming::new(11925, 11965)),
        "continuous mode should let a collapsed lexical word reclaim the minimum 40 ms from the following lexical span"
    );
    assert_eq!(
        collected[20],
        Some(WordTiming::new(11965, 12385)),
        "the following lexical word should keep the rest of its span after lending 20 ms back to the collapsed word"
    );
}

#[test]
fn test_postprocess_continuous_rebalances_near_zero_lexical_word_from_preceding_filler_span() {
    let mut chat = parse_chat(&proof_chat(
        "and &-um I first let it cool a little bit . \u{0015}10510_14824\u{0015}",
    ));
    let utt = get_test_utterance(&mut chat, 0);

    // Trace from align-regression-030: raw FA leaves "I" at 20 ms after a
    // leading filler, and continuous mode expands that filler right up to the
    // collapsed lexical word. The lexical word should reclaim enough time back
    // from the preceding filler span to reach the minimum duration floor.
    let timings = vec![
        Some(WordTiming::new(10503, 10883)),
        Some(WordTiming::new(11063, 11303)),
        Some(WordTiming::new(11903, 11923)),
        Some(WordTiming::new(13184, 13584)),
        Some(WordTiming::new(13984, 14124)),
        Some(WordTiming::new(14124, 14164)),
        Some(WordTiming::new(14204, 14444)),
        Some(WordTiming::new(14504, 14524)),
        Some(WordTiming::new(14524, 14724)),
        Some(WordTiming::new(14744, 14824)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    let dropped = postprocess_utterance_timings(utt, FaTimingMode::Continuous);
    assert_eq!(dropped, 0);

    let mut collected = Vec::new();
    postprocess::collect_word_timings(&utt.main.content.content, &mut collected);

    assert_eq!(
        collected[1],
        Some(WordTiming::new(11063, 11883)),
        "continuous mode should let the preceding filler lend 20 ms back to a collapsed lexical word"
    );
    assert_eq!(
        collected[2],
        Some(WordTiming::new(11883, 11923)),
        "the collapsed lexical word should reclaim enough duration from the preceding filler span to reach the 40 ms floor"
    );
}

#[test]
fn test_postprocess_with_existing_wor_does_not_clamp_final_word_to_near_zero_duration() {
    let input = "\
@UTF8\n\
@Begin\n\
@Languages:\teng\n\
@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|||||Target_Child|||\n\
@Media:\ttest, audio\n\
*CHI:\talpha beta gamma delta . \u{0015}0_1000\u{0015}\n\
%wor:\talpha \u{0015}100_200\u{0015} beta \u{0015}200_300\u{0015} gamma \u{0015}300_400\u{0015} delta \u{0015}400_500\u{0015} .\n\
@End\n\
";
    let mut chat = parse_chat(input);
    let utt = get_test_utterance(&mut chat, 0);

    let timings = vec![
        Some(WordTiming::new(100, 300)),
        Some(WordTiming::new(300, 500)),
        Some(WordTiming::new(500, 700)),
        Some(WordTiming::new(940, 1200)),
    ];
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_test_utterance(&mut chat, 0);
    let dropped = postprocess_utterance_timings(utt, FaTimingMode::Continuous);
    assert_eq!(dropped, 0);
    update_utterance_bullet(utt);

    let mut collected = Vec::new();
    postprocess::collect_word_timings(&utt.main.content.content, &mut collected);

    assert_eq!(
        collected[3],
        Some(WordTiming::new(940, 1200)),
        "a rerun with existing %wor must keep the worker's final-word duration when clamping would collapse it to a near-zero tail"
    );
    let bullet = utt
        .main
        .content
        .bullet
        .as_ref()
        .expect("bullet should remain");
    assert_eq!(
        bullet.timing.end_ms, 1200,
        "utterance bullet end should expand to the healed final word end"
    );
}

// ---------------------------------------------------------------------------
// Two-pass overlap UTR tests
// ---------------------------------------------------------------------------

/// Helper to make ASR tokens from a slice of (text, start_ms, end_ms).
fn make_utr_tokens(words_with_times: &[(&str, u64, u64)]) -> Vec<utr::AsrTimingToken> {
    words_with_times
        .iter()
        .map(|(text, start, end)| utr::AsrTimingToken {
            text: text.to_string(),
            start_ms: *start,
            end_ms: *end,
        })
        .collect()
}

/// Helper to extract the bullet from the nth utterance (by utterance index).
fn get_utterance_bullet(chat: &talkbank_model::model::ChatFile, idx: usize) -> Option<(u64, u64)> {
    let mut utt_idx = 0;
    for line in &chat.lines {
        if let Line::Utterance(utt) = line {
            if utt_idx == idx {
                return utt
                    .main
                    .content
                    .bullet
                    .as_ref()
                    .map(|b| (b.timing.start_ms, b.timing.end_ms));
            }
            utt_idx += 1;
        }
    }
    None
}

/// TwoPassOverlapUtr correctly times a `+<` backchannel by recovering it
/// from the previous utterance's audio window.
#[test]
fn test_two_pass_correctly_times_lazy_overlap() {
    use utr::UtrStrategy;
    let chat_text = include_str!("../../../../test-fixtures/utr_lazy_overlap_backchannel.cha");
    let mut chat = parse_chat(chat_text);
    let tokens = make_utr_tokens(&[
        // PAR's first utterance words
        ("I", 100, 300),
        ("went", 400, 800),
        ("to", 900, 1100),
        ("the", 1200, 1400),
        ("store", 1500, 2000),
        // INV's backchannel overlaps PAR's first utterance
        ("mhm", 1800, 2200),
        ("yesterday", 2300, 3000),
        // PAR's second utterance
        ("and", 5000, 5300),
        ("I", 5400, 5600),
        ("bought", 5700, 6200),
        ("some", 6300, 6600),
        ("groceries", 6700, 7500),
    ]);
    let result = utr::TwoPassOverlapUtr::new().inject(&mut chat, &tokens);

    // PAR's two utterances + INV's backchannel should all get timing
    assert_eq!(
        result.injected, 3,
        "all 3 untimed utterances should get timing"
    );
    assert_eq!(result.unmatched, 0);

    // Verify INV's "mhm" (utterance index 1) got correct timing
    let inv_bullet = get_utterance_bullet(&chat, 1).expect("INV +< mhm should have a bullet");
    assert!(
        inv_bullet.0 >= 1700 && inv_bullet.0 <= 1900,
        "INV start should be near 1800, got {}",
        inv_bullet.0,
    );
    assert!(
        inv_bullet.1 >= 2100 && inv_bullet.1 <= 2300,
        "INV end should be near 2200, got {}",
        inv_bullet.1,
    );
}

/// GlobalUtr cannot correctly time a `+<` backchannel — the global DP places
/// "mhm" after the main-speaker words, misaligning it.
#[test]
fn test_global_utr_misaligns_lazy_overlap_backchannel() {
    use utr::UtrStrategy;
    let chat_text = include_str!("../../../../test-fixtures/utr_lazy_overlap_backchannel.cha");
    let mut chat = parse_chat(chat_text);
    let tokens = make_utr_tokens(&[
        ("I", 100, 300),
        ("went", 400, 800),
        ("to", 900, 1100),
        ("the", 1200, 1400),
        ("store", 1500, 2000),
        ("mhm", 1800, 2200),
        ("yesterday", 2300, 3000),
        ("and", 5000, 5300),
        ("I", 5400, 5600),
        ("bought", 5700, 6200),
        ("some", 6300, 6600),
        ("groceries", 6700, 7500),
    ]);
    let result = utr::GlobalUtr.inject(&mut chat, &tokens);

    // GlobalUtr may still inject timing for INV, but the timing will be wrong:
    // the DP assigns "mhm" to the token at its position in the global sequence
    // (after "yesterday" or misplaced), not within the overlapping window.
    // We verify that at least the injected count covers all utterances.
    assert_eq!(
        result.injected + result.unmatched,
        3,
        "all 3 untimed utterances accounted for"
    );
}

/// When no `+<` utterances exist, TwoPassOverlapUtr produces identical results
/// to GlobalUtr.
#[test]
fn test_two_pass_identical_without_lazy_overlap() {
    use utr::UtrStrategy;
    let chat_text =
        include_str!("../../../../test-fixtures/fa_mixed_timed_untimed_interleaved.cha");

    // Build matching ASR tokens
    let tokens = make_utr_tokens(&[
        ("the", 10000, 10500),
        ("cat", 10600, 11000),
        ("is", 11200, 11500),
        ("here", 12000, 13000),
        ("she", 15500, 16000),
        ("is", 16200, 16500),
        ("looking", 16800, 17500),
        ("outside", 17800, 18500),
        ("there", 20500, 21000),
        ("is", 21200, 21500),
        ("a", 21800, 22000),
        ("path", 22200, 23000),
        ("I", 26000, 26500),
        ("do", 26800, 27000),
        ("not", 27200, 27500),
        ("know", 27800, 28500),
        ("but", 30000, 30500),
        ("there", 30800, 31200),
        ("is", 31500, 31800),
        ("a", 32000, 32200),
        ("building", 32500, 33500),
        ("okay", 40500, 41000),
        ("so", 41200, 41500),
        ("now", 41800, 42500),
    ]);

    let mut chat_global = parse_chat(chat_text);
    let mut chat_two_pass = parse_chat(chat_text);

    let r1 = utr::GlobalUtr.inject(&mut chat_global, &tokens);
    let r2 = utr::TwoPassOverlapUtr::new().inject(&mut chat_two_pass, &tokens);

    assert_eq!(r1.injected, r2.injected, "injected count should match");
    assert_eq!(r1.unmatched, r2.unmatched, "unmatched count should match");
    assert_eq!(r1.skipped, r2.skipped, "skipped count should match");

    // Compare bullets on each utterance
    for i in 0..6 {
        assert_eq!(
            get_utterance_bullet(&chat_global, i),
            get_utterance_bullet(&chat_two_pass, i),
            "utterance {i} bullets should match",
        );
    }
}

/// Dense backchannels: 4 consecutive `+<` utterances from INV during PAR's
/// narrative should all receive timing within PAR's audio range.
#[test]
fn test_two_pass_dense_backchannels() {
    use utr::UtrStrategy;
    let chat_text = include_str!("../../../../test-fixtures/utr_lazy_overlap_dense.cha");
    let mut chat = parse_chat(chat_text);

    // PAR's narrative spans 100-10000ms. INV's 4 backchannels are scattered within.
    let tokens = make_utr_tokens(&[
        // PAR's words
        ("I", 100, 300),
        ("grew", 400, 700),
        ("up", 800, 1000),
        ("in", 1100, 1300),
        ("Princeton", 1400, 2000),
        // INV backchannel 1: "oh okay"
        ("oh", 2100, 2300),
        ("okay", 2400, 2800),
        ("and", 2900, 3100),
        ("came", 3200, 3500),
        ("to", 3600, 3800),
        ("graduate", 3900, 4400),
        ("school", 4500, 5000),
        // INV backchannel 2: "mhm"
        ("mhm", 5100, 5400),
        ("at", 5500, 5700),
        ("Chapel", 5800, 6200),
        ("Hill", 6300, 6700),
        // INV backchannel 3: "oh"
        ("oh", 6800, 7100),
        ("in", 7200, 7400),
        ("ninety", 7500, 7900),
        ("one", 8000, 8300),
        // INV backchannel 4: "mhm"
        ("mhm", 8400, 8700),
        ("or", 8800, 9000),
        ("maybe", 9100, 9500),
        ("ninety", 9600, 9900),
        ("two", 10000, 10300),
    ]);

    let result = utr::TwoPassOverlapUtr::new().inject(&mut chat, &tokens);

    // PAR's utterance (1) + 4 INV backchannels = 5 injected
    assert_eq!(
        result.injected, 5,
        "PAR + 4 INV backchannels should be timed"
    );
    assert_eq!(result.unmatched, 0);

    // All 4 INV utterances (indices 1-4) should have bullets within PAR's range
    for inv_idx in 1..=4 {
        let bullet = get_utterance_bullet(&chat, inv_idx)
            .unwrap_or_else(|| panic!("INV utterance {inv_idx} should have a bullet"));
        assert!(
            bullet.0 >= 100 && bullet.1 <= 11000,
            "INV utterance {inv_idx} bullet {}-{} should be within PAR's range",
            bullet.0,
            bullet.1,
        );
    }
}

/// `select_strategy` returns TwoPassOverlapUtr for files with +<, GlobalUtr otherwise.
#[test]
fn test_select_strategy_chooses_correctly() {
    let with_overlap = include_str!("../../../../test-fixtures/utr_lazy_overlap_backchannel.cha");
    let without_overlap =
        include_str!("../../../../test-fixtures/fa_mixed_timed_untimed_interleaved.cha");

    let chat_overlap = parse_chat(with_overlap);
    let chat_no_overlap = parse_chat(without_overlap);

    // We can't check the concrete type directly, but we can verify behavior:
    // select_strategy on a +< file should produce TwoPassOverlapUtr results
    let strategy = utr::select_strategy(&chat_overlap, None);
    let mut chat = parse_chat(with_overlap);
    let tokens = make_utr_tokens(&[
        ("I", 100, 300),
        ("went", 400, 800),
        ("to", 900, 1100),
        ("the", 1200, 1400),
        ("store", 1500, 2000),
        ("mhm", 1800, 2200),
        ("yesterday", 2300, 3000),
        ("and", 5000, 5300),
        ("I", 5400, 5600),
        ("bought", 5700, 6200),
        ("some", 6300, 6600),
        ("groceries", 6700, 7500),
    ]);
    let result = strategy.inject(&mut chat, &tokens);
    assert_eq!(result.injected, 3, "should use two-pass and time all 3");

    // select_strategy on a non-+< file should use GlobalUtr
    let strategy = utr::select_strategy(&chat_no_overlap, None);
    let _ = strategy; // Just verify it compiles and returns
}

#[test]
fn snapshot_fa_infer_item() {
    let item = FaInferItem {
        words: vec!["hello".into(), "world".into()],
        word_ids: vec!["u0:w0".into(), "u0:w1".into()],
        word_utterance_indices: vec![0, 0],
        word_utterance_word_indices: vec![0, 1],
        audio_path: "/data/test.mp3".into(),
        audio_start_ms: 1500,
        audio_end_ms: 3200,
        timing_mode: FaTimingMode::WithPauses,
    };
    insta::assert_json_snapshot!(item);
}

// ---------------------------------------------------------------------------
// Bullet architecture tests — double-bullet bug and defense-in-depth
// ---------------------------------------------------------------------------

/// Helper: count InternalBullet items across all utterances in a ChatFile.
fn count_internal_bullets(chat: &talkbank_model::model::ChatFile) -> usize {
    let mut count = 0;
    for line in &chat.lines {
        if let Line::Utterance(utt) = line {
            for item in &utt.main.content.content.0 {
                if matches!(item, UtteranceContent::InternalBullet(_)) {
                    count += 1;
                }
            }
        }
    }
    count
}

/// Helper: count lines with multiple bullet pairs (\x15...\x15) in serialized CHAT.
fn count_double_bullet_lines(chat_text: &str) -> usize {
    chat_text
        .lines()
        .filter(|line| line.starts_with('*'))
        .filter(|line| {
            let bullet_count = line.matches('\x15').count() / 2; // each bullet is a pair
            bullet_count > 1
        })
        .count()
}

/// REGRESSION TEST: Verifies the CA terminator resolution prevents
/// InternalBullet misclassification during serialize→re-parse.
///
/// Before the talkbank-tools parser fix, re-parsing UTR output created stale
/// InternalBullet items. Now the parser's `resolve_ca_terminator()` correctly
/// promotes trailing bullets to terminal `TierContent.bullet`.
///
/// This test verifies the fix works for a simple (non-CA) file.
#[test]
fn utr_serialize_reparse_no_internal_bullets() {
    let input = include_str!("../../../../test-fixtures/fa_untimed_for_utr.cha");
    let mut chat = parse_chat(input);

    // Inject UTR timing (synthetic ASR tokens matching the words)
    let tokens = make_utr_tokens(&[
        ("hello", 1000, 1500),
        ("world", 1600, 2000),
        ("goodbye", 3000, 3500),
        ("world", 3600, 4000),
        ("more", 5000, 5400),
        ("words", 5500, 5800),
        ("here", 5900, 6200),
    ]);
    let result = utr::inject_utr_timing(&mut chat, &tokens);
    assert!(result.injected > 0, "UTR should inject timing");

    // At this point, ChatFile has TierContent.bullet set, but NO InternalBullet items
    assert_eq!(
        count_internal_bullets(&chat),
        0,
        "After UTR injection: should have zero InternalBullet items in AST"
    );

    // Serialize to CHAT text (this is what the old pipeline did)
    let serialized = crate::serialize::to_chat_string(&chat);

    // Re-parse (this is what FA did — the bug)
    let reparsed = parse_chat(&serialized);

    // With the CA terminator resolution fix, the parser correctly promotes
    // trailing bullets to terminal TierContent.bullet — no InternalBullets.
    let internal_count = count_internal_bullets(&reparsed);
    assert_eq!(
        internal_count, 0,
        "After serialize→re-parse: parser should promote all bullets to terminal \
         (CA terminator resolution). Found {internal_count} InternalBullet items."
    );
}

// (The former `diagnostic_sprott_output_ast_inspection` test used to
// live here. It parsed a specific pre-fix output file via
// `include_str!(".../experiments/align-regression-2026-03-30/...")`
// and printed bullet-classification statistics without asserting
// anything. The experiment directory was removed from this public
// repository as part of the 2026-04-10 public-info expunge, so the
// test is gone too. The underlying bug it diagnosed is covered by
// the assertions in the adjacent
// `apply_fa_produces_no_double_bullets_after_utr` test, which uses
// a committed public fixture and therefore still runs.)

/// DEFENSE-IN-DEPTH TEST: strip_internal_bullet_tokens in apply_fa_results
/// prevents double bullets even when InternalBullet items exist.
///
/// The talkbank-tools parser fix (CA terminator resolution) now correctly
/// promotes trailing InternalBullets to terminal bullets during parsing, so
/// the simple UTR→serialize→reparse path no longer creates stale
/// InternalBullets. However, LENA/HomeBank files legitimately have
/// InternalBullets (sub-utterance event timing on continuation lines), and
/// the defense-in-depth strip in apply_fa_results protects against any
/// future code path that might re-introduce stale InternalBullets.
///
/// This test verifies that apply_fa_results produces clean output (no
/// double bullets) on a file that has been through UTR injection.
#[test]
fn apply_fa_produces_no_double_bullets_after_utr() {
    let input = include_str!("../../../../test-fixtures/fa_untimed_for_utr.cha");
    let mut chat = parse_chat(input);
    let tokens = make_utr_tokens(&[
        ("hello", 1000, 1500),
        ("world", 1600, 2000),
        ("goodbye", 3000, 3500),
        ("world", 3600, 4000),
        ("more", 5000, 5400),
        ("words", 5500, 5800),
        ("here", 5900, 6200),
    ]);
    utr::inject_utr_timing(&mut chat, &tokens);

    // Group and create synthetic FA timings
    let groups = group_utterances(&chat, 30_000, Some(10_000));
    let responses: Vec<Vec<Option<WordTiming>>> = groups
        .iter()
        .map(|g| {
            let word_count: usize = g
                .utterance_indices
                .iter()
                .map(|&idx| {
                    let mut count = 0;
                    for (i, line) in chat.lines.iter().enumerate() {
                        if let Line::Utterance(u) = line
                            && crate::indices::UtteranceIdx(i) == idx
                        {
                            count = count_alignable_main_words(u);
                        }
                    }
                    count
                })
                .sum();
            (0..word_count)
                .map(|i| {
                    Some(WordTiming {
                        start_ms: (i as u64) * 500 + 1000,
                        end_ms: (i as u64) * 500 + 1400,
                    })
                })
                .collect()
        })
        .collect();

    apply_fa_results(
        &mut chat,
        &groups,
        &responses,
        FaTimingMode::Continuous,
        true,
    );

    let output = crate::serialize::to_chat_string(&chat);
    let double_bullets = count_double_bullet_lines(&output);
    assert_eq!(
        double_bullets, 0,
        "After apply_fa_results: should have zero double-bullet lines, got {double_bullets}.\n\
         Output:\n{output}"
    );
}

/// Compound fillers like `&-you_know` must be sent to FA as separate words
/// ("you", "know") so the DP aligner can match against Whisper's multi-word
/// output. After alignment, the timing for the parts merges back into the
/// single `&-you_know` token on %wor.
///
/// Bug report: a user's `17-3.cha` — `&-you_know` at utterance boundaries
/// gets cut off from %wor because FA can't align the compound token.
/// Input: `*PAR: &-you_know I see a boy who is on a stool . 23645_27265`
#[test]
fn compound_filler_extracted_as_separate_words_for_fa() {
    let input = include_str!("../../../../test-fixtures/fa_compound_filler.cha");
    let chat = parse_chat(input);

    let utt = chat
        .lines
        .iter()
        .find_map(|l| match l {
            Line::Utterance(u) => Some(u),
            _ => None,
        })
        .expect("fixture should have one utterance");

    let mut words = Vec::new();
    extraction::collect_fa_words(&utt.main.content.content, &mut words);

    // &-you_know should produce TWO words for FA: "you" and "know"
    // (not one compound "you_know" that Whisper can't match)
    assert!(
        words.contains(&"you".to_string()) && words.contains(&"know".to_string()),
        "compound filler &-you_know should be split into 'you' and 'know' for FA, got: {words:?}"
    );

    // &-um (if present) should remain as single "um"
    // Regular words should be unchanged
    assert!(
        words.contains(&"I".to_string()),
        "regular word 'I' missing: {words:?}"
    );
}

// ---------------------------------------------------------------------------
// Regression: injection/extraction desync for ReplacedWords (2026-04-08)
//
// Commit 369f1d4d changed collect_fa_words (extraction) to always send the
// original word for a ReplacedWord, but inject_timings_for_utterance still
// iterated replacement words — consuming N cursor positions when only 1 FA
// word was extracted.  For a 2-word replacement this shifts every subsequent
// word in the same FA group by one timing position, corrupting all later %wor.
// ---------------------------------------------------------------------------

/// Extraction must send the original word for a replaced word, not the
/// replacements.  This is the reference policy test; the timing injection
/// test below relies on this count.
#[test]
fn test_fa_extraction_replaced_word_uses_original() {
    // "foo [: bar baz] qux ." → extraction should see [foo, qux], NOT [bar, baz, qux]
    let chat = parse_chat(&proof_chat("foo [: bar baz] qux ."));
    let utt = get_utterance(&chat, 0);
    let fa_words = {
        let mut v = Vec::new();
        extraction::collect_fa_words(&utt.main.content.content, &mut v);
        v
    };
    assert_eq!(
        fa_words,
        vec!["foo", "qux"],
        "extraction should use original replaced word (foo), not replacements (bar baz)"
    );
}

/// Injection must consume exactly ONE cursor position for a replaced word and
/// set the timing on the original word, not the replacement words.
/// When a 2-word replacement consumed 2 cursor positions but only 1 word was
/// extracted, the next word in the same group received timing T[i+1] instead
/// of T[i].
#[test]
fn test_fa_injection_replaced_word_uses_original_and_cursor_stays_in_sync() {
    // "foo [: bar baz] qux ." — injection must advance cursor by exactly 2 total:
    //   slot 0 → timing for foo (original replaced word)
    //   slot 1 → timing for qux (plain word after the replacement)
    let timings = vec![
        Some(WordTiming {
            start_ms: 100,
            end_ms: 500,
        }), // slot 0: foo
        Some(WordTiming {
            start_ms: 600,
            end_ms: 1000,
        }), // slot 1: qux
    ];

    let mut chat = parse_chat(&proof_chat("foo [: bar baz] qux ."));
    let utt = get_test_utterance(&mut chat, 0);
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    // Cursor must advance by exactly 2 (one for the original replaced word,
    // one for qux). The old bug advanced by 3 (two for bar+baz, one for qux),
    // leaving qux to take timing slot 2 which is out-of-bounds → None.
    assert_eq!(
        offset, 2,
        "cursor must advance by 2 (1 for original + 1 for qux), not 3 (2 for replacements + 1 for qux)"
    );

    // Generate %wor and check that qux carries timing slot 1 (600_1000).
    add_wor_tier(get_test_utterance(&mut chat, 0));
    let output = chat.to_chat_string();
    assert!(
        output.contains("qux \u{15}600_1000\u{15}"),
        "qux should have timing 600_1000 (slot 1); old bug put it at slot 2 → untimed:\n{output}"
    );
}

/// When a FA group spans two utterances — one with a 2-word replaced word and
/// one plain — the cursor must stay in sync across the utterance boundary so
/// the second utterance's words get the correct timing positions.
#[test]
fn test_fa_injection_cursor_stays_in_sync_across_utterance_boundary_with_replaced_word() {
    // FA group: [utt0: "a [: x y] .", utt1: "hello world ."]
    // Extraction (new): [a, hello, world] = 3 words (original for utt0)
    // FA returns 3 timings.
    // Old injection for utt0 consumed 2 (x, y) + 0 for 'a' = 2 slots,
    // leaving hello at slot 2 (T_world) and world at slot 3 (None).
    // Fixed injection for utt0 consumes 1 (a) = 1 slot → hello at slot 1, world at slot 2.
    let timings = vec![
        Some(WordTiming {
            start_ms: 100,
            end_ms: 200,
        }), // slot 0: a
        Some(WordTiming {
            start_ms: 300,
            end_ms: 500,
        }), // slot 1: hello
        Some(WordTiming {
            start_ms: 600,
            end_ms: 900,
        }), // slot 2: world
    ];

    // Build a two-utterance chat file
    let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\ta [: x y] .\n*CHI:\thello world .\n@End\n".to_string();
    let mut chat = parse_chat(&chat_text);

    let mut offset = 0;
    {
        let utt0 = get_test_utterance(&mut chat, 0);
        inject_timings_for_utterance(utt0, &timings, &mut offset);
    }
    // After utt0: cursor should be at 1 (one original word consumed)
    assert_eq!(
        offset, 1,
        "cursor after utt0 must be 1 (only 'a' consumed), not 2 (old: bar+baz)"
    );

    {
        let utt1 = get_test_utterance(&mut chat, 1);
        inject_timings_for_utterance(utt1, &timings, &mut offset);
    }
    // After utt1: cursor at 3 (hello + world)
    assert_eq!(offset, 3, "cursor after utt1 must be 3");

    // Generate %wor for utt1 and verify hello/world timings are correct
    add_wor_tier(get_test_utterance(&mut chat, 1));
    let output = chat.to_chat_string();
    assert!(
        output.contains("hello \u{15}300_500\u{15}"),
        "hello should have timing 300_500 (slot 1):\n{output}"
    );
    assert!(
        output.contains("world \u{15}600_900\u{15}"),
        "world should have timing 600_900 (slot 2):\n{output}"
    );
}

/// UTR must not create a zero-duration bullet when the matched ASR tokens
/// have `start_ms == end_ms`.
///
/// Whisper ASR can return zero-duration word timestamps for very short words
/// (single 20ms frame backchannels like "mhm", "yeah"). If UTR injects these
/// as utterance bullets, the result is `•T_T•`. The FA postprocess then
/// clamps all word timings to the `[T, T]` range — dropping every timing —
/// and `update_utterance_bullet` has nothing to work from, so the
/// zero-duration bullet perpetuates across every subsequent `align` run.
///
/// The fix: UTR must not set a bullet when `start_ms >= end_ms`. It should
/// leave the utterance untimed so FA can assign a valid bullet from word
/// alignments instead. See OCSC corpus bug report (2026-04-08).
#[test]
fn test_utr_zero_duration_asr_token_does_not_create_zero_duration_bullet() {
    // Single-word backchannel — the canonical OCSC failure pattern.
    let mut chat = parse_chat(&proof_chat("mhm ."));

    // Simulate Whisper returning start==end for "mhm" (one 20ms frame).
    let tokens = make_utr_tokens(&[("mhm", 646025, 646025)]);

    let result = utr::inject_utr_timing(&mut chat, &tokens);

    // The utterance must NOT receive a zero-duration bullet.
    // Either it stays untimed (unmatched) or gets a non-zero-duration bullet.
    let bullet = get_utterance_bullet(&chat, 0);
    match bullet {
        Some((start, end)) => {
            assert!(
                start < end,
                "UTR injected a zero-duration bullet •{start}_{end}• for \"mhm\" — \
                 this is the OCSC bug: Whisper zero-duration ASR token propagated to \
                 utterance bullet, which perpetuates through all FA re-runs"
            );
        }
        None => {
            // Correct outcome: utterance is untimed; FA will assign a valid bullet.
            assert_eq!(
                result.unmatched, 1,
                "utterance should be counted as unmatched when only \
                 zero-duration tokens are available"
            );
        }
    }
}

/// `collect_existing_fa_word_timings` must return one entry per FA-alignable
/// word, using the original word for replaced words.  Before the fix it
/// iterated replacement words (bar, baz), returning [None, None, Some] for
/// "foo [: bar baz] qux .".  That miscount broke `collect_preserved_group_timings`
/// (len mismatch with group.words) so groups with ReplacedWord utterances
/// always bypassed the %wor preservation path and re-hit the FA worker.
#[test]
fn test_collect_existing_fa_word_timings_replaced_word_returns_one_entry_for_original() {
    // "foo [: bar baz] qux ." has 2 FA words (foo and qux) after my fix.
    // After injection: foo gets timing 100_500, qux gets timing 600_1000.
    // collect_existing_fa_word_timings must return exactly 2 entries —
    // not 3 (bar:None, baz:None, qux:Some) as the old code did.
    let timings = vec![
        Some(WordTiming {
            start_ms: 100,
            end_ms: 500,
        }), // foo
        Some(WordTiming {
            start_ms: 600,
            end_ms: 1000,
        }), // qux
    ];

    let mut chat = parse_chat(&proof_chat("foo [: bar baz] qux ."));
    let utt = get_test_utterance(&mut chat, 0);
    let mut offset = 0;
    inject_timings_for_utterance(utt, &timings, &mut offset);

    let utt = get_utterance(&chat, 0);
    let existing_timings = collect_existing_fa_word_timings(utt);

    assert_eq!(
        existing_timings.len(),
        2,
        "must return 2 entries (foo, qux) — old code returned 3 (bar:None, baz:None, qux:Some) causing collect_preserved_group_timings to return None on every run: {existing_timings:?}"
    );
    assert_eq!(
        existing_timings[0],
        Some(WordTiming {
            start_ms: 100,
            end_ms: 500
        }),
        "foo should carry its injected timing"
    );
    assert_eq!(
        existing_timings[1],
        Some(WordTiming {
            start_ms: 600,
            end_ms: 1000
        }),
        "qux should carry its injected timing"
    );
}

// ---------------------------------------------------------------------------
// Bug regression: UTR must produce strictly increasing start times
// ---------------------------------------------------------------------------

/// UTR must assign strictly increasing `start_ms` to adjacent non-overlap
/// utterances, even when their matched ASR tokens share the same start time.
///
/// Root cause of zero-duration bullets (deeper than the monotonicity fix):
/// Whisper's 20ms DTW grid can return multiple consecutive tokens all starting
/// at the same timestamp (e.g., two short backchannels "mhm" and "yeah" both
/// at 1000ms).  When the global DP aligns CHAT words to these tokens, adjacent
/// utterances receive the same `start_ms`.  `enforce_monotonicity`'s pass 2
/// then clamps `prev.end → next.start = prev.start` → zero-duration `•T_T•`.
///
/// The principled fix is in UTR: after assigning per-utterance token ranges,
/// walk adjacent non-overlap pairs and advance the next utterance's `start_ms`
/// to be strictly greater than the previous one's.  Using `prev.end_ms` as the
/// floor is a natural choice — the next utterance cannot start before the
/// previous one ended.
///
/// This test encodes the invariant: UTR output must have strictly monotonically
/// increasing `start_ms` for adjacent non-overlap utterances.
#[test]
fn test_utr_non_overlap_utterances_get_strictly_increasing_start_times() {
    use utr::UtrStrategy;
    // Two adjacent non-overlap utterances.  Both ASR tokens share start=1000ms
    // (the Whisper 20ms DTW artifact — both backchannels fall in the same frame).
    let chat_text = "\
@UTF8\n\
@Begin\n\
@Languages:\teng\n\
@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|2;0.0||||Target_Child|||\n\
*CHI:\tmhm .\n\
*CHI:\tyeah .\n\
@End\n\
";
    let mut chat = parse_chat(chat_text);
    let tokens = make_utr_tokens(&[
        ("mhm", 1000, 1500),
        ("yeah", 1000, 2000), // same start_ms as "mhm" — Whisper DTW collision
    ]);

    let _result = utr::TwoPassOverlapUtr::new().inject(&mut chat, &tokens);

    let b0 = get_utterance_bullet(&chat, 0);
    let b1 = get_utterance_bullet(&chat, 1);

    // Both utterances should be timed.
    let (start0, _end0) = b0.expect("utterance 0 (mhm) should have a bullet");
    let (start1, _end1) = b1.expect("utterance 1 (yeah) should have a bullet");

    // The core invariant: adjacent non-overlap utterances must have strictly
    // increasing start times so that monotonicity end-clamping can never
    // produce a zero-duration bullet.
    assert!(
        start1 > start0,
        "adjacent non-overlap utterances must have strictly increasing start_ms: \
         utt0.start={start0} utt1.start={start1} — identical start times from \
         Whisper DTW collision will cause enforce_monotonicity to produce •T_T•"
    );
}

// ---------------------------------------------------------------------------
// Bug regression: zero-duration bullets via monotonicity end-clamping
// ---------------------------------------------------------------------------

/// `enforce_monotonicity` must not produce zero-duration utterance bullets.
///
/// When two adjacent utterances have the same start time (a common UTR output
/// when overlapping ASR token ranges are assigned), the end-time clamp pass
/// sets prev.end = next.start = prev.start → zero-duration bullet.  The
/// validator then fires E362 on every such utterance.
///
/// The fix: after clamping, if clamped_end <= bullet.start_ms, strip the
/// bullet entirely rather than leaving a zero-duration or negative-duration
/// span behind.  Better untimed than invalid.
#[test]
fn test_monotonicity_clamp_does_not_create_zero_duration_bullet() {
    // Two utterances with identical start times — the UTR overlap scenario.
    // Utterance 0: •1000_1500• (start=1000, end=1500)
    // Utterance 1: •1000_2000• (start=1000 == utt0.start → monotonicity
    //              pass 1 leaves it because 1000 >= last_start_ms=1000,
    //              then pass 2 clamps utt0.end to 1000 → 1000_1000).
    let input = "\
@UTF8\n\
@Begin\n\
@Languages:\teng\n\
@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|2;0.0||||Target_Child|||\n\
*CHI:\thello . \u{0015}1000_1500\u{0015}\n\
*CHI:\tworld . \u{0015}1000_2000\u{0015}\n\
@End\n\
";
    let mut chat = parse_chat(input);
    enforce_monotonicity(&mut chat);

    // After monotonicity enforcement, no bullet may have start_ms >= end_ms.
    for (i, line) in chat.lines.iter().enumerate() {
        let talkbank_model::model::Line::Utterance(utt) = line else {
            continue;
        };
        if let Some(bullet) = &utt.main.content.bullet {
            assert!(
                bullet.timing.start_ms < bullet.timing.end_ms,
                "utterance at line {i} has zero-or-negative-duration bullet \
                 •{}_{} after monotonicity enforcement — this is the UTR overlap \
                 bug: identical start times cause end-clamping to produce •T_T•",
                bullet.timing.start_ms,
                bullet.timing.end_ms
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Bug regression: duplicate %x tiers after re-run
// ---------------------------------------------------------------------------

/// Re-running FA on a CHAT file that already has `%xalign`/`%xrev` from a
/// previous run must REMOVE the old tiers unconditionally — even when the new
/// run produces no decisions at all.
///
/// The bug: `inject_decision_tiers` (which strips old tiers) is gated on
/// `!all_decisions.is_empty()`.  A clean re-run with zero decisions skips the
/// call entirely, leaving stale `%xalign`/`%xrev` from the previous run in
/// the output file.  On a subsequent run that DOES produce decisions the old
/// tier and the new tier both appear → duplicate `%xalign` / `%xrev`.
///
/// The fix: strip decision tiers unconditionally at the start of `apply_fa_results`
/// (or at the top of the FA orchestration step), so old tiers are always removed
/// regardless of whether new ones will be written.
#[test]
fn test_rerun_fa_strips_stale_x_tiers_even_when_no_new_decisions() {
    // A pre-aligned file with existing %xalign / %xrev from a previous run.
    let input = "\
@UTF8\n\
@Begin\n\
@Languages:\teng\n\
@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|2;0.0||||Target_Child|||\n\
*CHI:\thello world . \u{0015}1000_3000\u{0015}\n\
%xalign:\tfa:old_decision old_reason_from_previous_run\n\
%xrev:\t[ok]\n\
@End\n\
";
    let mut chat = parse_chat(input);

    // Re-run: apply FA with clean word timings (no decisions expected).
    let groups = vec![FaGroup {
        audio_span: TimeSpan::new(0, 5000),
        words: vec![
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(0),
                text: "hello".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(1),
                text: "world".into(),
            },
        ],
        utterance_indices: vec![UtteranceIdx(0)],
    }];
    let responses = vec![vec![
        Some(WordTiming {
            start_ms: 1000,
            end_ms: 1500,
        }),
        Some(WordTiming {
            start_ms: 1500,
            end_ms: 3000,
        }),
    ]];
    let decisions = apply_fa_results(
        &mut chat,
        &groups,
        &responses,
        FaTimingMode::WithPauses,
        false,
    );

    // Simulate fa/mod.rs step 9d — the BUGGY path: only injects (and strips)
    // when decisions is non-empty.  Clean re-run → decisions is empty → no
    // strip → old tiers remain.
    if !decisions.is_empty() {
        crate::decisions::inject_decision_tiers(
            &mut chat,
            &decisions,
            crate::fa::ReviewLevel::LowConfidence,
        );
    }

    let output = chat.to_chat_string();
    let xalign_count = output.matches("%xalign:").count();
    let xrev_count = output.matches("%xrev:").count();

    // After a clean re-run, ALL old %xalign and %xrev tiers must be gone — even
    // when no new decisions were made.  Leaving stale tiers from the previous
    // run means the NEXT run that produces decisions will append to them,
    // producing duplicates.
    assert_eq!(
        xalign_count, 0,
        "stale %xalign from previous run must be stripped on re-run even with no new decisions; \
         got {xalign_count}:\n{output}"
    );
    assert_eq!(
        xrev_count, 0,
        "stale %xrev from previous run must be stripped on re-run even with no new decisions; \
         got {xrev_count}:\n{output}"
    );
}

// ---------------------------------------------------------------------------
// FA-authoritative utterance bullet: UTR hint vs word-derived span
// ---------------------------------------------------------------------------
//
// UTR sets utterance bullets as provisional grouping hints before FA runs.
// After FA injects word timings, the utterance bullet should be derived from
// actual word timings (FA-authoritative), not the UTR hint.
//
// This is the BA2-style self-healing property: valid word timings → valid
// utterance timing by construction, eliminating the UTR→FA dependency
// fragility where a wrong UTR window could survive into the output.
//
// Exception: when FA produces zero word timings for an utterance (total FA
// failure), the UTR hint is the only timing we have and must be preserved.

/// FA word timings must overwrite the UTR hint, not union-expand from it.
///
/// UTR set a wide provisional window (800_3000) on an initially-untimed
/// utterance. FA aligned both words to the narrower span 1000_2000. The
/// utterance bullet must reflect what FA actually aligned — not the UTR estimate.
///
/// The bullet is parsed from CHAT text and its source is then set to
/// `BulletSource::Utr` to simulate what UTR would have done at runtime.
#[test]
fn test_fa_bullet_overwrites_utr_hint_with_word_derived_timing() {
    use talkbank_model::model::BulletSource;

    let input = "\
@UTF8\n\
@Begin\n\
@Languages:\teng\n\
@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|2;0.0||||Target_Child|||\n\
*CHI:\thello world . \u{0015}800_3000\u{0015}\n\
@End\n\
";
    let mut chat = parse_chat(input);

    // Simulate what UTR does at runtime: mark the bullet as a provisional hint
    // so that update_utterance_bullet knows to overwrite it after FA.
    {
        let utt = get_test_utterance(&mut chat, 0);
        let bullet = utt
            .main
            .content
            .bullet
            .as_mut()
            .expect("test requires pre-existing UTR bullet");
        bullet.source = BulletSource::Utr;
    }

    let groups = vec![FaGroup {
        audio_span: TimeSpan::new(800, 3000),
        words: vec![
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(0),
                text: "hello".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(1),
                text: "world".into(),
            },
        ],
        utterance_indices: vec![UtteranceIdx(0)],
    }];

    let responses = vec![vec![
        Some(WordTiming::new(1000, 1500)),
        Some(WordTiming::new(1500, 2000)),
    ]];

    apply_fa_results(
        &mut chat,
        &groups,
        &responses,
        FaTimingMode::WithPauses,
        false,
    );

    let (start, end) =
        get_utterance_bullet(&chat, 0).expect("utterance must have a bullet after FA");
    assert_eq!(
        start, 1000,
        "FA word span must overwrite UTR hint start: expected 1000, got {start}. \
         UTR hint was 800 but FA aligned first word to 1000."
    );
    assert_eq!(
        end, 2000,
        "FA word span must overwrite UTR hint end: expected 2000, got {end}. \
         UTR hint was 3000 but FA aligned last word to 2000."
    );
}

/// When FA produces no word timings (total alignment failure), the UTR hint
/// must be preserved — it is the only timing information we have.
///
/// The bullet source is set to `BulletSource::Utr` to simulate runtime UTR
/// injection. Even with the overwrite semantics, the guard
/// `if let (Some(word_start), Some(word_end)) = ...` leaves the bullet
/// unchanged when FA produces no timings.
#[test]
fn test_fa_preserves_utr_hint_when_all_words_untimed() {
    use talkbank_model::model::BulletSource;

    let input = "\
@UTF8\n\
@Begin\n\
@Languages:\teng\n\
@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|2;0.0||||Target_Child|||\n\
*CHI:\thello world . \u{0015}1000_3000\u{0015}\n\
@End\n\
";
    let mut chat = parse_chat(input);

    // Simulate UTR having set this bullet as a provisional hint.
    {
        let utt = get_test_utterance(&mut chat, 0);
        let bullet = utt
            .main
            .content
            .bullet
            .as_mut()
            .expect("test requires pre-existing UTR bullet");
        bullet.source = BulletSource::Utr;
    }

    let groups = vec![FaGroup {
        audio_span: TimeSpan::new(1000, 3000),
        words: vec![
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(0),
                text: "hello".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(1),
                text: "world".into(),
            },
        ],
        utterance_indices: vec![UtteranceIdx(0)],
    }];

    // FA total failure: all words return None.
    let responses = vec![vec![None, None]];

    apply_fa_results(
        &mut chat,
        &groups,
        &responses,
        FaTimingMode::WithPauses,
        false,
    );

    let (start, end) = get_utterance_bullet(&chat, 0)
        .expect("UTR hint must survive when FA produced no word timings");
    assert_eq!(
        start, 1000,
        "UTR hint start must be preserved when FA produced no timings, got {start}"
    );
    assert_eq!(
        end, 3000,
        "UTR hint end must be preserved when FA produced no timings, got {end}"
    );
}

/// A rescued rerun bullet is still provisional until FA succeeds.
///
/// If rescue leaves it authoritative, rerun postprocess clamps fresh FA word
/// timings back into the still-too-narrow rescued span and drops them before
/// `update_utterance_bullet` can widen the utterance.
#[test]
fn test_rescued_rerun_bullet_does_not_clamp_new_fa_words() {
    let input = "\
@UTF8\n\
@Begin\n\
@Languages:\teng\n\
@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|2;0.0||||Target_Child|||\n\
*CHI:\tokay \u{0015}2200_2300\u{0015} thank \u{0015}2300_2450\u{0015} you \u{0015}2450_2600\u{0015} . \u{0015}1000_1300\u{0015}\n\
%wor:\tokay \u{0015}1100_1200\u{0015} thank \u{0015}1200_1300\u{0015} you .\n\
*CHI:\tbye . \u{0015}3000_3200\u{0015}\n\
@End\n\
";
    let mut chat = parse_chat(input);

    let decisions = rescue_narrow_bullets(&mut chat);
    assert_eq!(decisions.len(), 1, "narrow-bullet rescue should fire");

    {
        let utt = get_test_utterance(&mut chat, 0);
        let bullet = utt
            .main
            .content
            .bullet
            .as_ref()
            .expect("rescued utterance should still have a bullet");
        assert_eq!(
            bullet.source,
            talkbank_model::model::BulletSource::Utr,
            "rescued bullet must stay provisional so postprocess will not clamp FA back into the stale narrow span",
        );

        let dropped = postprocess_utterance_timings(utt, FaTimingMode::WithPauses);
        assert_eq!(
            dropped, 0,
            "rescued provisional bullet must not drop new FA timings during rerun postprocess",
        );
        update_utterance_bullet(utt);
    }

    let output = chat.to_chat_string();
    assert!(
        output.contains("okay \u{15}2200_2300\u{15} thank \u{15}2300_2450\u{15}"),
        "rescued rerun should keep FA word timings beyond the original narrow bullet:\n{output}",
    );
}

/// When there is no pre-existing bullet, FA word timings set the bullet from
/// scratch. This case is unchanged by the overwrite/union distinction —
/// both behaviors produce the same result when `existing` is None.
#[test]
fn test_fa_sets_bullet_from_word_span_when_no_prior_bullet() {
    let input = "\
@UTF8\n\
@Begin\n\
@Languages:\teng\n\
@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|2;0.0||||Target_Child|||\n\
*CHI:\thello world .\n\
@End\n\
";
    let mut chat = parse_chat(input);

    // No pre-existing bullet.
    assert!(
        get_utterance_bullet(&chat, 0).is_none(),
        "test requires utterance to have no bullet initially"
    );

    let groups = vec![FaGroup {
        audio_span: TimeSpan::new(0, 5000),
        words: vec![
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(0),
                text: "hello".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(1),
                text: "world".into(),
            },
        ],
        utterance_indices: vec![UtteranceIdx(0)],
    }];

    let responses = vec![vec![
        Some(WordTiming::new(1000, 1500)),
        Some(WordTiming::new(1500, 2000)),
    ]];

    apply_fa_results(
        &mut chat,
        &groups,
        &responses,
        FaTimingMode::WithPauses,
        false,
    );

    let (start, end) =
        get_utterance_bullet(&chat, 0).expect("utterance must have a bullet after FA");
    assert_eq!(
        start, 1000,
        "bullet start must be first word start, got {start}"
    );
    assert_eq!(end, 2000, "bullet end must be last word end, got {end}");
}

/// Regression test: OCSC/4/4024.cha pattern — a zero-duration utterance bullet
/// persisted from a previous buggy FA run is preserved on re-alignment because
/// FA produces no word timings for the utterance (single letter @l word).
///
/// `update_utterance_bullet` previously fell through to "no word timings →
/// keep existing bullet", propagating the invalid 0-duration bullet → E362.
///
/// Fix: when no word timings come from FA, a zero-duration existing bullet must
/// be cleared rather than kept. No bullet is valid CHAT; a zero-duration bullet
/// is E362.
#[test]
fn test_fa_clears_zero_duration_authoritative_bullet_when_fa_produces_no_word_timings() {
    // Simulate a file that had a zero-duration bullet (start == end) from a
    // previous buggy FA run. The bullet is parsed from the file, so it is
    // BulletSource::Authoritative (the default for Bullet::new / parsed bullets).
    let input = "\
@UTF8\n@Begin\n@Languages:\teng\n\
@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|2;0.0||||Target_Child|||\n\
*CHI:\tz@l . \u{0015}245986_245986\u{0015}\n\
@End\n";
    let mut chat = parse_chat(input);

    // Confirm the bullet is zero-duration before FA.
    let (pre_start, pre_end) =
        get_utterance_bullet(&chat, 0).expect("test setup: utterance must have a bullet before FA");
    assert_eq!(
        pre_start, pre_end,
        "test setup: bullet must be zero-duration"
    );

    // FA returns all None — e.g. the FA engine cannot align a single letter.
    let groups = vec![FaGroup {
        audio_span: TimeSpan::new(245000, 247000),
        words: vec![FaWord {
            utterance_index: UtteranceIdx(0),
            utterance_word_index: WordIdx(0),
            text: "z".into(),
        }],
        utterance_indices: vec![UtteranceIdx(0)],
    }];
    let responses = vec![vec![None]];

    apply_fa_results(
        &mut chat,
        &groups,
        &responses,
        FaTimingMode::WithPauses,
        false,
    );

    // The zero-duration bullet must be cleared, not preserved.
    assert!(
        get_utterance_bullet(&chat, 0).is_none(),
        "zero-duration authoritative bullet must be cleared when FA produces no word timings \
         (keeping it produces E362); found {:?}",
        get_utterance_bullet(&chat, 0)
    );
}

/// Confirms that `apply_fa_results` correctly handles utterances containing
/// `xxx` (unintelligible speech). Under the current policy, `xxx` has no known
/// phoneme sequence — the FA engine never receives it, and `%wor` does not
/// include a slot for it. Only the 5 phonologically real words receive timing.
///
/// This replaces the pre-policy-change test that expected `xxx` to appear in
/// `%wor` with `None` timing. The old behavior was wrong: sending an unknown
/// token to a CTC aligner wastes a cursor slot and shifts all subsequent
/// timings. `%wor` is a timing-annotation tier, not a 1-to-1 mirror of the
/// main tier; untranscribed tokens carry no alignable content.
#[test]
fn test_apply_fa_results_excludes_xxx_from_wor_tier() {
    // Fixture: an utterance with `xxx`. The stale %wor has 5 words (no xxx) —
    // this matches what the new policy will produce after FA.
    let input = concat!(
        "@UTF8\n",
        "@Begin\n",
        "@Languages:\teng\n",
        "@Participants:\tINV Investigator\n",
        "@ID:\teng|test|INV|||||Investigator|||\n",
        "*INV:\tlast time I saw you xxx . \u{0015}27602_28323\u{0015}\n",
        "%wor:\tlast \u{0015}27602_27762\u{0015} time \u{0015}27762_27942\u{0015} I \u{0015}27942_28002\u{0015} saw \u{0015}28002_28203\u{0015} you \u{0015}28203_28323\u{0015} .\n",
        "@End\n",
    );
    let mut chat = parse_chat(input);

    // FA group: 5 words extracted by collect_fa_words — xxx is excluded because
    // untranscribed tokens have no alignable phoneme sequence.
    let groups = vec![FaGroup {
        audio_span: TimeSpan::new(27602, 28323),
        words: vec![
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(0),
                text: "last".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(1),
                text: "time".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(2),
                text: "I".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(3),
                text: "saw".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(4),
                text: "you".into(),
            },
            // xxx is NOT in the FA group — not sent to the aligner.
        ],
        utterance_indices: vec![UtteranceIdx(0)],
    }];

    // FA response: 5 timings for the 5 real words.
    let responses = vec![vec![
        Some(WordTiming {
            start_ms: 27602,
            end_ms: 27762,
        }),
        Some(WordTiming {
            start_ms: 27762,
            end_ms: 27942,
        }),
        Some(WordTiming {
            start_ms: 27942,
            end_ms: 28002,
        }),
        Some(WordTiming {
            start_ms: 28002,
            end_ms: 28203,
        }),
        Some(WordTiming {
            start_ms: 28203,
            end_ms: 28323,
        }),
    ]];

    apply_fa_results(
        &mut chat,
        &groups,
        &responses,
        FaTimingMode::WithPauses,
        true,
    );

    let output = chat.to_chat_string();

    // The %wor tier must have 5 word entries — `xxx` is excluded, no slot for it.
    let post_wor: Vec<_> = get_utterance(&chat, 0)
        .wor_tier()
        .expect("output must contain a %wor tier after FA")
        .words()
        .map(|w| w.cleaned_text().to_string())
        .collect();
    assert_eq!(
        post_wor,
        vec!["last", "time", "I", "saw", "you"],
        "%wor tier must contain only the 5 real words (xxx excluded); \
         got: {post_wor:?}\nFull output:\n{output}"
    );

    // All 5 real words must have timing bullets.
    let wor = get_utterance(&chat, 0)
        .wor_tier()
        .expect("output must contain a %wor tier after FA");
    let wor_words: Vec<_> = wor.words().collect();
    assert!(
        wor_words[0].inline_bullet.is_some(),
        "`last` must have a timing bullet"
    );
    assert!(
        wor_words[4].inline_bullet.is_some(),
        "`you` must have a timing bullet"
    );
}

/// Regression test for APROCSA 2256_T4 (and similar multi-task protocol files):
/// FA aligns a scripted instruction block to the *wrong* audio window (an
/// earlier task repetition of the same phrases), producing a non-monotonic
/// start time that is less than the previous utterance's start time.
///
/// After `apply_fa_results` + `enforce_monotonicity`, the backward bullet must
/// be stripped — NOT written to the output file where it would cause E362/E704.
///
/// RED: currently fails because the backward bullet survives into the output.
#[test]
fn test_fa_backward_timestamp_from_wrong_audio_window_is_stripped() {
    // Simulate a previously-aligned CHAT file (BA2 bullets already present).
    // Two consecutive same-speaker utterances:
    //   utt 0: INV "alright ."       correctly aligned at 731556_733418
    //   utt 1: INV "so take a look"  FA ran on the wrong earlier window → 639095_640375
    let chat_text = concat!(
        "@UTF8\n",
        "@Begin\n",
        "@Languages:\teng\n",
        "@Participants:\tINV Investigator Adult_Unrelated\n",
        "@ID:\teng|test|INV||female|||Adult_Unrelated|||\n",
        "@Media:\ttest, audio\n",
        "*INV:\talright . \u{15}731556_733418\u{15}\n",
        "*INV:\tso take a look at all of them . \u{15}639095_640375\u{15}\n",
        "@End\n",
    );

    let mut chat = parse_chat(chat_text);

    // FA groups: each utterance is its own group.
    // Group 0 (correct): "alright" aligned to audio at 731556–733418.
    // Group 1 (wrong window): "so take a look …" aligned to earlier window at
    //   639000–641000ms — FA returns timings relative to that wrong window.
    let groups = vec![
        FaGroup {
            audio_span: TimeSpan::new(731000, 735000),
            words: vec![FaWord {
                utterance_index: UtteranceIdx(0),
                utterance_word_index: WordIdx(0),
                text: "alright".into(),
            }],
            utterance_indices: vec![UtteranceIdx(0)],
        },
        FaGroup {
            audio_span: TimeSpan::new(637000, 645000),
            words: vec![
                FaWord {
                    utterance_index: UtteranceIdx(1),
                    utterance_word_index: WordIdx(0),
                    text: "so".into(),
                },
                FaWord {
                    utterance_index: UtteranceIdx(1),
                    utterance_word_index: WordIdx(1),
                    text: "take".into(),
                },
                FaWord {
                    utterance_index: UtteranceIdx(1),
                    utterance_word_index: WordIdx(2),
                    text: "a".into(),
                },
                FaWord {
                    utterance_index: UtteranceIdx(1),
                    utterance_word_index: WordIdx(3),
                    text: "look".into(),
                },
                FaWord {
                    utterance_index: UtteranceIdx(1),
                    utterance_word_index: WordIdx(4),
                    text: "at".into(),
                },
                FaWord {
                    utterance_index: UtteranceIdx(1),
                    utterance_word_index: WordIdx(5),
                    text: "all".into(),
                },
                FaWord {
                    utterance_index: UtteranceIdx(1),
                    utterance_word_index: WordIdx(6),
                    text: "of".into(),
                },
                FaWord {
                    utterance_index: UtteranceIdx(1),
                    utterance_word_index: WordIdx(7),
                    text: "them".into(),
                },
            ],
            utterance_indices: vec![UtteranceIdx(1)],
        },
    ];

    // FA responses: group 1 returns timings from the wrong window (backward
    // relative to group 0's correct 731556ms start).
    let responses = vec![
        // Group 0: "alright" — correct.
        vec![Some(WordTiming::new(731556, 733418))],
        // Group 1: wrong window — all timings < 731556ms.
        vec![
            Some(WordTiming::new(639095, 639300)),
            Some(WordTiming::new(639400, 639600)),
            Some(WordTiming::new(639700, 639850)),
            Some(WordTiming::new(639900, 640050)),
            Some(WordTiming::new(640050, 640150)),
            Some(WordTiming::new(640150, 640250)),
            Some(WordTiming::new(640250, 640310)),
            Some(WordTiming::new(640310, 640375)),
        ],
    ];

    apply_fa_results(
        &mut chat,
        &groups,
        &responses,
        FaTimingMode::Continuous,
        false,
    );
    enforce_monotonicity(&mut chat);

    // utt 1 starts at 639095ms < utt 0's 731556ms → non-monotonic → must be stripped.
    let utt1 = get_utterance(&chat, 1);
    assert!(
        utt1.main.content.bullet.is_none(),
        "backward-timestamp utterance (FA wrong audio window) must have bullet stripped; \
         got: {:?}",
        utt1.main.content.bullet
    );

    // utt 0 must be unaffected — it was correctly timed.
    let utt0 = get_utterance(&chat, 0);
    let b0 = utt0
        .main
        .content
        .bullet
        .as_ref()
        .expect("correctly-timed utterance 0 must keep its bullet");
    assert_eq!(b0.timing.start_ms, 731556, "utt 0 start must be 731556ms");
}

// ---------------------------------------------------------------------------
// RED: %wor fast-path must call enforce_monotonicity + strip backward %wor
// ---------------------------------------------------------------------------
//
// When a file already has complete `%wor` timing (e.g. from a previous run),
// `run_fa_from_ast` takes a "fast path": it calls `refresh_existing_alignment`
// to reconstruct main-tier bullets from `%wor`, then returns immediately.
//
// If a previous run wrote backward `%wor` timestamps (as the APROCSA bug did),
// the fast path trusts those timestamps, reconstructs backward main-tier
// bullets, and returns a file that still fails E362 validation.
//
// The fix requires TWO steps after `refresh_existing_alignment`:
//   1. `enforce_monotonicity` — strips backward main-tier bullets.
//   2. `strip_wor_from_monotonicity_stripped_utterances` — removes `%wor`
//      from utterances whose bullets were stripped, so the NEXT re-run cannot
//      reconstruct the backward bullet from stale `%wor` timing again.
//
// Without step 2, the cycle repeats indefinitely: re-run → fast path →
// reconstruct backward bullet from backward %wor → enforce strips it → write
// file with no bullet but still-backward %wor → next re-run repeats.
//
// This test is RED until both steps are added to the fast path.

/// `refresh_existing_alignment` alone propagates backward `%wor` timestamps
/// into main-tier bullets; the fast path must call `enforce_monotonicity` and
/// strip backward `%wor` to break the re-run cycle.
///
/// Two INV utterances:
///   utt0 "alright"  → %wor at 731556 ms  (correct, forward)
///   utt1 "look"     → %wor at 639095 ms  (backward — earlier than utt0)
///
/// After the FIXED fast path:
///   - utt1 must have no main-tier bullet (backward bullet stripped)
///   - utt1 must have no %wor tier (stale backward %wor removed)
///
/// Without the fix, `refresh_existing_alignment` gives utt1 a backward bullet
/// and `%wor` stays, making every re-run perpetuate the violation.
#[test]
fn test_fast_path_strips_backward_wor_timestamps_and_removes_stale_wor_tier() {
    use talkbank_model::model::DependentTier;

    // Two utterances: utt0 correct (731556ms), utt1 backward (639095ms < 733418ms).
    let chat_text = concat!(
        "@UTF8\n",
        "@Begin\n",
        "@Languages:\teng\n",
        "@Participants:\tINV Investigator Adult_Unrelated\n",
        "@ID:\teng|test|INV||female|||Adult_Unrelated|||\n",
        "@Media:\ttest, audio\n",
        "*INV:\talright .\n",
        "%wor:\talright \u{15}731556_733418\u{15} .\n",
        "*INV:\tlook .\n",
        "%wor:\tlook \u{15}639095_639300\u{15} .\n",
        "@End\n",
    );
    let mut chat = parse_chat(chat_text);

    // Fast path precondition: %wor must be reusable for all utterances.
    assert!(
        has_reusable_wor_timing(&chat),
        "precondition: %wor must be complete and reusable"
    );

    // Step 1 (fast path): reconstruct main-tier bullets from %wor.
    refresh_existing_alignment(&mut chat, true);

    // After reconstruction, utt0 has a forward bullet and utt1 has a backward
    // bullet (639095ms < utt0's end time 733418ms).
    let utt1_after_refresh = get_utterance(&chat, 1);
    assert!(
        utt1_after_refresh.main.content.bullet.is_some(),
        "refresh_existing_alignment must reconstruct a bullet from %wor; \
         without fix the fast path returns here with a backward bullet"
    );

    // Step 2 (fast path FIX): call enforce_monotonicity to strip backward bullets.
    let decisions = enforce_monotonicity(&mut chat);

    // Step 3 (fast path FIX): remove %wor from utterances whose bullets were
    // stripped, so the next re-run cannot reconstruct the backward bullet again.
    strip_wor_from_monotonicity_stripped_utterances(&mut chat, &decisions);

    let utt0 = get_utterance(&chat, 0);
    let utt1 = get_utterance(&chat, 1);

    // utt0 retains its forward bullet.
    let b0 = utt0
        .main
        .content
        .bullet
        .as_ref()
        .expect("utt0 must keep its forward bullet");
    assert_eq!(b0.timing.start_ms, 731556, "utt0 start must be 731556ms");

    // utt1's backward bullet must be stripped.
    assert!(
        utt1.main.content.bullet.is_none(),
        "backward utt1 bullet (639095ms < utt0 end {}ms) must be stripped by \
         enforce_monotonicity; got {:?}",
        b0.timing.end_ms,
        utt1.main.content.bullet,
    );

    // utt1's %wor tier must be removed so the next re-run cannot reconstruct
    // the backward bullet from stale %wor timing.  This is the cycle-breaker.
    let utt1_has_wor = utt1
        .dependent_tiers
        .iter()
        .any(|t| matches!(t, DependentTier::Wor(_)));
    assert!(
        !utt1_has_wor,
        "backward %wor tier must be removed from utt1 after bullet is stripped; \
         leaving stale backward %wor causes every re-run to reconstruct the \
         backward bullet perpetuating the E362 violation cycle"
    );
}

// ---------------------------------------------------------------------------
// RED: postprocess must NOT clamp word timings to UTR-hinted bullets
// ---------------------------------------------------------------------------
//
// UTR injects provisional utterance bullets (BulletSource::Utr) based on
// rough ASR timestamps.  These hints can be much narrower than the actual
// speech — e.g., Rev.AI may only recognise the first word of a sentence and
// stamp 220ms for a sentence that actually spans 3 seconds.
//
// The FA engine runs on a wide audio window (the FA group spans many
// utterances).  FA returns correct word timings for every word in the
// utterance.  But `postprocess_utterance_timings` then CLAMPS all word
// timings to the utterance bullet range.  When the bullet is a UTR hint
// of 220ms, every word beyond the first is dropped:
//
//   Utterance: "ooh that happened to me Sunday"  24905_25125 (220ms, UTR)
//   FA returns: ooh→24990-25200, that→25200-26000, happened→26000-27000
//   CURRENTLY: "that" and "happened" are clamped and dropped (start > 25125)
//   FIXED:     no clamping against provisional UTR hints; all words survive
//
// The self-healing path (`update_utterance_bullet` overwrites UTR hints with
// the FA word span) only works when the words SURVIVE postprocessing.  When
// words are dropped the UTR hint persists as a narrow authoritative bullet on
// the next run, and the cycle repeats.
//
// Fix: in `postprocess_utterance_timings`, check `bullet.source` and skip the
// clamping step entirely when the source is `BulletSource::Utr`.

/// `postprocess_utterance_timings` drops words whose FA timings fall outside
/// the utterance bullet when that bullet is a UTR hint.  This test is currently
/// RED: 2 of 3 words are dropped.  The fix must leave all words intact.
#[test]
fn test_postprocess_does_not_clamp_word_timings_to_utr_hint_bullet() {
    use talkbank_model::alignment::helpers::{WordItemMut, walk_words_mut};
    use talkbank_model::model::{Bullet, BulletSource};

    // An utterance with a narrow UTR hint: 24905_25125 (220ms).
    // The real speech spans from ~24990 to ~27000ms — well beyond the hint.
    let input = concat!(
        "@UTF8\n",
        "@Begin\n",
        "@Languages:\teng\n",
        "@Participants:\tCHI Target_Child\n",
        "@ID:\teng|test|CHI||female|||Target_Child|||\n",
        "*CHI:\tooh that happened . \u{0015}24905_25125\u{0015}\n",
        "@End\n",
    );
    let mut chat = parse_chat(input);

    // Mark the bullet as a provisional UTR hint (simulating runtime UTR output).
    {
        let utt = get_test_utterance(&mut chat, 0);
        let bullet = utt
            .main
            .content
            .bullet
            .as_mut()
            .expect("test requires pre-existing UTR bullet");
        bullet.source = BulletSource::Utr;
    }

    // Inject FA word timings that extend well beyond the UTR hint window.
    // "ooh"      : 24990-25200  (partially inside, partially outside 25125)
    // "that"     : 25200-26000  (entirely outside 25125 — currently DROPPED)
    // "happened" : 26000-27000  (entirely outside 25125 — currently DROPPED)
    {
        let utt = get_test_utterance(&mut chat, 0);
        let timings = [
            Some((24990u64, 25200u64)),
            Some((25200, 26000)),
            Some((26000, 27000)),
        ];
        let mut idx = 0;
        walk_words_mut(&mut utt.main.content.content, None, &mut |leaf| {
            if let WordItemMut::Word(w) = leaf {
                if let Some(Some((s, e))) = timings.get(idx) {
                    w.inline_bullet = Some(Bullet::new(*s, *e));
                }
                idx += 1;
            }
        });
    }

    let utt = get_test_utterance(&mut chat, 0);
    let dropped = postprocess_utterance_timings(utt, FaTimingMode::WithPauses);

    // CURRENTLY RED: 2 words ("that" and "happened") are dropped because
    // their timings (25200ms and 26000ms) exceed the UTR hint boundary 25125ms.
    // AFTER FIX: 0 words dropped — UTR hints must not gate word timing acceptance.
    assert_eq!(
        dropped, 0,
        "postprocess must not drop words when the utterance bullet is a provisional \
         UTR hint; 'that' (25200ms) and 'happened' (26000ms) fall outside the \
         220ms UTR window but are valid FA timings that must be preserved"
    );

    // Verify all 3 words have their original FA timings intact.
    let utt = get_utterance(&chat, 0);
    let mut word_timings: Vec<Option<(u64, u64)>> = Vec::new();
    walk_words(&utt.main.content.content, None, &mut |leaf| {
        if let talkbank_model::alignment::helpers::WordItem::Word(w) = leaf {
            word_timings.push(
                w.inline_bullet
                    .as_ref()
                    .map(|b| (b.timing.start_ms, b.timing.end_ms)),
            );
        }
    });
    assert_eq!(
        word_timings,
        vec![
            Some((24990, 25200)),
            Some((25200, 26000)),
            Some((26000, 27000)),
        ],
        "all 3 words must retain their FA-assigned timings after postprocess"
    );
}

/// Regression test: first-time FA alignment must NOT clamp word timings to
/// the utterance bullet even when `BulletSource::Authoritative`.
///
/// When a file is produced by `transcribe` + `utseg`, utterance bullets come
/// from ASR token timestamps (Rev.AI, Whisper, etc.) via UTR.  These are
/// `BulletSource::Authoritative` once serialized and re-parsed (BulletSource
/// is not stored in CHAT text), but they are NOT FA-verified.
///
/// For an utterance like "ooh that happened .", Rev.AI may produce a 220ms
/// bullet (24905_25125) because it only matched "ooh".  FA, given a wider
/// audio window, correctly aligns all words — but postprocess then clamps them
/// to 25125ms, dropping the last 2 words.
///
/// The discriminator: if the utterance has NO `%wor` tier, it has never been
/// FA-aligned before.  The bullet comes from ASR/UTR and is NOT an FA-verified
/// boundary.  Clamping is only safe when `%wor` is present (indicating a
/// previous FA run set the bullet).
///
/// CURRENTLY RED: words dropped when timings fall outside the narrow
/// authoritative bullet, even on first-time alignment (no %wor).
/// AFTER FIX: 0 words dropped when %wor is absent.
#[test]
fn test_postprocess_does_not_clamp_word_timings_on_first_time_alignment_no_wor() {
    use talkbank_model::alignment::helpers::{WordItemMut, walk_words_mut};
    use talkbank_model::model::Bullet;

    // Utterance with a narrow ASR-derived bullet (220ms) and NO %wor tier.
    // This is the exact state produced by `transcribe` + `utseg` before the
    // first `align` run.  BulletSource is Authoritative (the default for
    // all parsed bullets — BulletSource is not persisted to CHAT text).
    let input = concat!(
        "@UTF8\n",
        "@Begin\n",
        "@Languages:\teng\n",
        "@Participants:\tPAR Participant\n",
        "@ID:\teng|test|PAR||female|||Participant|||\n",
        // narrow ASR bullet: 220ms for a ~3s sentence
        "*PAR:\tooh that happened . \u{0015}24905_25125\u{0015}\n",
        // no %wor tier — this is first-time alignment
        "@End\n",
    );
    let mut chat = parse_chat(input);

    // BulletSource is already Authoritative by default.
    // Confirm no %wor tier is present.
    {
        let utt = get_utterance(&chat, 0);
        assert!(
            utt.wor_tier().is_none(),
            "test precondition: utterance must have no %wor tier"
        );
    }

    // Inject FA word timings that extend well beyond the narrow ASR bullet.
    // "ooh"      : 24990-25200  (partially outside 25125ms boundary)
    // "that"     : 25200-26000  (entirely outside 25125ms — currently DROPPED)
    // "happened" : 26000-27000  (entirely outside 25125ms — currently DROPPED)
    {
        let utt = get_test_utterance(&mut chat, 0);
        let timings = [
            Some((24990u64, 25200u64)),
            Some((25200, 26000)),
            Some((26000, 27000)),
        ];
        let mut idx = 0;
        walk_words_mut(&mut utt.main.content.content, None, &mut |leaf| {
            if let WordItemMut::Word(w) = leaf {
                if let Some(Some((s, e))) = timings.get(idx) {
                    w.inline_bullet = Some(Bullet::new(*s, *e));
                }
                idx += 1;
            }
        });
    }

    let utt = get_test_utterance(&mut chat, 0);
    let dropped = postprocess_utterance_timings(utt, FaTimingMode::WithPauses);

    assert_eq!(
        dropped, 0,
        "first-time alignment (no %%wor tier): must not clamp FA word timings \
         to the ASR-derived utterance bullet; 'that' (25200ms) and 'happened' \
         (26000ms) are valid FA timings that must be preserved"
    );

    // Verify all 3 words have their original FA timings intact.
    let utt = get_utterance(&chat, 0);
    let mut word_timings: Vec<Option<(u64, u64)>> = Vec::new();
    walk_words(&utt.main.content.content, None, &mut |leaf| {
        if let WordItem::Word(w) = leaf {
            word_timings.push(
                w.inline_bullet
                    .as_ref()
                    .map(|b| (b.timing.start_ms, b.timing.end_ms)),
            );
        }
    });
    assert_eq!(
        word_timings,
        vec![
            Some((24990, 25200)),
            Some((25200, 26000)),
            Some((26000, 27000)),
        ],
        "all 3 words must retain their FA-assigned timings after postprocess"
    );
}
