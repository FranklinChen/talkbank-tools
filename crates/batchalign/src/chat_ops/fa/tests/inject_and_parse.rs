//! Per-utterance timing injection, FA result parsing (token-level + indexed), boundary estimation, fa-cache-key generation, snapshot probe, and bookkeeping miscellany.

#![allow(unused_imports, dead_code)]

use super::*;

use crate::chat_ops::fa::Placement;
use crate::chat_ops::fa::alignment::token_map::{UntimedReason, WordTimingOutcome};
use crate::chat_ops::fa::alignment::{IndexedWordTimings, apply_indexed_timings};
use crate::chat_ops::fa::coordinates::{FaWindow, FileMs, Ms, Recording};
use crate::chat_ops::fa::origin::{CharEdits, EngineId};
use crate::chat_ops::fa::timing::SpanFault;
use crate::chat_ops::nlp::FaIndexedTiming;
use talkbank_model::UtteranceIdx;
use talkbank_model::model::{Line, UtteranceContent, WriteChat};
use talkbank_parser::TreeSitterParser;

/// The span an estimate placed, failing the test if it refused.
///
/// `estimate_untimed_boundaries` returns a `Placement` because a run whose
/// remaining audio cannot physically hold its words has no span. These fixtures
/// are all comfortably placeable, so a refusal here is a real failure rather
/// than something to unwrap past.
fn placed(estimates: &[Placement], idx: usize) -> TimeSpan {
    match estimates[idx] {
        Placement::Placed(span) => span,
        Placement::Unplaceable(rate) => {
            panic!("estimate {idx} was refused as unplaceable: {rate}")
        }
    }
}

/// A window starting at `start_ms` inside a recording comfortably longer than
/// any timing these tests use.
///
/// The recording has to outlive the window because `FaWindow` refuses to open
/// past the end of its audio, which is the property under test elsewhere.
fn window_at(start_ms: u64) -> (Recording, FaWindow) {
    let recording = Recording::of_duration(Ms(600_000)).expect("non-zero");
    let window = FaWindow::within(
        &recording,
        FileMs::new(start_ms),
        FileMs::new(start_ms + 60_000),
    )
    .expect("window inside the recording");
    (recording, window)
}

fn fa_test_engine() -> EngineId {
    EngineId::new("test-fa")
}

#[test]
fn test_inject_timings_simple() {
    let input = include_str!("../../../../../../test-fixtures/fa_hello_world_timed.cha");
    let mut chat = parse_chat(input);
    let utt = get_test_utterance(&mut chat, 0);

    let timings = vec![
        WordTiming::fixture(100, 500),
        WordTiming::fixture(600, 1000),
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
        WordGapHealing::PreserveMeasured,
        crate::types::engines::FaEngineName::Whisper,
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
        WordGapHealing::PreserveMeasured,
        crate::types::engines::FaEngineName::Whisper,
    );
    assert_eq!(key, key2);

    // Different timing mode -> different key
    let key3 = cache_key(
        &words,
        &AudioIdentity::from_metadata("test.mp3", 1234, 5678),
        0,
        5000,
        WordGapHealing::Heal,
        crate::types::engines::FaEngineName::Whisper,
    );
    assert_ne!(key, key3);
}

#[test]
fn word_interval_cache_key_retires_scoreless_entries_only() {
    let words = vec!["hello".to_string(), "world".to_string()];
    let identity = AudioIdentity::from_metadata("test.mp3", 1234, 5678);
    let legacy_interval_key = crate::chat_ops::CacheKey::from_content(
        "test.mp3|1234|5678|0|5000|hello world|no_pauses|wav2vec_fa",
    );
    let score_bearing_interval_key = cache_key(
        &words,
        &identity,
        0,
        5000,
        WordGapHealing::PreserveMeasured,
        crate::types::engines::FaEngineName::Wave2Vec,
    );
    assert_ne!(score_bearing_interval_key, legacy_interval_key);

    // Whisper has no interval-model score to recover. Keep its established
    // cache namespace so this evidence upgrade does not buy nothing at GPU
    // cost for an onset-only response shape.
    let legacy_whisper_key = crate::chat_ops::CacheKey::from_content(
        "test.mp3|1234|5678|0|5000|hello world|preserve_measured|whisper_fa",
    );
    let current_whisper_key = cache_key(
        &words,
        &identity,
        0,
        5000,
        WordGapHealing::PreserveMeasured,
        crate::types::engines::FaEngineName::Whisper,
    );
    assert_eq!(current_whisper_key, legacy_whisper_key);
}

/// `%wor` is now written by the write phase
/// (`FaApplied::then_enforce_monotonicity`, reached via `then_finalize`),
/// not inline during `apply_fa_results` itself (2026-09-01 review, item 2):
/// a caller that discards the returned `FaApplied` without finalizing gets
/// no `%wor` at all, which is the point of carrying the write as a typed
/// obligation rather than a side effect of injection.
#[test]
fn test_apply_fa_results() {
    let input = include_str!("../../../../../../test-fixtures/fa_hello_world_goodbye_timed.cha");
    let mut chat = parse_chat(input);

    let groups = vec![FaGroup::test_fixture(
        TimeSpan::new(0, 10000),
        vec![
            FaWord {
                utterance_index: UtteranceIdx::new(0),
                utterance_word_index: WordIdx::new(0),
                text: "hello".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx::new(0),
                utterance_word_index: WordIdx::new(1),
                text: "world".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx::new(1),
                utterance_word_index: WordIdx::new(0),
                text: "goodbye".into(),
            },
        ],
        vec![UtteranceIdx::new(0), UtteranceIdx::new(1)],
    )];

    let responses = vec![vec![
        WordTiming::fixture(100, 1000),
        WordTiming::fixture(1500, 3000),
        WordTiming::fixture(5500, 8000),
    ]];

    let _finalized = apply_fa_results(
        &mut chat,
        &groups,
        &responses,
        WordEndPolicy::measured(WordGapHealing::PreserveMeasured),
        true,
    )
    .then_finalize(&mut chat, BulletRepairPolicy::Disabled);

    let output = chat.to_chat_string();
    assert!(output.contains("%wor:"), "Output should contain %wor tier");
}

#[test]
fn test_monotonicity_enforcement() {
    let input = include_str!("../../../../../../test-fixtures/fa_non_monotonic_bullets.cha");
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
        decisions.records().len(),
        1,
        "should have 1 decision for stripped utterance"
    );
    assert_eq!(
        decisions.records()[0].strategy.strategy_name(),
        "timing_stripped"
    );
    assert!(decisions.records()[0].needs_review);
    assert!(matches!(
        decisions.effects(),
        [MonotonicityEffect::StartRegressionStripped {
            utterance_idx,
            start_ms: 2_000,
            previous_start_ms: 5_000,
            previous_utterance_idx,
            ..
        }] if utterance_idx.raw() == 1 && previous_utterance_idx.raw() == 0
    ));
}

#[test]
fn test_monotonicity_clamps_overlapping_end_times() {
    let input = include_str!("../../../../../../test-fixtures/fa_overlapping_end_times.cha");
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
        .records()
        .iter()
        .filter(|d| d.strategy.strategy_name() == "end_clamped_coverage_only")
        .collect();
    assert_eq!(
        clamp_decisions.len(),
        2,
        "should have 2 end_clamped decisions"
    );
    // These fixture utterances have no `%wor` timing, so the clamp trims only
    // container coverage and does not cut a retained word observation.
    assert!(
        !clamp_decisions[0].needs_review,
        "a container-only end clamp must not request review"
    );
}

#[test]
fn monotonicity_clamp_that_cuts_wor_timing_requests_review() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|x|CHI|||||Child|||\n*CHI:\thello world . \u{15}1000_5000\u{15}\n%wor:\thello \u{15}1000_3000\u{15} world \u{15}3000_5000\u{15} .\n*CHI:\tnext . \u{15}4000_6000\u{15}\n%wor:\tnext \u{15}4000_6000\u{15} .\n@End\n";
    let mut chat = parse_chat(input);

    let decisions = enforce_monotonicity(&mut chat);

    let clamp = decisions
        .records()
        .iter()
        .find(|decision| decision.strategy.strategy_name() == "end_clamped_interleaved_words")
        .expect("overlap should produce an end clamp");
    assert!(
        clamp.needs_review,
        "cutting a retained %wor interval must request review"
    );
    assert!(clamp.reason.contains("cuts_word_timing=true"));
    assert!(!clamp.reason.contains("utr_token_range_overlap"));
}

#[test]
fn monotonicity_clamp_that_cuts_fresh_inline_word_timing_requests_review_without_wor() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|x|CHI|||||Child|||\n*CHI:\thello world . \u{15}1000_5000\u{15}\n*CHI:\tnext . \u{15}4000_6000\u{15}\n@End\n";
    let mut chat = parse_chat(input);
    let mut offset = 0;
    inject_timings_for_utterance(
        get_test_utterance(&mut chat, 0),
        &[
            WordTiming::fixture(1_000, 3_000),
            WordTiming::fixture(3_000, 5_000),
        ],
        &mut offset,
    );

    let decisions = enforce_monotonicity(&mut chat);

    let clamp = decisions
        .records()
        .iter()
        .find(|decision| decision.strategy.strategy_name() == "end_clamped_interleaved_words")
        .expect("overlap should produce an end clamp");
    assert!(
        clamp.needs_review,
        "fresh inline FA evidence must not become invisible merely because %wor output is disabled"
    );
    assert!(clamp.reason.contains("cuts_word_timing=true"));
}

#[test]
fn cross_speaker_overlap_policy_preserves_only_cross_speaker_end_overlap() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child, MOT Mother\n@ID:\teng|x|CHI|||||Child|||\n@ID:\teng|x|MOT|||||Mother|||\n*CHI:\tone . \u{15}1000_5000\u{15}\n*MOT:\ttwo . \u{15}4000_8000\u{15}\n*MOT:\tthree . \u{15}7000_9000\u{15}\n@End\n";
    let mut chat = parse_chat(input);

    let result =
        enforce_monotonicity_with_policy(&mut chat, EndOverlapPolicy::PreserveCrossSpeaker);

    assert_eq!(
        get_test_utterance(&mut chat, 0)
            .main
            .content
            .bullet
            .as_ref()
            .expect("cross-speaker timing retained")
            .timing
            .end_ms,
        5_000
    );
    assert_eq!(
        get_test_utterance(&mut chat, 1)
            .main
            .content
            .bullet
            .as_ref()
            .expect("same-speaker timing retained")
            .timing
            .end_ms,
        7_000
    );
    assert!(matches!(
        result.effects(),
        [MonotonicityEffect::EndClampedCoverageOnly { edge, .. }]
            if edge.utterance_idx.raw() == 1 && edge.next_utterance_idx.raw() == 2
    ));
}

#[test]
fn fa_applied_carries_end_overlap_policy_into_finalization() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child, MOT Mother\n@ID:\teng|x|CHI|||||Child|||\n@ID:\teng|x|MOT|||||Mother|||\n*CHI:\tone . \u{15}1000_5000\u{15}\n*MOT:\ttwo . \u{15}4000_8000\u{15}\n@End\n";
    let mut chat = parse_chat(input);
    let policy = FaProjectionPolicy::new(
        WordEndPolicy::measured(WordGapHealing::Heal),
        ExistingWorBoundaryPolicy::Preserve,
        EndOverlapPolicy::PreserveCrossSpeaker,
    );

    let _finalized = apply_fa_results_with_projection_policy(&mut chat, &[], &[], policy, false)
        .then_finalize(&mut chat, BulletRepairPolicy::Disabled);

    assert_eq!(
        get_test_utterance(&mut chat, 0)
            .main
            .content
            .bullet
            .as_ref()
            .expect("cross-speaker timing retained")
            .timing
            .end_ms,
        5_000
    );
}

#[test]
fn no_injection_projection_carries_end_overlap_policy_into_ordering() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child, MOT Mother\n@ID:\teng|x|CHI|||||Child|||\n@ID:\teng|x|MOT|||||Mother|||\n*CHI:\tone . \u{15}1000_5000\u{15}\n*MOT:\ttwo . \u{15}4000_8000\u{15}\n@End\n";
    let mut chat = parse_chat(input);
    let policy = FaProjectionPolicy::new(
        WordEndPolicy::measured(WordGapHealing::Heal),
        ExistingWorBoundaryPolicy::Preserve,
        EndOverlapPolicy::PreserveCrossSpeaker,
    );

    let _finalized = finalize_without_injection(&mut chat, policy, BulletRepairPolicy::Disabled);

    assert_eq!(
        get_test_utterance(&mut chat, 0)
            .main
            .content
            .bullet
            .as_ref()
            .expect("cross-speaker timing retained")
            .timing
            .end_ms,
        5_000
    );
}

#[test]
fn fa_finalization_runs_optional_repair_before_monotonicity() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child, MOT Mother\n@ID:\teng|x|CHI|||||Child|||\n@ID:\teng|x|MOT|||||Mother|||\n*CHI:\tone . \u{15}1000_5000\u{15}\n*MOT:\ttwo . \u{15}4800_8000\u{15}\n@End\n";
    let mut chat = parse_chat(input);
    let policy = FaProjectionPolicy::new(
        WordEndPolicy::measured(WordGapHealing::Heal),
        ExistingWorBoundaryPolicy::Preserve,
        EndOverlapPolicy::ClampAllAdjacent,
    );

    let finalized = apply_fa_results_with_projection_policy(&mut chat, &[], &[], policy, false)
        .then_finalize(&mut chat, BulletRepairPolicy::Enabled);

    assert_eq!(finalized.repair_stats().boundary_averaged, 1);
    assert_eq!(
        get_test_utterance(&mut chat, 0)
            .main
            .content
            .bullet
            .as_ref()
            .expect("earlier timing retained")
            .timing
            .end_ms,
        4_900
    );
    assert_eq!(
        get_test_utterance(&mut chat, 1)
            .main
            .content
            .bullet
            .as_ref()
            .expect("later timing retained")
            .timing
            .start_ms,
        4_900
    );
}

/// Every word the indexed path leaves untimed SAYS WHY, per word.
///
/// The fact was modelled twice and only one copy could be read: the token path
/// answered with a typed [`UntimedReason`] per word, while this path answered
/// with a `None` in a vector and a counter beside it, so "the host declined
/// this word", "the engine's span had no width" and "the engine answered about
/// audio it was not given" were one and the same `None` to every caller. Only
/// the aggregate counters told them apart, and only in a log line.
#[test]
fn test_apply_indexed_timings_says_why_each_word_has_no_timing() {
    let words = make_fa_words(&["alpha", "beta", "gamma", "delta"]);
    let indexed = vec![
        // Timed and usable.
        Some(FaIndexedTiming {
            start_ms: 100,
            end_ms: 400,
            confidence: None,
        }),
        // The host declined to time this word.
        None,
        // A span with no width: the engine answered and its answer is unusable.
        Some(FaIndexedTiming {
            start_ms: 500,
            end_ms: 500,
            confidence: None,
        }),
        // Past the end of the 60 s window the engine was handed, which is the
        // shape that wrote phantom speech into six sessions.
        Some(FaIndexedTiming {
            start_ms: 61_000,
            end_ms: 62_000,
            confidence: None,
        }),
    ];
    let (outcomes, _discarded) = apply_indexed_timings(
        IndexedWordTimings::pair(&words, &indexed).expect("one timing slot per word"),
        &window_at(0).1,
        &fa_test_engine(),
    );

    assert!(
        matches!(outcomes[0], WordTimingOutcome::Timed { .. }),
        "a usable timing must survive: {:?}",
        outcomes[0]
    );
    assert!(
        matches!(
            outcomes[1],
            WordTimingOutcome::Untimed {
                reason: UntimedReason::NoTimingFromHost
            }
        ),
        "the host declined this word: {:?}",
        outcomes[1]
    );
    assert!(
        matches!(
            outcomes[2],
            WordTimingOutcome::Untimed {
                reason: UntimedReason::SpanRefused(SpanFault::NoExtent { .. })
            }
        ),
        "a zero-width span is refused by width, not by absence: {:?}",
        outcomes[2]
    );
    assert!(
        matches!(
            outcomes[3],
            WordTimingOutcome::Untimed {
                reason: UntimedReason::ReportedOutsideWindow(_)
            }
        ),
        "a report past the end of the window says so: {:?}",
        outcomes[3]
    );
}

/// The warn line's per-word counters are DERIVED from those reasons.
///
/// The counters used to be hand-maintained fields incremented beside the
/// `None` they described, so nothing tied the number to the word. Deriving
/// them means a reason cannot be reported in one place and counted in another;
/// this pins the arithmetic on a group carrying one of each class.
#[test]
fn test_indexed_untimed_counters_are_derived_from_the_reasons() {
    let words = make_fa_words(&["alpha", "beta", "gamma", "delta"]);
    let indexed = vec![
        Some(FaIndexedTiming {
            start_ms: 100,
            end_ms: 400,
            confidence: None,
        }),
        None,
        Some(FaIndexedTiming {
            start_ms: 500,
            end_ms: 500,
            confidence: None,
        }),
        Some(FaIndexedTiming {
            start_ms: 61_000,
            end_ms: 62_000,
            confidence: None,
        }),
    ];
    let (outcomes, discarded) = apply_indexed_timings(
        IndexedWordTimings::pair(&words, &indexed).expect("one timing slot per word"),
        &window_at(0).1,
        &fa_test_engine(),
    );
    assert_eq!(
        discarded.tally(&outcomes).notable(),
        3,
        "one declined word, one zero-width span and one out-of-window report"
    );
}

/// A word the HOST returned no timing for is REPORTED, not silently skipped.
///
/// Downstream a word the host never timed is byte-identical to one whose
/// timing was REJECTED for landing outside the window, and only the second
/// warned. A Qwen3 group answering with three of ten words untimed emitted no
/// diagnostic at all, so nothing distinguished "the engine declined these
/// three" from "the engine timed everything".
///
/// It asserts the REASON now rather than a counter. The counter was a second
/// representation of this fact and was deleted with the field it lived on; the
/// warn line's `untimed_by_host` is counted from these reasons.
#[test]
fn test_apply_indexed_timings_reports_words_the_host_left_untimed() {
    let words = make_fa_words(&["alpha", "beta", "gamma"]);
    let indexed = vec![
        Some(FaIndexedTiming {
            start_ms: 100,
            end_ms: 400,
            confidence: None,
        }),
        None,
        None,
    ];
    let (outcomes, discarded) = apply_indexed_timings(
        IndexedWordTimings::pair(&words, &indexed).expect("one timing slot per word"),
        &window_at(0).1,
        &fa_test_engine(),
    );
    assert!(matches!(outcomes[0], WordTimingOutcome::Timed { .. }));
    let declined = outcomes
        .iter()
        .filter(|outcome| {
            matches!(
                outcome,
                WordTimingOutcome::Untimed {
                    reason: UntimedReason::NoTimingFromHost
                }
            )
        })
        .count();
    assert_eq!(declined, 2);
    assert_eq!(
        discarded.tally(&outcomes).notable(),
        2,
        "an untimed word is on its own enough to make the group worth reporting"
    );
}

/// Nothing to report when every word came back timed and usable.
#[test]
fn test_apply_indexed_timings_reports_nothing_when_every_word_is_timed() {
    let words = make_fa_words(&["alpha"]);
    let indexed = vec![Some(FaIndexedTiming {
        start_ms: 100,
        end_ms: 400,
        confidence: None,
    })];
    let (outcomes, discarded) = apply_indexed_timings(
        IndexedWordTimings::pair(&words, &indexed).expect("one timing slot per word"),
        &window_at(0).1,
        &fa_test_engine(),
    );
    assert!(matches!(outcomes[0], WordTimingOutcome::Timed { .. }));
    assert_eq!(discarded.tally(&outcomes).notable(), 0);
}

#[test]
fn test_parse_fa_response_token_level() {
    let json = r#"{"tokens": [
            {"text": "hello", "time_s": 0.1},
            {"text": "world", "time_s": 0.6}
        ]}"#;
    let words = make_fa_words(&["hello", "world"]);
    let timings = parse_fa_response(json, &words, &window_at(0).1, &fa_test_engine()).unwrap();
    assert_eq!(timings.len(), 2);
    // Until 2026-08-14 both words asserted `end_ms == start_ms`. That is not
    // a timing, it is the absence of one, and the six-week run of
    // zero-duration `%wor` tiers is what it looks like in the output. An
    // onset-only engine gives no end, so each word ends where the next
    // begins, and the last takes the named fallback because it has no
    // successor.
    // The origins are asserted, not just the numbers: word 0's end is the next
    // word's ONSET (an inference), and word 1's is the named fallback (an
    // invention). Neither is a measurement of when the word stopped, and the
    // output now says so.
    assert_eq!(
        timings[0],
        WordTiming::new(
            100,
            600,
            Origin::EngineMeasured {
                engine: fa_test_engine()
            },
            Origin::DerivedFromNextOnset
        )
    );
    assert_eq!(
        timings[1],
        WordTiming::new(
            600,
            1100,
            Origin::EngineMeasured {
                engine: fa_test_engine()
            },
            Origin::FallbackDuration {
                assumed: Ms(LAST_WORD_FALLBACK_MS)
            }
        )
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
    let timings = parse_fa_response(json, &words, &window_at(3000).1, &fa_test_engine()).unwrap();
    assert_eq!(
        timings[0],
        WordTiming::new(
            3100,
            3600,
            Origin::EngineMeasured {
                engine: fa_test_engine()
            },
            Origin::DerivedFromNextOnset
        )
    );
    assert_eq!(
        timings[1],
        WordTiming::new(
            3600,
            4100,
            Origin::EngineMeasured {
                engine: fa_test_engine()
            },
            Origin::FallbackDuration {
                assumed: Ms(LAST_WORD_FALLBACK_MS)
            }
        )
    );
}

/// An engine label the transcript does not contain must cost NOTHING but
/// itself.
///
/// The policy this test asserts CHANGED on 2026-09-07, and the docstring is
/// changed with it rather than left describing a loss. It used to end
/// `assert_eq!(timings[1], None)`: the deterministic stitch failed on "world"
/// (the extra label "there" is not a prefix of it), and a failed stitch ended
/// the group, so a word whose own onset the engine had reported perfectly well
/// came back untimed. That was the abort, and it is gone: the residue of a
/// broken stitch is now remapped by character alignment, which puts "world"
/// back on the label that spells it.
#[test]
fn test_parse_fa_response_token_level_mismatch_does_not_skip_tokens() {
    let json = r#"{"tokens": [
            {"text": "hello", "time_s": 0.1},
            {"text": "there", "time_s": 0.2},
            {"text": "world", "time_s": 0.6}
        ]}"#;
    let words = make_fa_words(&["hello", "world"]);
    let timings = parse_fa_response(json, &words, &window_at(0).1, &fa_test_engine()).unwrap();
    // "hello" ends at the next token's onset even though that token is the
    // unmatched "there": the end comes from the audio, not from whether
    // stitching went on to succeed.
    assert_eq!(
        timings[0],
        WordTiming::new(
            100,
            200,
            Origin::EngineMeasured {
                engine: fa_test_engine()
            },
            Origin::DerivedFromNextOnset
        )
    );
    // "world" is placed on its own label, and "there" is left owned by nobody.
    // Every character of the word and of that label matched, so NEITHER end is
    // wrapped: an exactly reconciled fold is the same fact the in-order stitch
    // proves, and claiming an attribution over it would report a guess that was
    // never made.
    //
    // Both ends carried `AttributedByCharAlignment { edits: ZERO }` until the
    // route was folded into the edit counts on 2026-09-07. That wrapper said
    // only WHICH CODE placed the word, and the reviewer's tally read it as our
    // own answer. Where a fold really is inexact, both ends are still wrapped,
    // which is what `both_boundaries_of_a_remapped_word_are_attributed` pins.
    assert_eq!(
        timings[1],
        WordTiming::new(
            600,
            1_100,
            Origin::EngineMeasured {
                engine: fa_test_engine()
            },
            Origin::FallbackDuration {
                assumed: Ms(LAST_WORD_FALLBACK_MS)
            }
        )
    );
}

#[test]
fn test_parse_fa_response_token_level_zero_width_word_does_not_erase_suffix() {
    let json = r#"{"tokens": [
            {"text": "once", "time_s": 0.1},
            {"text": "upon", "time_s": 0.1},
            {"text": "a", "time_s": 0.6},
            {"text": "time", "time_s": 0.8}
        ]}"#;
    let words = make_fa_words(&["once", "upon", "a", "time"]);
    let timings = parse_fa_response(json, &words, &window_at(0).1, &fa_test_engine()).unwrap();

    // Equal reported onsets cannot prove a positive interval for `once`, so it
    // remains untimed.  That local span failure is not a lexical mismatch:
    // every later word still has an independently stitched onset and must not
    // be erased with the suffix.
    assert_eq!(timings[0], None);
    assert_eq!(timings[1].as_ref().map(|timing| timing.start_ms), Some(100));
    assert_eq!(timings[1].as_ref().map(|timing| timing.end_ms), Some(600));
    assert_eq!(timings[2].as_ref().map(|timing| timing.start_ms), Some(600));
    assert_eq!(timings[2].as_ref().map(|timing| timing.end_ms), Some(800));
    assert_eq!(timings[3].as_ref().map(|timing| timing.start_ms), Some(800));
}

#[test]
fn test_parse_fa_response_indexed_word_level() {
    let json = r#"{"indexed_timings": [
            {"start_ms": 100, "end_ms": 500, "confidence": 0.73},
            {"start_ms": 600, "end_ms": 1000}
        ]}"#;
    let words = make_fa_words(&["hello", "world"]);
    let timings = parse_fa_response(json, &words, &window_at(5000).1, &fa_test_engine()).unwrap();
    assert_eq!(timings.len(), 2);
    assert_eq!(timings[0].as_ref().unwrap().start_ms, 5100);
    assert_eq!(timings[0].as_ref().unwrap().end_ms, 5500);
    assert_eq!(
        timings[0]
            .as_ref()
            .unwrap()
            .model_score()
            .unwrap()
            .millionths(),
        730_000
    );
    assert_eq!(timings[1].as_ref().unwrap().start_ms, 5600);
    assert_eq!(timings[1].as_ref().unwrap().end_ms, 6000);
    assert!(timings[1].as_ref().unwrap().model_score().is_none());
}

#[test]
fn test_parse_fa_response_indexed_length_mismatch_rejected() {
    use crate::chat_ops::fa::alignment::FaAlignmentError;
    let json = r#"{"indexed_timings": [{"start_ms": 100, "end_ms": 500}]}"#;
    let words = make_fa_words(&["hello", "world"]);
    let err = parse_fa_response(json, &words, &window_at(0).1, &fa_test_engine()).unwrap_err();
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

/// A gap whose bounds lie PAST the end of the recording must not yield an
/// inverted window.
///
/// The producer clamps both ends now. Before that it clamped only the end, so a
/// buffered start beyond the audio met an end cut down to it and
/// `TimeSpan::new(start, end)` was built with `end < start`, which its own
/// docstring says the caller must prevent. Transcript bullets past the end of
/// the recording are exactly the case the coordinate work exists for, so this
/// is reachable rather than theoretical.
#[test]
fn a_gap_beyond_the_recording_is_refused_not_inverted() {
    let input = "\
@UTF8\n\
@Begin\n\
@Languages:\teng\n\
@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|2;0.0||||Target_Child|||\n\
*CHI:\thello . \u{0015}90000_95000\u{0015}\n\
*CHI:\tuntimed words here .\n\
*CHI:\tworld . \u{0015}96000_99000\u{0015}\n\
@End\n\
";
    let chat = parse_chat(input);
    // A recording far shorter than the bullets claim.
    let estimated = estimate_untimed_boundaries(&chat, &test_recording(10_000));
    for (idx, placement) in estimated.placements.iter().enumerate() {
        if let Placement::Placed(span) = placement {
            assert!(
                span.end_ms >= span.start_ms,
                "placement {idx} is inverted: {}..{}",
                span.start_ms,
                span.end_ms
            );
        }
    }
}

#[test]
fn test_estimate_boundaries_proportional() {
    let input = include_str!("../../../../../../test-fixtures/fa_two_untimed_with_media.cha");
    let chat = parse_chat(input);
    // The count is asserted, not just the placements: `windows_clamped` exists
    // because `.min(total_audio_ms)` made "this window overshot the recording"
    // unobservable, and a counter nobody checks would reproduce that exactly.
    // This fixture's second estimate is buffered past 10 s and cut back to it.
    let estimated = estimate_untimed_boundaries(&chat, &test_recording(10_000));
    assert!(
        estimated.windows_clamped > 0,
        "a window buffered past the end of a 10 s recording must be counted as clamped"
    );
    let estimates = estimated.placements;
    assert_eq!(estimates.len(), 2);
    assert_eq!(placed(&estimates, 0).start_ms, 0);
    assert_eq!(placed(&estimates, 0).end_ms, 7000);
    assert_eq!(placed(&estimates, 1).start_ms, 3000);
    assert_eq!(placed(&estimates, 1).end_ms, 10000);
}

#[test]
fn test_estimate_boundaries_interpolates_from_neighbors() {
    let input =
        include_str!("../../../../../../test-fixtures/fa_mixed_timed_untimed_interleaved.cha");
    let chat = parse_chat(input);
    let estimates = estimate_untimed_boundaries(&chat, &test_recording(50_000)).placements;

    // 6 utterances total
    assert_eq!(estimates.len(), 6);

    // utt 0: timed (10000-15000), estimate mirrors real bullet
    assert_eq!(placed(&estimates, 0), TimeSpan::new(10000, 15000));

    // utt 1: untimed, between timed utt 0 (end=15000) and utt 2 (start=20000)
    // Gap = [15000, 20000], 4 words, only utterance in run
    // raw: 15000-20000, with 2s buffer: 13000-22000
    assert_eq!(placed(&estimates, 1).start_ms, 13000);
    assert_eq!(placed(&estimates, 1).end_ms, 22000);

    // utt 2: timed (20000-25000)
    assert_eq!(placed(&estimates, 2), TimeSpan::new(20000, 25000));

    // utt 3: untimed, in run [3,4] between timed utt 2 (end=25000) and utt 5 (start=40000)
    // Gap = [25000, 40000] = 15000ms, run_words = 4+5 = 9
    // utt 3 (4 words): raw 25000..31666, buffered 23000..33666
    assert_eq!(placed(&estimates, 3).start_ms, 23000);
    assert_eq!(placed(&estimates, 3).end_ms, 33666);

    // utt 4 (5 words): raw 31666..40000, buffered 29666..42000
    assert_eq!(placed(&estimates, 4).start_ms, 29666);
    assert_eq!(placed(&estimates, 4).end_ms, 42000);

    // utt 5: timed (40000-45000)
    assert_eq!(placed(&estimates, 5), TimeSpan::new(40000, 45000));
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
        gap_healing: WordGapHealing::PreserveMeasured,
    };
    insta::assert_json_snapshot!(item);
}

#[test]
fn test_apply_fa_results_excludes_xxx_from_wor_tier() {
    // Fixture: an utterance with `xxx`. The stale %wor has 5 words (no xxx)
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

    // FA group: 5 words extracted by collect_fa_words, xxx is excluded because
    // untranscribed tokens have no alignable phoneme sequence.
    let groups = vec![FaGroup::test_fixture(
        TimeSpan::new(27602, 28323),
        vec![
            FaWord {
                utterance_index: UtteranceIdx::new(0),
                utterance_word_index: WordIdx::new(0),
                text: "last".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx::new(0),
                utterance_word_index: WordIdx::new(1),
                text: "time".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx::new(0),
                utterance_word_index: WordIdx::new(2),
                text: "I".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx::new(0),
                utterance_word_index: WordIdx::new(3),
                text: "saw".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx::new(0),
                utterance_word_index: WordIdx::new(4),
                text: "you".into(),
            },
            // xxx is NOT in the FA group, not sent to the aligner.
        ],
        vec![UtteranceIdx::new(0)],
    )];

    // FA response: 5 timings for the 5 real words.
    let responses = vec![vec![
        WordTiming::fixture(27602, 27762),
        WordTiming::fixture(27762, 27942),
        WordTiming::fixture(27942, 28002),
        WordTiming::fixture(28002, 28203),
        WordTiming::fixture(28203, 28323),
    ]];

    let _ = apply_fa_results(
        &mut chat,
        &groups,
        &responses,
        WordEndPolicy::measured(WordGapHealing::PreserveMeasured),
        true,
    );

    let output = chat.to_chat_string();

    // The %wor tier must have 5 word entries, `xxx` is excluded, no slot for it.
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
