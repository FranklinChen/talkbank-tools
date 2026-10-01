//! Anchored splitting at its real boundaries: UTR's plan produces the
//! anchors, grouping turns an over-budget utterance into an anchored group or
//! an `anchor_gap` refusal, and an anchored group's timings inject exactly as
//! a single group's would.

use super::*;

use batchalign_transform::decisions::{DecisionStrategy, FaStrategy, RefusedWindow, SplitWindow};
use talkbank_model::UtteranceIdx;
use talkbank_model::model::WriteChat;

use crate::chat_ops::fa::coordinates::Ms;
use crate::chat_ops::fa::utr::{AsrTimingToken, inject_utr_timing};

const BUDGET_MS: u64 = 15_000;

const WORDS: [&str; 12] = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve",
];

/// One untimed utterance of twelve words, so UTR gives it its bullet.
fn untimed_twelve_words() -> String {
    format!(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n\
         @ID:\teng|test|PAR|||||Participant|||\n*PAR:\t{} .\n@End\n",
        WORDS.join(" ")
    )
}

/// One single-word ASR token per transcript word, at the given starts, each
/// 800 ms long.
fn tokens_at(starts_ms: &[u64]) -> Vec<AsrTimingToken> {
    WORDS
        .iter()
        .zip(starts_ms)
        .map(|(word, &start_ms)| AsrTimingToken {
            text: (*word).to_owned(),
            start_ms,
            end_ms: start_ms + 800,
        })
        .collect()
}

/// UTR over the fixture, then grouping with the anchors UTR observed.
fn group_after_utr(starts_ms: &[u64], recording_ms: u64) -> Grouping {
    let mut chat = parse_chat(&untimed_twelve_words());
    let mut utr = inject_utr_timing(&mut chat, &tokens_at(starts_ms));
    assert_eq!(utr.injected(), 1, "UTR placed the utterance");
    let anchors = utr.take_anchors();
    group_utterances(&chat, BUDGET_MS, &test_recording(recording_ms), &anchors)
}

/// Dense anchors, 3 s apart over 34 s: the utterance becomes one anchored
/// group whose pieces fit the budget and partition its words, and the split
/// is recorded as a decision that needs no review.
#[test]
fn an_over_budget_utterance_with_dense_anchors_becomes_an_anchored_group() {
    let starts: Vec<u64> = (0..12).map(|i| i * 3_000).collect();
    let grouped = group_after_utr(&starts, 60_000);

    assert_eq!(grouped.groups.len(), 1);
    let group = &grouped.groups[0];
    assert_eq!(group.utterance_indices(), [UtteranceIdx::new(0)]);
    assert_eq!(group.word_count(), 12);
    let GroupSpan::Anchored(split) = group.span() else {
        panic!("a 33.8 s utterance with anchors every 3 s is split");
    };
    let pieces: Vec<&AnchoredPiece> = split.pieces().collect();
    assert!(
        pieces
            .iter()
            .all(|piece| piece.window().len() <= Ms(BUDGET_MS))
    );
    assert_eq!(
        group
            .words()
            .map(|word| word.utterance_word_index.raw())
            .collect::<Vec<_>>(),
        (0..12).collect::<Vec<_>>()
    );
    // Every cut is the end of an anchored word: 800 ms after a 3 s start.
    for piece in &pieces[..pieces.len() - 1] {
        assert_eq!(piece.window().end().get() % 3_000, 800);
    }
    // The group spans the utterance window UTR gave it, and its last piece
    // is padded into the following silence as a single group's window is:
    // the 1.5 s cap, well inside the budget.
    assert_eq!(group.audio_start_ms(), 0);
    assert_eq!(group.audio_end_ms(), 35_300);

    assert_eq!(grouped.decisions.len(), 1);
    let decision = &grouped.decisions[0];
    assert_eq!(
        decision.strategy,
        DecisionStrategy::Fa(FaStrategy::WindowSplitAtAnchors(SplitWindow {
            start_ms: 0,
            end_ms: 33_800,
            budget_ms: BUDGET_MS,
            pieces: split.piece_count(),
        }))
    );
    assert!(!decision.needs_review);
}

/// Six words in the first six seconds and six more five minutes later: the
/// anchors leave a stretch no piece can cross, so the utterance is refused
/// as an `anchor_gap` naming that stretch, for review, and never aligned.
#[test]
fn a_minutes_long_anchor_gap_is_refused_for_review() {
    let starts: Vec<u64> = (0..6)
        .map(|i| i * 1_000)
        .chain((0..6).map(|i| 300_000 + i * 1_000))
        .collect();
    let grouped = group_after_utr(&starts, 400_000);

    assert!(grouped.groups.is_empty(), "nothing is sent to the aligner");
    assert_eq!(grouped.decisions.len(), 1);
    let expected = RefusedWindow::AnchorGap {
        start_ms: 0,
        end_ms: 305_800,
        budget_ms: BUDGET_MS,
        // The end of "six", then the end of "seven".
        gap_start_ms: 5_800,
        gap_end_ms: 300_800,
    };
    assert_eq!(
        grouped.decisions[0].strategy,
        DecisionStrategy::Fa(FaStrategy::WindowRefused(expected))
    );
    assert_eq!(grouped.decisions[0].reason, expected.to_string());
    assert!(grouped.decisions[0].needs_review);
}

/// Without a UTR pass there are no anchors, and an over-budget utterance is
/// refused exactly as it was before anchored splitting existed.
#[test]
fn without_anchors_an_over_budget_utterance_is_refused_as_before() {
    let mut chat = parse_chat(&untimed_twelve_words());
    let starts: Vec<u64> = (0..12).map(|i| i * 3_000).collect();
    let _ = inject_utr_timing(&mut chat, &tokens_at(&starts));
    let grouped = group_utterances(
        &chat,
        BUDGET_MS,
        &test_recording(60_000),
        &AnchorIndex::not_observed(),
    );
    assert!(grouped.groups.is_empty());
    assert_eq!(
        grouped.decisions[0].strategy,
        DecisionStrategy::Fa(FaStrategy::WindowRefused(RefusedWindow::OverBudget {
            start_ms: 0,
            end_ms: 33_800,
            budget_ms: BUDGET_MS,
        }))
    );
}

/// Only exact and case-insensitive matches to single-word tokens anchor: a
/// token holding two words says when the pair was said, not when either
/// word ended, so no anchor is taken from it.
#[test]
fn a_multi_word_token_is_not_an_anchor() {
    let mut chat = parse_chat(&untimed_twelve_words());
    let mut tokens = tokens_at(&(0..12).map(|i| i * 3_000).collect::<Vec<_>>());
    // "three four" as one provider segment.
    tokens[2].text = "three four".to_owned();
    tokens.remove(3);
    let anchors = inject_utr_timing(&mut chat, &tokens).take_anchors();
    let AnchorLookup::Anchored(utterance) = anchors.lookup(UtteranceIdx::new(0)) else {
        panic!("the single-word tokens still anchor");
    };
    let anchored: Vec<usize> = utterance
        .anchors()
        .iter()
        .map(|anchor| anchor.word().raw())
        .collect();
    assert!(!anchored.contains(&2) && !anchored.contains(&3));
    assert!(anchored.contains(&1) && anchored.contains(&4));
}

/// THE injection seam: an anchored group's timings, assembled in word order,
/// write exactly the transcript a single group over the same words writes.
#[test]
fn an_anchored_group_injects_exactly_as_a_single_group() {
    let text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n\
                @ID:\teng|test|PAR|||||Participant|||\n@Media:\ttest, audio\n\
                *PAR:\tone two three four five . \u{15}0_40000\u{15}\n@End\n";
    let anchors = AnchorIndex::fixture([(
        UtteranceIdx::new(0),
        UtteranceAnchors::fixture(
            5,
            vec![
                WordAnchor::fixture(1, 10_000, 13_000),
                WordAnchor::fixture(3, 24_000, 26_000),
            ],
        ),
    )]);
    let mut anchored_chat = parse_chat(text);
    let mut single_chat = parse_chat(text);

    let anchored =
        group_utterances(&anchored_chat, BUDGET_MS, &test_recording(40_000), &anchors).groups;
    assert!(matches!(anchored[0].span(), GroupSpan::Anchored(_)));
    let single = vec![FaGroup::test_fixture(
        TimeSpan::new(0, 40_000),
        anchored[0].words().cloned().collect(),
        vec![UtteranceIdx::new(0)],
    )];

    let timings = vec![vec![
        WordTiming::fixture(500, 9_000),
        WordTiming::fixture(10_000, 13_000),
        WordTiming::fixture(14_000, 20_000),
        WordTiming::fixture(24_000, 26_000),
        WordTiming::fixture(27_000, 39_000),
    ]];
    for (chat, groups) in [(&mut anchored_chat, &anchored), (&mut single_chat, &single)] {
        let _ = apply_fa_results(
            chat,
            groups,
            &timings,
            WordEndPolicy::measured(WordGapHealing::PreserveMeasured),
            true,
        )
        .then_finalize(chat, BulletRepairPolicy::Disabled)
        .expect("default policy finalizes");
    }

    assert_eq!(anchored_chat.to_chat_string(), single_chat.to_chat_string());
    assert!(anchored_chat.to_chat_string().contains("%wor:"));
}
