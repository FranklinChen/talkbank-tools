//! Anchored-region boundary tests: synthetic transcripts through the public
//! injection entry point, with the global problem's refusal shown directly.

use super::*;
use batchalign_transform::dp_align::interleaving::{LocalInterleaving, SpeakerTurn};
use batchalign_transform::parse_source_with_parser;
use talkbank_model::{
    NullErrorSink,
    model::{FileStem, TranscriptName},
};
use talkbank_parser::TreeSitterParser;

/// One synthetic turn: speaker, its distinct alphabetic words, the span it
/// was "spoken" in, and whether the transcript carries that span as a bullet.
struct Turn {
    speaker: &'static str,
    words: Vec<String>,
    start_ms: u64,
    end_ms: u64,
    timed: bool,
}

/// A distinct, CHAT-valid lowercase word for each ordinal.
fn word(ordinal: usize) -> String {
    let letters = |mut n: usize| {
        let mut out = Vec::new();
        for _ in 0..3 {
            out.push(b'a' + (n % 26) as u8);
            n /= 26;
        }
        String::from_utf8(out).expect("ascii")
    };
    format!("ka{}", letters(ordinal))
}

/// Alternating two-speaker turns of three words, one second apart.
/// `timed(index)` says which turns the transcript bullets.
fn dialogue(turns: usize, timed: impl Fn(usize) -> bool) -> Vec<Turn> {
    (0..turns)
        .map(|index| Turn {
            speaker: if index % 2 == 0 { "PAR" } else { "INV" },
            words: (0..3).map(|j| word(index * 3 + j)).collect(),
            start_ms: index as u64 * 1_000,
            end_ms: index as u64 * 1_000 + 900,
            timed: timed(index),
        })
        .collect()
}

fn chat(turns: &[Turn]) -> ChatFile {
    // A transcript with no bullet declares its media unlinked (E544).
    let linkage = if turns.iter().any(|turn| turn.timed) {
        ""
    } else {
        ", unlinked"
    };
    let mut text = format!(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant, INV Investigator\n\
         @ID:\teng|test|PAR|||||Participant|||\n@ID:\teng|test|INV|||||Investigator|||\n\
         @Media:\tinput, audio{linkage}\n",
    );
    for turn in turns {
        text.push_str(&format!("*{}:\t{} .", turn.speaker, turn.words.join(" ")));
        if turn.timed {
            text.push_str(&format!(" \u{15}{}_{}\u{15}", turn.start_ms, turn.end_ms));
        }
        text.push('\n');
    }
    text.push_str("@End\n");
    let parser = TreeSitterParser::new().expect("parser");
    parse_source_with_parser(&parser, &text)
        .admit(
            TranscriptName::Named(
                FileStem::from_path(std::path::Path::new("input.cha")).expect("name"),
            ),
            &NullErrorSink,
        )
        .expect("synthetic CHAT admission")
        .into_valid_file()
        .into_unchecked()
}

/// Every word heard where it was spoken, 300 ms apart inside its turn.
fn heard(turns: &[Turn]) -> Vec<AsrTimingToken> {
    turns
        .iter()
        .flat_map(|turn| {
            turn.words
                .iter()
                .enumerate()
                .map(|(j, text)| AsrTimingToken {
                    text: text.clone(),
                    start_ms: turn.start_ms + j as u64 * 300,
                    end_ms: turn.start_ms + j as u64 * 300 + 250,
                })
        })
        .collect()
}

fn bullets(chat: &ChatFile) -> Vec<Option<(u64, u64)>> {
    chat.lines
        .iter()
        .filter_map(|line| match line {
            Line::Utterance(utterance) => Some(
                utterance
                    .main
                    .content
                    .bullet
                    .as_ref()
                    .map(|bullet| (bullet.timing.start_ms, bullet.timing.end_ms)),
            ),
            _ => None,
        })
        .collect()
}

/// The pre-region formulation: one joint search over the whole file.
fn whole_file_search_is_refused(turns: &[Turn], tokens: &[AsrTimingToken]) -> bool {
    let reference: Vec<String> = tokens.iter().map(|token| token.text.clone()).collect();
    let speaker_turns: Vec<_> = turns
        .iter()
        .map(|turn| SpeakerTurn::new(turn.speaker, &turn.words))
        .collect();
    LocalInterleaving::observe(&speaker_turns, &reference, MatchMode::CaseInsensitive).is_err()
}

#[test]
fn a_file_over_the_global_budget_recovers_fully_through_its_anchors() {
    // 1,200 alternating turns; every other one timed. One joint search over
    // the whole file exceeds its fixed budget, which used to refuse every
    // utterance. Each anchored region is a few turns.
    let turns = dialogue(1_200, |index| index % 2 == 0);
    let tokens = heard(&turns);
    assert!(
        whole_file_search_is_refused(&turns, &tokens),
        "precondition: the whole-file problem exceeds the budget"
    );
    let mut document = chat(&turns);
    let result = inject_utr_timing(&mut document, &tokens);
    assert_eq!(
        (result.injected(), result.skipped(), result.unmatched()),
        (600, 600, 0),
        "every untimed turn recovered, every timed one kept"
    );
    for (turn, bullet) in turns.iter().zip(bullets(&document)) {
        let (start, end) = bullet.expect("every turn timed");
        assert!(
            start >= turn.start_ms && end <= turn.end_ms,
            "recovered span lies inside where the turn was spoken"
        );
    }
    let UtrAlignmentEvidence::Global { plan } = result.alignment() else {
        panic!("global plan");
    };
    assert_eq!(plan.regions().len(), 600, "one region per anchor");
    assert!(!plan.utterances().iter().any(|evidence| matches!(
        evidence,
        UtrUtteranceAlignmentEvidence::Refused { .. }
            | UtrUtteranceAlignmentEvidence::RetainedUnsearched { .. }
    )));
}

#[test]
fn one_oversized_region_is_refused_alone_and_its_neighbours_recover() {
    // Timed anchors around small gaps at both ends, and one stretch of
    // 1,200 untimed turns between two anchors in the middle.
    const BIG: std::ops::Range<usize> = 3..1_203;
    let total = BIG.end + 3;
    let timed = |index: usize| index.is_multiple_of(2) && !BIG.contains(&index);
    let turns = dialogue(total, |index| timed(index) || index == BIG.end);
    let tokens = heard(&turns);
    let mut document = chat(&turns);
    let before = bullets(&document);
    let result = inject_utr_timing(&mut document, &tokens);
    let after = bullets(&document);

    let UtrAlignmentEvidence::Global { plan } = result.alignment() else {
        panic!("global plan");
    };
    // The oversized region owns its leading anchor (turn 2) and the untimed
    // stretch; its search exceeded a budget, which every refusal names.
    let refused_region = plan
        .regions()
        .iter()
        .map(UtrRegionSummary::span)
        .find(|span| span.first_utterance().index() == 2)
        .expect("region opened by turn 2");
    assert_eq!(refused_region.last_utterance().index(), BIG.end - 1);
    for (index, evidence) in plan.utterances().iter().enumerate() {
        match evidence {
            UtrUtteranceAlignmentEvidence::Refused {
                reason: UtrCorrespondenceRefusal::BudgetExhausted(refusal),
                ..
            } => {
                assert!(BIG.contains(&index), "only the oversized region is refused");
                assert_eq!(refusal.region(), refused_region);
                assert!(matches!(
                    refusal.budget(),
                    UtrSearchBudget::InterleavingWork { .. }
                        | UtrSearchBudget::InterleavingMemory { .. }
                        | UtrSearchBudget::InterleavingCandidates { .. }
                ));
            }
            UtrUtteranceAlignmentEvidence::RetainedUnsearched { unsearched, .. } => {
                assert_eq!(index, 2, "the refused region's own anchor keeps its timing");
                assert_eq!(unsearched.region(), refused_region);
            }
            other => assert!(
                !BIG.contains(&index),
                "turn {index} in the oversized region has {other:?}"
            ),
        }
    }
    // Every timed bullet is unchanged, the refused stretch stays untimed and
    // the small gaps on either side are recovered.
    for (index, (was, now)) in before.iter().zip(&after).enumerate() {
        match was {
            Some(original) => assert_eq!(now.as_ref(), Some(original), "turn {index} kept"),
            None if BIG.contains(&index) => assert_eq!(*now, None, "turn {index} refused"),
            None => assert!(now.is_some(), "neighbouring turn {index} recovered"),
        }
    }
    assert_eq!(
        result.unmatched(),
        BIG.len(),
        "exactly the refused untimed stretch"
    );
    let budget_decisions = result
        .decisions()
        .iter()
        .filter(|decision| {
            matches!(
                decision.strategy,
                batchalign_transform::decisions::DecisionStrategy::Utr(
                    batchalign_transform::decisions::UtrStrategy::CorrespondenceBudgetExhausted
                )
            ) && decision.reason.contains("region=[utterances=2..=")
        })
        .count();
    assert_eq!(
        budget_decisions,
        BIG.len(),
        "each refusal is a visible decision"
    );
}

#[test]
fn interior_only_evidence_bounds_search_between_proved_neighbours_without_a_hint() {
    // Single speaker, so the monotonic proof runs. The middle utterance's
    // first word was never heard: no endpoint proof, so no hint, but its
    // neighbours' proved words bound where all of its words can be.
    let turns = vec![
        Turn {
            speaker: "PAR",
            words: vec!["alpha".into(), "beta".into()],
            start_ms: 100,
            end_ms: 300,
            timed: false,
        },
        Turn {
            speaker: "PAR",
            words: vec!["unheard".into(), "gamma".into(), "delta".into()],
            start_ms: 300,
            end_ms: 600,
            timed: false,
        },
        Turn {
            speaker: "PAR",
            words: vec!["epsilon".into(), "zeta".into()],
            start_ms: 700,
            end_ms: 900,
            timed: false,
        },
    ];
    let tokens: Vec<AsrTimingToken> = [
        ("alpha", 100),
        ("beta", 200),
        ("gamma", 400),
        ("delta", 500),
        ("epsilon", 700),
        ("zeta", 800),
    ]
    .into_iter()
    .map(|(text, start_ms)| AsrTimingToken {
        text: text.into(),
        start_ms,
        end_ms: start_ms + 80,
    })
    .collect();
    let mut document = chat(&turns);
    let result = inject_utr_timing(&mut document, &tokens);
    assert_eq!(bullets(&document)[1], None, "interior proof is not a hint");
    let UtrAlignmentEvidence::Global { plan } = result.alignment() else {
        panic!("global plan");
    };
    assert!(matches!(
        plan.utterances()[1],
        UtrUtteranceAlignmentEvidence::InteriorOnly { .. }
    ));
    insta::assert_json_snapshot!(plan.search_envelopes[1], @r#"
    {
      "words": [
        "unheard",
        "gamma",
        "delta"
      ],
      "floor_ms": 200,
      "ceiling_ms": 780,
      "scope": {
        "kind": "order_corridor"
      }
    }
    "#);
    assert!(result.decisions().iter().any(|decision| matches!(
        decision.strategy,
        batchalign_transform::decisions::DecisionStrategy::Utr(
            batchalign_transform::decisions::UtrStrategy::IncompleteBoundary
        )
    )
        && decision.reason.contains("bounded_search_window")));
}

/// Review 10 (cross-region claims). Neighbouring regions share the anchor
/// between them, so a token inside the anchor's span is in both windows. Here
/// an untimed turn before the anchor (one region) and an untimed turn of a
/// third speaker after it (the next region) both say the one word heard
/// inside the anchor's span, and each region alone proves its claim. The file
/// cannot have both, and the decomposition cannot say which: neither claim is
/// admitted, neither turn is hinted, and both regions record the withdrawal.
#[test]
fn a_token_claimed_across_an_anchor_by_two_regions_is_withdrawn_from_both() {
    let text = "@UTF8\n@Begin\n@Languages:\teng\n\
        @Participants:\tPAR Participant, INV Investigator, CHI Target_Child\n\
        @ID:\teng|test|PAR|||||Participant|||\n@ID:\teng|test|INV|||||Investigator|||\n\
        @ID:\teng|test|CHI|||||Target_Child|||\n@Media:\tinput, audio\n\
        *PAR:\tkaone katwo kathree . \u{15}0_900\u{15}\n\
        *INV:\tyes .\n\
        *PAR:\tkafour kafive kasix . \u{15}2000_2900\u{15}\n\
        *CHI:\tyes .\n\
        *PAR:\tkaseven kaeight kanine . \u{15}4000_4900\u{15}\n@End\n";
    let parser = TreeSitterParser::new().expect("parser");
    let mut chat = parse_source_with_parser(&parser, text)
        .admit(
            TranscriptName::Named(
                FileStem::from_path(std::path::Path::new("input.cha")).expect("name"),
            ),
            &NullErrorSink,
        )
        .expect("synthetic CHAT admission")
        .into_valid_file()
        .into_unchecked();
    let token = |text: &str, start_ms: u64| AsrTimingToken {
        text: text.to_owned(),
        start_ms,
        end_ms: start_ms + 200,
    };
    let tokens = vec![
        token("kaone", 0),
        token("katwo", 300),
        token("kathree", 600),
        token("kafour", 2000),
        token("kafive", 2250),
        // The one "yes", inside the middle anchor's span: token 5.
        token("yes", 2500),
        token("kasix", 2700),
        token("kaseven", 4000),
        token("kaeight", 4300),
        token("kanine", 4600),
    ];
    let result = inject_utr_timing(&mut chat, &tokens);
    let UtrAlignmentEvidence::Global { plan } = &result.alignment else {
        panic!("global plan");
    };
    assert_eq!(plan.order_model(), UtrOrderModel::Interleaved);
    let withdrawn: Vec<Vec<usize>> = plan
        .regions()
        .iter()
        .map(|region| {
            region
                .withdrawn_claims
                .iter()
                .map(|token| token.index())
                .collect()
        })
        .collect();
    assert_eq!(withdrawn, vec![vec![5], vec![5], vec![]]);
    let hints = bullets(&chat);
    assert_eq!(hints[1], None, "the turn before the anchor is not hinted");
    assert_eq!(hints[3], None, "the turn after the anchor is not hinted");
    assert_eq!(hints[2], Some((2000, 2900)), "the anchor keeps its bullet");
}

/// The same contest where the contested token is the ENDPOINT of a turn
/// that has other proved words. Withdrawing only the common match is not
/// enough: the endpoint must not be bounded by the contested token either,
/// or the hint's hull would still include it. The turn is interior-only, so
/// it gets no hint.
#[test]
fn a_withdrawn_endpoint_token_does_not_bound_its_turn() {
    let text = "@UTF8\n@Begin\n@Languages:\teng\n\
        @Participants:\tPAR Participant, INV Investigator, CHI Target_Child\n\
        @ID:\teng|test|PAR|||||Participant|||\n@ID:\teng|test|INV|||||Investigator|||\n\
        @ID:\teng|test|CHI|||||Target_Child|||\n@Media:\tinput, audio\n\
        *PAR:\tkaone katwo kathree . \u{15}0_900\u{15}\n\
        *INV:\tkaoh yes .\n\
        *PAR:\tkafour kafive kasix . \u{15}2000_2900\u{15}\n\
        *CHI:\tyes .\n\
        *PAR:\tkaseven kaeight kanine . \u{15}4000_4900\u{15}\n@End\n";
    let parser = TreeSitterParser::new().expect("parser");
    let mut chat = parse_source_with_parser(&parser, text)
        .admit(
            TranscriptName::Named(
                FileStem::from_path(std::path::Path::new("input.cha")).expect("name"),
            ),
            &NullErrorSink,
        )
        .expect("synthetic CHAT admission")
        .into_valid_file()
        .into_unchecked();
    let token = |text: &str, start_ms: u64| AsrTimingToken {
        text: text.to_owned(),
        start_ms,
        end_ms: start_ms + 200,
    };
    let tokens = vec![
        token("kaone", 0),
        token("katwo", 300),
        token("kathree", 600),
        token("kaoh", 1200),
        token("kafour", 2000),
        token("kafive", 2250),
        token("yes", 2500),
        token("kasix", 2700),
        token("kaseven", 4000),
        token("kaeight", 4300),
        token("kanine", 4600),
    ];
    let result = inject_utr_timing(&mut chat, &tokens);
    let UtrAlignmentEvidence::Global { plan } = &result.alignment else {
        panic!("global plan");
    };
    let withdrawn: Vec<Vec<usize>> = plan
        .regions()
        .iter()
        .map(|region| {
            region
                .withdrawn_claims
                .iter()
                .map(|token| token.index())
                .collect()
        })
        .collect();
    assert_eq!(
        withdrawn,
        vec![vec![6], vec![6], vec![]],
        "both regions claimed `yes`"
    );
    assert!(
        matches!(
            plan.utterances[1],
            UtrUtteranceAlignmentEvidence::InteriorOnly {
                missing_endpoints: MissingUtrEndpoints::Last,
                ..
            }
        ),
        "{:?}",
        plan.utterances[1]
    );
    let hints = bullets(&chat);
    assert_eq!(
        hints[1], None,
        "no hint whose hull reaches the withdrawn token"
    );
    assert_eq!(hints[3], None);
}
