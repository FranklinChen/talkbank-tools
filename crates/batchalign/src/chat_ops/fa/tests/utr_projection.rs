//! Checked UTR projection over fully admitted CHAT inputs.
//! Provider intervals are synthetic evidence, not acoustic accuracy gold.

use super::*;
use batchalign_transform::parse_source_with_parser;
use talkbank_model::model::TranscriptName;
use utr::{GlobalUtr, TwoPassOverlapUtr, UtrStrategy};

mod timing_hull;

fn selected_plan(result: &utr::UtrResult) -> &utr::UtrAlignmentPlan {
    match result.alignment() {
        utr::UtrAlignmentEvidence::Global { plan } => plan,
        utr::UtrAlignmentEvidence::TwoPass { first_pass, .. } => first_pass,
        utr::UtrAlignmentEvidence::NotRunNoUntimed => panic!("expected a selected pass"),
    }
}

enum FixtureTiming {
    Untimed,
    PartiallyTimed,
}

fn admitted_chat(turns: &str, timing: FixtureTiming) -> talkbank_model::model::ChatFile {
    let media = match timing {
        FixtureTiming::Untimed => "synthetic, audio, unlinked",
        FixtureTiming::PartiallyTimed => "synthetic, audio",
    };
    let source = format!(
        "@UTF8\n@Begin\n@Languages:\teng\n\
         @Participants:\tPAR Participant, INV Investigator\n\
         @ID:\teng|test|PAR|||||Participant|||\n\
         @ID:\teng|test|INV|||||Investigator|||\n\
         @Media:\t{media}\n{turns}@End\n"
    );
    let parser = TreeSitterParser::new().expect("parser");
    let errors = talkbank_model::ErrorCollector::new();
    parse_source_with_parser(&parser, &source)
        .admit(TranscriptName::Anonymous, &errors)
        .unwrap_or_else(|error| panic!("complete CHAT admission before UTR: {error}"))
        .into_valid_file()
        .into_unchecked()
}

#[test]
fn exhausted_global_and_two_pass_projection_leave_no_manufactured_hint() {
    for end_ms in [1_400, 1_500] {
        for strategy in [&GlobalUtr as &dyn UtrStrategy, &TwoPassOverlapUtr::new()] {
            let mut chat = admitted_chat(
                "*PAR:\thello . \u{15}0_1500\u{15}\n*PAR:\tworld .\n",
                FixtureTiming::PartiallyTimed,
            );
            let tokens = make_utr_tokens(&[("hello", 0, 1_500), ("world", 1_000, end_ms)]);
            let result = strategy.inject(&mut chat, &tokens);
            assert_eq!(get_utterance_bullet(&chat, 0), Some((0, 1_500)));
            assert_eq!(get_utterance_bullet(&chat, 1), None);
            assert_eq!(
                (result.injected(), result.skipped(), result.unmatched()),
                (0, 1, 1)
            );
            assert!(result.decisions().iter().any(|decision| {
                decision.strategy
                    == batchalign_transform::decisions::DecisionStrategy::Utr(
                        batchalign_transform::decisions::UtrStrategy::ProjectionExhausted,
                    )
                    && decision.needs_review
            }));
        }
    }
}

#[test]
fn surviving_projection_preserves_the_observed_end() {
    let mut chat = admitted_chat(
        "*PAR:\thello . \u{15}0_1500\u{15}\n*PAR:\tworld .\n",
        FixtureTiming::PartiallyTimed,
    );
    let tokens = make_utr_tokens(&[("hello", 0, 1_500), ("world", 1_000, 2_000)]);
    let result = GlobalUtr.inject(&mut chat, &tokens);
    assert_eq!(get_utterance_bullet(&chat, 0), Some((0, 1_500)));
    assert_eq!(get_utterance_bullet(&chat, 1), Some((1_500, 2_000)));
    assert_eq!(
        (result.injected(), result.skipped(), result.unmatched()),
        (1, 1, 0)
    );
}

#[test]
fn exhausted_new_hint_does_not_move_the_floor_for_later_evidence() {
    let mut chat = admitted_chat(
        "*PAR:\thello .\n*PAR:\tworld .\n*PAR:\tagain .\n",
        FixtureTiming::Untimed,
    );
    let tokens = make_utr_tokens(&[
        ("hello", 0, 1_500),
        ("world", 1_000, 1_500),
        ("again", 1_500, 2_000),
    ]);
    let result = GlobalUtr.inject(&mut chat, &tokens);
    assert_eq!(get_utterance_bullet(&chat, 0), Some((0, 1_500)));
    assert_eq!(get_utterance_bullet(&chat, 1), None);
    assert_eq!(get_utterance_bullet(&chat, 2), Some((1_500, 2_000)));
    insta::assert_json_snapshot!(serde_json::json!({
        "injected": result.injected(), "skipped": result.skipped(),
        "unmatched": result.unmatched(),
        "decision": result.decisions()[0].strategy.strategy_name(),
    }), @r###"
    {
      "decision": "projection_exhausted",
      "injected": 2,
      "skipped": 0,
      "unmatched": 1
    }
    "###);
}

#[test]
fn marked_overlap_keeps_its_interval_and_does_not_advance_the_floor() {
    let mut chat = admitted_chat(
        "*PAR:\thello . \u{15}0_1500\u{15}\n*INV:\t+< world .\n*PAR:\tagain .\n",
        FixtureTiming::PartiallyTimed,
    );
    let tokens = make_utr_tokens(&[
        ("hello", 0, 1_500),
        ("world", 1_000, 3_000),
        ("again", 1_500, 2_000),
    ]);
    let result = GlobalUtr.inject(&mut chat, &tokens);
    assert_eq!(get_utterance_bullet(&chat, 1), Some((1_000, 3_000)));
    assert_eq!(get_utterance_bullet(&chat, 2), Some((1_500, 2_000)));
    assert_eq!(
        (result.injected(), result.skipped(), result.unmatched()),
        (2, 1, 0)
    );
}

#[test]
fn ca_bottom_overlap_remains_exempt_from_non_overlap_clipping() {
    let mut chat = admitted_chat(
        "*PAR:\t⌈hello⌉ . \u{15}0_1500\u{15}\n*INV:\t⌊world⌋ .\n",
        FixtureTiming::PartiallyTimed,
    );
    let tokens = make_utr_tokens(&[("hello", 0, 1_500), ("world", 500, 900)]);
    let result = GlobalUtr.inject(&mut chat, &tokens);
    assert_eq!(get_utterance_bullet(&chat, 0), Some((0, 1_500)));
    assert_eq!(get_utterance_bullet(&chat, 1), Some((500, 900)));
    assert_eq!(
        (result.injected(), result.skipped(), result.unmatched()),
        (1, 1, 0)
    );
}

#[test]
fn two_pass_cannot_write_a_nonpositive_local_overlap_interval() {
    for end_ms in [500, 400] {
        let mut chat = admitted_chat(
            "*PAR:\thello . \u{15}0_1500\u{15}\n*INV:\t+< world .\n",
            FixtureTiming::PartiallyTimed,
        );
        let tokens = make_utr_tokens(&[("hello", 0, 1_500), ("world", 500, end_ms)]);
        let result = TwoPassOverlapUtr::new().inject(&mut chat, &tokens);
        assert_eq!(get_utterance_bullet(&chat, 0), Some((0, 1_500)));
        assert_eq!(get_utterance_bullet(&chat, 1), None);
        assert_eq!(
            (result.injected(), result.skipped(), result.unmatched()),
            (0, 1, 1)
        );
    }
}

#[test]
fn repeated_word_conflict_is_refused_without_hiding_the_selected_plan() {
    for strategy in [&GlobalUtr as &dyn UtrStrategy, &TwoPassOverlapUtr::new()] {
        let mut chat = admitted_chat(
            "*PAR:\thello .\n*PAR:\thello . \u{15}200_300\u{15}\n",
            FixtureTiming::PartiallyTimed,
        );
        let tokens = make_utr_tokens(&[("hello", 200, 300)]);
        let result = strategy.inject(&mut chat, &tokens);
        assert_eq!(get_utterance_bullet(&chat, 0), None);
        assert_eq!(get_utterance_bullet(&chat, 1), Some((200, 300)));
        assert_eq!(
            (result.injected(), result.skipped(), result.unmatched()),
            (0, 1, 1)
        );
        assert!(result.decisions().iter().any(|decision| {
            decision.strategy.strategy_name() == "ambiguous_correspondence"
                && decision.needs_review
                && decision.reason.contains("no_admitted_correspondence")
        }));
        // Selection is retained, but neither optimal assignment is forced.
        let plan = serde_json::to_value(selected_plan(&result)).expect("wire evidence");
        assert_eq!(plan["utterances"][0]["status"], "selected_only");
        assert_eq!(plan["utterances"][0]["reason"], "ambiguous");
        assert!(plan["utterances"][0].get("proposal").is_none());
        assert_eq!(
            plan["utterances"][0]["matches"]["first"]["token"]["token_index"],
            0
        );
        // The token lies inside the timed utterance's own bullet, so its
        // anchored region attributes it there; the untimed utterance before
        // it still sees the token as ambiguous and gets no hint.
        assert_eq!(plan["utterances"][1]["status"], "matched");
    }
}

#[test]
fn retained_neighbors_bound_hints_without_extending_provider_evidence() {
    for (start, end, expected) in [
        (200, 300, Some((200, 300))), // A legitimate gap survives unchanged.
        (50, 550, Some((100, 500))),  // Both bounds narrow, never extend.
        (500, 700, None),             // Touching the ceiling is exhausted.
        (20, 100, None),              // Touching the floor is exhausted.
    ] {
        for strategy in [&GlobalUtr as &dyn UtrStrategy, &TwoPassOverlapUtr::new()] {
            let mut chat = admitted_chat(
                "*PAR:\thello . \u{15}0_100\u{15}\n*PAR:\tworld .\n*PAR:\tagain . \u{15}500_600\u{15}\n",
                FixtureTiming::PartiallyTimed,
            );
            let tokens = make_utr_tokens(&[
                ("hello", 0, 100),
                ("world", start, end),
                ("again", 500, 600),
            ]);
            let result = strategy.inject(&mut chat, &tokens);
            assert_eq!(get_utterance_bullet(&chat, 0), Some((0, 100)));
            assert_eq!(get_utterance_bullet(&chat, 1), expected);
            assert_eq!(get_utterance_bullet(&chat, 2), Some((500, 600)));
            assert_eq!(
                (result.injected(), result.skipped(), result.unmatched()),
                (
                    usize::from(expected.is_some()),
                    2,
                    usize::from(expected.is_none())
                ),
            );
            let plan = serde_json::to_value(selected_plan(&result)).expect("wire evidence");
            assert_eq!(
                plan["utterances"][1]["proposal"],
                serde_json::json!({
                    "status": "positive", "start_ms": start, "end_ms": end,
                })
            );
        }
    }
}

#[test]
fn marked_overlap_hints_remain_exempt_from_the_following_retained_bound() {
    let mut chat = admitted_chat(
        "*PAR:\thello . \u{15}0_100\u{15}\n*INV:\t+< world .\n*PAR:\tagain . \u{15}500_600\u{15}\n",
        FixtureTiming::PartiallyTimed,
    );
    let tokens = make_utr_tokens(&[("hello", 0, 100), ("world", 50, 550), ("again", 500, 600)]);
    let result = GlobalUtr.inject(&mut chat, &tokens);
    assert_eq!(get_utterance_bullet(&chat, 1), Some((50, 550)));
    assert_eq!(
        (result.injected(), result.skipped(), result.unmatched()),
        (1, 2, 0)
    );
}

#[test]
fn retained_marked_overlap_does_not_supply_a_non_overlap_ceiling() {
    let mut chat = admitted_chat(
        "*PAR:\thello .\n*INV:\t+< world . \u{15}100_200\u{15}\n",
        FixtureTiming::PartiallyTimed,
    );
    let tokens = make_utr_tokens(&[("hello", 0, 300), ("world", 100, 200)]);
    let result = GlobalUtr.inject(&mut chat, &tokens);
    assert_eq!(get_utterance_bullet(&chat, 0), Some((0, 300)));
    assert_eq!(get_utterance_bullet(&chat, 1), Some((100, 200)));
    assert_eq!(
        (result.injected(), result.skipped(), result.unmatched()),
        (1, 1, 0)
    );
}
