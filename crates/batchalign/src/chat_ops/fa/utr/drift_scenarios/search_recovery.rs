//! Request-window controls, not acoustic-model accuracy or output permission.
//!
//! Keep the hint-only recovery assertions unchanged. This separate census asks
//! whether the final FA request crops still contain every fixture word, including
//! words whose correspondence is ambiguous or absent from the ASR observation.

use super::fixture::{DriftParams, OverlapConvention, build_scenario};
use crate::chat_ops::fa::coordinates::{Ms, Recording};
use crate::chat_ops::fa::group_utterances;
use crate::chat_ops::fa::grouping::{GroupUnit, GroupUnits};
use crate::chat_ops::fa::utr::{collect_utr_utterance_info, inject_utr_timing};

#[test]
#[ignore = "manual long request-crop recovery assessment; retained red until repair"]
fn joint_recovery_request_windows_retain_the_complete_source_population() {
    let controls = [
        ("short", 30, OverlapConvention::CaBracket, 0.2, 0.0),
        ("ca", 500, OverlapConvention::CaBracket, 0.60, 0.25),
        ("lazy", 500, OverlapConvention::LazyPrecedes, 0.45, 0.15),
        (
            "backchannels",
            500,
            OverlapConvention::InlineBackchannel,
            0.45,
            0.10,
        ),
        ("mixed", 500, OverlapConvention::Mixed, 0.60, 0.15),
    ];
    let mut violations = Vec::new();
    for (name, n_utts, convention, overlap_density, asr_missing_rate) in controls {
        let scenario = build_scenario(DriftParams {
            n_utts,
            convention,
            overlap_density,
            asr_missing_rate,
            stopword_only: false,
        });
        let mut chat = scenario.source.into_valid_file().into_unchecked();
        let census = collect_utr_utterance_info(&chat);
        assert_eq!(census.len(), scenario.expected.len(), "fixture population");
        let end = scenario
            .expected
            .iter()
            .map(|window| window.expected_end_ms)
            .max()
            .unwrap();
        let recording = Recording::of_duration(Ms(end)).expect("nonempty fixture recording");
        let mut observed = inject_utr_timing(&mut chat, &scenario.tokens);
        let anchors = observed.take_anchors();
        let plan = match &observed.alignment {
            super::super::UtrAlignmentEvidence::Global { plan } => Some(plan),
            _ => None,
        };
        println!("{name}: strategy={:?}", plan.map(|p| p.strategy));
        let grouping = group_utterances(&chat, 15_000, &recording, &anchors);
        let mut requested: Vec<Vec<usize>> = census
            .iter()
            .map(|utterance| vec![0; utterance.words.len()])
            .collect();
        let mut contained = 0usize;
        let mut outside = 0usize;
        let mut inspect = |unit: GroupUnit<'_>| {
            for word in unit.words {
                let utterance = word.utterance_index.raw();
                let index = word.utterance_word_index.raw();
                assert_eq!(
                    word.text, census[utterance].words[index],
                    "source-bound word census"
                );
                requested[utterance][index] += 1;
                let expected = scenario.expected[utterance];
                // The fixture producer spaces words evenly, retaining the last
                // word's exact end. Missing ASR words do not change this oracle.
                let step = (expected.expected_end_ms - expected.expected_start_ms)
                    / census[utterance].words.len() as u64;
                let start = expected.expected_start_ms + index as u64 * step;
                let end = if index + 1 == census[utterance].words.len() {
                    expected.expected_end_ms
                } else {
                    start + step
                };
                if unit.window.audio_start().get() <= start && unit.window.end().get() >= end {
                    contained += 1;
                } else {
                    outside += 1;
                    if outside <= 3 {
                        println!(
                            "{name} outside U{utterance}: envelope={:?}",
                            plan.and_then(|p| p.search_envelopes[utterance].as_ref())
                        );
                    }
                    violations.push(format!(
                        "{name} U{utterance} W{index}: [{start},{end}] outside request {:?}",
                        unit.window
                    ));
                }
            }
        };
        for group in &grouping.groups {
            match group.units() {
                GroupUnits::Whole(unit) => inspect(unit),
                GroupUnits::Pieces(pieces) => {
                    for piece in pieces {
                        inspect(GroupUnit {
                            words: piece.words(),
                            window: piece.window(),
                        });
                    }
                }
            }
        }
        let missing = requested
            .iter()
            .flatten()
            .filter(|&&count| count == 0)
            .count();
        let repeated = requested
            .iter()
            .flatten()
            .filter(|&&count| count > 1)
            .count();
        println!(
            "{name}: hints={} required_words={} contained={contained} outside={outside} unrequested={missing} repeated={repeated} groups={}",
            observed.injected(),
            requested.iter().map(Vec::len).sum::<usize>(),
            grouping.groups.len()
        );
        if missing != 0 || repeated != 0 {
            for (utterance, words) in requested
                .iter()
                .enumerate()
                .filter(|(_, words)| words.contains(&0))
                .take(5)
            {
                println!(
                    "{name} unrequested U{utterance}: count={} envelope={:?} anchors={:?}",
                    words.iter().filter(|&&n| n == 0).count(),
                    plan.and_then(|p| p.search_envelopes[utterance].as_ref()),
                    anchors.lookup(talkbank_model::UtteranceIdx::new(utterance))
                );
            }
            violations.push(format!(
                "{name}: {missing} missing requests, {repeated} duplicate requests"
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "FA recovery window failures ({}):\n{}",
        violations.len(),
        violations
            .iter()
            .take(30)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
