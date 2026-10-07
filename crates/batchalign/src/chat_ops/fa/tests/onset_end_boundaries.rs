//! Final word ends retain their evidence through injection and finalization.

use super::*;
use crate::chat_ops::fa::coordinates::Ms;
use crate::chat_ops::fa::origin::{EngineId, Origin};
use talkbank_model::model::Bullet;

#[test]
fn local_overlap_hint_remains_provisional_through_final_word_injection() {
    use crate::chat_ops::fa::utr::{TwoPassConfig, TwoPassOverlapUtr, UtrStrategy};
    use talkbank_model::model::{BulletSource, FileStem, TranscriptName};
    let source = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant, INV Investigator\n@ID:\teng|test|PAR|||||Participant|||\n@ID:\teng|test|INV|||||Investigator|||\n@Media:\tinput, audio, unlinked\n*PAR:\tnow middle .\n*INV:\t+< hello .\n*PAR:\ttail .\n@End\n";
    let parser = talkbank_parser::TreeSitterParser::new().expect("parser");
    let mut chat = batchalign_transform::parse_source_with_parser(&parser, source)
        .admit(
            TranscriptName::Named(
                FileStem::from_path(std::path::Path::new("input.cha")).expect("name"),
            ),
            &talkbank_model::NullErrorSink,
        )
        .expect("complete source admission")
        .into_valid_file()
        .into_unchecked();
    let tokens = make_utr_tokens(&[
        ("now", 100, 1000),
        ("middle", 1000, 3000),
        ("hello", 2260, 7000),
        ("tail", 7500, 8000),
    ]);
    let recovered = TwoPassOverlapUtr::new()
        .with_config(TwoPassConfig {
            max_exclusion_density: "1.0".parse().expect("checked density"),
            ..TwoPassConfig::default()
        })
        .inject(&mut chat, &tokens);
    let utr::UtrAlignmentEvidence::TwoPass {
        overlap_recoveries, ..
    } = recovered.alignment()
    else {
        panic!("two-pass proof");
    };
    assert_eq!(
        overlap_recoveries.len(),
        1,
        "exercise the actual local writer"
    );
    assert_eq!(
        get_utterance(&chat, 1)
            .main
            .content
            .bullet
            .as_ref()
            .expect("recovered hint")
            .source,
        BulletSource::Utr
    );

    let groups = vec![FaGroup::test_fixture(
        TimeSpan::new(0, 8500),
        vec![FaWord {
            utterance_index: UtteranceIdx::new(1),
            utterance_word_index: WordIdx::new(0),
            text: "hello".into(),
        }],
        vec![UtteranceIdx::new(1)],
    )];
    let timing = WordTiming::new(
        2260,
        2760,
        Origin::EngineMeasured {
            engine: EngineId::new("whisper_fa"),
        },
        Origin::FallbackDuration { assumed: Ms(500) },
    )
    .expect("positive measurement");
    let finalized = apply_fa_results(
        &mut chat,
        &groups,
        &[vec![Some(timing)]],
        WordEndPolicy::onset_only(WordGapHealing::Heal),
        true,
    )
    .then_finalize(&mut chat, BulletRepairPolicy::Disabled)
    .expect("finalization proof");
    let (records, _) = retain_decision_evidence(
        &mut chat,
        FaDecisions {
            rescue: Vec::new(),
            grouping: Vec::new(),
            finalized,
        },
    )
    .into_evidence();
    let utterance = get_utterance(&chat, 1);
    let word = utterance
        .wor_tier()
        .expect("written wor")
        .words()
        .next()
        .expect("aligned word");
    assert_eq!(
        word.inline_bullet
            .as_ref()
            .expect("word timing")
            .timing
            .end_ms,
        2760,
        "local UTR end must not replace the onset-only fallback with 7000"
    );
    assert_eq!(
        utterance
            .main
            .content
            .bullet
            .as_ref()
            .expect("final main bullet")
            .timing
            .end_ms,
        2760
    );
    assert!(
        records.iter().any(|record| record.needs_review
            && record.strategy
                == batchalign_transform::decisions::DecisionStrategy::Fa(
                    batchalign_transform::decisions::FaStrategy::TimingProvenance
                )),
        "onset-derived duration remains reported for review"
    );
}

#[test]
fn onset_end_boundary_controls_preserve_fallback_and_review_evidence() {
    struct Control {
        name: &'static str,
        bullet: Option<Bullet>,
        policy: WordEndPolicy,
        origin: Origin,
        expected_end: u64,
        expected_review: bool,
    }
    let fallback = || Origin::FallbackDuration { assumed: Ms(500) };
    let onset = WordEndPolicy::onset_only(WordGapHealing::Heal);
    let controls = [
        Control {
            name: "no transcript boundary",
            bullet: None,
            policy: onset,
            origin: fallback(),
            expected_end: 2760,
            expected_review: true,
        },
        Control {
            name: "long provisional UTR hint",
            bullet: Some(Bullet::utr_hint(0, 7000)),
            policy: onset,
            origin: fallback(),
            expected_end: 2760,
            expected_review: true,
        },
        Control {
            name: "short provisional UTR hint",
            bullet: Some(Bullet::utr_hint(0, 2300)),
            policy: onset,
            origin: fallback(),
            expected_end: 2760,
            expected_review: true,
        },
        Control {
            name: "retained transcript boundary",
            bullet: Some(Bullet::new(0, 7000)),
            policy: onset,
            origin: fallback(),
            expected_end: 7000,
            expected_review: false,
        },
        Control {
            name: "measured end",
            bullet: Some(Bullet::new(0, 7000)),
            policy: WordEndPolicy::measured(WordGapHealing::Heal),
            origin: Origin::EngineMeasured {
                engine: EngineId::new("whisper_fa"),
            },
            expected_end: 2760,
            expected_review: false,
        },
        Control {
            name: "preserved onset fallback",
            bullet: Some(Bullet::new(0, 7000)),
            policy: WordEndPolicy::onset_only(WordGapHealing::PreserveMeasured),
            origin: fallback(),
            expected_end: 2760,
            expected_review: true,
        },
    ];
    for control in controls {
        let mut chat = parse_chat(&proof_chat("hello ."));
        get_test_utterance(&mut chat, 0).main.content.bullet = control.bullet;
        let groups = vec![FaGroup::test_fixture(
            TimeSpan::new(0, 8500),
            vec![FaWord {
                utterance_index: UtteranceIdx::new(0),
                utterance_word_index: WordIdx::new(0),
                text: "hello".into(),
            }],
            vec![UtteranceIdx::new(0)],
        )];
        let timing = WordTiming::new(
            2260,
            2760,
            Origin::EngineMeasured {
                engine: EngineId::new("whisper_fa"),
            },
            control.origin,
        )
        .expect("positive producer interval");
        let finalized = apply_fa_results(
            &mut chat,
            &groups,
            &[vec![Some(timing)]],
            control.policy,
            true,
        )
        .then_finalize(&mut chat, BulletRepairPolicy::Disabled)
        .expect("consume injection and finalization proof");
        let (records, _) = retain_decision_evidence(
            &mut chat,
            FaDecisions {
                rescue: Vec::new(),
                grouping: Vec::new(),
                finalized,
            },
        )
        .into_evidence();
        let word = get_utterance(&chat, 0)
            .wor_tier()
            .expect("written wor")
            .words()
            .next()
            .expect("aligned word");
        assert_eq!(
            word.inline_bullet
                .as_ref()
                .expect("word timing")
                .timing
                .end_ms,
            control.expected_end,
            "{}",
            control.name
        );
        let provenance = records.iter().find(|record| {
            record.strategy
                == batchalign_transform::decisions::DecisionStrategy::Fa(
                    batchalign_transform::decisions::FaStrategy::TimingProvenance,
                )
        });
        assert_eq!(
            provenance.is_some_and(|record| record.needs_review),
            control.expected_review,
            "{}",
            control.name
        );
        if control.expected_review {
            assert!(
                provenance
                    .expect("assumed timing record")
                    .reason
                    .contains("assumed=1"),
                "{}",
                control.name
            );
        }
    }
}
