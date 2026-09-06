//! The write phase's `TimedUtterance` gate (2026-09-01 review, item 9): an
//! utterance in `WorPlan::Pending` must still carry at least one timed word
//! AT WRITE TIME, or no `%wor` tier is written for it at all. Reproduces the
//! real-data regression (Batch 1, `3009.cha`): utterances whose words never
//! got timed, or lost every timing before the write phase ran, were getting
//! an entirely-untimed `%wor` tier where the pre-fix pipeline wrote none.

#![allow(unused_imports, dead_code)]

use super::*;

/// Pass 1 (start-regression) strips an utterance outright. Its `%wor` must
/// not be written even though it was `touched` by fresh injection.
#[test]
fn pass1_stripped_utterance_gets_no_wor_tier() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|x|CHI|||||Child|||\n*CHI:\thello . \u{15}5000_6000\u{15}\n*CHI:\tworld . \u{15}2000_3000\u{15}\n@End\n";
    let mut chat = parse_chat(input);

    let groups = vec![FaGroup::test_fixture(
        TimeSpan::new(0, 10_000),
        vec![
            FaWord {
                utterance_index: UtteranceIdx::new(0),
                utterance_word_index: WordIdx::new(0),
                text: "hello".into(),
            },
            FaWord {
                utterance_index: UtteranceIdx::new(1),
                utterance_word_index: WordIdx::new(0),
                text: "world".into(),
            },
        ],
        vec![UtteranceIdx::new(0), UtteranceIdx::new(1)],
    )];
    let responses = vec![vec![
        WordTiming::fixture(5000, 5500),
        WordTiming::fixture(2000, 2500),
    ]];

    let _finalized = apply_fa_results(
        &mut chat,
        &groups,
        &responses,
        WordEndPolicy::measured(WordGapHealing::PreserveMeasured),
        true,
    )
    .then_finalize(&mut chat, BulletRepairPolicy::Disabled);

    let utt1 = get_test_utterance(&mut chat, 1);
    assert!(
        utt1.main.content.bullet.is_none(),
        "utterance 1 should have been stripped for a start regression"
    );
    assert!(
        utt1.wor_tier().is_none(),
        "a Pass-1-stripped utterance must get no %wor tier"
    );
}

/// The exact `3009.cha` line-58 shape: an utterance already carries an
/// authoritative bullet (inherited from an earlier pass), is `touched` by
/// this run (its word is part of a dispatched FA group), but the aligner
/// returns no timing for that word at all. `update_utterance_bullet_with_boundary_policy`'s
/// "no word timings: leave the existing bullet unchanged" branch (`mod.rs`)
/// is exactly why the bullet survives with zero timed words: there is
/// nothing to derive a new one from, so the inherited value is left in
/// place. Before this fix, that combination (bullet present, zero timed
/// words) still got a `%wor` tier written from the write phase; it must not.
#[test]
fn untimed_word_with_inherited_bullet_gets_no_wor_tier() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n@ID:\teng|x|CHI|||||Child|||\n*CHI:\thello . \u{15}29500_30980\u{15}\n@End\n";
    let mut chat = parse_chat(input);

    let groups = vec![FaGroup::test_fixture(
        TimeSpan::new(0, 40_000),
        vec![FaWord {
            utterance_index: UtteranceIdx::new(0),
            utterance_word_index: WordIdx::new(0),
            text: "hello".into(),
        }],
        vec![UtteranceIdx::new(0)],
    )];
    // The aligner returned nothing for the one word in this group.
    let responses = vec![vec![None]];

    let _finalized = apply_fa_results(
        &mut chat,
        &groups,
        &responses,
        WordEndPolicy::measured(WordGapHealing::PreserveMeasured),
        true,
    )
    .then_finalize(&mut chat, BulletRepairPolicy::Disabled);

    let utt0 = get_test_utterance(&mut chat, 0);
    assert_eq!(
        utt0.main
            .content
            .bullet
            .as_ref()
            .map(|b| (b.timing.start_ms, b.timing.end_ms)),
        Some((29500, 30980)),
        "the inherited bullet must survive unchanged: nothing to derive a new one from"
    );
    assert!(
        utt0.wor_tier().is_none(),
        "a bullet with zero timed words must get no %wor tier"
    );
}
