//! `--main-bullets keep`, ordering phases: monotonicity and bullet repair
//! never change a kept bullet. A revisable bullet next to a kept one is
//! resolved by moving only the revisable side; only a conflict between two
//! kept bullets is left in place and recorded. Every keep run goes through
//! [`run_under_both_policies`], which checks that EVERY given bullet held.

use super::*;

use batchalign_transform::decisions::{DecisionStrategy, FaStrategy, MonotonicityStrategy};
use talkbank_model::model::Bullet;

const UNRESOLVED: DecisionStrategy =
    DecisionStrategy::Monotonicity(MonotonicityStrategy::KeptBulletLeftUnresolved);

/// Finalize with no fresh injection.
fn finalize_as_is(
    chat: &mut talkbank_model::model::ChatFile,
    projection: FaProjection,
    repair: BulletRepairPolicy,
) -> (
    Vec<batchalign_transform::decisions::DecisionRecord>,
    Vec<MonotonicityEffect>,
    RepairStats,
) {
    let finalized =
        finalize_without_injection(chat, projection, repair).expect("every kept bullet holds");
    let stats = finalized.repair_stats().clone();
    let (records, effects) = retained_evidence(chat, finalized);
    (records, effects, stats)
}

/// Give an utterance the input left unbulleted a bullet, the way a
/// derivation would, after the bind (so it is revisable).
fn derive_bullet(chat: &mut talkbank_model::model::ChatFile, utt_idx: usize, start: u64, end: u64) {
    get_test_utterance(chat, utt_idx).main.content.bullet = Some(Bullet::new(start, end));
}

// ---------------------------------------------------------------------------
// Monotonicity, Pass 2 (end overlap)
// ---------------------------------------------------------------------------

/// Two same-speaker kept bullets overlap and their words interleave. The
/// default cuts the earlier bullet and its last word; keep changes neither,
/// records the conflict once (both sweeps visit the pair), and produces no
/// timing effect because it changed no timing.
#[test]
fn two_kept_bullets_overlapping_are_recorded_not_resolved() {
    let (derived, kept) = run_under_both_policies(
        &two_speaker_transcript(
            "*CHI:\thello there . \u{15}1000_5000\u{15}\n*CHI:\tgood bye . \u{15}4000_8000\u{15}\n",
        ),
        EndOverlapPolicy::ClampAllAdjacent,
        |chat, projection| {
            align_one_group(
                chat,
                projection,
                vec![
                    fa_word(0, 0, "hello"),
                    fa_word(0, 1, "there"),
                    fa_word(1, 0, "good"),
                    fa_word(1, 1, "bye"),
                ],
                &[0, 1],
                &[(1000, 2500), (2500, 4600), (4200, 5500), (5500, 7500)],
                true,
                BulletRepairPolicy::Disabled,
            )
        },
    );
    assert!(matches!(
        derived.outcome.1.as_slice(),
        [MonotonicityEffect::EndClampedInterleavedWords { .. }]
    ));

    let (records, effects) = kept.outcome;
    assert_eq!(
        wor_timings(&kept.chat, 0),
        vec![Some((1000, 2500)), Some((2500, 4600))]
    );
    assert!(effects.is_empty(), "no timing was changed: {effects:?}");
    let unresolved = records_with(&records, UNRESOLVED);
    assert_eq!(unresolved.len(), 1, "one record per pair, not per sweep");
    assert!(unresolved[0].needs_review, "a speaker overlapping themself");
}

/// A derived bullet overlapping a kept SUCCESSOR is resolved by cutting the
/// derived side only. The default moves the successor's start to its word
/// hull; keep cannot, so the derived bullet and its last word are clamped.
#[test]
fn derived_bullet_yields_to_a_kept_successor() {
    let (derived, kept) = run_under_both_policies(
        &two_speaker_transcript("*CHI:\tone two .\n*CHI:\tthree . \u{15}4000_8000\u{15}\n"),
        EndOverlapPolicy::ClampAllAdjacent,
        |chat, projection| {
            align_one_group(
                chat,
                projection,
                vec![
                    fa_word(0, 0, "one"),
                    fa_word(0, 1, "two"),
                    fa_word(1, 0, "three"),
                ],
                &[0, 1],
                &[(1000, 2000), (2000, 4500), (4600, 7000)],
                true,
                BulletRepairPolicy::Disabled,
            )
        },
    );
    assert_eq!(get_utterance_bullet(&derived.chat, 1), Some((4600, 8000)));

    assert_eq!(get_utterance_bullet(&kept.chat, 0), Some((1000, 4000)));
    assert_eq!(
        wor_timings(&kept.chat, 0),
        vec![Some((1000, 2000)), Some((2000, 4000))]
    );
    assert!(matches!(
        kept.outcome.1.as_slice(),
        [MonotonicityEffect::EndClampedInterleavedWords {
            clamped_to_ms: 4000,
            ..
        }]
    ));
}

/// A derived bullet overlapping a kept PREDECESSOR gives way by moving its
/// own start forward to the kept end and cutting its leading word, instead of
/// the default's cut of the predecessor.
#[test]
fn derived_successor_yields_to_a_kept_predecessor() {
    let (derived, kept) = run_under_both_policies(
        &two_speaker_transcript("*CHI:\thello there . \u{15}1000_5000\u{15}\n*CHI:\tgood bye .\n"),
        EndOverlapPolicy::ClampAllAdjacent,
        |chat, projection| {
            align_one_group(
                chat,
                projection,
                vec![
                    fa_word(0, 0, "hello"),
                    fa_word(0, 1, "there"),
                    fa_word(1, 0, "good"),
                    fa_word(1, 1, "bye"),
                ],
                &[0, 1],
                &[(1000, 2500), (2500, 4600), (4200, 5500), (5500, 7500)],
                true,
                BulletRepairPolicy::Disabled,
            )
        },
    );
    assert_eq!(get_utterance_bullet(&derived.chat, 0), Some((1000, 4200)));

    assert_eq!(get_utterance_bullet(&kept.chat, 1), Some((5000, 7500)));
    assert_eq!(
        wor_timings(&kept.chat, 1),
        vec![Some((5000, 5500)), Some((5500, 7500))]
    );
    let (records, effects) = kept.outcome;
    assert!(matches!(
        effects.as_slice(),
        [MonotonicityEffect::YieldedToKeptBullet {
            kept_boundary_ms: 5000,
            outcome: KeptBulletYield::StartMoved {
                from_ms: 4200,
                to_ms: 5000,
                words_trimmed: 1,
                ..
            },
            ..
        }]
    ));
    assert_eq!(
        records_with(
            &records,
            DecisionStrategy::Monotonicity(MonotonicityStrategy::YieldedToKeptBullet)
        )
        .len(),
        1
    );
}

// ---------------------------------------------------------------------------
// Monotonicity, Pass 1 (start order)
// ---------------------------------------------------------------------------

/// Two kept bullets out of start order: the default strips the later one;
/// keep leaves both and records the regression.
#[test]
fn kept_start_regressing_below_a_kept_start_is_recorded_not_stripped() {
    let (derived, kept) = run_under_both_policies(
        &two_speaker_transcript(
            "*CHI:\tfirst . \u{15}5000_6000\u{15}\n*MOT:\tsecond . \u{15}3000_4000\u{15}\n",
        ),
        EndOverlapPolicy::PreserveCrossSpeaker,
        |chat, projection| finalize_as_is(chat, projection, BulletRepairPolicy::Disabled),
    );
    assert_eq!(get_utterance_bullet(&derived.chat, 1), None);

    let (records, effects, _) = kept.outcome;
    assert!(effects.is_empty());
    let unresolved = records_with(&records, UNRESOLVED);
    assert_eq!(unresolved.len(), 1);
    assert!(
        unresolved[0]
            .reason
            .starts_with("non_monotonic kept_main_bullet")
    );
}

/// A derived bullet that starts AFTER the next kept start is the one outside
/// the kept skeleton, so it is stripped (having yielded to the kept start).
/// The default strips the kept one instead.
#[test]
fn derived_bullet_before_a_lower_kept_start_is_the_one_stripped() {
    let (derived, kept) = run_under_both_policies(
        &two_speaker_transcript(
            "*CHI:\tone . \u{15}1000_2000\u{15}\n*MOT:\ttwo .\n*CHI:\tthree . \u{15}5000_8000\u{15}\n",
        ),
        EndOverlapPolicy::PreserveCrossSpeaker,
        |chat, projection| {
            derive_bullet(chat, 1, 6000, 7000);
            finalize_as_is(chat, projection, BulletRepairPolicy::Disabled)
        },
    );
    assert_eq!(get_utterance_bullet(&derived.chat, 2), None);
    assert_eq!(get_utterance_bullet(&derived.chat, 1), Some((6000, 7000)));

    assert_eq!(get_utterance_bullet(&kept.chat, 1), None);
    let (records, effects, _) = kept.outcome;
    assert!(matches!(
        effects.as_slice(),
        [MonotonicityEffect::YieldedToKeptBullet {
            kept_boundary_ms: 5000,
            outcome: KeptBulletYield::Stripped {
                start_ms: 6000,
                end_ms: 7000
            },
            ..
        }]
    ));
    assert!(
        records
            .iter()
            .any(|r| r.reason.starts_with("yielded_to_kept_start"))
    );
}

// ---------------------------------------------------------------------------
// Bullet repair
// ---------------------------------------------------------------------------

/// Bullet repair averages a small cross-speaker overlap by default; with both
/// bullets kept it touches neither, and monotonicity records what remains
/// (not for review: cross-speaker overlap is ordinary conversation).
#[test]
fn bullet_repair_does_not_average_kept_bullets() {
    let (derived, kept) = run_under_both_policies(
        &two_speaker_transcript(
            "*CHI:\tone . \u{15}1000_5000\u{15}\n*MOT:\ttwo . \u{15}4800_8000\u{15}\n",
        ),
        EndOverlapPolicy::ClampAllAdjacent,
        |chat, projection| finalize_as_is(chat, projection, BulletRepairPolicy::Enabled),
    );
    assert_eq!(derived.outcome.2.boundary_averaged, 1);

    let (records, effects, stats) = kept.outcome;
    assert_eq!(stats.boundary_averaged, 0);
    assert!(effects.is_empty());
    let unresolved = records_with(&records, UNRESOLVED);
    assert_eq!(unresolved.len(), 1);
    assert!(!unresolved[0].needs_review);
}

/// A kept bullet is never a gap-fill target: the default snaps its start back
/// to the speaker's previous end; keep leaves it.
#[test]
fn a_kept_bullet_is_never_a_gap_fill_target() {
    let (derived, kept) = run_under_both_policies(
        &two_speaker_transcript(
            "*CHI:\tone . \u{15}1000_2000\u{15}\n*CHI:\ttwo . \u{15}2600_4000\u{15}\n",
        ),
        EndOverlapPolicy::ClampAllAdjacent,
        |chat, projection| finalize_as_is(chat, projection, BulletRepairPolicy::Enabled),
    );
    assert_eq!(derived.outcome.2.gaps_filled, 1);
    assert_eq!(get_utterance_bullet(&derived.chat, 1), Some((2000, 4000)));
    assert_eq!(kept.outcome.2.gaps_filled, 0);
}

/// A kept bullet outside its speaker's naive LIS: the default strips it; keep
/// makes it an anchor, so the derived bullets between the anchors that fall
/// outside them are the ones stripped.
#[test]
fn a_kept_bullet_outside_the_naive_lis_is_an_anchor() {
    let (derived, kept) = run_under_both_policies(
        &two_speaker_transcript(
            "*CHI:\ta . \u{15}1000_1500\u{15}\n*CHI:\tb .\n*CHI:\tc .\n*CHI:\td . \u{15}3000_3500\u{15}\n",
        ),
        EndOverlapPolicy::ClampAllAdjacent,
        |chat, projection| {
            derive_bullet(chat, 1, 5000, 5500);
            derive_bullet(chat, 2, 6000, 6500);
            finalize_as_is(chat, projection, BulletRepairPolicy::Enabled)
        },
    );
    assert_eq!(
        get_utterance_bullet(&derived.chat, 3),
        None,
        "naive LIS strips the kept one"
    );

    let (records, _, stats) = kept.outcome;
    assert_eq!(stats.timing_stripped, 2);
    assert_eq!(get_utterance_bullet(&kept.chat, 1), None);
    assert_eq!(get_utterance_bullet(&kept.chat, 2), None);
    assert_eq!(
        records_with(&records, DecisionStrategy::Fa(FaStrategy::LisRemoval)).len(),
        2
    );
}
