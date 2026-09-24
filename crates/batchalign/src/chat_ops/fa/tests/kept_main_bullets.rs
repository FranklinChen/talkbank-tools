//! `--main-bullets keep` and `exact`, binding and word fitting: a bullet the
//! input carried is written back exactly, word timings are fitted inside it on
//! both tiers, and an utterance the input left unbulleted still derives a
//! bullet under `keep` but stays without one under `exact`.
//! Every keep run goes through [`run_under_both_policies`] or calls
//! [`assert_given_bullets_held`] itself, so EVERY given bullet is checked,
//! not a chosen few. Ordering phases (monotonicity, repair) are in
//! `kept_main_bullets_ordering.rs`.

use super::*;

use batchalign_transform::decisions::{DecisionStrategy, FaStrategy};
use talkbank_model::model::{Bullet, BulletSource};

const CLAMPED: DecisionStrategy = DecisionStrategy::Fa(FaStrategy::WordsClampedToKeptBullet);

// ---------------------------------------------------------------------------
// Binding and the runtime check
// ---------------------------------------------------------------------------

/// An empty bullet (start equals end) is kept exactly as written, its words
/// are all untimed, and the run says so.
#[test]
fn empty_given_bullet_is_kept_exactly_with_its_words_untimed() {
    let input = two_speaker_transcript(
        "*CHI:\thello . \u{15}1000_2000\u{15}\n*CHI:\tthere . \u{15}3000_3000\u{15}\n",
    );
    let given = parse_chat(&input);
    let mut chat = parse_chat(&input);
    let keep = bound_projection(
        &chat,
        EndOverlapPolicy::ClampAllAdjacent,
        MainBulletPolicy::KeepGiven,
    );
    let (records, _) = align_one_group(
        &mut chat,
        keep,
        vec![fa_word(0, 0, "hello"), fa_word(1, 0, "there")],
        &[0, 1],
        &[(1100, 1900), (2900, 3400)],
        true,
        BulletRepairPolicy::Disabled,
    );

    assert_given_bullets_held(&given, &chat);
    assert_eq!(get_utterance_bullet(&chat, 1), Some((3000, 3000)));
    let clamps = records_with(&records, CLAMPED);
    assert_eq!(clamps.len(), 1);
    assert!(
        clamps[0]
            .reason
            .starts_with("kept_empty_main_bullet=3000_3000"),
        "{}",
        clamps[0].reason
    );
    assert!(clamps[0].needs_review);
}

/// A backward bullet cannot be kept, so binding refuses the file.
#[test]
fn backward_given_bullet_refuses_the_file_at_bind() {
    let chat = parse_chat(&two_speaker_transcript(
        "*CHI:\thello . \u{15}3000_2000\u{15}\n",
    ));
    assert!(MainBulletAuthority::bind(MainBulletPolicy::KeepGiven, &chat).is_err());
    assert_eq!(
        MainBulletAuthority::bind(MainBulletPolicy::DeriveFromWords, &chat),
        Ok(MainBulletAuthority::DeriveFromWords),
        "the default refers to no given bullet, so it has nothing to refuse"
    );
}

/// The runtime check itself: a kept bullet changed after binding fails it.
#[test]
fn held_check_fails_when_a_kept_bullet_is_changed() {
    let mut chat = parse_chat(&two_speaker_transcript(
        "*CHI:\thello . \u{15}1000_2000\u{15}\n*MOT:\tthere .\n",
    ));
    let authority =
        MainBulletAuthority::bind(MainBulletPolicy::KeepGiven, &chat).expect("forward bullets");
    assert_eq!(
        authority.verify_held(&chat).map(KeptBulletsHeld::kept),
        Ok(1)
    );

    get_test_utterance(&mut chat, 0)
        .main
        .content
        .bullet
        .as_mut()
        .expect("fixture is bulleted")
        .timing
        .end_ms = 2500;
    assert!(matches!(
        authority.verify_held(&chat),
        Err(KeptBulletError::Drifted {
            utterance_idx: 0,
            ..
        })
    ));

    get_test_utterance(&mut chat, 0).main.content.bullet = None;
    assert!(matches!(
        authority.verify_held(&chat),
        Err(KeptBulletError::Drifted {
            utterance_idx: 0,
            ..
        })
    ));
}

// ---------------------------------------------------------------------------
// Word fitting
// ---------------------------------------------------------------------------

/// Words straddling both edges: the default widens the bullet to their hull;
/// keep leaves the bullet and cuts each word to the edge it crossed.
#[test]
fn kept_bullet_is_not_widened_and_straddling_words_are_clamped() {
    let (derived, kept) = run_under_both_policies(
        &two_speaker_transcript("*CHI:\thello world . \u{15}1000_3000\u{15}\n"),
        EndOverlapPolicy::ClampAllAdjacent,
        |chat, projection| {
            align_one_group(
                chat,
                projection,
                make_fa_words(&["hello", "world"]),
                &[0],
                &[(800, 1500), (1500, 3400)],
                true,
                BulletRepairPolicy::Disabled,
            )
        },
    );
    assert_eq!(get_utterance_bullet(&derived.chat, 0), Some((800, 3400)));

    assert_eq!(
        wor_timings(&kept.chat, 0),
        vec![Some((1000, 1500)), Some((1500, 3000))],
        "each word is cut to the edge it crossed"
    );
    let (records, effects) = kept.outcome;
    let clamps = records_with(&records, CLAMPED);
    assert_eq!(clamps.len(), 1);
    assert!(clamps[0].reason.contains("words_trimmed=2 words_dropped=0"));
    assert!(!clamps[0].needs_review, "a trim keeps every timing");
    assert!(effects.is_empty());
}

/// A word wholly outside the kept bullet loses its timing and is written
/// untimed in `%wor`; the drop is recorded with its tier and lost span.
#[test]
fn word_wholly_outside_kept_bullet_becomes_untimed() {
    let (_, kept) = run_under_both_policies(
        &two_speaker_transcript("*CHI:\thello big world . \u{15}1000_3000\u{15}\n"),
        EndOverlapPolicy::ClampAllAdjacent,
        |chat, projection| {
            align_one_group(
                chat,
                projection,
                make_fa_words(&["hello", "big", "world"]),
                &[0],
                &[(1100, 1800), (1800, 2900), (3500, 4200)],
                true,
                BulletRepairPolicy::Disabled,
            )
        },
    );
    assert_eq!(
        wor_timings(&kept.chat, 0),
        vec![Some((1100, 1800)), Some((1800, 2900)), None]
    );
    let clamps = records_with(&kept.outcome.0, CLAMPED);
    assert_eq!(clamps.len(), 1);
    assert!(
        clamps[0]
            .reason
            .contains("words_dropped=1 dropped=[main:2:3500_4200]"),
        "reason: {}",
        clamps[0].reason
    );
    assert!(clamps[0].needs_review, "a lost measurement is flagged");
}

/// An utterance the input left unbulleted derives its bullet under keep
/// exactly as it does by default.
#[test]
fn unbulleted_utterance_still_derives_its_bullet_under_keep() {
    let (derived, kept) = run_under_both_policies(
        &two_speaker_transcript("*CHI:\thello . \u{15}1000_2000\u{15}\n*MOT:\tgood morning .\n"),
        EndOverlapPolicy::ClampAllAdjacent,
        |chat, projection| {
            align_one_group(
                chat,
                projection,
                vec![
                    fa_word(0, 0, "hello"),
                    fa_word(1, 0, "good"),
                    fa_word(1, 1, "morning"),
                ],
                &[0, 1],
                &[(900, 1900), (5000, 5400), (5400, 6100)],
                true,
                BulletRepairPolicy::Disabled,
            )
        },
    );
    assert_eq!(get_utterance_bullet(&kept.chat, 1), Some((5000, 6100)));
    assert_eq!(
        get_utterance_bullet(&kept.chat, 1),
        get_utterance_bullet(&derived.chat, 1)
    );
}

/// Under `exact`, the same unbulleted utterance stays without a bullet: its
/// words are aligned with the rest but lose their timings, on both tiers, and
/// the run records where the aligner had put them. The given bullet holds.
#[test]
fn unbulleted_utterance_stays_without_a_bullet_under_exact() {
    let input =
        two_speaker_transcript("*CHI:\thello . \u{15}1000_2000\u{15}\n*MOT:\tgood morning .\n");
    let given = parse_chat(&input);
    let mut chat = parse_chat(&input);
    let exact = bound_projection(
        &chat,
        EndOverlapPolicy::ClampAllAdjacent,
        MainBulletPolicy::KeepExact,
    );
    let (records, _) = align_one_group(
        &mut chat,
        exact,
        vec![
            fa_word(0, 0, "hello"),
            fa_word(1, 0, "good"),
            fa_word(1, 1, "morning"),
        ],
        &[0, 1],
        &[(900, 1900), (5000, 5400), (5400, 6100)],
        true,
        BulletRepairPolicy::Disabled,
    );

    assert_given_bullets_held(&given, &chat);
    assert_eq!(get_utterance_bullet(&chat, 1), None);
    assert!(
        wor_timings(&chat, 1).iter().all(Option::is_none),
        "no %wor timing a bullet could be derived from: {:?}",
        wor_timings(&chat, 1)
    );
    let untimed = records_with(
        &records,
        DecisionStrategy::Fa(FaStrategy::WordsUntimedForKeptAbsence),
    );
    assert_eq!(untimed.len(), 1);
    assert!(
        untimed[0]
            .reason
            .starts_with("kept_absent_main_bullet words_untimed="),
        "{}",
        untimed[0].reason
    );
    assert!(!untimed[0].needs_review);
}

/// The runtime check under `exact`: a kept absence that gained a bullet
/// fails it, and a held one is counted.
#[test]
fn held_check_fails_when_a_kept_absence_gains_a_bullet() {
    let mut chat = parse_chat(&two_speaker_transcript(
        "*CHI:\thello . \u{15}1000_2000\u{15}\n*MOT:\tthere .\n",
    ));
    let authority =
        MainBulletAuthority::bind(MainBulletPolicy::KeepExact, &chat).expect("forward bullets");
    let held = authority.verify_held(&chat).expect("nothing changed");
    assert_eq!((held.kept(), held.kept_absent()), (1, 1));

    get_test_utterance(&mut chat, 1).main.content.bullet = Some(Bullet::new(2500, 3000));
    assert!(matches!(
        authority.verify_held(&chat),
        Err(KeptBulletError::GainedBullet {
            utterance_idx: 1,
            start_ms: 2500,
            end_ms: 3000,
        })
    ));
}

/// `exact` admits no timing recovery; the other policies pass the request
/// through unchanged.
#[test]
fn exact_refuses_utterance_timing_recovery() {
    assert_eq!(
        MainBulletPolicy::KeepExact.admit_utr(Some("rev")),
        Err(crate::chat_ops::fa::ExactRefusesUtr)
    );
    assert_eq!(
        MainBulletPolicy::KeepExact.admit_utr(None::<&str>),
        Ok(None)
    );
    assert_eq!(
        MainBulletPolicy::KeepGiven.admit_utr(Some("rev")),
        Ok(Some("rev"))
    );
    assert_eq!(
        MainBulletPolicy::DeriveFromWords.admit_utr(Some("rev")),
        Ok(Some("rev"))
    );
}

/// The UTR pre-pass runs AFTER the bind and writes bullets onto unbulleted
/// utterances: a provisional hint, or (two-pass recovery) an ordinary
/// authoritative bullet. Neither is a given bullet, so each derives exactly
/// as it would by default.
#[test]
fn utr_bullets_written_after_the_bind_still_derive() {
    for utr_bullet in [Bullet::utr_hint(4800, 7000), Bullet::new(4800, 7000)] {
        let (derived, kept) = run_under_both_policies(
            &two_speaker_transcript(
                "*CHI:\thello . \u{15}1000_2000\u{15}\n*MOT:\tgood morning .\n",
            ),
            EndOverlapPolicy::ClampAllAdjacent,
            |chat, projection| {
                get_test_utterance(chat, 1).main.content.bullet = Some(utr_bullet.clone());
                align_one_group(
                    chat,
                    projection,
                    vec![
                        fa_word(0, 0, "hello"),
                        fa_word(1, 0, "good"),
                        fa_word(1, 1, "morning"),
                    ],
                    &[0, 1],
                    &[(1100, 1900), (5000, 5400), (5400, 6100)],
                    true,
                    BulletRepairPolicy::Disabled,
                )
            },
        );
        assert_eq!(
            get_utterance_bullet(&kept.chat, 1),
            get_utterance_bullet(&derived.chat, 1),
            "source {:?}",
            utr_bullet.source
        );
    }
}

/// A pre-grouping pass widening a kept bullet in the working document does
/// not survive: the bullet bound at the parse is the one written.
#[test]
fn a_kept_bullet_widened_before_grouping_is_restored() {
    let (_, kept) = run_under_both_policies(
        &two_speaker_transcript("*CHI:\thello world . \u{15}1000_3000\u{15}\n"),
        EndOverlapPolicy::ClampAllAdjacent,
        |chat, projection| {
            let bullet = get_test_utterance(chat, 0)
                .main
                .content
                .bullet
                .as_mut()
                .expect("fixture is bulleted");
            bullet.timing.end_ms = 6000;
            bullet.source = BulletSource::Utr;
            align_one_group(
                chat,
                projection,
                make_fa_words(&["hello", "world"]),
                &[0],
                &[(1200, 2000), (2000, 5500)],
                true,
                BulletRepairPolicy::Disabled,
            )
        },
    );
    assert_eq!(
        wor_timings(&kept.chat, 0),
        vec![Some((1200, 2000)), Some((2000, 3000))]
    );
}

/// Narrow-bullet rescue still widens a kept bullet's grouping window (the
/// aligner needs it), but records that as grouping-only, never as a
/// widening of the bullet, which the output does not have.
#[test]
fn narrow_kept_bullet_produces_no_widening_record() {
    let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello there friends . \u{15}1000_3000\u{15}\n*CHI:\tone two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twentyone twentytwo . \u{15}3500_3880\u{15}\n*CHI:\tafterward they went home . \u{15}20000_22000\u{15}\n@End\n";
    let given = parse_chat(input);
    let mut chat = parse_chat(input);
    let keep = bound_projection(
        &chat,
        EndOverlapPolicy::ClampAllAdjacent,
        MainBulletPolicy::KeepGiven,
    );

    let rescue = rescue_narrow_bullets(&mut chat, keep.main_bullets()).expect("in the input");
    assert!(
        records_with(
            &rescue,
            DecisionStrategy::Fa(FaStrategy::NarrowBulletRescued)
        )
        .is_empty(),
        "a kept bullet is never claimed widened"
    );
    let widened = records_with(
        &rescue,
        DecisionStrategy::Fa(FaStrategy::KeptBulletWindowWidened),
    );
    assert_eq!(widened.len(), 1);
    assert!(widened[0].reason.ends_with("scope=grouping_only"));

    let finalized = finalize_without_injection(&mut chat, keep, BulletRepairPolicy::Disabled)
        .expect("every kept bullet holds");
    retained_evidence(&mut chat, finalized);
    assert_given_bullets_held(&given, &chat);
}

// ---------------------------------------------------------------------------
// Routes to finalization
// ---------------------------------------------------------------------------

const WOR_OUTSIDE_BULLET: &str = "*CHI:\thello world . \u{15}1000_3000\u{15}\n%wor:\thello \u{15}900_1500\u{15} world \u{15}1500_3200\u{15} .\n";

/// The all-`%wor` rerun path refreshes the bullet from `%wor` (widening it
/// to the old word hull); keep restores the given bullet, and `%wor` is
/// rewritten from the clamped words.
#[test]
fn wor_reuse_path_keeps_the_given_bullet_and_refits_wor() {
    let (derived, kept) = run_under_both_policies(
        &two_speaker_transcript(WOR_OUTSIDE_BULLET),
        EndOverlapPolicy::ClampAllAdjacent,
        |chat, projection| {
            let touched = refresh_reusable_alignment(chat, ExistingWorBoundaryPolicy::Preserve);
            let finalized = projection_without_injection_with_touched(projection, true, touched)
                .then_finalize(chat, BulletRepairPolicy::Disabled)
                .expect("every kept bullet holds");
            retained_evidence(chat, finalized)
        },
    );
    assert_eq!(get_utterance_bullet(&derived.chat, 0), Some((900, 3200)));
    assert_eq!(
        wor_timings(&kept.chat, 0),
        vec![Some((1000, 1500)), Some((1500, 3000))]
    );
}

/// The same path with no `%wor` write requested (`--nowor`): the existing
/// `%wor` is clamped in place rather than left with words outside the bullet.
#[test]
fn wor_reuse_path_without_a_wor_write_clamps_wor_in_place() {
    let (_, kept) = run_under_both_policies(
        &two_speaker_transcript(WOR_OUTSIDE_BULLET),
        EndOverlapPolicy::ClampAllAdjacent,
        |chat, projection| {
            let touched = refresh_reusable_alignment(chat, ExistingWorBoundaryPolicy::Preserve);
            let finalized = projection_without_injection_with_touched(projection, false, touched)
                .then_finalize(chat, BulletRepairPolicy::Disabled)
                .expect("every kept bullet holds");
            retained_evidence(chat, finalized)
        },
    );
    assert_eq!(
        wor_timings(&kept.chat, 0),
        vec![Some((1000, 1500)), Some((1500, 3000))]
    );
}

/// A CA transcript: the align path never writes `%wor` for one, so a fresh
/// alignment runs with the write suppressed. The given bullet holds and the
/// `%wor` it already carries is clamped in place.
#[test]
fn ca_transcript_keeps_bullets_and_clamps_existing_wor_in_place() {
    let input = format!(
        "@UTF8\n@Begin\n@Languages:\teng\n@Options:\tCA\n@Participants:\tCHI Child\n@ID:\teng|x|CHI|||||Child|||\n{WOR_OUTSIDE_BULLET}@End\n"
    );
    let (_, kept) = run_under_both_policies(
        &input,
        EndOverlapPolicy::ClampAllAdjacent,
        |chat, projection| {
            align_one_group(
                chat,
                projection,
                make_fa_words(&["hello", "world"]),
                &[0],
                &[(950, 1500), (1500, 3100)],
                false,
                BulletRepairPolicy::Disabled,
            )
        },
    );
    assert_eq!(
        wor_timings(&kept.chat, 0),
        vec![Some((1000, 1500)), Some((1500, 3000))]
    );
}
