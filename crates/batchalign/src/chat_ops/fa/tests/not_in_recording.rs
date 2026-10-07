//! Utterances the transcript marks as not in the recording (`[+ diary]`):
//! no stage that places speech in the audio may place them, and none may
//! use them as evidence for their neighbours.

use super::*;

use crate::chat_ops::fa::utr::{AsrTimingToken, inject_utr_timing};

/// A transcript of invented utterances; `body` is its utterance lines.
fn transcript(body: &str) -> talkbank_model::model::ChatFile {
    parse_chat(&format!(
        "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tMOT Mother\n\
         @ID:\teng|test|MOT|||||Mother|||\n@Media:\ttest, audio\n{body}@End\n"
    ))
}

fn bullets(chat: &talkbank_model::model::ChatFile) -> Vec<Option<(u64, u64)>> {
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

/// Timing recovery gives a diary note no bullet even when the recording has
/// its words: here the mother SAYS the words the note wrote, and the spoken
/// utterance after the note is the one they belong to. Before, the note came
/// first in the matching and took them.
#[test]
fn recovery_matches_no_word_of_a_diary_note() {
    let mut chat = transcript(
        "*MOT:\tbig day today . [+ diary]\n\
         *MOT:\tbig day today .\n",
    );
    let tokens: Vec<AsrTimingToken> = [
        ("big", 1_000, 1_300),
        ("day", 1_300, 1_600),
        ("today", 1_600, 2_000),
    ]
    .into_iter()
    .map(|(text, start_ms, end_ms)| AsrTimingToken {
        text: text.to_owned(),
        start_ms,
        end_ms,
    })
    .collect();
    inject_utr_timing(&mut chat, &tokens);
    let [diary, spoken] = bullets(&chat)[..] else {
        panic!("two utterances");
    };
    assert_eq!(diary, None, "the note is not in the recording");
    assert!(spoken.is_some(), "the spoken utterance takes its own words");
}

/// Grouping makes no request for a diary note, and interpolation gives it no
/// share of the gap: the spoken utterance beside it is placed exactly as if
/// the note were not there.
#[test]
fn grouping_and_interpolation_leave_a_diary_note_out() {
    let with_note = transcript(
        "*MOT:\tgo there . \u{15}1000_2000\u{15}\n\
         *MOT:\twent to the park with grandma . [+ diary]\n\
         *MOT:\tcome here now .\n\
         *MOT:\tall done . \u{15}9000_10000\u{15}\n",
    );
    let without_note = transcript(
        "*MOT:\tgo there . \u{15}1000_2000\u{15}\n\
         *MOT:\tcome here now .\n\
         *MOT:\tall done . \u{15}9000_10000\u{15}\n",
    );
    let recording = test_recording(20_000);
    let placed_with = estimate_untimed_boundaries(&with_note, &recording).placements;
    let placed_without = estimate_untimed_boundaries(&without_note, &recording).placements;
    assert_eq!(
        placed_with[2], placed_without[1],
        "the note takes no share of the gap"
    );

    let grouping = group_utterances(
        &with_note,
        20_000,
        &recording,
        &crate::chat_ops::fa::AnchorIndex::not_observed(),
    );
    let requested: Vec<usize> = grouping
        .groups
        .iter()
        .flat_map(|group| group.utterance_indices().iter().map(|index| index.raw()))
        .collect();
    assert_eq!(
        requested,
        [0, 2, 3],
        "every spoken utterance, and not the note"
    );
    assert!(
        grouping.decisions.is_empty(),
        "leaving the note out is not a refusal"
    );
}

/// A diary note needs no timing: a file whose only untimed utterance is one
/// has nothing for recovery to do, and no window of audio to transcribe.
#[test]
fn a_diary_note_is_not_untimed_work() {
    let chat = transcript(
        "*MOT:\tgo there . \u{15}1000_2000\u{15}\n\
         *MOT:\twent to the park . [+ diary]\n\
         *MOT:\tall done . \u{15}9000_10000\u{15}\n",
    );
    assert_eq!(count_utterance_timing(&chat), (2, 0));
    let windows = find_untimed_windows(&chat, &test_recording(20_000), 500)
        .expect("windows inside the recording");
    assert!(windows.is_empty(), "no audio to transcribe for a note");
}

/// Under `keep`, a diary note's given bullet is written back exactly, but it
/// is no fixed point for its neighbours: here it overlaps the next spoken
/// utterance, which keeps the timing the aligner measured. Before, the note's
/// bullet was restored at imposition, ahead of monotonicity, which then cut
/// the spoken utterance back to the note's end and stripped it.
#[test]
fn a_kept_diary_bullet_is_restored_after_ordering_and_orders_nothing() {
    let input = parse_chat(&two_speaker_transcript(
        "*CHI:\thello there . \u{15}1000_3000\u{15}\n\
         *MOT:\tbig day today . [+ diary] \u{15}2500_6000\u{15}\n\
         *CHI:\tcome here .\n",
    ));
    for policy in [MainBulletPolicy::KeepGiven, MainBulletPolicy::KeepExact] {
        let keep = bound_projection(&input, EndOverlapPolicy::ClampAllAdjacent, policy);
        // As input admission leaves the working model.
        let mut chat = input.clone();
        crate::chat_ops::fa::strip_off_record_timing(&mut chat);
        if let MainBulletPolicy::KeepExact = policy {
            // `exact` keeps an unbulleted utterance unbulleted: give the
            // spoken one a bullet of its own so both policies time it.
            get_test_utterance(&mut chat, 2).main.content.bullet =
                Some(talkbank_model::model::Bullet::new(3_400, 4_600));
        }
        let keep = if let MainBulletPolicy::KeepExact = policy {
            let mut given = input.clone();
            get_test_utterance(&mut given, 2).main.content.bullet =
                Some(talkbank_model::model::Bullet::new(3_400, 4_600));
            bound_projection(&given, EndOverlapPolicy::ClampAllAdjacent, policy)
        } else {
            keep
        };
        align_one_group(
            &mut chat,
            keep,
            vec![
                fa_word(0, 0, "hello"),
                fa_word(0, 1, "there"),
                fa_word(2, 0, "come"),
                fa_word(2, 1, "here"),
            ],
            &[0, 2],
            &[
                (1_100, 1_900),
                (2_000, 2_900),
                (3_500, 4_000),
                (4_000, 4_500),
            ],
            true,
            BulletRepairPolicy::Enabled,
        );
        assert_eq!(
            get_utterance_bullet(&chat, 1),
            Some((2_500, 6_000)),
            "{policy:?}"
        );
        let spoken = get_utterance_bullet(&chat, 2).expect("the spoken utterance stays timed");
        assert!(
            spoken.0 < 6_000,
            "{policy:?}: not yielded to the note's end: {spoken:?}"
        );
    }
}
