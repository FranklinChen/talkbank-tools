//! Whether an utterance is speech in the recording being aligned.
//!
//! Some utterances of a transcript are not in its recording at all: a
//! `[+ diary]` utterance is a written diary note set into the transcript
//! (ruling of 2026-10-07). Alignment must never look for one in the audio.
//! Before this existed, timing recovery matched a diary note's words to
//! whatever speech sounded closest, and the note got a bullet: a placement in
//! the recording of something that is not there.
//!
//! [`RecordingPresence::of`] is the one owner of that rule. It reads Chatter's
//! typed postcodes (`main.content.postcodes`), never the utterance's text.
//! Every stage that would place an utterance in the audio asks it:
//!
//! | stage | an utterance not in the recording |
//! | --- | --- |
//! | input admission ([`strip_off_record_timing`]) | loses any bullet, word timing and `%wor` on the working model, so it is no stage's anchor |
//! | timing recovery census (`utr`) | contributes no words and no anchor |
//! | untimed-window search, timed/untimed count | is neither timed nor untimed: it needs no timing |
//! | interpolation (`estimate_untimed_boundaries`) | takes no share of a gap |
//! | grouping (`group_utterances`) | gets no window and no alignment request |
//! | incremental reuse | is not given the prior file's `%wor` |
//! | completion (`fa::completion`) | owes no timing and is listed as an exclusion, never a shortfall, so it does not diagnose the file; any timing on it is a producer fault |
//!
//! What happens to a bullet the INPUT gave such an utterance is the main-bullet
//! policy's, read through this rule: `derive` (the default) recomputes every
//! bullet from the utterance's aligned words, and this utterance has none to
//! align, so it is written without one; `keep` and `exact` keep every given
//! bullet exactly as given, this one too, and never use it as evidence for
//! another utterance: [`super::MainBulletAuthority`] captures it at the parse
//! as an off-record slot and restores it only after every phase that orders
//! or cuts bullets has run.

use talkbank_model::model::{ChatFile, Line, Utterance};

use crate::api::OffRecordPostcode;

/// Whether one utterance is speech in the recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordingPresence {
    /// Ordinary transcribed speech: alignment looks for it.
    InRecording,
    /// Marked by this postcode as not in the recording: alignment leaves it
    /// untimed.
    NotInRecording(OffRecordPostcode),
}

impl RecordingPresence {
    /// Read the utterance's typed postcodes. The first postcode that marks it
    /// as not in the recording decides; a postcode outside
    /// [`OffRecordPostcode`] has no bearing on alignment.
    pub fn of(utterance: &Utterance) -> Self {
        utterance
            .main
            .content
            .postcodes
            .iter()
            .find_map(|postcode| OffRecordPostcode::of_postcode_text(&postcode.text))
            .map_or(Self::InRecording, Self::NotInRecording)
    }
}

/// Remove all timing from every utterance not in the recording: its main-tier
/// bullet, its words' inline timing and its `%wor` tier.
///
/// Called once, at alignment's input admission, on the working model, AFTER
/// the main-bullet policy and the timing obligations were bound to the input
/// as parsed. From here on no stage can read such an utterance's timing as an
/// anchor, a window or reusable word timing. Returns what it removed.
pub(crate) fn strip_off_record_timing(chat_file: &mut ChatFile) -> StrippedOffRecordTiming {
    let mut stripped = StrippedOffRecordTiming {
        bullets: 0,
        timed_utterances: 0,
    };
    for line in &mut chat_file.lines {
        let Line::Utterance(utterance) = line else {
            continue;
        };
        match RecordingPresence::of(utterance) {
            RecordingPresence::InRecording => {}
            RecordingPresence::NotInRecording(_) => {
                if utterance.main.content.bullet.is_some() {
                    stripped.bullets += 1;
                }
                if carries_timing(utterance) {
                    stripped.timed_utterances += 1;
                }
                super::orchestrate::strip_utterance_timing(utterance);
            }
        }
    }
    stripped
}

/// What [`strip_off_record_timing`] removed from the utterances not in the
/// recording. Built only there, from what it saw before removing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StrippedOffRecordTiming {
    /// Main-tier bullets removed.
    bullets: usize,
    /// Utterances that carried any timing (a bullet, or timed words on the
    /// main tier or in `%wor`).
    timed_utterances: usize,
}

impl StrippedOffRecordTiming {
    /// Main-tier bullets removed, for the log.
    pub(crate) fn bullets(self) -> usize {
        self.bullets
    }

    /// What kind of timing was removed, or `None` when there was none: the
    /// one place the two counts are read together.
    pub(crate) fn removed(self) -> Option<crate::error::OffRecordTiming> {
        match (self.timed_utterances, self.bullets) {
            (0, _) => None,
            (_, 0) => Some(crate::error::OffRecordTiming::WordTimingOnly),
            (_, _) => Some(crate::error::OffRecordTiming::BulletNotKept),
        }
    }
}

/// Whether an utterance carries any timing: a main-tier bullet, or a timed
/// word on its main tier or in its `%wor`.
fn carries_timing(utterance: &Utterance) -> bool {
    utterance.main.content.bullet.is_some()
        || utterance
            .wor_tier()
            .is_some_and(|tier| tier.words().any(|word| word.inline_bullet.is_some()))
        || super::collect_existing_fa_word_timings(utterance)
            .iter()
            .any(Option::is_some)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utterances(body: &str) -> ChatFile {
        let text = format!(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tMOT Mother\n\
             @ID:\teng|test|MOT|||||Mother|||\n@Media:\ttest, audio\n{body}@End\n"
        );
        let (file, errors) =
            batchalign_transform::parse::parse_lenient(&crate::chat_parser(), &text);
        assert!(errors.is_empty(), "{errors:?}");
        file
    }

    fn presences(file: &ChatFile) -> Vec<RecordingPresence> {
        file.lines
            .iter()
            .filter_map(|line| match line {
                Line::Utterance(utterance) => Some(RecordingPresence::of(utterance)),
                _ => None,
            })
            .collect()
    }

    /// The postcode is read from the typed model: `[+ diary]` marks the
    /// utterance, another postcode does not, and the word "diary" in the
    /// utterance's own text is speech like any other.
    #[test]
    fn only_the_diary_postcode_marks_an_utterance_as_not_in_the_recording() {
        let file = utterances(
            "*MOT:\twent to the park today . [+ diary]\n\
             *MOT:\tjust a moment . [+ bch]\n\
             *MOT:\tI read your diary .\n\
             *MOT:\tshe said hello . [+ exc] [+ diary]\n",
        );
        assert_eq!(
            presences(&file),
            [
                RecordingPresence::NotInRecording(OffRecordPostcode::Diary),
                RecordingPresence::InRecording,
                RecordingPresence::InRecording,
                RecordingPresence::NotInRecording(OffRecordPostcode::Diary),
            ]
        );
    }

    /// Admission strips a diary note's bullet, word timing and `%wor`, and
    /// leaves every other utterance's timing exactly as it was.
    #[test]
    fn stripping_removes_all_timing_from_an_off_record_utterance_only() {
        let mut file = utterances(
            "*MOT:\tgo there . \u{15}100_900\u{15}\n%wor:\tgo \u{15}100_400\u{15} there \u{15}400_900\u{15} .\n\
             *MOT:\tbig day today . [+ diary] \u{15}1000_2000\u{15}\n%wor:\tbig \u{15}1000_1300\u{15} day \u{15}1300_1600\u{15} today \u{15}1600_2000\u{15} .\n",
        );
        let stripped = strip_off_record_timing(&mut file);
        assert_eq!(stripped.bullets(), 1);
        assert_eq!(
            stripped.removed(),
            Some(crate::error::OffRecordTiming::BulletNotKept)
        );
        let utterances: Vec<&Utterance> = file
            .lines
            .iter()
            .filter_map(|line| match line {
                Line::Utterance(utterance) => Some(utterance.as_ref()),
                _ => None,
            })
            .collect();
        assert!(utterances[0].main.content.bullet.is_some());
        assert!(utterances[0].wor_tier().is_some());
        assert!(utterances[1].main.content.bullet.is_none());
        assert!(utterances[1].wor_tier().is_none());
        assert_eq!(
            super::super::collect_existing_fa_word_timings(utterances[1]),
            [None, None, None]
        );
    }
}
