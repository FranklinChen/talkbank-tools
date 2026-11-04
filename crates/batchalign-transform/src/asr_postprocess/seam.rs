//! The record of a repetition at a Whisper window seam that was kept once.
//!
//! A long recording is decoded in windows that share some audio at each seam,
//! and the two decodes of the shared stretch are reconciled afterwards: a word
//! both windows heard, at the same time, in the shared audio, is one word.
//! Mostly the match is one to one. When it is not (two earlier observations
//! both fit the incoming one, two incoming ones claim the same earlier one, or
//! the matches cross), the reconciler cannot say WHICH observation is the
//! duplicate, only that one is. It keeps the earlier window's copies, drops
//! the later window's, and records that it did so here, with where the seam is
//! and what was repeated, so a reader of the transcript can check the one
//! place the recording was reconciled by rule rather than by evidence.
//!
//! Until 2026-11-03 that case refused the whole recording ("Whisper overlap
//! has ambiguous repeated observations"): a localized uncertainty about one
//! word cost the user every other word of the file. The rule now is the one
//! every localized problem in this pipeline follows: finish the file, say
//! what happened in the file beside the words it concerns (a `%com` tier on
//! the utterance that keeps the copy, rendered by [`SeamAmbiguity::comment_text`],
//! the same mechanism as a dropped recognizer token), and report it to the
//! job so the file finishes as diagnosed rather than clean.
//!
//! The record is born in the reconciler (`batchalign::worker::whisper_windows`,
//! the one place the stitch decision is made) and travels on the
//! `AsrResponse`, then is placed on the utterance holding the kept copy by
//! [`place_seam_ambiguities`] once the utterances exist. Placement is by the
//! kept copy's timing rather than by threading the record through every
//! word-level stage: retokenization splits, merges and rewrites words freely
//! (number expansion, compound merging, multi-word splitting), and a record
//! carried on a word would be lost silently at the first stage that rebuilt
//! the word without it. The kept copy always has timing, because only timed
//! observations can be duplicates of one another. The kept interval is the
//! reconciler's, taken BEFORE the monotone fit that settles overlapping chunk
//! boundaries, so it need not equal any word's final timing exactly; it
//! intersects the kept word's, which is what placement asks.

use batchalign_types::interval::AdmittedInterval;
use serde::{Deserialize, Serialize};

use super::Utterance;
use super::cleanup::{CommentQuoting, quoted_for_comment};

/// Which decode window of the recording, counting from zero in source order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WindowOrdinal(usize);

impl WindowOrdinal {
    /// The ordinal of one window in the recognizer's window sequence.
    pub fn new(ordinal: usize) -> Self {
        Self(ordinal)
    }

    /// The ordinal as a count from zero.
    pub fn as_usize(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for WindowOrdinal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One repeated observation at a window seam that was kept once.
///
/// Fields are private: the one way code makes one is the reconciler's
/// decision ([`Self::new`], called where the later copy is dropped), the one
/// way to read it into a transcript is [`Self::comment_text`]. Deserialization
/// is the second route in, for records retained in a run's
/// `_asr_response.json` and replayed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeamAmbiguity {
    /// Where the seam is: the first millisecond the later window owns, which
    /// is the boundary between the two windows' own stretches of the
    /// recording.
    seam_ms: i64,
    /// The repeated text, as the recognizer emitted it, trimmed.
    text: String,
    /// The window whose copy was dropped: the later of the two.
    dropped_from: WindowOrdinal,
    /// Where the copy that was kept (the earlier window's) lies. An admitted
    /// interval, so its end cannot precede its start; the reconciler takes it
    /// from positions it already holds.
    kept: AdmittedInterval,
}

impl SeamAmbiguity {
    /// Record that the later window's copy of `text` at seam `seam_ms` was
    /// dropped and the earlier window's copy at `kept` was kept.
    pub fn new(
        seam_ms: i64,
        text: &str,
        dropped_from: WindowOrdinal,
        kept: AdmittedInterval,
    ) -> Self {
        Self {
            seam_ms,
            text: text.trim().to_owned(),
            dropped_from,
            kept,
        }
    }

    /// Where the seam is, in milliseconds.
    pub fn seam_ms(&self) -> i64 {
        self.seam_ms
    }

    /// The repeated text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The window whose copy was dropped.
    pub fn dropped_from(&self) -> WindowOrdinal {
        self.dropped_from
    }

    /// Where the kept copy lies.
    pub fn kept(&self) -> AdmittedInterval {
        self.kept
    }

    /// Whether a word's timing intersects the kept copy by a positive length.
    ///
    /// Strict on both sides: a word that merely abuts the kept copy (its end
    /// is the copy's start) is the word BEFORE it, and placing the record by
    /// abutment put the comment on the previous utterance whenever an
    /// utterance boundary fell exactly there. The kept copy itself is
    /// strictly positive in seconds, and a sub-millisecond copy that rounds
    /// to zero length intersects nothing and takes the fallback placement.
    fn kept_overlaps(&self, word: &super::AsrWord) -> bool {
        matches!(
            (word.start_ms, word.end_ms),
            (Some(start), Some(end)) if start < self.kept.end_ms() && end > self.kept.start_ms()
        )
    }

    /// The one rendering of this record as CHAT comment text, for the `%com`
    /// tier on the utterance that keeps the copy:
    /// `batchalign window seam at 30120 ms: repeated "go" kept once, review here`.
    ///
    /// The text is quoted as a phrase: spaces stay (a repeated span may be
    /// several words) and a character that would break or restructure the
    /// comment line is written as its code point, by the same rule a dropped
    /// token's comment uses.
    pub fn comment_text(&self) -> String {
        format!(
            "batchalign window seam at {} ms: repeated {} kept once, review here",
            self.seam_ms,
            quoted_for_comment(&self.text, CommentQuoting::Phrase)
        )
    }
}

/// Why seam records could not be placed: there is no utterance to place them
/// on. A caller sees this only on a transcript with no utterances, which the
/// pipeline refuses as empty before the records matter; it is an error rather
/// than a silent return so a record can never be dropped on the floor.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0} window seam record(s) have no utterance to be placed on")]
pub struct UnplacedSeams(pub usize);

/// Put every seam record on the utterance that keeps its copy.
///
/// The utterance chosen is the first whose words intersect the kept copy's
/// interval by a positive length ([`SeamAmbiguity::kept_overlaps`]).
/// Retokenization keeps word timing while it rewrites word text, so the kept
/// copy's interval still finds its word, or the word it was merged into or
/// split from. Should no word intersect it (the copy was dropped later by the
/// CHAT-legality sanitizer, say), the record goes on the first utterance that
/// starts at or after the seam, and failing that on the last utterance, so it
/// is never lost; the comment carries the seam time, so a reader can still
/// find the place.
pub fn place_seam_ambiguities(
    utterances: &mut [Utterance],
    seams: Vec<SeamAmbiguity>,
) -> Result<(), UnplacedSeams> {
    if seams.is_empty() {
        return Ok(());
    }
    let Some(last) = utterances.len().checked_sub(1) else {
        return Err(UnplacedSeams(seams.len()));
    };
    for seam in seams {
        let overlapping = utterances
            .iter()
            .position(|utterance| utterance.words.iter().any(|word| seam.kept_overlaps(word)));
        let at_or_after_seam = || {
            utterances.iter().position(|utterance| {
                utterance
                    .words
                    .iter()
                    .any(|word| word.start_ms.is_some_and(|start| start >= seam.seam_ms()))
            })
        };
        let index = overlapping.or_else(at_or_after_seam).unwrap_or(last);
        utterances[index].seam_ambiguities.push(seam);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr_postprocess::{AsrWord, SpeakerIndex, UtteranceEnd};

    fn utterance(words: &[(&str, Option<i64>, Option<i64>)]) -> Utterance {
        Utterance {
            speaker: SpeakerIndex(0),
            words: words
                .iter()
                .map(|(text, start, end)| AsrWord::new(*text, *start, *end))
                .collect(),
            end: UtteranceEnd::observed(".").expect("a period is a terminator"),
            lang: None,
            dropped_tokens: Vec::new(),
            seam_ambiguities: Vec::new(),
        }
    }

    fn seam(seam_ms: i64, kept: (i64, i64)) -> SeamAmbiguity {
        SeamAmbiguity::new(
            seam_ms,
            "go",
            WindowOrdinal::new(1),
            AdmittedInterval::admit_millis(kept.0, kept.1).expect("an ordered fixture"),
        )
    }

    /// The record lands on the utterance whose words overlap the kept copy,
    /// not on the first or the last.
    #[test]
    fn a_seam_record_is_placed_on_the_utterance_that_keeps_the_copy() {
        let mut utterances = vec![
            utterance(&[("hello", Some(0), Some(500))]),
            utterance(&[("go", Some(20_000), Some(22_000))]),
            utterance(&[("bye", Some(30_000), Some(30_500))]),
        ];
        place_seam_ambiguities(&mut utterances, vec![seam(20_500, (20_000, 22_000))])
            .expect("three utterances to place on");
        assert!(utterances[0].seam_ambiguities.is_empty());
        assert_eq!(utterances[1].seam_ambiguities.len(), 1);
        assert!(utterances[2].seam_ambiguities.is_empty());
    }

    /// A word that ends exactly where the kept copy starts is the word before
    /// it, not a holder of it: with an utterance boundary there, the record
    /// goes on the utterance that begins with the copy, not the one that
    /// abuts it. The kept interval may also differ from the word's final
    /// timing by the monotone fit of overlapping chunks, which intersection
    /// tolerates.
    #[test]
    fn abutting_timing_does_not_claim_the_record_and_fitted_timing_still_does() {
        let mut utterances = vec![
            utterance(&[("so", Some(19_000), Some(20_000))]),
            utterance(&[
                ("go", Some(20_000), Some(21_250)),
                ("go", Some(21_250), Some(22_500)),
            ]),
        ];
        place_seam_ambiguities(&mut utterances, vec![seam(22_000, (20_000, 22_000))])
            .expect("two utterances to place on");
        assert!(
            utterances[0].seam_ambiguities.is_empty(),
            "abutment is not overlap"
        );
        assert_eq!(utterances[1].seam_ambiguities.len(), 1);
    }

    /// With no word overlapping the kept copy, the record is still written:
    /// on the first utterance at or after the seam, else the last.
    #[test]
    fn an_unmatched_seam_record_falls_back_to_the_seam_then_the_last_utterance() {
        let mut utterances = vec![
            utterance(&[("hello", Some(0), Some(500))]),
            utterance(&[("later", Some(40_000), Some(40_500))]),
        ];
        place_seam_ambiguities(&mut utterances, vec![seam(20_500, (20_000, 22_000))])
            .expect("two utterances to place on");
        assert_eq!(utterances[1].seam_ambiguities.len(), 1);

        let mut untimed = vec![
            utterance(&[("a", None, None)]),
            utterance(&[("b", None, None)]),
        ];
        place_seam_ambiguities(&mut untimed, vec![seam(20_500, (20_000, 22_000))])
            .expect("two utterances to place on");
        assert_eq!(untimed[1].seam_ambiguities.len(), 1);
    }

    /// No utterance at all is a typed error, never a dropped record.
    #[test]
    fn seam_records_without_utterances_are_refused_not_dropped() {
        assert_eq!(
            place_seam_ambiguities(&mut [], vec![seam(1, (0, 1))]),
            Err(UnplacedSeams(1))
        );
        assert_eq!(place_seam_ambiguities(&mut [], Vec::new()), Ok(()));
    }

    /// The comment names the seam, the text and the decision; a character
    /// that would break the line is written as its code point, a space stays.
    #[test]
    fn the_comment_names_the_seam_and_escapes_line_breakers() {
        assert_eq!(
            seam(30_120, (29_000, 30_500)).comment_text(),
            "batchalign window seam at 30120 ms: repeated \"go\" kept once, review here"
        );
        let broken = SeamAmbiguity::new(
            1,
            "a\u{15}b\nc d",
            WindowOrdinal::new(2),
            AdmittedInterval::admit_millis(0, 1).expect("an ordered fixture"),
        );
        assert_eq!(
            broken.comment_text(),
            "batchalign window seam at 1 ms: repeated \"aU+0015bU+000Ac d\" kept once, review here"
        );
    }
}
