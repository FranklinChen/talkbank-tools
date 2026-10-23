//! Reading timed utterances out of a parsed CHAT file.
//!
//! The one place this feature touches the CHAT model, and it touches it through
//! the typed AST: a `Line::Utterance`, its speaker, and its timing bullet. No
//! text is parsed, no marker is matched, and nothing is re-derived from a
//! serialized form.

use talkbank_model::model::{ChatFile, Line};

use crate::media::window::MediaWindow;
use crate::time::FileMs;

use super::run::{TranscriptUtterance, UtteranceTiming};

/// Which speaker tiers a run scores.
///
/// A sum rather than an empty `Vec` meaning "all": an empty selection and
/// "every tier" are different intentions, and a caller that passed an empty
/// list by accident should not silently get the whole transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TierSelection {
    /// Every speaker tier in the transcript.
    AllTiers,
    /// Only these speaker codes, given without the leading `*`.
    Named(Vec<String>),
}

impl TierSelection {
    /// Read the caller's `--tiers` list, where empty means every tier.
    ///
    /// The emptiness question is answered HERE, once, at the boundary, so no
    /// later code has to remember what an empty list meant.
    pub fn from_option(tiers: &[String]) -> Self {
        if tiers.is_empty() {
            Self::AllTiers
        } else {
            Self::Named(
                tiers
                    .iter()
                    .map(|tier| tier.trim().trim_start_matches('*').to_owned())
                    .collect(),
            )
        }
    }

    fn includes(&self, speaker: &str) -> bool {
        match self {
            Self::AllTiers => true,
            Self::Named(names) => names.iter().any(|name| name == speaker),
        }
    }

    /// The tiers as they are recorded in the evidence's provenance.
    pub fn recorded(&self) -> Vec<String> {
        match self {
            Self::AllTiers => vec!["*".to_owned()],
            Self::Named(names) => names.clone(),
        }
    }
}

/// Collect every utterance of the selected tiers, timed or not.
///
/// Untimed utterances are INCLUDED. They cannot be scored, and they are
/// reported `Unscored { NoBullet }` rather than omitted: a file that silently
/// dropped them would let a consumer conclude the transcript has fewer
/// utterances than it has, and "we could not measure this one" is exactly the
/// fact a reviewer needs.
///
/// `utterance_index` counts utterances of the SELECTED tiers in transcript
/// order, and `line` is the one-based line of the main tier, so a reader can go
/// straight to it.
pub fn read_utterances(chat_file: &ChatFile, tiers: &TierSelection) -> Vec<TranscriptUtterance> {
    let mut collected = Vec::new();
    for (line_idx, line) in chat_file.lines.as_slice().iter().enumerate() {
        let Line::Utterance(utterance) = line else {
            continue;
        };
        let speaker = utterance.main.speaker.as_str();
        if !tiers.includes(speaker) {
            continue;
        }
        let timing = match utterance.main.content.bullet.as_ref() {
            None => UtteranceTiming::Untimed,
            Some(bullet) => {
                let start = FileMs::new(bullet.timing.start_ms);
                let end = FileMs::new(bullet.timing.end_ms);
                match MediaWindow::new(start, end) {
                    Ok(window) => UtteranceTiming::Window(window),
                    Err(_) => UtteranceTiming::Invalid { start, end },
                }
            }
        };
        collected.push(TranscriptUtterance {
            utterance_index: collected.len(),
            line: line_idx + 1,
            speaker: speaker.to_owned(),
            timing,
        });
    }
    collected
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;

    /// The NAK byte CHAT wraps a timing bullet in. Functional data, so it is
    /// written as an escape and named rather than pasted as a raw control
    /// character nobody can see in a diff.
    const BULLET_DELIMITER: char = '\u{15}';

    fn sample() -> String {
        let bullet =
            |start: u32, end: u32| format!("{BULLET_DELIMITER}{start}_{end}{BULLET_DELIMITER}");
        [
            "@UTF8".to_owned(),
            "@Begin".to_owned(),
            "@Languages:\teng".to_owned(),
            "@Participants:\tINV Investigator, CHI Target_Child".to_owned(),
            "@ID:\teng|test|INV|||||Investigator|||".to_owned(),
            "@ID:\teng|test|CHI|||||Target_Child|||".to_owned(),
            format!("*INV:\thello there . {}", bullet(1000, 2000)),
            format!("*CHI:\thi . {}", bullet(2000, 3000)),
            "*CHI:\tno bullet here .".to_owned(),
            "@End".to_owned(),
            String::new(),
        ]
        .join("\n")
    }

    fn parsed() -> ChatFile {
        let parser = crate::chat_parser();
        let (chat_file, _) = batchalign_transform::parse::parse_lenient(&parser, &sample());
        chat_file
    }

    /// Bullets are read from the typed model, in file milliseconds.
    #[test]
    fn timed_utterances_carry_their_bullet_in_file_milliseconds() {
        let read = read_utterances(&parsed(), &TierSelection::AllTiers);
        assert_eq!(read.len(), 3);
        assert_eq!(read[0].speaker, "INV");
        assert_eq!(
            read[0].timing,
            UtteranceTiming::Window(
                MediaWindow::new(FileMs::new(1000), FileMs::new(2000))
                    .expect("test: a non-empty media window")
            )
        );
    }

    /// An untimed utterance is COLLECTED, not skipped, so the evidence can
    /// report why it was not scored instead of leaving a gap a reader would
    /// have to notice by counting.
    #[test]
    fn an_untimed_utterance_is_collected_with_no_bullet() {
        let read = read_utterances(&parsed(), &TierSelection::AllTiers);
        assert_eq!(read[2].timing, UtteranceTiming::Untimed);
        assert_eq!(read[2].speaker, "CHI");
    }

    /// A named selection scores only those tiers, and re-indexes so the
    /// evidence's `utterance_index` counts what it actually reports.
    #[test]
    fn a_named_selection_keeps_only_those_tiers() {
        let read = read_utterances(&parsed(), &TierSelection::Named(vec!["CHI".to_owned()]));
        assert_eq!(read.len(), 2);
        assert!(read.iter().all(|utterance| utterance.speaker == "CHI"));
        assert_eq!(read[0].utterance_index, 0);
        assert_eq!(read[1].utterance_index, 1);
    }

    /// The leading `*` is optional, because an operator reading a transcript
    /// sees `*CHI:` and will type what they see.
    #[test]
    fn a_tier_may_be_written_with_or_without_its_asterisk() {
        assert_eq!(
            TierSelection::from_option(&["*CHI".to_owned()]),
            TierSelection::Named(vec!["CHI".to_owned()])
        );
    }

    /// Empty means every tier, decided once here rather than remembered by
    /// every later reader of the list.
    #[test]
    fn an_empty_selection_means_every_tier() {
        assert_eq!(TierSelection::from_option(&[]), TierSelection::AllTiers);
        assert_eq!(TierSelection::AllTiers.recorded(), ["*"]);
    }

    /// A line number that sends a reviewer to the right line of the file.
    #[test]
    fn the_recorded_line_is_one_based_and_points_at_the_main_tier() {
        let read = read_utterances(&parsed(), &TierSelection::AllTiers);
        let text = sample();
        let Some(line) = text.lines().nth(read[0].line - 1) else {
            panic!("line {} is in the sample", read[0].line);
        };
        assert!(line.starts_with("*INV:"), "got {line:?}");
    }
}
