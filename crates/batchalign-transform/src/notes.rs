//! `%xbatchalign` notes: BA3's mark in the CHAT at an utterance it could not
//! fully process.
//!
//! A localized problem never fails a whole file. The utterance it touches is
//! left as written (no `%mor`, no timing, whatever the command could not
//! produce), the rest of the file is processed, and two things say where and
//! why: a warning in the job record, and a `%xbatchalign` tier on that
//! utterance, so a reader of the CHAT finds the spot without the job record.
//!
//! ```text
//! *CHI:    hello good world .
//! %xbatchalign:    morphotag: not analysed; the worker could not analyse this utterance (...)
//! ```
//!
//! (A tab, as in every CHAT tier, follows each colon.) One tier label for every command; the text starts with the command that
//! wrote it ([`NoteCommand`]). A command replaces its own earlier notes on
//! every run, so a note is never stale, and leaves other commands' notes
//! alone ([`replace_notes`] is the only writer). Distinct from the legacy
//! `%xalign` / `%xrev` decision tiers, which are provenance and are stripped
//! (see [`crate::decisions::strip_decision_tiers`]).

use talkbank_model::model::{
    ChatFile, DependentTier, DependentTierEntry, InvalidUserDefinedDependentTierLabel, Line,
    RecoveryText, UserDefinedDependentTier, UserDefinedDependentTierLabel,
};

use talkbank_model::UtteranceIdx;

/// The tier label, without `%` or colon.
pub const NOTE_TIER: &str = "xbatchalign";

/// The longest note text written, in characters; a longer reason is cut with
/// an ellipsis. The job record carries the bounded detail too.
const MAX_NOTE_CHARS: usize = 500;

/// The command a note belongs to: its text starts with this command's name,
/// and only this command replaces it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoteCommand {
    /// `morphotag`.
    Morphotag,
    /// `align`.
    Align,
}

impl NoteCommand {
    /// The name the note's text starts with.
    pub fn name(self) -> &'static str {
        match self {
            Self::Morphotag => "morphotag",
            Self::Align => "align",
        }
    }

    /// Whether `text` is a note this command wrote.
    fn wrote(self, text: &str) -> bool {
        text.strip_prefix(self.name())
            .is_some_and(|rest| rest.starts_with(':'))
    }
}

/// One note for one utterance: what happened there, in plain words. The
/// command's name is added when it is written.
///
/// The utterance is named by its position among the file's utterances, never
/// by a line index: a command inserts header lines (provenance stamps)
/// between finding an utterance and writing its note, which moves every
/// line after them, while no header edit changes which utterance is the nth.
/// [`replace_notes`] is the one place that turns the position into a line,
/// against the document as it is when the note is written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UtteranceNote {
    utterance: UtteranceIdx,
    text: String,
}

impl UtteranceNote {
    /// A note on the `utterance`th utterance of the file (from 0).
    pub fn new(utterance: UtteranceIdx, text: impl Into<String>) -> Self {
        Self {
            utterance,
            text: text.into(),
        }
    }
}

/// A note that could not be written.
#[derive(Debug, thiserror::Error)]
pub enum NoteError {
    /// The note names an utterance the file does not have.
    #[error("%{NOTE_TIER} note for utterance {utterance}, past the file's last utterance")]
    NoSuchUtterance {
        /// The position named, from 0.
        utterance: usize,
    },
    /// [`NOTE_TIER`] is not a label Chatter admits (a defect in this module,
    /// which the boundary test below rules out).
    #[error("the note tier label is not a custom tier label: {0}")]
    Label(#[from] InvalidUserDefinedDependentTierLabel),
}

/// Remove every note `command` wrote before, then write `notes`: the one
/// writer of `%xbatchalign`. Called on every run of the command, including a
/// run with no notes, which only clears the stale ones.
///
/// Every note is checked and built before the document is touched, so a
/// refused note leaves the file exactly as it was, never half rewritten.
pub fn replace_notes(
    chat: &mut ChatFile,
    command: NoteCommand,
    notes: impl IntoIterator<Item = UtteranceNote>,
) -> Result<(), NoteError> {
    let label = UserDefinedDependentTierLabel::new(NOTE_TIER)?;
    let utterance_count = chat
        .lines
        .iter()
        .filter(|line| matches!(line, Line::Utterance(_)))
        .count();
    // Each note's tier, by utterance position, in the order given.
    let mut written: Vec<(usize, DependentTierEntry)> = Vec::new();
    for note in notes {
        let position = note.utterance.raw();
        if position >= utterance_count {
            return Err(NoteError::NoSuchUtterance {
                utterance: position,
            });
        }
        written.push((
            position,
            DependentTierEntry::new(note_tier(label.clone(), command, &note.text)),
        ));
    }
    // One pass over the utterances, in order: clear this command's notes,
    // then write the ones for this position (stable sort keeps their order).
    written.sort_by_key(|(position, _)| *position);
    let mut pending = written.into_iter().peekable();
    let utterances = chat
        .lines
        .as_mut_slice()
        .iter_mut()
        .filter_map(|line| match line {
            Line::Utterance(utterance) => Some(utterance),
            _ => None,
        });
    for (position, utterance) in utterances.enumerate() {
        utterance
            .dependent_tiers
            .retain(|entry| !is_note_of(entry, command));
        while let Some((_, entry)) = pending.next_if(|(at, _)| *at == position) {
            utterance.dependent_tiers.push(entry);
        }
    }
    Ok(())
}

/// Whether `entry` is a `%xbatchalign` note `command` wrote.
fn is_note_of(entry: &DependentTierEntry, command: NoteCommand) -> bool {
    let DependentTier::UserDefined(tier) = &entry.tier else {
        return false;
    };
    tier.label.as_str() == NOTE_TIER
        && tier
            .content
            .as_ref()
            .is_some_and(|content| command.wrote(content.as_str()))
}

/// The tier for one note: `command: text`, on one line, bounded.
fn note_tier(
    label: UserDefinedDependentTierLabel,
    command: NoteCommand,
    text: &str,
) -> DependentTier {
    // One physical line: any line break or run of whitespace in a producer's
    // reason becomes one space.
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut body = format!("{}: {flat}", command.name());
    if let Some((cut, _)) = body.char_indices().nth(MAX_NOTE_CHARS) {
        body.truncate(cut);
        body.push('…');
    }
    DependentTier::UserDefined(UserDefinedDependentTier {
        label,
        content: Some(RecoveryText::observed(&body)),
        location: talkbank_model::Location::Unlocated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use talkbank_model::WriteChat;
    use talkbank_parser::TreeSitterParser;
    use talkbank_parser_tests::test_error::ParseProductTestExt;

    fn parse(text: &str) -> ChatFile {
        TreeSitterParser::new()
            .expect("parser")
            .parse_chat_file(text)
            .expect_built()
    }

    /// The file's first utterance.
    fn first() -> UtteranceIdx {
        UtteranceIdx::new(0)
    }

    /// The one boundary: a note is written as a `%xbatchalign` tier that
    /// names its command, a rerun replaces only that command's notes, and
    /// another command's note survives.
    #[test]
    fn a_rerun_replaces_its_own_notes_and_keeps_others() {
        let mut chat = parse(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
             @ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n@End\n",
        );
        replace_notes(
            &mut chat,
            NoteCommand::Align,
            [UtteranceNote::new(first(), "not timed")],
        )
        .expect("align note");
        replace_notes(
            &mut chat,
            NoteCommand::Morphotag,
            [UtteranceNote::new(first(), "not analysed;\nworker\tfailed")],
        )
        .expect("first morphotag note");
        let text = chat.to_chat_string();
        assert!(text.contains("%xbatchalign:\talign: not timed\n"), "{text}");
        assert!(
            text.contains("%xbatchalign:\tmorphotag: not analysed; worker failed\n"),
            "{text}"
        );

        // The next morphotag run has nothing to note: its old note goes, the
        // align note stays.
        replace_notes(&mut chat, NoteCommand::Morphotag, []).expect("rerun");
        let text = chat.to_chat_string();
        assert!(!text.contains("morphotag:"), "{text}");
        assert!(text.contains("%xbatchalign:\talign: not timed\n"), "{text}");
    }

    /// A rerun on the WRITTEN file: the note is reparsed from text, as a
    /// second morphotag run sees it, and is still recognized as morphotag's
    /// own, so the rerun leaves exactly one note, not two.
    #[test]
    fn a_rerun_on_the_reparsed_file_replaces_its_note() {
        let mut chat = parse(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
             @ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n@End\n",
        );
        replace_notes(
            &mut chat,
            NoteCommand::Morphotag,
            [UtteranceNote::new(first(), "not analysed; first run")],
        )
        .expect("first run");
        let mut reparsed = parse(&chat.to_chat_string());
        replace_notes(
            &mut reparsed,
            NoteCommand::Morphotag,
            [UtteranceNote::new(first(), "not analysed; second run")],
        )
        .expect("second run");
        let text = reparsed.to_chat_string();
        assert_eq!(
            text.matches("%xbatchalign:\tmorphotag:").count(),
            1,
            "{text}"
        );
        assert!(text.contains("second run"), "{text}");
        assert!(!text.contains("first run"), "{text}");
    }

    /// A refused note leaves the document as it was: the earlier note of
    /// the same command is not cleared, and no note is half written.
    #[test]
    fn a_refused_note_leaves_the_file_untouched() {
        let mut chat = parse(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
             @ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello .\n@End\n",
        );
        replace_notes(
            &mut chat,
            NoteCommand::Morphotag,
            [UtteranceNote::new(first(), "earlier")],
        )
        .expect("earlier run");
        let before = chat.to_chat_string();
        let refused = replace_notes(
            &mut chat,
            NoteCommand::Morphotag,
            [
                UtteranceNote::new(first(), "new"),
                UtteranceNote::new(UtteranceIdx::new(3), "out of range"),
            ],
        );
        assert!(matches!(
            refused,
            Err(NoteError::NoSuchUtterance { utterance: 3 })
        ));
        assert_eq!(chat.to_chat_string(), before);
    }

    #[test]
    fn a_note_past_the_last_utterance_is_refused() {
        let mut chat = parse(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
             @ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello .\n@End\n",
        );
        let refused = replace_notes(
            &mut chat,
            NoteCommand::Morphotag,
            [UtteranceNote::new(UtteranceIdx::new(1), "x")],
        );
        assert!(matches!(
            refused,
            Err(NoteError::NoSuchUtterance { utterance: 1 })
        ));
    }
}
