//! Build a CHAT file from a structured transcript description.
//!
//! This module constructs a [`ChatFile`] AST from a typed
//! [`TranscriptDescription`] (the Rust server's transcribe orchestrator
//! builds one from ASR utterances).
//!
//! The implementation is split by responsibility so contributors can find the
//! ASR bridge, header synthesis, and utterance assembly logic without paging
//! through the full end-to-end pipeline in one file.
//!
//! # Entry point
//!
//! - [`build_chat`]: takes a typed [`TranscriptDescription`]. Its languages
//!   ([`TranscriptLanguages`]) and words (`ChatWordText`) were parsed where
//!   they were built, so assembly parses neither again.
//!
//! # Convenience
//!
//! - [`transcript_from_asr_utterances`]: converts post-processed ASR
//!   utterances into a `TranscriptDescription` for CHAT assembly.

mod bridge;
mod headers;
mod schema;
#[cfg(test)]
mod tests;
mod utterances;

use talkbank_model::model::{ChatFile, Header, Line};

pub use bridge::{
    AsrDiagnosticError, AsrTranscript, LanguageInvalidWord, NamedAsrUtterances,
    TranscriptBuildError, transcript_from_asr_utterances,
};
pub use schema::{
    DescribedTiming, ParticipantDesc, TranscriptDescription, TranscriptLanguages, UtteranceBody,
    UtteranceDesc, WordDesc,
};
pub use utterances::tag_marker_separator;

use headers::{build_header_lines, build_participant_map};
use utterances::build_utterance_lines;

/// Why building a CHAT file from a [`TranscriptDescription`] failed.
///
/// These failure modes were previously flattened into `String`, which is the
/// primitive obsession rule 6 bans: a caller could not tell an unusable media
/// name (operator input, fixable) from a parser init failure (environment,
/// not fixable) without matching on prose. `chatter`'s equivalent builder has
/// carried a typed error since v0.9.0; this is the same shape.
#[derive(Debug, thiserror::Error)]
pub enum BuildChatError {
    /// CHAT requires at least one participant.
    #[error("at least one participant is required")]
    NoParticipants,

    /// The tree-sitter parser could not be created.
    #[error("failed to create parser: {0}")]
    ParserInit(String),

    /// An utterance body could not be parsed as CHAT.
    #[error("failed to parse utterance for speaker {speaker}: {message}")]
    Utterance {
        /// The speaker whose line failed.
        speaker: String,
        /// The parser's account.
        message: String,
    },

    /// The supplied media name cannot be written to an `@Media` header and
    /// read back unchanged.
    #[error("invalid @Media filename: {0}")]
    MediaFilename(#[from] talkbank_model::model::MediaFilenameError),

    /// The description carried no utterances at all.
    ///
    /// A CHAT document of headers and nothing else is not a transcript of
    /// anything, and it is indistinguishable from a transcript of a silent
    /// recording. Whatever produced no utterances has to say so; assembling
    /// the empty sequence into a plausible-looking file is how a run that
    /// recognized nothing was delivered as a success.
    #[error("a transcript needs at least one utterance, and this description carried none")]
    NoUtterances,

    /// Every described utterance reduced to no CHAT content.
    ///
    /// Distinct from [`NoUtterances`] because the emptiness was produced
    /// HERE rather than upstream: words arrived and none of them was content.
    /// The shape that does this is a transcript whose every token is a
    /// terminator or a separator, which is what a Cantonese engine returns
    /// when it recognizes only punctuation.
    ///
    /// [`NoUtterances`]: BuildChatError::NoUtterances
    #[error(
        "none of the {described} described utterance(s) produced any CHAT content: \
         every one held only terminators, separators or empty text"
    )]
    NoUtteranceSurvivedBuild {
        /// How many utterances the description carried.
        described: usize,
    },
}

/// A built CHAT file, with every utterance whose terminator the description's
/// producer synthesized: CHAT has no notation for a terminator nobody heard,
/// so this list is where the difference survives.
///
/// `#[must_use]`: a caller that wants only the file says so, in a line a
/// reader can see ([`Self::into_file`]).
#[must_use]
#[derive(Debug)]
pub struct BuiltChat {
    /// The CHAT file.
    pub file: ChatFile,
    /// The utterances written with a synthesized terminator, in order.
    pub synthesized_ends: Vec<SynthesizedEnd>,
}

impl BuiltChat {
    /// The file alone, the synthesized ends deliberately set aside.
    pub fn into_file(self) -> ChatFile {
        self.file
    }
}

/// An utterance written with a terminator this pipeline added, not one its
/// source observed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynthesizedEnd {
    /// Its position among the file's utterances, counting from 1.
    pub utterance: usize,
    /// Its speaker code (e.g. `"PAR1"`).
    pub speaker: String,
}

/// Build a CHAT file from a typed transcript description: the last
/// transition of transcript assembly. ASR utterances are named
/// (`NamedAsrUtterances`), admitted into a description whose languages and
/// words are parsed (`into_transcript`), and the description is consumed
/// here into the built file, so a description is built from once and never
/// re-checked.
pub fn build_chat(desc: TranscriptDescription) -> Result<BuiltChat, BuildChatError> {
    let desc = &desc;
    // An empty transcript before participants: a description with no
    // utterance also names no speaker (the ASR bridge lists only speakers it
    // observed), and "no participants" would hide what produced nothing.
    // The languages were parsed where the description was built.
    let parser = talkbank_parser::TreeSitterParser::new()
        .map_err(|e| BuildChatError::ParserInit(e.to_string()))?;
    if desc.utterances.is_empty() {
        return Err(BuildChatError::NoUtterances);
    }
    if desc.participants.is_empty() {
        return Err(BuildChatError::NoParticipants);
    }
    let mut lines = build_header_lines(desc)?;
    let utterances::BuiltUtterances {
        lines: utterances,
        synthesized_ends,
    } = build_utterance_lines(desc, &parser)?;
    // The built LINES are what decides, not the described utterances: a
    // description can carry words that all reduce to nothing, and it is the
    // absence of an utterance line that makes the output an empty transcript.
    if utterances.is_empty() {
        return Err(BuildChatError::NoUtteranceSurvivedBuild {
            described: desc.utterances.len(),
        });
    }
    lines.extend(utterances);
    lines.push(Line::header(
        talkbank_model::Location::Unlocated,
        Header::End,
    ));

    // Programmatically assembled CHAT carries the same typed participant
    // metadata as parsed CHAT; `ChatFile::new`'s empty map is for parser
    // intermediates only (see `build_participant_map`).
    let participants = build_participant_map(desc);
    Ok(BuiltChat {
        file: ChatFile::with_participants(lines, participants),
        synthesized_ends,
    })
}
