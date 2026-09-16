//! Build a CHAT file from a structured transcript description.
//!
//! This module constructs a [`ChatFile`] AST from structured input, either
//! a JSON transcript description (for PyO3 bridge compatibility) or typed
//! Rust structs (for the Rust server's transcribe orchestrator).
//!
//! The implementation is split by responsibility so contributors can find the
//! schema bridge, parser setup, header synthesis, and utterance assembly logic
//! without paging through the full end-to-end pipeline in one file.
//!
//! # Two entry points
//!
//! - [`build_chat`]: takes a typed [`TranscriptDescription`] struct
//! - [`build_chat_from_json`]: deserializes JSON into `TranscriptDescription`,
//!   then calls `build_chat`. Used by the PyO3 bridge to delegate here.
//!
//! # Convenience
//!
//! - [`transcript_from_asr_utterances`]: converts post-processed ASR
//!   utterances into a `TranscriptDescription` for CHAT assembly.

mod bridge;
mod headers;
mod parser;
mod schema;
#[cfg(test)]
mod tests;
mod utterances;

use talkbank_model::model::{ChatFile, Header, Line};

pub use bridge::{
    AsrDiagnosticError, AsrTranscript, LanguageInvalidWord, NamedAsrUtterances,
    TranscriptBuildError, build_chat_from_json, transcript_from_asr_utterances,
};
pub use schema::{ParticipantDesc, TranscriptDescription, UtteranceDesc, WordDesc};
pub use utterances::tag_marker_separator;

use headers::{build_header_lines, build_participant_map};
use parser::BuildChatContext;
use utterances::build_utterance_lines;

/// Why building a CHAT file from a [`TranscriptDescription`] failed.
///
/// These five modes were previously flattened into `String`, which is the
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

    /// A language code in `@Languages` or on an utterance is not valid.
    #[error("invalid language code {code:?}: {source}")]
    LanguageCode {
        /// The rejected code.
        code: String,
        /// Why the model rejected it.
        source: talkbank_model::model::LanguageCodeError,
    },

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

/// Build a CHAT file from a typed transcript description.
pub fn build_chat(desc: &TranscriptDescription) -> Result<ChatFile, BuildChatError> {
    if desc.participants.is_empty() {
        return Err(BuildChatError::NoParticipants);
    }

    let context = BuildChatContext::new(desc)?;
    // Headers first, so a document with a bad language code is still refused
    // for its language code rather than for the emptiness below.
    let mut lines = build_header_lines(desc, context.langs())?;
    let utterances = build_utterance_lines(
        desc,
        context.parser(),
        context.langs(),
        context.primary_lang(),
    )?;
    // The built LINES are what decides, not the described utterances: a
    // description can carry words that all reduce to nothing, and it is the
    // absence of an utterance line that makes the output an empty transcript.
    if utterances.is_empty() {
        return Err(if desc.utterances.is_empty() {
            BuildChatError::NoUtterances
        } else {
            BuildChatError::NoUtteranceSurvivedBuild {
                described: desc.utterances.len(),
            }
        });
    }
    lines.extend(utterances);
    lines.push(Line::header(Header::End));

    // Programmatically assembled CHAT carries the same typed participant
    // metadata as parsed CHAT; `ChatFile::new`'s empty map is for parser
    // intermediates only (see `build_participant_map`).
    let participants = build_participant_map(desc, context.langs());
    Ok(ChatFile::with_participants(lines, participants))
}
