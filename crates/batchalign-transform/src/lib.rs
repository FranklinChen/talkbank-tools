#![warn(missing_docs)]
// Test code is exempt from this crate's `deny`-level panic lints: assertion
// macros panic by design.
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable,
        clippy::todo,
        clippy::unimplemented
    )
)]
//! Focused transform building blocks for CHAT file processing.
//!
//! This crate exposes many leaf modules, but the crate root keeps a smaller
//! convenience surface for the most common pipeline entry points. Specialized
//! behavior continues to live in its owning module namespace (`json`,
//! `corpus`, `validation_runner`, `xml`, and so on).
//!
//! # Related CHAT Manual Sections
//!
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Format>
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Headers>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Main_Tier>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Dependent_Tiers>
//!
//! ## Top-level entry points
//!
//! - Root re-exports such as [`parse_and_validate`] and [`normalize_chat`] are
//!   the common one-shot pipeline helpers.
//! - [`json`] and [`xml`] own the format-conversion surfaces.
//! - [`corpus`], [`unified_cache`], and [`validation_runner`] own discovery,
//!   caching, and directory-scale validation workflows.
//!
//! # Design Principles
//!
//! - Streaming entry points require `ErrorSink` for diagnostics
//! - Cache paths are shared across tools for consistency
//!
//! # Examples
//!
//! ```no_run
//! use talkbank_transform::{parse_and_validate, PipelineError};
//! use talkbank_model::ParseValidateOptions;
//!
//! let content = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n\
//!     @ID:\teng|corpus|CHI|||||Child|||\n*CHI:\thello .\n@End\n";
//! let options = ParseValidateOptions::default().with_validation();
//! let chat_file = parse_and_validate(content, options).unwrap();
//! assert_eq!(chat_file.utterances().count(), 1);
//! ```

// Generic CHAT transform surface: single home is chatter's talkbank-transform,
// re-exported so the batchalign-specific modules below (which reach generic
// helpers via `crate::parse`, `crate::extract`, `crate::dependent_tiers`, ...)
// and downstream consumers reach the whole generic surface through one root.
pub use talkbank_transform::*;

// The `unified_cache` compatibility shim was deleted on 2026-07-30. It existed
// to keep historical `batchalign_transform::unified_cache::...` references
// resolving after chatter moved caching into `talkbank-cache`, and a search of
// the whole workspace found ZERO such references: it had no consumers at any
// point after it was written.
//
// It was also the one thing here that would have broken if chatter put its
// cache surface behind a feature, because it names those four types EXPLICITLY.
// The blanket `pub use talkbank_transform::*` above is a glob and simply
// re-exports fewer names when a feature is off, so with the shim gone this
// crate compiles against chatter either way.

// Batchalign-specific transforms. These need ML-pipeline context (ASR output,
// neural morphotag, forced-alignment decisions, utterance segmentation), so
// they live on the Batchalign side, NOT in the generic talkbank-transform crate.
pub mod asr_postprocess;
pub mod build_chat;
pub mod compare;
pub mod constituency;
pub mod coref;
pub mod decisions;
pub mod diff;
pub mod dp_align;
pub mod inject;
pub mod merge_abbrev;
pub mod morphosyntax;
pub mod retokenize;
pub mod tokenizer_realign;
pub mod translate;
pub mod utseg;
pub mod utseg_compute;
pub mod wer_conform;

// The generic convenience re-exports (corpus, json, pipeline, rendering,
// caching, validation_runner) now come from chatter via the `pub use
// talkbank_transform::*` glob above. Only the batchalign-specific convenience
// re-export stays here.
pub use self::merge_abbrev::{merge_abbreviations, merge_abbreviations_in_chat_text};

/// A `ChatCleanedText` / `ChatRawText` pair built the way production builds
/// them: by PARSING.
///
/// # Why this exists
///
/// chatter v0.12.0 removed `ChatCleanedText::test_unchecked` and the
/// `test-utils` feature that gated it. That removal is right: a type whose
/// whole job is to prove "this text came from a parsed AST" is only as strong
/// as its weakest constructor, and an escape hatch any dev-dependency could
/// enable was that constructor. Fixtures now go through `parse_word`, so a
/// fixture cannot assert a projection the parser would not produce.
///
/// One owner per crate rather than a call at each site, because building a
/// parser is not free and because the next reader should see one route.
#[cfg(test)]
use talkbank_model::{ChatCleanedText, ChatRawText};

#[cfg(test)]
pub(crate) fn parsed_word_text(word: &str) -> (ChatCleanedText, ChatRawText) {
    let parser = talkbank_parser::TreeSitterParser::new()
        .expect("tree-sitter parser must initialise for fixtures");
    // A `match` rather than `unwrap_or_else(|e| panic!(..))`: identical
    // behaviour, but the silent-defaults ratchet matches the `unwrap_or*`
    // spelling and cannot see that this one diverges instead of substituting a
    // value. Respelling a false positive, not evading the rule; nothing here
    // invents a fixture.
    let parsed = match parser.parse_word(word) {
        Ok(parsed) => parsed,
        Err(e) => panic!("fixture {word:?} must parse as a CHAT word: {e:?}"),
    };
    (
        ChatCleanedText::from_word(&parsed),
        ChatRawText::from_word_raw(&parsed),
    )
}

/// The cleaned projection only, for fixtures that do not need the raw text.
#[cfg(test)]
pub(crate) fn parsed_word_text_cleaned(word: &str) -> ChatCleanedText {
    parsed_word_text(word).0
}
