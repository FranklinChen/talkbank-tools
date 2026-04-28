#![warn(missing_docs)]
//! Pipeline functions for CHAT file processing.
//!
//! This module provides reusable pipeline functions that compose parsing,
//! validation, and transformation operations using the talkbank-parser
//! and talkbank-model libraries.
//!
//! # Related CHAT Manual Sections
//!
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Format>
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Headers>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Main_Tier>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Dependent_Tiers>
//!
//! ## Module Structure
//!
//! - **`pipeline`** - Core parse + validate + transform functions
//! - **`corpus`** - Corpus discovery and manifest operations
//! - **`unified_cache`** - SQLite-based caching for validation/roundtrip (pass/fail only)
//! - **`validation_runner`** - Parallel directory validation with streaming
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

// JSON serialization with schema validation (merged from talkbank-json)
pub mod json;
pub use json::*;

pub mod asr_postprocess;
pub mod benchmark;
pub mod build_chat;
pub mod compare;
pub mod constituency;
pub mod coref;
pub mod decisions;
pub mod dependent_tiers;
pub mod diff;
pub mod dp_align;
pub mod extract;
pub mod inject;
pub mod merge_abbrev;
pub mod morphosyntax;
pub mod parse;
pub mod retokenize;
pub mod serialize;
pub mod tokenizer_realign;
pub mod translate;
pub mod utseg;
pub mod utseg_compute;
pub mod validate;
pub mod wer_conform;
pub use merge_abbrev::merge_abbreviations;

// CHAT → TalkBank XML emission (Rust replacement for Java Chatter's XML output).
pub mod xml;

mod pipeline;
pub use pipeline::{
    PipelineError, chat_to_json, chat_to_json_unvalidated, normalize_chat, parse_and_validate,
    parse_and_validate_streaming, parse_and_validate_streaming_with_parser,
    parse_and_validate_with_parser, parse_file_and_validate,
};

// Internal lock helpers for poison recovery
mod lock_helpers;

// Miette error rendering (standalone, not tied to cache)
mod rendering;
pub use rendering::{
    render_error_with_miette, render_error_with_miette_with_named_source,
    render_error_with_miette_with_source, render_error_with_miette_with_source_colored,
};

// Unified cache infrastructure (shared by validation and roundtrip)
pub mod unified_cache;
pub use unified_cache::{CachePool, CacheStats, UnifiedCache};

// Corpus operations (discovery + manifest)
pub mod corpus;
pub use corpus::{
    CorpusEntry, CorpusManifest, FailureReason, FileEntry, FileStatus as CorpusFileStatus,
    ManifestError, build_manifest, corpus_summary, discover_corpora, format_manifest,
};

// Shared validation runner for CLI and GUI
pub mod validation_runner;
pub use validation_runner::{
    CacheMode, CacheOutcome, DirectoryMode, ErrorEvent, FileCompleteEvent, FileStatus, ParserKind,
    RoundtripEvent, ValidationCache, ValidationConfig, ValidationEvent, ValidationStats,
    ValidationStatsSnapshot, validate_directory_streaming,
};
