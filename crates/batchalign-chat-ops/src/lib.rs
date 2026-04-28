#![warn(missing_docs)]
//! CHAT format operations for the batchalign processing pipeline.
//!
//! This crate now contains the Batchalign-specific CHAT orchestration layer:
//! forced alignment, speaker reassignment, cache/runtime glue, and
//! compatibility facades over the shared deterministic CHAT/text logic that
//! has been promoted into `talkbank-transform`. It was originally extracted
//! from the PyO3 bridge (`batchalign-core`) so that both the PyO3 layer and
//! the standalone Rust server (`batchalign-server`) could share the same CHAT
//! manipulation code without duplication.
//!
//! # Design principle
//!
//! **No text hacking.** Every CHAT transformation goes through the typed
//! [`ChatFile`] AST from `talkbank-model`. This crate provides the
//! extract-modify-inject round-trip pattern that keeps CHAT serialization
//! correct even in the face of complex escaping, continuation lines,
//! multi-word tokens, and dependent tier alignment.
//!
//! # Parse, extract, modify, inject round-trip
//!
//! The fundamental workflow shared by all NLP tasks is:
//!
//! ```text
//!   CHAT text
//!       |  parse::parse_lenient()
//!       v
//!   ChatFile AST
//!       |  extract / collect_payloads()
//!       v
//!   NLP payloads (words, cache keys, positions)
//!       |  send to Python worker (batch_infer IPC)
//!       v
//!   NLP results (UdResponse, UtsegResponse, ...)
//!       |  inject / apply_*_results()
//!       v
//!   Modified ChatFile AST
//!       |  serialize::to_chat_string()
//!       v
//!   CHAT text (round-tripped)
//! ```
//!
//! Today, many of those task modules are thin re-export shims over
//! `talkbank-transform`; the remaining Batchalign-owned implementation is
//! concentrated in FA, speaker reassignment, and cache/runtime seams.
//!
//! # Example: morphosyntax round-trip
//!
//! ```rust,no_run
//! use batchalign_chat_ops::parse::{TreeSitterParser, parse_lenient};
//! use batchalign_chat_ops::morphosyntax::{
//!     collect_payloads, clear_morphosyntax,
//! };
//! use batchalign_chat_ops::serialize::to_chat_string;
//! use batchalign_chat_ops::LanguageCode;
//! use batchalign_chat_ops::morphosyntax::MultilingualPolicy;
//!
//! // 1. Parse CHAT text into an AST
//! let parser = TreeSitterParser::new().unwrap();
//! let chat_text = std::fs::read_to_string("example.cha").unwrap();
//! let (mut chat_file, _errors) = parse_lenient(&parser, &chat_text);
//!
//! // 2. Clear any existing %mor/%gra tiers
//! clear_morphosyntax(&mut chat_file);
//!
//! // 3. Extract NLP payloads (words + positions) for the worker
//! let primary = LanguageCode::new("eng");
//! let declared = vec![primary.clone()];
//! let collected = collect_payloads(
//!     &chat_file, &primary, &declared, MultilingualPolicy::ProcessAll,
//! );
//! let payloads = collected.batch_items;
//! // payloads: Vec<BatchItemWithPosition>
//! // Each item contains: words, terminator, special_forms, lang
//! // collected.not_applicable carries MorOutcomes for utterances that
//! // had zero Mor-alignable content (filler-only, empty, etc.)
//!
//! // 4. Send payloads to Python worker via batch_infer IPC,
//! //    receive Vec<UdResponse> back...
//! //    (server orchestrator handles this step)
//!
//! // 5. Inject NLP results back into the AST
//! // inject_results(&mut chat_file, &payloads, &ud_responses, retokenize);
//!
//! // 6. Serialize back to CHAT
//! let output = to_chat_string(&chat_file);
//! ```
//!
//! # Module map
//!
//! ## Parsing and serialization
//!
//! | Module         | Responsibility                                                    |
//! |----------------|-------------------------------------------------------------------|
//! | [`parse`]      | Lenient and strict CHAT parsing wrappers over tree-sitter         |
//! | [`serialize`]  | CHAT serialization (AST back to `.cha` text)                      |
//!
//! ## Word extraction and injection
//!
//! | Module         | Responsibility                                                    |
//! |----------------|-------------------------------------------------------------------|
//! | [`extract`]    | Compatibility facade over canonical AST word extraction in `talkbank-transform` |
//! | [`inject`]     | Compatibility facade over canonical tier injection helpers        |
//! | [`retokenize`] | Compatibility facade over canonical retokenization helpers       |
//!
//! ## NLP task modules (server-side orchestration payloads)
//!
//! | Module         | Responsibility                                                    |
//! |----------------|-------------------------------------------------------------------|
//! | [`morphosyntax`]| Compatibility surface over canonical `%mor`/`%gra` logic plus Batchalign runtime call sites |
//! | [`utseg`]      | Compatibility facade over canonical utterance-segmentation helpers |
//! | [`translate`]  | Compatibility facade over canonical translation helpers          |
//! | [`coref`]      | Compatibility facade over canonical coreference helpers          |
//! | [`fa`]         | Forced alignment: utterance grouping, DP alignment, timing injection, monotonicity |
//!
//! ## NLP support
//!
//! | Module         | Responsibility                                                    |
//! |----------------|-------------------------------------------------------------------|
//! | [`nlp`]        | Compatibility surface over canonical UD-to-CHAT mapping helpers  |
//! | [`dp_align`]   | Compatibility facade over canonical Hirschberg alignment         |
//! | [`tokenizer_realign`] | Compatibility facade over canonical tokenizer realignment |
//! | [`text_types`] | Compatibility re-exports of canonical provenance newtypes        |
//!
//! ## Evaluation
//!
//! | Module         | Responsibility                                                    |
//! |----------------|-------------------------------------------------------------------|
//! | [`wer_conform`]| Word normalization for WER benchmark comparison                   |
//! | [`benchmark`]  | Full WER computation: normalize, align, count errors, diff        |

pub mod asr_postprocess;
pub mod benchmark;
pub mod build_chat;
pub mod cache_key;
pub mod compare;
pub mod constituency;
pub mod coref;
pub mod decisions;
pub mod diff;
pub mod dp_align;
pub mod extract;
pub mod fa;
pub mod indices;
pub mod inject;
pub mod merge_abbrev;
pub mod morphosyntax;
pub mod nlp;
pub mod parse;
pub mod retokenize;
pub mod serialize;
pub mod speaker;
pub mod text_types;
pub mod tokenizer_realign;
pub mod translate;
pub mod utseg;
pub mod utseg_compute;
pub mod validate;
pub mod wer_conform;

// Re-export newtypes used by all NLP task modules and the server orchestrators.
pub use cache_key::{CacheKey, CacheTaskName};

// Re-export talkbank_model types commonly needed by downstream crates
// (e.g. batchalign-server) that shouldn't depend on talkbank_model directly.
pub use talkbank_model::ParseError;
pub use talkbank_model::Span;
pub use talkbank_model::alignment::helpers::TierDomain;
pub use talkbank_model::header::Header;
pub use talkbank_model::model::BulletContent;
pub use talkbank_model::model::{
    ChatFile, DependentTier, LanguageCode, Line, Linker, UserDefinedDependentTier, Utterance,
};
