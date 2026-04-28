#![warn(missing_docs)]
//! Shared Rev.AI integration for batchalign3.
//!
//! This crate exists so the Rust server control plane and the PyO3 extension
//! can share one typed HTTP client instead of routing non-model work through
//! Python worker processes. The long-term architectural goal is that Python
//! remains only the host for model SDK calls that cannot run elsewhere.
//!
//! The crate intentionally stays small:
//! - [`client`] owns the blocking HTTP client and transcript post-download flow
//! - [`types`] owns the typed Rev.AI request/response records
//!
//! The client is blocking by design because both current callers already have a
//! natural boundary for it:
//! - the PyO3 layer releases the GIL around each call
//! - the Rust server uses `tokio::task::spawn_blocking` for upload bursts

pub mod client;
pub mod types;

pub use client::{Result, RevAiClient, RevAiError, TranscriptResult, extract_timed_words};
pub use types::{
    Element, Job, JobStatus, LangIdResult, LanguageConfidence, Monologue, SubmitOptions, TimedWord,
    Transcript,
};
