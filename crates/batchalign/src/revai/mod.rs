//! Rust-owned Rev.AI integration for the server control plane.
//!
//! This module is where batchalign-app keeps Rev.AI work that does not belong
//! in Python:
//! - loading credentials for server-owned operations
//! - direct server-owned Rev.AI ASR inference for transcribe, benchmark, and
//!   Rev-backed UTR during align
//!
//! In server mode, Rev.AI is no longer a Python worker concern. The worker
//! boundary remains reserved for engines that genuinely require Python runtime
//! or model libraries.

mod asr;
mod client;
mod credentials;
mod evidence_cache;
mod poll;
#[cfg(test)]
mod scripted_server;
mod types;
mod utr;

pub use evidence_cache::RetainedRevLanguageRejection;

pub use asr::{RevAsrService, rev_evidence_to_asr_response};
pub use client::{Result, RevAiClient, TranscriptResult, extract_timed_words};
pub use credentials::{RevAiApiKey, load_revai_api_key};
pub use evidence_cache::*;
pub use poll::{Awaited, PollPolicy, SubmittedJob};
#[cfg(test)]
pub use types::RevTranscriptEvidence;
pub use types::{SubmitOptions, Transcript};
pub use utr::rev_evidence_to_utr_asr_response;
