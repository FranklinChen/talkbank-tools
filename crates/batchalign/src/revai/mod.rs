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
mod types;
mod utr;

pub use evidence_cache::RetainedRevLanguageRejection;

pub(crate) use asr::{RevAsrService, rev_evidence_to_asr_response};
pub(crate) use client::{Result, RevAiClient, TranscriptResult, extract_timed_words};
pub(crate) use credentials::{RevAiApiKey, load_revai_api_key};
pub(crate) use evidence_cache::*;
#[cfg(test)]
pub(crate) use types::RevTranscriptEvidence;
pub(crate) use types::{SubmitOptions, Transcript};
pub(crate) use utr::rev_evidence_to_utr_asr_response;
