//! Typed representation of UD/NLP responses consumed by Batchalign mapping logic.
//!
//! These types mirror the callback JSON contract used by Python pipeline engines,
//! but constrain ambiguous fields (`id`, `upos`, FA response variants) into Rust
//! enums before mapping starts.
//!
//! Keeping this layer strongly typed lets mapping code fail early with structured
//! errors instead of silently accepting malformed payloads.
//!
//! # Related CHAT Manual Sections
//!
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Format>
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Headers>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Main_Tier>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Dependent_Tiers>
//!
use serde::{Deserialize, Serialize};

pub use talkbank_transform::morphosyntax::{
    UdId, UdPunctable, UdResponse, UdSentence, UdWord, UniversalPos,
};

/// A raw token with its onset time, as returned by Whisper-style FA models.
///
/// Whisper produces token-level timestamps (one onset per sub-word token) rather
/// than word-level start/end pairs. The downstream DP aligner in `fa.rs`
/// reconstructs word boundaries by merging consecutive tokens and computing
/// durations from adjacent onsets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FaRawToken {
    /// The sub-word or word text fragment (e.g., " hello", " world").
    /// Leading whitespace is significant -- it indicates a word boundary in
    /// Whisper's byte-pair encoding.
    pub text: String,
    /// Onset time of this token in **seconds** (NOT milliseconds).
    /// Downstream code must convert to milliseconds (multiply by 1000) before
    /// injecting into CHAT timing bullets, which use integer milliseconds.
    pub time_s: f64,
}

/// Indexed timing produced when the callback already preserves word order.
///
/// This payload does not repeat word text; each entry corresponds to the same
/// index in the input `words` list supplied by Rust.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FaIndexedTiming {
    /// Start time in **milliseconds**.
    pub start_ms: u64,
    /// End time in **milliseconds**.
    pub end_ms: u64,
    /// Optional per-word confidence.
    pub confidence: Option<f64>,
}

/// Represents the raw data returned by a Forced Alignment "Passive Stub".
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum FaRawResponse {
    /// Indexed word-level timings aligned to callback input words by position.
    IndexedWordLevel {
        /// Per-index timing entries; `None` means no timing for that word.
        indexed_timings: Vec<Option<FaIndexedTiming>>,
    },
    /// Native Whisper format: list of (text, time)
    TokenLevel {
        /// Per-token BPE timing entries.
        tokens: Vec<FaRawToken>,
    },
}
