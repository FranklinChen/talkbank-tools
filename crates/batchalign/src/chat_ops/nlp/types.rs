//! Whisper-style FA raw-response types (`FaRawToken`, `FaIndexedTiming`,
//! `FaRawResponse`). The UD/NLP types these used to live alongside have
//! moved to `batchalign_transform::morphosyntax`; only the FA-specific shapes
//! remain here because they're audio/timing-coupled.

use serde::{Deserialize, Serialize};

use crate::api::AudioPositionSeconds;

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
    /// Onset of this token, a position in **seconds** (NOT milliseconds)
    /// from the start of the alignment window. Proven finite and
    /// non-negative where the worker's response was read;
    /// [`AudioPositionSeconds::whole_millis`] is the conversion into the
    /// integer milliseconds a window position is measured in.
    pub time_s: AudioPositionSeconds,
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
