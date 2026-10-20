//! How a Qwen recognizer runs: a control-plane decision the worker reads.
//!
//! The decoding window and the fate of recognized punctuation depend on the
//! language. The control plane decides them once, from the same table that
//! decides whether a transcription keeps the recognizer's own utterance
//! boundaries, and sends the decision to every worker it spawns under
//! [`QWEN_RECOGNITION_KEY`]. The worker never decides again from the language
//! code, so the route that relies on recognizer punctuation and the recognizer
//! that restores it cannot disagree.
//!
//! This is policy, not model identity: it is not part of the pinned model
//! composition, the loaded identity, or any cache namespace.

use serde::{Deserialize, Serialize};

/// The worker spawn key carrying a [`QwenRecognitionV2`] as JSON.
pub const QWEN_RECOGNITION_KEY: &str = "qwen_recognition";

/// How a Qwen recognizer turns audio into timed words.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
pub struct QwenRecognitionV2 {
    /// The decoding window the recording is split into.
    pub window: DecodeWindowV2,
    /// What becomes of the punctuation in the recognizer's text.
    pub punctuation: RecognitionPunctuationV2,
}

/// The decoding window a Qwen recognizer splits a recording into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DecodeWindowV2 {
    /// The chunker's own default (the forced aligner's input ceiling).
    ChunkerDefault,
    /// At most this many seconds per window, for languages whose spontaneous
    /// speech drifts into repetition in long windows.
    BoundedSeconds {
        /// The window ceiling.
        seconds: std::num::NonZeroU32,
    },
}

/// What becomes of the punctuation in a recognizer's text.
///
/// The forced aligner returns lexical units only, so recognized punctuation
/// survives only when deliberately restored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RecognitionPunctuationV2 {
    /// Attach recognized sentence and clause punctuation to the preceding
    /// aligned unit, after proving exact lexical correspondence. Used where the
    /// recognizer's punctuation is the utterance-boundary evidence.
    Restore,
    /// Keep only the aligned units; a segmentation model supplies boundaries.
    Discard,
}

#[cfg(test)]
#[allow(clippy::expect_used)] // fixture construction; the wire shape is the assertion
mod tests {
    use super::*;

    /// The wire shape the Python worker parses.
    #[test]
    fn the_wire_shape_is_stable() {
        let policy = QwenRecognitionV2 {
            window: DecodeWindowV2::BoundedSeconds {
                seconds: std::num::NonZeroU32::new(30).expect("nonzero"),
            },
            punctuation: RecognitionPunctuationV2::Restore,
        };
        let json = serde_json::to_string(&policy).expect("serializes");
        assert_eq!(
            json,
            r#"{"window":{"kind":"bounded_seconds","seconds":30},"punctuation":"restore"}"#
        );
        assert_eq!(
            serde_json::from_str::<QwenRecognitionV2>(&json).expect("round-trips"),
            policy
        );
        assert!(
            serde_json::from_str::<QwenRecognitionV2>(
                r#"{"window":{"kind":"bounded_seconds","seconds":0},"punctuation":"restore"}"#
            )
            .is_err(),
            "a zero-second window is not representable"
        );
    }
}
