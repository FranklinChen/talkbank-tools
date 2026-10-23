//! Which language a stock Whisper worker decodes: a control-plane decision the
//! worker reads.
//!
//! The control plane decides it once per worker, from the worker's language,
//! and sends it under [`WHISPER_DECODE_LANGUAGE_KEY`] to every stock Whisper
//! worker it spawns. The worker never maps a language code itself, so what
//! admission accepted and what the decoder is told cannot disagree. The same
//! table ([`WhisperLanguageToken`]) answers both.
//!
//! This is decoding policy, separate from pinned model composition and loaded
//! identity. A worker's key carries its language; response-cache producers
//! must account for changes to the decoding policy for that language.

use serde::{Deserialize, Serialize};

use crate::domain::{LanguageCode3, WorkerLanguage};
use crate::whisper_language::WhisperLanguageToken;

/// The worker spawn key carrying a [`WhisperDecodeLanguageV2`] as JSON.
pub const WHISPER_DECODE_LANGUAGE_KEY: &str = "whisper_decode_language";

/// How a stock Whisper worker is told its language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WhisperDecodeLanguageV2 {
    /// Whisper detects the language itself.
    Detect,
    /// Whisper decodes in this language and transcribes (never translates).
    Force {
        /// Whisper's own token for the language.
        token: WhisperLanguageToken,
    },
}

/// Why a stock Whisper worker cannot be given a decode language.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WhisperDecodeLanguageError {
    /// Whisper's vocabulary has no token for the language.
    #[error("stock Whisper has no language token for '{0}'")]
    NoToken(LanguageCode3),
    /// A per-file worker: it has no single job language.
    #[error("a stock Whisper worker needs one job language or auto-detection, not per-file")]
    PerFileWorker,
    /// A worker spawned without any language.
    #[error("a stock Whisper worker needs one job language or auto-detection, not none")]
    LanguagelessWorker,
}

impl WhisperDecodeLanguageV2 {
    /// The decision for a worker spawned for `language`.
    pub fn for_worker_language(
        language: &WorkerLanguage,
    ) -> Result<Self, WhisperDecodeLanguageError> {
        match language {
            WorkerLanguage::Auto => Ok(Self::Detect),
            WorkerLanguage::Resolved(code) => Self::for_language(code),
            WorkerLanguage::PerFile => Err(WhisperDecodeLanguageError::PerFileWorker),
            WorkerLanguage::Unspecified => Err(WhisperDecodeLanguageError::LanguagelessWorker),
        }
    }

    /// The decision for one concrete language.
    ///
    /// A resolved language always uses its Whisper token and transcription
    /// task. Only an explicitly automatic worker may detect its language.
    pub fn for_language(language: &LanguageCode3) -> Result<Self, WhisperDecodeLanguageError> {
        let token = WhisperLanguageToken::for_language(language)
            .ok_or_else(|| WhisperDecodeLanguageError::NoToken(language.clone()))?;
        Ok(Self::Force { token })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)] // fixture construction; the decision is the assertion
mod tests {
    use super::*;

    fn resolved(raw: &str) -> WorkerLanguage {
        WorkerLanguage::Resolved(LanguageCode3::try_new(raw).expect("valid fixture"))
    }

    #[test]
    fn each_worker_language_has_one_decision() {
        let greek = WhisperDecodeLanguageV2::for_worker_language(&resolved("ell"))
            .expect("Whisper has a Greek token");
        assert_eq!(
            serde_json::to_string(&greek).expect("serializes"),
            r#"{"kind":"force","token":"el"}"#
        );
        assert_eq!(
            WhisperDecodeLanguageV2::for_worker_language(&WorkerLanguage::Auto),
            Ok(WhisperDecodeLanguageV2::Detect)
        );
        assert_eq!(
            WhisperDecodeLanguageV2::for_worker_language(&resolved("yue")),
            Ok(WhisperDecodeLanguageV2::Force {
                token: WhisperLanguageToken::for_language(&LanguageCode3::yue())
                    .expect("Whisper supports Cantonese"),
            }),
            "a resolved Cantonese worker must transcribe Cantonese"
        );
        assert!(matches!(
            WhisperDecodeLanguageV2::for_worker_language(&resolved("nan")),
            Err(WhisperDecodeLanguageError::NoToken(_))
        ));
        assert_eq!(
            WhisperDecodeLanguageV2::for_worker_language(&WorkerLanguage::PerFile),
            Err(WhisperDecodeLanguageError::PerFileWorker)
        );
        assert_eq!(
            WhisperDecodeLanguageV2::for_worker_language(&WorkerLanguage::Unspecified),
            Err(WhisperDecodeLanguageError::LanguagelessWorker)
        );
    }

    /// The wire shape the Python worker parses, and the refusal of a token
    /// Whisper does not have.
    #[test]
    fn the_wire_shape_is_stable() {
        assert_eq!(
            serde_json::to_string(&WhisperDecodeLanguageV2::Detect).expect("serializes"),
            r#"{"kind":"detect"}"#
        );
        assert!(
            serde_json::from_str::<WhisperDecodeLanguageV2>(r#"{"kind":"force","token":"nan"}"#)
                .is_err()
        );
    }
}
