//! Typed errors for the native Whisper path.

use std::path::PathBuf;
use thiserror::Error;

/// Typed error for everything that can fail in the native Whisper path:
/// missing build features, model-file problems, audio decode/resample
/// failures, language-token issues, and upstream whisper-rs / rubato
/// errors (carried via `#[from]` so source chains stay intact).
#[derive(Debug, Error)]
pub enum WhisperNativeError {
    /// Build was compiled without the `whisper-rs-backend` Cargo feature.
    #[error(
        "native Whisper path is not available in this build (\
         the `whisper-rs-backend` Cargo feature was not enabled at compile time)"
    )]
    FeatureDisabled,

    /// `BATCHALIGN_WHISPER_RS_MODEL` env var not set and no model path
    /// configured.
    #[error(
        "no whisper-rs model path configured (set `BATCHALIGN_WHISPER_RS_MODEL` \
         to a `.bin` ggml model file)"
    )]
    ModelPathMissing,

    /// Model path is not valid UTF-8: whisper.cpp's C API can't accept it.
    #[error("whisper-rs model path is not valid UTF-8: {0}")]
    ModelPathNotUtf8(PathBuf),

    /// Audio decode failed (bad codec, missing file, etc.).
    #[error("audio decode failed for {path}: {reason}")]
    AudioDecode {
        /// The audio file path that failed to decode.
        path: PathBuf,
        /// Human-readable reason from symphonia.
        reason: String,
    },

    /// Audio resample failed. Carries a formatted rubato error message
    /// (rubato 4.0's error types are not carried structurally to keep this
    /// enum independent of the resampler crate's error surface).
    #[error("audio resample failed: {0}")]
    AudioResample(String),

    /// Audio resampler construction failed (mismatched parameters).
    #[error("audio resampler construction failed: {0}")]
    AudioResamplerCtor(String),

    /// whisper.cpp init / inference / state error.
    #[cfg(feature = "whisper-rs-backend")]
    #[error("whisper-rs error: {0}")]
    Whisper(#[from] whisper_rs::WhisperError),

    /// The configured language is not supported by this Whisper model
    /// (token lookup failed for `<|lang|>`).
    #[error("language `{0}` is not supported by this Whisper model")]
    UnsupportedLanguage(String),

    /// Auto-detection ran but whisper.cpp returned a language id with no
    /// known ISO 639-1 token.
    #[error("language auto-detection returned unknown whisper.cpp lang id {lang_id}")]
    LanguageDetectionFailed {
        /// The raw whisper.cpp language id.
        lang_id: i32,
    },

    /// Auto-detection produced a language outside the supported set (the
    /// same closed table `lang_to_iso2` enforces on explicit input).
    #[error("auto-detected language `{0}` is outside the supported set")]
    UnsupportedDetectedLanguage(String),

    /// Default-model resolution failed (download or cache error from
    /// hf-hub when no explicit model path was configured).
    #[error("default whisper-rs model resolution failed: {reason}")]
    ModelResolution {
        /// Human-readable detail from hf-hub.
        reason: String,
    },

    /// The loaded weights file could not be recorded as a model identity, so
    /// the result would have to name its model dishonestly or not at all.
    /// Refused instead: an ASR result carries the models behind it by
    /// construction, and a transcript nobody can attribute is worse than a
    /// failed job.
    #[error("native Whisper could not record which weights it loaded: {reason}")]
    ModelIdentity {
        /// Why the weights file could not be named.
        reason: String,
    },

    /// Token segment text was not valid UTF-8; should be impossible in
    /// practice with Whisper's vocab but kept as a strict guard.
    #[error("segment text was not valid UTF-8: {reason}")]
    SegmentTextNotUtf8 {
        /// Human-readable detail from the upstream UTF-8 error.
        reason: String,
    },

    /// whisper.cpp reported a segment timestamp that is not a time: negative
    /// centiseconds. Never seen in practice; refused rather than carried.
    #[error("segment timestamp is not a time: {centiseconds} cs")]
    SegmentTimestampNegative {
        /// The raw centisecond value whisper.cpp reported.
        centiseconds: i64,
    },

    /// The context cache reached a state that should be unreachable in
    /// practice (a concurrent `set` neither stored our value nor a matching
    /// one). Surfaced as a typed error rather than a panic because this crate
    /// denies `unreachable!`.
    #[error("whisper context cache invariant violated: {0}")]
    CacheInvariant(String),

    /// The process-cached `WhisperContext` was loaded from a different model
    /// path than the one this request is asking for. The cache is
    /// single-slot by design (one model per process), so a path change
    /// signals either misconfiguration or an attempt to hot-swap models;
    /// neither is supported. Restart the process with the new model path.
    #[error(
        "whisper-rs context cache is locked to model `{cached}` for this \
         process; refusing to load a second model `{requested}`. Restart to \
         change models."
    )]
    ModelPathChanged {
        /// The model path the cache was first initialized with.
        cached: PathBuf,
        /// The model path the new request is asking for.
        requested: PathBuf,
    },
}
