//! Rust-native Whisper inference, gated on the `whisper-rs-backend` Cargo
//! feature. Replaces the Python-worker inference path when the ASR engine is
//! `whisper_rs` (`AsrBackend::RustWhisperRs`).
//!
//! ## Production integration seam
//!
//! The transcribe dispatch path (`transcribe::infer::infer_whisper_rs_asr`)
//! routes Whisper jobs here when `opts.backend == AsrBackend::RustWhisperRs`.
//! That dispatch reads `BATCHALIGN_WHISPER_RS_MODEL` via
//! [`WhisperNativeConfig::from_env`], resolves the language, and calls
//! [`transcribe`] inside `tokio::task::spawn_blocking` so the executor is not
//! stalled by whisper.cpp's sync inference loop.
//!
//! The `WhisperContext` is held process-wide in a single-slot
//! `OnceLock`-backed cache (see `cache.rs`) so the ~3 s + 3 GB model
//! load happens once per process; subsequent calls clone an `Arc` and
//! create a per-job `WhisperState` (cheap). The cache is locked to one
//! model path per process; switching models requires a restart, which
//! matches fleet-host policy (one model per host via env var).

#[cfg(feature = "whisper-rs-backend")]
pub mod backend;

mod audio;
mod cache;
mod config;
mod error;
// Numeric FA core (median filter + standardization + DTW): lives in the
// leaf crate `batchalign-fa-core` (shared with the whisper pilot);
// re-exported here so existing `whisper_native::fa_dtw::...` paths keep
// working.
pub use batchalign_fa_core as fa_dtw;

/// Release the process-wide native-Whisper resources (the cached
/// `WhisperContext` and its Metal buffers). MUST run before process
/// exit on macOS: ggml's C++ static destructors assert that every Metal
/// resource was deallocated, and a Rust `static` cache would otherwise
/// outlive them (observed as SIGABRT-after-success, 2026-07-29).
/// Returns whether anything was released; safe to call repeatedly.
pub fn shutdown() -> bool {
    #[cfg(feature = "whisper-rs-backend")]
    {
        backend::shutdown_context_cache()
    }
    #[cfg(not(feature = "whisper-rs-backend"))]
    {
        false
    }
}

/// RAII form of [`shutdown`]: bind one of these in `main` (next to the
/// tracing guard) and the native-Whisper resources are released on any
/// path out of `main`, including early returns. This is the enforcement
/// shape for the invariant that a process which loaded a
/// `WhisperContext` must release it before ggml's own exit-time
/// destructors run; a bare `shutdown()` statement silently un-enforces
/// itself the first time someone adds an early exit.
pub struct ProcessGuard;

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        shutdown();
    }
}

pub use config::{WhisperModelSource, WhisperNativeConfig};
pub use error::WhisperNativeError;

use batchalign_types::api::LanguageCode3;
use batchalign_types::worker_v2::responses::WhisperChunkResultV2;
use std::path::Path;

/// Run native Whisper inference on the given audio file. Returns the
/// same shape as the Python sidecar's `WhisperChunkResultV2` so
/// downstream pipeline stages need no changes.
///
/// Returns [`WhisperNativeError::FeatureDisabled`] when the
/// `whisper-rs-backend` Cargo feature is not enabled at build time.
pub fn transcribe(
    audio_path: &Path,
    lang: Option<LanguageCode3>,
    cfg: &WhisperNativeConfig,
) -> Result<WhisperChunkResultV2, WhisperNativeError> {
    #[cfg(feature = "whisper-rs-backend")]
    {
        backend::transcribe_impl(audio_path, lang, cfg)
    }
    #[cfg(not(feature = "whisper-rs-backend"))]
    {
        let _ = (audio_path, lang, cfg);
        Err(WhisperNativeError::FeatureDisabled)
    }
}

#[cfg(all(test, not(feature = "whisper-rs-backend")))]
#[allow(clippy::unwrap_used, clippy::panic, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn returns_feature_disabled_when_not_built_in() {
        let cfg = WhisperNativeConfig::for_model(std::path::PathBuf::from("dummy.bin"));
        let err = transcribe(
            std::path::Path::new("nonexistent.wav"),
            Some(LanguageCode3::eng()),
            &cfg,
        )
        .unwrap_err();
        assert!(matches!(err, WhisperNativeError::FeatureDisabled));
    }
}
