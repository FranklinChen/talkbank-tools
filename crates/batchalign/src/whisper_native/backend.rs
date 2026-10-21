//! Concrete whisper-rs backend implementation. Compiled only when
//! `whisper-rs-backend` is enabled.

#![cfg(feature = "whisper-rs-backend")]

use std::path::Path;
use std::sync::Arc;

use batchalign_types::api::{AudioPositionSeconds, LanguageCode3};
use batchalign_types::worker_v2::requests::WhisperChunkSpanV2;
use batchalign_types::worker_v2::responses::WhisperChunkResultV2;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use super::audio::{TARGET_SAMPLE_RATE, pcm_decode, resample};
use super::cache::PathKeyedCache;
use super::config::WhisperNativeConfig;
use super::error::WhisperNativeError;
use super::language::{iso2_to_lang, lang_to_iso2};

/// One whisper.cpp segment timestamp (centiseconds) as a proven position.
fn segment_seconds(centiseconds: i64) -> Result<AudioPositionSeconds, WhisperNativeError> {
    AudioPositionSeconds::try_from(centiseconds as f64 / 100.0)
        .map_err(|_| WhisperNativeError::SegmentTimestampNegative { centiseconds })
}

/// Process-level cache for the loaded `WhisperContext`. First call
/// pays the ~3 s + 3 GB load; every subsequent call clones an `Arc`
/// and creates a per-job `WhisperState`.
static CONTEXT_CACHE: PathKeyedCache<WhisperContext> = PathKeyedCache::new();

/// Drop the process-wide cached `WhisperContext` (releasing its Metal
/// buffers) so ggml's own exit-time destructors find nothing resident.
/// Called from the binary's epilogue; a no-op when nothing was loaded.
pub(super) fn shutdown_context_cache() -> bool {
    CONTEXT_CACHE.shutdown()
}

/// Load `WhisperContext` from disk. Cold-load only; routed through
/// `CONTEXT_CACHE` so it runs once per process (per model path).
fn load_whisper_context(model_path: &Path) -> Result<Arc<WhisperContext>, WhisperNativeError> {
    let path_str = model_path
        .to_str()
        .ok_or_else(|| WhisperNativeError::ModelPathNotUtf8(model_path.to_path_buf()))?;
    let load_started = std::time::Instant::now();
    let ctx = WhisperContext::new_with_params(path_str, WhisperContextParameters::default())?;
    tracing::info!(
        model = %model_path.display(),
        elapsed_ms = load_started.elapsed().as_millis() as u64,
        "whisper-native: cold-loaded WhisperContext"
    );
    Ok(Arc::new(ctx))
}

/// End-to-end transcribe via whisper.cpp's standalone decoder.
///
/// `WhisperContext` is shared across calls via `CONTEXT_CACHE`: cold-load
/// once per process, every subsequent call clones an `Arc` and creates a
/// fresh `WhisperState` (cheap: just decoder bookkeeping). See
/// `super::cache` for the cache policy and race-window behavior.
///
/// **Async caveat**: this function is sync. Callers in async context
/// must dispatch via `crate::blocking::spawn_in_span` (or the existing
/// worker-pool `execute_v2` machinery, which already isolates
/// blocking work); otherwise inference will stall the executor for
/// minutes per call.
pub(super) fn transcribe_impl(
    audio_path: &Path,
    lang: Option<LanguageCode3>,
    cfg: &WhisperNativeConfig,
) -> Result<WhisperChunkResultV2, WhisperNativeError> {
    tracing::info!(
        model = %cfg.model_path.display(),
        audio = %audio_path.display(),
        lang = lang.as_deref().unwrap_or("auto"),
        "whisper-native: starting"
    );

    let (mut pcm, native_sr) = pcm_decode(audio_path)?;
    if native_sr != TARGET_SAMPLE_RATE {
        tracing::debug!(
            from = native_sr,
            to = TARGET_SAMPLE_RATE,
            samples_in = pcm.len(),
            "whisper-native: resampling"
        );
        pcm = resample(&pcm, native_sr, TARGET_SAMPLE_RATE)?;
    }

    let ctx = CONTEXT_CACHE.get_or_load(&cfg.model_path, load_whisper_context)?;
    let mut state = ctx.create_state()?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    // `None` engages whisper.cpp's language auto-detection ("auto" token);
    // the detected id is read back after the run.
    let lang_iso2 = match &lang {
        Some(code) => lang_to_iso2(code)?,
        None => "auto",
    };
    params.set_language(Some(lang_iso2));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    if let Some(n) = cfg.n_threads {
        params.set_n_threads(n);
    }
    if let Some(mc) = cfg.max_context {
        // Pilot finding: setting this to 0 resets decoder context per
        // 30 s chunk, suppressing tail-of-audio token-loop hallucinations
        // (and running ~45% faster: cross-chunk attention is expensive).
        params.set_n_max_text_ctx(mc);
    }
    params.set_translate(cfg.translate);

    state.full(params, &pcm)?;

    let n_segments = state.full_n_segments();
    let mut chunks: Vec<WhisperChunkSpanV2> = Vec::with_capacity(n_segments as usize);
    for segment in state.as_iter() {
        let text = segment
            .to_str_lossy()
            .map_err(|e| WhisperNativeError::SegmentTextNotUtf8 {
                reason: e.to_string(),
            })?;
        let trimmed = text.trim();
        if trimmed.is_empty() {
            continue;
        }
        // whisper.cpp returns timestamps in centiseconds (10 ms units). A
        // negative one is a library defect, refused as such rather than
        // carried into the pipeline as a time.
        chunks.push(WhisperChunkSpanV2 {
            text: trimmed.to_owned(),
            start_s: Some(segment_seconds(segment.start_timestamp())?),
            end_s: Some(segment_seconds(segment.end_timestamp())?),
        });
    }

    let text = chunks
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    // Resolve the result language: the caller's, or the one whisper.cpp
    // detected during the run (auto mode).
    let lang = match lang {
        Some(code) => code,
        None => {
            let id = state.full_lang_id_from_state();
            let iso2 = whisper_rs::get_lang_str(id)
                .ok_or(WhisperNativeError::LanguageDetectionFailed { lang_id: id })?;
            iso2_to_lang(iso2)?
        }
    };

    tracing::info!(
        chunks = chunks.len(),
        chars = text.len(),
        lang = %lang,
        "whisper-native: done"
    );

    Ok(WhisperChunkResultV2 {
        lang,
        text,
        chunks: chunks.into(),
        // Built from the file this run actually loaded, not from a constant:
        // the path may be the auto-fetched default or one the host named, and
        // the result must say which weights produced it either way.
        model: crate::model_manifest::native_whisper_identity(&cfg.model_path, cfg.source)
            .map_err(|error| WhisperNativeError::ModelIdentity {
                reason: error.to_string(),
            })?,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_are_pilot_recommended() {
        let cfg = WhisperNativeConfig::for_model(std::path::PathBuf::from("dummy.bin"));
        assert_eq!(cfg.n_threads, Some(8));
        assert_eq!(cfg.max_context, Some(0));
        assert!(!cfg.translate);
    }
}
