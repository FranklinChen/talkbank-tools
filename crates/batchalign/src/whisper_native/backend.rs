//! Concrete whisper-rs backend implementation. Compiled only when
//! `whisper-rs-backend` is enabled.

#![cfg(feature = "whisper-rs-backend")]

use std::path::Path;
use std::sync::Arc;

use batchalign_types::api::{DurationSeconds, LanguageCode3};
use batchalign_types::worker_v2::requests::WhisperChunkSpanV2;
use batchalign_types::worker_v2::responses::WhisperChunkResultV2;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use super::audio::{TARGET_SAMPLE_RATE, pcm_decode, resample};
use super::cache::PathKeyedCache;
use super::config::WhisperNativeConfig;
use super::error::WhisperNativeError;

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
/// must dispatch via `tokio::task::spawn_blocking` (or the existing
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
        // whisper.cpp returns timestamps in centiseconds (10 ms units).
        chunks.push(WhisperChunkSpanV2 {
            text: trimmed.to_owned(),
            start_s: DurationSeconds(segment.start_timestamp() as f64 / 100.0),
            end_s: DurationSeconds(segment.end_timestamp() as f64 / 100.0),
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

    Ok(WhisperChunkResultV2 { lang, text, chunks })
}

/// One table, both directions: canonical ISO 639-3 (TalkBank internal)
/// paired with ISO 639-1 (whisper.cpp's language token). Restricted to
/// the languages whisper.cpp's large models genuinely support well; a
/// silent default-to-English would mask real misconfiguration. The
/// Rev.AI submission preflight at
/// `crates/batchalign/src/types/revai_language.rs`
/// (`try_revai_language_hint`) keeps a fuller table; lifting both into a shared
/// `LanguageCode3::to_iso_639_1()` on `batchalign-types` is the right
/// follow-up (tracked in the whisper-asr book page). A round-trip test
/// below keeps the two directions from drifting.
const LANG_TABLE: &[(&str, &str)] = &[
    ("eng", "en"),
    ("yue", "yue"),
    ("cmn", "zh"),
    ("fra", "fr"),
    ("deu", "de"),
    ("spa", "es"),
    ("ita", "it"),
    ("jpn", "ja"),
    ("kor", "ko"),
    ("nld", "nl"),
    ("por", "pt"),
    ("rus", "ru"),
    ("tur", "tr"),
    ("swe", "sv"),
    ("nor", "no"),
    ("dan", "da"),
    ("fin", "fi"),
    ("pol", "pl"),
    ("ell", "el"),
    ("hun", "hu"),
    ("heb", "he"),
    ("ara", "ar"),
];

/// Legacy ISO 639-2/B aliases accepted on INPUT (normalized to the
/// canonical 639-3 code before the table lookup). Note the asymmetry
/// this creates for auto-detect: an explicit `zho` job stays `zho`, but
/// a detected `zh` comes back as the canonical `cmn`.
const LANG_INPUT_ALIASES: &[(&str, &str)] = &[
    ("zho", "cmn"),
    ("fre", "fra"),
    ("ger", "deu"),
    ("dut", "nld"),
    ("gre", "ell"),
];

fn lang_to_iso2(lang: &LanguageCode3) -> Result<&'static str, WhisperNativeError> {
    let canonical = LANG_INPUT_ALIASES
        .iter()
        .find(|(alias, _)| *alias == &**lang)
        .map_or(&**lang, |(_, canon)| *canon);
    LANG_TABLE
        .iter()
        .find(|(iso3, _)| *iso3 == canonical)
        .map(|(_, iso2)| *iso2)
        .ok_or_else(|| WhisperNativeError::UnsupportedLanguage((**lang).to_owned()))
}

/// Map whisper.cpp's detected ISO 639-1 token back to TalkBank's ISO
/// 639-3 canonical code: the reverse direction of `LANG_TABLE`,
/// restricted to the same supported set so detection cannot smuggle in
/// a language the rest of the pipeline has no support for.
fn iso2_to_lang(iso2: &str) -> Result<LanguageCode3, WhisperNativeError> {
    let (iso3, _) = LANG_TABLE
        .iter()
        .find(|(_, i2)| *i2 == iso2)
        .ok_or_else(|| WhisperNativeError::UnsupportedDetectedLanguage(iso2.to_owned()))?;
    // The table holds only valid three-letter codes; a TryFrom failure
    // here would mean the table itself is malformed.
    LanguageCode3::try_from(*iso3)
        .map_err(|_| WhisperNativeError::UnsupportedDetectedLanguage(iso2.to_owned()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn lang_map_known_codes() {
        assert_eq!(lang_to_iso2(&LanguageCode3::eng()).unwrap(), "en");
    }

    #[test]
    fn lang_table_round_trips() {
        // Both directions come from one table; every entry must survive
        // iso3 -> iso2 -> iso3 unchanged.
        for (iso3, iso2) in LANG_TABLE {
            let fwd = lang_to_iso2(&LanguageCode3::try_from(*iso3).unwrap()).unwrap();
            assert_eq!(fwd, *iso2);
            let back = iso2_to_lang(iso2).unwrap();
            assert_eq!(&*back, *iso3);
        }
    }

    #[test]
    fn lang_aliases_normalize_on_input() {
        // Legacy 639-2/B aliases are accepted forward but never produced
        // in reverse (the documented auto-detect asymmetry).
        for (alias, canonical) in LANG_INPUT_ALIASES {
            let via_alias = lang_to_iso2(&LanguageCode3::try_from(*alias).unwrap()).unwrap();
            let via_canonical =
                lang_to_iso2(&LanguageCode3::try_from(*canonical).unwrap()).unwrap();
            assert_eq!(via_alias, via_canonical);
        }
    }

    #[test]
    fn lang_map_unknown_returns_error() {
        let bogus = LanguageCode3("xxx".to_owned());
        let err = lang_to_iso2(&bogus).unwrap_err();
        assert!(matches!(err, WhisperNativeError::UnsupportedLanguage(_)));
    }

    #[test]
    fn config_defaults_are_pilot_recommended() {
        let cfg = WhisperNativeConfig::for_model(std::path::PathBuf::from("dummy.bin"));
        assert_eq!(cfg.n_threads, Some(8));
        assert_eq!(cfg.max_context, Some(0));
        assert!(!cfg.translate);
    }
}
