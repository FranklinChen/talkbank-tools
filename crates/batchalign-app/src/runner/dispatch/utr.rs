//! UTR (untimed utterance timing recovery) orchestration.
//!
//! This module keeps the full CHAT-level timing-recovery algorithm in Rust:
//! parse CHAT, decide between partial-window and full-file recovery, fetch raw
//! timed tokens from the selected backend, and inject timing bullets back into
//! the AST. Python is only used for the worker-hosted ASR path.

use std::path::Path;

use crate::api::{DurationMs, DurationSeconds, LanguageCode3, NumSpeakers};
use crate::cache::CacheBackend;
use crate::options::{UtrEngine, UtrOverlapStrategy};
use crate::params::CachePolicy;
use crate::pipeline::PipelineServices;
use crate::runner::debug_dumper::DebugDumper;
use tracing::{info, warn};

/// Immutable runtime inputs for one UTR execution.
#[derive(Clone, Copy)]
pub(in crate::runner) struct UtrPassContext<'a> {
    /// Audio file used to recover utterance timing.
    pub audio_path: &'a Path,
    /// CHAT language for ASR/UTR normalization.
    pub lang: &'a LanguageCode3,
    /// Shared worker pool/cache handles for the current pipeline stage.
    pub services: PipelineServices<'a>,
    /// Audio identity used to key UTR cache entries.
    pub audio_identity: &'a batchalign_chat_ops::fa::AudioIdentity,
    /// Cache policy selected for the current job.
    pub cache_policy: CachePolicy,
    /// Total audio duration in milliseconds when known.
    pub total_audio_ms: Option<DurationMs>,
    /// Maximum FA group duration in milliseconds. Used by the two-pass UTR
    /// strategy to compare FA grouping outcomes and detect the wider-window
    /// regression on non-English files.
    pub max_group_ms: Option<DurationMs>,
    /// Display filename for logging.
    pub filename: &'a str,
    /// Selected UTR backend.
    pub engine: &'a UtrEngine,
    /// Overlap strategy for `+<` utterances.
    pub overlap_strategy: UtrOverlapStrategy,
    /// Optional Rev.AI job ID produced by preflight submission.
    pub rev_job_id: Option<&'a str>,
    /// Debug artifact writer for offline replay.
    pub dumper: &'a DebugDumper,
}

impl<'a> UtrPassContext<'a> {
    /// Return the same context without a preflight job ID.
    ///
    /// Segment-level partial UTR extracts fresh temporary WAV files, so those
    /// requests can never reuse a full-file Rev.AI preflight submission.
    fn without_rev_job_id(self) -> Self {
        Self {
            rev_job_id: None,
            ..self
        }
    }
}

/// Resolve the UTR overlap strategy for a specific CHAT file.
///
/// `Auto` inspects the file for `+<` linkers and selects accordingly,
/// with a **language-aware gate**: non-English files always use `GlobalUtr`
/// because the two-pass strategy regresses on languages where ASR quality
/// is lower (wider FA groups cause the overlap recovery pass to misalign).
/// English files fall through to `select_strategy()` which inspects the
/// file for overlap markers.
///
/// `Global` and `TwoPass` are used as-is regardless of file content or
/// language — they are explicit overrides.
///
/// When `total_audio_ms` and `max_group_ms` are both available, a
/// [`GroupingContext`](batchalign_chat_ops::fa::GroupingContext) is passed to
/// the two-pass strategy so it can compare FA group counts and avoid the
/// wider-window regression on non-English files.
fn resolve_strategy(
    strategy: UtrOverlapStrategy,
    _chat_file: &batchalign_chat_ops::ChatFile,
    context: &UtrPassContext<'_>,
) -> Box<dyn batchalign_chat_ops::fa::UtrStrategy> {
    let grouping_context = match (context.total_audio_ms, context.max_group_ms) {
        (Some(total_audio_ms), Some(max_group_ms)) => {
            Some(batchalign_chat_ops::fa::GroupingContext {
                total_audio_ms: total_audio_ms.0,
                max_group_ms: max_group_ms.0,
            })
        }
        _ => None,
    };

    match strategy {
        UtrOverlapStrategy::Auto => {
            // Two-pass overlap strategy is experimental and gated behind
            // --utr-strategy two-pass.  Auto always uses GlobalUtr until
            // the two-pass algorithm is validated on an operator's problem files
            // and the end-time overlap bug is resolved.
            //
            // Previous behavior: auto-selected TwoPassOverlapUtr for English
            // files with +< or ⌊ markers.  Disabled 2026-03-30 because:
            // 1. an operator reported alignment regressions on real files.
            // 2. enforce_monotonicity() only checks start times, not end
            //    times, so overlapping utterance bullets go uncorrected.
            // 3. Two-pass was tuned on 4 corpora but not broadly validated.
            Box::new(batchalign_chat_ops::fa::GlobalUtr)
        }
        UtrOverlapStrategy::Global => Box::new(batchalign_chat_ops::fa::GlobalUtr),
        UtrOverlapStrategy::TwoPass => Box::new(batchalign_chat_ops::fa::TwoPassOverlapUtr {
            grouping_context,
            config: batchalign_chat_ops::fa::TwoPassConfig::default(),
        }),
    }
}

/// Run ASR and inject UTR timing into a parsed `ChatFile`.
///
/// Returns `Ok((updated_chat_text, utr_result))` on success, or
/// `Err(original_chat_text)` on inference failure.
///
/// When `progress` is provided, per-window updates are sent during partial UTR
/// so frontends can show "Recovering utterance timing 2/5" etc.
///
/// Mutates `chat_file` in place — no serialize/re-parse cycle. The caller owns
/// the AST and can pass it directly to FA without a round-trip through text.
pub(in crate::runner) async fn run_utr_pass(
    chat_file: &mut batchalign_chat_ops::ChatFile,
    context: UtrPassContext<'_>,
    progress: Option<&super::super::util::ProgressSender>,
) -> Result<batchalign_chat_ops::fa::utr::UtrResult, crate::error::ServerError> {
    use batchalign_chat_ops::CacheTaskName;

    let (timed, untimed) = batchalign_chat_ops::fa::count_utterance_timing(chat_file);
    let total_utts = timed + untimed;

    if untimed == 0 {
        info!(context.filename, "UTR pass: no untimed utterances");
        return Ok(batchalign_chat_ops::fa::utr::UtrResult {
            injected: 0,
            skipped: timed,
            unmatched: 0,
            decisions: Vec::new(),
        });
    }

    info!(
        context.filename,
        timed,
        untimed,
        engine = context.engine.as_wire_name(),
        "UTR pass: running timing recovery"
    );

    // Partial-window UTR is useful for worker-hosted ASR because it can avoid
    // sending already-timed regions through local model inference. For the
    // Rust-owned Rev.AI path, full-file polling is the better boundary: one
    // provider job, one transcript projection, no segment upload fan-out.
    let untimed_ratio = if total_utts > 0 {
        untimed as f64 / total_utts as f64
    } else {
        1.0
    };
    let use_partial = context.engine.supports_partial_windows()
        && untimed_ratio < 0.5
        && context.total_audio_ms.is_some_and(|ms| ms.0 > 60_000);

    if use_partial {
        let audio_ms = context
            .total_audio_ms
            .expect("partial UTR requires audio length");
        let windows = batchalign_chat_ops::fa::find_untimed_windows(chat_file, audio_ms.0, 500);

        if windows.is_empty() {
            info!(
                context.filename,
                "Partial UTR: no windows found, falling back to full-file recovery"
            );
        } else {
            info!(
                context.filename,
                windows = windows.len(),
                "Partial UTR: running ASR on untimed windows only"
            );

            let mut all_tokens: Vec<batchalign_chat_ops::fa::utr::AsrTimingToken> = Vec::new();
            let total_windows = windows.len() as i64;

            for (window_idx, &(start_ms, end_ms)) in windows.iter().enumerate() {
                let seg_cache_key = batchalign_chat_ops::fa::utr_asr_segment_cache_key(
                    context.audio_identity,
                    start_ms,
                    end_ms,
                    context.lang,
                );
                let cached_seg = if context.cache_policy != CachePolicy::SkipCache {
                    match context
                        .services
                        .cache
                        .get(
                            seg_cache_key.as_str(),
                            CacheTaskName::UtrAsr.as_str(),
                            context.services.engine_version,
                        )
                        .await
                    {
                        Ok(Some(value)) => {
                            info!(context.filename, start_ms, end_ms, "UTR segment cache hit");
                            serde_json::from_value::<crate::transcribe::AsrResponse>(value).ok()
                        }
                        _ => None,
                    }
                } else {
                    None
                };

                let seg_response = if let Some(cached) = cached_seg {
                    cached
                } else {
                    let segment_path = match crate::ensure_wav::extract_audio_segment(
                        context.audio_path,
                        start_ms,
                        end_ms,
                    )
                    .await
                    {
                        Ok(path) => path,
                        Err(error) => {
                            warn!(
                                context.filename,
                                error = %error,
                                start_ms,
                                end_ms,
                                "Failed to extract audio segment, falling back to full UTR"
                            );
                            return run_utr_pass_full(chat_file, context).await;
                        }
                    };

                    match infer_utr_asr_response(&segment_path, context.without_rev_job_id()).await
                    {
                        Ok(response) => {
                            let ba_version = env!("CARGO_PKG_VERSION");
                            if let Ok(value) = serde_json::to_value(&response)
                                && let Err(error) = context
                                    .services
                                    .cache
                                    .put(
                                        seg_cache_key.as_str(),
                                        CacheTaskName::UtrAsr.as_str(),
                                        context.services.engine_version,
                                        ba_version,
                                        &value,
                                    )
                                    .await
                            {
                                warn!(
                                    context.filename,
                                    error = %error,
                                    "Failed to cache UTR segment (non-fatal)"
                                );
                            }
                            response
                        }
                        Err(error) => {
                            warn!(
                                context.filename,
                                error = %error,
                                "UTR segment ASR failed, falling back to full-file recovery"
                            );
                            return run_utr_pass_full(chat_file, context).await;
                        }
                    }
                };

                all_tokens.extend(asr_response_to_utr_tokens(&seg_response, start_ms));

                // Report per-window progress so the frontend shows "Recovering
                // utterance timing 2/5" as each window's ASR completes.
                if let Some(tx) = progress {
                    use super::super::util::{FileStage, ProgressUpdate};
                    let _ = tx.send(ProgressUpdate::new(
                        FileStage::RecoveringUtteranceTiming,
                        Some(window_idx as i64 + 1),
                        Some(total_windows),
                    ));
                }
            }

            all_tokens.sort_by_key(|token| token.start_ms);

            if context.dumper.is_enabled() {
                let text = batchalign_chat_ops::serialize::to_chat_string(chat_file);
                context.dumper.dump_utr_input(context.filename, &text);
            }
            context
                .dumper
                .dump_utr_tokens(context.filename, &all_tokens);

            let strategy = resolve_strategy(context.overlap_strategy, chat_file, &context);
            let utr_result = strategy.inject(chat_file, &all_tokens);

            info!(
                context.filename,
                injected = utr_result.injected,
                skipped = utr_result.skipped,
                unmatched = utr_result.unmatched,
                "UTR partial pass complete"
            );

            if context.dumper.is_enabled() {
                let text = batchalign_chat_ops::serialize::to_chat_string(chat_file);
                context
                    .dumper
                    .dump_utr_output(context.filename, &text, &utr_result);
            }
            return Ok(utr_result);
        }
    }

    // Full-file path: signal 0/1 so the frontend knows it's a single-window pass.
    if let Some(tx) = progress {
        use super::super::util::{FileStage, ProgressUpdate};
        let _ = tx.send(ProgressUpdate::new(
            FileStage::RecoveringUtteranceTiming,
            Some(0),
            Some(1),
        ));
    }

    run_utr_pass_full(chat_file, context).await
}

/// Run the full-file UTR path with cache reuse.
async fn run_utr_pass_full(
    chat_file: &mut batchalign_chat_ops::ChatFile,
    context: UtrPassContext<'_>,
) -> Result<batchalign_chat_ops::fa::utr::UtrResult, crate::error::ServerError> {
    use batchalign_chat_ops::CacheTaskName;

    let cache_key =
        batchalign_chat_ops::fa::utr_asr_cache_key(context.audio_identity, context.lang);
    let cached_response = if context.cache_policy != CachePolicy::SkipCache {
        match context
            .services
            .cache
            .get(
                cache_key.as_str(),
                CacheTaskName::UtrAsr.as_str(),
                context.services.engine_version,
            )
            .await
        {
            Ok(Some(value)) => {
                info!(context.filename, "UTR ASR cache hit");
                serde_json::from_value::<crate::transcribe::AsrResponse>(value).ok()
            }
            Ok(None) => None,
            Err(error) => {
                warn!(
                    context.filename,
                    error = %error,
                    "UTR ASR cache lookup failed (non-fatal)"
                );
                None
            }
        }
    } else {
        None
    };

    let asr_response = if let Some(cached) = cached_response {
        cached
    } else {
        info!(
            context.filename,
            engine = context.engine.as_wire_name(),
            "UTR ASR cache miss, running inference"
        );
        match infer_utr_asr_response(context.audio_path, context).await {
            Ok(response) => {
                let ba_version = env!("CARGO_PKG_VERSION");
                if let Ok(value) = serde_json::to_value(&response)
                    && let Err(error) = context
                        .services
                        .cache
                        .put(
                            cache_key.as_str(),
                            CacheTaskName::UtrAsr.as_str(),
                            context.services.engine_version,
                            ba_version,
                            &value,
                        )
                        .await
                {
                    warn!(
                        context.filename,
                        error = %error,
                        "Failed to cache UTR ASR result (non-fatal)"
                    );
                }
                response
            }
            Err(error) => {
                warn!(context.filename, error = %error, "UTR ASR inference failed");
                return Err(error);
            }
        }
    };

    let asr_tokens = asr_response_to_utr_tokens(&asr_response, 0);

    if context.dumper.is_enabled() {
        let text = batchalign_chat_ops::serialize::to_chat_string(chat_file);
        context.dumper.dump_utr_input(context.filename, &text);
    }
    context
        .dumper
        .dump_utr_tokens(context.filename, &asr_tokens);

    let strategy = resolve_strategy(context.overlap_strategy, chat_file, &context);
    let utr_result = strategy.inject(chat_file, &asr_tokens);

    info!(
        context.filename,
        injected = utr_result.injected,
        skipped = utr_result.skipped,
        unmatched = utr_result.unmatched,
        "UTR pass complete"
    );

    if context.dumper.is_enabled() {
        let text = batchalign_chat_ops::serialize::to_chat_string(chat_file);
        context
            .dumper
            .dump_utr_output(context.filename, &text, &utr_result);
    }
    Ok(utr_result)
}

/// Fetch one UTR ASR response from the selected backend and map it into the
/// shared `AsrResponse` cache format.
async fn infer_utr_asr_response(
    audio_path: &Path,
    context: UtrPassContext<'_>,
) -> Result<crate::transcribe::AsrResponse, crate::error::ServerError> {
    match context.engine {
        UtrEngine::RevAi => {
            let tokens =
                crate::revai::infer_revai_utr(audio_path, context.lang, context.rev_job_id).await?;
            Ok(crate::transcribe::AsrResponse {
                tokens: tokens
                    .into_iter()
                    .map(|token| crate::transcribe::AsrToken {
                        text: token.text,
                        start_s: Some(DurationSeconds(token.start_ms as f64 / 1000.0)),
                        end_s: Some(DurationSeconds(token.end_ms as f64 / 1000.0)),
                        speaker: None,
                        confidence: None,
                    })
                    .collect(),
                lang: context.lang.clone(),
                source_monologues: None,
            })
        }
        UtrEngine::Whisper | UtrEngine::HkTencent => {
            crate::transcribe::infer_asr(
                context.services.pool,
                &crate::transcribe::AsrInferParams {
                    backend: crate::transcribe::AsrBackend::Worker(
                        crate::transcribe::AsrWorkerMode::LocalWhisperV2,
                    ),
                    audio_path,
                    lang: &crate::api::LanguageSpec::Resolved(context.lang.clone()),
                    num_speakers: NumSpeakers(1),
                    rev_job_id: None,
                },
            )
            .await
        }
    }
}

/// Convert cached/shared ASR responses into the timing-token shape consumed by
/// the UTR injector, applying an optional window offset in milliseconds.
///
/// Zero-duration tokens (`start_ms >= end_ms`) are filtered out here.
/// Whisper's DTW timestamp extraction works at 20ms resolution and can assign
/// `start == end` to very short words (single-frame backchannels like "mhm",
/// "yeah"). Such tokens carry no useful interval information and, if allowed
/// through, cause UTR to create `•T_T•` utterance bullets that the FA
/// postprocess then perpetuates indefinitely (see OCSC bug, 2026-04-08).
fn asr_response_to_utr_tokens(
    asr_response: &crate::transcribe::AsrResponse,
    offset_ms: u64,
) -> Vec<batchalign_chat_ops::fa::utr::AsrTimingToken> {
    asr_response
        .tokens
        .iter()
        .filter_map(|token| {
            let start_ms = (token.start_s?.0 * 1000.0).round() as u64 + offset_ms;
            let end_ms = (token.end_s?.0 * 1000.0).round() as u64 + offset_ms;
            // Drop zero-duration tokens: they carry no useful timing information
            // and propagate to zero-duration utterance bullets if allowed through.
            if end_ms <= start_ms {
                return None;
            }
            Some(batchalign_chat_ops::fa::utr::AsrTimingToken {
                text: token.text.clone(),
                start_ms,
                end_ms,
            })
        })
        .collect()
}
