//! Transcribe pipeline built on the internal stage runner.

use crate::chat_ops::morphosyntax_ops::{MultilingualPolicy, TokenizationMode};
use crate::chat_ops::speaker::{
    SpeakerSegment as ChatSpeakerSegment, project_speakers_onto_chunks,
};
use batchalign_transform::asr_postprocess::{
    self, AsrPipelineSnapshot, AsrWord, PreparedMonologueChunk, Utterance,
};
use batchalign_transform::build_chat;
use batchalign_transform::serialize::to_chat_string;
use batchalign_transform::utseg::UtsegBatchItem;
use std::path::Path;

use tracing::info;

use crate::api::{ChatText, LanguageCode3, LanguageSpec, NumSpeakers, WorkerLanguage};
use crate::error::ServerError;
use crate::params::{MorphosyntaxParams, UtsegFallbackPolicy};
use crate::pipeline::PipelineServices;
use crate::pipeline::plan::{PipelinePlan, StageFuture, StageId, StageSpec, run_plan};
use crate::revai::{
    PreparedRevProviderMedia, RevAsrEvidenceInference, RevAsrEvidenceRequest, RevAsrEvidenceSource,
    RevAsrEvidenceTrace, RevAsrModelRevision, RevAsrProjectionRevision, RevAsrService,
    resolve_rev_asr_evidence, rev_asr_resolution_error_to_server_error,
    rev_evidence_to_asr_response,
};
use crate::runner::debug_dumper::DebugDumper;
use crate::runner::util::{FileStage, ProgressSender, ProgressUpdate};
use crate::transcribe::replay::AdmittedLegacyTranscribeReplay;
use crate::transcribe::{
    AsrInferParams, AsrResponse, SpeakerEvidenceRunParams, SpeakerEvidenceSource,
    TranscribeOptions, build_empty_chat_text, convert_asr_response, generate_participant_ids,
    infer_asr, resolve_speaker_evidence_for_audio,
};
use crate::types::worker_v2::{SpeakerBackendV2, SpeakerSegmentV2};
use crate::utseg::TranscribeUtsegExecution;
use crate::utseg_evidence::{UtsegEvidencePhase, UtsegEvidenceSink, UtsegEvidenceTrace};

static PRODUCTION_REV_INFERENCE: RevAsrService = RevAsrService;

/// Mutually exclusive evidence capabilities for one transcribe execution.
///
/// The replay variant carries admitted evidence but no provider-inference
/// capability. Code executing a replay therefore cannot call Rev.AI or a
/// speaker backend without first changing this exhaustive match.
enum TranscribeEvidenceInput<'a> {
    Live {
        rev_inference: &'a dyn RevAsrEvidenceInference,
    },
    LegacyReplay {
        replay: AdmittedLegacyTranscribeReplay,
    },
}

/// Per-file transcribe pipeline state.
pub(crate) struct TranscribePipelineContext<'a> {
    /// Shared services for the run.
    pub services: PipelineServices<'a>,
    /// Immutable transcribe options.
    pub opts: &'a TranscribeOptions,
    /// Audio path being processed.
    pub audio_path: &'a Path,
    /// Raw ASR worker response.
    pub asr_response: Option<AsrResponse>,
    /// Postprocessed utterances.
    pub utterances: Option<Vec<Utterance>>,
    /// Dedicated diarization segments when Rust composes the speaker task.
    pub speaker_segments: Option<Vec<SpeakerSegmentV2>>,
    /// Current serialized CHAT text.
    pub chat_text: Option<String>,
    /// Debug artifact writer for offline replay.
    pub dumper: DebugDumper,
    /// Fail-closed destination for versioned utterance-boundary evidence.
    utseg_evidence_sink: UtsegEvidenceSink,
    /// Live-inference capability or fingerprinted offline replay evidence.
    evidence_input: TranscribeEvidenceInput<'a>,
    /// Explicit pass topology and policy for utterance segmentation.
    utseg_execution: TranscribeUtsegExecution,
    /// Typed causal receipt for the Rev projection used by this run.
    rev_evidence: Option<RevAsrEvidenceTrace>,
    /// Language resolved after ASR detection. When `opts.lang` is `Auto`,
    /// this is set by `stage_build_chat` to the ASR-detected language.
    /// Post-ASR stages (utseg, morphotag) use this for concrete dispatch.
    pub resolved_lang: Option<LanguageCode3>,
    /// Per-stage ASR pipeline snapshot. Populated when
    /// `BA3_DUMP_ASR_PIPELINE` is set, otherwise `None`. Captures the
    /// stage outputs that `AsrPipelineTrace` (the dashboard-facing
    /// type) would render. See `crate::types::results::snapshot_into_pipeline_trace`
    /// for the conversion.
    pub asr_pipeline_snapshot: Option<AsrPipelineSnapshot>,
}

impl<'a> TranscribePipelineContext<'a> {
    #[cfg(test)]
    fn new_with_rev_inference(
        audio_path: &'a Path,
        services: PipelineServices<'a>,
        opts: &'a TranscribeOptions,
        dumper: DebugDumper,
        utseg_evidence_sink: UtsegEvidenceSink,
        rev_inference: &'a dyn RevAsrEvidenceInference,
    ) -> Self {
        Self::new_with_evidence_input(
            audio_path,
            services,
            opts,
            dumper,
            utseg_evidence_sink,
            TranscribeEvidenceInput::Live { rev_inference },
            TranscribeUtsegExecution::production(opts.with_utseg),
        )
    }

    fn new_with_evidence_input(
        audio_path: &'a Path,
        services: PipelineServices<'a>,
        opts: &'a TranscribeOptions,
        dumper: DebugDumper,
        utseg_evidence_sink: UtsegEvidenceSink,
        evidence_input: TranscribeEvidenceInput<'a>,
        utseg_execution: TranscribeUtsegExecution,
    ) -> Self {
        Self {
            services,
            opts,
            audio_path,
            asr_response: None,
            utterances: None,
            speaker_segments: None,
            chat_text: None,
            dumper,
            utseg_evidence_sink,
            evidence_input,
            utseg_execution,
            rev_evidence: None,
            resolved_lang: None,
            asr_pipeline_snapshot: std::env::var("BA3_DUMP_ASR_PIPELINE")
                .ok()
                .map(|_| AsrPipelineSnapshot::default()),
        }
    }

    /// Return the resolved language code for NLP stages (utseg, morphotag).
    ///
    /// After ASR, `resolved_lang` is populated from the ASR response's
    /// detected language. If `opts.lang` was already resolved (not Auto),
    /// it's used directly. Returns an error if called before resolution
    /// this is a structural guarantee that the pipeline runs ASR before NLP.
    fn lang_for_nlp(&self) -> Result<&LanguageCode3, ServerError> {
        if let Some(ref resolved) = self.resolved_lang {
            return Ok(resolved);
        }
        match &self.opts.lang {
            LanguageSpec::Resolved(code) => Ok(code),
            LanguageSpec::Auto => Err(ServerError::Validation(
                "lang_for_nlp() called with unresolved Auto language, \
                 ASR must resolve the language before NLP stages run"
                    .into(),
            )),
            LanguageSpec::PerFile => Err(ServerError::Validation(
                "lang_for_nlp() called with PerFile language, transcribe \
                 takes an explicit `--lang` and never carries a per-file \
                 language spec; this state should have been rejected at \
                 submission validation"
                    .into(),
            )),
        }
    }

    fn asr_provenance_name(&self) -> &'static str {
        if let TranscribeEvidenceInput::LegacyReplay { replay, .. } = &self.evidence_input {
            return replay.producer().provenance_name();
        }
        self.opts.backend.provenance_name()
    }
}

/// Run the transcribe pipeline for a single audio file.
pub(crate) async fn run_transcribe_pipeline(
    audio_path: &Path,
    services: PipelineServices<'_>,
    opts: &TranscribeOptions,
    progress: Option<ProgressSender>,
    debug_dir: Option<&Path>,
) -> Result<String, ServerError> {
    let completed = run_transcribe_pipeline_with_rev_inference(
        audio_path,
        services,
        opts,
        progress,
        debug_dir,
        &PRODUCTION_REV_INFERENCE,
    )
    .await?;
    let CompletedTranscribePipeline {
        chat_text,
        rev_evidence: _rev_evidence,
    } = completed;
    Ok(chat_text)
}

#[derive(Debug)]
struct CompletedTranscribePipeline {
    chat_text: String,
    rev_evidence: Option<RevAsrEvidenceTrace>,
}

async fn run_transcribe_pipeline_with_rev_inference<'a>(
    audio_path: &'a Path,
    services: PipelineServices<'a>,
    opts: &'a TranscribeOptions,
    progress: Option<ProgressSender>,
    debug_dir: Option<&Path>,
    rev_inference: &'a dyn RevAsrEvidenceInference,
) -> Result<CompletedTranscribePipeline, ServerError> {
    run_transcribe_pipeline_with_evidence_input(
        audio_path,
        services,
        opts,
        progress,
        debug_dir,
        TranscribeEvidenceInput::Live { rev_inference },
        TranscribeUtsegExecution::production(opts.with_utseg),
    )
    .await
}

/// Replay fingerprinted projected evidence through the current local Rust and
/// worker post-processing stages. This does not enter either paid cache.
pub(crate) async fn run_transcribe_pipeline_with_legacy_replay<'a>(
    replay: AdmittedLegacyTranscribeReplay,
    utseg_execution: TranscribeUtsegExecution,
    services: PipelineServices<'a>,
    opts: &'a TranscribeOptions,
    progress: Option<ProgressSender>,
    debug_dir: Option<&Path>,
) -> Result<String, ServerError> {
    let audio_path = replay.media_path().to_owned();
    let completed = run_transcribe_pipeline_with_evidence_input(
        &audio_path,
        services,
        opts,
        progress,
        debug_dir,
        TranscribeEvidenceInput::LegacyReplay { replay },
        utseg_execution,
    )
    .await?;
    Ok(completed.chat_text)
}

async fn run_transcribe_pipeline_with_evidence_input<'a>(
    audio_path: &'a Path,
    services: PipelineServices<'a>,
    opts: &'a TranscribeOptions,
    progress: Option<ProgressSender>,
    debug_dir: Option<&Path>,
    evidence_input: TranscribeEvidenceInput<'a>,
    utseg_execution: TranscribeUtsegExecution,
) -> Result<CompletedTranscribePipeline, ServerError> {
    // Plan-time language gate: if the user resolved a language Stanza
    // can't handle, drop the optional Stanza-backed stages at plan
    // build time so the dep graph stays internally consistent. The
    // runtime registry (populated from the worker's resources.json) is
    // authoritative when present; the hardcoded chat-ops list is the
    // pre-warmup fallback. Auto-detect stays optimistic, the worker's
    // typed UnsupportedLanguageError catches the resolved-to-unsupported
    // case if it arises.
    let stanza_supported = match &opts.lang {
        crate::types::domain::LanguageSpec::Resolved(code) => {
            if let Some(reg) = services.pool.stanza_registry() {
                reg.supports_morphosyntax(code.as_ref())
            } else {
                // Fallible in chatter 0.3.0; stringified error
                // (`LanguageCodeError` not re-exported upstream).
                let chat_lang = crate::chat_ops::LanguageCode::new(code.as_ref()).map_err(|e| {
                    ServerError::Validation(format!(
                        "transcribe: invalid language code {:?}: {e}",
                        code.as_ref()
                    ))
                })?;
                crate::chat_ops::morphosyntax_ops::is_stanza_supported(&chat_lang)
            }
        }
        // Auto stays optimistic; the worker's UnsupportedLanguageError catches
        // resolved-to-unsupported cases at runtime.
        crate::types::domain::LanguageSpec::Auto => true,
        // PerFile is not a transcribe state, submission validation should
        // have rejected it. Be optimistic here so a regression in validation
        // doesn't silently disable Stanza-backed transcribe stages; the
        // resolved-language path will trip its own typed error if reached.
        crate::types::domain::LanguageSpec::PerFile => true,
    };
    let with_post_chat_utseg = utseg_execution.post_chat_policy().is_some() && stanza_supported;
    let with_morphosyntax = opts.with_morphosyntax && stanza_supported;
    if !stanza_supported {
        info!(
            lang = ?opts.lang,
            skipped_post_chat_utseg = utseg_execution.post_chat_policy().is_some(),
            pre_chat_utseg_still_requested = utseg_execution.pre_chat_policy().is_some(),
            skipped_morphosyntax = opts.with_morphosyntax,
            "Skipping requested Stanza-backed post-CHAT stages: no Stanza pipeline for this language."
        );
    }
    let plan = transcribe_plan(opts.diarize, with_post_chat_utseg, with_morphosyntax);
    let dumper = DebugDumper::new(debug_dir);
    let utseg_evidence_sink = UtsegEvidenceSink::new(debug_dir);
    let mut ctx = TranscribePipelineContext::new_with_evidence_input(
        audio_path,
        services,
        opts,
        dumper,
        utseg_evidence_sink,
        evidence_input,
        utseg_execution,
    );

    // Build stage-level progress callback if a sender is provided.
    let on_stage = progress.map(|tx| {
        move |stage: StageId, done: usize, total: usize| {
            let _ = tx.send(ProgressUpdate::new(
                progress_stage_for_stage(stage),
                Some(done as i64),
                Some(total as i64),
            ));
        }
    });

    let on_stage_ref: Option<&(dyn Fn(StageId, usize, usize) + Send + Sync)> =
        on_stage.as_ref().map(|cb| cb as _);
    let _ = run_plan("transcribe", &plan, &mut ctx, on_stage_ref).await?;

    let chat_text = ctx.chat_text.ok_or_else(|| {
        ServerError::Validation("transcribe pipeline completed without output".to_string())
    })?;
    Ok(CompletedTranscribePipeline {
        chat_text,
        rev_evidence: ctx.rev_evidence,
    })
}

/// Map transcribe-pipeline stage ids onto the shared file-progress stage
/// vocabulary.
///
/// This match is intentionally explicit. If the transcribe plan adds a new
/// stage, contributors should decide its operator-facing stage here rather
/// than silently falling back to a generic string.
fn progress_stage_for_stage(stage: StageId) -> FileStage {
    // Plan invariant: `transcribe_plan` (below) only emits stages
    // from the `StageId` set listed above. New `StageId` variants
    // not handled here will fail this match, caught by the
    // catalog test in `recipe_runner/catalog.rs` before reaching
    // production.
    #[allow(clippy::unreachable)]
    match stage {
        StageId::AsrInfer => FileStage::Transcribing,
        StageId::SpeakerDiarization => FileStage::PostProcessing,
        StageId::AsrPostprocess => FileStage::PostProcessing,
        StageId::BuildChat => FileStage::BuildingChat,
        StageId::OptionalUtseg => FileStage::SegmentingUtterances,
        StageId::OptionalMorphosyntax => FileStage::AnalyzingMorphosyntax,
        StageId::Serialize => FileStage::Finalizing,
        _ => unreachable!("transcribe plan emitted unsupported stage id {stage}"),
    }
}

fn transcribe_plan<'a>(
    diarize: bool,
    with_post_chat_utseg: bool,
    with_morphosyntax: bool,
) -> PipelinePlan<TranscribePipelineContext<'a>> {
    let postprocess_dep = if diarize {
        StageId::SpeakerDiarization
    } else {
        StageId::AsrInfer
    };
    let mut stages = vec![
        StageSpec::new(StageId::AsrInfer, vec![], always_enabled, stage_asr_infer),
        StageSpec::new(
            StageId::SpeakerDiarization,
            vec![StageId::AsrInfer],
            diarization_requested,
            stage_speaker_diarization,
        ),
        StageSpec::new(
            StageId::AsrPostprocess,
            vec![postprocess_dep],
            always_enabled,
            stage_asr_postprocess,
        ),
        StageSpec::new(
            StageId::BuildChat,
            vec![StageId::AsrPostprocess],
            always_enabled,
            stage_build_chat,
        ),
    ];

    if with_post_chat_utseg {
        stages.push(StageSpec::new(
            StageId::OptionalUtseg,
            vec![StageId::BuildChat],
            always_enabled,
            stage_run_utseg,
        ));
    }

    if with_morphosyntax {
        let dep = if with_post_chat_utseg {
            StageId::OptionalUtseg
        } else {
            StageId::BuildChat
        };
        stages.push(StageSpec::new(
            StageId::OptionalMorphosyntax,
            vec![dep],
            always_enabled,
            stage_run_morphosyntax,
        ));
    }

    let final_dep = if with_morphosyntax {
        StageId::OptionalMorphosyntax
    } else if with_post_chat_utseg {
        StageId::OptionalUtseg
    } else {
        StageId::BuildChat
    };
    stages.push(StageSpec::new(
        StageId::Serialize,
        vec![final_dep],
        always_enabled,
        stage_serialize,
    ));

    PipelinePlan::new(stages)
}

fn always_enabled(_: &TranscribePipelineContext<'_>) -> bool {
    true
}

fn diarization_requested(ctx: &TranscribePipelineContext<'_>) -> bool {
    ctx.opts.diarize
}

/// Whether the dedicated post-ASR speaker diarization stage should run.
///
/// BA2-jan9 semantics are explicit: `transcribe_s` means "run the separate
/// speaker backend as a post-processing step" even when the ASR engine already
/// returned first-pass speaker labels. The default non-diarized Rev path still
/// uses ASR labels directly; this helper only governs the opt-in `--diarize`
/// stage.
fn should_run_dedicated_speaker_diarization(
    response: &AsrResponse,
    speaker_backend: Option<SpeakerBackendV2>,
) -> bool {
    !response.tokens.is_empty() && speaker_backend.is_some()
}

fn stage_asr_infer<'a, 'ctx>(ctx: &'a mut TranscribePipelineContext<'ctx>) -> StageFuture<'a> {
    Box::pin(async move {
        info!(
            audio_path = %ctx.audio_path.display(),
            lang = %ctx.opts.lang,
            num_speakers = ctx.opts.num_speakers,
            "Starting ASR inference"
        );

        if let TranscribeEvidenceInput::LegacyReplay { replay } = &ctx.evidence_input {
            info!(
                recording_id = replay.recording_id(),
                manifest_blake3 = replay.manifest_blake3(),
                "Replaying fingerprinted legacy projected ASR evidence"
            );
            ctx.asr_response = Some(replay.asr_response().clone());
            if ctx.opts.diarize {
                let segments = replay.speaker_segments().ok_or_else(|| {
                    ServerError::Validation(format!(
                        "replay {} requested diarization but its manifest has no speaker-turn artifact",
                        replay.recording_id()
                    ))
                })?;
                ctx.speaker_segments = Some(segments.to_vec());
            }
            return Ok(());
        }

        let rev_inference = match &ctx.evidence_input {
            TranscribeEvidenceInput::Live { rev_inference } => *rev_inference,
            TranscribeEvidenceInput::LegacyReplay { .. } => {
                return Err(ServerError::Validation(
                    "legacy replay reached live ASR inference after replay admission".into(),
                ));
            }
        };
        let num_speakers = NumSpeakers(ctx.opts.num_speakers as u32);
        let response = if let Some(backend) = ctx.opts.backend.as_non_rev() {
            if ctx.opts.auto_speakers {
                return Err(ServerError::Validation("automatic speaker counts currently require the Rev.AI ASR engine".into()));
            }
            infer_asr(
                ctx.services.pool,
                &AsrInferParams {
                    backend,
                    audio_path: ctx.audio_path,
                    lang: &ctx.opts.lang,
                    num_speakers,
                    extras: &ctx.opts.engine_extras,
                },
            )
            .await?
        } else {
            let provider_media = PreparedRevProviderMedia::from_source(ctx.audio_path)
                .await
                .map_err(|error| ServerError::Persistence(error.to_string()))?;
            let request = RevAsrEvidenceRequest::new(
                provider_media,
                &ctx.opts.lang,
                ctx.opts.expected_speakers(),
                &RevAsrModelRevision::current(),
            )
            .map_err(|error| ServerError::Persistence(error.to_string()))?;
            let resolution = resolve_rev_asr_evidence(
                &request,
                ctx.services.cache,
                ctx.opts.cache_policies.rev_asr,
                rev_inference,
            )
            .await
            .map_err(rev_asr_resolution_error_to_server_error)?;
            let trace = resolution.trace(RevAsrProjectionRevision::AsrResponseV1);
            if ctx.dumper.is_enabled() {
                ctx.dumper
                    .dump_rev_evidence(ctx.audio_path.to_string_lossy().as_ref(), &trace)
                    .map_err(|error| ServerError::Persistence(error.to_string()))?;
            }
            ctx.rev_evidence = Some(trace);
            match resolution.source() {
                RevAsrEvidenceSource::Replayed => {
                    info!(
                        cache_key = %request.cache_key(),
                        "Replaying validated raw Rev.AI transcript evidence"
                    );
                }
                RevAsrEvidenceSource::Inferred(reason) => {
                    info!(
                        cache_key = %request.cache_key(),
                        reason = ?reason,
                        "Committed fresh raw Rev.AI transcript evidence"
                    );
                }
            }
            let evidence = resolution.into_evidence();
            rev_evidence_to_asr_response(&evidence)
        };
        let filename = ctx
            .audio_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        ctx.dumper.dump_asr_response(filename, &response);
        ctx.asr_response = Some(response);
        Ok(())
    })
}

fn stage_asr_postprocess<'a, 'ctx>(
    ctx: &'a mut TranscribePipelineContext<'ctx>,
) -> StageFuture<'a> {
    Box::pin(async move {
        let response = ctx.asr_response.as_ref().ok_or_else(|| {
            ServerError::Validation("ASR response missing before post-processing".to_string())
        })?;

        if response.tokens.is_empty() {
            return Ok(());
        }

        let asr_output = convert_asr_response(response);
        info!(
            num_tokens = response.tokens.len(),
            num_monologues = asr_output.monologues.len(),
            "ASR response received, starting post-processing"
        );

        let resolved_lang = resolved_asr_language(ctx.opts, response)?;
        ctx.resolved_lang = Some(resolved_lang.clone());
        let utterances =
            process_asr_with_prechat_segmentation(ctx, &asr_output, &resolved_lang).await?;
        info!(
            num_utterances = utterances.len(),
            "Post-processing complete, building CHAT"
        );

        // Optional diagnostic: when `BA3_DUMP_ASR_PIPELINE=/path/to/file.json`
        // is set, write the per-stage snapshot to disk for inspection.
        if let (Ok(path), Some(snapshot)) = (
            std::env::var("BA3_DUMP_ASR_PIPELINE"),
            ctx.asr_pipeline_snapshot.as_ref(),
        ) {
            let trace = crate::types::results::snapshot_into_pipeline_trace(snapshot.clone());
            if let Ok(json) = serde_json::to_string_pretty(&trace) {
                let _ = std::fs::write(&path, json);
                tracing::warn!(
                    path = %path,
                    "BA3_DUMP_ASR_PIPELINE wrote per-stage AsrPipelineTrace JSON",
                );
            }
        }

        ctx.utterances = Some(utterances);
        Ok(())
    })
}

/// Decide the post-ASR language used for CHAT headers and NLP dispatch.
///
/// Errors when the language cannot be honestly resolved:
///   - `Auto` with an ASR response that does not carry a usable language code.
///   - `PerFile` reaching transcribe at all (transcribe carries a real
///     `--lang`; submission validation must reject `PerFile` before this
///     code runs).
///
/// No silent fallback to English. CHAT files must declare a real
/// `@Languages:` value, and downstream NLP needs the real code; pretending
/// the language is English when it is not is exactly the kind of provenance
/// corruption the 2026-05-03 morphotag incident punished.
fn resolved_asr_language(
    opts: &TranscribeOptions,
    response: &AsrResponse,
) -> Result<LanguageCode3, ServerError> {
    match &opts.lang {
        LanguageSpec::Auto => {
            let detected = response.lang.clone();
            if &*detected == "auto" || detected.is_empty() {
                // ASR did not return a usable language. Try off-line detection
                // on the transcript text. If that also fails, error out, do
                // NOT silently stamp the file as English.
                let all_text: String = response
                    .tokens
                    .iter()
                    .map(|t| t.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                let detected_iso3 =
                    batchalign_transform::asr_postprocess::lang_detect::detect_primary_language(&[
                        &all_text,
                    ])
                    .ok_or_else(|| {
                        ServerError::Validation(
                            "ASR returned no language and offline detection on the transcript \
                             text failed; cannot stamp `@Languages:` honestly. Re-run with an \
                             explicit `--lang <iso3>` instead of `--lang auto`."
                                .into(),
                        )
                    })?;
                LanguageCode3::try_new(&detected_iso3).map_err(|err| {
                    ServerError::Validation(format!(
                        "offline language detection produced invalid ISO 639-3 code \
                         '{detected_iso3}': {err}",
                    ))
                })
            } else {
                Ok(detected)
            }
        }
        LanguageSpec::Resolved(code) => Ok(code.clone()),
        // Transcribe never legitimately carries `PerFile`. Submission
        // validation rejects `PerFile` for transcribe before this code runs;
        // if we ever land here it is a bug in submission validation, not a
        // user error. Surface a typed Validation error so the failure is
        // observable instead of pretending the language is English.
        LanguageSpec::PerFile => Err(ServerError::Validation(
            "transcribe pipeline received LanguageSpec::PerFile, which is reserved for \
             morphotag/translate/coref. This is a submission-validation bug, please \
             file a bug report."
                .into(),
        )),
    }
}

fn uses_prechat_utterance_model(lang: &LanguageCode3) -> bool {
    matches!(lang.as_ref(), "eng" | "cmn" | "zho" | "yue")
}

fn build_prechat_utseg_items(chunks: &[PreparedMonologueChunk]) -> Vec<UtsegBatchItem> {
    chunks
        .iter()
        .map(|chunk| {
            let words: Vec<String> = chunk
                .words
                .iter()
                .map(|word| word.text.as_str().to_string())
                .collect();
            UtsegBatchItem {
                text: words.join(" "),
                words,
            }
        })
        .collect()
}

fn apply_prechat_assignments(
    chunks: &[PreparedMonologueChunk],
    predictions: &[crate::utseg::AdmittedUtsegPrediction],
) -> Vec<PreparedMonologueChunk> {
    chunks
        .iter()
        .zip(predictions.iter())
        .flat_map(|(chunk, prediction)| {
            asr_postprocess::split_prepared_chunk_by_assignments(
                chunk,
                &prediction.response().assignments,
            )
        })
        .collect()
}

async fn process_asr_with_prechat_segmentation(
    ctx: &mut TranscribePipelineContext<'_>,
    asr_output: &batchalign_transform::asr_postprocess::AsrOutput,
    resolved_lang: &LanguageCode3,
) -> Result<Vec<Utterance>, ServerError> {
    let lang_str = resolved_lang.to_string();
    let project_speakers = |chunks| {
        let Some(segments) = ctx.speaker_segments.as_deref() else {
            return chunks;
        };
        let segments: Vec<ChatSpeakerSegment> = segments
            .iter()
            .map(|segment| ChatSpeakerSegment {
                start_ms: segment.start_ms.0,
                end_ms: segment.end_ms.0,
                speaker: segment.speaker.clone(),
            })
            .collect();
        let projection = project_speakers_onto_chunks(chunks, &segments);
        info!(
            contested_timed_words = projection.stats.contested_timed_words,
            unattested_timed_words = projection.stats.unattested_timed_words,
            speaker_boundaries = projection.stats.speaker_boundaries,
            "Projected diarization onto timed ASR words"
        );
        projection.chunks
    };
    let Some(pre_chat_policy) = ctx.utseg_execution.pre_chat_policy() else {
        let chunks = project_speakers(prepare_asr_chunks_with_snapshot(
            asr_output,
            &lang_str,
            ctx.asr_pipeline_snapshot.as_mut(),
        ));
        let mut utterances = asr_postprocess::utterances_from_prepared_chunks(chunks);
        asr_postprocess::finalize_utterances(&mut utterances, &lang_str);
        if let Some(s) = ctx.asr_pipeline_snapshot.as_mut() {
            s.final_utterances = utterances.clone();
        }
        return Ok(utterances);
    };

    if !uses_prechat_utterance_model(resolved_lang) {
        let chunks = project_speakers(prepare_asr_chunks_with_snapshot(
            asr_output,
            &lang_str,
            ctx.asr_pipeline_snapshot.as_mut(),
        ));
        let mut utterances = asr_postprocess::utterances_from_prepared_chunks(chunks);
        asr_postprocess::finalize_utterances(&mut utterances, &lang_str);
        if let Some(s) = ctx.asr_pipeline_snapshot.as_mut() {
            s.final_utterances = utterances.clone();
        }
        return Ok(utterances);
    }

    let prepared_chunks = project_speakers(prepare_asr_chunks_with_snapshot(
        asr_output,
        &lang_str,
        ctx.asr_pipeline_snapshot.as_mut(),
    ));
    if prepared_chunks.is_empty() {
        return Ok(Vec::new());
    }

    let items = build_prechat_utseg_items(&prepared_chunks);
    let predictions = crate::utseg::infer_utseg_predictions_with_policy(
        ctx.services.pool,
        resolved_lang,
        &items,
        ctx.opts.allow_stanza_fallback_utseg,
        pre_chat_policy,
        // `TranscribePipelineContext` has no job cancellation token wired
        // yet (see `stage_run_morphosyntax`'s identical note); genuinely
        // NotWired rather than a fabricated stand-in.
        crate::infer_retry::Cancellation::NotWired {
            reason: "TranscribePipelineContext has no job cancellation token wired yet",
        },
    )
    .await?;
    let split_chunks = apply_prechat_assignments(&prepared_chunks, &predictions);
    let indexed_items: Vec<_> = items.iter().cloned().enumerate().collect();
    let evidence = UtsegEvidenceTrace::from_predictions(
        UtsegEvidencePhase::PreChat,
        resolved_lang.as_ref(),
        ctx.services.engine_version.as_ref(),
        &indexed_items,
        &predictions,
    )
    .map_err(|error| ServerError::Validation(error.to_string()))?;
    ctx.utseg_evidence_sink
        .write(ctx.audio_path.to_string_lossy().as_ref(), &evidence)
        .map_err(|error| {
            ServerError::Persistence(format!(
                "could not retain requested pre-CHAT utseg evidence for {}: {error}",
                ctx.audio_path.display()
            ))
        })?;
    let mut utterances = asr_postprocess::utterances_from_prepared_chunks(split_chunks);
    asr_postprocess::finalize_utterances(&mut utterances, &lang_str);
    if let Some(s) = ctx.asr_pipeline_snapshot.as_mut() {
        s.final_utterances = utterances.clone();
    }
    Ok(utterances)
}

/// Prepare ASR chunks fully in Rust.
///
/// Stages 1-3 (compound merge, timed-word extraction, multi-word split) run per
/// monologue. Number expansion is then applied per word via
/// `asr_postprocess::expand_number`. After expansion a whitespace-split pass
/// widens multi-word expansions into separate tokens. Stages 4b-5b (Cantonese
/// normalization, long-turn / pause splitting) finalize per monologue.
#[allow(dead_code)]
fn prepare_asr_chunks(
    asr_output: &batchalign_transform::asr_postprocess::AsrOutput,
    lang: &str,
) -> Vec<PreparedMonologueChunk> {
    prepare_asr_chunks_with_snapshot(asr_output, lang, None)
}

/// Snapshot-aware variant of [`prepare_asr_chunks`].
///
/// When `snapshot` is `Some`, populates per-stage trace fields:
/// `raw_elements`, `after_compound_merge`, `after_timing_extract`,
/// `after_multiword_split`, `after_number_expand`,
/// `after_cantonese_norm` (yue only), `after_long_turn_split`. The
/// `final_utterances` field is filled by the caller after `retokenize`.
///
/// Multi-monologue inputs concatenate their stage outputs into the
/// snapshot's flat fields (other than `after_long_turn_split` which
/// is `Vec<Vec<...>>` and accumulates chunks in order).
fn prepare_asr_chunks_with_snapshot(
    asr_output: &batchalign_transform::asr_postprocess::AsrOutput,
    lang: &str,
    mut snapshot: Option<&mut AsrPipelineSnapshot>,
) -> Vec<PreparedMonologueChunk> {
    if let Some(ref mut s) = snapshot {
        for m in &asr_output.monologues {
            s.raw_elements.extend_from_slice(&m.elements);
        }
    }

    let mut monologue_words: Vec<(asr_postprocess::SpeakerIndex, Vec<AsrWord>)> = asr_output
        .monologues
        .iter()
        .map(|m| {
            let mut sub = AsrPipelineSnapshot::default();
            let cap = snapshot.is_some().then_some(&mut sub);
            let words =
                asr_postprocess::prepare_words_pre_expansion_with_snapshot(&m.elements, lang, cap);
            if let Some(ref mut s) = snapshot {
                s.after_compound_merge.extend(sub.after_compound_merge);
                s.after_timing_extract.extend(sub.after_timing_extract);
                s.after_multiword_split.extend(sub.after_multiword_split);
            }
            (m.speaker, words)
        })
        .collect();

    for (_speaker, words) in &mut monologue_words {
        for word in words.iter_mut() {
            let text = word.text.as_str();
            // Fast path: tokens with no ASCII digit can never expand
            // (every expander: NUM2LANG, num2chinese, currency,
            // ordinal/decade: requires a digit somewhere in the input).
            if !text.bytes().any(|b| b.is_ascii_digit()) {
                continue;
            }
            let expanded = asr_postprocess::expand_number(text, lang);
            if expanded != text {
                word.text = asr_postprocess::AsrNormalizedText::new(expanded);
            }
        }
        asr_postprocess::split_words_with_whitespace(words);
    }

    if let Some(ref mut s) = snapshot {
        for (_, words) in &monologue_words {
            s.after_number_expand.extend_from_slice(words);
        }
    }

    let mut prepared = Vec::new();
    for (speaker, words) in monologue_words {
        let mut sub = AsrPipelineSnapshot::default();
        let cap = snapshot.is_some().then_some(&mut sub);
        prepared.extend(asr_postprocess::finalize_words_to_chunks_with_snapshot(
            words, speaker, lang, cap,
        ));
        if let Some(ref mut s) = snapshot {
            if let Some(yue) = sub.after_cantonese_norm {
                s.after_cantonese_norm
                    .get_or_insert_with(Vec::new)
                    .extend(yue);
            }
            s.after_long_turn_split.extend(sub.after_long_turn_split);
        }
    }
    prepared
}

fn stage_speaker_diarization<'a, 'ctx>(
    ctx: &'a mut TranscribePipelineContext<'ctx>,
) -> StageFuture<'a> {
    Box::pin(async move {
        if matches!(
            ctx.evidence_input,
            TranscribeEvidenceInput::LegacyReplay { .. }
        ) {
            // The ASR stage admitted and installed the manifest-bound turns.
            // No live speaker-inference capability exists in this state.
            return Ok(());
        }
        let response = ctx.asr_response.as_ref().ok_or_else(|| {
            ServerError::Validation("ASR response missing before speaker diarization".to_string())
        })?;

        if !should_run_dedicated_speaker_diarization(response, ctx.opts.speaker_backend) {
            return Ok(());
        }

        // Control-flow invariant: `should_run_dedicated_speaker_diarization`
        // immediately above returns false when `ctx.opts.speaker_backend`
        // is `None`, taking the early-return branch. Reaching this
        // line therefore guarantees `Some(...)`.
        #[allow(clippy::expect_used)]
        let speaker_backend = ctx
            .opts
            .speaker_backend
            .expect("speaker backend presence checked above");

        // Speaker workers require a concrete routing language. `opts.lang`
        // remains `Auto` even after ASR, so derive a resolved value from the
        // response here rather than relying on a pipeline-order comment that
        // the type did not enforce.
        let speaker_worker_lang = WorkerLanguage::from(resolved_asr_language(ctx.opts, response)?);
        info!(
            audio_path = %ctx.audio_path.display(),
            speaker_backend = ?speaker_backend,
            num_speakers = ctx.opts.num_speakers,
            "Running dedicated speaker diarization"
        );
        let expected_speakers = ctx.opts.expected_speakers();
        let cache_policy = ctx.opts.cache_policies.speaker;
        let resolution = resolve_speaker_evidence_for_audio(
            ctx.services.pool,
            ctx.services.cache,
            speaker_worker_lang,
            SpeakerEvidenceRunParams {
                audio_path: ctx.audio_path,
                backend: speaker_backend,
                expected_speakers,
                cache_policy,
            },
        )
        .await?;
        let evidence_identity = ctx.audio_path.to_string_lossy().into_owned();
        let trace = resolution.trace();
        ctx.dumper
            .dump_speaker_evidence(&evidence_identity, &trace)
            .map_err(|error| {
                ServerError::Persistence(format!(
                    "could not retain requested speaker evidence trace for {evidence_identity}: {error}"
                ))
            })?;
        let num_segments = resolution.segments().len();
        match resolution.source() {
            SpeakerEvidenceSource::ReplayedDerived => {
                info!(
                    cache_key = %resolution.cache_key(),
                    num_segments,
                    "Replaying validated speaker diarization evidence"
                );
            }
            SpeakerEvidenceSource::DerivedFromRaw => {
                info!(
                    cache_key = %resolution.cache_key(),
                    num_segments,
                    "Derived speaker segments from retained raw evidence"
                );
            }
            SpeakerEvidenceSource::Inferred(reason) => {
                info!(
                    cache_key = %resolution.cache_key(),
                    reason = ?reason,
                    num_segments,
                    "Committed fresh speaker diarization evidence"
                );
            }
        }
        let segments = resolution.into_segments();
        info!(
            num_segments = segments.len(),
            "Speaker diarization complete"
        );
        ctx.dumper
            .dump_speaker_turns(&evidence_identity, speaker_backend, &segments)
            .map_err(|error| {
                ServerError::Persistence(format!(
                    "could not retain requested same-job diarization evidence for {evidence_identity}: {error}"
                ))
            })?;
        ctx.speaker_segments = Some(segments);
        Ok(())
    })
}

fn stage_build_chat<'a, 'ctx>(ctx: &'a mut TranscribePipelineContext<'ctx>) -> StageFuture<'a> {
    Box::pin(async move {
        let response = ctx.asr_response.as_ref().ok_or_else(|| {
            ServerError::Validation("ASR response missing before CHAT build".to_string())
        })?;

        // Resolve Auto → ASR-detected language for CHAT headers and NLP.
        // When the user passed --lang auto, opts.lang is Auto. The ASR
        // response carries the engine's detected language code (e.g. "spa").
        // Store the resolved language so post-ASR stages (utseg, morphotag)
        // use the real language, not Auto.
        let resolved_lang = resolved_asr_language(ctx.opts, response)?;
        ctx.resolved_lang = Some(resolved_lang.clone());

        if response.tokens.is_empty() {
            // Build empty CHAT with resolved language.
            let mut opts_resolved = ctx.opts.clone();
            opts_resolved.lang = LanguageSpec::Resolved(resolved_lang.clone());
            ctx.chat_text = Some(build_empty_chat_text(&opts_resolved)?);
            return Ok(());
        }

        let utterances = ctx.utterances.as_mut().ok_or_else(|| {
            ServerError::Validation("Utterances missing before CHAT build".to_string())
        })?;

        // When auto-detecting language, run per-utterance language detection
        // for code-switching markup and multi-language headers.
        let is_auto = matches!(&ctx.opts.lang, LanguageSpec::Auto);
        let langs: Vec<String> = if is_auto {
            use batchalign_transform::asr_postprocess::lang_detect;

            // Concatenate each utterance's words for language detection
            let utt_texts: Vec<String> = utterances
                .iter()
                .map(|utt| {
                    utt.words
                        .iter()
                        .map(|w| w.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect();
            let utt_text_refs: Vec<&str> = utt_texts.iter().map(String::as_str).collect();

            // Tag each utterance with its detected language
            for (utt, text) in utterances.iter_mut().zip(utt_text_refs.iter()) {
                utt.lang = lang_detect::detect_utterance_language(text);
            }

            // Collect all detected languages for @Languages header
            lang_detect::collect_detected_languages(&utt_text_refs, &resolved_lang)
        } else {
            vec![resolved_lang.to_string()]
        };

        let diarization_speaker_count = ctx
            .speaker_segments
            .as_deref()
            .map(unique_diarization_speaker_count)
            .unwrap_or(0);
        let participant_ids = generate_participant_ids(
            utterances,
            if ctx.opts.auto_speakers {
                diarization_speaker_count
            } else {
                ctx.opts.num_speakers.max(diarization_speaker_count)
            },
        );
        let transcript = build_chat::transcript_from_asr_utterances(
            utterances,
            &participant_ids,
            &langs,
            ctx.opts.media_name.as_deref(),
            ctx.opts.write_wor,
        )
        .map_err(|e| {
            ServerError::Validation(format!(
                "Failed to build transcript description \
                 (ASR token failed CHAT-legality): {e}"
            ))
        })?;

        // The gate hands back every token it refused and we emitted anyway.
        // Saying so here is the point: an operator reading this run now learns
        // that the transcript is knowingly invalid CHAT and why, instead of
        // finding out from `chatter validate` days later with no way to tell
        // an ASR artifact from a transcription error. The tokens themselves
        // stay verbatim, which is this pipeline's standing policy: what the
        // speaker actually said is a human's call, not the pipeline's.
        report_language_invalid_words(ctx, &transcript.language_invalid);
        let desc = transcript.description;

        let mut chat_file = build_chat::build_chat(&desc)
            .map_err(|e| ServerError::Validation(format!("Failed to build CHAT: {e}")))?;
        // Inject processing provenance comment.
        let asr_engine = ctx.asr_provenance_name();
        let provenance = crate::provenance::transcribe_provenance(
            resolved_lang.as_ref(),
            asr_engine,
            ctx.opts.diarize,
            ctx.opts.write_wor,
        );
        crate::provenance::inject_provenance(&mut chat_file, &provenance);

        // Inject human-readable "unchecked ASR" warning (a user's workflow depends on this).
        crate::provenance::inject_unchecked_warning(&mut chat_file, asr_engine);

        let chat_text = to_chat_string(&chat_file);
        let filename = ctx
            .audio_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        ctx.dumper.dump_post_asr_chat(filename, &chat_text);
        ctx.chat_text = Some(chat_text);
        Ok(())
    })
}

/// One refused token, in the shape written to the run's debug artifacts.
#[derive(serde::Serialize)]
struct LanguageInvalidWordRecord<'a> {
    utterance: usize,
    word: usize,
    speaker: &'a str,
    /// The provider's surface, verbatim, exactly as it was emitted.
    text: &'a str,
    lang: &'a str,
    /// chatter error codes, e.g. `E220` (digits) or `E241` (reserved
    /// untranscribed marker).
    codes: Vec<&'a str>,
    messages: Vec<&'a str>,
}

/// Narrate, and durably record, every word the CHAT-legality gate refused.
///
/// Deliberately not fatal: emitting the provider's surface for human review is
/// the pipeline's standing policy. What was missing until 2026-09-03 was any
/// record at all, so `Www` (E241) and `b2` (E220) reached a corpus with the
/// only evidence in a log line nobody was reading.
fn report_language_invalid_words(
    ctx: &TranscribePipelineContext<'_>,
    refused: &[build_chat::LanguageInvalidWord],
) {
    if refused.is_empty() {
        return;
    }

    let records: Vec<LanguageInvalidWordRecord<'_>> = refused
        .iter()
        .map(|word| LanguageInvalidWordRecord {
            utterance: word.utt_idx,
            word: word.word_idx,
            speaker: word.speaker_id.as_str(),
            text: word.text.as_str(),
            lang: word.lang.as_str(),
            codes: word
                .parse_errors
                .iter()
                .map(|error| error.code.as_str())
                .collect(),
            messages: word
                .parse_errors
                .iter()
                .map(|error| error.message.as_str())
                .collect(),
        })
        .collect();

    let mut codes: Vec<&str> = records
        .iter()
        .flat_map(|r| r.codes.iter().copied())
        .collect();
    codes.sort_unstable();
    codes.dedup();

    let filename = ctx
        .audio_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown");
    tracing::warn!(
        %filename,
        refused_words = records.len(),
        codes = %codes.join(","),
        "transcript emitted with words the CHAT language rules refuse; \
         surfaces kept verbatim for human review"
    );
    ctx.dumper.dump_language_invalid_words(filename, &records);
}

fn unique_diarization_speaker_count(segments: &[SpeakerSegmentV2]) -> usize {
    let mut seen: Vec<&str> = Vec::new();
    for segment in segments {
        if !seen.contains(&segment.speaker.as_str()) {
            seen.push(segment.speaker.as_str());
        }
    }
    seen.len()
}

fn stage_run_utseg<'a, 'ctx>(ctx: &'a mut TranscribePipelineContext<'ctx>) -> StageFuture<'a> {
    Box::pin(async move {
        info!("Running utterance segmentation");
        let input = ctx
            .chat_text
            .as_deref()
            .ok_or_else(|| ServerError::Validation("CHAT text missing before utseg".to_string()))?;
        let filename = ctx
            .audio_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        ctx.dumper.dump_pre_utseg_chat(filename, input);
        let utseg_lang = ctx.lang_for_nlp()?.clone();
        let evidence_filename = ctx.audio_path.to_string_lossy();
        let post_chat_policy = ctx.utseg_execution.post_chat_policy().ok_or_else(|| {
            ServerError::Validation(
                "post-CHAT utseg stage exists without a post-CHAT execution policy".into(),
            )
        })?;
        let result = crate::utseg::process_utseg_with_evidence(
            crate::utseg::EvidenceRetainingUtsegRequest {
                chat_text: ChatText::from(input),
                lang: &utseg_lang,
                services: ctx.services,
                fallback_policy: UtsegFallbackPolicy::from(ctx.opts.allow_stanza_fallback_utseg),
                decision_policy: post_chat_policy,
                evidence_filename: evidence_filename.as_ref(),
                evidence_sink: &ctx.utseg_evidence_sink,
                cancellation: crate::infer_retry::Cancellation::NotWired {
                    reason: "TranscribePipelineContext has no job cancellation token wired yet",
                },
            },
        )
        .await?;
        ctx.dumper.dump_post_utseg_chat(filename, &result);
        ctx.chat_text = Some(result);
        Ok(())
    })
}

fn stage_run_morphosyntax<'a, 'ctx>(
    ctx: &'a mut TranscribePipelineContext<'ctx>,
) -> StageFuture<'a> {
    Box::pin(async move {
        info!("Running morphosyntax");
        let input = ctx.chat_text.as_deref().ok_or_else(|| {
            ServerError::Validation("CHAT text missing before morphosyntax".to_string())
        })?;
        let filename = ctx
            .audio_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");
        ctx.dumper.dump_pre_morphosyntax_chat(filename, input);
        let empty_mwt = std::collections::BTreeMap::new();
        let mor_lang = ctx.lang_for_nlp()?.clone();
        let mor_params = MorphosyntaxParams {
            lang: &mor_lang,
            tokenization_mode: TokenizationMode::Preserve,
            multilingual_policy: MultilingualPolicy::ProcessAll,
            mwt: &empty_mwt,
            policy: crate::params::MorphotagExecutionPolicy {
                l2: crate::params::L2MorphotagPolicy::Placeholder,
                pos_hints: crate::params::PosHintPolicy::Ignore,
                ca_policy: crate::options::CaMorphotagPolicy::Honor,
            },
            // Transcribe's morphotag sub-step never surfaces review tiers.
            review_level: crate::chat_ops::fa::ReviewLevel::None,
            // No job-level reporter on this path: see the field doc.
            progress: None,
            // `TranscribePipelineContext` does not carry the job's
            // cancellation token yet (see `process_one_transcribe_file`,
            // which has one, and the several wrapper layers between it and
            // here that would need a new field each); genuinely NotWired
            // rather than a fabricated stand-in. Follow-up: thread a real
            // token through `TranscribePipelineContext` the same way
            // `services`/`opts` are threaded.
            cancellation: crate::infer_retry::Cancellation::NotWired {
                reason: "TranscribePipelineContext has no job cancellation token wired yet",
            },
        };
        ctx.chat_text = Some(
            crate::morphosyntax::process_morphosyntax(input, ctx.services, &mor_params)
                .await?
                // The transcribe pipeline threads CHAT text between stages and
                // gates its own final output; the morphotag proof is
                // discharged into that text here.
                .into_text(),
        );
        Ok(())
    })
}

fn stage_serialize<'a, 'ctx>(ctx: &'a mut TranscribePipelineContext<'ctx>) -> StageFuture<'a> {
    Box::pin(async move {
        if ctx.chat_text.is_none() {
            return Err(ServerError::Validation(
                "CHAT text missing before serialize".to_string(),
            ));
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{DurationSeconds, EngineVersion};
    use crate::cache::UtteranceCache;
    use crate::revai::{
        AuthorizedRevEvidenceRun, CompletedRevAsrEvidence, RevAsrEvidenceCacheOutcome,
        RevAsrEvidenceInference, RevTranscriptEvidence,
    };
    use crate::transcribe::replay::{
        LegacyProjectedAsrProducer, LegacyReplayManifestRequest, admit_legacy_replay_manifest,
        write_legacy_replay_manifest,
    };
    use crate::transcribe::{AsrBackend, AsrToken};
    use crate::types::worker_v2::SpeakerBackendV2;
    use crate::worker::pool::{PoolConfig, WorkerPool};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn transcribe_stage_progress_labels_are_stable() {
        assert_eq!(
            progress_stage_for_stage(StageId::AsrInfer),
            FileStage::Transcribing
        );
        assert_eq!(
            progress_stage_for_stage(StageId::SpeakerDiarization),
            FileStage::PostProcessing
        );
        assert_eq!(
            progress_stage_for_stage(StageId::AsrPostprocess),
            FileStage::PostProcessing
        );
        assert_eq!(
            progress_stage_for_stage(StageId::BuildChat),
            FileStage::BuildingChat
        );
        assert_eq!(
            progress_stage_for_stage(StageId::OptionalUtseg),
            FileStage::SegmentingUtterances
        );
        assert_eq!(
            progress_stage_for_stage(StageId::OptionalMorphosyntax),
            FileStage::AnalyzingMorphosyntax
        );
        assert_eq!(
            progress_stage_for_stage(StageId::Serialize),
            FileStage::Finalizing
        );
    }

    #[test]
    fn prechat_utterance_models_cover_documented_supported_codes() {
        assert!(uses_prechat_utterance_model(&LanguageCode3::eng()));
        let cmn = LanguageCode3::try_new("cmn").expect("cmn should be a valid ISO-639-3 code");
        assert!(uses_prechat_utterance_model(&cmn));
        assert!(uses_prechat_utterance_model(&LanguageCode3::zho()));
        assert!(uses_prechat_utterance_model(&LanguageCode3::yue()));
    }

    fn test_transcribe_options(speaker_backend: Option<SpeakerBackendV2>) -> TranscribeOptions {
        TranscribeOptions {
            auto_speakers: false,
            backend: AsrBackend::RustRevAi,
            diarize: true,
            speaker_backend,
            lang: LanguageCode3::eng().into(),
            num_speakers: 2,
            with_utseg: false,
            with_morphosyntax: false,
            cache_policies: crate::transcribe::TranscribeCachePolicies::uniform(
                crate::params::CachePolicy::UseCache,
            ),
            allow_stanza_fallback_utseg: false,
            write_wor: false,
            media_name: Some("sample".into()),
            engine_extras: std::collections::BTreeMap::new(),
        }
    }

    struct CountingRevInference {
        calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl RevAsrEvidenceInference for CountingRevInference {
        async fn infer(
            &self,
            _run: AuthorizedRevEvidenceRun,
        ) -> Result<crate::revai::RevAsrInferenceOutcome, ServerError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(crate::revai::RevAsrInferenceOutcome::Completed(CompletedRevAsrEvidence {
                transcript_evidence: RevTranscriptEvidence::from_provider_json(
                    r#"{"monologues":[{"speaker":0,"elements":[{"type":"text","value":"hello","ts":0.1,"end_ts":0.5,"confidence":0.9},{"type":"punct","value":".","ts":null,"end_ts":null,"confidence":null}]},{"speaker":1,"elements":[{"type":"text","value":"there","ts":0.6,"end_ts":1.0,"confidence":0.8},{"type":"punct","value":"?","ts":null,"end_ts":null,"confidence":null}]}]}"#
                        .to_owned(),
                )
                .expect("valid provider transcript fixture"),
                resolved_language: LanguageCode3::eng(),
            }))
        }
    }

    fn only_debug_artifact(dir: &Path, suffix: &str) -> std::path::PathBuf {
        let matches = std::fs::read_dir(dir)
            .expect("read debug directory")
            .map(|entry| entry.expect("debug directory entry").path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with(suffix))
            })
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 1, "expected one {suffix} artifact");
        matches.into_iter().next().expect("one debug artifact")
    }

    /// Assert that two transcribe outputs differ, if at all, only in the
    /// execution timestamp of an otherwise identical provenance receipt.
    fn assert_same_transcribe_semantics(left: &str, right: &str) {
        let left_provenance = crate::provenance::extract_provenance(left);
        let right_provenance = crate::provenance::extract_provenance(right);
        assert_eq!(
            left_provenance.len(),
            1,
            "expected one left provenance receipt"
        );
        assert_eq!(
            right_provenance.len(),
            1,
            "expected one right provenance receipt"
        );
        assert_eq!(left_provenance[0].command, "transcribe");
        assert_eq!(right_provenance[0].command, "transcribe");
        assert_eq!(left_provenance[0].fields, right_provenance[0].fields);
        assert!(
            left == right
                || crate::provenance::is_provenance_only_difference(
                    left,
                    right,
                    crate::api::ReleasedCommand::Transcribe,
                ),
            "transcribe outputs differ outside the execution timestamp"
        );
    }

    /// A durable Rev cache hit must replay the same evidence through the full
    /// Rust post-processing and CHAT construction path. The causal receipt is
    /// intentionally different (`inferred_not_found` versus `replayed`), but
    /// its typed semantic projection and every downstream output are stable.
    #[tokio::test]
    async fn rev_cold_and_replayed_transcribe_are_semantically_identical() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio_path = tempdir.path().join("sample.wav");
        tokio::fs::write(&audio_path, b"provider media")
            .await
            .expect("write provider media");
        let cache_dir = tempdir.path().join("cache");
        let cold_debug = tempdir.path().join("cold-debug");
        let replay_debug = tempdir.path().join("replay-debug");
        let pool = WorkerPool::new(PoolConfig::default());
        let engine_version = EngineVersion::from("test-asr");
        let inference = CountingRevInference {
            calls: AtomicUsize::new(0),
        };
        let mut opts = test_transcribe_options(None);
        opts.diarize = false;

        let cache = UtteranceCache::sqlite(Some(cache_dir.clone()))
            .await
            .expect("cold cache");
        let cold = run_transcribe_pipeline_with_rev_inference(
            &audio_path,
            PipelineServices::new(&pool, &cache, &engine_version),
            &opts,
            None,
            Some(&cold_debug),
            &inference,
        )
        .await
        .expect("cold transcribe");
        drop(cache);

        let reopened = UtteranceCache::sqlite(Some(cache_dir))
            .await
            .expect("reopened cache");
        let replayed = run_transcribe_pipeline_with_rev_inference(
            &audio_path,
            PipelineServices::new(&pool, &reopened, &engine_version),
            &opts,
            None,
            Some(&replay_debug),
            &inference,
        )
        .await
        .expect("replayed transcribe");

        assert_eq!(inference.calls.load(Ordering::SeqCst), 1);
        assert_same_transcribe_semantics(&cold.chat_text, &replayed.chat_text);
        assert_eq!(
            std::fs::read(only_debug_artifact(&cold_debug, "_asr_response.json"))
                .expect("cold ASR artifact"),
            std::fs::read(only_debug_artifact(&replay_debug, "_asr_response.json"))
                .expect("replayed ASR artifact")
        );

        let cold_trace = cold.rev_evidence.expect("cold Rev trace");
        let replay_trace = replayed.rev_evidence.expect("replayed Rev trace");
        assert_eq!(
            cold_trace.cache_outcome(),
            RevAsrEvidenceCacheOutcome::InferredNotFound
        );
        assert_eq!(
            replay_trace.cache_outcome(),
            RevAsrEvidenceCacheOutcome::Replayed
        );
        assert_eq!(
            cold_trace.semantic_projection(),
            replay_trace.semantic_projection()
        );
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(
                &std::fs::read(only_debug_artifact(&cold_debug, "_rev_evidence.json"))
                    .expect("cold Rev trace artifact")
            )
            .expect("cold Rev trace JSON"),
            serde_json::to_value(&cold_trace).expect("cold typed trace JSON")
        );
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(
                &std::fs::read(only_debug_artifact(&replay_debug, "_rev_evidence.json"))
                    .expect("replayed Rev trace artifact")
            )
            .expect("replayed Rev trace JSON"),
            serde_json::to_value(&replay_trace).expect("replayed typed trace JSON")
        );
    }

    /// A projected-evidence replay enters the current word-level speaker
    /// projection and CHAT builder without possessing a live provider
    /// capability. This is the end-to-end guard for the research replay path,
    /// not merely a manifest-parser test.
    #[tokio::test]
    async fn legacy_replay_runs_current_speaker_projection_without_live_inference() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let media = tempdir.path().join("sample.wav");
        let asr = tempdir.path().join("sample_asr_response.json");
        let turns = tempdir.path().join("sample.turns.json");
        let manifest = tempdir.path().join("sample.replay.json");
        std::fs::write(&media, b"fingerprinted media").expect("media");
        std::fs::write(
            &asr,
            serde_json::to_vec_pretty(&AsrResponse {
                tokens: vec![
                    AsrToken {
                        text: "bonjour".into(),
                        start_s: Some(DurationSeconds(0.0)),
                        end_s: Some(DurationSeconds(0.5)),
                        speaker: Some("ASR_0".into()),
                        confidence: Some(0.9),
                    },
                    AsrToken {
                        text: "oui".into(),
                        start_s: Some(DurationSeconds(0.5)),
                        end_s: Some(DurationSeconds(1.0)),
                        speaker: Some("ASR_0".into()),
                        confidence: Some(0.8),
                    },
                    AsrToken {
                        text: ".".into(),
                        start_s: None,
                        end_s: None,
                        speaker: Some("ASR_0".into()),
                        confidence: None,
                    },
                ],
                lang: LanguageCode3::fra(),
                source_monologues: None,
            })
            .expect("ASR JSON"),
        )
        .expect("ASR artifact");
        std::fs::write(
            &turns,
            br#"{"source":"batchalign3:pyannote_ai:precision-2","turns":[{"track":"PAR0","start_ms":0,"end_ms":500},{"track":"PAR1","start_ms":500,"end_ms":1000}]}"#,
        )
        .expect("turns");
        write_legacy_replay_manifest(
            LegacyReplayManifestRequest {
                recording_id: "sample",
                media_path: &media,
                asr_response_path: &asr,
                speaker_turns_path: Some(&turns),
                producer: LegacyProjectedAsrProducer::RevAi,
            },
            &manifest,
        )
        .expect("manifest");
        let replay = admit_legacy_replay_manifest(&manifest).expect("admitted replay");

        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let pool = WorkerPool::new(PoolConfig::default());
        let engine_version = EngineVersion::from("test-replay");
        let mut opts = test_transcribe_options(Some(SpeakerBackendV2::PyannoteAi));
        opts.lang = LanguageCode3::fra().into();
        let chat = run_transcribe_pipeline_with_legacy_replay(
            replay,
            TranscribeUtsegExecution::production(opts.with_utseg),
            PipelineServices::new(&pool, &cache, &engine_version),
            &opts,
            None,
            None,
        )
        .await
        .expect("offline replay");

        assert!(chat.contains("*PAR0:\tbonjour ."), "{chat}");
        assert!(chat.contains("*PAR1:\toui ."), "{chat}");
        assert!(chat.contains("asr=rev"), "{chat}");
    }

    #[test]
    fn dedicated_speaker_diarization_runs_when_backend_is_available_even_if_asr_has_labels() {
        let response = AsrResponse {
            tokens: vec![AsrToken {
                text: "hello".into(),
                start_s: Some(DurationSeconds(0.0)),
                end_s: Some(DurationSeconds(0.5)),
                speaker: Some("SPEAKER_1".into()),
                confidence: None,
            }],
            lang: LanguageCode3::eng(),
            source_monologues: None,
        };

        assert!(
            should_run_dedicated_speaker_diarization(&response, Some(SpeakerBackendV2::Pyannote)),
            "explicit diarization should still run even when ASR already carries first-pass speaker labels"
        );
    }

    #[test]
    fn dedicated_speaker_diarization_skips_when_response_is_empty() {
        let response = AsrResponse {
            tokens: vec![],
            lang: LanguageCode3::eng(),
            source_monologues: None,
        };

        assert!(
            !should_run_dedicated_speaker_diarization(&response, Some(SpeakerBackendV2::Pyannote)),
            "empty ASR responses should not trigger dedicated speaker diarization"
        );
    }

    #[tokio::test]
    async fn speaker_diarization_stage_skips_when_backend_is_unavailable() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let pool = WorkerPool::new(PoolConfig::default());
        let engine_version = EngineVersion::from("test-asr");
        let services = PipelineServices::new(&pool, &cache, &engine_version);
        let audio_path = tempdir.path().join("sample.wav");
        let opts = test_transcribe_options(None);
        let mut ctx = TranscribePipelineContext::new_with_rev_inference(
            &audio_path,
            services,
            &opts,
            DebugDumper::disabled(),
            UtsegEvidenceSink::Disabled,
            &PRODUCTION_REV_INFERENCE,
        );
        ctx.asr_response = Some(AsrResponse {
            tokens: vec![AsrToken {
                text: "hello".into(),
                start_s: Some(DurationSeconds(0.0)),
                end_s: Some(DurationSeconds(0.5)),
                speaker: None,
                confidence: None,
            }],
            lang: LanguageCode3::eng(),
            source_monologues: None,
        });

        stage_speaker_diarization(&mut ctx)
            .await
            .expect("speaker stage should succeed");

        assert!(
            ctx.speaker_segments.is_none(),
            "dedicated speaker inference should be skipped when no speaker backend is configured"
        );
    }

    /// `--no-utseg` controls the whole transcribe segmentation execution, not
    /// only the optional CHAT-level stage. English is deliberately used here:
    /// with segmentation enabled these two words require a worker request, and
    /// this pool has no worker to satisfy one.
    #[tokio::test]
    async fn no_utseg_bypasses_the_pre_chat_worker_for_a_supported_language() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let pool = WorkerPool::new(PoolConfig::default());
        let engine_version = EngineVersion::from("test-asr");
        let services = PipelineServices::new(&pool, &cache, &engine_version);
        let audio_path = tempdir.path().join("sample.wav");
        let mut opts = test_transcribe_options(None);
        opts.diarize = false;
        opts.with_utseg = false;
        let mut ctx = TranscribePipelineContext::new_with_rev_inference(
            &audio_path,
            services,
            &opts,
            DebugDumper::disabled(),
            UtsegEvidenceSink::Disabled,
            &PRODUCTION_REV_INFERENCE,
        );
        ctx.asr_response = Some(AsrResponse {
            tokens: vec![
                AsrToken {
                    text: "hello".into(),
                    start_s: Some(DurationSeconds(0.0)),
                    end_s: Some(DurationSeconds(0.5)),
                    speaker: None,
                    confidence: None,
                },
                AsrToken {
                    text: "there".into(),
                    start_s: Some(DurationSeconds(0.5)),
                    end_s: Some(DurationSeconds(1.0)),
                    speaker: None,
                    confidence: None,
                },
            ],
            lang: LanguageCode3::eng(),
            source_monologues: None,
        });

        stage_asr_postprocess(&mut ctx)
            .await
            .expect("disabled segmentation must not dispatch a worker");

        assert_eq!(ctx.utterances.as_ref().map(Vec::len), Some(1));
    }

    /// Dedicated diarization is available while ASR words still carry their
    /// observed timings. A speaker boundary between two words must therefore
    /// constrain utterance segmentation before CHAT is built, rather than
    /// relabeling the already-mixed utterance afterward.
    #[tokio::test]
    async fn diarization_boundary_splits_timed_asr_words_before_chat_build() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let pool = WorkerPool::new(PoolConfig::default());
        let engine_version = EngineVersion::from("test-asr");
        let services = PipelineServices::new(&pool, &cache, &engine_version);
        let audio_path = tempdir.path().join("sample.wav");
        let mut opts = test_transcribe_options(Some(SpeakerBackendV2::Pyannote));
        opts.lang = LanguageCode3::fra().into();

        let mut ctx = TranscribePipelineContext::new_with_rev_inference(
            &audio_path,
            services,
            &opts,
            DebugDumper::disabled(),
            UtsegEvidenceSink::Disabled,
            &PRODUCTION_REV_INFERENCE,
        );
        ctx.asr_response = Some(AsrResponse {
            tokens: vec![
                AsrToken {
                    text: "bonjour".into(),
                    start_s: Some(DurationSeconds(0.0)),
                    end_s: Some(DurationSeconds(0.5)),
                    speaker: Some("ASR_0".into()),
                    confidence: None,
                },
                AsrToken {
                    text: "oui".into(),
                    start_s: Some(DurationSeconds(0.5)),
                    end_s: Some(DurationSeconds(1.0)),
                    speaker: Some("ASR_0".into()),
                    confidence: None,
                },
                AsrToken {
                    text: ".".into(),
                    start_s: None,
                    end_s: None,
                    speaker: Some("ASR_0".into()),
                    confidence: None,
                },
            ],
            lang: LanguageCode3::fra(),
            source_monologues: None,
        });
        ctx.speaker_segments = Some(vec![
            SpeakerSegmentV2 {
                start_ms: crate::api::DurationMs(0),
                end_ms: crate::api::DurationMs(500),
                speaker: "HUMAN_A".into(),
            },
            SpeakerSegmentV2 {
                start_ms: crate::api::DurationMs(500),
                end_ms: crate::api::DurationMs(1_000),
                speaker: "HUMAN_B".into(),
            },
        ]);

        stage_asr_postprocess(&mut ctx).await.expect("postprocess");
        stage_build_chat(&mut ctx).await.expect("build chat");

        let chat = ctx.chat_text.expect("CHAT output");
        assert!(chat.contains("*PAR0:\tbonjour ."), "{chat}");
        assert!(chat.contains("*PAR1:\toui ."), "{chat}");
        assert_eq!(chat.lines().filter(|line| line.starts_with('*')).count(), 2);
    }

    /// When opts.lang is "auto", stage_build_chat must resolve to the
    /// ASR-detected language for CHAT headers (regression test for job
    /// 696870c7-02b where `@Languages: auto` leaked into output).
    #[tokio::test]
    async fn build_chat_stage_resolves_auto_to_detected_language() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let pool = WorkerPool::new(PoolConfig::default());
        let engine_version = EngineVersion::from("test-asr");
        let services = PipelineServices::new(&pool, &cache, &engine_version);
        let audio_path = tempdir.path().join("sample.wav");

        // Opts with lang="auto": simulates --lang auto from CLI
        let mut opts = test_transcribe_options(None);
        opts.lang = LanguageSpec::Auto;
        opts.diarize = false;

        let mut ctx = TranscribePipelineContext::new_with_rev_inference(
            &audio_path,
            services,
            &opts,
            DebugDumper::disabled(),
            UtsegEvidenceSink::Disabled,
            &PRODUCTION_REV_INFERENCE,
        );

        // ASR response with detected language "spa"
        ctx.asr_response = Some(AsrResponse {
            tokens: vec![AsrToken {
                text: "hola".into(),
                start_s: Some(DurationSeconds(0.0)),
                end_s: Some(DurationSeconds(0.5)),
                speaker: None,
                confidence: None,
            }],
            lang: LanguageCode3::spa(),
            source_monologues: None,
        });

        // Run post-processing to generate utterances
        stage_asr_postprocess(&mut ctx).await.expect("postprocess");

        // Run build_chat; this should resolve "auto" → "spa"
        stage_build_chat(&mut ctx).await.expect("build_chat");

        let chat_text = ctx.chat_text.as_deref().expect("CHAT text should be set");

        // The @Languages header must contain the detected language, NOT "auto"
        let languages_line = chat_text
            .lines()
            .find(|l| l.starts_with("@Languages:"))
            .expect("@Languages header missing");
        assert!(
            languages_line.contains("spa"),
            "@Languages should contain detected 'spa', got: {languages_line}"
        );
        assert!(
            !languages_line.contains("auto"),
            "@Languages must NOT contain sentinel 'auto', got: {languages_line}"
        );
    }

    #[tokio::test]
    async fn postprocess_stage_resolves_auto_before_chat_build() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let pool = WorkerPool::new(PoolConfig::default());
        let engine_version = EngineVersion::from("test-asr");
        let services = PipelineServices::new(&pool, &cache, &engine_version);
        let audio_path = tempdir.path().join("sample.wav");

        let mut opts = test_transcribe_options(None);
        opts.lang = LanguageSpec::Auto;
        opts.diarize = false;

        let mut ctx = TranscribePipelineContext::new_with_rev_inference(
            &audio_path,
            services,
            &opts,
            DebugDumper::disabled(),
            UtsegEvidenceSink::Disabled,
            &PRODUCTION_REV_INFERENCE,
        );
        ctx.asr_response = Some(AsrResponse {
            tokens: vec![AsrToken {
                text: "hola".into(),
                start_s: Some(DurationSeconds(0.0)),
                end_s: Some(DurationSeconds(0.5)),
                speaker: None,
                confidence: None,
            }],
            lang: LanguageCode3::spa(),
            source_monologues: None,
        });

        stage_asr_postprocess(&mut ctx).await.expect("postprocess");
        assert_eq!(ctx.resolved_lang, Some(LanguageCode3::spa()));
    }

    /// When opts.lang is "auto" and ASR returns empty tokens,
    /// build_chat should still resolve to the ASR response language.
    #[tokio::test]
    async fn build_chat_stage_resolves_auto_for_empty_response() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let pool = WorkerPool::new(PoolConfig::default());
        let engine_version = EngineVersion::from("test-asr");
        let services = PipelineServices::new(&pool, &cache, &engine_version);
        let audio_path = tempdir.path().join("sample.wav");

        let mut opts = test_transcribe_options(None);
        opts.lang = LanguageSpec::Auto;

        let mut ctx = TranscribePipelineContext::new_with_rev_inference(
            &audio_path,
            services,
            &opts,
            DebugDumper::disabled(),
            UtsegEvidenceSink::Disabled,
            &PRODUCTION_REV_INFERENCE,
        );
        ctx.asr_response = Some(AsrResponse {
            tokens: vec![],
            lang: LanguageCode3::fra(),
            source_monologues: None,
        });

        stage_build_chat(&mut ctx).await.expect("build_chat");

        let chat_text = ctx.chat_text.as_deref().expect("CHAT text should be set");
        let languages_line = chat_text
            .lines()
            .find(|l| l.starts_with("@Languages:"))
            .expect("@Languages header missing");
        assert!(
            languages_line.contains("fra"),
            "empty-response @Languages should contain 'fra', got: {languages_line}"
        );
        assert!(
            !languages_line.contains("auto"),
            "empty-response @Languages must NOT contain 'auto', got: {languages_line}"
        );
    }
}
