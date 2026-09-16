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
use crate::error::{EmptyTranscription, ServerError};
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
    TranscribeOptions, convert_asr_response, infer_asr, resolve_speaker_evidence_for_audio,
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
    /// Language resolved after ASR, by whichever post-ASR stage reaches
    /// [`TranscribePipelineContext::resolve_lang`] first. Every later stage,
    /// `lang_for_nlp` included, reads what that one recorded.
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

    /// The post-ASR language, resolved ONCE per file.
    ///
    /// Three stages need it (post-processing, dedicated diarization, CHAT
    /// assembly) and each used to resolve it for itself, two of them writing
    /// the answer back into this context separately. Under `--lang auto` with
    /// an engine that reported no language, resolving joins every token in the
    /// transcript into one string and runs offline detection over the join, so
    /// a file paid for that up to three times and could in principle record
    /// three different answers. The first caller resolves and records; the rest
    /// read the record.
    fn resolve_lang(&mut self) -> Result<LanguageCode3, ServerError> {
        if let Some(resolved) = &self.resolved_lang {
            return Ok(resolved.clone());
        }
        let resolved = {
            let response = self.asr_response.as_ref().ok_or_else(|| {
                ServerError::Validation(
                    "ASR response missing before the transcript language could be resolved"
                        .to_string(),
                )
            })?;
            resolved_asr_language(self.opts, response)?
        };
        self.resolved_lang = Some(resolved.clone());
        Ok(resolved)
    }

    /// The ASR identity transcript provenance records: the engine, plus the
    /// checkpoint when the request selected one through the engine's override
    /// key, so a Paraformer run no longer reads as plain `funaudio`.
    fn asr_identity(&self) -> crate::transcribe::types::AsrIdentity {
        match &self.evidence_input {
            TranscribeEvidenceInput::LegacyReplay { replay, .. } => {
                crate::transcribe::types::AsrIdentity::of_replay(replay.producer())
            }
            TranscribeEvidenceInput::Live { .. } => {
                let planned = self.opts.asr.identity();
                // The response is the only witness to what actually loaded, and
                // it is populated well before CHAT is built, so the stamp names
                // the models that ran rather than the ones the plan hoped for.
                match self.asr_response.as_ref().and_then(|r| r.model.clone()) {
                    Some(model) => planned.with_loaded_models(model),
                    None => planned,
                }
            }
        }
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
            expected_speakers = ?ctx.opts.expected_speakers(),
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
        let response = match ctx.opts.asr {
            crate::transcribe::TranscribeAsrPlan::NonRev {
                backend, speakers, ..
            } => {
                infer_asr(
                    ctx.services.pool,
                    &AsrInferParams {
                        backend,
                        audio_path: ctx.audio_path,
                        lang: &ctx.opts.lang,
                        num_speakers: NumSpeakers(speakers.get()),
                        extras: &ctx.opts.engine_extras,
                    },
                )
                .await?
            }
            crate::transcribe::TranscribeAsrPlan::RevAi(_) => {
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
            }
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

        // Nothing to post-process means nothing to transcribe. This returned
        // `Ok(())` until 2026-09-16, and `stage_build_chat` then wrote a
        // headers-only CHAT file, so a run that recognized not one word
        // finished as a completed job carrying an empty transcript. Silence
        // reported as success is the defect; the job now says which stage came
        // up empty.
        if response.tokens.is_empty() {
            return Err(ServerError::EmptyTranscription(EmptyTranscription::Asr));
        }

        let asr_output = convert_asr_response(response);
        info!(
            num_tokens = response.tokens.len(),
            num_monologues = asr_output.monologues.len(),
            "ASR response received, starting post-processing"
        );

        let resolved_lang = ctx.resolve_lang()?;
        let utterances =
            process_asr_with_prechat_segmentation(ctx, &asr_output, &resolved_lang).await?;
        // The engine returned words and post-processing kept no utterance from
        // them. Its own empty-chunk path returns an empty Vec here, so without
        // this the emptiness travelled on to CHAT assembly unremarked.
        if utterances.is_empty() {
            return Err(ServerError::EmptyTranscription(
                EmptyTranscription::Postprocess,
            ));
        }
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

/// How this run will segment utterances, decided from the resolved language.
///
/// This replaces a bare `matches!(lang.as_ref(), "eng" | "cmn" | "zho" | "yue")`
/// that was one of two copies of the boundary-model language set, the other
/// being the Python resolver's key set. The route now comes from the one table
/// in [`crate::utseg_route`].
///
/// Fallible because a language with no boundary model and no authorized
/// fallback has no segmenter at all. For an explicit `--lang` that case was
/// already refused at plan time, before ASR; reaching it here means `--lang
/// auto` detected such a language, which cannot be known any earlier.
fn resolve_utseg_route(
    opts: &TranscribeOptions,
    lang: &LanguageCode3,
) -> Result<crate::utseg_route::UtsegRoute, ServerError> {
    crate::utseg_route::UtsegRoute::resolve(
        lang,
        crate::params::UtsegFallbackPolicy::from(opts.allow_stanza_fallback_utseg),
    )
    .map_err(|unavailable| ServerError::Validation(unavailable.to_string()))
}

pub(crate) fn build_prechat_utseg_items(chunks: &[PreparedMonologueChunk]) -> Vec<UtsegBatchItem> {
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

pub(crate) fn apply_prechat_assignments(
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
        )?);
        let mut utterances = asr_postprocess::utterances_from_prepared_chunks(chunks);
        asr_postprocess::finalize_utterances(&mut utterances, &lang_str);
        if let Some(s) = ctx.asr_pipeline_snapshot.as_mut() {
            s.final_utterances = utterances.clone();
        }
        return Ok(utterances);
    };

    // The pre-CHAT pass runs the boundary model or nothing: there is no
    // pre-CHAT Stanza path, so an authorized Stanza fallback segments only
    // after CHAT is built, and this pass hands its chunks to punctuation
    // retokenization exactly as it always did for a language with no model.
    if !matches!(
        resolve_utseg_route(ctx.opts, resolved_lang)?,
        crate::utseg_route::UtsegRoute::BoundaryModel
    ) {
        let chunks = project_speakers(prepare_asr_chunks_with_snapshot(
            asr_output,
            &lang_str,
            ctx.asr_pipeline_snapshot.as_mut(),
        )?);
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
    )?);
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
/// Stages 1-3 (compound merge, timed-word extraction, Cantonese normalization,
/// multi-word split) run per monologue. Number expansion is then applied per
/// word via `asr_postprocess::expand_number`. After expansion a whitespace-split
/// pass widens multi-word expansions into separate tokens. Stages 5-5b
/// (long-turn and pause splitting) finalize per monologue.
///
/// The entry point for any caller that wants what transcribe produces without
/// a per-stage trace, including the offline `eval utseg-replay` pre-ASR pass.
/// A replay that prepared its chunks some other way would be comparing two
/// implementations rather than replaying one.
pub(crate) fn prepare_asr_chunks(
    asr_output: &batchalign_transform::asr_postprocess::AsrOutput,
    lang: &str,
) -> Result<Vec<PreparedMonologueChunk>, ServerError> {
    prepare_asr_chunks_with_snapshot(asr_output, lang, None)
}

/// A Cantonese normalization that changed a monologue's character count refuses
/// the file.
///
/// The alternative would be re-cutting the words around the new characters,
/// after which each word's timing would belong to characters it no longer
/// holds. A transcript mistimed that way looks right and reads wrong, which is
/// worse than a named failure.
fn cantonese_normalization_refused(
    refusal: batchalign_transform::asr_postprocess::NormalizationChangedLength,
) -> ServerError {
    ServerError::Validation(format!("ASR post-processing refused this file: {refusal}"))
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
) -> Result<Vec<PreparedMonologueChunk>, ServerError> {
    let Some(_) = snapshot.as_ref() else {
        // One implementation of this step, not two. Without a trace to fill in
        // there is nothing this function adds over the transform's own
        // preparation, and keeping a second copy here is exactly the drift
        // `eval utseg-replay` exists to catch. The traced path below stays
        // separate only because it must record each stage as it goes;
        // `snapshot_and_plain_preparation_agree` holds the two to one answer.
        return asr_postprocess::prepare_asr_chunks(asr_output, lang)
            .map_err(cantonese_normalization_refused);
    };

    if let Some(ref mut s) = snapshot {
        for m in &asr_output.monologues {
            s.raw_elements.extend_from_slice(&m.elements);
        }
    }

    let mut monologue_words: Vec<(asr_postprocess::SpeakerIndex, Vec<AsrWord>)> =
        Vec::with_capacity(asr_output.monologues.len());
    for m in &asr_output.monologues {
        let mut sub = AsrPipelineSnapshot::default();
        let cap = snapshot.is_some().then_some(&mut sub);
        let words =
            asr_postprocess::prepare_words_pre_expansion_with_snapshot(&m.elements, lang, cap)
                .map_err(cantonese_normalization_refused)?;
        if let Some(ref mut s) = snapshot {
            s.after_compound_merge.extend(sub.after_compound_merge);
            s.after_timing_extract.extend(sub.after_timing_extract);
            // Cantonese normalization is captured here now: it runs inside
            // preparation, before the multi-word split, not after expansion.
            if let Some(yue) = sub.after_cantonese_norm {
                s.after_cantonese_norm
                    .get_or_insert_with(Vec::new)
                    .extend(yue);
            }
            s.after_multiword_split.extend(sub.after_multiword_split);
        }
        monologue_words.push((m.speaker, words));
    }

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
            s.after_long_turn_split.extend(sub.after_long_turn_split);
        }
    }
    Ok(prepared)
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
        // remains `Auto` even after ASR, so the value comes from the response,
        // through the context's one resolution rather than a pipeline-order
        // comment the type did not enforce.
        let speaker_worker_lang = WorkerLanguage::from(ctx.resolve_lang()?);
        info!(
            audio_path = %ctx.audio_path.display(),
            speaker_backend = ?speaker_backend,
            expected_speakers = ?ctx.opts.expected_speakers(),
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
        // Auto becomes the ASR-detected language for CHAT headers and NLP.
        // Resolved once per file and recorded on the context, so utseg and
        // morphotag dispatch on the real language rather than on Auto.
        let resolved_lang = ctx.resolve_lang()?;

        // No empty-transcript branch here any more: `stage_asr_postprocess`
        // refuses a response with no tokens before this stage runs, so there is
        // no longer a shape this stage could answer by writing a file with no
        // utterances in it.
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

        let transcript = build_chat::NamedAsrUtterances::numbered(utterances).into_transcript(
            &langs,
            ctx.opts.media_name.as_deref(),
            ctx.opts.write_wor,
        )?;

        // The gate hands back every token it refused and we emitted anyway.
        // Saying so here is the point: an operator reading this run now learns
        // that the transcript is knowingly invalid CHAT and why, instead of
        // finding out from `chatter validate` days later with no way to tell
        // an ASR artifact from a transcription error. The tokens themselves
        // stay verbatim, which is this pipeline's standing policy: what the
        // speaker actually said is a human's call, not the pipeline's.
        report_language_invalid_words(ctx, &transcript.language_invalid);
        let desc = transcript.description;

        let mut chat_file = build_chat::build_chat(&desc).map_err(|error| match error {
            // Neither malformed input nor an internal fault: words reached CHAT
            // assembly and none of them was content, which is what a Cantonese
            // engine that recognized only punctuation produces. Carried as the
            // typed emptiness so the failure names the stage, rather than being
            // flattened into a validation message.
            build_chat::BuildChatError::NoUtterances => {
                ServerError::EmptyTranscription(EmptyTranscription::ChatBuild { described: 0 })
            }
            build_chat::BuildChatError::NoUtteranceSurvivedBuild { described } => {
                ServerError::EmptyTranscription(EmptyTranscription::ChatBuild { described })
            }
            other => ServerError::Validation(format!("Failed to build CHAT: {other}")),
        })?;
        // Inject processing provenance comment.
        let asr = ctx.asr_identity();
        let provenance = crate::provenance::transcribe_provenance(
            &resolved_lang,
            &asr,
            ctx.opts.diarize,
            ctx.opts.write_wor,
        );
        crate::provenance::inject_provenance(&mut chat_file, &provenance);

        // Inject human-readable "unchecked ASR" warning (a user's workflow depends on this).
        crate::provenance::inject_unchecked_warning(&mut chat_file, &asr);

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
    use crate::api::DurationSeconds;
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

    fn test_transcribe_options(speaker_backend: Option<SpeakerBackendV2>) -> TranscribeOptions {
        TranscribeOptions {
            asr: crate::transcribe::TranscribeAsrPlan::from_request(
                AsrBackend::RustRevAi,
                false,
                2,
                &std::collections::BTreeMap::new(),
            )
            .unwrap(),
            diarize: true,
            speaker_backend,
            lang: LanguageCode3::eng().into(),
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
        let left_provenance =
            crate::provenance::extract_provenance(left).expect("left stamps parse");
        let right_provenance =
            crate::provenance::extract_provenance(right).expect("right stamps parse");
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
            PipelineServices::new(&pool, &cache),
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
            PipelineServices::new(&pool, &reopened),
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
                model: None,
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
        let mut opts = test_transcribe_options(Some(SpeakerBackendV2::PyannoteAi));
        opts.lang = LanguageCode3::fra().into();
        let chat = run_transcribe_pipeline_with_legacy_replay(
            replay,
            TranscribeUtsegExecution::production(opts.with_utseg),
            PipelineServices::new(&pool, &cache),
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
            model: None,
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
            model: None,
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
        let services = PipelineServices::new(&pool, &cache);
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
            model: None,
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
        let services = PipelineServices::new(&pool, &cache);
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
            model: None,
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
        let services = PipelineServices::new(&pool, &cache);
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
            model: None,
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

    /// Both preparation entry points give one answer.
    ///
    /// The untraced path IS the transform's function now, so this holds the
    /// traced path to it. Number expansion is where the two could drift: the
    /// traced path skips `expand_number` for tokens carrying no ASCII digit, so
    /// a token that expanded without one would diverge silently.
    #[test]
    fn snapshot_and_plain_preparation_agree() {
        use batchalign_transform::asr_postprocess::{
            AsrElement, AsrElementKind, AsrMonologue, AsrOutput, AsrRawText, AsrTimestampSecs,
            SpeakerIndex,
        };

        let element = |value: &str, ts: f64, end_ts: f64| AsrElement {
            value: AsrRawText::new(value),
            ts: AsrTimestampSecs::from(Some(ts)),
            end_ts: AsrTimestampSecs::from(Some(end_ts)),
            kind: AsrElementKind::Text,
        };
        let output = AsrOutput {
            monologues: vec![AsrMonologue {
                speaker: SpeakerIndex(0),
                elements: vec![
                    element("I", 0.0, 0.2),
                    element("counted", 0.2, 0.6),
                    element("100", 0.6, 1.0),
                    element("sheep", 1.0, 1.4),
                    element("in", 1.4, 1.6),
                    element("2001", 1.6, 2.2),
                    element(".", 2.2, 2.3),
                ],
            }],
        };

        let plain = prepare_asr_chunks(&output, "eng")
            .expect("test: ASR post-processing must not refuse this input");
        let mut snapshot = AsrPipelineSnapshot::default();
        let traced = prepare_asr_chunks_with_snapshot(&output, "eng", Some(&mut snapshot))
            .expect("test: ASR post-processing must not refuse this input");

        assert!(
            plain.iter().any(|chunk| chunk
                .words
                .iter()
                .any(|word| word.text.as_str() == "hundred")),
            "the fixture must actually expand a number, or it proves nothing: {plain:?}"
        );
        assert_eq!(plain, traced, "one preparation step, two entry points");
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
        let services = PipelineServices::new(&pool, &cache);
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
            model: None,
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
        let services = PipelineServices::new(&pool, &cache);
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
            model: None,
            source_monologues: None,
        });

        stage_asr_postprocess(&mut ctx).await.expect("postprocess");
        assert_eq!(ctx.resolved_lang, Some(LanguageCode3::spa()));
    }

    /// RED FIRST (2026-09-16): an ASR response with no tokens refuses the job,
    /// naming the stage that produced nothing.
    ///
    /// This used to return `Ok(())`, and `stage_build_chat` then wrote a
    /// headers-only CHAT file, so a run that recognized not one word finished
    /// as a completed job carrying an empty transcript.
    ///
    /// It replaces `build_chat_stage_resolves_auto_for_empty_response`, which
    /// asserted that the empty file carried the resolved language rather than
    /// `auto`. There is no empty file to make that assertion about any more.
    /// The property that test guarded, that `auto` is never stamped into
    /// `@Languages`, is held on the path that still writes a file by
    /// `build_chat_stage_resolves_auto_to_detected_language`.
    #[tokio::test]
    async fn an_asr_response_with_no_tokens_is_refused_naming_the_asr_stage() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let pool = WorkerPool::new(PoolConfig::default());
        let services = PipelineServices::new(&pool, &cache);
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
            model: None,
            source_monologues: None,
        });

        let refusal = stage_asr_postprocess(&mut ctx)
            .await
            .expect_err("an ASR response with no tokens must refuse the job");
        assert!(
            matches!(
                refusal,
                ServerError::EmptyTranscription(EmptyTranscription::Asr)
            ),
            "the refusal must name the ASR stage, got: {refusal:?}"
        );
        assert!(
            ctx.chat_text.is_none(),
            "nothing may be written for a recording with no recognized words"
        );
    }
}
