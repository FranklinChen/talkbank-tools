//! Transcribe pipeline built on the internal stage runner.

#[cfg(test)]
use crate::revai::FetchedRevAsrEvidence;

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

use crate::api::{
    AsrLanguageRequest, ChatText, LanguageCode3, NumSpeakers, TranscriptLanguage, WorkerLanguage,
};
use crate::error::{EmptyTranscription, ServerError};
use crate::params::{MorphosyntaxParams, UtsegFallbackPolicy};
use crate::pipeline::PipelineServices;
use crate::pipeline::plan::{StageId, observe_stage};
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
use crate::transcribe::{ReplayAsrPlan, TranscribeAsrPlan, TranscribePlan};
use crate::types::worker_v2::{SpeakerBackendV2, SpeakerSegmentV2};
use crate::utseg::TranscribeUtsegExecution;
use crate::utseg_evidence::{UtsegEvidencePhase, UtsegEvidenceSink, UtsegEvidenceTrace};

static PRODUCTION_REV_INFERENCE: RevAsrService = RevAsrService;

struct PendingAsr;

/// The ASR producer resolves language and identity before any downstream stage.
struct Recognized {
    response: AsrResponse,
    language: TranscriptLanguage,
    identity: crate::transcribe::types::AsrIdentity,
}

struct Postprocessed {
    asr: Recognized,
    utterances: Vec<Utterance>,
}

struct ChatReady {
    asr: Recognized,
    text: String,
}

trait HasAsr {
    fn asr(&self) -> &Recognized;
}
impl HasAsr for Recognized {
    fn asr(&self) -> &Recognized {
        self
    }
}
impl HasAsr for Postprocessed {
    fn asr(&self) -> &Recognized {
        &self.asr
    }
}
impl HasAsr for ChatReady {
    fn asr(&self) -> &Recognized {
        &self.asr
    }
}

/// A consuming transition replaces the state; prerequisites are not optional.
struct TranscribePipelineContext<'a, S, P: TranscribePlan = TranscribeAsrPlan> {
    services: PipelineServices<'a>,
    opts: &'a TranscribeOptions<P>,
    audio_path: &'a Path,
    speaker_segments: Option<Vec<SpeakerSegmentV2>>,
    dumper: DebugDumper,
    utseg_evidence_sink: UtsegEvidenceSink,
    utseg_execution: TranscribeUtsegExecution,
    rev_evidence: Option<RevAsrEvidenceTrace>,
    asr_pipeline_snapshot: Option<AsrPipelineSnapshot>,
    state: S,
}

impl<'a, S, P: TranscribePlan> TranscribePipelineContext<'a, S, P> {
    fn map_state<T>(self, transition: impl FnOnce(S) -> T) -> TranscribePipelineContext<'a, T, P> {
        let Self {
            services,
            opts,
            audio_path,
            speaker_segments,
            dumper,
            utseg_evidence_sink,
            utseg_execution,
            rev_evidence,
            asr_pipeline_snapshot,
            state,
        } = self;
        TranscribePipelineContext {
            services,
            opts,
            audio_path,
            speaker_segments,
            dumper,
            utseg_evidence_sink,
            utseg_execution,
            rev_evidence,
            asr_pipeline_snapshot,
            state: transition(state),
        }
    }

    fn new(
        audio_path: &'a Path,
        services: PipelineServices<'a>,
        opts: &'a TranscribeOptions<P>,
        dumper: DebugDumper,
        utseg_evidence_sink: UtsegEvidenceSink,
        utseg_execution: TranscribeUtsegExecution,
        state: S,
    ) -> Self {
        Self {
            services,
            opts,
            audio_path,
            speaker_segments: None,
            dumper,
            utseg_evidence_sink,
            utseg_execution,
            rev_evidence: None,
            asr_pipeline_snapshot: std::env::var("BA3_DUMP_ASR_PIPELINE")
                .ok()
                .map(|_| AsrPipelineSnapshot::default()),
            state,
        }
    }
}

impl<S: HasAsr, P: TranscribePlan> TranscribePipelineContext<'_, S, P> {
    fn asr_identity(&self) -> crate::transcribe::types::AsrIdentity {
        self.state.asr().identity.clone()
    }
}

impl<'a> TranscribePipelineContext<'a, PendingAsr> {
    #[cfg(test)]
    fn new_with_rev_inference(
        audio_path: &'a Path,
        services: PipelineServices<'a>,
        opts: &'a TranscribeOptions,
        dumper: DebugDumper,
        sink: UtsegEvidenceSink,
        _inference: &'a dyn RevAsrEvidenceInference,
    ) -> Self {
        Self::new(
            audio_path,
            services,
            opts,
            dumper,
            sink,
            TranscribeUtsegExecution::production(opts.with_utseg),
            PendingAsr,
        )
    }

    #[cfg(test)]
    fn admit_response(
        self,
        response: AsrResponse,
    ) -> Result<TranscribePipelineContext<'a, Recognized>, ServerError> {
        let language = resolved_asr_language(self.opts, &response)?;
        let identity = self.opts.asr.identity();
        Ok(self.map_state(|_| Recognized {
            response,
            language,
            identity,
        }))
    }
}

/// The observer preserves stage progress and timing without erasing transitions.
struct TranscribeProgress {
    sender: Option<ProgressSender>,
    completed: usize,
    total: usize,
}

impl TranscribeProgress {
    async fn run<T>(
        &mut self,
        stage: StageId,
        transition: impl std::future::Future<Output = Result<T, ServerError>>,
    ) -> Result<T, ServerError> {
        if let Some(sender) = &self.sender {
            let _ = sender.send(ProgressUpdate::new(
                progress_stage_for_stage(stage),
                Some(self.completed as i64),
                Some(self.total as i64),
            ));
        }
        let result = observe_stage("transcribe", stage, transition).await?;
        self.completed += 1;
        Ok(result)
    }
}

struct TranscribeTail {
    post_chat: Option<crate::utseg::UtsegDecisionPolicy>,
    morphosyntax: bool,
}

fn prepare_tail<P: TranscribePlan>(
    services: &PipelineServices<'_>,
    opts: &TranscribeOptions<P>,
    execution: TranscribeUtsegExecution,
    progress: Option<ProgressSender>,
) -> Result<(TranscribeTail, TranscribeProgress), ServerError> {
    let stanza_supported = match opts.language().primary_if_known() {
        Some(code) => stanza_supports(services, code)?,
        None => true,
    };
    let post_chat = if stanza_supported {
        execution.post_chat_policy()
    } else {
        None
    };
    let morphosyntax = opts.with_morphosyntax && stanza_supported;
    if !stanza_supported {
        info!(lang = %opts.language(), "Skipping requested Stanza-backed stages: no Stanza pipeline for this language");
    }
    let total = 4
        + usize::from(opts.diarize)
        + usize::from(post_chat.is_some())
        + usize::from(morphosyntax);
    Ok((
        TranscribeTail {
            post_chat,
            morphosyntax,
        },
        TranscribeProgress {
            sender: progress,
            completed: 0,
            total,
        },
    ))
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
    let execution = TranscribeUtsegExecution::production(opts.with_utseg);
    let (tail, mut progress) = prepare_tail(&services, opts, execution, progress)?;
    let ctx = TranscribePipelineContext::new(
        audio_path,
        services,
        opts,
        DebugDumper::new(debug_dir),
        UtsegEvidenceSink::new(debug_dir),
        execution,
        PendingAsr,
    );
    let ctx = progress
        .run(StageId::AsrInfer, stage_asr_infer(ctx, rev_inference))
        .await?;
    finish_transcribe(ctx, tail, progress).await
}

/// Replay options cannot be supplied to the live entry point.
pub(crate) async fn run_transcribe_pipeline_with_legacy_replay<'a>(
    replay: AdmittedLegacyTranscribeReplay,
    execution: TranscribeUtsegExecution,
    services: PipelineServices<'a>,
    opts: &'a TranscribeOptions<ReplayAsrPlan>,
    progress: Option<ProgressSender>,
    debug_dir: Option<&Path>,
) -> Result<String, ServerError> {
    let audio_path = replay.media_path().to_owned();
    let (tail, mut progress) = prepare_tail(&services, opts, execution, progress)?;
    let ctx = progress.run(StageId::AsrInfer, async {
        let response = replay.asr_response().clone();
        let language = resolved_asr_language(opts, &response)?;
        let identity = crate::transcribe::types::AsrIdentity::of_replay(replay.producer());
        let mut ctx = TranscribePipelineContext::new(&audio_path, services, opts, DebugDumper::new(debug_dir),
            UtsegEvidenceSink::new(debug_dir), execution, Recognized { response, language, identity });
        if opts.diarize {
            ctx.speaker_segments = Some(replay.speaker_segments().ok_or_else(|| ServerError::Validation(
                format!("replay {} requested diarization but its manifest has no speaker-turn artifact", replay.recording_id())
            ))?.to_vec());
        }
        Ok(ctx)
    }).await?;
    Ok(finish_transcribe(ctx, tail, progress).await?.chat_text)
}

async fn finish_transcribe<P: TranscribePlan>(
    mut ctx: TranscribePipelineContext<'_, Recognized, P>,
    tail: TranscribeTail,
    mut progress: TranscribeProgress,
) -> Result<CompletedTranscribePipeline, ServerError> {
    if ctx.opts.diarize {
        progress
            .run(
                StageId::SpeakerDiarization,
                stage_speaker_diarization(&mut ctx),
            )
            .await?;
    }
    let ctx = progress
        .run(StageId::AsrPostprocess, stage_asr_postprocess(ctx))
        .await?;
    let mut ctx = progress
        .run(StageId::BuildChat, stage_build_chat(ctx))
        .await?;
    if let Some(policy) = tail.post_chat {
        progress
            .run(StageId::OptionalUtseg, stage_run_utseg(&mut ctx, policy))
            .await?;
    }
    if tail.morphosyntax {
        progress
            .run(
                StageId::OptionalMorphosyntax,
                stage_run_morphosyntax(&mut ctx),
            )
            .await?;
    }
    progress
        .run(StageId::Serialize, async {
            Ok(CompletedTranscribePipeline {
                chat_text: ctx.state.text,
                rev_evidence: ctx.rev_evidence,
            })
        })
        .await
}

/// Whether Stanza has a pipeline for one language: the runtime registry when
/// the worker has reported one, else the hardcoded pre-warmup table.
fn stanza_supports(
    services: &PipelineServices<'_>,
    code: &LanguageCode3,
) -> Result<bool, ServerError> {
    match services.pool.stanza_registry() {
        Some(registry) => Ok(registry.supports_morphosyntax(code.as_ref())),
        None => {
            // Fallible in chatter 0.3.0; stringified error
            // (`LanguageCodeError` not re-exported upstream).
            let chat_lang = crate::chat_ops::LanguageCode::new(code.as_ref()).map_err(|e| {
                ServerError::Validation(format!(
                    "transcribe: invalid language code {:?}: {e}",
                    code.as_ref()
                ))
            })?;
            Ok(crate::chat_ops::morphosyntax_ops::is_stanza_supported(
                &chat_lang,
            ))
        }
    }
}

/// Map transcribe-pipeline stage ids onto the shared file-progress stage
/// vocabulary.
///
/// This match is intentionally explicit. If the transcribe plan adds a new
/// stage, contributors should decide its operator-facing stage here rather
/// than silently falling back to a generic string.
fn progress_stage_for_stage(stage: StageId) -> FileStage {
    // The typed transcribe sequence only emits stages
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

async fn stage_asr_infer<'a>(
    mut ctx: TranscribePipelineContext<'a, PendingAsr>,
    rev_inference: &dyn RevAsrEvidenceInference,
) -> Result<TranscribePipelineContext<'a, Recognized>, ServerError> {
    info!(
        audio_path = %ctx.audio_path.display(),
                                lang = %ctx.opts.language(),
        expected_speakers = ?ctx.opts.expected_speakers(),
        "Starting ASR inference"
    );

    let opts = ctx.opts;
    let mut evidence_language = None;
    let response = match opts.asr.inference() {
        crate::transcribe::AsrInference::NonRev {
            backend,
            speakers,
            language,
        } => {
            infer_asr(
                ctx.services.pool,
                &AsrInferParams {
                    backend,
                    audio_path: ctx.audio_path,
                    lang: language,
                    num_speakers: NumSpeakers(speakers.get()),
                    extras: &ctx.opts.engine_extras,
                },
            )
            .await?
        }
        crate::transcribe::AsrInference::RevAi { language } => {
            let provider_media = PreparedRevProviderMedia::from_source(ctx.audio_path)
                .await
                .map_err(|error| ServerError::Persistence(error.to_string()))?;
            let request = RevAsrEvidenceRequest::new(
                provider_media,
                language,
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
            // The evidence says what language its transcript is in; that is
            // the transcript's language, not a second reading of the request.
            evidence_language = Some(evidence.resolved_language().clone());
            rev_evidence_to_asr_response(&evidence)
        }
    };
    let filename = ctx
        .audio_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    ctx.dumper.dump_asr_response(filename, &response);
    let language = match evidence_language {
        Some(language) => language,
        None => resolved_asr_language(opts, &response)?,
    };
    let planned = opts.asr.identity();
    let identity = match response.model.clone() {
        Some(model) => planned.with_loaded_models(model),
        None => planned,
    };
    Ok(ctx.map_state(|_| Recognized {
        response,
        language,
        identity,
    }))
}

async fn stage_asr_postprocess<'a, P: TranscribePlan>(
    mut ctx: TranscribePipelineContext<'a, Recognized, P>,
) -> Result<TranscribePipelineContext<'a, Postprocessed, P>, ServerError> {
    let response = &ctx.state.response;
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

    let resolved_lang = ctx.state.language.clone();
    let utterances =
        process_asr_with_prechat_segmentation(&mut ctx, &asr_output, &resolved_lang).await?;
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

    Ok(ctx.map_state(|asr| Postprocessed { asr, utterances }))
}

/// Decide the transcript's language(s) for CHAT headers and NLP dispatch.
///
/// One language or a pair is what was requested. Detection resolves to the
/// language the engine reported, or, when it reported none, to offline
/// detection over the transcript text, and errors when that fails too.
///
/// No silent fallback to English. CHAT files must declare a real
/// `@Languages:` value, and downstream NLP needs the real code; pretending
/// the language is English when it is not is exactly the kind of provenance
/// corruption the 2026-05-03 morphotag incident punished.
fn resolved_asr_language<P: TranscribePlan>(
    opts: &TranscribeOptions<P>,
    response: &AsrResponse,
) -> Result<TranscriptLanguage, ServerError> {
    match opts.language() {
        AsrLanguageRequest::Detect => {
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
                LanguageCode3::try_new(&detected_iso3)
                    .map(TranscriptLanguage::One)
                    .map_err(|err| {
                        ServerError::Validation(format!(
                            "offline language detection produced invalid ISO 639-3 code \
                             '{detected_iso3}': {err}",
                        ))
                    })
            } else {
                Ok(TranscriptLanguage::One(detected))
            }
        }
        AsrLanguageRequest::One(code) => Ok(TranscriptLanguage::One(code)),
        AsrLanguageRequest::Pair(pair) => Ok(TranscriptLanguage::Pair(pair)),
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
fn resolve_utseg_route<P: TranscribePlan>(
    opts: &TranscribeOptions<P>,
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

async fn process_asr_with_prechat_segmentation<P: TranscribePlan>(
    ctx: &mut TranscribePipelineContext<'_, Recognized, P>,
    asr_output: &batchalign_transform::asr_postprocess::AsrOutput,
    language: &TranscriptLanguage,
) -> Result<Vec<Utterance>, ServerError> {
    // Segmentation and the structural cleanup rules run under the primary
    // language; for a pair, the steps that write words keep what was
    // recognized (see `AsrTextLanguage`).
    let resolved_lang = language.primary();
    let lang_str = resolved_lang.to_string();
    let text_language = match language {
        TranscriptLanguage::One(_) => asr_postprocess::AsrTextLanguage::One(&lang_str),
        TranscriptLanguage::Pair(_) => {
            asr_postprocess::AsrTextLanguage::CodeSwitched { primary: &lang_str }
        }
    };
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
            text_language,
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
            text_language,
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
        text_language,
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
pub(crate) fn prepare_asr_chunks<'a>(
    asr_output: &batchalign_transform::asr_postprocess::AsrOutput,
    language: impl Into<asr_postprocess::AsrTextLanguage<'a>>,
) -> Result<Vec<PreparedMonologueChunk>, ServerError> {
    prepare_asr_chunks_with_snapshot(asr_output, language.into(), None)
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
    language: asr_postprocess::AsrTextLanguage<'_>,
    mut snapshot: Option<&mut AsrPipelineSnapshot>,
) -> Result<Vec<PreparedMonologueChunk>, ServerError> {
    let lang = language.rules();
    let Some(_) = snapshot.as_ref() else {
        // One implementation of this step, not two. Without a trace to fill in
        // there is nothing this function adds over the transform's own
        // preparation, and keeping a second copy here is exactly the drift
        // `eval utseg-replay` exists to catch. The traced path below stays
        // separate only because it must record each stage as it goes;
        // `snapshot_and_plain_preparation_agree` holds the two to one answer.
        return asr_postprocess::prepare_asr_chunks(asr_output, language)
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
            asr_postprocess::prepare_words_pre_expansion_with_snapshot(&m.elements, language, cap)
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

    // Number expansion writes words in a language, so it runs only for text
    // in one language; code-switched text keeps its digits (see
    // `AsrTextLanguage`), exactly as the untraced path does.
    match language {
        asr_postprocess::AsrTextLanguage::One(numeral_lang) => {
            for (_speaker, words) in &mut monologue_words {
                for word in words.iter_mut() {
                    let text = word.text.as_str();
                    // Fast path: tokens with no ASCII digit can never expand
                    // (every expander: NUM2LANG, num2chinese, currency,
                    // ordinal/decade: requires a digit somewhere in the input).
                    if !text.bytes().any(|b| b.is_ascii_digit()) {
                        continue;
                    }
                    let expanded = asr_postprocess::expand_number(text, numeral_lang);
                    if expanded != text {
                        word.text = asr_postprocess::AsrNormalizedText::new(expanded);
                    }
                }
                asr_postprocess::split_words_with_whitespace(words);
            }
        }
        asr_postprocess::AsrTextLanguage::CodeSwitched { .. } => {}
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

async fn stage_speaker_diarization<P: TranscribePlan>(
    ctx: &mut TranscribePipelineContext<'_, Recognized, P>,
) -> Result<(), ServerError> {
    if P::REPLAY {
        return Ok(());
    }
    let response = &ctx.state.response;
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

    // Speaker workers require a concrete routing language. The request may
    // be detection even after ASR, so the value comes from the context's
    // one resolution: the transcript's primary language.
    let speaker_worker_lang = WorkerLanguage::from(ctx.state.language.primary());
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
}

async fn stage_build_chat<'a, P: TranscribePlan>(
    mut ctx: TranscribePipelineContext<'a, Postprocessed, P>,
) -> Result<TranscribePipelineContext<'a, ChatReady, P>, ServerError> {
    let resolved_lang = ctx.state.asr.language.clone();
    let utterances = &mut ctx.state.utterances;
    // Under detection, run per-utterance language detection for
    // code-switching markup and multi-language headers. A requested
    // language or pair is declared as requested.
    let langs: Vec<String> = match ctx.opts.language() {
        AsrLanguageRequest::Detect => {
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
            lang_detect::collect_detected_languages(&utt_text_refs, resolved_lang.primary())
        }
        AsrLanguageRequest::One(_) | AsrLanguageRequest::Pair(_) => resolved_lang
            .declared()
            .into_iter()
            .map(ToString::to_string)
            .collect(),
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
    report_language_invalid_words(&ctx, &transcript.language_invalid);
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
    Ok(ctx.map_state(|state| ChatReady {
        asr: state.asr,
        text: chat_text,
    }))
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
fn report_language_invalid_words<S, P: TranscribePlan>(
    ctx: &TranscribePipelineContext<'_, S, P>,
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

async fn stage_run_utseg<P: TranscribePlan>(
    ctx: &mut TranscribePipelineContext<'_, ChatReady, P>,
    post_chat_policy: crate::utseg::UtsegDecisionPolicy,
) -> Result<(), ServerError> {
    let utseg_lang = ctx.state.asr.language.primary().clone();
    let input = ctx.state.text.as_str();
    let filename = ctx
        .audio_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    ctx.dumper.dump_pre_utseg_chat(filename, input);
    let evidence_filename = ctx.audio_path.to_string_lossy();
    let result =
        crate::utseg::process_utseg_with_evidence(crate::utseg::EvidenceRetainingUtsegRequest {
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
        })
        .await?;
    ctx.dumper.dump_post_utseg_chat(filename, &result);
    ctx.state.text = result;
    Ok(())
}

async fn stage_run_morphosyntax<P: TranscribePlan>(
    ctx: &mut TranscribePipelineContext<'_, ChatReady, P>,
) -> Result<(), ServerError> {
    let mor_lang = ctx.state.asr.language.primary().clone();
    let input = ctx.state.text.as_str();
    let filename = ctx
        .audio_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    ctx.dumper.dump_pre_morphosyntax_chat(filename, input);
    let empty_mwt = std::collections::BTreeMap::new();
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
    ctx.state.text = crate::morphosyntax::process_morphosyntax(input, ctx.services, &mor_params)
        .await?
        // The transcribe pipeline threads CHAT text between stages and
        // gates its own final output; the morphotag proof is
        // discharged into that text here.
        .into_text();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::DurationSeconds;
    use crate::cache::UtteranceCache;
    use crate::revai::{
        AuthorizedRevEvidenceRun, RevAsrEvidenceCacheOutcome, RevAsrEvidenceInference,
        RevTranscriptEvidence,
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
        test_transcribe_options_in(speaker_backend, LanguageCode3::eng().into())
    }

    fn test_transcribe_options_in(
        speaker_backend: Option<SpeakerBackendV2>,
        language: crate::api::LanguageSpec,
    ) -> TranscribeOptions {
        TranscribeOptions {
            asr: crate::transcribe::TranscribeAsrPlan::from_request(
                AsrBackend::RustRevAi,
                false,
                2,
                &std::collections::BTreeMap::new(),
                &language,
            )
            .unwrap(),
            diarize: true,
            speaker_backend,
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
            Ok(crate::revai::RevAsrInferenceOutcome::Fetched(FetchedRevAsrEvidence {
                transcript_evidence: RevTranscriptEvidence::from_provider_json(
                    r#"{"monologues":[{"speaker":0,"elements":[{"type":"text","value":"hello","ts":0.1,"end_ts":0.5,"confidence":0.9},{"type":"punct","value":".","ts":null,"end_ts":null,"confidence":null}]},{"speaker":1,"elements":[{"type":"text","value":"there","ts":0.6,"end_ts":1.0,"confidence":0.8},{"type":"punct","value":"?","ts":null,"end_ts":null,"confidence":null}]}]}"#
                        .to_owned(),
                )
                .expect("valid provider transcript fixture"),
                resolved_language: TranscriptLanguage::One(LanguageCode3::eng()),
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

    /// Rev.AI evidence for a code-switched English/Spanish request.
    struct EnglishSpanishRevInference;

    #[async_trait::async_trait]
    impl RevAsrEvidenceInference for EnglishSpanishRevInference {
        async fn infer(
            &self,
            _run: AuthorizedRevEvidenceRun,
        ) -> Result<crate::revai::RevAsrInferenceOutcome, ServerError> {
            Ok(crate::revai::RevAsrInferenceOutcome::Fetched(FetchedRevAsrEvidence {
                transcript_evidence: RevTranscriptEvidence::from_provider_json(
                    r#"{"monologues":[{"speaker":0,"elements":[{"type":"text","value":"quiero","ts":0.1,"end_ts":0.4,"confidence":0.9},{"type":"text","value":"water","ts":0.5,"end_ts":0.9,"confidence":0.8},{"type":"text","value":"25","ts":0.9,"end_ts":1.2,"confidence":0.8},{"type":"punct","value":".","ts":null,"end_ts":null,"confidence":null}]}]}"#
                        .to_owned(),
                )
                .expect("valid provider transcript fixture"),
                resolved_language: TranscriptLanguage::Pair(
                    crate::api::LanguagePair::new(LanguageCode3::eng(), LanguageCode3::spa())
                        .expect("two languages"),
                ),
            }))
        }
    }

    /// A code-switched request declares both of its languages: `@Languages`
    /// in the pair's order, and one `lang=` provenance field naming the pair.
    #[tokio::test]
    async fn a_language_pair_is_declared_in_languages_and_provenance() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio_path = tempdir.path().join("sample.wav");
        tokio::fs::write(&audio_path, b"provider media")
            .await
            .expect("write provider media");
        let pool = WorkerPool::new(PoolConfig::default());
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let mut opts = test_transcribe_options_in(
            None,
            crate::api::LanguageSpec::try_from("eng,spa").expect("a pair"),
        );
        opts.diarize = false;

        let completed = run_transcribe_pipeline_with_rev_inference(
            &audio_path,
            PipelineServices::new(&pool, &cache),
            &opts,
            None,
            None,
            &EnglishSpanishRevInference,
        )
        .await
        .expect("transcribe");
        let chat = completed.chat_text;

        assert!(
            chat.lines().any(|line| line == "@Languages:\teng, spa"),
            "{chat}"
        );
        let stamp = chat
            .lines()
            .find(|line| line.contains("fc-ba3 transcribe"))
            .expect("provenance stamp");
        assert!(stamp.contains("lang=eng,spa"), "{stamp}");
        // No language is known for `25`, so it is not spelled out in either.
        let main_tier = chat
            .lines()
            .find(|line| line.starts_with('*'))
            .expect("an utterance");
        assert!(main_tier.contains("25"), "{main_tier}");
        assert!(!main_tier.contains("twenty"), "{main_tier}");
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
        let opts = test_transcribe_options_in(
            Some(SpeakerBackendV2::PyannoteAi),
            LanguageCode3::fra().into(),
        );
        let opts = TranscribeOptions {
            asr: ReplayAsrPlan::for_legacy_replay(2, &LanguageCode3::fra().into()).unwrap(),
            diarize: opts.diarize,
            speaker_backend: opts.speaker_backend,
            with_utseg: opts.with_utseg,
            with_morphosyntax: opts.with_morphosyntax,
            cache_policies: opts.cache_policies,
            allow_stanza_fallback_utseg: opts.allow_stanza_fallback_utseg,
            write_wor: opts.write_wor,
            media_name: opts.media_name,
            engine_extras: opts.engine_extras,
        };
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
        let ctx = TranscribePipelineContext::new_with_rev_inference(
            &audio_path,
            services,
            &opts,
            DebugDumper::disabled(),
            UtsegEvidenceSink::Disabled,
            &PRODUCTION_REV_INFERENCE,
        );
        let mut ctx = ctx
            .admit_response(AsrResponse {
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
            })
            .expect("resolved ASR response");

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
        let ctx = TranscribePipelineContext::new_with_rev_inference(
            &audio_path,
            services,
            &opts,
            DebugDumper::disabled(),
            UtsegEvidenceSink::Disabled,
            &PRODUCTION_REV_INFERENCE,
        );
        let ctx = ctx
            .admit_response(AsrResponse {
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
            })
            .expect("resolved ASR response");

        let ctx = stage_asr_postprocess(ctx)
            .await
            .expect("disabled segmentation must not dispatch a worker");

        assert_eq!(Some(ctx.state.utterances.len()), Some(1));
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
        let opts = test_transcribe_options_in(
            Some(SpeakerBackendV2::Pyannote),
            LanguageCode3::fra().into(),
        );

        let ctx = TranscribePipelineContext::new_with_rev_inference(
            &audio_path,
            services,
            &opts,
            DebugDumper::disabled(),
            UtsegEvidenceSink::Disabled,
            &PRODUCTION_REV_INFERENCE,
        );
        let mut ctx = ctx
            .admit_response(AsrResponse {
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
            })
            .expect("resolved ASR response");
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

        let ctx = stage_asr_postprocess(ctx).await.expect("postprocess");
        let ctx = stage_build_chat(ctx).await.expect("build chat");

        let chat = ctx.state.text;
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
        let traced = prepare_asr_chunks_with_snapshot(
            &output,
            asr_postprocess::AsrTextLanguage::One("eng"),
            Some(&mut snapshot),
        )
        .expect("test: ASR post-processing must not refuse this input");

        assert!(
            plain.iter().any(|chunk| chunk
                .words
                .iter()
                .any(|word| word.text.as_str() == "hundred")),
            "the fixture must actually expand a number, or it proves nothing: {plain:?}"
        );
        assert_eq!(plain, traced, "one preparation step, two entry points");

        // The same holds for code-switched text, where both entry points must
        // keep the numerals rather than write them out.
        let switched = asr_postprocess::AsrTextLanguage::CodeSwitched { primary: "eng" };
        let plain = prepare_asr_chunks(&output, switched)
            .expect("test: ASR post-processing must not refuse this input");
        let mut snapshot = AsrPipelineSnapshot::default();
        let traced = prepare_asr_chunks_with_snapshot(&output, switched, Some(&mut snapshot))
            .expect("test: ASR post-processing must not refuse this input");
        assert!(
            plain
                .iter()
                .any(|chunk| chunk.words.iter().any(|word| word.text.as_str() == "100")),
            "{plain:?}"
        );
        assert_eq!(plain, traced, "one preparation step, two entry points");
    }

    /// When the request is detection, stage_build_chat must resolve to the
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
        let mut opts = test_transcribe_options_in(None, crate::api::LanguageSpec::Auto);
        opts.diarize = false;

        let ctx = TranscribePipelineContext::new_with_rev_inference(
            &audio_path,
            services,
            &opts,
            DebugDumper::disabled(),
            UtsegEvidenceSink::Disabled,
            &PRODUCTION_REV_INFERENCE,
        );

        // ASR response with detected language "spa"
        let ctx = ctx
            .admit_response(AsrResponse {
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
            })
            .expect("resolved ASR response");

        // Run post-processing to generate utterances
        let ctx = stage_asr_postprocess(ctx).await.expect("postprocess");

        // Run build_chat; this should resolve "auto" → "spa"
        let ctx = stage_build_chat(ctx).await.expect("build_chat");

        let chat_text = ctx.state.text.as_str();

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

        let mut opts = test_transcribe_options_in(None, crate::api::LanguageSpec::Auto);
        opts.diarize = false;

        let ctx = TranscribePipelineContext::new_with_rev_inference(
            &audio_path,
            services,
            &opts,
            DebugDumper::disabled(),
            UtsegEvidenceSink::Disabled,
            &PRODUCTION_REV_INFERENCE,
        );
        let ctx = ctx
            .admit_response(AsrResponse {
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
            })
            .expect("resolved ASR response");

        let ctx = stage_asr_postprocess(ctx).await.expect("postprocess");
        assert_eq!(
            ctx.state.asr.language,
            TranscriptLanguage::One(LanguageCode3::spa())
        );
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

        let opts = test_transcribe_options_in(None, crate::api::LanguageSpec::Auto);

        let ctx = TranscribePipelineContext::new_with_rev_inference(
            &audio_path,
            services,
            &opts,
            DebugDumper::disabled(),
            UtsegEvidenceSink::Disabled,
            &PRODUCTION_REV_INFERENCE,
        );
        let ctx = ctx
            .admit_response(AsrResponse {
                tokens: vec![],
                lang: LanguageCode3::fra(),
                model: None,
                source_monologues: None,
            })
            .expect("resolved ASR response");

        let refusal = stage_asr_postprocess(ctx)
            .await
            .err()
            .expect("an ASR response with no tokens must refuse the job");
        assert!(
            matches!(
                refusal,
                ServerError::EmptyTranscription(EmptyTranscription::Asr)
            ),
            "the refusal must name the ASR stage, got: {refusal:?}"
        );
    }
}
