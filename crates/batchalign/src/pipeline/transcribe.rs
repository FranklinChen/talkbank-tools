//! Transcribe pipeline built on the internal stage runner.

#[cfg(test)]
use crate::revai::FetchedRevAsrEvidence;

use crate::chat_ops::morphosyntax_ops::{MultilingualPolicy, TokenizationMode};
use crate::chat_ops::speaker::{
    AdmittedSpeakerProjectionEvidence, SpeakerProjectionPolicy,
    SpeakerSegment as ChatSpeakerSegment, project_speakers_onto_chunks,
};
use batchalign_transform::asr_postprocess::{
    self, AsrPipelineSnapshot, AsrWord, PreparedMonologueChunk, Utterance,
};
use batchalign_transform::build_chat;
use batchalign_transform::utseg::UtsegBatchItem;
use std::path::Path;

use tracing::info;

use crate::api::{
    AsrLanguageRequest, LanguageCode3, NumSpeakers, StageRefusalRecord, TranscriptLanguage,
    WorkerLanguage,
};
use crate::error::{EmptyTranscription, OutputAdmissionRefusal, ServerError};
use crate::params::MorphosyntaxParams;
#[cfg(test)]
use crate::params::UtsegFallbackPolicy;
use crate::pipeline::PipelineServices;
use crate::pipeline::plan::{StageId, observe_stage};
use crate::pipeline::post_validate::{OptionalStage, PostValidated, ProducedOutput, Shortfall};
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
    AsrInferParams, AsrResponse, NonRevAsrBackend, SpeakerEvidenceRunParams, SpeakerEvidenceSource,
    TranscribeOptions, convert_asr_response, infer_asr, resolve_speaker_evidence_for_audio,
};
use crate::transcribe::{ReplayAsrPlan, TranscribeAsrPlan, TranscribePlan};
use crate::types::worker_v2::{DecodeBudgetSeconds, SpeakerBackendV2, SpeakerSegmentV2};
use crate::utseg::TranscribeUtsegExecution;
use crate::utseg_evidence::{UtsegEvidencePhase, UtsegEvidenceSink, UtsegEvidenceTrace};

static PRODUCTION_REV_INFERENCE: RevAsrService = RevAsrService;

struct PendingAsr;

/// The ASR producer resolves language and identity before any downstream stage.
struct Recognized {
    response: AsrResponse,
    language: TranscriptLanguage,
    segmentation: Option<crate::utseg_route::UtsegRoute>,
    identity: crate::transcribe::types::AsrIdentity,
}

impl Recognized {
    fn admit<P: TranscribePlan>(
        response: AsrResponse,
        language: TranscriptLanguage,
        identity: crate::transcribe::types::AsrIdentity,
        plan: &crate::transcribe::types::AdmittedTranscribePlan<P>,
    ) -> Result<Self, ServerError> {
        let segmentation = plan
            .resolve_segmentation(&language)
            .map_err(|error| ServerError::Validation(error.to_string()))?;
        Ok(Self {
            response,
            language,
            segmentation,
            identity,
        })
    }
}

struct Postprocessed {
    asr: Recognized,
    utterances: Vec<Utterance>,
    speaker_assignments: SpeakerAssignmentOutcome,
}

/// The generated document, judged once at assembly by the producer
/// transition: admitted, or written with its diagnostics. A sum, so the
/// optional stages below are reachable only from the admitted arm.
enum Built {
    /// Admitted: the optional stages may run.
    Admitted(ChatReady),
    /// Diagnosed: the per-utterance stages (segmentation, morphosyntax) run
    /// outside the faulty utterances when the findings are confined to them,
    /// and are skipped, with the skips recorded, when they are not.
    Diagnosed(ChatDiagnosed),
}

/// An admitted document, ready for stages that require admission.
struct ChatReady {
    asr: Recognized,
    document: PostValidated,
    speaker_assignments: SpeakerAssignmentOutcome,
    /// Optional stages that ran and whose output was refused, so this
    /// document is the one from before them.
    shortfalls: Vec<Shortfall>,
}

/// A generated document that did not pass admission. It is written with its
/// diagnostics; only a per-utterance stage may still run on it, outside the
/// utterances its findings belong to.
struct ChatDiagnosed {
    /// The recognition, kept for a per-utterance stage that can still run on
    /// the utterances the findings do not belong to.
    asr: Recognized,
    document: crate::pipeline::post_validate::DiagnosedOutput,
    speaker_assignments: SpeakerAssignmentOutcome,
}

/// The admitted document after the last optional stage. Every stage ends in
/// the strict gate (or kept an admitted predecessor), so it needs no second
/// judgement before writing.
struct Analyzed {
    document: PostValidated,
    speaker_assignments: SpeakerAssignmentOutcome,
    shortfalls: Vec<Shortfall>,
}

/// What transcription hands the writer: the judged document and every piece
/// of requested work it does not carry. The shortfalls are facts about this
/// run, so they travel here, beside the document, and are reported whatever
/// its standing.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TranscribeOutput {
    /// The generated document, admitted or diagnosed.
    pub(crate) document: ProducedOutput,
    /// Requested work it does not carry, in the order it was recorded.
    pub(crate) shortfalls: Vec<Shortfall>,
}

/// No dedicated operation is distinct from an operation with empty evidence.
/// Every postprocessing route must retain its actual producer outcome.
#[derive(Debug)]
enum SpeakerAssignmentOutcome {
    AsrOnly,
    Dedicated(AdmittedSpeakerProjectionEvidence),
}

impl SpeakerAssignmentOutcome {
    fn review_warning(&self) -> Option<String> {
        match self {
            Self::AsrOnly => None,
            Self::Dedicated(evidence) => evidence.review_warning(),
        }
    }
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
    /// Take the state out, leaving the context to be given a new one with
    /// [`Self::map_state`]: how a branch on the state keeps the context.
    fn split_state(self) -> (S, TranscribePipelineContext<'a, (), P>) {
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
        (
            state,
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
                state: (),
            },
        )
    }

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
            TranscribeUtsegExecution::production(opts.plan.with_utseg()),
            PendingAsr,
        )
    }

    #[cfg(test)]
    fn admit_response(
        self,
        response: AsrResponse,
    ) -> Result<TranscribePipelineContext<'a, Recognized>, ServerError> {
        let language = resolved_asr_language(self.opts, &response)?;
        let identity = self.opts.plan.asr().identity();
        let recognized = Recognized::admit(response, language, identity, &self.opts.plan)?;
        Ok(self.map_state(|_| recognized))
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
        transition: crate::pipeline::plan::StageFuture<'_, T>,
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

    /// Count a planned stage that will not run, so the progress total stays
    /// honest; the reason is recorded on the file's diagnosed outcome.
    fn skip(&mut self, stage: StageId) {
        info!(stage = %stage, "Skipping transcribe stage: the generated document was not admitted");
        self.completed += 1;
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
) -> Result<TranscribeOutput, ServerError> {
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
        output,
        rev_evidence: _rev_evidence,
        speaker_assignments: _speaker_assignments,
    } = completed;
    Ok(output)
}

#[derive(Debug)]
struct CompletedTranscribePipeline {
    output: TranscribeOutput,
    rev_evidence: Option<RevAsrEvidenceTrace>,
    speaker_assignments: SpeakerAssignmentOutcome,
}

async fn run_transcribe_pipeline_with_rev_inference<'a>(
    audio_path: &'a Path,
    services: PipelineServices<'a>,
    opts: &'a TranscribeOptions,
    progress: Option<ProgressSender>,
    debug_dir: Option<&Path>,
    rev_inference: &'a dyn RevAsrEvidenceInference,
) -> Result<CompletedTranscribePipeline, ServerError> {
    let execution = TranscribeUtsegExecution::production(opts.plan.with_utseg());
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
        .run(
            StageId::AsrInfer,
            Box::pin(stage_asr_infer(ctx, rev_inference)),
        )
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
) -> Result<TranscribeOutput, ServerError> {
    let audio_path = replay.media_path().to_owned();
    let (tail, mut progress) = prepare_tail(&services, opts, execution, progress)?;
    let ctx = progress.run(StageId::AsrInfer, Box::pin(async {
        let response = replay.asr_response().clone();
        let language = resolved_asr_language(opts, &response)?;
        let identity = crate::transcribe::types::AsrIdentity::of_replay(replay.producer());
        let recognized = Recognized::admit(response, language, identity, &opts.plan)?;
        let mut ctx = TranscribePipelineContext::new(&audio_path, services, opts, DebugDumper::new(debug_dir),
            UtsegEvidenceSink::new(debug_dir), execution, recognized);
        if opts.diarize {
            ctx.speaker_segments = Some(replay.speaker_segments().ok_or_else(|| ServerError::Validation(
                format!("replay {} requested diarization but its manifest has no speaker-turn artifact", replay.recording_id())
            ))?.to_vec());
        }
        Ok(ctx)
    })).await?;
    Ok(finish_transcribe(ctx, tail, progress).await?.output)
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
                Box::pin(stage_speaker_diarization(&mut ctx)),
            )
            .await?;
    }
    let ctx = progress
        .run(
            StageId::AsrPostprocess,
            Box::pin(stage_asr_postprocess(ctx)),
        )
        .await?;
    let ctx = progress
        .run(StageId::BuildChat, Box::pin(stage_build_chat(ctx)))
        .await?;
    let (built, ctx) = ctx.split_state();
    match built {
        Built::Admitted(ready) => {
            let mut ctx = ctx.map_state(|()| ready);
            if let Some(policy) = tail.post_chat {
                ctx = progress
                    .run(
                        StageId::OptionalUtseg,
                        Box::pin(stage_run_utseg(ctx, policy)),
                    )
                    .await?;
            }
            finish_admitted(ctx, tail.morphosyntax, progress).await
        }
        Built::Diagnosed(ChatDiagnosed {
            asr,
            document,
            speaker_assignments,
        }) => {
            let mut shortfalls = Vec::new();
            // Utterance segmentation works utterance by utterance, so it
            // refuses only the utterances the findings belong to: when they
            // can be confined to some utterances, every other one is
            // segmented. Findings about the document as a whole leave
            // nothing to confine them to, and the stage is skipped.
            let document = match tail.post_chat {
                None => ProducedOutput::Diagnosed(document),
                Some(policy) => match document.localize() {
                    Ok(localized) => {
                        let (segmented, shortfall) = progress
                            .run(
                                StageId::OptionalUtseg,
                                Box::pin(stage_run_localized_utseg(&ctx, &asr, localized, policy)),
                            )
                            .await?;
                        shortfalls.push(shortfall);
                        segmented
                    }
                    Err(document) => {
                        progress.skip(StageId::OptionalUtseg);
                        shortfalls.push(Shortfall::StageSkipped {
                            stage: OptionalStage::UtteranceSegmentation,
                        });
                        ProducedOutput::Diagnosed(document)
                    }
                },
            };
            let diagnosed = match document {
                // Segmentation removed every finding's cause: the document
                // is admitted now, and continues as one.
                ProducedOutput::Admitted(document) => {
                    let ctx = ctx.map_state(|()| ChatReady {
                        asr,
                        document,
                        speaker_assignments,
                        shortfalls,
                    });
                    return finish_admitted(ctx, tail.morphosyntax, progress).await;
                }
                ProducedOutput::Diagnosed(diagnosed) => diagnosed,
            };
            // Morphosyntax works utterance by utterance too: the document,
            // as segmentation left it, is localized afresh (segmentation
            // renumbered its utterances) and every utterance but the faulty
            // ones is analyzed. Findings about the document as a whole skip
            // the stage, counted in progress and recorded.
            let document = if tail.morphosyntax {
                match diagnosed.localize() {
                    Ok(localized) => {
                        let (document, shortfall) = progress
                            .run(
                                StageId::OptionalMorphosyntax,
                                Box::pin(stage_run_localized_morphosyntax(&ctx, localized)),
                            )
                            .await?;
                        shortfalls.extend(shortfall);
                        document
                    }
                    Err(diagnosed) => {
                        progress.skip(StageId::OptionalMorphosyntax);
                        shortfalls.push(Shortfall::StageSkipped {
                            stage: OptionalStage::Morphosyntax,
                        });
                        ProducedOutput::Diagnosed(diagnosed)
                    }
                }
            } else {
                ProducedOutput::Diagnosed(diagnosed)
            };
            progress
                .run(
                    StageId::Serialize,
                    Box::pin(async {
                        Ok(CompletedTranscribePipeline {
                            output: TranscribeOutput {
                                document,
                                shortfalls,
                            },
                            rev_evidence: ctx.rev_evidence,
                            speaker_assignments,
                        })
                    }),
                )
                .await
        }
    }
}

/// The admitted document's remaining stages: morphosyntax, if planned, then
/// serialization.
async fn finish_admitted<P: TranscribePlan>(
    ctx: TranscribePipelineContext<'_, ChatReady, P>,
    morphosyntax: bool,
    mut progress: TranscribeProgress,
) -> Result<CompletedTranscribePipeline, ServerError> {
    let ctx = if morphosyntax {
        progress
            .run(
                StageId::OptionalMorphosyntax,
                Box::pin(stage_run_morphosyntax(ctx)),
            )
            .await?
    } else {
        ctx.map_state(|ready| Analyzed {
            document: ready.document,
            speaker_assignments: ready.speaker_assignments,
            shortfalls: ready.shortfalls,
        })
    };
    progress
        .run(
            StageId::Serialize,
            Box::pin(async {
                // Admitted already: assembly was judged by the producer
                // transition, and every later stage ends in the strict gate or
                // kept its admitted predecessor.
                Ok(CompletedTranscribePipeline {
                    output: TranscribeOutput {
                        document: ProducedOutput::Admitted(ctx.state.document),
                        shortfalls: ctx.state.shortfalls,
                    },
                    rev_evidence: ctx.rev_evidence,
                    speaker_assignments: ctx.state.speaker_assignments,
                })
            }),
        )
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

async fn stage_run_utseg<'a, P: TranscribePlan>(
    ctx: TranscribePipelineContext<'a, ChatReady, P>,
    post_chat_policy: crate::utseg::UtsegDecisionPolicy,
) -> Result<TranscribePipelineContext<'a, ChatReady, P>, ServerError> {
    let (
        ChatReady {
            asr,
            document,
            speaker_assignments,
            mut shortfalls,
        },
        ctx,
    ) = ctx.split_state();
    let route = asr.segmentation.as_ref().ok_or_else(|| {
        ServerError::Validation("post-CHAT segmentation requested from a disabled plan".into())
    })?;
    let filename = ctx
        .audio_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    ctx.dumper.dump_pre_utseg_chat(filename, document.as_str());
    let evidence_filename = ctx.audio_path.to_string_lossy();
    // Kept so a refused segmentation leaves the admitted document it started
    // from: one clone per file, against losing the whole transcript.
    let before = document.clone();
    let outcome =
        crate::utseg::process_utseg_with_evidence(crate::utseg::EvidenceRetainingUtsegRequest {
            document,
            lang: route.language(),
            services: ctx.services,
            fallback_policy: route.fallback(),
            decision_policy: post_chat_policy,
            evidence_filename: evidence_filename.as_ref(),
            evidence_sink: &ctx.utseg_evidence_sink,
            cancellation: crate::infer_retry::Cancellation::NotWired {
                reason: "TranscribePipelineContext has no job cancellation token wired yet",
            },
        })
        .await;
    let document = match not_applied(OptionalStage::UtteranceSegmentation, outcome)? {
        StageOutcome::Applied(result) => {
            ctx.dumper.dump_post_utseg_chat(filename, result.as_str());
            result
        }
        StageOutcome::NotApplied(shortfall) => {
            tracing::warn!(%filename, %shortfall, "transcribe kept its pre-segmentation document");
            shortfalls.push(shortfall);
            before
        }
    };
    Ok(ctx.map_state(|()| ChatReady {
        asr,
        document,
        speaker_assignments,
        shortfalls,
    }))
}

/// Post-CHAT segmentation of a diagnosed transcript, outside the utterances
/// its findings belong to. The held-out utterances are not sent to the model
/// and keep their generated form; the result is judged afresh.
///
/// Returns the document to continue with and what the file's outcome must
/// say, as the morphosyntax stage does: the held-out shortfall when the
/// stage applied; the stage's refusal, with the document from before it,
/// when segmentation added a finding of its own outside the held-out
/// utterances (as the strict gate refuses an admitted document's stage
/// output).
async fn stage_run_localized_utseg<P: TranscribePlan>(
    ctx: &TranscribePipelineContext<'_, (), P>,
    asr: &Recognized,
    localized: crate::pipeline::post_validate::LocalizedDiagnosis,
    post_chat_policy: crate::utseg::UtsegDecisionPolicy,
) -> Result<(ProducedOutput, Shortfall), ServerError> {
    let route = asr.segmentation.as_ref().ok_or_else(|| {
        ServerError::Validation("post-CHAT segmentation requested from a disabled plan".into())
    })?;
    let filename = ctx
        .audio_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    ctx.dumper.dump_pre_utseg_chat(filename, localized.as_str());
    let evidence_filename = ctx.audio_path.to_string_lossy();
    let held_out = held_out_shortfall(OptionalStage::UtteranceSegmentation, localized.held_out());
    // Kept so a refused segmentation leaves the document it started from.
    let before = localized.clone().into_diagnosed();
    let segmented = crate::utseg::process_localized_utseg_with_evidence(
        crate::utseg::EvidenceRetainingUtsegRequest {
            document: localized,
            lang: route.language(),
            services: ctx.services,
            fallback_policy: route.fallback(),
            decision_policy: post_chat_policy,
            evidence_filename: evidence_filename.as_ref(),
            evidence_sink: &ctx.utseg_evidence_sink,
            cancellation: crate::infer_retry::Cancellation::NotWired {
                reason: "TranscribePipelineContext has no job cancellation token wired yet",
            },
        },
    )
    .await;
    Ok(
        match not_applied(OptionalStage::UtteranceSegmentation, segmented)? {
            StageOutcome::Applied(segmented) => {
                ctx.dumper
                    .dump_post_utseg_chat(filename, segmented.as_str());
                (segmented, held_out)
            }
            StageOutcome::NotApplied(shortfall) => {
                tracing::warn!(%filename, %shortfall, "transcribe kept its pre-segmentation document");
                (ProducedOutput::Diagnosed(before), shortfall)
            }
        },
    )
}

/// What one optional stage did with its document: by default an admitted
/// one, whose stage output must pass the strict gate.
enum StageOutcome<T = PostValidated> {
    /// The stage's output passed its judgement.
    Applied(T),
    /// The stage's own output could not be admitted; the caller keeps the
    /// document it started from and records this.
    NotApplied(Shortfall),
}

/// Classify an optional stage's result. Only a refusal of the stage's OWN
/// output (`ServerError::OutputAdmission`) is a shortfall: the admitted
/// document before the stage is still good, so losing it would discard a
/// whole transcript for one stage's defect. Every other failure (a worker
/// that died, cancellation, persistence) is still the file's error, so the
/// runner's retry and failure policy apply to it unchanged.
fn not_applied<T>(
    stage: OptionalStage,
    result: Result<T, ServerError>,
) -> Result<StageOutcome<T>, ServerError> {
    match result {
        Ok(output) => Ok(StageOutcome::Applied(output)),
        Err(ServerError::OutputAdmission { command, details }) => {
            // The full refusal goes to the server log once; the shortfall,
            // which every poll copies, keeps a bounded record of it.
            tracing::warn!(%command, stage = stage.name(), refusal = %details, "optional stage output refused");
            let refusal = match details {
                OutputAdmissionRefusal::Judged { bar, first, rest } => {
                    StageRefusalRecord::judged(bar, std::iter::once(&first).chain(&rest))
                }
                OutputAdmissionRefusal::Unestablished(reason) => {
                    StageRefusalRecord::Unestablished { reason }
                }
            };
            Ok(StageOutcome::NotApplied(Shortfall::StageNotApplied {
                stage,
                refusal,
            }))
        }
        Err(other) => Err(other),
    }
}

/// The shortfall of a per-utterance stage that ran outside the utterances a
/// diagnosed document's findings are confined to.
fn held_out_shortfall(
    stage: OptionalStage,
    held_out: &crate::pipeline::post_validate::HeldOutUtterances,
) -> Shortfall {
    Shortfall::StageHeldOut {
        stage,
        held_out_utterances: held_out.len() as u64,
        first_held_out: held_out
            .ordinals()
            .take(crate::api::FileOutputDiagnostics::FIRST_FINDINGS)
            .map(|ordinal| ordinal as u64 + 1)
            .collect(),
    }
}

/// Morphosyntax on a diagnosed transcript, outside the utterances its
/// findings belong to: those are never sent to a worker and keep their form.
///
/// Returns the document to continue with and what the file's outcome must
/// say: the held-out shortfall when the stage applied; the stage's refusal,
/// with the document from before it, when the analysis added a finding of its
/// own (as the strict gate refuses an admitted document's stage output); and
/// nothing when an `@Options: CA` document is declined, as the admitted path
/// declines one.
async fn stage_run_localized_morphosyntax<P: TranscribePlan>(
    ctx: &TranscribePipelineContext<'_, (), P>,
    localized: crate::pipeline::post_validate::LocalizedDiagnosis,
) -> Result<(ProducedOutput, Option<Shortfall>), ServerError> {
    let filename = ctx
        .audio_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    ctx.dumper
        .dump_pre_morphosyntax_chat(filename, localized.as_str());
    let held_out = held_out_shortfall(OptionalStage::Morphosyntax, localized.held_out());
    // Kept so a refused analysis leaves the document it started from.
    let before = localized.clone().into_diagnosed();
    let empty_mwt = std::collections::BTreeMap::new();
    let mor_params = transcribe_morphosyntax_params(&empty_mwt);
    let analyzed = crate::pipeline::morphosyntax::run_localized_morphosyntax(
        localized,
        ctx.services,
        &mor_params,
    )
    .await;
    Ok(match not_applied(OptionalStage::Morphosyntax, analyzed)? {
        StageOutcome::Applied(crate::pipeline::morphosyntax::LocalizedMorphosyntax::Analyzed(
            document,
        )) => (document, Some(held_out)),
        StageOutcome::Applied(crate::pipeline::morphosyntax::LocalizedMorphosyntax::Declined(
            document,
        )) => (ProducedOutput::Diagnosed(document), None),
        StageOutcome::NotApplied(shortfall) => {
            tracing::warn!(%filename, %shortfall, "transcribe kept its pre-morphosyntax document");
            (ProducedOutput::Diagnosed(before), Some(shortfall))
        }
    })
}

/// Transcribe's morphosyntax parameters, whatever the document's standing.
fn transcribe_morphosyntax_params(
    empty_mwt: &crate::chat_ops::morphosyntax_ops::MwtDict,
) -> MorphosyntaxParams<'_> {
    MorphosyntaxParams {
        tokenization_mode: TokenizationMode::Preserve,
        multilingual_policy: MultilingualPolicy::ProcessAll,
        mwt: empty_mwt,
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
    }
}

async fn stage_run_morphosyntax<'a, P: TranscribePlan>(
    ctx: TranscribePipelineContext<'a, ChatReady, P>,
) -> Result<TranscribePipelineContext<'a, Analyzed, P>, ServerError> {
    let input = ctx.state.document.as_str();
    let filename = ctx
        .audio_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown");
    ctx.dumper.dump_pre_morphosyntax_chat(filename, input);
    let empty_mwt = std::collections::BTreeMap::new();
    let mor_params = transcribe_morphosyntax_params(&empty_mwt);
    let (ready, ctx) = ctx.split_state();
    let ChatReady {
        document: before,
        speaker_assignments,
        mut shortfalls,
        ..
    } = ready;
    // Kept so refused analysis leaves the admitted document it started from.
    let analyzed = async {
        let parsed = crate::pipeline::morphosyntax::ParsedFile::from_output(
            before.clone(),
            mor_params.policy.ca_policy,
        )?;
        crate::pipeline::morphosyntax::run_admitted_morphosyntax(parsed, ctx.services, &mor_params)
            .await
    }
    .await;
    let document = match not_applied(OptionalStage::Morphosyntax, analyzed)? {
        StageOutcome::Applied(document) => document,
        StageOutcome::NotApplied(shortfall) => {
            tracing::warn!(%filename, %shortfall, "transcribe kept its pre-morphosyntax document");
            shortfalls.push(shortfall);
            before
        }
    };
    Ok(ctx.map_state(|()| Analyzed {
        document,
        speaker_assignments,
        shortfalls,
    }))
}

mod assembly;
mod inference;
mod preprocessing;
use assembly::stage_build_chat;
use inference::{stage_asr_infer, stage_speaker_diarization};
#[cfg(test)]
use preprocessing::prepare_asr_chunks_with_snapshot;
use preprocessing::stage_asr_postprocess;
pub(crate) use preprocessing::{
    apply_prechat_assignments, build_prechat_utseg_items, prepare_asr_chunks,
};
#[cfg(test)]
mod tests;
