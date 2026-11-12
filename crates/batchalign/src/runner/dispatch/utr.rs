//! UTR (untimed utterance timing recovery) orchestration.
//!
//! This module keeps the full CHAT-level timing-recovery algorithm in Rust:
//! parse CHAT, decide between partial-window and full-file recovery, fetch raw
//! timed tokens from the selected backend, and inject timing bullets back into
//! the AST. Python is only used for the worker-hosted ASR path.

#[cfg(test)]
use crate::revai::FetchedRevAsrEvidence;

use std::path::Path;

use super::options::ResolvedUtrStrategy;
use crate::api::LanguageCode3;
use crate::cache::{
    InferenceLease, UtrAsrCacheEligibility, UtrAsrCacheNamespace, UtteranceCache, tasks,
};
use crate::chat_ops::CacheKey;
use crate::chat_ops::fa::coordinates::{FaWindow, FileMs, Recording, WindowMs};
use crate::chat_ops::fa::origin::EngineId;
use crate::chat_ops::fa::timing::{SpanRejections, WordSpan};
use crate::media::probe::AudioDuration;
use crate::options::UtrEngine;
use crate::params::CachePolicy;
use crate::provenance::{
    UtrPassOutcome, UtrRecognitionWindow, UtrTokenRejection, UtrTokenRejectionReason,
};
use crate::runner::debug_dumper::DebugDumper;
use crate::types::worker_v2::DecodeBudgetSeconds;
use batchalign_types::interval::AdmittedInterval;
use tracing::{info, warn};

#[path = "utr_full.rs"]
mod full;
#[path = "utr_partial.rs"]
mod partial;
#[path = "utr_work.rs"]
mod work;
pub(crate) use work::CompletedRecognition;
#[cfg(test)]
pub(crate) use work::PreparedRecognition;

/// How the recognizer inputs for a completed correspondence were obtained.
#[derive(Debug, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum UtrRecognitionRoute {
    Full,
    Partial,
    FullAfterPartial {
        #[serde(serialize_with = "serialize_abandoned")]
        abandoned: std::sync::Arc<AbandonedPartialAttempt>,
    },
}

#[derive(Clone)]
enum UtrFullRoute {
    Direct,
    AfterPartial(std::sync::Arc<AbandonedPartialAttempt>),
}

fn serialize_abandoned<S: serde::Serializer>(
    abandoned: &std::sync::Arc<AbandonedPartialAttempt>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serde::Serialize::serialize(abandoned.as_ref(), serializer)
}

impl From<UtrFullRoute> for UtrRecognitionRoute {
    fn from(route: UtrFullRoute) -> Self {
        match route {
            UtrFullRoute::Direct => Self::Full,
            UtrFullRoute::AfterPartial(abandoned) => Self::FullAfterPartial { abandoned },
        }
    }
}

#[derive(Debug, serde::Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub(crate) enum AbandonedPartialAttempt {
    PlanRefused {
        reason: PartialPlanRefusal,
    },
    ExecutionStopped {
        planned_windows: Vec<UtrWindowBounds>,
        completed_windows: Vec<AbandonedRecognitionWindow>,
        reason: PartialExecutionStop,
    },
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct AbandonedRecognitionWindow {
    accepted_tokens: Vec<crate::chat_ops::fa::utr::AsrTimingToken>,
    evidence: UtrRecognitionWindow,
}

#[derive(Debug, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum PartialPlanRefusal {
    Recording { error: String },
    WindowPlanning { error: String },
}

#[derive(Debug, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum PartialExecutionStop {
    NoWindows,
    SegmentExtraction {
        window: UtrWindowBounds,
        error: String,
    },
    Recognition {
        window: UtrWindowBounds,
        error: String,
    },
}

#[derive(Debug)]
pub(crate) struct UtrWindowBounds(FaWindow);

impl From<FaWindow> for UtrWindowBounds {
    fn from(window: FaWindow) -> Self {
        Self(window)
    }
}

impl serde::Serialize for UtrWindowBounds {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("UtrWindowBounds", 2)?;
        state.serialize_field("start_ms", &self.0.audio_start().get())?;
        state.serialize_field("end_ms", &self.0.end().get())?;
        state.end()
    }
}

/// Immutable runtime inputs for one UTR execution.
#[derive(Clone, Copy)]
pub(in crate::runner) struct UtrPassContext<'a> {
    /// Which source recovery transition owns this attempt.
    pub pass: crate::api::UtrPass,
    /// Audio file used to recover utterance timing.
    pub audio_path: &'a Path,
    /// CHAT language for ASR/UTR normalization.
    pub lang: &'a LanguageCode3,
    /// The align job's services: worker pool and cache, plus the job's GPU
    /// model plan, by which worker-hosted UTR requests are routed to the
    /// worker the job's FA requests also reach.
    pub services: crate::fa::FaServices<'a>,
    /// Audio identity used to key UTR cache entries.
    pub audio_identity: &'a crate::chat_ops::fa::AudioIdentity,
    /// Cache policy selected for the current job.
    pub cache_policy: crate::params::UtrCachePolicies,
    /// How long the audio runs, when the pipeline already probed it.
    pub audio_duration: Option<AudioDuration>,
    /// FA aggregation preference. Used by the two-pass UTR
    /// strategy to compare FA grouping outcomes and detect the wider-window
    /// regression on non-English files.
    pub aggregation_target: Option<crate::types::engines::FaAggregationTarget>,
    /// Display filename for logging.
    pub filename: &'a str,
    /// Selected UTR backend.
    pub engine: &'a UtrEngine,
    /// Resolved strategy and its submitted tuning, shared by both recovery paths.
    pub strategy: &'a ResolvedUtrStrategy,
    /// Debug artifact writer for offline replay.
    pub dumper: &'a DebugDumper,
}

impl<'a> UtrPassContext<'a> {
    /// The language UTR requests its recognition in: the one value both the
    /// worker route (and so the decode decision a stock Whisper worker is
    /// spawned with) and the UTR cache namespace are derived from.
    fn asr_language(&self) -> crate::transcribe::SingleAsrLanguage {
        crate::transcribe::SingleAsrLanguage::One(self.lang.clone())
    }

    /// The recording this pass is working against.
    ///
    /// Delegates to [`AudioContext::recording`] rather than repeating its
    /// probe-or-use logic. This context carries the same two facts an
    /// `AudioContext` does (`audio_path`, `audio_duration`), so a second
    /// derivation here would be the duplication that method exists to end, with
    /// its own divergent error message. One question, one owner.
    async fn recording(&self) -> Result<Recording, crate::error::ServerError> {
        crate::params::AudioContext {
            audio_path: self.audio_path,
            audio_identity: self.audio_identity,
            audio_duration: self.audio_duration,
        }
        .recording()
        .await
    }
}

/// Instantiate the already-resolved strategy with this recording's grouping
/// limits. Auto has been resolved to global at option extraction; two-pass
/// carries the submitted configuration instead of constructing defaults here.
///
/// Takes the recording the pass already derived rather than deriving another
/// from the context, so the grouping comparison and the token bounds are
/// checked against one value.
fn resolve_strategy(
    context: &UtrPassContext<'_>,
    recording: Recording,
) -> Box<dyn crate::chat_ops::fa::UtrStrategy> {
    let grouping_context =
        context
            .aggregation_target
            .map(|aggregation_target| crate::chat_ops::fa::GroupingContext {
                recording,
                aggregation_target,
            });

    match context.strategy {
        ResolvedUtrStrategy::Global => Box::new(crate::chat_ops::fa::GlobalUtr),
        ResolvedUtrStrategy::TwoPass(config) => Box::new(crate::chat_ops::fa::TwoPassOverlapUtr {
            grouping_context,
            config: config.clone(),
        }),
    }
}

/// Run ASR and inject UTR timing into a parsed `ChatFile`.
///
/// Returns the completed recovery result, or the typed failure. Callers must
/// record that outcome through `UtrRecovery` before continuing FA. Failed
/// inference leaves the document unchanged; owned recovery commits only a
/// completed candidate.
///
/// When `progress` is provided, per-window updates are sent during partial UTR
/// so frontends can show "Recovering utterance timing 2/5" etc.
///
/// Observes `chat_file` without mutation or a serialize/re-parse cycle. The
/// caller retains the source AST and can pass it directly to FA.
pub(in crate::runner) async fn run_utr_pass(
    source: &crate::chat_ops::fa::SourceBoundUtrInput<'_>,
    context: UtrPassContext<'_>,
    progress: Option<&super::super::util::ProgressSender>,
) -> Result<UtrPassOutcome, crate::error::ServerError> {
    let chat_file = source.document();
    let (timed, untimed) = crate::chat_ops::fa::count_utterance_timing(chat_file);
    let total_utts = timed + untimed;

    if untimed == 0 {
        info!(context.filename, "UTR pass: no untimed utterances");
        return Ok(UtrPassOutcome::NothingUntimed(
            crate::chat_ops::fa::utr::UtrResult::not_run_no_untimed(timed),
        ));
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
        && context
            .audio_duration
            .is_some_and(|duration| duration.length().0 > 60_000);

    if use_partial {
        // The recording every window and every recovered token is checked
        // against. Asked of the context so this path shares the crate's single
        // derivation; a zero-length or unprobeable recording degrades to
        // full-file recovery rather than failing, because partial UTR is an
        // optimisation and the full path does not need the bound.
        let recording = match context.recording().await {
            Ok(recording) => recording,
            Err(why) => {
                warn!(
                    context.filename,
                    error = %why,
                    "Partial UTR needs a non-empty recording, falling back to full-file recovery"
                );
                let route = UtrFullRoute::AfterPartial(std::sync::Arc::new(
                    AbandonedPartialAttempt::PlanRefused {
                        reason: PartialPlanRefusal::Recording {
                            error: why.to_string(),
                        },
                    },
                ));
                return run_utr_pass_full(source, context, progress, route).await;
            }
        };
        // The recording built above is the single derivation of the audio's
        // length on this path. It used to be derived a SECOND time three lines
        // later, from the context's raw milliseconds through an `expect` whose
        // safety rested on a control-flow invariant argued in a four-line
        // comment.
        // Passing the recording deletes the second derivation, the panic and
        // the argument together.
        //
        // The windows come back as `FaWindow`s, proven inside this recording,
        // and stay that way through the segment extraction (whose transcode
        // then needs no probe of the source to bound a damaged decode) and the
        // token conversion below. Partial UTR is an optimisation, so a window
        // that is somehow not inside the recording degrades to full-file
        // recovery like the other failures here.
        let windows = match crate::chat_ops::fa::find_untimed_windows(chat_file, &recording, 500) {
            Ok(windows) => windows,
            Err(why) => {
                warn!(
                    context.filename,
                    error = %why,
                    "Partial UTR window is not inside the recording, falling back to full-file recovery"
                );
                let route = UtrFullRoute::AfterPartial(std::sync::Arc::new(
                    AbandonedPartialAttempt::PlanRefused {
                        reason: PartialPlanRefusal::WindowPlanning {
                            error: why.to_string(),
                        },
                    },
                ));
                return run_utr_pass_full(source, context, progress, route).await;
            }
        };

        let total_windows = windows.len() as i64;
        let mut pending = match partial::PendingWindow::start(windows, context) {
            Ok(pending) => pending,
            Err(route) => {
                info!(
                    context.filename,
                    "Partial UTR: no windows found, falling back to full-file recovery"
                );
                if let Some(tx) = progress {
                    use super::super::util::{FileStage, ProgressUpdate};
                    let _ = tx.send(ProgressUpdate::new(
                        FileStage::RecoveringUtteranceTiming,
                        Some(0),
                        Some(1),
                    ));
                }
                return run_utr_pass_full(source, context, progress, route).await;
            }
        };
        info!(
            context.filename,
            windows = total_windows,
            "Partial UTR: running ASR on untimed windows only"
        );

        let mut completed_count = 0;
        let completed = loop {
            let step = match pending.run().await? {
                partial::WindowResult::Abandoned(route) => {
                    return run_utr_pass_full(source, context, progress, route).await;
                }
                partial::WindowResult::Recognized(step) => step,
            };
            completed_count += 1;
            if let Some(tx) = progress {
                use super::super::util::{FileStage, ProgressUpdate};
                let _ = tx.send(ProgressUpdate::new(
                    FileStage::RecoveringUtteranceTiming,
                    Some(completed_count),
                    Some(total_windows),
                ));
            }
            match *step {
                partial::RecognizedWindow::Next(next) => pending = next,
                partial::RecognizedWindow::Completed(completed) => break completed,
            }
        };
        let prepared = work::PreparedRecognition::partial(completed)?;

        if context.dumper.is_enabled() {
            let text = batchalign_transform::serialize::to_chat_string(chat_file);
            context.dumper.dump_utr_input(context.filename, &text);
        }
        context
            .dumper
            .dump_utr_tokens(context.filename, prepared.tokens().tokens());

        let strategy = resolve_strategy(&context, recording);
        let completed = prepared.run(source, strategy, progress).await?;
        let utr_result = completed.observation().result();

        info!(
            context.filename,
            candidate_injected = utr_result.injected(),
            skipped = utr_result.skipped(),
            unmatched = utr_result.unmatched(),
            "UTR partial observation complete"
        );

        if context.dumper.is_enabled() {
            context
                .dumper
                .dump_utr_observation(context.filename, utr_result);
        }
        return Ok(UtrPassOutcome::Recovered(completed));
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

    run_utr_pass_full(source, context, progress, UtrFullRoute::Direct).await
}

/// Run the full-file UTR path with cache reuse.
async fn run_utr_pass_full(
    source: &crate::chat_ops::fa::SourceBoundUtrInput<'_>,
    context: UtrPassContext<'_>,
    progress: Option<&super::super::util::ProgressSender>,
    route: UtrFullRoute,
) -> Result<UtrPassOutcome, crate::error::ServerError> {
    let result = run_utr_pass_full_inner(source, context, progress, route.clone()).await;
    match (result, route) {
        (Err(cause), UtrFullRoute::AfterPartial(abandoned)) => {
            match context.dumper.dump_utr_failed_fallback(
                context.filename,
                context.pass,
                context.engine,
                &abandoned,
                &cause,
            ) {
                Ok(_) => Err(cause),
                Err(failure) => Err(crate::error::UtrEvidenceRetentionFailure::new(
                    cause,
                    failure.to_string(),
                )
                .into()),
            }
        }
        (result, _) => result,
    }
}

async fn run_utr_pass_full_inner(
    source: &crate::chat_ops::fa::SourceBoundUtrInput<'_>,
    context: UtrPassContext<'_>,
    progress: Option<&super::super::util::ProgressSender>,
    route: UtrFullRoute,
) -> Result<UtrPassOutcome, crate::error::ServerError> {
    let chat_file = source.document();
    let recognition = full::FullRecognition::run(context, route).await?;
    let recording = recognition.recording();
    let prepared = work::PreparedRecognition::full(recognition)?;

    if context.dumper.is_enabled() {
        let text = batchalign_transform::serialize::to_chat_string(chat_file);
        context.dumper.dump_utr_input(context.filename, &text);
    }
    context
        .dumper
        .dump_utr_tokens(context.filename, prepared.tokens().tokens());

    let strategy = resolve_strategy(&context, recording);
    let completed = prepared.run(source, strategy, progress).await?;
    let utr_result = completed.observation().result();

    info!(
        context.filename,
        candidate_injected = utr_result.injected(),
        skipped = utr_result.skipped(),
        unmatched = utr_result.unmatched(),
        "UTR observation complete"
    );

    if context.dumper.is_enabled() {
        context
            .dumper
            .dump_utr_observation(context.filename, utr_result);
    }
    Ok(UtrPassOutcome::Recovered(completed))
}

/// Fetch one UTR ASR response from the selected backend and map it into the
/// shared `AsrResponse` cache format.
fn infer_utr_asr_response<'run>(
    audio_path: &'run Path,
    decode_budget: Option<DecodeBudgetSeconds>,
    context: UtrPassContext<'run>,
) -> futures::future::BoxFuture<
    'run,
    Result<crate::transcribe::AsrResponse, crate::error::ServerError>,
> {
    Box::pin(async move {
        match crate::transcribe::AsrBackend::from(context.engine).as_non_rev() {
            None => {
                let lang = crate::types::revai_language::RevLanguage::admit(
                    &crate::api::AsrLanguageRequest::One(context.lang.clone()),
                )
                .map_err(|refusal| crate::error::ServerError::Validation(refusal.to_string()))?;
                let provider_media =
                    crate::revai::PreparedRevProviderMedia::from_source(audio_path)
                        .await
                        .map_err(|error| {
                            crate::error::ServerError::Persistence(error.to_string())
                        })?;
                let request = crate::revai::RevAsrEvidenceRequest::new(
                    provider_media,
                    &lang,
                    // One speaker: UTR recovers timing for a transcript whose
                    // speakers are already known, and asks Rev.AI to separate none.
                    crate::transcribe::SpeakerCount::Known(std::num::NonZeroU8::MIN),
                    &crate::revai::RevAsrModelRevision::current(),
                )
                .map_err(|error| crate::error::ServerError::Persistence(error.to_string()))?;
                let service = crate::revai::RevAsrService::new();
                let evidence_identity =
                    rev_utr_evidence_identity(context.filename, request.cache_key());
                resolve_rev_utr_asr_response(
                    &request,
                    context.services.pipeline.cache,
                    context.cache_policy.rev_evidence(),
                    &service,
                    context.dumper,
                    &evidence_identity,
                )
                .await
            }
            Some(backend) => {
                // UTR runs each engine's default configuration: no extras.
                // `GpuModelPlan::alignment` plans the same, and the route check
                // refuses a UTR request whose extras differ from the plan's.
                let no_extras = std::collections::BTreeMap::new();
                crate::transcribe::infer_asr(
                    context.services.pipeline.pool,
                    &crate::transcribe::AsrInferParams {
                        backend,
                        audio_path,
                        lang: &context.asr_language(),
                        // UTR never asks a provider to separate speakers: the
                        // count of one it used to pass meant exactly this.
                        diarization: crate::types::worker_v2::ProviderDiarizationV2::NotRequested,
                        extras: &no_extras,
                        gpu: context.services.gpu,
                        decode_budget,
                    },
                )
                .await
            }
        }
    })
}

/// A normalized UTR cache read has exactly two safe outcomes: replay a valid
/// response, or authorize inference. Storage errors and malformed values are
/// neither, so they fail closed instead of silently becoming paid work.
#[derive(Debug)]
enum UtrAsrCacheLookup {
    /// Boxed: a stored response dwarfs the miss arm, so an unboxed enum would
    /// make every cache miss pay the size of a hit it did not get.
    Hit(Box<crate::transcribe::AsrResponse>),
    Miss(UtrAsrCacheMiss),
}

/// Proof that the normalized UTR cache was safely absent or explicitly
/// bypassed. Consuming it is the only route to inference, and inference is the
/// only route to [`UtrAsrInferred`], whose commit is the only write.
#[derive(Debug)]
struct UtrAsrCacheMiss {
    /// Whether, and how, the result may be stored and shared.
    authority: UtrAsrWriteAuthority,
}

/// What a miss is allowed to do with the response it produces.
///
/// A sum rather than an `Option<namespace>`: the two arms are different facts
/// about the run, and the store path must say which one it is in.
#[derive(Debug)]
enum UtrAsrWriteAuthority {
    /// Every model is pinned: the response is stored under `namespace`, and
    /// `lease` makes this the ONE inference for the cache key in this process.
    /// A concurrent identical request waits on the lease and then replays
    /// what this one stores, rather than running the same inference again.
    Shared {
        namespace: UtrAsrCacheNamespace,
        lease: InferenceLease,
    },
    /// A floating plan: nothing is stored, because no later run could prove a
    /// stored row came from the same weights.
    Unshared,
}

impl UtrAsrCacheMiss {
    async fn infer(
        self,
        audio_path: &Path,
        decode_budget: Option<DecodeBudgetSeconds>,
        context: UtrPassContext<'_>,
    ) -> Result<UtrAsrInferred, crate::error::ServerError> {
        match (context.cache_policy.response(), context.engine) {
            (CachePolicy::RequireCache, UtrEngine::Whisper | UtrEngine::HkTencent) => {
                Err(crate::error::ServerError::Persistence(
                    "required UTR ASR evidence is missing for the selected ASR backend".to_owned(),
                ))
            }
            // The normalized UTR entry is derived. Rev may rebuild it from the
            // separately retained raw transcript, whose own required-cache
            // gate still refuses a provider call on a raw miss.
            (CachePolicy::RequireCache, UtrEngine::RevAi)
            | (CachePolicy::UseCache | CachePolicy::SkipCache, _) => {
                self.run(infer_utr_asr_response(audio_path, decode_budget, context))
                    .await
            }
        }
    }

    /// Run one inference under this miss's authority.
    ///
    /// The inference is a parameter so the typestate (miss, then inferred,
    /// then committed) is the same whichever backend produced the response.
    async fn run(
        self,
        inference: impl Future<
            Output = Result<crate::transcribe::AsrResponse, crate::error::ServerError>,
        >,
    ) -> Result<UtrAsrInferred, crate::error::ServerError> {
        let response = inference.await?;
        Ok(UtrAsrInferred {
            response,
            authority: self.authority,
        })
    }
}

/// A response produced under a miss's authority, not yet stored.
#[derive(Debug)]
struct UtrAsrInferred {
    response: crate::transcribe::AsrResponse,
    authority: UtrAsrWriteAuthority,
}

impl UtrAsrInferred {
    /// Store the response (when the plan is pinned) and hand it back.
    ///
    /// The one write both recovery paths use. A failed write (or a response
    /// that does not serialize) is logged and otherwise ignored: the response
    /// is still used for this run, and a later run recomputes what was not
    /// stored.
    async fn commit(
        self,
        cache: &UtteranceCache,
        cache_key: &CacheKey,
        filename: &str,
    ) -> crate::transcribe::AsrResponse {
        match self.authority {
            // Nothing is written for a floating plan. A row whose models
            // cannot be named would be unreadable by construction anyway.
            UtrAsrWriteAuthority::Unshared => {}
            UtrAsrWriteAuthority::Shared { namespace, lease } => {
                let stored = match serde_json::to_value(&self.response) {
                    Ok(value) => cache
                        .put(cache_key.as_str(), tasks::UTR_ASR, &namespace, &value)
                        .await
                        .map_err(|error| error.to_string()),
                    Err(error) => Err(error.to_string()),
                };
                match stored {
                    // Waiters re-read the cache after this lease drops; the
                    // mark lets a forced refresh replay this fresh result.
                    Ok(()) => lease.mark_committed(),
                    // A waiter then finds no row and runs its own inference:
                    // the failure costs a repeat, never a wrong answer.
                    Err(error) => {
                        warn!(filename, %error, "Failed to cache UTR ASR response (non-fatal)");
                    }
                }
            }
        }
        self.response
    }
}

/// Whether this pass may use the UTR ASR cache, and under which namespace.
///
/// Resolved once per pass from the engine and the language, so both recovery
/// paths key their rows the same way rather than each deriving it.
///
/// A plan that cannot be resolved is treated as uncacheable rather than as a
/// job failure: recovery is an optimisation, and when the models cannot be
/// named the safe direction is to recompute, never to reuse rows that may have
/// come from other weights.
fn utr_cache_eligibility(context: UtrPassContext<'_>) -> UtrAsrCacheEligibility {
    let models = match crate::model_manifest::utr_pinned_models(
        context.engine,
        context.lang,
        context.services.pipeline.pool.tencent_engine(),
    ) {
        Ok(models) => models,
        Err(error) => {
            warn!(
                context.filename,
                engine = context.engine.as_wire_name(),
                %error,
                "UTR ASR models could not be resolved; running without cache reuse"
            );
            return UtrAsrCacheEligibility::Floating;
        }
    };
    match UtrAsrCacheNamespace::for_pinned_plan(context.engine, &models, &context.asr_language()) {
        Ok(eligibility) => eligibility,
        Err(error) => {
            warn!(
                context.filename,
                engine = context.engine.as_wire_name(),
                %error,
                "UTR ASR producer could not be named; running without cache reuse"
            );
            UtrAsrCacheEligibility::Floating
        }
    }
}

/// Look up one UTR ASR response, or take the single right to infer it.
///
/// For a pinned plan the lookup runs under an [`InferenceLease`] keyed by the
/// same cache key the row is stored under (the key string already names the
/// engine, the audio and the language, which determine the namespace). A
/// second identical request therefore waits for the first to commit and then
/// replays its row, instead of each running the same inference.
async fn lookup_utr_asr_cache(
    cache: &UtteranceCache,
    cache_key: &CacheKey,
    eligibility: &UtrAsrCacheEligibility,
    policy: CachePolicy,
) -> Result<UtrAsrCacheLookup, crate::error::ServerError> {
    let namespace = match eligibility {
        // A floating plan cannot promise a stored row came from the same
        // weights, so there is nothing safe to read and nothing to share.
        UtrAsrCacheEligibility::Floating => {
            return Ok(UtrAsrCacheLookup::Miss(UtrAsrCacheMiss {
                authority: UtrAsrWriteAuthority::Unshared,
            }));
        }
        UtrAsrCacheEligibility::Pinned(namespace) => namespace.clone(),
    };
    // A warm row needs no lease: hits stay lock-free, and only a miss pays
    // for single flight (then reads again under the lease, below).
    match policy {
        CachePolicy::UseCache | CachePolicy::RequireCache => {
            if let Some(response) = read_utr_asr_cache(cache, cache_key, &namespace).await? {
                return Ok(UtrAsrCacheLookup::Hit(Box::new(response)));
            }
        }
        CachePolicy::SkipCache => {}
    }
    let lease = InferenceLease::acquire(cache_key, cache).await;
    match policy {
        // A forced refresh still replays a result another request committed
        // while this one waited: that result is exactly as fresh.
        CachePolicy::SkipCache if !lease.observed_commit_while_waiting() => {
            return Ok(UtrAsrCacheLookup::Miss(UtrAsrCacheMiss {
                authority: UtrAsrWriteAuthority::Shared { namespace, lease },
            }));
        }
        CachePolicy::UseCache | CachePolicy::RequireCache | CachePolicy::SkipCache => {}
    }
    match read_utr_asr_cache(cache, cache_key, &namespace).await? {
        Some(response) => Ok(UtrAsrCacheLookup::Hit(Box::new(response))),
        None => Ok(UtrAsrCacheLookup::Miss(UtrAsrCacheMiss {
            authority: UtrAsrWriteAuthority::Shared { namespace, lease },
        })),
    }
}

/// Read one stored UTR ASR response. A malformed row fails closed.
async fn read_utr_asr_cache(
    cache: &UtteranceCache,
    cache_key: &CacheKey,
    namespace: &UtrAsrCacheNamespace,
) -> Result<Option<crate::transcribe::AsrResponse>, crate::error::ServerError> {
    let stored = cache
        .get(cache_key.as_str(), tasks::UTR_ASR, namespace)
        .await
        .map_err(|error| crate::error::ServerError::Persistence(error.to_string()))?;
    let Some(stored) = stored else {
        return Ok(None);
    };
    serde_json::from_value(stored).map(Some).map_err(|error| {
        crate::error::ServerError::Persistence(format!(
            "invalid cached UTR ASR response for key {cache_key}: {error}"
        ))
    })
}

async fn resolve_rev_utr_asr_response<I: crate::revai::RevAsrEvidenceInference>(
    request: &crate::revai::RevAsrEvidenceRequest,
    cache: &UtteranceCache,
    policy: CachePolicy,
    inference: &I,
    dumper: &DebugDumper,
    evidence_identity: &str,
) -> Result<crate::transcribe::AsrResponse, crate::error::ServerError> {
    let resolution = crate::revai::resolve_rev_asr_evidence(request, cache, policy, inference)
        .await
        .map_err(crate::revai::rev_asr_resolution_error_to_server_error)?;
    if dumper.is_enabled() {
        let trace = resolution.trace(crate::revai::RevAsrProjectionRevision::UtrAsrResponseV1);
        dumper
            .dump_rev_evidence(evidence_identity, &trace)
            .map_err(|error| crate::error::ServerError::Persistence(error.to_string()))?;
    }
    let evidence = resolution.into_evidence();
    Ok(crate::revai::rev_evidence_to_utr_asr_response(&evidence))
}

fn rev_utr_evidence_identity(filename: &str, cache_key: &CacheKey) -> String {
    let stem = Path::new(filename)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("unknown");
    let key_prefix = cache_key.as_str().get(..12).unwrap_or(cache_key.as_str());
    format!("{filename}/{stem}-utr-rev-{key_prefix}")
}

/// What a conversion kept, and what it discarded.
///
/// The two discard reasons are different facts about the run: an engine that
/// reports no word timings and an engine whose timings all collapsed both
/// leave UTR with nothing to inject, and a bare token list cannot say which.
pub(crate) struct UtrTokenConversion {
    window: FaWindow,
    response: crate::transcribe::AsrResponse,
    tokens: Vec<crate::chat_ops::fa::utr::AsrTimingToken>,
    rejections: Vec<UtrTokenRejection>,
}

impl UtrTokenConversion {
    fn into_evidence(
        self,
    ) -> (
        Vec<crate::chat_ops::fa::utr::AsrTimingToken>,
        UtrRecognitionWindow,
    ) {
        UtrRecognitionWindow::from_conversion(self)
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        FaWindow,
        crate::transcribe::AsrResponse,
        Vec<crate::chat_ops::fa::utr::AsrTimingToken>,
        Vec<UtrTokenRejection>,
    ) {
        (self.window, self.response, self.tokens, self.rejections)
    }

    fn summary(&self) -> (usize, SpanRejections) {
        let mut untimed = 0;
        let mut rejected = SpanRejections::default();
        for refusal in &self.rejections {
            match refusal.reason {
                UtrTokenRejectionReason::Untimed => untimed += 1,
                UtrTokenRejectionReason::NoExtent => rejected.no_extent += 1,
                UtrTokenRejectionReason::InvertedInterval
                | UtrTokenRejectionReason::InvertedSpan => rejected.inverted += 1,
                UtrTokenRejectionReason::NotFinite
                | UtrTokenRejectionReason::Negative
                | UtrTokenRejectionReason::OutOfRange
                | UtrTokenRejectionReason::OffsetOverflow => rejected.inadmissible += 1,
                UtrTokenRejectionReason::OutsideWindow { overshoot_ms } => {
                    rejected.outside_window += 1;
                    rejected.worst_overshoot = rejected
                        .worst_overshoot
                        .max(crate::chat_ops::fa::coordinates::Ms(overshoot_ms));
                }
            }
        }
        (untimed, rejected)
    }

    /// Log at most one line describing what was discarded.
    fn warn_if_lossy(&self, context: &UtrPassContext<'_>) {
        // Nothing discarded is the common case and deserves no log line.
        // Written as a total over the three counts rather than a three-way
        // pattern: the question is "was anything lost", and summing says that
        // once instead of enumerating the ways it could be false.
        let (dropped_untimed, rejected) = self.summary();
        let discarded = dropped_untimed + rejected.total();
        if discarded == 0 {
            return;
        }
        // Field names match the FA path's line, so "how many out-of-window
        // rejections did this file have" is one query rather than two.
        tracing::warn!(
            filename = context.filename,
            engine = context.engine.as_wire_name(),
            total = self.tokens.len() + discarded,
            dropped_untimed,
            no_extent = rejected.no_extent,
            inverted = rejected.inverted,
            outside_window = rejected.outside_window,
            inadmissible = rejected.inadmissible,
            worst_overshoot_ms = rejected.worst_overshoot.0,
            "ASR tokens discarded before UTR"
        );
    }
}

/// Convert an ASR response over one extracted audio segment into the timing
/// tokens the UTR injector consumes, moving them from SEGMENT coordinates into
/// FILE coordinates.
///
/// # Why this takes a window rather than an offset
///
/// It took a bare `offset_ms: u64` until 2026-08-15, added it to the engine's
/// reported times, and validated only that the result's end exceeded its start.
/// That is a relation among the engine's own numbers and says nothing about the
/// audio: an engine that loses alignment on a short segment reports positions
/// past the end of the slice it was handed, and the offset then places them
/// past the end of the whole recording. Six of 226 screened sessions carried
/// timings up to 28.2 seconds beyond their own media because of it, and nothing
/// in the pipeline was in a position to notice, because the segment's length
/// was never passed in.
///
/// Taking the window makes the containment question askable, and
/// [`FaWindow::to_file`] makes it unskippable: it is the only route from
/// [`WindowMs`] to a file-coordinate instant.
///
/// Zero-duration tokens are still dropped. Whisper's DTW timestamp extraction
/// works at 20ms resolution and can assign `start == end` to very short words
/// (single-frame backchannels like "mhm", "yeah"). Such tokens carry no useful
/// interval information and, if allowed through, cause UTR to create `•T_T•`
/// utterance bullets that the FA postprocess then perpetuates indefinitely
/// (see OCSC bug, 2026-04-08).
fn asr_response_to_utr_tokens(
    asr_response: crate::transcribe::AsrResponse,
    window: FaWindow,
    engine: &EngineId,
) -> UtrTokenConversion {
    let mut conversion = UtrTokenConversion {
        tokens: Vec::with_capacity(asr_response.tokens.len()),
        rejections: Vec::new(),
        window,
        response: asr_response,
    };
    for (index, token) in conversion.response.tokens.iter().enumerate() {
        let (Some(start_s), Some(end_s)) = (token.start_s, token.end_s) else {
            // The engine reported no span for this token. Distinct from a span
            // it reported as empty: that is a measurement, this is a silence.
            conversion.rejections.push(UtrTokenRejection {
                index,
                reason: UtrTokenRejectionReason::Untimed,
            });
            continue;
        };
        // The pair is admitted once as an interval: rounded to the nearest
        // millisecond, ordered, and within the admissible range (a bound
        // past it, as an epoch timestamp is, used to saturate into a number).
        let interval = match AdmittedInterval::admit_positions(start_s, end_s) {
            Ok(interval) => interval,
            Err(refusal) => {
                let reason = match refusal {
                    batchalign_types::interval::IntervalRefusal::Inverted { .. } => {
                        UtrTokenRejectionReason::InvertedInterval
                    }
                    batchalign_types::interval::IntervalRefusal::NotFinite { .. } => {
                        UtrTokenRejectionReason::NotFinite
                    }
                    batchalign_types::interval::IntervalRefusal::Negative { .. } => {
                        UtrTokenRejectionReason::Negative
                    }
                    batchalign_types::interval::IntervalRefusal::OutOfRange { .. } => {
                        UtrTokenRejectionReason::OutOfRange
                    }
                    batchalign_types::interval::IntervalRefusal::OffsetOverflow { .. } => {
                        UtrTokenRejectionReason::OffsetOverflow
                    }
                };
                conversion
                    .rejections
                    .push(UtrTokenRejection { index, reason });
                continue;
            }
        };
        // Milliseconds relative to the extracted segment. Naming the space is
        // the point: these cannot be written to a transcript without
        // conversion, because a transcript's timings are not measured from a
        // segment.
        let reported_start = WindowMs::reported(interval.start_millis());
        let reported_end = WindowMs::reported(interval.end_millis());

        match (
            window.to_file(reported_start, engine),
            window.to_file(reported_end, engine),
        ) {
            // Built through `WordSpan` rather than by comparing the two ends
            // here. A bare `end > start` is a bool over exactly the relation
            // `SpanFault` splits into `NoExtent` and `Inverted`, so testing it
            // by hand both re-implements the constructor and throws away which
            // of the two happened, leaving this path's counters incomparable
            // with the FA path's for the same engine failure.
            (Ok(start), Ok(end)) => match WordSpan::measured(start, end) {
                Ok(span) => conversion
                    .tokens
                    .push(crate::chat_ops::fa::utr::AsrTimingToken {
                        text: token.text.clone(),
                        start_ms: span.start().at().get(),
                        end_ms: span.end().at().get(),
                    }),
                Err(fault) => {
                    let reason = match fault {
                        crate::chat_ops::fa::timing::SpanFault::NoExtent { .. } => {
                            UtrTokenRejectionReason::NoExtent
                        }
                        crate::chat_ops::fa::timing::SpanFault::Inverted { .. } => {
                            UtrTokenRejectionReason::InvertedSpan
                        }
                    };
                    conversion
                        .rejections
                        .push(UtrTokenRejection { index, reason });
                }
            },
            // Either end past the segment condemns the token: half of a span is
            // not a span, and keeping the surviving end would invent the other.
            (Err(fault), _) | (Ok(_), Err(fault)) => {
                let crate::chat_ops::fa::coordinates::OutsideWindow::BeyondWindowEnd {
                    exceeds_by,
                    ..
                } = fault;
                conversion.rejections.push(UtrTokenRejection {
                    index,
                    reason: UtrTokenRejectionReason::OutsideWindow {
                        overshoot_ms: exceeds_by.0,
                    },
                });
            }
        }
    }
    conversion
}

#[cfg(test)]
pub(crate) fn convert_utr_for_test(
    response: crate::transcribe::AsrResponse,
    window: FaWindow,
) -> UtrTokenConversion {
    asr_response_to_utr_tokens(response, window, &EngineId::new("test-asr"))
}

#[cfg(test)]
pub(crate) fn prepare_after_partial_failure_for_test(
    windows: &[FaWindow],
    completed: UtrTokenConversion,
    failed_window: FaWindow,
    full: UtrTokenConversion,
) -> Result<work::PreparedRecognition, crate::error::ServerError> {
    let route = partial::abandon_for_test(windows, completed, failed_window);
    work::PreparedRecognition::full_for_test(
        full,
        &UtrEngine::Whisper,
        &LanguageCode3::eng(),
        route,
    )
}

/// Prepare recovered tokens for the job's language: its script, once over
/// the whole run, and the lexical units both sides are matched in.
///
/// UTR matches ASR text against the transcript's OWN words, so a Cantonese
/// transcript (HK traditional) or a Taiwanese one (Taiwan traditional) and a
/// provider's simplified output would miss each other word for word. This
/// happens here, once, for every engine, through
/// [`PreparedUtrTokens::prepare`](crate::chat_ops::fa::utr::PreparedUtrTokens::prepare);
/// a normalization that changed the character count refuses rather than
/// re-cutting the tokens away from their timings.
fn normalize_utr_tokens(
    tokens: Vec<crate::chat_ops::fa::utr::AsrTimingToken>,
    lang: &LanguageCode3,
) -> Result<crate::chat_ops::fa::utr::PreparedUtrTokens, crate::error::ServerError> {
    crate::chat_ops::fa::utr::PreparedUtrTokens::prepare(tokens, lang).map_err(|refusal| {
        crate::error::ServerError::Validation(format!(
            "UTR could not normalize its tokens: {refusal}"
        ))
    })
}

#[cfg(test)]
mod utr_token_conversion_tests {
    use super::*;
    use crate::api::AudioPositionSeconds;
    use crate::chat_ops::fa::coordinates::Ms;
    use crate::transcribe::{AsrResponse, AsrToken};

    fn response(tokens: Vec<AsrToken>) -> AsrResponse {
        AsrResponse {
            seam_ambiguities: Vec::new(),
            tokens,
            lang: LanguageCode3::eng(),
            model: None,
            source_monologues: None,
        }
    }

    /// Taiwanese provider tokens arrive simplified; UTR compares them with a
    /// traditional-script transcript, so they are converted, each token
    /// keeping its own interval. A language without a normalization is left
    /// exactly as recognized.
    #[test]
    fn utr_tokens_take_the_transcript_script_and_keep_their_timing() {
        let timed = |text: &str, start_ms: u64| crate::chat_ops::fa::utr::AsrTimingToken {
            text: text.to_owned(),
            start_ms,
            end_ms: start_ms + 100,
        };
        let nan = LanguageCode3::try_new("nan").expect("nan is a language code");
        let converted = normalize_utr_tokens(vec![timed("这个", 0), timed("来", 200)], &nan)
            .expect("conversion preserves character counts");
        let summary: Vec<_> = converted
            .tokens()
            .iter()
            .map(|t| (t.text.as_str(), t.start_ms, t.end_ms))
            .collect();
        assert_eq!(summary, [("這個", 0, 100), ("來", 200, 300)]);

        let untouched = normalize_utr_tokens(vec![timed("这个", 0)], &LanguageCode3::eng())
            .expect("no normalization to refuse");
        assert_eq!(untouched.tokens()[0].text, "这个");
    }

    fn token(text: &str, start_s: Option<f64>, end_s: Option<f64>) -> AsrToken {
        AsrToken {
            text: text.to_owned(),
            start_s: start_s.map(|s| AudioPositionSeconds::try_from(s).expect("fixture position")),
            end_s: end_s.map(|s| AudioPositionSeconds::try_from(s).expect("fixture position")),
            speaker: None,
            confidence: None,
        }
    }

    /// The distinction this type exists for.
    ///
    /// Both discards used to produce a shorter vector and nothing else, so an
    /// engine reporting no word timings and an engine whose timings collapsed
    /// were indistinguishable at the call site: each left UTR with nothing to
    /// inject and no way to say which had happened.
    /// A recording long enough that the windows below sit comfortably inside
    /// it, so these tests exercise conversion rather than containment.
    fn wide_window(start_ms: u64, end_ms: u64) -> (Recording, FaWindow) {
        let recording = Recording::of_duration(Ms(600_000)).expect("non-zero");
        let window = FaWindow::within(&recording, FileMs::new(start_ms), FileMs::new(end_ms))
            .expect("window inside the recording");
        (recording, window)
    }

    fn test_engine() -> EngineId {
        EngineId::new("test-asr")
    }

    #[test]
    fn untimed_and_degenerate_discards_are_counted_apart() {
        let (_rec, window) = wide_window(0, 60_000);
        let converted = asr_response_to_utr_tokens(
            response(vec![
                token("kept", Some(1.0), Some(1.5)),
                token("no start", None, Some(2.0)),
                token("no end", Some(2.0), None),
                token("empty span", Some(3.0), Some(3.0)),
                token("inverted", Some(5.0), Some(4.0)),
            ]),
            window,
            &test_engine(),
        );

        assert_eq!(converted.tokens.len(), 1);
        assert_eq!(converted.tokens[0].text, "kept");
        let (untimed, rejected) = converted.summary();
        assert_eq!(untimed, 2);
        // These two were one counter (`dropped_degenerate == 2`) until
        // 2026-08-15, when this path started building spans through
        // `WordSpan::measured` instead of testing `end > start` by hand. The
        // expectation changed because the CONFLATION was the defect: an empty
        // span is a model that found nothing, an inverted one is a model whose
        // two answers contradict each other, and the FA path had always kept
        // them apart. One number for both made the two paths' logs
        // incomparable for the same engine failure.
        assert_eq!(rejected.no_extent, 1);
        assert_eq!(rejected.inverted, 1);
        assert_eq!(rejected.outside_window, 0);
        let (_, evidence) = converted.into_evidence();
        let evidence = serde_json::to_value(evidence).unwrap();
        assert_eq!(evidence["response"]["tokens"][1]["text"], "no start");
        assert_eq!(evidence["response"]["tokens"][3]["text"], "empty span");
        assert_eq!(evidence["rejections"][0]["index"], 1);
        assert_eq!(evidence["rejections"][0]["reason"]["kind"], "untimed");
        assert_eq!(evidence["rejections"][2]["index"], 3);
        assert_eq!(evidence["rejections"][2]["reason"]["kind"], "no_extent");
        assert_eq!(evidence["rejections"][3]["index"], 4);
        assert_eq!(
            evidence["rejections"][3]["reason"]["kind"],
            "inverted_interval"
        );
    }

    /// A token whose ends lie past the admissible range (an epoch timestamp
    /// in seconds) is refused as an interval and counted as inadmissible. Its
    /// ends used to saturate into millisecond counts and be counted as a
    /// report outside the window.
    #[test]
    fn a_token_past_the_admissible_range_is_inadmissible() {
        let (_rec, window) = wide_window(0, 60_000);
        let epoch = 1_700_000_000.0;
        let converted = asr_response_to_utr_tokens(
            response(vec![
                token("kept", Some(1.0), Some(1.5)),
                token("epoch", Some(epoch), Some(epoch + 0.5)),
            ]),
            window,
            &test_engine(),
        );
        assert_eq!(converted.tokens.len(), 1);
        let (_, rejected) = converted.summary();
        assert_eq!(rejected.inadmissible, 1);
        assert_eq!(rejected.outside_window, 0);
    }

    #[test]
    fn a_clean_response_reports_no_losses() {
        let (_rec, window) = wide_window(0, 60_000);
        let converted = asr_response_to_utr_tokens(
            response(vec![
                token("one", Some(0.0), Some(0.5)),
                token("two", Some(0.5), Some(1.0)),
            ]),
            window,
            &test_engine(),
        );
        assert_eq!(converted.tokens.len(), 2);
        let (untimed, rejected) = converted.summary();
        assert_eq!(untimed, 0);
        assert_eq!(rejected.no_extent, 0);
        assert_eq!(rejected.outside_window, 0);
    }

    #[test]
    fn the_offset_applies_to_both_ends() {
        let (_rec, window) = wide_window(10_000, 70_000);
        let converted = asr_response_to_utr_tokens(
            response(vec![token("w", Some(1.0), Some(2.0))]),
            window,
            &test_engine(),
        );
        assert_eq!(converted.tokens[0].start_ms, 11_000);
        assert_eq!(converted.tokens[0].end_ms, 12_000);
    }

    /// The defect this signature change exists to prevent.
    ///
    /// UTR extracts a physical audio slice and asks an engine about it. An
    /// engine that loses alignment on a short slice reports positions past the
    /// end of the audio it was handed; the old signature took a bare offset,
    /// added it, checked only that end exceeded start, and wrote the result
    /// out. That is how six of 226 screened sessions came to carry timings up
    /// to 28.2 seconds past the end of their own media.
    ///
    /// The numbers here are the real ones: a 4.803 second window at the very
    /// end of a 1259.968 second recording, and an engine reporting 33.02
    /// seconds into it.
    #[test]
    fn a_token_past_the_end_of_its_own_segment_is_refused_not_offset() {
        let recording = Recording::of_duration(Ms(1_259_968)).expect("non-zero");
        let window = FaWindow::within(&recording, FileMs::new(1_255_165), FileMs::new(1_259_968))
            .expect("window inside the recording");

        let converted = asr_response_to_utr_tokens(
            response(vec![
                token("real", Some(1.0), Some(2.0)),
                token("hallucinated", Some(33.02), Some(33.5)),
            ]),
            window,
            &test_engine(),
        );

        // The plausible token survives; the impossible one is discarded and
        // COUNTED, rather than being written out as a measurement.
        assert_eq!(converted.tokens.len(), 1);
        assert_eq!(converted.tokens[0].text, "real");
        let (_, rejected) = converted.summary();
        assert_eq!(rejected.outside_window, 1);
        assert_eq!(rejected.worst_overshoot, Ms(28_217));

        // The property that actually matters, stated directly: nothing this
        // conversion emits can name a moment the recording does not contain.
        for kept in &converted.tokens {
            assert!(kept.end_ms <= recording.duration().get());
        }
    }
}

#[cfg(test)]
mod utr_evidence_cache_tests {
    use super::*;
    use crate::cache::UtteranceCache;
    use crate::chat_ops::CacheKey;
    use crate::error::ServerError;
    use crate::revai::{
        AuthorizedRevEvidenceRun, RevAsrEvidenceInference, RevAsrEvidenceRequest,
        RevAsrModelRevision,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    enum FullFallbackCase {
        DirectSuccess,
        DirectUnprobeable,
        PartialSuccess,
        CachedSuccess,
        CacheMiss,
        CacheMissUnwritableEvidence,
    }

    async fn partial_cache_hit_then_extraction_failure(case: FullFallbackCase) {
        use crate::api::AudioPositionSeconds;
        use crate::chat_ops::fa::AudioIdentity;
        use crate::engine_reports::FaCacheNamespace;
        use crate::pipeline::PipelineServices;
        use crate::types::engines::FaEngineName;
        use crate::worker::pool::{GpuModelPlan, PoolConfig, WorkerPool};
        use talkbank_parser::TreeSitterParser;
        use talkbank_parser_tests::test_error::ParseProductTestExt;

        let directory = tempfile::tempdir().unwrap();
        let audio_path = directory.path().join("recording.mp3");
        crate::media::probe::fixtures::write_unpadded_cbr_mp3(&audio_path);
        let duration = crate::runner::util::probe_audio_duration(&audio_path)
            .await
            .expect("fixture has a proven duration");
        std::fs::remove_file(&audio_path).unwrap();
        let mut chat = TreeSitterParser::new()
            .unwrap()
            .parse_chat_file(
                "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tSPK Speaker\n\
                 @ID:\teng|test|SPK|||||Speaker|||\n@Media:\trecording, audio\n\
                 *SPK:\tone . \u{15}0_5000\u{15}\n\
                 *SPK:\thello .\n\
                 *SPK:\ttwo . \u{15}15000_20000\u{15}\n\
                 *SPK:\tworld .\n\
                 *SPK:\tthree . \u{15}40000_45000\u{15}\n@End\n",
            )
            .expect_built();
        let windows =
            crate::chat_ops::fa::find_untimed_windows(&chat, &Recording::of_audio(duration), 500)
                .unwrap();
        assert_eq!(windows.len(), 2);
        let token = |text: &str, start: f64, end: f64| crate::transcribe::AsrToken {
            text: text.into(),
            start_s: Some(AudioPositionSeconds::try_from(start).unwrap()),
            end_s: Some(AudioPositionSeconds::try_from(end).unwrap()),
            speaker: None,
            confidence: None,
        };
        let response = |tokens| crate::transcribe::AsrResponse {
            tokens,
            lang: LanguageCode3::eng(),
            seam_ambiguities: Vec::new(),
            source_monologues: None,
            model: None,
        };
        let partial = response(vec![
            token("hello", 0.1, 0.2),
            crate::transcribe::AsrToken {
                text: "untimed".into(),
                start_s: None,
                end_s: None,
                speaker: None,
                confidence: None,
            },
        ]);
        let full = response(vec![token("hello", 6.0, 6.1), token("world", 25.0, 25.1)]);
        let cache = UtteranceCache::sqlite(Some(directory.path().join("cache")))
            .await
            .unwrap();
        let engine = UtrEngine::Whisper;
        let language = LanguageCode3::eng();
        let identity = AudioIdentity::from_metadata("recording.mp3", 0, 0);
        let UtrAsrCacheEligibility::Pinned(namespace) = test_eligibility(&engine) else {
            panic!("test requires pinned cache")
        };
        let first = windows[0];
        let segment_key = crate::chat_ops::fa::utr_asr_segment_cache_key(
            &identity,
            &engine,
            first.audio_start().get(),
            first.end().get(),
            &language,
        );
        let full_key = crate::chat_ops::fa::utr_asr_cache_key(&identity, &engine, &language);
        cache
            .put(
                segment_key.as_str(),
                tasks::UTR_ASR,
                &namespace,
                &serde_json::to_value(partial).unwrap(),
            )
            .await
            .unwrap();
        if matches!(case, FullFallbackCase::PartialSuccess) {
            let second = windows[1];
            let key = crate::chat_ops::fa::utr_asr_segment_cache_key(
                &identity,
                &engine,
                second.audio_start().get(),
                second.end().get(),
                &language,
            );
            cache
                .put(
                    key.as_str(),
                    tasks::UTR_ASR,
                    &namespace,
                    &serde_json::to_value(response(vec![token("world", 0.3, 0.4)])).unwrap(),
                )
                .await
                .unwrap();
        }
        if matches!(
            case,
            FullFallbackCase::CachedSuccess | FullFallbackCase::DirectSuccess
        ) {
            cache
                .put(
                    full_key.as_str(),
                    tasks::UTR_ASR,
                    &namespace,
                    &serde_json::to_value(full).unwrap(),
                )
                .await
                .unwrap();
        }
        if matches!(case, FullFallbackCase::DirectUnprobeable) {
            cache
                .put(
                    full_key.as_str(),
                    tasks::UTR_ASR,
                    &namespace,
                    &serde_json::json!({"not": "an ASR response"}),
                )
                .await
                .unwrap();
        }
        let pool = WorkerPool::new(PoolConfig::for_tests());
        let fa_namespace = FaCacheNamespace::for_test("test-fa");
        let gpu = GpuModelPlan::alignment(FaEngineName::Wave2Vec, None);
        let debug_path = directory.path().join("debug");
        if matches!(case, FullFallbackCase::CacheMissUnwritableEvidence) {
            std::fs::write(&debug_path, b"not a directory").unwrap();
        }
        let dumper = DebugDumper::new(Some(&debug_path));
        let strategy = ResolvedUtrStrategy::Global;
        let overrides =
            crate::params::CacheOverrides::from_common(&crate::options::CommonOptions {
                require_media_cache: true,
                ..Default::default()
            })
            .unwrap();
        let context = UtrPassContext {
            pass: crate::api::UtrPass::Initial,
            audio_path: &audio_path,
            lang: &language,
            services: crate::fa::FaServices {
                pipeline: PipelineServices::new(&pool, &cache),
                cache_namespace: &fa_namespace,
                gpu: &gpu,
                attention_capture_dir: None,
            },
            audio_identity: &identity,
            cache_policy: crate::params::UtrCachePolicies::from_overrides(&overrides),
            audio_duration: match case {
                FullFallbackCase::DirectUnprobeable => None,
                _ => Some(duration),
            },
            aggregation_target: None,
            filename: "sample.cha",
            engine: &engine,
            strategy: &strategy,
            dumper: &dumper,
        };
        if matches!(case, FullFallbackCase::DirectSuccess) {
            chat = TreeSitterParser::new()
                .unwrap()
                .parse_chat_file(
                    "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tSPK Speaker\n\
                     @ID:\teng|test|SPK|||||Speaker|||\n@Media:\trecording, audio\n\
                     *SPK:\thello .\n*SPK:\tworld .\n@End\n",
                )
                .expect_built();
        }
        let source = crate::chat_ops::fa::SourceBoundUtrInput::for_test(&chat);
        let result = run_utr_pass(&source, context, None).await;
        if matches!(case, FullFallbackCase::DirectUnprobeable) {
            assert!(matches!(
                result,
                Err(ServerError::RecordingDuration(
                    crate::error::RecordingDurationError::Probe(_)
                ))
            ));
            return;
        }
        if matches!(case, FullFallbackCase::CacheMissUnwritableEvidence) {
            let Err(ServerError::UtrEvidenceRetention(failure)) = result else {
                panic!("requested evidence failure must be fatal")
            };
            let text = failure.to_string();
            assert!(text.contains("required UTR ASR evidence is missing"));
            assert!(text.contains("could not retain failed UTR evidence"));
            return;
        }
        if matches!(case, FullFallbackCase::CacheMiss) {
            let Err(ServerError::Persistence(cause)) = result else {
                panic!("expected original cache-only failure")
            };
            assert!(cause.contains("required UTR ASR evidence is missing"));
            let artifact: serde_json::Value = serde_json::from_slice(
                &std::fs::read(debug_path.join("sample_utr_initial_failed_evidence.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(artifact["failure_category"], "system");
            assert!(artifact["failure"].as_str().unwrap().contains(&cause));
            let partial = &artifact["abandoned_partial"];
            assert_eq!(partial["reason"]["kind"], "segment_extraction");
            let evidence = &partial["completed_windows"][0]["evidence"];
            assert_eq!(evidence["response"]["tokens"][1]["text"], "untimed");
            assert_eq!(evidence["rejections"][0]["index"], 1);
            assert_eq!(evidence["rejections"][0]["reason"]["kind"], "untimed");
            return;
        }
        let UtrPassOutcome::Recovered(completed) = result.unwrap() else {
            panic!("expected full recovery")
        };
        let (observation, _, inputs) = completed.into_parts();
        let evidence = serde_json::to_value(inputs).unwrap();
        if matches!(case, FullFallbackCase::DirectSuccess) {
            assert_eq!(evidence["route"]["kind"], "full");
            assert_eq!(evidence["first"]["start_ms"], 0);
            assert_eq!(
                evidence["first"]["end_ms"],
                Recording::of_audio(duration).duration().get()
            );
            assert_eq!(evidence["first"]["response"]["tokens"][0]["text"], "hello");
            assert_eq!(evidence["first"]["response"]["tokens"][1]["text"], "world");
            assert_eq!(observation.tokens().tokens()[0].start_ms, 6000);
            return;
        }
        if matches!(case, FullFallbackCase::PartialSuccess) {
            assert_eq!(evidence["route"]["kind"], "partial");
            assert_eq!(
                evidence["first"]["start_ms"],
                windows[0].audio_start().get()
            );
            assert_eq!(evidence["first"]["response"]["tokens"][0]["text"], "hello");
            assert_eq!(
                evidence["remaining"][0]["start_ms"],
                windows[1].audio_start().get()
            );
            assert_eq!(
                evidence["remaining"][0]["response"]["tokens"][0]["text"],
                "world"
            );
            assert_eq!(
                observation.tokens().tokens()[1].start_ms,
                windows[1].audio_start().get() + 300
            );
            return;
        }
        let route = &evidence["route"];
        assert_eq!(route["kind"], "full_after_partial");
        assert_eq!(route["abandoned"]["reason"]["kind"], "segment_extraction");
        assert_eq!(
            route["abandoned"]["reason"]["window"]["start_ms"],
            windows[1].audio_start().get()
        );
        assert_eq!(
            route["abandoned"]["completed_windows"][0]["accepted_tokens"][0]["text"],
            "hello"
        );
        assert_eq!(
            route["abandoned"]["planned_windows"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(observation.tokens().tokens()[0].start_ms, 6000);
    }

    #[tokio::test]
    async fn full_cached_request_owns_its_response_and_recording_window() {
        partial_cache_hit_then_extraction_failure(FullFallbackCase::DirectSuccess).await;
    }

    #[tokio::test]
    async fn full_request_requires_recording_before_reading_cache() {
        partial_cache_hit_then_extraction_failure(FullFallbackCase::DirectUnprobeable).await;
    }

    #[tokio::test]
    async fn partial_cached_windows_keep_each_response_with_its_request() {
        partial_cache_hit_then_extraction_failure(FullFallbackCase::PartialSuccess).await;
    }

    #[tokio::test]
    async fn partial_cache_hit_then_extraction_failure_retains_the_attempt() {
        partial_cache_hit_then_extraction_failure(FullFallbackCase::CachedSuccess).await;
    }

    #[tokio::test]
    async fn failed_full_fallback_retains_partial_evidence_or_reports_write_failure() {
        partial_cache_hit_then_extraction_failure(FullFallbackCase::CacheMiss).await;
        partial_cache_hit_then_extraction_failure(FullFallbackCase::CacheMissUnwritableEvidence)
            .await;
    }

    struct CountingRevService {
        calls: AtomicUsize,
    }

    struct UnavailableRevService;

    #[async_trait::async_trait]
    impl RevAsrEvidenceInference for UnavailableRevService {
        async fn infer(
            &self,
            _: AuthorizedRevEvidenceRun,
        ) -> Result<crate::revai::RevAsrInferenceOutcome, ServerError> {
            Err(ServerError::Worker(
                crate::worker::error::WorkerError::WorkerResponse(
                    "controlled provider unavailable".into(),
                ),
            ))
        }
    }

    /// Real evidence resolution and the admitted-document recovery transition,
    /// with a controlled failing provider rather than a paid network call.
    #[tokio::test]
    async fn unavailable_provider_retains_admitted_document_and_records_the_cause() {
        let temp = tempfile::tempdir().unwrap();
        let audio = temp.path().join("audio.wav");
        crate::media::probe::fixtures::write_pcm_wav(&audio, 0);
        let cache = UtteranceCache::sqlite(Some(temp.path().join("cache")))
            .await
            .unwrap();
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &crate::types::revai_language::RevLanguage::admit(
                &crate::api::AsrLanguageRequest::One(LanguageCode3::eng()),
            )
            .unwrap(),
            crate::transcribe::SpeakerCount::Known(std::num::NonZeroU8::MIN),
            &RevAsrModelRevision::current(),
        )
        .await
        .unwrap();
        let mut model = crate::fa::read_fa_source(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n@Media:\taudio, audio\n*PAR:\thello . \u{15}100_400\u{15}\n*PAR:\tgoodbye .\n@End\n"
        ).unwrap();
        let active = model.active_mut().unwrap();
        let before = batchalign_transform::serialize::to_chat_string(active.document());
        let dumper = DebugDumper::new(None);
        let recovered = active
            .recover_timing(crate::api::UtrPass::Initial, async |_| {
                resolve_rev_utr_asr_response(
                    &request,
                    &cache,
                    CachePolicy::UseCache,
                    &UnavailableRevService,
                    &dumper,
                    "sample.cha",
                )
                .await
                .map(|_| panic!("the controlled provider cannot return timing"))
            })
            .await;
        assert_eq!(
            batchalign_transform::serialize::to_chat_string(active.document()),
            before
        );
        let mut recovery =
            crate::provenance::UtrRecovery::from_initial(&UtrEngine::RevAi, recovered).unwrap();
        assert!(recovery.fallback().is_some());
        let report = recovery.contribution();
        assert!(
            report
                .completed()
                .map(crate::provenance::UtrProducer::engine)
                .is_none()
        );
        assert!(matches!(
            &report.shortfalls()[..],
            [crate::api::OutputShortfallRecord::UtrUnavailable {
                pass: crate::api::UtrPass::Initial,
                category: crate::scheduling::FailureCategory::ProviderTransient,
                ..
            }]
        ));
    }

    #[async_trait::async_trait]
    impl RevAsrEvidenceInference for CountingRevService {
        async fn infer(
            &self,
            _run: AuthorizedRevEvidenceRun,
        ) -> Result<crate::revai::RevAsrInferenceOutcome, ServerError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(crate::revai::RevAsrInferenceOutcome::Fetched(
                FetchedRevAsrEvidence {
                    transcript_evidence:
                        crate::revai::RevTranscriptEvidence::from_legacy_transcript(
                            serde_json::from_str(
                                r#"{
                        "monologues": [{
                            "speaker": 0,
                            "elements": [{
                                "type": "text",
                                "value": "hello",
                                "ts": 0.1,
                                "end_ts": 0.5,
                                "confidence": 0.9
                            }]
                        }]
                    }"#,
                            )
                            .expect("valid Rev transcript"),
                        ),
                    resolved_language: crate::api::TranscriptLanguage::One(LanguageCode3::eng()),
                },
            ))
        }
    }

    #[tokio::test]
    async fn rev_utr_replays_raw_evidence_without_a_second_paid_call() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio = tempdir.path().join("audio.wav");
        crate::media::probe::fixtures::write_pcm_wav(&audio, 0);
        let cache_dir = tempdir.path().join("cache");
        let cache = UtteranceCache::sqlite(Some(cache_dir.clone()))
            .await
            .expect("cache");
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &crate::types::revai_language::RevLanguage::admit(
                &crate::api::AsrLanguageRequest::One(LanguageCode3::eng()),
            )
            .expect("Rev.AI recognizes English"),
            crate::transcribe::SpeakerCount::Known(std::num::NonZeroU8::MIN),
            &RevAsrModelRevision::current(),
        )
        .await
        .expect("request");
        let service = CountingRevService {
            calls: AtomicUsize::new(0),
        };
        let debug_dir = tempdir.path().join("debug");
        let dumper = DebugDumper::new(Some(&debug_dir));
        let evidence_identity = rev_utr_evidence_identity("sample.cha", request.cache_key());

        let cold = resolve_rev_utr_asr_response(
            &request,
            &cache,
            CachePolicy::UseCache,
            &service,
            &dumper,
            &evidence_identity,
        )
        .await
        .expect("cold resolution");
        drop(cache);

        let reopened = UtteranceCache::sqlite(Some(cache_dir))
            .await
            .expect("reopen");
        let warm = resolve_rev_utr_asr_response(
            &request,
            &reopened,
            CachePolicy::UseCache,
            &service,
            &dumper,
            &evidence_identity,
        )
        .await
        .expect("warm resolution");

        assert_eq!(
            serde_json::to_value(cold).expect("cold JSON"),
            serde_json::to_value(warm).expect("warm JSON")
        );
        assert_eq!(service.calls.load(Ordering::SeqCst), 1);
        let trace_paths: Vec<_> = std::fs::read_dir(&debug_dir)
            .expect("read debug dir")
            .map(|entry| entry.expect("debug entry").path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with("_rev_evidence.json"))
            })
            .collect();
        assert_eq!(trace_paths.len(), 1);
        let trace: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&trace_paths[0]).expect("read trace"))
                .expect("parse trace");
        assert_eq!(trace["cache_outcome"], "replayed");
        assert_eq!(
            trace["projection_revision"],
            "rev-transcript-to-utr-asr-response-v1"
        );
    }

    #[tokio::test]
    async fn task_scoped_required_asr_cold_miss_never_calls_provider() {
        let root = tempfile::tempdir().unwrap();
        let audio = root.path().join("audio.wav");
        crate::media::probe::fixtures::write_pcm_wav(&audio, 0);
        let request = RevAsrEvidenceRequest::from_audio(
            &audio,
            &crate::types::revai_language::RevLanguage::admit(
                &crate::api::AsrLanguageRequest::One(LanguageCode3::eng()),
            )
            .unwrap(),
            crate::transcribe::SpeakerCount::Known(std::num::NonZeroU8::MIN),
            &RevAsrModelRevision::current(),
        )
        .await
        .unwrap();
        let overrides =
            crate::params::CacheOverrides::from_common(&crate::options::CommonOptions {
                require_media_cache_tasks: vec!["utr_asr".into()],
                override_media_cache_tasks: vec!["forced_alignment".into()],
                ..Default::default()
            })
            .unwrap();
        let policies = crate::params::UtrCachePolicies::from_overrides(&overrides);
        let cache = UtteranceCache::sqlite(Some(root.path().join("cache")))
            .await
            .unwrap();
        let service = CountingRevService {
            calls: AtomicUsize::new(0),
        };
        let dumper = DebugDumper::new(None);
        let result = resolve_rev_utr_asr_response(
            &request,
            &cache,
            policies.rev_evidence(),
            &service,
            &dumper,
            "synthetic",
        )
        .await;
        assert!(result.is_err());
        assert_eq!(service.calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            overrides.policy_for(crate::chat_ops::CacheTaskName::ForcedAlignment),
            CachePolicy::SkipCache
        );
    }

    /// The cache eligibility one engine resolves for a test, which must be the
    /// pinned arm: every UTR engine's composition is fully pinned in this
    /// build, and a test that silently ran against `Floating` would exercise
    /// the no-cache path while claiming to test the cache.
    fn test_eligibility(engine: &UtrEngine) -> UtrAsrCacheEligibility {
        let lang = crate::api::LanguageCode3::eng();
        let models = crate::model_manifest::utr_pinned_models(
            engine,
            &lang,
            &crate::model_manifest::TencentEngineSetting::LARGE_MODEL_2,
        )
        .expect("a UTR engine resolves a pinned composition");
        let eligibility = UtrAsrCacheNamespace::for_pinned_plan(
            engine,
            &models,
            &crate::transcribe::SingleAsrLanguage::One(lang.clone()),
        )
        .expect("the engine can decode English");
        assert!(
            matches!(eligibility, UtrAsrCacheEligibility::Pinned(_)),
            "every UTR engine composition is fully pinned in this build"
        );
        eligibility
    }

    #[tokio::test]
    async fn corrupt_derived_utr_cache_fails_closed() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let key = CacheKey::from_content("corrupt UTR cache test");
        let eligibility = test_eligibility(&UtrEngine::Whisper);
        // The seed write takes the namespace itself; the lookup under test
        // takes the eligibility. Binding the pinned arm here makes the two
        // halves provably the same namespace, which is what puts the corrupt
        // row where the read below will actually find it.
        let UtrAsrCacheEligibility::Pinned(namespace) = &eligibility else {
            panic!("every UTR engine composition is fully pinned in this build")
        };
        cache
            .put(
                key.as_str(),
                tasks::UTR_ASR,
                namespace,
                &serde_json::json!({"not": "an AsrResponse"}),
            )
            .await
            .expect("seed corrupt cache value");

        let error = lookup_utr_asr_cache(&cache, &key, &eligibility, CachePolicy::UseCache)
            .await
            .expect_err("corrupt cache must not become an inference-authorizing miss");
        assert!(matches!(error, ServerError::Persistence(_)));
    }

    /// Two jobs asking for the same recording's UTR ASR at the same moment must
    /// share ONE inference (two jobs over the same recordings each logged "UTR ASR
    /// cache miss, running inference" within a millisecond of each other, and
    /// both ran the same long Whisper pass per file).
    ///
    /// The first lookup is the in-flight leader: it holds a miss. A second
    /// identical lookup must PARK behind it rather than authorize a second
    /// inference, and must then replay what the leader stored. The wait is
    /// observed through the lease registry, never a clock: before the fix the
    /// second lookup finishes on its own with a miss, which ends the loop.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_identical_utr_lookups_share_one_inference() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let cache = std::sync::Arc::new(
            UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
                .await
                .expect("cache"),
        );
        let key = CacheKey::from_content("single-flight UTR ASR lookup");
        let eligibility = std::sync::Arc::new(test_eligibility(&UtrEngine::Whisper));

        let leader = lookup_utr_asr_cache(&cache, &key, &eligibility, CachePolicy::UseCache)
            .await
            .expect("cold lookup");
        let UtrAsrCacheLookup::Miss(leader) = leader else {
            panic!("an empty cache must authorize the first inference")
        };

        let follower = {
            let (cache, key, eligibility) = (cache.clone(), key.clone(), eligibility.clone());
            tokio::spawn(async move {
                lookup_utr_asr_cache(&cache, &key, &eligibility, CachePolicy::UseCache).await
            })
        };
        // Either the follower parks behind the leader's lease (two contenders),
        // or it finishes by itself, which is the defect.
        while !follower.is_finished() && crate::cache::InferenceLease::contenders_for_test(&key) < 2
        {
            tokio::task::yield_now().await;
        }
        assert!(
            !follower.is_finished(),
            "a second identical lookup completed while the first inference was \
             still in flight: {:?}; both would run the same inference",
            follower.await.expect("follower task")
        );

        let response = crate::transcribe::AsrResponse {
            seam_ambiguities: Vec::new(),
            tokens: Vec::new(),
            lang: LanguageCode3::eng(),
            source_monologues: None,
            model: None,
        };
        let stored = leader
            .run(async { Ok(response.clone()) })
            .await
            .expect("leader inference")
            .commit(&cache, &key, "leader.cha")
            .await;
        assert_eq!(
            serde_json::to_value(&stored).expect("stored JSON"),
            serde_json::to_value(&response).expect("response JSON")
        );

        let replay = follower
            .await
            .expect("follower task")
            .expect("follower lookup");
        let UtrAsrCacheLookup::Hit(replayed) = replay else {
            panic!("the follower must replay the leader's result, not infer again")
        };
        assert_eq!(
            serde_json::to_value(*replayed).expect("replayed JSON"),
            serde_json::to_value(&response).expect("response JSON")
        );
    }

    #[tokio::test]
    async fn required_utr_derived_cache_miss_can_reach_raw_evidence_resolution() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("cache");
        let key = CacheKey::from_content("required UTR cache test");
        let eligibility = test_eligibility(&UtrEngine::RevAi);

        let lookup = lookup_utr_asr_cache(&cache, &key, &eligibility, CachePolicy::RequireCache)
            .await
            .expect("a derived miss must remain distinguishable from a raw evidence miss");

        assert!(matches!(lookup, UtrAsrCacheLookup::Miss(_)));
    }
}
