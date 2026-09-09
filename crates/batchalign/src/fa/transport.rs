//! Transport adapter for forced-alignment worker inference.
//!
//! The FA pipeline delegates worker interaction through this module so the
//! orchestration code can ask for "timings for these miss groups" without
//! depending on the concrete worker-protocol V2 request-building details.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::api::{DurationMs, WorkerLanguage};
use crate::chat_ops::fa::coordinates::FaWindow;
use crate::chat_ops::fa::origin::EngineId;
use crate::chat_ops::fa::{FaGroup, FaInferItem, WordGapHealing, WordTiming};
use crate::error::{MissingForcedAlignmentEvidence, MissingRequiredEvidence, ServerError};
use crate::params::CachePolicy;
use crate::pipeline::PipelineServices;
use crate::types::engines::SelectableEngine;
use crate::types::traces::FaFallbackEventTrace;
use crate::worker::artifacts_v2::PreparedArtifactRuntimeV2;
use crate::worker::fa_result_v2::parse_forced_alignment_result_v2;
use crate::worker::request_builder_v2::{
    ForcedAlignmentBuildInputV2, ForcedAlignmentRequestBuildErrorV2, PreparedFaRequestIdsV2,
    build_forced_alignment_request_v2,
};
use tracing::warn;

use super::raw_evidence::{
    ExpectedFaWords, FaEvidenceRoute, FaRawEvidence, ReplayableFaRawEvidence,
};

static NEXT_FA_REQUEST_NAMESPACE: AtomicU64 = AtomicU64::new(1);

/// Unchecked parallel inputs for one FA worker batch.
///
/// This state can be assembled by orchestration but cannot be dispatched.
/// [`UncheckedFaWorkerBatch::admit`] is the only constructor for the worker's
/// cardinality-checked [`FaWorkerBatch`] state.
pub(crate) struct UncheckedFaWorkerBatch<'a> {
    /// Precomputed cleaned word texts keyed by group index.
    pub word_texts: &'a [Vec<String>],
    /// FA groups for the current file.
    pub groups: &'a [FaGroup],
    /// Semantic cache identities corresponding one-to-one with `groups`.
    pub cache_keys: &'a [crate::chat_ops::CacheKey],
    /// Proof that the cache policy permits inference for these miss groups.
    pub authorization: FaInferenceAuthorization<'a>,
    /// Source audio path for the current file.
    pub audio_path: &'a Path,
    /// Worker-runtime language hint for FA model bootstrap.
    pub worker_lang: WorkerLanguage,
    /// FA backend selected by the Rust control plane.
    pub engine: crate::types::engines::FaEngineName,
    /// Gap-healing policy for every group in this batch.
    pub gap_healing: WordGapHealing,
}

/// FA worker batch whose parallel group facts have one proven cardinality.
#[derive(Debug)]
pub(crate) struct FaWorkerBatch<'a> {
    word_texts: &'a [Vec<String>],
    groups: &'a [FaGroup],
    cache_keys: &'a [crate::chat_ops::CacheKey],
    authorization: FaInferenceAuthorization<'a>,
    audio_path: &'a Path,
    worker_lang: WorkerLanguage,
    engine: crate::types::engines::FaEngineName,
    gap_healing: WordGapHealing,
}

impl<'a> UncheckedFaWorkerBatch<'a> {
    /// Prove every parallel group input and miss index before dispatch.
    pub(crate) fn admit(self) -> Result<FaWorkerBatch<'a>, ServerError> {
        if self.word_texts.len() != self.groups.len() || self.cache_keys.len() != self.groups.len()
        {
            return Err(ServerError::Validation(format!(
                "FA worker batch cardinality drift: groups={}, word_texts={}, cache_keys={}",
                self.groups.len(),
                self.word_texts.len(),
                self.cache_keys.len()
            )));
        }
        if let Some(group_index) = self
            .authorization
            .miss_indices
            .iter()
            .copied()
            .find(|group_index| *group_index >= self.groups.len())
        {
            return Err(ServerError::Validation(format!(
                "FA worker batch miss index {group_index} exceeds {} groups",
                self.groups.len()
            )));
        }
        Ok(FaWorkerBatch {
            word_texts: self.word_texts,
            groups: self.groups,
            cache_keys: self.cache_keys,
            authorization: self.authorization,
            audio_path: self.audio_path,
            worker_lang: self.worker_lang,
            engine: self.engine,
            gap_healing: self.gap_healing,
        })
    }
}

/// The only value that permits FA worker inference for cache misses.
///
/// Its field is private and [`plan_fa_inference`] is its only constructor, so
/// a required-cache miss has no route to [`FaWorkerBatch`].
#[derive(Debug)]
pub(crate) struct FaInferenceAuthorization<'a> {
    miss_indices: &'a [usize],
}

/// Whether one cache partition needs and permits worker inference.
#[derive(Debug)]
pub(crate) enum FaInferencePlan<'a> {
    /// Every group was satisfied by reusable or cached evidence.
    NothingToInfer,
    /// Cache misses exist and policy permits worker inference.
    Authorized(FaInferenceAuthorization<'a>),
}

/// Convert cache misses into a worker-inference capability, or refuse when
/// the job required complete reusable evidence.
pub(crate) fn plan_fa_inference(
    policy: CachePolicy,
    miss_indices: &[usize],
) -> Result<FaInferencePlan<'_>, ServerError> {
    let Some((&first_miss, remaining_misses)) = miss_indices.split_first() else {
        return Ok(FaInferencePlan::NothingToInfer);
    };
    match policy {
        CachePolicy::RequireCache => {
            return Err(ServerError::RequiredEvidenceUnavailable(
                MissingRequiredEvidence::ForcedAlignment(MissingForcedAlignmentEvidence::new(
                    first_miss,
                    remaining_misses,
                )),
            ));
        }
        CachePolicy::UseCache | CachePolicy::SkipCache => {}
    }
    Ok(FaInferencePlan::Authorized(FaInferenceAuthorization {
        miss_indices,
    }))
}

/// Result of attempting worker inference for one FA group.
///
/// Successful model evidence and an intentional unaligned fallback are
/// distinct states. Only the former can cross the raw-evidence cache boundary.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum FaWorkerGroupResult {
    /// A successful worker response admitted against the request and parsed.
    Evidence(Box<FaWorkerEvidenceResult>),
    /// A group-local failure deliberately represented as unaligned words.
    Unaligned(FaWorkerUnalignedResult),
}

/// Timing projection and provenance issued together by the worker outcome.
/// Callers cannot construct a projection with a fabricated evidence source.
pub(crate) struct FaWorkerProjection {
    /// Original group index in the file.
    pub group_index: usize,
    /// One optional timing per requested word.
    pub timings: Vec<Option<WordTiming>>,
    /// Replayable direct evidence, when admitted by the worker boundary.
    pub raw_evidence: Option<ReplayableFaRawEvidence>,
    /// Engine fallback associated with this evidence, if any.
    pub fallback_event: Option<FaFallbackEventTrace>,
    source: crate::types::traces::FaEvidenceSourceTrace,
}

impl FaWorkerProjection {
    /// Provenance of this projection, including the absence of worker evidence.
    pub fn source(&self) -> crate::types::traces::FaEvidenceSourceTrace {
        self.source
    }
}

impl FaWorkerGroupResult {
    /// Preserve the outcome distinction when materializing unaligned words.
    pub fn into_projection(self) -> FaWorkerProjection {
        use crate::types::traces::FaEvidenceSourceTrace;
        match self {
            Self::Evidence(evidence) => FaWorkerProjection {
                group_index: evidence.group_index,
                timings: evidence.timings,
                raw_evidence: evidence.raw_evidence,
                fallback_event: evidence.fallback_event,
                source: FaEvidenceSourceTrace::Inference,
            },
            Self::Unaligned(unaligned) => FaWorkerProjection {
                group_index: unaligned.group_index,
                timings: vec![None; unaligned.word_count],
                raw_evidence: None,
                fallback_event: None,
                source: FaEvidenceSourceTrace::Unaligned,
            },
        }
    }
}

/// Successful FA evidence paired inseparably with its parsed timing projection.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FaWorkerEvidenceResult {
    /// Original group index inside the current file.
    pub group_index: usize,
    /// Parsed timings in the established Rust FA timing domain.
    pub timings: Vec<Option<WordTiming>>,
    /// Immutable worker response admitted against engine and word cardinality.
    pub raw_evidence: Option<ReplayableFaRawEvidence>,
    /// Fallback event metadata when this group had to retry with another engine.
    pub fallback_event: Option<FaFallbackEventTrace>,
}

/// A group-local failure that is safe to retain as explicitly unaligned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FaWorkerUnalignedResult {
    /// Original group index inside the current file.
    pub group_index: usize,
    /// Exact number of unaligned words to materialize.
    pub word_count: usize,
}

/// Narrow transport adapter for FA worker inference.
#[derive(Clone, Copy)]
pub(crate) enum FaWorkerTransport<'a> {
    /// Live typed worker-protocol V2 transport using prepared artifacts.
    V2 {
        /// Shared pipeline services with worker-pool access.
        services: PipelineServices<'a>,
    },
}

impl<'a> FaWorkerTransport<'a> {
    /// Return the production FA worker transport.
    pub(crate) fn production(services: PipelineServices<'a>) -> Self {
        Self::V2 { services }
    }

    /// Infer timings for the requested FA miss groups.
    pub(crate) async fn infer_groups(
        self,
        batch: FaWorkerBatch<'_>,
    ) -> Result<Vec<FaWorkerGroupResult>, ServerError> {
        match self {
            Self::V2 { services } => infer_groups_v2(services, batch).await,
        }
    }
}

/// Dispatch staged worker-protocol V2 requests for each FA miss group and
/// parse successful results back into the established Rust timing domain.
async fn infer_groups_v2(
    services: PipelineServices<'_>,
    batch: FaWorkerBatch<'_>,
) -> Result<Vec<FaWorkerGroupResult>, ServerError> {
    let request_namespace = NEXT_FA_REQUEST_NAMESPACE.fetch_add(1, Ordering::Relaxed);
    let artifacts = PreparedArtifactRuntimeV2::new("fa_v2").map_err(|error| {
        ServerError::Validation(format!("failed to create FA V2 artifact runtime: {error}"))
    })?;

    let mut parsed_results = Vec::with_capacity(batch.authorization.miss_indices.len());
    for group_index in batch.authorization.miss_indices.iter().copied() {
        let group = &batch.groups[group_index];

        // Grouping carries the admitted recording window through dispatch.
        let window = group.window();

        let response = match dispatch_group_request(
            services,
            &artifacts,
            &batch,
            request_namespace,
            group_index,
            batch.engine,
        )
        .await
        {
            Ok(response) => response,
            Err(ServerError::EmptyFaAudioSegment(ref segment)) => {
                // Extraction produced no frames. That does not prove the
                // window is past EOF; very short in-range windows can do this.
                // Leave this group's words unaligned rather than failing the
                // whole file: the transcript is still useful without timing.
                warn!(
                    group = group_index,
                    start_ms = segment.window.start().get(),
                    end_ms = segment.window.end().get(),
                    path = %segment.path,
                    "FA group decoded no audio samples; leaving words unaligned"
                );
                parsed_results.push(unaligned_group_result(group_index, group));
                continue;
            }
            // Process exit proves that this request lost its worker, not why
            // the process exited or whether this input caused it. Preserve the
            // group as unaligned; the pool replaces a dead shared generation
            // before later requests. Do not label an unknown exit as OOM.
            Err(ref e) if is_worker_process_crash(e) => {
                warn!(
                    group = group_index,
                    start_ms = group.audio_start_ms(),
                    end_ms = group.audio_end_ms(),
                    error = %e,
                    "Worker process exited during FA group; leaving words unaligned"
                );
                parsed_results.push(unaligned_group_result(group_index, group));
                continue;
            }
            Err(other) => return Err(other),
        };

        // Bind every fact that identifies this group before interpreting any
        // response. Primary and fallback responses must travel through the
        // same capability, so a later branch cannot accidentally pair a
        // response with another group's key, window, or word cardinality.
        let admission = FaGroupEvidenceAdmission {
            requested_engine: batch.engine,
            request_engine_version: services.engine_version,
            cache_key: &batch.cache_keys[group_index],
            group_index,
            group,
            window: &window,
        };

        match admission.admit(&response, FaEvidenceRoute::Direct) {
            Ok(parsed) => parsed_results.push(FaWorkerGroupResult::Evidence(Box::new(parsed))),
            Err(error) => {
                let Some(retry) = fa_group_retry(batch.engine, &error) else {
                    // Before propagating to the file level, check whether this is a
                    // data-driven RuntimeFailure. RuntimeFailure means the model failed
                    // on this group's specific input, other groups are unaffected, so
                    // the correct recovery is a group-level skip, not a file abort.
                    if is_fa_runtime_failure(&error) {
                        warn!(
                            group = group_index,
                            start_ms = group.audio_start_ms(),
                            end_ms = group.audio_end_ms(),
                            error = %error,
                            "FA group failed with model RuntimeFailure (data-driven); \
                             leaving words unaligned"
                        );
                        parsed_results.push(unaligned_group_result(group_index, group));
                        continue;
                    }
                    return Err(error);
                };
                warn!(
                    group = group_index,
                    start_ms = group.audio_start_ms(),
                    end_ms = group.audio_end_ms(),
                    reason = retry.reason,
                    failed_engine = batch.engine.selection_name(),
                    retry_engine = retry.target.selection_name(),
                    "FA engine hit a recoverable target constraint; retrying group on its \
                     fallback engine"
                );
                let fallback_namespace = NEXT_FA_REQUEST_NAMESPACE.fetch_add(1, Ordering::Relaxed);
                let fallback_response = dispatch_group_request(
                    services,
                    &artifacts,
                    &batch,
                    fallback_namespace,
                    group_index,
                    retry.target,
                )
                .await?;
                match admission.admit(
                    &fallback_response,
                    FaEvidenceRoute::Fallback {
                        reason: retry.reason,
                    },
                ) {
                    Ok(parsed) => parsed_results.push(FaWorkerGroupResult::Evidence(Box::new(
                        parsed.with_fallback_event(build_fallback_event(
                            group_index,
                            group,
                            batch.engine,
                            retry.target,
                            retry.reason,
                        )),
                    ))),
                    // The fallback model is not loaded in this worker (capability
                    // gap, not a data error).  Leave the group's words unaligned
                    // rather than aborting the whole file, the surrounding
                    // utterances still have valid timing.
                    Err(ref error) if is_whisper_model_unavailable(error) => {
                        warn!(
                            group = group_index,
                            start_ms = group.audio_start_ms(),
                            end_ms = group.audio_end_ms(),
                            retry_engine = retry.target.selection_name(),
                            "fallback FA engine unavailable (worker has no such model \
                             loaded); leaving group words unaligned"
                        );
                        parsed_results.push(unaligned_group_result(group_index, group));
                    }
                    // The fallback itself hit a data-driven RuntimeFailure
                    // (e.g. the group is still too long for the fallback's own
                    // context after the retry).  Same treatment: leave the group
                    // unaligned rather than aborting the file.
                    Err(ref error) if is_fa_runtime_failure(error) => {
                        warn!(
                            group = group_index,
                            start_ms = group.audio_start_ms(),
                            end_ms = group.audio_end_ms(),
                            error = %error,
                            retry_engine = retry.target.selection_name(),
                            "fallback FA engine also failed with model RuntimeFailure; \
                             leaving group words unaligned"
                        );
                        parsed_results.push(unaligned_group_result(group_index, group));
                    }
                    Err(error) => return Err(error),
                }
            }
        }
    }

    Ok(parsed_results)
}

/// Construct a group result with every word timing left unaligned (`None`).
///
/// All group-level skip paths (empty audio, model unavailability, data-driven
/// RuntimeFailure) return this same shape so the orchestrator can continue to
/// the next group without aborting the file.
fn unaligned_group_result(group_index: usize, group: &FaGroup) -> FaWorkerGroupResult {
    FaWorkerGroupResult::Unaligned(FaWorkerUnalignedResult {
        group_index,
        word_count: group.words.len(),
    })
}

async fn dispatch_group_request(
    services: PipelineServices<'_>,
    artifacts: &PreparedArtifactRuntimeV2,
    batch: &FaWorkerBatch<'_>,
    request_namespace: u64,
    group_index: usize,
    engine: crate::types::engines::FaEngineName,
) -> Result<crate::types::worker_v2::ExecuteResponseV2, ServerError> {
    let infer_item = build_fa_infer_item(batch, group_index);
    let request_ids = build_fa_request_ids(request_namespace, group_index);
    let request = build_forced_alignment_request_v2(
        artifacts.store(),
        ForcedAlignmentBuildInputV2 {
            ids: &request_ids,
            infer_item: &infer_item,
            engine,
        },
    )
    .await
    .map_err(|error| match error {
        // Empty audio is a skip signal, not a fatal failure.  Propagate as a
        // dedicated error so the caller can leave the group unaligned instead
        // of failing the whole file.
        // Whole, again. This arm used to rebuild three fields AND silently
        // change their type from `u64` to `DurationMs` on the way through.
        ForcedAlignmentRequestBuildErrorV2::EmptyAudioSegment(segment) => {
            ServerError::EmptyFaAudioSegment(segment)
        }
        other => ServerError::Validation(format!(
            "failed to build worker protocol V2 FA request for group {group_index}: {other}"
        )),
    })?;

    services
        .pool
        .dispatch_execute_v2(&batch.worker_lang, &request)
        .await
        .map_err(ServerError::Worker)
}

/// Lower one worker response into the FA timing domain.
///
/// Takes the group's window RATHER THAN rebuilding it from the recording. The
/// window is proved once, by the caller, before inference is dispatched; a
/// second construction here would be the same check in a second place, and the
/// failure would arrive after the expensive part had already been paid for.
fn parse_group_response(
    response: &crate::types::worker_v2::ExecuteResponseV2,
    group_index: usize,
    group: &FaGroup,
    window: &FaWindow,
    engine: &EngineId,
) -> Result<Vec<Option<WordTiming>>, ServerError> {
    parse_forced_alignment_result_v2(response, &group.words, window, engine).map_err(|error| {
        ServerError::Validation(format!(
            "failed to parse worker protocol V2 FA response for group {group_index} ({}..{} ms): {error}",
            group.audio_start_ms(),
            group.audio_end_ms(),
        ))
    })
}

impl FaWorkerEvidenceResult {
    fn with_fallback_event(mut self, fallback_event: FaFallbackEventTrace) -> Self {
        self.fallback_event = Some(fallback_event);
        self
    }
}

/// Capability binding every current-request fact needed to admit one group's
/// worker response.
///
/// Keeping these values together prevents the primary and fallback branches
/// from independently reconstructing a six-value relationship by convention.
struct FaGroupEvidenceAdmission<'a> {
    requested_engine: crate::types::engines::FaEngineName,
    request_engine_version: &'a crate::api::EngineVersion,
    cache_key: &'a crate::chat_ops::CacheKey,
    group_index: usize,
    group: &'a FaGroup,
    window: &'a FaWindow,
}

impl FaGroupEvidenceAdmission<'_> {
    fn admit(
        &self,
        response: &crate::types::worker_v2::ExecuteResponseV2,
        route: FaEvidenceRoute<'_>,
    ) -> Result<FaWorkerEvidenceResult, ServerError> {
        let raw_evidence = FaRawEvidence::admit_requested(
            response,
            self.requested_engine,
            self.request_engine_version,
            ExpectedFaWords::new(self.group.words.len()),
            self.cache_key,
            route,
        )
        .map_err(|error| {
            ServerError::Validation(format!(
                "failed to admit worker protocol V2 FA evidence for group {}: {error}",
                self.group_index
            ))
        })?;
        let timings = parse_group_response(
            raw_evidence.response(),
            self.group_index,
            self.group,
            self.window,
            &EngineId::new(raw_evidence.effective_engine().as_wire_name()),
        )?;
        let replayable_raw_evidence = match raw_evidence.into_replayable() {
            Ok(replayable) => Some(replayable),
            Err(super::raw_evidence::FaRawEvidenceError::UnversionedFallbackEvidence) => None,
            Err(error) => {
                return Err(ServerError::Validation(format!(
                    "failed to close FA evidence replay state for group {}: {error}",
                    self.group_index
                )));
            }
        };
        Ok(FaWorkerEvidenceResult {
            group_index: self.group_index,
            timings,
            raw_evidence: replayable_raw_evidence,
            fallback_event: None,
        })
    }
}

/// Reparse one admitted cached response without invoking a model worker.
pub(super) fn replay_group_evidence(
    raw_evidence: ReplayableFaRawEvidence,
    group_index: usize,
    group: &FaGroup,
) -> Result<FaWorkerEvidenceResult, ServerError> {
    let raw_evidence = raw_evidence.into_inner();
    let window = group.window();
    let timings = parse_group_response(
        raw_evidence.response(),
        group_index,
        group,
        &window,
        &EngineId::new(raw_evidence.effective_engine().as_wire_name()),
    )?;
    let fallback_event = raw_evidence.fallback_reason().map(|reason| {
        build_fallback_event(
            group_index,
            group,
            raw_evidence.requested_engine(),
            raw_evidence.effective_engine(),
            reason,
        )
    });
    Ok(FaWorkerEvidenceResult {
        group_index,
        timings,
        raw_evidence: None,
        fallback_event,
    })
}

fn build_fallback_event(
    group_index: usize,
    group: &FaGroup,
    from_engine: crate::types::engines::FaEngineName,
    to_engine: crate::types::engines::FaEngineName,
    reason: &str,
) -> FaFallbackEventTrace {
    FaFallbackEventTrace {
        group_index,
        from_engine: {
            use crate::types::engines::EngineBackend;
            from_engine.wire_name().to_string()
        },
        to_engine: {
            use crate::types::engines::EngineBackend;
            to_engine.wire_name().to_string()
        },
        reason: reason.to_string(),
        audio_start_ms: DurationMs(group.audio_start_ms()),
        audio_end_ms: DurationMs(group.audio_end_ms()),
    }
}

/// Returns `true` when the FA group error is a data-driven `RuntimeFailure`
/// the worker received and understood the request but the model raised a Python
/// exception on this specific input (token overflow, shape mismatch, OOM, etc.).
///
/// # Why this is always group-local
///
/// A `RuntimeFailure` means the worker successfully parsed the request and
/// attempted inference, then the model failed. The failure is caused by the
/// content of *this* group's words and audio. Other groups have different
/// words and different audio; they will not trigger the same failure. The error
/// is therefore inherently group-scoped and should be demoted to a group-level
/// warning (leave words unaligned, continue) rather than propagating as a
/// file-level failure.
///
/// # Contrast with infrastructure failures
///
/// `ProcessExited` (worker crash) and `Protocol` (IPC deserialization failure)
/// are not data-driven: if the worker crashed or the protocol is broken, every
/// subsequent call will also fail. Those errors still propagate to the file
/// level so the retry loop and fallback UTR path can attempt recovery.
///
/// # Detection
///
/// The substring `"RuntimeFailure:"` is inserted by
/// `parse_forced_alignment_result_v2()` when formatting a
/// `ProtocolErrorCodeV2::RuntimeFailure` response from the Python worker.
/// It does not appear in `ModelUnavailable`, `Protocol`, or IPC parse errors,
/// so the match is specific to data-driven model failures.
fn is_fa_runtime_failure(error: &ServerError) -> bool {
    matches!(
        error,
        ServerError::Validation(msg) if msg.contains("RuntimeFailure:")
    )
}

/// Returns `true` when `error` is a worker process crash, the Python child
/// process was killed by a signal (`exit code: None` = SIGKILL from the kernel
/// OOM-killer, or SIGSEGV/SIGABRT from a C-extension crash in torchaudio).
///
/// # Why this is always group-local
///
/// A process crash is triggered by the *content* of the request: a specific
/// combination of audio length and word count can push the Wave2Vec or Whisper
/// model into OOM or cause a C-extension assertion failure.  The crash is
/// deterministic on the same group, retrying with a fresh worker on the same
/// group will produce the same crash.  Other groups have different audio and
/// words; they will not trigger the crash and can be processed normally by the
/// replacement worker that the pool automatically spawns.
///
/// The correct recovery is therefore **group-level skip**, not file-level abort
/// followed by retry (which just respawns workers that crash on the same group).
///
/// # Contrast with infrastructure failures
///
/// `WorkerError::SpawnFailed`, `Protocol`, and `NoWorker` are environmental
/// failures that affect every subsequent dispatch.  Those still propagate to
/// the file level so the retry loop and fallback UTR path can attempt recovery.
/// `ProcessExited` is different: it is caused by this specific group's input,
/// not by a broken environment.
fn is_worker_process_crash(error: &ServerError) -> bool {
    matches!(
        error,
        ServerError::Worker(crate::worker::error::WorkerError::ProcessExited { .. })
    )
}

/// Returns true when `error` was produced because the worker that handled the
/// Whisper FA fallback request does not have a Whisper FA model loaded.
///
/// This is a **worker capability gap**, not a data quality failure.  When this
/// is true the FA group should be left unaligned (same as an empty audio
/// segment) rather than propagating a file-level error.
///
/// The distinctive substring comes from the Python worker's
/// `execute_forced_alignment_request_v2` in `crates/batchalign-pyo3/src/worker_fa_exec.rs`,
/// which returns `ProtocolErrorCodeV2::ModelUnavailable` with message
/// `"no whisper FA host loaded for worker protocol V2"` when
/// `whisper_runner` is `None`.  `parse_forced_alignment_result_v2` formats
/// this as `"… with ModelUnavailable: no whisper FA host loaded …"`.
fn is_whisper_model_unavailable(error: &ServerError) -> bool {
    matches!(
        error,
        ServerError::Validation(msg) if msg.contains("ModelUnavailable: no whisper FA host loaded")
    )
}

/// A recoverable FA failure and where the group goes next.
///
/// One value rather than a bare reason, because the reason and the engine the
/// retry runs on are one decision: the reason comes from the error, the target
/// comes from the failing engine's own row, and a caller holding only the
/// reason has to name the target itself. It did, as a literal `Whisper`, and
/// the warn line beside it said "Wave2Vec FA" for every retry, so a Cantonese
/// group reported the wrong engine to whoever read the log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FaGroupRetry {
    /// The engine the group is re-dispatched to.
    ///
    /// Read from [`FaFallbackPolicy::RetryGroupOn`], never named here. That
    /// row's target is const-checked to be language-general, which is what
    /// makes re-dispatching without a second language admission sound.
    target: crate::types::engines::FaEngineName,
    /// The exact constraint the engine reported, quoted into the warn line and
    /// into the fallback event.
    reason: &'static str,
}

/// Whether this failure is one the engine's row says to retry, and on what.
fn fa_group_retry(
    engine: crate::types::engines::FaEngineName,
    error: &ServerError,
) -> Option<FaGroupRetry> {
    // The policy is a FIELD on the engine's row in `FA_ENGINES`, not a match
    // over the roster here. The roster match, and the paragraph explaining why
    // the Qwen3 aligner has nothing to fall back to, moved to that row: a new
    // engine still cannot compile without stating a policy, and it now states
    // it beside everything else it is.
    use crate::types::engines::FaFallbackPolicy;
    let target = match engine.fallback_policy() {
        FaFallbackPolicy::RetryGroupOn(target) => target,
        FaFallbackPolicy::NoFallback => return None,
    };

    let reason = match error {
        ServerError::Validation(message)
            if message.contains("targets length is too long for CTC") =>
        {
            "targets length is too long for CTC"
        }
        ServerError::Validation(message)
            if message.contains("targets Tensor shouldn't contain blank index") =>
        {
            "targets Tensor shouldn't contain blank index"
        }
        // Wave2Vec MMS_FA has 7 conv layers (kernels [10,3,3,3,3,2,2], strides
        // [5,2,2,2,2,2,2]).  A group shorter than ~400 samples (25ms @ 16 kHz)
        // can produce fewer than 2 samples after layer 6 so layer 7 (kernel=2)
        // crashes with "Kernel size can't be greater than actual input size".
        // Whisper pads all input to 30 seconds before computing features and
        // can handle any non-zero audio length.
        ServerError::Validation(message)
            if message.contains("Kernel size can't be greater than actual input size") =>
        {
            "audio segment too short for Wave2Vec feature extractor"
        }
        _ => return None,
    };
    Some(FaGroupRetry { target, reason })
}

/// Build one production-domain `FaInferItem` from the transport-neutral batch
/// view.
fn build_fa_infer_item(batch: &FaWorkerBatch<'_>, group_index: usize) -> FaInferItem {
    let group = &batch.groups[group_index];
    FaInferItem {
        words: batch.word_texts[group_index].clone(),
        word_ids: group.words.iter().map(|word| word.stable_id()).collect(),
        word_utterance_indices: group
            .words
            .iter()
            .map(|word| word.utterance_index.raw())
            .collect(),
        word_utterance_word_indices: group
            .words
            .iter()
            .map(|word| word.utterance_word_index.raw())
            .collect(),
        audio_path: batch.audio_path.to_string_lossy().into_owned(),
        audio_start_ms: group.audio_start_ms(),
        audio_end_ms: group.audio_end_ms(),
        gap_healing: batch.gap_healing,
    }
}

/// Build unique request and artifact ids for one FA V2 request.
///
/// The request namespace is allocated once per `infer_groups_v2` call so two
/// concurrent files cannot collide on `fa-v2-request-0`, `fa-v2-request-1`,
/// and so on while sharing the same GPU worker.
fn build_fa_request_ids(request_namespace: u64, group_index: usize) -> PreparedFaRequestIdsV2 {
    PreparedFaRequestIdsV2::new(
        format!("fa-v2-request-{request_namespace}-{group_index}"),
        format!("fa-v2-payload-{request_namespace}-{group_index}"),
        format!("fa-v2-audio-{request_namespace}-{group_index}"),
    )
}

#[cfg(test)]
mod tests {
    use crate::chat_ops::fa::coordinates::{FileMs, Recording};
    use crate::chat_ops::fa::{FaWord, TimeSpan};
    use crate::chat_ops::{UtteranceIdx, WordIdx};

    use super::*;
    use crate::api::DurationSeconds;
    use crate::types::worker_v2::{
        ExecuteResponseV2, TaskResultV2, TranslationItemResultV2, TranslationResultV2,
        WorkerRequestIdV2,
    };
    use crate::worker::error::WorkerError;

    #[test]
    fn worker_outcomes_serialize_distinct_evidence_provenance() {
        // A failed call and a successful response that aligned no words have
        // the same timing projection. The outcome, not those timings, owns
        // the distinction recorded in the evidence artifact.
        let outcomes = [
            (
                FaWorkerGroupResult::Unaligned(FaWorkerUnalignedResult {
                    group_index: 7,
                    word_count: 2,
                }),
                "unaligned",
            ),
            (
                FaWorkerGroupResult::Evidence(Box::new(FaWorkerEvidenceResult {
                    group_index: 7,
                    timings: vec![None; 2],
                    raw_evidence: None,
                    fallback_event: None,
                })),
                "inference",
            ),
        ];
        for (outcome, expected) in outcomes {
            let projection = outcome.into_projection();
            let encoded = serde_json::to_value(projection.source()).expect("serialize source");
            assert_eq!(encoded, expected);
            let decoded: crate::types::traces::FaEvidenceSourceTrace =
                serde_json::from_value(encoded).expect("read recorded source");
            assert_eq!(serde_json::to_value(decoded).unwrap(), expected);
            assert_eq!(projection.group_index, 7);
            assert_eq!(projection.timings, vec![None; 2]);
        }
    }

    /// Build a small FA word for transport unit tests.
    fn make_word(index: usize, text: &str) -> FaWord {
        FaWord {
            utterance_index: UtteranceIdx::new(0),
            utterance_word_index: WordIdx::new(index),
            text: text.into(),
        }
    }

    /// A recording long enough to contain every window these tests build.
    fn test_recording() -> Recording {
        Recording::of_duration(crate::chat_ops::fa::coordinates::Ms(600_000)).expect("non-zero")
    }

    #[test]
    fn builds_fa_infer_item_from_transport_neutral_batch() {
        let word_texts = vec![vec!["hello".to_string(), "world".to_string()]];
        let groups = vec![FaGroup::test_fixture(
            TimeSpan::new(100, 900),
            vec![make_word(0, "hello"), make_word(1, "world")],
            vec![UtteranceIdx::new(0)],
        )];
        let misses = [0];
        let authorization = match plan_fa_inference(CachePolicy::UseCache, &misses)
            .expect("ordinary cache misses may infer")
        {
            FaInferencePlan::Authorized(authorization) => authorization,
            FaInferencePlan::NothingToInfer => panic!("one miss must require inference"),
        };
        let cache_keys = vec![crate::chat_ops::CacheKey::from_content("test-group")];
        let batch = UncheckedFaWorkerBatch {
            word_texts: &word_texts,
            groups: &groups,
            cache_keys: &cache_keys,
            authorization,
            audio_path: Path::new("/tmp/input.wav"),
            worker_lang: WorkerLanguage::from(crate::api::LanguageCode3::eng()),
            engine: crate::types::engines::FaEngineName::Whisper,
            gap_healing: WordGapHealing::PreserveMeasured,
        }
        .admit()
        .expect("parallel batch inputs agree");

        let item = build_fa_infer_item(&batch, 0);
        assert_eq!(item.words, vec!["hello".to_string(), "world".to_string()]);
        assert_eq!(
            item.word_ids,
            vec!["u0:w0".to_string(), "u0:w1".to_string()]
        );
        assert_eq!(item.word_utterance_indices, vec![0, 0]);
        assert_eq!(item.word_utterance_word_indices, vec![0, 1]);
        assert_eq!(item.audio_path, "/tmp/input.wav");
        assert_eq!(item.audio_start_ms, 100);
        assert_eq!(item.audio_end_ms, 900);
        assert_eq!(item.gap_healing, WordGapHealing::PreserveMeasured);
    }

    #[test]
    fn worker_batch_refuses_parallel_group_identity_drift() {
        let groups = vec![FaGroup::test_fixture(
            TimeSpan::new(100, 900),
            vec![make_word(0, "hello")],
            vec![UtteranceIdx::new(0)],
        )];
        let word_texts = Vec::new();
        let cache_keys = vec![crate::chat_ops::CacheKey::from_content("test-group")];
        let misses = [0];
        let authorization = match plan_fa_inference(CachePolicy::UseCache, &misses)
            .expect("ordinary cache misses may infer")
        {
            FaInferencePlan::Authorized(authorization) => authorization,
            FaInferencePlan::NothingToInfer => panic!("one miss must require inference"),
        };

        let error = UncheckedFaWorkerBatch {
            word_texts: &word_texts,
            groups: &groups,
            cache_keys: &cache_keys,
            authorization,
            audio_path: Path::new("/tmp/input.wav"),
            worker_lang: WorkerLanguage::from(crate::api::LanguageCode3::eng()),
            engine: crate::types::engines::FaEngineName::Whisper,
            gap_healing: WordGapHealing::PreserveMeasured,
        }
        .admit()
        .expect_err("parallel group inputs must not drift");

        assert!(error.to_string().contains("cardinality drift"));
    }

    #[test]
    fn required_cache_misses_never_produce_fa_inference_authorization() {
        let error = plan_fa_inference(CachePolicy::RequireCache, &[1, 3])
            .expect_err("required evidence misses must refuse worker inference");

        let ServerError::RequiredEvidenceUnavailable(missing) = error else {
            panic!("cache precondition must have a typed refusal");
        };
        let MissingRequiredEvidence::ForcedAlignment(missing) = missing else {
            panic!("expected forced-alignment evidence refusal");
        };
        assert_eq!(missing.group_indices(), &[1, 3]);
    }

    #[test]
    fn reusable_fa_evidence_needs_no_authorization_even_when_required() {
        let plan = plan_fa_inference(CachePolicy::RequireCache, &[])
            .expect("no misses satisfy required-cache policy");

        assert!(matches!(plan, FaInferencePlan::NothingToInfer));
    }

    #[test]
    fn builds_namespaced_v2_request_ids_from_group_index() {
        let ids = build_fa_request_ids(42, 7);
        assert_eq!(&*ids.request_id, "fa-v2-request-42-7");
        assert_eq!(&*ids.payload_ref_id, "fa-v2-payload-42-7");
        assert_eq!(&*ids.audio_ref_id, "fa-v2-audio-42-7");
    }

    #[test]
    fn namespaces_v2_request_ids_across_concurrent_files() {
        let first = build_fa_request_ids(1, 0);
        let second = build_fa_request_ids(2, 0);

        assert_ne!(first.request_id, second.request_id);
        assert_ne!(first.payload_ref_id, second.payload_ref_id);
        assert_ne!(first.audio_ref_id, second.audio_ref_id);
    }

    #[test]
    fn parse_group_response_reports_parser_failure_with_group_context() {
        let group = FaGroup::test_fixture(
            TimeSpan::new(100, 900),
            vec![make_word(0, "hello")],
            vec![UtteranceIdx::new(0)],
        );
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("req-fa-v2-bad"),
            TaskResultV2::TranslationResult(TranslationResultV2 {
                items: vec![TranslationItemResultV2 {
                    raw_translation: Some("hola".into()),
                    error: None,
                }],
            }),
            DurationSeconds(0.01),
        );

        let recording = test_recording();
        let window = FaWindow::within(
            &recording,
            FileMs::new(group.audio_start_ms()),
            FileMs::new(group.audio_end_ms()),
        )
        .expect("test group lies inside the test recording");
        let error = parse_group_response(&response, 13, &group, &window, &EngineId::new("test-fa"))
            .expect_err("non-FA payload should fail immediately");

        assert!(
            error
                .to_string()
                .contains("failed to parse worker protocol V2 FA response for group 13")
        );
        assert!(error.to_string().contains("translation data"));
    }

    #[test]
    fn whisper_fallback_triggers_for_known_wave2vec_target_failures() {
        let overflow = ServerError::Validation(
            "failed to parse worker protocol V2 FA response for group 13 (175765..176365 ms): \
             worker protocol V2 forced-alignment request failed with RuntimeFailure: \
             targets length is too long for CTC"
                .into(),
        );
        assert_eq!(
            fa_group_retry(crate::types::engines::FaEngineName::Wave2Vec, &overflow),
            Some(FaGroupRetry {
                target: crate::types::engines::FaEngineName::Whisper,
                reason: "targets length is too long for CTC",
            })
        );
        assert_eq!(
            fa_group_retry(crate::types::engines::FaEngineName::Whisper, &overflow),
            None
        );

        let blank_index = ServerError::Validation(
            "failed to parse worker protocol V2 FA response for group 56 (754285..767165 ms): \
             worker protocol V2 forced-alignment request failed with RuntimeFailure: \
             ValueError: targets Tensor shouldn't contain blank index. Found tensor([[20, 5, 10, 10]])"
                .into(),
        );
        assert_eq!(
            fa_group_retry(crate::types::engines::FaEngineName::Wave2Vec, &blank_index),
            Some(FaGroupRetry {
                target: crate::types::engines::FaEngineName::Whisper,
                reason: "targets Tensor shouldn't contain blank index",
            })
        );
        assert_eq!(
            fa_group_retry(crate::types::engines::FaEngineName::Whisper, &blank_index),
            None
        );

        // Wave2Vec conv layer 7 (kernel=2) crashes when input shrinks to 1 sample.
        // Observed: group 27 (296480..296500 ms = 20 ms window = 320 samples),
        // job 2afcc302-bfd, file 28-NM-63-4.cha.
        let kernel_too_large = ServerError::Validation(
            "failed to parse worker protocol V2 FA response for group 27 (296480..296500 ms): \
             worker protocol V2 forced-alignment request failed with RuntimeFailure: \
             RuntimeError: Calculated padded input size per channel: (1). \
             Kernel size: (2). Kernel size can't be greater than actual input size"
                .into(),
        );
        assert_eq!(
            fa_group_retry(
                crate::types::engines::FaEngineName::Wave2Vec,
                &kernel_too_large
            ),
            Some(FaGroupRetry {
                target: crate::types::engines::FaEngineName::Whisper,
                reason: "audio segment too short for Wave2Vec feature extractor",
            })
        );
        assert_eq!(
            fa_group_retry(
                crate::types::engines::FaEngineName::Whisper,
                &kernel_too_large
            ),
            None
        );

        let other = ServerError::Validation("some other parse failure".into());
        assert_eq!(
            fa_group_retry(crate::types::engines::FaEngineName::Wave2Vec, &other),
            None
        );
    }

    /// The retry each engine offers, written out per variant.
    ///
    /// An exhaustive `match` and not a read of `fallback_policy()`, which is
    /// the field the function under test reads: an expectation derived from
    /// that field stays green for ANY value it holds. A new engine fails to
    /// compile here, next to the values it must state.
    ///
    /// The self-retry case this used to be the only guard against, a row
    /// naming its own engine as its target, is refused by the `const` block
    /// beside `FA_ENGINES` and can no longer be written at all. What is left
    /// for this match is the pair no type pins: that the target and the reason
    /// the transport hands back are the ones the row declares.
    fn expected_retry(engine: crate::types::engines::FaEngineName) -> Option<FaGroupRetry> {
        use crate::types::engines::FaEngineName;
        const RECOVERABLE: &str = "targets length is too long for CTC";
        match engine {
            FaEngineName::Wave2Vec => Some(FaGroupRetry {
                target: FaEngineName::Whisper,
                reason: RECOVERABLE,
            }),
            FaEngineName::Wav2vecCanto => Some(FaGroupRetry {
                target: FaEngineName::Whisper,
                reason: RECOVERABLE,
            }),
            // It IS the fallback target and cannot retry itself.
            FaEngineName::Whisper => None,
            // Its aligner is not a CTC decoder, so no recoverable CTC
            // constraint is reachable for it, and a whole-group retry would
            // replace measured Qwen3 timings with another model's.
            FaEngineName::Qwen3 => None,
        }
    }

    /// Every engine offers exactly the retry written out above, on a failure
    /// all of them could see.
    ///
    /// A ROUNDTRIP between the declaration table and the transport that reads
    /// it, which no signature pins: the row says "retry on Whisper", and only
    /// running the function shows that a Whisper request is what comes back.
    #[test]
    fn every_fa_engine_falls_back_exactly_as_its_row_declares() {
        use crate::types::engines::{FaEngineName, SelectableEngine};

        let recoverable = ServerError::Validation(
            "worker protocol V2 forced-alignment request failed with RuntimeFailure: \
             targets length is too long for CTC"
                .into(),
        );
        for engine in FaEngineName::ALL.iter().copied() {
            assert_eq!(
                fa_group_retry(engine, &recoverable),
                expected_retry(engine),
                "{engine:?} did not offer the retry this test writes out for it"
            );
        }
    }

    /// a user's bug (2026-04-08): `batchalign3 align` silently drops files
    /// when Wave2Vec falls back to Whisper FA but the worker has no Whisper
    /// model loaded.
    ///
    /// Repro:
    ///   batchalign3 align ~/ba_data/input ~/ba_data/output
    ///   → Job submitted for 4 files; 45-3.cha and 86-3.cha missing from output
    ///
    /// Server log:
    ///   Wave2Vec FA hit recoverable target constraint; retrying group with Whisper FA
    ///   FA error (raw): ModelUnavailable: no whisper FA host loaded for worker protocol V2
    ///
    /// Root cause: `infer_groups_v2` dispatches the Whisper fallback, but when
    /// the worker returns `ModelUnavailable` (because `whisper_runner = None`),
    /// `parse_group_response` wraps the error into a `ServerError::Validation`
    /// that looks identical to a fatal data error.  The `?` on the fallback call
    /// propagates it as a **file-level** failure instead of leaving the group
    /// unaligned the way empty audio segments are handled.
    ///
    /// Fix: implement `is_whisper_model_unavailable` so `infer_groups_v2` can
    /// detect the capability gap and leave the group with `None` timings.
    #[test]
    fn whisper_fallback_model_unavailable_is_detectable_as_capability_gap_not_data_error() {
        // This is the exact error produced in production:
        // parse_group_response wraps the ModelUnavailable worker response into
        // a ServerError::Validation with this message.
        let model_unavailable = ServerError::Validation(
            "failed to parse worker protocol V2 FA response for group 24 (379515..381395 ms): \
             worker protocol V2 forced-alignment request failed with ModelUnavailable: \
             no whisper FA host loaded for worker protocol V2"
                .into(),
        );

        // The test asserts that is_whisper_model_unavailable can distinguish
        // this worker-capability error from an ordinary data-quality error.
        // Without this predicate, infer_groups_v2 has no way to leave the
        // group unaligned instead of killing the file.
        //
        // This assertion is currently RED: is_whisper_model_unavailable always
        // returns false (stub).  Implementing it makes the test GREEN.
        assert!(
            is_whisper_model_unavailable(&model_unavailable),
            "ModelUnavailable from Whisper fallback should be detectable \
             so infer_groups_v2 can leave the group unaligned"
        );

        // A plain data-quality error must NOT be mistaken for a capability gap.
        let data_error = ServerError::Validation(
            "failed to parse worker protocol V2 FA response for group 5 (1000..2000 ms): \
             some parse error from the model output"
                .into(),
        );
        assert!(
            !is_whisper_model_unavailable(&data_error),
            "ordinary data errors must not be treated as ModelUnavailable"
        );
    }

    /// Any `RuntimeFailure` from the FA worker is data-driven: the model
    /// failed on *this group's* specific input. Other groups have different
    /// words and audio and will not trigger the same failure. These errors
    /// must be demoted to group-level skips, never propagated as file-level
    /// failures. This test verifies the detection predicate covers:
    ///
    /// - The exact 448-token overflow seen on `biling-data/DiazCollazos`
    /// - Generic RuntimeFailure variants (shape errors, OOM, etc.)
    /// - Does NOT fire on infrastructure errors (IPC parse, model unavailable)
    #[test]
    fn fa_runtime_failure_is_detectable_as_data_driven_group_error() {
        // Exact message from production: biling-data/DiazCollazos/09.cha, job ad9eb6ba,
        // group 0 (0..10970 ms), 2043 chars > 448 token limit.
        let overflow_448 = ServerError::Validation(
            "failed to parse worker protocol V2 FA response for group 0 (0..10970 ms): \
             worker protocol V2 forced-alignment request failed with RuntimeFailure: \
             ValueError: Labels' sequence length 2043 cannot exceed the maximum \
             allowed length of 448 tokens."
                .into(),
        );
        assert!(
            is_fa_runtime_failure(&overflow_448),
            "448-token Whisper CTC overflow must be detectable as a RuntimeFailure"
        );

        // Generic RuntimeFailure (shape error, device mismatch, etc.), also group-local.
        let shape_error = ServerError::Validation(
            "failed to parse worker protocol V2 FA response for group 3 (50000..70000 ms): \
             worker protocol V2 forced-alignment request failed with RuntimeFailure: \
             RuntimeError: Expected all tensors to be on the same device"
                .into(),
        );
        assert!(
            is_fa_runtime_failure(&shape_error),
            "generic RuntimeFailure must also be detectable"
        );

        // Infrastructure error (IPC parse failure, no RuntimeFailure code), must NOT match.
        let ipc_error = ServerError::Validation(
            "failed to parse worker protocol V2 FA response for group 5 (0..5000 ms): \
             translation data (not FA)"
                .into(),
        );
        assert!(
            !is_fa_runtime_failure(&ipc_error),
            "non-RuntimeFailure IPC error must not be mistaken for a data-driven group error"
        );

        // ModelUnavailable: a capability gap, not a RuntimeFailure.
        let model_unavail = ServerError::Validation(
            "failed to parse worker protocol V2 FA response for group 24 (379515..381395 ms): \
             worker protocol V2 forced-alignment request failed with ModelUnavailable: \
             no whisper FA host loaded for worker protocol V2"
                .into(),
        );
        assert!(
            !is_fa_runtime_failure(&model_unavail),
            "ModelUnavailable must not be mistaken for a data-driven RuntimeFailure"
        );
    }

    /// When Wave2Vec falls back to Whisper FA and Whisper itself hits a RuntimeFailure
    /// (e.g., the group is too long even for Whisper), that fallback error must also be
    /// detectable and demoted to a group-level skip rather than a file-level abort.
    #[test]
    fn fa_runtime_failure_from_whisper_fallback_is_detectable() {
        let whisper_overflow = ServerError::Validation(
            "failed to parse worker protocol V2 FA response for group 7 (100000..111000 ms): \
             worker protocol V2 forced-alignment request failed with RuntimeFailure: \
             ValueError: Labels' sequence length 512 cannot exceed the maximum \
             allowed length of 448 tokens."
                .into(),
        );
        assert!(is_fa_runtime_failure(&whisper_overflow));
        // Must not be confused with the capability-gap path.
        assert!(!is_whisper_model_unavailable(&whisper_overflow));
    }

    /// Worker process crashes (SIGKILL, C-extension SIGSEGV) must be detectable
    /// as a group-local signal so `infer_groups_v2` can leave the crashing group
    /// unaligned and continue processing remaining groups rather than aborting
    /// the entire file.
    ///
    /// Root cause confirmed for job 1020067a-85f: one GPU worker (pid=12656)
    /// crashed with `exit code: None` (SIGKILL) while processing groups from 3
    /// concurrent files.  Every retry hit the same crashing group, died within
    /// 1-4 s, and all 3 files failed.  The fix: treat `ProcessExited` as a
    /// group-level skip, identical to `EmptyFaAudioSegment` and `RuntimeFailure`.
    #[test]
    fn worker_process_crash_is_detectable_as_group_local_signal() {
        // exit code: None = SIGKILL (kernel OOM-killer) or C-extension crash
        // that kills the process before Python's exception handling can run.
        let sigkill = ServerError::Worker(WorkerError::ProcessExited {
            code: None,
            stderr: None,
        });
        assert!(
            is_worker_process_crash(&sigkill),
            "SIGKILL (exit code None) must be detectable as a process crash"
        );

        // Explicit non-zero exit code (e.g., SIGSEGV = 139, SIGABRT = 134)
        // still classifies as a crash, the model died on this group's content.
        let sigsegv = ServerError::Worker(WorkerError::ProcessExited {
            code: Some(139),
            stderr: Some("Segmentation fault: 11".to_string()),
        });
        assert!(
            is_worker_process_crash(&sigsegv),
            "Non-zero exit code (SIGSEGV=139) must also be detectable as a process crash"
        );

        // Infrastructure failures that are NOT process crashes must not be
        // conflated; they should still propagate to the file level.
        let data_error = ServerError::Validation(
            "failed to parse worker protocol V2 FA response for group 5 (0..5000 ms): \
             some parse error from the model output"
                .into(),
        );
        assert!(
            !is_worker_process_crash(&data_error),
            "a Validation error must not be mistaken for a process crash"
        );
    }

    #[test]
    fn build_fallback_event_captures_group_and_engine_metadata() {
        let group = FaGroup::test_fixture(
            TimeSpan::new(175_765, 176_365),
            vec![make_word(0, "hello")],
            vec![UtteranceIdx::new(0)],
        );

        let event = build_fallback_event(
            13,
            &group,
            crate::types::engines::FaEngineName::Wave2Vec,
            crate::types::engines::FaEngineName::Whisper,
            "targets length is too long for CTC",
        );

        assert_eq!(event.group_index, 13);
        assert_eq!(event.from_engine, "wav2vec_fa");
        assert_eq!(event.to_engine, "whisper_fa");
        assert_eq!(event.reason, "targets length is too long for CTC");
        assert_eq!(event.audio_start_ms.0, 175_765);
        assert_eq!(event.audio_end_ms.0, 176_365);
    }
}
