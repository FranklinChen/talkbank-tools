//! Transport adapter for forced-alignment worker inference.
//!
//! The FA pipeline delegates worker interaction through this module so the
//! orchestration code can ask for "timings for these missed requests" without
//! depending on the concrete worker-protocol V2 request-building details.
//!
//! Everything here is per DISPATCH UNIT (one request: a group's words, or one
//! piece of an anchored group, with its window and cache key), never per
//! group; see `super::units`.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::api::{DurationMs, WorkerLanguage};
use crate::chat_ops::fa::alignment::FaAlignmentError;
use crate::chat_ops::fa::origin::EngineId;
use crate::chat_ops::fa::{FaInferItem, WhisperTextMode, WordTiming};
use crate::error::{MissingForcedAlignmentEvidence, MissingRequiredEvidence, ServerError};
use crate::params::CachePolicy;
use crate::types::engines::SelectableEngine;
use crate::types::traces::FaFallbackEventTrace;
use crate::worker::artifacts_v2::PreparedArtifactRuntimeV2;
use crate::worker::fa_result_v2::{FaEngineConstraint, FaFailureClass, FaWorkerFailure};
use crate::worker::request_builder_v2::{
    ForcedAlignmentBuildInputV2, ForcedAlignmentRequestBuildErrorV2, PreparedFaRequestIdsV2,
    build_forced_alignment_request_v2,
};
use tracing::warn;

use super::raw_evidence::{
    ExpectedFaWords, FaEvidenceRoute, FaRawEvidence, FaRawEvidenceError, ReplayableFaRawEvidence,
};
use super::units::{DispatchUnit, UnitOrdinal};

static NEXT_FA_REQUEST_NAMESPACE: AtomicU64 = AtomicU64::new(1);

/// One FA worker batch: the authorized missed requests and the run facts
/// every request is built with.
///
/// Built directly: the authorization already holds the missed units
/// themselves, borrowed from the plan that laid them out, so there is no
/// index to check against a separate list of units.
#[derive(Debug)]
pub(crate) struct FaWorkerBatch<'a> {
    /// Proof that the cache policy permits inference for these requests.
    pub authorization: FaInferenceAuthorization<'a>,
    /// Source audio path for the current file.
    pub audio_path: &'a Path,
    /// Worker-runtime language hint for FA model bootstrap.
    pub worker_lang: WorkerLanguage,
    /// FA backend selected by the Rust control plane.
    pub engine: crate::types::engines::FaEngineName,
    /// Whisper input shaping for every request in this batch.
    pub whisper_text: WhisperTextMode,
}

/// The only value that permits FA worker inference for cache misses.
///
/// Its field is private and [`plan_fa_inference`] is its only constructor, so
/// a required-cache miss has no route to [`FaWorkerBatch`].
#[derive(Debug)]
pub(crate) struct FaInferenceAuthorization<'a> {
    misses: Vec<&'a DispatchUnit<'a>>,
}

/// Whether one cache partition needs and permits worker inference.
#[derive(Debug)]
pub(crate) enum FaInferencePlan<'a> {
    /// Every request was satisfied by reusable or cached evidence.
    NothingToInfer,
    /// Cache misses exist and policy permits worker inference.
    Authorized(FaInferenceAuthorization<'a>),
}

/// Convert cache misses into a worker-inference capability, or refuse when
/// the job required complete reusable evidence.
pub(crate) fn plan_fa_inference<'a>(
    policy: CachePolicy,
    misses: Vec<&'a DispatchUnit<'a>>,
) -> Result<FaInferencePlan<'a>, ServerError> {
    let Some((first_miss, remaining_misses)) = misses.split_first() else {
        return Ok(FaInferencePlan::NothingToInfer);
    };
    match policy {
        CachePolicy::RequireCache => {
            return Err(ServerError::RequiredEvidenceUnavailable(
                MissingRequiredEvidence::ForcedAlignment(MissingForcedAlignmentEvidence::new(
                    first_miss.ordinal().index(),
                    remaining_misses.iter().map(|unit| unit.ordinal().index()),
                )),
            ));
        }
        CachePolicy::UseCache | CachePolicy::SkipCache => {}
    }
    Ok(FaInferencePlan::Authorized(FaInferenceAuthorization {
        misses,
    }))
}

/// Result of attempting worker inference for one FA request.
///
/// Successful model evidence and an intentional unaligned fallback are
/// distinct states. Only the former can cross the raw-evidence cache boundary.
#[derive(Debug)]
pub(crate) enum FaWorkerUnitResult<'a> {
    /// A successful worker response admitted against the request and parsed.
    Evidence(Box<FaWorkerEvidenceResult<'a>>),
    /// A request-local failure deliberately represented as unaligned words.
    Unaligned(FaWorkerUnalignedResult<'a>),
}

/// Timing projection and provenance issued together by the worker outcome.
/// Callers cannot construct a projection with a fabricated evidence source.
pub(crate) struct FaWorkerProjection<'a> {
    /// The request these timings answer.
    pub unit: &'a DispatchUnit<'a>,
    /// One optional timing per requested word.
    pub timings: Vec<Option<WordTiming>>,
    /// Replayable direct evidence, when admitted by the worker boundary.
    pub raw_evidence: Option<ReplayableFaRawEvidence>,
    /// Engine fallback associated with this evidence, if any.
    pub fallback_event: Option<FaFallbackEventTrace>,
    source: crate::types::traces::FaEvidenceSourceTrace,
}

impl FaWorkerProjection<'_> {
    /// Provenance of this projection, including the absence of worker evidence.
    pub fn source(&self) -> crate::types::traces::FaEvidenceSourceTrace {
        self.source
    }
}

impl<'a> FaWorkerUnitResult<'a> {
    /// Preserve the outcome distinction when materializing unaligned words.
    pub fn into_projection(self) -> FaWorkerProjection<'a> {
        use crate::types::traces::FaEvidenceSourceTrace;
        match self {
            Self::Evidence(evidence) => FaWorkerProjection {
                unit: evidence.unit,
                timings: evidence.timings,
                raw_evidence: evidence.raw_evidence,
                fallback_event: evidence.fallback_event,
                source: FaEvidenceSourceTrace::Inference,
            },
            Self::Unaligned(unaligned) => FaWorkerProjection {
                unit: unaligned.unit,
                timings: vec![None; unaligned.unit.words().len()],
                raw_evidence: None,
                fallback_event: None,
                source: FaEvidenceSourceTrace::Unaligned,
            },
        }
    }
}

/// Successful FA evidence paired inseparably with its parsed timing projection.
#[derive(Debug)]
pub(crate) struct FaWorkerEvidenceResult<'a> {
    /// The request these timings answer.
    pub unit: &'a DispatchUnit<'a>,
    /// Parsed timings in the established Rust FA timing domain.
    pub timings: Vec<Option<WordTiming>>,
    /// Immutable worker response admitted against engine and word cardinality.
    pub raw_evidence: Option<ReplayableFaRawEvidence>,
    /// Fallback event metadata when this request had to retry with another engine.
    pub fallback_event: Option<FaFallbackEventTrace>,
}

/// A request-local failure that is safe to retain as explicitly unaligned:
/// every one of the request's words, and no more, is left without timing.
#[derive(Debug)]
pub(crate) struct FaWorkerUnalignedResult<'a> {
    /// The request left unaligned.
    pub unit: &'a DispatchUnit<'a>,
}

/// Narrow transport adapter for FA worker inference.
#[derive(Clone, Copy)]
pub(crate) enum FaWorkerTransport<'a> {
    /// Live typed worker-protocol V2 transport using prepared artifacts.
    V2 {
        /// FA services: worker-pool access plus the namespace admitted
        /// evidence is recorded under.
        services: super::FaServices<'a>,
    },
}

impl<'a> FaWorkerTransport<'a> {
    /// Return the production FA worker transport.
    pub(crate) fn production(services: super::FaServices<'a>) -> Self {
        Self::V2 { services }
    }

    /// Infer timings for the requested FA missed requests.
    pub(crate) async fn infer_units<'b>(
        self,
        batch: FaWorkerBatch<'b>,
    ) -> Result<Vec<FaWorkerUnitResult<'b>>, ServerError> {
        match self {
            Self::V2 { services } => infer_units_v2(services, batch).await,
        }
    }
}

/// Dispatch staged worker-protocol V2 requests for each missed FA request and
/// parse successful results back into the established Rust timing domain.
async fn infer_units_v2<'b>(
    services: super::FaServices<'_>,
    batch: FaWorkerBatch<'b>,
) -> Result<Vec<FaWorkerUnitResult<'b>>, ServerError> {
    let request_namespace = NEXT_FA_REQUEST_NAMESPACE.fetch_add(1, Ordering::Relaxed);
    let artifacts = PreparedArtifactRuntimeV2::new("fa_v2").map_err(|error| {
        ServerError::Validation(format!("failed to create FA V2 artifact runtime: {error}"))
    })?;

    // One route for every unit of the batch: the job's plan, the FA task and
    // the batch's language are the same for all of them.
    let route = services.pipeline.pool.gpu_route(
        services.gpu,
        crate::worker::pool::GpuTask::Fa,
        batch.worker_lang.clone(),
    )?;

    let mut parsed_results = Vec::with_capacity(batch.authorization.misses.len());
    for &unit in &batch.authorization.misses {
        let ordinal = unit.ordinal();
        // Grouping carries the admitted recording window through dispatch:
        // the group's window, or this piece's.
        let window = unit.window();
        let group = unit.group();

        let response = match dispatch_unit_request(
            services,
            &artifacts,
            &batch,
            request_namespace,
            unit,
            batch.engine,
            &route,
        )
        .await
        {
            Ok(response) => response,
            Err(ServerError::EmptyFaAudioSegment(ref segment)) => {
                // The window is an `FaWindow`, admitted inside the probed
                // recording, so past-EOF cannot be the cause; report its
                // length, the recording's, and what the decode measured.
                // Leave the group's words unaligned; the transcript is still
                // useful without timing.
                warn!(
                    request = ordinal.index(),
                    group = group.index(),
                    start_ms = segment.window.audio_start().get(),
                    end_ms = segment.window.end().get(),
                    window_ms = window.len().0,
                    recording_ms = window.recording().duration().get(),
                    decoded = %segment.reason,
                    path = %segment.path,
                    "FA request window inside the recording decoded to no whole audio sample; \
                     leaving words unaligned"
                );
                parsed_results.push(unaligned_unit_result(unit));
                continue;
            }
            // Process exit proves that this request lost its worker, not why
            // the process exited or whether this input caused it. Preserve the
            // group as unaligned; the pool replaces a dead shared generation
            // before later requests. Do not label an unknown exit as OOM.
            Err(ref e) if is_worker_process_crash(e) => {
                warn!(
                    request = ordinal.index(),
                    group = group.index(),
                    start_ms = window.audio_start().get(),
                    end_ms = window.end().get(),
                    error = %e,
                    "Worker process exited during FA request; leaving words unaligned"
                );
                parsed_results.push(unaligned_unit_result(unit));
                continue;
            }
            Err(other) => return Err(other),
        };

        // Bind every fact that identifies this request before interpreting
        // any response. Primary and fallback responses must travel through
        // the same capability, so a later branch cannot accidentally pair a
        // response with another request's key, window, or word cardinality.
        let admission = FaUnitEvidenceAdmission {
            requested_engine: batch.engine,
            cache_namespace: services.cache_namespace,
            unit,
        };

        let failure = match admission.admit(&response, FaEvidenceRoute::Direct) {
            Ok(parsed) => {
                parsed_results.push(FaWorkerUnitResult::Evidence(Box::new(parsed)));
                continue;
            }
            Err(failure) => failure,
        };
        let retry = match PrimaryStep::after(batch.engine, &failure) {
            PrimaryStep::Retry(retry) => retry,
            // The model failed on this request's own input. Other requests
            // have different words and audio, so the request is left
            // unaligned and the file continues.
            PrimaryStep::LeaveUnaligned => {
                warn!(
                    request = ordinal.index(),
                    group = group.index(),
                    start_ms = window.audio_start().get(),
                    end_ms = window.end().get(),
                    error = %failure,
                    "FA engine constraint has no fallback; \
                     leaving words unaligned"
                );
                parsed_results.push(unaligned_unit_result(unit));
                continue;
            }
            PrimaryStep::FailFile => return Err(failure.into()),
        };
        warn!(
            request = ordinal.index(),
            group = group.index(),
            start_ms = window.audio_start().get(),
            end_ms = window.end().get(),
            reason = retry.constraint.reason(),
            failed_engine = batch.engine.selection_name(),
            retry_engine = retry.target.selection_name(),
            "FA engine hit a recoverable target constraint; retrying request on its \
             fallback engine"
        );
        let fallback_namespace = NEXT_FA_REQUEST_NAMESPACE.fetch_add(1, Ordering::Relaxed);
        // Built only on this rare path: most batches never retry.
        let fallback_route = services.pipeline.pool.gpu_route(
            services.gpu,
            crate::worker::pool::GpuTask::FaFallback,
            batch.worker_lang.clone(),
        )?;
        let fallback_response = dispatch_unit_request(
            services,
            &artifacts,
            &batch,
            fallback_namespace,
            unit,
            retry.target,
            &fallback_route,
        )
        .await?;
        match admission.admit(
            &fallback_response,
            FaEvidenceRoute::Fallback {
                constraint: retry.constraint,
            },
        ) {
            Ok(parsed) => parsed_results.push(FaWorkerUnitResult::Evidence(Box::new(
                parsed.with_fallback_event(build_fallback_event(
                    unit,
                    batch.engine,
                    retry.target,
                    retry.constraint,
                )),
            ))),
            Err(failure) => match FallbackStep::after(&failure) {
                // The fallback model failed on this input too (the request is
                // still too long for its own context), or the worker has no
                // fallback model loaded: a capability gap, not a data error.
                // Either way the surrounding requests keep their timing.
                FallbackStep::LeaveUnaligned => {
                    warn!(
                        request = ordinal.index(),
                        group = group.index(),
                        start_ms = window.audio_start().get(),
                        end_ms = window.end().get(),
                        error = %failure,
                        retry_engine = retry.target.selection_name(),
                        "fallback FA engine could not align the request either; \
                         leaving request words unaligned"
                    );
                    parsed_results.push(unaligned_unit_result(unit));
                }
                FallbackStep::FailFile => return Err(failure.into()),
            },
        }
    }

    Ok(parsed_results)
}

/// Construct a request result with every word timing left unaligned (`None`).
///
/// All request-level skip paths (empty audio, model unavailability,
/// recognised engine constraints) return this same shape so the orchestrator can
/// continue to the next request without aborting the file. For an anchored
/// group that leaves one piece untimed and its neighbours aligned.
fn unaligned_unit_result<'a>(unit: &'a DispatchUnit<'a>) -> FaWorkerUnitResult<'a> {
    FaWorkerUnitResult::Unaligned(FaWorkerUnalignedResult { unit })
}

async fn dispatch_unit_request(
    services: super::FaServices<'_>,
    artifacts: &PreparedArtifactRuntimeV2,
    batch: &FaWorkerBatch<'_>,
    request_namespace: u64,
    unit: &DispatchUnit<'_>,
    engine: crate::types::engines::FaEngineName,
    route: &crate::worker::pool::GpuRoute,
) -> Result<crate::types::worker_v2::ExecuteResponseV2, ServerError> {
    let infer_item = build_fa_infer_item(batch, unit);
    let request_ids = build_fa_request_ids(request_namespace, unit.ordinal());
    let mut request = build_forced_alignment_request_v2(
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
            "failed to build worker protocol V2 FA request {} (group {}): {other}",
            unit.ordinal(),
            unit.group()
        )),
    })?;

    let capture = services
        .attention_capture_dir
        .filter(|_| engine == crate::types::engines::FaEngineName::Whisper)
        .map(|dir| super::attention_capture::request(dir, unit.cache_key().as_str()))
        .transpose()?;
    if let crate::types::worker_v2::TaskRequestV2::ForcedAlignment(payload) = &mut request.payload {
        payload.attention_capture = capture.clone();
    }
    let response = services
        .pipeline
        .pool
        .dispatch_routed_v2(route, &request)
        .await
        .map_err(|error| {
            if capture.is_some() {
                ServerError::Validation(format!("attention capture worker failed: {error}"))
            } else {
                ServerError::Worker(error)
            }
        })?;
    if let Some(capture) = capture {
        super::attention_capture::admit_response(&capture, &request.request_id, &response).await?;
    }
    Ok(response)
}

/// Lower one worker response into the FA timing domain.
///
/// Takes the unit's window RATHER THAN rebuilding it from the recording. The
/// window is proved once, by grouping, before inference is dispatched; a
/// second construction here would be the same check in a second place, and the
/// failure would arrive after the expensive part had already been paid for.
/// Window-relative engine reports cross into file coordinates through THIS
/// window, a piece's own for a piece, exactly as a single group's do.
///
/// Takes the evidence's admitted payload, never the response: admission read
/// the response once, so a failed or wrong-kind response never reaches here,
/// and what is left to go wrong is pairing the timings with the words.
fn parse_unit_response(
    raw_evidence: &FaRawEvidence,
    unit: &DispatchUnit<'_>,
) -> Result<Vec<Option<WordTiming>>, FaUnitFailure> {
    let window = unit.window();
    crate::chat_ops::fa::align_fa_response(
        raw_evidence.payload().clone(),
        unit.words(),
        &window,
        &EngineId::new(raw_evidence.effective_engine().as_wire_name()),
    )
    .map_err(|source| FaUnitFailure::Response {
        request: unit.ordinal().index(),
        group: unit.group().index(),
        start_ms: window.audio_start().get(),
        end_ms: window.end().get(),
        source,
    })
}

impl FaWorkerEvidenceResult<'_> {
    fn with_fallback_event(mut self, fallback_event: FaFallbackEventTrace) -> Self {
        self.fallback_event = Some(fallback_event);
        self
    }
}

/// Capability binding every current-request fact needed to admit one
/// request's worker response.
///
/// Keeping these values together prevents the primary and fallback branches
/// from independently reconstructing the relationship by convention. The
/// unit carries the key, the window and the words, so they cannot belong to
/// different requests.
struct FaUnitEvidenceAdmission<'s, 'a> {
    requested_engine: crate::types::engines::FaEngineName,
    cache_namespace: &'s crate::engine_reports::FaCacheNamespace,
    unit: &'a DispatchUnit<'a>,
}

impl<'a> FaUnitEvidenceAdmission<'_, 'a> {
    fn admit(
        &self,
        response: &crate::types::worker_v2::ExecuteResponseV2,
        route: FaEvidenceRoute,
    ) -> Result<FaWorkerEvidenceResult<'a>, FaUnitFailure> {
        let raw_evidence = FaRawEvidence::admit_requested(
            response,
            self.requested_engine,
            self.cache_namespace,
            ExpectedFaWords::new(self.unit.words().len()),
            self.unit.cache_key(),
            route,
        )
        .map_err(|source| match source {
            FaRawEvidenceError::WorkerFailure(failure) => FaUnitFailure::worker(self.unit, failure),
            source => FaUnitFailure::Evidence {
                request: self.unit.ordinal().index(),
                source,
            },
        })?;
        let timings = parse_unit_response(&raw_evidence, self.unit)?;
        // Closing admitted evidence refuses only fallback output, which is
        // valid for this run and never persisted.
        let replayable_raw_evidence = raw_evidence.into_replayable().ok();
        Ok(FaWorkerEvidenceResult {
            unit: self.unit,
            timings,
            raw_evidence: replayable_raw_evidence,
            fallback_event: None,
        })
    }
}

/// Reparse one admitted cached response without invoking a model worker.
pub(super) fn replay_unit_evidence<'a>(
    raw_evidence: ReplayableFaRawEvidence,
    unit: &'a DispatchUnit<'a>,
) -> Result<FaWorkerEvidenceResult<'a>, ServerError> {
    let raw_evidence = raw_evidence.into_inner();
    let timings = parse_unit_response(&raw_evidence, unit)?;
    // Replayable evidence is never fallback evidence, so it has no fallback
    // event to report.
    Ok(FaWorkerEvidenceResult {
        unit,
        timings,
        raw_evidence: None,
        fallback_event: None,
    })
}

/// The fallback event for one request: its group, and the window the
/// fallback engine was actually given (a piece's, for a piece).
fn build_fallback_event(
    unit: &DispatchUnit<'_>,
    from_engine: crate::types::engines::FaEngineName,
    to_engine: crate::types::engines::FaEngineName,
    constraint: FaEngineConstraint,
) -> FaFallbackEventTrace {
    FaFallbackEventTrace {
        group_index: unit.group().index(),
        from_engine: {
            use crate::types::engines::EngineBackend;
            from_engine.wire_name().to_string()
        },
        to_engine: {
            use crate::types::engines::EngineBackend;
            to_engine.wire_name().to_string()
        },
        reason: constraint.reason().to_string(),
        audio_start_ms: DurationMs(unit.window().audio_start().get()),
        audio_end_ms: DurationMs(unit.window().end().get()),
    }
}

/// Why one request's worker response was not admitted as timing evidence.
///
/// Typed so that what a failed request does next is decided by matching it
/// ([`PrimaryStep::after`], [`FallbackStep::after`]), and so the file-level
/// error it may become keeps the worker's own category. It used to be a
/// `ServerError::Validation` string the transport searched for
/// `"RuntimeFailure:"` and `"ModelUnavailable: no whisper FA host loaded"`,
/// so the decision depended on how the message happened to be worded, and
/// a file that did fail was reported to the operator as bad input.
#[derive(Debug, thiserror::Error)]
pub(super) enum FaUnitFailure {
    /// The worker answered with a failure, classified by its protocol code.
    #[error("FA request {request} (group {group}): {failure}")]
    Worker {
        request: usize,
        group: usize,
        failure: FaWorkerFailure,
    },
    /// The response did not admit as evidence for this request: another
    /// engine's shape, another word count, another task's result.
    #[error("failed to admit worker protocol V2 FA evidence for request {request}: {source}")]
    Evidence {
        request: usize,
        source: FaRawEvidenceError,
    },
    /// The admitted response could not be lowered into this request's
    /// timings.
    #[error(
        "failed to parse worker protocol V2 FA response for request {request} (group {group}) \
         ({start_ms}..{end_ms} ms): {source}"
    )]
    Response {
        request: usize,
        group: usize,
        start_ms: u64,
        end_ms: u64,
        source: FaAlignmentError,
    },
}

impl FaUnitFailure {
    fn worker(unit: &DispatchUnit<'_>, failure: FaWorkerFailure) -> Self {
        Self::Worker {
            request: unit.ordinal().index(),
            group: unit.group().index(),
            failure,
        }
    }

    /// The worker's own report, when that is what this failure is.
    fn worker_failure(&self) -> Option<&FaWorkerFailure> {
        match self {
            Self::Worker { failure, .. } => Some(failure),
            Self::Evidence { .. } | Self::Response { .. } => None,
        }
    }
}

impl From<FaUnitFailure> for ServerError {
    /// A worker-reported failure keeps the category its protocol code names
    /// (a runtime failure, a model that would not load, a refused request);
    /// a response that does not fit its request is the worker breaking the
    /// protocol. Neither is invalid client input.
    fn from(failure: FaUnitFailure) -> Self {
        match failure {
            FaUnitFailure::Worker {
                request,
                group,
                failure,
            } => failure
                .into_report()
                .in_context(format_args!("FA request {request} (group {group})"))
                // Every forced-alignment engine runs on this machine.
                .into_server_error(crate::worker::execute_result_v2::ModelLocality::Local),
            FaUnitFailure::Evidence { .. } | FaUnitFailure::Response { .. } => ServerError::Worker(
                crate::worker::error::WorkerError::Protocol(failure.to_string()),
            ),
        }
    }
}

/// What a request whose primary engine failed does next.
///
/// Decided by [`Self::after`] from the typed failure alone. A retry exists
/// only here: the fallback's own failure has [`FallbackStep`], which cannot
/// express one, so a retry cannot chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrimaryStep {
    /// Re-dispatch the request on the engine row's fallback target.
    Retry(FaGroupRetry),
    /// The model failed on this request's own input: leave its words
    /// unaligned and continue the file.
    LeaveUnaligned,
    /// Not this request's input: fail the file with the failure's category.
    FailFile,
}

impl PrimaryStep {
    fn after(engine: crate::types::engines::FaEngineName, failure: &FaUnitFailure) -> Self {
        match failure.worker_failure().map(FaWorkerFailure::class) {
            // The policy is a FIELD on the engine's row in `FA_ENGINES`, not a
            // match over the roster here: a new engine cannot compile without
            // stating it, beside everything else it is.
            Some(FaFailureClass::EngineConstraint(constraint)) => {
                match engine.fallback_policy().retry_target() {
                    Some(target) => Self::Retry(FaGroupRetry { target, constraint }),
                    // A constraint the engine has no fallback for is still a
                    // failure on this request's input.
                    None => Self::LeaveUnaligned,
                }
            }
            // Missing models, refused requests, unclassified runtime failures
            // and malformed responses reach the file policy. That policy
            // distinguishes bounded retries from typed permanent refusals.
            Some(FaFailureClass::ModelUnavailable | FaFailureClass::FileFailure) | None => {
                Self::FailFile
            }
        }
    }
}

/// What a request whose fallback engine also failed does next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FallbackStep {
    /// Leave the request's words unaligned and continue the file.
    LeaveUnaligned,
    /// Fail the file with the failure's category.
    FailFile,
}

impl FallbackStep {
    fn after(failure: &FaUnitFailure) -> Self {
        match failure.worker_failure().map(FaWorkerFailure::class) {
            // The fallback failed on this input too (still too long for its
            // own context), or the worker that took it has no model for it: a
            // capability gap, not a data error. Either way the surrounding
            // requests keep their timing rather than the file failing.
            Some(FaFailureClass::EngineConstraint(_) | FaFailureClass::ModelUnavailable) => {
                Self::LeaveUnaligned
            }
            Some(FaFailureClass::FileFailure) | None => Self::FailFile,
        }
    }
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

/// A recoverable FA failure and where the group goes next.
///
/// One value rather than a bare reason, because the reason and the engine the
/// retry runs on are one decision: the reason comes from the failure, the
/// target comes from the failing engine's own row, and a caller holding only
/// the reason has to name the target itself. It did, as a literal `Whisper`,
/// and the warn line beside it said "Wave2Vec FA" for every retry, so a
/// Cantonese group reported the wrong engine to whoever read the log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FaGroupRetry {
    /// The engine the group is re-dispatched to.
    ///
    /// Read from [`FaFallbackPolicy::RetryGroupOn`], never named here. That
    /// row's target is const-checked to be language-general, which is what
    /// makes re-dispatching without a second language admission sound.
    ///
    /// [`FaFallbackPolicy::RetryGroupOn`]: crate::types::engines::FaFallbackPolicy::RetryGroupOn
    target: crate::types::engines::FaEngineName,
    /// The constraint the failing engine reported. Its recorded reason
    /// (`FaEngineConstraint::reason`) is rendered only where the warn line,
    /// the fallback event and the fallback evidence are written.
    constraint: FaEngineConstraint,
}

/// Build one production-domain `FaInferItem` for one request.
fn build_fa_infer_item(batch: &FaWorkerBatch<'_>, unit: &DispatchUnit<'_>) -> FaInferItem {
    let words = unit.words();
    FaInferItem {
        words: unit.texts(),
        word_ids: words.iter().map(|word| word.stable_id()).collect(),
        word_utterance_indices: words
            .iter()
            .map(|word| word.utterance_index.raw())
            .collect(),
        word_utterance_word_indices: words
            .iter()
            .map(|word| word.utterance_word_index.raw())
            .collect(),
        audio_path: batch.audio_path.to_string_lossy().into_owned(),
        // The unit's own window, proof included: nothing is lowered to
        // integers here for the request builder to rebuild.
        window: unit.window(),
        whisper_text: batch.whisper_text,
    }
}

/// Build unique request and artifact ids for one FA V2 request.
///
/// The request namespace is allocated once per `infer_units_v2` call so two
/// concurrent files cannot collide on `fa-v2-request-0`, `fa-v2-request-1`,
/// and so on while sharing the same GPU worker.
fn build_fa_request_ids(request_namespace: u64, unit: UnitOrdinal) -> PreparedFaRequestIdsV2 {
    let unit = unit.index();
    PreparedFaRequestIdsV2::new(
        format!("fa-v2-request-{request_namespace}-{unit}"),
        format!("fa-v2-payload-{request_namespace}-{unit}"),
        format!("fa-v2-audio-{request_namespace}-{unit}"),
    )
}

#[cfg(test)]
mod tests {
    use crate::chat_ops::fa::{FaGroup, FaWord, TimeSpan};
    use crate::chat_ops::{UtteranceIdx, WordIdx};
    use crate::fa::units::test_support::{anchored_group, plan_for};

    use super::*;
    use crate::api::NonNegativeSeconds;
    use crate::types::worker_v2::{
        ExecuteResponseV2, ProtocolErrorCodeV2, TaskResultV2, TranslationItemResultV2,
        TranslationResultV2, WorkerRequestIdV2,
    };
    use crate::worker::error::WorkerError;

    /// Build a small FA word for transport unit tests.
    fn make_word(index: usize, text: &str) -> FaWord {
        FaWord {
            utterance_index: UtteranceIdx::new(0),
            utterance_word_index: WordIdx::new(index),
            text: text.into(),
        }
    }

    /// One single group per window, for plans with several requests.
    fn single_groups(windows: &[(u64, u64)]) -> Vec<FaGroup> {
        windows
            .iter()
            .enumerate()
            .map(|(utterance, &(start, end))| {
                FaGroup::test_fixture(
                    TimeSpan::new(start, end),
                    vec![FaWord {
                        utterance_index: UtteranceIdx::new(utterance),
                        utterance_word_index: WordIdx::new(0),
                        text: "hello".into(),
                    }],
                    vec![UtteranceIdx::new(utterance)],
                )
            })
            .collect()
    }

    #[test]
    fn worker_outcomes_serialize_distinct_evidence_provenance() {
        // A failed call and a successful response that aligned no words have
        // the same timing projection. The outcome, not those timings, owns
        // the distinction recorded in the evidence artifact.
        let groups = vec![FaGroup::test_fixture(
            TimeSpan::new(100, 900),
            vec![make_word(0, "hello"), make_word(1, "world")],
            vec![UtteranceIdx::new(0)],
        )];
        let plan = plan_for(&groups);
        let unit = plan.units().next().expect("one request");
        let outcomes = [
            (
                FaWorkerUnitResult::Unaligned(FaWorkerUnalignedResult { unit }),
                "unaligned",
            ),
            (
                FaWorkerUnitResult::Evidence(Box::new(FaWorkerEvidenceResult {
                    unit,
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
            assert_eq!(projection.unit.ordinal(), unit.ordinal());
            assert_eq!(projection.timings, vec![None; 2]);
        }
    }

    fn batch<'a>(misses: Vec<&'a DispatchUnit<'a>>) -> FaWorkerBatch<'a> {
        let authorization = match plan_fa_inference(CachePolicy::UseCache, misses)
            .expect("ordinary cache misses may infer")
        {
            FaInferencePlan::Authorized(authorization) => authorization,
            FaInferencePlan::NothingToInfer => panic!("one miss must require inference"),
        };
        FaWorkerBatch {
            authorization,
            audio_path: Path::new("/tmp/input.wav"),
            worker_lang: WorkerLanguage::from(crate::api::LanguageCode3::eng()),
            engine: crate::types::engines::FaEngineName::Whisper,
            whisper_text: WhisperTextMode::CharSpaced,
        }
    }

    #[test]
    fn builds_fa_infer_item_from_one_request() {
        let groups = vec![FaGroup::test_fixture(
            TimeSpan::new(100, 900),
            vec![make_word(0, "hello"), make_word(1, "world")],
            vec![UtteranceIdx::new(0)],
        )];
        let plan = plan_for(&groups);
        let unit = plan.units().next().expect("one request");
        let batch = batch(vec![unit]);

        let item = build_fa_infer_item(&batch, unit);
        assert_eq!(item.words, vec!["hello".to_string(), "world".to_string()]);
        assert_eq!(
            item.word_ids,
            vec!["u0:w0".to_string(), "u0:w1".to_string()]
        );
        assert_eq!(item.word_utterance_indices, vec![0, 0]);
        assert_eq!(item.word_utterance_word_indices, vec![0, 1]);
        assert_eq!(item.audio_path, "/tmp/input.wav");
        assert_eq!(item.window.audio_start().get(), 100);
        assert_eq!(item.window.end().get(), 900);
        assert_eq!(item.whisper_text, WhisperTextMode::CharSpaced);
    }

    /// A piece of an anchored group is requested with only its own words
    /// and its own window: the request is the piece, not the utterance.
    #[test]
    fn an_anchored_piece_is_requested_with_its_own_words_and_window() {
        let groups = vec![anchored_group()];
        let plan = plan_for(&groups);
        let piece = plan.units().nth(1).expect("three pieces");
        let batch = batch(vec![piece]);

        let item = build_fa_infer_item(&batch, piece);
        assert_eq!(item.words, vec!["w2".to_string(), "w3".to_string()]);
        assert_eq!(item.word_utterance_word_indices, vec![2, 3]);
        assert_eq!(item.window.audio_start().get(), 13_000);
        assert_eq!(item.window.end().get(), 26_000);
    }

    #[test]
    fn required_cache_misses_never_produce_fa_inference_authorization() {
        let groups = single_groups(&[(0, 100), (100, 200), (200, 300), (300, 400)]);
        let plan = plan_for(&groups);
        let units: Vec<&DispatchUnit<'_>> = plan.units().collect();
        let error = plan_fa_inference(CachePolicy::RequireCache, vec![units[1], units[3]])
            .expect_err("required evidence misses must refuse worker inference");

        let ServerError::RequiredEvidenceUnavailable(missing) = error else {
            panic!("cache precondition must have a typed refusal");
        };
        let MissingRequiredEvidence::ForcedAlignment(missing) = missing else {
            panic!("expected forced-alignment evidence refusal");
        };
        assert_eq!(missing.request_indices(), &[1, 3]);
    }

    #[test]
    fn reusable_fa_evidence_needs_no_authorization_even_when_required() {
        let plan = plan_fa_inference(CachePolicy::RequireCache, Vec::new())
            .expect("no misses satisfy required-cache policy");

        assert!(matches!(plan, FaInferencePlan::NothingToInfer));
    }

    #[test]
    fn builds_namespaced_v2_request_ids_from_request_ordinal() {
        let ids = build_fa_request_ids(42, UnitOrdinal::fixture(7));
        assert_eq!(&*ids.request_id, "fa-v2-request-42-7");
        assert_eq!(&*ids.payload_ref_id, "fa-v2-payload-42-7");
        assert_eq!(&*ids.audio_ref_id, "fa-v2-audio-42-7");
    }

    #[test]
    fn namespaces_v2_request_ids_across_concurrent_files() {
        let first = build_fa_request_ids(1, UnitOrdinal::fixture(0));
        let second = build_fa_request_ids(2, UnitOrdinal::fixture(0));

        assert_ne!(first.request_id, second.request_id);
        assert_ne!(first.payload_ref_id, second.payload_ref_id);
        assert_ne!(first.audio_ref_id, second.audio_ref_id);
    }

    /// A response of another task's kind is refused where the response is
    /// read, once, as evidence for no request; it never reaches alignment
    /// (which used to read the response a second time and carried arms for
    /// a failure and a wrong kind it could no longer meet).
    #[test]
    fn a_non_fa_payload_is_refused_at_admission_with_request_context() {
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("req-fa-v2-bad"),
            TaskResultV2::TranslationResult(TranslationResultV2 {
                items: vec![TranslationItemResultV2::BlankInput],
            }),
            NonNegativeSeconds::try_from(0.01).expect("fixture elapsed"),
        );

        let error = admitted_failure(crate::types::engines::FaEngineName::Wave2Vec, &response);

        assert!(
            matches!(
                error,
                FaUnitFailure::Evidence {
                    request: 0,
                    source: FaRawEvidenceError::UnexpectedResult,
                }
            ),
            "{error}"
        );
    }

    /// The real reader: one worker response through the admission every
    /// request's response takes, for a request on `engine`.
    fn admitted_failure(
        engine: crate::types::engines::FaEngineName,
        response: &ExecuteResponseV2,
    ) -> FaUnitFailure {
        let groups = single_groups(&[(100, 900)]);
        let plan = plan_for(&groups);
        let unit = plan.units().next().expect("one request");
        let namespace = crate::engine_reports::FaCacheNamespace::for_test("test-fa");
        let admission = FaUnitEvidenceAdmission {
            requested_engine: engine,
            cache_namespace: &namespace,
            unit,
        };
        match admission.admit(response, FaEvidenceRoute::Direct) {
            Ok(_) => panic!("a failed response must not admit"),
            Err(failure) => failure,
        }
    }

    fn failed(code: ProtocolErrorCodeV2, message: &str) -> ExecuteResponseV2 {
        ExecuteResponseV2::failure(
            WorkerRequestIdV2::from("req-fa-v2-failed"),
            code,
            message.to_owned(),
            NonNegativeSeconds::try_from(0.01).expect("fixture elapsed"),
        )
    }

    /// RED FIRST: the retry was chosen by searching a rendered
    /// `ServerError::Validation` string. It is now chosen from the
    /// constraint the response reader classified, and carries the same
    /// recorded reasons as before.
    #[test]
    fn known_engine_constraints_retry_on_the_rows_fallback_engine() {
        use crate::types::engines::FaEngineName;
        for (message, constraint, reason) in [
            (
                "targets length is too long for CTC",
                FaEngineConstraint::CtcTargetsTooLong,
                "targets length is too long for CTC",
            ),
            (
                "ValueError: targets Tensor shouldn't contain blank index. Found tensor([[20, 5]])",
                FaEngineConstraint::CtcBlankIndexInTargets,
                "targets Tensor shouldn't contain blank index",
            ),
            (
                "RuntimeError: Calculated padded input size per channel: (1). Kernel size: (2). \
                 Kernel size can't be greater than actual input size",
                FaEngineConstraint::InputShorterThanKernel,
                "audio segment too short for Wave2Vec feature extractor",
            ),
        ] {
            // The recorded reason is the vocabulary fallback evidence has
            // always carried.
            assert_eq!(constraint.reason(), reason);
            let response = failed(ProtocolErrorCodeV2::RuntimeFailure, message);
            assert_eq!(
                PrimaryStep::after(
                    FaEngineName::Wave2Vec,
                    &admitted_failure(FaEngineName::Wave2Vec, &response)
                ),
                PrimaryStep::Retry(FaGroupRetry {
                    target: FaEngineName::Whisper,
                    constraint,
                }),
                "{message}"
            );
            // Whisper is the fallback target and has no retry of its own: the
            // constraint is still a failure on this request's input.
            assert_eq!(
                PrimaryStep::after(
                    FaEngineName::Whisper,
                    &admitted_failure(FaEngineName::Whisper, &response)
                ),
                PrimaryStep::LeaveUnaligned,
                "{message}"
            );
        }
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
    fn expected_step(engine: crate::types::engines::FaEngineName) -> PrimaryStep {
        use crate::types::engines::FaEngineName;
        const RECOVERABLE: FaEngineConstraint = FaEngineConstraint::CtcTargetsTooLong;
        match engine {
            FaEngineName::Wave2Vec => PrimaryStep::Retry(FaGroupRetry {
                target: FaEngineName::Whisper,
                constraint: RECOVERABLE,
            }),
            FaEngineName::Wav2vecCanto => PrimaryStep::Retry(FaGroupRetry {
                target: FaEngineName::Whisper,
                constraint: RECOVERABLE,
            }),
            // It IS the fallback target and cannot retry itself.
            FaEngineName::Whisper => PrimaryStep::LeaveUnaligned,
            // Its aligner is not a CTC decoder, so no recoverable CTC
            // constraint is reachable for it, and a whole-group retry would
            // replace measured Qwen3 timings with another model's.
            FaEngineName::Qwen3 => PrimaryStep::LeaveUnaligned,
        }
    }

    /// Every engine offers exactly the step written out above, on a failure
    /// all of them could see.
    ///
    /// A ROUNDTRIP between the declaration table and the transport that reads
    /// it, which no signature pins: the row says "retry on Whisper", and only
    /// running the function shows that a Whisper request is what comes back.
    #[test]
    fn every_fa_engine_falls_back_exactly_as_its_row_declares() {
        use crate::types::engines::{FaEngineName, SelectableEngine};

        let recoverable = failed(
            ProtocolErrorCodeV2::RuntimeFailure,
            "targets length is too long for CTC",
        );
        for engine in FaEngineName::ALL.iter().copied() {
            assert_eq!(
                PrimaryStep::after(engine, &admitted_failure(engine, &recoverable)),
                expected_step(engine),
                "{engine:?} did not offer the step this test writes out for it"
            );
        }
    }

    /// An unclassified wire failure reaches the file policy on either engine.
    /// Its wording cannot establish OOM or a permanent input defect.
    #[test]
    fn a_runtime_failure_reaches_bounded_file_retry() {
        use crate::types::engines::FaEngineName;
        for message in [
            "ValueError: Labels' sequence length 2043 cannot exceed the maximum allowed \
             length of 448 tokens.",
            "RuntimeError: Expected all tensors to be on the same device",
            "RuntimeError: MPS backend out of memory",
            "unclassified local failure",
        ] {
            let failure = admitted_failure(
                FaEngineName::Wave2Vec,
                &failed(ProtocolErrorCodeV2::RuntimeFailure, message),
            );
            assert_eq!(
                PrimaryStep::after(FaEngineName::Wave2Vec, &failure),
                PrimaryStep::FailFile
            );
            assert_eq!(FallbackStep::after(&failure), FallbackStep::FailFile);
            let error = ServerError::from(failure);
            let category = crate::runner::util::classify_server_error(&error);
            assert_eq!(category, crate::scheduling::FailureCategory::LocalRuntime);
            assert!(error.to_string().contains(message));
            let policy = crate::scheduling::RetryPolicy::default();
            assert!(matches!(
                policy.after_failure(1, category),
                crate::scheduling::NextAttempt::RetryAfter(_)
            ));
            assert_eq!(
                policy.after_failure(policy.max_attempts, category),
                crate::scheduling::NextAttempt::Stop(crate::scheduling::StopReason::BudgetSpent)
            );
        }
    }

    /// The user bug of 2026-04-08: a fallback taken by a worker with no
    /// fallback model loaded failed the whole file. A missing fallback model
    /// is a capability gap and leaves the request unaligned; a missing
    /// PRIMARY model would meet every request, so it fails the file, with
    /// the bootstrap category its protocol code names.
    #[test]
    fn a_missing_model_is_a_gap_on_the_fallback_and_a_file_failure_on_the_primary() {
        use crate::types::engines::FaEngineName;
        let failure = admitted_failure(
            FaEngineName::Whisper,
            &failed(
                ProtocolErrorCodeV2::ModelUnavailable,
                "no whisper FA host loaded for worker protocol V2",
            ),
        );
        assert_eq!(FallbackStep::after(&failure), FallbackStep::LeaveUnaligned);
        assert_eq!(
            PrimaryStep::after(FaEngineName::Whisper, &failure),
            PrimaryStep::FailFile
        );
        assert!(matches!(
            ServerError::from(failure),
            ServerError::Worker(WorkerError::Bootstrap { .. })
        ));
    }

    /// RED FIRST: every FA failure that reached the file level was
    /// `ServerError::Validation`, which the dashboard reports as bad input.
    /// A worker failure keeps its protocol category, and a response that
    /// does not fit its request is a protocol fault.
    #[test]
    fn a_failed_request_reaches_the_file_with_its_own_category() {
        use crate::runner::util::classify_server_error;
        use crate::scheduling::FailureCategory;
        use crate::types::engines::FaEngineName;

        let refused = admitted_failure(
            FaEngineName::Wave2Vec,
            &failed(ProtocolErrorCodeV2::InvalidPayload, "bad request"),
        );
        assert_eq!(
            PrimaryStep::after(FaEngineName::Wave2Vec, &refused),
            PrimaryStep::FailFile
        );
        assert_eq!(FallbackStep::after(&refused), FallbackStep::FailFile);
        assert_eq!(
            classify_server_error(&ServerError::from(refused)),
            FailureCategory::WorkerProtocol
        );

        let denied = admitted_failure(
            FaEngineName::Wave2Vec,
            &failed(ProtocolErrorCodeV2::ModelAccessDenied, "gated repository"),
        );
        assert_eq!(
            classify_server_error(&ServerError::from(denied)),
            FailureCategory::ModelAccessDenied
        );

        let wrong_task = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("req-fa-v2-wrong-task"),
            TaskResultV2::TranslationResult(TranslationResultV2 {
                items: vec![TranslationItemResultV2::BlankInput],
            }),
            NonNegativeSeconds::try_from(0.01).expect("fixture elapsed"),
        );
        let protocol = admitted_failure(FaEngineName::Wave2Vec, &wrong_task);
        assert_eq!(
            PrimaryStep::after(FaEngineName::Wave2Vec, &protocol),
            PrimaryStep::FailFile
        );
        assert_eq!(
            classify_server_error(&ServerError::from(protocol)),
            FailureCategory::WorkerProtocol
        );
    }

    /// Worker process crashes (SIGKILL, C-extension SIGSEGV) must be detectable
    /// as a group-local signal so `infer_units_v2` can leave the crashing group
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
        let groups = vec![FaGroup::test_fixture(
            TimeSpan::new(175_765, 176_365),
            vec![make_word(0, "hello")],
            vec![UtteranceIdx::new(0)],
        )];
        let plan = plan_for(&groups);

        let event = build_fallback_event(
            plan.units().next().expect("one request"),
            crate::types::engines::FaEngineName::Wave2Vec,
            crate::types::engines::FaEngineName::Whisper,
            FaEngineConstraint::CtcTargetsTooLong,
        );

        assert_eq!(event.group_index, 0);
        assert_eq!(event.from_engine, "wav2vec_fa");
        assert_eq!(event.to_engine, "whisper_fa");
        assert_eq!(event.reason, "targets length is too long for CTC");
        assert_eq!(event.audio_start_ms.0, 175_765);
        assert_eq!(event.audio_end_ms.0, 176_365);
    }

    /// A fallback on one piece reports the piece's window, which is the
    /// audio the fallback engine was actually given, under its group.
    #[test]
    fn a_piece_fallback_reports_the_piece_window() {
        let groups = vec![
            FaGroup::test_fixture(
                TimeSpan::new(0, 900),
                vec![make_word(0, "hello")],
                vec![UtteranceIdx::new(0)],
            ),
            anchored_group(),
        ];
        let plan = plan_for(&groups);
        let event = build_fallback_event(
            plan.units().nth(3).expect("one request and three pieces"),
            crate::types::engines::FaEngineName::Wave2Vec,
            crate::types::engines::FaEngineName::Whisper,
            FaEngineConstraint::CtcTargetsTooLong,
        );
        assert_eq!(event.group_index, 1);
        assert_eq!(event.audio_start_ms.0, 26_000);
        assert_eq!(event.audio_end_ms.0, 40_000);
    }
}
