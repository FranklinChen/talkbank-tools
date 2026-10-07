//! Server-side forced alignment orchestrator.
//!
//! Owns the full CHAT lifecycle for FA jobs:
//! parse → group → cache check → infer (audio chunks) → DP-align → inject →
//! postprocess → %wor → monotonicity/E704 → serialize.
//!
//! # Call path
//!
//! `batchalign-cli`/API submission
//! → `runner::dispatch_fa_infer`
//! → [`run_fa_from_ast`]
//! → `crate::chat_ops::fa::{group_utterances, parse_fa_response, apply_fa_results}`
//! → FA worker transport adapter
//! → validation + serialization.
//!
//! # Key differences from morphosyntax/utseg/translate/coref
//!
//! - **Per-file, not cross-file**: Each file has its own audio, so no cross-file batching.
//! - **Multiple groups per file**: Utterances are grouped by time window; each group is one
//!   infer item, except an over-budget utterance split at UTR word anchors, which is one group
//!   executed as one infer item per piece (see `units`).
//! - **Audio access**: Workers need the audio file path and time range, not just text.
//! - **DP alignment in Rust**: Model output is aligned to transcript words via Hirschberg.
//!
//! # Invariants for contributors
//!
//! - FA worker timestamps are chunk-relative; `parse_fa_response` must convert
//!   them to file-absolute ms through `FaWindow::to_file`.
//! - `apply_fa_results` ordering is load-bearing:
//!   inject → postprocess → utterance bullet update → `%wor` generation
//!   → monotonicity (E362) → same-speaker overlap enforcement (E704).
//! - Cache keys must include audio identity, time window, text, timing mode
//!   and engine; changing dimensions changes cache compatibility. They are per
//!   REQUEST (dispatch unit), so a single group keys exactly as it always did
//!   and each piece of an anchored group keys its own words and window.

mod completion;
mod input;
mod raw_evidence;
mod transport;
mod units;
use input::FaAdmission;
#[cfg(test)]
pub(crate) use input::read_fa_source;
pub(crate) use input::{
    FaWorkingDocument, OutstandingTiming, ReconciledOutput, read_fa_source_named,
};

use crate::cache::tasks::{FORCED_ALIGNMENT, FORCED_ALIGNMENT_RAW_EVIDENCE};
use crate::chat_ops::CacheKey;
use crate::chat_ops::fa::{
    BulletRepairPolicy, WordTiming, apply_fa_results_with_projection_policy,
    expand_bullets_for_edge_fillers, finalize_without_injection, find_reusable_utterance_indices,
    group_utterances, has_reusable_wor_timing, projection_without_injection_with_touched,
    refresh_reusable_alignment, refresh_reusable_utterances, rescue_narrow_bullets,
    strip_wor_from_monotonicity_stripped_utterances,
};
use crate::engine_reports::FaCacheNamespace;
use crate::params::{AudioContext, FaParams};
use crate::pipeline::PipelineServices;
use crate::pipeline::post_validate::PostValidated;
use batchalign_transform::parse::is_ca;
use tracing::info;

use crate::chat_ops::fa::Grouping;
use crate::error::ServerError;
use crate::runner::util::ProgressSender;
use crate::types::results::{FaOutput, FaResult};
use units::{DispatchUnit, FaDispatchInputs, resolve_group_timings};

/// What forced alignment runs with: the shared pool and cache, plus the
/// namespace every FA cache row and FA evidence envelope is written and
/// admitted under.
///
/// The namespace is the FA engine the selected worker reported, byte for byte,
/// so evidence cached by earlier runs stays admissible. It is a separate field
/// rather than a version on [`PipelineServices`] because it belongs to this
/// stage alone: the UTR pass that runs inside align shares the pool and cache
/// but caches under its own engine's namespace.
#[derive(Clone, Copy)]
pub(crate) struct FaServices<'a> {
    /// Worker pool and cache shared with every other stage.
    pub(crate) pipeline: PipelineServices<'a>,
    /// The FA engine the selected worker reported.
    pub(crate) cache_namespace: &'a FaCacheNamespace,
}

/// Source admission carried only by the owning active working document.
/// Every processed output nevertheless requires complete construction
/// admission in finish; the caller cannot choose its output validity level.
/// A no-op instead consumes strict, source-bound unchanged-byte admission.
impl FaAdmission {
    /// Return an unchanged, strictly admitted source.
    /// The source owns its exact bytes and model; neither a caller-selected
    /// validity level nor an independently supplied model can grant permission.
    pub(super) fn pass_through(
        source: batchalign_transform::AdmittedSourceChat<'_>,
        gap_healing: crate::chat_ops::fa::WordGapHealing,
        engine: &str,
        cache_namespace: &FaCacheNamespace,
    ) -> AdmittedFaResult {
        let chat_file = source.document().clone();
        let document = PostValidated::pass_through(source, crate::api::ReleasedCommand::Align);
        AdmittedFaResult {
            // Written out rather than hidden behind a `FaResult::pass_through`
            // constructor: that constructor was the hole this type exists to
            // close, and there is nothing left for it to be reused by.
            evidence: AdmittedFaEvidence::Preserved(FaResult {
                output: FaOutput::PassThrough(chat_file),
                group_evidence: Vec::new(),
                engine: engine.to_owned(),
                cache_namespace: cache_namespace.clone(),
                decisions: Vec::new(),
                timing_decisions: Vec::new(),
                gap_healing,
                fallback_events: Vec::new(),
            }),
            document,
        }
    }

    /// Gate a finished FA result and hand back the value the caller returns.
    ///
    /// Timing completeness is a fact about the result, never a reason to
    /// discard it: a result whose words are not all timed is admitted as
    /// [`AdmittedFaEvidence::Partial`], written with the timing it has, and
    /// reported through [`AdmittedFaResult::shortfalls`] so the file is
    /// diagnosed rather than clean. Only an output that fails CHAT admission,
    /// a changed lexical structure, or an unfulfilled source timing obligation
    /// refuses the file.
    ///
    /// Fail-closed: a file whose aligned output fails the gate fails THIS file
    /// (`OutputAdmission` classifies as `FailureCategory::System`),
    /// so `finalize_success` never runs and nothing is written. A refusal is
    /// therefore reported through the file's failure rather than through a
    /// trace nobody would read, which is why an `FaResult` carries no
    /// violations at all: the wire format's field is filled in, empty, where
    /// the trace is built.
    ///
    /// The gate's PROOF is kept, not dropped. It used to be discarded and the
    /// dispatch seam re-serialized `output.to_chat_string()` afterwards, so
    /// the L2-gated bytes were never the bytes written; the writer then
    /// manufactured a second, weaker proof of its own over text this one had
    /// never seen.
    /// It consumes this attempt's admission. Retry attempts clone the working
    /// state's obligation; each finished result still requires its own complete
    /// output proof. This token alone does not prove input/output identity.
    ///
    /// It takes the DRAFT: the media/timing transition runs here, against
    /// the declaration this source was admitted with, and nowhere else.
    pub(super) fn finish(
        self,
        draft: FaResult<crate::chat_ops::ChatFile>,
    ) -> Result<AdmittedFaResult, ServerError> {
        let result = self.reconcile_output(draft)?;
        self.require_output_timing(result.output.as_chat_file())?;
        let document = PostValidated::gate(
            result.output.as_chat_file(),
            crate::api::ReleasedCommand::Align,
        )
        .map_err(|failure| explain_gate_refusal(result.output.as_chat_file(), failure))?;
        let evidence = match self.complete_result(result)? {
            completion::FaCompletion::Complete(complete) => AdmittedFaEvidence::Complete(complete),
            completion::FaCompletion::Partial(partial) => {
                // The complete account, once; the record keeps a bounded copy.
                tracing::warn!(account = %partial.account(), "forced alignment left words untimed; writing the partial result");
                AdmittedFaEvidence::Partial(partial)
            }
        };
        if let Some(exclusions) = evidence.exclusions() {
            // The complete account, once; the record keeps a bounded copy.
            tracing::info!(%exclusions, "forced alignment left utterances out by design");
        }
        let document = self.discharge(document);
        Ok(AdmittedFaResult { evidence, document })
    }
}

/// Name the cause of an output refusal when it is a kept bullet on an
/// utterance not in the recording.
///
/// Under `--main-bullets keep` or `exact` such a bullet is written back
/// exactly as given, but it is never an anchor: nothing ordered the timing
/// aligned around it against it. If it then conflicts with that timing
/// (start order, or a same-speaker overlap), the output fails its gate. The
/// gate is the judge: the output is judged again without those bullets, and
/// only if THAT passes is the refusal theirs, reported as the input-and-policy
/// conflict it is, with the remedy, instead of as an internal fault.
fn explain_gate_refusal(
    output: &crate::chat_ops::ChatFile,
    failure: crate::pipeline::post_validate::PostValidationFailure,
) -> ServerError {
    let mut without = output.clone();
    let mut utterances = Vec::new();
    let mut ordinal = 0;
    for line in &mut without.lines {
        if let talkbank_model::model::Line::Utterance(utterance) = line {
            ordinal += 1;
            match crate::chat_ops::fa::RecordingPresence::of(utterance) {
                crate::chat_ops::fa::RecordingPresence::InRecording => {}
                crate::chat_ops::fa::RecordingPresence::NotInRecording(_) => {
                    if utterance.main.content.bullet.take().is_some() {
                        utterances.push(ordinal);
                    }
                }
            }
        }
    }
    if utterances.is_empty()
        || PostValidated::gate_owned(without, crate::api::ReleasedCommand::Align).is_err()
    {
        return failure.into_server_error();
    }
    ServerError::KeptOffRecordBulletConflict { utterances }
}

/// A finished FA run whose CHAT bytes are already proven writable.
///
/// The phase type that closes the loop the admission opens: an `FaResult` is a
/// document plus its evidence, and this is that pair AFTER
/// [`FaAdmission::finish`] gated it or [`FaAdmission::pass_through`] declared
/// it untouched. Its fields are private to this module, so the only values of
/// this type in the program are the ones those two functions returned.
pub(crate) struct AdmittedFaResult {
    /// The run's evidence, for the dashboard timeline.
    evidence: AdmittedFaEvidence,
    /// The bytes the writer may persist, and the proof that it may.
    document: PostValidated,
}

/// An admitted alignment's output, with everything the writer must report
/// beside it. The two reports are separate channels: a shortfall diagnoses
/// the file, an exclusion is information and does not.
pub(crate) struct AlignedOutput {
    /// The proven bytes.
    pub(crate) document: PostValidated,
    /// Requested work the document does not carry: words owed timing that
    /// have none, bounded.
    pub(crate) shortfalls: Vec<crate::pipeline::post_validate::Shortfall>,
    /// Utterances left out of alignment on purpose, bounded.
    pub(crate) exclusions: Vec<crate::pipeline::post_validate::Exclusion>,
}

/// Declined alignment retains source evidence; actual timing work must retain
/// its producer-issued completion payload, not a caller-selected success flag.
enum AdmittedFaEvidence {
    Preserved(FaResult),
    Complete(completion::CompleteFaResult),
    /// Written with the timing it has; the account is the file's shortfall.
    Partial(completion::PartialFaResult),
}

impl AdmittedFaEvidence {
    fn into_timeline_trace(self) -> crate::types::traces::FaTimelineTrace {
        match self {
            Self::Preserved(result) => result.into_timeline_trace(),
            Self::Complete(complete) => complete.into_timeline_trace(),
            Self::Partial(partial) => partial.into_timeline_trace(),
        }
    }
}

impl AdmittedFaEvidence {
    /// Requested work the written document does not carry: the untimed
    /// words of a partial result, bounded.
    fn shortfalls(&self) -> Vec<crate::pipeline::post_validate::Shortfall> {
        match self {
            Self::Preserved(_) | Self::Complete(_) => Vec::new(),
            Self::Partial(partial) => vec![partial.account().record()],
        }
    }

    /// The utterances alignment left out on purpose. A preserved source was
    /// not aligned, so it left nothing out; a measured result carries the
    /// account its admission made.
    fn exclusions(&self) -> Option<&completion::ExclusionAccount> {
        match self {
            Self::Preserved(_) => None,
            Self::Complete(complete) => complete.exclusions(),
            Self::Partial(partial) => partial.exclusions(),
        }
    }

    /// The bounded records of [`Self::exclusions`].
    fn exclusion_records(&self) -> Vec<crate::pipeline::post_validate::Exclusion> {
        self.exclusions()
            .map(completion::ExclusionAccount::record)
            .into_iter()
            .collect()
    }
}

impl AdmittedFaResult {
    /// Take the proven bytes with their reports, discarding the evidence
    /// timeline. The reports leave WITH the bytes, so a partial result
    /// cannot reach the writer without saying what it lacks, nor an
    /// exclusion without being listed.
    pub(crate) fn into_document(self) -> AlignedOutput {
        AlignedOutput {
            shortfalls: self.evidence.shortfalls(),
            exclusions: self.evidence.exclusion_records(),
            document: self.document,
        }
    }

    /// Split the proven bytes and their reports from the evidence timeline.
    ///
    /// Both leave together because a caller that wants the timeline still
    /// has to write the document, and handing out the timeline alone would
    /// leave the bytes with no route to disk.
    pub(crate) fn into_document_and_timeline(
        self,
    ) -> (AlignedOutput, crate::types::traces::FaTimelineTrace) {
        let output = AlignedOutput {
            shortfalls: self.evidence.shortfalls(),
            exclusions: self.evidence.exclusion_records(),
            document: self.document,
        };
        (output, self.evidence.into_timeline_trace())
    }
}

const FA_DERIVED_EVIDENCE_SCHEMA_VERSION: u8 = 1;

/// Persisted local timing projection with the request facts needed for replay.
///
/// The old cache stored a bare timing vector. It could not prove whether the
/// vector came from the requested engine or an unversioned fallback, so this
/// envelope deliberately invalidates that legacy shape.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct VersionedCachedFaTimings {
    schema_version: u8,
    requested_engine: crate::types::engines::FaEngineName,
    request_engine_version: crate::api::ReportedEngineName,
    expected_words: usize,
    cache_key: CacheKey,
    timings: Vec<Option<WordTiming>>,
}

#[derive(Debug, thiserror::Error)]
enum FaDerivedEvidenceError {
    #[error("forced-alignment derived evidence JSON is invalid: {0}")]
    InvalidJson(#[from] serde_json::Error),
    #[error("unsupported forced-alignment derived evidence schema version {0}")]
    SchemaVersion(u8),
    #[error("derived evidence requested {cached:?}, not current engine {current:?}")]
    EngineDrift {
        cached: crate::types::engines::FaEngineName,
        current: crate::types::engines::FaEngineName,
    },
    #[error("derived evidence worker version {cached} does not match current version {current}")]
    EngineVersionDrift {
        cached: crate::api::ReportedEngineName,
        current: crate::api::ReportedEngineName,
    },
    #[error("derived evidence belongs to a different semantic cache key")]
    CacheKeyDrift,
    #[error("derived evidence expected {cached} words, not current cardinality {current}")]
    ExpectedWordsDrift { cached: usize, current: usize },
    #[error("derived evidence contains {actual} timings for {expected} request words")]
    WordCardinality { expected: usize, actual: usize },
}

/// Cached timings proven to correspond one-to-one with a current FA group.
#[derive(Debug)]
struct AdmittedCachedFaTimings(Vec<Option<WordTiming>>);

impl AdmittedCachedFaTimings {
    fn encode_from_raw(
        timings: Vec<Option<WordTiming>>,
        raw: &raw_evidence::ReplayableFaRawEvidence,
    ) -> Result<serde_json::Value, FaDerivedEvidenceError> {
        let expected_words = raw.expected_words().get();
        if timings.len() != expected_words {
            return Err(FaDerivedEvidenceError::WordCardinality {
                expected: expected_words,
                actual: timings.len(),
            });
        }
        Ok(serde_json::to_value(VersionedCachedFaTimings {
            schema_version: FA_DERIVED_EVIDENCE_SCHEMA_VERSION,
            requested_engine: raw.requested_engine(),
            request_engine_version: raw.request_engine_version().name().clone(),
            expected_words,
            cache_key: raw.cache_key().clone(),
            timings,
        })?)
    }

    fn decode(
        value: serde_json::Value,
        requested_engine: crate::types::engines::FaEngineName,
        cache_namespace: &FaCacheNamespace,
        expected_words: usize,
        cache_key: &CacheKey,
    ) -> Result<Self, FaDerivedEvidenceError> {
        let cached: VersionedCachedFaTimings = serde_json::from_value(value)?;
        if cached.schema_version != FA_DERIVED_EVIDENCE_SCHEMA_VERSION {
            return Err(FaDerivedEvidenceError::SchemaVersion(cached.schema_version));
        }
        if cached.requested_engine != requested_engine {
            return Err(FaDerivedEvidenceError::EngineDrift {
                cached: cached.requested_engine,
                current: requested_engine,
            });
        }
        if &cached.request_engine_version != cache_namespace.name() {
            return Err(FaDerivedEvidenceError::EngineVersionDrift {
                cached: cached.request_engine_version,
                current: cache_namespace.name().clone(),
            });
        }
        if &cached.cache_key != cache_key {
            return Err(FaDerivedEvidenceError::CacheKeyDrift);
        }
        if cached.expected_words != expected_words {
            return Err(FaDerivedEvidenceError::ExpectedWordsDrift {
                cached: cached.expected_words,
                current: expected_words,
            });
        }
        if cached.timings.len() != expected_words {
            return Err(FaDerivedEvidenceError::WordCardinality {
                expected: expected_words,
                actual: cached.timings.len(),
            });
        }
        Ok(Self(cached.timings))
    }

    fn into_timings(self) -> Vec<Option<WordTiming>> {
        self.0
    }
}

/// Re-admit and locally reparse immutable FA worker evidence for one request.
fn replay_cached_raw_evidence<'a>(
    value: serde_json::Value,
    engine: crate::types::engines::FaEngineName,
    cache_namespace: &FaCacheNamespace,
    unit: &'a DispatchUnit<'a>,
) -> Result<transport::FaWorkerEvidenceResult<'a>, ServerError> {
    let evidence = raw_evidence::ReplayableFaRawEvidence::decode(
        value,
        engine,
        cache_namespace,
        raw_evidence::ExpectedFaWords::new(unit.words().len()),
        unit.cache_key(),
    )
    .map_err(|error| {
        ServerError::Validation(format!(
            "cached raw FA evidence for request {} was refused: {error}",
            unit.ordinal()
        ))
    })?;
    transport::replay_unit_evidence(evidence, unit)
}

/// A cache layer whose value was present but could not be admitted.
#[derive(Debug)]
struct RefusedFaCacheLayer {
    layer: &'static str,
    error: String,
}

/// The admitted cache state for one current FA request.
///
/// Raw worker evidence is intentionally tried first. Replaying it through the
/// current Rust projection is what lets alignment-algorithm experiments reuse
/// model inference. The derived timing layer remains a compatibility and
/// resilience fallback when raw evidence is absent or corrupt.
#[derive(Debug)]
enum AdmittedFaCacheGroup<'a> {
    RawEvidence(Box<transport::FaWorkerEvidenceResult<'a>>),
    DerivedTimings(AdmittedCachedFaTimings),
    Miss,
}

/// Result of checking both cache layers for one current FA request.
#[derive(Debug)]
struct FaCacheResolution<'a> {
    admitted: AdmittedFaCacheGroup<'a>,
    refusals: Vec<RefusedFaCacheLayer>,
}

/// Capability binding one current FA request to the exact facts against
/// which both cache layers must be admitted.
///
/// The unit carries its own key, words and window, so raw and derived
/// candidates are admitted against one request.
struct FaCacheUnitAdmission<'a> {
    unit: &'a DispatchUnit<'a>,
    engine: crate::types::engines::FaEngineName,
    cache_namespace: &'a FaCacheNamespace,
}

impl<'a> FaCacheUnitAdmission<'a> {
    fn new(
        unit: &'a DispatchUnit<'a>,
        engine: crate::types::engines::FaEngineName,
        cache_namespace: &'a FaCacheNamespace,
    ) -> Self {
        Self {
            unit,
            engine,
            cache_namespace,
        }
    }

    fn resolve(
        &self,
        raw_value: Option<&serde_json::Value>,
        derived_value: Option<&serde_json::Value>,
    ) -> FaCacheResolution<'a> {
        let mut refusals = Vec::new();

        if let Some(value) = raw_value {
            match replay_cached_raw_evidence(
                value.clone(),
                self.engine,
                self.cache_namespace,
                self.unit,
            ) {
                Ok(evidence) => {
                    return FaCacheResolution {
                        admitted: AdmittedFaCacheGroup::RawEvidence(Box::new(evidence)),
                        refusals,
                    };
                }
                Err(error) => refusals.push(RefusedFaCacheLayer {
                    layer: FORCED_ALIGNMENT_RAW_EVIDENCE.name().as_str(),
                    error: error.to_string(),
                }),
            }
        }

        if let Some(value) = derived_value {
            match AdmittedCachedFaTimings::decode(
                value.clone(),
                self.engine,
                self.cache_namespace,
                self.unit.words().len(),
                self.unit.cache_key(),
            ) {
                Ok(timings) => {
                    return FaCacheResolution {
                        admitted: AdmittedFaCacheGroup::DerivedTimings(timings),
                        refusals,
                    };
                }
                Err(error) => refusals.push(RefusedFaCacheLayer {
                    layer: FORCED_ALIGNMENT.name().as_str(),
                    error: error.to_string(),
                }),
            }
        }

        FaCacheResolution {
            admitted: AdmittedFaCacheGroup::Miss,
            refusals,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-file FA processing
// ---------------------------------------------------------------------------

/// An attempt produced only by the source-admitted working document.
pub(crate) enum FaInputDocument<'a> {
    Preserved(batchalign_transform::AdmittedSourceChat<'static>),
    Active(ActiveFaInput<'a>),
}

/// Active attempt: the producer's admission travels with its mutation target.
/// There is no free constructor accepting unrelated model and parse evidence.
pub(crate) struct ActiveFaInput<'a> {
    chat_file: crate::chat_ops::ChatFile,
    admission: FaAdmission,
    /// The run's main-bullet policy, bound to the input's bullets at the same
    /// parse, before the UTR pre-pass could write any (see
    /// `chat_ops::fa::main_bullets`).
    main_bullets: crate::chat_ops::fa::MainBulletAuthority,
    /// Word anchors the UTR pre-pass observed on THIS model's words, or
    /// [`crate::chat_ops::fa::AnchorIndex::not_observed`] when it did not run.
    /// Carried with the model rather than beside it because they describe its
    /// words: grouping cuts an over-budget utterance only where they say a
    /// word was heard ending. Borrowed: every attempt at the file reads the
    /// same anchors.
    anchors: &'a crate::chat_ops::fa::AnchorIndex,
}

/// Lower a kept-bullet failure to the error that fails this file.
///
/// Every variant is an internal invariant failure (a kept bullet did not
/// hold, or could not be attributed), so the file fails rather than being
/// written with a moved bullet.
pub(super) fn kept_bullets_failed(error: crate::chat_ops::fa::KeptBulletError) -> ServerError {
    ServerError::Validation(error.to_string())
}

/// Run forced alignment on a pre-parsed `ChatFile`.
///
/// THE FA entry point, and the only one. There used to be a `process_fa(&str)`
/// beside it that parsed a string and delegated here; its last caller was the
/// incremental path, which held the model all along and serialized it so that
/// this function could parse it back.
///
/// Returns a structured [`FaResult`] containing a reconciled CHAT output
/// state, group info, timing data, and validation results. The runner owns the
/// sole serialization boundary and decides which evidence to persist.
///
/// Algorithm outline:
/// 1. Consume the already source-admitted active or preserved disposition.
/// 2. Group utterances into FA windows.
/// 3. Resolve cache hits/misses per group.
/// 4. Send miss groups through the FA worker transport adapter.
/// 5. Parse responses and align to transcript words in Rust.
/// 6. Apply timings + postprocessing (`apply_fa_results`).
/// 7. Hand the draft to `FaAdmission::finish`: media/timing transition, then
///    full post-validation.
pub(crate) async fn run_fa_from_ast(
    document: FaInputDocument<'_>,
    audio: &AudioContext<'_>,
    worker_lang: &crate::api::LanguageCode3,
    services: FaServices<'_>,
    fa_params: &FaParams,
    progress: Option<&ProgressSender>,
) -> Result<AdmittedFaResult, ServerError> {
    let active = match document {
        FaInputDocument::Preserved(source) => {
            return Ok(FaAdmission::pass_through(
                source,
                fa_params.gap_healing,
                fa_params.engine.as_wire_name(),
                services.cache_namespace,
            ));
        }
        FaInputDocument::Active(active) => active,
    };
    let ActiveFaInput {
        mut chat_file,
        admission,
        main_bullets,
        anchors,
    } = active;
    // 1a′. Suppress %wor for Conversation Analysis transcripts.
    // CA transcripts (@Options: CA) use prosodic notation (⌈⌉⌊⌋, arrows,
    // lengthening marks) that %wor cannot represent. Generating %wor for
    // these files adds noise that CA researchers must manually remove.
    let write_wor = if is_ca(&chat_file) {
        info!("@Options: CA detected: suppressing %wor generation");
        false
    } else {
        fa_params.wor_tier.should_write()
    };

    // 1d'. Pair the projection policy with the main bullets bound at the
    // parse. Every finalization route below takes this one value.
    let projection =
        crate::chat_ops::fa::FaProjection::new(fa_params.projection_policy(), main_bullets);

    // 1e. Cheap rerun path: if the file already has complete, reusable `%wor`
    // timing, rebuild main-tier bullets and optionally regenerate `%wor`
    // without sending audio back through FA.
    if has_reusable_wor_timing(&chat_file) {
        info!("FA fast path: reusing existing %wor timing");
        // Mechanical refresh only (no `%wor` write): the touched utterances
        // are folded into the SAME `FaApplied` write phase that runs
        // monotonicity below, so `%wor` (when requested) is always written
        // after same-speaker overlaps are resolved, never before (2026-09-01
        // review, item 2).
        let touched = refresh_reusable_alignment(&mut chat_file, fa_params.existing_wor_boundaries);

        // A previous run may have written backward `%wor` timestamps (e.g.
        // APROCSA 2256_T4.cha: UTR anchor drift placed utterances from task N
        // into task N-1's audio window).  Without these two steps, every re-run
        // reconstructs the backward main-tier bullet from the stale `%wor`
        // data, and the E362 violation persists indefinitely.
        //
        // Step 1: strip backward main-tier bullets.
        let finalized = projection_without_injection_with_touched(projection, write_wor, touched)
            .then_finalize(
                &mut chat_file,
                BulletRepairPolicy::from(fa_params.bullet_repair),
            )
            .map_err(kept_bullets_failed)?;
        if fa_params.bullet_repair {
            tracing::info!(stats = %finalized.repair_stats(), "bullet repair applied");
        }
        // Step 2: remove `%wor` from stripped utterances so the next run goes
        // through full FA rather than reconstructing the backward bullet again.
        strip_wor_from_monotonicity_stripped_utterances(&mut chat_file, finalized.monotonicity());

        let written = crate::chat_ops::fa::retain_decision_evidence(
            &mut chat_file,
            crate::chat_ops::fa::FaDecisions::without_injection(Vec::new(), Vec::new(), finalized),
        );

        return admission.finish(
            FaResult::without_groups(
                chat_file,
                fa_params.gap_healing,
                fa_params.engine.as_wire_name(),
                services.cache_namespace,
            )
            .with_written_decisions(written),
        );
    }

    // 1f. Per-utterance partial reuse: when some (but not all) utterances have
    // clean %wor, refresh those and track them so their FA groups can be skipped.
    // `partially_reused_touched` is folded into the write phase around
    // `apply_fa_results_with_projection_policy` below via `also_touched`, so
    // their `%wor` is (re)written from the SAME post-monotonicity state as
    // everything that run injected fresh (2026-09-01 review, item 2).
    let mut partially_reused_touched: Vec<crate::chat_ops::UtteranceIdx> = Vec::new();
    let reusable_indices = find_reusable_utterance_indices(&chat_file);
    if !reusable_indices.is_empty() {
        info!(
            reusable = reusable_indices.len(),
            "FA partial reuse: refreshing utterances with clean %wor"
        );
        // This is a pre-grouping reconstruction, so it MUST preserve the
        // inherited main bullets. Rebuilding here changes group windows and
        // therefore raw-evidence cache keys. The explicit projection policy is
        // applied later, after evidence collection, to every group. Mechanical
        // refresh only: `%wor` is not written here; `partially_reused_touched`
        // is folded into the write phase around `apply_fa_results_with_projection_policy`
        // below via `FaApplied::also_touched`.
        partially_reused_touched = refresh_reusable_utterances(&mut chat_file, &reusable_indices);
    }

    // 2a. Rescue catastrophically narrow utterance bullets before grouping.
    //
    // When `transcribe` writes a bullet that is physically too narrow to
    // contain its words (e.g., 22 words in 380 ms = 58 wps, impossible),
    // FA cannot align the words against that audio range. Wave2Vec rejects
    // the group with "targets length is too long for CTC" because the
    // encoder produces too few frames for the target labels, and the
    // Whisper FA fallback path produces degenerate token-level timings
    // (zero-duration words, words past the bullet end). The user sees a
    // CHAT file with a `%wor` tier full of broken timings.
    //
    // The rescue pre-pass detects under-budgeted bullets and expands them
    // into the trailing inter-utterance gap, giving FA a wide-enough audio
    // window to find the actual speech. After FA finishes,
    // `update_utterance_bullet` overwrites the rescued range with the FA
    // word span (which is tighter), so the rescue is self-healing.
    //
    // Covered by the private regression fixture set under
    // `test-fixtures/align/regressions/` (gitignored; see
    // `book/src/batchalign/developer/regression-fixtures.md`).
    let rescue_decisions = rescue_narrow_bullets(&mut chat_file, projection.main_bullets())
        .map_err(kept_bullets_failed)?;

    // 2b. Expand utterance bullets to cover edge fillers in inter-utterance gaps.
    // UTR-assigned bullets may be too narrow to include trailing/leading fillers
    // whose audio lives in the gap between utterances.
    expand_bullets_for_edge_fillers(&mut chat_file);

    // Resolved once, here, and used for BOTH grouping and the containment
    // checks on what the engine returns. Grouping used to take an
    // `Option<u64>` and invent its own behaviour when it was absent; there is
    // one recording and one answer.
    let recording = audio.recording().await?;
    // 2c. Group utterances. An utterance over the engine budget is split at
    // the UTR anchors when they allow it; see `chat_ops::fa::split`.
    let (admission, grouping) = admission.admit_grouping(
        group_utterances(&chat_file, fa_params.max_group_ms().0, &recording, anchors),
        &chat_file,
    )?;
    let Grouping {
        groups,
        decisions: grouping_decisions,
        windows_clamped,
    } = grouping;

    if groups.is_empty() {
        // `partially_reused_touched` may be non-empty here too (1f refreshed
        // some utterances, but grouping still found nothing left to send to
        // FA workers): fold it in so its `%wor` is still written, once, after
        // monotonicity (2026-09-01 review, item 2). `finalize_without_injection`
        // stays the right call for the ordinary no-touched case (no
        // allocation for an empty `Vec` beyond what it already does).
        let finalized = if partially_reused_touched.is_empty() {
            finalize_without_injection(
                &mut chat_file,
                projection,
                BulletRepairPolicy::from(fa_params.bullet_repair),
            )
        } else {
            projection_without_injection_with_touched(
                projection,
                write_wor,
                partially_reused_touched,
            )
            .then_finalize(
                &mut chat_file,
                BulletRepairPolicy::from(fa_params.bullet_repair),
            )
        }
        .map_err(kept_bullets_failed)?;
        if fa_params.bullet_repair {
            tracing::info!(stats = %finalized.repair_stats(), "bullet repair applied");
        }
        strip_wor_from_monotonicity_stripped_utterances(&mut chat_file, finalized.monotonicity());
        let written = crate::chat_ops::fa::retain_decision_evidence(
            &mut chat_file,
            crate::chat_ops::fa::FaDecisions::without_injection(
                rescue_decisions,
                grouping_decisions,
                finalized,
            ),
        );
        return admission.finish(
            FaResult::without_groups(
                chat_file,
                fa_params.gap_healing,
                fa_params.engine.as_wire_name(),
                services.cache_namespace,
            )
            .with_written_decisions(written),
        );
    }

    info!(
        num_groups = groups.len(),
        total_words = groups.iter().map(|g| g.word_count()).sum::<usize>(),
        // How many over-budget utterances were split at UTR anchors, out of
        // how many utterances UTR anchored at all.
        anchored_groups = groups
            .iter()
            .filter(|g| matches!(g.span(), crate::chat_ops::fa::GroupSpan::Anchored(_)))
            .count(),
        anchored_utterances = anchors.anchored_utterances(),
        // Reported beside the other grouping facts rather than only as its own
        // warning, so a reader of one line sees whether our gap arithmetic
        // overshot the audio.
        windows_clamped,
        "FA grouping complete"
    );

    // 3-7. Resolve every group's timings: `%wor` reuse for whole groups, then
    // cache and worker inference per request (an anchored group is one
    // request per piece), then reassembly into group timings in word order.
    let mut resolved = resolve_group_timings(FaDispatchInputs {
        groups: &groups,
        chat_file: &chat_file,
        reusable_utterances: &reusable_indices,
        audio,
        worker_lang,
        services,
        fa_params,
        progress,
        context: "forced alignment",
    })
    .await?;
    let fallback_events = std::mem::take(&mut resolved.fallback_events);
    // Injection consumes the timings; the evidence keeps the pre-injection
    // snapshot taken from them inside `into_parts`.
    let (final_timings, group_evidence) = resolved.into_parts();

    let fa_applied = apply_fa_results_with_projection_policy(
        &mut chat_file,
        &groups,
        &final_timings,
        projection,
        write_wor,
    )
    .map_err(kept_bullets_failed)?
    // Utterances 1f refreshed from reusable `%wor` before grouping: their
    // `%wor` (if requested) is written by the SAME phase this run's own
    // fresh injections are, after monotonicity resolves.
    .also_touched(partially_reused_touched);

    // 9. Apply optional repair, then enforce monotonicity: strip non-monotonic start times and clamp
    //    end-time overlaps. The old enforcement was removed (see comment in
    //    apply_fa_results) because it stripped too aggressively. The current
    //    version strips every start-time regression and clamps end times to
    //    the next utterance's start. Timing removal is now retained as a typed
    //    decision in both the optional CHAT projection and durable evidence.
    let finalized = fa_applied
        .then_finalize(
            &mut chat_file,
            BulletRepairPolicy::from(fa_params.bullet_repair),
        )
        .map_err(kept_bullets_failed)?;
    if fa_params.bullet_repair {
        tracing::info!(stats = %finalized.repair_stats(), "bullet repair applied");
    }

    // 9d. Retain decision provenance for all pipeline decisions that altered
    //    the output and strip abandoned review tiers. Ordering and retention
    //    live in `retain_decision_evidence`; this states the SOURCES only, and
    //    a sixth would not compile until both FA paths named it.
    let written_decisions = crate::chat_ops::fa::retain_decision_evidence(
        &mut chat_file,
        crate::chat_ops::fa::FaDecisions {
            rescue: rescue_decisions,
            grouping: grouping_decisions,
            finalized,
        },
    );
    let (decision_records, timing_effects) = written_decisions.into_evidence();
    let decision_traces = decision_records.into_iter().map(Into::into).collect();
    let timing_decisions = timing_effects.into_iter().map(Into::into).collect();

    // 10. Every processed return requires complete typed construction
    // admission; `finish` also takes the draft through the media/timing
    // transition its source was admitted for.
    admission.finish(FaResult {
        output: chat_file,
        group_evidence,
        engine: fa_params.engine.as_wire_name().to_owned(),
        cache_namespace: services.cache_namespace.clone(),
        decisions: decision_traces,
        timing_decisions,
        gap_healing: fa_params.gap_healing,
        fallback_events,
    })
}

mod incremental;
pub(crate) use incremental::{RetainedFaPrior, process_fa_incremental};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_ops::fa::WordGapHealing;
    use batchalign_transform::validate::ValidityLevel;
    use talkbank_model::model::Line;

    /// An L2-valid file: participants, languages, terminator, main-tier
    /// content, and a `@Media` line so the media/timing transition admits it.
    const ALIGNED: &str = "@UTF8\n@Begin\n@Languages:\teng\n\
@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|||||Target_Child|||\n\
@Media:\tsample, audio\n\
*CHI:\thello world . \u{15}0_500\u{15}\n\
@End\n";

    fn parse_aligned(text: &str) -> crate::chat_ops::ChatFile {
        let parser = crate::chat_parser();
        let (file, errors) = batchalign_transform::parse::parse_lenient(&parser, text);
        assert!(errors.is_empty(), "fixture must parse cleanly: {errors:?}");
        file
    }

    fn source_admission() -> FaAdmission {
        let source = read_fa_source(ALIGNED).expect("the source is completely valid");
        match source.attempt() {
            FaInputDocument::Active(active) => active.admission,
            FaInputDocument::Preserved(_) => panic!("fixture must use active alignment"),
        }
    }

    /// A fast-path-shaped result: no groups, built exactly as the `%wor`-reuse
    /// rerun route builds one.
    fn fast_path_result(file: crate::chat_ops::ChatFile) -> FaResult<crate::chat_ops::ChatFile> {
        FaResult::without_groups(
            file,
            WordGapHealing::PreserveMeasured,
            "test_engine",
            &FaCacheNamespace::for_test("test-build"),
        )
    }

    /// RED FIRST (2026-09-07 review, item 1): a document `align` declares it
    /// will not touch must be written back BYTE-IDENTICAL.
    ///
    /// Complete source admission retains original bytes alongside its parsed
    /// document. Serializing the model instead would lose the trailing spaces
    /// on the comment, despite the command promising unchanged output.
    ///
    /// The `assert_ne!` is the precondition and it is what makes this test
    /// worth having: without a construct the round trip actually changes, the
    /// assertion below would hold under both behaviours.
    #[test]
    fn a_declared_pass_through_carries_the_input_bytes_not_a_reserialization() {
        const DUMMY: &str = "@UTF8\n@Begin\n@Languages:\teng\n\
@Participants:\tCHI Child\n@Options:\tNoAlign\n\
@ID:\teng|test|CHI|||||Child|||\n@Comment:\tkept   \n\
*CHI:\thello .\n@End\n";
        let read = read_fa_source(DUMMY).expect("the NoAlign source is valid");
        assert_ne!(
            batchalign_transform::serialize::to_chat_string(read.document()),
            DUMMY,
            "precondition: re-serializing this model is NOT the identity, so a \
             proof built from the model cannot be the input's bytes"
        );

        let admitted = FaAdmission::pass_through(
            read.unchanged()
                .expect("NoAlign carries source admission")
                .clone(),
            WordGapHealing::PreserveMeasured,
            "test_engine",
            &FaCacheNamespace::for_test("test-build"),
        );

        assert_eq!(
            admitted.into_document().document.as_str(),
            DUMMY,
            "a document align declines to touch must be written back unchanged"
        );
    }

    /// RED FIRST (review item 1): the `%wor`-reuse and no-groups fast paths
    /// used to return `Ok(FaResult ...)` without running the gate at all, and
    /// the `%wor`-reuse path is the ordinary rerun route. A finalized model
    /// that fails `validate_output("align")` must be REFUSED, not returned.
    #[test]
    fn a_fast_path_result_failing_the_align_output_check_is_refused() {
        let admission = source_admission();

        let mut degraded = parse_aligned(ALIGNED);
        for line in &mut degraded.lines {
            if let Line::Utterance(utt) = line {
                utt.main.content.terminator = None;
            }
        }

        let Err(failure) = admission.finish(fast_path_result(degraded)) else {
            panic!("a fast-path output that lost a terminator must be refused");
        };
        let rendered = failure.to_string();
        assert!(
            rendered.contains("lost its terminator"),
            "the refusal must name what broke, got: {rendered}"
        );
    }

    /// Complete construction refuses an empty main tier even when the legacy
    /// structural-only workflow check would accept it.
    #[test]
    fn output_degraded_from_l2_to_l1_is_refused() {
        let admission = source_admission();

        let mut degraded = parse_aligned(ALIGNED);
        for line in &mut degraded.lines {
            if let Line::Utterance(utt) = line {
                // Empty main tier: still L1 (speaker declared, terminator
                // present), no longer L2.
                utt.main.content.content = talkbank_model::model::TierContentItems::new(Vec::new());
            }
        }
        assert!(
            batchalign_transform::validate::validate_to_level(
                &degraded,
                &[],
                ValidityLevel::StructurallyComplete
            )
            .is_ok(),
            "precondition: the degraded output still satisfies L1, so only a \
             complete construction must not rely on this partial check"
        );

        let Err(failure) = admission.finish(fast_path_result(degraded)) else {
            panic!("output degraded below its admission level must be refused");
        };
        assert!(
            failure.to_string().contains("E306"),
            "the refusal must retain Chatter's empty-utterance diagnostic, got: {failure}"
        );
    }

    /// An output that still meets the bar its input was admitted at passes.
    #[test]
    fn an_undegraded_fast_path_result_is_admitted() {
        let admission = source_admission();
        const COMPLETE: &str = "@UTF8\n@Begin\n@Languages:\teng\n\
@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|||||Target_Child|||\n\
@Media:\tsample, audio\n\
*CHI:\thello world . \u{15}0_500\u{15}\n\
%wor:\thello \u{15}0_250\u{15} world \u{15}250_500\u{15} .\n\
@End\n";
        assert!(
            admission
                .finish(fast_path_result(parse_aligned(COMPLETE)))
                .is_ok(),
            "complete retained timing must still pass its own gate"
        );
    }

    /// THE BOUNDARY (align recovery step 2): a file where part of the
    /// transcript cannot be aligned runs through the real FA entry point and
    /// comes back admitted and partial. The timed utterance keeps its timing
    /// (reused from its `%wor`, so no worker is needed), the untimed one stays
    /// in the transcript without bullets, and the shortfall names it with the
    /// window grouping refused. The writer's report is diagnosed, not clean.
    #[tokio::test]
    async fn a_partly_alignable_file_is_written_with_its_untimed_account() {
        use crate::pipeline::post_validate::{OutputReport, ProducedOutput};

        const PARTLY: &str = "@UTF8\n@Begin\n@Languages:\teng\n\
@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|||||Target_Child|||\n\
@Media:\tsample, audio\n\
*CHI:\thello world . \u{15}0_500\u{15}\n\
%wor:\thello \u{15}0_200\u{15} world \u{15}200_500\u{15} .\n\
*CHI:\tmore words here .\n\
@End\n";
        let admitted = run_on_a_long_recording(PARTLY)
            .await
            .expect("a partly alignable file is written, not refused");

        let crate::fa::AlignedOutput {
            document,
            shortfalls,
            exclusions,
        } = admitted.into_document();
        assert!(exclusions.is_empty(), "no utterance is left out on purpose");
        insta::assert_json_snapshot!(shortfalls, @r#"
        [
          {
            "kind": "timing_incomplete",
            "required_words": 5,
            "untimed_words": 3,
            "untimed_utterances": 1,
            "first_untimed": [
              {
                "utterance": 2,
                "words": 3,
                "untimed_words": 3,
                "cause": {
                  "kind": "window_refused",
                  "window": {
                    "cause": "over_budget",
                    "start_ms": 0,
                    "end_ms": 300017,
                    "budget_ms": 15000
                  }
                }
              }
            ]
          }
        ]
        "#);
        let text = document.as_str();
        assert!(
            text.contains("%wor:\thello \u{15}0_200\u{15} world \u{15}200_500\u{15} ."),
            "{text}"
        );
        assert!(text.contains("*CHI:\tmore words here .\n"), "{text}");
        assert!(matches!(
            OutputReport::of(&ProducedOutput::from(document), &shortfalls),
            OutputReport::Diagnosed(_)
        ));
    }

    /// Run the real FA entry point on `source` against a real 300 s
    /// recording, with a pool that has no workers: every case using it is
    /// decided without inference. An untimed utterance's estimated window
    /// spans the whole recording, far over the 15 s budget, so grouping
    /// refuses it.
    async fn run_on_a_long_recording(source: &str) -> Result<AdmittedFaResult, ServerError> {
        use crate::cache::UtteranceCache;
        use crate::worker::pool::{PoolConfig, WorkerPool};
        let dir = tempfile::tempdir().expect("tempdir");
        let audio_path = dir.path().join("sample.mp3");
        crate::media::probe::fixtures::write_unpadded_cbr_mp3(&audio_path);
        let identity = crate::chat_ops::fa::AudioIdentity::from_metadata("sample.mp3", 0, 0);
        let audio = AudioContext {
            audio_path: &audio_path,
            audio_identity: &identity,
            audio_duration: None,
        };
        let pool = WorkerPool::new(PoolConfig::default());
        let cache = UtteranceCache::sqlite(Some(dir.path().join("cache")))
            .await
            .expect("cache");
        let namespace = FaCacheNamespace::for_test("test-fa-wave-v1");
        let services = FaServices {
            pipeline: PipelineServices::new(&pool, &cache),
            cache_namespace: &namespace,
        };
        let params = FaParams {
            gap_healing: WordGapHealing::Heal,
            existing_wor_boundaries: crate::chat_ops::fa::ExistingWorBoundaryPolicy::Preserve,
            end_overlap_policy: crate::chat_ops::fa::EndOverlapPolicy::ClampAllAdjacent,
            main_bullets: crate::chat_ops::fa::MainBulletPolicy::DeriveFromWords,
            engine: crate::types::engines::FaEngineName::Wave2Vec,
            cache_policy: crate::types::params::CachePolicy::UseCache,
            wor_tier: crate::types::params::WorTierPolicy::Include,
            bullet_repair: true,
            review_level: crate::chat_ops::fa::ReviewLevel::None,
        };
        let source = read_fa_source_named(
            source,
            talkbank_model::model::TranscriptName::for_path(std::path::Path::new("sample.cha")),
            crate::chat_ops::fa::DEFAULT_MAIN_BULLET_POLICY,
        )?;
        run_fa_from_ast(
            source.attempt(),
            &audio,
            &crate::api::LanguageCode3::eng(),
            services,
            &params,
            None,
        )
        .await
    }

    /// A never-aligned transcript of `sample.cha`, two untimed utterances,
    /// with the case's `@Media` line.
    fn never_aligned(media_line: &str) -> String {
        format!(
            "@UTF8\n@Begin\n@Languages:\teng\n\
@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|||||Target_Child|||\n\
{media_line}*CHI:\thello world .\n\
*CHI:\tmore words .\n@End\n"
        )
    }

    /// Through the real entry point: a linked
    /// transcript whose only timing was the bullet on a `[+ diary]` note.
    /// Align never uses that bullet and `derive` does not write it, so when
    /// no speech can be aligned (every window is refused here) the output
    /// would be linked and untimed. It used to fail the output gate with a
    /// generic E544, reported as an internal fault; it is refused before any
    /// inference as unavailable evidence, naming the note and the remedies.
    #[tokio::test]
    async fn a_file_timed_only_on_a_note_bullet_is_refused_with_its_remedy() {
        let source = never_aligned("@Media:\tsample, audio\n").replace(
            "*CHI:\tmore words .\n",
            "*CHI:\tmore words .\n*CHI:\tthe note . [+ diary] \u{15}1000_2000\u{15}\n",
        );
        let Err(refused) = run_on_a_long_recording(&source).await else {
            panic!("a linked transcript that gains no timing cannot be written");
        };
        assert!(
            matches!(
                &refused,
                ServerError::RequiredEvidenceUnavailable(
                    crate::error::MissingRequiredEvidence::TimingRegeneration(_)
                )
            ),
            "{refused}"
        );
        assert_eq!(
            crate::runner::util::classify_server_error(&refused),
            crate::scheduling::FailureCategory::EvidenceUnavailable
        );
        let message = refused.to_string();
        for phrase in ["[+ diary]", "--main-bullets keep", ", unlinked", "E544"] {
            assert!(message.contains(phrase), "{phrase}: {message}");
        }
        assert!(!message.contains("internal"), "{message}");
    }

    /// THE BOUNDARY for a never-aligned transcript, through the real entry
    /// point. Neither declaration is refused at input any more; what each
    /// one owes decides the outcome when alignment finds no admissible
    /// window. `unlinked` is written untimed, still `unlinked`, diagnosed.
    /// Linked owes timing (E544), so it is refused, as unavailable evidence
    /// with the header change named, never as an internal error.
    #[tokio::test]
    async fn a_never_aligned_transcript_reaches_alignment_under_either_declaration() {
        let unlinked =
            run_on_a_long_recording(&never_aligned("@Media:\tsample, audio, unlinked\n"))
                .await
                .expect("an unlinked never-aligned transcript is written");
        let crate::fa::AlignedOutput {
            document,
            shortfalls,
            ..
        } = unlinked.into_document();
        assert!(
            document
                .as_str()
                .contains("@Media:\tsample, audio, unlinked\n"),
            "untimed output stays unlinked: {}",
            document.as_str()
        );
        assert_eq!(shortfalls.len(), 1, "its untimed words are its shortfall");

        let Err(linked) = run_on_a_long_recording(&never_aligned("@Media:\tsample, audio\n")).await
        else {
            panic!("a linked transcript that gains no timing cannot be written");
        };
        let category = crate::runner::util::classify_server_error(&linked);
        assert_eq!(
            category,
            crate::scheduling::FailureCategory::EvidenceUnavailable
        );
        let message = linked.to_string();
        assert!(message.contains("add `, unlinked`"), "{message}");
        assert!(message.contains("no output was written"), "{message}");
        assert!(!message.contains("regeneration"), "{message}");

        let Err(undeclared) = run_on_a_long_recording(&never_aligned("")).await else {
            panic!("no @Media is refused");
        };
        assert!(
            matches!(undeclared, ServerError::AlignmentMedia(_)),
            "refused at admission, not after alignment: {undeclared}"
        );
    }

    /// Timing written into a never-aligned source takes the media transition
    /// its admission proved: `unlinked` is consumed, a plain declaration
    /// stays, and the written document is valid CHAT. The same untimed
    /// output from a linked source is refused, because E544 forbids writing it.
    #[test]
    fn timing_written_into_a_never_aligned_source_links_its_media() {
        const TIMED: &str = "*CHI:\thello world . \u{15}0_500\u{15}\n\
%wor:\thello \u{15}0_200\u{15} world \u{15}200_500\u{15} .\n\
*CHI:\tmore words . \u{15}600_900\u{15}\n\
%wor:\tmore \u{15}600_700\u{15} words \u{15}700_900\u{15} .\n@End\n";
        let untimed_body = "*CHI:\thello world .\n*CHI:\tmore words .\n@End\n";
        let attempt = |source: &str| {
            let working = read_fa_source_named(
                source,
                talkbank_model::model::TranscriptName::for_path(std::path::Path::new("sample.cha")),
                crate::chat_ops::fa::DEFAULT_MAIN_BULLET_POLICY,
            )
            .expect("a never-aligned source is admitted");
            match working.attempt() {
                FaInputDocument::Active(active) => active.admission,
                FaInputDocument::Preserved(_) => panic!("a never-aligned source is aligned"),
            }
        };
        for (declared, written) in [
            ("sample, audio, unlinked", "sample, audio"),
            ("sample, audio", "sample, audio"),
            ("sample, video, unlinked", "sample, video"),
        ] {
            let source = never_aligned(&format!("@Media:\t{declared}\n"));
            let output = source.replace(untimed_body, TIMED);
            let admitted = attempt(&source)
                .finish(fast_path_result(parse_aligned(&output)))
                .expect("timed output of an admitted source is written");
            let document = admitted.into_document().document;
            let text = document.as_str();
            assert!(text.contains(&format!("@Media:\t{written}\n")), "{text}");
            assert!(!text.contains("unlinked"), "{text}");
            batchalign_transform::parse_and_validate(
                text,
                talkbank_model::ParseValidateOptions::default().with_validation(),
            )
            .expect("aligned output is valid CHAT");
        }

        let linked = never_aligned("@Media:\tsample, audio\n");
        let Err(refused) = attempt(&linked).finish(fast_path_result(parse_aligned(&linked))) else {
            panic!("an untimed output of a linked source cannot be written");
        };
        assert!(
            matches!(
                refused,
                ServerError::RequiredEvidenceUnavailable(
                    crate::error::MissingRequiredEvidence::TimingRegeneration(_)
                )
            ),
            "{refused}"
        );
        assert!(
            refused
                .to_string()
                .starts_with("alignment produced no timing"),
            "{refused}"
        );
    }

    #[test]
    fn cache_task_name_is_stable() {
        assert_eq!(FORCED_ALIGNMENT.name().as_str(), "forced_alignment");
        assert_eq!(
            FORCED_ALIGNMENT_RAW_EVIDENCE.name().as_str(),
            "forced_alignment_raw_evidence"
        );
    }

    #[test]
    fn cached_group_timing_admission_refuses_legacy_bare_vectors() {
        let value = serde_json::json!([null]);

        let error = AdmittedCachedFaTimings::decode(
            value,
            crate::types::engines::FaEngineName::Wave2Vec,
            &FaCacheNamespace::for_test("test-fa-wave-v1"),
            1,
            &CacheKey::from_content("legacy-derived"),
        )
        .expect_err("an unversioned bare vector must not become replayable evidence");

        assert!(matches!(error, FaDerivedEvidenceError::InvalidJson(_)));
    }

    #[test]
    fn cached_group_timing_admission_accepts_an_exact_versioned_envelope() {
        use crate::api::NonNegativeSeconds;
        use crate::types::engines::FaEngineName;
        use crate::types::worker_v2::{
            ExecuteResponseV2, IndexedWordTimingResultV2, TaskResultV2, WorkerRequestIdV2,
        };

        let key = CacheKey::from_content("derived-exact");
        let engine_version = FaCacheNamespace::for_test("test-fa-wave-v1");
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("derived-exact"),
            TaskResultV2::IndexedWordTimingResult(IndexedWordTimingResultV2 {
                indexed_timings: vec![None, None],
            }),
            NonNegativeSeconds::try_from(0.01).expect("fixture elapsed"),
        );
        let raw = raw_evidence::FaRawEvidence::admit_requested(
            &response,
            FaEngineName::Wave2Vec,
            &engine_version,
            raw_evidence::ExpectedFaWords::new(2),
            &key,
            raw_evidence::FaEvidenceRoute::Direct,
        )
        .expect("fixture raw evidence")
        .into_replayable()
        .expect("direct evidence is replayable");
        let value = AdmittedCachedFaTimings::encode_from_raw(vec![None, None], &raw)
            .expect("encode derived evidence");

        let timings = AdmittedCachedFaTimings::decode(
            value,
            FaEngineName::Wave2Vec,
            &engine_version,
            2,
            &key,
        )
        .expect("an exact cache vector should be admitted")
        .into_timings();

        assert_eq!(timings, vec![None, None]);
    }

    #[test]
    fn cached_group_timing_admission_refuses_worker_version_drift() {
        let key = CacheKey::from_content("derived-version-drift");
        let mut value = serde_json::to_value(VersionedCachedFaTimings {
            schema_version: FA_DERIVED_EVIDENCE_SCHEMA_VERSION,
            requested_engine: crate::types::engines::FaEngineName::Wave2Vec,
            request_engine_version: crate::api::ReportedEngineName::try_from("test-fa-wave-v1")
                .expect("valid engine name"),
            expected_words: 1,
            cache_key: key.clone(),
            timings: vec![None],
        })
        .expect("serialize versioned timing fixture");
        value["request_engine_version"] = serde_json::json!("test-fa-wave-v0");

        let error = AdmittedCachedFaTimings::decode(
            value,
            crate::types::engines::FaEngineName::Wave2Vec,
            &FaCacheNamespace::for_test("test-fa-wave-v1"),
            1,
            &key,
        )
        .expect_err("derived timings from another worker build must not replay");

        assert!(matches!(
            error,
            FaDerivedEvidenceError::EngineVersionDrift { .. }
        ));
    }

    #[test]
    fn raw_evidence_is_replayed_before_a_derived_timing_hit() {
        use crate::api::NonNegativeSeconds;
        use crate::chat_ops::fa::{FaGroup, FaWord, TimeSpan};
        use crate::chat_ops::{UtteranceIdx, WordIdx};
        use crate::types::engines::FaEngineName;
        use crate::types::worker_v2::{
            ExecuteResponseV2, IndexedWordTimingResultV2, TaskResultV2, WorkerRequestIdV2,
        };

        let groups = vec![FaGroup::test_fixture(
            TimeSpan::new(100, 900),
            vec![FaWord {
                utterance_index: UtteranceIdx::new(0),
                utterance_word_index: WordIdx::new(0),
                text: "hello".to_owned(),
            }],
            vec![UtteranceIdx::new(0)],
        )];
        let plan = units::test_support::plan_for(&groups);
        let unit = plan.units().next().expect("one request");
        let key = unit.cache_key().clone();
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("raw-first"),
            TaskResultV2::IndexedWordTimingResult(IndexedWordTimingResultV2 {
                indexed_timings: vec![None],
            }),
            NonNegativeSeconds::try_from(0.01).expect("fixture elapsed"),
        );
        let raw = raw_evidence::FaRawEvidence::admit_requested(
            &response,
            FaEngineName::Wave2Vec,
            &FaCacheNamespace::for_test("test-fa-wave-v1"),
            raw_evidence::ExpectedFaWords::new(1),
            &key,
            raw_evidence::FaEvidenceRoute::Direct,
        )
        .expect("fixture evidence is valid")
        .into_replayable()
        .expect("direct evidence is replayable");
        let raw_json = serde_json::to_value(raw).expect("serialize raw evidence");
        let derived_timing = WordTiming::new(
            110,
            180,
            crate::chat_ops::fa::origin::Origin::TranscriptBullet,
            crate::chat_ops::fa::origin::Origin::TranscriptBullet,
        )
        .expect("positive derived timing");
        let derived_json =
            serde_json::to_value(vec![Some(derived_timing)]).expect("serialize derived timing");

        let namespace = FaCacheNamespace::for_test("test-fa-wave-v1");
        let resolution = FaCacheUnitAdmission::new(unit, FaEngineName::Wave2Vec, &namespace)
            .resolve(Some(&raw_json), Some(&derived_json));

        assert!(resolution.refusals.is_empty());
        match resolution.admitted {
            AdmittedFaCacheGroup::RawEvidence(evidence) => {
                assert_eq!(evidence.timings, vec![None]);
            }
            other => panic!("raw evidence must win over derived timings, got {other:?}"),
        }
    }
}
