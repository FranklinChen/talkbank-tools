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
//! - **Multiple groups per file**: Utterances are grouped by time window; each group is one infer item.
//! - **Audio access**: Workers need the audio file path and time range, not just text.
//! - **DP alignment in Rust**: Model output is aligned to transcript words via Hirschberg.
//!
//! # Invariants for contributors
//!
//! - FA worker timestamps are chunk-relative; `parse_fa_response` must convert
//!   them to file-absolute ms with `audio_start_ms`.
//! - `apply_fa_results` ordering is load-bearing:
//!   inject → postprocess → utterance bullet update → `%wor` generation
//!   → monotonicity (E362) → same-speaker overlap enforcement (E704).
//! - Cache keys must include audio identity + time window + text + timing mode
//!   + engine; changing dimensions changes cache compatibility.

mod raw_evidence;
mod transport;

use crate::cache::CacheBackend;
use crate::chat_ops::fa::{
    BulletRepairPolicy, WordTiming, apply_fa_results_with_projection_policy, cache_key,
    expand_bullets_for_edge_fillers, finalize_without_injection, find_reusable_utterance_indices,
    group_utterances, has_reusable_wor_timing, projection_without_injection_with_touched,
    refresh_reusable_alignment, refresh_reusable_utterances, rescue_narrow_bullets,
    strip_wor_from_monotonicity_stripped_utterances,
};
use crate::chat_ops::{CacheKey, CacheTaskName};
use crate::params::{AudioContext, FaParams};
use crate::pipeline::PipelineServices;
use crate::pipeline::post_validate::PostValidated;
use batchalign_transform::parse::{is_ca, is_dummy, is_no_align};
use batchalign_transform::validate::{ValidityLevel, validate_to_level};
use tracing::{info, warn};

use crate::api::DurationMs;
use crate::chat_ops::fa::Grouping;
use crate::error::ServerError;
use crate::runner::util::{FileStage, ProgressSender, ProgressUpdate};
use crate::types::results::{FaGroupEvidence, FaOutput, FaResult};
use crate::types::traces::{FaEvidenceSourceTrace, FaGroupTrace, TimingTrace};
use transport::{FaInferencePlan, FaWorkerTransport, UncheckedFaWorkerBatch, plan_fa_inference};

/// The validity level forced alignment admits an input at.
///
/// Stated ONCE. Its output gate reads the level off the [`FaAdmission`] the
/// admission check produced, so there is no second place for a bar to be
/// written down and no way for the two to disagree.
const FA_ADMISSION_LEVEL: ValidityLevel = ValidityLevel::MainTierValid;

/// Proof that an FA input cleared pre-validation, carrying the level it
/// cleared and the exclusive route to a returned [`FaResult`].
///
/// # Why this is a type
///
/// Two defects lived in the gap this closes, and both were invisible in
/// review because the correct-looking code was spread over three functions.
///
/// 1. The gate ran at ONE of the four `Ok(FaResult ...)` returns. The
///    `%wor`-reuse fast path, the no-groups path and the incremental
///    no-groups path each finalized a model (bullet repair, monotonicity
///    stripping, decision retention) and returned it having judged nothing.
///    That fast path is the ordinary rerun route, so the gate was absent from
///    the commonest way FA output reaches disk. This type is now the only way
///    to build the returned value, so a new early return cannot skip it: it
///    has nothing to return. There are exactly two routes, [`Self::finish`]
///    for a document FA processed and [`Self::pass_through`] for a declared
///    `@Options: dummy` / `NoAlign` document, and both hand back an
///    [`AdmittedFaResult`] whose fields are private to this module.
/// 2. Where the gate did run it restated its level as
///    `ValidityLevel::StructurallyComplete`, while admission demanded
///    `MainTierValid`. `ValidityLevel` is `Ord`, so that is strictly lower:
///    output degraded from L2 to L1 by the run itself passed. The level is
///    [`FA_ADMISSION_LEVEL`] on both sides now, read from the constant by both
///    [`Self::admit`] and [`Self::finish`], so restating it is not something a
///    caller can do.
///
/// # Why it carries nothing
///
/// It held a `level: ValidityLevel` field that was assigned
/// `FA_ADMISSION_LEVEL` at its one construction and read back in `finish`: a
/// copy of a constant, kept in step by hand, and a value a future caller could
/// have set to something else. The proof is the EXISTENCE of the value, not
/// anything inside it, so the type is zero-sized and `finish` reads the
/// constant directly.
///
/// The private `()` field is what keeps it unforgeable. A unit struct
/// `FaAdmission;` would be constructible anywhere the name is visible, which
/// is the whole crate, and this type's entire job is to be obtainable only by
/// passing the gate.
pub(super) struct FaAdmission(());

impl FaAdmission {
    /// Run FA's pre-validation gate. The only constructor.
    pub(super) fn admit(
        file: &crate::chat_ops::ChatFile,
        parse_errors: &[crate::chat_ops::ParseError],
    ) -> Result<Self, ServerError> {
        match validate_to_level(file, parse_errors, FA_ADMISSION_LEVEL) {
            Ok(()) => Ok(Self(())),
            Err(errors) => {
                let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
                Err(ServerError::Validation(format!(
                    "align pre-validation failed: {}",
                    msgs.join("; ")
                )))
            }
        }
    }

    /// Admit a `@Options: dummy` / `NoAlign` document the command refuses to
    /// touch, and hand back the value the caller returns.
    ///
    /// The second and last route to an [`AdmittedFaResult`]. It exists because
    /// `FaResult::pass_through` used to be a `pub(crate)` constructor that
    /// built the returned `Ok` value directly at three sites, which made this
    /// type's own claim to be the only route false. The proof it carries is
    /// `PostValidated::pass_through`: NOT gated, because gating a dummy file
    /// re-judges the INPUT against a bar the input never had to meet (it has
    /// no `@Participants`, so it fails L1) and would refuse a document the
    /// researcher asked us to leave alone. That refusal is not hypothetical:
    /// it is what the writer's own second gate did until this proof started
    /// travelling to it.
    ///
    /// # It takes the input TEXT, and that is the point
    ///
    /// `align` promises a `@Options: dummy` or `NoAlign` document back
    /// UNCHANGED, and the book says `NoAlign` is a strict pass-through with
    /// zero modifications. Until 2026-09-07 the proof was built with
    /// `PostValidated::declined_stripping_decision_tiers` (then named
    /// `declined_document`), which SERIALIZES THE MODEL, so the
    /// bytes written were a round trip through the parser and the serializer
    /// and every difference that round trip makes reached disk unexamined:
    /// `@Comment:\tkept   ` came back without its trailing spaces. Nothing was
    /// gating those bytes, because this is the one route that judges nothing,
    /// so it is the one route where a silent rewrite could not be caught.
    ///
    /// The text therefore travels with the model from the seam that READ it:
    /// the dispatch task carries the bytes it read off disk, and both entry
    /// points take the same [`FaInputDocument`], so the incremental path hands
    /// on the very bytes the full path would.
    pub(super) fn pass_through(
        chat_file: crate::chat_ops::ChatFile,
        original_text: &str,
        gap_healing: crate::chat_ops::fa::WordGapHealing,
        engine: &str,
        engine_version: &str,
    ) -> AdmittedFaResult {
        let document =
            PostValidated::pass_through(original_text, crate::api::ReleasedCommand::Align);
        AdmittedFaResult {
            // Written out rather than hidden behind a `FaResult::pass_through`
            // constructor: that constructor was the hole this type exists to
            // close, and there is nothing left for it to be reused by.
            result: FaResult {
                output: FaOutput::PassThrough(chat_file),
                group_evidence: Vec::new(),
                engine: engine.to_owned(),
                engine_version: engine_version.to_owned(),
                decisions: Vec::new(),
                timing_decisions: Vec::new(),
                gap_healing,
                fallback_events: Vec::new(),
            },
            document,
        }
    }

    /// Gate a finished FA result and hand back the value the caller returns.
    ///
    /// Fail-closed: a file whose aligned output fails the gate fails THIS file
    /// (`ServerError::Validation` classifies as `FailureCategory::Validation`),
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
    /// It CONSUMES the admission, so one admitted input yields at most one
    /// finished result. Taking `&self` let `admit(&a)` be followed by
    /// `finish(result_for_b)`, and by a second `finish` after that; nothing is
    /// deleted by the change, but the mismatched pair stops type-checking.
    pub(super) fn finish(self, result: FaResult) -> Result<AdmittedFaResult, ServerError> {
        let document = PostValidated::gate(
            result.output.as_chat_file(),
            FA_ADMISSION_LEVEL,
            crate::api::ReleasedCommand::Align,
        )
        .map_err(|failure| ServerError::Validation(failure.to_string()))?;
        Ok(AdmittedFaResult { result, document })
    }
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
    result: FaResult,
    /// The bytes the writer may persist, and the proof that it may.
    document: PostValidated,
}

impl AdmittedFaResult {
    /// Take the proven bytes, discarding the evidence timeline.
    pub(crate) fn into_document(self) -> PostValidated {
        self.document
    }

    /// Split the proven bytes from the evidence timeline.
    ///
    /// Both halves leave together because a caller that wants the timeline
    /// still has to write the document, and handing out the timeline alone
    /// would leave the bytes with no route to disk.
    pub(crate) fn into_document_and_timeline(
        self,
    ) -> (PostValidated, crate::types::traces::FaTimelineTrace) {
        (self.document, self.result.into_timeline_trace())
    }
}

/// Cache task name for FA results.
const CACHE_TASK: CacheTaskName = CacheTaskName::ForcedAlignment;
/// Cache namespace for immutable worker responses before local reconciliation.
const RAW_EVIDENCE_CACHE_TASK: CacheTaskName = CacheTaskName::ForcedAlignmentRawEvidence;

pub(super) fn collect_final_timings(
    all_timings: Vec<Option<Vec<Option<WordTiming>>>>,
    context: &str,
) -> Result<Vec<Vec<Option<WordTiming>>>, ServerError> {
    let missing_groups: Vec<usize> = all_timings
        .iter()
        .enumerate()
        .filter_map(|(index, timings)| timings.is_none().then_some(index))
        .collect();
    if !missing_groups.is_empty() {
        return Err(ServerError::Validation(format!(
            "{context} completed without timings for group(s): {missing_groups:?}"
        )));
    }

    // Safety: the None check above returned Err for any missing groups,
    // so all remaining elements are guaranteed Some.
    Ok(all_timings.into_iter().flatten().collect())
}

pub(super) fn collect_evidence_sources(
    sources: Vec<Option<FaEvidenceSourceTrace>>,
    context: &str,
) -> Result<Vec<FaEvidenceSourceTrace>, ServerError> {
    let missing_groups: Vec<usize> = sources
        .iter()
        .enumerate()
        .filter_map(|(index, source)| source.is_none().then_some(index))
        .collect();
    if !missing_groups.is_empty() {
        return Err(ServerError::Validation(format!(
            "{context} completed without an evidence source for group(s): {missing_groups:?}"
        )));
    }
    Ok(sources.into_iter().flatten().collect())
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
    request_engine_version: crate::api::EngineVersion,
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
        cached: crate::api::EngineVersion,
        current: crate::api::EngineVersion,
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
            request_engine_version: raw.request_engine_version().clone(),
            expected_words,
            cache_key: raw.cache_key().clone(),
            timings,
        })?)
    }

    fn decode(
        value: serde_json::Value,
        requested_engine: crate::types::engines::FaEngineName,
        current_engine_version: &crate::api::EngineVersion,
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
        if &cached.request_engine_version != current_engine_version {
            return Err(FaDerivedEvidenceError::EngineVersionDrift {
                cached: cached.request_engine_version,
                current: current_engine_version.clone(),
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

/// Re-admit and locally reparse immutable FA worker evidence for one group.
fn replay_cached_raw_evidence(
    value: serde_json::Value,
    cache_key: &CacheKey,
    engine: crate::types::engines::FaEngineName,
    engine_version: &crate::api::EngineVersion,
    group_index: usize,
    group: &crate::chat_ops::fa::FaGroup,
) -> Result<transport::FaWorkerEvidenceResult, ServerError> {
    let evidence = raw_evidence::ReplayableFaRawEvidence::decode(
        value,
        engine,
        engine_version,
        raw_evidence::ExpectedFaWords::new(group.words.len()),
        cache_key,
    )
    .map_err(|error| {
        ServerError::Validation(format!(
            "cached raw FA evidence for group {group_index} was refused: {error}"
        ))
    })?;
    transport::replay_group_evidence(evidence, group_index, group)
}

/// A cache layer whose value was present but could not be admitted.
#[derive(Debug)]
struct RefusedFaCacheLayer {
    layer: &'static str,
    error: String,
}

/// The admitted cache state for one current FA group.
///
/// Raw worker evidence is intentionally tried first. Replaying it through the
/// current Rust projection is what lets alignment-algorithm experiments reuse
/// model inference. The derived timing layer remains a compatibility and
/// resilience fallback when raw evidence is absent or corrupt.
#[derive(Debug)]
enum AdmittedFaCacheGroup {
    RawEvidence(Box<transport::FaWorkerEvidenceResult>),
    DerivedTimings(AdmittedCachedFaTimings),
    Miss,
}

/// Result of checking both cache layers for one current FA group.
#[derive(Debug)]
struct FaCacheResolution {
    admitted: AdmittedFaCacheGroup,
    refusals: Vec<RefusedFaCacheLayer>,
}

/// Capability binding one current FA group to the exact facts against which
/// both cache layers must be admitted.
///
/// Lookup code constructs this where the group index and semantic key are
/// born. Raw and derived candidates then share the same relationship instead
/// of receiving six parallel facts independently.
struct FaCacheGroupAdmission<'a> {
    cache_key: &'a CacheKey,
    engine: crate::types::engines::FaEngineName,
    engine_version: &'a crate::api::EngineVersion,
    group_index: usize,
    group: &'a crate::chat_ops::fa::FaGroup,
}

impl<'a> FaCacheGroupAdmission<'a> {
    fn new(
        cache_key: &'a CacheKey,
        engine: crate::types::engines::FaEngineName,
        engine_version: &'a crate::api::EngineVersion,
        group_index: usize,
        group: &'a crate::chat_ops::fa::FaGroup,
    ) -> Self {
        Self {
            cache_key,
            engine,
            engine_version,
            group_index,
            group,
        }
    }

    fn resolve(
        &self,
        raw_value: Option<&serde_json::Value>,
        derived_value: Option<&serde_json::Value>,
    ) -> FaCacheResolution {
        let mut refusals = Vec::new();

        if let Some(value) = raw_value {
            match replay_cached_raw_evidence(
                value.clone(),
                self.cache_key,
                self.engine,
                self.engine_version,
                self.group_index,
                self.group,
            ) {
                Ok(evidence) => {
                    return FaCacheResolution {
                        admitted: AdmittedFaCacheGroup::RawEvidence(Box::new(evidence)),
                        refusals,
                    };
                }
                Err(error) => refusals.push(RefusedFaCacheLayer {
                    layer: RAW_EVIDENCE_CACHE_TASK.as_str(),
                    error: error.to_string(),
                }),
            }
        }

        if let Some(value) = derived_value {
            match AdmittedCachedFaTimings::decode(
                value.clone(),
                self.engine,
                self.engine_version,
                self.group.words.len(),
                self.cache_key,
            ) {
                Ok(timings) => {
                    return FaCacheResolution {
                        admitted: AdmittedFaCacheGroup::DerivedTimings(timings),
                        refusals,
                    };
                }
                Err(error) => refusals.push(RefusedFaCacheLayer {
                    layer: CACHE_TASK.as_str(),
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

/// Close the temporary indexed algorithm state into evidence that cannot
/// mispair one group's provenance, cache identity, or worker timing.
fn assemble_group_evidence(
    groups: Vec<FaGroupTrace>,
    evidence_sources: Vec<FaEvidenceSourceTrace>,
    cache_keys: Vec<String>,
    pre_injection_timings: Vec<Vec<Option<TimingTrace>>>,
) -> Result<Vec<FaGroupEvidence>, ServerError> {
    let lengths = [
        groups.len(),
        evidence_sources.len(),
        cache_keys.len(),
        pre_injection_timings.len(),
    ];
    if lengths.iter().any(|length| *length != lengths[0]) {
        return Err(ServerError::Validation(format!(
            "FA evidence cardinality drift: groups={}, sources={}, keys={}, timings={}",
            lengths[0], lengths[1], lengths[2], lengths[3]
        )));
    }
    Ok(groups
        .into_iter()
        .zip(evidence_sources)
        .zip(cache_keys)
        .zip(pre_injection_timings)
        .map(
            |(((group, source), cache_key), pre_injection_timings)| FaGroupEvidence {
                group,
                source,
                cache_key,
                pre_injection_timings,
            },
        )
        .collect())
}

// ---------------------------------------------------------------------------
// Per-file FA processing
// ---------------------------------------------------------------------------

/// A CHAT document as forced alignment received it.
///
/// The model, the parse errors that came with it, and the BYTES it was read
/// as, in one value, because the three are only ever right together: the
/// errors describe that parse and no other, and the text is the only thing
/// that can keep align's promise to hand a `@Options: dummy` or `NoAlign`
/// document back byte-identical. As three parameters, nothing stopped a
/// caller pairing one file's model with another file's text, and the text was
/// simply absent, which is how the pass-through came to be a re-serialization.
pub(crate) struct FaInputDocument<'a> {
    /// The parsed document FA works on.
    chat_file: crate::chat_ops::ChatFile,
    /// The errors that parse reported, carried so admission judges the same
    /// parse the model came from.
    parse_errors: Vec<crate::chat_ops::ParseError>,
    /// The bytes the document was read as.
    ///
    /// NOT a serialization of `chat_file`, and not required to still describe
    /// it: the dispatch path runs the UTR pre-pass over the model between the
    /// read and this call, so the two legitimately diverge. What it is for is
    /// the one route that applies NOTHING, where the input's own bytes are the
    /// correct output.
    text: &'a str,
}

impl<'a> FaInputDocument<'a> {
    /// Bundle a parsed document with the bytes it was read as.
    ///
    /// Deliberately not `parse(text)`: the dispatch path hands over a model
    /// the UTR pre-pass has already edited, so deriving one from the other
    /// here would either undo that work or make a false claim about it.
    pub(crate) fn new(
        chat_file: crate::chat_ops::ChatFile,
        parse_errors: Vec<crate::chat_ops::ParseError>,
        text: &'a str,
    ) -> Self {
        Self {
            chat_file,
            parse_errors,
            text,
        }
    }
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
/// 1. Pre-validate the document it was handed (`MainTierValid`).
/// 2. Group utterances into FA windows.
/// 3. Resolve cache hits/misses per group.
/// 4. Send miss groups through the FA worker transport adapter.
/// 5. Parse responses and align to transcript words in Rust.
/// 6. Apply timings + postprocessing (`apply_fa_results`).
/// 7. Reconcile media/timing typestate and run full post-validation.
pub(crate) async fn run_fa_from_ast(
    document: FaInputDocument<'_>,
    audio: &AudioContext<'_>,
    worker_lang: &crate::api::LanguageCode3,
    services: PipelineServices<'_>,
    fa_params: &FaParams,
    progress: Option<&ProgressSender>,
) -> Result<AdmittedFaResult, ServerError> {
    let FaInputDocument {
        mut chat_file,
        parse_errors,
        text: chat_text,
    } = document;
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

    // 1b. Skip dummy files
    if is_dummy(&chat_file) {
        return Ok(FaAdmission::pass_through(
            chat_file,
            chat_text,
            fa_params.gap_healing,
            fa_params.engine.as_wire_name(),
            services.engine_version.as_ref(),
        ));
    }

    // 1c. @Options: NoAlign: strict pass-through, zero modifications.
    //
    // A researcher who sets this option has opted the file out of all
    // alignment processing.  The file is returned EXACTLY as parsed:
    // no timestamps added, removed, or adjusted, no %wor generated,
    // no decision tiers written.  This includes cleanup passes that
    // might seem safe (e.g., monotonicity enforcement); those are
    // the researcher's responsibility.
    //
    // See book/src/batchalign/developer/commands/align.md: "NoAlign: strict pass-through".
    if is_no_align(&chat_file) {
        return Ok(FaAdmission::pass_through(
            chat_file,
            chat_text,
            fa_params.gap_healing,
            fa_params.engine.as_wire_name(),
            services.engine_version.as_ref(),
        ));
    }

    // 1d. Pre-validation gate. The level lives on `FaAdmission`, and the
    // proof it returns is what the output gate later reads its bar from.
    let admission = FaAdmission::admit(&chat_file, &parse_errors)?;

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
        let finalized = projection_without_injection_with_touched(
            fa_params.projection_policy(),
            write_wor,
            touched,
        )
        .then_finalize(
            &mut chat_file,
            BulletRepairPolicy::from(fa_params.bullet_repair),
        );
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
                services.engine_version.as_ref(),
            )?
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
    let rescue_decisions = rescue_narrow_bullets(&mut chat_file);

    // 2b. Expand utterance bullets to cover edge fillers in inter-utterance gaps.
    // UTR-assigned bullets may be too narrow to include trailing/leading fillers
    // whose audio lives in the gap between utterances.
    expand_bullets_for_edge_fillers(&mut chat_file);

    // Resolved once, here, and used for BOTH grouping and the containment
    // checks on what the engine returns. Grouping used to take an
    // `Option<u64>` and invent its own behaviour when it was absent; there is
    // one recording and one answer.
    let recording = audio.recording().await?;
    // 2c. Group utterances
    let Grouping {
        groups,
        refusals: grouping_decisions,
        windows_clamped,
    } = group_utterances(&chat_file, fa_params.max_group_ms().0, &recording);

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
                fa_params.projection_policy(),
                BulletRepairPolicy::from(fa_params.bullet_repair),
            )
        } else {
            projection_without_injection_with_touched(
                fa_params.projection_policy(),
                write_wor,
                partially_reused_touched,
            )
            .then_finalize(
                &mut chat_file,
                BulletRepairPolicy::from(fa_params.bullet_repair),
            )
        };
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
                services.engine_version.as_ref(),
            )?
            .with_written_decisions(written),
        );
    }

    info!(
        num_groups = groups.len(),
        total_words = groups.iter().map(|g| g.words.len()).sum::<usize>(),
        // Reported beside the other grouping facts rather than only as its own
        // warning, so a reader of one line sees whether our gap arithmetic
        // overshot the audio.
        windows_clamped,
        "FA grouping complete"
    );

    if let Some(tx) = progress {
        let _ = tx.send(ProgressUpdate::new(
            FileStage::CheckingCache,
            Some(0),
            Some(groups.len() as i64),
        ));
    }

    // 3. For each group: compute cache key, check cache
    let word_texts: Vec<Vec<String>> = groups
        .iter()
        .map(|g| g.words.iter().map(|w| w.text.clone()).collect())
        .collect();

    let cache_keys: Vec<CacheKey> = groups
        .iter()
        .zip(word_texts.iter())
        .map(|(g, words)| {
            cache_key(
                words,
                audio.audio_identity,
                g.audio_start_ms(),
                g.audio_end_ms(),
                fa_params.gap_healing,
                fa_params.engine,
            )
        })
        .collect();

    // 4. Cache lookup
    let key_strings: Vec<String> = cache_keys.iter().map(|k| k.as_str().to_string()).collect();
    let cached = match fa_params.cache_policy {
        crate::params::CachePolicy::SkipCache => std::collections::HashMap::new(),
        crate::params::CachePolicy::UseCache | crate::params::CachePolicy::RequireCache => {
            match services
                .cache
                .get_batch(&key_strings, CACHE_TASK.as_str(), services.engine_version)
                .await
            {
                Ok(map) => map,
                Err(e) => {
                    warn!(error = %e, "FA cache batch lookup failed (treating all as misses)");
                    std::collections::HashMap::new()
                }
            }
        }
    };
    let cached_raw = match fa_params.cache_policy {
        crate::params::CachePolicy::SkipCache => std::collections::HashMap::new(),
        crate::params::CachePolicy::UseCache | crate::params::CachePolicy::RequireCache => {
            match services
                .cache
                .get_batch(
                    &key_strings,
                    RAW_EVIDENCE_CACHE_TASK.as_str(),
                    services.engine_version,
                )
                .await
            {
                Ok(map) => map,
                Err(error) => {
                    warn!(error = %error, "Raw FA evidence cache batch lookup failed");
                    std::collections::HashMap::new()
                }
            }
        }
    };

    // 5. Partition into reused (from %wor), cache hits, and misses
    let mut all_timings: Vec<Option<Vec<Option<WordTiming>>>> = vec![None; groups.len()];
    let mut evidence_sources: Vec<Option<FaEvidenceSourceTrace>> = vec![None; groups.len()];
    let mut miss_indices: Vec<usize> = Vec::new();
    let mut reused_group_count = 0usize;
    let mut fallback_events = Vec::new();

    for (i, key) in cache_keys.iter().enumerate() {
        // Tier 1: group fully reusable from %wor (all utterances have clean timing)
        if !reusable_indices.is_empty()
            && groups[i]
                .utterance_indices
                .iter()
                .all(|idx| reusable_indices.contains(&idx.raw()))
            && let Some(timings) =
                incremental::collect_preserved_group_timings(&chat_file, &groups[i])
        {
            all_timings[i] = Some(timings);
            evidence_sources[i] = Some(FaEvidenceSourceTrace::WorReuse);
            reused_group_count += 1;
            continue;
        }

        // Tier 2: replay immutable worker evidence through current Rust logic.
        // Tier 3: fall back to the admitted derived timing cache when raw
        // evidence is unavailable. This ordering is what makes local algorithm
        // experiments inference-free.
        let resolution = FaCacheGroupAdmission::new(
            key,
            fa_params.engine,
            services.engine_version,
            i,
            &groups[i],
        )
        .resolve(cached_raw.get(key.as_str()), cached.get(key.as_str()));
        for refusal in resolution.refusals {
            warn!(
                error = %refusal.error,
                cache_layer = refusal.layer,
                group = i,
                "Cached FA evidence was refused"
            );
        }
        match resolution.admitted {
            AdmittedFaCacheGroup::RawEvidence(evidence) => {
                let evidence = *evidence;
                if let Some(event) = evidence.fallback_event {
                    fallback_events.push(event);
                }
                all_timings[i] = Some(evidence.timings);
                evidence_sources[i] = Some(FaEvidenceSourceTrace::RawEvidenceReplay);
            }
            AdmittedFaCacheGroup::DerivedTimings(timings) => {
                all_timings[i] = Some(timings.into_timings());
                evidence_sources[i] = Some(FaEvidenceSourceTrace::Cache);
            }
            AdmittedFaCacheGroup::Miss => {
                // Tier 4: no admitted cache evidence, so inference is needed.
                miss_indices.push(i);
            }
        }
    }

    let cache_hits = groups.len() - miss_indices.len() - reused_group_count;
    let reused_or_cached_groups = reused_group_count + cache_hits;
    if cache_hits > 0 || reused_group_count > 0 {
        info!(
            reused = reused_group_count,
            cache_hits = cache_hits,
            misses = miss_indices.len(),
            "FA partition (reused from %wor / cache hits / misses)"
        );
    }

    if let Some(tx) = progress {
        let _ = tx.send(ProgressUpdate::new(
            FileStage::Aligning,
            Some(reused_or_cached_groups as i64),
            Some(groups.len() as i64),
        ));
    }

    let transport = FaWorkerTransport::production(services);

    // 6. Dispatch miss groups through the FA worker transport adapter
    if let FaInferencePlan::Authorized(authorization) =
        plan_fa_inference(fa_params.cache_policy, &miss_indices)?
    {
        // Every group already owns the recording-bound window admitted by
        // grouping; live inference and cache replay consume that same proof.
        let parsed_results = transport
            .infer_groups(
                UncheckedFaWorkerBatch {
                    word_texts: &word_texts,
                    groups: &groups,
                    cache_keys: &cache_keys,
                    authorization,
                    audio_path: audio.audio_path,
                    worker_lang: worker_lang.into(),
                    engine: fa_params.engine,
                    gap_healing: fa_params.gap_healing,
                }
                .admit()?,
            )
            .await?;

        for (parsed_idx, parsed_result) in parsed_results.into_iter().enumerate() {
            let projection = parsed_result.into_projection();
            let miss_idx = projection.group_index;
            evidence_sources[miss_idx] = Some(projection.source());
            let timings = projection.timings;
            let raw_evidence = projection.raw_evidence;
            let fallback_event = projection.fallback_event;
            if let Some(event) = fallback_event {
                fallback_events.push(event);
            }

            // Only direct, version-identified evidence can enter either cache
            // layer. Fallback and unaligned results remain valid for this run
            // but are deliberately recomputed later: the fallback model is
            // outside the primary request's version namespace.
            let ba_version = env!("CARGO_PKG_VERSION");
            if let Some(raw_evidence) = raw_evidence {
                match AdmittedCachedFaTimings::encode_from_raw(timings.clone(), &raw_evidence) {
                    Ok(cache_data) => {
                        if let Err(error) = services
                            .cache
                            .put_batch(
                                &[(cache_keys[miss_idx].as_str().to_string(), cache_data)],
                                CACHE_TASK.as_str(),
                                services.engine_version,
                                ba_version,
                            )
                            .await
                        {
                            warn!(error = %error, "Failed to cache derived FA evidence (non-fatal)");
                        }
                    }
                    Err(error) => {
                        warn!(error = %error, "Failed to encode derived FA evidence (non-fatal)");
                    }
                }
                match serde_json::to_value(raw_evidence) {
                    Ok(cache_data) => {
                        if let Err(error) = services
                            .cache
                            .put_batch(
                                &[(cache_keys[miss_idx].as_str().to_string(), cache_data)],
                                RAW_EVIDENCE_CACHE_TASK.as_str(),
                                services.engine_version,
                                ba_version,
                            )
                            .await
                        {
                            warn!(error = %error, "Failed to cache raw FA evidence (non-fatal)");
                        }
                    }
                    Err(error) => {
                        warn!(error = %error, "Failed to serialize raw FA evidence (non-fatal)");
                    }
                }
            }
            all_timings[miss_idx] = Some(timings);

            if let Some(tx) = progress {
                let done = reused_or_cached_groups + parsed_idx + 1;
                let _ = tx.send(ProgressUpdate::new(
                    FileStage::Aligning,
                    Some(done as i64),
                    Some(groups.len() as i64),
                ));
            }
        }
    }

    // 8. Apply all results
    if let Some(tx) = progress {
        let _ = tx.send(ProgressUpdate::new(
            FileStage::ApplyingResults,
            Some(groups.len() as i64),
            Some(groups.len() as i64),
        ));
    }

    let final_timings = collect_final_timings(all_timings, "forced alignment")?;
    let evidence_sources = collect_evidence_sources(evidence_sources, "forced alignment")?;

    // Snapshot pre-injection timings (before apply_fa_results consumes them)
    let pre_injection_timings: Vec<Vec<Option<TimingTrace>>> = final_timings
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|t| t.as_ref().map(TimingTrace::from_word_timing))
                .collect()
        })
        .collect();

    let fa_applied = apply_fa_results_with_projection_policy(
        &mut chat_file,
        &groups,
        &final_timings,
        fa_params.projection_policy(),
        write_wor,
    )
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
    let finalized = fa_applied.then_finalize(
        &mut chat_file,
        BulletRepairPolicy::from(fa_params.bullet_repair),
    );
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
            unplaceable: grouping_decisions,
            finalized,
        },
    );
    let (decision_records, timing_effects) = written_decisions.into_evidence();
    let decision_traces = decision_records.into_iter().map(Into::into).collect();
    let timing_decisions = timing_effects.into_iter().map(Into::into).collect();

    // 10. Post-validation runs in `FaAdmission::finish`, below, at the level
    //    this file was ADMITTED at, together with every other `Ok` return.
    let output = FaOutput::processed(chat_file)?;

    // 10. Build group traces
    let group_traces: Vec<FaGroupTrace> = groups
        .iter()
        .map(|g| FaGroupTrace {
            audio_start_ms: DurationMs(g.audio_start_ms()),
            audio_end_ms: DurationMs(g.audio_end_ms()),
            utterance_indices: g.utterance_indices.iter().map(|idx| idx.raw()).collect(),
            words: g.words.iter().map(|w| w.text.clone()).collect(),
            word_ids: g.words.iter().map(|word| word.stable_id()).collect(),
        })
        .collect();

    // 11. Serialize and return structured result
    let group_evidence = assemble_group_evidence(
        group_traces,
        evidence_sources,
        cache_keys
            .iter()
            .map(|key| key.as_str().to_owned())
            .collect(),
        pre_injection_timings,
    )?;

    admission.finish(FaResult {
        output,
        group_evidence,
        engine: fa_params.engine.as_wire_name().to_owned(),
        engine_version: services.engine_version.as_ref().to_owned(),
        decisions: decision_traces,
        timing_decisions,
        gap_healing: fa_params.gap_healing,
        fallback_events,
    })
}

mod incremental;
pub(crate) use incremental::process_fa_incremental;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_ops::fa::WordGapHealing;
    use talkbank_model::model::Line;

    /// An L2-valid file: participants, languages, terminator, main-tier
    /// content, and a `@Media` line so `FaOutput::processed` can reconcile.
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

    /// A fast-path-shaped result: no groups, built exactly as the `%wor`-reuse
    /// rerun route builds one.
    fn fast_path_result(file: crate::chat_ops::ChatFile) -> FaResult {
        FaResult::without_groups(
            file,
            WordGapHealing::PreserveMeasured,
            "test_engine",
            "test-build",
        )
        .expect("the fixture has one usable @Media declaration")
    }

    /// RED FIRST (2026-09-07 review, item 1): a document `align` declares it
    /// will not touch must be written back BYTE-IDENTICAL.
    ///
    /// The pass-through proof was built with the constructor now called
    /// `PostValidated::declined_stripping_decision_tiers`, which serializes
    /// the MODEL, so the
    /// bytes written were a parse-and-serialize round trip of the input rather
    /// than the input. This route judges nothing by design, so nothing was
    /// looking, and the trailing spaces on the `@Comment` line here were gone
    /// from the file the researcher got back after asking us to leave it alone.
    ///
    /// The `assert_ne!` is the precondition and it is what makes this test
    /// worth having: without a construct the round trip actually changes, the
    /// assertion below would hold under both behaviours.
    #[test]
    fn a_declared_pass_through_carries_the_input_bytes_not_a_reserialization() {
        const DUMMY: &str = "@UTF8\n@Begin\n@Options:\tdummy\n@Comment:\tkept   \n\
*PAR:\thello .\n@End\n";
        let parser = crate::chat_parser();
        let (chat_file, _errors) = batchalign_transform::parse::parse_lenient(&parser, DUMMY);
        assert_ne!(
            batchalign_transform::serialize::to_chat_string(&chat_file),
            DUMMY,
            "precondition: re-serializing this model is NOT the identity, so a \
             proof built from the model cannot be the input's bytes"
        );

        let admitted = FaAdmission::pass_through(
            chat_file,
            DUMMY,
            WordGapHealing::PreserveMeasured,
            "test_engine",
            "test-build",
        );

        assert_eq!(
            admitted.into_document().as_str(),
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
        let admitted = parse_aligned(ALIGNED);
        let admission = FaAdmission::admit(&admitted, &[]).expect("the fixture is L2-valid");

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

    /// RED FIRST (review item 2): the gate used to restate its level as
    /// `StructurallyComplete` while admission demanded `MainTierValid`, so a
    /// run that degraded its own input from L2 to L1 passed. The level now
    /// comes from the admission, so this output is refused.
    #[test]
    fn output_degraded_from_l2_to_l1_is_refused() {
        let admitted = parse_aligned(ALIGNED);
        let admission = FaAdmission::admit(&admitted, &[]).expect("the fixture is L2-valid");

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
             gate at the ADMITTED level can catch it"
        );

        let Err(failure) = admission.finish(fast_path_result(degraded)) else {
            panic!("output degraded below its admission level must be refused");
        };
        assert!(
            failure.to_string().contains("empty main tier"),
            "the refusal must name the L2 failure, got: {failure}"
        );
    }

    /// An output that still meets the bar its input was admitted at passes.
    #[test]
    fn an_undegraded_fast_path_result_is_admitted() {
        let admitted = parse_aligned(ALIGNED);
        let admission = FaAdmission::admit(&admitted, &[]).expect("the fixture is L2-valid");
        assert!(
            admission
                .finish(fast_path_result(parse_aligned(ALIGNED)))
                .is_ok(),
            "an unchanged output must still pass its own gate"
        );
    }

    #[test]
    fn cache_task_name_is_stable() {
        assert_eq!(CACHE_TASK.as_str(), "forced_alignment");
        assert_eq!(
            RAW_EVIDENCE_CACHE_TASK.as_str(),
            "forced_alignment_raw_evidence"
        );
    }

    #[test]
    fn collect_final_timings_rejects_missing_groups() {
        let error = collect_final_timings(vec![Some(Vec::new()), None], "forced alignment")
            .expect_err("missing timing groups should fail");
        assert!(
            error
                .to_string()
                .contains("completed without timings for group(s): [1]")
        );
    }

    #[test]
    fn collect_evidence_sources_rejects_missing_groups() {
        let error = collect_evidence_sources(
            vec![Some(FaEvidenceSourceTrace::Cache), None],
            "forced alignment",
        )
        .expect_err("missing evidence-source groups should fail");
        assert!(
            error
                .to_string()
                .contains("completed without an evidence source for group(s): [1]")
        );
    }

    #[test]
    fn group_evidence_assembly_rejects_parallel_cardinality_drift() {
        let error = assemble_group_evidence(
            Vec::new(),
            vec![FaEvidenceSourceTrace::Cache],
            Vec::new(),
            Vec::new(),
        )
        .expect_err("parallel FA evidence with different lengths must fail");

        assert!(error.to_string().contains("FA evidence cardinality drift"));
    }

    #[test]
    fn cached_group_timing_admission_refuses_legacy_bare_vectors() {
        let value = serde_json::json!([null]);

        let error = AdmittedCachedFaTimings::decode(
            value,
            crate::types::engines::FaEngineName::Wave2Vec,
            &crate::api::EngineVersion::from("test-fa-wave-v1"),
            1,
            &CacheKey::from_content("legacy-derived"),
        )
        .expect_err("an unversioned bare vector must not become replayable evidence");

        assert!(matches!(error, FaDerivedEvidenceError::InvalidJson(_)));
    }

    #[test]
    fn cached_group_timing_admission_accepts_an_exact_versioned_envelope() {
        use crate::api::DurationSeconds;
        use crate::types::engines::FaEngineName;
        use crate::types::worker_v2::{
            ExecuteResponseV2, IndexedWordTimingResultV2, TaskResultV2, WorkerRequestIdV2,
        };

        let key = CacheKey::from_content("derived-exact");
        let engine_version = crate::api::EngineVersion::from("test-fa-wave-v1");
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("derived-exact"),
            TaskResultV2::IndexedWordTimingResult(IndexedWordTimingResultV2 {
                indexed_timings: vec![None, None],
            }),
            DurationSeconds(0.01),
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
            request_engine_version: crate::api::EngineVersion::from("test-fa-wave-v1"),
            expected_words: 1,
            cache_key: key.clone(),
            timings: vec![None],
        })
        .expect("serialize versioned timing fixture");
        value["request_engine_version"] = serde_json::json!("test-fa-wave-v0");

        let error = AdmittedCachedFaTimings::decode(
            value,
            crate::types::engines::FaEngineName::Wave2Vec,
            &crate::api::EngineVersion::from("test-fa-wave-v1"),
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
        use crate::api::DurationSeconds;
        use crate::chat_ops::fa::{FaGroup, FaWord, TimeSpan};
        use crate::chat_ops::{UtteranceIdx, WordIdx};
        use crate::types::engines::FaEngineName;
        use crate::types::worker_v2::{
            ExecuteResponseV2, IndexedWordTimingResultV2, TaskResultV2, WorkerRequestIdV2,
        };

        let key = CacheKey::from_content("raw-first");
        let group = FaGroup::test_fixture(
            TimeSpan::new(100, 900),
            vec![FaWord {
                utterance_index: UtteranceIdx::new(0),
                utterance_word_index: WordIdx::new(0),
                text: "hello".to_owned(),
            }],
            vec![UtteranceIdx::new(0)],
        );
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("raw-first"),
            TaskResultV2::IndexedWordTimingResult(IndexedWordTimingResultV2 {
                indexed_timings: vec![None],
            }),
            DurationSeconds(0.01),
        );
        let raw = raw_evidence::FaRawEvidence::admit_requested(
            &response,
            FaEngineName::Wave2Vec,
            &crate::api::EngineVersion::from("test-fa-wave-v1"),
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

        let resolution = FaCacheGroupAdmission::new(
            &key,
            FaEngineName::Wave2Vec,
            &crate::api::EngineVersion::from("test-fa-wave-v1"),
            0,
            &group,
        )
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
