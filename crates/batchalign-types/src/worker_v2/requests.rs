//! Worker protocol V2 schema types shared across batchalign crates.
//!
//! These types define the next worker boundary described in
//! `book/src/batchalign/developer/worker-protocol-v2.md`. Unlike the current
//! JSON-lines protocol in [`super::worker`], this schema is intentionally
//! staged for migration:
//!
//! - the types are drift-tested against Python
//! - canonical fixtures live under `tests/fixtures/worker_protocol_v2/`
//! - production code now dispatches FA, ASR, and speaker requests through
//!   these typed envelopes, while the remaining tasks are still staged
//!
//! The design goal is to keep Python as a thin model host while Rust owns
//! preprocessing, postprocessing, document semantics, and cache policy.
//!
//! ## Timing field validation contract
//!
//! Several response structs carry floating-point or integer timing fields
//! (`start_s`, `end_s`, `time_s`, `start_ms`, `end_ms`).  On the Python side,
//! Pydantic V2 models in `_types_v2.py` enforce upstream validation:
//! non-finite values (NaN, ±Inf) are rejected, and reversed ranges
//! (`start > end`) are rejected via `@model_validator`.  Rust deserializes
//! these fields permissively: `serde_json` will accept any valid JSON number
//!, because the Python worker has already sanitised the data before it
//! reaches the wire.  If a new producer is added that bypasses Python
//! validation, Rust-side checks must be added to the affected structs.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::api::{DurationSeconds, EngineVersion, LanguageCode3, NumSpeakers, WorkerLanguage};
use crate::worker::WorkerPid;

string_id!(
    /// Stable identifier for one V2 protocol request/response pair.
    pub WorkerRequestIdV2
);

string_id!(
    /// Stable identifier for one prepared worker artifact.
    pub WorkerArtifactIdV2
);

string_id!(
    /// Filesystem path to a prepared worker artifact.
    pub WorkerArtifactPathV2
);

string_id!(
    /// Caller-chosen name for one span of a speaker-embedding request.
    ///
    /// Echoed verbatim in the response so the control plane pairs an outcome
    /// with the span it asked about BY NAME. Pairing two sequences by position
    /// is the shape that silently attributes one speaker's acoustic evidence
    /// to another utterance, and no arity check catches a reordering.
    pub SpeakerEmbeddingSpanIdV2
);

numeric_id!(
    /// Worker protocol major version.
    pub WorkerProtocolVersionV2(u16) [Eq]
);

numeric_id!(
    /// Audio sample rate in Hz carried by prepared artifacts.
    pub SampleRateHzV2(u32) [Eq]
);

numeric_id!(
    /// Number of channels in a prepared audio artifact.
    pub ChannelCountV2(u16) [Eq]
);

numeric_id!(
    /// Number of audio frames in a prepared artifact.
    pub FrameCountV2(u64) [Eq]
);

numeric_id!(
    /// Byte offset inside a prepared artifact file.
    pub ByteOffsetV2(u64) [Eq]
);

numeric_id!(
    /// Byte length inside a prepared artifact file.
    pub ByteLengthV2(u64) [Eq]
);

/// Wall-clock decode budget for one ASR request, in seconds.
///
/// Named apart from a bare `DurationSeconds` because this quantity
/// carries a specific provenance contract: it is derived once, at
/// request-build time, from the audio's own duration and
/// [`DEADLINE_REALTIME_FACTOR`], and it is the value both the Python
/// decode loop (`_qwen_chunking.DecodeBudget`) and the Rust transport
/// ceiling (`TaskRequestV2::timeout_seconds_with_config`) are computed
/// FROM. A generic duration carries no such contract, so reusing one
/// here would let a caller pass an unrelated seconds value where this
/// one is required.
///
/// HAND-WRITTEN rather than built with `numeric_id!`: that macro gives its
/// type a PUBLIC inner field plus `From<f64>` and `Default` (a zero budget
/// any caller could construct from nothing), which would make the "only
/// sanctioned constructor" claim on [`Self::for_audio`] false -- a
/// `DecodeBudgetSeconds(1.0)` or `DecodeBudgetSeconds::default()` written
/// anywhere in the crate would compile and produce a budget with no
/// audio duration behind it. The field here is private; [`Self::for_audio`]
/// and [`Self::for_duration_ms`] are the only ways to construct one, and
/// [`Self::as_seconds`] is the only way to read one back out.
///
/// The tuple constructor is unreachable from outside this module (proved
/// by the compile_fail doctest below); `Default` and `From<f64>` are
/// proved absent crate-wide by a `static_assertions::assert_not_impl_any!`
/// beneath this type, which runs on every `cargo test --lib`.
///
/// ```compile_fail
/// use batchalign_types::worker_v2::DecodeBudgetSeconds;
/// // The inner field is private: this must not compile from outside
/// // `batchalign_types::worker_v2::requests`.
/// let _ = DecodeBudgetSeconds(1.0);
/// ```
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    PartialOrd,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
#[serde(transparent)]
pub struct DecodeBudgetSeconds(f64);

// Compile-time proof, checked on every `cargo test`/`cargo check --tests`
// (dev-dependency, so `#[cfg(test)]`; the plain `cargo build`/library path
// does not compile this crate's dev-dependencies at all): neither
// `Default` nor `From<f64>` exists for this type. Either would let a
// caller fabricate a budget with no audio duration behind it -- `Default`
// a silent zero, `From<f64>` an arbitrary one -- exactly the hole
// `numeric_id!` used to leave open. If either is ever (re)implemented,
// this line stops compiling, immediately, the next time anyone runs
// `cargo test -p batchalign-types --lib` (already one of this crate's
// standing gates, not a separate harness to remember).
#[cfg(test)]
static_assertions::assert_not_impl_any!(DecodeBudgetSeconds: Default, From<f64>);

/// Wall-clock decode-budget seconds per second of audio.
///
/// Must equal Python's `DEADLINE_REALTIME_FACTOR` in
/// `batchalign/inference/languages/cantonese/_qwen_chunking.py`
/// (`_MEASURED_REALTIME_FACTOR * _REALTIME_FACTOR_SAFETY_MARGIN` there).
/// The two definitions are pinned together by a conformance test in
/// `batchalign/tests/test_ipc_type_conformance.py`
/// (`test_decode_budget_realtime_factor_matches_rust`) rather than by a
/// single generated source: the Python V2 models are deliberately
/// hand-written (see `scripts/generate_ipc_types.sh`), so there is no
/// existing codegen seam that would carry a bare numeric constant across
/// the language boundary. Update both together.
pub const DEADLINE_REALTIME_FACTOR: f64 = 12.8;

/// Seconds added atop a request's own decode budget when deriving the
/// worker-transport read timeout.
///
/// The decode budget alone bounds only the model's own decode loop; the
/// transport ceiling additionally has to cover audio load, alignment
/// postprocessing, and IPC framing around that loop, so it must always be
/// strictly larger than the budget it was computed from.
pub const TRANSPORT_CEILING_MARGIN_SECONDS: u64 = 300;

/// Transport ceiling used for an ASR request whose audio duration is not
/// knowable to Rust before dispatch (a provider-media request whose file
/// could not be probed).
///
/// Generous on purpose, the same posture the flat 1800s ceiling this
/// replaces used to take for every request: this is a fallback for the
/// case where [`DecodeBudgetSeconds`] genuinely cannot be derived, not a
/// normal-case value, so it errs toward "wait longer" rather than toward a
/// tight number a real long file could exceed.
pub const PROVIDER_MEDIA_ASR_TRANSPORT_CEILING_SECONDS: u64 = 3600;

impl DecodeBudgetSeconds {
    /// Derive a request's decode budget from a prepared audio artifact's
    /// own duration.
    ///
    /// The only sanctioned constructor: every caller with a known audio
    /// duration goes through this so the budget and the realtime factor it
    /// was computed from can never drift apart at different call sites. A
    /// zero sample rate (never a real artifact, but not excluded by
    /// [`SampleRateHzV2`]'s own type) yields a zero budget rather than
    /// dividing by zero.
    pub fn for_audio(frame_count: FrameCountV2, sample_rate_hz: SampleRateHzV2) -> Self {
        let audio_seconds = if sample_rate_hz.0 == 0 {
            0.0
        } else {
            frame_count.0 as f64 / f64::from(sample_rate_hz.0)
        };
        Self(audio_seconds * DEADLINE_REALTIME_FACTOR)
    }

    /// Derive a request's decode budget from a probed duration in
    /// milliseconds (the shape [`crate::api::DurationMs`]-probing call
    /// sites already have).
    pub fn for_duration_ms(duration_ms: u64) -> Self {
        Self((duration_ms as f64 / 1000.0) * DEADLINE_REALTIME_FACTOR)
    }

    /// The transport read-timeout ceiling for a request carrying this
    /// budget: the budget itself plus the named margin, rounded up to the
    /// next whole second so the transport is never shorter than the
    /// fractional-second budget it was computed from.
    pub fn transport_ceiling_seconds(self) -> u64 {
        let budget_seconds = self.as_seconds().max(0.0).ceil() as u64;
        budget_seconds.saturating_add(TRANSPORT_CEILING_MARGIN_SECONDS)
    }

    /// The budget's raw seconds value.
    ///
    /// The only sanctioned way to read a budget back out: needed by
    /// `transport_ceiling_seconds` above and by the PyO3 bridge
    /// (`batchalign-pyo3::worker_asr_exec`), which forwards this number
    /// to Python's `AsrBatchItem.decode_budget_seconds` verbatim. A read
    /// accessor does not reopen the construction hole the private field
    /// closes: it cannot manufacture a budget, only report one that was
    /// already built by `for_audio` or `for_duration_ms`.
    pub fn as_seconds(self) -> f64 {
        self.0
    }
}

/// Worker role selected during the protocol handshake.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WorkerKindV2 {
    /// Stateless inference worker process.
    Infer,
}

/// High-level V2 task family.
#[derive(
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum InferenceTaskV2 {
    /// Morphosyntax tagging.
    Morphosyntax,
    /// Utterance segmentation.
    Utseg,
    /// Machine translation.
    Translate,
    /// Coreference annotation.
    Coref,
    /// Automatic speech recognition.
    Asr,
    /// Forced alignment.
    ForcedAlignment,
    /// Speaker diarization.
    Speaker,
    /// Speaker embedding over named spans of one prepared recording.
    SpeakerEmbedding,
    /// OpenSMILE feature extraction.
    Opensmile,
    /// AVQI feature extraction.
    Avqi,
}

/// ASR backend selected by the Rust control plane.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AsrBackendV2 {
    /// Local Whisper runtime hosted in Python.
    LocalWhisper,
    /// HuggingFace Whisper community fine-tune, resolved per-language.
    /// The same worker-side ``WhisperASRHandle`` hosts the model; this
    /// enum variant exists so the control-plane pool key and the worker
    /// bootstrap can select the fine-tune loader over the stock loader.
    WhisperHub,
    /// Tencent Cantonese ASR provider.
    HkTencent,
    /// Aliyun Cantonese ASR provider.
    HkAliyun,
    /// FunASR Cantonese provider.
    HkFunaudio,
    /// Qwen3-ASR Cantonese provider (local model via qwen-asr package).
    HkQwen,
    /// Rev.AI provider.
    Revai,
}

impl AsrBackendV2 {
    /// Every backend the ASR wire can name.
    ///
    /// THE owner of "which ASR backends exist", for the same reason
    /// [`FaBackendV2::ALL`] is: the fixture-coverage test used to iterate four
    /// string literals, so three of these seven had no request fixture and
    /// nothing said so.
    pub const ALL: [Self; 7] = [
        Self::LocalWhisper,
        Self::WhisperHub,
        Self::HkTencent,
        Self::HkAliyun,
        Self::HkFunaudio,
        Self::HkQwen,
        Self::Revai,
    ];
}

/// Forced-alignment backend selected by the Rust control plane.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FaBackendV2 {
    /// Whisper token-timestamp alignment.
    Whisper,
    /// MMS Wave2Vec forced alignment.
    Wave2vec,
    /// Cantonese Wave2Vec forced alignment.
    Wav2vecCanto,
    /// Qwen3 forced alignment, the standalone form of the aligner the
    /// Qwen3-ASR engine loads for its own word timestamps.
    Qwen3,
}

impl FaBackendV2 {
    /// Every backend the FA wire can name.
    ///
    /// THE owner of "which FA backends exist" for tests that must cover the
    /// whole matrix. The fixture-coverage test in
    /// `crates/batchalign/tests/worker_protocol_matrix.rs` iterates this
    /// instead of a hand-written list of three literals, which is how a fourth
    /// backend came to have no request fixture while the test stayed green.
    pub const ALL: [Self; 4] = [
        Self::Whisper,
        Self::Wave2vec,
        Self::Wav2vecCanto,
        Self::Qwen3,
    ];
}

/// Speaker diarization backend selected by Rust.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerBackendV2 {
    /// pyannoteAI cloud diarization using the Precision-2 model.
    PyannoteAi,
    /// Pyannote diarization backend.
    Pyannote,
    /// NeMo diarization backend.
    Nemo,
}

impl SpeakerBackendV2 {
    /// Every backend the speaker wire can name.
    ///
    /// THE owner of "which speaker backends exist"; the fixture-coverage test
    /// iterated two literals and so never noticed that the DEFAULT engine,
    /// `pyannote_ai`, had no request fixture.
    pub const ALL: [Self; 3] = [Self::PyannoteAi, Self::Pyannote, Self::Nemo];
}

/// Speaker-embedding backend selected by the Rust control plane.
///
/// One variant today, and deliberately an enum rather than an implied
/// constant: the wire already has to say which acoustic model produced a
/// vector, because vectors from two models are not comparable and a cache that
/// could not tell them apart would replay one against the other.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerEmbeddingBackendV2 {
    /// The embedding node of the released local Pyannote model graph.
    Pyannote,
}

/// Small artifact-kind vocabulary advertised in task capabilities.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WorkerAttachmentKindV2 {
    /// File-backed prepared PCM audio.
    PreparedAudio,
    /// File-backed prepared text/JSON.
    PreparedText,
    /// Inline JSON attachment carried inside the envelope.
    InlineJson,
    /// Provider-local media path that Rust still has not replaced.
    ProviderMedia,
    /// Previously submitted provider job identifier.
    SubmittedJob,
}

/// PCM encoding used for prepared audio artifacts.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PreparedAudioEncodingV2 {
    /// Little-endian float32 PCM frames.
    PcmF32le,
}

/// Encoding used for prepared text artifacts.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PreparedTextEncodingV2 {
    /// UTF-8 JSON text stored on disk.
    Utf8Json,
}

/// Error category for protocol-level failures.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProtocolErrorCodeV2 {
    /// Worker/runtime does not understand the requested protocol version.
    UnsupportedProtocol,
    /// Request payload shape was invalid for the task.
    InvalidPayload,
    /// Required attachment was not supplied.
    MissingAttachment,
    /// Attachment existed logically but could not be read.
    AttachmentUnreadable,
    /// Model or SDK runtime for the task is unavailable.
    ModelUnavailable,
    /// Runtime failed while executing the task.
    RuntimeFailure,
    /// A pinned Hugging Face Hub artifact refused this machine's request: a
    /// gated repository requiring accepted terms, a missing/invalid token,
    /// or no cached copy while offline. Distinct from `RuntimeFailure` so
    /// callers can categorize it as a configuration/credential condition on
    /// the OPERATOR's machine rather than a batchalign defect.
    ModelAccessDenied,
}

impl ProtocolErrorCodeV2 {
    /// The wire spelling of this code, as it appears in serialized responses.
    ///
    /// Exists so callers that must hand the category to another language (the
    /// PyO3 boundary passes it to the Python worker layer) can name it without
    /// inventing a second spelling. The test below pins every variant against
    /// what serde actually emits, which is the one case where a second
    /// representation earns a test: this is a WIRE FORMAT, and no type can
    /// state that a `#[serde(rename_all)]` attribute and a match arm agree.
    #[must_use]
    pub const fn as_wire_str(self) -> &'static str {
        match self {
            Self::UnsupportedProtocol => "unsupported_protocol",
            Self::InvalidPayload => "invalid_payload",
            Self::MissingAttachment => "missing_attachment",
            Self::AttachmentUnreadable => "attachment_unreadable",
            Self::ModelUnavailable => "model_unavailable",
            Self::RuntimeFailure => "runtime_failure",
            Self::ModelAccessDenied => "model_access_denied",
        }
    }
}

#[cfg(test)]
mod protocol_error_code_tests {
    use super::ProtocolErrorCodeV2;

    /// Every variant's `as_wire_str` must equal what serde serializes it to.
    /// Exhaustive by construction: adding a variant without extending the list
    /// leaves the new arm unproven, and adding one without extending
    /// `as_wire_str` fails to compile.
    #[test]
    fn wire_spelling_matches_serde() {
        let all = [
            ProtocolErrorCodeV2::UnsupportedProtocol,
            ProtocolErrorCodeV2::InvalidPayload,
            ProtocolErrorCodeV2::MissingAttachment,
            ProtocolErrorCodeV2::AttachmentUnreadable,
            ProtocolErrorCodeV2::ModelUnavailable,
            ProtocolErrorCodeV2::RuntimeFailure,
            ProtocolErrorCodeV2::ModelAccessDenied,
        ];
        for code in all {
            // Compared as `Option`, so a serialization failure shows up as a
            // mismatch rather than needing an `expect` the crate's lints ban.
            assert_eq!(
                serde_json::to_string(&code).ok(),
                Some(format!("\"{}\"", code.as_wire_str())),
                "wire spelling drifted for {code:?}"
            );
        }
    }
}

/// Text-joining mode for forced-alignment payloads.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FaTextModeV2 {
    /// Join words with spaces before model invocation.
    SpaceJoined,
    /// Join words as character stream.
    CharJoined,
    /// Space-join the words, then separate every CHARACTER with a space.
    ///
    /// Gives the decoder one token per character, so its onsets land inside
    /// words rather than only at their starts, which is what makes a real
    /// silence visible as a gap. Selected by `--pauses`.
    ///
    /// This was a second boolean on the request until 2026-08-14, applied by
    /// the Python host AFTER the mode above had already joined the text. Two
    /// fields decided one thing, and the boolean was documented as preserving
    /// pause markers, which it never did.
    CharSpaced,
}

/// Runtime information returned during the V2 handshake.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct WorkerRuntimeInfoV2 {
    /// Python runtime version used by the worker.
    pub python_version: String,
    /// Whether the runtime is free-threaded.
    pub free_threaded: bool,
}

/// Initial V2 handshake request sent by Rust.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct HelloRequestV2 {
    /// Requested protocol version.
    pub protocol_version: WorkerProtocolVersionV2,
    /// Worker role the parent process expects.
    pub worker_kind: WorkerKindV2,
}

/// Initial V2 handshake response sent by Python.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct HelloResponseV2 {
    /// Agreed protocol version.
    pub protocol_version: WorkerProtocolVersionV2,
    /// OS process id of the worker.
    pub worker_pid: WorkerPid,
    /// Runtime metadata needed by the pool.
    pub runtime: WorkerRuntimeInfoV2,
}

/// Request for task capability metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct CapabilitiesRequestV2 {
    /// Correlation id for the capability lookup.
    pub request_id: WorkerRequestIdV2,
}

/// One task capability advertised by a V2 worker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct TaskCapabilityV2 {
    /// Task family supported by the worker.
    pub task: InferenceTaskV2,
    /// Attachment/input kinds the task can consume.
    pub accepted_inputs: Vec<WorkerAttachmentKindV2>,
    /// Whether the task can emit progress events.
    pub supports_progress_events: bool,
}

/// Response describing task capabilities for the worker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct CapabilitiesResponseV2 {
    /// Correlation id that matches the request.
    pub request_id: WorkerRequestIdV2,
    /// Task capabilities advertised by the runtime.
    pub tasks: Vec<TaskCapabilityV2>,
    /// Engine version strings keyed by task name.
    pub engine_versions: BTreeMap<String, EngineVersion>,
}

/// File-backed prepared audio artifact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct PreparedAudioRefV2 {
    /// Stable artifact id referenced by request payloads.
    pub id: WorkerArtifactIdV2,
    /// Filesystem path to the prepared artifact.
    pub path: WorkerArtifactPathV2,
    /// PCM encoding for the prepared audio.
    pub encoding: PreparedAudioEncodingV2,
    /// Number of channels in the artifact view.
    pub channels: ChannelCountV2,
    /// Sample rate in Hz.
    pub sample_rate_hz: SampleRateHzV2,
    /// Number of frames in the view.
    pub frame_count: FrameCountV2,
    /// Byte offset inside the artifact file.
    pub byte_offset: ByteOffsetV2,
    /// Byte length inside the artifact file.
    pub byte_len: ByteLengthV2,
}

/// File-backed prepared text artifact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct PreparedTextRefV2 {
    /// Stable artifact id referenced by request payloads.
    pub id: WorkerArtifactIdV2,
    /// Filesystem path to the prepared artifact.
    pub path: WorkerArtifactPathV2,
    /// Encoding used by the file content.
    pub encoding: PreparedTextEncodingV2,
    /// Byte offset inside the artifact file.
    pub byte_offset: ByteOffsetV2,
    /// Byte length inside the artifact file.
    pub byte_len: ByteLengthV2,
}

/// Small inline JSON attachment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct InlineJsonRefV2 {
    /// Stable artifact id referenced by request payloads.
    pub id: WorkerArtifactIdV2,
    /// Inline JSON payload carried with the envelope.
    pub value: serde_json::Value,
}

/// Prepared artifact reference carried alongside one execute request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ArtifactRefV2 {
    /// Prepared PCM audio view.
    PreparedAudio(PreparedAudioRefV2),
    /// Prepared UTF-8 JSON or text view.
    PreparedText(PreparedTextRefV2),
    /// Small inline JSON attachment.
    InlineJson(InlineJsonRefV2),
}

/// Request-time reference to a prepared audio artifact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct PreparedAudioInputV2 {
    /// Artifact id of the audio descriptor included in `attachments`.
    pub audio_ref_id: WorkerArtifactIdV2,
}

/// Temporary cloud-provider media input retained during migration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct ProviderMediaInputV2 {
    /// Media file path readable by the worker host.
    pub media_path: WorkerArtifactPathV2,
    /// Expected number of speakers for diarization-aware providers.
    pub num_speakers: NumSpeakers,
}

/// Previously submitted provider job id.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct SubmittedJobInputV2 {
    /// Provider job identifier to poll.
    pub provider_job_id: WorkerArtifactIdV2,
}

/// ASR input variants for V2.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AsrInputV2 {
    /// Local prepared audio path.
    PreparedAudio(PreparedAudioInputV2),
    /// Provider-local media path.
    ProviderMedia(ProviderMediaInputV2),
    /// Previously submitted provider job.
    SubmittedJob(SubmittedJobInputV2),
}

/// V2 ASR request payload.
///
/// No `Eq`: `decode_budget_seconds` is derived from a floating-point audio
/// duration, and `f64` cannot support a total-equality contract (NaN != NaN).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct AsrRequestV2 {
    /// Worker-runtime language hint for the transcript.
    ///
    /// This may be a concrete ISO 639-3 code or the `"auto"` worker sentinel
    /// used by local Whisper auto-detect.
    pub lang: WorkerLanguage,
    /// Backend selected by Rust.
    pub backend: AsrBackendV2,
    /// Backend-specific input transport.
    pub input: AsrInputV2,
    /// Per-engine configuration extras (e.g. `qwen_model`,
    /// `qwen_device`, `funaudio_model`). Opaque string-keyed map carried
    /// verbatim from the user's `--engine-overrides` JSON through every
    /// dispatch layer down to the worker spawn argv and the Python
    /// engine-load function. Empty when no per-engine knob is set.
    ///
    /// Why this lives on the typed V2 request rather than only on
    /// `EngineOverrides`: a CLI override like
    /// `{"asr":"qwen","qwen_model":"Qwen/Qwen3-ASR-0.6B"}` would otherwise
    /// be silently truncated to `{"asr":"qwen"}` at this typed boundary,
    /// and the worker would default to the 1.7B model regardless of what
    /// the user asked for, the bug fixed 2026-05-27. The `#[serde(default)]`
    /// keeps older daemons that don't emit the field forward-compatible.
    #[serde(
        default,
        deserialize_with = "null_as_empty_string_map",
        skip_serializing_if = "std::collections::BTreeMap::is_empty"
    )]
    pub extras: std::collections::BTreeMap<String, String>,
    /// The request's own wall-clock decode budget, derived once at
    /// request-build time from the audio's duration and
    /// [`DEADLINE_REALTIME_FACTOR`].
    ///
    /// `None` means the audio duration was not knowable to Rust before
    /// dispatch (see the request builder in
    /// `batchalign::worker::asr_request_v2`), never "no budget applies";
    /// the transport ceiling falls back to
    /// [`PROVIDER_MEDIA_ASR_TRANSPORT_CEILING_SECONDS`] in that case. When
    /// present, this is also the value Python's native Qwen3-ASR decode
    /// loop uses in place of re-deriving its own budget from the file it
    /// is given (`_qwen_chunking.DecodeBudget`), so the two ceilings are
    /// computed from one duration instead of drifting apart.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_budget_seconds: Option<DecodeBudgetSeconds>,
}

/// Deserialize an optional string map where JSON `null` means "empty".
///
/// The schemars output for a `default + skip_serializing_if` map field is
/// an OPTIONAL field, and the Python models generated from that schema
/// spell "absent" as `None`, which crosses the PyO3 JSON bridge as
/// explicit `null`. Plain `#[serde(default)]` only covers the MISSING
/// case, so `null` used to fail with "invalid type: null, expected a
/// map"; both spellings must mean "no extras".
fn null_as_empty_string_map<'de, D>(
    deserializer: D,
) -> Result<std::collections::BTreeMap<String, String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<std::collections::BTreeMap<String, String>>::deserialize(deserializer)?;
    Ok(value.unwrap_or_default())
}

/// V2 forced-alignment request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct ForcedAlignmentRequestV2 {
    /// Backend selected by Rust.
    pub backend: FaBackendV2,
    /// Reference to the prepared text/JSON payload for the word arrays.
    pub payload_ref_id: WorkerArtifactIdV2,
    /// Reference to the prepared audio span.
    pub audio_ref_id: WorkerArtifactIdV2,
    /// Text shaping mode requested by Rust.
    pub text_mode: FaTextModeV2,
}

/// V2 morphosyntax request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct MorphosyntaxRequestV2 {
    /// Primary language routed by Rust.
    pub lang: LanguageCode3,
    /// Reference to the prepared text batch payload.
    pub payload_ref_id: WorkerArtifactIdV2,
    /// Number of utterance items frozen into the prepared batch payload.
    pub item_count: u32,
    /// Whether Stanza/PyCantonese should re-tokenize the input.
    ///
    /// When `true`, CJK word segmentation is applied before POS tagging:
    /// - Cantonese (`yue`): PyCantonese `segment()` groups characters into words
    /// - Mandarin (`cmn`/`zho`): Stanza neural tokenizer segments the text
    ///
    /// Defaults to `false` for backward compatibility with workers that do not
    /// yet understand this field.
    #[serde(default)]
    pub retokenize: bool,
}

/// V2 utterance-segmentation request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct UtsegRequestV2 {
    /// Primary language routed by Rust.
    pub lang: LanguageCode3,
    /// Reference to the prepared text batch payload.
    pub payload_ref_id: WorkerArtifactIdV2,
    /// Number of utterance items frozen into the prepared batch payload.
    pub item_count: u32,
    /// Operator opt-in to the legacy Stanza constituency-parser
    /// fallback for unsupported languages. Set by the
    /// `--utseg-fallback-stanza` CLI flag. Defaults to `false` so
    /// older clients (and any non-CLI caller) refuse silent
    /// substitution by default.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub allow_stanza_fallback: bool,
}

/// V2 translation request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct TranslateRequestV2 {
    /// Source language determined by Rust.
    pub source_lang: LanguageCode3,
    /// Target language requested by Rust.
    pub target_lang: LanguageCode3,
    /// Reference to the prepared text batch payload.
    pub payload_ref_id: WorkerArtifactIdV2,
    /// Number of utterance items frozen into the prepared batch payload.
    pub item_count: u32,
}

/// V2 coreference request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct CorefRequestV2 {
    /// Primary language routed by Rust.
    pub lang: LanguageCode3,
    /// Reference to the prepared text batch payload.
    pub payload_ref_id: WorkerArtifactIdV2,
    /// Number of document items frozen into the prepared batch payload.
    pub item_count: u32,
}

/// V2 speaker diarization request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct SpeakerRequestV2 {
    /// Backend selected by Rust.
    pub backend: SpeakerBackendV2,
    /// Input transport for the speaker runtime.
    pub input: SpeakerInputV2,
    /// Expected number of speakers when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_speakers: Option<NumSpeakers>,
}

/// One span of the prepared recording to embed, in PREPARED-AUDIO FRAMES.
///
/// Frames, not milliseconds. The prepared PCM view is the only coordinate
/// system the worker holds, and the conversion from the recording's file
/// milliseconds happens once, on the Rust side, at the single place that owns
/// both the sample rate and the frame count. A request carrying milliseconds
/// would make the worker re-derive a frame index from a rate it was told about
/// separately, which is the same number arriving twice by two routes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct SpeakerEmbeddingSpanV2 {
    /// Caller-chosen name echoed in the response.
    pub span_id: SpeakerEmbeddingSpanIdV2,
    /// First frame of the span, inclusive.
    pub start_frame: FrameCountV2,
    /// Last frame of the span, exclusive.
    pub end_frame: FrameCountV2,
}

/// V2 speaker-embedding request payload.
///
/// Exactly ONE prepared-audio attachment carries the whole recording, and
/// every span indexes into it. That is not a transport optimisation: two
/// embeddings computed from separately decoded files can differ for reasons
/// that have nothing to do with who was speaking, so enrollment material and
/// the utterances scored against it have to come from the same decode.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct SpeakerEmbeddingRequestV2 {
    /// Backend selected by Rust.
    pub backend: SpeakerEmbeddingBackendV2,
    /// Artifact id of the prepared mono PCM view of the whole recording.
    pub audio_ref_id: WorkerArtifactIdV2,
    /// Spans to embed, in the order the response must echo them.
    pub spans: Vec<SpeakerEmbeddingSpanV2>,
}

/// V2 openSMILE request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct OpenSmileRequestV2 {
    /// Reference to the prepared audio attachment.
    pub audio_ref_id: WorkerArtifactIdV2,
    /// Requested openSMILE feature-set name.
    pub feature_set: String,
    /// Requested openSMILE feature-level name.
    pub feature_level: String,
}

/// V2 AVQI request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct AvqiRequestV2 {
    /// Reference to the prepared continuous-speech audio attachment.
    pub cs_audio_ref_id: WorkerArtifactIdV2,
    /// Reference to the prepared sustained-vowel audio attachment.
    pub sv_audio_ref_id: WorkerArtifactIdV2,
}

/// Prepared-audio speaker input owned by Rust.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct SpeakerPreparedAudioInputV2 {
    /// Artifact id of the prepared mono PCM audio view.
    pub audio_ref_id: WorkerArtifactIdV2,
}

/// Current input variants for speaker diarization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpeakerInputV2 {
    /// Prepared mono PCM audio owned by Rust.
    PreparedAudio(SpeakerPreparedAudioInputV2),
}

/// Typed execute payload carried by one V2 request.
///
/// No `Eq`: the `Asr` variant carries a floating-point decode budget (see
/// `AsrRequestV2`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskRequestV2 {
    /// Automatic speech recognition request.
    Asr(AsrRequestV2),
    /// Forced-alignment request.
    ForcedAlignment(ForcedAlignmentRequestV2),
    /// Morphosyntax request.
    Morphosyntax(MorphosyntaxRequestV2),
    /// Utterance-segmentation request.
    Utseg(UtsegRequestV2),
    /// Translation request.
    Translate(TranslateRequestV2),
    /// Coreference request.
    Coref(CorefRequestV2),
    /// Speaker diarization request.
    Speaker(SpeakerRequestV2),
    /// Speaker-embedding request.
    SpeakerEmbedding(SpeakerEmbeddingRequestV2),
    /// OpenSMILE request.
    Opensmile(OpenSmileRequestV2),
    /// AVQI request.
    Avqi(AvqiRequestV2),
}

/// One top-level V2 execution request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct ExecuteRequestV2 {
    /// Correlation id for the request.
    pub request_id: WorkerRequestIdV2,
    /// Task family being executed.
    pub task: InferenceTaskV2,
    /// Typed task payload.
    pub payload: TaskRequestV2,
    /// Prepared artifacts attached to the request.
    pub attachments: Vec<ArtifactRefV2>,
}

impl ExecuteRequestV2 {
    /// Return the timeout budget this request should receive on the worker
    /// transport.
    pub fn timeout_seconds(&self) -> u64 {
        self.payload.timeout_seconds()
    }

    /// Return the timeout with optional config overrides for audio and
    /// analysis tasks.
    pub fn timeout_seconds_with_config(
        &self,
        audio_timeout_s: u64,
        analysis_timeout_s: u64,
    ) -> u64 {
        self.payload
            .timeout_seconds_with_config(audio_timeout_s, analysis_timeout_s)
    }
}

impl TaskRequestV2 {
    /// The task family this payload belongs to, derived from the variant.
    ///
    /// [`ExecuteRequestV2`] carries the discriminant twice: as `task` and again
    /// as the payload variant. This accessor is what lets the agreement between
    /// them be checked in ONE place (the executor control plane) instead of
    /// re-checked by every executor with its own message.
    pub fn task(&self) -> InferenceTaskV2 {
        match self {
            Self::Asr(_) => InferenceTaskV2::Asr,
            Self::ForcedAlignment(_) => InferenceTaskV2::ForcedAlignment,
            Self::Morphosyntax(_) => InferenceTaskV2::Morphosyntax,
            Self::Utseg(_) => InferenceTaskV2::Utseg,
            Self::Translate(_) => InferenceTaskV2::Translate,
            Self::Coref(_) => InferenceTaskV2::Coref,
            Self::Speaker(_) => InferenceTaskV2::Speaker,
            Self::SpeakerEmbedding(_) => InferenceTaskV2::SpeakerEmbedding,
            Self::Opensmile(_) => InferenceTaskV2::Opensmile,
            Self::Avqi(_) => InferenceTaskV2::Avqi,
        }
    }

    /// Return the timeout budget this task family should receive on the worker
    /// transport.
    pub fn timeout_seconds(&self) -> u64 {
        self.timeout_seconds_with_config(0, 0)
    }

    /// Return the timeout with optional config overrides.
    ///
    /// When `audio_timeout_s` or `analysis_timeout_s` is 0, the built-in
    /// defaults (1800 and 120) are used.
    pub fn timeout_seconds_with_config(
        &self,
        audio_timeout_s: u64,
        analysis_timeout_s: u64,
    ) -> u64 {
        match self {
            Self::Morphosyntax(request) => batched_text_timeout_seconds(request.item_count),
            Self::Utseg(request) => batched_text_timeout_seconds(request.item_count),
            Self::Translate(request) => batched_text_timeout_seconds(request.item_count),
            Self::Coref(request) => batched_text_timeout_seconds(request.item_count),
            // The ASR request carries its own decode budget, derived from
            // the audio's actual duration (see `DecodeBudgetSeconds`), so
            // its transport ceiling scales with the file rather than using
            // one flat number for a 30s clip and a 2-hour recording alike.
            // `audio_timeout_s` remains available as an operator override,
            // but it can only RAISE the ceiling above what the request's
            // own budget needs, never lower it below.
            Self::Asr(request) => {
                let derived = match request.decode_budget_seconds {
                    Some(budget) => budget.transport_ceiling_seconds(),
                    None => PROVIDER_MEDIA_ASR_TRANSPORT_CEILING_SECONDS,
                };
                derived.max(audio_timeout_s)
            }
            // Forced alignment and speaker diarization do not yet carry a
            // request-level duration the way ASR now does. A generous flat
            // ceiling remains here as their transport bound; scoped out of
            // the ASR decode-budget change (2026-09-02).
            Self::ForcedAlignment(_) | Self::Speaker(_) => {
                if audio_timeout_s > 0 {
                    audio_timeout_s
                } else {
                    1800
                }
            }
            // Speaker embedding scales with the NUMBER OF SPANS rather than
            // with the recording, because the model runs once per span over a
            // few seconds of audio each and never over the whole file. The
            // shape is the batched-text one, not the whole-recording one, so
            // it reuses that budget rather than inheriting a flat 1800 that
            // says nothing about the work actually asked for.
            Self::SpeakerEmbedding(request) => {
                let per_span = (request.spans.len() as u64).saturating_mul(5);
                per_span.max(120).max(audio_timeout_s)
            }
            // Lightweight audio analysis: 120s is sufficient.
            Self::Opensmile(_) | Self::Avqi(_) => {
                if analysis_timeout_s > 0 {
                    analysis_timeout_s
                } else {
                    120
                }
            }
        }
    }
}

/// Return the timeout budget for one batched text-inference request.
fn batched_text_timeout_seconds(item_count: u32) -> u64 {
    u64::from(item_count).saturating_mul(5).max(120)
}

/// One raw Whisper chunk span returned by Python.
///
/// Timing fields validated upstream by Python Pydantic models (see module docs).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct WhisperChunkSpanV2 {
    /// Surface text for the chunk.
    pub text: String,
    /// Start timestamp in seconds.
    pub start_s: DurationSeconds,
    /// End timestamp in seconds.
    pub end_s: DurationSeconds,
}

#[cfg(test)]
// Test code: the panic-family lints are relaxed in source by house policy;
// the workspace [lints] table holds production code to deny. This module was
// missing the header its siblings carry, so it warned on every clippy run.
#[allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
mod tests {
    #[test]
    fn asr_request_extras_accepts_null_missing_and_map() {
        // Python spells "no extras" as explicit null (Optional field in
        // the schema-generated models); Rust must treat null, missing,
        // and {} identically as the empty map.
        for extras_json in ["\"extras\": null,", "\"extras\": {},", ""] {
            let json = format!(
                "{{ {extras_json} \"kind\": \"asr\", \"lang\": \"eng\",                  \"backend\": \"local_whisper\",                  \"input\": {{\"kind\": \"prepared_audio\", \"audio_ref_id\": \"a-1\"}} }}"
            );
            let request: super::AsrRequestV2 = serde_json::from_str(&json)
                .unwrap_or_else(|error| panic!("extras form {extras_json:?} rejected: {error}"));
            assert!(request.extras.is_empty());
        }
    }

    use super::*;

    #[test]
    fn morphosyntax_request_v2_retokenize_round_trips() -> Result<(), Box<dyn std::error::Error>> {
        let req = MorphosyntaxRequestV2 {
            lang: LanguageCode3::yue(),
            payload_ref_id: WorkerArtifactIdV2::from("payload-1"),
            item_count: 3,
            retokenize: true,
        };
        let json = serde_json::to_string(&req)?;
        let deserialized: MorphosyntaxRequestV2 = serde_json::from_str(&json)?;
        assert!(deserialized.retokenize);
        assert_eq!(deserialized.lang.as_ref(), "yue");
        Ok(())
    }

    #[test]
    fn morphosyntax_request_v2_retokenize_defaults_false() -> Result<(), Box<dyn std::error::Error>>
    {
        let json = r#"{"lang":"eng","payload_ref_id":"p1","item_count":1}"#;
        let req: MorphosyntaxRequestV2 = serde_json::from_str(json)?;
        assert!(
            !req.retokenize,
            "retokenize must default to false for backward compat"
        );
        Ok(())
    }

    fn asr_request(decode_budget_seconds: Option<DecodeBudgetSeconds>) -> AsrRequestV2 {
        AsrRequestV2 {
            lang: crate::api::WorkerLanguage::from(LanguageCode3::eng()),
            backend: AsrBackendV2::LocalWhisper,
            input: AsrInputV2::PreparedAudio(PreparedAudioInputV2 {
                audio_ref_id: WorkerArtifactIdV2::from("audio-1"),
            }),
            extras: std::collections::BTreeMap::new(),
            decode_budget_seconds,
        }
    }

    #[test]
    fn asr_ceiling_scales_with_decode_budget_not_a_flat_1800() {
        let short = asr_request(Some(DecodeBudgetSeconds::for_audio(
            FrameCountV2(30 * 16_000),
            SampleRateHzV2(16_000),
        )));
        let long = asr_request(Some(DecodeBudgetSeconds::for_audio(
            FrameCountV2(900 * 16_000),
            SampleRateHzV2(16_000),
        )));
        let short_ceiling = TaskRequestV2::Asr(short).timeout_seconds_with_config(0, 0);
        let long_ceiling = TaskRequestV2::Asr(long).timeout_seconds_with_config(0, 0);
        assert!(
            short_ceiling < long_ceiling,
            "a 30s file's ceiling ({short_ceiling}) must be below a 900s file's ({long_ceiling})"
        );
        assert_ne!(
            short_ceiling, 1800,
            "the flat 1800 ceiling must be gone for a request with a known decode budget"
        );
        assert_ne!(
            long_ceiling, 1800,
            "the flat 1800 ceiling must be gone for a request with a known decode budget"
        );
    }

    #[test]
    fn asr_ceiling_never_below_the_sent_budget() {
        let request = asr_request(Some(DecodeBudgetSeconds::for_audio(
            FrameCountV2(400 * 16_000),
            SampleRateHzV2(16_000),
        )));
        let sent_budget = request.decode_budget_seconds.expect("budget set above");
        let ceiling = TaskRequestV2::Asr(request).timeout_seconds_with_config(0, 0);
        assert!(
            (ceiling as f64) >= sent_budget.as_seconds(),
            "transport ceiling ({ceiling}) must never be shorter than the decode budget it sent ({})",
            sent_budget.as_seconds()
        );
    }

    #[test]
    fn asr_ceiling_falls_back_to_named_constant_when_budget_unknown() {
        let request = asr_request(None);
        let ceiling = TaskRequestV2::Asr(request).timeout_seconds_with_config(0, 0);
        assert_eq!(ceiling, PROVIDER_MEDIA_ASR_TRANSPORT_CEILING_SECONDS);
    }

    #[test]
    fn asr_operator_override_can_only_raise_the_ceiling() {
        let request = asr_request(Some(DecodeBudgetSeconds::for_audio(
            FrameCountV2(30 * 16_000),
            SampleRateHzV2(16_000),
        )));
        let task = TaskRequestV2::Asr(request);
        let derived = task.timeout_seconds_with_config(0, 0);

        // A tiny override below the derived ceiling must not lower it.
        let overridden_low = task.timeout_seconds_with_config(1, 0);
        assert_eq!(
            overridden_low, derived,
            "an operator override below the derived ceiling must not shorten it"
        );

        // A large override must raise the ceiling.
        let overridden_high = task.timeout_seconds_with_config(derived + 500, 0);
        assert_eq!(overridden_high, derived + 500);
    }

    #[test]
    fn decode_budget_seconds_round_trips_over_json() {
        let request = asr_request(Some(DecodeBudgetSeconds::for_audio(
            FrameCountV2(60 * 16_000),
            SampleRateHzV2(16_000),
        )));
        let json = serde_json::to_string(&request).expect("serializes");
        let round_tripped: AsrRequestV2 = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(
            round_tripped.decode_budget_seconds,
            request.decode_budget_seconds
        );
    }

    #[test]
    fn decode_budget_seconds_absent_from_wire_when_none() {
        let request = asr_request(None);
        let json = serde_json::to_string(&request).expect("serializes");
        assert!(
            !json.contains("decode_budget_seconds"),
            "an unknown budget must not be serialized as a fabricated value on the wire"
        );
    }
}
