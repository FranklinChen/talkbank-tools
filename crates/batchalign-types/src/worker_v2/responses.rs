//! Worker protocol V2 response and result types.

use serde::{Deserialize, Serialize};

use crate::api::{DurationMs, DurationSeconds, LanguageCode3, ReportedEngineName};

use super::asr_model::AsrModelIdentityV2;
use super::requests::{
    FrameCountV2, ProtocolErrorCodeV2, SpeakerEmbeddingSpanIdV2, WhisperChunkSpanV2,
    WorkerRequestIdV2,
};
use super::utseg_evidence::UtsegBoundaryModelEvidenceV2;

/// One ASR result built from raw Whisper chunks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct WhisperChunkResultV2 {
    /// Transcript language.
    pub lang: LanguageCode3,
    /// Concatenated transcript text.
    pub text: String,
    /// Raw chunk spans.
    pub chunks: Vec<WhisperChunkSpanV2>,
    /// The models that produced this result, as the runtime observed them.
    ///
    /// Required, and deliberately not defaulted: a transcript whose models are
    /// unknown cannot be stamped honestly or cached safely, and an optional
    /// field would let every producer forget to fill it while still
    /// compiling. The only route to a value is a loader that actually loaded
    /// something, so "which models produced this text" is answered by
    /// construction rather than by a downstream lookup that can miss.
    pub model: AsrModelIdentityV2,
}

/// Stable vocabulary for one monologue element returned by an ASR provider.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AsrElementKindV2 {
    /// Lexical content that should become transcript tokens.
    Text,
    /// Punctuation emitted by the provider.
    Punctuation,
}

/// One raw ASR element inside a speaker monologue.
///
/// Timing fields validated upstream by Python Pydantic models (see module docs).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct AsrElementV2 {
    /// Surface token or punctuation value.
    pub value: String,
    /// Start timestamp in seconds when the provider exposes one.
    #[serde(default)]
    pub start_s: Option<DurationSeconds>,
    /// End timestamp in seconds when the provider exposes one.
    #[serde(default)]
    pub end_s: Option<DurationSeconds>,
    /// Stable element kind selected by the worker adapter.
    pub kind: AsrElementKindV2,
    /// Optional model/provider confidence score.
    #[serde(default)]
    pub confidence: Option<f64>,
}

/// A provider's OWN label for one speaker, admitted as text that names
/// somebody.
///
/// Non-empty and free of surrounding whitespace, because a label is used to
/// tell one voice from another: an empty or blank one distinguishes nothing
/// while looking like an answer.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct ProviderSpeakerLabelV2(String);

impl ProviderSpeakerLabelV2 {
    /// The label, byte for byte as the provider spelled it.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ProviderSpeakerLabelV2 {
    type Error = InvalidProviderSpeakerLabel;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.is_empty() || value.trim() != value {
            return Err(InvalidProviderSpeakerLabel { value });
        }
        Ok(Self(value))
    }
}

impl TryFrom<&str> for ProviderSpeakerLabelV2 {
    type Error = InvalidProviderSpeakerLabel;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::try_from(value.to_owned())
    }
}

impl std::fmt::Display for ProviderSpeakerLabelV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ProviderSpeakerLabelV2 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::try_from(raw).map_err(serde::de::Error::custom)
    }
}

impl schemars::JsonSchema for ProviderSpeakerLabelV2 {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "ProviderSpeakerLabelV2".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "minLength": 1,
            "description": "A provider's own label for one speaker: non-empty, with no surrounding whitespace.",
        })
    }
}

/// A speaker label that names nobody.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("a provider speaker label must be non-empty and unpadded, and {value:?} is not")]
pub struct InvalidProviderSpeakerLabel {
    /// The label as the provider sent it.
    pub value: String,
}

/// Who a provider attributed one monologue to.
///
/// Two states, because there are two facts, and the wire used to have a
/// spelling for only one of them. `speaker` was a bare string, so an engine
/// that separates no speakers at all (Aliyun, FunASR, Whisper, Qwen) had to
/// write SOMETHING, and what it wrote was `"0"`: a label indistinguishable
/// from a provider's real first speaker. Downstream that became
/// `SpeakerIndex(0)` and a `PAR0` tier, so "nobody was diarized" and "the
/// provider said speaker zero" produced identical CHAT.
///
/// `Undiarized` is now a state of its own, and an ABSENT label where
/// separation WAS requested is a refusal rather than a track number.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpeakerAttributionV2 {
    /// The provider named a speaker; this is its own label for them.
    Attributed {
        /// The provider's label, exactly as it spelled it.
        label: ProviderSpeakerLabelV2,
    },
    /// The provider separates no speakers and named none. This is a claim
    /// about the ENGINE, not a guess about the recording.
    Undiarized,
}

/// One speaker-attributed monologue returned by a provider ASR backend.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct AsrMonologueV2 {
    /// Who the provider attributed this span to, if anyone.
    pub speaker: SpeakerAttributionV2,
    /// Ordered elements inside the monologue.
    pub elements: Vec<AsrElementV2>,
}

/// One ASR result built from provider-shaped speaker monologues.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct MonologueAsrResultV2 {
    /// Transcript language reported by the provider adapter.
    pub lang: LanguageCode3,
    /// Speaker-grouped ASR output.
    pub monologues: Vec<AsrMonologueV2>,
    /// The models that produced this result, as the runtime observed them.
    ///
    /// Required for the same reason as on [`WhisperChunkResultV2`]. For a
    /// cloud provider the composition names the service and the parameter it
    /// was called with, never an account credential: an appkey identifies who
    /// paid, not what ran.
    pub model: AsrModelIdentityV2,
}

/// One raw Whisper forced-alignment token span returned by Python.
///
/// Timing fields validated upstream by Python Pydantic models (see module docs).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct WhisperTokenTimingV2 {
    /// Surface token text returned by the FA runtime.
    pub text: String,
    /// Token onset timestamp in seconds.
    pub time_s: DurationSeconds,
}

/// Forced-alignment token response returned before Rust token-to-word
/// reconciliation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct WhisperTokenTimingResultV2 {
    /// Raw token timings in model order.
    pub tokens: Vec<WhisperTokenTimingV2>,
}

/// One word-level timing result.
///
/// Timing fields validated upstream by Python Pydantic models (see module docs).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct IndexedWordTimingV2 {
    /// Start time in milliseconds.
    pub start_ms: DurationMs,
    /// End time in milliseconds.
    pub end_ms: DurationMs,
    /// Optional model confidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

/// Forced-alignment indexed response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct IndexedWordTimingResultV2 {
    /// Indexed timing results aligned to the request words.
    pub indexed_timings: Vec<Option<IndexedWordTimingV2>>,
}

/// The Stanza model that analyzed one morphosyntax item, as reported by the
/// worker that ran it.
///
/// Carried per item rather than once per worker because one batch can span
/// language groups, and because the process that ran the item is the only
/// honest witness: a capability snapshot taken from another process (inside
/// `transcribe`, the ASR worker) describes a different runtime.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, schemars::JsonSchema,
)]
pub struct MorphosyntaxModelIdentityV2 {
    /// Installed Stanza package version (`stanza.__version__`).
    pub stanza_version: ReportedEngineName,
    /// Language whose Stanza pipeline analyzed the item.
    pub lang: LanguageCode3,
    /// Which pipeline variant ran, so provenance does not hide a
    /// retokenizing or POS-overriding path behind the plain Stanza name.
    pub pipeline: MorphosyntaxPipelineV2,
}

/// The morphosyntax pipeline variant that analyzed an item. Closed: a new
/// variant must be named here before a worker can report it.
///
/// Each variant has ONE name, [`Self::wire_name`]. The serde form, the JSON
/// Schema constants and the name a provenance stamp writes
/// ([`Self::stamp_name`]) are all read from it, so they cannot drift apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MorphosyntaxPipelineV2 {
    /// Stanza on the transcript's own tokens.
    Standard,
    /// Stanza's neural tokenizer re-segmenting Mandarin text.
    MandarinRetokenize,
    /// Stanza with PyCantonese part-of-speech tags replacing Stanza's.
    CantonesePycantonesePos,
}

impl MorphosyntaxPipelineV2 {
    /// Every variant, in declaration order.
    pub const ALL: [Self; 3] = [
        Self::Standard,
        Self::MandarinRetokenize,
        Self::CantonesePycantonesePos,
    ];

    /// The variant's one name: its wire form, its schema constant, and the
    /// text a provenance stamp writes for it.
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::MandarinRetokenize => "mandarin_retokenize",
            Self::CantonesePycantonesePos => "cantonese_pycantonese_pos",
        }
    }

    /// The name as stamp-safe text. Each arm is checked at compile time.
    pub fn stamp_name(self) -> crate::domain::StampSafeText {
        use crate::domain::StampSafeText;
        match self {
            Self::Standard => const { StampSafeText::from_static(Self::Standard.wire_name()) },
            Self::MandarinRetokenize => {
                const { StampSafeText::from_static(Self::MandarinRetokenize.wire_name()) }
            }
            Self::CantonesePycantonesePos => {
                const { StampSafeText::from_static(Self::CantonesePycantonesePos.wire_name()) }
            }
        }
    }
}

impl serde::Serialize for MorphosyntaxPipelineV2 {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.wire_name())
    }
}

impl<'de> serde::Deserialize<'de> for MorphosyntaxPipelineV2 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = <std::borrow::Cow<'de, str> as serde::Deserialize>::deserialize(deserializer)?;
        Self::ALL
            .into_iter()
            .find(|pipeline| pipeline.wire_name() == name)
            .ok_or_else(|| {
                serde::de::Error::custom(format!("unknown morphosyntax pipeline {name:?}"))
            })
    }
}

impl schemars::JsonSchema for MorphosyntaxPipelineV2 {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "MorphosyntaxPipelineV2".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let variants: Vec<serde_json::Value> = Self::ALL
            .iter()
            .map(|pipeline| serde_json::json!({"type": "string", "const": pipeline.wire_name()}))
            .collect();
        schemars::json_schema!({
            "description": "The morphosyntax pipeline variant that analyzed an item. Closed: a new variant must be named here before a worker can report it.",
            "oneOf": variants,
        })
    }
}

/// Why one Universal Dependencies relation a worker received from Stanza is
/// not the relation it applied.
///
/// Closed: a rewrite a worker can make must be named here before it can be
/// reported, so a new one cannot reach a transcript under a name nothing
/// downstream knows. Each variant has ONE name, [`Self::wire_name`], which the
/// serde form and the JSON Schema constants are both read from, so they cannot
/// drift apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UdRelationRepairKindV2 {
    /// The model emitted a padding label (`<PAD>`, `<UNK>`), which is not a
    /// relation at all. Replaced by `dep`.
    PadRelation,
    /// A UD relation in the wrong case (`NSUBJ`). Lowercased.
    RelationCase,
    /// A known non-UD spelling of a UD relation (`iob`). Replaced by the UD
    /// relation it means (`iobj`).
    RelationAlias,
    /// A label that is no UD relation and has no known equivalent. Replaced by
    /// `dep`, which is a real UD relation.
    UnknownRelation,
}

impl UdRelationRepairKindV2 {
    /// Every variant, in declaration order.
    pub const ALL: [Self; 4] = [
        Self::PadRelation,
        Self::RelationCase,
        Self::RelationAlias,
        Self::UnknownRelation,
    ];

    /// The variant's one name: its wire form and its schema constant.
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::PadRelation => "pad_relation",
            Self::RelationCase => "relation_case",
            Self::RelationAlias => "relation_alias",
            Self::UnknownRelation => "unknown_relation",
        }
    }
}

impl serde::Serialize for UdRelationRepairKindV2 {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.wire_name())
    }
}

impl<'de> serde::Deserialize<'de> for UdRelationRepairKindV2 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = <std::borrow::Cow<'de, str> as serde::Deserialize>::deserialize(deserializer)?;
        Self::ALL
            .into_iter()
            .find(|kind| kind.wire_name() == name)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown UD relation repair {name:?}")))
    }
}

impl schemars::JsonSchema for UdRelationRepairKindV2 {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "UdRelationRepairKindV2".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let variants: Vec<serde_json::Value> = Self::ALL
            .iter()
            .map(|kind| serde_json::json!({"type": "string", "const": kind.wire_name()}))
            .collect();
        schemars::json_schema!({
            "description": "Why one Universal Dependencies relation a worker received from Stanza is not the relation it applied. Closed: a worker cannot report a repair not named here.",
            "oneOf": variants,
        })
    }
}

/// One relation rewrite a worker made inside an analysis it returned.
///
/// Carries what it takes to attribute the rewrite as well as count it: the
/// word it was made on, and both relations. The worker is the only witness,
/// because the relation Stanza produced does not survive into the analysis it
/// sends: by the time the server sees the item, the repaired relation is the
/// only one there.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct UdRelationRepairV2 {
    /// Which rewrite this is.
    pub kind: UdRelationRepairKindV2,
    /// Surface form of the word whose relation was rewritten, so a reader can
    /// find it in the transcript.
    pub word: String,
    /// The relation the model produced, verbatim.
    pub from_relation: String,
    /// The relation applied in its place, which the worker admits is a UD
    /// relation before it builds this value.
    pub to_relation: String,
}

/// One morphosyntax item result returned by Python.
///
/// A tagged union, so an analysis cannot exist without the model that
/// produced it and an item cannot be both an analysis and an error.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MorphosyntaxItemResultV2 {
    /// Stanza analyzed the item.
    Analyzed {
        /// Raw Stanza `doc.to_dict()` sentence arrays.
        raw_sentences: Vec<serde_json::Value>,
        /// The model that produced them.
        model: MorphosyntaxModelIdentityV2,
        /// Every relation the worker rewrote inside `raw_sentences`, in the
        /// order it made them. Empty means it rewrote nothing.
        ///
        /// Required, with no serde default, for the reason the model is:
        /// a default would make "this worker reports no repairs" and "this
        /// worker does not report repairs" the same value on the wire, which
        /// is the shape that let a rewrite live only in a log line.
        repairs: Vec<UdRelationRepairV2>,
    },
    /// The item had no words, so no model ran and there is nothing to apply.
    NoWords,
    /// The item failed; the file it belongs to fails with this message.
    Failed {
        /// Operator-facing failure message.
        error: String,
    },
}

/// Batched morphosyntax response payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct MorphosyntaxResultV2 {
    /// Item results aligned to the prepared batch payload order.
    pub items: Vec<MorphosyntaxItemResultV2>,
}

/// One utseg item result returned by Python.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct UtsegItemResultV2 {
    /// Direct word-group assignments when inference succeeded without raw trees.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assignments: Option<Vec<usize>>,
    /// Raw constituency trees when inference succeeded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trees: Option<Vec<String>>,
    /// Provenance-bearing classifier evidence parallel to the request words.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boundary_model_evidence: Option<UtsegBoundaryModelEvidenceV2>,
    /// Optional per-item runtime error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Batched utterance-segmentation response payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct UtsegResultV2 {
    /// Item results aligned to the prepared batch payload order.
    pub items: Vec<UtsegItemResultV2>,
}

/// One translation item result returned by Python.
///
/// A tagged union: a translation always names the engine that produced it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TranslationItemResultV2 {
    /// The engine translated the item.
    Translated {
        /// Raw model translation.
        raw_translation: String,
        /// The translation engine that produced it.
        engine: ReportedEngineName,
    },
    /// The item's text was blank, so no engine ran.
    BlankInput,
    /// The item failed; the file it belongs to fails with this message.
    Failed {
        /// Operator-facing failure message.
        error: String,
    },
}

/// Batched translation response payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct TranslationResultV2 {
    /// Item results aligned to the prepared batch payload order.
    pub items: Vec<TranslationItemResultV2>,
}

/// One structured coreference chain reference returned by Python.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct CorefChainRefV2 {
    /// Chain identifier assigned by the coreference runtime.
    pub chain_id: usize,
    /// Whether the current word starts this mention.
    pub is_start: bool,
    /// Whether the current word ends this mention.
    pub is_end: bool,
}

/// One per-sentence coreference annotation returned by Python.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct CorefAnnotationV2 {
    /// Sentence index inside the corresponding document item.
    pub sentence_idx: usize,
    /// Per-word chain references parallel to the sentence words.
    pub words: Vec<Vec<CorefChainRefV2>>,
}

/// One coreference item result returned by Python.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CorefItemResultV2 {
    /// The engine resolved the document.
    Resolved {
        /// Structured sparse annotations (empty when no chains were found).
        annotations: Vec<CorefAnnotationV2>,
        /// The coreference engine that produced them.
        engine: ReportedEngineName,
    },
    /// The document had no sentences, so no engine ran.
    NoSentences,
    /// The item failed; the file it belongs to fails with this message.
    Failed {
        /// Operator-facing failure message.
        error: String,
    },
}

/// Batched coreference response payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct CorefResultV2 {
    /// Item results aligned to the prepared batch payload order.
    pub items: Vec<CorefItemResultV2>,
}

/// One raw speaker diarization segment returned by Python.
///
/// Timing fields validated upstream by Python Pydantic models (see module docs).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct SpeakerSegmentV2 {
    /// Segment start in milliseconds.
    pub start_ms: DurationMs,
    /// Segment end in milliseconds.
    pub end_ms: DurationMs,
    /// Stable speaker label chosen by the model adapter.
    pub speaker: String,
}

/// Provider job identifier attached to durable raw speaker evidence.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, schemars::JsonSchema,
)]
#[serde(transparent)]
pub struct SpeakerProviderJobIdV2(String);

impl From<&str> for SpeakerProviderJobIdV2 {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl SpeakerProviderJobIdV2 {
    /// Borrow the provider's identifier.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Raw evidence returned by one speaker-inference backend.
///
/// The three variants make backend provenance structural. In particular, a
/// local backend cannot construct a pyannoteAI result without also supplying
/// the completed provider job evidence, and cloud evidence cannot masquerade
/// as a normalized-only local segment list.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpeakerInferenceEvidenceV2 {
    /// Complete successful pyannoteAI job output, before normalization.
    PyannoteAi {
        /// Provider job which produced the output.
        job_id: SpeakerProviderJobIdV2,
        /// Provider-shaped completed-job output.
        output: serde_json::Map<String, serde_json::Value>,
        /// Optional provider warning returned with the successful job.
        warning: Option<String>,
    },
    /// Normalized segments from the local pyannote runtime.
    Pyannote {
        /// Ordered speaker segments.
        segments: Vec<SpeakerSegmentV2>,
    },
    /// Normalized segments from the local NeMo runtime.
    Nemo {
        /// Ordered speaker segments.
        segments: Vec<SpeakerSegmentV2>,
    },
}

/// Raw speaker diarization evidence returned by the model host.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct SpeakerResultV2 {
    /// Backend-specific evidence before shared normalization.
    pub evidence: SpeakerInferenceEvidenceV2,
}

/// What the embedding model had to say about one requested span.
///
/// Two variants, because "we could not measure this" and "we measured this"
/// are different facts and only one of them is a vector. Below its own minimum
/// input length the pinned model does not raise: it returns a correctly shaped
/// float array whose every component is NaN, which no type can distinguish
/// from a real embedding and which compares false against every threshold. A
/// caller handed that would report an utterance as UNMATCHED when the honest
/// answer is UNMEASURABLE. The worker refuses instead, and says how short.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpeakerEmbeddingOutcomeV2 {
    /// The model measured this span.
    Embedded {
        /// Fixed-width acoustic vector, whose width the envelope declares.
        vector: Vec<f64>,
    },
    /// The span is shorter than the model's own minimum input length.
    TooShort {
        /// How many frames the span held. The minimum it fell short of is on
        /// the envelope, which owns that fact once for the whole response.
        frame_count: FrameCountV2,
    },
}

/// One requested span's outcome, echoing the id it was requested under.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct SpeakerEmbeddingSpanResultV2 {
    /// The requested span's name, echoed verbatim.
    pub span_id: SpeakerEmbeddingSpanIdV2,
    /// What the model had to say about it.
    pub outcome: SpeakerEmbeddingOutcomeV2,
}

/// Speaker embeddings for every requested span, plus the model's own bounds.
///
/// `dimension` and `minimum_frames` are REPORTED by the worker rather than
/// assumed by the reader. They are properties of the loaded model file, so a
/// constant on this side would be a second place the truth lives and would go
/// on agreeing with a model that had moved.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct SpeakerEmbeddingResultV2 {
    /// Width of every `Embedded` vector in this response.
    pub dimension: u32,
    /// Shortest span the model will measure, in prepared-audio frames.
    pub minimum_frames: FrameCountV2,
    /// One outcome per requested span, in the order requested.
    pub spans: Vec<SpeakerEmbeddingSpanResultV2>,
}

/// Raw openSMILE output returned by the model host.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct OpenSmileResultV2 {
    /// Requested feature-set name.
    pub feature_set: String,
    /// Requested feature-level name.
    pub feature_level: String,
    /// Number of extracted feature columns.
    pub num_features: u64,
    /// Number of result rows/segments.
    pub duration_segments: u64,
    /// Source audio identifier echoed by the worker.
    pub audio_file: String,
    /// Tabular feature rows keyed by feature name.
    pub rows: Vec<std::collections::BTreeMap<String, f64>>,
    /// Whether the underlying runtime succeeded.
    pub success: bool,
    /// Optional runtime error detail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Raw AVQI output returned by the model host.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct AvqiResultV2 {
    /// AVQI score.
    pub avqi: f64,
    /// Cepstral Peak Prominence Smoothed.
    pub cpps: f64,
    /// Harmonics-to-noise ratio.
    pub hnr: f64,
    /// Local shimmer percentage.
    pub shimmer_local: f64,
    /// Local shimmer in dB.
    pub shimmer_local_db: f64,
    /// LTAS slope.
    pub slope: f64,
    /// LTAS tilt.
    pub tilt: f64,
    /// Continuous-speech file label echoed by the worker.
    pub cs_file: String,
    /// Sustained-vowel file label echoed by the worker.
    pub sv_file: String,
    /// Whether the runtime succeeded.
    pub success: bool,
    /// Optional runtime error detail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Typed execute result payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskResultV2 {
    /// Whisper chunk output.
    WhisperChunkResult(WhisperChunkResultV2),
    /// Provider-shaped speaker monologue output.
    MonologueAsrResult(MonologueAsrResultV2),
    /// Raw Whisper FA token timings.
    WhisperTokenTimingResult(WhisperTokenTimingResultV2),
    /// Forced-alignment indexed timings.
    IndexedWordTimingResult(IndexedWordTimingResultV2),
    /// Batched morphosyntax result.
    MorphosyntaxResult(MorphosyntaxResultV2),
    /// Batched utterance-segmentation result.
    UtsegResult(UtsegResultV2),
    /// Batched translation result.
    TranslationResult(TranslationResultV2),
    /// Batched coreference result.
    CorefResult(CorefResultV2),
    /// Raw speaker diarization result.
    SpeakerResult(SpeakerResultV2),
    /// Speaker embeddings for named spans of one prepared recording.
    SpeakerEmbeddingResult(SpeakerEmbeddingResultV2),
    /// Raw openSMILE result.
    OpensmileResult(OpenSmileResultV2),
    /// Raw AVQI result.
    AvqiResult(AvqiResultV2),
}

/// Top-level execute outcome.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecuteOutcomeV2 {
    /// Request completed successfully.
    Success,
    /// Request failed at the protocol/runtime boundary.
    Error {
        /// Stable protocol error category.
        code: ProtocolErrorCodeV2,
        /// Human-readable detail for logs and tests.
        message: String,
    },
}

/// Top-level V2 execute response.
///
/// STRUCTURAL (ruled 2026-08-21): the outcome and payload are stored as ONE
/// private enum, so a success without a payload, or an error carrying one, has
/// no representation at all. The two constructors build the two legal shapes;
/// deserialization goes through a validating wire shape and refuses the
/// disagreeing pairings by name; `Serialize` and the JSON Schema still present
/// the wire's four fields. What holds Serialize to the wire shape is the
/// roundtrip test below (the schema gate never observes Serialize; it
/// regenerates from the wire struct and byte-compares the committed files).
#[derive(Debug, Clone, PartialEq)]
pub struct ExecuteResponseV2 {
    request_id: WorkerRequestIdV2,
    body: ExecuteResponseBodyV2,
    elapsed_s: DurationSeconds,
}

/// The validated interior: outcome and payload as one fact. Private, so the
/// pairing can neither be built nor read apart.
#[derive(Debug, Clone, PartialEq)]
enum ExecuteResponseBodyV2 {
    /// Succeeded, with the payload success promises.
    Success(TaskResultV2),
    /// Failed, with the code and message a failure promises, and nothing else.
    Failure {
        code: ProtocolErrorCodeV2,
        message: String,
    },
}

// The unvalidated wire shape `ExecuteResponseV2` deserializes THROUGH, and
// the single source of the published JSON Schema (schemars is renamed onto
// the public type's identity). Private: no code outside this module can hold
// the unvalidated pairing. Its DOC comments ship as the schema descriptions,
// so they must stay byte-identical to the published schema; the drift gate
// compares against the committed schema files and caught the first draft
// shipping a maintenance note as the type description.
#[derive(Deserialize, schemars::JsonSchema)]
#[schemars(rename = "ExecuteResponseV2")]
/// Top-level V2 execute response.
struct ExecuteResponseWireV2 {
    /// Correlation id for the request.
    request_id: WorkerRequestIdV2,
    /// Success or typed protocol/runtime error.
    outcome: ExecuteOutcomeV2,
    /// Typed task result when execution succeeded.
    #[serde(default)]
    result: Option<TaskResultV2>,
    /// Execution time in seconds.
    elapsed_s: DurationSeconds,
}

impl<'de> Deserialize<'de> for ExecuteResponseV2 {
    /// Refuse the two pairings the type makes unrepresentable, at the only
    /// remaining door (the wire). A worker that claims success and sends
    /// nothing, or reports an error while also sending a payload, is a worker
    /// bug, and it is reported as a parse failure naming the disagreement
    /// rather than surfacing downstream as a half-read response.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ExecuteResponseWireV2::deserialize(deserializer)?;
        let body = match (wire.outcome, wire.result) {
            (ExecuteOutcomeV2::Success, Some(result)) => ExecuteResponseBodyV2::Success(result),
            (ExecuteOutcomeV2::Success, None) => {
                return Err(serde::de::Error::custom(
                    "execute response claimed success but carried no result payload",
                ));
            }
            (ExecuteOutcomeV2::Error { code, message }, None) => {
                ExecuteResponseBodyV2::Failure { code, message }
            }
            (ExecuteOutcomeV2::Error { .. }, Some(_)) => {
                return Err(serde::de::Error::custom(
                    "execute response reported an error but also carried a result payload",
                ));
            }
        };
        Ok(Self {
            request_id: wire.request_id,
            body,
            elapsed_s: wire.elapsed_s,
        })
    }
}

impl Serialize for ExecuteResponseV2 {
    /// Emit the wire's four-field shape from the validated interior. The
    /// borrowed outcome mirror below must serialize identically to
    /// [`ExecuteOutcomeV2`]; the byte-stable roundtrip test in this file is
    /// what holds that equivalence (the schema drift gate cannot: it never
    /// observes Serialize).
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        /// Borrowed serialization mirror of [`ExecuteOutcomeV2`].
        #[derive(Serialize)]
        #[serde(tag = "kind", rename_all = "snake_case")]
        enum OutcomeWireRef<'a> {
            Success,
            Error {
                code: ProtocolErrorCodeV2,
                message: &'a str,
            },
        }

        let (outcome, result) = match &self.body {
            ExecuteResponseBodyV2::Success(result) => (OutcomeWireRef::Success, Some(result)),
            ExecuteResponseBodyV2::Failure { code, message } => (
                OutcomeWireRef::Error {
                    code: *code,
                    message,
                },
                None,
            ),
        };

        let mut state = serializer.serialize_struct("ExecuteResponseV2", 4)?;
        state.serialize_field("request_id", &self.request_id)?;
        state.serialize_field("outcome", &outcome)?;
        state.serialize_field("result", &result)?;
        state.serialize_field("elapsed_s", &self.elapsed_s)?;
        state.end()
    }
}

impl schemars::JsonSchema for ExecuteResponseV2 {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        ExecuteResponseWireV2::schema_name()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        ExecuteResponseWireV2::schema_id()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        ExecuteResponseWireV2::json_schema(generator)
    }
}

/// What a response actually says, with its outcome and payload read as ONE
/// thing.
///
/// [`ExecuteResponseV2`] used to pair an [`ExecuteOutcomeV2`] with an
/// `Option<TaskResultV2>` in two public fields the wire format could not keep
/// in agreement, so every consumer checked both by hand in an order nobody
/// stated. Four got the order right and two got it wrong, which is the cost of
/// a rule that lives in a reader's head: the openSMILE and AVQI dispatch paths
/// tested the payload first, and since a failed request carries no payload,
/// every typed error response was reported as "missing a result payload" with
/// the code and message the worker actually sent thrown away.
///
/// The pairing is now enforced structurally, so this view is total: there is
/// no malformed case left for a consumer to name.
// No `Eq`: task results carry floating-point measurements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecuteOutcomeRef<'a> {
    /// Succeeded, and the payload it produced.
    Success(&'a TaskResultV2),
    /// Failed, with the protocol category and detail the worker reported.
    Failed {
        /// Stable protocol error category.
        code: ProtocolErrorCodeV2,
        /// Human-readable detail for logs and tests.
        message: &'a str,
    },
}

impl ExecuteResponseV2 {
    /// Build the one legal success shape: `Success` WITH a payload.
    #[must_use]
    pub fn success(
        request_id: WorkerRequestIdV2,
        result: TaskResultV2,
        elapsed_s: DurationSeconds,
    ) -> Self {
        Self {
            request_id,
            body: ExecuteResponseBodyV2::Success(result),
            elapsed_s,
        }
    }

    /// Build the one legal failure shape: an error code and message, and no
    /// payload. Takes the code and message rather than an [`ExecuteOutcomeV2`]
    /// so `failure(Success)` has no signature to travel through.
    #[must_use]
    pub fn failure(
        request_id: WorkerRequestIdV2,
        code: ProtocolErrorCodeV2,
        message: String,
        elapsed_s: DurationSeconds,
    ) -> Self {
        Self {
            request_id,
            body: ExecuteResponseBodyV2::Failure { code, message },
            elapsed_s,
        }
    }

    /// Correlation id for the request this response answers.
    #[must_use]
    pub fn request_id(&self) -> &WorkerRequestIdV2 {
        &self.request_id
    }

    /// Execution time in seconds, as the worker measured it.
    #[must_use]
    pub fn elapsed_s(&self) -> DurationSeconds {
        self.elapsed_s
    }

    /// Read the outcome and payload together.
    #[must_use]
    pub fn read(&self) -> ExecuteOutcomeRef<'_> {
        match &self.body {
            ExecuteResponseBodyV2::Success(result) => ExecuteOutcomeRef::Success(result),
            ExecuteResponseBodyV2::Failure { code, message } => ExecuteOutcomeRef::Failed {
                code: *code,
                message,
            },
        }
    }

    /// Take the payload out of a success, or the failure's code and message.
    ///
    /// The owned counterpart of [`Self::read`], for a consumer that moves the
    /// payload's parts into its own types instead of cloning them.
    #[must_use]
    pub fn into_outcome(self) -> Result<TaskResultV2, (ProtocolErrorCodeV2, String)> {
        match self.body {
            ExecuteResponseBodyV2::Success(result) => Ok(result),
            ExecuteResponseBodyV2::Failure { code, message } => Err((code, message)),
        }
    }
}

/// Progress event emitted by long-running V2 tasks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct ProgressEventV2 {
    /// Correlation id of the request being updated.
    pub request_id: WorkerRequestIdV2,
    /// Completed units.
    pub completed: u32,
    /// Total units expected.
    pub total: u32,
    /// Stable progress stage label.
    pub stage: String,
}

/// Shutdown request sent to a V2 worker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct ShutdownRequestV2 {
    /// Correlation id for the shutdown message.
    pub request_id: WorkerRequestIdV2,
}

#[cfg(test)]
#[allow(clippy::expect_used)] // fixture parses and refusals are the assertions themselves
mod tests {
    use super::*;

    /// WIRE FORMAT: paid pyannoteAI output crosses the worker boundary before
    /// normalization, so a later normalization revision can replay it without
    /// another service call.
    #[test]
    fn speaker_result_retains_provider_shaped_evidence() {
        let result = SpeakerResultV2 {
            evidence: SpeakerInferenceEvidenceV2::PyannoteAi {
                job_id: SpeakerProviderJobIdV2::from("job-1"),
                output: serde_json::from_value(serde_json::json!({
                    "exclusiveDiarization": [
                        {"start": 0.0, "end": 0.75, "speaker": "SPEAKER_00"}
                    ]
                }))
                .expect("provider output object"),
                warning: None,
            },
        };

        let value = serde_json::to_value(result).expect("provider evidence serializes");
        assert_eq!(value["evidence"]["kind"], "pyannote_ai");
        assert_eq!(value["evidence"]["job_id"], "job-1");
        assert!(value["evidence"]["output"]["exclusiveDiarization"].is_array());
    }

    /// WIRE FORMAT: a speaker-embedding response roundtrips, and the two
    /// span outcomes stay distinguishable on the wire.
    ///
    /// This is a wire-format test rather than something a type could delete:
    /// Rust and a hand-written Pydantic model on the other side are two
    /// separate functions that have to agree, and only a roundtrip says so.
    /// What it pins beyond shape: a span the model could not measure
    /// serializes as its own tagged variant carrying the length that was too
    /// short, and never as an empty, zero or NaN-filled vector.
    #[test]
    fn speaker_embedding_result_keeps_unmeasurable_spans_distinguishable() {
        let result = SpeakerEmbeddingResultV2 {
            dimension: 3,
            minimum_frames: FrameCountV2(1680),
            spans: vec![
                SpeakerEmbeddingSpanResultV2 {
                    span_id: SpeakerEmbeddingSpanIdV2::from("enroll-INV"),
                    outcome: SpeakerEmbeddingOutcomeV2::Embedded {
                        vector: vec![0.5, -0.25, 0.125],
                    },
                },
                SpeakerEmbeddingSpanResultV2 {
                    span_id: SpeakerEmbeddingSpanIdV2::from("utt-7"),
                    outcome: SpeakerEmbeddingOutcomeV2::TooShort {
                        frame_count: FrameCountV2(400),
                    },
                },
            ],
        };

        let json = serde_json::to_string(&result).expect("embedding result serializes");
        let back: SpeakerEmbeddingResultV2 =
            serde_json::from_str(&json).expect("embedding result deserializes");
        assert_eq!(back, result);

        let value: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        assert_eq!(value["spans"][0]["outcome"]["kind"], "embedded");
        assert_eq!(value["spans"][1]["outcome"]["kind"], "too_short");
        assert_eq!(value["spans"][1]["outcome"]["frame_count"], 400);
        assert!(
            value["spans"][1]["outcome"].get("vector").is_none(),
            "an unmeasurable span must not carry a vector field at all"
        );
    }

    /// WIRE FORMAT: the embedding result travels inside the shared task-result
    /// envelope under its own tag, so a reader cannot mistake it for
    /// diarization evidence.
    #[test]
    fn speaker_embedding_result_has_its_own_task_result_tag() {
        let payload = TaskResultV2::SpeakerEmbeddingResult(SpeakerEmbeddingResultV2 {
            dimension: 1,
            minimum_frames: FrameCountV2(1680),
            spans: Vec::new(),
        });
        let value = serde_json::to_value(&payload).expect("task result serializes");
        assert_eq!(value["kind"], "speaker_embedding_result");

        let back: TaskResultV2 = serde_json::from_value(value).expect("task result deserializes");
        assert_eq!(back, payload);
    }

    /// WIRE FORMAT: both legal response shapes roundtrip byte-stably through
    /// the manual `Serialize`/`Deserialize` pair, which is exactly the kind of
    /// property no type can pin (two separate functions must agree).
    #[test]
    fn legal_response_shapes_roundtrip_byte_stably() {
        let success = serde_json::json!({
            "request_id": "req-1",
            "outcome": {"kind": "success"},
            "result": {
                "kind": "speaker_result",
                "evidence": {"kind": "pyannote", "segments": []}
            },
            "elapsed_s": 0.5
        });
        let failure = serde_json::json!({
            "request_id": "req-2",
            "outcome": {"kind": "error", "code": "runtime_failure", "message": "boom"},
            "result": null,
            "elapsed_s": 0.25
        });
        for wire in [success, failure] {
            let parsed: ExecuteResponseV2 =
                serde_json::from_value(wire.clone()).expect("legal shape must parse");
            let emitted = serde_json::to_value(&parsed).expect("must serialize");
            assert_eq!(emitted, wire);
        }
    }

    /// The two pairings the structural type exists to refuse (ruled
    /// 2026-08-21): a success with no payload, and an error carrying one. Both
    /// used to deserialize and travel until a consumer noticed, or did not.
    #[test]
    fn disagreeing_pairings_are_refused_at_the_wire() {
        let success_without_result = serde_json::json!({
            "request_id": "req-3",
            "outcome": {"kind": "success"},
            "elapsed_s": 0.5
        });
        let error_with_result = serde_json::json!({
            "request_id": "req-4",
            "outcome": {"kind": "error", "code": "runtime_failure", "message": "boom"},
            "result": {
                "kind": "speaker_result",
                "evidence": {"kind": "pyannote", "segments": []}
            },
            "elapsed_s": 0.5
        });

        let error = serde_json::from_value::<ExecuteResponseV2>(success_without_result)
            .expect_err("success without a result must be refused");
        assert!(
            error.to_string().contains("carried no result payload"),
            "{error}"
        );

        let error = serde_json::from_value::<ExecuteResponseV2>(error_with_result)
            .expect_err("an error carrying a result must be refused");
        assert!(
            error.to_string().contains("also carried a result payload"),
            "{error}"
        );
    }
}
