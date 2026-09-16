//! Pinned ASR model identity: what a plan asked a worker to load, and what the
//! worker reported loading.
//!
//! # Why an identity is pinned rather than only reported
//!
//! Most local ASR models used to load by NAME. A hub repository that moves
//! therefore changed what a run produced, with nothing in the transcript or
//! the cache to say so, and no identity was knowable before a model had
//! loaded. The identity here travels WITH the plan: the control plane resolves
//! an exact revision from its own manifest, the request carries it, the worker
//! loads exactly that revision and reports what it loaded, and
//! [`AsrModelIdentityV2::admit`] refuses a disagreement by name.
//!
//! # The three spaces, and why they are different types
//!
//! | Space | Type | Meaning |
//! |---|---|---|
//! | asked for | [`RequestedModelV2`] | the id and revision the plan chose |
//! | loaded | [`LoadedModelV2`] | the same pair, plus what the runtime saw |
//! | composition | [`AsrRequestedModelsV2`] / [`AsrModelIdentityV2`] | which models an engine needs, by role |
//!
//! Requested and observed are deliberately NOT one field. A ModelScope loader
//! is given a tag and can report no commit at all; a cloud provider is given a
//! parameter and reports nothing. Collapsing the two would force those cases to
//! invent an observation, which is the defect this module exists to remove.
//!
//! # Composition is closed, and its required fields are structural
//!
//! An engine needs a fixed set of models: Qwen3-ASR cannot run without its
//! forced aligner, Paraformer without its voice-activity and punctuation
//! models. Those are struct variants with required fields, so "Qwen without an
//! aligner" has no representation and no downstream check can forget it.
//!
//! Closed means closed in both directions: every [`AsrRequestedModelsV2`]
//! variant has a producer in the manifest's per-backend resolver. In-process
//! whisper.cpp is the case worth stating, because it is the one composition
//! that appears on the LOADED side alone: it is not a worker backend, so no
//! [`AsrBackendV2`] selects it and nothing can request it. A requested arm for
//! it had no constructor anywhere and admitted nothing, which made "the plan
//! pinned a native-Whisper composition" a state the type permitted and the
//! program could never reach.

use serde::{Deserialize, Serialize};

use super::requests::AsrBackendV2;
use crate::domain::{InvalidStampSafeText, StampJoiner, StampSafeText};

// ---------------------------------------------------------------------------
// Identifier and revision newtypes
// ---------------------------------------------------------------------------

/// A model identifier as BA3 names it: a Hugging Face repo id, a ModelScope
/// repo id, a ggml file name, or a provider's own service name.
///
/// Stamp-safe text, because provenance records it byte for byte.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct ModelIdV2(StampSafeText);

impl ModelIdV2 {
    /// Admit a compile-time literal. An unsafe literal is a compile error.
    #[must_use]
    pub const fn from_static(text: &'static str) -> Self {
        Self(StampSafeText::from_static(text))
    }

    /// The identifier, byte for byte.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// The identifier as stamp-safe text, for writing into a provenance field.
    #[must_use]
    pub fn as_stamp_text(&self) -> &StampSafeText {
        &self.0
    }
}

impl TryFrom<String> for ModelIdV2 {
    type Error = InvalidStampSafeText;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        StampSafeText::try_from(value).map(Self)
    }
}

impl TryFrom<&str> for ModelIdV2 {
    type Error = InvalidStampSafeText;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::try_from(value.to_owned())
    }
}

impl std::fmt::Display for ModelIdV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ModelIdV2 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::try_from(raw).map_err(serde::de::Error::custom)
    }
}

impl schemars::JsonSchema for ModelIdV2 {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "ModelIdV2".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "pattern": StampSafeText::json_schema_pattern(),
            "description": "A model identifier as BA3 names it: a hub repo id, a ModelScope repo id, a ggml file name, or a provider service name. Stamp-safe text, because provenance records it byte for byte.",
        })
    }
}

/// A lowercase-hexadecimal digest of a fixed width, the shared representation
/// behind [`HubCommitV2`] and [`ContentDigestV2`].
///
/// Private: the two digests are different facts (where a repository stood
/// versus what a file contains) and must not be interchangeable, so each keeps
/// its own type and only the character rule is shared.
fn admit_lowercase_hex(value: &str, width: usize, label: &str) -> Result<(), InvalidRevision> {
    if value.len() != width || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(InvalidRevision {
            detail: format!("{label} must be {width} lowercase hexadecimal characters"),
        });
    }
    if value.bytes().any(|byte| byte.is_ascii_uppercase()) {
        return Err(InvalidRevision {
            detail: format!("{label} must be lowercase"),
        });
    }
    Ok(())
}

/// Why a revision value cannot be admitted.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{detail}")]
pub struct InvalidRevision {
    /// What was wrong with the value.
    pub detail: String,
}

impl From<InvalidStampSafeText> for InvalidRevision {
    fn from(error: InvalidStampSafeText) -> Self {
        Self {
            detail: error.to_string(),
        }
    }
}

/// Width of a git commit as the hubs publish it.
const HUB_COMMIT_WIDTH: usize = 40;
/// Width of a SHA-256 content digest.
const CONTENT_DIGEST_WIDTH: usize = 64;

/// An exact hub commit: 40 lowercase hexadecimal characters.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct HubCommitV2(String);

impl HubCommitV2 {
    /// The commit, byte for byte.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for HubCommitV2 {
    type Error = InvalidRevision;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        admit_lowercase_hex(&value, HUB_COMMIT_WIDTH, "a hub commit")?;
        Ok(Self(value))
    }
}

impl TryFrom<&str> for HubCommitV2 {
    type Error = InvalidRevision;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::try_from(value.to_owned())
    }
}

impl std::fmt::Display for HubCommitV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for HubCommitV2 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::try_from(raw).map_err(serde::de::Error::custom)
    }
}

impl schemars::JsonSchema for HubCommitV2 {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "HubCommitV2".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "description": "An exact hub commit: 40 lowercase hexadecimal characters.",
            "type": "string",
            "pattern": "^[0-9a-f]{40}$",
        })
    }
}

/// A SHA-256 content digest: 64 lowercase hexadecimal characters.
///
/// The identity of a model that is ONE FILE rather than a repository, which is
/// the native Whisper case: whisper.cpp is handed a `.bin` path and a path is
/// not an identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct ContentDigestV2(String);

impl ContentDigestV2 {
    /// The digest, byte for byte.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ContentDigestV2 {
    type Error = InvalidRevision;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        admit_lowercase_hex(&value, CONTENT_DIGEST_WIDTH, "a content digest")?;
        Ok(Self(value))
    }
}

impl TryFrom<&str> for ContentDigestV2 {
    type Error = InvalidRevision;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::try_from(value.to_owned())
    }
}

impl std::fmt::Display for ContentDigestV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ContentDigestV2 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::try_from(raw).map_err(serde::de::Error::custom)
    }
}

impl schemars::JsonSchema for ContentDigestV2 {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "ContentDigestV2".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "description": "A SHA-256 content digest: 64 lowercase hexadecimal characters.",
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        })
    }
}

/// A release tag as a hub publishes it, for example ModelScope's `v2.0.4`.
///
/// A tag is NOT a commit and is never recorded as one. ModelScope's public API
/// exposes no commit behind a tag (checked 2026-09-15: the revisions endpoint
/// returns tag names and timestamps only, and the tag and branch endpoints do
/// not exist), so a tag is the most exact thing that can honestly be said
/// about those models. If ModelScope later exposes the commit, the manifest
/// pins that instead and this variant stops being used for it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct RevisionTagV2(StampSafeText);

impl RevisionTagV2 {
    /// Admit a compile-time literal.
    #[must_use]
    pub const fn from_static(text: &'static str) -> Self {
        Self(StampSafeText::from_static(text))
    }

    /// The tag, byte for byte.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// The tag as stamp-safe text.
    #[must_use]
    pub fn as_stamp_text(&self) -> &StampSafeText {
        &self.0
    }
}

impl TryFrom<String> for RevisionTagV2 {
    type Error = InvalidRevision;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Ok(Self(StampSafeText::try_from(value)?))
    }
}

impl std::fmt::Display for RevisionTagV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for RevisionTagV2 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::try_from(raw).map_err(serde::de::Error::custom)
    }
}

impl schemars::JsonSchema for RevisionTagV2 {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "RevisionTagV2".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "pattern": StampSafeText::json_schema_pattern(),
            "description": "A release tag as a hub publishes it (for example ModelScope's `v2.0.4`). A tag is never recorded as a commit.",
        })
    }
}

/// A provider's own model-selection parameter: Tencent's `EngineModelType`,
/// Aliyun's service identity, Rev.AI's constant.
///
/// The provider chooses what runs behind it and never says what that was, so
/// this is the whole identity for a cloud engine. Never a credential: an
/// Aliyun appkey identifies an ACCOUNT, not a model, and must not appear here.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct ProviderParameterV2(StampSafeText);

impl ProviderParameterV2 {
    /// Admit a compile-time literal.
    #[must_use]
    pub const fn from_static(text: &'static str) -> Self {
        Self(StampSafeText::from_static(text))
    }

    /// The parameter, byte for byte.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// The parameter as stamp-safe text.
    #[must_use]
    pub fn as_stamp_text(&self) -> &StampSafeText {
        &self.0
    }
}

impl TryFrom<String> for ProviderParameterV2 {
    type Error = InvalidRevision;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Ok(Self(StampSafeText::try_from(value)?))
    }
}

impl std::fmt::Display for ProviderParameterV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ProviderParameterV2 {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::try_from(raw).map_err(serde::de::Error::custom)
    }
}

impl schemars::JsonSchema for ProviderParameterV2 {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "ProviderParameterV2".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "pattern": StampSafeText::json_schema_pattern(),
            "description": "A provider's own model-selection parameter (Tencent's EngineModelType, Aliyun's service identity, Rev.AI's constant). Never a credential.",
        })
    }
}

// ---------------------------------------------------------------------------
// Requested and observed revisions
// ---------------------------------------------------------------------------

/// What the plan asked a worker to load.
///
/// [`Self::Unpinned`] is a real state, not a placeholder: a user may name an
/// arbitrary hub model this build has no pin for, and if plan-time resolution
/// cannot reach the hub the honest answer is "load the default and tell me
/// what you got". It has its own admission rule (the worker MUST report a
/// commit) and it can never build a cache key.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[derive(schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RequestedRevisionV2 {
    /// Load exactly this hub commit.
    Commit {
        /// The pinned commit.
        commit: HubCommitV2,
    },
    /// Load exactly this published tag. Recorded as a tag, never as a commit.
    Tag {
        /// The pinned tag.
        tag: RevisionTagV2,
    },
    /// Load the file with exactly these contents.
    ContentDigest {
        /// The pinned digest.
        digest: ContentDigestV2,
    },
    /// Select the provider's model with this parameter.
    ProviderParameter {
        /// The parameter sent to the provider.
        parameter: ProviderParameterV2,
    },
    /// The plan could not pin this model, so the worker loads the hub default
    /// and must report the commit it observed.
    Unpinned,
}

/// What the runtime actually loaded, as the worker observed it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[derive(schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ObservedRevisionV2 {
    /// The runtime reported the hub commit it loaded.
    Commit {
        /// The observed commit.
        commit: HubCommitV2,
    },
    /// The runtime loaded a file with these contents.
    ContentDigest {
        /// The observed digest.
        digest: ContentDigestV2,
    },
    /// The runtime exposes no revision at all. True of every ModelScope loader
    /// and every cloud provider; provider-side drift is invisible, and this
    /// variant is how the type says so rather than inventing a value.
    NotExposed,
}

// ---------------------------------------------------------------------------
// One model, in each of its two spaces
// ---------------------------------------------------------------------------

/// One model the plan pinned, before anything loaded it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[derive(schemars::JsonSchema)]
pub struct RequestedModelV2 {
    /// Which model.
    pub id: ModelIdV2,
    /// Which revision of it.
    pub revision: RequestedRevisionV2,
}

/// One model a worker loaded: what was asked for, and what was seen.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[derive(schemars::JsonSchema)]
pub struct LoadedModelV2 {
    /// Which model.
    pub id: ModelIdV2,
    /// The revision the plan asked for.
    pub requested: RequestedRevisionV2,
    /// The revision the runtime reported.
    pub observed: ObservedRevisionV2,
}

impl LoadedModelV2 {
    /// The revision a provenance stamp records: what was OBSERVED when the
    /// runtime exposed it, otherwise what was requested.
    ///
    /// Total, and honest in both directions. Admission has already proved the
    /// two agree wherever both exist, so this never has to choose between
    /// conflicting facts; it only decides which of two agreeing facts is the
    /// more specific one to write.
    #[must_use]
    pub fn stamp_revision(&self) -> Option<StampSafeText> {
        match &self.observed {
            ObservedRevisionV2::Commit { commit } => {
                StampSafeText::try_from(commit.as_str()).ok()
            }
            ObservedRevisionV2::ContentDigest { digest } => {
                StampSafeText::try_from(digest.as_str()).ok()
            }
            ObservedRevisionV2::NotExposed => match &self.requested {
                RequestedRevisionV2::Commit { commit } => {
                    StampSafeText::try_from(commit.as_str()).ok()
                }
                RequestedRevisionV2::Tag { tag } => Some(tag.as_stamp_text().clone()),
                RequestedRevisionV2::ContentDigest { digest } => {
                    StampSafeText::try_from(digest.as_str()).ok()
                }
                RequestedRevisionV2::ProviderParameter { parameter } => {
                    Some(parameter.as_stamp_text().clone())
                }
                // Nothing was pinned and nothing was reported, so there is no
                // revision to name. The stamp writes the id alone rather than
                // a placeholder.
                RequestedRevisionV2::Unpinned => None,
            },
        }
    }

    /// The `<id>@<revision>` a provenance stamp writes for this model, or the
    /// id alone when neither side named a revision.
    #[must_use]
    pub fn stamp_name(&self) -> StampSafeText {
        match self.stamp_revision() {
            Some(revision) => {
                StampSafeText::join(self.id.as_stamp_text(), [&revision], StampJoiner::At)
            }
            None => self.id.as_stamp_text().clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Roles and composition shapes
// ---------------------------------------------------------------------------

/// The role one model plays inside an engine's composition.
///
/// Closed, and each variant has ONE name ([`Self::wire_name`]) that the
/// provenance stamp also uses, so there is no second table to drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AsrModelRoleV2 {
    /// The speech-recognition model itself.
    Asr,
    /// A forced aligner loaded for word timings.
    Aligner,
    /// A voice-activity model.
    Vad,
    /// A punctuation model.
    Punc,
    /// A ggml weights file loaded in-process by whisper.cpp.
    Ggml,
    /// A provider's engine-model selector.
    EngineModelType,
    /// A provider's service identity.
    Service,
    /// A provider constant that is the whole identity.
    Provider,
}

impl AsrModelRoleV2 {
    /// Every role, in declaration order.
    pub const ALL: [Self; 8] = [
        Self::Asr,
        Self::Aligner,
        Self::Vad,
        Self::Punc,
        Self::Ggml,
        Self::EngineModelType,
        Self::Service,
        Self::Provider,
    ];

    /// The role's one name: its wire form and the text a stamp writes.
    #[must_use]
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::Asr => "asr",
            Self::Aligner => "aligner",
            Self::Vad => "vad",
            Self::Punc => "punc",
            Self::Ggml => "ggml",
            Self::EngineModelType => "engine_model_type",
            Self::Service => "service",
            Self::Provider => "provider",
        }
    }

    /// The role name as stamp-safe text, checked at compile time.
    #[must_use]
    pub fn stamp_name(self) -> StampSafeText {
        match self {
            Self::Asr => const { StampSafeText::from_static(AsrModelRoleV2::Asr.wire_name()) },
            Self::Aligner => {
                const { StampSafeText::from_static(AsrModelRoleV2::Aligner.wire_name()) }
            }
            Self::Vad => const { StampSafeText::from_static(AsrModelRoleV2::Vad.wire_name()) },
            Self::Punc => const { StampSafeText::from_static(AsrModelRoleV2::Punc.wire_name()) },
            Self::Ggml => const { StampSafeText::from_static(AsrModelRoleV2::Ggml.wire_name()) },
            Self::EngineModelType => {
                const { StampSafeText::from_static(AsrModelRoleV2::EngineModelType.wire_name()) }
            }
            Self::Service => {
                const { StampSafeText::from_static(AsrModelRoleV2::Service.wire_name()) }
            }
            Self::Provider => {
                const { StampSafeText::from_static(AsrModelRoleV2::Provider.wire_name()) }
            }
        }
    }
}

impl std::fmt::Display for AsrModelRoleV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.wire_name())
    }
}

/// Which engine's composition a model set describes.
///
/// THE owner of "which compositions exist". Both the requested and the loaded
/// composition report their shape through it, so the two enums cannot drift
/// into describing different engines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AsrCompositionShapeV2 {
    /// Hugging Face Whisper, stock or fine-tune: one model.
    Whisper,
    /// Qwen3-ASR: the model and its forced aligner.
    Qwen,
    /// FunASR Paraformer: the model, its voice-activity and punctuation models.
    Paraformer,
    /// FunASR SenseVoice: the model and its voice-activity model.
    SenseVoice,
    /// whisper.cpp in-process: one ggml weights file.
    NativeWhisper,
    /// Tencent: the requested engine-model type.
    Tencent,
    /// Aliyun: a fixed service identity.
    Aliyun,
    /// Rev.AI: a code constant.
    Rev,
}

impl AsrCompositionShapeV2 {
    /// The shapes one backend may report.
    ///
    /// Exhaustive over the backend, with no catch-all, so a new ASR backend
    /// must say which compositions it can produce. FunAudio is the one backend
    /// with two: `funaudio` spans materially different models, and which one
    /// runs depends on the checkpoint the request selected.
    #[must_use]
    pub const fn for_backend(backend: AsrBackendV2) -> &'static [Self] {
        match backend {
            AsrBackendV2::LocalWhisper | AsrBackendV2::WhisperHub => &[Self::Whisper],
            AsrBackendV2::HkQwen => &[Self::Qwen],
            AsrBackendV2::HkFunaudio => &[Self::Paraformer, Self::SenseVoice],
            AsrBackendV2::HkTencent => &[Self::Tencent],
            AsrBackendV2::HkAliyun => &[Self::Aliyun],
            AsrBackendV2::Revai => &[Self::Rev],
        }
    }

    /// Whether `backend` may report this composition.
    #[must_use]
    pub fn admits_backend(self, backend: AsrBackendV2) -> bool {
        Self::for_backend(backend).contains(&self)
    }

    /// The shape's wire name, for diagnostics.
    #[must_use]
    pub const fn wire_name(self) -> &'static str {
        match self {
            Self::Whisper => "whisper",
            Self::Qwen => "qwen",
            Self::Paraformer => "paraformer",
            Self::SenseVoice => "sense_voice",
            Self::NativeWhisper => "native_whisper",
            Self::Tencent => "tencent",
            Self::Aliyun => "aliyun",
            Self::Rev => "rev",
        }
    }
}

impl std::fmt::Display for AsrCompositionShapeV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.wire_name())
    }
}

// ---------------------------------------------------------------------------
// The two compositions
// ---------------------------------------------------------------------------

/// Every model of one engine's composition, as the plan requested them.
///
/// Carried on the ASR request. Struct variants with required fields: a Qwen
/// request without an aligner, or a cloud request carrying a hub commit, has
/// no representation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[derive(schemars::JsonSchema)]
#[serde(tag = "engine", rename_all = "snake_case")]
pub enum AsrRequestedModelsV2 {
    /// Hugging Face Whisper, stock or fine-tune.
    Whisper {
        /// The Whisper checkpoint.
        asr: RequestedModelV2,
    },
    /// Qwen3-ASR and the forced aligner it needs for word timings.
    Qwen {
        /// The Qwen3-ASR checkpoint.
        asr: RequestedModelV2,
        /// The forced aligner loaded with it.
        aligner: RequestedModelV2,
    },
    /// FunASR Paraformer, with its voice-activity and punctuation models.
    Paraformer {
        /// The Paraformer checkpoint.
        asr: RequestedModelV2,
        /// The voice-activity model.
        vad: RequestedModelV2,
        /// The punctuation model.
        punc: RequestedModelV2,
    },
    /// FunASR SenseVoice, with its voice-activity model.
    SenseVoice {
        /// The SenseVoice checkpoint.
        asr: RequestedModelV2,
        /// The voice-activity model.
        vad: RequestedModelV2,
    },
    /// Tencent cloud ASR.
    Tencent {
        /// The requested engine-model type.
        engine_model_type: RequestedModelV2,
    },
    /// Aliyun cloud ASR.
    Aliyun {
        /// The fixed service identity.
        service: RequestedModelV2,
    },
    /// Rev.AI cloud ASR.
    Rev {
        /// The provider constant.
        provider: RequestedModelV2,
    },
}

impl AsrRequestedModelsV2 {
    /// Which composition this is.
    #[must_use]
    pub const fn shape(&self) -> AsrCompositionShapeV2 {
        match self {
            Self::Whisper { .. } => AsrCompositionShapeV2::Whisper,
            Self::Qwen { .. } => AsrCompositionShapeV2::Qwen,
            Self::Paraformer { .. } => AsrCompositionShapeV2::Paraformer,
            Self::SenseVoice { .. } => AsrCompositionShapeV2::SenseVoice,
            Self::Tencent { .. } => AsrCompositionShapeV2::Tencent,
            Self::Aliyun { .. } => AsrCompositionShapeV2::Aliyun,
            Self::Rev { .. } => AsrCompositionShapeV2::Rev,
        }
    }

    /// Every model, paired with the role it plays, primary role first.
    #[must_use]
    pub fn models(&self) -> Vec<(AsrModelRoleV2, &RequestedModelV2)> {
        match self {
            Self::Whisper { asr } => vec![(AsrModelRoleV2::Asr, asr)],
            Self::Qwen { asr, aligner } => vec![
                (AsrModelRoleV2::Asr, asr),
                (AsrModelRoleV2::Aligner, aligner),
            ],
            Self::Paraformer { asr, vad, punc } => vec![
                (AsrModelRoleV2::Asr, asr),
                (AsrModelRoleV2::Vad, vad),
                (AsrModelRoleV2::Punc, punc),
            ],
            Self::SenseVoice { asr, vad } => {
                vec![(AsrModelRoleV2::Asr, asr), (AsrModelRoleV2::Vad, vad)]
            }
            Self::Tencent { engine_model_type } => {
                vec![(AsrModelRoleV2::EngineModelType, engine_model_type)]
            }
            Self::Aliyun { service } => vec![(AsrModelRoleV2::Service, service)],
            Self::Rev { provider } => vec![(AsrModelRoleV2::Provider, provider)],
        }
    }

    /// Whether every model of this composition names an exact revision.
    ///
    /// The predicate behind the pinned-plan proof: a composition with any
    /// [`RequestedRevisionV2::Unpinned`] member can never build a cache key.
    #[must_use]
    pub fn is_fully_pinned(&self) -> bool {
        self.models()
            .iter()
            .all(|(_, model)| model.revision != RequestedRevisionV2::Unpinned)
    }

    /// The text a cache namespace records for this plan, when every member is
    /// pinned.
    ///
    /// Shaped `<composition>|<role>=<id>@<revision>|...` with the roles in
    /// composition order, so two plans differing in ANY member land in
    /// different namespaces and a row produced under one can never be read back
    /// for the other. That is the whole job: a cache hit is only sound when the
    /// stored row provably came from the same weights.
    ///
    /// `None` when any member floats. An unpinned model cannot make that
    /// promise, so such a plan gets no namespace at all rather than a namespace
    /// that silently groups different weights together.
    #[must_use]
    pub fn pinned_namespace_text(&self) -> Option<String> {
        let mut text = self.shape().to_string();
        for (role, model) in self.models() {
            let revision = match &model.revision {
                RequestedRevisionV2::Commit { commit } => commit.as_str().to_owned(),
                RequestedRevisionV2::Tag { tag } => tag.as_str().to_owned(),
                RequestedRevisionV2::ContentDigest { digest } => digest.as_str().to_owned(),
                RequestedRevisionV2::ProviderParameter { parameter } => {
                    parameter.as_str().to_owned()
                }
                // The one exit. Checked HERE rather than by calling
                // `is_fully_pinned` first, so the predicate is not evaluated
                // twice and cannot drift from what this function accepts.
                RequestedRevisionV2::Unpinned => return None,
            };
            text.push('|');
            text.push_str(&format!("{role}={}@{revision}", model.id));
        }
        Some(text)
    }
}

/// Every model a worker loaded for one ASR response, with what was asked for
/// and what was seen.
///
/// Required on both ASR result types, so a response can never be applied
/// without naming the models behind it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[derive(schemars::JsonSchema)]
#[serde(tag = "engine", rename_all = "snake_case")]
pub enum AsrModelIdentityV2 {
    /// Hugging Face Whisper, stock or fine-tune.
    Whisper {
        /// The Whisper checkpoint that ran.
        asr: LoadedModelV2,
    },
    /// Qwen3-ASR and its forced aligner.
    Qwen {
        /// The Qwen3-ASR checkpoint that ran.
        asr: LoadedModelV2,
        /// The forced aligner loaded with it.
        aligner: LoadedModelV2,
    },
    /// FunASR Paraformer, with its voice-activity and punctuation models.
    Paraformer {
        /// The Paraformer checkpoint that ran.
        asr: LoadedModelV2,
        /// The voice-activity model.
        vad: LoadedModelV2,
        /// The punctuation model.
        punc: LoadedModelV2,
    },
    /// FunASR SenseVoice, with its voice-activity model.
    SenseVoice {
        /// The SenseVoice checkpoint that ran.
        asr: LoadedModelV2,
        /// The voice-activity model.
        vad: LoadedModelV2,
    },
    /// whisper.cpp run in-process from one ggml weights file.
    NativeWhisper {
        /// The ggml weights file that was loaded.
        ggml: LoadedModelV2,
    },
    /// Tencent cloud ASR.
    Tencent {
        /// The engine-model type the request named.
        engine_model_type: LoadedModelV2,
    },
    /// Aliyun cloud ASR.
    Aliyun {
        /// The service identity.
        service: LoadedModelV2,
    },
    /// Rev.AI cloud ASR.
    Rev {
        /// The provider constant.
        provider: LoadedModelV2,
    },
}

impl AsrModelIdentityV2 {
    /// Which composition this is.
    #[must_use]
    pub const fn shape(&self) -> AsrCompositionShapeV2 {
        match self {
            Self::Whisper { .. } => AsrCompositionShapeV2::Whisper,
            Self::Qwen { .. } => AsrCompositionShapeV2::Qwen,
            Self::Paraformer { .. } => AsrCompositionShapeV2::Paraformer,
            Self::SenseVoice { .. } => AsrCompositionShapeV2::SenseVoice,
            Self::NativeWhisper { .. } => AsrCompositionShapeV2::NativeWhisper,
            Self::Tencent { .. } => AsrCompositionShapeV2::Tencent,
            Self::Aliyun { .. } => AsrCompositionShapeV2::Aliyun,
            Self::Rev { .. } => AsrCompositionShapeV2::Rev,
        }
    }

    /// Every model, paired with the role it plays, primary role first.
    #[must_use]
    pub fn models(&self) -> Vec<(AsrModelRoleV2, &LoadedModelV2)> {
        match self {
            Self::Whisper { asr } => vec![(AsrModelRoleV2::Asr, asr)],
            Self::Qwen { asr, aligner } => vec![
                (AsrModelRoleV2::Asr, asr),
                (AsrModelRoleV2::Aligner, aligner),
            ],
            Self::Paraformer { asr, vad, punc } => vec![
                (AsrModelRoleV2::Asr, asr),
                (AsrModelRoleV2::Vad, vad),
                (AsrModelRoleV2::Punc, punc),
            ],
            Self::SenseVoice { asr, vad } => {
                vec![(AsrModelRoleV2::Asr, asr), (AsrModelRoleV2::Vad, vad)]
            }
            Self::NativeWhisper { ggml } => vec![(AsrModelRoleV2::Ggml, ggml)],
            Self::Tencent { engine_model_type } => {
                vec![(AsrModelRoleV2::EngineModelType, engine_model_type)]
            }
            Self::Aliyun { service } => vec![(AsrModelRoleV2::Service, service)],
            Self::Rev { provider } => vec![(AsrModelRoleV2::Provider, provider)],
        }
    }

    /// The primary model: the one `asr_model=` names before its auxiliaries.
    ///
    /// Total: every composition has at least one model, and `models()` puts
    /// the primary role first in every arm.
    #[must_use]
    pub fn primary(&self) -> (AsrModelRoleV2, &LoadedModelV2) {
        match self {
            Self::Whisper { asr }
            | Self::Qwen { asr, .. }
            | Self::Paraformer { asr, .. }
            | Self::SenseVoice { asr, .. } => (AsrModelRoleV2::Asr, asr),
            Self::NativeWhisper { ggml } => (AsrModelRoleV2::Ggml, ggml),
            Self::Tencent { engine_model_type } => {
                (AsrModelRoleV2::EngineModelType, engine_model_type)
            }
            Self::Aliyun { service } => (AsrModelRoleV2::Service, service),
            Self::Rev { provider } => (AsrModelRoleV2::Provider, provider),
        }
    }

    /// The auxiliary models, in role order: everything but the primary.
    #[must_use]
    pub fn auxiliaries(&self) -> Vec<(AsrModelRoleV2, &LoadedModelV2)> {
        let (primary_role, _) = self.primary();
        self.models()
            .into_iter()
            .filter(|(role, _)| *role != primary_role)
            .collect()
    }

    /// The value `asr_model=` records: `<id>@<revision>`, with
    /// `+<role>:<id>@<revision>` appended for each auxiliary model.
    ///
    /// Total, because every part is already stamp-safe text and the joins add
    /// only `@`, `+` and `:`.
    #[must_use]
    pub fn stamp_value(&self) -> StampSafeText {
        let (_, primary) = self.primary();
        let mut value = primary.stamp_name();
        for (role, model) in self.auxiliaries() {
            let qualified = StampSafeText::join(
                &role.stamp_name(),
                [&model.stamp_name()],
                StampJoiner::Colon,
            );
            value = StampSafeText::join(&value, [&qualified], StampJoiner::Plus);
        }
        value
    }

    /// Refuse a response whose models are not the ones the plan asked for.
    ///
    /// Four ways a response can disagree, each named with BOTH sides so an
    /// operator can see what moved. The per-model checks run in role order, so
    /// the first disagreement reported is the outermost one.
    pub fn admit(
        &self,
        requested: &AsrRequestedModelsV2,
        backend: AsrBackendV2,
    ) -> Result<(), AsrIdentityMismatchV2> {
        if !self.shape().admits_backend(backend) {
            return Err(AsrIdentityMismatchV2::BackendComposition {
                backend,
                reported: self.shape(),
            });
        }
        if self.shape() != requested.shape() {
            return Err(AsrIdentityMismatchV2::Composition {
                requested: requested.shape(),
                reported: self.shape(),
            });
        }

        // The shapes agree, so `models()` returns the same roles in the same
        // order on both sides; zipping them cannot mispair.
        for ((role, reported), (_, asked)) in self.models().into_iter().zip(requested.models()) {
            if reported.id != asked.id {
                return Err(AsrIdentityMismatchV2::ModelId {
                    role,
                    requested: asked.id.clone(),
                    reported: reported.id.clone(),
                });
            }
            if reported.requested != asked.revision {
                return Err(AsrIdentityMismatchV2::EchoedRequest {
                    role,
                    requested: asked.revision.clone(),
                    reported: reported.requested.clone(),
                });
            }
            admit_observation(role, &asked.revision, &reported.observed)?;
        }
        Ok(())
    }
}

/// Whether what a worker observed can follow from what was asked.
///
/// The whole matrix, written out. An exact pin must be observed exactly; a tag
/// may come back unexposed (ModelScope) or as the commit behind it if a hub
/// ever exposes one; a provider parameter can only come back unexposed; and an
/// unpinned model MUST come back with a commit, which is the entire point of
/// leaving it unpinned rather than refusing the job.
fn admit_observation(
    role: AsrModelRoleV2,
    requested: &RequestedRevisionV2,
    observed: &ObservedRevisionV2,
) -> Result<(), AsrIdentityMismatchV2> {
    let agrees = match (requested, observed) {
        (
            RequestedRevisionV2::Commit { commit },
            ObservedRevisionV2::Commit {
                commit: seen_commit,
            },
        ) => commit == seen_commit,
        (
            RequestedRevisionV2::ContentDigest { digest },
            ObservedRevisionV2::ContentDigest {
                digest: seen_digest,
            },
        ) => digest == seen_digest,
        (RequestedRevisionV2::Tag { .. }, ObservedRevisionV2::NotExposed)
        | (RequestedRevisionV2::Tag { .. }, ObservedRevisionV2::Commit { .. })
        | (RequestedRevisionV2::ProviderParameter { .. }, ObservedRevisionV2::NotExposed)
        | (RequestedRevisionV2::Unpinned, ObservedRevisionV2::Commit { .. })
        // An unpinned model on a hub that exposes NOTHING. ModelScope is the
        // case: a user may name a Paraformer checkpoint this build does not
        // pin, and that loader can report no commit for it, so the honest
        // record is "nothing was asked for and nothing was told". Weak, and
        // deliberately so: such a composition is not fully pinned, and
        // `is_fully_pinned` already stops it building a cache key.
        | (RequestedRevisionV2::Unpinned, ObservedRevisionV2::NotExposed) => true,
        (RequestedRevisionV2::Commit { .. }, _)
        | (RequestedRevisionV2::ContentDigest { .. }, _)
        | (RequestedRevisionV2::Tag { .. }, ObservedRevisionV2::ContentDigest { .. })
        | (RequestedRevisionV2::ProviderParameter { .. }, _)
        | (RequestedRevisionV2::Unpinned, ObservedRevisionV2::ContentDigest { .. }) => false,
    };
    if agrees {
        return Ok(());
    }
    Err(AsrIdentityMismatchV2::Observation {
        role,
        requested: requested.clone(),
        observed: observed.clone(),
    })
}

/// Why an ASR response's models were refused.
///
/// Every variant names both sides. A refusal that said only "model mismatch"
/// would leave an operator unable to tell a moved hub repository from a worker
/// running a different build.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AsrIdentityMismatchV2 {
    /// The response describes a composition the requested backend cannot run.
    #[error(
        "ASR response reports a {reported} model composition, which the {backend:?} backend \
         cannot produce"
    )]
    BackendComposition {
        /// The backend the request selected.
        backend: AsrBackendV2,
        /// The composition the response reported.
        reported: AsrCompositionShapeV2,
    },
    /// The response describes a different composition than the plan pinned.
    #[error("ASR response reports a {reported} model composition; the plan pinned {requested}")]
    Composition {
        /// The composition the plan pinned.
        requested: AsrCompositionShapeV2,
        /// The composition the response reported.
        reported: AsrCompositionShapeV2,
    },
    /// One role names a different model than the plan pinned.
    #[error("ASR response loaded {reported} for the {role} model; the plan pinned {requested}")]
    ModelId {
        /// Which model.
        role: AsrModelRoleV2,
        /// The id the plan pinned.
        requested: ModelIdV2,
        /// The id the worker reported.
        reported: ModelIdV2,
    },
    /// The worker echoed back a different request than it was sent, which means
    /// it is not answering this plan.
    #[error(
        "ASR response echoes a different requested revision for the {role} model: the plan asked \
         for {requested:?} and the worker echoed {reported:?}"
    )]
    EchoedRequest {
        /// Which model.
        role: AsrModelRoleV2,
        /// The revision the plan asked for.
        requested: RequestedRevisionV2,
        /// The revision the worker echoed.
        reported: RequestedRevisionV2,
    },
    /// The worker loaded something other than what it was asked to load.
    #[error(
        "ASR response loaded {observed:?} for the {role} model, which does not follow from the \
         requested {requested:?}"
    )]
    Observation {
        /// Which model.
        role: AsrModelRoleV2,
        /// The revision the plan asked for.
        requested: RequestedRevisionV2,
        /// The revision the worker observed.
        observed: ObservedRevisionV2,
    },
}

#[cfg(test)]
#[allow(clippy::expect_used)] // fixture construction; the refusals are the assertions
mod tests {
    use super::*;

    fn commit(hex: &str) -> RequestedRevisionV2 {
        RequestedRevisionV2::Commit {
            commit: HubCommitV2::try_from(hex).expect("valid commit"),
        }
    }

    fn observed_commit(hex: &str) -> ObservedRevisionV2 {
        ObservedRevisionV2::Commit {
            commit: HubCommitV2::try_from(hex).expect("valid commit"),
        }
    }

    const WHISPER: &str = "06f233fe06e710322aca913c1bc4249a0d71fce1";
    const ALIGNER: &str = "c07281df297b9905d24a508279258cccf987a064";

    fn whisper_requested() -> AsrRequestedModelsV2 {
        AsrRequestedModelsV2::Whisper {
            asr: RequestedModelV2 {
                id: ModelIdV2::from_static("openai/whisper-large-v3"),
                revision: commit(WHISPER),
            },
        }
    }

    fn whisper_loaded() -> AsrModelIdentityV2 {
        AsrModelIdentityV2::Whisper {
            asr: LoadedModelV2 {
                id: ModelIdV2::from_static("openai/whisper-large-v3"),
                requested: commit(WHISPER),
                observed: observed_commit(WHISPER),
            },
        }
    }

    /// A digest is a fixed-width lowercase hex string; anything else is not a
    /// revision at all, and is refused where it enters rather than travelling.
    #[test]
    fn revision_newtypes_refuse_anything_but_their_own_shape() {
        assert!(HubCommitV2::try_from(WHISPER).is_ok());
        assert!(HubCommitV2::try_from("06F233FE06E710322ACA913C1BC4249A0D71FCE1").is_err());
        assert!(HubCommitV2::try_from("06f233fe").is_err());
        assert!(
            HubCommitV2::try_from(
                "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2"
            )
            .is_err(),
            "a 64-character content digest is not a 40-character commit"
        );
        assert!(
            ContentDigestV2::try_from(
                "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2"
            )
            .is_ok()
        );
        assert!(ContentDigestV2::try_from(WHISPER).is_err());
    }

    /// The cache namespace text is pinned BYTE FOR BYTE, because that is what
    /// it is for: rows one run writes are read back by later runs only if these
    /// bytes match, so a change here is a silent cache-wide invalidation and
    /// must be a deliberate edit to this assertion rather than a side effect.
    #[test]
    fn a_pinned_composition_renders_its_namespace_text() {
        assert_eq!(
            whisper_requested().pinned_namespace_text(),
            Some(format!("whisper|asr=openai/whisper-large-v3@{WHISPER}"))
        );
    }

    /// A floating member yields no namespace at all. An unpinned model cannot
    /// promise a stored row came from the same weights, so such a plan gets no
    /// key rather than one that would pool different weights together.
    #[test]
    fn a_floating_composition_has_no_namespace_text() {
        let floating = AsrRequestedModelsV2::Whisper {
            asr: RequestedModelV2 {
                id: ModelIdV2::from_static("openai/whisper-large-v3"),
                revision: RequestedRevisionV2::Unpinned,
            },
        };
        assert_eq!(floating.pinned_namespace_text(), None);
    }

    /// Moving any revision moves the namespace. This is the property the cache
    /// depends on, and it is what the engine name alone could never provide.
    #[test]
    fn changing_a_revision_changes_the_namespace_text() {
        let moved = AsrRequestedModelsV2::Whisper {
            asr: RequestedModelV2 {
                id: ModelIdV2::from_static("openai/whisper-large-v3"),
                revision: commit(ALIGNER),
            },
        };
        assert_ne!(
            whisper_requested().pinned_namespace_text(),
            moved.pinned_namespace_text()
        );
    }

    /// The happy path: the worker loaded what it was told to load.
    #[test]
    fn a_matching_response_is_admitted() {
        whisper_loaded()
            .admit(&whisper_requested(), AsrBackendV2::LocalWhisper)
            .expect("the worker loaded exactly the pinned revision");
    }

    /// A worker that loaded a DIFFERENT commit than the one pinned is the
    /// whole failure this workstream exists to catch, and the refusal must
    /// name both revisions.
    #[test]
    fn a_worker_that_loaded_another_commit_is_refused_naming_both() {
        let drifted = AsrModelIdentityV2::Whisper {
            asr: LoadedModelV2 {
                id: ModelIdV2::from_static("openai/whisper-large-v3"),
                requested: commit(WHISPER),
                observed: observed_commit(ALIGNER),
            },
        };
        let refusal = drifted
            .admit(&whisper_requested(), AsrBackendV2::LocalWhisper)
            .expect_err("a different loaded commit must be refused");
        let message = refusal.to_string();
        assert!(message.contains(ALIGNER), "{message}");
        assert!(message.contains(WHISPER), "{message}");
    }

    /// A composition the backend cannot produce is refused before any
    /// per-model comparison: a cloud engine cannot report hub models.
    #[test]
    fn a_composition_the_backend_cannot_produce_is_refused() {
        let refusal = whisper_loaded()
            .admit(&whisper_requested(), AsrBackendV2::HkAliyun)
            .expect_err("Aliyun cannot report a Whisper composition");
        assert!(matches!(
            refusal,
            AsrIdentityMismatchV2::BackendComposition { .. }
        ));
    }

    /// FunAudio is the one backend with two legal compositions, because the
    /// checkpoint decides which models load.
    #[test]
    fn funaudio_admits_both_of_its_compositions_and_nothing_else() {
        let shapes = AsrCompositionShapeV2::for_backend(AsrBackendV2::HkFunaudio);
        assert!(shapes.contains(&AsrCompositionShapeV2::Paraformer));
        assert!(shapes.contains(&AsrCompositionShapeV2::SenseVoice));
        assert!(!shapes.contains(&AsrCompositionShapeV2::Whisper));
        for backend in AsrBackendV2::ALL {
            assert!(
                !AsrCompositionShapeV2::for_backend(backend).is_empty(),
                "every backend must name at least one composition"
            );
        }
    }

    /// A tag may come back unexposed, and that is not a defect: ModelScope
    /// tells the loader nothing. A provider parameter likewise. But an exact
    /// pin that comes back unexposed IS a defect, because the loader was
    /// supposed to prove what it loaded.
    #[test]
    fn observation_rules_follow_from_what_was_requested() {
        let tag = RequestedRevisionV2::Tag {
            tag: RevisionTagV2::from_static("v2.0.4"),
        };
        assert!(admit_observation(AsrModelRoleV2::Asr, &tag, &ObservedRevisionV2::NotExposed).is_ok());
        assert!(
            admit_observation(
                AsrModelRoleV2::Asr,
                &commit(WHISPER),
                &ObservedRevisionV2::NotExposed
            )
            .is_err(),
            "a pinned commit must be proved, not assumed"
        );
        assert!(
            admit_observation(
                AsrModelRoleV2::Asr,
                &RequestedRevisionV2::Unpinned,
                &observed_commit(WHISPER)
            )
            .is_ok(),
            "an unpinned model on a hub that exposes commits must report one"
        );
        // An unpinned model on a hub that exposes NOTHING (ModelScope) can
        // only be recorded as unpinned and unexposed. Deliberately weak, and
        // unreachable for a cache key: such a composition is not fully pinned.
        assert!(
            admit_observation(
                AsrModelRoleV2::Asr,
                &RequestedRevisionV2::Unpinned,
                &ObservedRevisionV2::NotExposed
            )
            .is_ok()
        );
        // The rule still has teeth: nothing unpinned was ever a FILE, so a
        // content digest there means the two sides disagree about the model.
        assert!(
            admit_observation(
                AsrModelRoleV2::Asr,
                &RequestedRevisionV2::Unpinned,
                &ObservedRevisionV2::ContentDigest {
                    digest: ContentDigestV2::try_from(
                        "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2"
                    )
                    .expect("valid digest"),
                }
            )
            .is_err()
        );
    }

    /// The pinned-plan predicate: one unpinned member makes the whole
    /// composition unpinned, and therefore uncacheable.
    #[test]
    fn a_composition_is_pinned_only_when_every_member_is() {
        assert!(whisper_requested().is_fully_pinned());
        let floating = AsrRequestedModelsV2::Qwen {
            asr: RequestedModelV2 {
                id: ModelIdV2::from_static("someone/a-qwen-fine-tune"),
                revision: RequestedRevisionV2::Unpinned,
            },
            aligner: RequestedModelV2 {
                id: ModelIdV2::from_static("Qwen/Qwen3-ForcedAligner-0.6B-hf"),
                revision: commit(ALIGNER),
            },
        };
        assert!(!floating.is_fully_pinned());
    }

    /// The stamp value: the primary model, then each auxiliary qualified by
    /// its role. This is the text `asr_model=` carries.
    #[test]
    fn the_stamp_value_names_the_primary_model_then_its_auxiliaries() {
        let qwen = AsrModelIdentityV2::Qwen {
            asr: LoadedModelV2 {
                id: ModelIdV2::from_static("Qwen/Qwen3-ASR-1.7B-hf"),
                requested: commit(WHISPER),
                observed: observed_commit(WHISPER),
            },
            aligner: LoadedModelV2 {
                id: ModelIdV2::from_static("Qwen/Qwen3-ForcedAligner-0.6B-hf"),
                requested: commit(ALIGNER),
                observed: observed_commit(ALIGNER),
            },
        };
        assert_eq!(
            qwen.stamp_value().as_str(),
            format!(
                "Qwen/Qwen3-ASR-1.7B-hf@{WHISPER}+aligner:Qwen/Qwen3-ForcedAligner-0.6B-hf@{ALIGNER}"
            )
        );
    }

    /// A ModelScope model stamps the TAG it was pinned to, because that is the
    /// most exact thing that can honestly be said about it.
    #[test]
    fn an_unexposed_revision_stamps_what_was_requested() {
        let paraformer = LoadedModelV2 {
            id: ModelIdV2::from_static(
                "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            ),
            requested: RequestedRevisionV2::Tag {
                tag: RevisionTagV2::from_static("v2.0.4"),
            },
            observed: ObservedRevisionV2::NotExposed,
        };
        assert!(paraformer.stamp_name().as_str().ends_with("@v2.0.4"));
    }

    /// WIRE FORMAT: the identity roundtrips, and the two revision spaces stay
    /// distinguishable on the wire (two separate functions must agree, which
    /// no type can pin).
    #[test]
    fn identity_roundtrips_and_keeps_requested_and_observed_apart() {
        let identity = whisper_loaded();
        let json = serde_json::to_string(&identity).expect("serializes");
        let back: AsrModelIdentityV2 = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back, identity);

        let value: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        assert_eq!(value["engine"], "whisper");
        assert_eq!(value["asr"]["requested"]["kind"], "commit");
        assert_eq!(value["asr"]["observed"]["kind"], "commit");
        assert_eq!(value["asr"]["id"], "openai/whisper-large-v3");
    }
}
