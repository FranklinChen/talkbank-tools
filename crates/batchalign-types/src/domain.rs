//! Domain newtypes and small enums shared across batchalign crates.
//!
//! These are re-exported from [`super::api`] for backward compatibility.

use std::borrow::Cow;
use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

// ---------------------------------------------------------------------------
// Domain newtypes (shared across modules, re-exported from lib.rs)
// ---------------------------------------------------------------------------

validated_string_id!(
    /// Server-assigned identifier for a job (non-empty).
    pub JobId
);

/// Closed released command vocabulary used at all Rust seams.
///
/// This is the single canonical command type. Unknown command strings are
/// rejected at deserialization boundaries (HTTP 422, DB recovery skip).
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    serde::Deserialize,
    utoipa::ToSchema,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ReleasedCommand {
    Align,
    Transcribe,
    TranscribeS,
    Translate,
    Morphotag,
    Coref,
    Utseg,
    Benchmark,
    Opensmile,
    Compare,
    Avqi,
    Diarize,
    SpeakerIdentify,
}

/// Error returned when one string is not a released command name.
#[derive(Debug, Clone, thiserror::Error)]
#[error("unknown released command \"{0}\"")]
pub struct InvalidReleasedCommand(pub String);

impl ReleasedCommand {
    /// All released commands in a stable contributor-facing order.
    pub const ALL: [Self; 13] = [
        Self::Align,
        Self::Transcribe,
        Self::TranscribeS,
        Self::Translate,
        Self::Morphotag,
        Self::Coref,
        Self::Utseg,
        Self::Benchmark,
        Self::Opensmile,
        Self::Compare,
        Self::Avqi,
        Self::Diarize,
        Self::SpeakerIdentify,
    ];

    /// Parse one untrusted released-command token.
    pub fn parse_untrusted(value: &str) -> Result<Self, InvalidReleasedCommand> {
        Self::try_from(value.trim())
    }

    /// Return the canonical snake_case released command name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Align => "align",
            Self::Transcribe => "transcribe",
            Self::TranscribeS => "transcribe_s",
            Self::Translate => "translate",
            Self::Morphotag => "morphotag",
            Self::Coref => "coref",
            Self::Utseg => "utseg",
            Self::Benchmark => "benchmark",
            Self::Opensmile => "opensmile",
            Self::Compare => "compare",
            Self::Avqi => "avqi",
            Self::Diarize => "diarize",
            Self::SpeakerIdentify => "speaker_identify",
        }
    }

    /// Return the canonical wire/storage spelling.
    pub const fn as_wire_name(self) -> &'static str {
        self.as_str()
    }

    /// Return whether this released command requires client-local audio access.
    pub const fn uses_local_audio(self) -> bool {
        matches!(
            self,
            Self::Transcribe
                | Self::TranscribeS
                | Self::Benchmark
                | Self::Avqi
                | Self::Diarize
                | Self::SpeakerIdentify
        )
    }
}

impl std::fmt::Display for ReleasedCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl AsRef<str> for ReleasedCommand {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl PartialEq<&str> for ReleasedCommand {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl TryFrom<&str> for ReleasedCommand {
    type Error = InvalidReleasedCommand;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "align" => Ok(Self::Align),
            "transcribe" => Ok(Self::Transcribe),
            "transcribe_s" => Ok(Self::TranscribeS),
            "translate" => Ok(Self::Translate),
            "morphotag" => Ok(Self::Morphotag),
            "coref" => Ok(Self::Coref),
            "utseg" => Ok(Self::Utseg),
            "benchmark" => Ok(Self::Benchmark),
            "opensmile" => Ok(Self::Opensmile),
            "compare" => Ok(Self::Compare),
            "avqi" => Ok(Self::Avqi),
            "diarize" => Ok(Self::Diarize),
            "speaker_identify" => Ok(Self::SpeakerIdentify),
            other => Err(InvalidReleasedCommand(other.to_owned())),
        }
    }
}

/// Borrowed CHAT document text at a contributor-facing boundary.
///
/// This wrapper is intentionally lightweight: it prevents workflow/request
/// types from collapsing back into raw `&str` while still borrowing the
/// underlying document text without allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChatText<'a>(&'a str);

impl<'a> ChatText<'a> {
    /// Wrap one borrowed CHAT document string.
    pub fn new(text: &'a str) -> Self {
        Self(text)
    }

    /// Borrow the underlying CHAT string.
    pub fn as_str(self) -> &'a str {
        self.0
    }
}

impl std::fmt::Display for ChatText<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl<'a> From<&'a str> for ChatText<'a> {
    fn from(value: &'a str) -> Self {
        Self::new(value)
    }
}

impl std::ops::Deref for ChatText<'_> {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl AsRef<str> for ChatText<'_> {
    fn as_ref(&self) -> &str {
        self.0
    }
}

// ---------------------------------------------------------------------------
// LanguageCode3: validated 3-letter ISO 639-3 language code
// ---------------------------------------------------------------------------

/// 3-letter ISO 639-3 language code (e.g. `"eng"`, `"spa"`).
///
/// Construction validates that the value is exactly 3 ASCII alphabetic
/// characters, lowercased. Sentinel values like `"auto"` are rejected, use
/// [`LanguageSpec`] at boundaries where auto-detection is meaningful.
// `PartialOrd`/`Ord` are derived (added 2026-07-29) so a validated code can key
// a `BTreeMap` directly, which is what `BatchInferProgress.language_groups`
// needs for deterministic JSON. Ordering is the underlying string's, and since
// construction lowercases and length-checks, it is a total order over exactly
// the valid codes.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    serde::Serialize,
    utoipa::ToSchema,
    schemars::JsonSchema,
)]
#[serde(transparent)]
pub struct LanguageCode3(String);

/// Error returned when a string is not a valid 3-letter ISO 639-3 code.
#[derive(Debug, Clone, thiserror::Error)]
#[error("invalid language code \"{0}\": expected 3 ASCII letters (e.g. \"eng\", \"spa\")")]
pub struct InvalidLanguageCode(pub String);

impl LanguageCode3 {
    // -- Well-known language codes (use these instead of string literals) --

    /// English (`"eng"`).
    pub fn eng() -> Self {
        Self("eng".to_owned())
    }
    /// Spanish (`"spa"`).
    pub fn spa() -> Self {
        Self("spa".to_owned())
    }
    /// French (`"fra"`).
    pub fn fra() -> Self {
        Self("fra".to_owned())
    }
    /// Chinese / Mandarin (`"zho"`).
    pub fn zho() -> Self {
        Self("zho".to_owned())
    }
    /// Cantonese (`"yue"`).
    pub fn yue() -> Self {
        Self("yue".to_owned())
    }
    /// Japanese (`"jpn"`).
    pub fn jpn() -> Self {
        Self("jpn".to_owned())
    }
    /// German (`"deu"`).
    pub fn deu() -> Self {
        Self("deu".to_owned())
    }

    // -- Construction --

    /// Try to create a validated language code.
    ///
    /// Validation: exactly 3 ASCII alphabetic characters, lowercased.
    /// Rejects `"auto"`, `""`, `"en"`, `"english"`, etc.
    ///
    /// This is the **only** way to construct a `LanguageCode3` from
    /// untrusted input. Use well-known constants (e.g. [`Self::eng()`])
    /// for compile-time-known values.
    pub fn try_new(s: &str) -> Result<Self, InvalidLanguageCode> {
        let s = s.trim();
        if s.len() == 3 && s.bytes().all(|b| b.is_ascii_alphabetic()) {
            Ok(Self(s.to_ascii_lowercase()))
        } else {
            Err(InvalidLanguageCode(s.to_string()))
        }
    }

    // -- Conversion --

    /// This language's ISO 639-1 two-letter code, when the standard assigns
    /// one.
    ///
    /// PARTIAL, and the return type says so: most ISO 639-3 languages have no
    /// two-letter form, Mandarin (`cmn`) and Cantonese (`yue`) among them. The
    /// result is a closed sum rather than an `Option` so no caller can
    /// substitute the three-letter code for a missing two-letter one, which is
    /// how provider requests came to carry codes like `cmn` that no provider
    /// defines. See [`crate::iso639_part1`] for the single owner of this
    /// conversion and where its table comes from.
    #[must_use]
    pub fn to_iso_639_1(&self) -> crate::iso639_part1::Iso639Part1Lookup {
        crate::iso639_part1::lookup(&self.0)
    }
}

impl std::fmt::Display for LanguageCode3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for LanguageCode3 {
    type Error = InvalidLanguageCode;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::try_new(&s)
    }
}

impl TryFrom<&str> for LanguageCode3 {
    type Error = InvalidLanguageCode;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::try_new(s)
    }
}

impl From<LanguageCode3> for String {
    fn from(v: LanguageCode3) -> String {
        v.0
    }
}

impl std::ops::Deref for LanguageCode3 {
    type Target = str;
    fn deref(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for LanguageCode3 {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl PartialEq<&str> for LanguageCode3 {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

impl std::borrow::Borrow<str> for LanguageCode3 {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl Default for LanguageCode3 {
    fn default() -> Self {
        Self::eng()
    }
}

impl<'de> serde::Deserialize<'de> for LanguageCode3 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::try_new(&s).map_err(serde::de::Error::custom)
    }
}

// ---------------------------------------------------------------------------
// WorkerLanguage: worker-runtime language routing, not a domain language code
// ---------------------------------------------------------------------------

/// Worker-runtime language routed to Python workers.
///
/// This is intentionally distinct from [`LanguageCode3`]. The worker runtime
/// accepts a small sentinel vocabulary that is meaningful only at the process
/// bootstrap/dispatch boundary:
///
/// - `Resolved(code)` for a concrete ISO 639-3 language
/// - `Auto` for ASR auto-detection
/// - `PerFile` for text-NLP commands (morphotag/translate/coref) that
///   resolve language per-file from each CHAT file's `@Languages:`
///   header. Distinct from `Auto`: the Python worker must NOT try to
///   load language-specific models for a `PerFile` worker, language
///   pipelines are loaded lazily as files dispatch.
/// - `Unspecified` when the worker task does not consume a language hint
#[derive(Debug, Clone, PartialEq, Eq, Hash, utoipa::ToSchema)]
pub enum WorkerLanguage {
    /// Concrete ISO 639-3 language code.
    Resolved(LanguageCode3),
    /// ASR auto-detection sentinel.
    Auto,
    /// Per-file language resolution (no job-level language).
    PerFile,
    /// No worker language hint should be provided.
    Unspecified,
}

/// Error returned when a worker-runtime language string is invalid.
#[derive(Debug, Clone, thiserror::Error)]
#[error(
    "invalid worker language \"{0}\": expected 3 ASCII letters, \"auto\", \"per-file\", or an empty string"
)]
pub struct InvalidWorkerLanguage(pub String);

impl WorkerLanguage {
    /// Parse one untrusted worker-runtime language string.
    pub fn parse_untrusted(s: &str) -> Result<Self, InvalidWorkerLanguage> {
        let s = s.trim();
        if s.is_empty() {
            Ok(Self::Unspecified)
        } else if s.eq_ignore_ascii_case("auto") {
            Ok(Self::Auto)
        } else if s.eq_ignore_ascii_case("per-file") {
            Ok(Self::PerFile)
        } else {
            LanguageCode3::try_new(s)
                .map(Self::Resolved)
                .map_err(|_| InvalidWorkerLanguage(s.to_string()))
        }
    }

    /// Return the CLI/registry string form used by the worker runtime.
    pub fn as_worker_arg(&self) -> &str {
        match self {
            Self::Resolved(code) => code.as_ref(),
            Self::Auto => "auto",
            Self::PerFile => "per-file",
            Self::Unspecified => "",
        }
    }

    /// Return the resolved ISO language code, if present.
    pub fn as_resolved(&self) -> Option<&LanguageCode3> {
        match self {
            Self::Resolved(code) => Some(code),
            Self::Auto | Self::PerFile | Self::Unspecified => None,
        }
    }

    /// Return `true` when the worker should auto-detect the language.
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }

    /// Return `true` when the worker has no job-level language and
    /// should resolve per-file.
    pub fn is_per_file(&self) -> bool {
        matches!(self, Self::PerFile)
    }

    /// Return `true` when the worker should receive no language hint.
    pub fn is_unspecified(&self) -> bool {
        matches!(self, Self::Unspecified)
    }
}

impl std::fmt::Display for WorkerLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_worker_arg())
    }
}

impl TryFrom<String> for WorkerLanguage {
    type Error = InvalidWorkerLanguage;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse_untrusted(&value)
    }
}

impl TryFrom<&str> for WorkerLanguage {
    type Error = InvalidWorkerLanguage;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::parse_untrusted(value)
    }
}

impl From<LanguageCode3> for WorkerLanguage {
    fn from(code: LanguageCode3) -> Self {
        Self::Resolved(code)
    }
}

impl From<&LanguageCode3> for WorkerLanguage {
    fn from(code: &LanguageCode3) -> Self {
        Self::Resolved(code.clone())
    }
}

impl From<&WorkerLanguage> for WorkerLanguage {
    fn from(value: &WorkerLanguage) -> Self {
        value.clone()
    }
}

impl serde::Serialize for WorkerLanguage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_worker_arg())
    }
}

impl<'de> serde::Deserialize<'de> for WorkerLanguage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::parse_untrusted(&s).map_err(serde::de::Error::custom)
    }
}

impl schemars::JsonSchema for WorkerLanguage {
    fn schema_name() -> Cow<'static, str> {
        "WorkerLanguage".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "description": "Worker-runtime language string: ISO 639-3 code, \"auto\", or empty string."
        })
    }
}

// ---------------------------------------------------------------------------
// LanguagePair: the two languages of one code-switched recording
// ---------------------------------------------------------------------------

/// Two different languages spoken in one recording, primary first.
///
/// The order is CHAT's `@Languages` order: the primary language is the one an
/// unmarked utterance is in, and the secondary is the one a bare `@s` switches
/// to. Built only by [`LanguagePair::new`] and [`LanguagePair::parse`], both of
/// which refuse the same language twice, so no pair is one language in
/// disguise.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LanguagePair {
    primary: LanguageCode3,
    secondary: LanguageCode3,
}

/// Why text or two codes do not make a [`LanguagePair`].
#[derive(Debug, Clone, thiserror::Error)]
pub enum InvalidLanguagePair {
    /// Both positions name the same language.
    #[error("a language pair names \"{0}\" twice; name two different languages")]
    SameLanguage(LanguageCode3),
    /// The text is not two codes separated by one comma.
    #[error(
        "expected two language codes separated by a comma, primary first (for example \"eng,spa\"); got \"{0}\""
    )]
    Shape(String),
    /// One of the two codes is not a valid ISO 639-3 code.
    #[error(transparent)]
    Code(#[from] InvalidLanguageCode),
}

impl LanguagePair {
    /// The separator between the two codes wherever a pair is written as text:
    /// `--lang eng,spa`, the job record and the provenance stamp.
    pub const SEPARATOR: char = ',';

    /// Pair two languages, primary first.
    pub fn new(
        primary: LanguageCode3,
        secondary: LanguageCode3,
    ) -> Result<Self, InvalidLanguagePair> {
        match primary == secondary {
            true => Err(InvalidLanguagePair::SameLanguage(primary)),
            false => Ok(Self { primary, secondary }),
        }
    }

    /// Read a pair written as `primary,secondary`.
    pub fn parse(text: &str) -> Result<Self, InvalidLanguagePair> {
        match text.split_once(Self::SEPARATOR) {
            Some((primary, secondary)) if !secondary.contains(Self::SEPARATOR) => Self::new(
                LanguageCode3::try_new(primary)?,
                LanguageCode3::try_new(secondary)?,
            ),
            Some(_) | None => Err(InvalidLanguagePair::Shape(text.to_owned())),
        }
    }

    /// The language an unmarked utterance is in.
    pub fn primary(&self) -> &LanguageCode3 {
        &self.primary
    }

    /// The other language.
    pub fn secondary(&self) -> &LanguageCode3 {
        &self.secondary
    }

    /// Both languages, in `@Languages` order.
    pub fn declared(&self) -> [&LanguageCode3; 2] {
        [&self.primary, &self.secondary]
    }
}

impl std::fmt::Display for LanguagePair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}{}", self.primary, Self::SEPARATOR, self.secondary)
    }
}

// ---------------------------------------------------------------------------
// LanguageSpec: Auto, one resolved code, a pair, or per-file
// ---------------------------------------------------------------------------

/// Language specification from the CLI or job submission.
///
/// `Auto` means the ASR engine should detect the language. This variant must
/// be resolved to a concrete [`LanguageCode3`] before any CHAT construction
/// or NLP dispatch that requires a known language.
///
/// `Pair` declares a code-switched recording in two languages, primary first.
/// Only transcription accepts it; submission validation refuses it for every
/// other command, and for engines that recognize one language at a time.
///
/// `PerFile` means the command has no job-level language at all: each input
/// file's processing language is read from its `@Languages:` header at the
/// start of the per-file pipeline. This is distinct from `Auto`: `Auto` is an
/// ASR-engine signal asking the model to detect the spoken language;
/// `PerFile` is a routing signal for text-NLP commands (morphotag, translate,
/// coref) whose language source is the CHAT file itself, not the job
/// submission. The 2026-05-03 morphotag incident happened because these
/// commands were forced to carry a placeholder `Resolved(eng)` value that
/// then leaked into the job record, the dashboard, and the Stanza
/// pre-warming key. `PerFile` makes the absence of a job-level language a
/// first-class state in the type system.
///
/// Written as text (the wire, the job record, `--lang`) it is `auto`,
/// `per-file`, a code such as `eng`, or a pair such as `eng,spa`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LanguageSpec {
    /// Let the ASR engine auto-detect the language.
    Auto,
    /// A concrete ISO 639-3 language code.
    Resolved(LanguageCode3),
    /// Two languages of one code-switched recording, primary first.
    Pair(LanguagePair),
    /// No job-level language; resolve per-file from each CHAT file's
    /// `@Languages:` header. Used by morphotag, translate, and coref
    /// none of which take a `--lang` CLI flag.
    PerFile,
}

/// Why text is not a [`LanguageSpec`].
#[derive(Debug, Clone, thiserror::Error)]
pub enum InvalidLanguageSpec {
    /// Not `auto`, `per-file` or a pair, and not a valid code either.
    #[error(transparent)]
    Code(#[from] InvalidLanguageCode),
    /// Written as a pair, but not a valid one.
    #[error(transparent)]
    Pair(#[from] InvalidLanguagePair),
}

impl LanguageSpec {
    /// Return the one resolved language code, or `None` for `Auto`, `Pair`
    /// and `PerFile`.
    ///
    /// None of those three has a single job-level code, each for its own
    /// reason. Callers that need to tell them apart must match on the variant
    /// directly; a caller that can receive a `Pair` must not use this.
    pub fn as_resolved(&self) -> Option<&LanguageCode3> {
        match self {
            Self::Auto | Self::Pair(_) | Self::PerFile => None,
            Self::Resolved(code) => Some(code),
        }
    }

    /// Return `true` if this is `PerFile`.
    pub fn is_per_file(&self) -> bool {
        matches!(self, Self::PerFile)
    }

    /// Convert this submission/runtime language into the worker-runtime
    /// language domain.
    ///
    /// Each `LanguageSpec` variant maps to its `WorkerLanguage`
    /// counterpart. `PerFile` does **not** collapse into `Auto`: those are
    /// semantically different states (Auto = "ASR detect a single
    /// language for this whole job"; PerFile = "no job-level language at
    /// all, dispatch per-file") and the wire format must distinguish
    /// them. Otherwise the Python worker, which parses `--lang` as a
    /// plain string: would receive `"auto"` for both cases and try to
    /// load Stanza models for the literal string `"auto"`, crashing
    /// before ready.
    ///
    /// A `Pair` maps to its PRIMARY language, deliberately: workers are keyed
    /// by the language their models load for, and a code-switched transcript's
    /// language-bearing worker stages (segmentation, morphosyntax) run under
    /// its primary language. Nothing marks an utterance or word as the
    /// secondary language yet, so those stages treat the whole transcript as
    /// the primary.
    pub fn to_worker_language(&self) -> WorkerLanguage {
        match self {
            Self::Auto => WorkerLanguage::Auto,
            Self::Resolved(code) => WorkerLanguage::Resolved(code.clone()),
            Self::Pair(pair) => WorkerLanguage::Resolved(pair.primary().clone()),
            Self::PerFile => WorkerLanguage::PerFile,
        }
    }

    /// Parse text written by [`Display`](std::fmt::Display): `auto`,
    /// `per-file`, a pair such as `eng,spa`, or a code.
    fn parse(text: &str) -> Result<Self, InvalidLanguageSpec> {
        match text.trim() {
            text if text.eq_ignore_ascii_case("auto") => Ok(Self::Auto),
            text if text.eq_ignore_ascii_case("per-file") => Ok(Self::PerFile),
            text if text.contains(LanguagePair::SEPARATOR) => {
                Ok(Self::Pair(LanguagePair::parse(text)?))
            }
            text => Ok(Self::Resolved(LanguageCode3::try_new(text)?)),
        }
    }

    /// Parse from a DB string column. `"auto"` → `Auto`, `"per-file"`
    /// → `PerFile`, `"eng,spa"` → `Pair`, anything else → `Resolved`.
    ///
    /// Returns `(spec, true)` if the value was valid, `(spec, false)` if
    /// the stored value was invalid and fell back to `eng`. Callers should
    /// log the fallback so corrupt DB values are visible.
    pub fn parse_from_db(s: &str) -> (Self, bool) {
        match Self::parse(s) {
            Ok(spec) => (spec, true),
            Err(_) => (Self::Resolved(LanguageCode3::eng()), false),
        }
    }
}

impl std::fmt::Display for LanguageSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Resolved(code) => write!(f, "{code}"),
            Self::Pair(pair) => write!(f, "{pair}"),
            Self::PerFile => write!(f, "per-file"),
        }
    }
}

impl From<LanguageCode3> for LanguageSpec {
    fn from(code: LanguageCode3) -> Self {
        Self::Resolved(code)
    }
}

impl TryFrom<&str> for LanguageSpec {
    type Error = InvalidLanguageSpec;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::parse(s)
    }
}

impl Serialize for LanguageSpec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_str(self)
    }
}

/// The schema of what `LanguageSpec` writes: one string.
///
/// Written by hand because a derived schema describes the Rust enum
/// (`{"Resolved": "eng"}`), which is not what serde writes and never was.
impl utoipa::PartialSchema for LanguageSpec {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        utoipa::openapi::ObjectBuilder::new()
            .schema_type(utoipa::openapi::schema::Type::String)
            .description(Some(
                "Job language: `auto` (the ASR engine detects one language), `per-file` \
                 (each file's `@Languages:` header decides), an ISO 639-3 code such as \
                 `eng`, or a code-switched pair such as `eng,spa`, primary language first.",
            ))
            .examples(["eng", "auto", "per-file", "eng,spa"])
            .into()
    }
}

impl utoipa::ToSchema for LanguageSpec {
    fn name() -> Cow<'static, str> {
        Cow::Borrowed("LanguageSpec")
    }
}

impl<'de> Deserialize<'de> for LanguageSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Self::parse(&s).map_err(serde::de::Error::custom)
    }
}

// ---------------------------------------------------------------------------
// AsrLanguageRequest and TranscriptLanguage: speech recognition's language,
// before and after recognition
// ---------------------------------------------------------------------------

/// What speech recognition is asked to recognize: one language, detection, or
/// a declared pair.
///
/// The ASR side of [`LanguageSpec`], without `PerFile`: a recognizer always
/// has a job-level answer, so the per-file state is refused once, where this
/// is built ([`TryFrom<&LanguageSpec>`]), instead of being re-checked by every
/// stage that reads it. Written as text it is exactly what the matching
/// `LanguageSpec` writes (`eng`, `auto`, `eng,spa`), which matters: evidence
/// cache keys are built from that text, and a request must key where the same
/// request always has.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AsrLanguageRequest {
    /// One known language.
    One(LanguageCode3),
    /// The recognizer detects the language.
    Detect,
    /// A code-switched recording in two languages, primary first.
    Pair(LanguagePair),
}

/// A per-file language spec reached a speech recognizer.
#[derive(Debug, Clone, thiserror::Error)]
#[error(
    "speech recognition needs a job-level language (a code, `auto`, or a pair such as \"eng,spa\"); \
     `per-file` is only for commands that read `@Languages:` from each file"
)]
pub struct PerFileHasNoAsrLanguage;

impl TryFrom<&LanguageSpec> for AsrLanguageRequest {
    type Error = PerFileHasNoAsrLanguage;

    fn try_from(spec: &LanguageSpec) -> Result<Self, Self::Error> {
        match spec {
            LanguageSpec::Resolved(code) => Ok(Self::One(code.clone())),
            LanguageSpec::Auto => Ok(Self::Detect),
            LanguageSpec::Pair(pair) => Ok(Self::Pair(pair.clone())),
            LanguageSpec::PerFile => Err(PerFileHasNoAsrLanguage),
        }
    }
}

impl From<&AsrLanguageRequest> for LanguageSpec {
    fn from(request: &AsrLanguageRequest) -> Self {
        match request {
            AsrLanguageRequest::One(code) => Self::Resolved(code.clone()),
            AsrLanguageRequest::Detect => Self::Auto,
            AsrLanguageRequest::Pair(pair) => Self::Pair(pair.clone()),
        }
    }
}

impl AsrLanguageRequest {
    /// The language processing will run under when it is known before
    /// recognition: the one requested, or a pair's primary. `None` only for
    /// detection.
    pub fn primary_if_known(&self) -> Option<&LanguageCode3> {
        match self {
            Self::One(code) => Some(code),
            Self::Pair(pair) => Some(pair.primary()),
            Self::Detect => None,
        }
    }
}

impl std::fmt::Display for AsrLanguageRequest {
    /// Exactly the text of the matching [`LanguageSpec`], by construction.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        LanguageSpec::from(self).fmt(f)
    }
}

/// The language or languages of a finished transcript.
///
/// What recognition resolves an [`AsrLanguageRequest`] to: detection becomes
/// the language detected, and one language or a pair stays what was asked.
/// There is no undetermined state; a recognizer that could not say which
/// language it heard is refused before one of these exists. Written as text it
/// is a code (`eng`) or a pair (`eng,spa`), so a value stored before pairs
/// existed still reads.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TranscriptLanguage {
    /// One language.
    One(LanguageCode3),
    /// Two languages, primary first.
    Pair(LanguagePair),
}

impl TranscriptLanguage {
    /// The language an unmarked utterance is in, and the one the transcript's
    /// language-bearing processing (number expansion, segmentation,
    /// morphosyntax) runs under.
    pub fn primary(&self) -> &LanguageCode3 {
        match self {
            Self::One(code) => code,
            Self::Pair(pair) => pair.primary(),
        }
    }

    /// Every language, in `@Languages` order.
    pub fn declared(&self) -> Vec<&LanguageCode3> {
        match self {
            Self::One(code) => vec![code],
            Self::Pair(pair) => pair.declared().to_vec(),
        }
    }
}

impl From<&TranscriptLanguage> for LanguageSpec {
    fn from(language: &TranscriptLanguage) -> Self {
        match language {
            TranscriptLanguage::One(code) => Self::Resolved(code.clone()),
            TranscriptLanguage::Pair(pair) => Self::Pair(pair.clone()),
        }
    }
}

/// A language spec that is not a transcript's language: `auto` or `per-file`.
#[derive(Debug, Clone, thiserror::Error)]
#[error(
    "\"{0}\" is not a transcript language: expected a code such as \"eng\" or a pair such as \"eng,spa\""
)]
pub struct NotATranscriptLanguage(LanguageSpec);

impl TryFrom<LanguageSpec> for TranscriptLanguage {
    type Error = NotATranscriptLanguage;

    fn try_from(spec: LanguageSpec) -> Result<Self, Self::Error> {
        match spec {
            LanguageSpec::Resolved(code) => Ok(Self::One(code)),
            LanguageSpec::Pair(pair) => Ok(Self::Pair(pair)),
            LanguageSpec::Auto | LanguageSpec::PerFile => Err(NotATranscriptLanguage(spec)),
        }
    }
}

impl std::fmt::Display for TranscriptLanguage {
    /// Exactly the text of the matching [`LanguageSpec`], by construction.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        LanguageSpec::from(self).fmt(f)
    }
}

impl Serialize for TranscriptLanguage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for TranscriptLanguage {
    /// Read through [`LanguageSpec`]'s one parser.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Self::try_from(LanguageSpec::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

impl schemars::JsonSchema for LanguageSpec {
    fn schema_name() -> Cow<'static, str> {
        "LanguageSpec".into()
    }

    fn json_schema(g: &mut schemars::SchemaGenerator) -> schemars::Schema {
        // Reuse the string schema, "auto" or a 3-letter code.
        <String as schemars::JsonSchema>::json_schema(g)
    }
}

// ---------------------------------------------------------------------------
// DisplayPath: display-oriented file path within a job
// ---------------------------------------------------------------------------

/// Display path for a file within a job: either a bare basename
/// (`"sample.cha"`) for single-file input or a relative forward-slash path
/// (`"PWA/TYO_a1.cha"`) for directory input with subdirectories.
///
/// Backslashes are normalized to forward slashes on construction so the value
/// is platform-independent regardless of whether the CLI ran on Windows.
///
/// This type replaces the former `FileName` which incorrectly rejected path
/// separators during deserialization even though the system routinely carries
/// relative paths.
// `PartialOrd`/`Ord` are derived (added 2026-07-29) so a file identity can key a
// `BTreeMap`, which batch progress needs to aggregate per input file with
// deterministic iteration order.
#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    serde::Serialize,
    utoipa::ToSchema,
    schemars::JsonSchema,
)]
#[serde(transparent)]
pub struct DisplayPath(pub String);

/// Error returned when a display path is empty.
#[derive(Debug, Clone, thiserror::Error)]
#[error("empty display path: \"{0}\"")]
pub struct InvalidDisplayPath(pub String);

impl DisplayPath {
    /// Try to create a validated display path.
    ///
    /// Validation: non-empty after trimming. Backslashes are normalized to
    /// forward slashes for cross-platform safety.
    pub fn try_new(s: &str) -> Result<Self, InvalidDisplayPath> {
        if s.is_empty() {
            return Err(InvalidDisplayPath(s.to_owned()));
        }
        Ok(Self(normalize_backslashes(s)))
    }
}

/// Normalize Windows-style backslashes to forward slashes.
fn normalize_backslashes(s: &str) -> String {
    if s.contains('\\') {
        s.replace('\\', "/")
    } else {
        s.to_owned()
    }
}

impl std::fmt::Display for DisplayPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for DisplayPath {
    fn from(s: String) -> Self {
        if s.contains('\\') {
            Self(s.replace('\\', "/"))
        } else {
            Self(s)
        }
    }
}

impl From<&str> for DisplayPath {
    fn from(s: &str) -> Self {
        Self(normalize_backslashes(s))
    }
}

impl From<DisplayPath> for String {
    fn from(v: DisplayPath) -> String {
        v.0
    }
}

impl std::ops::Deref for DisplayPath {
    type Target = str;
    fn deref(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for DisplayPath {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl PartialEq<&str> for DisplayPath {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

impl std::borrow::Borrow<str> for DisplayPath {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl<'de> serde::Deserialize<'de> for DisplayPath {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        if s.is_empty() {
            return Err(serde::de::Error::custom("DisplayPath must not be empty"));
        }
        Ok(Self(normalize_backslashes(&s)))
    }
}

string_id!(
    /// Identifier of a server/fleet node.
    /// Empty when the node does not report an identity (older server versions).
    pub NodeId
);

numeric_id!(
    /// Number of speakers in a recording.
    pub NumSpeakers(u32) [Eq]
);

numeric_id!(
    /// A count of utterances.
    ///
    /// Distinct from every other count in the pipeline: utterances are the unit
    /// a text backend reports progress in (one `%mor` line's worth of work),
    /// which is neither files (`NumWorkers`-bounded fanout) nor words nor
    /// `%mor` items. Introduced for batch progress, where mixing an utterance
    /// count with a file count is exactly the confusion that made the old
    /// display read `453/274`.
    pub UtteranceCount(u64) [Eq]
);

numeric_id!(
    /// Duration measured in fractional seconds.
    pub DurationSeconds(f64)
);

numeric_id!(
    /// Unix timestamp as fractional seconds since epoch.
    pub UnixTimestamp(f64)
);

numeric_id!(
    /// A LENGTH of time in milliseconds. Not a position.
    ///
    /// This docstring read "Duration or audio position" until 2026-08-15, and
    /// the `or` was the defect: one type standing for two facts means nothing
    /// can notice a length passed where an offset belongs. `MediaWindow` stored
    /// its two file offsets in it for exactly that reason.
    ///
    /// A POSITION measured from the start of a recording is
    /// `batchalign::time::FileMs`; one measured from the start of an alignment
    /// window is `batchalign::time::WindowMs`. Use those for offsets, and this
    /// only for genuine durations (`max_group_ms`, `tight_buffer_ms`).
    ///
    /// Still carried by the `worker_v2` IPC types, whose `start_ms` / `end_ms`
    /// fields ARE positions and are not yet converted. Two DIFFERENT decisions
    /// hide behind that, and only one waits on anyone else: the wire type here
    /// mirrors the Python side, but the hop after it
    /// (`batchalign::chat_ops::nlp::types::FaIndexedTiming`) is entirely ours,
    /// is not Python-facing, and serializes identically to a `WindowMs`. Typing
    /// that one needs no coordination. See `batchalign::time` for what else
    /// blocks retiring the duplicate `Ms`, and for which direction a merge
    /// would have to go.
    pub DurationMs(u64) [Eq]
);

numeric_id!(
    /// Physical memory quantity in megabytes.
    ///
    /// Used for memory gate thresholds and health-response memory readings.
    pub MemoryMb(u64) [Eq]
);

// ---------------------------------------------------------------------------
// StampSafeText: text that cannot change a provenance stamp's structure
// ---------------------------------------------------------------------------

/// Text that cannot change the structure of a provenance stamp.
///
/// A stamp is one `@Comment` line, `[<name> <command> | key=value ; key=value |
/// <timestamp>]`, and the text written into its fields (engine names, a
/// language, a checkpoint) goes in byte for byte. Such text is refused when it
/// is blank, when it has surrounding whitespace (the grammar's spacing would
/// swallow it), or when it contains a character the grammar uses as structure:
/// `|`, `;`, `]` or a line break. The characters are refused on their own, not
/// only in their spaced forms: a value ending in ` |` would still form a
/// separator with the space the writer puts after it.
///
/// Three routes build one, and each keeps the invariant:
/// - [`TryFrom`] checks runtime text, and is what deserialization uses;
/// - [`Self::from_static`] checks a literal at compile time, when evaluated in
///   a `const` context (every caller writes it inside `const { ... }`);
/// - [`Self::join`] composes values that are already safe with a
///   [`StampJoiner`] that is itself safe, so the result needs no second check.
///
/// "Whitespace" means exactly [`Self::WHITESPACE`], the Unicode `White_Space`
/// characters written out, so the Python producer and the JSON Schema pattern
/// ([`Self::json_schema_pattern`]) refuse the same set rather than each
/// language's own idea of whitespace.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(try_from = "String", into = "String")]
pub struct StampSafeText(Cow<'static, str>);

/// Why text cannot be written into a provenance stamp.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InvalidStampSafeText {
    /// The text was empty or only whitespace.
    #[error("stamp text is blank")]
    Blank,
    /// The text had leading or trailing whitespace.
    #[error("stamp text {0:?} has surrounding whitespace")]
    SurroundingWhitespace(String),
    /// The text contained a character the stamp grammar uses as structure.
    #[error(
        "stamp text {0:?} contains a provenance stamp character (`|`, `;`, `]` or a line break)"
    )]
    StampStructure(String),
}

/// A separator [`StampSafeText::join`] may place between safe parts. None of
/// them is whitespace or stamp structure, which is what makes a join safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StampJoiner {
    /// Nothing: the parts are concatenated.
    Concat,
    /// `+`, between the entries of an engine list.
    Plus,
    /// `:`, between the facets of one engine identity.
    Colon,
    /// `@`, between a model id and its revision.
    At,
}

impl StampJoiner {
    /// The separator text, for a reader that splits what a join wrote.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Concat => "",
            Self::Plus => "+",
            Self::Colon => ":",
            Self::At => "@",
        }
    }
}

impl StampSafeText {
    /// The characters the stamp grammar uses as structure.
    pub const STAMP_STRUCTURE: [char; 5] = ['|', ';', ']', '\n', '\r'];

    /// What counts as whitespace: the Unicode `White_Space` characters.
    pub const WHITESPACE: [char; 25] = [
        '\t', '\n', '\u{0B}', '\u{0C}', '\r', ' ', '\u{85}', '\u{A0}', '\u{1680}', '\u{2000}',
        '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}', '\u{2006}', '\u{2007}',
        '\u{2008}', '\u{2009}', '\u{200A}', '\u{2028}', '\u{2029}', '\u{202F}', '\u{205F}',
        '\u{3000}',
    ];

    fn is_whitespace(character: char) -> bool {
        Self::WHITESPACE.contains(&character)
    }

    /// Admit a literal. In a `const` context an unsafe literal is a compile
    /// error, which is the only way this constructor may be used: it accepts
    /// ASCII only, so the check stays exact without decoding UTF-8 in `const`.
    pub const fn from_static(text: &'static str) -> Self {
        assert!(
            static_text_is_stamp_safe(text),
            "a static stamp text must be non-blank ASCII with no surrounding whitespace and \
             no `|`, `;`, `]` or line break"
        );
        Self(Cow::Borrowed(text))
    }

    /// Join `first` and `rest` with `joiner`. Total: every part is already
    /// safe, and the joiner is neither whitespace nor structure, so the result
    /// begins with `first`'s first character and ends with the last part's last.
    pub fn join<'a>(
        first: &StampSafeText,
        rest: impl IntoIterator<Item = &'a StampSafeText>,
        joiner: StampJoiner,
    ) -> Self {
        let mut joined = String::from(first.as_str());
        for part in rest {
            joined.push_str(joiner.as_str());
            joined.push_str(part.as_str());
        }
        Self(Cow::Owned(joined))
    }

    /// The text, byte for byte.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// A decimal count, as stamp text.
    ///
    /// Total, and the fourth route to a value: the decimal digits of a count
    /// are ASCII, none of them is whitespace or stamp structure, and a
    /// non-zero count has at least one of them, so the result cannot change a
    /// stamp's shape and needs no runtime check.
    ///
    /// [`NonZeroUsize`] rather than `usize` on purpose. A count field whose
    /// ABSENCE states "none" must not also be writable as `0`, or the same
    /// fact would have two spellings in the grammar and a reader would have to
    /// know they mean the same thing.
    pub fn from_count(count: NonZeroUsize) -> Self {
        Self(Cow::Owned(count.to_string()))
    }

    /// The JSON Schema `pattern` equivalent to the check, generated from
    /// [`Self::WHITESPACE`] and [`Self::STAMP_STRUCTURE`] so the two cannot
    /// disagree: a first and last character that are neither whitespace nor
    /// structure, and no structure character anywhere between.
    pub fn json_schema_pattern() -> String {
        use std::fmt::Write as _;
        let mut edge = String::new();
        for character in Self::WHITESPACE.iter().chain(Self::STAMP_STRUCTURE.iter()) {
            let _ = write!(edge, "\\u{:04X}", u32::from(*character));
        }
        let mut inner = String::new();
        for character in Self::STAMP_STRUCTURE {
            let _ = write!(inner, "\\u{:04X}", u32::from(character));
        }
        format!("^[^{edge}](?:[^{inner}]*[^{edge}])?$")
    }
}

/// The compile-time check behind [`StampSafeText::from_static`], over ASCII.
const fn static_text_is_stamp_safe(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if !byte.is_ascii() || byte == b'|' || byte == b';' || byte == b']' {
            return false;
        }
        if (index == 0 || index == bytes.len() - 1) && (byte.is_ascii_whitespace() || byte == 0x0B)
        {
            return false;
        }
        if byte == b'\n' || byte == b'\r' {
            return false;
        }
        index += 1;
    }
    true
}

impl TryFrom<String> for StampSafeText {
    type Error = InvalidStampSafeText;

    fn try_from(text: String) -> Result<Self, Self::Error> {
        if text.chars().all(Self::is_whitespace) {
            return Err(InvalidStampSafeText::Blank);
        }
        let surrounded = text.chars().next().is_some_and(Self::is_whitespace)
            || text.chars().next_back().is_some_and(Self::is_whitespace);
        if surrounded {
            return Err(InvalidStampSafeText::SurroundingWhitespace(text));
        }
        if text.contains(Self::STAMP_STRUCTURE) {
            return Err(InvalidStampSafeText::StampStructure(text));
        }
        Ok(Self(Cow::Owned(text)))
    }
}

impl TryFrom<&str> for StampSafeText {
    type Error = InvalidStampSafeText;

    fn try_from(text: &str) -> Result<Self, Self::Error> {
        Self::try_from(text.to_owned())
    }
}

impl From<StampSafeText> for String {
    fn from(text: StampSafeText) -> Self {
        text.0.into_owned()
    }
}

impl From<&LanguageCode3> for StampSafeText {
    /// Total: a language code is three ASCII letters, which can never be blank,
    /// padded or structure.
    fn from(code: &LanguageCode3) -> Self {
        Self(Cow::Owned(code.0.clone()))
    }
}

impl From<&LanguagePair> for StampSafeText {
    /// Total: two language codes and [`LanguagePair::SEPARATOR`], none of which
    /// is whitespace or stamp structure. The pair's own text (`eng,spa`).
    fn from(pair: &LanguagePair) -> Self {
        Self(Cow::Owned(pair.to_string()))
    }
}

impl AsRef<str> for StampSafeText {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for StampSafeText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl schemars::JsonSchema for StampSafeText {
    fn schema_name() -> Cow<'static, str> {
        "StampSafeText".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "pattern": StampSafeText::json_schema_pattern(),
            "description": "Text that cannot change a provenance stamp's structure: not blank, no surrounding whitespace, and none of `|`, `;`, `]` or a line break.",
        })
    }
}

/// An engine identity as a worker reported it: in a capability report, or on
/// a result the worker produced (for example `wave2vec-fa-v1` or
/// `googletrans-v1`).
///
/// Stamp-safe text ([`StampSafeText`]), because a reported name is written into
/// cache namespaces and provenance fields byte for byte and must not be able
/// to change their structure. The only constructor is [`TryFrom`], shared by
/// deserialization, so every route in admits the same values. There is no
/// infallible `From`.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(try_from = "String", into = "String")]
pub struct ReportedEngineName(StampSafeText);

impl ReportedEngineName {
    /// The name as the worker reported it, byte for byte.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// The name as stamp-safe text, for writing into a provenance field.
    pub fn as_stamp_text(&self) -> &StampSafeText {
        &self.0
    }
}

impl TryFrom<String> for ReportedEngineName {
    type Error = InvalidStampSafeText;

    fn try_from(name: String) -> Result<Self, Self::Error> {
        StampSafeText::try_from(name).map(Self)
    }
}

impl TryFrom<&str> for ReportedEngineName {
    type Error = InvalidStampSafeText;

    fn try_from(name: &str) -> Result<Self, Self::Error> {
        Self::try_from(name.to_owned())
    }
}

impl From<ReportedEngineName> for String {
    fn from(name: ReportedEngineName) -> Self {
        name.0.into()
    }
}

impl AsRef<str> for ReportedEngineName {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for ReportedEngineName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl schemars::JsonSchema for ReportedEngineName {
    fn schema_name() -> Cow<'static, str> {
        "ReportedEngineName".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "pattern": StampSafeText::json_schema_pattern(),
            "description": "An engine identity as a worker reported it, written into cache namespaces and provenance fields byte for byte: not blank, no surrounding whitespace, and none of `|`, `;`, `]` or a line break.",
        })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)] // fixture construction; the refusals are the assertions
mod stamp_safe_text_tests {
    use super::{
        InvalidStampSafeText, LanguageCode3, ReportedEngineName, StampJoiner, StampSafeText,
    };

    #[test]
    fn admits_text_byte_for_byte_and_refuses_what_could_corrupt_a_stamp() {
        for name in [
            "stanza-1.11.1",
            "wave2vec-fa-v1",
            "facebook/nllb-200-distilled-1.3B",
        ] {
            assert_eq!(
                ReportedEngineName::try_from(name).map(String::from),
                Ok(name.to_owned())
            );
        }
        assert_eq!(
            ReportedEngineName::try_from(" "),
            Err(InvalidStampSafeText::Blank)
        );
        assert_eq!(
            ReportedEngineName::try_from(""),
            Err(InvalidStampSafeText::Blank)
        );
        assert!(matches!(
            ReportedEngineName::try_from("wave2vec "),
            Err(InvalidStampSafeText::SurroundingWhitespace(_))
        ));
        for name in ["a | b", "a ; b", "a]b", "a\nb", "x |", "a|b", "a;b"] {
            assert!(
                matches!(
                    ReportedEngineName::try_from(name),
                    Err(InvalidStampSafeText::StampStructure(_))
                ),
                "{name:?}"
            );
        }
        // Deserialization is the same constructor.
        assert!(serde_json::from_str::<ReportedEngineName>("\"\"").is_err());
        assert_eq!(
            serde_json::from_str::<ReportedEngineName>("\"stanza-1.11.1\"")
                .ok()
                .map(String::from),
            Some("stanza-1.11.1".to_owned())
        );
        assert!(serde_json::from_str::<ReportedEngineName>("\"a|b\"").is_err());
    }

    /// The cases the Python producer's check is held to as well
    /// (`batchalign/tests/test_stamp_safe_text_conformance.py` reads the same
    /// file), so the two languages agree case by case, including whitespace
    /// that only one of them would otherwise count.
    #[test]
    fn the_shared_conformance_cases_are_decided_as_recorded() {
        #[derive(serde::Deserialize)]
        struct Case {
            text: String,
            admitted: bool,
        }
        let cases: Vec<Case> = serde_json::from_str(include_str!(
            "../../../tests/fixtures/stamp_safe_text_cases.json"
        ))
        .expect("the conformance cases parse");
        assert!(!cases.is_empty());
        for case in cases {
            assert_eq!(
                StampSafeText::try_from(case.text.as_str()).is_ok(),
                case.admitted,
                "{:?}",
                case.text
            );
        }
    }

    #[test]
    fn joins_safe_parts_without_a_second_check() {
        let stanza = const { StampSafeText::from_static("stanza-") };
        let version = StampSafeText::try_from("1.14.0").expect("safe");
        let identity = StampSafeText::join(&stanza, [&version], StampJoiner::Concat);
        let lang = StampSafeText::from(&LanguageCode3::eng());
        let facets = StampSafeText::join(&identity, [&lang], StampJoiner::Colon);
        assert_eq!(facets.as_str(), "stanza-1.14.0:eng");
        assert_eq!(
            StampSafeText::join(&facets, [&facets], StampJoiner::Plus).as_str(),
            "stanza-1.14.0:eng+stanza-1.14.0:eng"
        );
        // Every join result is itself admissible text.
        assert!(StampSafeText::try_from(facets.as_str()).is_ok());
    }

    #[test]
    fn the_schema_pattern_names_every_refused_character() {
        let pattern = StampSafeText::json_schema_pattern();
        for character in StampSafeText::WHITESPACE
            .iter()
            .chain(StampSafeText::STAMP_STRUCTURE.iter())
        {
            assert!(
                pattern.contains(&format!("\\u{:04X}", u32::from(*character))),
                "{character:?}"
            );
        }
    }
}

validated_string_id!(
    /// Correlation ID for tracing a job across log entries (non-empty).
    ///
    /// Usually the same as `JobId` but may differ for retried or cloned jobs.
    pub CorrelationId
);

numeric_id!(
    /// Number of parallel file-processing workers for a job.
    ///
    /// Computed by `compute_job_workers()` based on available memory and CPU.
    /// Used in dispatch runtime structs to bound concurrency via a semaphore.
    pub NumWorkers(usize) [Eq]
);

validated_string_id!(
    /// A Rev.AI server-side job identifier returned after audio submission (non-empty).
    ///
    /// Obtained during preflight batch upload and passed to polling calls so
    /// individual file tasks can retrieve results without re-uploading audio.
    pub RevAiJobId
);

/// Engine category that supports backend overrides.
///
/// Currently only ASR and FA have multiple engine backends.
/// Other inference tasks (morphosyntax, utseg, translate, coref)
/// always use their single built-in engine.
/// MIME-like content discriminator for file results.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ContentType {
    /// CHAT format output.
    #[default]
    Chat,
    /// Tabular CSV output (e.g. opensmile features).
    Csv,
    /// Plain text output (e.g. AVQI voice quality reports).
    Text,
    /// JSON document output (e.g. diarization speaker-turns files).
    Json,
}

impl std::fmt::Display for ContentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chat => write!(f, "chat"),
            Self::Csv => write!(f, "csv"),
            Self::Text => write!(f, "text"),
            Self::Json => write!(f, "json"),
        }
    }
}

// ---------------------------------------------------------------------------
// Cancellation provenance
// ---------------------------------------------------------------------------

/// Where a job-cancellation request originated. No `Default` impl
/// every caller must explicitly state what kind of actor it is, so a
/// future "anonymous cancel" path cannot silently slip through as
/// `Api`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum CancelSource {
    /// Interactive TUI cancel (user pressed `c` then `y`).
    Tui,
    /// Non-TUI CLI cancel (e.g., a `--cancel` flag invocation).
    Cli,
    /// Web dashboard cancel button.
    Dashboard,
    /// Staging orchestrator forwarded a cancel.
    Staging,
    /// Direct REST API cancel with no caller hint (raw curl, scripts).
    Api,
    /// SIGTERM-driven graceful shutdown cancelled in-flight work.
    Signal,
}

/// Returned when a string cannot be parsed into a `CancelSource`.
#[derive(Debug, Clone, thiserror::Error)]
#[error("invalid cancel source \"{0}\": expected one of tui, cli, dashboard, staging, api, signal")]
pub struct InvalidCancelSource(pub String);

impl std::fmt::Display for CancelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tui => write!(f, "tui"),
            Self::Cli => write!(f, "cli"),
            Self::Dashboard => write!(f, "dashboard"),
            Self::Staging => write!(f, "staging"),
            Self::Api => write!(f, "api"),
            Self::Signal => write!(f, "signal"),
        }
    }
}

impl std::str::FromStr for CancelSource {
    type Err = InvalidCancelSource;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "tui" => Ok(Self::Tui),
            "cli" => Ok(Self::Cli),
            "dashboard" => Ok(Self::Dashboard),
            "staging" => Ok(Self::Staging),
            "api" => Ok(Self::Api),
            "signal" => Ok(Self::Signal),
            other => Err(InvalidCancelSource(other.to_string())),
        }
    }
}

string_id!(
    /// Hostname or remote IP of the caller who issued a cancel.
    ///
    /// Empty when the source did not report identity (older clients,
    /// localhost defaults). Persisted in `cancellations.host` and
    /// projected onto `jobs.last_cancelled_host`.
    pub CallerHost
);

string_id!(
    /// Free-form reason text attached to a cancel request.
    ///
    /// Examples: `"user-pressed-cancel"`, `"ctrl-c-shutdown"`,
    /// `"too-slow-aborting"`. Empty is allowed.
    pub CancelReason
);

impl CancelReason {
    /// The reason recorded when a `cancel_all` runs as part of graceful
    /// server shutdown (i.e. not a user gesture).
    pub fn server_cancel_all() -> Self {
        Self::from("server-cancel-all")
    }
}

numeric_id!(
    /// Process identifier of the caller that issued a cancel.
    ///
    /// Unix PIDs fit in u32 on every platform we support. Persisted in
    /// `cancellations.pid` for forensics across multi-machine setups
    /// (helps distinguish an operator-laptop cancel from a fleet-internal one).
    pub CallerPid(u32) [Eq]
);

/// Server health status.
///
/// Deliberately has NO `Default`, and that is the whole point rather than an
/// oversight. While it derived one, `#[serde(default)]` on
/// `HealthResponse::status` was writable, and it was written: every field of
/// that struct defaulted, so any JSON object deserialized into a healthy
/// response and the probe's `status != Ok` guard could never be false. On
/// 2026-08-27 an unrelated local service holding port 8000 was accepted as a
/// Batchalign server and had jobs dispatched to it.
///
/// Removing the `#[serde(default)]` fixed that VALUE; removing this `Default`
/// is what stops it being re-expressible, because re-adding the attribute now
/// fails to compile instead of silently restoring the behaviour. A `Default`
/// on a domain type is an affordance for fabrication: audit the `Default`, not
/// only the call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    /// Server is accepting work.
    Ok,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ok => write!(f, "ok"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- LanguageCode3 validation ----

    #[test]
    fn language_code3_valid() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(LanguageCode3::try_new("eng")?.0, "eng");
        assert_eq!(LanguageCode3::try_new("SPA")?.0, "spa");
        assert_eq!(LanguageCode3::try_new("Zho")?.0, "zho");
        Ok(())
    }

    #[test]
    fn language_code3_rejects_auto() {
        assert!(LanguageCode3::try_new("auto").is_err());
    }

    #[test]
    fn language_code3_rejects_empty() {
        assert!(LanguageCode3::try_new("").is_err());
    }

    #[test]
    fn language_code3_rejects_two_letter() {
        assert!(LanguageCode3::try_new("en").is_err());
    }

    #[test]
    fn language_code3_rejects_four_letter() {
        assert!(LanguageCode3::try_new("engl").is_err());
    }

    #[test]
    fn language_code3_rejects_digits() {
        assert!(LanguageCode3::try_new("e1g").is_err());
    }

    #[test]
    fn language_code3_serde_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let code = LanguageCode3::eng();
        let json = serde_json::to_string(&code)?;
        assert_eq!(json, "\"eng\"");
        let back: LanguageCode3 = serde_json::from_str(&json)?;
        assert_eq!(back, code);
        Ok(())
    }

    #[test]
    fn language_code3_deserialize_rejects_auto() {
        let result: Result<LanguageCode3, _> = serde_json::from_str("\"auto\"");
        assert!(result.is_err());
    }

    #[test]
    fn language_code3_try_from_str_rejects_auto() {
        assert!(LanguageCode3::try_from("auto").is_err());
    }

    #[test]
    fn language_code3_try_from_string_rejects_auto() {
        assert!(LanguageCode3::try_from("auto".to_string()).is_err());
    }

    // ---- WorkerLanguage ----

    #[test]
    fn worker_language_parses_resolved_auto_and_unspecified()
    -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(
            WorkerLanguage::parse_untrusted("eng")?,
            WorkerLanguage::Resolved(LanguageCode3::eng())
        );
        assert_eq!(
            WorkerLanguage::parse_untrusted("AUTO")?,
            WorkerLanguage::Auto
        );
        assert_eq!(
            WorkerLanguage::parse_untrusted("")?,
            WorkerLanguage::Unspecified
        );
        Ok(())
    }

    #[test]
    fn worker_language_rejects_invalid_values() {
        assert!(WorkerLanguage::parse_untrusted("english").is_err());
        assert!(WorkerLanguage::parse_untrusted("12").is_err());
    }

    #[test]
    fn worker_language_serde_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let auto = WorkerLanguage::Auto;
        assert_eq!(serde_json::to_string(&auto)?, "\"auto\"");
        assert_eq!(
            serde_json::from_str::<WorkerLanguage>("\"\"")?,
            WorkerLanguage::Unspecified
        );
        assert_eq!(
            serde_json::from_str::<WorkerLanguage>("\"yue\"")?,
            WorkerLanguage::Resolved(LanguageCode3::yue())
        );
        Ok(())
    }

    // ---- LanguageSpec ----

    #[test]
    fn language_spec_deserializes_auto() -> Result<(), Box<dyn std::error::Error>> {
        let spec: LanguageSpec = serde_json::from_str("\"auto\"")?;
        assert_eq!(spec, LanguageSpec::Auto);
        Ok(())
    }

    #[test]
    fn language_spec_deserializes_auto_case_insensitive() -> Result<(), Box<dyn std::error::Error>>
    {
        let spec: LanguageSpec = serde_json::from_str("\"AUTO\"")?;
        assert_eq!(spec, LanguageSpec::Auto);
        Ok(())
    }

    #[test]
    fn language_spec_deserializes_resolved() -> Result<(), Box<dyn std::error::Error>> {
        let spec: LanguageSpec = serde_json::from_str("\"eng\"")?;
        assert_eq!(spec, LanguageSpec::Resolved(LanguageCode3::eng()));
        Ok(())
    }

    #[test]
    fn language_spec_serializes_auto() -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string(&LanguageSpec::Auto)?;
        assert_eq!(json, "\"auto\"");
        Ok(())
    }

    #[test]
    fn language_spec_serializes_resolved() -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string(&LanguageSpec::Resolved(LanguageCode3::spa()))?;
        assert_eq!(json, "\"spa\"");
        Ok(())
    }

    #[test]
    fn language_spec_roundtrip_auto() -> Result<(), Box<dyn std::error::Error>> {
        let spec = LanguageSpec::Auto;
        let json = serde_json::to_string(&spec)?;
        let back: LanguageSpec = serde_json::from_str(&json)?;
        assert_eq!(spec, back);
        Ok(())
    }

    #[test]
    fn language_spec_roundtrip_resolved() -> Result<(), Box<dyn std::error::Error>> {
        let spec = LanguageSpec::Resolved(LanguageCode3::fra());
        let json = serde_json::to_string(&spec)?;
        let back: LanguageSpec = serde_json::from_str(&json)?;
        assert_eq!(spec, back);
        Ok(())
    }

    #[test]
    fn language_spec_rejects_invalid_code() {
        let result: Result<LanguageSpec, _> = serde_json::from_str("\"xx\"");
        assert!(result.is_err());
    }

    #[test]
    fn language_spec_try_from_str_auto() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(LanguageSpec::try_from("auto")?, LanguageSpec::Auto);
        Ok(())
    }

    #[test]
    fn language_spec_try_from_str_resolved() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(
            LanguageSpec::try_from("eng")?,
            LanguageSpec::Resolved(LanguageCode3::eng())
        );
        Ok(())
    }

    #[test]
    fn language_spec_display() {
        assert_eq!(LanguageSpec::Auto.to_string(), "auto");
        assert_eq!(
            LanguageSpec::Resolved(LanguageCode3::eng()).to_string(),
            "eng"
        );
    }

    #[test]
    fn language_spec_parse_from_db_valid() {
        let (spec, valid) = LanguageSpec::parse_from_db("auto");
        assert_eq!(spec, LanguageSpec::Auto);
        assert!(valid);

        let (spec, valid) = LanguageSpec::parse_from_db("eng");
        assert_eq!(spec, LanguageSpec::Resolved(LanguageCode3::eng()));
        assert!(valid);
    }

    #[test]
    fn language_spec_parse_from_db_invalid_falls_back() {
        let (spec, valid) = LanguageSpec::parse_from_db("not-a-lang");
        assert_eq!(spec, LanguageSpec::Resolved(LanguageCode3::eng()));
        assert!(!valid, "invalid DB value should report fallback");
    }

    #[test]
    fn language_spec_maps_to_worker_language() {
        assert_eq!(
            LanguageSpec::Auto.to_worker_language(),
            WorkerLanguage::Auto
        );
        assert_eq!(
            LanguageSpec::Resolved(LanguageCode3::eng()).to_worker_language(),
            WorkerLanguage::Resolved(LanguageCode3::eng())
        );
        assert_eq!(
            LanguageSpec::PerFile.to_worker_language(),
            WorkerLanguage::PerFile,
            "PerFile must NOT collapse into Auto; those are semantically \
             distinct states and the wire format must distinguish them",
        );
    }

    // ---- LanguageSpec::PerFile ----
    //
    // `PerFile` exists for commands whose processing language is not a
    // job-level concept but is resolved per-file from each CHAT file's
    // `@Languages:` header (morphotag, translate, coref). It is NOT the same
    // as `Auto` (which is an ASR-engine signal: "let the model detect the
    // spoken language"). The two must serialize differently so the wire
    // format and the dashboard distinguish "no job-level lang" from "ASR
    // auto-detect".

    #[test]
    fn language_spec_per_file_displays_as_per_file() {
        assert_eq!(LanguageSpec::PerFile.to_string(), "per-file");
    }

    #[test]
    fn language_spec_per_file_serializes_as_per_file_string()
    -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string(&LanguageSpec::PerFile)?;
        assert_eq!(json, "\"per-file\"");
        Ok(())
    }

    #[test]
    fn language_spec_per_file_round_trips_json() -> Result<(), Box<dyn std::error::Error>> {
        let spec = LanguageSpec::PerFile;
        let json = serde_json::to_string(&spec)?;
        let back: LanguageSpec = serde_json::from_str(&json)?;
        assert_eq!(back, spec);
        Ok(())
    }

    #[test]
    fn language_spec_parse_from_db_per_file() {
        let (spec, valid) = LanguageSpec::parse_from_db("per-file");
        assert_eq!(spec, LanguageSpec::PerFile);
        assert!(valid);
    }

    #[test]
    fn language_spec_per_file_try_from_str() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(LanguageSpec::try_from("per-file")?, LanguageSpec::PerFile);
        Ok(())
    }

    #[test]
    fn language_spec_per_file_distinct_from_auto() {
        // ASR auto-detect ≠ per-file lang resolution. Code that branches on
        // these states must not collapse them into `Option<LanguageCode3>`.
        assert_ne!(LanguageSpec::PerFile, LanguageSpec::Auto);
    }

    #[test]
    fn language_spec_per_file_as_resolved_is_none() {
        assert_eq!(LanguageSpec::PerFile.as_resolved(), None);
    }

    // ---- LanguagePair, AsrLanguageRequest, TranscriptLanguage ----

    /// A pair is two different valid codes separated by one comma, primary
    /// first; anything else is refused with the reason.
    #[test]
    fn a_language_pair_is_two_different_codes() {
        let pair = LanguagePair::parse("eng,spa").expect("a pair");
        assert_eq!(pair.primary(), &LanguageCode3::eng());
        assert_eq!(pair.secondary(), &LanguageCode3::spa());
        assert_eq!(pair.to_string(), "eng,spa");
        assert_eq!(
            LanguagePair::parse("SPA, eng")
                .expect("case and spacing")
                .to_string(),
            "spa,eng"
        );

        assert!(matches!(
            LanguagePair::parse("eng,eng"),
            Err(InvalidLanguagePair::SameLanguage(_))
        ));
        assert!(matches!(
            LanguagePair::parse("eng,spa,fra"),
            Err(InvalidLanguagePair::Shape(_))
        ));
        assert!(matches!(
            LanguagePair::parse("eng,sp"),
            Err(InvalidLanguagePair::Code(_))
        ));
    }

    /// A pair round-trips through every text form a job record is kept in:
    /// display, JSON and the database column.
    #[test]
    fn a_language_spec_pair_round_trips_through_its_text_forms()
    -> Result<(), Box<dyn std::error::Error>> {
        let spec = LanguageSpec::try_from("eng,spa")?;
        assert!(matches!(spec, LanguageSpec::Pair(_)));
        assert_eq!(spec.to_string(), "eng,spa");
        let json = serde_json::to_string(&spec)?;
        assert_eq!(json, "\"eng,spa\"");
        assert_eq!(serde_json::from_str::<LanguageSpec>(&json)?, spec);
        assert_eq!(LanguageSpec::parse_from_db("eng,spa"), (spec.clone(), true));
        assert!(serde_json::from_str::<LanguageSpec>("\"eng,eng\"").is_err());
        Ok(())
    }

    /// A pair has no single code, and its workers load for its primary.
    #[test]
    fn a_language_spec_pair_has_no_single_code_and_routes_workers_by_primary() {
        let spec = LanguageSpec::try_from("spa,eng").expect("a pair");
        assert_eq!(spec.as_resolved(), None);
        assert_eq!(
            spec.to_worker_language(),
            WorkerLanguage::Resolved(LanguageCode3::spa())
        );
    }

    /// A transcript language stored before pairs existed, a bare code, still
    /// reads; a pair reads as a pair; `auto` is never a transcript language.
    #[test]
    fn a_transcript_language_reads_old_codes_and_pairs() -> Result<(), Box<dyn std::error::Error>> {
        let old: TranscriptLanguage = serde_json::from_str("\"eng\"")?;
        assert_eq!(old, TranscriptLanguage::One(LanguageCode3::eng()));
        assert_eq!(old.declared(), vec![&LanguageCode3::eng()]);

        let pair: TranscriptLanguage = serde_json::from_str("\"spa,eng\"")?;
        assert_eq!(pair.primary(), &LanguageCode3::spa());
        assert_eq!(
            pair.declared(),
            vec![&LanguageCode3::spa(), &LanguageCode3::eng()]
        );
        assert_eq!(serde_json::to_string(&pair)?, "\"spa,eng\"");

        assert!(serde_json::from_str::<TranscriptLanguage>("\"auto\"").is_err());
        Ok(())
    }

    // ---- DisplayPath ----

    #[test]
    fn display_path_accepts_bare_basename() -> Result<(), Box<dyn std::error::Error>> {
        let p = DisplayPath::try_new("sample.cha")?;
        assert_eq!(&*p, "sample.cha");
        Ok(())
    }

    #[test]
    fn display_path_accepts_relative_path() -> Result<(), Box<dyn std::error::Error>> {
        let p = DisplayPath::try_new("PWA/TYO_a1.cha")?;
        assert_eq!(&*p, "PWA/TYO_a1.cha");
        Ok(())
    }

    #[test]
    fn display_path_rejects_empty() {
        assert!(DisplayPath::try_new("").is_err());
    }

    #[test]
    fn display_path_normalizes_backslash_in_from() {
        let p = DisplayPath::from("PWA\\TYO.cha");
        assert_eq!(&*p, "PWA/TYO.cha");
    }

    #[test]
    fn display_path_serde_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let p = DisplayPath::try_new("sub/dir/file.cha")?;
        let json = serde_json::to_string(&p)?;
        let back: DisplayPath = serde_json::from_str(&json)?;
        assert_eq!(p, back);
        Ok(())
    }

    #[test]
    fn display_path_deserialize_rejects_empty() {
        let result: Result<DisplayPath, _> = serde_json::from_str("\"\"");
        assert!(result.is_err());
    }

    #[test]
    fn display_path_deserialize_normalizes_backslash() -> Result<(), Box<dyn std::error::Error>> {
        let p: DisplayPath = serde_json::from_str("\"PWA\\\\TYO.cha\"")?;
        assert_eq!(&*p, "PWA/TYO.cha");
        Ok(())
    }
}
