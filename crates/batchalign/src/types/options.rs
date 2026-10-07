//! Typed per-command options for job submission and processing.
//!
//! Replaces the stringly-typed `HashMap<String, serde_json::Value>` that
//! previously carried command options through the system. Each command has
//! a dedicated struct with compile-time checked fields and serde defaults
//! matching the CLI defaults.
//!
//! # Wire format
//!
//! [`CommandOptions`] serializes as an internally-tagged JSON object:
//!
//! ```json
//! {
//!   "command": "morphotag",
//!   "retokenize": true,
//!   "skipmultilang": false,
//!   "merge_abbrev": false,
//!   "override_media_cache": false,
//!   "engine_overrides": {}
//! }
//! ```
//!
//! The `command` tag doubles as the command name for routing.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub use super::params::{MergeAbbrevPolicy, UtsegFallbackPolicy, WorTierPolicy};

mod diarize;
pub use diarize::{DiarizeOutputMode, SpeakerTrackMapping, SpeakerTrackMappingRefusal};

// ---------------------------------------------------------------------------
// Default helpers
// ---------------------------------------------------------------------------

/// Default forced-alignment engine for serialized command options.
///
/// Wave2Vec, because it reports a word's END as well as its start.
///
/// Whisper FA reports token onsets only, so selecting it without deriving an
/// end yields zero-duration words. See the FA reference chapter for the
/// per-engine comparison.
///
/// DERIVED, not restated: the CLI flag's default comes from the same constant,
/// so the value a user gets by omitting `--fa-engine` and the value a
/// deserialized `AlignOptions` gets by omitting the field cannot disagree.
fn default_fa_engine() -> FaEngineName {
    FaEngineName::DEFAULT
}

/// Default ASR engine for serialized command options.
fn default_asr_engine() -> AsrEngineName {
    AsrEngineName::DEFAULT
}

/// Default translation engine for serialized command options.
///
/// Google preserves the fleet's historical behavior. Operators on
/// hosts where Google Translate is unreachable (mainland-China sites
/// behind the Great Firewall) pass `--translate-engine seamless`
/// explicitly; there is no per-host config-file default by design,
/// because hidden host-specific behavior is the failure mode this
/// project rules out (see the no-config-junk discussion in
/// `book/src/batchalign/user-guide/commands/translate.md`).
fn default_translate_engine() -> TranslateEngineName {
    TranslateEngineName::Google
}

/// Default Whisper batch size.
fn default_batch_size() -> i32 {
    8
}

/// Default `%wor` policy for commands that enable the tier by default.
fn default_wor_tier_include() -> WorTierPolicy {
    WorTierPolicy::Include
}

/// Default openSMILE feature set.
fn default_feature_set() -> String {
    "eGeMAPSv02".to_string()
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn is_ca_policy_honor(value: &CaMorphotagPolicy) -> bool {
    *value == CaMorphotagPolicy::Honor
}

/// Shared behavior for all engine backend selectors.
///
/// Implement this on each engine enum (`AsrEngineName`, `FaEngineName`,
/// `UtrEngine`) so generic code can work across engine categories without
/// knowing which specific enum it holds.
pub use super::engines::*;

// ---------------------------------------------------------------------------
// CommonOptions
// ---------------------------------------------------------------------------

/// Options shared by all processing commands.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CommonOptions {
    /// Bypass the media analysis cache.
    #[serde(default)]
    pub override_media_cache: bool,

    /// Require reusable media evidence and refuse inference on a cache miss.
    #[serde(default, skip_serializing_if = "is_false")]
    pub require_media_cache: bool,

    /// Engine overrides selected for this job (e.g., ASR=tencent, FA=cantonese_fa).
    /// Typed struct with `Option<AsrEngineName>` and `Option<FaEngineName>`.
    #[serde(default, skip_serializing_if = "EngineOverrides::is_empty")]
    pub engine_overrides: EngineOverrides,

    /// Multi-word token (MWT) lexicon: maps a surface form (e.g. "gonna")
    /// to its expansion tokens (e.g. `["going", "to"]`).
    /// Loaded from `--lexicon` CSV on the CLI side.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub mwt: BTreeMap<String, Vec<String>>,

    /// Optional directory for pipeline debug artifact dumps. Always carries
    /// an absolute path: the CLI canonicalizes the user-supplied
    /// `--debug-dir` value via `canonicalize_debug_dir` in
    /// `batchalign-cli::args::options` before constructing this struct, so
    /// the server never has to resolve a relative path against its own
    /// (opaque) working directory. `serde` serializes `PathBuf` as a JSON
    /// string, so the wire format is unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub debug_dir: Option<PathBuf>,

    /// Per-task cache override specifications (comma-separated task names).
    /// When non-empty, only the listed tasks skip cache; others use cache normally.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub override_media_cache_tasks: Vec<String>,
    // `batch_window` used to live here (removed 2026-07-30). It was parsed from
    // `--batch-window`, stored, serialized and read by NOTHING in the execution
    // path: its help text promised "smaller windows show progress sooner", which
    // was untrue at every setting. The windowing it named belonged to the
    // cross-file pooled batching that per-file fanout replaced. There is no
    // `deny_unknown_fields` here, so a client still sending the field is
    // ignored rather than rejected.
}

// ---------------------------------------------------------------------------
// Per-command option structs
// ---------------------------------------------------------------------------

/// How `+<` overlap utterances are handled during UTR.
///
/// Selects the alignment strategy for utterance timing recovery. The trait-based
/// architecture in `batchalign-chat-ops` allows plugging in different strategies
/// at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UtrOverlapStrategy {
    /// Automatically select: two-pass when `+<` utterances are present,
    /// global otherwise.
    #[default]
    Auto,
    /// Single global DP pass. `+<` utterances get no special treatment.
    Global,
    /// Two-pass overlap-aware strategy. Pass 1 excludes `+<` utterances,
    /// pass 2 recovers their timing from the predecessor's audio window.
    TwoPass,
}

/// Utterance-timing-recovery policy persisted with an `align` job.
///
/// The fields remain flattened on the wire for compatibility, but Rust code
/// cannot detach the selected engine from the strategy and two-pass tuning it
/// governs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AlignUtrOptions {
    /// UTR engine selection; `None` disables the UTR pass.
    #[serde(
        default,
        rename = "utr_engine",
        skip_serializing_if = "Option::is_none"
    )]
    pub engine: Option<UtrEngine>,

    /// How `+<` overlap utterances are handled during UTR.
    #[serde(default, rename = "utr_overlap_strategy")]
    pub overlap_strategy: UtrOverlapStrategy,

    /// Two-pass UTR configuration (CA markers, density threshold, buffers).
    #[serde(default, rename = "utr_two_pass")]
    pub two_pass: crate::chat_ops::fa::TwoPassConfig,
}

/// Boundary-projection policy persisted with an `align` job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlignBoundaryOptions {
    /// How fresh FA evidence interacts with boundaries from an existing `%wor` run.
    #[serde(default)]
    pub existing_wor_boundaries: crate::chat_ops::fa::ExistingWorBoundaryPolicy,

    /// How adjacent utterance end overlap is projected after alignment.
    #[serde(default = "crate::chat_ops::fa::default_end_overlap_policy_serde")]
    pub end_overlap_policy: crate::chat_ops::fa::EndOverlapPolicy,

    /// Whether projection may change a main bullet the input carried. A job
    /// stored before this field existed reads as the default, `derive`, which
    /// is exactly what such a job ran.
    #[serde(default = "crate::chat_ops::fa::default_main_bullet_policy_serde")]
    pub main_bullets: crate::chat_ops::fa::MainBulletPolicy,
}

/// `EndOverlapPolicy` deliberately carries no `Default` impl (see
/// `DEFAULT_END_OVERLAP_POLICY`'s doc), so this struct's default cannot be
/// derived field-wise; it reads the same one constant every other caller of
/// "the default" reads, rather than restating the choice here.
impl Default for AlignBoundaryOptions {
    fn default() -> Self {
        Self {
            existing_wor_boundaries: Default::default(),
            end_overlap_policy: crate::chat_ops::fa::DEFAULT_END_OVERLAP_POLICY,
            main_bullets: crate::chat_ops::fa::DEFAULT_MAIN_BULLET_POLICY,
        }
    }
}

/// Options for the `transcribe` and `transcribe_s` commands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscribeOptions {
    /// Infer speaker count rather than imposing the job's numeric default.
    #[serde(default)]
    pub auto_speakers: bool,
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// ASR engine selector (`rev`, `whisper`, `whisperx`, `whisper_oai`, or
    /// plugin name).
    #[serde(default = "default_asr_engine")]
    pub asr_engine: AsrEngineName,

    /// Enable speaker diarization.
    #[serde(default)]
    pub diarize: bool,

    /// Generate `%wor` tier with word-level timing bullets.
    #[serde(default)]
    pub wor: WorTierPolicy,

    /// Merge abbreviated forms during processing.
    #[serde(default)]
    pub merge_abbrev: MergeAbbrevPolicy,

    /// Operator opt-in to the legacy Stanza constituency-parser
    /// fallback for utseg when no language-specific TalkBank BERT
    /// model is configured for the job's language. Driven by the
    /// `--utseg-fallback-stanza` CLI flag; default refuses
    /// substitution.
    #[serde(default)]
    pub utseg_fallback: UtsegFallbackPolicy,

    /// Whisper batch size.
    #[serde(default = "default_batch_size")]
    pub batch_size: i32,
}

impl TranscribeOptions {
    /// Return the effective ASR engine after applying any shared `asr` override.
    pub fn effective_asr_engine(&self) -> AsrEngineName {
        self.common
            .engine_overrides
            .asr
            .clone()
            .unwrap_or_else(|| self.asr_engine.clone())
    }
}

/// How `morphotag` treats a transcript carrying `@Options: CA`.
///
/// The transcript header remains authoritative by default. `Analyze` is an
/// explicit operator choice for corpus-reconstruction work where automatic
/// morphology is intentionally added without deleting or rewriting the CA
/// declaration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum CaMorphotagPolicy {
    /// Honor `@Options: CA` and pass the transcript through unchanged.
    #[default]
    Honor,
    /// Run morphotag while preserving the transcript's `@Options: CA` header.
    Analyze,
}

/// Options for the `morphotag` command.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MorphotagOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// Re-tokenize words before morphosyntactic analysis.
    #[serde(default)]
    pub retokenize: bool,

    /// Skip files with multiple `@Languages`.
    #[serde(default)]
    pub skipmultilang: bool,

    /// Merge abbreviated forms during processing.
    #[serde(default)]
    pub merge_abbrev: MergeAbbrevPolicy,

    /// Opt-out: if `true`, suppress the default L2 dispatch and emit
    /// `L2|xxx` placeholders for `@s` (code-switched) words. Default
    /// `false`: L2 dispatch is on.
    #[serde(default)]
    pub no_l2_morphotag: bool,

    /// Opt-out: if `true`, suppress the default `$POS`-hint
    /// post-pass over the morphotagged `%mor` tier. By default,
    /// every main-tier word carrying a `$POS` suffix has its CLAN
    /// tag mapped to a UD UPOS
    /// (`talkbank_model::...::clan_to_ud_upos`); on disagreement
    /// with Stanza's UPOS the `%mor` POS is overridden. Lemma and
    /// features from Stanza are preserved. Default `false`, POS
    /// hints are respected; set via `--no-pos-hints` to opt out.
    #[serde(default)]
    pub no_pos_hints: bool,

    /// Policy for transcripts carrying `@Options: CA`.
    #[serde(default, skip_serializing_if = "is_ca_policy_honor")]
    pub ca_policy: CaMorphotagPolicy,

    /// Legacy review-tier request retained for wire compatibility. No value
    /// emits `%xalign` or `%xrev`; structured evidence retains the decisions.
    #[serde(default)]
    pub review_level: crate::chat_ops::fa::ReviewLevel,
}

/// Options for the `translate` command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslateOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// Translation engine selector. Default Google preserves the
    /// fleet's historical behavior; pass `--translate-engine seamless`
    /// to opt into the local Meta SeamlessM4T model.
    #[serde(default = "default_translate_engine")]
    pub translate_engine: TranslateEngineName,

    /// Requested translation target. Older stored jobs retain English.
    #[serde(default = "crate::api::LanguageCode3::eng")]
    pub target: crate::api::LanguageCode3,

    /// Merge abbreviated forms during processing.
    #[serde(default)]
    pub merge_abbrev: MergeAbbrevPolicy,
}

impl Default for TranslateOptions {
    fn default() -> Self {
        Self {
            common: CommonOptions::default(),
            translate_engine: default_translate_engine(),
            target: crate::api::LanguageCode3::eng(),
            merge_abbrev: MergeAbbrevPolicy::default(),
        }
    }
}

impl TranslateOptions {
    /// Return the effective translation engine after applying any
    /// shared `translate` override.
    ///
    /// Precedence mirrors `AlignOptions::effective_fa_engine`:
    /// `--engine-overrides '{"translate":"..."}'` beats the dedicated
    /// `--translate-engine` flag.
    pub fn effective_translate_engine(&self) -> TranslateEngineName {
        self.common
            .engine_overrides
            .translate
            .clone()
            .unwrap_or_else(|| self.translate_engine.clone())
    }
}

/// Options for the `coref` command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorefOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// Merge abbreviated forms during processing.
    #[serde(default)]
    pub merge_abbrev: MergeAbbrevPolicy,
}

/// Options for the `utseg` command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UtsegOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// Merge abbreviated forms during processing.
    #[serde(default)]
    pub merge_abbrev: MergeAbbrevPolicy,

    /// Operator opt-in to the legacy Stanza constituency-parser
    /// fallback for utseg when no language-specific TalkBank BERT
    /// model is configured for the job's language. Driven by the
    /// `--utseg-fallback-stanza` CLI flag; default refuses
    /// substitution.
    #[serde(default)]
    pub utseg_fallback: UtsegFallbackPolicy,
}

/// Options for the `benchmark` command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// ASR engine selector.
    #[serde(default = "default_asr_engine")]
    pub asr_engine: AsrEngineName,

    /// Generate `%wor` tier with word-level timing bullets. Defaults to
    /// `Include` because the benchmark pipeline always runs forced
    /// alignment (it's the comparison anchor against the gold), so the
    /// word timings already exist, omitting them from the serialized
    /// output throws away alignment data the comparison just computed.
    /// Mirrors `AlignOptions::wor`, not `TranscribeOptions::wor`.
    #[serde(default = "default_wor_tier_include")]
    pub wor: WorTierPolicy,

    /// Merge abbreviated forms during processing.
    #[serde(default)]
    pub merge_abbrev: MergeAbbrevPolicy,
}

impl BenchmarkOptions {
    /// Return the effective ASR engine after applying any shared `asr` override.
    pub fn effective_asr_engine(&self) -> AsrEngineName {
        self.common
            .engine_overrides
            .asr
            .clone()
            .unwrap_or_else(|| self.asr_engine.clone())
    }
}

/// Options for the `opensmile` command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpensmileOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// Feature set to extract (e.g. `"eGeMAPSv02"`, `"ComParE_2016"`).
    #[serde(default = "default_feature_set")]
    pub feature_set: String,
}

/// How many speakers a diarizer is asked to find.
///
/// Exactly N, and N is at least two. A count of ONE has no representation,
/// because asking a diarizer to separate speakers while asserting the recording
/// has one is a contradiction rather than a request: it is not "detect the
/// count" (that is the absence of this value) and it is not a separation
/// problem. Before this type, a count of 1 was representable and was obeyed:
/// diarized runs forced every utterance onto a single track, which is why
/// `--diarization enabled` transcripts came back with one `PAR0`.
///
/// Policy: "a diarization speaker count is either
/// exactly N (N at least 2) or automatic, for every diarizer."
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "u32", into = "u32")]
pub struct DiarizationSpeakerCount(u32);

impl DiarizationSpeakerCount {
    /// The smallest count that asks a diarizer a question it can answer.
    pub const MINIMUM: u32 = 2;

    /// The count, as the diarizer's own parameter.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl TryFrom<u32> for DiarizationSpeakerCount {
    type Error = InvalidDiarizationSpeakerCount;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value < Self::MINIMUM {
            return Err(InvalidDiarizationSpeakerCount { value });
        }
        Ok(Self(value))
    }
}

impl From<DiarizationSpeakerCount> for u32 {
    fn from(value: DiarizationSpeakerCount) -> Self {
        value.0
    }
}

impl std::fmt::Display for DiarizationSpeakerCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A speaker count that cannot be asked of a diarizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "a diarization speaker count must be at least {minimum}, and {value} was given. \
     Omit the count to have the diarizer detect it, which is the recommended mode.",
    minimum = DiarizationSpeakerCount::MINIMUM
)]
pub struct InvalidDiarizationSpeakerCount {
    /// The count that was asked for.
    pub value: u32,
}

/// Options for the `diarize` command (standalone speaker diarization).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiarizeOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// Speaker backend selected for this standalone diarization run.
    ///
    /// This is concrete rather than `Option`: by the time a job is persisted,
    /// the choice between local and paid inference must already be made.
    /// Older persisted jobs deserialize to the historical local backend.
    #[serde(default = "default_standalone_speaker_engine")]
    pub speaker_engine: SpeakerEngineName,

    /// Expected speaker count when the caller knows it; `None` lets the
    /// diarizer auto-detect (the normal mode, and pyannote's strength).
    ///
    /// Typed so that a count of one has no representation: see
    /// [`DiarizationSpeakerCount`].
    #[serde(default)]
    pub expected_speakers: Option<DiarizationSpeakerCount>,

    /// Source/output mode; legacy persisted jobs retain media-to-turns behavior.
    #[serde(default)]
    pub output_mode: DiarizeOutputMode,
}

fn default_standalone_speaker_engine() -> SpeakerEngineName {
    SpeakerEngineName::Pyannote
}

/// Options for the `speaker-identify` command.
///
/// # Why there is no default threshold, and no `Default` on this struct
///
/// The threshold is a decision about how much acoustic agreement counts as
/// the same person. It depends on the recording, the microphone, how much
/// enrollment audio there is, and what the caller intends to do with a wrong
/// answer, and nothing in this crate knows any of that. A default would make
/// every run produce confident verdicts under a number nobody chose, with no
/// way for a reader to tell it had been chosen by accident.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerIdentifyOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// Every enrolled span, already validated.
    ///
    /// An `EnrollmentSet` rather than a list of raw arguments: reading a
    /// persisted job goes through the same construction a command line takes,
    /// so a job row cannot reconstitute a set the CLI would have refused
    /// (empty, duplicate-labelled, or overlapping).
    pub enrollments: crate::chat_ops::speaker_identity::EnrollmentSet,

    /// The similarity at or above which an enrolled voice is called a match.
    pub threshold: crate::chat_ops::speaker_identity::MatchThreshold,

    /// Speaker tiers whose utterances are scored; empty means every tier.
    #[serde(default)]
    pub tiers: Vec<String>,

    /// Seed and count of the permutation test behind the track contrasts.
    ///
    /// A job row written before the track verdict existed carries none, and
    /// re-running such a row is what `documented_plan` is for: it is the
    /// plan the CLI flags name in their help text, so a re-run and a fresh
    /// run under default flags produce the same file.
    #[serde(default = "crate::chat_ops::speaker_identity::documented_permutation_plan")]
    pub permutation: crate::chat_ops::speaker_identity::PermutationPlan,
}

/// Options for the `compare` command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompareOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// Merge abbreviated forms during processing.
    #[serde(default)]
    pub merge_abbrev: MergeAbbrevPolicy,
}

/// Options for the `avqi` command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AvqiOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,
}

// ---------------------------------------------------------------------------
// CommandOptions tagged enum
// ---------------------------------------------------------------------------

/// Options for native export. The encoding is required, never guessed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvertOptions {
    /// Shared scheduling and job options.
    #[serde(flatten)]
    pub common: CommonOptions,
    /// Selected preservation-oriented encoding.
    pub format: crate::media::export::AudioExportFormat,
}

/// Typed per-command options with an internally-tagged `command` discriminator.
///
/// Each variant holds a struct with all options for that command. The `command`
/// tag in the JSON matches the job submission command name, enabling
/// deserialization from the wire format without a separate `command` field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "lowercase")]
pub enum CommandOptions {
    /// `align`: forced alignment.
    Align(AlignOptions),
    /// `transcribe`: ASR transcription.
    Transcribe(TranscribeOptions),
    /// `transcribe_s`: ASR with speaker diarization.
    #[serde(rename = "transcribe_s")]
    TranscribeS(TranscribeOptions),
    /// `translate`: translation.
    Translate(TranslateOptions),
    /// `morphotag`: morphosyntactic analysis.
    Morphotag(MorphotagOptions),
    /// `coref`: coreference resolution.
    Coref(CorefOptions),
    /// `utseg`: utterance segmentation.
    Utseg(UtsegOptions),
    /// `benchmark`: ASR benchmarking.
    Benchmark(BenchmarkOptions),
    /// `opensmile`: audio feature extraction.
    Opensmile(OpensmileOptions),
    /// `compare`: transcript comparison against gold standard.
    Compare(CompareOptions),
    /// `avqi`: voice quality index.
    Avqi(AvqiOptions),
    /// `diarize`: standalone speaker diarization to turns JSON.
    Diarize(DiarizeOptions),
    /// `speaker_identify`: score utterances against enrolled voices.
    #[serde(rename = "speaker_identify")]
    SpeakerIdentify(SpeakerIdentifyOptions),
    /// `convert`: standalone native audio export.
    Convert(ConvertOptions),
}

impl CommandOptions {
    /// Get the common options shared by all commands.
    pub fn common(&self) -> &CommonOptions {
        match self {
            Self::Align(o) => &o.common,
            Self::Transcribe(o) | Self::TranscribeS(o) => &o.common,
            Self::Translate(o) => &o.common,
            Self::Morphotag(o) => &o.common,
            Self::Coref(o) => &o.common,
            Self::Utseg(o) => &o.common,
            Self::Benchmark(o) => &o.common,
            Self::Opensmile(o) => &o.common,
            Self::Compare(o) => &o.common,
            Self::Avqi(o) => &o.common,
            Self::Diarize(o) => &o.common,
            Self::SpeakerIdentify(o) => &o.common,
            Self::Convert(o) => &o.common,
        }
    }

    /// Get a mutable reference to the common options.
    pub fn common_mut(&mut self) -> &mut CommonOptions {
        match self {
            Self::Align(o) => &mut o.common,
            Self::Transcribe(o) | Self::TranscribeS(o) => &mut o.common,
            Self::Translate(o) => &mut o.common,
            Self::Morphotag(o) => &mut o.common,
            Self::Coref(o) => &mut o.common,
            Self::Utseg(o) => &mut o.common,
            Self::Benchmark(o) => &mut o.common,
            Self::Opensmile(o) => &mut o.common,
            Self::Compare(o) => &mut o.common,
            Self::Avqi(o) => &mut o.common,
            Self::Diarize(o) => &mut o.common,
            Self::SpeakerIdentify(o) => &mut o.common,
            Self::Convert(o) => &mut o.common,
        }
    }

    /// Abbreviation-merging policy for this command.
    ///
    /// Commands without this option use [`MergeAbbrevPolicy::Keep`].
    pub fn merge_abbrev_policy(&self) -> MergeAbbrevPolicy {
        match self {
            Self::Align(o) => o.merge_abbrev,
            Self::Transcribe(o) | Self::TranscribeS(o) => o.merge_abbrev,
            Self::Translate(o) => o.merge_abbrev,
            Self::Morphotag(o) => o.merge_abbrev,
            Self::Coref(o) => o.merge_abbrev,
            Self::Utseg(o) => o.merge_abbrev,
            Self::Benchmark(o) => o.merge_abbrev,
            Self::Compare(o) => o.merge_abbrev,
            // Never merges: this command does not write CHAT at all, so
            // there is no document for the policy to apply to.
            Self::Opensmile(_)
            | Self::Avqi(_)
            | Self::Diarize(_)
            | Self::SpeakerIdentify(_)
            | Self::Convert(_) => MergeAbbrevPolicy::Keep,
        }
    }

    /// Whether abbreviation merging is enabled for this command.
    pub fn merge_abbrev(&self) -> bool {
        self.merge_abbrev_policy().should_merge()
    }

    /// Operator opt-in to the Stanza utseg fallback for this command.
    ///
    /// Commands without an utseg surface return
    /// [`UtsegFallbackPolicy::Refuse`].
    pub fn utseg_fallback_policy(&self) -> UtsegFallbackPolicy {
        match self {
            Self::Transcribe(o) | Self::TranscribeS(o) => o.utseg_fallback,
            Self::Utseg(o) => o.utseg_fallback,
            _ => UtsegFallbackPolicy::Refuse,
        }
    }

    /// The translate options, when this is a translate command.
    ///
    /// `None` for every other command rather than a defaulted engine: a
    /// translate dispatcher handed another command's options must refuse,
    /// not translate with whatever the default happens to be.
    pub fn as_translate(&self) -> Option<&TranslateOptions> {
        match self {
            Self::Translate(options) => Some(options),
            _ => None,
        }
    }

    /// Get the command name as a string (matches the serde tag value).
    pub fn command_name(&self) -> &'static str {
        self.command().as_str()
    }

    /// Command identity derived from the options variant, not a second token.
    pub fn command(&self) -> crate::ReleasedCommand {
        use crate::ReleasedCommand as Command;
        match self {
            Self::Align(_) => Command::Align,
            Self::Transcribe(_) => Command::Transcribe,
            Self::TranscribeS(_) => Command::TranscribeS,
            Self::Translate(_) => Command::Translate,
            Self::Morphotag(_) => Command::Morphotag,
            Self::Coref(_) => Command::Coref,
            Self::Utseg(_) => Command::Utseg,
            Self::Benchmark(_) => Command::Benchmark,
            Self::Opensmile(_) => Command::Opensmile,
            Self::Compare(_) => Command::Compare,
            Self::Avqi(_) => Command::Avqi,
            Self::Diarize(_) => Command::Diarize,
            Self::SpeakerIdentify(_) => Command::SpeakerIdentify,
            Self::Convert(_) => Command::Convert,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

mod align;
mod media_root;
pub use align::AlignOptions;
pub use media_root::{AbsoluteMediaRoot, MediaRootDeclaration, MediaRootRefusal};
#[cfg(test)]
mod tests;
