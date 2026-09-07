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
        }
    }
}

/// Options for the `align` command.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignOptions {
    /// Shared options.
    #[serde(flatten)]
    pub common: CommonOptions,

    /// FA engine selector (`wav2vec_fa`, `whisper_fa`, or plugin name).
    #[serde(default = "default_fa_engine")]
    pub fa_engine: FaEngineName,

    /// Utterance-timing-recovery selection and tuning.
    #[serde(flatten)]
    pub utr: AlignUtrOptions,

    /// Include pause durations in forced alignment.
    #[serde(default)]
    pub pauses: bool,

    /// Existing- and cross-utterance boundary projection policy.
    #[serde(flatten)]
    pub boundaries: AlignBoundaryOptions,

    /// Generate `%wor` tier with word-level timing bullets.
    #[serde(default = "default_wor_tier_include")]
    pub wor: WorTierPolicy,

    /// Merge abbreviated forms during processing.
    #[serde(default)]
    pub merge_abbrev: MergeAbbrevPolicy,

    /// Apply post-FA bullet repair to fix timing violations.
    ///
    /// Uses boundary averaging, gap filling, and selective removal instead
    /// of CLAN FIXBULLETS. Experimental.
    #[serde(default)]
    pub bullet_repair: bool,

    /// Review tier verbosity (none / low-confidence / all).
    #[serde(default)]
    pub review_level: crate::chat_ops::fa::ReviewLevel,

    /// Directory to search for media files (audio/video).
    /// When set, the aligner looks here in addition to the standard
    /// media resolution paths (alongside .cha file, server media roots).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_dir: Option<String>,
}

impl Default for AlignOptions {
    fn default() -> Self {
        Self {
            common: CommonOptions::default(),
            fa_engine: default_fa_engine(),
            utr: AlignUtrOptions::default(),
            pauses: false,
            boundaries: AlignBoundaryOptions::default(),
            wor: default_wor_tier_include(),
            merge_abbrev: MergeAbbrevPolicy::default(),
            bullet_repair: false,
            review_level: crate::chat_ops::fa::ReviewLevel::default(),
            media_dir: None,
        }
    }
}

impl AlignOptions {
    /// Return the effective FA engine after applying any shared `fa` override.
    pub fn effective_fa_engine(&self) -> FaEngineName {
        self.common.engine_overrides.fa.unwrap_or(self.fa_engine)
    }

    /// Return the effective UTR engine after applying any shared `utr`
    /// override, or `None` when no UTR pass was asked for.
    ///
    /// The `None` here means "no UTR pass", which is a real answer. An override
    /// cannot conjure a pass that was not requested: `--engine-overrides
    /// '{"utr":...}'` says WHICH engine, not WHETHER, exactly as the `fa` and
    /// `asr` overrides do.
    pub fn effective_utr_engine(&self) -> Option<UtrEngine> {
        let requested = self.utr.engine.as_ref()?;
        Some(
            self.common
                .engine_overrides
                .utr
                .clone()
                .unwrap_or_else(|| requested.clone()),
        )
    }
}

/// Options for the `transcribe` and `transcribe_s` commands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscribeOptions {
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

    /// Merge abbreviated forms during processing.
    #[serde(default)]
    pub merge_abbrev: MergeAbbrevPolicy,
}

impl Default for TranslateOptions {
    fn default() -> Self {
        Self {
            common: CommonOptions::default(),
            translate_engine: default_translate_engine(),
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
    #[serde(default)]
    pub expected_speakers: Option<crate::api::NumSpeakers>,
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
            Self::Opensmile(_) | Self::Avqi(_) | Self::Diarize(_) | Self::SpeakerIdentify(_) => {
                MergeAbbrevPolicy::Keep
            }
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

    /// Get the command name as a string (matches the serde tag value).
    pub fn command_name(&self) -> &'static str {
        match self {
            Self::Align(_) => "align",
            Self::Transcribe(_) => "transcribe",
            Self::TranscribeS(_) => "transcribe_s",
            Self::Translate(_) => "translate",
            Self::Morphotag(_) => "morphotag",
            Self::Coref(_) => "coref",
            Self::Utseg(_) => "utseg",
            Self::Benchmark(_) => "benchmark",
            Self::Opensmile(_) => "opensmile",
            Self::Compare(_) => "compare",
            Self::Avqi(_) => "avqi",
            Self::Diarize(_) => "diarize",
            Self::SpeakerIdentify(_) => "speaker_identify",
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morphotag_roundtrip() {
        let opts = CommandOptions::Morphotag(MorphotagOptions {
            common: CommonOptions::default(),
            retokenize: true,

            ..Default::default()
        });
        let json = serde_json::to_string(&opts).unwrap();
        assert!(
            !json.contains("ca_policy"),
            "the default policy stays wire-compatible with older jobs: {json}"
        );
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn morphotag_ca_analyze_policy_roundtrips_explicitly() {
        let opts = CommandOptions::Morphotag(MorphotagOptions {
            ca_policy: CaMorphotagPolicy::Analyze,
            ..Default::default()
        });
        let json = serde_json::to_string(&opts).unwrap();
        assert!(json.contains("\"ca_policy\":\"analyze\""), "json: {json}");
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn align_roundtrip() {
        let opts = CommandOptions::Align(AlignOptions {
            common: CommonOptions::default(),
            fa_engine: FaEngineName::Whisper,
            utr: AlignUtrOptions {
                engine: Some(UtrEngine::RevAi),
                ..Default::default()
            },
            pauses: true,
            boundaries: AlignBoundaryOptions {
                existing_wor_boundaries: Default::default(),
                end_overlap_policy: crate::chat_ops::fa::DEFAULT_END_OVERLAP_POLICY,
            },
            wor: true.into(),
            merge_abbrev: false.into(),
            bullet_repair: false,
            review_level: Default::default(),
            media_dir: None,
        });
        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn transcribe_roundtrip() {
        let opts = CommandOptions::Transcribe(TranscribeOptions {
            common: CommonOptions::default(),
            asr_engine: AsrEngineName::WhisperX,
            diarize: true,
            wor: false.into(),
            merge_abbrev: false.into(),
            utseg_fallback: false.into(),
            batch_size: 16,
        });
        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn transcribe_s_roundtrip() {
        let opts = CommandOptions::TranscribeS(TranscribeOptions {
            common: CommonOptions::default(),
            asr_engine: AsrEngineName::RevAi,
            diarize: true,
            wor: false.into(),
            merge_abbrev: false.into(),
            utseg_fallback: false.into(),
            batch_size: 8,
        });
        let json = serde_json::to_string(&opts).unwrap();
        assert!(json.contains("\"command\":\"transcribe_s\""));
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn command_name_matches_tag() {
        let cases: Vec<(CommandOptions, &str)> = vec![
            (
                CommandOptions::Align(AlignOptions {
                    common: CommonOptions::default(),
                    fa_engine: FaEngineName::Wave2Vec,
                    utr: Default::default(),
                    pauses: false,
                    boundaries: Default::default(),
                    wor: true.into(),
                    merge_abbrev: false.into(),
                    bullet_repair: false,
                    review_level: Default::default(),
                    media_dir: None,
                }),
                "align",
            ),
            (
                CommandOptions::Morphotag(MorphotagOptions {
                    common: CommonOptions::default(),

                    ..Default::default()
                }),
                "morphotag",
            ),
            (
                CommandOptions::Opensmile(OpensmileOptions {
                    common: CommonOptions::default(),
                    feature_set: "eGeMAPSv02".into(),
                }),
                "opensmile",
            ),
            (
                CommandOptions::Compare(CompareOptions {
                    common: CommonOptions::default(),
                    merge_abbrev: false.into(),
                }),
                "compare",
            ),
            (
                CommandOptions::Avqi(AvqiOptions {
                    common: CommonOptions::default(),
                }),
                "avqi",
            ),
        ];

        for (opts, expected_name) in cases {
            assert_eq!(opts.command_name(), expected_name);
            let json = serde_json::to_string(&opts).unwrap();
            assert!(
                json.contains(&format!("\"command\":\"{expected_name}\"")),
                "JSON should contain command tag '{expected_name}': {json}"
            );
        }
    }

    #[test]
    fn common_accessor() {
        let opts = CommandOptions::Morphotag(MorphotagOptions {
            common: CommonOptions {
                override_media_cache: true,
                engine_overrides: EngineOverrides::default(),
                mwt: BTreeMap::new(),
                ..Default::default()
            },
            retokenize: true,

            ..Default::default()
        });
        assert!(opts.common().override_media_cache);
    }

    #[test]
    fn required_media_cache_wire_field_is_explicit_and_backward_compatible() {
        let required = CommonOptions {
            require_media_cache: true,
            ..Default::default()
        };
        let json = serde_json::to_string(&required).expect("serialize required policy");
        assert!(json.contains(r#""require_media_cache":true"#));

        let legacy: CommonOptions = serde_json::from_str(r#"{"override_media_cache":false}"#)
            .expect("deserialize legacy common options");
        assert!(!legacy.require_media_cache);
    }

    #[test]
    fn engine_overrides_roundtrip() {
        let overrides = EngineOverrides {
            asr: Some(AsrEngineName::HkTencent),
            fa: Some(FaEngineName::Wav2vecCanto),
            translate: None,
            ..Default::default()
        };

        let opts = CommandOptions::Align(AlignOptions {
            common: CommonOptions {
                override_media_cache: false,
                engine_overrides: overrides.clone(),
                mwt: BTreeMap::new(),
                ..Default::default()
            },
            fa_engine: FaEngineName::Wav2vecCanto,
            ..AlignOptions::default()
        });

        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(back.common().engine_overrides, overrides);
    }

    /// `CommonOptions::debug_dir` is typed as `Option<PathBuf>`, but the wire
    /// format must stay a JSON string so existing clients (and the dashboard
    /// schema) keep working. Lock that invariant here.
    #[test]
    fn debug_dir_serializes_as_json_string() {
        let opts = CommonOptions {
            debug_dir: Some(PathBuf::from("/tmp/some/abs/path")),
            ..Default::default()
        };
        let json = serde_json::to_string(&opts).unwrap();
        assert!(
            json.contains(r#""debug_dir":"/tmp/some/abs/path""#),
            "expected debug_dir as JSON string, got: {json}"
        );
        let back: CommonOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(back.debug_dir, Some(PathBuf::from("/tmp/some/abs/path")));
    }

    #[test]
    fn transcribe_asr_override_effective_engine_prefers_override() {
        let overrides = EngineOverrides {
            asr: Some(AsrEngineName::HkTencent),
            fa: None,
            translate: None,
            ..Default::default()
        };
        let opts = TranscribeOptions {
            common: CommonOptions {
                engine_overrides: overrides,
                ..CommonOptions::default()
            },
            asr_engine: AsrEngineName::RevAi,
            diarize: false,
            wor: false.into(),
            merge_abbrev: false.into(),
            utseg_fallback: false.into(),
            batch_size: 8,
        };

        assert_eq!(opts.effective_asr_engine(), AsrEngineName::HkTencent);
    }

    #[test]
    fn benchmark_asr_override_effective_engine_prefers_override() {
        let overrides = EngineOverrides {
            asr: Some(AsrEngineName::HkAliyun),
            fa: None,
            translate: None,
            ..Default::default()
        };
        let opts = BenchmarkOptions {
            common: CommonOptions {
                engine_overrides: overrides,
                ..CommonOptions::default()
            },
            asr_engine: AsrEngineName::RevAi,
            wor: true.into(),
            merge_abbrev: false.into(),
        };

        assert_eq!(opts.effective_asr_engine(), AsrEngineName::HkAliyun);
    }

    #[test]
    fn minimal_json_deserializes_with_defaults() {
        let json = r#"{"command": "morphotag"}"#;
        let opts: CommandOptions = serde_json::from_str(json).unwrap();
        assert_eq!(opts.command_name(), "morphotag");
        if let CommandOptions::Morphotag(m) = &opts {
            assert!(!m.retokenize);
            assert!(!m.skipmultilang);
            assert!(!m.merge_abbrev.should_merge());
        } else {
            panic!("expected Morphotag");
        }
    }

    #[test]
    fn avqi_roundtrip() {
        let opts = CommandOptions::Avqi(AvqiOptions {
            common: CommonOptions::default(),
        });
        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn legacy_diarize_job_defaults_to_historical_local_backend() {
        let options: CommandOptions = serde_json::from_str(r#"{"command":"diarize"}"#)
            .expect("legacy standalone diarize job remains readable");
        let CommandOptions::Diarize(options) = options else {
            panic!("expected diarize options");
        };
        assert_eq!(options.speaker_engine, SpeakerEngineName::Pyannote);
        assert_eq!(options.expected_speakers, None);
    }

    #[test]
    fn standalone_paid_speaker_backend_roundtrips_by_wire_name() {
        let options = CommandOptions::Diarize(DiarizeOptions {
            common: CommonOptions::default(),
            speaker_engine: SpeakerEngineName::PyannoteAi,
            expected_speakers: Some(crate::api::NumSpeakers(3)),
        });
        let json = serde_json::to_string(&options).expect("serialize diarize options");
        assert!(json.contains(r#""speaker_engine":"pyannote_ai""#));
        let back = serde_json::from_str(&json).expect("deserialize diarize options");
        assert_eq!(options, back);
    }

    #[test]
    fn compare_roundtrip() {
        let opts = CommandOptions::Compare(CompareOptions {
            common: CommonOptions::default(),
            merge_abbrev: true.into(),
        });
        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn translate_roundtrip() {
        let opts = CommandOptions::Translate(TranslateOptions {
            common: CommonOptions::default(),
            translate_engine: TranslateEngineName::Google,
            merge_abbrev: true.into(),
        });
        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn translate_options_default_engine_is_google() {
        // Default preserves the fleet's historical behavior. Operators
        // who want Seamless pass `--translate-engine seamless`
        // explicitly; there is no per-host config-file default.
        let opts = TranslateOptions::default();
        assert_eq!(opts.translate_engine, TranslateEngineName::Google);
    }

    #[test]
    fn translate_options_effective_engine_prefers_explicit_field() {
        let opts = TranslateOptions {
            common: CommonOptions::default(),
            translate_engine: TranslateEngineName::Seamless,
            merge_abbrev: false.into(),
        };
        assert_eq!(
            opts.effective_translate_engine(),
            TranslateEngineName::Seamless,
        );
    }

    #[test]
    fn translate_options_effective_engine_override_wins() {
        // Shared --engine-overrides '{"translate":"seamless"}' beats
        // the dedicated --translate-engine flag, mirroring
        // effective_fa_engine / effective_asr_engine.
        let mut common = CommonOptions::default();
        common.engine_overrides.translate = Some(TranslateEngineName::Seamless);
        let opts = TranslateOptions {
            common,
            translate_engine: TranslateEngineName::Google,
            merge_abbrev: false.into(),
        };
        assert_eq!(
            opts.effective_translate_engine(),
            TranslateEngineName::Seamless,
        );
    }

    #[test]
    fn translate_options_serializes_seamless_engine() {
        let opts = TranslateOptions {
            common: CommonOptions::default(),
            translate_engine: TranslateEngineName::Seamless,
            merge_abbrev: false.into(),
        };
        let json = serde_json::to_string(&opts).unwrap();
        assert!(
            json.contains("\"translate_engine\":\"seamless\""),
            "expected serialized form to contain the Seamless wire token, got: {json}"
        );
    }

    #[test]
    fn coref_roundtrip() {
        let opts = CommandOptions::Coref(CorefOptions {
            common: CommonOptions::default(),
            merge_abbrev: false.into(),
        });
        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn utseg_roundtrip() {
        let opts = CommandOptions::Utseg(UtsegOptions {
            common: CommonOptions::default(),
            merge_abbrev: true.into(),
            utseg_fallback: false.into(),
        });
        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn benchmark_roundtrip() {
        let opts = CommandOptions::Benchmark(BenchmarkOptions {
            common: CommonOptions::default(),
            asr_engine: AsrEngineName::WhisperOai,
            wor: true.into(),
            merge_abbrev: false.into(),
        });
        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn opensmile_roundtrip() {
        let opts = CommandOptions::Opensmile(OpensmileOptions {
            common: CommonOptions::default(),
            feature_set: "ComParE_2016".into(),
        });
        let json = serde_json::to_string(&opts).unwrap();
        let back: CommandOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(opts, back);
    }

    #[test]
    fn utr_engine_roundtrip_preserves_wire_names() {
        let rev_json = serde_json::to_string(&UtrEngine::RevAi).unwrap();
        let whisper_json = serde_json::to_string(&UtrEngine::Whisper).unwrap();
        let custom_json = serde_json::to_string(&UtrEngine::HkTencent).unwrap();

        assert_eq!(rev_json, "\"rev_utr\"");
        assert_eq!(whisper_json, "\"whisper_utr\"");
        assert_eq!(custom_json, "\"tencent_utr\"");

        assert_eq!(
            serde_json::from_str::<UtrEngine>(&rev_json).unwrap(),
            UtrEngine::RevAi
        );
        assert_eq!(
            serde_json::from_str::<UtrEngine>(&whisper_json).unwrap(),
            UtrEngine::Whisper
        );
        assert_eq!(
            serde_json::from_str::<UtrEngine>(&custom_json).unwrap(),
            UtrEngine::HkTencent
        );
    }

    #[test]
    fn fa_engine_roundtrip_preserves_wire_names() {
        let wav2vec_json = serde_json::to_string(&FaEngineName::Wave2Vec).unwrap();
        let whisper_json = serde_json::to_string(&FaEngineName::Whisper).unwrap();
        let custom_json = serde_json::to_string(&FaEngineName::Wav2vecCanto).unwrap();

        assert_eq!(wav2vec_json, "\"wav2vec_fa\"");
        assert_eq!(whisper_json, "\"whisper_fa\"");
        assert_eq!(custom_json, "\"cantonese_fa\"");

        assert_eq!(
            serde_json::from_str::<FaEngineName>(&wav2vec_json).unwrap(),
            FaEngineName::Wave2Vec
        );
        assert_eq!(
            serde_json::from_str::<FaEngineName>(&whisper_json).unwrap(),
            FaEngineName::Whisper
        );
        assert_eq!(
            serde_json::from_str::<FaEngineName>(&custom_json).unwrap(),
            FaEngineName::Wav2vecCanto
        );
    }

    #[test]
    fn asr_engine_roundtrip_preserves_wire_names() {
        let rev_json = serde_json::to_string(&AsrEngineName::RevAi).unwrap();
        let whisperx_json = serde_json::to_string(&AsrEngineName::WhisperX).unwrap();
        let custom_json = serde_json::to_string(&AsrEngineName::HkTencent).unwrap();

        assert_eq!(rev_json, "\"rev\"");
        assert_eq!(whisperx_json, "\"whisperx\"");
        assert_eq!(custom_json, "\"tencent\"");

        assert_eq!(
            serde_json::from_str::<AsrEngineName>(&rev_json).unwrap(),
            AsrEngineName::RevAi
        );
        assert_eq!(
            serde_json::from_str::<AsrEngineName>(&whisperx_json).unwrap(),
            AsrEngineName::WhisperX
        );
        assert_eq!(
            serde_json::from_str::<AsrEngineName>(&custom_json).unwrap(),
            AsrEngineName::HkTencent
        );
    }
}
