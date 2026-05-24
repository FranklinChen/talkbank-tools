//! Public JSON contracts shared by editor integrations and the LSP backend.
//!
//! These types describe the stable JSON payloads that cross the extension/server
//! boundary. They are intentionally transport-shaped, derive `JsonSchema`, and
//! are reused directly by the backend decoder so schema generation and runtime
//! decoding stay in sync.

use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use talkbank_clan::database::{DatabaseFilter, Gender};
use talkbank_clan::service_types::AnalysisCommandName;

/// Demographic filter payload used by `talkbank/analyze`.
#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema)]
#[serde(default)]
pub struct AnalysisDatabaseFilterPayload {
    /// Optional language code such as `eng`.
    pub language: Option<String>,
    /// Optional demographic group such as `TD`.
    pub group: Option<String>,
    /// Optional gender selector.
    pub gender: Option<Gender>,
    /// Optional lower age bound in months.
    pub age_from_months: Option<u32>,
    /// Optional upper age bound in months.
    pub age_to_months: Option<u32>,
    /// Optional speaker-code subset.
    pub speaker_codes: Vec<String>,
}

impl From<AnalysisDatabaseFilterPayload> for DatabaseFilter {
    fn from(value: AnalysisDatabaseFilterPayload) -> Self {
        Self {
            language: value.language,
            group: value.group,
            gender: value.gender,
            age_from_months: value.age_from_months,
            age_to_months: value.age_to_months,
            speaker_codes: value.speaker_codes,
        }
    }
}

/// Command-specific option payload for `talkbank/analyze`.
#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema)]
#[serde(default)]
pub struct AnalysisOptionsPayload {
    /// Whether to run `freq` against `%mor`.
    pub mor: bool,
    /// Whether `mlu` should count words instead of morphemes.
    pub words: bool,
    /// Whether `wdsize` should read from the main tier.
    #[serde(rename = "mainTier")]
    pub main_tier: bool,
    /// Result limit for commands such as `maxwd`.
    pub limit: Option<usize>,
    /// Keyword list for `kwal` and `keymap`.
    pub keywords: Vec<String>,
    /// Search expressions for `combo`.
    pub search: Vec<String>,
    /// Maximum code depth for `codes`.
    #[serde(rename = "maxDepth")]
    pub max_depth: Option<usize>,
    /// Tier selector used by several analysis commands.
    pub tier: Option<String>,
    /// Frequency threshold for `corelex`.
    pub threshold: Option<u64>,
    /// Shared max-utterance limit used by several analyzers.
    #[serde(rename = "maxUtterances")]
    pub max_utterances: Option<usize>,
    /// Optional normative database path.
    #[serde(rename = "databasePath")]
    pub database_path: Option<PathBuf>,
    /// Optional normative database demographic filter.
    #[serde(rename = "databaseFilter")]
    pub database_filter: Option<AnalysisDatabaseFilterPayload>,
    /// Whether `flucalc` should use syllable mode.
    #[serde(rename = "syllableMode")]
    pub syllable_mode: bool,
    /// KidEval DSS utterance cap.
    #[serde(rename = "dssMaxUtterances")]
    pub dss_max_utterances: Option<usize>,
    /// KidEval IPSyn utterance cap.
    #[serde(rename = "ipsynMaxUtterances")]
    pub ipsyn_max_utterances: Option<usize>,
    /// Mortable script path.
    #[serde(rename = "scriptPath")]
    pub script_path: Option<PathBuf>,
    /// Secondary file URI for `rely`.
    #[serde(rename = "secondFile")]
    pub second_file: Option<String>,
    /// Template file path for `script`.
    #[serde(rename = "templatePath")]
    pub template_path: Option<PathBuf>,
    /// Minimum utterance count for `sugar`.
    #[serde(rename = "minUtterances")]
    pub min_utterances: Option<usize>,
    /// First tier for `trnfix`.
    pub tier1: Option<String>,
    /// Second tier for `trnfix`.
    pub tier2: Option<String>,
    /// Whether `uniq` should sort by frequency.
    #[serde(rename = "sortByFrequency")]
    pub sort_by_frequency: bool,
    /// Words that — when an utterance consists solely of them —
    /// cause the utterance to be excluded from `mlu` / `mlt`
    /// counts. Maps CLAN's command-specific `+gS` overload for
    /// those subcommands; distinct from the inherited `--gem`
    /// gem-segment filter.
    #[serde(default, rename = "soloWordExclusions")]
    pub solo_word_exclusions: Vec<String>,
    /// COMBO exclude-search expressions (CLAN: `-sS` for combo).
    /// Utterances matching any of these compound expressions are
    /// dropped from COMBO output. Default empty.
    #[serde(default, rename = "excludeSearch")]
    pub exclude_search: Vec<String>,
    /// DIST `+g`: count each word at most once per turn.
    #[serde(default, rename = "oncePerTurn")]
    pub once_per_turn: bool,
    /// FREQ / VOCD `+c` / `+c0` / `+c1` capitalization predicate.
    /// Serialized as the same kebab-case strings the CLI accepts:
    /// `"initial"` (CLAN `+c`/`+c0`) or `"mid"` (CLAN `+c1`).
    #[serde(default, rename = "capitalization")]
    pub capitalization: Option<String>,
    /// COMBO `+g3`: only report the first matching expression per
    /// utterance.
    #[serde(default, rename = "firstMatchOnly")]
    pub first_match_only: bool,
    /// COMBO `+g7`: deduplicate repeated word matches within an
    /// utterance.
    #[serde(default, rename = "dedupeMatches")]
    pub dedupe_matches: bool,
    /// KWAL `+b`: keyword must be the only countable word on the
    /// tier (single-word utterance match).
    #[serde(default, rename = "strictMatch")]
    pub strict_match: bool,
    /// MAXWD `+xN` (repeatable): drop words of length N from
    /// the output.
    #[serde(default, rename = "excludeLengths")]
    pub exclude_lengths: Vec<usize>,
}

/// Canonical object payload for `talkbank/analyze`.
///
/// The LSP wire protocol still carries this object inside the
/// `workspace/executeCommand` argument vector, but the logical payload is a
/// single object rather than a positional tuple. This shape is the source of
/// truth for schema generation and editor/server type sharing.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
pub struct AnalyzeCommandPayload {
    /// CLAN analysis command identifier.
    #[serde(rename = "commandName")]
    pub command_name: AnalysisCommandName,
    /// File or directory URI to analyze.
    #[serde(rename = "targetUri")]
    pub target_uri: String,
    /// Command-specific options payload.
    #[serde(default)]
    pub options: AnalysisOptionsPayload,
}
