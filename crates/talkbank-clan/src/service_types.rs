//! High-level library service for executing CLAN analyses.
//!
//! `talkbank-clan` already exposes low-level command traits and runner plumbing in
//! [`crate::framework`], but editor integrations should not need to import every
//! individual command type just to execute a named analysis. This module keeps the
//! higher-level analysis execution boundary inside the library so CLI and LSP
//! wrappers can stay focused on adapting outer request shapes. That boundary
//! now also includes a typed analysis-command identifier so outer adapters do
//! not need to route analysis work through raw command-name strings.

use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use thiserror::Error;

use crate::commands::chains::ChainsConfig;
use crate::commands::codes::CodesConfig;
use crate::commands::combo::ComboConfig;
use crate::commands::cooccur::CooccurConfig;
use crate::commands::corelex::CorelexConfig;
use crate::commands::dist::DistConfig;
use crate::commands::dss::DssConfig;
use crate::commands::eval::EvalConfig;
use crate::commands::flucalc::FlucalcConfig;
use crate::commands::freq::FreqConfig;
use crate::commands::freqpos::FreqposConfig;
use crate::commands::ipsyn::IpsynConfig;
use crate::commands::keymap::KeymapConfig;
use crate::commands::kideval::KidevalConfig;
use crate::commands::kwal::KwalConfig;
use crate::commands::maxwd::MaxwdConfig;
use crate::commands::mlt::MltConfig;
use crate::commands::mlu::MluConfig;
use crate::commands::mortable::MortableConfig;
use crate::commands::rely::RelyConfig;
use crate::commands::script::ScriptConfig;
use crate::commands::sugar::SugarConfig;
use crate::commands::trnfix::TrnfixConfig;
use crate::commands::uniq::UniqConfig;
use crate::commands::vocd::VocdConfig;
use crate::commands::wdsize::WdsizeConfig;
use crate::framework::{RunnerError, TransformError};

/// Typed identifier for one supported CLAN analysis command.
///
/// Outer adapters such as the CLI and LSP should parse raw command-name
/// strings into this enum at their boundary, then pass the typed identifier
/// through the shared builder and service layers.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum AnalysisCommandName {
    /// `freq`
    Freq,
    /// `mlu`
    Mlu,
    /// `mlt`
    Mlt,
    /// `wdlen`
    Wdlen,
    /// `wdsize`
    Wdsize,
    /// `maxwd`
    Maxwd,
    /// `freqpos`
    Freqpos,
    /// `timedur`
    Timedur,
    /// `kwal`
    Kwal,
    /// `gemlist`
    Gemlist,
    /// `combo`
    Combo,
    /// `cooccur`
    Cooccur,
    /// `dist`
    Dist,
    /// `chip`
    Chip,
    /// `phonfreq`
    Phonfreq,
    /// `modrep`
    Modrep,
    /// `vocd`
    Vocd,
    /// `codes`
    Codes,
    /// `chains`
    Chains,
    /// `complexity`
    Complexity,
    /// `corelex`
    Corelex,
    /// `dss`
    Dss,
    /// `eval`
    Eval,
    /// `eval-d`
    #[serde(rename = "eval-d")]
    EvalDialect,
    /// `flucalc`
    Flucalc,
    /// `ipsyn`
    Ipsyn,
    /// `keymap`
    Keymap,
    /// `kideval`
    Kideval,
    /// `mortable`
    Mortable,
    /// `rely`
    Rely,
    /// `script`
    Script,
    /// `sugar`
    Sugar,
    /// `trnfix`
    Trnfix,
    /// `uniq`
    Uniq,
}

impl AnalysisCommandName {
    /// Return the stable wire-format name used by CLI and editor adapters.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Freq => "freq",
            Self::Mlu => "mlu",
            Self::Mlt => "mlt",
            Self::Wdlen => "wdlen",
            Self::Wdsize => "wdsize",
            Self::Maxwd => "maxwd",
            Self::Freqpos => "freqpos",
            Self::Timedur => "timedur",
            Self::Kwal => "kwal",
            Self::Gemlist => "gemlist",
            Self::Combo => "combo",
            Self::Cooccur => "cooccur",
            Self::Dist => "dist",
            Self::Chip => "chip",
            Self::Phonfreq => "phonfreq",
            Self::Modrep => "modrep",
            Self::Vocd => "vocd",
            Self::Codes => "codes",
            Self::Chains => "chains",
            Self::Complexity => "complexity",
            Self::Corelex => "corelex",
            Self::Dss => "dss",
            Self::Eval => "eval",
            Self::EvalDialect => "eval-d",
            Self::Flucalc => "flucalc",
            Self::Ipsyn => "ipsyn",
            Self::Keymap => "keymap",
            Self::Kideval => "kideval",
            Self::Mortable => "mortable",
            Self::Rely => "rely",
            Self::Script => "script",
            Self::Sugar => "sugar",
            Self::Trnfix => "trnfix",
            Self::Uniq => "uniq",
        }
    }

    /// Return the CLAN banner scope mode for this command.
    ///
    /// CLAN's `cutt.cpp` mainloop emits one of three scope shapes
    /// depending on the `nomain` and `tct` flags the command sets in
    /// its arg parser. Each chatter command must select the same mode
    /// as its CLAN counterpart for byte-level banner parity.
    pub const fn clan_scope_mode(self) -> ClanScopeMode {
        match self {
            // Dependent-tier-only commands (CLAN `nomain=TRUE` and a
            // single `tct` entry): banner emits just `ONLY dependent
            // tiers matching: %X;` with no speaker-tier prefix.
            Self::Mlu | Self::Vocd => ClanScopeMode::DependentOnly("mor"),

            // Combined commands (CLAN `nomain=FALSE` with a `tct`
            // dependent-tier filter): banner emits `ALL speaker
            // tiers` followed by `and those speakers' ONLY dependent
            // tiers matching: %X;` on a continuation line.
            //
            // `maxwd` was previously here but its CLAN banner is
            // main-only (it counts characters on the main tier, not
            // morphemes on %mor) — moved to MainOnly below.
            Self::Wdlen
            | Self::Wdsize
            | Self::Dss
            | Self::Ipsyn
            | Self::Mortable
            | Self::Corelex
            | Self::Eval
            | Self::EvalDialect
            | Self::Kideval
            | Self::Sugar => ClanScopeMode::MainAndDependent("mor"),
            // `complexity` reads %gra rather than %mor, and CLAN
            // emits a 4th-banner-shape `and ONLY header tiers
            // matching: @ID:;` continuation after the dep-tier line.
            // The header-filter continuation isn't yet modelled in
            // `ClanScopeMode`; the MainAndDependent("gra:") here
            // only captures the dep-tier dimension. Tracked in
            // scripts/clan-parity/STATUS.md.
            Self::Complexity => ClanScopeMode::MainAndDependent("gra:"),
            Self::Phonfreq => ClanScopeMode::MainAndDependent("pho:"),

            // Main-tier-only commands: banner emits just `ALL speaker
            // tiers` (or the explicit speaker-tier filter set).
            // `freqpos` is in this group despite consuming `%mor`
            // because CLAN's freqpos emits the main-only banner.
            Self::Freq
            | Self::Mlt
            | Self::Maxwd
            | Self::Kwal
            | Self::Combo
            | Self::Cooccur
            | Self::Dist
            | Self::Gemlist
            | Self::Chip
            | Self::Modrep
            | Self::Codes
            | Self::Chains
            | Self::Timedur
            | Self::Freqpos
            | Self::Flucalc
            | Self::Keymap
            | Self::Rely
            | Self::Script
            | Self::Trnfix
            | Self::Uniq => ClanScopeMode::MainOnly,
        }
    }
}

/// Selects which scope text CLAN's banner emits.
///
/// CLAN's `cutt.cpp` mainloop branches on `nomain` (does the command
/// consume main tier at all?) and `tct` (is a `+t%X` dependent-tier
/// filter active?). The combinations produce three distinct banner
/// shapes; chatter mirrors that taxonomy here so the
/// banner-emission code can stay in one helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClanScopeMode {
    /// `ALL speaker tiers` — main-tier-only commands.
    MainOnly,
    /// `ONLY dependent tiers matching: %X;` — no main tier, single
    /// dependent tier. The carried `&str` is the tier name without
    /// the leading `%` (e.g. `"mor"`).
    DependentOnly(&'static str),
    /// `ALL speaker tiers\n\tand those speakers' ONLY dependent tiers
    /// matching: %X;` — main tier plus a dependent-tier filter. The
    /// carried `&str` is the tier name (e.g. `"mor"`); for phonfreq
    /// CLAN includes a trailing colon (`%PHO:;`), so the carried
    /// value is `"pho:"`.
    MainAndDependent(&'static str),
}

impl fmt::Display for AnalysisCommandName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error returned when a raw outer-layer command name is not recognized.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error("Unknown analysis command: {command_name}")]
pub struct ParseAnalysisCommandNameError {
    command_name: String,
}

impl FromStr for AnalysisCommandName {
    type Err = ParseAnalysisCommandNameError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "freq" => Ok(Self::Freq),
            "mlu" => Ok(Self::Mlu),
            "mlt" => Ok(Self::Mlt),
            "wdlen" => Ok(Self::Wdlen),
            "wdsize" => Ok(Self::Wdsize),
            "maxwd" => Ok(Self::Maxwd),
            "freqpos" => Ok(Self::Freqpos),
            "timedur" => Ok(Self::Timedur),
            "kwal" => Ok(Self::Kwal),
            "gemlist" => Ok(Self::Gemlist),
            "combo" => Ok(Self::Combo),
            "cooccur" => Ok(Self::Cooccur),
            "dist" => Ok(Self::Dist),
            "chip" => Ok(Self::Chip),
            "phonfreq" => Ok(Self::Phonfreq),
            "modrep" => Ok(Self::Modrep),
            "vocd" => Ok(Self::Vocd),
            "codes" => Ok(Self::Codes),
            "chains" => Ok(Self::Chains),
            "complexity" => Ok(Self::Complexity),
            "corelex" => Ok(Self::Corelex),
            "dss" => Ok(Self::Dss),
            "eval" => Ok(Self::Eval),
            "eval-d" => Ok(Self::EvalDialect),
            "flucalc" => Ok(Self::Flucalc),
            "ipsyn" => Ok(Self::Ipsyn),
            "keymap" => Ok(Self::Keymap),
            "kideval" => Ok(Self::Kideval),
            "mortable" => Ok(Self::Mortable),
            "rely" => Ok(Self::Rely),
            "script" => Ok(Self::Script),
            "sugar" => Ok(Self::Sugar),
            "trnfix" => Ok(Self::Trnfix),
            "uniq" => Ok(Self::Uniq),
            _ => Err(ParseAnalysisCommandNameError {
                command_name: value.to_owned(),
            }),
        }
    }
}

impl<'de> Deserialize<'de> for AnalysisCommandName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

/// Typed, library-owned request for a CLAN analysis command.
///
/// This enum is the stable integration boundary for higher-level consumers such
/// as the CLI and LSP. It keeps command-specific configuration typed without
/// forcing those outer layers to import and execute each command type directly.
#[derive(Debug)]
pub enum AnalysisRequest {
    /// `freq`
    Freq(FreqConfig),
    /// `mlu`
    Mlu(MluConfig),
    /// `mlt`
    Mlt(MltConfig),
    /// `wdlen`
    Wdlen,
    /// `wdsize`
    Wdsize(WdsizeConfig),
    /// `maxwd`
    Maxwd(MaxwdConfig),
    /// `freqpos`
    Freqpos(FreqposConfig),
    /// `timedur`
    Timedur,
    /// `kwal`
    Kwal(KwalConfig),
    /// `gemlist`
    Gemlist,
    /// `combo`
    Combo(ComboConfig),
    /// `cooccur`
    Cooccur(CooccurConfig),
    /// `dist`
    Dist(DistConfig),
    /// `chip`
    Chip,
    /// `phonfreq`
    Phonfreq,
    /// `modrep`
    Modrep,
    /// `vocd`
    Vocd(VocdConfig),
    /// `codes`
    Codes(CodesConfig),
    /// `chains`
    Chains(ChainsConfig),
    /// `complexity`
    Complexity,
    /// `corelex`
    Corelex(CorelexConfig),
    /// `dss`
    Dss(DssConfig),
    /// `eval`
    Eval(EvalConfig),
    /// `flucalc`
    Flucalc(FlucalcConfig),
    /// `ipsyn`
    Ipsyn(IpsynConfig),
    /// `keymap`
    Keymap(KeymapConfig),
    /// `kideval`
    Kideval(KidevalConfig),
    /// `mortable`
    Mortable(MortableConfig),
    /// `script`
    Script(ScriptConfig),
    /// `sugar`
    Sugar(SugarConfig),
    /// `trnfix`
    Trnfix(TrnfixConfig),
    /// `uniq`
    Uniq(UniqConfig),
}

/// FREQ-specific raw input. See [`AnalysisOptions`].
#[derive(Debug, Clone, Default)]
pub struct FreqOptions {
    /// Run `freq` against the `%mor` tier instead of the main tier.
    pub mor: bool,
    /// CLAN `+c` / `+c0` / `+c1` capitalization filter. Default
    /// (`Any`) counts every countable word.
    pub capitalization: crate::framework::CapitalizationFilter,
    /// CLAN `+o1`: sort entries by reverse concordance.
    pub reverse_concordance: bool,
    /// CLAN `+d1`: emit alphabetized deduped word list only.
    pub word_list_only: bool,
    /// CLAN `+d4`: emit only per-speaker type/token/TTR summary.
    pub types_tokens_only: bool,
    /// CLAN `+k`: case-sensitive keying.
    pub case_sensitive: bool,
}

/// MLU-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct MluOptions {
    /// Count words instead of morphemes.
    pub words: bool,
    /// CLAN `+gS`: drop utterances consisting solely of these words.
    pub solo_word_exclusions: Vec<String>,
}

/// MLT-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct MltOptions {
    /// CLAN `+gS`: drop utterances consisting solely of these words.
    pub solo_word_exclusions: Vec<String>,
}

/// WDSIZE-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct WdsizeOptions {
    /// Read from the main tier instead of the `%mor` tier.
    pub main_tier: bool,
    /// CLAN `+w[>|<|=]N`: include only words whose character
    /// length satisfies the comparison.
    pub length_filter: Option<crate::commands::wdsize::LengthFilter>,
}

/// MAXWD-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct MaxwdOptions {
    /// Result limit (CLAN: `+cN`). `None` ⇒ apply `MaxwdConfig`
    /// default.
    pub limit: Option<crate::framework::WordLimit>,
    /// CLAN `+a`: restrict to words whose length is unique within
    /// a speaker's lexicon.
    pub unique_length_only: bool,
    /// CLAN `+xN` (repeatable): drop words of length N. Each
    /// `+xN` on the CLI appends one entry.
    pub exclude_lengths: Vec<usize>,
    /// CLAN `+k`: case-sensitive word keying.
    pub case_sensitive: bool,
}

/// KWAL-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct KwalOptions {
    /// Keyword search list.
    pub keywords: Vec<crate::framework::KeywordPattern>,
    /// CLAN `+b`: keyword must be the only countable word on
    /// the tier (single-word utterance match).
    pub strict_match: bool,
    /// CLAN `+k`: case-sensitive keyword matching. Default
    /// (`false`) lowercases both sides before comparison.
    pub case_sensitive: bool,
    /// CLAN `+d` (no N): emit matching utterances as legal CHAT
    /// (drop the location decoration).
    pub legal_chat: bool,
    /// CLAN `-wN`: pre-match context lines.
    pub context_before: u32,
    /// CLAN `+wN`: post-match context lines.
    pub context_after: u32,
}

/// COMBO-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct ComboOptions {
    /// Search expressions (parsed downstream by
    /// `commands::combo::SearchExpr::parse`).
    pub search: Vec<String>,
    /// Exclude search expressions (CLAN: `-sS`).
    pub exclude_search: Vec<String>,
    /// CLAN `+g3`: only the first matching expression per utterance.
    pub first_match_only: bool,
    /// CLAN `+g7`: deduplicate repeated matched words.
    pub dedupe_matches: bool,
    /// CLAN `+k`: case-sensitive matching. When `true`, the
    /// `SearchExpr::parse_with_case` step preserves case in the
    /// stored terms, and `process_utterance` populates words via
    /// `cleaned_text()` instead of `NormalizedWord::from_word`.
    pub case_sensitive: bool,
    /// CLAN `-wN`: pre-match context lines.
    pub context_before: u32,
    /// CLAN `+wN`: post-match context lines.
    pub context_after: u32,
}

/// DIST-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct DistOptions {
    /// CLAN `+g`: count each word at most once per turn.
    pub once_per_turn: bool,
    /// CLAN `+k`: case-sensitive word keying.
    pub case_sensitive: bool,
}

/// COOCCUR-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct CooccurOptions {
    /// CLAN `+d`: render output without the leading count column.
    pub no_frequency_counts: bool,
    /// CLAN `+nN`: cluster size (number of adjacent words per
    /// row). `0` falls back to the `CooccurConfig` default of 2.
    pub cluster_size: u8,
}

/// FREQPOS-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct FreqposOptions {
    /// CLAN `+d`: switch position classification from
    /// first/last/other to first/second/other.
    pub position_classification: crate::commands::freqpos::PositionClassification,
    /// CLAN `+k`: case-sensitive word keying.
    pub case_sensitive: bool,
}

/// VOCD-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct VocdOptions {
    /// CLAN `+c` / `+c0` / `+c1` capitalization filter. Default
    /// (`Any`) feeds every countable word to the D-statistic
    /// sampler.
    pub capitalization: crate::framework::CapitalizationFilter,
    /// CLAN `+k`: case-sensitive token keying.
    pub case_sensitive: bool,
}

/// CODES-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct CodesOptions {
    /// Maximum hierarchical code depth. `None` ⇒ default.
    pub max_depth: Option<crate::framework::CodeDepth>,
}

/// CHAINS-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct ChainsOptions {
    /// Tier to walk (defaults to `CodesConfig` default).
    pub tier: Option<crate::framework::TierKind>,
}

/// CORELEX-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct CorelexOptions {
    /// Minimum frequency for core classification.
    pub threshold: Option<crate::framework::FrequencyThreshold>,
}

/// DSS-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct DssOptions {
    /// Override the bundled DSS rules file.
    pub rules_path: Option<PathBuf>,
    /// Cap on utterances scored.
    pub max_utterances: Option<crate::framework::UtteranceLimit>,
}

/// EVAL-specific raw input (shared by `eval` and `eval-dialect`).
#[derive(Debug, Clone, Default)]
pub struct EvalOptions {
    /// Optional normative database path.
    pub database_path: Option<PathBuf>,
    /// Optional normative database demographic filter.
    pub database_filter: Option<crate::database::DatabaseFilter>,
}

/// FLUCALC-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct FlucalcOptions {
    /// Use syllable counts instead of word counts.
    pub syllable_mode: bool,
}

/// IPSYN-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct IpsynOptions {
    /// Override the bundled IPSyn rules file.
    pub rules_path: Option<PathBuf>,
    /// Cap on utterances scored.
    pub max_utterances: Option<crate::framework::UtteranceLimit>,
}

/// KEYMAP-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct KeymapOptions {
    /// Keyword search list.
    pub keywords: Vec<crate::framework::KeywordPattern>,
    /// Tier to scan (defaults to `KeymapConfig::default().tier`).
    pub tier: Option<crate::framework::TierKind>,
}

/// KIDEVAL-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct KidevalOptions {
    /// Override the bundled DSS rules file.
    pub dss_rules_path: Option<PathBuf>,
    /// Override the bundled IPSyn rules file.
    pub ipsyn_rules_path: Option<PathBuf>,
    /// Cap on utterances scored by the embedded DSS sub-analysis.
    pub dss_max_utterances: Option<crate::framework::UtteranceLimit>,
    /// Cap on utterances scored by the embedded IPSyn sub-analysis.
    pub ipsyn_max_utterances: Option<crate::framework::UtteranceLimit>,
    /// Optional normative database path.
    pub database_path: Option<PathBuf>,
    /// Optional normative database demographic filter.
    pub database_filter: Option<crate::database::DatabaseFilter>,
}

/// MORTABLE-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct MortableOptions {
    /// Path to the language-script `.cut` file (required).
    pub script_path: Option<PathBuf>,
}

/// RELY-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct RelyOptions {
    /// Path to the comparison file (required).
    pub second_file: Option<PathBuf>,
    /// Tier to align (defaults to `RelyConfig::default().tier`).
    pub tier: Option<crate::framework::TierKind>,
}

/// SCRIPT-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct ScriptOptions {
    /// Path to the template file (required).
    pub template_path: Option<PathBuf>,
}

/// SUGAR-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct SugarOptions {
    /// Minimum utterance count threshold.
    pub min_utterances: Option<crate::framework::UtteranceLimit>,
}

/// TRNFIX-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct TrnfixOptions {
    /// First tier of the swap.
    pub tier1: Option<crate::framework::TierKind>,
    /// Second tier of the swap.
    pub tier2: Option<crate::framework::TierKind>,
}

/// UNIQ-specific raw input.
#[derive(Debug, Clone, Default)]
pub struct UniqOptions {
    /// Sort by descending frequency instead of alphabetical order.
    pub sort_by_frequency: bool,
}

/// Raw analysis options supplied by outer adapters before defaults
/// are applied. Variant carries the per-command `*Options` for
/// commands that take input, or is unit for commands that don't.
///
/// The variant doubles as the command discriminator: the builder
/// no longer needs a separate [`AnalysisCommandName`] parameter
/// because [`Self::command_name`] derives it from the variant.
///
/// Note: `Eval` and `EvalDialect` share the same `EvalOptions`
/// shape but are distinct variants so the dispatcher can pick the
/// right `EvalVariant` downstream.
#[derive(Debug, Clone)]
pub enum AnalysisOptions {
    /// FREQ.
    Freq(FreqOptions),
    /// MLU.
    Mlu(MluOptions),
    /// MLT.
    Mlt(MltOptions),
    /// WDLEN — no input options.
    Wdlen,
    /// WDSIZE.
    Wdsize(WdsizeOptions),
    /// MAXWD.
    Maxwd(MaxwdOptions),
    /// FREQPOS.
    Freqpos(FreqposOptions),
    /// TIMEDUR — no input options.
    Timedur,
    /// KWAL.
    Kwal(KwalOptions),
    /// GEMLIST — no input options.
    Gemlist,
    /// COMBO.
    Combo(ComboOptions),
    /// COOCCUR.
    Cooccur(CooccurOptions),
    /// DIST.
    Dist(DistOptions),
    /// CHIP — no input options.
    Chip,
    /// PHONFREQ — no input options.
    Phonfreq,
    /// MODREP — no input options.
    Modrep,
    /// VOCD.
    Vocd(VocdOptions),
    /// CODES.
    Codes(CodesOptions),
    /// CHAINS.
    Chains(ChainsOptions),
    /// COMPLEXITY — no input options.
    Complexity,
    /// CORELEX.
    Corelex(CorelexOptions),
    /// DSS.
    Dss(DssOptions),
    /// EVAL.
    Eval(EvalOptions),
    /// EVAL-DIALECT (shares `EvalOptions` shape with `Eval`).
    EvalDialect(EvalOptions),
    /// FLUCALC.
    Flucalc(FlucalcOptions),
    /// IPSYN.
    Ipsyn(IpsynOptions),
    /// KEYMAP.
    Keymap(KeymapOptions),
    /// KIDEVAL.
    Kideval(KidevalOptions),
    /// MORTABLE.
    Mortable(MortableOptions),
    /// RELY.
    Rely(RelyOptions),
    /// SCRIPT.
    Script(ScriptOptions),
    /// SUGAR.
    Sugar(SugarOptions),
    /// TRNFIX.
    Trnfix(TrnfixOptions),
    /// UNIQ.
    Uniq(UniqOptions),
}

impl AnalysisOptions {
    /// Derive the command-identity tag from the variant. Used by
    /// callers (banner rendering, scope determination) that need a
    /// stable name string independent of the option payload.
    pub fn command_name(&self) -> AnalysisCommandName {
        match self {
            AnalysisOptions::Freq(_) => AnalysisCommandName::Freq,
            AnalysisOptions::Mlu(_) => AnalysisCommandName::Mlu,
            AnalysisOptions::Mlt(_) => AnalysisCommandName::Mlt,
            AnalysisOptions::Wdlen => AnalysisCommandName::Wdlen,
            AnalysisOptions::Wdsize(_) => AnalysisCommandName::Wdsize,
            AnalysisOptions::Maxwd(_) => AnalysisCommandName::Maxwd,
            AnalysisOptions::Freqpos(_) => AnalysisCommandName::Freqpos,
            AnalysisOptions::Timedur => AnalysisCommandName::Timedur,
            AnalysisOptions::Kwal(_) => AnalysisCommandName::Kwal,
            AnalysisOptions::Gemlist => AnalysisCommandName::Gemlist,
            AnalysisOptions::Combo(_) => AnalysisCommandName::Combo,
            AnalysisOptions::Cooccur(_) => AnalysisCommandName::Cooccur,
            AnalysisOptions::Dist(_) => AnalysisCommandName::Dist,
            AnalysisOptions::Chip => AnalysisCommandName::Chip,
            AnalysisOptions::Phonfreq => AnalysisCommandName::Phonfreq,
            AnalysisOptions::Modrep => AnalysisCommandName::Modrep,
            AnalysisOptions::Vocd(_) => AnalysisCommandName::Vocd,
            AnalysisOptions::Codes(_) => AnalysisCommandName::Codes,
            AnalysisOptions::Chains(_) => AnalysisCommandName::Chains,
            AnalysisOptions::Complexity => AnalysisCommandName::Complexity,
            AnalysisOptions::Corelex(_) => AnalysisCommandName::Corelex,
            AnalysisOptions::Dss(_) => AnalysisCommandName::Dss,
            AnalysisOptions::Eval(_) => AnalysisCommandName::Eval,
            AnalysisOptions::EvalDialect(_) => AnalysisCommandName::EvalDialect,
            AnalysisOptions::Flucalc(_) => AnalysisCommandName::Flucalc,
            AnalysisOptions::Ipsyn(_) => AnalysisCommandName::Ipsyn,
            AnalysisOptions::Keymap(_) => AnalysisCommandName::Keymap,
            AnalysisOptions::Kideval(_) => AnalysisCommandName::Kideval,
            AnalysisOptions::Mortable(_) => AnalysisCommandName::Mortable,
            AnalysisOptions::Rely(_) => AnalysisCommandName::Rely,
            AnalysisOptions::Script(_) => AnalysisCommandName::Script,
            AnalysisOptions::Sugar(_) => AnalysisCommandName::Sugar,
            AnalysisOptions::Trnfix(_) => AnalysisCommandName::Trnfix,
            AnalysisOptions::Uniq(_) => AnalysisCommandName::Uniq,
        }
    }
}

/// Built analysis plan after library-owned defaults and validation are applied.
#[derive(Debug)]
pub enum AnalysisPlan {
    /// Standard request executed through [`AnalysisService`].
    Service(AnalysisRequest),
    /// `rely` still uses an explicit two-file execution path.
    Rely(RelyRequest),
}

/// Typed request for `rely`.
#[derive(Debug)]
pub struct RelyRequest {
    /// Parsed secondary file path.
    pub secondary_file: PathBuf,
    /// Validated RELY configuration.
    pub config: RelyConfig,
}

/// Builder that turns raw outer-layer options into typed library requests.
pub struct AnalysisRequestBuilder {
    options: AnalysisOptions,
}

impl AnalysisRequestBuilder {
    /// Create a builder. The variant encodes the command identity;
    /// callers that previously passed an `AnalysisCommandName`
    /// separately should construct the matching variant instead.
    pub fn new(options: AnalysisOptions) -> Self {
        Self { options }
    }

    /// Validate, apply library defaults, and build the analysis plan.
    pub fn build(self) -> Result<AnalysisPlan, AnalysisServiceError> {
        match self.options {
            AnalysisOptions::Freq(o) => {
                Ok(AnalysisPlan::Service(AnalysisRequest::Freq(FreqConfig {
                    use_mor: o.mor,
                    capitalization: o.capitalization,
                    reverse_concordance: o.reverse_concordance,
                    word_list_only: o.word_list_only,
                    types_tokens_only: o.types_tokens_only,
                    case_sensitive: o.case_sensitive,
                })))
            }
            AnalysisOptions::Mlu(o) => Ok(AnalysisPlan::Service(AnalysisRequest::Mlu(MluConfig {
                words_only: o.words,
                solo_word_exclusions: o.solo_word_exclusions,
            }))),
            AnalysisOptions::Mlt(o) => Ok(AnalysisPlan::Service(AnalysisRequest::Mlt(MltConfig {
                solo_word_exclusions: o.solo_word_exclusions,
            }))),
            AnalysisOptions::Wdlen => Ok(AnalysisPlan::Service(AnalysisRequest::Wdlen)),
            AnalysisOptions::Wdsize(o) => Ok(AnalysisPlan::Service(AnalysisRequest::Wdsize(
                WdsizeConfig {
                    use_main_tier: o.main_tier,
                    length_filter: o.length_filter,
                },
            ))),
            AnalysisOptions::Maxwd(o) => {
                let default = MaxwdConfig::default();
                Ok(AnalysisPlan::Service(AnalysisRequest::Maxwd(MaxwdConfig {
                    limit: o.limit.unwrap_or(default.limit),
                    unique_length_only: o.unique_length_only,
                    exclude_lengths: o.exclude_lengths,
                    case_sensitive: o.case_sensitive,
                })))
            }
            AnalysisOptions::Freqpos(o) => Ok(AnalysisPlan::Service(AnalysisRequest::Freqpos(
                FreqposConfig {
                    position_classification: o.position_classification,
                    case_sensitive: o.case_sensitive,
                },
            ))),
            AnalysisOptions::Timedur => Ok(AnalysisPlan::Service(AnalysisRequest::Timedur)),
            AnalysisOptions::Kwal(o) => {
                Ok(AnalysisPlan::Service(AnalysisRequest::kwal(KwalConfig {
                    keywords: o.keywords,
                    strict_match: o.strict_match,
                    case_sensitive: o.case_sensitive,
                    legal_chat: o.legal_chat,
                    context_before: o.context_before,
                    context_after: o.context_after,
                })?))
            }
            AnalysisOptions::Gemlist => Ok(AnalysisPlan::Service(AnalysisRequest::Gemlist)),
            AnalysisOptions::Combo(o) => {
                let case_sensitive = o.case_sensitive;
                let search: Vec<crate::commands::combo::SearchExpr> = o
                    .search
                    .iter()
                    .map(|expr| {
                        crate::commands::combo::SearchExpr::parse_with_case(expr, case_sensitive)
                    })
                    .collect();
                if search.is_empty() {
                    return Err(AnalysisServiceError::InvalidRequest(
                        "combo requires at least one search expression".to_owned(),
                    ));
                }
                let exclude: Vec<crate::commands::combo::SearchExpr> = o
                    .exclude_search
                    .iter()
                    .map(|expr| {
                        crate::commands::combo::SearchExpr::parse_with_case(expr, case_sensitive)
                    })
                    .collect();
                Ok(AnalysisPlan::Service(AnalysisRequest::Combo(ComboConfig {
                    search,
                    exclude,
                    first_match_only: o.first_match_only,
                    dedupe_matches: o.dedupe_matches,
                    case_sensitive,
                    context_before: o.context_before,
                    context_after: o.context_after,
                })))
            }
            AnalysisOptions::Cooccur(o) => {
                let default = CooccurConfig::default();
                Ok(AnalysisPlan::Service(AnalysisRequest::Cooccur(
                    CooccurConfig {
                        no_frequency_counts: o.no_frequency_counts,
                        cluster_size: if o.cluster_size == 0 {
                            default.cluster_size
                        } else {
                            o.cluster_size
                        },
                    },
                )))
            }
            AnalysisOptions::Dist(o) => {
                Ok(AnalysisPlan::Service(AnalysisRequest::Dist(DistConfig {
                    once_per_turn: o.once_per_turn,
                    case_sensitive: o.case_sensitive,
                })))
            }
            AnalysisOptions::Chip => Ok(AnalysisPlan::Service(AnalysisRequest::Chip)),
            AnalysisOptions::Phonfreq => Ok(AnalysisPlan::Service(AnalysisRequest::Phonfreq)),
            AnalysisOptions::Modrep => Ok(AnalysisPlan::Service(AnalysisRequest::Modrep)),
            AnalysisOptions::Vocd(o) => {
                let default = VocdConfig::default();
                Ok(AnalysisPlan::Service(AnalysisRequest::Vocd(VocdConfig {
                    capitalization: o.capitalization,
                    case_sensitive: o.case_sensitive,
                    ..default
                })))
            }
            AnalysisOptions::Codes(o) => {
                let default = CodesConfig::default();
                Ok(AnalysisPlan::Service(AnalysisRequest::Codes(CodesConfig {
                    max_depth: o.max_depth.unwrap_or(default.max_depth),
                })))
            }
            AnalysisOptions::Chains(o) => {
                let default = ChainsConfig::default();
                Ok(AnalysisPlan::Service(AnalysisRequest::Chains(
                    ChainsConfig {
                        tier: o.tier.unwrap_or(default.tier),
                    },
                )))
            }
            AnalysisOptions::Complexity => Ok(AnalysisPlan::Service(AnalysisRequest::Complexity)),
            AnalysisOptions::Corelex(o) => {
                let default = CorelexConfig::default();
                Ok(AnalysisPlan::Service(AnalysisRequest::Corelex(
                    CorelexConfig {
                        min_frequency: o.threshold.unwrap_or(default.min_frequency),
                    },
                )))
            }
            AnalysisOptions::Dss(o) => {
                let default = DssConfig::default();
                Ok(AnalysisPlan::Service(AnalysisRequest::Dss(DssConfig {
                    rules_path: o.rules_path,
                    max_utterances: o.max_utterances.unwrap_or(default.max_utterances),
                })))
            }
            AnalysisOptions::Eval(o) => {
                let default = EvalConfig::default();
                Ok(AnalysisPlan::Service(AnalysisRequest::Eval(EvalConfig {
                    database_path: o.database_path,
                    database_filter: o.database_filter,
                    ..default
                })))
            }
            AnalysisOptions::EvalDialect(o) => {
                Ok(AnalysisPlan::Service(AnalysisRequest::Eval(EvalConfig {
                    database_path: o.database_path,
                    database_filter: o.database_filter,
                    variant: crate::commands::eval::EvalVariant::Dialect,
                })))
            }
            AnalysisOptions::Flucalc(o) => Ok(AnalysisPlan::Service(AnalysisRequest::Flucalc(
                FlucalcConfig {
                    syllable_mode: o.syllable_mode,
                },
            ))),
            AnalysisOptions::Ipsyn(o) => {
                let default = IpsynConfig::default();
                Ok(AnalysisPlan::Service(AnalysisRequest::Ipsyn(IpsynConfig {
                    rules_path: o.rules_path,
                    max_utterances: o.max_utterances.unwrap_or(default.max_utterances),
                })))
            }
            AnalysisOptions::Keymap(o) => {
                let tier = o.tier.unwrap_or_else(|| KeymapConfig::default().tier);
                Ok(AnalysisPlan::Service(AnalysisRequest::keymap(
                    o.keywords, tier,
                )?))
            }
            AnalysisOptions::Kideval(o) => {
                let default = KidevalConfig::default();
                Ok(AnalysisPlan::Service(AnalysisRequest::Kideval(
                    KidevalConfig {
                        dss_rules_path: o.dss_rules_path,
                        ipsyn_rules_path: o.ipsyn_rules_path,
                        dss_max_utterances: o
                            .dss_max_utterances
                            .unwrap_or(default.dss_max_utterances),
                        ipsyn_max_utterances: o
                            .ipsyn_max_utterances
                            .unwrap_or(default.ipsyn_max_utterances),
                        database_path: o.database_path,
                        database_filter: o.database_filter,
                    },
                )))
            }
            AnalysisOptions::Mortable(o) => {
                let script_path = o.script_path.ok_or_else(|| {
                    AnalysisServiceError::InvalidRequest(
                        "mortable requires a scriptPath option".to_owned(),
                    )
                })?;
                Ok(AnalysisPlan::Service(AnalysisRequest::Mortable(
                    MortableConfig { script_path },
                )))
            }
            AnalysisOptions::Rely(o) => {
                let secondary_file = o.second_file.ok_or_else(|| {
                    AnalysisServiceError::InvalidRequest(
                        "rely requires a secondFile option".to_owned(),
                    )
                })?;
                let tier = o.tier.unwrap_or_else(|| RelyConfig::default().tier);
                Ok(AnalysisPlan::Rely(RelyRequest {
                    secondary_file,
                    config: RelyConfig { tier },
                }))
            }
            AnalysisOptions::Script(o) => {
                let template_path = o.template_path.ok_or_else(|| {
                    AnalysisServiceError::InvalidRequest(
                        "script requires a templatePath option".to_owned(),
                    )
                })?;
                Ok(AnalysisPlan::Service(AnalysisRequest::Script(
                    ScriptConfig { template_path },
                )))
            }
            AnalysisOptions::Sugar(o) => {
                let default = SugarConfig::default();
                Ok(AnalysisPlan::Service(AnalysisRequest::Sugar(SugarConfig {
                    min_utterances: o.min_utterances.unwrap_or(default.min_utterances),
                })))
            }
            AnalysisOptions::Trnfix(o) => {
                let default = TrnfixConfig::default();
                Ok(AnalysisPlan::Service(AnalysisRequest::Trnfix(
                    TrnfixConfig {
                        tier1: o.tier1.unwrap_or(default.tier1),
                        tier2: o.tier2.unwrap_or(default.tier2),
                    },
                )))
            }
            AnalysisOptions::Uniq(o) => {
                Ok(AnalysisPlan::Service(AnalysisRequest::Uniq(UniqConfig {
                    sort_by_frequency: o.sort_by_frequency,
                })))
            }
        }
    }
}

impl AnalysisRequest {
    /// Validate and construct a `kwal` request. The caller assembles the
    /// `KwalConfig` (likely from a `KwalOptions`); this function's only
    /// job is the non-empty-keywords check.
    pub fn kwal(config: KwalConfig) -> Result<Self, AnalysisServiceError> {
        if config.keywords.is_empty() {
            return Err(AnalysisServiceError::InvalidRequest(
                "kwal requires at least one keyword".to_owned(),
            ));
        }
        Ok(Self::Kwal(config))
    }

    /// Validate and construct a `keymap` request.
    pub fn keymap(
        keywords: Vec<crate::framework::KeywordPattern>,
        tier: crate::framework::TierKind,
    ) -> Result<Self, AnalysisServiceError> {
        if keywords.is_empty() {
            return Err(AnalysisServiceError::InvalidRequest(
                "keymap requires at least one keyword".to_owned(),
            ));
        }

        Ok(Self::Keymap(KeymapConfig { keywords, tier }))
    }
}

/// Error from the high-level analysis service boundary.
#[derive(Debug, Error)]
pub enum AnalysisServiceError {
    /// Invalid request shape or unsupported option combination.
    #[error("{0}")]
    InvalidRequest(String),
    /// Underlying transform failure used by non-runner commands such as `rely`.
    #[error(transparent)]
    Transform(#[from] TransformError),
    /// Underlying runner failure.
    #[error(transparent)]
    Runner(#[from] RunnerError),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `AnalysisOptions::command_name` derives the
    /// `AnalysisCommandName` from the variant. Spot-check several
    /// variants to pin the mapping; the full set is exhaustive by
    /// construction inside the method.
    #[test]
    fn analysis_options_command_name_freq() {
        let opts = AnalysisOptions::Freq(FreqOptions::default());
        assert_eq!(opts.command_name(), AnalysisCommandName::Freq);
    }

    #[test]
    fn analysis_options_command_name_combo() {
        let opts = AnalysisOptions::Combo(ComboOptions::default());
        assert_eq!(opts.command_name(), AnalysisCommandName::Combo);
    }

    #[test]
    fn analysis_options_command_name_dist() {
        let opts = AnalysisOptions::Dist(DistOptions::default());
        assert_eq!(opts.command_name(), AnalysisCommandName::Dist);
    }

    /// No-options unit variants still map to their command names.
    #[test]
    fn analysis_options_command_name_unit_variants() {
        assert_eq!(
            AnalysisOptions::Wdlen.command_name(),
            AnalysisCommandName::Wdlen
        );
        assert_eq!(
            AnalysisOptions::Cooccur(CooccurOptions::default()).command_name(),
            AnalysisCommandName::Cooccur,
        );
        assert_eq!(
            AnalysisOptions::Chip.command_name(),
            AnalysisCommandName::Chip
        );
        assert_eq!(
            AnalysisOptions::Complexity.command_name(),
            AnalysisCommandName::Complexity
        );
    }

    /// `Eval` and `EvalDialect` share the `EvalOptions` payload
    /// shape but produce distinct command names. The variant
    /// discrimination is load-bearing because the builder routes
    /// `EvalDialect` to `EvalConfig { variant: Dialect, .. }`.
    #[test]
    fn analysis_options_command_name_distinguishes_eval_dialect() {
        let plain = AnalysisOptions::Eval(EvalOptions::default());
        let dialect = AnalysisOptions::EvalDialect(EvalOptions::default());
        assert_eq!(plain.command_name(), AnalysisCommandName::Eval);
        assert_eq!(dialect.command_name(), AnalysisCommandName::EvalDialect);
    }

    /// Builder consumes the DIST variant and threads
    /// `once_per_turn` into the resulting `DistConfig`.
    #[test]
    fn builder_threads_dist_once_per_turn() {
        let plan = AnalysisRequestBuilder::new(AnalysisOptions::Dist(DistOptions {
            once_per_turn: true,
            case_sensitive: false,
        }))
        .build()
        .expect("dist should build");
        match plan {
            AnalysisPlan::Service(AnalysisRequest::Dist(config)) => {
                assert!(config.once_per_turn);
            }
            other => panic!("unexpected plan: {other:?}"),
        }
    }

    /// Builder threads `FreqOptions::capitalization` into
    /// `FreqConfig::capitalization`.
    #[test]
    fn builder_threads_freq_capitalization() {
        let plan = AnalysisRequestBuilder::new(AnalysisOptions::Freq(FreqOptions {
            mor: false,
            capitalization: crate::framework::CapitalizationFilter::MidUpper,
            reverse_concordance: false,
            word_list_only: false,
            types_tokens_only: false,
            case_sensitive: false,
        }))
        .build()
        .expect("freq should build");
        match plan {
            AnalysisPlan::Service(AnalysisRequest::Freq(config)) => {
                assert_eq!(
                    config.capitalization,
                    crate::framework::CapitalizationFilter::MidUpper
                );
            }
            other => panic!("unexpected plan: {other:?}"),
        }
    }

    /// Builder threads `VocdOptions::capitalization` into
    /// `VocdConfig::capitalization`. Distinct from the FREQ test
    /// because each command has its own `*Config` value but the
    /// `CapitalizationFilter` is shared.
    #[test]
    fn builder_threads_vocd_capitalization() {
        let plan = AnalysisRequestBuilder::new(AnalysisOptions::Vocd(VocdOptions {
            capitalization: crate::framework::CapitalizationFilter::InitialUpper,
            case_sensitive: false,
        }))
        .build()
        .expect("vocd should build");
        match plan {
            AnalysisPlan::Service(AnalysisRequest::Vocd(config)) => {
                assert_eq!(
                    config.capitalization,
                    crate::framework::CapitalizationFilter::InitialUpper
                );
            }
            other => panic!("unexpected plan: {other:?}"),
        }
    }

    /// Builder threads COMBO's `first_match_only` and
    /// `dedupe_matches` into the resulting `ComboConfig`. The
    /// search expression is required (builder errors without it),
    /// so the test provides one.
    #[test]
    fn builder_threads_combo_first_match_and_dedupe() {
        let plan = AnalysisRequestBuilder::new(AnalysisOptions::Combo(ComboOptions {
            search: vec!["want".to_owned()],
            exclude_search: vec![],
            first_match_only: true,
            dedupe_matches: true,
            case_sensitive: false,
            context_before: 0,
            context_after: 0,
        }))
        .build()
        .expect("combo should build");
        match plan {
            AnalysisPlan::Service(AnalysisRequest::Combo(config)) => {
                assert!(config.first_match_only);
                assert!(config.dedupe_matches);
            }
            other => panic!("unexpected plan: {other:?}"),
        }
    }

    /// Builder maps `EvalDialect` variant to
    /// `EvalConfig::variant = Dialect`, not the default `Eval`
    /// variant. Pins the variant routing introduced in the enum
    /// refactor.
    #[test]
    fn builder_eval_dialect_variant_is_dialect() {
        let plan =
            AnalysisRequestBuilder::new(AnalysisOptions::EvalDialect(EvalOptions::default()))
                .build()
                .expect("eval-dialect should build");
        match plan {
            AnalysisPlan::Service(AnalysisRequest::Eval(config)) => {
                assert_eq!(config.variant, crate::commands::eval::EvalVariant::Dialect);
            }
            other => panic!("unexpected plan: {other:?}"),
        }
    }
}
