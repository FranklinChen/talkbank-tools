//! Per-command argument structs and supporting enums.
//!
//! Each processing command (align, transcribe, morphotag, etc.) has its own
//! `*Args` struct embedding [`CommonOpts`](super::CommonOpts) for shared file
//! I/O flags. Utility commands (serve, jobs, logs, cache, etc.) have
//! their own structs and sub-enums here as well.

use clap::{Args, Subcommand, ValueEnum};

use crate::chat_ops::fa::{UtrFuzzyThreshold, UtrOverlapDensityThreshold};
use crate::types::engines::{
    AsrEngineName, AsrSelection, FaEngineName, SelectableEngine, SpeakerEngineName,
    TranslateEngineName, UtrEngine as AppUtrEngine,
};

use super::{CommonOpts, IncrementalOpts};

// ---------------------------------------------------------------------------
// Per-command option enums
//
// Engine choices are NOT here: they live on the domain enums in
// `types::engines` and reach the CLI through `SelectableEngine`.
// ---------------------------------------------------------------------------

/// How `utr` resolves overlapping candidate regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum UtrOverlapStrategy {
    /// Currently equivalent to `global`, the language/content-aware
    /// gate was disabled 2026-03-30 because the two-pass algorithm had
    /// not been validated on operator-reported regression files. See
    /// the book chapter on align for the historical context. Pass
    /// `two-pass` explicitly to opt into the
    /// experimental TwoPassOverlapUtr path.
    #[default]
    Auto,
    /// Single global DP pass (original algorithm). All utterances
    /// participate in one alignment. `+<` utterances get no special
    /// treatment.
    Global,
    /// Two-pass overlap-aware strategy. Pass 1 excludes `+<` utterances
    /// from the global DP. Pass 2 recovers `+<` timing from the
    /// previous utterance's audio window.
    TwoPass,
}

/// Whether CA overlap markers (⌈⌉⌊⌋) are used for alignment windowing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum CaMarkerPolicy {
    /// Use CA markers for onset windowing when present (default).
    #[default]
    Enabled,
    /// Ignore CA markers: treat all overlaps as `+<` only.
    Disabled,
}

/// Legacy review-tier request accepted for command compatibility.
///
/// No value emits `%xalign` or `%xrev`; decisions are retained in structured
/// evidence instead. This enum remains so older scripts still parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum CliReviewLevel {
    /// No review tiers (default).
    #[default]
    None,
    /// Legacy value; retained for compatibility and emits no CHAT tiers.
    #[value(name = "low-confidence")]
    LowConfidence,
    /// Legacy value; retained for compatibility and emits no CHAT tiers.
    All,
}

/// Whether a transcription run attempts speaker diarization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum DiarizationMode {
    /// Automatic (currently defaults to disabled).
    #[default]
    Auto,
    /// Enable speaker diarization.
    Enabled,
    /// Disable speaker diarization.
    Disabled,
}

/// Which utterance-timing-recovery pass an `align` invocation requests.
#[derive(Args, Debug, Clone)]
pub struct AlignUtrSelectionArgs {
    /// UTR engine to recover utterance timings with.
    #[arg(
        long,
        default_value = AppUtrEngine::DEFAULT.selection_name(),
        value_parser = engine_selection_parser::<AppUtrEngine>(),
    )]
    pub utr_engine: AppUtrEngine,

    /// Deprecated alias for `--utr-engine`; hidden, still honoured.
    #[arg(long, hide = true, value_parser = engine_selection_parser::<AppUtrEngine>())]
    pub utr_engine_custom: Option<AppUtrEngine>,

    /// Include utterance timing recovery before forced alignment.
    #[arg(long, default_value_t = true)]
    pub utr: bool,

    /// Skip UTR (faster, but untimed files may get incomplete alignment).
    #[arg(long, conflicts_with = "utr")]
    pub no_utr: bool,

    /// BA2 compat: use --utr-engine whisper instead.
    #[arg(long, hide = true)]
    pub whisper: bool,

    /// BA2 compat: use --utr-engine rev instead.
    #[arg(long, hide = true)]
    pub rev: bool,
}

/// How an enabled utterance-timing-recovery pass operates.
#[derive(Args, Debug, Clone)]
pub struct AlignUtrTuningArgs {
    /// UTR overlap strategy: auto (default), global, or two-pass.
    #[arg(long, value_enum, default_value_t)]
    pub utr_strategy: UtrOverlapStrategy,

    /// Use CA overlap markers (⌈⌉⌊⌋) for alignment windowing.
    #[arg(long, value_enum, default_value_t)]
    pub utr_ca_markers: CaMarkerPolicy,

    /// Max overlap density before skipping pass-1 exclusion (0.0-1.0).
    #[arg(long, default_value = "0.30")]
    pub utr_density_threshold: UtrOverlapDensityThreshold,

    /// Tight window buffer for pass-2 backchannel recovery (ms).
    #[arg(long, default_value_t = 500)]
    pub utr_tight_buffer: u64,

    /// Two-pass UTR word matching threshold (default 0.85); 1.0 requests
    /// exact matching. Applies only with --utr-strategy two-pass. Global
    /// and auto use case-insensitive exact matching.
    #[arg(long)]
    pub utr_fuzzy: Option<UtrFuzzyThreshold>,
}

/// Complete UTR CLI policy, lowered immediately to
/// [`AlignUtrOptions`](crate::types::options::AlignUtrOptions).
///
/// Both parts are flattened, preserving the historical command line while
/// preventing selection/compatibility state from becoming one bag with
/// algorithm tuning.
#[derive(Args, Debug, Clone)]
pub struct AlignUtrArgs {
    /// Whether and which UTR pass is requested.
    #[command(flatten)]
    pub selection: AlignUtrSelectionArgs,

    /// Algorithm policy for an enabled UTR pass.
    #[command(flatten)]
    pub tuning: AlignUtrTuningArgs,
}

/// Existing- and cross-utterance boundary policies for `align`.
#[derive(Args, Debug, Clone)]
pub struct AlignBoundaryArgs {
    /// Treatment of boundaries inherited from an existing `%wor` run.
    #[arg(long, value_enum, default_value_t)]
    pub existing_wor_boundaries: crate::chat_ops::fa::ExistingWorBoundaryPolicy,

    /// Treatment of an earlier utterance end that crosses the next start.
    #[arg(long, value_enum, default_value_t = crate::chat_ops::fa::DEFAULT_END_OVERLAP_POLICY)]
    pub end_overlap_policy: crate::chat_ops::fa::EndOverlapPolicy,
}

/// Arguments for the `align` subcommand (forced alignment).
#[derive(Args, Debug, Clone)]
pub struct AlignArgs {
    /// Shared file I/O options.
    #[command(flatten)]
    pub common: CommonOpts,

    /// Incremental-processing options.
    #[command(flatten)]
    pub incremental: IncrementalOpts,

    /// Utterance-timing-recovery selection and tuning.
    #[command(flatten)]
    pub utr_args: AlignUtrArgs,

    /// Forced-alignment engine.
    #[arg(
        long,
        default_value = FaEngineName::DEFAULT.selection_name(),
        value_parser = engine_selection_parser::<FaEngineName>(),
    )]
    pub fa_engine: FaEngineName,

    /// Deprecated alias for `--fa-engine`; hidden, still honoured.
    #[arg(long, hide = true, value_parser = engine_selection_parser::<FaEngineName>())]
    pub fa_engine_custom: Option<FaEngineName>,

    /// Directory containing media files for alignment.
    /// Matches by filename stem (file.cha looks for file.mp3/mp4/wav).
    #[arg(long, value_name = "PATH")]
    pub media_dir: Option<String>,

    /// Apply post-FA bullet repair to fix timing violations.
    ///
    /// Uses boundary averaging (small overlaps), gap filling (same-speaker),
    /// and selective removal (large violations) instead of CLAN FIXBULLETS.
    /// Experimental: test on real data before enabling in production.
    #[arg(long)]
    pub bullet_repair: bool,

    /// Legacy review-tier setting; retained for command compatibility.
    ///
    /// No value writes `%xalign` or `%xrev`. Machine decisions are retained in
    /// structured run evidence.
    #[arg(long, value_enum, default_value_t)]
    pub review_level: CliReviewLevel,

    /// Try to add pauses between words by grouping them.
    #[arg(long)]
    pub pauses: bool,

    /// Word- and utterance-boundary projection policies.
    #[command(flatten)]
    pub boundaries: AlignBoundaryArgs,

    /// Write word-level alignment (%wor) tier.
    #[arg(long, default_value_t = true)]
    pub wor: bool,

    /// Disable %wor tier.
    #[arg(long, conflicts_with = "wor")]
    pub nowor: bool,

    /// Merge abbreviations in output.
    #[arg(long, conflicts_with = "no_merge_abbrev")]
    pub merge_abbrev: bool,

    /// Do not merge abbreviations in output (default).
    #[arg(long = "no-merge-abbrev", conflicts_with = "merge_abbrev")]
    pub no_merge_abbrev: bool,

    // -- Hidden BA2 compatibility aliases --
    /// BA2 compat: use --fa-engine whisper instead.
    #[arg(long, hide = true)]
    pub whisper_fa: bool,

    /// BA2 compat: use --fa-engine wav2vec instead.
    #[arg(long, hide = true)]
    pub wav2vec: bool,
}

/// Clap value parser producing an [`AsrSelection`] straight from the flag.
///
/// Parsing at the BOUNDARY, so no later stage holds an engine name that might
/// not name an engine. The accepted set is derived from `AsrEngineName`, which
/// does two jobs at once: clap renders it into `--help`, and clap rejects
/// anything else with an error naming what the user typed and listing what is
/// valid. The surface this replaced did neither, so a typo was accepted by the
/// flag and then discarded in silence further down.
/// Value parser for any engine-selection flag.
///
/// Takes NO parameters: everything it needs (the shown names, the hidden
/// aliases, the resolver, the category) travels with the type. The earlier
/// version took those four as separate arguments, none constrained by the
/// others, so nothing stopped a caller pairing one category's names with
/// another's resolver and producing a flag whose help advertised engines it
/// would then reject.
///
/// Parsing at the BOUNDARY, so no later stage holds an engine name that might
/// not name an engine. Clap renders the accepted set into `--help` AND rejects
/// anything else with an error naming what the user typed, so the advertised
/// set and the accepted set are one list by construction.
fn engine_selection_parser<E: SelectableEngine>()
-> impl clap::builder::TypedValueParser<Value = E::Selected> + 'static {
    use clap::builder::TypedValueParser as _;
    let shown = E::selectable_names().map(clap::builder::PossibleValue::new);
    // Historical spellings stay accepted but hidden, so help shows one name per
    // engine while nobody's existing command line breaks.
    let hidden =
        E::hidden_alias_names().map(|alias| clap::builder::PossibleValue::new(alias).hide(true));
    clap::builder::PossibleValuesParser::new(shown.chain(hidden).collect::<Vec<_>>()).try_map(
        move |name: String| {
            // Unreachable in practice: clap has already restricted the value to
            // the list above. An error rather than an unwrap keeps the
            // impossible case impossible without a panic in a shipped binary.
            E::resolve(&name).ok_or_else(|| {
                clap::Error::raw(
                    clap::error::ErrorKind::InvalidValue,
                    format!("unknown {} engine {name:?}\n", E::CATEGORY),
                )
            })
        },
    )
}

impl AlignArgs {
    /// Which UTR engine this invocation selected, or `None` for no UTR pass.
    ///
    /// INFALLIBLE for the same reason [`AsrSelectionArgs::selection`] is: every
    /// field is already a typed engine, so there is no unparsed name left to
    /// fail on. The `None` here means "the user asked for no UTR", which is a
    /// real answer, NOT the old "a name failed to resolve and we said nothing".
    ///
    /// Precedence, widest override first:
    /// 1. `--no-utr`, which turns the pass off outright.
    /// 2. `--utr-engine-custom`, the hidden legacy flag.
    /// 3. the BA2 compatibility switch `--whisper` (unless `--rev` is also set).
    /// 4. `--utr-engine`, which has a default and so is always present.
    pub fn utr_selection(&self) -> Option<AppUtrEngine> {
        if !self.utr_args.selection.utr || self.utr_args.selection.no_utr {
            return None;
        }
        if let Some(ref engine) = self.utr_args.selection.utr_engine_custom {
            return Some(engine.clone());
        }
        if self.utr_args.selection.whisper && !self.utr_args.selection.rev {
            return Some(AppUtrEngine::Whisper);
        }
        Some(self.utr_args.selection.utr_engine.clone())
    }

    /// Which forced-alignment engine this invocation selected.
    ///
    /// Precedence, widest override first:
    /// 1. `--fa-engine-custom`, the hidden legacy flag.
    /// 2. the BA2 compatibility switches, which are explicit user intent while
    ///    `--fa-engine` always carries a default.
    /// 3. `--fa-engine`.
    ///
    /// `--wav2vec` was dead before 2026-07-01: nothing consulted the flag, so
    /// it silently fell through to whatever the default happened to be. That
    /// was masked while the default WAS Wav2Vec, and exposed the moment the
    /// default became Whisper. It is consulted here.
    pub fn fa_selection(&self) -> FaEngineName {
        if let Some(ref engine) = self.fa_engine_custom {
            return *engine;
        }
        if self.whisper_fa {
            return FaEngineName::Whisper;
        }
        if self.wav2vec {
            return FaEngineName::Wave2Vec;
        }
        self.fa_engine
    }
}

/// The ASR engine selection surface, shared by every command that transcribes.
///
/// One struct, flattened into each command, because there is one question here
/// and it had been answered twice. `transcribe` and `benchmark` each carried
/// their own engine flag, their own hand-written value list (five of ten and
/// three of ten), their own copy of the BA2 compatibility ladder, and their own
/// `--asr-engine-custom`. Fixing the resolution in one of them left the other
/// holding the original defect, including an unknown name resolving to nothing
/// at all, in silence.
///
/// Now the flag, the accepted values, the precedence and the resolution have a
/// single definition, and a new command gets them by flattening this.
#[derive(Args, Debug, Clone)]
pub struct AsrSelectionArgs {
    // Keep the doc comment SHORT: clap renders it verbatim as the flag's help,
    // so rationale written here is shown to users instead of the value list.
    /// ASR engine to transcribe with.
    #[arg(
        long,
        default_value = AsrEngineName::DEFAULT.selection_name(),
        value_parser = engine_selection_parser::<AsrEngineName>(),
    )]
    pub asr_engine: AsrSelection,

    /// Deprecated alias for `--asr-engine`; hidden, still honoured.
    ///
    /// Kept so existing scripts and book examples keep working. Hidden from
    /// help because two doors onto one choice is what let the visible one go
    /// stale. It is parsed by the SAME parser, so it can no longer accept a
    /// name that `--asr-engine` would reject.
    #[arg(long, hide = true, value_parser = engine_selection_parser::<AsrEngineName>())]
    pub asr_engine_custom: Option<AsrSelection>,

    /// BA2 compat: use --asr-engine whisper instead.
    #[arg(long, hide = true)]
    pub whisper: bool,

    /// BA2 compat: use --asr-engine whisperx instead.
    #[arg(long, hide = true)]
    pub whisperx: bool,

    /// BA2 compat: use --asr-engine whisper_oai instead.
    #[arg(long, hide = true)]
    pub whisper_oai: bool,

    /// BA2 compat: use --asr-engine rev instead.
    #[arg(long, hide = true)]
    pub rev: bool,
}

impl AsrSelectionArgs {
    /// Which ASR engine this invocation selected.
    ///
    /// INFALLIBLE, which is the point: every field is already an
    /// [`AsrSelection`] or a bool, so there is no unparsed name left to fail
    /// on and no `None` for a caller to mishandle. Clap rejected an unknown
    /// name at parse time, with the valid list, before this runs.
    ///
    /// Precedence, widest override first:
    /// 1. `--asr-engine-custom`, the hidden legacy flag.
    /// 2. the BA2 compatibility switches.
    /// 3. `--asr-engine`, which has a default and so is always present.
    pub fn selection(&self) -> AsrSelection {
        use crate::types::engines::AsrEngineName;
        if let Some(ref selection) = self.asr_engine_custom {
            return selection.clone();
        }
        // Compat switches are checked before the flag because they are explicit
        // user intent while `--asr-engine` always carries a default. Built from
        // the enum, not by re-parsing a string: these four engines are known at
        // compile time, so a wire-name change must not be able to turn them
        // into a silent no-op.
        for (set, engine) in [
            (self.whisperx, AsrEngineName::WhisperX),
            (self.whisper_oai, AsrEngineName::WhisperOai),
            (self.whisper, AsrEngineName::Whisper),
            (self.rev, AsrEngineName::RevAi),
        ] {
            if set {
                return AsrSelection::from_engine(engine);
            }
        }
        self.asr_engine.clone()
    }
}

/// Arguments for the `transcribe` command.
#[derive(Args, Debug, Clone)]
pub struct TranscribeArgs {
    /// Shared file I/O options.
    #[command(flatten)]
    pub common: CommonOpts,

    /// ASR engine selection (flag, legacy alias, BA2 switches).
    #[command(flatten)]
    pub asr: AsrSelectionArgs,

    /// Speaker diarization mode: auto (default), enabled, or disabled.
    #[arg(long, value_enum, default_value_t)]
    pub diarization: DiarizationMode,

    /// Dedicated diarization engine. Defaults to pyannoteAI when diarization is enabled.
    #[arg(long, value_parser = engine_selection_parser::<SpeakerEngineName>())]
    pub speaker_engine: Option<SpeakerEngineName>,

    /// Write word-level alignment (%wor) tier.
    #[arg(long, conflicts_with = "nowor")]
    pub wor: bool,

    /// Disable %wor tier (default).
    #[arg(long, conflicts_with = "wor")]
    pub nowor: bool,

    /// Merge abbreviations in output.
    #[arg(long, conflicts_with = "no_merge_abbrev")]
    pub merge_abbrev: bool,

    /// Do not merge abbreviations in output (default).
    #[arg(long = "no-merge-abbrev", conflicts_with = "merge_abbrev")]
    pub no_merge_abbrev: bool,

    /// Opt in to the legacy Stanza constituency-parser fallback for
    /// utterance segmentation when no language-specific TalkBank BERT
    /// model is configured for `--lang`. Default refuses substitution;
    /// pass this flag to permit the same Stanza-based segmenter that
    /// Batchalign 2 used for unsupported languages (quality varies).
    #[arg(long)]
    pub utseg_fallback_stanza: bool,

    /// Language (3-letter ISO code).
    #[arg(long, default_value = "eng")]
    pub lang: String,

    /// Expected number of speakers. NOT a worker count: see `--workers`.
    /// No short flag by design; the book explains why `-n` was removed.
    #[arg(long, default_value_t = 2)]
    pub num_speakers: u32,

    // -- Hidden BA2 compatibility aliases --
    /// BA2 compat: use --diarization enabled instead.
    #[arg(long, hide = true)]
    pub diarize: bool,

    /// BA2 compat: use --diarization disabled instead.
    #[arg(long, hide = true)]
    pub nodiarize: bool,
}

/// Arguments for the `translate` command.
///
/// **No `--lang` flag.** BA2 parity (`~/batchalign2-master/batchalign/cli/cli.py`
/// `translate` command takes no `--lang`). Source language is read per-file
/// from the CHAT file's `@Languages:` header (BA2
/// `pipelines/translate/seamless.py:40` uses `doc.langs[0]`); the
/// translation target is hardcoded to English (BA2 `seamless.py:41`
/// `tgt_lang="eng"`). The 2026-05-03 morphotag incident showed that a
/// job-level lang sentinel silently overrides per-file routing, do not
/// re-introduce `--lang` here without re-reading that postmortem.
#[derive(Args, Debug, Clone)]
pub struct TranslateArgs {
    /// Shared file I/O options.
    #[command(flatten)]
    pub common: CommonOpts,

    // Keep this SHORT: clap renders it verbatim above the value list. The
    // prose here used to name four of the five engines, omitting `aliyun`,
    // which is the same stale-hand-written-list defect the value list now
    // fixes. Per-engine tradeoffs belong in the book, not in `--help`.
    /// Translation engine.
    #[arg(
        long,
        default_value = TranslateEngineName::DEFAULT.selection_name(),
        value_parser = engine_selection_parser::<TranslateEngineName>(),
    )]
    pub translate_engine: TranslateEngineName,

    /// Merge abbreviations in output.
    #[arg(long, conflicts_with = "no_merge_abbrev")]
    pub merge_abbrev: bool,

    /// Do not merge abbreviations in output (default).
    #[arg(long = "no-merge-abbrev", conflicts_with = "merge_abbrev")]
    pub no_merge_abbrev: bool,
}

/// Morphotag analysis-policy switches lowered into typed runtime policy at the
/// CLI boundary.
#[derive(Args, Debug, Clone)]
pub struct MorphotagPolicyArgs {
    /// Opt out of L2 dispatch for `@s` (code-switched) words.
    ///
    /// By default morphotag routes `@s` words to the secondary-language
    /// Stanza path and splices the resulting morphology back into `%mor`.
    /// Pass `--no-l2-morphotag` to keep the legacy `L2|xxx` placeholders
    /// instead.
    #[arg(long, default_value_t = false)]
    pub no_l2_morphotag: bool,

    /// Opt out of transcriber-supplied `$POS` hint respect.
    #[arg(long, default_value_t = false)]
    pub no_pos_hints: bool,

    /// Choose whether `@Options: CA` files remain pass-through or are
    /// intentionally analyzed while retaining the CA header.
    #[arg(
        long,
        value_enum,
        default_value_t = crate::options::CaMorphotagPolicy::Honor
    )]
    pub ca_policy: crate::options::CaMorphotagPolicy,
}

/// Arguments for the `morphotag` command.
///
/// **No `--lang` flag.** BA2 parity (`~/batchalign2-master/batchalign/cli/cli.py`
/// `morphotag` command takes no `--lang`). The processing language is read
/// per-file from the CHAT file's `@Languages:` header (see
/// `pipeline/morphosyntax.rs::stage_parse`). Files whose primary language
/// is not Stanza-supported hard-error out; the daemon does not silently
/// rewrite them with English morphotag (the 2026-05-03 incident).
#[derive(Args, Debug, Clone)]
pub struct MorphotagArgs {
    /// Shared file I/O options.
    #[command(flatten)]
    pub common: CommonOpts,

    /// Incremental-processing options.
    #[command(flatten)]
    pub incremental: IncrementalOpts,

    /// Retokenize the main line to fit UD tokenizations.
    ///
    /// WARNING: This modifies the main tier text to match Stanza's UD
    /// tokenization (splitting/merging words). Existing word-level timing
    /// bullets and %wor tiers may become stale. Use --before for incremental
    /// processing to preserve unaffected utterances.
    #[arg(long, conflicts_with = "keeptokens")]
    pub retokenize: bool,

    /// Keep existing tokenization (default).
    #[arg(long, conflicts_with = "retokenize")]
    pub keeptokens: bool,

    /// Skip code switching.
    #[arg(long, conflicts_with = "multilang")]
    pub skipmultilang: bool,

    /// Keep multilingual spans (default).
    #[arg(long, conflicts_with = "skipmultilang")]
    pub multilang: bool,

    /// Morphotag analysis-policy switches.
    #[command(flatten)]
    pub policy: MorphotagPolicyArgs,

    /// Comma-separated manual lexicon override file.
    #[arg(long)]
    pub lexicon: Option<String>,

    /// Merge abbreviations in output.
    #[arg(long, conflicts_with = "no_merge_abbrev")]
    pub merge_abbrev: bool,

    /// Do not merge abbreviations in output (default).
    #[arg(long = "no-merge-abbrev", conflicts_with = "merge_abbrev")]
    pub no_merge_abbrev: bool,

    /// Legacy review-tier setting; retained for command compatibility.
    ///
    /// No value writes `%xalign` or `%xrev`. Machine decisions are retained in
    /// structured run evidence.
    #[arg(long, value_enum, default_value_t)]
    pub review_level: CliReviewLevel,
}

/// Arguments for the `coref` command.
/// Arguments for the `coref` command.
///
/// **No `--lang` flag.** BA2 parity (`~/batchalign2-master/batchalign/cli/cli.py`
/// `coref` command takes no `--lang`). Coref is English-only, non-English
/// files pass through unchanged based on the per-file `@Languages:` header
/// (see `coref.rs::file_has_english`). Re-introducing `--lang` here would
/// recreate the 2026-05-03 morphotag failure mode where a job-level sentinel
/// silently overrode per-file language routing.
#[derive(Args, Debug, Clone)]
pub struct CorefArgs {
    /// Shared file I/O options.
    #[command(flatten)]
    pub common: CommonOpts,

    /// Merge abbreviations in output.
    #[arg(long, conflicts_with = "no_merge_abbrev")]
    pub merge_abbrev: bool,

    /// Do not merge abbreviations in output (default).
    #[arg(long = "no-merge-abbrev", conflicts_with = "merge_abbrev")]
    pub no_merge_abbrev: bool,
}

/// Arguments for the `compare` command.
#[derive(Args, Debug, Clone)]
pub struct CompareArgs {
    /// Shared file I/O options.
    #[command(flatten)]
    pub common: CommonOpts,

    /// Language (3-letter ISO code).
    #[arg(long, default_value = "eng")]
    pub lang: String,

    /// Expected number of speakers. NOT a worker count: see `--workers`.
    /// No short flag by design; the book explains why `-n` was removed.
    #[arg(long, default_value_t = 2)]
    pub num_speakers: u32,

    /// Merge abbreviations in output.
    #[arg(long, conflicts_with = "no_merge_abbrev")]
    pub merge_abbrev: bool,

    /// Do not merge abbreviations in output (default).
    #[arg(long = "no-merge-abbrev", conflicts_with = "merge_abbrev")]
    pub no_merge_abbrev: bool,
}

/// Arguments for the `utseg` command.
#[derive(Args, Debug, Clone)]
pub struct UtsegArgs {
    /// Shared file I/O options.
    #[command(flatten)]
    pub common: CommonOpts,

    /// Language (3-letter ISO code).
    #[arg(long, default_value = "eng")]
    pub lang: String,

    /// Expected number of speakers. NOT a worker count: see `--workers`.
    /// No short flag by design; the book explains why `-n` was removed.
    #[arg(long, default_value_t = 2)]
    pub num_speakers: u32,

    /// Merge abbreviations in output.
    #[arg(long, conflicts_with = "no_merge_abbrev")]
    pub merge_abbrev: bool,

    /// Do not merge abbreviations in output (default).
    #[arg(long = "no-merge-abbrev", conflicts_with = "merge_abbrev")]
    pub no_merge_abbrev: bool,

    /// Opt in to the legacy Stanza constituency-parser fallback for
    /// utterance segmentation when no language-specific TalkBank BERT
    /// model is configured for `--lang`. Default refuses substitution;
    /// pass this flag to permit the same Stanza-based segmenter that
    /// Batchalign 2 used for unsupported languages (quality varies).
    #[arg(long)]
    pub utseg_fallback_stanza: bool,
}

/// Arguments for the `benchmark` command.
#[derive(Args, Debug, Clone)]
pub struct BenchmarkArgs {
    /// Shared file I/O options.
    #[command(flatten)]
    pub common: CommonOpts,

    /// ASR engine selection (flag, legacy alias, BA2 switches).
    ///
    /// The same surface `transcribe` has. It previously advertised three of the
    /// ten engines through its own enum, which was a restriction with nothing
    /// behind it: `BenchmarkOptions::asr_engine` was already the full
    /// `AsrEngineName`, and `--asr-engine-custom` routed around the enum anyway.
    #[command(flatten)]
    pub asr: AsrSelectionArgs,

    /// Language (3-letter ISO code).
    #[arg(long, default_value = "eng")]
    pub lang: String,

    /// Expected number of speakers. NOT a worker count: see `--workers`.
    /// No short flag by design; the book explains why `-n` was removed.
    #[arg(long, default_value_t = 2)]
    pub num_speakers: u32,

    /// Write word-level alignment (%wor) tier. See
    /// [`BenchmarkOptions::wor`](crate::types::options::BenchmarkOptions::wor)
    /// for rationale.
    #[arg(long, conflicts_with = "nowor", default_value_t = true)]
    pub wor: bool,

    /// Disable %wor tier.
    #[arg(long, conflicts_with = "wor")]
    pub nowor: bool,

    /// Merge abbreviations in output.
    #[arg(long, conflicts_with = "no_merge_abbrev")]
    pub merge_abbrev: bool,

    /// Do not merge abbreviations in output (default).
    #[arg(long = "no-merge-abbrev", conflicts_with = "merge_abbrev")]
    pub no_merge_abbrev: bool,

    /// Server media bank name (from server.yaml media_mappings).
    #[arg(long)]
    pub bank: Option<String>,

    /// Subdirectory under the bank.
    #[arg(long)]
    pub subdir: Option<String>,
}

/// Arguments for the `opensmile` command.
#[derive(Args, Debug, Clone)]
pub struct OpensmileArgs {
    /// Input directory.
    pub input_dir: std::path::PathBuf,
    /// Output directory.
    pub output_dir: std::path::PathBuf,

    /// Feature set to extract.
    #[arg(long, default_value = "eGeMAPSv02",
          value_parser = ["eGeMAPSv02", "eGeMAPSv01b", "GeMAPSv01b", "ComParE_2016"])]
    pub feature_set: String,

    /// Language (3-letter ISO code).
    #[arg(long, default_value = "eng")]
    pub lang: String,

    /// Server media bank name.
    #[arg(long)]
    pub bank: Option<String>,

    /// Subdirectory under the bank.
    #[arg(long)]
    pub subdir: Option<String>,
}

/// Arguments for the `diarize` command.
#[derive(Args, Debug, Clone)]
pub struct DiarizeArgs {
    /// Shared file-I/O options (input media paths, output directory).
    #[command(flatten)]
    pub common: CommonOpts,

    /// Speaker diarization engine. The paid pyannoteAI path is opt-in.
    #[arg(
        long,
        value_parser = engine_selection_parser::<SpeakerEngineName>(),
        default_value = "pyannote"
    )]
    pub speaker_engine: SpeakerEngineName,

    /// Expected number of speakers. Omit to auto-detect (recommended).
    ///
    /// NOT a worker count: see `--workers`. No short flag by design.
    #[arg(long)]
    pub num_speakers: Option<u32>,

    /// Language (3-letter ISO code). Worker-pool selection only;
    /// diarization itself is language-independent.
    #[arg(long, default_value = "eng")]
    pub lang: String,
}

/// Read one `<start_ms>-<end_ms>:<label>` enrollment argument.
///
/// A clap value parser rather than a `String` field, so a malformed span is
/// refused by argument parsing with the offending text quoted, and nothing
/// downstream ever holds an unvalidated enrollment.
fn parse_enrollment(
    value: &str,
) -> Result<
    crate::chat_ops::speaker_identity::EnrollmentSpec,
    crate::chat_ops::speaker_identity::InvalidEnrollment,
> {
    crate::chat_ops::speaker_identity::EnrollmentSpec::parse(value)
}

/// Why a CLI threshold argument could not become a domain threshold.
#[derive(Debug, thiserror::Error)]
enum ThresholdArgumentError {
    /// The argument was not a floating-point number.
    #[error("{value:?} is not a number")]
    Parse {
        /// The original CLI spelling.
        value: String,
        /// The standard parser's diagnostic.
        #[source]
        source: std::num::ParseFloatError,
    },
    /// The number was outside the cosine-similarity domain.
    #[error(transparent)]
    Range(#[from] crate::chat_ops::speaker_identity::NotAThreshold),
}

/// Read the `--threshold` value into the range a cosine similarity can reach.
fn parse_threshold(
    value: &str,
) -> Result<crate::chat_ops::speaker_identity::MatchThreshold, ThresholdArgumentError> {
    let parsed: f64 = value
        .parse()
        .map_err(|source| ThresholdArgumentError::Parse {
            value: value.to_owned(),
            source,
        })?;
    Ok(crate::chat_ops::speaker_identity::MatchThreshold::try_from(
        parsed,
    )?)
}

/// Arguments for the `speaker-identify` command.
#[derive(Args, Debug, Clone)]
pub struct SpeakerIdentifyArgs {
    /// Shared file-I/O options (input transcripts, output directory).
    #[command(flatten)]
    pub common: CommonOpts,

    /// A span known to contain one speaker alone, as
    /// `<start_ms>-<end_ms>:<label>`. Repeat once per known voice.
    ///
    /// Example: `--enroll 1500-9000:INV`. The milliseconds are positions in
    /// the recording, the same coordinates a CHAT timing bullet uses.
    #[arg(long = "enroll", required = true, value_parser = parse_enrollment)]
    pub enroll: Vec<crate::chat_ops::speaker_identity::EnrollmentSpec>,

    /// Speaker tiers to score, comma-separated or repeated. Omit for all.
    #[arg(long, value_delimiter = ',')]
    pub tiers: Vec<String>,

    /// Similarity at or above which an enrolled voice counts as a match.
    ///
    /// REQUIRED, and deliberately has no default: how much acoustic agreement
    /// counts as the same person depends on the recording, the microphone and
    /// how much enrollment audio there is, none of which this tool knows. A
    /// default would produce confident verdicts under a number nobody chose.
    #[arg(long, required = true, value_parser = parse_threshold)]
    pub threshold: crate::chat_ops::speaker_identity::MatchThreshold,

    /// Language (3-letter ISO code). Worker-pool selection only; speaker
    /// embedding itself is language-independent.
    #[arg(long, default_value = "eng")]
    pub lang: String,
}

impl SpeakerIdentifyArgs {
    /// Validate the enrollments as a SET: non-empty, uniquely labelled, and
    /// non-overlapping.
    ///
    /// Clap validates each `--enroll` on its own; these three are relations
    /// BETWEEN them, which no per-value parser can see.
    pub fn enrollment_set(
        &self,
    ) -> Result<
        crate::chat_ops::speaker_identity::EnrollmentSet,
        crate::chat_ops::speaker_identity::InvalidEnrollmentSet,
    > {
        crate::chat_ops::speaker_identity::EnrollmentSet::new(self.enroll.clone())
    }
}

/// Arguments for the `avqi` command.
#[derive(Args, Debug, Clone)]
pub struct AvqiArgs {
    /// Input directory containing paired .cs/.sv audio files.
    pub input_dir: std::path::PathBuf,
    /// Output directory.
    pub output_dir: std::path::PathBuf,

    /// Language (3-letter ISO code).
    #[arg(long, default_value = "eng")]
    pub lang: String,
}

/// ASR engine choice for the `setup` command.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetupEngine {
    /// Rev.AI ASR engine.
    Rev,
    /// Huggingface Whisper ASR engine.
    Whisper,
}

/// Processing command target for the `bench` subcommand.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchTarget {
    /// Forced alignment.
    Align,
    /// Transcription (single speaker).
    Transcribe,
    /// Transcription with speaker diarization.
    #[value(name = "transcribe_s")]
    TranscribeS,
    /// Morphosyntactic tagging.
    Morphotag,
    /// Translation.
    Translate,
    /// Utterance segmentation.
    Utseg,
    /// WER benchmarking.
    Benchmark,
    /// OpenSMILE feature extraction.
    Opensmile,
    /// Coreference resolution.
    Coref,
    /// Transcript comparison against gold standard.
    Compare,
}

impl BenchTarget {
    /// Return the server-side command name string for this target.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Align => "align",
            Self::Transcribe => "transcribe",
            Self::TranscribeS => "transcribe_s",
            Self::Morphotag => "morphotag",
            Self::Translate => "translate",
            Self::Utseg => "utseg",
            Self::Benchmark => "benchmark",
            Self::Opensmile => "opensmile",
            Self::Coref => "coref",
            Self::Compare => "compare",
        }
    }
}

/// Arguments for the `bench` subcommand (repeated benchmark runs).
#[derive(Args, Debug, Clone)]
pub struct BenchArgs {
    /// Command to benchmark.
    pub command: BenchTarget,

    /// Input directory.
    pub in_dir: std::path::PathBuf,

    /// Output directory.
    pub out_dir: std::path::PathBuf,

    /// Number of benchmark runs.
    #[arg(long, default_value_t = 1)]
    pub runs: usize,

    /// Dataset label for structured output.
    #[arg(long)]
    pub dataset: Option<String>,

    /// Number of workers to use.
    #[arg(long)]
    pub workers: Option<usize>,

    /// Use cache for benchmark runs (default is to bypass cache).
    #[arg(long)]
    pub use_cache: bool,
}

/// Arguments for the `models` subcommand (model training utilities).
#[derive(Args, Debug, Clone)]
pub struct ModelsArgs {
    /// Subcommand.
    #[command(subcommand)]
    pub action: ModelsAction,
}

/// Model training subcommands.
#[derive(Subcommand, Debug, Clone)]
pub enum ModelsAction {
    /// Extract training text from CHAT files (Rust-native, no CLAN needed).
    Prep(ModelsPrepArgs),
    /// Train a model (forwards to Python training runtime).
    Train(ModelsTrainArgs),
}

/// Arguments for `models prep`.
#[derive(Args, Debug, Clone)]
pub struct ModelsPrepArgs {
    /// Run name (used as prefix for output files).
    pub run_name: String,

    /// Input directory containing .cha files.
    pub input_dir: std::path::PathBuf,

    /// Output directory for prepared .train.txt and .val.txt files.
    pub output_dir: std::path::PathBuf,

    /// Minimum word count per utterance (shorter utterances are excluded).
    #[arg(long, default_value_t = 10)]
    pub min_length: usize,

    /// Separate validation directory. If not given, splits from input.
    #[arg(long)]
    pub val_dir: Option<String>,

    /// Fraction of data to use for validation when --val-dir is not given.
    #[arg(long, default_value_t = 0.1)]
    pub val_fraction: f64,
}

/// Arguments for `models train` (forwarded to Python).
#[derive(Args, Debug, Clone)]
pub struct ModelsTrainArgs {
    /// Arguments passed through to `python -m batchalign.models.training.run`.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub args: Vec<String>,
}

/// Arguments for the `setup` subcommand (initialize `~/.batchalign.ini`).
#[derive(Args, Debug, Clone)]
pub struct SetupArgs {
    /// Default ASR engine to persist in ~/.batchalign.ini.
    #[arg(long, value_enum)]
    pub engine: Option<SetupEngine>,

    /// Rev.ai API key (required with --engine rev in non-interactive mode).
    #[arg(long)]
    pub rev_key: Option<String>,

    /// Disable prompts and rely only on flags.
    #[arg(long)]
    pub non_interactive: bool,

    /// Download (or verify) the whisper_rs default model now, so first
    /// use does not pay a ~3.1 GB download inside a job.
    #[arg(long)]
    pub prefetch_whisper_rs: bool,
}

/// Arguments for the offline `compare-runs` utility family.
#[derive(Args, Debug, Clone)]
pub struct CompareRunsArgs {
    /// Offline comparison action.
    #[command(subcommand)]
    pub action: CompareRunsAction,
}

/// Offline manifest authoring and comparison actions.
#[derive(Subcommand, Debug, Clone)]
pub enum CompareRunsAction {
    /// Create a canonical immutable run manifest.
    Manifest(CompareRunsManifestArgs),
    /// Compare transcription agreement (never labeled accuracy).
    Transcribe(CompareRunsExecuteArgs),
    /// Compare structured morphology and dependency annotations.
    Morphotag(CompareRunsExecuteArgs),
    /// Compare `%wor` timings for identical normalized tokens.
    Align(CompareRunsExecuteArgs),
}

/// Identity-specific manifest authoring action.
#[derive(Args, Debug, Clone)]
pub struct CompareRunsManifestArgs {
    /// Producer identity kind.
    #[command(subcommand)]
    pub identity: CompareRunsManifestIdentity,
}

/// Typed producer identity used to author a manifest.
#[derive(Subcommand, Debug, Clone)]
pub enum CompareRunsManifestIdentity {
    /// Artifacts produced by an executable implementation.
    Machine(MachineManifestArgs),
    /// Artifacts produced under a human review protocol.
    Human(HumanManifestArgs),
}

/// Shared manifest fields.
#[derive(Args, Debug, Clone)]
pub struct ManifestCommonArgs {
    /// Directory containing immutable artifacts to inventory.
    #[arg(long)]
    pub artifacts: std::path::PathBuf,
    /// Manifest JSON destination; must be outside the artifact root.
    #[arg(long)]
    pub output: std::path::PathBuf,
    /// Stable identity for this produced run.
    #[arg(long)]
    pub run_id: String,
    /// Stable source media or source CHAT identity.
    #[arg(long)]
    pub source_id: String,
    /// Normalized producer argument as KEY=VALUE; repeat as needed.
    #[arg(long = "argument", value_name = "KEY=VALUE")]
    pub arguments: Vec<String>,
}

/// Machine-producer manifest fields.
#[derive(Args, Debug, Clone)]
pub struct MachineManifestArgs {
    /// Shared manifest fields.
    #[command(flatten)]
    pub common: ManifestCommonArgs,
    /// Stable implementation label.
    #[arg(long)]
    pub implementation: String,
    /// Command family that produced these artifacts.
    #[arg(long)]
    pub command: String,
    /// Reproducible source/build identity.
    #[arg(long)]
    pub build: String,
}

/// Human-producer manifest fields.
#[derive(Args, Debug, Clone)]
pub struct HumanManifestArgs {
    /// Shared manifest fields.
    #[command(flatten)]
    pub common: ManifestCommonArgs,
    /// Review protocol version.
    #[arg(long)]
    pub protocol: String,
    /// Reviewer cohort label.
    #[arg(long)]
    pub cohort: String,
}

/// Shared execution arguments for all three offline comparison modes.
#[derive(Args, Debug, Clone)]
pub struct CompareRunsExecuteArgs {
    /// TOML comparison plan.
    #[arg(long)]
    pub plan: std::path::PathBuf,
    /// Bypass content-addressed pair caches and regenerate them atomically.
    #[arg(long)]
    pub recompute: bool,
}

/// Cache behavior for an offline comparison execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareRunsCachePolicy {
    /// Reuse an exact content-addressed pair result when available.
    Reuse,
    /// Recompute pair results even when an exact cache entry exists.
    Recompute,
}

impl CompareRunsExecuteArgs {
    /// Convert the CLI compatibility flag into a typed cache policy at the boundary.
    pub fn cache_policy(&self) -> CompareRunsCachePolicy {
        if self.recompute {
            CompareRunsCachePolicy::Recompute
        } else {
            CompareRunsCachePolicy::Reuse
        }
    }
}

// ---------------------------------------------------------------------------
// Utility commands
// ---------------------------------------------------------------------------

/// Arguments for the `serve` subcommand.
#[derive(Args, Debug, Clone)]
pub struct ServeArgs {
    /// Serve action (start, stop, status).
    #[command(subcommand)]
    pub action: ServeAction,
}

/// Server lifecycle actions.
#[derive(Subcommand, Debug, Clone)]
pub enum ServeAction {
    /// Start the processing server.
    Start(ServeStartArgs),
    /// Stop the processing server.
    Stop,
    /// Check server health and status.
    Status(ServeStatusArgs),
}

/// Arguments for `serve start`.
#[derive(Args, Debug, Clone)]
pub struct ServeStartArgs {
    /// Which handshake slot this server publishes to.
    ///
    /// Hidden: set by the CLI when it spawns a daemon, not by people. Two
    /// servers can run at once and each must publish its own record; nothing
    /// can infer which one this process is, so it is stated rather than
    /// guessed.
    #[arg(
        long,
        hide = true,
        value_enum,
        default_value = crate::server_handshake::HandshakeSlot::Main.as_arg(),
    )]
    pub handshake_slot: crate::server_handshake::HandshakeSlot,

    /// Port to listen on (defaults to server.yaml or 8000).
    #[arg(long)]
    pub port: Option<u16>,

    /// Host to bind to (defaults to server.yaml or 0.0.0.0).
    #[arg(long)]
    pub host: Option<String>,

    /// Path to server.yaml config file.
    #[arg(long)]
    pub config: Option<String>,

    /// Python executable used to spawn worker processes.
    #[arg(long, env = "BATCHALIGN_PYTHON")]
    pub python: Option<String>,

    /// Start workers in test-echo mode (debugging only; no ML models).
    #[arg(long)]
    pub test_echo: bool,

    /// Run in foreground (don't daemonize).
    #[arg(long)]
    pub foreground: bool,

    /// Maximum concurrent files per job. Overrides the `max_workers_per_job`
    /// value from `server.yaml`. 0 = auto-tune.
    #[arg(long)]
    pub workers: Option<usize>,

    /// Inference timeout in seconds for audio tasks (ASR, FA, speaker).
    /// Increase for very long recordings. Default: 1800 (30 minutes).
    #[arg(long)]
    pub timeout: Option<u64>,
}

/// Arguments for `serve status`.
#[derive(Args, Debug, Clone)]
pub struct ServeStatusArgs {
    /// Server URL to check.
    #[arg(long)]
    pub server: Option<String>,
}

/// Arguments for the `jobs` subcommand.
#[derive(Args, Debug, Clone)]
pub struct JobsArgs {
    /// Sub-action (e.g. `cancellations`). Backward-compat: no
    /// sub-action falls through to the original positional /
    /// flag-driven `list` / `show` behaviour.
    #[command(subcommand)]
    pub action: Option<JobsAction>,

    /// Job ID to inspect (legacy positional form, equivalent to
    /// `jobs show <id>`). Without `--server`, this inspects local
    /// job artifacts.
    pub job_id: Option<String>,

    /// Server URL (or set BATCHALIGN_SERVER env var) for remote job listing/detail.
    #[arg(long, env = "BATCHALIGN_SERVER")]
    pub server: Option<String>,

    /// Emit machine-readable JSON instead of the default human-readable summary.
    #[arg(long)]
    pub json: bool,
}

/// `jobs` sub-actions. New work nests here; legacy callers using
/// the positional `jobs <id>` form continue working through the
/// `action: None` fallback in `JobsArgs`.
#[derive(Subcommand, Debug, Clone)]
pub enum JobsAction {
    /// Print the cancellation audit history for one job. Use this
    /// when a user reports "I didn't cancel that job", every
    /// cancel attempt is recorded with `source` (tui / api /
    /// dashboard / staging / signal), `host`, `pid`, `reason`, and
    /// `in_flight_filename`.
    Cancellations(JobsCancellationsArgs),
}

/// Arguments for `jobs cancellations <id>`.
#[derive(Args, Debug, Clone)]
pub struct JobsCancellationsArgs {
    /// Job ID whose cancellation history should be printed.
    pub job_id: String,

    /// Server URL (or set BATCHALIGN_SERVER env var).
    #[arg(long, env = "BATCHALIGN_SERVER")]
    pub server: Option<String>,

    /// Emit machine-readable JSON instead of the default human-readable summary.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the `logs` subcommand.
#[derive(Args, Debug, Clone)]
pub struct LogsArgs {
    /// Show the most recent run log.
    #[arg(long)]
    pub last: bool,

    /// Show raw JSONL event lines (with --last).
    #[arg(long)]
    pub raw: bool,

    /// Export recent logs to a zip file.
    #[arg(long)]
    pub export: bool,

    /// Delete all log files.
    #[arg(long)]
    pub clear: bool,

    /// Live-tail the newest log file (Ctrl-C to stop).
    #[arg(long)]
    pub follow: bool,

    /// Number of recent runs to list.
    #[arg(short = 'n', long, default_value_t = 10)]
    pub count: usize,
}

/// Arguments for the `openapi` subcommand.
#[derive(Args, Debug, Clone)]
pub struct OpenapiArgs {
    /// Output path for OpenAPI JSON.
    ///
    /// In normal mode, if omitted, schema is written to stdout.
    /// In `--check` mode, if omitted, defaults to `openapi.json`.
    #[arg(short, long)]
    pub output: Option<String>,

    /// Verify that the target file already matches the generated schema.
    ///
    /// This mode does not modify files and exits non-zero on schema drift.
    #[arg(long)]
    pub check: bool,
}

/// Arguments for the `ipc-schema` subcommand.
#[derive(Args, Debug, Clone)]
pub struct IpcSchemaArgs {
    /// Output directory for JSON Schema files.
    ///
    /// If omitted, schemas are written to stdout as a single JSON object.
    #[arg(short, long)]
    pub output: Option<String>,

    /// Verify that the target directory already matches the generated schemas.
    ///
    /// This mode does not modify files and exits non-zero on schema drift.
    #[arg(long)]
    pub check: bool,
}

/// Arguments for the `cache` subcommand.
#[derive(Args, Debug, Clone)]
pub struct CacheArgs {
    /// Cache action (stats or clear).
    #[command(subcommand)]
    pub action: Option<CacheAction>,

    /// Show cache statistics (BA2-compatible flag form).
    #[arg(long)]
    pub stats: bool,

    /// Clear cache (BA2-compatible flag form).
    #[arg(long)]
    pub clear: bool,

    /// Also remove permanent UTR cache entries (with --clear).
    #[arg(long, requires = "clear")]
    pub all: bool,

    /// Skip confirmation prompt (with --clear).
    #[arg(short = 'y', long, requires = "clear")]
    pub yes: bool,
}

/// Cache management actions.
#[derive(Subcommand, Debug, Clone)]
pub enum CacheAction {
    /// Show cache statistics.
    Stats,
    /// Clear cached data.
    Clear(CacheClearArgs),
}

/// Arguments for `cache clear`.
#[derive(Args, Debug, Clone)]
pub struct CacheClearArgs {
    /// Also remove permanent UTR cache entries.
    #[arg(long)]
    pub all: bool,
    /// Skip confirmation prompt.
    #[arg(short = 'y', long)]
    pub yes: bool,
}

// ---------------------------------------------------------------------------
// Worker daemon management
// ---------------------------------------------------------------------------

/// Arguments for `batchalign3 worker`.
#[derive(Args, Debug, Clone)]
pub struct WorkerArgs {
    /// Worker action (start, list, stop).
    #[command(subcommand)]
    pub action: WorkerAction,
}

/// Worker management actions.
#[derive(Subcommand, Debug, Clone)]
pub enum WorkerAction {
    /// Start a worker as a foreground daemon.
    Start(WorkerStartArgs),
    /// List active workers from the registry.
    List,
    /// Stop one or all workers.
    Stop(WorkerStopArgs),
}

/// Arguments for `worker start`.
#[derive(Args, Debug, Clone)]
pub struct WorkerStartArgs {
    /// Worker profile: gpu, stanza, or io.
    #[arg(long)]
    pub profile: String,
    /// 3-letter ISO language code (e.g. eng, fra, yue).
    #[arg(long, default_value = "eng")]
    pub lang: String,
    /// TCP port to listen on (0 = auto-assign from 9100-9199).
    #[arg(long, default_value_t = 0)]
    pub port: u16,
    /// TCP bind address.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// Engine overrides as JSON (e.g. '{"asr":"tencent"}').
    #[arg(long, default_value = "")]
    pub engine_overrides: String,
}

/// Arguments for `worker stop`.
#[derive(Args, Debug, Clone)]
pub struct WorkerStopArgs {
    /// Stop the worker on this port.
    #[arg(long, default_value_t = 0)]
    pub port: u16,
    /// Stop all workers matching this profile.
    #[arg(long, default_value = "")]
    pub profile: String,
    /// Stop all workers matching this language.
    #[arg(long, default_value = "")]
    pub lang: String,
    /// Stop all registered workers.
    #[arg(long)]
    pub all: bool,
}

// ---------------------------------------------------------------------------
// Doctor command
// ---------------------------------------------------------------------------

/// Pre-flight diagnostic arguments.
#[derive(Args, Debug, Clone)]
pub struct DoctorArgs {
    /// Language to test (default: eng).
    #[arg(long, default_value = "eng")]
    pub lang: String,

    /// Output format.
    #[arg(long, default_value = "human")]
    pub format: DoctorFormat,

    /// Custom Python path (overrides BATCHALIGN_PYTHON).
    #[arg(long)]
    pub python: Option<String>,

    /// Skip the Python worker-pipeline checks; only inspect host
    /// facts and validate the deployed `server.yaml`. Fast (no
    /// Python spawn, no model load) and intended for operators
    /// verifying config sanity before deploying or restarting.
    /// Exits non-zero on host-facts validation errors.
    #[arg(long)]
    pub check: bool,

    /// Trace why one resolved knob has its current value. Prints the
    /// resolved value, whether it came from an operator override or
    /// the host-facts recommendation, the rule that produced the
    /// recommendation, and the relevant detected facts. Implies
    /// `--check` (skips worker pipeline). Valid knob names:
    /// `gpu_thread_pool_size`, `force_cpu`, `max_total_workers`,
    /// `max_concurrent_jobs`, `max_workers_per_key`,
    /// `memory_gate_mb`.
    #[arg(long, value_name = "KNOB")]
    pub explain: Option<String>,

    /// Treat host-facts validation warnings as fatal: exit non-zero
    /// when any warning fires, not only when an error fires.
    /// Intended for CI gates that want zero-warning deployments.
    /// Has no effect outside `--check` and `--explain` paths
    /// (the worker-pipeline path doesn't run host-facts validation
    /// today).
    #[arg(long)]
    pub warnings_as_errors: bool,
}

/// Doctor output format.
#[derive(Debug, Clone, clap::ValueEnum)]
pub enum DoctorFormat {
    /// Human-readable summary.
    Human,
    /// JSON output for machine consumption.
    Json,
}

// ---------------------------------------------------------------------------
// Replay command
// ---------------------------------------------------------------------------

/// Replay a captured failed IPC request.
#[derive(Args, Debug, Clone)]
pub struct ReplayArgs {
    /// Path to a failed_ipc_*.json dump file.
    pub dump_file: std::path::PathBuf,

    /// Language override (uses dump file's worker label if omitted).
    #[arg(long)]
    pub lang: Option<String>,

    /// Custom Python path (overrides BATCHALIGN_PYTHON).
    #[arg(long)]
    pub python: Option<String>,
}

// ---------------------------------------------------------------------------
// Eval subcommand tree
// ---------------------------------------------------------------------------

/// Arguments for `batchalign3 eval`.
#[derive(Args, Debug, Clone)]
pub struct EvalArgs {
    /// Evaluation action.
    #[command(subcommand)]
    pub action: EvalAction,
}

/// Evaluation actions: starts with `l2-morphotag`; more can land here.
#[derive(Subcommand, Debug, Clone)]
pub enum EvalAction {
    /// L2 morphotag evaluation: pair `@s` words with `%mor` / `%gra` items
    /// using a typed AST walk (supersedes `scripts/l2-eval/analyze.py`).
    #[command(name = "l2-morphotag")]
    L2Morphotag(L2MorphotagEvalArgs),
    /// Replay fingerprinted legacy projected ASR evidence through the current
    /// local transcribe post-processing pipeline without provider inference.
    #[command(name = "transcribe-replay")]
    TranscribeReplay(TranscribeReplayArgs),
    /// Replay global UTR word-to-token matching without inference or CHAT
    /// mutation, retaining exact match and timing-proposal evidence.
    #[command(name = "utr-alignment")]
    UtrAlignment(UtrAlignmentEvalArgs),
}

/// Arguments for `eval utr-alignment`.
#[derive(Args, Debug, Clone)]
pub struct UtrAlignmentEvalArgs {
    /// Exact CHAT document to align against retained UTR tokens.
    #[arg(long)]
    pub chat: std::path::PathBuf,
    /// Retained `_utr_tokens.json` artifact.
    #[arg(long)]
    pub tokens: std::path::PathBuf,
    /// Fresh JSON report destination; an existing path is refused.
    #[arg(long)]
    pub output: std::path::PathBuf,
    /// Jaro-Winkler threshold for fuzzy matching. Omit for case-insensitive
    /// exact matching.
    #[arg(long)]
    pub fuzzy_threshold: Option<UtrFuzzyThreshold>,
    /// Which utterances participate in the global alignment payload.
    #[arg(long, value_enum, default_value_t)]
    pub participation: UtrAlignmentParticipation,
}

/// Utterance population replayed by `eval utr-alignment`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum UtrAlignmentParticipation {
    /// Include every main-tier utterance in document order.
    #[default]
    AllUtterances,
    /// Exclude marked overlaps, matching the first pass of two-pass UTR.
    ExcludeMarkedOverlap,
}

/// Offline transcribe-replay utility family.
#[derive(Args, Debug, Clone)]
pub struct TranscribeReplayArgs {
    /// Manifest authoring or replay execution.
    #[command(subcommand)]
    pub action: TranscribeReplayAction,
}

/// Offline transcribe-replay actions.
#[derive(Subcommand, Debug, Clone)]
pub enum TranscribeReplayAction {
    /// Fingerprint one media/ASR/turns evidence set into an immutable manifest.
    Manifest(TranscribeReplayManifestArgs),
    /// Replay one or more admitted manifests with one warm worker pool.
    Run(TranscribeReplayRunArgs),
}

/// Arguments for `eval transcribe-replay manifest`.
#[derive(Args, Debug, Clone)]
pub struct TranscribeReplayManifestArgs {
    /// Stable recording identifier used for the output CHAT basename.
    #[arg(long)]
    pub recording_id: String,
    /// Exact source media corresponding to the retained ASR and turns.
    #[arg(long)]
    pub media: std::path::PathBuf,
    /// Retained BA3 projected `_asr_response.json` artifact.
    #[arg(long)]
    pub asr_response: std::path::PathBuf,
    /// Optional canonical BA3 speaker-turns JSON artifact.
    #[arg(long)]
    pub speaker_turns: Option<std::path::PathBuf>,
    /// Destination for the fingerprinted replay manifest.
    #[arg(long)]
    pub output: std::path::PathBuf,
}

/// Arguments for `eval transcribe-replay run`.
#[derive(Args, Debug, Clone)]
pub struct TranscribeReplayRunArgs {
    /// Fingerprinted replay manifests. All are admitted before models load.
    #[arg(required = true)]
    pub manifests: Vec<std::path::PathBuf>,
    /// Directory for replayed CHAT and run receipts.
    #[arg(short, long)]
    pub output: std::path::PathBuf,
    /// Resolved three-letter language code used by local NLP stages.
    #[arg(long, default_value = "eng")]
    pub lang: String,
    /// Expected speaker count used for CHAT participant capacity.
    #[arg(long, default_value_t = 2)]
    pub num_speakers: usize,
    /// Apply each manifest's canonical speaker turns before segmentation.
    #[arg(long)]
    pub diarize: bool,
    /// Skip the current utterance-segmentation model.
    #[arg(long)]
    pub no_utseg: bool,
    /// Closed local policy used to turn retained raw model actions into
    /// utterance assignments. The experimental choice repeats no inference.
    #[arg(long, value_enum, default_value_t)]
    pub utseg_policy: TranscribeReplayUtsegPolicy,
    /// Which transcribe segmentation pass or passes receive the selected
    /// policy. This is an offline evaluation control, not a production flag.
    #[arg(long, value_enum, default_value_t)]
    pub utseg_passes: TranscribeReplayUtsegPasses,
    /// Generate `%wor` tiers from retained ASR word timings.
    #[arg(long)]
    pub wor: bool,
    /// Opt in to the legacy Stanza fallback for unsupported utseg languages.
    #[arg(long)]
    pub utseg_fallback_stanza: bool,
    /// Optional directory for the current pipeline's debug evidence.
    #[arg(long)]
    pub debug_dir: Option<std::path::PathBuf>,
}

/// Utterance-boundary decision policy for a controlled offline replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum TranscribeReplayUtsegPolicy {
    /// Preserve the assignments declared by the worker.
    #[default]
    WorkerDeclared,
    /// Suppress only an earlier true boundary when two true boundaries are
    /// adjacent, preserving a boundary before a capitalized onset.
    SuppressEarlierAdjacentBoundariesV1,
    /// Apply the boundary-only adjacency policy while preserving exact
    /// repeated n-grams for CHAT retrace recognition.
    SuppressEarlierAdjacentBoundariesPreserveExactRetracesV1,
}

/// Explicit pass topology for controlled offline transcribe replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum TranscribeReplayUtsegPasses {
    /// Apply the selected policy before and after CHAT, matching production's
    /// current two-pass topology.
    #[default]
    Both,
    /// Apply the selected policy once before CHAT and omit the post-CHAT pass.
    PreChatOnly,
    /// Run both passes, apply the selected policy only before CHAT, and keep
    /// the worker-declared production policy after CHAT.
    PolicyOnPreChatOnly,
    /// Preserve the worker-declared production pre-CHAT pass and apply the
    /// selected policy only to the post-CHAT pass.
    PolicyOnPostChatOnly,
}

/// Arguments for `batchalign3 eval l2-morphotag`.
///
/// The typical workflow:
/// 1. Run `batchalign3 morphotag` (L2 dispatch is on by default) on the
///    eval set and place the output CHAT files in a directory.
/// 2. Run `batchalign3 eval l2-morphotag --eval-set eval-set.jsonl
///    --morphotag-output <dir> --output <report-dir>/`.
///
/// The eval set is a JSONL file with one `{ "path": ..., "pair_key": ... }`
/// object per line (produced by `scripts/l2-eval/select_eval_set.py`).
#[derive(Args, Debug, Clone)]
pub struct L2MorphotagEvalArgs {
    /// JSONL file listing input CHAT files with their `pair_key` labels.
    #[arg(long, value_name = "JSONL")]
    pub eval_set: std::path::PathBuf,

    /// Directory (flat or nested) of post-morphotag CHAT files.
    /// Matched against the eval set by filename basename.
    #[arg(long, value_name = "DIR")]
    pub morphotag_output: std::path::PathBuf,

    /// Directory to write `per-word.csv`, `per-pair.csv`,
    /// `flagged.csv`, and `summary.md`.
    #[arg(long, value_name = "DIR")]
    pub output: std::path::PathBuf,
}

/// Arguments for the `merge-verify` command.
#[derive(Args, Debug, Clone)]
pub struct MergeVerifyArgs {
    /// Merged draft directory (one `<session>.cha` per session named in
    /// the verdicts document).
    #[arg(long)]
    pub draft: std::path::PathBuf,

    /// Engine-verdicts JSON (per-line category, FA score, pitch band,
    /// machine-ear answer), produced by the verify engines or replayed
    /// from a cache.
    #[arg(long)]
    pub verdicts: std::path::PathBuf,

    /// Output directory for rewritten drafts and `review-queue.json`.
    #[arg(long)]
    pub out: std::path::PathBuf,

    /// Prefix identifying verify flags among `%com` tiers. The corpus
    /// seam owns the vocabulary; this pass only needs the prefix.
    #[arg(long, default_value = "verify")]
    pub flag_prefix: String,
}
