//! Engine backend types and traits.
//!
//! Closed enum sets for ASR, FA, and UTR engine selection.
//! No external plugin system, all engines are built-in.
//! The [`EngineBackend`] trait provides a common interface.

use batchalign_types::worker_v2::FaBackendV2;
use serde::{Deserialize, Serialize};

/// Shared behavior for all engine backend selectors.
///
/// Implement this on each engine enum so generic code can work across
/// engine categories without knowing which specific enum it holds.
pub trait EngineBackend: std::fmt::Debug + Clone + Send + Sync + 'static {
    /// Stable wire-format name used in JSON, CLI args, and SQLite.
    ///
    /// `&'static str`, not a borrow of `self`: every implementation returns a
    /// literal, and the narrower signature meant a category whose selection
    /// name IS its wire name could not say so without copying the table.
    fn wire_name(&self) -> &'static str;

    /// Whether this engine's inference is fully Rust-owned (no Python worker).
    fn is_rust_owned(&self) -> bool;

    /// Parse a wire-format name. Returns `None` for unrecognized names.
    fn try_from_wire_name(name: &str) -> Option<Self>
    where
        Self: Sized;
}

/// An engine category a user can choose from on the command line.
///
/// # Why this exists
///
/// There are four engine categories (ASR, UTR, FA, translate) and they had four
/// hand-written answers to the same five questions: which engines exist, what
/// is the default, what does a user type, what historical spellings still work,
/// and what does a name resolve to. Each answer was restated at the CLI, and
/// three of the four restatements had gone stale in the same direction: the
/// flag advertised a SUBSET, and the engines it left out were reachable only
/// through a second, differently-named flag taking an unvalidated string. That
/// is how the Cantonese engines stayed hidden from the users who needed them.
///
/// Fixing them one at a time did not work either: ASR was fixed first, and the
/// report that prompted this landed on UTR, one of the two still broken.
///
/// # What it buys
///
/// [`engine_selection_parser`] takes no free parameters. It previously took the
/// shown names, the hidden names, a resolver and a category string as four
/// unrelated arguments, which meant nothing stopped a caller pairing FA's names
/// with UTR's resolver and producing a flag whose help advertised three engines
/// it would then reject. Now the four facts travel together, from the type.
///
/// [`engine_selection_parser`]: crate::cli::args::commands
pub trait SelectableEngine: EngineBackend + Sized {
    /// What choosing a name yields.
    ///
    /// Usually `Self`. ASR is the exception: some of its names are an engine
    /// PLUS a checkpoint (`paraformer` is `funaudio` carrying `paraformer-zh`),
    /// so it selects an [`AsrSelection`] rather than a bare variant. Without
    /// this associated type the trait would either exclude ASR or force the
    /// other three into a wrapper they do not need.
    type Selected: Clone + Send + Sync + 'static;

    /// Every engine in this category, in help-display order.
    ///
    /// THE owner of "which engines exist" for the category.
    const ALL: &'static [Self];

    /// The engine used when the flag is omitted.
    const DEFAULT: Self;

    /// Human-readable category name, for diagnostics.
    ///
    /// What [`UnknownEngineName::category`] reports, via
    /// [`parse_wire_name`](Self::parse_wire_name), so each category string has
    /// one owner rather than one per error site.
    const CATEGORY: &'static str;

    /// The single name this engine is advertised under.
    ///
    /// Distinct from [`EngineBackend::wire_name`], which is persisted in JSON
    /// and SQLite and therefore cannot change. `--utr-engine tencent` beside
    /// `--asr-engine tencent` is one concept spelled one way, even though UTR's
    /// wire name is `tencent_utr`.
    fn selection_name(&self) -> &'static str;

    /// Every spelling accepted, canonical and historical, and what it means.
    ///
    /// COMPLETE: every canonical [`selection_name`](Self::selection_name)
    /// appears here, followed by that engine's historical spellings. The first
    /// draft of this trait let the table mean something different per category
    /// (UTR listed only wire names, FA listed some canonical names and not
    /// others), which forced three different resolvers and quietly weakened
    /// the coherence test to two of four categories.
    ///
    /// One table per category, read by the resolver, by
    /// [`try_from_wire_name`](EngineBackend::try_from_wire_name) and by the
    /// CLI's hidden list, so a spelling cannot be accepted by one and rejected
    /// by another. Clap rejects anything outside its list BEFORE the resolver
    /// runs, so a hidden list that falls short of the resolver silently breaks
    /// old command lines.
    fn accepted_names() -> &'static [(&'static str, Self)];

    /// Resolve a user-typed name to a variant.
    ///
    /// Provided, because with a complete table there is one answer. Three
    /// hand-written bodies preceded this, two of which did the same two lookups
    /// in opposite orders.
    fn resolve_variant(name: &str) -> Option<Self> {
        Self::accepted_names()
            .iter()
            .find(|(accepted, _)| *accepted == name)
            .map(|(_, engine)| engine.clone())
    }

    /// Resolve a user-typed name to whatever this category selects.
    ///
    /// Defaults to [`resolve_variant`](Self::resolve_variant). ASR overrides
    /// it, because a name there can carry a checkpoint as well as an engine.
    fn resolve(name: &str) -> Option<Self::Selected>;

    /// The names shown in `--help`.
    ///
    /// Overridable: ASR appends `paraformer`, which names a selection rather
    /// than a variant.
    fn selectable_names() -> impl Iterator<Item = &'static str> {
        Self::ALL.iter().map(Self::selection_name)
    }

    /// The names accepted but not advertised: everything that is not the one
    /// canonical selection name for its engine.
    ///
    /// Derived, so the hidden set cannot fall short of the resolver.
    fn hidden_alias_names() -> impl Iterator<Item = &'static str> {
        Self::accepted_names()
            .iter()
            .filter(|(name, engine)| *name != engine.selection_name())
            .map(|(name, _)| *name)
    }

    /// Parse a persisted wire-format token, reporting the category on failure.
    ///
    /// Each category had its own copy of this body and its own copy of the
    /// category literal, four of each. The literal now comes from
    /// [`CATEGORY`](Self::CATEGORY).
    fn parse_wire_name(name: &str) -> Result<Self, UnknownEngineName> {
        Self::try_from_wire_name(name).ok_or_else(|| UnknownEngineName {
            name: name.to_owned(),
            category: Self::CATEGORY,
        })
    }
}

impl SelectableEngine for AsrEngineName {
    /// ASR selects an [`AsrSelection`], not a bare variant: `paraformer` is
    /// `funaudio` plus a checkpoint, so a name is not always an engine.
    type Selected = AsrSelection;
    const ALL: &'static [Self] = &[
        Self::RevAi,
        Self::Whisper,
        Self::WhisperHub,
        Self::WhisperX,
        Self::WhisperOai,
        Self::WhisperRs,
        Self::HkTencent,
        Self::HkAliyun,
        Self::HkFunaudio,
        Self::HkQwen,
    ];
    const DEFAULT: Self = Self::RevAi;
    const CATEGORY: &'static str = "ASR";

    fn selection_name(&self) -> &'static str {
        // Identical to the wire name for this category, so there is one table.
        self.wire_name_const()
    }

    fn accepted_names() -> &'static [(&'static str, Self)] {
        Self::ACCEPTED_NAMES
    }

    fn resolve(name: &str) -> Option<AsrSelection> {
        AsrSelection::parse(name)
    }

    /// Overridden to append `paraformer`, which names a SELECTION rather than
    /// a variant and so cannot come from `ALL`.
    fn selectable_names() -> impl Iterator<Item = &'static str> {
        Self::ALL
            .iter()
            .map(|engine| engine.wire_name_const())
            .chain(std::iter::once(PARAFORMER_SELECTION_NAME))
    }
}

impl AsrEngineName {
    /// Every spelling accepted. One historical entry: `whisper-oai` was the
    /// CLI's spelling of the `whisper_oai` wire name, which deriving the value
    /// list from the enum immediately exposed as two public names for one
    /// engine.
    const ACCEPTED_NAMES: &'static [(&'static str, Self)] = &[
        ("rev", Self::RevAi),
        ("whisper", Self::Whisper),
        ("whisper_hub", Self::WhisperHub),
        ("whisperx", Self::WhisperX),
        ("whisper_oai", Self::WhisperOai),
        ("whisper-oai", Self::WhisperOai),
        ("whisper_rs", Self::WhisperRs),
        ("tencent", Self::HkTencent),
        ("aliyun", Self::HkAliyun),
        ("funaudio", Self::HkFunaudio),
        ("qwen", Self::HkQwen),
    ];
}

impl SelectableEngine for TranslateEngineName {
    type Selected = Self;
    const ALL: &'static [Self] = &[
        Self::Google,
        Self::Seamless,
        Self::Nllb,
        Self::Tencent,
        Self::Aliyun,
    ];
    const DEFAULT: Self = Self::Google;
    const CATEGORY: &'static str = "translate";

    fn selection_name(&self) -> &'static str {
        // Identical to the wire name for this category, so there is one table.
        self.wire_name()
    }

    fn accepted_names() -> &'static [(&'static str, Self)] {
        Self::ACCEPTED_NAMES
    }

    fn resolve(name: &str) -> Option<Self> {
        Self::resolve_variant(name)
    }
}

impl TranslateEngineName {
    /// Every spelling accepted. No historical aliases: the CLI names and the
    /// wire names were already identical, which is why this category's
    /// duplicate CLI enum stayed silently in agreement rather than drifting
    /// like the other three.
    const ACCEPTED_NAMES: &'static [(&'static str, Self)] = &[
        ("google", Self::Google),
        ("seamless", Self::Seamless),
        ("nllb", Self::Nllb),
        ("tencent", Self::Tencent),
        ("aliyun", Self::Aliyun),
    ];
}

/// Error returned when a wire-format engine name is not recognized.
#[derive(Debug, Clone, thiserror::Error)]
#[error("unknown engine name \"{name}\" for {category}")]
pub struct UnknownEngineName {
    /// The unrecognized wire name.
    pub name: String,
    /// Which engine category was being parsed (e.g. "ASR", "FA", "UTR").
    pub category: &'static str,
}

/// Error returned when a RECOGNIZED engine name selects an engine this build
/// does not implement.
///
/// Deliberately distinct from [`UnknownEngineName`], which is a name nothing
/// recognizes. This one parses, round-trips through the wire format, appears
/// in `--help`, and has no runtime behind it: `whisperx` and `whisper_oai`
/// are the standing cases, since nothing in this workspace implements either
/// WhisperX or the OpenAI Whisper API.
///
/// It exists because backend selection used to end in a catch-all arm that
/// mapped every unhandled name to stock local Whisper, so a job asking for
/// one of those two ran a different engine and wrote that other engine's name
/// into provenance. The catch-all is gone; the selector is now a total,
/// exhaustive match over [`AsrEngineName`] whose refusal case is this type,
/// so adding an engine variant without implementing it fails to compile
/// rather than silently resolving to Whisper.
///
/// The message names only the engine that was refused. The list of engines
/// that DO work is derived by the caller from the same selection function
/// that produced this error, never restated as prose, so a refusal cannot
/// recommend an engine that is itself unimplemented.
#[derive(Debug, Clone, thiserror::Error)]
#[error("ASR engine \"{}\" is accepted as a name but not implemented in this build", .engine.as_wire_name())]
pub struct EngineNotImplemented {
    /// The recognized but unimplemented engine.
    pub engine: AsrEngineName,
}

/// Typed UTR engine selector.
///
/// The wire format still uses the legacy string tokens (`"rev_utr"`,
/// `"whisper_utr"`, or a plugin-provided name), but the server runtime works
/// with this enum so the control plane stops branching on anonymous strings.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UtrEngine {
    /// Rust-owned Rev.AI timed-word path.
    RevAi,
    /// Python-worker ASR path with the built-in Whisper profile.
    Whisper,
    /// Tencent UTR (HK/Cantonese).
    HkTencent,
}

impl EngineBackend for UtrEngine {
    fn wire_name(&self) -> &'static str {
        match self {
            Self::RevAi => "rev_utr",
            Self::Whisper => "whisper_utr",
            Self::HkTencent => "tencent_utr",
        }
    }

    fn is_rust_owned(&self) -> bool {
        matches!(self, Self::RevAi)
    }

    fn try_from_wire_name(name: &str) -> Option<Self> {
        // The one table, so a spelling cannot be accepted here and rejected by
        // the CLI resolver, or the reverse.
        Self::resolve_variant(name)
    }
}

impl SelectableEngine for UtrEngine {
    type Selected = Self;
    const ALL: &'static [Self] = &[Self::RevAi, Self::Whisper, Self::HkTencent];
    const DEFAULT: Self = Self::RevAi;
    const CATEGORY: &'static str = "UTR";

    fn selection_name(&self) -> &'static str {
        match self {
            Self::RevAi => "rev",
            Self::Whisper => "whisper",
            Self::HkTencent => "tencent",
        }
    }

    fn accepted_names() -> &'static [(&'static str, Self)] {
        Self::ACCEPTED_NAMES
    }

    fn resolve(name: &str) -> Option<Self> {
        Self::resolve_variant(name)
    }
}

impl UtrEngine {
    /// Canonical names first, then the wire names, which are the historical
    /// half: persisted in JSON and SQLite so they cannot change, but also not
    /// what a user should have to type.
    const ACCEPTED_NAMES: &'static [(&'static str, Self)] = &[
        ("rev", Self::RevAi),
        ("whisper", Self::Whisper),
        ("tencent", Self::HkTencent),
        ("rev_utr", Self::RevAi),
        ("whisper_utr", Self::Whisper),
        ("tencent_utr", Self::HkTencent),
    ];

    /// Parse one persisted wire-format token.
    ///
    /// Forwards to [`SelectableEngine::parse_wire_name`], which owns both
    /// this body and the category name. All four categories had their own
    /// copy of each.
    pub fn from_wire_name(name: &str) -> Result<Self, UnknownEngineName> {
        <Self as SelectableEngine>::parse_wire_name(name)
    }

    /// Borrow the wire-format token for JSON/SQLite.
    ///
    /// Returns `&'static str`, not `&str`: every wire name is a compile-time
    /// literal and `wire_name` already says so, so the elided lifetime here was
    /// narrowing a fact the callee had established. That cost real work, since a
    /// caller wanting to store the name (an `EngineId` on a per-word provenance
    /// value) had to allocate a `String` to escape a borrow that never needed to
    /// exist.
    pub fn as_wire_name(&self) -> &'static str {
        self.wire_name()
    }

    /// Whether the current engine can reuse the worker-side segment strategy
    /// for partial-window UTR.
    pub fn supports_partial_windows(&self) -> bool {
        !self.is_rust_owned()
    }
}

impl Serialize for UtrEngine {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_wire_name())
    }
}

impl<'de> Deserialize<'de> for UtrEngine {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Self::from_wire_name(&name).map_err(serde::de::Error::custom)
    }
}

/// Typed forced-alignment engine selector.
///
/// The wire format still uses the legacy string tokens (`"wav2vec_fa"`,
/// `"whisper_fa"`, or a plugin-provided name), but the control plane works
/// with this enum so dispatch does not branch on anonymous strings.
///
/// Everything an engine IS lives in [`FA_ENGINES`], one row per variant,
/// reached through [`FaEngineName::spec`]. Adding a variant is a compile error
/// in exactly one place, the pairing list passed to [`fa_engine_table`], which
/// is also what builds the table; adding a FIELD is a compile error in every
/// row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FaEngineName {
    /// MMS Wave2Vec forced alignment.
    Wave2Vec,
    /// Whisper token-timestamp forced alignment.
    Whisper,
    /// Wav2Vec Cantonese forced alignment (HK).
    Wav2vecCanto,
    /// Qwen3 forced alignment: the standalone form of the aligner the
    /// Qwen3-ASR engine already loads for its own word timestamps
    /// (`Qwen/Qwen3-ForcedAligner-0.6B-hf`). Unlike the other three it is
    /// NOT language-general; see its row in [`FA_ENGINES`].
    Qwen3,
}

/// What an alignment engine can say about a word's extent.
///
/// The distinction decides whether a `%wor` tier carries measured durations or
/// derived ones, so it belongs to the engine rather than to any consumer. It
/// had previously been restated in prose in several places, in one case
/// inverted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaTimingResolution {
    /// Reports a word's start AND its end; durations are measured.
    WordIntervals,
    /// Reports only when a word STARTS; an end must be derived from the next
    /// onset, and a consumer that skips that step produces zero-duration words.
    TokenOnsets,
}

/// Which languages an engine can be asked for.
///
/// One field instead of the two functions this replaces: a `FaLanguageScope`
/// enum that named the distinction, and a per-engine code list consulted only
/// in the narrow case. The scope WAS the shape of the list, so the shape is
/// now the variant and there is nothing left to keep in step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguageSupport {
    /// The engine aligns against its own label set and never sees the
    /// `@Languages:` header, so no declared code can make it wrong.
    Any,
    /// The engine is handed ONE language label for a whole group, so every
    /// language whose words can appear in a group must be one of these
    /// ISO-639-3 codes.
    Only(&'static [&'static str]),
}

impl LanguageSupport {
    /// Whether one ISO-639-3 code is covered.
    pub fn covers(self, code: &str) -> bool {
        match self {
            Self::Any => true,
            Self::Only(codes) => codes.contains(&code),
        }
    }

    /// Whether the engine reads the language declaration at all.
    ///
    /// The question admission asks first: a general engine can skip the whole
    /// per-entry walk, including entries that do not parse.
    ///
    /// `const` so the fallback-target check below can run at compile time
    /// rather than as a test nobody would write.
    pub const fn is_language_general(self) -> bool {
        matches!(self, Self::Any)
    }
}

/// What happens to an FA group this engine failed on.
///
/// A policy, not a fact about the model, which is why it is a named variant
/// rather than a boolean: "we do not retry" and "we cannot retry" read the
/// same from a call site and mean different things to whoever adds the next
/// engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaFallbackPolicy {
    /// A recoverable CTC-decoder constraint (target too long, blank index in
    /// the targets, a window shorter than the feature extractor's receptive
    /// field) retries the whole group on the named engine.
    ///
    /// The target is carried rather than baked into the variant name, because
    /// the retry does NOT go through [`validate_fa_language_support`]: the
    /// group is re-dispatched on the target directly, so an engine admitted
    /// for a file's languages can hand its work to one that never was. Naming
    /// the target here makes that engine a value the compiler can check, and
    /// the check is the `const` block beside [`FA_ENGINES`], which refuses a
    /// target that is the failing engine itself, one that is not
    /// [`LanguageSupport::Any`], and one whose
    /// [`FaTimingResolution`] the evidence admission cannot read as a
    /// fallback. Each refusal and why it exists is written out there.
    ///
    /// [`validate_fa_language_support`]: crate::types::request
    RetryGroupOn(FaEngineName),
    /// The group is left unaligned. Either the engine IS a fallback target,
    /// or retrying would silently substitute another model's timings.
    NoFallback,
}

/// Everything one forced-alignment engine is, stated once.
///
/// # Why this exists
///
/// Adding `Qwen3` cost one enum variant and then fifteen separate per-engine
/// edits spread over five files and two crates: seven `match`es in this module
/// alone, the wire-backend enum next door, the two direction maps, the
/// fallback policy, and the language scope plus its code list. Every one of
/// them was exhaustive, so the compiler did demand an answer; it demanded them
/// one file at a time, days apart, with no single place that says what an
/// engine IS.
///
/// # What is compiler-enforced, exactly
///
/// A new engine must be unable to compile until its author has stated every
/// fact, and each half of that is enforced by a different mechanism:
///
/// - **A new FIELD breaks every row.** This struct has every field required
///   and **no `Default` impl**, so adding one is a compile error at each of
///   the row constants until each states it. No plausible-looking blank is
///   available.
/// - **A new VARIANT breaks the pairing list.** [`FaEngineName::spec`] is
///   generated by [`fa_engine_table`] from the same one-line-per-engine list
///   that builds [`FA_ENGINES`], and its match is exhaustive over
///   [`FaEngineName`], so a variant absent from that list does not compile and
///   a variant present in it is necessarily present in the table. This was NOT
///   true until 2026-09-07: `spec()` was a hand-written match returning
///   standalone row constants, so a variant could compile with a row the table
///   never saw, and every derivation from the table would then omit it.
///
/// What is NOT enforced by a type, and is checked by the `const` block beside
/// [`FA_ENGINES`] instead: that no engine has two rows, that no spelling is
/// accepted by two rows, and that every fallback target is language-general.
#[derive(Debug, PartialEq, Eq)]
pub struct FaEngineSpec {
    /// The variant this row describes. Read by the derivations below, and by
    /// the test that pins each row to the variant whose `spec()` returns it.
    pub engine: FaEngineName,

    /// Stable wire-format token, persisted in JSON and SQLite and therefore
    /// unchangeable.
    pub wire_name: &'static str,

    /// The single name this engine is advertised under in `--help`.
    pub selection_name: &'static str,

    /// Historical spellings that still resolve, NOT including
    /// [`Self::selection_name`], which is added by the derivation.
    ///
    /// The wire name belongs here whenever it differs from the selection name,
    /// since a persisted job option must still parse.
    pub aliases: &'static [&'static str],

    /// The name the Python worker sees in its `--engine-overrides` JSON and
    /// uses to select which FA model to load.
    ///
    /// Must match `resolve_fa_engine` on the Python side.
    pub dispatch_override_name: &'static str,

    /// What this engine reports about a word's extent.
    pub timing_resolution: FaTimingResolution,

    /// Longest audio window handed to this engine in one dispatch.
    pub max_group: batchalign_types::domain::DurationMs,

    /// Resident memory footprint estimate for one worker process running this
    /// engine, in MB. Feeds the admission gate's engine-aware reservation, see
    /// [`super::super::worker::pool::memory_gate::engine_aware_startup_reservation_mb`].
    pub resident_memory_mb: u64,

    /// Which languages this engine can be asked for.
    pub language_support: LanguageSupport,

    /// What happens to a group this engine failed on.
    pub fallback: FaFallbackPolicy,

    /// The backend token this engine travels to the worker as.
    pub worker_backend: FaBackendV2,
}

/// The ISO-639-3 codes the Qwen3 forced aligner is wired for.
///
/// MUST stay in step with `QWEN_LANG_LABELS` in
/// `batchalign/inference/qwen_forced_alignment.py`, which is the only producer
/// of the language label the model is handed. The Python side raises on a code
/// it cannot map, so a drift between the two lists is a loud worker failure
/// rather than a wrong alignment; this list exists so the refusal happens at
/// admission instead, before a model is downloaded.
///
/// Narrower than the checkpoint's own advertised set (which also covers
/// French, German, Italian, Japanese, Korean, Portuguese, Russian and
/// Spanish). Widening it is a measurement, and Japanese and Korean
/// additionally need optional tokenizer packages, so neither list guesses.
const FA_QWEN3_LANGUAGES: &[&str] = &["yue", "zho", "cmn", "eng"];

/// The MMS Wave2Vec aligner.
const WAVE2VEC_FA_SPEC: FaEngineSpec = FaEngineSpec {
    engine: FaEngineName::Wave2Vec,
    wire_name: "wav2vec_fa",
    selection_name: "wav2vec",
    aliases: &["wav2vec_fa", "wave2vec"],
    dispatch_override_name: "wave2vec",
    timing_resolution: FaTimingResolution::WordIntervals,
    // The CTC decoder's target length grows with the audio window, so the
    // wav2vec family takes the shorter one.
    max_group: batchalign_types::domain::DurationMs(15_000),
    // MMS / torchaudio Wave2Vec FA models: ~1.2 GB + runtime margin.
    resident_memory_mb: WAVE2VEC_FA_RSS_MB,
    // Aligns against its own label set and never sees the declaration.
    language_support: LanguageSupport::Any,
    fallback: FaFallbackPolicy::RetryGroupOn(FaEngineName::Whisper),
    worker_backend: FaBackendV2::Wave2vec,
};

/// Whisper token-timestamp alignment, which is also the fallback target.
const WHISPER_FA_SPEC: FaEngineSpec = FaEngineSpec {
    engine: FaEngineName::Whisper,
    wire_name: "whisper_fa",
    selection_name: "whisper",
    aliases: &["whisper_fa"],
    dispatch_override_name: "whisper",
    timing_resolution: FaTimingResolution::TokenOnsets,
    max_group: batchalign_types::domain::DurationMs(20_000),
    // Whisper-large-v2 FA: ~3 GB weights + tokenizer + Python runtime. Same
    // shape as Whisper-large-v3 ASR, hence the shared constant.
    resident_memory_mb: WHISPER_LARGE_V3_RSS_MB,
    // Language-general in the same way its ASR is.
    language_support: LanguageSupport::Any,
    // It is the fallback target and cannot fall back to itself.
    fallback: FaFallbackPolicy::NoFallback,
    worker_backend: FaBackendV2::Whisper,
};

/// The Cantonese Wave2Vec aligner.
const WAV2VEC_CANTO_FA_SPEC: FaEngineSpec = FaEngineSpec {
    engine: FaEngineName::Wav2vecCanto,
    wire_name: "cantonese_fa",
    selection_name: "cantonese",
    // This engine answered to three spellings before it had a canonical name,
    // and the book used a fourth.
    aliases: &["cantonese_fa", "wav2vec_canto", "wav2vec_fa_canto"],
    dispatch_override_name: "wav2vec_canto",
    timing_resolution: FaTimingResolution::WordIntervals,
    max_group: batchalign_types::domain::DurationMs(15_000),
    // Same model shape as the general wave2vec aligner.
    resident_memory_mb: WAVE2VEC_FA_RSS_MB,
    // It only adds a romanization step, which is itself gated on `yue`.
    language_support: LanguageSupport::Any,
    fallback: FaFallbackPolicy::RetryGroupOn(FaEngineName::Whisper),
    worker_backend: FaBackendV2::Wav2vecCanto,
};

/// The Qwen3 forced aligner.
const QWEN3_FA_SPEC: FaEngineSpec = FaEngineSpec {
    engine: FaEngineName::Qwen3,
    wire_name: "qwen3_fa",
    // `qwen3_fa` is canonical, so it is also the selection name. The other two
    // spellings are conveniences and NOT a house style: no other FA engine
    // carries a hyphenated alias, and the underscore spellings elsewhere in
    // this table are historical rather than parallel. Read the rows, never
    // generalise from one of them.
    selection_name: "qwen3_fa",
    aliases: &["qwen3-fa", "qwen3"],
    dispatch_override_name: "qwen3_fa",
    // The aligner returns `start_time` AND `end_time` per word
    // (`decode_forced_alignment`), so its durations are measured, not derived
    // from the next onset.
    timing_resolution: FaTimingResolution::WordIntervals,
    // The Qwen aligner is an encoder over the whole window with no CTC
    // target-length limit, and its ASR sibling already runs 180 s chunks. It
    // takes the wav2vec window anyway: an FA group is a grouping decision
    // about the transcript, not about the model, and a wider window here would
    // change which words are grouped together without any measurement saying
    // it should.
    max_group: batchalign_types::domain::DurationMs(15_000),
    // Qwen3-ForcedAligner-0.6B: ~2.4 GB of float32 weights plus the processor
    // and the Python runtime.
    resident_memory_mb: QWEN3_FA_RSS_MB,
    // Handed ONE language label for a whole group.
    language_support: LanguageSupport::Only(FA_QWEN3_LANGUAGES),
    // No fallback, and that is a decision rather than an omission. Two halves:
    //
    // 1. None of the recoverable CTC failures is reachable for it. Its aligner
    //    is a transformer encoder scored per token, not a CTC decoder, so
    //    there is no target-length limit and no blank index, and its feature
    //    extractor pads short windows.
    // 2. Its own characteristic failure, a word its tokenizer keeps no
    //    character of, is not a group-level failure at all: the Python host
    //    folds the aligner's units back onto the requested words and returns
    //    that word UNTIMED, timing the rest. Retrying the whole group on
    //    Whisper would replace measured Qwen3 timings with Whisper ones for
    //    every word that DID align, and the wire records timings per word
    //    without recording which engine produced each, so the substitution
    //    would be invisible afterwards. One untimed word is the smaller and
    //    the honest loss.
    fallback: FaFallbackPolicy::NoFallback,
    worker_backend: FaBackendV2::Qwen3,
};

/// Pair every [`FaEngineName`] variant with its row, once.
///
/// # Why a macro, when nothing else in this file needs one
///
/// The table only enforces what something reads it. Before this macro,
/// [`FA_ENGINES`] was a hand-written list and `spec()` was a hand-written
/// exhaustive match returning standalone row constants, so a new variant
/// compiled perfectly well by returning a new constant nobody added to the
/// list. Everything derived FROM the list (`ALL`, the accepted-name table, the
/// language-general remedy line) then silently omitted that engine: it could
/// not be selected by name, its persisted wire name failed to parse, and it
/// appeared in no diagnostic. The test that was supposed to catch this
/// iterated `FaEngineName::ALL`, which IS the list, so the missing variant was
/// invisible to it.
///
/// Rust cannot enumerate an enum's variants without a macro or a derive. Of
/// the three available routes this is the one that keeps the enum readable:
/// the variants and their prose stay hand-written above, each row stays a
/// named constant with its own reasoning, and only the one-line-per-engine
/// PAIRING lives here. The alternatives were a macro that also emits the enum
/// (airtight, but the reader can no longer see the enum), or `strum::EnumIter`
/// (a compiler-generated witness, but a new direct dependency, and it yields a
/// runtime iteration rather than a compile error).
///
/// # What it enforces
///
/// The match it generates is exhaustive over [`FaEngineName`], so a variant
/// missing from this list does not compile; and the list it generates is the
/// same list, so a variant present in the match is necessarily present in
/// [`FA_ENGINES`]. The `const` block additionally pins each row's own
/// [`FaEngineSpec::engine`] field to the variant it was paired with, so a row
/// cannot be attached to the wrong engine.
macro_rules! fa_engine_table {
    ($($variant:ident => $spec:ident),+ $(,)?) => {
        /// Every forced-alignment engine, in help-display order.
        ///
        /// THE declaration. `ALL`, the accepted-name table, and the
        /// language-general remedy list are all DERIVED from it below rather
        /// than restated.
        pub const FA_ENGINES: &[FaEngineSpec] = &[$($spec),+];

        impl FaEngineName {
            /// This engine's row in [`FA_ENGINES`].
            ///
            /// THE one exhaustive match over the FA roster, generated from the
            /// same list that builds the table, so the row it returns is
            /// necessarily a row of the table. Every other per-engine answer
            /// in this workspace reads a field off what this returns.
            pub const fn spec(self) -> &'static FaEngineSpec {
                match self {
                    $(Self::$variant => &$spec,)+
                }
            }
        }

        /// Each row names the variant it was paired with.
        const _: () = {
            $(
                assert!(
                    matches!($spec.engine, FaEngineName::$variant),
                    concat!(
                        "the FA row paired with FaEngineName::",
                        stringify!($variant),
                        " declares a different engine in its own `engine` field",
                    ),
                );
            )+
        };
    };
}

fa_engine_table! {
    Wave2Vec => WAVE2VEC_FA_SPEC,
    Whisper => WHISPER_FA_SPEC,
    Wav2vecCanto => WAV2VEC_CANTO_FA_SPEC,
    Qwen3 => QWEN3_FA_SPEC,
}

/// How many engines the table declares.
const FA_ENGINE_COUNT: usize = FA_ENGINES.len();

/// `ALL` derived from the table, so the two cannot disagree.
///
/// A `const fn` rather than a second hand-written list: the previous `ALL` was
/// the third place a new variant had to be named, and the coherence test that
/// catches an omission is a runtime check for something that can be a
/// derivation.
const fn engines_from_table() -> [FaEngineName; FA_ENGINE_COUNT] {
    // Every slot is overwritten by the loop; the fill value is never observed,
    // and the table is non-empty by construction (an empty one would fail to
    // index here, at compile time).
    let mut out = [FA_ENGINES[0].engine; FA_ENGINE_COUNT];
    let mut i = 0;
    while i < FA_ENGINE_COUNT {
        out[i] = FA_ENGINES[i].engine;
        i += 1;
    }
    out
}

/// Backing storage for [`SelectableEngine::ALL`].
const FA_ENGINE_ALL: [FaEngineName; FA_ENGINE_COUNT] = engines_from_table();

/// How many accepted spellings the table declares: one canonical selection
/// name per engine, plus that engine's aliases.
const fn accepted_name_count() -> usize {
    let mut count = 0;
    let mut i = 0;
    while i < FA_ENGINE_COUNT {
        count += 1 + FA_ENGINES[i].aliases.len();
        i += 1;
    }
    count
}

/// The size of the derived accepted-name table.
const FA_ACCEPTED_NAME_COUNT: usize = accepted_name_count();

/// The accepted-name table derived from the rows.
///
/// Canonical name first for each engine, then its aliases. `SelectableEngine`
/// requires this to be COMPLETE (every canonical name present), which a
/// derivation cannot get wrong and a hand-written list did get wrong in three
/// of the four categories.
const fn accepted_names_from_table() -> [(&'static str, FaEngineName); FA_ACCEPTED_NAME_COUNT] {
    // As above: every slot is overwritten before it is read.
    let mut out = [("", FA_ENGINES[0].engine); FA_ACCEPTED_NAME_COUNT];
    let mut out_index = 0;
    let mut i = 0;
    while i < FA_ENGINE_COUNT {
        let spec = &FA_ENGINES[i];
        out[out_index] = (spec.selection_name, spec.engine);
        out_index += 1;
        let mut alias = 0;
        while alias < spec.aliases.len() {
            out[out_index] = (spec.aliases[alias], spec.engine);
            out_index += 1;
            alias += 1;
        }
        i += 1;
    }
    out
}

/// Backing storage for [`SelectableEngine::accepted_names`].
const FA_ACCEPTED_NAMES: [(&str, FaEngineName); FA_ACCEPTED_NAME_COUNT] =
    accepted_names_from_table();

/// `str` equality in a `const` context, which [`str::eq`] is not.
const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Facts about the FA table that a reader would otherwise have to hold in
/// their head, checked while the crate compiles.
///
/// A `const` block rather than a test, because each is a property of the
/// DECLARATION and has no input: there is nothing to arrange, nothing to run,
/// and a test would only report at `cargo test` what the compiler can refuse
/// outright.
///
/// Engine identity is compared here as `engine as u8`, the discriminant, not
/// with `==`: [`FaEngineName`] derives [`PartialEq`], but `PartialEq::eq` is
/// not a `const fn`, so it is unavailable in this context. The enum is
/// fieldless, so the discriminant IS the identity and the cast is exact.
///
/// 1. **No engine has two rows.** The macro's match would merely warn about an
///    unreachable arm, while `ALL` and the accepted-name table would carry the
///    engine twice.
/// 2. **No spelling is accepted by two rows.**
///    [`SelectableEngine::resolve_variant`] is FIRST MATCH, so a duplicate
///    silently belongs to whichever row is listed earlier: adding `"whisper"`
///    to the Qwen3 row's aliases would change what `--fa-engine whisper`
///    means, with every existing test still green.
/// 3. **No engine is its own fallback target.** A retry re-dispatches the same
///    group to the same model, which fails the same way, so the row is a
///    guaranteed waste of one worker call and a whole audio window.
/// 4. **Every fallback target is language-general.** A retry is re-dispatched
///    on the target WITHOUT a second pass through
///    `validate_fa_language_support`, so a language-restricted target would
///    align a file with an engine nobody admitted for its language. Today that
///    holds by accident, because the only target is Whisper and Whisper is
///    [`LanguageSupport::Any`]; here it holds because it is checked.
/// 5. **Every fallback target reports TOKEN ONSETS.** The policy generalised
///    to "retry on the engine this row names" while the evidence admission it
///    feeds stayed specific to one response SHAPE, and this is where the two
///    are tied back together. See the assertion's own message.
const _: () = {
    let mut i = 0;
    while i < FA_ENGINE_COUNT {
        let mut j = i + 1;
        while j < FA_ENGINE_COUNT {
            assert!(
                FA_ENGINES[i].engine as u8 != FA_ENGINES[j].engine as u8,
                "two rows of FA_ENGINES declare the same engine",
            );
            j += 1;
        }
        match FA_ENGINES[i].fallback {
            FaFallbackPolicy::RetryGroupOn(target) => {
                assert!(
                    FA_ENGINES[i].engine as u8 != target as u8,
                    "an FA engine must not name itself as its own retry target: the retry \
                     re-dispatches the identical group to the identical model, so it fails \
                     the identical way, and the evidence admission would then see an \
                     effective engine equal to the requested one and refuse the response as \
                     an ineffective fallback",
                );
                assert!(
                    target.spec().language_support.is_language_general(),
                    "an FA fallback target must be language-general: the retry does not \
                     re-run language admission, so a restricted target would align a file \
                     with an engine nobody admitted for its language",
                );
                assert!(
                    matches!(
                        target.spec().timing_resolution,
                        FaTimingResolution::TokenOnsets,
                    ),
                    "an FA fallback target must report TOKEN ONSETS, because the fallback \
                     POLICY and the evidence ADMISSION are one mechanism split across two \
                     files. `FaRawEvidence::admit_requested` recognises a fallback only by \
                     the response SHAPE: a token-onset payload proves an effective engine \
                     other than the one requested, while an interval payload is attributed \
                     to the requested engine itself. Retrying on an interval engine would \
                     therefore make a SUCCESSFUL retry indistinguishable from no fallback \
                     at all, and the Fallback route would refuse it as ineffective and \
                     throw its timings away",
                );
            }
            FaFallbackPolicy::NoFallback => {}
        }
        i += 1;
    }

    let mut i = 0;
    while i < FA_ACCEPTED_NAME_COUNT {
        let mut j = i + 1;
        while j < FA_ACCEPTED_NAME_COUNT {
            assert!(
                !str_eq(FA_ACCEPTED_NAMES[i].0, FA_ACCEPTED_NAMES[j].0),
                "two FA engine rows accept the same spelling; the resolver is first \
                 match, so the later row would never receive it",
            );
            j += 1;
        }
        i += 1;
    }
};

impl FaEngineName {
    /// The engine behind one wire backend token.
    ///
    /// The one match this table could NOT absorb, and deliberately so: it is
    /// exhaustive over [`FaBackendV2`], which is a different enum in a
    /// different crate, and a new backend on the wire is its own obligation to
    /// route. The forward direction is [`FaEngineSpec::worker_backend`]; the
    /// two are pinned as a bijection by
    /// `fa_engine_and_worker_backend_are_a_bijection`.
    pub const fn from_worker_backend(backend: FaBackendV2) -> Self {
        match backend {
            FaBackendV2::Whisper => Self::Whisper,
            FaBackendV2::Wave2vec => Self::Wave2Vec,
            FaBackendV2::Wav2vecCanto => Self::Wav2vecCanto,
            FaBackendV2::Qwen3 => Self::Qwen3,
        }
    }

    /// What this engine reports about a word's extent.
    pub fn timing_resolution(&self) -> FaTimingResolution {
        self.spec().timing_resolution
    }

    /// Longest audio window handed to this engine in one dispatch.
    pub fn max_group_ms(&self) -> batchalign_types::domain::DurationMs {
        self.spec().max_group
    }

    /// Which languages this engine can be asked for.
    pub fn language_support(&self) -> LanguageSupport {
        self.spec().language_support
    }

    /// What happens to a group this engine failed on.
    pub fn fallback_policy(&self) -> FaFallbackPolicy {
        self.spec().fallback
    }

    /// The backend token this engine travels to the worker as.
    pub fn worker_backend(&self) -> FaBackendV2 {
        self.spec().worker_backend
    }
}

impl EngineBackend for FaEngineName {
    fn wire_name(&self) -> &'static str {
        self.spec().wire_name
    }

    fn is_rust_owned(&self) -> bool {
        // Not a per-engine fact: no FA engine runs in-process today, so there
        // is nothing for the table to state.
        false
    }

    fn try_from_wire_name(name: &str) -> Option<Self> {
        Self::resolve_variant(name)
    }
}

impl SelectableEngine for FaEngineName {
    type Selected = Self;
    const ALL: &'static [Self] = &FA_ENGINE_ALL;
    // Wave2Vec returns word-level start AND end; Whisper FA returns token
    // onsets only, so an end has to be derived from the next onset and the
    // last word of a group has none to derive from. Measured beats derived,
    // which is why the default is the engine that measures. Pinned by
    // `default_fa_engine_reports_word_intervals`, which asserts the PROPERTY
    // rather than the variant.
    const DEFAULT: Self = Self::Wave2Vec;
    const CATEGORY: &'static str = "FA";

    fn selection_name(&self) -> &'static str {
        self.spec().selection_name
    }

    fn accepted_names() -> &'static [(&'static str, Self)] {
        &FA_ACCEPTED_NAMES
    }

    fn resolve(name: &str) -> Option<Self> {
        Self::resolve_variant(name)
    }
}

impl FaEngineName {
    /// The override name used in worker pool keys for dispatch.
    ///
    /// Reads [`FaEngineSpec::dispatch_override_name`], which carries the
    /// contract. The doc here used to name `fa_backend_override_name()` in
    /// `worker/pool/execute_v2.rs` as the function that must agree; no such
    /// function exists any more, and the party that has to agree is
    /// `resolve_fa_engine` in
    /// `batchalign/worker/_model_loading/forced_alignment.py`.
    pub fn dispatch_override_name(&self) -> &'static str {
        self.spec().dispatch_override_name
    }

    /// Parse one persisted wire-format token.
    ///
    /// Forwards to [`SelectableEngine::parse_wire_name`], which owns both
    /// this body and the category name. All four categories had their own
    /// copy of each.
    pub fn from_wire_name(name: &str) -> Result<Self, UnknownEngineName> {
        <Self as SelectableEngine>::parse_wire_name(name)
    }

    /// Borrow the wire-format token for JSON/SQLite.
    ///
    /// Returns `&'static str`, not `&str`: every wire name is a compile-time
    /// literal and `wire_name` already says so, so the elided lifetime here was
    /// narrowing a fact the callee had established. That cost real work, since a
    /// caller wanting to store the name (an `EngineId` on a per-word provenance
    /// value) had to allocate a `String` to escape a borrow that never needed to
    /// exist.
    pub fn as_wire_name(&self) -> &'static str {
        self.wire_name()
    }

    /// Resident memory footprint estimate for one worker process running
    /// this FA engine, in MB. Used by the admission gate to reserve enough
    /// headroom for engines whose actual RSS exceeds the default GPU-profile
    /// reservation (``tier.gpu_startup_mb``: 6 GB Small / 3 GB Medium /
    /// 16 GB Large+Fleet). See
    /// [`super::super::worker::pool::memory_gate::engine_aware_startup_reservation_mb`].
    pub fn resident_memory_mb(&self) -> u64 {
        self.spec().resident_memory_mb
    }
}

impl Serialize for FaEngineName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_wire_name())
    }
}

impl<'de> Deserialize<'de> for FaEngineName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Self::from_wire_name(&name).map_err(serde::de::Error::custom)
    }
}

/// Typed ASR engine selector.
///
/// The wire format still uses the legacy string tokens (`"rev"`,
/// `"whisper"`, `"whisperx"`, `"whisper_oai"`, or a plugin-provided name), but
/// the control plane works with this enum so backend selection is explicit.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AsrEngineName {
    /// Rust-owned Rev.AI backend.
    RevAi,
    /// Local Whisper worker backend.
    Whisper,
    /// HuggingFace Whisper fine-tune backend. Loads community fine-tunes
    /// by model_id (resolved per-language, with an explicit override in
    /// ``engine_overrides.model_id``). See
    /// ``book/src/batchalign/reference/whisper-hub-asr.md``.
    WhisperHub,
    /// WhisperX worker backend.
    WhisperX,
    /// OpenAI Whisper API backend.
    WhisperOai,
    /// Rust-native Whisper backend (whisper.cpp via whisper-rs), run
    /// in-process instead of through the Python worker. Rust-owned; the
    /// `whisper-rs-backend` Cargo feature is DEFAULT since 2026-07-28, the
    /// model auto-resolves (``BATCHALIGN_WHISPER_RS_MODEL`` override, else
    /// ggml-large-v3 fetched once via hf-hub), and language `Auto` engages
    /// whisper.cpp's own detection. See ``book/src/batchalign/reference/whisper-asr.md``.
    WhisperRs,
    /// Tencent Cloud ASR (HK/Cantonese).
    HkTencent,
    /// Aliyun ASR (HK/Cantonese).
    HkAliyun,
    /// FunAudio ASR (HK/Cantonese).
    HkFunaudio,
    /// Qwen3-ASR (Alibaba, HK/Cantonese). Local model loaded via the
    /// ``qwen-asr`` Python package. Open-weight Cantonese-capable ASR;
    /// external evaluations report competitive CER on per-utterance
    /// child speech.
    HkQwen,
}

impl AsrEngineName {
    /// The wire name, usable in a `const` context.
    ///
    /// THE owner of the table; [`EngineBackend::wire_name`] delegates here. It
    /// is inherent and `const` so a constant can be DERIVED from a variant
    /// rather than restating its spelling: `DEFAULT_ASR_SELECTION_NAME` was
    /// the literal `"rev"` written beside `AsrEngineName::RevAi`, and nothing
    /// tied the two together.
    pub const fn wire_name_const(&self) -> &'static str {
        match self {
            Self::RevAi => "rev",
            Self::Whisper => "whisper",
            Self::WhisperHub => "whisper_hub",
            Self::WhisperX => "whisperx",
            Self::WhisperOai => "whisper_oai",
            Self::WhisperRs => "whisper_rs",
            Self::HkTencent => "tencent",
            Self::HkAliyun => "aliyun",
            Self::HkFunaudio => "funaudio",
            Self::HkQwen => "qwen",
        }
    }
}

impl EngineBackend for AsrEngineName {
    fn wire_name(&self) -> &'static str {
        self.wire_name_const()
    }

    fn is_rust_owned(&self) -> bool {
        matches!(self, Self::RevAi | Self::WhisperRs)
    }

    fn try_from_wire_name(name: &str) -> Option<Self> {
        match name {
            "rev" => Some(Self::RevAi),
            "whisper" => Some(Self::Whisper),
            "whisper_hub" => Some(Self::WhisperHub),
            "whisperx" => Some(Self::WhisperX),
            "whisper_oai" => Some(Self::WhisperOai),
            "whisper_rs" => Some(Self::WhisperRs),
            "tencent" => Some(Self::HkTencent),
            "aliyun" => Some(Self::HkAliyun),
            "funaudio" => Some(Self::HkFunaudio),
            "qwen" => Some(Self::HkQwen),
            _ => None,
        }
    }
}

impl AsrEngineName {
    // `ALL` and `selectable_names` used to live here as inherent items, and
    // this diff added them AGAIN as part of the `SelectableEngine` impl. Two
    // owners of one list, which is the defect the trait exists to remove, and
    // the worse half is that inherent items WIN method resolution over trait
    // items: `AsrEngineName::ALL` silently meant the inherent copy, so the two
    // could drift with nothing to notice. The trait impl is the only one now.

    /// The override name used in worker pool keys for dispatch, or `None` for
    /// cloud-only engines (Rev.AI) that don't need a local worker.
    ///
    /// `execute_v2` reaches this through `EngineSelection` rather than keeping
    /// its own copy, so there is no second table to keep in step.
    pub fn dispatch_override_name(&self) -> Option<&'static str> {
        match self {
            Self::Whisper => Some("whisper"),
            Self::WhisperHub => Some("whisper_hub"),
            Self::HkTencent => Some("tencent"),
            Self::HkAliyun => Some("aliyun"),
            Self::HkFunaudio => Some("funaudio"),
            Self::HkQwen => Some("qwen"),
            // Rust-owned in-process paths (no pool-managed Python worker):
            // Rev.AI and WhisperRs; plus the cloud HTTP engines.
            Self::RevAi | Self::WhisperX | Self::WhisperOai | Self::WhisperRs => None,
        }
    }

    /// Parse one persisted wire-format token.
    ///
    /// Forwards to [`SelectableEngine::parse_wire_name`], which owns both
    /// this body and the category name. All four categories had their own
    /// copy of each.
    pub fn from_wire_name(name: &str) -> Result<Self, UnknownEngineName> {
        <Self as SelectableEngine>::parse_wire_name(name)
    }

    /// Borrow the wire-format token for JSON/SQLite.
    ///
    /// Returns `&'static str`, not `&str`: every wire name is a compile-time
    /// literal and `wire_name` already says so, so the elided lifetime here was
    /// narrowing a fact the callee had established. That cost real work, since a
    /// caller wanting to store the name (an `EngineId` on a per-word provenance
    /// value) had to allocate a `String` to escape a borrow that never needed to
    /// exist.
    pub fn as_wire_name(&self) -> &'static str {
        self.wire_name()
    }

    /// Resident memory footprint estimate for one worker process running
    /// this ASR engine, in MB. Used by the admission gate to reserve
    /// enough headroom for engines whose actual RSS exceeds the default
    /// GPU-profile reservation (``tier.gpu_startup_mb``: 6 GB Small /
    /// 3 GB Medium / 16 GB Large+Fleet). See
    /// [`super::super::worker::pool::memory_gate::engine_aware_startup_reservation_mb`].
    pub fn resident_memory_mb(&self) -> u64 {
        match self {
            // Whisper-large-v3 (and its WhisperHub fine-tunes): ~3 GB
            // model + tokenizer + Python runtime. WhisperX is included
            // here for symmetry/future-proofing even though
            // ``dispatch_override_name`` returns ``None`` for it today
            // (it doesn't get a pool-managed Python worker), so the
            // admission gate never observes this value in production.
            Self::Whisper | Self::WhisperHub | Self::WhisperX => WHISPER_LARGE_V3_RSS_MB,
            // whisper.cpp large-v3 loaded in-process (Rust). Same RSS class as
            // the Python Whisper worker. Runs in the main process (no pool
            // worker), so the worker-admission gate never observes this value;
            // classified here for symmetry.
            Self::WhisperRs => WHISPER_LARGE_V3_RSS_MB,
            // Local model: Qwen3-ASR-1.7B weights (~3.4 GB fp16 /
            // ~7 GB fp32) + tokenizer + Python runtime. Same RSS
            // class as Whisper-large-v3; pinned via the
            // ``asr_engine_qwen_resident_memory_matches_local_model_footprint``
            // test in this module.
            Self::HkQwen => WHISPER_LARGE_V3_RSS_MB,
            // Cloud HTTP clients with no local model. FunASR is
            // grouped here for historical reasons even though
            // SenseVoiceSmall is a local model; the wrapper's
            // resident footprint is closer to a cloud client because
            // it offloads to ModelScope's cached model server.
            // Re-classify if a long-form FunASR run on a tight host
            // ever OOM-kills.
            Self::RevAi
            | Self::WhisperOai
            | Self::HkTencent
            | Self::HkAliyun
            | Self::HkFunaudio => HTTP_CLIENT_BASELINE_RSS_MB,
        }
    }

    /// Whether this engine is the Rust-owned Rev.AI path.
    pub fn is_revai(&self) -> bool {
        matches!(self, Self::RevAi)
    }
}

impl Serialize for AsrEngineName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_wire_name())
    }
}

impl<'de> Deserialize<'de> for AsrEngineName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Self::from_wire_name(&name).map_err(serde::de::Error::custom)
    }
}

/// Typed translation engine selector.
///
/// The wire format uses the lowercase tokens ``"google"``,
/// ``"seamless"``, ``"nllb"``, ``"tencent"``, and ``"aliyun"``; the
/// Python worker's ``resolve_translate_engine``
/// (``batchalign/worker/_model_loading/translation.py``) matches on
/// those exact strings. Any change here must be mirrored on the Python
/// side or dispatch breaks silently.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TranslateEngineName {
    /// Public Google Translate via the ``googletrans`` library. Requires
    /// outbound reachability to ``translate.google.com``, unsuitable
    /// behind the Great Firewall without a VPN.
    Google,
    /// Local Meta SeamlessM4T model, loaded from HuggingFace and run
    /// in-process in the Python worker. No outbound network at
    /// inference time. Retained for back-compat with BA2 callers;
    /// short-CJK quality is poor, prefer ``Nllb`` or ``Tencent`` for
    /// new work.
    Seamless,
    /// Local Meta NLLB-200-distilled-1.3B (~5 GB), text-MT-native.
    /// No outbound network at inference time. Self-hosted fallback
    /// that handles Cantonese first-class (Tencent does not).
    Nllb,
    /// Tencent Cloud TMT (Text Translation), cloud-API engine.
    /// Strong quality on Mandarin (``zh→en``); does NOT support
    /// Cantonese (``yue``). Requires CAM credentials with
    /// ``tmt:TextTranslate`` permission in ``~/.batchalign.ini``
    /// or via ``BATCHALIGN_TENCENT_{ID,KEY,REGION}`` environment
    /// variables. Free tier 5M chars/month.
    Tencent,
    /// Aliyun (Alibaba Cloud) Machine Translation, cloud-API engine.
    /// Supports Cantonese (``yue``) as a source language, which Tencent
    /// TMT does not: the canonical cloud translate option for HK
    /// Cantonese material. Requires access-key credentials in
    /// ``~/.batchalign.ini`` ``[asr]`` section
    /// (``engine.aliyun.id``/``key``/``region``, shared with the Aliyun
    /// ASR backend) or via ``BATCHALIGN_ALIYUN_{ID,KEY,REGION}``
    /// environment variables. Quotas and pricing per Aliyun MT service
    /// terms.
    Aliyun,
}

impl EngineBackend for TranslateEngineName {
    fn wire_name(&self) -> &'static str {
        match self {
            Self::Google => "google",
            Self::Seamless => "seamless",
            Self::Nllb => "nllb",
            Self::Tencent => "tencent",
            Self::Aliyun => "aliyun",
        }
    }

    fn is_rust_owned(&self) -> bool {
        // All backends run in the Python worker. No Rust-owned
        // translate path exists today.
        false
    }

    fn try_from_wire_name(name: &str) -> Option<Self> {
        // The one table, so a spelling cannot be accepted here and rejected by
        // the CLI resolver, or the reverse.
        Self::resolve_variant(name)
    }
}

impl TranslateEngineName {
    /// The override name used in worker pool keys for dispatch.
    ///
    /// Identical to ``wire_name``: translate has no legacy alias
    /// divergence between dispatch and wire today. Provided for
    /// shape-parity with ``AsrEngineName`` and ``FaEngineName``.
    pub fn dispatch_override_name(&self) -> &'static str {
        // Identical to the wire name, which its previous doc admitted; a third
        // copy of one five-string table is three places to mistype it.
        self.wire_name()
    }

    /// Parse one persisted wire-format token.
    ///
    /// Forwards to [`SelectableEngine::parse_wire_name`], which owns both
    /// this body and the category name. All four categories had their own
    /// copy of each.
    pub fn from_wire_name(name: &str) -> Result<Self, UnknownEngineName> {
        <Self as SelectableEngine>::parse_wire_name(name)
    }

    /// Borrow the wire-format token for JSON/SQLite.
    ///
    /// Returns `&'static str`, not `&str`: every wire name is a compile-time
    /// literal and `wire_name` already says so, so the elided lifetime here was
    /// narrowing a fact the callee had established. That cost real work, since a
    /// caller wanting to store the name (an `EngineId` on a per-word provenance
    /// value) had to allocate a `String` to escape a borrow that never needed to
    /// exist.
    pub fn as_wire_name(&self) -> &'static str {
        self.wire_name()
    }

    /// Resident memory footprint estimate for one worker process
    /// running this translate engine, in MB. Used by the admission
    /// gate to reserve enough headroom for engines whose actual RSS
    /// exceeds the default IO-profile reservation
    /// (``tier.io_startup_mb``: 2 GB Small/Medium, 4 GB Large/Fleet).
    /// The estimate is the observed model + tokenizer + Python
    /// runtime footprint with a modest margin; conservative on the
    /// side of over-reserving so the OS OOM killer isn't the fallback
    /// safety mechanism. Related but distinct from the *on-disk*
    /// model-size hints used by the Python progress events
    /// (``batchalign/worker/_progress.py::_HF_SIZE_HINTS_GB``).
    pub fn resident_memory_mb(&self) -> u64 {
        match self {
            // googletrans + Tencent TMT + Aliyun MT are all thin
            // HTTP-client engines with no local model loaded, same
            // baseline. The Aliyun MT REST client and ``googletrans``
            // both wrap ``requests``/``aiohttp``-style transports;
            // there is no per-process model state to account for.
            Self::Google | Self::Tencent | Self::Aliyun => HTTP_CLIENT_BASELINE_RSS_MB,
            Self::Seamless => SEAMLESS_M4T_MEDIUM_RSS_MB,
            Self::Nllb => NLLB_200_DISTILLED_1_3B_RSS_MB,
        }
    }
}

/// Resident memory estimate for any worker that runs a thin HTTP-client
/// engine with no local model loaded, googletrans for translate, and
/// the cloud ASR engines (Rev.AI, WhisperOai, HkTencent, HkAliyun,
/// HkFunaudio). Baseline Python + worker scaffolding only.
pub(crate) const HTTP_CLIENT_BASELINE_RSS_MB: u64 = 200;

/// Resident memory estimate for a worker running the local
/// SeamlessM4T-medium model: ~2.4 GB weights + tokenizer + runtime,
/// with margin.
pub(crate) const SEAMLESS_M4T_MEDIUM_RSS_MB: u64 = 2_900;

/// Resident memory estimate for a worker running the local
/// NLLB-200-distilled-1.3B model: ~5 GB weights + tokenizer +
/// runtime, with margin.
pub(crate) const NLLB_200_DISTILLED_1_3B_RSS_MB: u64 = 5_500;

/// Resident memory estimate for a worker running the Whisper-large-v3
/// ASR model or the Whisper-large-v2 FA model (same shape). ~3 GB
/// weights + tokenizer + Python runtime + margin.
pub(crate) const WHISPER_LARGE_V3_RSS_MB: u64 = 3_500;

/// Resident memory estimate for a worker running an MMS / Wave2Vec
/// forced-alignment model (including the Cantonese variant): ~1.2 GB
/// torchaudio weights + runtime margin.
pub(crate) const WAVE2VEC_FA_RSS_MB: u64 = 1_800;

/// Resident memory estimate for a worker running the Qwen3 forced aligner
/// (`Qwen/Qwen3-ForcedAligner-0.6B-hf`): ~0.6 B parameters at float32 is
/// ~2.4 GB of weights, plus the processor and the Python runtime. Between the
/// wave2vec and Whisper classes, so it gets its own constant rather than
/// borrowing one that would understate the reservation.
pub(crate) const QWEN3_FA_RSS_MB: u64 = 3_000;

impl Serialize for TranslateEngineName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_wire_name())
    }
}

impl<'de> Deserialize<'de> for TranslateEngineName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Self::from_wire_name(&name).map_err(serde::de::Error::custom)
    }
}

/// Typed speaker-diarization engine selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SpeakerEngineName {
    /// pyannoteAI Precision-2 cloud diarization.
    PyannoteAi,
    /// Local TalkBank-pinned Pyannote diarization.
    Pyannote,
    /// Local NeMo diarization.
    Nemo,
}

impl EngineBackend for SpeakerEngineName {
    fn wire_name(&self) -> &'static str {
        match self {
            Self::PyannoteAi => "pyannote_ai",
            Self::Pyannote => "pyannote",
            Self::Nemo => "nemo",
        }
    }

    fn is_rust_owned(&self) -> bool {
        false
    }

    fn try_from_wire_name(name: &str) -> Option<Self> {
        Self::resolve_variant(name)
    }
}

impl SelectableEngine for SpeakerEngineName {
    type Selected = Self;
    const ALL: &'static [Self] = &[Self::PyannoteAi, Self::Pyannote, Self::Nemo];
    const DEFAULT: Self = Self::PyannoteAi;
    const CATEGORY: &'static str = "speaker diarization";

    fn selection_name(&self) -> &'static str {
        match self {
            Self::PyannoteAi => "pyannote-ai",
            Self::Pyannote => "pyannote",
            Self::Nemo => "nemo",
        }
    }

    fn accepted_names() -> &'static [(&'static str, Self)] {
        &[
            ("pyannote-ai", Self::PyannoteAi),
            ("pyannote_ai", Self::PyannoteAi),
            ("pyannote", Self::Pyannote),
            ("nemo", Self::Nemo),
        ]
    }

    fn resolve(name: &str) -> Option<Self> {
        Self::resolve_variant(name)
    }
}

impl SpeakerEngineName {
    /// Parse one persisted wire token.
    pub fn from_wire_name(name: &str) -> Result<Self, UnknownEngineName> {
        <Self as SelectableEngine>::parse_wire_name(name)
    }
}

impl Serialize for SpeakerEngineName {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.wire_name())
    }
}

impl<'de> Deserialize<'de> for SpeakerEngineName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Self::from_wire_name(&name).map_err(serde::de::Error::custom)
    }
}

// ---------------------------------------------------------------------------
// EngineOverrides: typed engine override selection
// ---------------------------------------------------------------------------

/// Typed engine overrides for one job or worker spawn.
///
/// Replaces `BTreeMap<String, String>` in `CommonOptions.engine_overrides`.
/// Only populated fields are serialized; empty overrides produce `{}`.
///
/// Engine-selection fields are typed because they pick which engine runs.
/// Any other key is preserved
/// as an opaque per-engine configuration extra in [`Self::extras`].
/// This is how the Python worker receives per-engine knobs such as
/// ``qwen_model``, ``qwen_device``, ``funaudio_*``, etc., adding a
/// new engine knob does NOT require a Rust schema change, but a typo
/// in a knob name will reach Python where the engine loader chooses
/// whether to use a default or error. (A future engine registry
/// task #66 / Phase 5c, replaces this string-keyed map with typed
/// per-engine payload structs.)
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct EngineOverrides {
    /// ASR engine override (e.g., `AsrEngineName::HkTencent`).
    pub asr: Option<AsrEngineName>,
    /// FA engine override (e.g., `FaEngineName::Wav2vecCanto`).
    pub fa: Option<FaEngineName>,
    /// UTR engine override (e.g., `UtrEngine::HkTencent`).
    ///
    /// Added late, and its absence was a silent-discard bug rather than a
    /// deliberate omission: `utr` is not one of the typed keys, so
    /// `--engine-overrides '{"utr":"whisper_utr"}'` fell through to
    /// [`Self::extras`] and was shipped to a Python worker that has no say in
    /// UTR engine selection at all. The user's choice vanished with no message.
    pub utr: Option<UtrEngine>,
    /// Translate engine override (e.g., `TranslateEngineName::Seamless`).
    pub translate: Option<TranslateEngineName>,
    /// Dedicated speaker diarization engine.
    pub speaker: Option<SpeakerEngineName>,
    /// Opaque per-engine configuration knobs (e.g., ``qwen_model``,
    /// ``qwen_device``). Round-trips verbatim through the JSON
    /// boundary so the Python worker bootstrap can read them by name.
    pub extras: std::collections::BTreeMap<String, String>,
}

impl EngineOverrides {
    /// Return `true` when no overrides are set.
    pub fn is_empty(&self) -> bool {
        self.utr.is_none()
            && self.asr.is_none()
            && self.fa.is_none()
            && self.translate.is_none()
            && self.speaker.is_none()
            && self.extras.is_empty()
    }

    /// Serialize to a JSON string in the PERSISTENCE wire format
    /// (`wire_name()` tokens). For anything that reaches a worker
    /// (pool keys, capability-discovery spawns, worker argv), use
    /// [`Self::to_dispatch_json_string`] instead.
    ///
    /// Returns empty string when no overrides are set.
    pub fn to_json_string(&self) -> String {
        if self.is_empty() {
            String::new()
        } else {
            serde_json::to_string(self).unwrap_or_else(|e| format!("<serialization failed: {e}>"))
        }
    }

    /// Produce the worker-facing override map, using the DISPATCH names the
    /// Python engine loaders accept (`dispatch_override_name()`), NOT the
    /// persistence wire names (`wire_name()`).
    ///
    /// The two schemes differ for every FA engine ("wav2vec_fa" /
    /// "whisper_fa" / "cantonese_fa" persisted vs "wave2vec" /
    /// "whisper" / "wav2vec_canto" dispatched). Sending a persistence
    /// name kills the worker at bootstrap: `resolve_fa_engine` raises
    /// before the ready signal, which failed four consecutive align
    /// jobs on a fleet host on 2026-06-11.
    ///
    /// Cloud-only ASR engines with no local worker (Rev.AI, WhisperX,
    /// WhisperOai) have no dispatch name and are omitted. Extras
    /// round-trip verbatim, exactly as in [`Self::to_json_string`]
    /// (the 2026-05-27 `qwen_model` lesson).
    ///
    pub fn dispatch_overrides(&self) -> std::collections::BTreeMap<String, String> {
        let mut map = std::collections::BTreeMap::new();
        if let Some(ref asr) = self.asr
            && let Some(name) = asr.dispatch_override_name()
        {
            map.insert("asr".to_owned(), name.to_owned());
        }
        if let Some(ref fa) = self.fa {
            map.insert("fa".to_owned(), fa.dispatch_override_name().to_owned());
        }
        if let Some(ref translate) = self.translate {
            map.insert(
                "translate".to_owned(),
                translate.dispatch_override_name().to_owned(),
            );
        }
        if let Some(ref speaker) = self.speaker {
            map.insert("speaker".to_owned(), speaker.wire_name().to_owned());
        }
        for (key, value) in &self.extras {
            map.insert(key.clone(), value.clone());
        }
        map
    }

    /// Serialize [`Self::dispatch_overrides`] at the Rust/Python worker
    /// boundary.
    ///
    /// Returns an empty string when no overrides are set.
    pub fn to_dispatch_json_string(&self) -> String {
        let map = self.dispatch_overrides();
        if map.is_empty() {
            return String::new();
        }
        serde_json::to_string(&map).unwrap_or_else(|e| format!("<serialization failed: {e}>"))
    }
}

impl Serialize for EngineOverrides {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        let count = self.asr.is_some() as usize
            + self.fa.is_some() as usize
            + self.utr.is_some() as usize
            + self.translate.is_some() as usize
            + self.speaker.is_some() as usize
            + self.extras.len();
        let mut map = serializer.serialize_map(Some(count))?;
        if let Some(ref asr) = self.asr {
            map.serialize_entry("asr", asr.as_wire_name())?;
        }
        if let Some(ref fa) = self.fa {
            map.serialize_entry("fa", fa.as_wire_name())?;
        }
        if let Some(ref utr) = self.utr {
            map.serialize_entry("utr", utr.as_wire_name())?;
        }
        if let Some(ref translate) = self.translate {
            map.serialize_entry("translate", translate.as_wire_name())?;
        }
        if let Some(ref speaker) = self.speaker {
            map.serialize_entry("speaker", speaker.wire_name())?;
        }
        for (key, value) in &self.extras {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for EngineOverrides {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let map: std::collections::BTreeMap<String, String> =
            std::collections::BTreeMap::deserialize(deserializer)?;
        let mut overrides = Self::default();
        for (key, value) in map {
            match key.as_str() {
                "asr" => {
                    overrides.asr = Some(
                        AsrEngineName::from_wire_name(&value).map_err(serde::de::Error::custom)?,
                    );
                }
                "fa" => {
                    overrides.fa = Some(
                        FaEngineName::from_wire_name(&value).map_err(serde::de::Error::custom)?,
                    );
                }
                "utr" => {
                    overrides.utr =
                        Some(UtrEngine::from_wire_name(&value).map_err(serde::de::Error::custom)?);
                }
                "translate" => {
                    overrides.translate = Some(
                        TranslateEngineName::from_wire_name(&value)
                            .map_err(serde::de::Error::custom)?,
                    );
                }
                "speaker" => {
                    overrides.speaker = Some(
                        SpeakerEngineName::from_wire_name(&value)
                            .map_err(serde::de::Error::custom)?,
                    );
                }
                _other => {
                    // Per-engine configuration knob. The set of valid
                    // keys is engine-specific and validated on the
                    // Python side at load time; an unknown knob falls
                    // through to engine defaults rather than rejecting
                    // the entire CLI invocation. See the doc comment
                    // on EngineOverrides for the rationale.
                    overrides.extras.insert(key, value);
                }
            }
        }
        Ok(overrides)
    }
}

#[cfg(test)]
mod tests {
    //! Wire-name / dispatch-key roundtrip coverage for ``AsrEngineName``.
    //!
    //! The wire name is the single source of truth shared between
    //! Rust (``AsrEngineName`` here, ``AsrBackendV2`` in
    //! ``batchalign-types``), Python (``AsrEngine`` enum in
    //! ``batchalign/worker/_types.py``), the CLI flag parser, and SQLite
    //! job persistence. A mismatch in any one of those locations breaks
    //! dispatch silently. These tests pin the contract at the Rust
    //! entry point.
    use super::*;

    #[test]
    fn whisper_hub_wire_roundtrip() {
        assert_eq!(AsrEngineName::WhisperHub.wire_name(), "whisper_hub");
        assert_eq!(
            AsrEngineName::try_from_wire_name("whisper_hub"),
            Some(AsrEngineName::WhisperHub),
        );
    }

    #[test]
    fn whisper_rs_wire_roundtrip() {
        assert_eq!(AsrEngineName::WhisperRs.wire_name(), "whisper_rs");
        assert_eq!(
            AsrEngineName::try_from_wire_name("whisper_rs"),
            Some(AsrEngineName::WhisperRs),
        );
    }

    #[test]
    fn whisper_rs_is_rust_owned_but_not_revai() {
        // The native whisper.cpp path runs in-process (Rust-owned), like
        // Rev.AI, so it has no pool-managed Python worker.
        assert!(AsrEngineName::WhisperRs.is_rust_owned());
        assert!(!AsrEngineName::WhisperRs.is_revai());
        assert_eq!(AsrEngineName::WhisperRs.dispatch_override_name(), None);
    }

    #[test]
    fn whisper_hub_is_not_rust_owned() {
        // Rust-owned engines run inference from the server process directly
        // (Rev.AI and the native whisper-rs path today). whisper_hub runs in
        // a Python worker like stock Whisper / WhisperX / HK engines.
        assert!(!AsrEngineName::WhisperHub.is_rust_owned());
        assert!(!AsrEngineName::WhisperHub.is_revai());
    }

    #[test]
    fn whisper_hub_dispatch_override_name_matches_wire_name() {
        // Worker pool keys must match the wire name so the Python worker
        // bootstrap sees ``engine_overrides["asr"] == "whisper_hub"`` and
        // routes to the fine-tune loader in ``_model_loading/asr.py``.
        assert_eq!(
            AsrEngineName::WhisperHub.dispatch_override_name(),
            Some("whisper_hub"),
        );
    }

    // ---- TranslateEngineName ----
    //
    // Pinned because the Python worker's `resolve_translate_engine`
    // (`batchalign/worker/_model_loading/translation.py`) matches on the
    // exact strings "google" and "seamless". A typo here would
    // silently fall through to the default engine on the Python side.

    #[test]
    fn translate_engine_google_wire_roundtrip() {
        assert_eq!(TranslateEngineName::Google.wire_name(), "google");
        assert_eq!(
            TranslateEngineName::try_from_wire_name("google"),
            Some(TranslateEngineName::Google),
        );
    }

    #[test]
    fn translate_engine_seamless_wire_roundtrip() {
        assert_eq!(TranslateEngineName::Seamless.wire_name(), "seamless");
        assert_eq!(
            TranslateEngineName::try_from_wire_name("seamless"),
            Some(TranslateEngineName::Seamless),
        );
    }

    #[test]
    fn translate_engine_nllb_wire_roundtrip() {
        assert_eq!(TranslateEngineName::Nllb.wire_name(), "nllb");
        assert_eq!(
            TranslateEngineName::try_from_wire_name("nllb"),
            Some(TranslateEngineName::Nllb),
        );
    }

    #[test]
    fn translate_engine_tencent_wire_roundtrip() {
        assert_eq!(TranslateEngineName::Tencent.wire_name(), "tencent");
        assert_eq!(
            TranslateEngineName::try_from_wire_name("tencent"),
            Some(TranslateEngineName::Tencent),
        );
    }

    #[test]
    fn translate_engine_aliyun_wire_roundtrip() {
        // Aliyun Machine Translation is the cloud-API translate engine
        // for Cantonese (``yue``) and other Asian-language source codes
        // that Tencent TMT does not list. The wire name ``"aliyun"``
        // must match the Python worker's ``TranslationBackend.ALIYUN``
        // value in ``batchalign/inference/_domain_types.py`` exactly,
        // since the resolver in
        // ``batchalign/worker/_model_loading/translation.py`` matches
        // on string equality.
        assert_eq!(TranslateEngineName::Aliyun.wire_name(), "aliyun");
        assert_eq!(
            TranslateEngineName::try_from_wire_name("aliyun"),
            Some(TranslateEngineName::Aliyun),
        );
    }

    #[test]
    fn translate_engine_unknown_wire_name_is_rejected() {
        assert_eq!(TranslateEngineName::try_from_wire_name("gogle"), None);
        let err = TranslateEngineName::from_wire_name("gogle").unwrap_err();
        assert_eq!(err.category, "translate");
        assert_eq!(err.name, "gogle");
    }

    #[test]
    fn translate_engine_resident_memory_ordering() {
        // Pins the physical ordering, Google (HTTP client) <
        // Seamless (~2.4 GB) < NLLB (~5 GB), that the admission-gate
        // engine-aware reservation
        // (``worker::pool::memory_gate::engine_aware_startup_reservation_mb``)
        // relies on. A typo here would silently re-introduce under-
        // reservation for the heavier engines.
        let google_mb = TranslateEngineName::Google.resident_memory_mb();
        let seamless_mb = TranslateEngineName::Seamless.resident_memory_mb();
        let nllb_mb = TranslateEngineName::Nllb.resident_memory_mb();
        assert!(
            google_mb < seamless_mb,
            "Google ({google_mb} MB) must be smaller than Seamless ({seamless_mb} MB)"
        );
        assert!(
            seamless_mb < nllb_mb,
            "Seamless ({seamless_mb} MB) must be smaller than NLLB ({nllb_mb} MB)"
        );
    }

    #[test]
    fn asr_engine_resident_memory_partitions_local_vs_cloud() {
        // Local Whisper variants must all match the heavy-model
        // footprint; cloud HTTP clients must all match the cheap
        // baseline. The admission gate's engine-aware reservation
        // depends on this partition being clean.
        assert_eq!(
            AsrEngineName::Whisper.resident_memory_mb(),
            WHISPER_LARGE_V3_RSS_MB
        );
        assert_eq!(
            AsrEngineName::WhisperHub.resident_memory_mb(),
            WHISPER_LARGE_V3_RSS_MB
        );
        assert_eq!(
            AsrEngineName::WhisperX.resident_memory_mb(),
            WHISPER_LARGE_V3_RSS_MB
        );
        for cloud in [
            AsrEngineName::RevAi,
            AsrEngineName::WhisperOai,
            AsrEngineName::HkTencent,
            AsrEngineName::HkAliyun,
            AsrEngineName::HkFunaudio,
        ] {
            assert_eq!(
                cloud.resident_memory_mb(),
                HTTP_CLIENT_BASELINE_RSS_MB,
                "{cloud:?} should match the cloud HTTP-client baseline"
            );
        }
        const _: () = assert!(HTTP_CLIENT_BASELINE_RSS_MB < WHISPER_LARGE_V3_RSS_MB);
    }

    #[test]
    fn asr_engine_qwen_wire_roundtrip() {
        // ``HkQwen`` wires as ``"qwen"`` across the JSON and the
        // engine-overrides knob. Round-trip pinned so a future rename
        // breaks visibly.
        let engine = AsrEngineName::HkQwen;
        assert_eq!(engine.wire_name(), "qwen");
        assert_eq!(
            AsrEngineName::try_from_wire_name("qwen"),
            Some(AsrEngineName::HkQwen)
        );
        assert_eq!(engine.dispatch_override_name(), Some("qwen"));
    }

    #[test]
    fn asr_engine_qwen_resident_memory_matches_local_model_footprint() {
        // Qwen3-ASR-1.7B is a local model, not a cloud HTTP client.
        // Its resident footprint must reserve enough headroom for the
        // weights + tokenizer + Python runtime. We pin it to the same
        // class as Whisper-large-v3, both are local ~1.5-3 GB
        // models with similar Python-side overhead. Wrong-side
        // partitioning (treating Qwen as a cloud HTTP client) would
        // under-reserve memory and trigger admission-gate OOM kills
        // on tight hosts.
        let qwen_mb = AsrEngineName::HkQwen.resident_memory_mb();
        assert!(
            qwen_mb >= WHISPER_LARGE_V3_RSS_MB,
            "Qwen ({qwen_mb} MB) must reserve at least the local-model baseline ({WHISPER_LARGE_V3_RSS_MB} MB)"
        );
        assert!(
            qwen_mb > HTTP_CLIENT_BASELINE_RSS_MB,
            "Qwen must NOT be partitioned as a cloud HTTP client ({HTTP_CLIENT_BASELINE_RSS_MB} MB)"
        );
    }

    #[test]
    fn fa_engine_resident_memory_separates_whisper_from_wave2vec() {
        assert_eq!(
            FaEngineName::Whisper.resident_memory_mb(),
            WHISPER_LARGE_V3_RSS_MB
        );
        assert_eq!(
            FaEngineName::Wave2Vec.resident_memory_mb(),
            WAVE2VEC_FA_RSS_MB
        );
        assert_eq!(
            FaEngineName::Wav2vecCanto.resident_memory_mb(),
            WAVE2VEC_FA_RSS_MB
        );
        const _: () = assert!(WAVE2VEC_FA_RSS_MB < WHISPER_LARGE_V3_RSS_MB);
    }

    #[test]
    fn translate_engine_tencent_matches_http_client_baseline() {
        // Tencent TMT is a thin HTTP-client engine, no local model
        // loaded, so its resident footprint is the same as Google's
        // and Seamless's lightweight baseline. Pinned to prevent
        // accidental inflation (which would over-reserve memory and
        // refuse spawns on hosts that can comfortably run Tencent
        // translate workers).
        assert_eq!(
            TranslateEngineName::Tencent.resident_memory_mb(),
            HTTP_CLIENT_BASELINE_RSS_MB
        );
        assert_eq!(
            TranslateEngineName::Tencent.resident_memory_mb(),
            TranslateEngineName::Google.resident_memory_mb()
        );
    }

    #[test]
    fn translate_engine_no_variant_is_rust_owned() {
        // All backends run in the Python worker, none talk to a
        // provider directly from the Rust server.
        assert!(!TranslateEngineName::Google.is_rust_owned());
        assert!(!TranslateEngineName::Seamless.is_rust_owned());
        assert!(!TranslateEngineName::Nllb.is_rust_owned());
        assert!(!TranslateEngineName::Tencent.is_rust_owned());
    }

    #[test]
    fn translate_engine_serializes_as_wire_string() {
        let json = serde_json::to_string(&TranslateEngineName::Seamless).unwrap();
        assert_eq!(json, "\"seamless\"");
    }

    #[test]
    fn translate_engine_deserializes_from_wire_string() {
        let parsed: TranslateEngineName = serde_json::from_str("\"seamless\"").unwrap();
        assert_eq!(parsed, TranslateEngineName::Seamless);
    }

    #[test]
    fn translate_engine_deserialize_rejects_unknown_variant() {
        let err = serde_json::from_str::<TranslateEngineName>("\"gogle\"").unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("gogle"),
            "expected error to mention the bad name, got: {message}"
        );
    }

    // ---- EngineOverrides translate field ----

    #[test]
    fn engine_overrides_serializes_translate_field() {
        let overrides = EngineOverrides {
            asr: None,
            fa: None,
            translate: Some(TranslateEngineName::Seamless),
            ..Default::default()
        };
        let json = overrides.to_json_string();
        assert_eq!(json, "{\"translate\":\"seamless\"}");
    }

    #[test]
    fn engine_overrides_deserializes_translate_field() {
        let parsed: EngineOverrides = serde_json::from_str("{\"translate\":\"seamless\"}").unwrap();
        assert_eq!(parsed.translate, Some(TranslateEngineName::Seamless));
        assert!(parsed.asr.is_none());
        assert!(parsed.fa.is_none());
    }

    #[test]
    fn engine_overrides_translate_only_is_not_empty() {
        let overrides = EngineOverrides {
            asr: None,
            fa: None,
            translate: Some(TranslateEngineName::Seamless),
            ..Default::default()
        };
        assert!(!overrides.is_empty());
    }

    #[test]
    fn engine_overrides_all_none_is_still_empty() {
        let overrides = EngineOverrides::default();
        assert!(overrides.is_empty());
        assert_eq!(overrides.to_json_string(), "");
    }

    // ---- EngineOverrides extras (per-engine knobs) ----

    #[test]
    fn engine_overrides_extras_round_trip_unknown_keys() {
        // Drill-down regression guard for Fix 1 (the starter test
        // lives in cli/args/tests.rs and exercises the full
        // Cli::parse_from → build_typed_options → to_json_string
        // path). This pins the deserialize/serialize layer in
        // isolation so a future refactor that moves the JSON shape
        // can't silently drop extras.
        let parsed: EngineOverrides = serde_json::from_str(
            r#"{"asr":"qwen","qwen_model":"Qwen/Qwen3-ASR-0.6B","qwen_device":"cuda"}"#,
        )
        .unwrap();
        assert_eq!(parsed.asr, Some(AsrEngineName::HkQwen));
        assert_eq!(
            parsed.extras.get("qwen_model").map(String::as_str),
            Some("Qwen/Qwen3-ASR-0.6B")
        );
        assert_eq!(
            parsed.extras.get("qwen_device").map(String::as_str),
            Some("cuda")
        );

        let json = parsed.to_json_string();
        let reparsed: EngineOverrides = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, reparsed, "round-trip must be lossless");
    }

    #[test]
    fn engine_overrides_extras_only_is_not_empty() {
        // An override payload of just per-engine knobs (no explicit
        // engine selection) is still a meaningful payload that must
        // reach the worker: ``is_empty`` must reflect that or the
        // ``--engine-overrides`` flag drops out before reaching the
        // worker spawn arg (see ``worker/handle/spawn.rs:61``).
        let parsed: EngineOverrides =
            serde_json::from_str(r#"{"qwen_model":"Qwen/Qwen3-ASR-0.6B"}"#).unwrap();
        assert_eq!(parsed.asr, None);
        assert!(!parsed.is_empty());
    }

    #[test]
    fn engine_overrides_known_engine_validation_still_fires() {
        // Unknown values for KNOWN keys (asr/fa/translate) still
        // error: Fix 1 relaxed schema strictness only for unknown
        // KEYS. A typo in an engine name is still loud.
        let err = serde_json::from_str::<EngineOverrides>(r#"{"asr":"wisper"}"#).unwrap_err();
        assert!(
            err.to_string().contains("wisper"),
            "expected engine-name validation error, got: {err}"
        );
    }
}

/// The name a user reaches for when they want Mandarin/Cantonese Paraformer.
///
/// Not an [`AsrEngineName`] variant: Paraformer is the FunAudio engine loading
/// a particular checkpoint, so promoting it to its own backend would duplicate
/// the funaudio dispatch path for no gain. It is a selection name instead.
pub const PARAFORMER_SELECTION_NAME: &str = "paraformer";

/// The FunASR checkpoint that makes FunAudio behave as Paraformer.
///
/// This is the checkpoint the FunASR ecosystem publishes under the Paraformer
/// name for Chinese, so `--asr-engine paraformer` and a hand-written
/// `funaudio_model=paraformer-zh` load exactly the same model.
pub const PARAFORMER_CHECKPOINT: &str = "paraformer-zh";

/// The override key FunAudio reads its checkpoint from.
pub const FUNAUDIO_MODEL_OVERRIDE_KEY: &str = "funaudio_model";

/// A selection that implies nothing beyond its engine.
const NO_IMPLIED_OVERRIDES: &[(&str, &str)] = &[];

/// What selecting `paraformer` implies: the FunASR checkpoint that makes the
/// FunAudio engine behave as Paraformer.
const PARAFORMER_IMPLIED_OVERRIDES: &[(&str, &str)] =
    &[(FUNAUDIO_MODEL_OVERRIDE_KEY, PARAFORMER_CHECKPOINT)];

/// An ASR engine choice as made by a user, with anything that choice implies.
///
/// Constructed only by [`parse`](Self::parse), so a selection cannot be
/// assembled from an engine and an unrelated set of overrides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsrSelection {
    engine: AsrEngineName,
    implied_overrides: &'static [(&'static str, &'static str)],
}

impl AsrSelection {
    /// A selection that is just an engine, with nothing implied.
    ///
    /// For engines known at COMPILE time, so they never travel through a
    /// string. The BA2 compatibility switches used this type by calling
    /// `parse("whisperx")` and friends, which meant a wire-name change would
    /// have turned four working flags into silent no-ops with nothing to catch
    /// it; taking the variant makes that a compile error instead.
    pub fn from_engine(engine: AsrEngineName) -> Self {
        Self {
            engine,
            implied_overrides: NO_IMPLIED_OVERRIDES,
        }
    }

    /// Resolve a user-facing engine name.
    ///
    /// Accepts every [`AsrEngineName`] wire name plus
    /// [`PARAFORMER_SELECTION_NAME`]. Returns `None` for anything else, and
    /// the caller must REPORT that rather than proceeding: an unrecognised
    /// engine name used to be swallowed by an `Option`-returning resolver, so a
    /// typo silently produced no engine at all.
    pub fn parse(name: &str) -> Option<Self> {
        if name == PARAFORMER_SELECTION_NAME {
            return Some(Self {
                engine: AsrEngineName::HkFunaudio,
                implied_overrides: PARAFORMER_IMPLIED_OVERRIDES,
            });
        }
        // The SAME table the CLI derives its hidden-alias list from. These were
        // two tables (a separate `LEGACY_SELECTION_ALIASES` const), so a
        // spelling could be advertised by one and rejected by the other, which
        // is precisely what `accepted_names` documents as impossible.
        AsrEngineName::resolve_variant(name).map(Self::from_engine)
    }

    /// The backend this selection runs on.
    pub fn engine(&self) -> AsrEngineName {
        self.engine.clone()
    }

    /// Overrides the NAME implies, which an explicit `--engine-overrides` wins
    /// over: the user's typed JSON is more specific than the name's default.
    pub fn implied_overrides(&self) -> &'static [(&'static str, &'static str)] {
        self.implied_overrides
    }

    /// Apply what the NAME implies, without beating what the user typed.
    ///
    /// Lives on the selection because the merge is part of what selecting a
    /// name MEANS. Every transcribing command had its own copy of this loop,
    /// which is the same shape the shared resolver removed one level up: an
    /// arm that forgets it makes `--asr-engine paraformer` parse and then run
    /// plain funaudio, which is the exact defect the name exists to fix.
    ///
    /// An explicit `--engine-overrides` wins: a user naming a checkpoint is
    /// being more specific than the name's default, not less.
    pub fn apply_implied(&self, overrides: &mut EngineOverrides) {
        for (key, value) in self.implied_overrides {
            overrides
                .extras
                .entry((*key).to_string())
                .or_insert_with(|| (*value).to_string());
        }
    }
}

#[cfg(test)]
mod asr_selection_tests {
    use super::*;

    /// Every engine in the owner list resolves by its own wire name.
    ///
    /// This is what keeps the CLI's accepted set honest: a new variant that is
    /// not in `ALL` is caught here rather than by a user who cannot reach it.
    #[test]
    fn every_engine_in_all_parses_from_its_wire_name() {
        for engine in AsrEngineName::ALL {
            let selection = AsrSelection::parse(engine.wire_name())
                .unwrap_or_else(|| panic!("{} must parse", engine.wire_name()));
            assert_eq!(&selection.engine(), engine);
            assert!(
                selection.implied_overrides().is_empty(),
                "a plain engine name implies no overrides"
            );
        }
    }

    /// Every entry in `ALL` names a distinct engine.
    ///
    /// The COUNT is carried by the array type (`[Self; 10]`), so adding a
    /// variant without extending `ALL` is a compile error and needs no test.
    /// What a type cannot check is that the ten entries are ten DIFFERENT
    /// engines, which is what this asserts.
    #[test]
    fn all_names_are_distinct() {
        let mut names: Vec<&str> = AsrEngineName::ALL.iter().map(|e| e.wire_name()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), AsrEngineName::ALL.len(), "duplicate wire name");
        assert_eq!(names.len(), 10, "ALL must list all ten engines");
    }

    /// Paraformer resolves to funaudio carrying its checkpoint.
    #[test]
    fn paraformer_resolves_to_funaudio_plus_checkpoint() {
        let selection = AsrSelection::parse(PARAFORMER_SELECTION_NAME).expect("selectable");
        assert_eq!(selection.engine(), AsrEngineName::HkFunaudio);
        assert_eq!(selection.implied_overrides(), PARAFORMER_IMPLIED_OVERRIDES);
    }

    /// An unknown name yields nothing, so the caller has to report it.
    #[test]
    fn an_unknown_name_does_not_resolve() {
        assert_eq!(AsrSelection::parse("paraformr"), None);
        assert_eq!(AsrSelection::parse(""), None);
    }

    /// The user-facing list is the owner list plus paraformer, with no gaps.
    #[test]
    fn selectable_names_covers_all_engines_and_paraformer() {
        let names: Vec<&str> = AsrEngineName::selectable_names().collect();
        assert_eq!(names.len(), AsrEngineName::ALL.len() + 1);
        assert!(names.contains(&PARAFORMER_SELECTION_NAME));
        for engine in AsrEngineName::ALL {
            assert!(names.contains(&engine.wire_name()));
        }
    }
}

#[cfg(test)]
mod selectable_engine_tests {
    use super::*;

    /// Every engine in a category is advertised, resolves, and resolves back to
    /// itself, checked generically so no category can be left out.
    ///
    /// This is the property the four hand-written surfaces kept failing: three
    /// of four advertised a SUBSET of what they accepted. It survives as a test
    /// rather than a type because "the advertised list and the resolver agree"
    /// is a relationship between two functions, which no signature states.
    fn assert_category_is_coherent<E>()
    where
        E: SelectableEngine + PartialEq + std::fmt::Debug,
    {
        // Written against `resolve_variant`, which returns `Option<Self>` for
        // EVERY category including ASR. The first version was bounded on
        // `Selected = E`, which silently exempted ASR, the only category with
        // ten engines and the one whose list had already gone stale once.

        // The load-bearing check, and the ONLY one here that can catch an
        // engine missing from `ALL`. `ACCEPTED_NAMES` is written independently
        // of `ALL`, so it is a second witness; anything derived from `ALL` and
        // compared against `ALL` is circular. An earlier version asserted
        // `advertised.len() == ALL.len()`, which passed cleanly with a variant
        // deleted from `ALL`, because deleting it shrank both sides.
        for (name, engine) in E::accepted_names() {
            assert!(
                E::ALL.contains(engine),
                "{}: {name} resolves to {engine:?}, which is missing from ALL, \
                 so that engine is unreachable from the flag",
                E::CATEGORY
            );
        }

        // Every engine's canonical name is in the table. This is what makes
        // `accepted_names` mean the same thing in all four categories; three
        // of them used to omit some or all canonical names, which forced three
        // different resolvers and weakened the check above.
        for engine in E::ALL {
            let name = engine.selection_name();
            assert_eq!(
                E::resolve_variant(name).as_ref(),
                Some(engine),
                "{}: canonical name {name} must resolve back to itself",
                E::CATEGORY
            );
        }

        // Hidden aliases must resolve: clap rejects anything outside the
        // shown-plus-hidden list before the resolver runs, so an alias the
        // resolver does not know is a value clap accepts and then errors on.
        for alias in E::hidden_alias_names() {
            assert!(
                E::resolve_variant(alias).is_some(),
                "{}: hidden alias {alias} does not resolve",
                E::CATEGORY
            );
        }

        // The wire name round-trips, so persisted job options still parse.
        for engine in E::ALL {
            assert_eq!(
                E::try_from_wire_name(engine.wire_name()).as_ref(),
                Some(engine),
                "{}: wire name {} must parse back",
                E::CATEGORY,
                engine.wire_name()
            );
        }

        // One canonical name per engine.
        let mut advertised: Vec<&str> = E::ALL.iter().map(E::selection_name).collect();
        let count = advertised.len();
        advertised.sort_unstable();
        advertised.dedup();
        assert_eq!(
            advertised.len(),
            count,
            "{}: two engines share a selection name",
            E::CATEGORY
        );
    }

    #[test]
    fn utr_category_is_coherent() {
        assert_category_is_coherent::<UtrEngine>();
    }

    #[test]
    fn fa_category_is_coherent() {
        assert_category_is_coherent::<FaEngineName>();
    }

    #[test]
    fn translate_category_is_coherent() {
        assert_category_is_coherent::<TranslateEngineName>();
    }

    /// ASR is no longer exempt.
    #[test]
    fn asr_category_is_coherent() {
        assert_category_is_coherent::<AsrEngineName>();
    }

    /// ASR additionally advertises `paraformer`, a selection rather than a
    /// variant, and it must resolve through the CLI's own resolver.
    #[test]
    fn asr_advertises_paraformer_on_top_of_its_engines() {
        let advertised: Vec<&str> = AsrEngineName::selectable_names().collect();
        assert_eq!(advertised.len(), AsrEngineName::ALL.len() + 1);
        assert!(advertised.contains(&PARAFORMER_SELECTION_NAME));
        for name in &advertised {
            assert!(
                AsrEngineName::resolve(name).is_some(),
                "advertised {name} does not resolve"
            );
        }
    }

    /// Every spelling of the Qwen3 aligner reaches the Qwen3 aligner.
    ///
    /// A wire format, so it stays a test: these tokens are persisted in stored
    /// job options and typed on a command line, and no type of ours pins what
    /// a past build wrote or what a user types. The round-trip is the point:
    /// what `as_wire_name` emits must be what `from_wire_name` accepts.
    #[test]
    fn every_spelling_of_qwen3_fa_resolves_to_it() {
        for name in ["qwen3_fa", "qwen3-fa", "qwen3"] {
            assert_eq!(
                FaEngineName::resolve(name),
                Some(FaEngineName::Qwen3),
                "{name} did not resolve to the Qwen3 aligner"
            );
        }
        assert_eq!(FaEngineName::Qwen3.as_wire_name(), "qwen3_fa");
        assert_eq!(
            FaEngineName::from_wire_name(FaEngineName::Qwen3.as_wire_name()).unwrap(),
            FaEngineName::Qwen3
        );
    }

    /// The default aligner must report word intervals, not bare onsets.
    ///
    /// POLICY: an onset-only engine is a legitimate engine, it just cannot be
    /// the default without an onset-to-interval step, because a word's end
    /// would otherwise equal its start. Asserting the PROPERTY rather than a
    /// particular variant means a new interval-reporting engine may become the
    /// default without editing this test, while an onset-only one may not.
    #[test]
    fn default_fa_engine_reports_word_intervals() {
        assert_eq!(
            FaEngineName::DEFAULT.timing_resolution(),
            FaTimingResolution::WordIntervals,
            "the default aligner must report a word's end, not only its start"
        );
    }

    /// Every FA engine reports the shape its model actually produces.
    ///
    /// `Wav2vecCanto` is the case worth pinning: it is a wav2vec model and
    /// returns index-aligned word spans, but its wire name is `cantonese_fa`.
    /// Classification used to be a substring test for "wav2vec" on that name,
    /// so it fell through to the onset-only branch and its word spans were
    /// read as token onsets on the wrong grouping window.
    #[test]
    fn every_fa_engine_reports_the_shape_its_model_produces() {
        for (engine, expected) in [
            (FaEngineName::Wave2Vec, FaTimingResolution::WordIntervals),
            (
                FaEngineName::Wav2vecCanto,
                FaTimingResolution::WordIntervals,
            ),
            // Qwen3's aligner reports both ends of every word, so it is an
            // interval engine. Pinned because the whole reason it is worth
            // exposing standalone is that its durations are measured.
            (FaEngineName::Qwen3, FaTimingResolution::WordIntervals),
            (FaEngineName::Whisper, FaTimingResolution::TokenOnsets),
        ] {
            assert_eq!(
                engine.timing_resolution(),
                expected,
                "{} classified wrongly",
                engine.wire_name()
            );
        }
    }

    /// The default is a member of its own category.
    #[test]
    fn every_default_is_an_engine_that_exists() {
        assert!(UtrEngine::ALL.contains(&UtrEngine::DEFAULT));
        assert!(FaEngineName::ALL.contains(&FaEngineName::DEFAULT));
        assert!(AsrEngineName::ALL.contains(&AsrEngineName::DEFAULT));
        assert!(TranslateEngineName::ALL.contains(&TranslateEngineName::DEFAULT));
    }

    /// The engine roster and the wire backend vocabulary invert each other.
    ///
    /// A ROUNDTRIP between two functions in two crates, which is what a test
    /// is legitimately for: `FaEngineSpec::worker_backend` is a field in
    /// `batchalign`, `FaEngineName::from_worker_backend` is an exhaustive
    /// match over an enum owned by `batchalign-types`, and no signature says
    /// they are inverses. Both directions are checked, so neither a new engine
    /// pointed at an occupied backend nor a new backend routed to the wrong
    /// engine can pass.
    #[test]
    fn fa_engine_and_worker_backend_are_a_bijection() {
        for &engine in FaEngineName::ALL {
            assert_eq!(
                FaEngineName::from_worker_backend(engine.worker_backend()),
                engine,
                "{engine:?} does not survive a round trip through its wire backend"
            );
        }
        for backend in batchalign_types::worker_v2::FaBackendV2::ALL {
            assert_eq!(
                FaEngineName::from_worker_backend(backend).worker_backend(),
                backend,
                "{backend:?} does not survive a round trip through its engine"
            );
        }
    }
}
