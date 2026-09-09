//! Parameter structs and enums that replace long function parameter lists
//! and eliminate boolean blindness.
//!
//! These types group related parameters that are always passed together,
//! reducing function signatures from 10-15 parameters to 3-5.

use std::collections::BTreeSet;
use std::path::Path;

use crate::api::{DurationMs, LanguageCode3};
use crate::chat_ops::CacheTaskName;
use crate::chat_ops::fa::coordinates::{Ms, Recording};
use crate::chat_ops::fa::{AudioIdentity, WordGapHealing};
use crate::chat_ops::morphosyntax_ops::{MultilingualPolicy, MwtDict, TokenizationMode};
use crate::error::ServerError;
use crate::infer_retry::Cancellation;
use crate::types::engines::FaEngineName;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Boolean blindness elimination
// ---------------------------------------------------------------------------

/// Cache lookup policy for audio-task NLP processing (FA, UTR ASR,
/// media analysis). Text-NLP tasks bypass the cache entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePolicy {
    /// Use the cache normally (check for hits, store new results).
    UseCache,
    /// Skip cache lookups (always recompute; still stores results for future use).
    SkipCache,
    /// Require reusable evidence. A miss is an error and cannot authorize
    /// inference.
    RequireCache,
}

/// Whether to generate the `%wor` (word-level timing) dependent tier.
///
/// Replaces the `write_wor: bool` parameter in FA processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "bool", from = "bool")]
#[derive(Default)]
pub enum WorTierPolicy {
    /// Generate `%wor` tier with word-level timing bullets.
    Include,
    /// Omit `%wor` tier from output.
    #[default]
    Omit,
}

impl WorTierPolicy {
    /// Returns `true` when the `%wor` tier should be generated.
    pub fn should_write(&self) -> bool {
        matches!(self, Self::Include)
    }
}

impl From<bool> for WorTierPolicy {
    /// Converts from legacy `write_wor: bool`, `true` means include.
    fn from(write_wor: bool) -> Self {
        if write_wor { Self::Include } else { Self::Omit }
    }
}

impl From<WorTierPolicy> for bool {
    fn from(policy: WorTierPolicy) -> Self {
        policy.should_write()
    }
}

/// Whether abbreviation merging should run before writing output.
///
/// Replaces the `merge_abbrev: bool` option-family flag with an explicit policy
/// that still serializes to the existing boolean wire format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "bool", from = "bool")]
#[derive(Default)]
pub enum MergeAbbrevPolicy {
    /// Merge abbreviated forms into their expanded spellings.
    Merge,
    /// Preserve abbreviated forms as written.
    #[default]
    Keep,
}

impl MergeAbbrevPolicy {
    /// Returns `true` when abbreviation merging should run.
    pub fn should_merge(&self) -> bool {
        matches!(self, Self::Merge)
    }
}

impl From<bool> for MergeAbbrevPolicy {
    /// Converts from legacy `merge_abbrev: bool`, `true` means merge.
    fn from(merge_abbrev: bool) -> Self {
        if merge_abbrev {
            Self::Merge
        } else {
            Self::Keep
        }
    }
}

impl From<MergeAbbrevPolicy> for bool {
    fn from(policy: MergeAbbrevPolicy) -> Self {
        policy.should_merge()
    }
}

/// Operator opt-in to the legacy Stanza constituency-parser fallback
/// for utterance segmentation when no language-specific TalkBank BERT
/// utseg model is configured for the requested language.
///
/// The default is [`Refuse`], which mirrors the
/// `WhisperHubModelNotFoundError` pattern: silent substitution of one
/// model for another is the foot-gun this enum exists to prevent.
/// Operators who want the previous behavior pass
/// `--utseg-fallback-stanza` on the CLI; this is surfaced on every
/// utseg-invoking subcommand (transcribe, transcribe-s, utseg).
///
/// Serialized as a plain JSON `bool` (`#[serde(into = "bool", from
/// = "bool")]`) so persisted job documents that predate this field
/// (`#[serde(default)]` on the surrounding option structs) load as
/// `Refuse` without a schema migration.
///
/// [`Refuse`]: UtsegFallbackPolicy::Refuse
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "bool", from = "bool")]
#[derive(Default)]
pub enum UtsegFallbackPolicy {
    /// Refuse to segment when no language-specific BERT model is
    /// configured; raise a typed error pointing the user at
    /// `--utseg-fallback-stanza` and at the resolver-entry workflow.
    #[default]
    Refuse,
    /// Use Stanza constituency parsing as the segmenter for languages
    /// without a BERT model. Quality varies by Stanza language pack.
    AllowStanza,
}

impl UtsegFallbackPolicy {
    /// Returns `true` when the Stanza fallback should be permitted.
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::AllowStanza)
    }
}

impl From<bool> for UtsegFallbackPolicy {
    /// Converts from the CLI `--utseg-fallback-stanza` flag, `true`
    /// means the operator has opted in to the Stanza substitution.
    fn from(allow: bool) -> Self {
        if allow {
            Self::AllowStanza
        } else {
            Self::Refuse
        }
    }
}

impl From<UtsegFallbackPolicy> for bool {
    fn from(policy: UtsegFallbackPolicy) -> Self {
        policy.is_allowed()
    }
}

// ---------------------------------------------------------------------------
// Fine-grained cache overrides
// ---------------------------------------------------------------------------

/// Typed cache override policy supporting per-task granularity.
///
/// Replaces the binary `override_media_cache: bool` with a richer model
/// that avoids boolean blindness and enables experiment-grade control.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum CacheOverrides {
    /// Use cache normally for all tasks (default).
    #[default]
    None,
    /// Skip cache for all tasks (`--override-media-cache`).
    All,
    /// Skip cache only for listed tasks (`--override-media-cache-tasks`).
    Tasks(BTreeSet<CacheTaskName>),
    /// Require reusable evidence for every cache-backed task
    /// (`--require-media-cache`).
    RequireAll,
}

impl CacheOverrides {
    /// Resolve the effective cache policy for a specific task.
    pub fn policy_for(&self, task: CacheTaskName) -> CachePolicy {
        match self {
            Self::None => CachePolicy::UseCache,
            Self::All => CachePolicy::SkipCache,
            Self::RequireAll => CachePolicy::RequireCache,
            Self::Tasks(set) => {
                if set.contains(&task) {
                    CachePolicy::SkipCache
                } else {
                    CachePolicy::UseCache
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Parameter grouping structs
// ---------------------------------------------------------------------------

/// Morphosyntax-specific processing parameters.
///
/// Groups the 5 morphosyntax-specific parameters that are always passed
/// together through `process_morphosyntax`, `process_morphosyntax_batch`,
/// and `process_morphosyntax_incremental`.
pub struct MorphosyntaxParams<'a> {
    /// ISO 639-3 language code (e.g., `"eng"`, `"spa"`).
    pub lang: &'a LanguageCode3,
    /// How to handle Stanza retokenization during injection.
    pub tokenization_mode: TokenizationMode,
    /// How to handle non-primary languages in multilingual files.
    pub multilingual_policy: MultilingualPolicy,
    /// Multi-word token lexicon for retokenization overrides.
    pub mwt: &'a MwtDict,
    /// Analysis policy resolved at the command boundary.
    pub policy: MorphotagExecutionPolicy,
    /// Legacy review-tier request retained for wire compatibility. No value
    /// emits `%xalign` or `%xrev`; structured evidence retains the decisions.
    ///
    /// [`ReviewLevel::None`]: crate::chat_ops::fa::ReviewLevel::None
    pub review_level: crate::chat_ops::fa::ReviewLevel,
    /// Where to report backend progress for this file, if anyone is listening.
    ///
    /// `Some` only on the job path, where a `BatchProgressReporter` exists to
    /// aggregate and publish; `None` for the CLI's direct path, `compare`'s
    /// internal morphotag, and tests. Carried in the params bundle rather than
    /// as another positional argument because it travels the same route as
    /// every other per-file morphotag input.
    /// `pub(crate)`, unlike its neighbours: the port is internal wiring between
    /// the dispatcher and the worker call, not part of any published surface.
    pub(crate) progress: Option<&'a crate::execution::morphotag::progress::BackendProgressPort>,
    /// The job's cancellation token, when this dispatch has one.
    ///
    /// Carried in the params bundle rather than as another positional
    /// argument, same rationale as `progress`: it travels the same route
    /// as every other per-file morphotag input, all the way down to
    /// `infer_batch`'s retry loop.
    pub cancellation: Cancellation<'a>,
}

/// Runtime policy choices that travel together through morphotag processing.
///
/// Keeping these choices in one value prevents each new policy from widening
/// every pipeline parameter bundle independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MorphotagExecutionPolicy {
    /// Treatment of `@s` words.
    pub l2: L2MorphotagPolicy,
    /// Treatment of transcriber-supplied `$POS` hints.
    pub pos_hints: PosHintPolicy,
    /// Policy for a parsed transcript carrying `@Options: CA`.
    pub ca_policy: crate::options::CaMorphotagPolicy,
}

/// Runtime treatment of `@s` code-switched words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum L2MorphotagPolicy {
    /// Route words to the secondary-language Stanza model.
    Analyze,
    /// Preserve the legacy `L2|xxx` placeholder behavior.
    Placeholder,
}

impl L2MorphotagPolicy {
    /// Whether secondary-language Stanza inference should run.
    pub fn should_analyze(self) -> bool {
        matches!(self, Self::Analyze)
    }
}

/// Runtime treatment of transcriber-supplied `$POS` hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PosHintPolicy {
    /// Apply hints after Stanza and any secondary-language splicing.
    Honor,
    /// Leave Stanza's POS categories unchanged.
    Ignore,
}

impl PosHintPolicy {
    /// Whether the post-Stanza hint pass should run.
    pub fn should_apply(self) -> bool {
        matches!(self, Self::Honor)
    }
}

impl From<bool> for PosHintPolicy {
    fn from(honor: bool) -> Self {
        if honor { Self::Honor } else { Self::Ignore }
    }
}

impl From<bool> for L2MorphotagPolicy {
    fn from(analyze: bool) -> Self {
        if analyze {
            Self::Analyze
        } else {
            Self::Placeholder
        }
    }
}

/// Audio file context for forced alignment and transcription.
///
/// Groups the audio-related parameters that are always passed together
/// to FA processing functions.
pub struct AudioContext<'a> {
    /// Path to the audio file on disk.
    pub audio_path: &'a Path,
    /// Content-based identity for cache keying (hash of audio content).
    pub audio_identity: &'a AudioIdentity,
    /// Total duration of the audio file in milliseconds, if known.
    pub total_audio_ms: Option<DurationMs>,
}

impl AudioContext<'_> {
    /// The recording this context describes, probing when the pipeline did not
    /// already carry its duration.
    ///
    /// # Why this exists rather than each caller unwrapping the `Option`
    ///
    /// `total_audio_ms` being optional made "we do not know how long the audio
    /// is" a legal state, and each consumer invented its own answer for it: FA
    /// grouping skipped untimed utterances, the final window declined to
    /// extend, and the timing conversion simply went unbounded, which is how
    /// word timings came to be written 28.2 seconds past the end of a
    /// recording. None of those is a better answer than the others, because
    /// there is no good behaviour for an unknown duration.
    ///
    /// Forced alignment always has an audio file and the engine must read the
    /// same bytes, so the duration is always obtainable. This makes the
    /// question cheap to answer in one place instead of unanswerable in
    /// several. Being the single owner is the point: three call sites deriving
    /// this independently is the duplication that produced the divergent
    /// fallbacks.
    pub async fn recording(&self) -> Result<Recording, ServerError> {
        let duration = match self.total_audio_ms {
            Some(known) => known,
            None => crate::media::probe::MediaProbe::new(self.audio_path)
                .duration()
                .await
                .map_err(|why| {
                    ServerError::RecordingDuration(format!(
                        "probing {} failed: {why}",
                        self.audio_path.display()
                    ))
                })?,
        };
        Recording::of_duration(Ms(duration.0)).map_err(|why| {
            ServerError::RecordingDuration(format!("{}: {why}", self.audio_path.display()))
        })
    }
}

/// Forced alignment processing parameters.
///
/// Groups the 5 FA-specific parameters that are always passed together
/// through `run_fa_from_ast` and `process_fa_incremental`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaParams {
    /// Whether post-processing heals small gaps between words (`--pauses`
    /// selects `PreserveMeasured`).
    pub gap_healing: WordGapHealing,
    /// How admitted word evidence treats boundaries inherited from prior `%wor`.
    pub existing_wor_boundaries: crate::chat_ops::fa::ExistingWorBoundaryPolicy,
    /// How adjacent utterance end overlap is projected.
    pub end_overlap_policy: crate::chat_ops::fa::EndOverlapPolicy,
    /// Which FA engine to use: the engine itself, never a strategy derived
    /// from it. See [`FaEngineName`] for why that distinction matters.
    pub engine: FaEngineName,
    /// Cache lookup policy.
    pub cache_policy: CachePolicy,
    /// Whether to generate `%wor` tier.
    pub wor_tier: WorTierPolicy,
    /// Whether to apply post-FA bullet repair.
    pub bullet_repair: bool,
    /// Review tier verbosity.
    pub review_level: crate::chat_ops::fa::ReviewLevel,
}

impl FaParams {
    /// What post-processing may do to a word's end, for THIS run's engine.
    ///
    /// Derived rather than stored: the engine is already here, so the pairing
    /// cannot disagree with it.
    pub fn word_end_policy(&self) -> crate::chat_ops::fa::WordEndPolicy {
        crate::chat_ops::fa::WordEndPolicy::for_engine(self.engine, self.gap_healing)
    }

    /// Complete downstream projection policy for this run.
    pub fn projection_policy(&self) -> crate::chat_ops::fa::FaProjectionPolicy {
        crate::chat_ops::fa::FaProjectionPolicy::new(
            self.word_end_policy(),
            self.existing_wor_boundaries,
            self.end_overlap_policy,
        )
    }

    /// Longest audio window handed to this run's engine in one dispatch.
    ///
    /// Derived rather than stored, so it cannot disagree with the engine
    /// beside it. It was a field until 2026-08-14, which meant the value could
    /// be set from one engine and read beside another, and made the test that
    /// checked them equal a test of nothing.
    pub fn max_group_ms(&self) -> DurationMs {
        self.engine.max_group_ms()
    }
}

#[cfg(test)]
mod tests {
    use super::{CacheOverrides, CachePolicy, MergeAbbrevPolicy, WorTierPolicy};

    #[test]
    fn wor_tier_policy_uses_bool_wire_format() {
        assert_eq!(
            serde_json::to_string(&WorTierPolicy::Include).unwrap(),
            "true"
        );
        assert_eq!(
            serde_json::to_string(&WorTierPolicy::Omit).unwrap(),
            "false"
        );
        assert_eq!(
            serde_json::from_str::<WorTierPolicy>("true").unwrap(),
            WorTierPolicy::Include
        );
        assert_eq!(
            serde_json::from_str::<WorTierPolicy>("false").unwrap(),
            WorTierPolicy::Omit
        );
    }

    #[test]
    fn cache_overrides_none_uses_cache_for_all() {
        let overrides = CacheOverrides::None;
        assert_eq!(
            overrides.policy_for(crate::chat_ops::CacheTaskName::UtrAsr),
            CachePolicy::UseCache,
        );
        assert_eq!(
            overrides.policy_for(crate::chat_ops::CacheTaskName::ForcedAlignment),
            CachePolicy::UseCache,
        );
    }

    #[test]
    fn cache_overrides_all_skips_cache_for_all() {
        let overrides = CacheOverrides::All;
        assert_eq!(
            overrides.policy_for(crate::chat_ops::CacheTaskName::ForcedAlignment),
            CachePolicy::SkipCache,
        );
        assert_eq!(
            overrides.policy_for(crate::chat_ops::CacheTaskName::UtrAsr),
            CachePolicy::SkipCache,
        );
    }

    #[test]
    fn cache_overrides_tasks_selective() {
        use std::collections::BTreeSet;
        let mut tasks = BTreeSet::new();
        tasks.insert(crate::chat_ops::CacheTaskName::UtrAsr);
        let overrides = CacheOverrides::Tasks(tasks);

        assert_eq!(
            overrides.policy_for(crate::chat_ops::CacheTaskName::UtrAsr),
            CachePolicy::SkipCache,
        );
        assert_eq!(
            overrides.policy_for(crate::chat_ops::CacheTaskName::ForcedAlignment),
            CachePolicy::UseCache,
        );
    }

    #[test]
    fn cache_overrides_default_is_none() {
        assert_eq!(CacheOverrides::default(), CacheOverrides::None);
    }

    #[test]
    fn merge_abbrev_policy_uses_bool_wire_format() {
        assert_eq!(
            serde_json::to_string(&MergeAbbrevPolicy::Merge).unwrap(),
            "true"
        );
        assert_eq!(
            serde_json::to_string(&MergeAbbrevPolicy::Keep).unwrap(),
            "false"
        );
        assert_eq!(
            serde_json::from_str::<MergeAbbrevPolicy>("true").unwrap(),
            MergeAbbrevPolicy::Merge
        );
        assert_eq!(
            serde_json::from_str::<MergeAbbrevPolicy>("false").unwrap(),
            MergeAbbrevPolicy::Keep
        );
    }
}
