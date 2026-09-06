//! Forced alignment orchestration for CHAT files.
//!
//! Extracts pure-Rust FA logic from the PyO3 bridge (`batchalign-core`) so that
//! both the PyO3 layer and the root Rust workspace can share it.
//!
//! Pipeline: parse -> group utterances -> dispatch inference -> parse responses
//! -> inject timings -> postprocess -> generate %wor -> enforce monotonicity/E704.

pub mod alignment;
pub mod coordinates;
mod expand_for_fillers;
mod extraction;
mod grouping;
mod injection;
mod orchestrate;
pub mod origin;
pub mod outcome;
mod postprocess;
pub mod repair;
mod rescue_narrow_bullets;
pub mod speech_rate;
pub mod timing;
pub mod utr;

#[cfg(test)]
mod tests;

use self::origin::Origin;
use crate::types::engines::FaTimingResolution;
use serde::{Deserialize, Serialize};
use talkbank_model::alignment::CompleteWorTimingSlot;
use talkbank_model::alignment::helpers::{TierDomain, WordItem, counts_for_tier, walk_words};
use talkbank_model::model::{
    Bullet, ChatFile, DependentTier, Line, Utterance, UtteranceContent, Word, WordCategory,
};
use talkbank_model::{UtteranceIdx, WordIdx};

// Re-export public API so that `crate::chat_ops::fa::Foo` paths remain unchanged.
pub use self::alignment::parse_fa_response;
pub use self::expand_for_fillers::expand_bullets_for_edge_fillers;
pub use self::extraction::collect_fa_words;
pub use self::grouping::{
    Estimates, FaGroup, Grouping, Placement, WHISPER_FA_MAX_LABEL_TOKENS, count_utterance_timing,
    estimate_untimed_boundaries, group_utterances,
};
pub use self::injection::inject_timings_for_utterance;
pub(crate) use self::orchestrate::WrittenFaDecisions;
pub use self::orchestrate::{
    DroppedWordTiming, FaApplied, FaDecisions, FaFinalized, MonotonicityEffect, MonotonicityResult,
    OverlapEdge, WordTier, apply_fa_results, apply_fa_results_with_projection_policy,
    enforce_monotonicity, enforce_monotonicity_with_policy, finalize_without_injection,
    has_reusable_wor_timing_for_utterance, projection_without_injection_with_touched,
    refresh_existing_alignment_for_utterance, refresh_reusable_alignment,
    refresh_reusable_utterances, retain_decision_evidence, strip_e704_same_speaker_overlaps,
    strip_timing_from_content, strip_wor_from_monotonicity_stripped_utterances,
};
// `#[cfg(test)]` (2026-09-01 review, item 12): both names are test-fixture
// convenience only, with no production caller; see their own doc comments
// in `orchestrate.rs`.
#[cfg(test)]
pub use self::orchestrate::{
    refresh_existing_alignment, refresh_existing_alignment_with_boundary_policy,
};
pub use self::postprocess::{
    postprocess_utterance_timings, postprocess_utterance_timings_with_boundary_policy,
};
pub use self::repair::{
    BulletRepairPolicy, RepairDecision, RepairResult, RepairStats, repair_bullets,
};
pub use self::rescue_narrow_bullets::rescue_narrow_bullets;
// `ReviewLevel` remains on the wire surface for stored-job compatibility. It
// does not reach CHAT serialization; `retain_decision_evidence` always strips
// the two abandoned review tiers and returns typed evidence.
pub use self::utr::{
    CaMarkerPolicy, GlobalUtr, GroupingContext, TwoPassConfig, TwoPassOverlapUtr,
    UtrFuzzyThreshold, UtrMatchMode, UtrOverlapDensityThreshold, UtrStrategy, find_untimed_windows,
    select_strategy, utr_asr_cache_key, utr_asr_segment_cache_key,
};
pub use batchalign_transform::decisions::ReviewLevel;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A time interval in milliseconds.
///
/// NOT guaranteed `start <= end`: [`Self::new`] accepts any pair. Callers that
/// require a real extent must check.
///
/// The constructor cannot enforce it because post-processing holds spans whose
/// extent is not resolved yet, and those are legitimate. [`WordTiming`] is the
/// other half of that split: it is what a settled extent looks like, and it is
/// the only thing a word's bullet can be written from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeSpan {
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
}

impl TimeSpan {
    /// Create a new time span. Caller is responsible for ensuring start <= end.
    pub fn new(start_ms: u64, end_ms: u64) -> Self {
        Self { start_ms, end_ms }
    }

    /// Duration in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// A word's timing, guaranteed to cover a positive extent.
///
/// [`TimeSpan`] deliberately carries no invariant, because post-processing
/// works with spans whose extent is not settled yet. This is the type that
/// leaves that computation: a word occupies time, so `end > start` holds for
/// every value that reaches a `%wor` tier or the cache.
///
/// The invariant is enforced at CONSTRUCTION, which is the only way in;
/// `TimeSpan::new` accepts any pair and always did. Reads are unchanged: this
/// derefs to the span it wraps.
///
/// `postprocess_continuous::a_word_is_not_healed_backwards_into_an_earlier_neighbour`
/// is the executable account of why this type exists.
/// # Every timing says how it was made
///
/// The `origin` is not decoration and not a log line: it is the answer to "was
/// this measured, or did we compute it?", and it is the question a corpus
/// consumer most needs and could least ask. On one merged corpus roughly 37% of
/// timings were interpolated rather than measured, indistinguishable in the
/// output, so a reference comparison reported 37.2% agreement by time against
/// 76.4% by text: the gap was almost entirely our own arithmetic being scored as
/// observation.
///
/// It is carried on the value rather than beside it because anything beside it
/// is a second thing to keep true, and the two drift. There is deliberately no
/// constructor that omits it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "WordTimingWire", into = "WordTimingWire")]
pub struct WordTiming {
    span: TimeSpan,
    /// Optional score emitted by the alignment model for this word.
    ///
    /// This is deliberately not called a probability or accuracy confidence:
    /// MMS_FA supplies a CTC path score, which still needs empirical
    /// calibration against human boundary judgments.
    model_score: Option<ModelAlignmentScore>,
    /// How the word's START was produced.
    start_origin: Origin,
    /// How the word's END was produced.
    ///
    /// Separate from the start's because a word has TWO boundaries and they are
    /// routinely produced differently. On an onset-only engine the start is
    /// MEASURED (it is the token onset) and the end is inferred from the next
    /// word; carrying one origin for both reported such a word as wholly
    /// derived, understating what was actually observed by half. `WordSpan`
    /// already modelled both ends correctly and the lowering here threw one
    /// away.
    end_origin: Origin,
}

/// The stored form of a [`WordTiming`].
///
/// Exists so the extent invariant is checked on the way IN from the cache, not
/// only at construction. Adding the `origin` field replaced a
/// `#[serde(try_from = "TimeSpan")]` with a plain derive, which silently
/// dropped that check: a stored `{"span":{"start_ms":5,"end_ms":5}}` began
/// deserializing into a zero-width timing again, and the docstring below still
/// claimed it could not. A wire type keeps the two facts together, so the
/// invariant holds at every boundary the value crosses.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WordTimingWire {
    span: TimeSpan,
    #[serde(default)]
    model_score: Option<ModelAlignmentScore>,
    start_origin: Origin,
    end_origin: Origin,
}

impl From<WordTiming> for WordTimingWire {
    fn from(timing: WordTiming) -> Self {
        Self {
            span: timing.span,
            model_score: timing.model_score,
            start_origin: timing.start_origin,
            end_origin: timing.end_origin,
        }
    }
}

impl TryFrom<WordTimingWire> for WordTiming {
    type Error = DegenerateWordTiming;

    fn try_from(wire: WordTimingWire) -> Result<Self, Self::Error> {
        Self::new(
            wire.span.start_ms,
            wire.span.end_ms,
            wire.start_origin,
            wire.end_origin,
        )
        .map(|timing| timing.with_model_score_option(wire.model_score))
        .ok_or(DegenerateWordTiming {
            start_ms: wire.span.start_ms,
            end_ms: wire.span.end_ms,
        })
    }
}

impl WordTiming {
    /// A timing for a word that occupies `start_ms..end_ms`, produced as
    /// `origin` describes.
    ///
    /// `None` when the pair does not describe a positive extent. A caller that
    /// cannot build one has learned something real: the engine gave no usable
    /// timing for that word, and the honest result is no timing rather than a
    /// zero-length or backwards one.
    pub fn new(
        start_ms: u64,
        end_ms: u64,
        start_origin: Origin,
        end_origin: Origin,
    ) -> Option<Self> {
        (end_ms > start_ms).then_some(Self {
            span: TimeSpan::new(start_ms, end_ms),
            model_score: None,
            start_origin,
            end_origin,
        })
    }

    /// Attach the model's own alignment score to this otherwise valid timing.
    pub fn with_model_score(mut self, score: ModelAlignmentScore) -> Self {
        self.model_score = Some(score);
        self
    }

    fn with_model_score_option(mut self, score: Option<ModelAlignmentScore>) -> Self {
        self.model_score = score;
        self
    }

    /// The aligner's score for this word, when its response shape supplies one.
    pub const fn model_score(&self) -> Option<ModelAlignmentScore> {
        self.model_score
    }

    /// How the word's start was produced.
    pub fn start_origin(&self) -> &Origin {
        &self.start_origin
    }

    /// How the word's end was produced.
    pub fn end_origin(&self) -> &Origin {
        &self.end_origin
    }

    /// A timing read from a span already present in the transcript.
    ///
    /// Replaces the old `TryFrom<TimeSpan>`, which could turn any pair of
    /// integers into a timing with no statement of where they came from. Naming
    /// the route makes the provenance automatic: a span lifted off a `%wor`
    /// tier is an observation, because a human or an earlier run put it there,
    /// and it is not one this run measured.
    pub fn from_transcript(span: TimeSpan) -> Result<Self, DegenerateWordTiming> {
        Self::new(
            span.start_ms,
            span.end_ms,
            Origin::TranscriptBullet,
            Origin::TranscriptBullet,
        )
        .ok_or(DegenerateWordTiming {
            start_ms: span.start_ms,
            end_ms: span.end_ms,
        })
    }

    /// A timing lifted from a lexically corroborated, complete `%wor` slot.
    ///
    /// Accepting the complete-state slot, rather than two raw integers, makes
    /// this conversion infallible: Chatter has already proved that the
    /// interval is present and positive and that it belongs to the matching
    /// main-tier word.
    pub fn from_complete_wor_slot(slot: &CompleteWorTimingSlot<'_>) -> Self {
        let interval = slot.timing();
        Self {
            span: TimeSpan::new(interval.start().get(), interval.end().get()),
            model_score: None,
            start_origin: Origin::TranscriptBullet,
            end_origin: Origin::TranscriptBullet,
        }
    }
}

/// A finite model-emitted forced-alignment score in the closed interval 0..=1.
///
/// Stored as millionths so timings retain `Eq` and cache comparisons remain
/// deterministic. The quantization is far finer than the model evidence can
/// justify; it is storage discipline, not an accuracy claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "u32", into = "u32")]
pub struct ModelAlignmentScore(u32);

impl ModelAlignmentScore {
    const SCALE: f64 = 1_000_000.0;

    /// Construct from a model score, rejecting non-finite or out-of-range data.
    pub fn try_from_f64(score: f64) -> Result<Self, InvalidModelAlignmentScore> {
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            return Err(InvalidModelAlignmentScore { score });
        }
        Ok(Self((score * Self::SCALE).round() as u32))
    }

    /// Stable integer representation used by evidence artifacts.
    pub const fn millionths(self) -> u32 {
        self.0
    }

    /// Floating representation for summaries and thresholds.
    pub fn as_f64(self) -> f64 {
        f64::from(self.0) / Self::SCALE
    }

    pub(crate) fn from_weighted_millionths(
        weighted_millionths: u128,
        duration_ms: u128,
    ) -> Option<Self> {
        if duration_ms == 0 {
            return None;
        }
        let rounded = (weighted_millionths + duration_ms / 2) / duration_ms;
        let millionths = u32::try_from(rounded).ok()?;
        (millionths <= 1_000_000).then_some(Self(millionths))
    }
}

impl TryFrom<u32> for ModelAlignmentScore {
    type Error = InvalidStoredModelAlignmentScore;

    fn try_from(millionths: u32) -> Result<Self, Self::Error> {
        (millionths <= 1_000_000)
            .then_some(Self(millionths))
            .ok_or(InvalidStoredModelAlignmentScore { millionths })
    }
}

impl From<ModelAlignmentScore> for u32 {
    fn from(score: ModelAlignmentScore) -> Self {
        score.millionths()
    }
}

/// A worker supplied a value that cannot be an alignment score.
#[derive(Debug, thiserror::Error)]
#[error("model alignment score must be finite and between 0 and 1, got {score}")]
pub struct InvalidModelAlignmentScore {
    score: f64,
}

/// Persisted millionths that fall outside the model score's closed interval.
#[derive(Debug, thiserror::Error)]
#[error("stored model alignment score must be at most 1000000 millionths, got {millionths}")]
pub struct InvalidStoredModelAlignmentScore {
    millionths: u32,
}

#[cfg(test)]
mod model_alignment_score_tests {
    use super::ModelAlignmentScore;

    #[test]
    fn persisted_score_rejects_values_above_the_closed_interval() {
        let error = serde_json::from_str::<ModelAlignmentScore>("1000001")
            .expect_err("cached data must not bypass the score invariant");

        assert!(error.to_string().contains("at most 1000000"));
    }

    #[test]
    fn persisted_score_round_trips_at_both_interval_edges() {
        for millionths in [0, 1_000_000] {
            let score = ModelAlignmentScore::try_from(millionths).expect("valid edge");
            let json = serde_json::to_string(&score).expect("serialize score");
            assert_eq!(
                serde_json::from_str::<ModelAlignmentScore>(&json).expect("deserialize score"),
                score
            );
        }
    }
}

#[cfg(test)]
impl WordTiming {
    /// A timing for a test fixture.
    ///
    /// Stands for a span the transcript already carried, which is what almost
    /// every fixture here means: the test is exercising a LATER pass and needs
    /// some prior timing to feed it. Tests that care about provenance call
    /// [`WordTiming::new`] with the origin they mean, and a few do.
    ///
    /// Test-only on purpose. Production code cannot reach it, so no real timing
    /// can acquire a provenance nobody chose; that is the difference between a
    /// convenience and a hole in the proof.
    pub(crate) fn fixture(start_ms: u64, end_ms: u64) -> Option<Self> {
        Self::new(
            start_ms,
            end_ms,
            Origin::TranscriptBullet,
            Origin::TranscriptBullet,
        )
    }
}

impl std::ops::Deref for WordTiming {
    type Target = TimeSpan;

    fn deref(&self) -> &Self::Target {
        &self.span
    }
}

impl From<WordTiming> for TimeSpan {
    fn from(timing: WordTiming) -> Self {
        timing.span
    }
}

/// Rejects a stored span that does not describe a positive extent.
///
/// This is what makes the FA cache self-cleaning: an entry holding
/// zero-duration timings fails to deserialize (via `WordTimingWire`), the read
/// path treats that as a miss, and the group is recomputed. The same mechanism
/// retires entries written before timings carried an origin, which is why
/// adding that field needed no migration: an old entry lacks `origin`, fails to
/// deserialize, and simply misses.
#[derive(Debug, thiserror::Error)]
#[error("a word timing must cover a positive extent, got {start_ms}..{end_ms}")]
pub struct DegenerateWordTiming {
    /// Start of the rejected span.
    pub start_ms: u64,
    /// End of the rejected span.
    pub end_ms: u64,
}

/// Split a compound filler's cleaned text at underscores, or return the
/// text as a single element. Only applies to `WordCategory::Filler` words;
/// regular compounds are unchanged.
///
/// Both extraction and injection must agree on the split count, extraction
/// sends N parts to FA, injection consumes N timings from the cursor. This
/// shared helper is the single source of truth for the splitting rule.
pub fn split_compound_filler(word: &talkbank_model::model::Word) -> Vec<String> {
    use talkbank_model::model::WordCategory;
    let text = word.cleaned_text();
    if word.category == Some(WordCategory::Filler) && text.contains('_') {
        text.split('_')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    } else {
        vec![text.to_string()]
    }
}

/// A word extracted for forced alignment, with its position in the AST.
#[derive(Debug, Clone)]
pub struct FaWord {
    /// Index of the utterance in the file (among utterances only).
    pub utterance_index: UtteranceIdx,
    /// Index among alignable words within the utterance.
    pub utterance_word_index: WordIdx,
    /// Cleaned text for the FA model.
    pub text: String,
}

impl FaWord {
    /// Stable word identifier for callback protocols.
    pub fn stable_id(&self) -> String {
        format!("u{}:w{}", self.utterance_index, self.utterance_word_index)
    }
}

/// Duration given to a word whose end nothing could measure or derive.
///
/// The last word of a group has no successor onset, and an utterance may have
/// no bullet to borrow an end from. It is a stand-in for an unmeasured
/// quantity, which is why it is named once here rather than written as a bare
/// number at each of the places that need it.
pub const LAST_WORD_FALLBACK_MS: u64 = 500;

/// Whether FA post-processing heals small gaps between consecutive words.
///
/// This used to be `FaTimingMode`, a name that covered two unrelated
/// decisions: how the token-level parser derived a word's END, and whether
/// the gap-healing pass ran. One flag drove both, so asking for one asked for
/// the other. That is how `--fa-engine whisper` without `--pauses` came to
/// request word ends set to their own starts, which is the shape of a `%wor`
/// tier full of 0-duration words.
///
/// The parser no longer has a choice to make: an onset-only engine reports no
/// end, so the next word's onset is the only end available, and it always uses
/// it. What remains is genuinely a policy, and this type names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WordGapHealing {
    /// Extend a word to the next word's start when the gap between them is
    /// small and plausible, so words read as contiguous speech.
    ///
    /// See `postprocess::postprocess_utterance_timings` for the plausibility
    /// caps; the healing is bounded, not unconditional.
    Heal,
    /// Leave each word ending where the engine said it ended, so silence
    /// between words survives into `%wor`.
    PreserveMeasured,
}

/// How an alignment projection treats main-tier boundaries from a prior `%wor` run.
///
/// This policy changes only the projection of admitted raw evidence into CHAT;
/// it does not participate in the raw FA cache key and cannot authorize model
/// inference.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ExistingWorBoundaryPolicy {
    /// Keep compatibility with established rerun behavior by clamping fresh
    /// word timings to the previous authoritative utterance bullet.
    #[default]
    Preserve,
    /// Treat the prior `%wor` and main bullet as revisable projections: retain
    /// admitted word extents, then rebuild the main bullet from their hull.
    RebuildFromEvidence,
}

/// How monotonicity projection treats an earlier utterance end that crosses
/// the following utterance's start.
///
/// This is downstream projection policy. It never changes the raw FA evidence
/// or its cache key. Cross-speaker overlap can be ordinary conversation, while
/// same-speaker overlap still indicates incompatible segmentation or timing.
///
/// # Why there is no "flag the overlap instead of cutting it" policy
///
/// Assessed 2026-09-02 and recorded here so it is not re-derived: when
/// `EndOverlapResolution::InterleavedWords` cuts a measured word, the natural
/// alternative is to keep the timing and mark the overlap in the transcript.
/// No typed CHAT construct expresses that, for two independent reasons.
///
/// First, the discarded timings live on `%wor`, and a `%wor` tier is a flat
/// list of words and tag-marker separators: it carries no annotations,
/// groups or events at all, so the slot whose timing is dropped has nowhere
/// to say why. Second, CHAT's overlap annotations (`[<]` and `[>]`, typed in
/// the model as scoped content annotations on a main-tier word) assert
/// simultaneous speech by two DIFFERENT speakers. Under the default policy
/// this resolution only ever fires on a pair sharing one speaker code, and a
/// speaker overlapping themself is exactly what the validator's
/// speaker-self-overlap rule exists to reject. Writing the marker there would
/// state something the recording does not support.
///
/// So the cut stays, and the information it removes is preserved where it can
/// be: every discarded extent is recorded in the run's evidence artifact (see
/// the `dropped_word_timings` section of the FA timeline trace), rather than
/// surviving only as a count. An untimed `%wor` slot honestly says "we do not
/// know when this was said", which is true when segmentation and alignment
/// disagree, and is better than a timing asserting an answer we know is wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum EndOverlapPolicy {
    /// Clamp every adjacent overlap regardless of speaker. Legacy
    /// compatibility behavior; NOT the default (see
    /// [`DEFAULT_END_OVERLAP_POLICY`]) because it also clamps ordinary
    /// conversational overlap between two different speakers.
    ClampAllAdjacent,
    /// Preserve cross-speaker overlap, while retaining the existing clamp for
    /// adjacent utterances carrying the same speaker code.
    PreserveCrossSpeaker,
}

/// The one owner of `EndOverlapPolicy`'s default (2026-09-01 review, item 8).
///
/// `EndOverlapPolicy` deliberately does NOT derive `Default`: a `Default` on
/// this enum is exactly the affordance CLAUDE.md's practice 14 warns about,
/// and it was two independent, silently-agreeing-by-luck literals before this
/// constant existed (the CLI's `default_value_t` reading the derive, and
/// `AlignBoundaryOptions`'s own derived `Default` reading it too), so a plain
/// `batchalign3 align` clamped every adjacent SAME- or CROSS-speaker overlap.
/// Cross-speaker overlap is ordinary conversation (see the LIS-removal
/// comment on `fa/repair.rs`'s `find_boundary_averages`), so the default
/// leaves it alone. Every construction site that needs "the default", not an
/// explicit deliberate choice, reads this constant; `rg
/// 'EndOverlapPolicy::PreserveCrossSpeaker\b'` should show only this
/// definition and this crate's own compatibility-shim call sites (which name
/// `ClampAllAdjacent` explicitly, on purpose, for old behavior).
pub const DEFAULT_END_OVERLAP_POLICY: EndOverlapPolicy = EndOverlapPolicy::PreserveCrossSpeaker;

/// `serde(default = "...")` needs a function, not a const path; this is that
/// function, reading the one constant so a deserialized job that omits the
/// field gets the same default every other caller does.
pub fn default_end_overlap_policy_serde() -> EndOverlapPolicy {
    DEFAULT_END_OVERLAP_POLICY
}

impl EndOverlapPolicy {
    fn should_clamp(self, earlier_speaker: &str, next_speaker: &str) -> bool {
        match self {
            Self::ClampAllAdjacent => true,
            Self::PreserveCrossSpeaker => earlier_speaker == next_speaker,
        }
    }
}

/// What post-processing may do to a word's end time.
///
/// Two independent facts that must describe the SAME run: whether the user
/// asked for gaps to be healed, and whether this engine measured word ends or
/// had them derived from the next onset. They were separate arguments until
/// 2026-08-14, which type-checked happily with one engine's healing policy and
/// another engine's resolution.
///
/// Constructed where both facts are already in hand, so a consumer cannot pair
/// them wrongly: [`crate::types::params::FaParams::word_end_policy`] derives
/// the resolution from the engine it already holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WordEndPolicy {
    gap_healing: WordGapHealing,
    resolution: crate::types::engines::FaTimingResolution,
}

/// Complete policy for projecting one run's raw FA evidence into CHAT.
///
/// Keeping these decisions in one value prevents the full and incremental
/// paths from applying different interpretations to the same evidence. This
/// type deliberately contains no cache or inference policy: projection is a
/// downstream transformation and must never change the raw-evidence key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaProjectionPolicy {
    word_ends: WordEndPolicy,
    existing_wor_boundaries: ExistingWorBoundaryPolicy,
    end_overlaps: EndOverlapPolicy,
}

impl FaProjectionPolicy {
    /// Construct a complete projection policy.
    pub fn new(
        word_ends: WordEndPolicy,
        existing_wor_boundaries: ExistingWorBoundaryPolicy,
        end_overlaps: EndOverlapPolicy,
    ) -> Self {
        Self {
            word_ends,
            existing_wor_boundaries,
            end_overlaps,
        }
    }

    /// Word-end and gap-healing policy for this projection.
    pub fn word_ends(self) -> WordEndPolicy {
        self.word_ends
    }

    /// Treatment of boundaries inherited from an earlier `%wor` projection.
    pub fn existing_wor_boundaries(self) -> ExistingWorBoundaryPolicy {
        self.existing_wor_boundaries
    }

    /// Treatment of one utterance ending after the next begins.
    pub fn end_overlaps(self) -> EndOverlapPolicy {
        self.end_overlaps
    }
}

impl WordEndPolicy {
    /// Derive the policy for an engine, given what the user asked for.
    pub fn for_engine(
        engine: crate::types::engines::FaEngineName,
        gap_healing: WordGapHealing,
    ) -> Self {
        Self {
            gap_healing,
            resolution: engine.timing_resolution(),
        }
    }

    /// A word-interval engine: ends came from the model.
    #[cfg(test)]
    pub fn measured(gap_healing: WordGapHealing) -> Self {
        Self {
            gap_healing,
            resolution: crate::types::engines::FaTimingResolution::WordIntervals,
        }
    }

    /// An onset-only engine: ends were derived from the next onset.
    #[cfg(test)]
    pub fn onset_only(gap_healing: WordGapHealing) -> Self {
        Self {
            gap_healing,
            resolution: crate::types::engines::FaTimingResolution::TokenOnsets,
        }
    }

    /// Whether small gaps between words may be closed.
    pub fn heals(self) -> bool {
        self.gap_healing == WordGapHealing::Heal
    }

    /// Whether this engine's word ends were derived rather than measured.
    ///
    /// A derived end may be replaced by a better one, such as the utterance
    /// bullet; a measured end is the model's own answer and is left alone.
    pub fn ends_are_derived(self) -> bool {
        self.resolution == crate::types::engines::FaTimingResolution::TokenOnsets
    }
}

/// Words that lost their timing during post-processing, by cause.
///
/// Two counts rather than one total, because a reviewer reads the reason. The
/// caller used to receive a bare `usize` and stamp
/// `reason=clamped_to_utterance_boundary` on every one, which became false as
/// soon as a second cause existed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DroppedWordTimings {
    /// Clamping to an authoritative utterance bullet left no extent.
    pub clamped_to_utterance_boundary: usize,
    /// The span reaching the write-back did not describe an extent.
    pub without_extent: usize,
}

impl DroppedWordTimings {
    /// Whether any word lost its timing.
    pub fn any(self) -> bool {
        self.total() > 0
    }

    /// How many words lost their timing, for any reason.
    pub fn total(self) -> usize {
        self.clamped_to_utterance_boundary + self.without_extent
    }

    /// A decision-record reason naming only the causes that actually fired.
    pub fn reason(self) -> String {
        let mut causes = Vec::new();
        if self.clamped_to_utterance_boundary > 0 {
            causes.push(format!(
                "clamped_to_utterance_boundary={}",
                self.clamped_to_utterance_boundary
            ));
        }
        if self.without_extent > 0 {
            causes.push(format!("no_extent={}", self.without_extent));
        }
        format!("count={} {}", self.total(), causes.join(" "))
    }
}

/// Wire type for the FA infer protocol -- one group sent to a Python worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaInferItem {
    /// Words to align (cleaned text).
    pub words: Vec<String>,
    /// Stable word IDs aligned 1:1 with `words`.
    pub word_ids: Vec<String>,
    /// Utterance indices aligned 1:1 with `words`.
    pub word_utterance_indices: Vec<usize>,
    /// Word indices inside each utterance aligned 1:1 with `words`.
    pub word_utterance_word_indices: Vec<usize>,
    /// Path to the audio file.
    pub audio_path: String,
    /// Start of the audio window (ms).
    pub audio_start_ms: u64,
    /// End of the audio window (ms).
    pub audio_end_ms: u64,
    /// How to handle word end times during post-processing.
    pub gap_healing: WordGapHealing,
}

impl FaInferItem {
    /// Audio window as a [`TimeSpan`].
    pub fn audio_span(&self) -> TimeSpan {
        TimeSpan::new(self.audio_start_ms, self.audio_end_ms)
    }
}

// ---------------------------------------------------------------------------
// %wor tier management
// ---------------------------------------------------------------------------

/// Remove existing %wor tier from an utterance (if any).
pub fn remove_wor_tier(utterance: &mut Utterance) {
    utterance
        .dependent_tiers
        .retain(|t| !matches!(t.tier, DependentTier::Wor(_)));
}

/// Add a `%wor` tier generated from the inline bullets on words.
///
/// If the utterance already has a `%wor` tier, it is replaced **in place**,
/// preserving its position among other dependent tiers. If it has none,
/// the new tier is appended. Previously this called
/// [`remove_wor_tier`] followed by `push`, which destroyed the original
/// position and produced noisy git diffs on every file whose `%wor`
/// originally sat before `%mor` / `%gra` or in some other non-default
/// slot. The position-preserving replace fixes that entire class of
/// diff noise.
///
/// `pub(crate)`, not `pub` (2026-09-01 review, item 2): this crate's ONLY
/// production route to a `%wor` WRITE for a real alignment or refresh run is
/// `orchestrate::FaApplied::then_enforce_monotonicity`, reached through
/// `then_finalize` or called directly, always AFTER monotonicity has
/// resolved same-speaker overlaps (see `WorPlan`). Every whole-file refresh
/// helper in `orchestrate.rs` that has a PRODUCTION caller
/// (`refresh_reusable_alignment`, `refresh_reusable_utterances`) is
/// mechanical only and returns its touched utterances for a caller to fold
/// into that SAME write phase; neither calls this function directly.
///
/// The remaining direct callers: unit tests of `%wor` GENERATION shape
/// itself (`chat_ops/fa/tests/replaced_word_and_compound.rs`,
/// `chat_ops/morphosyntax_ops/tests.rs`), which build a fixture and are not
/// claiming anything about monotonicity ordering; and
/// `orchestrate::refresh_existing_alignment_with_boundary_policy` (and its
/// `refresh_existing_alignment` wrapper), which have NO production caller
/// and exist only to build test fixtures with `%wor` written from
/// refreshed-but-not-yet-resolved state, documented at their own
/// definition.
pub(crate) fn add_wor_tier(utterance: &mut Utterance) {
    let wor_tier = utterance.main.generate_wor_tier();
    batchalign_transform::inject::replace_or_add_tier(
        &mut utterance.dependent_tiers,
        DependentTier::Wor(wor_tier),
    );
}

/// Return `true` when every alignable FA word in the file already has reusable
/// `%wor` timing.
///
/// This intentionally does **not** look at `main` tier `inline_bullet` alone.
/// After a parse roundtrip, main-tier word timing may be represented as
/// `InternalBullet` tokens, while `%wor` carries the durable first-class timing
/// bullets. For the cheap rerun path we therefore verify that `%wor` fully and
/// cleanly aligns back to the main tier.
pub fn has_reusable_wor_timing(chat_file: &ChatFile) -> bool {
    let mut saw_alignable_word = false;
    let reusable = find_reusable_utterance_indices(chat_file);
    let mut expected_reusable = 0usize;
    let mut utt_idx = 0usize;

    for line in &chat_file.lines {
        let Line::Utterance(utterance) = line else {
            continue;
        };

        let main_word_count = count_alignable_main_words(utterance);
        if main_word_count == 0 {
            utt_idx += 1;
            continue;
        }
        saw_alignable_word = true;
        expected_reusable += 1;
        if !reusable.contains(&utt_idx) {
            return false;
        }
        utt_idx += 1;
    }

    saw_alignable_word && reusable.len() == expected_reusable
}

/// Find utterance indices that have reusable `%wor` timing.
///
/// Returns a set of utterance ordinal indices where
/// [`has_reusable_wor_timing_for_utterance()`] succeeds. Used by the plain
/// rerun path to selectively skip FA for utterances whose `%wor` is still
/// clean after manual edits to other utterances.
pub fn find_reusable_utterance_indices(chat_file: &ChatFile) -> std::collections::HashSet<usize> {
    struct ReuseCandidate {
        utt_idx: usize,
        has_alignable_words: bool,
        wor_span: Option<WordTiming>,
        main_start_ms: Option<u64>,
    }

    let mut candidates = Vec::new();
    let mut reusable = std::collections::HashSet::new();
    let mut utt_idx = 0usize;
    for line in &chat_file.lines {
        let Line::Utterance(utterance) = line else {
            continue;
        };
        let has_alignable_words = count_alignable_main_words(utterance) > 0;
        candidates.push(ReuseCandidate {
            utt_idx,
            has_alignable_words,
            wor_span: if has_alignable_words {
                self::orchestrate::collect_wor_backed_span(utterance)
            } else {
                None
            },
            main_start_ms: utterance
                .main
                .content
                .bullet
                .as_ref()
                .map(|bullet| bullet.timing.start_ms),
        });
        utt_idx += 1;
    }

    let mut next_timed_start = None;
    let mut next_timed_start_after = vec![None; candidates.len()];
    for i in (0..candidates.len()).rev() {
        next_timed_start_after[i] = next_timed_start;
        if let Some(start_ms) = candidates[i].main_start_ms {
            next_timed_start = Some(start_ms);
        }
    }

    for (i, candidate) in candidates.iter().enumerate() {
        if !candidate.has_alignable_words {
            continue;
        }
        let Some(ref span) = candidate.wor_span else {
            continue;
        };
        if let Some(next_start_ms) = next_timed_start_after[i]
            && span.end_ms > next_start_ms
        {
            continue;
        }
        reusable.insert(candidate.utt_idx);
    }

    reusable
}

/// Count Wor-alignable words in the main tier.
pub(crate) fn count_alignable_main_words(utterance: &Utterance) -> usize {
    let mut count = 0usize;
    walk_words(
        &utterance.main.content.content,
        None,
        &mut |leaf| match leaf {
            WordItem::Word(word) => {
                if counts_for_tier(word, TierDomain::Wor) {
                    count += 1;
                }
            }
            WordItem::ReplacedWord(replaced) => {
                // Must mirror extraction policy: the original word is the FA
                // unit, not the replacement words.  Counting replacement words
                // here would overcount vs. extraction and desync the cursor.
                if counts_for_tier(&replaced.word, TierDomain::Wor) {
                    count += 1;
                }
            }
            WordItem::Separator(_) => {}
        },
    );
    count
}

/// Update the utterance-level bullet from word timings.
///
/// The behavior depends on the bullet's provenance (`BulletSource`):
///
/// - **No pre-existing bullet**, sets bullet directly from the FA word span.
///
/// - **`BulletSource::Utr`**: overwrites with the FA word span. The UTR
///   bullet was a provisional grouping hint; once FA has produced word
///   timings the hint is discarded and the FA span is authoritative.
///
/// - **`BulletSource::Authoritative`**: usually unions the existing bullet
///   with the FA word span when that preserves plausible leading/trailing
///   coverage. On reruns with an existing `%wor` tier, a large lead before the
///   first aligned word is only preserved when untimed leading filler coverage
///   still remains. Otherwise it is treated as stale inherited timing and reset
///   to the FA word start. Trailing coverage still never shrinks so gestures
///   and similar non-alignable material are preserved.
///
/// When FA produces no word timings at all (all `None`), the existing bullet
/// is left unchanged: the UTR hint is the only timing we have and must be
/// preserved.
///
/// # Invariant
///
/// Every bullet written by this function has `BulletSource::Authoritative`,
/// marking it as FA-derived (no longer a provisional UTR hint).
pub fn update_utterance_bullet(utterance: &mut Utterance) {
    update_utterance_bullet_with_boundary_policy(utterance, ExistingWorBoundaryPolicy::Preserve);
}

/// Update the utterance bullet under an explicit prior-boundary policy.
///
/// [`ExistingWorBoundaryPolicy::RebuildFromEvidence`] writes the exact hull of
/// the admitted word evidence. This can shrink prior leading or trailing
/// coverage, so callers must opt into it explicitly; the compatibility entry
/// point [`update_utterance_bullet`] continues to preserve that coverage.
pub fn update_utterance_bullet_with_boundary_policy(
    utterance: &mut Utterance,
    existing_wor_boundaries: ExistingWorBoundaryPolicy,
) {
    use talkbank_model::model::BulletSource;

    const MAX_AUTHORITATIVE_START_LEAD_MS: u64 = 2_000;

    let mut first_start: Option<u64> = None;
    let mut last_end: Option<u64> = None;

    let mut timings: Vec<Option<postprocess::PendingTiming>> = Vec::new();
    postprocess::collect_word_timings(&utterance.main.content.content, &mut timings);
    let has_fa_wor = utterance.wor_tier().is_some();
    let has_untimed_leading_filler_coverage =
        has_untimed_leading_filler_coverage(&utterance.main.content.content);

    for span in timings.iter().flatten() {
        if first_start.is_none_or(|s| span.start_ms < s) {
            first_start = Some(span.start_ms);
        }
        if last_end.is_none_or(|e| span.end_ms > e) {
            last_end = Some(span.end_ms);
        }
    }

    if let (Some(word_start), Some(word_end)) = (first_start, last_end) {
        let (final_start, final_end) =
            match (existing_wor_boundaries, &utterance.main.content.bullet) {
                // Research projection: prior boundaries are revisable output, not
                // constraints on newly admitted evidence.
                (ExistingWorBoundaryPolicy::RebuildFromEvidence, _) => (word_start, word_end),
                // Provisional UTR hint: FA word span is authoritative, overwrite.
                (_, Some(existing)) if existing.source == BulletSource::Utr => {
                    (word_start, word_end)
                }
                // Authoritative hand-linked/FA bullet: reruns with an existing %wor
                // can preserve stale starts from a previous pass. If that lead is
                // implausibly large and there is no untimed leading filler coverage
                // left to preserve, snap the start back to the FA word span.
                // Otherwise keep the old leading coverage.
                (_, Some(existing)) => {
                    let start_lead_ms = word_start.saturating_sub(existing.timing.start_ms);
                    let final_start = if !has_fa_wor
                        || has_untimed_leading_filler_coverage
                        || start_lead_ms <= MAX_AUTHORITATIVE_START_LEAD_MS
                    {
                        word_start.min(existing.timing.start_ms)
                    } else {
                        word_start
                    };
                    (final_start, word_end.max(existing.timing.end_ms))
                }
                // No pre-existing bullet: set from word span.
                (_, None) => (word_start, word_end),
            };
        // The resulting bullet is authoritative (FA-derived).
        utterance.main.content.bullet = Some(Bullet::new(final_start, final_end));
    }
    // If no word timings: leave existing bullet unchanged, UNLESS it is
    // zero-duration (start >= end), which is an invalid timing that would
    // produce E362. A zero-duration bullet from a previous buggy run must be
    // cleared; no bullet is valid CHAT, an invalid bullet is not.
    if let Some(ref existing) = utterance.main.content.bullet
        && existing.timing.start_ms >= existing.timing.end_ms
    {
        utterance.main.content.bullet = None;
    }
}

fn has_untimed_leading_filler_coverage(content: &[UtteranceContent]) -> bool {
    let mut before_first_timed_word = true;
    let mut has_untimed_leading_filler = false;

    walk_words(content, Some(TierDomain::Wor), &mut |leaf| {
        if !before_first_timed_word {
            return;
        }
        let word = match leaf {
            WordItem::Word(word) => word,
            WordItem::ReplacedWord(replaced) => &replaced.word,
            WordItem::Separator(_) => return,
        };
        if get_word_timing(word).is_some() {
            before_first_timed_word = false;
            return;
        }
        if word.category == Some(WordCategory::Filler) {
            has_untimed_leading_filler = true;
        }
    });

    has_untimed_leading_filler
}

/// Collect current main-tier word timings in the exact order FA uses for
/// extraction and injection.
///
/// This is the stable timing surface for selective reuse: when an utterance has
/// already been refreshed from `%wor`, the returned vector can be stitched
/// directly into a preserved FA group without a worker roundtrip.
pub fn collect_existing_fa_word_timings(utterance: &Utterance) -> Vec<Option<WordTiming>> {
    let mut timings = Vec::new();
    walk_words(
        &utterance.main.content.content,
        None,
        &mut |leaf| match leaf {
            WordItem::Word(word) => {
                if counts_for_tier(word, TierDomain::Wor) {
                    timings.push(
                        get_word_timing(word)
                            .and_then(|span| WordTiming::from_transcript(span).ok()),
                    );
                }
            }
            WordItem::ReplacedWord(replaced) => {
                // Mirror extraction policy: the original word is the FA unit.
                // Replacement words never receive inline_bullet after injection,
                // so collecting them would always produce None for each and
                // miscount vs. collect_fa_words → collect_preserved_group_timings
                // would return None and needlessly bypass the %wor preservation path.
                if counts_for_tier(&replaced.word, TierDomain::Wor) {
                    timings.push(
                        get_word_timing(&replaced.word)
                            .and_then(|span| WordTiming::from_transcript(span).ok()),
                    );
                }
            }
            WordItem::Separator(_) => {}
        },
    );
    timings
}

// ---------------------------------------------------------------------------
// Helpers shared across submodules
// ---------------------------------------------------------------------------

/// The LINE index of the nth utterance, or `None` if there is no nth utterance.
///
/// `UtteranceIdx` counts only utterances; `LineIdx` counts every line, headers
/// included. The two never coincide, because a CHAT file always opens with
/// headers, so converting between them requires walking the file. That is why
/// this is a function taking the file rather than a cast: a producer holding
/// an ordinal cannot fabricate a line index without looking.
pub(super) fn utterance_line_idx(
    chat_file: &talkbank_model::model::ChatFile,
    utt_idx: UtteranceIdx,
) -> Option<batchalign_transform::decisions::LineIdx> {
    use talkbank_model::model::Line;
    let mut seen = 0;
    for (line_idx, line) in chat_file.lines.as_slice().iter().enumerate() {
        if matches!(line, Line::Utterance(_)) {
            if seen == utt_idx.raw() {
                return Some(batchalign_transform::decisions::LineIdx::new(line_idx));
            }
            seen += 1;
        }
    }
    None
}

/// Get a mutable reference to the nth utterance in the file.
pub(super) fn get_utterance_mut(
    chat_file: &mut talkbank_model::model::ChatFile,
    utt_idx: UtteranceIdx,
) -> Option<&mut Utterance> {
    use talkbank_model::model::Line;
    let mut current = 0;
    for line in &mut chat_file.lines {
        if let Line::Utterance(utt) = line {
            if current == utt_idx.raw() {
                return Some(utt);
            }
            current += 1;
        }
    }
    None
}

/// Get the inline timing from a word, if present.
pub(super) fn get_word_timing(word: &Word) -> Option<TimeSpan> {
    word.inline_bullet
        .as_ref()
        .map(|b| TimeSpan::new(b.timing.start_ms, b.timing.end_ms))
}

// ---------------------------------------------------------------------------
// AudioIdentity
// ---------------------------------------------------------------------------

/// Content identity for an audio file used in FA cache keys.
///
/// # Invariant
///
/// Format: `"{resolved_path}|{mtime_secs}|{file_size}"`. Fast identity
/// based on filesystem metadata (no file content hashing). Created by
/// [`AudioIdentity::from_metadata`] in the server runner.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AudioIdentity(String);

impl AudioIdentity {
    /// Build an identity from resolved path + filesystem metadata.
    pub fn from_metadata(path: &str, mtime_secs: u64, size: u64) -> Self {
        Self(format!("{path}|{mtime_secs}|{size}"))
    }

    /// Access the raw identity string (for display/logging).
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for AudioIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// Cache key computation
// ---------------------------------------------------------------------------

/// Compute cache key for an FA result.
///
/// Key = `BLAKE3("{audio_identity}|{start}|{end}|{text}|{healing_flag}|{engine}{schema}")`.
pub fn cache_key(
    words: &[String],
    audio_identity: &AudioIdentity,
    start_ms: u64,
    end_ms: u64,
    gap_healing: WordGapHealing,
    engine: crate::types::engines::FaEngineName,
) -> crate::chat_ops::CacheKey {
    let text = words.join(" ");
    // What is CACHED is the parsed `Vec<Option<WordTiming>>`, not the raw
    // worker response, so this key must cover everything those timings depend
    // on and nothing they do not. Gap healing is applied later, in
    // post-processing, and never reaches the stored value.
    //
    // The two resolutions therefore differ, and collapsing them is what makes
    // the difference expensive:
    //
    // - WORD-INTERVAL engines take their start and end from the model. The
    //   stored timings do not depend on the healing policy at all, and
    //   the healing policy cannot change what the model returns, so one
    //   constant is correct. (The Cantonese request does carry `pauses` on the
    //   wire; its runner binds it unused, so it changes no output today.) It keeps the literal the old code always hashed for these
    //   engines, which is what lets the fleet's existing entries stay
    //   reachable: re-running them would spend GPU hours to recompute
    //   byte-identical values.
    //
    // - TOKEN-ONSET engines get their ends derived by the parser, and that
    //   derivation changed on 2026-08-14: entries written before it hold ends
    //   equal to their own starts, which is no duration at all. New strings
    //   retire exactly those, and the flag additionally stands in for the
    //   tokenization, since the same `--pauses` value selects it.
    let healing_flag = match engine.timing_resolution() {
        FaTimingResolution::WordIntervals => "no_pauses",
        FaTimingResolution::TokenOnsets => match gap_healing {
            WordGapHealing::Heal => "heal_gaps",
            WordGapHealing::PreserveMeasured => "preserve_measured",
        },
    };
    // The ENGINE, so two models that share a parse strategy cannot share
    // cache entries for the same audio and words.
    let engine_str = {
        use crate::types::engines::EngineBackend;
        engine.wire_name()
    };
    // Word-interval responses now preserve the model's own per-word score.
    // Older cache entries deserialize (the field is optional for wire
    // compatibility) but cannot supply that evidence, so they must not satisfy
    // a new interval request. Onset-only Whisper has no corresponding score
    // and keeps its established namespace rather than paying to recompute the
    // same evidence.
    let result_schema = match engine.timing_resolution() {
        FaTimingResolution::WordIntervals => "|model_score_v1",
        FaTimingResolution::TokenOnsets => "",
    };
    let input = format!(
        "{}|{start_ms}|{end_ms}|{text}|{healing_flag}|{engine_str}{result_schema}",
        audio_identity.0
    );
    crate::chat_ops::CacheKey::from_content(&input)
}

#[cfg(test)]
mod utterance_line_idx_tests {
    use super::*;

    /// The conversion must SKIP headers, which is the whole reason it exists.
    ///
    /// `orchestrate` previously stored an utterance ordinal in a field that
    /// indexes lines. It compiled because both were `usize`, and the effect was
    /// invisible: the decision either vanished or attached to the wrong
    /// utterance. `LineIdx` now rejects the assignment, and this pins the
    /// conversion that replaced it.
    #[test]
    fn converts_an_utterance_ordinal_to_a_line_index() {
        let chat_text = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI|2;0.0||||Target_Child|||
*CHI:\thello .
*CHI:\tworld .
@End
";
        let parser = talkbank_parser::TreeSitterParser::new().expect("parser init");
        let chat = parser.parse_chat_file(chat_text).expect_built();

        // Five headers precede the first utterance, so the spaces differ.
        assert_eq!(
            utterance_line_idx(&chat, UtteranceIdx::new(0)).map(|l| l.raw()),
            Some(5)
        );
        assert_eq!(
            utterance_line_idx(&chat, UtteranceIdx::new(1)).map(|l| l.raw()),
            Some(6)
        );
        // Past the end is None, not a fabricated index.
        assert_eq!(utterance_line_idx(&chat, UtteranceIdx::new(2)), None);
    }
}
