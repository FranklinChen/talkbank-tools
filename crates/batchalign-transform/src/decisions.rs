//! Pipeline decision provenance: tracking machine decisions for user review.
//!
//! Every batchalign3 command makes decisions that alter output: clamping
//! timestamps, stripping timing, skipping utterances, defaulting values,
//! normalizing text. These decisions are currently logged via `tracing` but
//! invisible to the user in the output CHAT file.
//!
//! This module defines `DecisionRecord`, a structured representation of a
//! machine decision retained in structured run evidence so users and tools
//! can review what the pipeline did and why without cluttering CHAT output.
//!
//! # Architecture
//!
//! Each pipeline stage (FA, UTR, morphosyntax, utseg, etc.) collects
//! `Vec<DecisionRecord>` during processing. The command orchestrator retains
//! those records in durable evidence and strips legacy `%xalign` / `%xrev`
//! tiers before serialization. `ReviewLevel` remains on the compatibility
//! wire surface for now, but no value authorizes CHAT-tier generation.

use talkbank_model::model::{ChatFile, DependentTier, Line};

/// Which pipeline module made the decision.
///
/// Derivable from a [`DecisionStrategy`] via [`DecisionStrategy::module`],
/// but retained as its own type for call sites that want to filter or
/// display by module without caring about the specific strategy variant
/// (e.g. "show me all FA decisions").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionModule {
    /// Forced alignment (grouping, injection, postprocessing).
    Fa,
    /// Utterance timing recovery.
    Utr,
    /// Monotonicity enforcement (end-time clamping, start-time stripping).
    Monotonicity,
    /// Morphosyntax (Stanza mapping, retokenization).
    Morphosyntax,
    /// Coreference resolution (sparse `%xcoref` injection).
    Coref,
    /// Utterance segmentation.
    Utseg,
}

impl DecisionModule {
    /// Stable label for tracing and structured evidence.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fa => "fa",
            Self::Utr => "utr",
            Self::Monotonicity => "monotonicity",
            Self::Morphosyntax => "morphosyntax",
            Self::Coref => "coref",
            Self::Utseg => "utseg",
        }
    }
}

// ---------------------------------------------------------------------------
// Typed decision strategies
//
// Every strategy the pipeline can emit is declared as a variant of one of
// the per-module enums below, then wrapped in [`DecisionStrategy`] at the
// boundary with [`DecisionRecord`]. This replaces the previous stringly
// typed `strategy: &'static str` field so that:
//
// - Typos at construction sites fail to compile instead of producing a
//   novel strategy label consumers silently can't match.
// - Consumers can match exhaustively on the strategy set per module.
// - Adding a new strategy requires declaring its name in exactly one
//   place, and serialization + tracing derive from that declaration.
//
// The `as_str()` name on each per-module enum is the *label* that was
// previously typed as a string literal. Migration rule: if downstream
// consumers read `record.strategy == "end_clamped"`, their new read is
// `matches!(record.strategy, DecisionStrategy::Monotonicity(MonotonicityStrategy::EndClampedCoverageOnly
// | MonotonicityStrategy::EndClampedBoundaryFromWords | MonotonicityStrategy::EndClampedInterleavedWords))`.
// ---------------------------------------------------------------------------

/// Forced-alignment repair strategies (`fa::repair`, `fa::orchestrate`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaStrategy {
    /// Same-speaker gap filling narrowed an overlap into a gap-fill.
    GapFilled,
    /// Two bullets' overlap was split at the midpoint.
    BoundaryAveraged,
    /// Longest-increasing-subsequence selective timing removal.
    LisRemoval,
    /// Utterance-bullet timing was stripped under a monotonicity violation.
    TimingStripped,
    /// Per-word timings were dropped (e.g. clamped to utterance boundary).
    WordsTimingDropped,
    /// A too-narrow utterance bullet was expanded to fit its word count.
    NarrowBulletRescued,
    /// Under `align --main-bullets keep`, narrow-bullet rescue widened the
    /// grouping window of a kept bullet so the aligner could find its words.
    /// The window is used for grouping only: the transcript keeps the given
    /// bullet, which is why this is not [`Self::NarrowBulletRescued`].
    KeptBulletWindowWidened,
    /// Word timings were cut to fit an utterance bullet the input carried and
    /// the run was told to keep (`align --main-bullets keep`).
    ///
    /// Distinct from [`Self::WordsTimingDropped`] because the bullet here is a
    /// fixed authority, not a projection this run derived: a word straddling
    /// its edge is trimmed to it, and a word wholly outside it (or left with no
    /// extent) loses its timing.
    WordsClampedToKeptBullet,
    /// The input gave this utterance no bullet and the run was told to keep
    /// that absence (`align --main-bullets exact`): the aligner's word
    /// timings for it are removed, so no bullet can be derived from them.
    ///
    /// Distinct from [`Self::WordsClampedToKeptBullet`], where the words are
    /// fitted into a bullet the input gave; here there is no bullet to fit.
    WordsUntimedForKeptAbsence,
    /// How this utterance's word timings were produced.
    ///
    /// Not a decision in the sense the others are: nothing was changed. It
    /// reports what a `Bullet` cannot carry, so a reader can tell a measured
    /// timing from one this pipeline inferred or invented. Without it the
    /// distinction exists only inside the process that made it.
    TimingProvenance,
    /// A run of untimed utterances was left unaligned because the audio
    /// remaining for it could not physically contain its words.
    ///
    /// Distinct from every other variant here: the rest describe a timing this
    /// pipeline ADJUSTED, this one describes words that will have NO timing at
    /// all, permanently. A reader of the transcript cannot otherwise tell that
    /// from an alignment failure, which is why it is a recorded decision rather
    /// than a log line.
    UnplaceableRun,
    /// No request was made because the utterance window cannot fit the recording or engine budget.
    ///
    /// Carries the refused window itself, so the evidence file states its
    /// bounds and the cause as data rather than only as prose in `reason`.
    WindowRefused(RefusedWindow),
    /// An utterance window longer than the engine budget was aligned in
    /// pieces, each cut at the end of a word utterance timing recovery had
    /// matched to a timed ASR token, and each within the budget.
    ///
    /// Not a review item: every cut point is an acoustic observation and the
    /// aligner still measured every word inside its piece. It is recorded so a
    /// reader of the evidence can tell which utterances were aligned in pieces
    /// rather than in one request, and how many.
    WindowSplitAtAnchors(SplitWindow),
}

/// An over-budget utterance window that was aligned as several pieces.
///
/// Plain `u64` milliseconds for the same reason as [`RefusedWindow`]: this
/// crate cannot see batchalign's time newtypes and the wire form is numbers.
/// The [`std::fmt::Display`] impl is the single source of the human reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitWindow {
    /// Start of the utterance window, in file milliseconds.
    pub start_ms: u64,
    /// End of the utterance window, in file milliseconds.
    pub end_ms: u64,
    /// The engine budget every piece fits, in milliseconds.
    pub budget_ms: u64,
    /// How many pieces the window was aligned as; at least two, since one
    /// piece would be the whole window and the whole window is over budget.
    pub pieces: usize,
}

impl std::fmt::Display for SplitWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "audio window {}ms-{}ms ({}ms) exceeds alignment budget {}ms; aligned as {} pieces cut at recovered word anchors",
            self.start_ms,
            self.end_ms,
            self.end_ms.saturating_sub(self.start_ms),
            self.budget_ms,
            self.pieces
        )
    }
}

/// An audio window the grouping stage refused to send to the aligner.
///
/// One variant per cause, each holding only the numbers that exist for it, so
/// a window cannot contradict its cause (an `Inverted` whose start precedes
/// its end, an `Empty` with two different bounds). Plain `u64` milliseconds
/// with a `_ms` suffix: this crate cannot see batchalign's `Ms`/`FileMs`
/// newtypes, and the wire form is plain numbers. The [`std::fmt::Display`]
/// impl is the single source of the human reason text, so the prose and the
/// data cannot disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusedWindow {
    /// Longer than the engine's alignment budget.
    OverBudget {
        /// Start of the window, in file milliseconds.
        start_ms: u64,
        /// End of the window, in file milliseconds.
        end_ms: u64,
        /// The budget the window exceeded, in milliseconds.
        budget_ms: u64,
    },
    /// The window holds no audio (start equals end).
    Empty {
        /// The position at which the window starts and ends.
        at_ms: u64,
    },
    /// The end precedes the start.
    Inverted {
        /// The proposed start.
        start_ms: u64,
        /// The proposed end, which precedes the start.
        end_ms: u64,
    },
    /// The window ends past the end of the recording.
    PastRecording {
        /// Start of the window, in file milliseconds.
        start_ms: u64,
        /// End of the window, in file milliseconds.
        end_ms: u64,
        /// How far past the recording's end the window falls, in milliseconds.
        exceeds_by_ms: u64,
    },
    /// Longer than the engine's budget, and the words utterance timing
    /// recovery matched inside it leave a stretch longer than the budget with
    /// no anchor to cut at.
    ///
    /// Distinct from [`Self::OverBudget`], which means there was no usable
    /// anchor evidence at all. Here the evidence exists and is itself the
    /// problem: consecutive matched words farther apart than any one request
    /// may span almost always means the recovered placement, and so the
    /// utterance's bullet, is wrong, not that someone paused that long
    /// mid-utterance. It is the signal that needs review; the words are never
    /// aligned across it.
    AnchorGap {
        /// Start of the utterance window, in file milliseconds.
        start_ms: u64,
        /// End of the utterance window, in file milliseconds.
        end_ms: u64,
        /// The budget no piece could stay within, in milliseconds.
        budget_ms: u64,
        /// Start of the widest uncrossable stretch: the window start or the
        /// end of an anchored word.
        gap_start_ms: u64,
        /// End of that stretch: the end of the next anchored word, or the
        /// window end.
        gap_end_ms: u64,
    },
    /// Longer than the engine's budget, and utterance timing recovery DID
    /// match this utterance, but its matches give no usable cut point; the
    /// cause says why.
    ///
    /// Distinct from [`Self::OverBudget`], which means recovery has nothing to
    /// say about this utterance (it did not run, had no tokens, or matched
    /// none of its words). Here there is evidence and it could not be used,
    /// which a reviewer may want to look at.
    AnchorsUnusable {
        /// Start of the utterance window, in file milliseconds.
        start_ms: u64,
        /// End of the utterance window, in file milliseconds.
        end_ms: u64,
        /// The budget the window exceeded, in milliseconds.
        budget_ms: u64,
        /// Why the matches could not be cut at.
        cause: UnusableAnchors,
    },
}

/// Why utterance timing recovery's matches for an over-budget utterance
/// could not be used to split it. A closed set: a new reason is a new
/// variant, with its own wire name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnusableAnchors {
    /// Every match was fuzzy or to a multi-word token, so none says when a
    /// word ended.
    NoReliableAnchors,
    /// The anchors were refused as a set: out of word order, overlapping in
    /// time, inverted, or naming a token absent from the stream.
    AnchorsRefused,
    /// The anchors were read off a different word list than the utterance
    /// grouping holds.
    AnchorsDescribeOtherWords,
    /// Every anchor lies on the last word or at or outside the window's
    /// edges, so none can be a cut.
    NoInteriorCut,
}

impl UnusableAnchors {
    /// Stable wire and prose label.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoReliableAnchors => "no_reliable_anchors",
            Self::AnchorsRefused => "anchors_refused",
            Self::AnchorsDescribeOtherWords => "anchors_describe_other_words",
            Self::NoInteriorCut => "no_interior_cut",
        }
    }
}

impl std::fmt::Display for RefusedWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::OverBudget {
                start_ms,
                end_ms,
                budget_ms,
            } => write!(
                f,
                "audio window {start_ms}ms-{end_ms}ms ({}ms) exceeds alignment budget {budget_ms}ms; narrower evidence is required",
                end_ms.saturating_sub(start_ms)
            ),
            Self::Empty { at_ms } => {
                write!(f, "audio window at {at_ms}ms has no positive extent")
            }
            Self::Inverted { start_ms, end_ms } => {
                write!(
                    f,
                    "audio window start {start_ms}ms is after its end {end_ms}ms"
                )
            }
            Self::PastRecording {
                start_ms,
                end_ms,
                exceeds_by_ms,
            } => write!(
                f,
                "audio window {start_ms}ms-{end_ms}ms ends {exceeds_by_ms}ms past the end of the recording"
            ),
            Self::AnchorGap {
                start_ms,
                end_ms,
                budget_ms,
                gap_start_ms,
                gap_end_ms,
            } => write!(
                f,
                "audio window {start_ms}ms-{end_ms}ms ({}ms) exceeds alignment budget {budget_ms}ms, and its recovered word anchors leave {gap_start_ms}ms-{gap_end_ms}ms ({}ms) with no cut point; the utterance placement needs review",
                end_ms.saturating_sub(start_ms),
                gap_end_ms.saturating_sub(gap_start_ms)
            ),
            Self::AnchorsUnusable {
                start_ms,
                end_ms,
                budget_ms,
                cause,
            } => write!(
                f,
                "audio window {start_ms}ms-{end_ms}ms ({}ms) exceeds alignment budget {budget_ms}ms, and its recovered word anchors cannot split it ({})",
                end_ms.saturating_sub(start_ms),
                cause.as_str()
            ),
        }
    }
}

impl FaStrategy {
    /// Stable wire/tracing label for structured evidence.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GapFilled => "gap_filled",
            Self::BoundaryAveraged => "boundary_averaged",
            Self::LisRemoval => "lis_removal",
            Self::TimingStripped => "timing_stripped",
            Self::TimingProvenance => "timing_provenance",
            Self::UnplaceableRun => "unplaceable_run",
            Self::WindowRefused(_) => "window_refused",
            Self::WindowSplitAtAnchors(_) => "window_split_at_anchors",
            Self::WordsTimingDropped => "words_timing_dropped",
            Self::NarrowBulletRescued => "narrow_bullet_rescued",
            Self::WordsClampedToKeptBullet => "words_clamped_to_kept_bullet",
            Self::WordsUntimedForKeptAbsence => "words_untimed_for_kept_absence",
            Self::KeptBulletWindowWidened => "kept_bullet_window_widened",
        }
    }
}

/// Utterance-timing-recovery (UTR) strategies (`fa::utr`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UtrStrategy {
    /// Untimed utterance matched a zero-duration span and was left alone.
    ZeroDurationSkipped,
    /// No ASR alignment found for an untimed utterance.
    Unmatched,
}

impl UtrStrategy {
    /// Stable wire/tracing label for structured evidence.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ZeroDurationSkipped => "zero_duration_skipped",
            Self::Unmatched => "unmatched",
        }
    }
}

/// Monotonicity-enforcement strategies applied to utterance bullets.
///
/// The three `EndClamped*` variants (2026-09-01 review, item 4) replace a
/// single `EndClamped` label that carried its real distinction only in the
/// free-text `reason` string (`resolution=coverage_only|boundary_from_words
/// |interleaved_words`), matching `chat_ops::fa::orchestrate`'s
/// `EndOverlapResolution` and `MonotonicityEffect` one-for-one so a consumer
/// filtering on `strategy` alone (not parsing `reason`) sees the same three
/// cases those types do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonotonicityStrategy {
    /// Only the bullet's inherited coverage overshot the next utterance's
    /// start; no measured word conflicted. Words untouched.
    EndClampedCoverageOnly,
    /// Both bullets' inherited boundary replaced by their measured word
    /// hulls; the words themselves never conflicted.
    EndClampedBoundaryFromWords,
    /// The words themselves interleave, or the next utterance has none: a
    /// genuine conflict. The bullet and every word past the bound were
    /// clamped together.
    EndClampedInterleavedWords,
    /// Bullet timing stripped because monotonicity could not be restored.
    TimingStripped,
    /// Two utterance bullets the input carried conflict (one overlaps or
    /// starts before the other) and the run was told to keep given bullets
    /// (`align --main-bullets keep`), so both were left exactly as given and
    /// the conflict is recorded here instead of resolved.
    KeptBulletLeftUnresolved,
    /// Under `align --main-bullets keep`, a derived bullet overlapping a kept
    /// one gave way to it: its start moved forward to the kept end, and any
    /// leading word reaching back before that was cut.
    YieldedToKeptBullet,
}

impl MonotonicityStrategy {
    /// Stable wire/tracing label for structured evidence.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::EndClampedCoverageOnly => "end_clamped_coverage_only",
            Self::EndClampedBoundaryFromWords => "end_clamped_boundary_from_words",
            Self::EndClampedInterleavedWords => "end_clamped_interleaved_words",
            Self::TimingStripped => "timing_stripped",
            Self::KeptBulletLeftUnresolved => "kept_bullet_left_unresolved",
            Self::YieldedToKeptBullet => "yielded_to_kept_bullet",
        }
    }
}

/// Morphosyntax (Stanza mapping / injection) strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphosyntaxStrategy {
    /// Utterance had no Mor-alignable content.
    NotApplicable,
    /// 1-to-1 invariant violated post-mapping.
    MisalignmentBug,
    /// UD→Mor mapping returned an error (e.g. multi-root UD).
    MappingFailed,
    /// Stanza retokenization rewrite failed.
    RetokenizationFailed,
    /// `inject_morphosyntax` rejected the utterance (re-raised as a decision).
    InjectionFailed,
    /// Stanza returned zero sentences for the dispatched utterance.
    NlpNoSentences,
}

impl MorphosyntaxStrategy {
    /// Stable wire/tracing label for structured evidence.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NotApplicable => "not_applicable",
            Self::MisalignmentBug => "misalignment_bug",
            Self::MappingFailed => "mapping_failed",
            Self::RetokenizationFailed => "retokenization_failed",
            Self::InjectionFailed => "injection_failed",
            Self::NlpNoSentences => "nlp_no_sentences",
        }
    }
}

/// Utterance-segmentation strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UtsegStrategy {
    /// Single-word or empty utterance, not dispatched.
    NotApplicable,
    /// Worker returned the wrong number of assignments.
    MisalignmentBug,
}

impl UtsegStrategy {
    /// Stable wire/tracing label for structured evidence.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NotApplicable => "not_applicable",
            Self::MisalignmentBug => "misalignment_bug",
        }
    }
}

/// Coreference-injection strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorefStrategy {
    /// Worker returned a sentence_idx that doesn't map to a valid line.
    SentenceIndexOutOfBounds,
    /// `%xcoref` tier construction failed (NonEmptyString, etc.).
    InjectionFailed,
}

impl CorefStrategy {
    /// Stable wire/tracing label for structured evidence.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SentenceIndexOutOfBounds => "sentence_index_out_of_bounds",
            Self::InjectionFailed => "injection_failed",
        }
    }
}

/// The typed strategy carried by a [`DecisionRecord`].
///
/// Subsumes the previous `(module: DecisionModule, strategy: &'static str)`
/// pair into a single enum. [`DecisionStrategy::module`] recovers the
/// module when a consumer wants that level of grouping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionStrategy {
    /// Forced alignment.
    Fa(FaStrategy),
    /// Utterance timing recovery.
    Utr(UtrStrategy),
    /// Monotonicity enforcement.
    Monotonicity(MonotonicityStrategy),
    /// Morphosyntax.
    Morphosyntax(MorphosyntaxStrategy),
    /// Coreference.
    Coref(CorefStrategy),
    /// Utterance segmentation.
    Utseg(UtsegStrategy),
}

impl DecisionStrategy {
    /// The pipeline module this strategy belongs to.
    pub fn module(&self) -> DecisionModule {
        match self {
            Self::Fa(_) => DecisionModule::Fa,
            Self::Utr(_) => DecisionModule::Utr,
            Self::Monotonicity(_) => DecisionModule::Monotonicity,
            Self::Morphosyntax(_) => DecisionModule::Morphosyntax,
            Self::Coref(_) => DecisionModule::Coref,
            Self::Utseg(_) => DecisionModule::Utseg,
        }
    }

    /// Stable label for structured tracing and persisted decision evidence.
    pub fn strategy_name(&self) -> &'static str {
        match self {
            Self::Fa(s) => s.as_str(),
            Self::Utr(s) => s.as_str(),
            Self::Monotonicity(s) => s.as_str(),
            Self::Morphosyntax(s) => s.as_str(),
            Self::Coref(s) => s.as_str(),
            Self::Utseg(s) => s.as_str(),
        }
    }
}

/// Index of a LINE in `ChatFile.lines`, counting headers.
///
/// Distinct from `UtteranceIdx`, which counts only utterances, because the two
/// never coincide: every CHAT file opens with headers, so utterance 0 is line 5
/// or later. They were both bare `usize` and a producer assigned one to the
/// other, which silently dropped a decision (the consumer looked at a header
/// line and skipped it) or attached it to the wrong utterance. Nothing noticed,
/// because nothing could: `usize` accepts either.
///
/// Converting between the spaces genuinely requires the file, so the
/// conversion is a function that takes one, not a cast.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct LineIdx(usize);

impl LineIdx {
    /// Wraps a 0-based index into `ChatFile.lines`.
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    /// The wrapped index.
    pub fn raw(self) -> usize {
        self.0
    }
}

/// A machine decision that altered the output and should be reviewable.
///
/// Every silent clamp, skip, default, or normalization that changes the
/// output should produce a `DecisionRecord`. These are collected during
/// processing and persisted as structured evidence.
#[derive(Debug, Clone)]
pub struct DecisionRecord {
    /// Index into `ChatFile.lines` (the affected utterance).
    pub line_idx: LineIdx,
    /// Speaker code for the affected utterance.
    pub speaker: String,
    /// Typed module + strategy. Replaces the prior separate `module` /
    /// `strategy: &'static str` fields; use `strategy.module()` and
    /// `strategy.strategy_name()` to recover the old components.
    pub strategy: DecisionStrategy,
    /// Structured key=value reason retained in evidence.
    ///
    /// Example: `"overlap=1200ms prev_end=5000 next_start=3800"`
    pub reason: String,
    /// Whether a human should review this decision.
    pub needs_review: bool,
}

impl DecisionRecord {
    /// Format a stable human-readable evidence summary.
    pub fn evidence_summary(&self) -> String {
        format!(
            "{}:{} {}",
            self.strategy.module().as_str(),
            self.strategy.strategy_name(),
            self.reason
        )
    }

    /// Emit a structured tracing event for this decision.
    ///
    /// This is the single logging point, callers should NOT separately call
    /// `tracing::warn!` with the same information. The decision record is the
    /// source of truth; tracing and durable evidence are derived outputs.
    pub fn trace(&self) {
        let module = self.strategy.module().as_str();
        let strategy = self.strategy.strategy_name();
        if self.needs_review {
            tracing::warn!(
                line_idx = self.line_idx.raw(),
                module,
                strategy,
                speaker = %self.speaker,
                reason = %self.reason,
                "pipeline decision (needs review)"
            );
        } else {
            tracing::info!(
                module,
                strategy,
                speaker = %self.speaker,
                line_idx = self.line_idx.raw(),
                reason = %self.reason,
                "pipeline decision"
            );
        }
    }

    /// Create a decision, emit its trace, and return it.
    ///
    /// Convenience for the common pattern at decision points:
    /// ```ignore
    /// decisions.push(DecisionRecord::new_and_trace(...));
    /// ```
    pub fn new_and_trace(
        line_idx: usize,
        speaker: String,
        strategy: DecisionStrategy,
        reason: String,
        needs_review: bool,
    ) -> Self {
        let record = Self {
            line_idx: LineIdx::new(line_idx),
            speaker,
            strategy,
            reason,
            needs_review,
        };
        record.trace();
        record
    }
}

/// Legacy review-tier request retained for wire compatibility.
///
/// Batchalign3 no longer injects `%xalign` or `%xrev` for any value. The enum
/// remains deserializable so stored jobs and older clients do not fail merely
/// because the presentation policy changed. Decisions themselves remain in
/// structured evidence.
///
/// [`None`]: ReviewLevel::None
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewLevel {
    /// Legacy default request value.
    #[default]
    None,
    /// Legacy low-confidence request value; emits no CHAT tiers.
    LowConfidence,
    /// Legacy all-decisions request value; emits no CHAT tiers.
    All,
}

/// Remove all `%xalign` and `%xrev` tiers from every utterance in the file.
///
/// Called by every pipeline serialization path so legacy scaffolding cannot
/// survive into new output.
pub fn strip_decision_tiers(chat_file: &mut ChatFile) {
    for line in &mut chat_file.lines {
        let Line::Utterance(utt) = line else {
            continue;
        };
        utt.dependent_tiers.retain(|tier| {
            !matches!(
                &tier.tier,
                DependentTier::UserDefined(t)
                    if t.label.as_str() == "xalign" || t.label.as_str() == "xrev"
            )
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use talkbank_model::WriteChat;
    use talkbank_parser::TreeSitterParser;

    fn parse_chat(text: &str) -> ChatFile {
        let parser = TreeSitterParser::new().expect("parser init");
        parser.parse_chat_file(text).expect_built()
    }

    #[test]
    fn decision_record_evidence_summary_format() {
        let d = DecisionRecord {
            line_idx: LineIdx::new(5),
            speaker: "CHI".into(),
            strategy: DecisionStrategy::Monotonicity(MonotonicityStrategy::EndClampedCoverageOnly),
            reason: "overlap=1200ms prev_end=5000 next_start=3800".into(),
            needs_review: false,
        };
        assert_eq!(
            d.evidence_summary(),
            "monotonicity:end_clamped_coverage_only overlap=1200ms prev_end=5000 next_start=3800"
        );
    }

    #[test]
    fn refused_window_prose_has_no_doubled_unit() {
        let refused = RefusedWindow::OverBudget {
            start_ms: 12_000,
            end_ms: 38_520,
            budget_ms: 15_000,
        };
        assert_eq!(
            refused.to_string(),
            "audio window 12000ms-38520ms (26520ms) exceeds alignment budget 15000ms; narrower evidence is required"
        );
        assert_eq!(
            FaStrategy::WindowRefused(refused).as_str(),
            "window_refused"
        );
    }

    /// The anchor-gap refusal names both the window and the stretch that
    /// could not be crossed, so a reviewer can go straight to the gap.
    #[test]
    fn anchor_gap_prose_names_the_window_and_the_uncrossable_stretch() {
        let refused = RefusedWindow::AnchorGap {
            start_ms: 10_000,
            end_ms: 400_000,
            budget_ms: 15_000,
            gap_start_ms: 12_500,
            gap_end_ms: 380_000,
        };
        assert_eq!(
            refused.to_string(),
            "audio window 10000ms-400000ms (390000ms) exceeds alignment budget 15000ms, and its recovered word anchors leave 12500ms-380000ms (367500ms) with no cut point; the utterance placement needs review"
        );
        assert_eq!(
            FaStrategy::WindowRefused(refused).as_str(),
            "window_refused"
        );
    }

    #[test]
    fn unusable_anchors_prose_names_the_cause() {
        let refused = RefusedWindow::AnchorsUnusable {
            start_ms: 0,
            end_ms: 20_000,
            budget_ms: 15_000,
            cause: UnusableAnchors::NoInteriorCut,
        };
        assert_eq!(
            refused.to_string(),
            "audio window 0ms-20000ms (20000ms) exceeds alignment budget 15000ms, and its recovered word anchors cannot split it (no_interior_cut)"
        );
    }

    #[test]
    fn split_window_prose_and_label() {
        let split = SplitWindow {
            start_ms: 1_000,
            end_ms: 31_000,
            budget_ms: 15_000,
            pieces: 3,
        };
        assert_eq!(
            split.to_string(),
            "audio window 1000ms-31000ms (30000ms) exceeds alignment budget 15000ms; aligned as 3 pieces cut at recovered word anchors"
        );
        assert_eq!(
            FaStrategy::WindowSplitAtAnchors(split).as_str(),
            "window_split_at_anchors"
        );
    }

    #[test]
    fn fa_decision_record_carries_strategy_metadata() {
        let decision = DecisionRecord {
            line_idx: LineIdx::new(3),
            speaker: "MOT".into(),
            strategy: DecisionStrategy::Fa(FaStrategy::GapFilled),
            reason: "gap=500ms".into(),
            needs_review: true,
        };
        assert_eq!(decision.strategy.module(), DecisionModule::Fa);
        assert_eq!(decision.strategy.strategy_name(), "gap_filled");
        assert_eq!(decision.evidence_summary(), "fa:gap_filled gap=500ms");
    }

    /// Running a pipeline command removes legacy review scaffolding and does
    /// not replace it with fresh CHAT tiers. The decisions remain available in
    /// the structured run evidence tested by the orchestration layer.
    #[test]
    fn decision_tier_policy_strips_legacy_tiers() {
        let chat_text = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI|2;0.0||||Target_Child|||
*CHI:\thello . \u{0015}1000_2000\u{0015}
%xalign:\told_decision
%xrev:\t[ok]
@End
";
        let mut chat = parse_chat(chat_text);

        strip_decision_tiers(&mut chat);

        let output = chat.to_chat_string();
        assert!(!output.contains("%xalign:"), "output:\n{output}");
        assert!(!output.contains("%xrev:"), "output:\n{output}");
    }
}
