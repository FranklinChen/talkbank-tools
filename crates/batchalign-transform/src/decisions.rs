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
    WindowRefused,
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
            Self::WindowRefused => "window_refused",
            Self::WordsTimingDropped => "words_timing_dropped",
            Self::NarrowBulletRescued => "narrow_bullet_rescued",
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
}

impl MonotonicityStrategy {
    /// Stable wire/tracing label for structured evidence.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::EndClampedCoverageOnly => "end_clamped_coverage_only",
            Self::EndClampedBoundaryFromWords => "end_clamped_boundary_from_words",
            Self::EndClampedInterleavedWords => "end_clamped_interleaved_words",
            Self::TimingStripped => "timing_stripped",
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
