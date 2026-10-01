//! Trace data structures for algorithm visualization.
//!
//! Orchestrators return structured result types (e.g. [`super::results::FaResult`])
//! that always carry intermediate data.  When `debug_traces` is enabled for a
//! job, the dispatch layer converts these results into trace structs and stores
//! them in the ephemeral [`crate::trace_store::TraceStore`].  The dashboard
//! fetches them via `GET /jobs/{id}/traces`.
//!
//! When `debug_traces` is off (the default), structured results are still
//! returned but traces are not stored, no extra memory is used.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::api::{DurationMs, DurationSeconds};
use crate::chat_ops::fa::origin::{ClampBound, Origin};

// ---------------------------------------------------------------------------
// Top-level containers
// ---------------------------------------------------------------------------

/// All algorithm traces collected for a completed job.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct JobTraces {
    /// Per-file traces, keyed by file index (0-based).
    pub files: BTreeMap<usize, FileTraces>,
}

/// Algorithm traces for a single file within a job.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct FileTraces {
    /// Original filename (e.g. "01DM_18.cha").
    pub filename: crate::api::DisplayPath,
    /// DP alignment traces (one per alignment call: FA, retokenize, WER).
    pub dp_alignments: Vec<DpAlignmentTrace>,
    /// ASR post-processing pipeline trace (transcribe jobs only).
    pub asr_pipeline: Option<AsrPipelineTrace>,
    /// Forced alignment timeline trace (align jobs only).
    pub fa_timeline: Option<FaTimelineTrace>,
    /// Retokenization traces (one per utterance that was retokenized).
    pub retokenizations: Vec<RetokenizationTrace>,
}

// ---------------------------------------------------------------------------
// DP Alignment
// ---------------------------------------------------------------------------

/// Full matrix + traceback for a single `align_small` invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct DpAlignmentTrace {
    /// What triggered this alignment (e.g. "fa_whisper", "retokenize", "wer").
    pub context: String,
    /// Payload sequence (left side).
    pub payload: Vec<String>,
    /// Reference sequence (top side).
    pub reference: Vec<String>,
    /// Match mode used ("exact" or "case_insensitive").
    pub match_mode: String,
    /// Number of prefix elements stripped before DP.
    pub prefix_stripped: usize,
    /// Number of suffix elements stripped before DP.
    pub suffix_stripped: usize,
    /// Flat cost matrix (row-major, `(ref_len+1) * (pay_len+1)` entries).
    pub cost_matrix: Vec<usize>,
    /// Traceback path through the cost matrix.
    pub traceback: Vec<AlignStepTrace>,
    /// Final alignment result.
    pub result: Vec<AlignResultTrace>,
}

/// A single step in the DP traceback path.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct AlignStepTrace {
    /// Action taken: "match", "substitution", "extra_payload", "extra_reference".
    pub action: String,
    /// Row index in the cost matrix.
    pub i: usize,
    /// Column index in the cost matrix.
    pub j: usize,
}

/// A single item in the alignment result (matches `AlignResult` enum).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct AlignResultTrace {
    /// "match", "extra_payload", or "extra_reference".
    pub kind: String,
    /// The string key.
    pub key: String,
    /// Index into payload (present for match and extra_payload).
    pub payload_idx: Option<usize>,
    /// Index into reference (present for match and extra_reference).
    pub reference_idx: Option<usize>,
}

// ---------------------------------------------------------------------------
// ASR Pipeline
// ---------------------------------------------------------------------------

/// Intermediate word lists at each stage of ASR post-processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct AsrPipelineTrace {
    /// Stage 0: raw tokens from the ASR worker.
    pub raw_tokens: Vec<AsrTokenTrace>,
    /// Stage 1: after compound merging.
    pub after_compound_merge: Vec<WordTrace>,
    /// Stage 2: after timed word extraction (seconds → ms).
    pub after_timing_extract: Vec<TimedWordTrace>,
    /// Stage 3: after multi-word splitting.
    pub after_multiword_split: Vec<TimedWordTrace>,
    /// Stage 4: after number expansion.
    pub after_number_expand: Vec<TimedWordTrace>,
    /// Stage 2d: after Cantonese normalization (only if lang=yue).
    ///
    /// Normalization runs once per monologue, before the multi-word split, so
    /// this sits between `after_timing_extract` and `after_multiword_split`
    /// even though the field is listed after `after_number_expand` for wire
    /// compatibility. It ran per word after number expansion (as stage 4b)
    /// until 2026-09-16.
    pub after_cantonese_norm: Option<Vec<TimedWordTrace>>,
    /// Stage 5: after long-turn splitting (nested by turn).
    pub after_long_turn_split: Vec<Vec<TimedWordTrace>>,
    /// Stage 6: final utterances.
    pub final_utterances: Vec<UtteranceTrace>,
}

/// A raw ASR token (stage 0).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct AsrTokenTrace {
    /// Token text.
    pub value: String,
    /// Provider start time in seconds; null when no endpoint was supplied.
    pub ts: Option<DurationSeconds>,
    /// Provider end time in seconds; null when no endpoint was supplied.
    pub end_ts: Option<DurationSeconds>,
    /// Token type ("text", "punctuation", etc.).
    pub token_type: String,
}

/// A word without timing (e.g. after compound merge).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct WordTrace {
    /// Word text.
    pub text: String,
}

/// A word with optional timing in milliseconds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct TimedWordTrace {
    /// Word text.
    pub text: String,
    /// Start time in ms (if known).
    pub start_ms: Option<i64>,
    /// End time in ms (if known).
    pub end_ms: Option<i64>,
}

/// A final utterance (stage 6).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct UtteranceTrace {
    /// Speaker index (0-based).
    pub speaker: usize,
    /// Words in the utterance.
    pub words: Vec<TimedWordTrace>,
}

// ---------------------------------------------------------------------------
// FA Timeline
// ---------------------------------------------------------------------------

/// Schema written by the current [`FaTimelineTrace`] producer.
pub const CURRENT_FA_EVIDENCE_SCHEMA_VERSION: u32 = 6;

/// Forced alignment trace: grouping, timing injection, and post-processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct FaTimelineTrace {
    /// Evidence schema revision. Version 1 records pre-injection timing,
    /// confidence, and full boundary provenance. Version 2 additionally
    /// records the typed post-injection decisions that altered or removed
    /// timing. Version 3 adds stable utterance ordinals to numeric
    /// monotonicity effects. Version 4 adds the flat `dropped_word_timings`
    /// section, so a discarded measurement is readable without walking the
    /// tagged effect union and joining each drop back to its parent;
    /// `post_injection_timings` remains reserved for a later complete
    /// per-word outcome projection. Version 5 adds `refused_window` to
    /// `window_refused` decisions: a `cause`-tagged object carrying that
    /// cause's own bounds and figures, so a refused window is data instead of
    /// prose in `reason`. Version 6 records how each group was EXECUTED: every
    /// group carries a `span`, `single` (one request, with its evidence
    /// source and cache key) or `anchored` (an over-budget utterance aligned
    /// as several requests cut at recovered word anchors, each piece with its
    /// window, first and last word, source and cache key). The top-level
    /// `evidence_sources` and `cache_keys` arrays moved into those spans,
    /// since an anchored group has one of each per piece; a
    /// `window_split_at_anchors` decision carries `split_window`, and
    /// `refused_window` gains the `anchor_gap` cause.
    #[serde(default)]
    pub evidence_schema_version: u32,
    /// Forced-alignment engine selected for this run.
    #[serde(default)]
    pub engine: String,
    /// The FA engine name the selected worker reported after FA loaded
    /// (`FaCacheNamespace`), byte for byte: the namespace this run's FA
    /// cache rows and evidence envelopes are written and admitted under.
    #[serde(default)]
    pub engine_version: String,
    /// Utterance groups for batched FA, each with how it was executed and
    /// where each request's evidence came from.
    pub groups: Vec<FaGroupTrace>,
    /// Pre-injection timings per group, per word (None = untimed).
    pub pre_injection_timings: Vec<Vec<Option<TimingTrace>>>,
    /// Post-injection timings after post-processing fixes.
    pub post_injection_timings: Vec<Vec<Option<TimingTrace>>>,
    /// Typed decisions made after inference, including timing removal and
    /// clamping that cannot be reconstructed from final CHAT alone.
    #[serde(default)]
    pub decisions: Vec<FaDecisionTrace>,
    /// Structured numeric facts for every monotonicity decision. This is
    /// deliberately separate from the human-readable decision reason.
    #[serde(default)]
    pub timing_decisions: Vec<FaTimingDecisionTrace>,
    /// Every word timing this run discarded outright, flattened out of
    /// [`Self::timing_decisions`] so each measurement stands on its own.
    ///
    /// DERIVED, never assembled by hand: `FaResult::into_timeline_trace`
    /// builds it from the timing decisions above via
    /// [`FaTimingDecisionTrace::dropped_word_timings`], so the two cannot
    /// disagree and no producer can add a drop to one without the other.
    /// What it adds over the nested form is the JOIN a reviewer would
    /// otherwise do by hand: the speaker, the utterance and the bound live
    /// on the parent effect, so a dropped span read out of `timing_decisions`
    /// cannot say whose word it was or what it exceeded.
    ///
    /// Always written, empty when nothing was discarded. An absent key and an
    /// empty one read identically to a consumer that does not know which
    /// schema version produced the file, and "this run discarded nothing" is
    /// a fact worth stating rather than one to infer from silence.
    #[serde(default)]
    pub dropped_word_timings: Vec<DroppedWordTimingRecord>,
    /// Gap-healing policy, as the `Debug` spelling of `WordGapHealing`
    /// (`"Heal"` / `"PreserveMeasured"`). A string because this trace is a
    /// serialization boundary shared with the dashboard.
    pub gap_healing: String,
    /// Validation violations detected (e.g. E362, E704).
    ///
    /// Empty on every trace FA now emits: align post-validation is a
    /// fail-closed gate, so a file with violations fails and never reaches
    /// the point where a timeline is written. Retained because this struct is
    /// a serialization boundary shared with the dashboard, which still
    /// declares the field.
    pub violations: Vec<ViolationTrace>,
    /// Engine fallback events that occurred while aligning this file.
    pub fallback_events: Vec<FaFallbackEventTrace>,
}

/// One typed pipeline decision retained in a forced-alignment evidence file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct FaDecisionTrace {
    /// Index into `ChatFile.lines`, including headers.
    pub line_idx: usize,
    /// Speaker code on the affected utterance.
    pub speaker: String,
    /// Typed producer module rendered using its stable wire name.
    pub module: String,
    /// Typed strategy rendered using its stable wire name.
    pub strategy: String,
    /// Structured key/value explanation emitted by the decision point.
    pub reason: String,
    /// Whether the decision requires human review.
    pub needs_review: bool,
    /// The refused audio window, present exactly for `window_refused`
    /// decisions. Populated only in `From<DecisionRecord>`, from the typed
    /// strategy, so no other strategy can carry it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refused_window: Option<RefusedWindowTrace>,
    /// The split window, present exactly for `window_split_at_anchors`
    /// decisions, populated the same way as `refused_window`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split_window: Option<SplitWindowTrace>,
}

/// Wire form of [`batchalign_transform::decisions::SplitWindow`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct SplitWindowTrace {
    /// Start of the utterance window, in file milliseconds.
    pub start_ms: u64,
    /// End of the utterance window, in file milliseconds.
    pub end_ms: u64,
    /// The engine budget every piece fits, in milliseconds.
    pub budget_ms: u64,
    /// How many pieces the window was aligned as.
    pub pieces: usize,
}

impl From<batchalign_transform::decisions::SplitWindow> for SplitWindowTrace {
    fn from(split: batchalign_transform::decisions::SplitWindow) -> Self {
        let batchalign_transform::decisions::SplitWindow {
            start_ms,
            end_ms,
            budget_ms,
            pieces,
        } = split;
        Self {
            start_ms,
            end_ms,
            budget_ms,
            pieces,
        }
    }
}

/// Wire form of [`batchalign_transform::decisions::RefusedWindow`]: one
/// variant per cause, tagged by `cause`, each carrying only its own numbers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(tag = "cause", rename_all = "snake_case")]
pub enum RefusedWindowTrace {
    /// Longer than the engine's alignment budget.
    OverBudget {
        /// Start of the window, in file milliseconds.
        start_ms: u64,
        /// End of the window, in file milliseconds.
        end_ms: u64,
        /// The budget exceeded, in milliseconds.
        budget_ms: u64,
    },
    /// The window holds no audio.
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
        /// How far past the recording's end, in milliseconds.
        exceeds_by_ms: u64,
    },
    /// Over budget, and the recovered word anchors leave a stretch longer
    /// than the budget to cut across: the utterance's placement needs review.
    AnchorGap {
        /// Start of the window, in file milliseconds.
        start_ms: u64,
        /// End of the window, in file milliseconds.
        end_ms: u64,
        /// The budget exceeded, in milliseconds.
        budget_ms: u64,
        /// Start of the widest uncrossable stretch, in file milliseconds.
        gap_start_ms: u64,
        /// End of that stretch, in file milliseconds.
        gap_end_ms: u64,
    },
    /// Over budget; recovery matched the utterance but gave no usable cut.
    AnchorsUnusable {
        /// Start of the window, in file milliseconds.
        start_ms: u64,
        /// End of the window, in file milliseconds.
        end_ms: u64,
        /// The budget exceeded, in milliseconds.
        budget_ms: u64,
        /// Why the matches could not be cut at. Not `cause`, which is this
        /// object's own tag.
        unusable: UnusableAnchorsTrace,
    },
}

/// Wire form of [`batchalign_transform::decisions::UnusableAnchors`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum UnusableAnchorsTrace {
    /// Every match was fuzzy or to a multi-word token.
    NoReliableAnchors,
    /// The anchors were refused as a set.
    AnchorsRefused,
    /// The anchors describe a different word list.
    AnchorsDescribeOtherWords,
    /// No anchor can be a cut.
    NoInteriorCut,
}

impl From<batchalign_transform::decisions::UnusableAnchors> for UnusableAnchorsTrace {
    fn from(cause: batchalign_transform::decisions::UnusableAnchors) -> Self {
        use batchalign_transform::decisions::UnusableAnchors;
        match cause {
            UnusableAnchors::NoReliableAnchors => Self::NoReliableAnchors,
            UnusableAnchors::AnchorsRefused => Self::AnchorsRefused,
            UnusableAnchors::AnchorsDescribeOtherWords => Self::AnchorsDescribeOtherWords,
            UnusableAnchors::NoInteriorCut => Self::NoInteriorCut,
        }
    }
}

impl From<batchalign_transform::decisions::RefusedWindow> for RefusedWindowTrace {
    fn from(window: batchalign_transform::decisions::RefusedWindow) -> Self {
        use batchalign_transform::decisions::RefusedWindow;
        match window {
            RefusedWindow::OverBudget {
                start_ms,
                end_ms,
                budget_ms,
            } => Self::OverBudget {
                start_ms,
                end_ms,
                budget_ms,
            },
            RefusedWindow::Empty { at_ms } => Self::Empty { at_ms },
            RefusedWindow::Inverted { start_ms, end_ms } => Self::Inverted { start_ms, end_ms },
            RefusedWindow::PastRecording {
                start_ms,
                end_ms,
                exceeds_by_ms,
            } => Self::PastRecording {
                start_ms,
                end_ms,
                exceeds_by_ms,
            },
            RefusedWindow::AnchorGap {
                start_ms,
                end_ms,
                budget_ms,
                gap_start_ms,
                gap_end_ms,
            } => Self::AnchorGap {
                start_ms,
                end_ms,
                budget_ms,
                gap_start_ms,
                gap_end_ms,
            },
            RefusedWindow::AnchorsUnusable {
                start_ms,
                end_ms,
                budget_ms,
                cause,
            } => Self::AnchorsUnusable {
                start_ms,
                end_ms,
                budget_ms,
                unusable: cause.into(),
            },
        }
    }
}

impl From<batchalign_transform::decisions::DecisionRecord> for FaDecisionTrace {
    fn from(record: batchalign_transform::decisions::DecisionRecord) -> Self {
        use batchalign_transform::decisions::{DecisionStrategy, FaStrategy};
        let (refused_window, split_window) = match record.strategy {
            DecisionStrategy::Fa(FaStrategy::WindowRefused(window)) => (Some(window.into()), None),
            DecisionStrategy::Fa(FaStrategy::WindowSplitAtAnchors(split)) => {
                (None, Some(split.into()))
            }
            // Every other FA strategy listed, not `Fa(_)`: a new one that
            // carries data must decide here whether the trace records it.
            DecisionStrategy::Fa(
                FaStrategy::GapFilled
                | FaStrategy::BoundaryAveraged
                | FaStrategy::LisRemoval
                | FaStrategy::TimingStripped
                | FaStrategy::WordsTimingDropped
                | FaStrategy::NarrowBulletRescued
                | FaStrategy::KeptBulletWindowWidened
                | FaStrategy::WordsClampedToKeptBullet
                | FaStrategy::WordsUntimedForKeptAbsence
                | FaStrategy::TimingProvenance
                | FaStrategy::UnplaceableRun,
            )
            | DecisionStrategy::Utr(_)
            | DecisionStrategy::Monotonicity(_)
            | DecisionStrategy::Morphosyntax(_)
            | DecisionStrategy::Coref(_)
            | DecisionStrategy::Utseg(_) => (None, None),
        };
        Self {
            line_idx: record.line_idx.raw(),
            speaker: record.speaker,
            module: record.strategy.module().as_str().to_owned(),
            strategy: record.strategy.strategy_name().to_owned(),
            reason: record.reason,
            needs_review: record.needs_review,
            refused_window,
            split_window,
        }
    }
}

/// Machine-readable numeric effect of one monotonicity decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FaTimingDecisionTrace {
    /// A later utterance's start went backward in document order.
    StartRegressionStripped {
        /// Affected line index, including headers.
        line_idx: usize,
        /// Stable ordinal among utterances only.
        utterance_idx: usize,
        /// Speaker on the affected utterance.
        speaker: String,
        /// Measured start that regressed.
        start_ms: u64,
        /// Greatest preceding start in document order.
        previous_start_ms: u64,
        /// Line that supplied `previous_start_ms`.
        previous_line_idx: usize,
        /// Stable ordinal of the preceding utterance.
        previous_utterance_idx: usize,
        /// Speaker on the preceding line.
        previous_speaker: String,
    },
    /// An earlier utterance could not be end-clamped without becoming empty.
    ZeroDurationClampStripped {
        /// Affected line index, including headers.
        line_idx: usize,
        /// Stable ordinal among utterances only.
        utterance_idx: usize,
        /// Speaker on the affected utterance.
        speaker: String,
        /// Original start of the earlier utterance.
        start_ms: u64,
        /// Original end of the earlier utterance.
        original_end_ms: u64,
        /// Following start that made a positive clamp impossible.
        next_start_ms: u64,
        /// Following line that supplied `next_start_ms`.
        next_line_idx: usize,
        /// Stable ordinal of the following timed utterance.
        next_utterance_idx: usize,
        /// Speaker on the following line.
        next_speaker: String,
    },
    /// A same-speaker end overlap was resolved because only the bullet's
    /// inherited coverage overshot the next utterance's start; no measured
    /// word conflicted, and no word timing changed.
    EndClampedCoverageOnly {
        /// The two utterances involved. Flattened so this variant's OWN
        /// field set (`line_idx`, `utterance_idx`, `speaker`,
        /// `original_end_ms`, `next_line_idx`, `next_utterance_idx`,
        /// `next_speaker`, `clamped_to_ms`) matches exactly what the single
        /// pre-2026-09-01 `end_clamped` tag emitted (2026-09-01 review, item
        /// 5 factored the SEVEN shared fields into `OverlapEdgeTrace`
        /// without renaming or reshaping any of them here).
        #[serde(flatten)]
        edge: OverlapEdgeTrace,
        /// The end this bullet was clamped to.
        clamped_to_ms: u64,
    },
    /// A same-speaker end overlap was resolved by replacing both
    /// utterances' inherited boundary with their measured word hulls.
    EndClampedBoundaryFromWords {
        /// The two utterances involved.
        #[serde(flatten)]
        edge: OverlapEdgeTrace,
        /// The previous utterance's measured last-word end, its new bullet end.
        prev_hull_end_ms: u64,
        /// The next utterance's measured first-word start, its new bullet start.
        next_hull_start_ms: u64,
    },
    /// A same-speaker end overlap could not be resolved from measurement
    /// alone: the words themselves interleave, or the next utterance has
    /// none. The bullet and every previous-utterance word past the bound
    /// were clamped.
    EndClampedInterleavedWords {
        /// The two utterances involved.
        #[serde(flatten)]
        edge: OverlapEdgeTrace,
        /// Following start used as the new end.
        clamped_to_ms: u64,
        /// Words cut to a shorter positive extent, still keeping a timing.
        words_trimmed: usize,
        /// Words whose start was at or past the bound: no timing survived.
        /// One record per word (2026-09-02), not just a count: each is a
        /// MEASURED extent this run threw away, not the same fact as
        /// `words_trimmed` (formerly folded into one `words_clamped` count
        /// that could not tell the two apart, then briefly into a
        /// `words_dropped: usize` that could say how many but not which).
        words_dropped: Vec<DroppedWordTimingTrace>,
    },
    /// Under `--main-bullets keep`, a derived bullet gave way to a kept
    /// neighbour's fixed boundary (its end, or its start); the kept bullet did
    /// not change.
    YieldedToKeptBullet {
        /// Yielding line index, including headers.
        line_idx: usize,
        /// Stable ordinal of the yielding utterance.
        utterance_idx: usize,
        /// Speaker on the yielding utterance.
        speaker: String,
        /// Line of the kept utterance.
        kept_line_idx: usize,
        /// Stable ordinal of the kept utterance.
        kept_utterance_idx: usize,
        /// Speaker on the kept utterance.
        kept_speaker: String,
        /// The kept boundary yielded to.
        kept_boundary_ms: u64,
        /// What happened to the yielding utterance.
        outcome: KeptBulletYieldTrace,
    },
}

/// Wire form of `chat_ops::fa::KeptBulletYield`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KeptBulletYieldTrace {
    /// The start moved forward to the kept end; leading words were cut.
    StartMoved {
        /// Start before the move.
        from_ms: u64,
        /// Start after the move.
        to_ms: u64,
        /// Leading words cut to a shorter positive extent.
        words_trimmed: usize,
        /// Leading words that lost their timing.
        words_dropped: Vec<DroppedWordTimingTrace>,
    },
    /// No valid bullet could remain: timing stripped.
    Stripped {
        /// Start of the stripped bullet.
        start_ms: u64,
        /// End of the stripped bullet.
        end_ms: u64,
    },
}

/// Wire form of `chat_ops::fa::orchestrate::DroppedWordTiming`: a word whose
/// timing was thrown away entirely, past the clamp bound, with the extent
/// it lost. `tier` follows the same convention as `OriginTrace::ClampedTo`'s
/// `bound` field: a plain string rather than a nested enum, since nothing
/// downstream of this wire boundary needs to match on it structurally.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct DroppedWordTimingTrace {
    /// This word's position among the words visited on its own tier, in
    /// document order.
    pub word_index: usize,
    /// Which tier this word's dropped timing lived on: `"main_tier"` or
    /// `"wor"`.
    pub tier: String,
    /// Start of the extent that was lost.
    pub start_ms: u64,
    /// End of the extent that was lost.
    pub end_ms: u64,
}

impl From<&crate::chat_ops::fa::DroppedWordTiming> for DroppedWordTimingTrace {
    fn from(dropped: &crate::chat_ops::fa::DroppedWordTiming) -> Self {
        use crate::chat_ops::fa::WordTier;
        Self {
            word_index: dropped.word_index,
            tier: match dropped.tier {
                WordTier::MainTier => "main_tier",
                WordTier::Wor => "wor",
            }
            .to_owned(),
            start_ms: dropped.measured.start_ms,
            end_ms: dropped.measured.end_ms,
        }
    }
}

/// One word timing this run threw away, as a record that needs no parent.
///
/// [`DroppedWordTimingTrace`] carries the extent and the position but lives
/// nested inside one variant of [`FaTimingDecisionTrace`], so the speaker,
/// the utterance and the bound that cut it are only reachable from the
/// enclosing effect. This is the same fact with that join already done, which
/// is what makes it usable: a reviewer asking "what did this run measure and
/// then discard?" reads one flat list instead of walking a tagged union.
///
/// Not a second source of truth: it is derived from the effects by
/// [`FaTimingDecisionTrace::dropped_word_timings`] at the moment the trace is
/// assembled, and there is no constructor that takes a bare span.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct DroppedWordTimingRecord {
    /// Index into `ChatFile.lines` of the utterance that lost the timing.
    pub line_idx: usize,
    /// Stable ordinal among utterances only.
    pub utterance_idx: usize,
    /// Speaker code on that utterance.
    pub speaker: String,
    /// Which tier the dropped timing lived on: `"main_tier"` or `"wor"`.
    pub tier: String,
    /// This word's position among the words on its own tier, in document
    /// order.
    pub word_index: usize,
    /// Start of the measured extent that was lost.
    pub start_ms: u64,
    /// End of the measured extent that was lost.
    pub end_ms: u64,
    /// The clamp bound this measurement exceeded, which is why it was cut.
    pub bound_ms: u64,
}

impl FaTimingDecisionTrace {
    /// The word timings this one decision discarded outright.
    ///
    /// Written as an exhaustive match with every non-dropping variant named,
    /// rather than a catch-all: a future effect that CAN discard a
    /// measurement must then fail to compile here instead of silently
    /// reporting none.
    pub fn dropped_word_timings(&self) -> Vec<DroppedWordTimingRecord> {
        match self {
            Self::StartRegressionStripped { .. }
            | Self::ZeroDurationClampStripped { .. }
            | Self::EndClampedCoverageOnly { .. }
            | Self::EndClampedBoundaryFromWords { .. }
            | Self::YieldedToKeptBullet {
                outcome: KeptBulletYieldTrace::Stripped { .. },
                ..
            } => Vec::new(),
            Self::YieldedToKeptBullet {
                line_idx,
                utterance_idx,
                speaker,
                outcome:
                    KeptBulletYieldTrace::StartMoved {
                        to_ms,
                        words_dropped,
                        ..
                    },
                ..
            } => words_dropped
                .iter()
                .map(|dropped| DroppedWordTimingRecord {
                    line_idx: *line_idx,
                    utterance_idx: *utterance_idx,
                    speaker: speaker.clone(),
                    tier: dropped.tier.clone(),
                    word_index: dropped.word_index,
                    start_ms: dropped.start_ms,
                    end_ms: dropped.end_ms,
                    bound_ms: *to_ms,
                })
                .collect(),
            Self::EndClampedInterleavedWords {
                edge,
                clamped_to_ms,
                words_trimmed: _,
                words_dropped,
            } => words_dropped
                .iter()
                .map(|dropped| DroppedWordTimingRecord {
                    line_idx: edge.line_idx,
                    utterance_idx: edge.utterance_idx,
                    speaker: edge.speaker.clone(),
                    tier: dropped.tier.clone(),
                    word_index: dropped.word_index,
                    start_ms: dropped.start_ms,
                    end_ms: dropped.end_ms,
                    bound_ms: *clamped_to_ms,
                })
                .collect(),
        }
    }
}

/// Wire form of `chat_ops::fa::orchestrate::OverlapEdge`: the same seven
/// fields (`line_idx`, `utterance_idx`, `speaker`, `original_end_ms`,
/// `next_line_idx`, `next_utterance_idx`, `next_speaker`), flattened into
/// each `EndClamped*` variant above, unchanged in name and type from before
/// the three arms shared this type (2026-09-01 review, item 5).
///
/// The WIRE SCHEMA as a whole is NOT unchanged from before this session's
/// work, and this type does not claim it is (2026-09-01 review, item 14):
/// the single pre-session `kind: "end_clamped"` became three tag values
/// (`end_clamped_coverage_only` / `_boundary_from_words` /
/// `_interleaved_words`), and `EndClampedBoundaryFromWords` /
/// `EndClampedInterleavedWords` each carry fields the pre-session shape did
/// not (`prev_hull_end_ms` + `next_hull_start_ms` on the former,
/// `words_clamped` on the latter). Item 5's factoring, the change this
/// comment is actually about, is scoped to the SEVEN shared fields only:
/// it changed how they are DECLARED (one struct, flattened three times)
/// without changing what any of the three already-three-way-split variants
/// emitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct OverlapEdgeTrace {
    /// Affected (previous) line index, including headers.
    pub line_idx: usize,
    /// Stable ordinal of the affected (previous) utterance.
    pub utterance_idx: usize,
    /// Speaker on the affected (previous) utterance.
    pub speaker: String,
    /// The previous utterance's end before clamping.
    pub original_end_ms: u64,
    /// Following line that supplied the clamp boundary.
    pub next_line_idx: usize,
    /// Stable ordinal of the following timed utterance.
    pub next_utterance_idx: usize,
    /// Speaker on the following line.
    pub next_speaker: String,
}

impl From<crate::chat_ops::fa::OverlapEdge> for OverlapEdgeTrace {
    fn from(edge: crate::chat_ops::fa::OverlapEdge) -> Self {
        Self {
            line_idx: edge.line_idx,
            utterance_idx: edge.utterance_idx.raw(),
            speaker: edge.speaker,
            original_end_ms: edge.original_end_ms,
            next_line_idx: edge.next_line_idx,
            next_utterance_idx: edge.next_utterance_idx.raw(),
            next_speaker: edge.next_speaker,
        }
    }
}

impl From<crate::chat_ops::fa::MonotonicityEffect> for FaTimingDecisionTrace {
    fn from(effect: crate::chat_ops::fa::MonotonicityEffect) -> Self {
        use crate::chat_ops::fa::MonotonicityEffect;
        match effect {
            MonotonicityEffect::StartRegressionStripped {
                line_idx,
                utterance_idx,
                speaker,
                start_ms,
                previous_start_ms,
                previous_line_idx,
                previous_utterance_idx,
                previous_speaker,
            } => Self::StartRegressionStripped {
                line_idx,
                utterance_idx: utterance_idx.raw(),
                speaker,
                start_ms,
                previous_start_ms,
                previous_line_idx,
                previous_utterance_idx: previous_utterance_idx.raw(),
                previous_speaker,
            },
            MonotonicityEffect::ZeroDurationClampStripped {
                line_idx,
                utterance_idx,
                speaker,
                start_ms,
                original_end_ms,
                next_start_ms,
                next_line_idx,
                next_utterance_idx,
                next_speaker,
            } => Self::ZeroDurationClampStripped {
                line_idx,
                utterance_idx: utterance_idx.raw(),
                speaker,
                start_ms,
                original_end_ms,
                next_start_ms,
                next_line_idx,
                next_utterance_idx: next_utterance_idx.raw(),
                next_speaker,
            },
            MonotonicityEffect::EndClampedCoverageOnly {
                edge,
                clamped_to_ms,
            } => Self::EndClampedCoverageOnly {
                edge: edge.into(),
                clamped_to_ms,
            },
            MonotonicityEffect::EndClampedBoundaryFromWords {
                edge,
                prev_hull_end_ms,
                next_hull_start_ms,
            } => Self::EndClampedBoundaryFromWords {
                edge: edge.into(),
                prev_hull_end_ms,
                next_hull_start_ms,
            },
            MonotonicityEffect::EndClampedInterleavedWords {
                edge,
                clamped_to_ms,
                words_trimmed,
                words_dropped,
            } => Self::EndClampedInterleavedWords {
                edge: edge.into(),
                clamped_to_ms,
                words_trimmed,
                words_dropped: words_dropped
                    .iter()
                    .map(DroppedWordTimingTrace::from)
                    .collect(),
            },
            MonotonicityEffect::YieldedToKeptBullet {
                line_idx,
                utterance_idx,
                speaker,
                kept_line_idx,
                kept_utterance_idx,
                kept_speaker,
                kept_boundary_ms,
                outcome,
            } => Self::YieldedToKeptBullet {
                line_idx,
                utterance_idx: utterance_idx.raw(),
                speaker,
                kept_line_idx,
                kept_utterance_idx: kept_utterance_idx.raw(),
                kept_speaker,
                kept_boundary_ms,
                outcome: match outcome {
                    crate::chat_ops::fa::KeptBulletYield::StartMoved {
                        from_ms,
                        to_ms,
                        words_trimmed,
                        words_dropped,
                    } => KeptBulletYieldTrace::StartMoved {
                        from_ms,
                        to_ms,
                        words_trimmed,
                        words_dropped: words_dropped
                            .iter()
                            .map(DroppedWordTimingTrace::from)
                            .collect(),
                    },
                    crate::chat_ops::fa::KeptBulletYield::Stripped { start_ms, end_ms } => {
                        KeptBulletYieldTrace::Stripped { start_ms, end_ms }
                    }
                },
            },
        }
    }
}

/// A single FA group (time-windowed batch of utterances).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct FaGroupTrace {
    /// Audio start time in ms.
    pub audio_start_ms: DurationMs,
    /// Audio end time in ms.
    pub audio_end_ms: DurationMs,
    /// Utterance indices covered by this group.
    pub utterance_indices: Vec<usize>,
    /// Words in this group.
    pub words: Vec<String>,
    /// Stable AST-derived identity corresponding one-to-one with `words`.
    #[serde(default)]
    pub word_ids: Vec<String>,
    /// How the group was executed and where each request's evidence came
    /// from (schema 6). Required: there is no honest value to give a group
    /// written before it, whose evidence lived in the top-level arrays.
    pub span: FaGroupSpanTrace,
}

/// How one group's audio was presented to the aligner, with the evidence of
/// each request it took.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FaGroupSpanTrace {
    /// One request over the group's whole window.
    Single {
        /// How the request's timing evidence was obtained.
        source: FaEvidenceSourceTrace,
        /// Content-addressed FA cache key of the request.
        cache_key: String,
    },
    /// One over-budget utterance aligned as several requests, cut at the
    /// ends of words utterance timing recovery heard.
    Anchored {
        /// The pieces in word and time order; they partition the group's
        /// window and words.
        pieces: Vec<FaPieceTrace>,
    },
}

/// One piece of an anchored group: its request's window, words and evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct FaPieceTrace {
    /// Start of the piece's window, in file milliseconds.
    pub start_ms: u64,
    /// End of the piece's window, in file milliseconds.
    pub end_ms: u64,
    /// The first word of the piece, as its index among the utterance's
    /// alignable words.
    pub first_word: usize,
    /// The last word of the piece, likewise.
    pub last_word: usize,
    /// How the piece's timing evidence was obtained.
    pub source: FaEvidenceSourceTrace,
    /// Content-addressed FA cache key of the piece's request.
    pub cache_key: String,
}

/// How one group's word timing evidence was obtained.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum FaEvidenceSourceTrace {
    /// Reprojected from an existing healthy `%wor` tier; original provenance
    /// is unavailable because CHAT bullets cannot store it.
    WorReuse,
    /// Replayed from the content-addressed FA cache.
    Cache,
    /// Reparsed locally from an immutable cached worker response.
    RawEvidenceReplay,
    /// Produced by a worker call during this run.
    Inference,
    /// No usable worker evidence was obtained; words were left unaligned.
    Unaligned,
}

/// One forced-alignment engine fallback that occurred for a single group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct FaFallbackEventTrace {
    /// Group index within the file-local FA grouping.
    pub group_index: usize,
    /// Engine originally requested by the Rust control plane.
    pub from_engine: String,
    /// Engine actually used for the retry.
    pub to_engine: String,
    /// Human-readable reason why the fallback was triggered.
    pub reason: String,
    /// Start of the window the fallback engine was given, in ms: the group's
    /// window, or the piece's for a piece of an anchored group.
    pub audio_start_ms: DurationMs,
    /// End of that window, in ms.
    pub audio_end_ms: DurationMs,
}

/// Start/end timing for a single word.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct TimingTrace {
    /// Start time in ms.
    pub start_ms: i64,
    /// End time in ms.
    pub end_ms: i64,
    /// Model-emitted alignment score, quantized to millionths. This is not a
    /// calibrated probability of boundary accuracy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_score_millionths: Option<u32>,
    /// Full provenance chain for the start boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_origin: Option<OriginTrace>,
    /// Full provenance chain for the end boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_origin: Option<OriginTrace>,
}

impl TimingTrace {
    /// Preserve all timing evidence that survives in `WordTiming` before CHAT
    /// serialization lowers it to two integers.
    pub fn from_word_timing(timing: &crate::chat_ops::fa::WordTiming) -> Self {
        Self {
            start_ms: timing.start_ms as i64,
            end_ms: timing.end_ms as i64,
            model_score_millionths: timing.model_score().map(|score| score.millionths()),
            start_origin: Some(OriginTrace::from(timing.start_origin())),
            end_origin: Some(OriginTrace::from(timing.end_origin())),
        }
    }
}

/// Serializable mirror of the complete forced-alignment origin chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OriginTrace {
    /// A model measured this boundary.
    EngineMeasured {
        /// Engine identity carried by the measurement.
        engine: String,
    },
    /// Timing was already present in the input CHAT.
    TranscriptBullet,
    /// A boundary was reduced to a typed bound.
    ClampedTo {
        /// Typed bound applied to the value.
        bound: String,
        /// Provenance before clamping.
        was: Box<OriginTrace>,
        /// Boundary before clamping.
        original_ms: u64,
        /// Distance beyond the bound.
        overshoot_ms: u64,
    },
    /// A boundary between two words was moved so a word that had collapsed
    /// to near-zero duration keeps a usable extent.
    ///
    /// The alias is load-bearing, not decoration: this tag was written
    /// `repaired_for_order` for the whole life of the stored traces on disk,
    /// and `OriginTrace` is a WIRE type. Renaming it without the alias made
    /// every stored trace carrying the old tag fail to deserialize, so a
    /// dashboard read of any earlier run's evidence broke. The rename is a
    /// change of NAME, not of meaning, so the old spelling still reads.
    /// Contrast `estimated_from_word_count`, which is retired rather than
    /// renamed and must stay refused; see
    /// [`tests::a_retired_word_count_estimate_no_longer_deserializes`].
    #[serde(alias = "repaired_for_order")]
    RebalancedWithNeighbour {
        /// Provenance before the rebalance.
        was: Box<OriginTrace>,
        /// Boundary before the rebalance.
        original_ms: u64,
    },
    /// A boundary was copied from a neighbor.
    InheritedFromNeighbour {
        /// Neighbor boundary that was copied.
        from_ms: u64,
    },
    /// An envelope was built over multiple measured spans.
    MergedFromParts {
        /// Number of spans covered by the envelope.
        parts: usize,
    },
    /// An onset from the next token supplied this end.
    DerivedFromNextOnset,
    /// A constant duration supplied this end.
    FallbackDuration {
        /// Constant duration supplied by the fallback.
        assumed_ms: u64,
    },
    /// A measured onset was attached to this word by character alignment
    /// rather than by matching the label to the word.
    ///
    /// The instant underneath is the engine's; the ATTRIBUTION is ours, so the
    /// nesting is the point: a reviewer can see both that a measurement exists
    /// and that we chose which word it belongs to. The two counts say how far
    /// the fit was from exact.
    AttributedByCharAlignment {
        /// Provenance of the instant, before attribution.
        was: Box<OriginTrace>,
        /// Characters of the word no label character matched.
        transcript_only: usize,
        /// Characters of the labels no word character matched.
        label_only: usize,
    },
}

impl From<&Origin> for OriginTrace {
    fn from(origin: &Origin) -> Self {
        match origin {
            Origin::EngineMeasured { engine } => Self::EngineMeasured {
                engine: engine.to_string(),
            },
            Origin::TranscriptBullet => Self::TranscriptBullet,
            Origin::ClampedTo {
                bound,
                was,
                original,
                overshoot,
            } => Self::ClampedTo {
                bound: match bound {
                    ClampBound::RecordingEnd => "recording_end",
                    ClampBound::UtteranceBullet => "utterance_bullet",
                    ClampBound::NextOnset => "next_onset",
                }
                .to_owned(),
                was: Box::new(Self::from(was.as_ref())),
                original_ms: original.get(),
                overshoot_ms: overshoot.0,
            },
            Origin::RebalancedWithNeighbour { was, original } => Self::RebalancedWithNeighbour {
                was: Box::new(Self::from(was.as_ref())),
                original_ms: original.get(),
            },
            Origin::InheritedFromNeighbour { from } => Self::InheritedFromNeighbour {
                from_ms: from.get(),
            },
            Origin::MergedFromParts { parts } => Self::MergedFromParts { parts: *parts },
            Origin::DerivedFromNextOnset => Self::DerivedFromNextOnset,
            Origin::FallbackDuration { assumed } => Self::FallbackDuration {
                assumed_ms: assumed.0,
            },
            Origin::AttributedByCharAlignment { was, edits } => Self::AttributedByCharAlignment {
                was: Box::new(Self::from(was.as_ref())),
                transcript_only: edits.transcript_only,
                label_only: edits.label_only,
            },
        }
    }
}

/// A validation violation detected during FA.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct ViolationTrace {
    /// Error code (e.g. "E362", "E704").
    pub code: String,
    /// Human-readable description.
    pub message: String,
    /// Utterance index where the violation was found.
    pub utterance_index: Option<usize>,
}

// ---------------------------------------------------------------------------
// Retokenization
// ---------------------------------------------------------------------------

/// Retokenization trace for a single utterance.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
pub struct RetokenizationTrace {
    /// Utterance index in the file (0-based).
    pub utterance_index: usize,
    /// Original CHAT words.
    pub original_words: Vec<String>,
    /// Stanza tokens after retokenization.
    pub stanza_tokens: Vec<String>,
    /// Normalized concatenation of original words.
    pub normalized_original: String,
    /// Normalized concatenation of Stanza tokens.
    pub normalized_tokens: String,
    /// Word→token index mapping: `mapping[word_idx]` = list of token indices.
    pub mapping: Vec<Vec<usize>>,
    /// Whether the fallback (length-proportional) mapping was used.
    pub used_fallback: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_ops::fa::origin::EngineId;
    use crate::chat_ops::fa::{ModelAlignmentScore, WordTiming};
    use crate::time::{FileMs, Ms};

    #[test]
    fn refused_window_is_serialized_only_for_window_refused() {
        use batchalign_transform::decisions::{
            DecisionRecord, DecisionStrategy, FaStrategy, RefusedWindow,
        };
        let refused = RefusedWindow::OverBudget {
            start_ms: 12_000,
            end_ms: 38_520,
            budget_ms: 15_000,
        };
        let record = DecisionRecord::new_and_trace(
            3,
            "CHI".to_owned(),
            DecisionStrategy::Fa(FaStrategy::WindowRefused(refused)),
            refused.to_string(),
            true,
        );
        let json = serde_json::to_value(FaDecisionTrace::from(record)).expect("serializes");
        assert_eq!(json["strategy"], "window_refused");
        assert_eq!(json["refused_window"]["start_ms"], 12_000);
        assert_eq!(json["refused_window"]["end_ms"], 38_520);
        assert_eq!(json["refused_window"]["cause"], "over_budget");
        assert_eq!(json["refused_window"]["budget_ms"], 15_000);

        let other = DecisionRecord::new_and_trace(
            4,
            "CHI".to_owned(),
            DecisionStrategy::Fa(FaStrategy::GapFilled),
            "gap=500ms".to_owned(),
            false,
        );
        let json = serde_json::to_value(FaDecisionTrace::from(other)).expect("serializes");
        assert!(json.get("refused_window").is_none());
        assert!(json.get("split_window").is_none());
    }

    /// Schema 6: an anchor-gap refusal states its uncrossable stretch as data,
    /// and a split carries its piece count, each only on its own strategy.
    #[test]
    fn anchor_gap_and_split_decisions_are_serialized_as_data() {
        use batchalign_transform::decisions::{
            DecisionRecord, DecisionStrategy, FaStrategy, RefusedWindow, SplitWindow,
        };
        let gap = RefusedWindow::AnchorGap {
            start_ms: 0,
            end_ms: 305_800,
            budget_ms: 15_000,
            gap_start_ms: 5_800,
            gap_end_ms: 300_800,
        };
        let record = DecisionRecord::new_and_trace(
            6,
            "PAR".to_owned(),
            DecisionStrategy::Fa(FaStrategy::WindowRefused(gap)),
            gap.to_string(),
            true,
        );
        let json = serde_json::to_value(FaDecisionTrace::from(record)).expect("serializes");
        assert_eq!(json["refused_window"]["cause"], "anchor_gap");
        assert_eq!(json["refused_window"]["gap_start_ms"], 5_800);
        assert_eq!(json["refused_window"]["gap_end_ms"], 300_800);
        assert!(json.get("split_window").is_none());

        let split = SplitWindow {
            start_ms: 0,
            end_ms: 33_800,
            budget_ms: 15_000,
            pieces: 3,
        };
        let record = DecisionRecord::new_and_trace(
            6,
            "PAR".to_owned(),
            DecisionStrategy::Fa(FaStrategy::WindowSplitAtAnchors(split)),
            split.to_string(),
            false,
        );
        let json = serde_json::to_value(FaDecisionTrace::from(record)).expect("serializes");
        assert_eq!(json["strategy"], "window_split_at_anchors");
        assert_eq!(json["split_window"]["pieces"], 3);
        assert_eq!(json["needs_review"], false);
        assert!(json.get("refused_window").is_none());
    }

    #[test]
    fn timing_trace_preserves_score_and_complete_origin_chain() {
        let measured = Origin::EngineMeasured {
            engine: EngineId::new("wav2vec_fa"),
        };
        let adjusted_start = Origin::ClampedTo {
            bound: ClampBound::UtteranceBullet,
            was: Box::new(measured.clone()),
            original: FileMs::new(90),
            overshoot: Ms(10),
        };
        let adjusted_end = Origin::RebalancedWithNeighbour {
            was: Box::new(measured),
            original: FileMs::new(220),
        };
        let score = ModelAlignmentScore::try_from_f64(0.812_345).expect("fixture score is valid");
        let timing = WordTiming::new(100, 200, adjusted_start, adjusted_end)
            .expect("fixture timing has positive extent")
            .with_model_score(score);

        let trace = TimingTrace::from_word_timing(&timing);
        let json = serde_json::to_value(trace).expect("timing trace should serialize");

        assert_eq!(json["start_ms"], 100);
        assert_eq!(json["end_ms"], 200);
        assert_eq!(json["model_score_millionths"], 812_345);
        assert_eq!(json["start_origin"]["kind"], "clamped_to");
        assert_eq!(json["start_origin"]["bound"], "utterance_bullet");
        assert_eq!(json["start_origin"]["was"]["kind"], "engine_measured");
        assert_eq!(json["start_origin"]["was"]["engine"], "wav2vec_fa");
        assert_eq!(json["end_origin"]["kind"], "rebalanced_with_neighbour");
        assert_eq!(json["end_origin"]["was"]["kind"], "engine_measured");
    }

    /// The word-count estimate was RETIRED (2026-09-07) because nothing in
    /// the pipeline ever produced it: the word-count arithmetic in
    /// `chat_ops::fa::grouping` yields an audio WINDOW, and the timings that
    /// come back inside that window are measured by the engine.
    ///
    /// `OriginTrace` is a wire type, so the retirement is asserted here rather
    /// than by a type: a running program can only observe the removal by the
    /// tag failing to deserialize. A trace bearing this kind can now only be
    /// corrupt or hand-forged, and reading it back would reintroduce exactly
    /// the fabricated provenance the `Origin` type exists to prevent.
    #[test]
    fn a_retired_word_count_estimate_no_longer_deserializes() {
        let stored = r#"{"kind":"estimated_from_word_count","gap_ms":290,"words_before":40,"words_total":180}"#;
        assert!(
            serde_json::from_str::<OriginTrace>(stored).is_err(),
            "the retired word-count estimate must not deserialize back into existence"
        );
    }

    /// RED FIRST: a trace stored under the OLD tag must still deserialize.
    /// The rename `repaired_for_order` -> `rebalanced_with_neighbour` is a
    /// change of name, not of meaning, and this is a wire type: without the
    /// alias every trace already on disk stopped reading back.
    #[test]
    fn a_trace_stored_under_the_old_order_repair_tag_still_deserializes() {
        let stored =
            r#"{"kind":"repaired_for_order","was":{"kind":"transcript_bullet"},"original_ms":220}"#;
        let decoded: OriginTrace =
            serde_json::from_str(stored).expect("the old tag must still read back");
        match decoded {
            OriginTrace::RebalancedWithNeighbour { was, original_ms } => {
                assert_eq!(original_ms, 220);
                assert!(matches!(*was, OriginTrace::TranscriptBullet));
            }
            other => panic!("the old tag must decode to the renamed variant, got: {other:?}"),
        }
    }

    /// A trimmed word and a dropped word are different facts (2026-09-02):
    /// the trace must carry both separately, and the dropped side must
    /// carry the full per-word record (which word, what was measured), not
    /// merely how many, since that is exactly the information practice 15
    /// says must not die at a `usize` boundary.
    #[test]
    fn interleaved_words_trace_carries_trimmed_count_and_dropped_records() {
        use crate::chat_ops::fa::TimeSpan;
        use crate::chat_ops::fa::{DroppedWordTiming, MonotonicityEffect, OverlapEdge, WordTier};
        use talkbank_model::UtteranceIdx;

        let effect = MonotonicityEffect::EndClampedInterleavedWords {
            edge: OverlapEdge {
                line_idx: 5,
                utterance_idx: UtteranceIdx::new(2),
                speaker: "CHI".to_string(),
                original_end_ms: 5_000,
                next_line_idx: 7,
                next_utterance_idx: UtteranceIdx::new(3),
                next_speaker: "CHI".to_string(),
            },
            clamped_to_ms: 4_000,
            words_trimmed: 1,
            words_dropped: vec![DroppedWordTiming {
                word_index: 2,
                tier: WordTier::Wor,
                measured: TimeSpan::new(4_200, 4_800),
            }],
        };

        let trace: FaTimingDecisionTrace = effect.into();
        let json = serde_json::to_value(trace).expect("trace should serialize");

        assert_eq!(json["kind"], "end_clamped_interleaved_words");
        assert_eq!(json["clamped_to_ms"], 4_000);
        assert_eq!(
            json["words_trimmed"], 1,
            "the trimmed count must be present and separate: {json}"
        );
        assert_eq!(
            json["words_dropped"].as_array().map(Vec::len),
            Some(1),
            "the dropped side is a list of records, not a count: {json}"
        );
        assert_eq!(json["words_dropped"][0]["word_index"], 2);
        assert_eq!(json["words_dropped"][0]["tier"], "wor");
        assert_eq!(
            json["words_dropped"][0]["start_ms"], 4_200,
            "the dropped word's measured span must survive to the trace: {json}"
        );
        assert_eq!(json["words_dropped"][0]["end_ms"], 4_800);
    }
}
