//! Utterance Timing Recovery (UTR): inject ASR-derived timing into untimed CHAT utterances.
//!
//! When a CHAT file has a mix of timed and untimed utterances, UTR uses ASR
//! output to recover utterance-level bullets for the untimed ones. This is a
//! pre-pass before forced alignment, FA then operates on a fully-timed file.
//!
//! Algorithm:
//! 0. Partition the census into anchored regions (`regions`): the bullets the
//!    file already has bound both the utterances and the ASR onsets each
//!    region may match, and every step below runs once per region, within
//!    that region's own budget.
//! 1. Bind the region's participating utterance words and speaker identities
//!    to one source census. Adjacent different-speaker turns use a joint bounded
//!    singleton/speaker-run composition, preserving each turn's order and assigning
//!    each ASR word at most once within the region.
//! 2. A single-speaker region retains the cheap O(n+m) fast path: if the words are a
//!    *uniquely embedded* exact monotonic subsequence of the region's tokens,
//!    use that directly.
//! 3. For that monotonic region, if the subsequence is missing or ambiguous, use one
//!    Hirschberg DP alignment of the region's words against its tokens
//!    (`dp_align::CorrespondenceAnalysis::observe`, which runs
//!    `dp_align::align`).
//! 4. Retain selection separately from correspondences common to every optimum
//!    of the declared order model. Only common endpoint proof permits a hint;
//!    a complete candidate hull can bound FA search without choosing a repeat,
//!    and interior-only proof bounds FA search between the neighbouring proved
//!    timings without becoming a hint.
//! 5. Set `utterance.main.content.bullet` from admitted tokens' timing hull
//!    (untimed only), intersected with retained non-overlap timing bounds.
//!    Already-timed utterances are left unchanged; exhausted hints are refused.
//!
//! The joint model permits disjoint adjacent-speaker-run interleavings, not arbitrary
//! transcript reordering or unrestricted multi-speaker overlap. Budget refusal
//! refuses its region only, and grants neither fallback timing authority nor an
//! assertion of lexical absence.

use talkbank_model::model::{ChatFile, Line};

use batchalign_transform::dp_align::{self, MatchMode};

use tracing::debug;

use super::coordinates::{FaWindow, FileMs, Recording, WindowFault};

use super::extraction::collect_fa_words;
use super::presence::RecordingPresence;

mod anchors;
#[cfg(test)]
mod correspondence_tests;
mod evidence;
mod interior;
pub mod interleaving;
mod lexical;
pub mod overlap_markers;
#[cfg(test)]
mod region_tests;
mod regions;
mod search;
mod two_pass;

pub use anchors::{
    AlignableWords, AnchorDisorder, AnchorIndex, AnchorLookup, UtteranceAnchors, WordAnchor,
};
use evidence::UtrAsrTokenOrdinal;
use evidence::UtrWordOrdinal;
pub use evidence::{
    NonEmptyUtrWordMatches, PositiveUtrInterval, UtrAlignmentEvidence, UtrAlignmentPlan,
    UtrAlignmentStrategy, UtrAsrTokenAddress, UtrBudgetRefusal, UtrCorrespondenceRefusal,
    UtrLexicalRelation, UtrOrderModel, UtrOverlapRecovery, UtrRegionSummary, UtrResult,
    UtrSearchBudget, UtrTimingProposal, UtrUtteranceAlignmentEvidence, UtrUtteranceOrdinal,
    UtrWordAddress, UtrWordMatch,
};
pub use lexical::{
    AdmittedUtrWordMatch, EndpointBoundUtrWordMatches, MissingUtrEndpoints,
    NonEmptyAdmittedUtrWordMatches,
};
pub use regions::{UtrRegionEdge, UtrRegionSpan};

/// Synthetic drift-class regression scenarios. Public entry point is
/// [`inject_utr_timing`]; the scenarios generate CHAT + ASR in-memory and
/// assert monotonicity / non-silent-strip invariants on the output. The
/// four long scenarios are retained opt-in assessments. A passing no-drift
/// assertion does not establish complete recovery: the printed summary also
/// records missing turns. Run the selected long deck explicitly, not the sweep.
#[cfg(test)]
mod drift_scenarios;

/// A provider ASR token or segment with timing, used as retained input for UTR.
///
/// This is intentionally a simple struct; it can be constructed from
/// any ASR response format (Python worker `AsrToken`, or any other source).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AsrTimingToken {
    /// Provider text, which may contain multiple words. UTR projects lexical
    /// words while retaining this token's original timing interval.
    pub text: String,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
}

/// Strategy trait for UTR injection.
///
/// Implementations determine how ASR tokens are aligned with CHAT utterances
/// to recover utterance-level timing bullets.
pub trait UtrStrategy: Send + Sync {
    /// Inject utterance-level timing from ASR tokens into untimed CHAT utterances.
    fn inject(&self, chat_file: &mut ChatFile, asr_tokens: &[AsrTimingToken]) -> UtrResult;
}

/// Global single-pass UTR strategy.
///
/// Flattens all utterance words into one reference sequence and runs a single
/// alignment pass (exact-subsequence fast path or Hirschberg DP fallback).
/// This is the original UTR algorithm, monotonic, works well when transcript
/// order matches audio order, but cannot correctly place `+<` overlap
/// backchannels whose words appear at the wrong position in the global sequence.
pub struct GlobalUtr;

pub use two_pass::{
    CaMarkerPolicy, GroupingContext, TwoPassConfig, TwoPassOverlapUtr, UtrFuzzyThreshold,
    UtrMatchMode, UtrOverlapDensityThreshold,
};

/// Select the best UTR strategy for a given CHAT file.
///
/// Returns [`TwoPassOverlapUtr`] when any utterance has a `+<` lazy overlap
/// linker or CA overlap markers (⌊), [`GlobalUtr`] otherwise. When no overlap
/// utterances exist, pass 2 is a no-op, but we avoid the overhead entirely.
///
/// When `grouping_context` is provided, the two-pass strategy uses FA group
/// counts to detect and avoid the wider-window regression on non-English files.
pub fn select_strategy(
    chat_file: &ChatFile,
    grouping_context: Option<GroupingContext>,
) -> Box<dyn UtrStrategy> {
    let has_overlap = chat_file.lines.iter().any(|line| {
        if let Line::Utterance(utt) = line {
            // Check for +< linker (explicit overlap marker)
            if utt
                .main
                .content
                .linkers
                .iter()
                .any(|l| l.kind == talkbank_model::model::LinkerKind::LazyOverlapPrecedes)
            {
                return true;
            }
            // Check for ⌊ CA overlap markers (bottom overlap = overlapping speaker)
            let info = overlap_markers::extract_overlap_info(&utt.main.content.content);
            info.has_bottom_overlap()
        } else {
            false
        }
    });
    if has_overlap {
        Box::new(TwoPassOverlapUtr {
            grouping_context,
            config: two_pass::TwoPassConfig::default(),
        })
    } else {
        Box::new(GlobalUtr)
    }
}

/// Pre-extracted utterance metadata used while planning one UTR pass.
#[derive(Debug, Clone)]
pub(super) struct UtrUtteranceInfo {
    /// Alignable words from the utterance in transcript order.
    pub(super) words: Vec<String>,
    /// Original source timing, not an independently maintained presence flag.
    retained_timing: Option<lexical::ObservedWordTiming>,
    /// Whether the utterance has a `+<` lazy overlap linker.
    pub(super) has_lazy_overlap: bool,
    /// Whether this utterance contains ⌊ (bottom overlap) markers,
    /// indicating it overlaps with a preceding utterance's ⌈ markers.
    pub(super) has_ca_overlap: bool,
    /// For utterances with ⌈ (top overlap begin): the proportional position
    /// of the first ⌈ among the utterance's alignable words (0.0-1.0).
    /// Used by pass 2 to narrow the backchannel recovery window.
    pub(super) overlap_onset_fraction: Option<f64>,
    /// Speaker code for cross-utterance matching.
    pub(super) speaker: String,
    /// Whether the utterance is speech in the recording. One that is not has
    /// no words here and no retained timing, and is neither matched nor
    /// counted unmatched: it is not recovery's to time.
    pub(super) presence: RecordingPresence,
    /// Indices of bottom overlap regions (for index-aware matching with
    /// predecessor tops). `None` = unindexed, `Some(n)` = indexed.
    pub(super) bottom_indices: Vec<Option<talkbank_model::model::OverlapIndex>>,
    /// Per-top-region onset fractions with their indices (for index-aware
    /// predecessor lookup).
    pub(super) top_onsets: Vec<(Option<talkbank_model::model::OverlapIndex>, f64)>,
}

/// Inject utterance-level timing from ASR tokens into untimed CHAT utterances.
///
/// Backward-compatible entry point that delegates to [`GlobalUtr`]. New callers
/// should prefer [`select_strategy`] to automatically choose the best strategy.
pub fn inject_utr_timing(chat_file: &mut ChatFile, asr_tokens: &[AsrTimingToken]) -> UtrResult {
    GlobalUtr.inject(chat_file, asr_tokens)
}

impl UtrStrategy for GlobalUtr {
    /// Global single-pass UTR: flatten all words, align once, assign bullets.
    ///
    /// Attempts a cheap exact-subsequence fast path first. Falls back to a
    /// single global Hirschberg DP alignment when the fast path is ambiguous.
    fn inject(&self, chat_file: &mut ChatFile, asr_tokens: &[AsrTimingToken]) -> UtrResult {
        run_global_utr(
            chat_file,
            asr_tokens,
            GlobalUtrParticipation::AllUtterances,
            MatchMode::CaseInsensitive,
        )
    }
}

/// Which utterances participate in the global UTR alignment payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GlobalUtrParticipation {
    /// Include every utterance in document order.
    AllUtterances,
    /// Exclude marked overlap utterances for later local recovery.
    ExcludeMarkedOverlap,
}

/// Recompute global UTR word-to-token evidence without changing CHAT timing.
///
/// This is the offline replay seam used by evaluation tools. It runs the same
/// planner production UTR consumes, but it returns the plan without applying
/// any bullet projection.
pub fn observe_global_utr_alignment(
    chat_file: &ChatFile,
    asr_tokens: &[AsrTimingToken],
    match_mode: UtrMatchMode,
    participation: GlobalUtrParticipation,
) -> UtrAlignmentPlan {
    plan_global_utr_alignment(
        chat_file,
        asr_tokens,
        match_mode.to_dp_match_mode(),
        participation,
    )
}

/// Core global UTR implementation shared by [`GlobalUtr`] and the first pass
/// of [`TwoPassOverlapUtr`].
///
/// Under [`GlobalUtrParticipation::ExcludeMarkedOverlap`], `+<` utterances are
/// excluded from the flattened word sequence (their words do not participate
/// in the global DP) but are still counted as unmatched. Pass 2 of the
/// two-pass strategy handles them separately.
pub(super) fn run_global_utr(
    chat_file: &mut ChatFile,
    asr_tokens: &[AsrTimingToken],
    participation: GlobalUtrParticipation,
    dp_match_mode: MatchMode,
) -> UtrResult {
    if asr_tokens.is_empty() {
        let utt_infos = collect_utr_utterance_info(chat_file);
        // Nothing can match. The plan still records every utterance as
        // unmatched (or excluded, or wordless) so the evidence is complete,
        // but no decision is written and no bullet is touched: the count is
        // the whole outcome.
        let plan = UtrAlignmentPlan::assemble(
            UtrOrderModel::of(&utt_infos, participation),
            regions::partition(&utt_infos).iter().map(|region| {
                build_alignment_plan(
                    UtrAlignmentStrategy::GlobalDp,
                    &[],
                    &lexical::UtrLexicalStream::from_tokens(asr_tokens),
                    UtrMatchSelection {
                        selected: Vec::new(),
                        admission: CorrespondenceProof::Complete(Vec::new()),
                        per_utterance: region
                            .searched(&utt_infos)
                            .map(|_| UtteranceSelection::UNSEARCHED),
                    },
                    participation,
                    &ContestedTokens::none(),
                )
            }),
        );
        let skipped = utt_infos
            .iter()
            .filter(|info| info.retained_timing.is_some())
            .count();
        return UtrResult {
            injected: 0,
            skipped,
            unmatched: utt_infos.len() - skipped,
            alignment: UtrAlignmentEvidence::Global { plan },
            decisions: Vec::new(),
            // No tokens, so nothing was heard: an index that observed nothing,
            // which never supersedes an earlier pass's anchors.
            anchors: AnchorIndex::not_observed(),
        };
    }
    PreparedGlobalUtr::plan(chat_file, asr_tokens, participation, dp_match_mode).project()
}

/// A selected pass owns the source borrow and every observation derived from
/// that source. No consumer can pair a plan/census with a different document,
/// or change the document between planning and projection.
struct PreparedGlobalUtr<'a> {
    source: &'a mut ChatFile,
    census: Vec<UtrUtteranceInfo>,
    plan: UtrAlignmentPlan,
    anchors: AnchorIndex,
    line_indices: Vec<usize>,
    following_starts: Vec<FollowingStarts>,
}

/// The nearest retained non-overlap start after one utterance, under both
/// order models. The pass's order model chooses which one bounds its hints:
/// a monotonic pass is bounded by any following turn, an interleaved one
/// only by the same speaker's.
#[derive(Debug, Clone, Copy)]
struct FollowingStarts {
    document: Option<u64>,
    same_speaker: Option<u64>,
}

impl FollowingStarts {
    fn under(self, order: UtrOrderModel) -> Option<u64> {
        match order {
            UtrOrderModel::Interleaved => self.same_speaker,
            UtrOrderModel::Monotonic => self.document,
        }
    }
}

impl UtrOrderModel {
    /// The census's order model: interleaved when two adjacent
    /// participating turns belong to different speakers. The one owner of
    /// that rule for planning and projection alike.
    fn of(census: &[UtrUtteranceInfo], participation: GlobalUtrParticipation) -> Self {
        let interleaves = census.windows(2).any(|pair| {
            pair[0].speaker != pair[1].speaker
                && !pair[0].excluded_from(participation)
                && !pair[1].excluded_from(participation)
        });
        if interleaves {
            Self::Interleaved
        } else {
            Self::Monotonic
        }
    }
}

/// Geometry for a non-overlap hint, admitted from retained source boundaries.
/// Exhaustion is a state, never a fabricated positive interval.
enum UtrTimingCorridor {
    Open { floor_end_ms: u64 },
    Bounded(PositiveUtrInterval),
    Exhausted,
}

impl UtrTimingCorridor {
    fn admit(floor_end_ms: u64, ceiling_start_ms: Option<u64>) -> Self {
        match ceiling_start_ms {
            None => Self::Open { floor_end_ms },
            Some(ceiling) => match PositiveUtrInterval::admit(floor_end_ms, ceiling) {
                Some(interval) => Self::Bounded(interval),
                None => Self::Exhausted,
            },
        }
    }

    fn intersect(self, proposal: PositiveUtrInterval) -> Option<PositiveUtrInterval> {
        match self {
            Self::Open { floor_end_ms } => proposal.after(floor_end_ms),
            Self::Bounded(corridor) => PositiveUtrInterval::admit(
                proposal.start_ms().max(corridor.start_ms()),
                proposal.end_ms().min(corridor.end_ms()),
            ),
            Self::Exhausted => None,
        }
    }
}

impl<'a> PreparedGlobalUtr<'a> {
    fn plan(
        source: &'a mut ChatFile,
        tokens: &[AsrTimingToken],
        participation: GlobalUtrParticipation,
        match_mode: MatchMode,
    ) -> Self {
        let census = collect_utr_utterance_info(source);
        let plan = plan_global_utr_alignment_for(&census, tokens, match_mode, participation);
        let anchors = AnchorIndex::from_plan(&plan);
        let line_indices: Vec<_> = source
            .lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| matches!(line, Line::Utterance(_)).then_some(index))
            .collect();
        let mut following_starts = vec![
            FollowingStarts {
                document: None,
                same_speaker: None,
            };
            census.len()
        ];
        let mut ceiling: Option<u64> = None;
        let mut speaker_ceilings = std::collections::BTreeMap::<&str, u64>::new();
        for (ordinal, line_index) in line_indices.iter().enumerate().rev() {
            let info = &census[ordinal];
            following_starts[ordinal] = FollowingStarts {
                document: ceiling,
                same_speaker: speaker_ceilings.get(info.speaker.as_str()).copied(),
            };
            if !info.has_lazy_overlap
                && !info.has_ca_overlap
                && let Line::Utterance(utterance) = &source.lines[*line_index]
                && let Some(bullet) = &utterance.main.content.bullet
            {
                ceiling = Some(ceiling.map_or(bullet.timing.start_ms, |previous| {
                    previous.min(bullet.timing.start_ms)
                }));
                speaker_ceilings
                    .entry(info.speaker.as_str())
                    .and_modify(|previous| *previous = (*previous).min(bullet.timing.start_ms))
                    .or_insert(bullet.timing.start_ms);
            }
        }
        Self {
            source,
            census,
            plan,
            anchors,
            line_indices,
            following_starts,
        }
    }

    /// Consume exactly the source-bound pass; raw tokens are no longer present.
    fn project(self) -> UtrResult {
        let Self {
            source: chat_file,
            census: utt_infos,
            plan,
            anchors,
            line_indices: utt_line_indices,
            following_starts,
        } = self;
        let mut result = UtrResult {
            injected: 0,
            skipped: 0,
            unmatched: 0,
            alignment: UtrAlignmentEvidence::NotRunNoUntimed,
            decisions: Vec::new(),
            anchors,
        };

        let decision = |utt_idx: usize,
                        strategy: batchalign_transform::decisions::UtrStrategy,
                        reason: String,
                        needs_review: bool|
         -> Option<batchalign_transform::decisions::DecisionRecord> {
            let line_idx = *utt_line_indices.get(utt_idx)?;
            let Some(Line::Utterance(utt)) = chat_file.lines.get(line_idx) else {
                return None;
            };
            Some(batchalign_transform::decisions::DecisionRecord {
                line_idx: batchalign_transform::decisions::LineIdx::new(line_idx),
                speaker: utt.main.speaker.as_str().to_string(),
                strategy: batchalign_transform::decisions::DecisionStrategy::Utr(strategy),
                reason,
                needs_review,
            })
        };

        // Project proposals onto untimed utterances only; timed ones are kept.
        // Both floors are kept; the pass's order model chooses which one
        // bounds each hint (see `FollowingStarts`).
        let mut bullets_to_set: Vec<Option<PositiveUtrInterval>> = vec![None; utt_infos.len()];
        let mut document_floor_ms = 0;
        let mut speaker_floors = std::collections::BTreeMap::<&str, u64>::new();
        let order = plan.strategy;
        for (utt_idx, (info, evidence)) in utt_infos.iter().zip(&plan.utterances).enumerate() {
            // Not in the recording: not recovery's to time, so neither
            // counted (skipped, injected or unmatched) nor reviewed, whatever
            // its evidence says (an overlap-marked note is "excluded" in a
            // two-pass first pass).
            match info.presence {
                RecordingPresence::InRecording => {}
                RecordingPresence::NotInRecording(_) => continue,
            }
            let speaker_floor_ms = speaker_floors
                .get(info.speaker.as_str())
                .copied()
                .unwrap_or(0);
            let floor_end_ms = match order {
                UtrOrderModel::Interleaved => speaker_floor_ms,
                UtrOrderModel::Monotonic => document_floor_ms,
            };
            let ceiling_start_ms = following_starts[utt_idx].under(order);
            let overlaps = info.has_lazy_overlap || info.has_ca_overlap;
            if info.retained_timing.is_some() {
                result.skipped += 1;
                if !overlaps
                    && let Some(Line::Utterance(utt)) =
                        chat_file.lines.get(utt_line_indices[utt_idx])
                    && let Some(bullet) = &utt.main.content.bullet
                {
                    document_floor_ms = document_floor_ms.max(bullet.timing.end_ms);
                    speaker_floors.insert(
                        info.speaker.as_str(),
                        speaker_floor_ms.max(bullet.timing.end_ms),
                    );
                }
                continue;
            }
            match evidence {
                UtrUtteranceAlignmentEvidence::Matched {
                    proposal: UtrTimingProposal::Positive { interval },
                    admitted_matches: matches,
                    alignable_words,
                    ..
                } => {
                    let projected = if overlaps {
                        Some(*interval)
                    } else {
                        UtrTimingCorridor::admit(floor_end_ms, ceiling_start_ms)
                            .intersect(*interval)
                    };
                    if let Some(projected) = projected {
                        bullets_to_set[utt_idx] = Some(projected);
                        if !overlaps {
                            document_floor_ms = document_floor_ms.max(projected.end_ms());
                            speaker_floors.insert(
                                info.speaker.as_str(),
                                speaker_floor_ms.max(projected.end_ms()),
                            );
                        }
                    } else {
                        result.unmatched += 1;
                        let (min_asr, max_asr) = matches.token_extent();
                        result.decisions.extend(decision(
                            utt_idx,
                            batchalign_transform::decisions::UtrStrategy::ProjectionExhausted,
                            format!(
                                "words={alignable_words} asr_range=[{},{}] \
                                 start_ms={} end_ms={} floor_end_ms={floor_end_ms} \
                                 ceiling_start_ms={:?} \
                                 reason=monotonic_projection_exhausted",
                                min_asr.index(),
                                max_asr.index(),
                                interval.start_ms(),
                                interval.end_ms(),
                                ceiling_start_ms,
                            ),
                            true,
                        ));
                    }
                }
                UtrUtteranceAlignmentEvidence::Matched {
                    proposal: UtrTimingProposal::NonPositive { start_ms, end_ms },
                    admitted_matches: matches,
                    alignable_words,
                    ..
                } => {
                    // A zero- or negative-duration span, produced by Whisper for
                    // very short words (single 20ms frame backchannels like "mhm",
                    // "yeah"). Creating a •T_T• utterance bullet would be actively
                    // harmful: the FA postprocess bounds word timings to the
                    // utterance range, clamping every word timing to the empty
                    // [T,T] interval and dropping them, so the zero-duration
                    // bullet then perpetuates across every subsequent `align`
                    // re-run. Leave the utterance untimed; FA will assign a valid
                    // bullet from the word-level forced alignment instead.
                    result.unmatched += 1;
                    let (min_asr, max_asr) = matches.token_extent();
                    result.decisions.extend(decision(
                        utt_idx,
                        batchalign_transform::decisions::UtrStrategy::ZeroDurationSkipped,
                        format!(
                            "words={alignable_words} asr_range=[{},{}] \
                             start_ms={start_ms} end_ms={end_ms} \
                             reason=zero_or_negative_duration",
                            min_asr.index(),
                            max_asr.index()
                        ),
                        false,
                    ));
                }
                UtrUtteranceAlignmentEvidence::InteriorOnly {
                    alignable_words,
                    missing_endpoints,
                    ..
                } => {
                    result.unmatched += 1;
                    // A bounded search window, when the proof yields one, is
                    // forced alignment's to use; it is never a hint.
                    let search = match plan.search_envelopes.get(utt_idx) {
                        Some(Some(_)) => "bounded_search_window",
                        Some(None) | None => "no_search_window",
                    };
                    result.decisions.extend(decision(
                        utt_idx,
                        batchalign_transform::decisions::UtrStrategy::IncompleteBoundary,
                        format!(
                            "words={alignable_words} missing_endpoints={missing_endpoints:?} \
                             interior_anchors_retained {search}"
                        ),
                        true,
                    ));
                }
                UtrUtteranceAlignmentEvidence::SelectedOnly {
                    alignable_words,
                    reason,
                    ..
                }
                | UtrUtteranceAlignmentEvidence::Refused {
                    alignable_words,
                    reason,
                    ..
                } => {
                    result.unmatched += 1;
                    let (strategy, reason) = match reason {
                        UtrCorrespondenceRefusal::Ambiguous => (
                            batchalign_transform::decisions::UtrStrategy::AmbiguousCorrespondence,
                            "ambiguous".to_owned(),
                        ),
                        UtrCorrespondenceRefusal::BudgetExhausted(refusal) => (
                            batchalign_transform::decisions::UtrStrategy::CorrespondenceBudgetExhausted,
                            format!(
                                "budget_exhausted budget={:?} region=[{}]",
                                refusal.budget(),
                                refusal.region().describe()
                            ),
                        ),
                    };
                    result.decisions.extend(decision(
                        utt_idx,
                        strategy,
                        format!(
                            "words={alignable_words} reason={reason} no_admitted_correspondence"
                        ),
                        true,
                    ));
                }
                UtrUtteranceAlignmentEvidence::RetainedUnsearched { .. } => {
                    // Only built for an utterance with retained timing, which
                    // the retained-timing branch above has already counted
                    // and kept. Reaching here means the census and the plan
                    // disagree about that timing; the utterance stays as it
                    // is and is counted unmatched rather than guessed at.
                    result.unmatched += 1;
                    tracing::error!(
                        utterance = utt_idx,
                        "UTR plan marks an untimed utterance as retained; left untimed"
                    );
                }
                UtrUtteranceAlignmentEvidence::Unmatched {
                    alignable_words, ..
                }
                | UtrUtteranceAlignmentEvidence::ExcludedMarkedOverlap {
                    alignable_words, ..
                } => {
                    // `+<` utterances skipped in pass 1 count as unmatched here;
                    // the two-pass caller handles them in pass 2.
                    result.unmatched += 1;
                    result.decisions.extend(decision(
                        utt_idx,
                        batchalign_transform::decisions::UtrStrategy::Unmatched,
                        format!("words={alignable_words} no_asr_match"),
                        true,
                    ));
                }
                UtrUtteranceAlignmentEvidence::NoAlignableWords { .. } => {
                    result.unmatched += 1;
                    result.decisions.extend(decision(
                        utt_idx,
                        batchalign_transform::decisions::UtrStrategy::Unmatched,
                        "words=0 no_asr_match".to_string(),
                        true,
                    ));
                }
            }
        }
        result.alignment = UtrAlignmentEvidence::Global { plan };

        // Apply bullets to the actual ChatFile utterances
        let mut utt_idx = 0;
        for line in &mut chat_file.lines {
            if let Line::Utterance(utt) = line {
                if let Some(interval) = bullets_to_set[utt_idx] {
                    // Mark as a provisional UTR hint so that update_utterance_bullet
                    // (called after FA injection) overwrites this bullet with the
                    // FA word span instead of union-expanding from it.
                    utt.main.content.bullet = Some(interval.into_hint());
                    result.injected += 1;
                }
                utt_idx += 1;
            }
        }

        result
    }
}

/// Extract alignable words, bullet presence, `+<` linker status, and CA
/// overlap marker info for every utterance in the order UTR sees them.
impl UtrUtteranceInfo {
    /// Whether a global pass with this participation leaves the utterance
    /// out of its flattened word sequence: the one owner of that rule, used
    /// both to build the payload and to label the evidence.
    fn excluded_from(&self, participation: GlobalUtrParticipation) -> bool {
        participation == GlobalUtrParticipation::ExcludeMarkedOverlap && self.is_marked_overlap()
    }

    /// Whether this is a marked overlap recovery may handle on its own: speech
    /// with a `+<` linker or a bottom CA overlap. An utterance not in the
    /// recording never is, whatever it is marked with.
    pub(super) fn is_marked_overlap(&self) -> bool {
        match self.presence {
            RecordingPresence::NotInRecording(_) => false,
            RecordingPresence::InRecording => self.has_lazy_overlap || self.has_ca_overlap,
        }
    }
}

pub(super) fn collect_utr_utterance_info(chat_file: &ChatFile) -> Vec<UtrUtteranceInfo> {
    let mut utt_infos = Vec::new();
    for line in &chat_file.lines {
        if let Line::Utterance(utt) = line {
            // An utterance not in the recording keeps its place in the census
            // (ordinals are utterance indices) but offers recovery no word to
            // match and no anchor: its words are not in the audio, and a
            // bullet on it locates nothing there.
            let presence = RecordingPresence::of(utt);
            let mut words = Vec::new();
            let retained_timing = match presence {
                RecordingPresence::NotInRecording(_) => None,
                RecordingPresence::InRecording => {
                    collect_fa_words(&utt.main.content.content, &mut words);
                    utt.main
                        .content
                        .bullet
                        .as_ref()
                        .map(|bullet| lexical::ObservedWordTiming {
                            start_ms: bullet.timing.start_ms,
                            end_ms: bullet.timing.end_ms,
                        })
                }
            };
            let has_lazy_overlap = utt
                .main
                .content
                .linkers
                .iter()
                .any(|l| l.kind == talkbank_model::model::LinkerKind::LazyOverlapPrecedes);
            let overlap_info = overlap_markers::extract_overlap_info(&utt.main.content.content);

            // Collect bottom region indices for index-aware matching.
            let bottom_indices: Vec<_> = overlap_info
                .regions
                .iter()
                .filter(|r| {
                    r.kind == talkbank_model::alignment::helpers::OverlapRegionKind::Bottom
                        && r.has_begin()
                })
                .map(|r| r.index)
                .collect();

            // Collect per-top-region onset fractions with their indices.
            let top_onsets: Vec<_> = overlap_info
                .regions
                .iter()
                .filter(|r| {
                    r.kind == talkbank_model::alignment::helpers::OverlapRegionKind::Top
                        && r.has_begin()
                })
                .filter_map(|r| {
                    let word_pos = r.begin_at_word?;
                    if overlap_info.total_words == 0 {
                        return None;
                    }
                    let fraction = word_pos as f64 / overlap_info.total_words as f64;
                    Some((r.index, fraction))
                })
                .collect();

            utt_infos.push(UtrUtteranceInfo {
                words,
                retained_timing,
                presence,
                has_lazy_overlap,
                has_ca_overlap: overlap_info.has_bottom_overlap(),
                overlap_onset_fraction: overlap_info.top_onset_fraction(),
                speaker: utt.main.speaker.to_string(),
                bottom_indices,
                top_onsets,
            });
        }
    }
    utt_infos
}

/// Plan the per-utterance ASR token ranges for one UTR pass.
///
/// This first tries the cheap exact-subsequence fast path. If every transcript
/// word appears in ASR order *and* that embedding is unique, UTR can avoid DP.
/// Any missing word or repeated-token ambiguity falls back to the global
/// Hirschberg alignment.
pub(super) fn plan_global_utr_alignment(
    chat_file: &ChatFile,
    asr_tokens: &[AsrTimingToken],
    dp_match_mode: MatchMode,
    participation: GlobalUtrParticipation,
) -> UtrAlignmentPlan {
    let utt_infos = collect_utr_utterance_info(chat_file);
    plan_global_utr_alignment_for(&utt_infos, asr_tokens, dp_match_mode, participation)
}

/// Plan against utterance information already collected, so a caller that
/// also projects the plan uses ONE census for both and the plan's utterance
/// population is the projection's by construction.
///
/// The census is partitioned into anchored regions and each region is
/// planned on its own, within its own budget (see `regions`).
fn plan_global_utr_alignment_for(
    utt_infos: &[UtrUtteranceInfo],
    asr_tokens: &[AsrTimingToken],
    dp_match_mode: MatchMode,
    participation: GlobalUtrParticipation,
) -> UtrAlignmentPlan {
    let order = UtrOrderModel::of(utt_infos, participation);
    let regions = regions::partition(utt_infos);
    let plan = |region: &regions::UtrRegion, contested: &ContestedTokens| {
        let lexical = lexical::UtrLexicalStream::within_region(asr_tokens, region);
        plan_region(
            order,
            region.searched(utt_infos),
            &lexical,
            dp_match_mode,
            participation,
            contested,
        )
    };
    let uncontested = ContestedTokens::none();
    let first: Vec<_> = regions
        .iter()
        .map(|region| plan(region, &uncontested))
        .collect();
    // Neighbouring regions share the anchor between them: tokens inside its
    // span are in both windows. A token claimed by owned words of both is
    // one the regional search cannot assign, so the regions that claimed it
    // are planned again with that claim withdrawn. Rare, and only those
    // regions pay for it.
    let contested = ContestedTokens::between(&first);
    if contested.is_empty() {
        return UtrAlignmentPlan::assemble(order, first);
    }
    UtrAlignmentPlan::assemble(
        order,
        regions.iter().zip(first).map(|(region, local)| {
            if contested.touches(&local) {
                plan(region, &contested)
            } else {
                local
            }
        }),
    )
}

/// Provider tokens whose correspondence the regional decomposition cannot
/// assign: admitted by owned words of two different regions.
///
/// Regions partition the transcript, but neighbouring regions' ASR windows
/// overlap on the anchor between them. In an interleaved file an untimed turn
/// before the anchor (in one region) and one after it (in the next) may each
/// claim the same token inside the anchor's span, and each region's claim is
/// common to every optimum of ITS problem only. Which claim the whole file
/// would admit, if either, the decomposition cannot say, so neither is
/// admitted: both utterances keep their other evidence, and each region
/// records the claims it withdrew. Within a region a token is claimed at most
/// once already.
struct ContestedTokens(std::collections::BTreeSet<UtrAsrTokenOrdinal>);

impl ContestedTokens {
    /// No token is contested: the first planning pass.
    fn none() -> Self {
        Self(std::collections::BTreeSet::new())
    }

    /// The tokens two or more of `plans` claim with their owned words.
    fn between(plans: &[evidence::LocalRegionPlan<'_>]) -> Self {
        let mut claimed_by = std::collections::BTreeMap::<UtrAsrTokenOrdinal, usize>::new();
        let mut contested = std::collections::BTreeSet::new();
        for (region, plan) in plans.iter().enumerate() {
            for token in plan.owned_claims() {
                match claimed_by.insert(token, region) {
                    Some(other) if other != region => {
                        contested.insert(token);
                    }
                    Some(_) | None => {}
                }
            }
        }
        Self(contested)
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn contains(&self, token: UtrAsrTokenOrdinal) -> bool {
        self.0.contains(&token)
    }

    /// Whether `plan`'s owned words claim any contested token.
    fn touches(&self, plan: &evidence::LocalRegionPlan<'_>) -> bool {
        plan.owned_claims().any(|token| self.contains(token))
    }
}

/// Plan one region's searched census (owned utterances and trailing context)
/// against the region's tokens, in the region's local numbering.
///
/// The order model is the file's: an interleaved file's regions all use the
/// joint composition (on a single-speaker region it admits exactly the
/// monotonic orders), so regions bound the work without changing the model.
/// A region with no tokens matches nothing under either model.
fn plan_region<'c>(
    order: UtrOrderModel,
    searched: regions::SearchedCensus<'c>,
    lexical: &lexical::UtrLexicalStream<'_>,
    dp_match_mode: MatchMode,
    participation: GlobalUtrParticipation,
    contested: &ContestedTokens,
) -> evidence::LocalRegionPlan<'c> {
    match order {
        UtrOrderModel::Interleaved if !lexical.is_empty() => {
            return lexical.plan_interleaving(searched, dp_match_mode, participation, contested);
        }
        UtrOrderModel::Interleaved | UtrOrderModel::Monotonic => {}
    }
    let mut payload = Vec::new();
    for (utterance_index, info) in searched.infos().iter().enumerate() {
        if info.excluded_from(participation) {
            continue;
        }
        for (word_index, word) in info.words.iter().enumerate() {
            payload.push(UtrPayloadWord {
                text: word.clone(),
                address: UtrWordAddress {
                    utterance_index: UtrUtteranceOrdinal(utterance_index),
                    word_index: UtrWordOrdinal(word_index),
                },
            });
        }
    }
    plan_utr_alignment(
        &payload,
        lexical,
        searched,
        dp_match_mode,
        participation,
        contested,
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UtrPayloadWord {
    text: String,
    address: UtrWordAddress,
}

fn plan_utr_alignment<'c>(
    payload: &[UtrPayloadWord],
    lexical: &lexical::UtrLexicalStream<'_>,
    searched: regions::SearchedCensus<'c>,
    dp_match_mode: MatchMode,
    participation: GlobalUtrParticipation,
    contested: &ContestedTokens,
) -> evidence::LocalRegionPlan<'c> {
    let region = searched.span();
    let all_words = payload
        .iter()
        .map(|word| word.text.clone())
        .collect::<Vec<_>>();
    let asr_texts = lexical.texts();
    let embeddings = matches!(dp_match_mode, MatchMode::CaseInsensitive)
        .then(|| search::CompleteExactEmbeddings::observe(&all_words, &asr_texts))
        .flatten();
    // The monotonic producers observe common correspondences only, never
    // every optimum's candidates, so they bound no ambiguous endpoint.
    let per_utterance = match embeddings.as_ref() {
        Some(complete) => complete
            .envelopes(payload, searched, lexical)
            .map_values(|envelope| UtteranceSelection {
                envelope,
                endpoints: lexical::EndpointExtents::UNOBSERVED,
            }),
        None => searched.map(|_| UtteranceSelection::UNSEARCHED),
    };

    // This fast path deliberately uses case-folded comparison, so it is only
    // valid for the case-insensitive DP policy. An exact caller must reach the
    // exact DP relation instead of being silently weakened here.
    if let Some(complete) = embeddings
        && complete.earliest == complete.latest
    {
        let reference_indices = UniqueEmbedding {
            indices: complete.earliest,
        };
        let admitted = lexical.unique_embedding_matches(payload, &reference_indices);
        return build_alignment_plan(
            UtrAlignmentStrategy::UniqueExactSubsequence,
            payload,
            lexical,
            UtrMatchSelection {
                selected: reference_indices.indices.into_iter().enumerate().collect(),
                admission: CorrespondenceProof::Complete(admitted),
                per_utterance,
            },
            participation,
            contested,
        );
    }

    let alignment =
        dp_align::CorrespondenceAnalysis::observe(&all_words, &asr_texts, dp_match_mode);
    let admission = match alignment.admission() {
        dp_align::CorrespondenceAdmission::Complete(common) => {
            CorrespondenceProof::Complete(lexical.common_matches(common, payload))
        }
        dp_align::CorrespondenceAdmission::BudgetExhausted(budget) => {
            let refusal = UtrBudgetRefusal {
                region,
                budget: UtrSearchBudget::monotonic(*budget),
            };
            tracing::warn!(
                region = %region.describe(),
                budget = ?refusal.budget,
                "UTR correspondence proof refused for this region; other regions are unaffected"
            );
            CorrespondenceProof::BudgetExhausted(refusal)
        }
    };
    let matched_indices = alignment
        .selected()
        .iter()
        .filter_map(|item| match item {
            dp_align::AlignResult::Match {
                payload_idx,
                reference_idx,
                ..
            } => Some((*payload_idx, *reference_idx)),
            dp_align::AlignResult::ExtraPayload { .. }
            | dp_align::AlignResult::ExtraReference { .. } => None,
        })
        .collect();
    let mut plan = build_alignment_plan(
        UtrAlignmentStrategy::GlobalDp,
        payload,
        lexical,
        UtrMatchSelection {
            selected: matched_indices,
            admission,
            per_utterance,
        },
        participation,
        contested,
    );
    interior::bound_interior_only_searches(&mut plan, lexical);
    plan
}

/// Attempt the exact-subsequence fast path for UTR.
///
/// The fast path is accepted only when the transcript words have exactly one
/// monotonic embedding into the ASR stream. Repeated-token ambiguity therefore
/// forces a DP fallback instead of silently accepting an arbitrary greedy path.
struct UniqueEmbedding {
    indices: Vec<usize>,
}

/// Return the earliest monotonic exact-subsequence match indices for the
/// payload words.
fn greedy_forward_match_indices(payload: &[String], reference: &[String]) -> Option<Vec<usize>> {
    let mut reference_idx = 0;
    let mut matches = Vec::with_capacity(payload.len());

    for payload_word in payload {
        while reference_idx < reference.len()
            && !payload_word.eq_ignore_ascii_case(&reference[reference_idx])
        {
            reference_idx += 1;
        }
        if reference_idx == reference.len() {
            return None;
        }
        matches.push(reference_idx);
        reference_idx += 1;
    }

    Some(matches)
}

/// Return the latest monotonic exact-subsequence match indices for the payload
/// words.
fn greedy_reverse_match_indices(payload: &[String], reference: &[String]) -> Option<Vec<usize>> {
    let mut reference_idx = reference.len();
    let mut matches = vec![0; payload.len()];

    for (payload_idx, payload_word) in payload.iter().enumerate().rev() {
        let mut found = None;
        while reference_idx > 0 {
            reference_idx -= 1;
            if payload_word.eq_ignore_ascii_case(&reference[reference_idx]) {
                found = Some(reference_idx);
                break;
            }
        }
        matches[payload_idx] = found?;
    }

    Some(matches)
}

struct UtrMatchSelection<'c> {
    selected: Vec<(usize, usize)>,
    admission: CorrespondenceProof,
    /// What the producer knows about each searched utterance, carried with
    /// the census it describes.
    per_utterance: regions::PerSearched<'c, UtteranceSelection>,
}

/// A producer's per-utterance facts beyond its correspondences: where forced
/// alignment may search for the utterance, and the extents of its endpoint
/// words when the producer observed every optimum.
struct UtteranceSelection {
    envelope: Option<search::FaSearchEnvelope>,
    endpoints: lexical::EndpointExtents,
}

impl UtteranceSelection {
    /// Nothing searched: no envelope, no endpoint extents.
    const UNSEARCHED: Self = Self {
        envelope: None,
        endpoints: lexical::EndpointExtents::UNOBSERVED,
    };
}

enum CorrespondenceProof {
    Complete(Vec<AdmittedUtrWordMatch>),
    /// The region's proof work exceeded a budget; carries which region and
    /// which budget, so every refusal it causes can name both.
    BudgetExhausted(UtrBudgetRefusal),
}

fn build_alignment_plan<'c>(
    strategy: UtrAlignmentStrategy,
    payload: &[UtrPayloadWord],
    asr_tokens: &lexical::UtrLexicalStream<'_>,
    selection: UtrMatchSelection<'c>,
    participation: GlobalUtrParticipation,
    contested: &ContestedTokens,
) -> evidence::LocalRegionPlan<'c> {
    let UtrMatchSelection {
        selected,
        admission,
        per_utterance,
    } = selection;
    let searched = per_utterance.searched().infos().len();
    let (admitted, refusal) = match admission {
        CorrespondenceProof::Complete(admitted) => (admitted, UtrCorrespondenceRefusal::Ambiguous),
        CorrespondenceProof::BudgetExhausted(refusal) => (
            Vec::new(),
            UtrCorrespondenceRefusal::BudgetExhausted(refusal),
        ),
    };
    let mut by_utterance = vec![Vec::new(); searched];
    let mut admitted_by_utterance = vec![Vec::new(); searched];
    let mut withdrawn_claims = Vec::new();
    for matched in admitted {
        let token = matched.matched().token.token_index;
        if contested.contains(token) {
            withdrawn_claims.push(token);
            continue;
        }
        admitted_by_utterance[matched.matched().word.utterance_index()].push(matched);
    }
    for (payload_idx, reference_idx) in selected {
        let payload_word = &payload[payload_idx];
        by_utterance[payload_word.address.utterance_index.index()].push(asr_tokens.matched_word(
            reference_idx,
            payload_word.address,
            &payload_word.text,
        ));
    }

    evidence::LocalRegionPlan::from_searched(
        strategy,
        withdrawn_claims,
        per_utterance,
        |utterance_index,
         info,
         UtteranceSelection {
             envelope,
             endpoints,
         }| {
            let evidence = utterance_evidence(
                UtrUtteranceOrdinal(utterance_index),
                info,
                std::mem::take(&mut by_utterance[utterance_index]),
                std::mem::take(&mut admitted_by_utterance[utterance_index]),
                refusal,
                endpoints,
                participation,
            );
            (evidence, envelope)
        },
    )
}

/// One searched utterance's correspondence evidence, from its selected and
/// admitted matches, the region's refusal (if any) and its endpoint extents.
fn utterance_evidence(
    utterance_index: UtrUtteranceOrdinal,
    info: &UtrUtteranceInfo,
    matches: Vec<UtrWordMatch>,
    admitted: Vec<AdmittedUtrWordMatch>,
    refusal: UtrCorrespondenceRefusal,
    endpoints: lexical::EndpointExtents,
    participation: GlobalUtrParticipation,
) -> UtrUtteranceAlignmentEvidence {
    let alignable_words = info.words.len();
    if info.excluded_from(participation) && matches.is_empty() {
        return UtrUtteranceAlignmentEvidence::ExcludedMarkedOverlap {
            utterance_index,
            alignable_words,
        };
    }
    if info.words.is_empty() {
        return UtrUtteranceAlignmentEvidence::NoAlignableWords { utterance_index };
    }
    // A budget refusal refuses correspondence, never timing: an utterance
    // that already has its bullet keeps it and says so.
    if let (UtrCorrespondenceRefusal::BudgetExhausted(unsearched), Some(_)) =
        (refusal, info.retained_timing)
    {
        return UtrUtteranceAlignmentEvidence::RetainedUnsearched {
            utterance_index,
            alignable_words,
            unsearched,
        };
    }
    let Some(matches) = NonEmptyUtrWordMatches::from_vec(matches) else {
        return match refusal {
            UtrCorrespondenceRefusal::BudgetExhausted(_) => {
                UtrUtteranceAlignmentEvidence::Refused {
                    utterance_index,
                    alignable_words,
                    reason: refusal,
                }
            }
            UtrCorrespondenceRefusal::Ambiguous => UtrUtteranceAlignmentEvidence::Unmatched {
                utterance_index,
                alignable_words,
            },
        };
    };
    let Some(admitted_matches) = NonEmptyAdmittedUtrWordMatches::from_vec(admitted) else {
        return UtrUtteranceAlignmentEvidence::SelectedOnly {
            utterance_index,
            alignable_words,
            matches,
            reason: refusal,
        };
    };
    match admitted_matches.admit_endpoints(alignable_words, endpoints) {
        lexical::BoundaryAdmission::Bound(admitted_matches) => {
            let proposal = admitted_matches.proposal();
            UtrUtteranceAlignmentEvidence::Matched {
                utterance_index,
                alignable_words,
                matches,
                admitted_matches,
                proposal,
            }
        }
        lexical::BoundaryAdmission::Interior {
            matches: admitted_matches,
            missing,
        } => UtrUtteranceAlignmentEvidence::InteriorOnly {
            utterance_index,
            alignable_words,
            matches,
            admitted_matches,
            missing_endpoints: missing,
        },
    }
}

fn lexical_relation(chat_text: &str, asr_text: &str) -> UtrLexicalRelation {
    if chat_text == asr_text {
        UtrLexicalRelation::Exact
    } else if chat_text.eq_ignore_ascii_case(asr_text) {
        UtrLexicalRelation::CaseInsensitive
    } else {
        let similarity = strsim::jaro_winkler(&chat_text.to_lowercase(), &asr_text.to_lowercase());
        UtrLexicalRelation::Fuzzy {
            similarity_millionths: (similarity * 1_000_000.0).round() as u32,
        }
    }
}

/// Cache key for a full-file UTR ASR result.
///
/// The versioned identity includes the selected ASR provider. Legacy keys
/// lacked provider provenance and are deliberately not replayed.
pub fn utr_asr_cache_key(
    audio_identity: &super::AudioIdentity,
    engine: &crate::options::UtrEngine,
    lang: &str,
) -> crate::chat_ops::CacheKey {
    let input = format!(
        "utr_asr_v2|{}|{}|{lang}",
        engine.as_wire_name(),
        audio_identity.as_str()
    );
    crate::chat_ops::CacheKey::from_content(&input)
}

/// Cache key for a segment-level UTR ASR result (partial-window mode).
///
/// Includes the selected ASR provider and window. Legacy provider-agnostic
/// keys cannot safely establish which backend produced their response.
pub fn utr_asr_segment_cache_key(
    audio_identity: &super::AudioIdentity,
    engine: &crate::options::UtrEngine,
    start_ms: u64,
    end_ms: u64,
    lang: &str,
) -> crate::chat_ops::CacheKey {
    let input = format!(
        "utr_asr_segment_v2|{}|{}|{start_ms}|{end_ms}|{lang}",
        engine.as_wire_name(),
        audio_identity.as_str()
    );
    crate::chat_ops::CacheKey::from_content(&input)
}

/// Where one edge of an untimed run's window came from.
///
/// # Why this is a type and not an `unwrap_or`
///
/// Both edges used to be `find_map(...).unwrap_or(fallback)`, which is correct
/// arithmetic and silent about which case occurred. The two are different
/// facts: a boundary taken from a timed neighbour is derived from a MEASURED
/// bullet, while the fallback is a boundary of the FILE ITSELF, chosen because
/// there is no neighbour on that side. Rewriting the `unwrap_or` as a bare
/// `match` would only spell the same collapse differently, so the distinction
/// is carried in the value instead: a caller reading `FileEdge` can see that
/// nothing measured it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunBoundary {
    /// The nearest timed utterance's bullet edge on that side.
    TimedNeighbour(u64),
    /// No timed utterance on that side, so the window runs to the edge of the
    /// audio. Not a measurement of speech, and not an invented one either: it
    /// is the extent of the file.
    FileEdge(u64),
}

/// Which way a window edge is pushed to leave room around the speech.
#[derive(Debug, Clone, Copy)]
enum Padding {
    /// Move the edge earlier, never before the start of the file.
    Earlier(u64),
    /// Move the edge later, never past the end of the recording.
    Later(u64),
}

impl RunBoundary {
    /// Classify a neighbour search's answer, naming the fallback's meaning.
    fn from_neighbour(found: Option<u64>, file_edge: u64) -> Self {
        match found {
            Some(ms) => Self::TimedNeighbour(ms),
            None => Self::FileEdge(file_edge),
        }
    }

    /// This edge, moved outward to leave room, kept inside the recording.
    ///
    /// The padding and the bound live here rather than at the call site because
    /// BOTH edges must be clamped and for a while only the end was, which is
    /// what let a run whose bullet starts past the end of the audio produce an
    /// inverted window. One owner means the two edges cannot disagree again.
    ///
    /// Asks the [`Recording`] rather than writing `.min(total_audio_ms)`: the
    /// recording owns its own bound, and a bare `.min()` cannot say whether it
    /// did anything.
    fn padded(self, by: Padding, recording: &Recording) -> u64 {
        let from = match self {
            Self::TimedNeighbour(ms) | Self::FileEdge(ms) => ms,
        };
        let moved = match by {
            Padding::Earlier(ms) => from.saturating_sub(ms),
            Padding::Later(ms) => from.saturating_add(ms),
        };
        match recording.overshoot_of(FileMs::new(moved)) {
            None => moved,
            // Past the end of the audio, so the edge becomes the end of the
            // audio. Nothing is measured here; this is the file's own extent.
            Some(_) => recording.duration().get(),
        }
    }
}

/// Identify audio windows covering untimed utterances.
///
/// Each window spans from the preceding timed utterance's end to the
/// following timed utterance's start (with `padding_ms` on each side).
/// Adjacent untimed utterances are merged into a single window.
/// Windows of audio with no timing, for UTR to transcribe.
///
/// Returns [`FaWindow`]s, each proven inside `recording`, so the proof reaches
/// the audio extraction and its transcode rather than being rebuilt (or
/// dropped) on the way. The caller used to convert each bare `MediaWindow`
/// back with `FaWindow::over`, a second fallible step with a `continue` arm for
/// a failure that could not occur.
///
/// This used to hand back bare `(u64, u64)` pairs, and
/// it is the one place that can build a degenerate one: `padded_end` is clamped
/// to `total_audio_ms` while `padded_start` was not, so an utterance whose
/// bullet starts past the end of the audio produced an INVERTED window. Every
/// consumer then re-checked, three of them, in two subsystems, and the check
/// nearest the origin did not exist.
///
/// Degenerate windows are dropped here rather than passed on, because a window
/// of zero or negative length contains no audio to transcribe. The count is
/// logged: a silent drop and an empty result are the same thing to a reader,
/// which is the failure this crate keeps finding.
///
/// # Errors
///
/// [`WindowFault`] if a merged window is inverted or not inside `recording`.
/// Both edges are clamped to it and empties are skipped, so this is not
/// expected to occur; it is surfaced, not dropped, so that a change to the clamping breaks
/// loudly instead of silently shrinking the set of windows.
pub fn find_untimed_windows(
    chat_file: &ChatFile,
    recording: &Recording,
    padding_ms: u64,
) -> Result<Vec<FaWindow>, WindowFault> {
    // Collect bullet info for each utterance in order. One not in the
    // recording needs no timing and bounds no window: it is left out.
    let mut utt_bullets: Vec<Option<(u64, u64)>> = Vec::new();
    for line in &chat_file.lines {
        if let Line::Utterance(utt) = line {
            match RecordingPresence::of(utt) {
                RecordingPresence::InRecording => {}
                RecordingPresence::NotInRecording(_) => continue,
            }
            utt_bullets.push(
                utt.main
                    .content
                    .bullet
                    .as_ref()
                    .map(|b| (b.timing.start_ms, b.timing.end_ms)),
            );
        }
    }

    if utt_bullets.is_empty() {
        return Ok(Vec::new());
    }

    // Find contiguous runs of untimed utterances and compute their windows
    let mut raw_windows: Vec<(u64, u64)> = Vec::new();
    let mut i = 0;
    while i < utt_bullets.len() {
        if utt_bullets[i].is_some() {
            i += 1;
            continue;
        }

        // Start of an untimed run
        let run_start = i;
        while i < utt_bullets.len() && utt_bullets[i].is_none() {
            i += 1;
        }
        // run_start..i is the untimed run

        // The nearest timed neighbour on each side bounds the window. Both
        // searches range over a possibly-EMPTY span, which already answers
        // `None`, so the `if run_start > 0` / `if i < len` guards these used to
        // carry restated what the iterator already knew, and each then spelled
        // its answer twice: once in the `else` arm and once in an `unwrap_or`.
        let window_start = RunBoundary::from_neighbour(
            (0..run_start)
                .rev()
                .find_map(|j| utt_bullets[j].map(|(_, end)| end)),
            0,
        );
        let window_end = RunBoundary::from_neighbour(
            (i..utt_bullets.len()).find_map(|j| utt_bullets[j].map(|(start, _)| start)),
            recording.duration().get(),
        );

        // Padding and clamping both belong to the boundary now; see
        // `RunBoundary::padded` for why one owner matters here.
        let padded_start = window_start.padded(Padding::Earlier(padding_ms), recording);
        let padded_end = window_end.padded(Padding::Later(padding_ms), recording);

        raw_windows.push((padded_start, padded_end));
    }

    // Merge overlapping windows
    if raw_windows.is_empty() {
        return Ok(Vec::new());
    }
    raw_windows.sort_by_key(|&(start, _)| start);
    let mut merged: Vec<(u64, u64)> = vec![raw_windows[0]];
    for &(start, end) in &raw_windows[1..] {
        // SAFETY: `merged` is initialized with `vec![raw_windows[0]]`, so it is
        // always non-empty at this point.
        #[allow(clippy::unwrap_used)]
        let last = merged.last_mut().unwrap();
        if start <= last.1 {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }

    // The merged pairs become proven windows here, at the last moment they are
    // still pairs. A window holding nothing cannot be transcribed, and
    // `FaWindow::within` refuses it as `WindowFault::Empty`: it is dropped and
    // counted rather than passed on. Any other fault is returned.
    let total = merged.len();
    let mut windows: Vec<FaWindow> = Vec::with_capacity(total);
    for (start, end) in merged {
        match FaWindow::within(recording, FileMs::new(start), FileMs::new(end)) {
            Ok(window) => windows.push(window),
            Err(WindowFault::Empty { .. }) => {}
            Err(fault @ (WindowFault::Inverted { .. } | WindowFault::PastRecording { .. })) => {
                return Err(fault);
            }
        }
    }
    if windows.len() != total {
        debug!(
            dropped = total - windows.len(),
            kept = windows.len(),
            "Dropped degenerate untimed windows (zero-length after clamping)"
        );
    }
    Ok(windows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use talkbank_parser::TreeSitterParser;

    fn parse_chat(text: &str) -> ChatFile {
        let parser = TreeSitterParser::new().unwrap();
        parser.parse_chat_file(text).expect_built()
    }

    fn recording(duration_ms: u64) -> Recording {
        Recording::of_duration(super::super::coordinates::Ms(duration_ms))
            .expect("test recordings are non-empty")
    }

    fn test_utt_infos(word_counts: &[usize]) -> Vec<UtrUtteranceInfo> {
        word_counts
            .iter()
            .enumerate()
            .map(|(utterance_index, count)| UtrUtteranceInfo {
                words: (0..*count)
                    .map(|word_index| format!("u{utterance_index}-w{word_index}"))
                    .collect(),
                retained_timing: None,
                has_lazy_overlap: false,
                has_ca_overlap: false,
                overlap_onset_fraction: None,
                speaker: "PAR".to_string(),
                presence: RecordingPresence::InRecording,
                bottom_indices: Vec::new(),
                top_onsets: Vec::new(),
            })
            .collect()
    }

    /// Plan a payload over an untimed census, which is one region spanning
    /// the whole stream, through the same assembly production uses.
    fn plan_untimed_payload(
        payload: &[UtrPayloadWord],
        asr_tokens: &[AsrTimingToken],
        utt_infos: &[UtrUtteranceInfo],
        mode: MatchMode,
    ) -> UtrAlignmentPlan {
        let regions = regions::partition(utt_infos);
        assert_eq!(regions.len(), 1, "an untimed census is one region");
        let lexical = lexical::UtrLexicalStream::from_tokens(asr_tokens);
        let local = plan_utr_alignment(
            payload,
            &lexical,
            regions[0].searched(utt_infos),
            mode,
            GlobalUtrParticipation::AllUtterances,
            &ContestedTokens::none(),
        );
        UtrAlignmentPlan::assemble(UtrOrderModel::Monotonic, [local])
    }

    #[test]
    fn a_padded_window_never_leaves_the_recording() {
        // Both edges clamp, and the type owns it. Only the END used to be
        // clamped, so a bullet naming a moment past the end of the audio
        // produced a window whose start exceeded its end. `MediaWindow::new`
        // then refused it and the run silently lost a window.
        let rec = recording(10_000);
        let past_the_end = RunBoundary::FileEdge(12_000);
        assert_eq!(past_the_end.padded(Padding::Later(500), &rec), 10_000);
        assert_eq!(past_the_end.padded(Padding::Earlier(500), &rec), 10_000);

        // An edge inside the recording is padded normally, and an edge near
        // zero saturates rather than wrapping.
        let inside = RunBoundary::TimedNeighbour(4_000);
        assert_eq!(inside.padded(Padding::Earlier(500), &rec), 3_500);
        assert_eq!(inside.padded(Padding::Later(500), &rec), 4_500);
        assert_eq!(
            RunBoundary::FileEdge(0).padded(Padding::Earlier(500), &rec),
            0
        );
    }

    fn make_asr_tokens(words_with_times: &[(&str, u64, u64)]) -> Vec<AsrTimingToken> {
        words_with_times
            .iter()
            .map(|(text, start, end)| AsrTimingToken {
                text: text.to_string(),
                start_ms: *start,
                end_ms: *end,
            })
            .collect()
    }

    /// A prepared pass owns producer observations: later token changes cannot
    /// alter its proposal, and nonpositive evidence remains a typed refusal.
    #[test]
    fn projection_consumes_the_plans_proposal_not_the_raw_tokens() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n*CHI:\tmhm .\n@End\n";
        let parser = TreeSitterParser::new().expect("parser");
        let errors = talkbank_model::ErrorCollector::new();
        let mut chat = batchalign_transform::parse_source_with_parser(&parser, chat_text)
            .admit(talkbank_model::model::TranscriptName::Anonymous, &errors)
            .expect("complete source admission")
            .into_valid_file()
            .into_unchecked();
        let mut tokens = make_asr_tokens(&[
            ("hello", 100, 500),
            ("world", 500, 900),
            ("mhm", 1_000, 1_000),
        ]);
        let prepared = PreparedGlobalUtr::plan(
            &mut chat,
            &tokens,
            GlobalUtrParticipation::AllUtterances,
            MatchMode::CaseInsensitive,
        );
        // The prepared pass owns its observations, not a token-stream borrow.
        // Changing provider data now cannot change the selected proposal.
        tokens[0].start_ms = 42;
        tokens[1].end_ms = 9_000;
        tokens[2].end_ms = 2_000;
        let result = prepared.project();
        assert_eq!(
            (result.injected(), result.skipped(), result.unmatched()),
            (1, 0, 1)
        );
        assert_eq!(utterance_bullets(&chat), vec![Some((100, 900)), None]);
        assert!(result.decisions().iter().any(|decision| matches!(
            decision.strategy,
            batchalign_transform::decisions::DecisionStrategy::Utr(
                batchalign_transform::decisions::UtrStrategy::ZeroDurationSkipped
            )
        )));
    }

    #[test]
    fn timing_corridor_intersection_is_positive_bounded_and_never_extended() {
        for floor in [0, 100, 200, u64::MAX - 1, u64::MAX] {
            for ceiling in [None, Some(0), Some(100), Some(200), Some(u64::MAX)] {
                for (start, end) in [(0, 100), (100, 200), (u64::MAX - 1, u64::MAX)] {
                    let proposal = PositiveUtrInterval::admit(start, end).expect("positive");
                    let actual = UtrTimingCorridor::admit(floor, ceiling).intersect(proposal);
                    let expected_start = start.max(floor);
                    let expected_end = ceiling.map_or(end, |bound| end.min(bound));
                    if expected_start < expected_end {
                        let interval = actual.expect("nonempty intersection survives");
                        assert_eq!(
                            (interval.start_ms(), interval.end_ms()),
                            (expected_start, expected_end),
                        );
                    } else {
                        assert_eq!(actual, None, "exhausted or contradictory corridor");
                    }
                }
            }
        }
    }

    /// Every utterance's terminal bullet, in document order.
    fn utterance_bullets(chat: &ChatFile) -> Vec<Option<(u64, u64)>> {
        chat.lines
            .iter()
            .filter_map(|line| match line {
                Line::Utterance(utt) => Some(
                    utt.main
                        .content
                        .bullet
                        .as_ref()
                        .map(|bullet| (bullet.timing.start_ms, bullet.timing.end_ms)),
                ),
                _ => None,
            })
            .collect()
    }

    fn test_word_address(utterance_index: usize, word_index: usize) -> UtrWordAddress {
        UtrWordAddress {
            utterance_index: UtrUtteranceOrdinal(utterance_index),
            word_index: UtrWordOrdinal(word_index),
        }
    }

    fn test_token_address(token_index: usize) -> UtrAsrTokenAddress {
        UtrAsrTokenAddress {
            token_index: UtrAsrTokenOrdinal(token_index),
            word_index: evidence::UtrAsrWordOrdinal(0),
        }
    }

    /// Captured old-batchalign output for the trimmed 407 regression fixture.
    ///
    /// The fixture stores one row per utterance from the legacy reference
    /// output so the regression test can verify both coverage and timing
    /// neighborhood, not only the presence of a bullet.
    #[derive(Debug, serde::Deserialize)]
    struct ExpectedUtrFixture {
        /// Speaker code from the captured reference output.
        speaker: String,
        /// Expected utterance start time in milliseconds, when one existed.
        start_ms: Option<u64>,
        /// Expected utterance end time in milliseconds, when one existed.
        end_ms: Option<u64>,
        /// Main-tier text from the captured reference output.
        text: String,
    }

    /// Return `true` when two utterance spans land in the same timing
    /// neighborhood.
    ///
    /// Exact equality is too brittle for ASR-derived regression fixtures, but
    /// a restored global-DP UTR pass should still land on substantially the same
    /// interval. The spans must therefore either overlap or both endpoints must
    /// be within a small tolerance.
    fn spans_roughly_agree(actual: (u64, u64), expected: (u64, u64)) -> bool {
        const ENDPOINT_TOLERANCE_MS: u64 = 1_500;

        let overlaps = actual.0 <= expected.1 && expected.0 <= actual.1;
        let start_close = actual.0.abs_diff(expected.0) <= ENDPOINT_TOLERANCE_MS;
        let end_close = actual.1.abs_diff(expected.1) <= ENDPOINT_TOLERANCE_MS;

        overlaps || (start_close && end_close)
    }

    #[test]
    fn test_inject_utr_all_timed_is_noop() {
        let input = include_str!("../../../../../test-fixtures/fa_two_timed_utterances.cha");
        let mut chat = parse_chat(input);
        let tokens = make_asr_tokens(&[("hello", 0, 500), ("world", 600, 1000)]);
        let result = inject_utr_timing(&mut chat, &tokens);
        assert_eq!(result.skipped, 2);
        assert_eq!(result.injected, 0);
        assert_eq!(result.unmatched, 0);
    }

    #[test]
    fn test_inject_utr_empty_tokens() {
        let input = include_str!("../../../../../test-fixtures/fa_two_untimed_with_media.cha");
        let mut chat = parse_chat(input);
        let result = inject_utr_timing(&mut chat, &[]);
        assert_eq!(result.unmatched, 2);
        assert_eq!(result.injected, 0);
    }

    #[test]
    fn test_inject_utr_untimed_gets_timing() {
        // Use a file with mixed timed/untimed utterances
        let input =
            include_str!("../../../../../test-fixtures/fa_mixed_timed_untimed_interleaved.cha");
        let mut chat = parse_chat(input);

        // Count before
        let (timed_before, untimed_before) = super::super::grouping::count_utterance_timing(&chat);
        assert!(untimed_before > 0, "test fixture should have untimed utts");

        // Build ASR tokens matching the fixture's actual words:
        // utt 0 (timed): "the cat is here"
        // utt 1 (untimed): "she is looking outside"
        // utt 2 (timed): "there is a path"
        // utt 3 (untimed): "I do not know"
        // utt 4 (untimed): "but there is a building"
        // utt 5 (timed): "okay so now"
        let tokens = make_asr_tokens(&[
            // utt 0 (timed): cursor advance
            ("the", 10000, 10500),
            ("cat", 10600, 11000),
            ("is", 11200, 11500),
            ("here", 12000, 13000),
            // utt 1 (untimed): "she is looking outside"
            ("she", 15500, 16000),
            ("is", 16200, 16500),
            ("looking", 16800, 17500),
            ("outside", 17800, 18500),
            // utt 2 (timed): cursor advance
            ("there", 20500, 21000),
            ("is", 21200, 21500),
            ("a", 21800, 22000),
            ("path", 22200, 23000),
            // utt 3 (untimed): "I do not know"
            ("I", 26000, 26500),
            ("do", 26800, 27000),
            ("not", 27200, 27500),
            ("know", 27800, 28500),
            // utt 4 (untimed): "but there is a building"
            ("but", 30000, 30500),
            ("there", 30800, 31200),
            ("is", 31500, 31800),
            ("a", 32000, 32200),
            ("building", 32500, 33500),
            // utt 5 (timed): cursor advance
            ("okay", 40500, 41000),
            ("so", 41200, 41500),
            ("now", 41800, 42500),
        ]);

        let result = inject_utr_timing(&mut chat, &tokens);
        assert_eq!(result.skipped, 3, "3 already-timed utterances");
        assert_eq!(result.injected, 3, "3 untimed utterances should get timing");
        assert_eq!(result.unmatched, 0);

        // Verify all utterances now have bullets
        let (timed_after, untimed_after) = super::super::grouping::count_utterance_timing(&chat);
        assert_eq!(untimed_after, 0, "all utterances should now be timed");
        assert_eq!(timed_after, timed_before + untimed_before);
    }

    #[test]
    fn test_plan_utr_alignment_uses_unique_exact_subsequence_fast_path() {
        let payload = vec![
            UtrPayloadWord {
                text: "the".to_string(),
                address: test_word_address(0, 0),
            },
            UtrPayloadWord {
                text: "cat".to_string(),
                address: test_word_address(0, 1),
            },
            UtrPayloadWord {
                text: "sat".to_string(),
                address: test_word_address(1, 0),
            },
            UtrPayloadWord {
                text: "down".to_string(),
                address: test_word_address(1, 1),
            },
        ];
        let asr_tokens = make_asr_tokens(&[
            ("noise", 0, 10),
            ("the", 10, 20),
            ("cat", 20, 30),
            ("sat", 30, 40),
            ("down", 40, 50),
            ("tail", 50, 60),
        ]);
        let utt_infos = test_utt_infos(&[2, 2]);

        let plan = plan_untimed_payload(
            &payload,
            &asr_tokens,
            &utt_infos,
            MatchMode::CaseInsensitive,
        );

        assert_eq!(
            plan.regions[0].strategy,
            UtrAlignmentStrategy::UniqueExactSubsequence
        );
        assert_eq!(plan.token_extents(), vec![Some((1, 2)), Some((3, 4))]);
        assert_eq!(
            plan.utterances[0],
            UtrUtteranceAlignmentEvidence::Matched {
                utterance_index: UtrUtteranceOrdinal(0),
                alignable_words: 2,
                matches: NonEmptyUtrWordMatches {
                    first: UtrWordMatch {
                        word: payload[0].address,
                        token: test_token_address(1),
                        chat_text: "the".to_string(),
                        asr_text: "the".to_string(),
                        relation: UtrLexicalRelation::Exact,
                    },
                    rest: vec![UtrWordMatch {
                        word: payload[1].address,
                        token: test_token_address(2),
                        chat_text: "cat".to_string(),
                        asr_text: "cat".to_string(),
                        relation: UtrLexicalRelation::Exact,
                    }],
                },
                admitted_matches: match NonEmptyAdmittedUtrWordMatches::from_vec(
                    lexical::UtrLexicalStream::from_tokens(&asr_tokens).unique_embedding_matches(
                        &payload[..2],
                        &UniqueEmbedding {
                            indices: vec![1, 2]
                        },
                    ),
                )
                .expect("nonempty producer proof")
                .admit_endpoints(2, lexical::EndpointExtents::UNOBSERVED)
                {
                    lexical::BoundaryAdmission::Bound(matches) => matches,
                    lexical::BoundaryAdmission::Interior { .. } => panic!("both endpoints proved"),
                },
                proposal: UtrTimingProposal::Positive {
                    interval: PositiveUtrInterval::admit(10, 30).expect("positive fixture"),
                },
            }
        );
    }

    #[test]
    fn test_plan_utr_alignment_falls_back_to_dp_when_exact_match_is_ambiguous() {
        let payload = vec![
            UtrPayloadWord {
                text: "hello".to_string(),
                address: test_word_address(0, 0),
            },
            UtrPayloadWord {
                text: "world".to_string(),
                address: test_word_address(0, 1),
            },
        ];
        let asr_tokens = make_asr_tokens(&[
            ("hello", 0, 10),
            ("noise", 10, 20),
            ("hello", 20, 30),
            ("world", 30, 40),
        ]);
        let utt_infos = test_utt_infos(&[2]);

        let plan = plan_untimed_payload(
            &payload,
            &asr_tokens,
            &utt_infos,
            MatchMode::CaseInsensitive,
        );

        assert_eq!(plan.regions[0].strategy, UtrAlignmentStrategy::GlobalDp);
        let range = plan.token_extents()[0].expect("DP fallback should still time the utterance");
        assert_eq!(range.1, 3, "Range should reach the aligned final token");
    }

    #[test]
    fn exact_dp_mode_does_not_use_the_case_insensitive_fast_path() {
        let payload = vec![UtrPayloadWord {
            text: "Hello".to_string(),
            address: test_word_address(0, 0),
        }];
        let asr_tokens = make_asr_tokens(&[("hello", 10, 20)]);
        let utt_infos = test_utt_infos(&[1]);

        let plan = plan_untimed_payload(&payload, &asr_tokens, &utt_infos, MatchMode::Exact);

        assert!(matches!(
            plan.utterances[0],
            UtrUtteranceAlignmentEvidence::Unmatched { .. }
        ));
    }

    #[test]
    fn global_plan_retains_a_proposal_for_an_already_timed_utterance() {
        let input = include_str!("../../../../../test-fixtures/fa_two_timed_utterances.cha");
        let chat = parse_chat(input);
        // Tokens agree with the fixture's bullets (0_5000, 5000_10000): each
        // timed utterance anchors a region, and a token is searched only in
        // the regions whose onset window holds it.
        let tokens = make_asr_tokens(&[
            ("hello", 100, 300),
            ("world", 500, 900),
            ("I", 5_100, 5_200),
            ("want", 5_300, 5_500),
            ("cookie", 5_600, 5_900),
        ]);

        let plan = plan_global_utr_alignment(
            &chat,
            &tokens,
            MatchMode::CaseInsensitive,
            GlobalUtrParticipation::AllUtterances,
        );

        assert_eq!(plan.token_extents(), vec![Some((0, 1)), Some((2, 4))]);
        assert!(matches!(
            plan.utterances[0],
            UtrUtteranceAlignmentEvidence::Matched {
                proposal: UtrTimingProposal::Positive {
                    interval
                },
                ..
            } if interval.start_ms() == 100 && interval.end_ms() == 900
        ));
    }

    #[test]
    fn global_plan_distinguishes_excluded_overlap_from_unmatched() {
        let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant, INV Investigator\n@ID:\teng|test|PAR|||||Participant|||\n@ID:\teng|test|INV|||||Investigator|||\n*PAR:\thello .\n*INV:\t+< mhm .\n@End\n";
        let chat = parse_chat(input);
        let tokens = make_asr_tokens(&[("hello", 100, 300), ("mhm", 200, 250)]);

        let plan = plan_global_utr_alignment(
            &chat,
            &tokens,
            MatchMode::CaseInsensitive,
            GlobalUtrParticipation::ExcludeMarkedOverlap,
        );

        assert!(matches!(
            plan.utterances[1],
            UtrUtteranceAlignmentEvidence::ExcludedMarkedOverlap {
                utterance_index: UtrUtteranceOrdinal(1),
                alignable_words: 1
            }
        ));
    }

    #[test]
    fn global_plan_keeps_fuzzy_and_exact_relations_distinct() {
        let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\tgonna now .\n@End\n";
        let chat = parse_chat(input);
        let tokens = make_asr_tokens(&[("gona", 100, 300), ("now", 400, 600)]);

        let plan = plan_global_utr_alignment(
            &chat,
            &tokens,
            MatchMode::Fuzzy { threshold: 0.85 },
            GlobalUtrParticipation::AllUtterances,
        );
        let UtrUtteranceAlignmentEvidence::Matched { matches, .. } = &plan.utterances[0] else {
            panic!("fuzzy plan should match the utterance");
        };

        assert!(matches!(
            matches.first.relation,
            UtrLexicalRelation::Fuzzy { .. }
        ));
        assert_eq!(matches.rest[0].relation, UtrLexicalRelation::Exact);
    }

    #[test]
    fn global_plan_retains_nonpositive_provider_timing_as_a_refusal_state() {
        let input = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n*PAR:\tmhm .\n@End\n";
        let chat = parse_chat(input);
        let tokens = make_asr_tokens(&[("mhm", 1_000, 1_000)]);

        let plan = plan_global_utr_alignment(
            &chat,
            &tokens,
            MatchMode::CaseInsensitive,
            GlobalUtrParticipation::AllUtterances,
        );

        assert!(matches!(
            plan.utterances[0],
            UtrUtteranceAlignmentEvidence::Matched {
                proposal: UtrTimingProposal::NonPositive {
                    start_ms: 1_000,
                    end_ms: 1_000
                },
                ..
            }
        ));
    }

    #[test]
    fn test_count_utterance_timing() {
        let input =
            include_str!("../../../../../test-fixtures/fa_mixed_timed_untimed_interleaved.cha");
        let chat = parse_chat(input);
        let (timed, untimed) = super::super::grouping::count_utterance_timing(&chat);
        assert_eq!(timed, 3);
        assert_eq!(untimed, 3);
    }

    #[test]
    fn test_count_utterance_timing_all_timed() {
        let input = include_str!("../../../../../test-fixtures/fa_two_timed_utterances.cha");
        let chat = parse_chat(input);
        let (timed, untimed) = super::super::grouping::count_utterance_timing(&chat);
        assert_eq!(timed, 2);
        assert_eq!(untimed, 0);
    }

    /// Regression test: a real-world trimmed UTR fixture from a hand-edited
    /// transcript failure report.
    ///
    /// This is the real token-starvation case: earlier utterances used
    /// to consume ASR tokens that later utterances needed. The regression must
    /// therefore prove two things:
    ///
    /// 1. coverage matches the legacy output, except where the legacy hint
    ///    was a crop our recovery deliberately declines, and
    /// 2. recovered bullets still land in the same timing neighborhood.
    ///
    /// The legacy output is evidence, not gold. Two of its hints were built
    /// from interior words alone, in utterances whose endpoint word the ASR
    /// stream does not contain in any reading: U5 "and dad's driving" (no
    /// "and" was recognised before "Dad's") and U11 "and she's blowing
    /// bubblegum" (the provider wrote "bubble" "gumm"). A hint there is the
    /// hull of the interior tokens, which excludes the endpoint's speech; our
    /// recovery gives those utterances no hint, keeps their interior anchors,
    /// and lets forced alignment search an order corridor instead
    /// (`interior_only_regression_utterances_get_windows_containing_the_legacy_span`).
    /// They are pinned here by utterance and missing endpoint: any other
    /// legacy-timed utterance without a hint is a coverage regression, and so
    /// is either of these two gaining or losing a hint without this list
    /// changing.
    #[test]
    fn test_utr_real_world_trimmed_regression() {
        let chat_input =
            include_str!("../../../../../test-fixtures/utr_real_world_regression_input.cha");
        let tokens_json =
            include_str!("../../../../../test-fixtures/utr_real_world_regression_tokens.json");
        let expected_json =
            include_str!("../../../../../test-fixtures/utr_real_world_regression_expected.json");

        let parser = TreeSitterParser::new().expect("parser");
        let mut chat = batchalign_transform::parse_source_with_parser(&parser, chat_input)
            .admit(
                talkbank_model::model::TranscriptName::Named(
                    talkbank_model::model::FileStem::from_path(std::path::Path::new(
                        "utr_real_world_regression_input.cha",
                    ))
                    .expect("fixture name"),
                ),
                &talkbank_model::NullErrorSink,
            )
            .expect("complete named fixture admission")
            .into_valid_file()
            .into_unchecked();
        let tokens: Vec<AsrTimingToken> = serde_json::from_str(tokens_json).unwrap();

        let expected: Vec<ExpectedUtrFixture> = serde_json::from_str(expected_json).unwrap();

        let result = inject_utr_timing(&mut chat, &tokens);

        // Collect actual timing
        let mut actual_timing: Vec<Option<(u64, u64)>> = Vec::new();
        for line in &chat.lines {
            if let Line::Utterance(utt) = line {
                actual_timing.push(
                    utt.main
                        .content
                        .bullet
                        .as_ref()
                        .map(|b| (b.timing.start_ms, b.timing.end_ms)),
                );
            }
        }

        assert_eq!(
            actual_timing.len(),
            expected.len(),
            "utterance count should match"
        );

        // Check each utterance
        let mut timed_count = 0;
        let mut coverage_regressions = Vec::new();
        let mut timing_regressions = Vec::new();
        for (i, (actual, exp)) in actual_timing.iter().zip(expected.iter()).enumerate() {
            let old_had_timing = exp.start_ms.is_some();
            let new_has_timing = actual.is_some();

            if new_has_timing {
                timed_count += 1;
            }

            if old_had_timing && !new_has_timing {
                coverage_regressions.push(format!(
                    "  U{} {}: expected {}-{}, ba3 has NONE: {}",
                    i + 1,
                    exp.speaker,
                    exp.start_ms.unwrap(),
                    exp.end_ms.unwrap(),
                    &exp.text[..exp.text.len().min(60)]
                ));
                continue;
            }

            if let (Some(actual), Some(start_ms), Some(end_ms)) = (actual, exp.start_ms, exp.end_ms)
                && !spans_roughly_agree(*actual, (start_ms, end_ms))
            {
                timing_regressions.push(format!(
                    "  U{} {}: expected {}-{}, ba3 got {}-{}: {}",
                    i + 1,
                    exp.speaker,
                    start_ms,
                    end_ms,
                    actual.0,
                    actual.1,
                    &exp.text[..exp.text.len().min(60)]
                ));
            }
        }

        // Declined deliberately: (utterance ordinal, the endpoint no reading
        // of the ASR stream contains). See the doc comment.
        let declined = [
            (4, MissingUtrEndpoints::First),
            (10, MissingUtrEndpoints::Last),
        ];
        let UtrAlignmentEvidence::Global { plan } = &result.alignment else {
            panic!("global plan");
        };
        for (index, endpoint) in declined {
            assert!(
                matches!(
                    plan.utterances[index],
                    UtrUtteranceAlignmentEvidence::InteriorOnly { missing_endpoints, .. }
                        if missing_endpoints == endpoint
                ),
                "U{} must be declined for its {endpoint:?} endpoint, got {:?}",
                index + 1,
                plan.utterances[index]
            );
            assert_eq!(actual_timing[index], None, "U{} has no hint", index + 1);
        }
        coverage_regressions.retain(|line| {
            !declined
                .iter()
                .any(|(index, _)| line.starts_with(&format!("  U{} ", index + 1)))
        });

        let old_timed = expected.iter().filter(|e| e.start_ms.is_some()).count();

        // The goal: match or exceed old batchalign's coverage, apart from the
        // declined crops, which are pinned above.
        assert!(
            coverage_regressions.is_empty() && timed_count + declined.len() >= old_timed,
            "UTR regression: old batchalign timed {old_timed}/{total} utterances, \
             ba3 only timed {timed_count}/{total}.\n\
             Regressions ({n_reg}):\n{details}",
            total = expected.len(),
            n_reg = coverage_regressions.len(),
            details = coverage_regressions.join("\n")
        );

        assert!(
            timing_regressions.is_empty(),
            "UTR timing regression on 407 trimmed fixture.\n\
             Expected the restored global-DP path to stay in the same timing \
             neighborhood as the captured reference output.\n\
             Regressions ({n_reg}):\n{details}",
            n_reg = timing_regressions.len(),
            details = timing_regressions.join("\n")
        );

        // Also verify the overall result counters are consistent
        assert_eq!(
            result.injected + result.skipped + result.unmatched,
            expected.len(),
            "result counters should sum to total utterances"
        );
    }

    /// The two legacy-timed utterances the regression above no longer hints
    /// (U5, U11) have interior-only proof: an unproved first or last word,
    /// so no hint. Their neighbours' proved timings still bound where all of
    /// their words can be, and that bound reaches forced alignment as a
    /// search window, never as a bullet. Each window contains the legacy span.
    #[test]
    fn interior_only_regression_utterances_get_windows_containing_the_legacy_span() {
        let parser = TreeSitterParser::new().expect("parser");
        let mut chat = batchalign_transform::parse_source_with_parser(
            &parser,
            include_str!("../../../../../test-fixtures/utr_real_world_regression_input.cha"),
        )
        .admit(
            talkbank_model::model::TranscriptName::Named(
                talkbank_model::model::FileStem::from_path(std::path::Path::new(
                    "utr_real_world_regression_input.cha",
                ))
                .expect("fixture name"),
            ),
            &talkbank_model::NullErrorSink,
        )
        .expect("complete named fixture admission")
        .into_valid_file()
        .into_unchecked();
        let tokens: Vec<AsrTimingToken> = serde_json::from_str(include_str!(
            "../../../../../test-fixtures/utr_real_world_regression_tokens.json"
        ))
        .expect("tokens");
        let expected: Vec<ExpectedUtrFixture> = serde_json::from_str(include_str!(
            "../../../../../test-fixtures/utr_real_world_regression_expected.json"
        ))
        .expect("expected");
        let result = inject_utr_timing(&mut chat, &tokens);
        let UtrAlignmentEvidence::Global { plan } = &result.alignment else {
            panic!("global plan");
        };
        for index in [4, 10] {
            assert!(matches!(
                plan.utterances[index],
                UtrUtteranceAlignmentEvidence::InteriorOnly { .. }
            ));
            assert_eq!(utterance_bullets(&chat)[index], None, "no hint");
            let window = serde_json::to_value(&plan.search_envelopes[index]).expect("wire");
            assert_eq!(window["scope"]["kind"], "order_corridor");
            let (floor, ceiling) = (
                window["floor_ms"].as_u64().expect("floor"),
                window["ceiling_ms"].as_u64().expect("bounded ceiling"),
            );
            let legacy = &expected[index];
            assert!(
                floor <= legacy.start_ms.expect("legacy start")
                    && ceiling >= legacy.end_ms.expect("legacy end"),
                "U{} window [{floor},{ceiling}] contains the legacy span",
                index + 1
            );
        }
    }

    #[test]
    fn test_utr_asr_cache_key_deterministic() {
        use super::super::AudioIdentity;
        let identity = AudioIdentity::from_metadata("/tmp/audio.wav", 1234, 5678);
        let a = super::utr_asr_cache_key(&identity, &crate::options::UtrEngine::Whisper, "eng");
        let b = super::utr_asr_cache_key(&identity, &crate::options::UtrEngine::Whisper, "eng");
        assert_eq!(a, b);
    }

    #[test]
    fn test_utr_asr_cache_key_differs_for_different_inputs() {
        use super::super::AudioIdentity;
        let id1 = AudioIdentity::from_metadata("/tmp/a.wav", 1234, 5678);
        let id2 = AudioIdentity::from_metadata("/tmp/b.wav", 1234, 5678);
        let key1 = super::utr_asr_cache_key(&id1, &crate::options::UtrEngine::Whisper, "eng");
        let key2 = super::utr_asr_cache_key(&id2, &crate::options::UtrEngine::Whisper, "eng");
        assert_ne!(key1, key2, "different audio should produce different keys");

        let key3 = super::utr_asr_cache_key(&id1, &crate::options::UtrEngine::Whisper, "spa");
        assert_ne!(key1, key3, "different lang should produce different keys");
        for engine in [
            crate::options::UtrEngine::RevAi,
            crate::options::UtrEngine::HkTencent,
        ] {
            let other = super::utr_asr_cache_key(&id1, &engine, "eng");
            assert_ne!(
                key1, other,
                "different ASR providers must never share evidence"
            );
        }
    }

    #[test]
    fn test_utr_asr_segment_cache_key_differs_for_windows() {
        use super::super::AudioIdentity;
        let identity = AudioIdentity::from_metadata("/tmp/audio.wav", 1234, 5678);
        let a = super::utr_asr_segment_cache_key(
            &identity,
            &crate::options::UtrEngine::Whisper,
            0,
            5000,
            "eng",
        );
        let b = super::utr_asr_segment_cache_key(
            &identity,
            &crate::options::UtrEngine::Whisper,
            5000,
            10000,
            "eng",
        );
        assert_ne!(a, b, "different windows should produce different keys");
        let other = super::utr_asr_segment_cache_key(
            &identity,
            &crate::options::UtrEngine::HkTencent,
            0,
            5000,
            "eng",
        );
        assert_ne!(a, other, "segment evidence belongs to its ASR provider");
    }

    #[test]
    fn test_find_untimed_windows_all_timed() {
        let input = include_str!("../../../../../test-fixtures/fa_two_timed_utterances.cha");
        let chat = parse_chat(input);
        let windows = super::find_untimed_windows(&chat, &recording(60000), 500)
            .expect("clamped windows lie inside the recording");
        assert!(windows.is_empty(), "all timed → no windows");
    }

    #[test]
    fn test_find_untimed_windows_mixed() {
        let input =
            include_str!("../../../../../test-fixtures/fa_mixed_timed_untimed_interleaved.cha");
        let chat = parse_chat(input);
        // This fixture has 3 timed and 3 untimed utterances interleaved
        let windows = super::find_untimed_windows(&chat, &recording(60000), 500)
            .expect("clamped windows lie inside the recording");
        assert!(!windows.is_empty(), "should find untimed windows");
        // Windows should be non-overlapping and ordered
        for w in windows.windows(2) {
            assert!(
                w[0].end() <= w[1].audio_start(),
                "windows should be non-overlapping"
            );
        }
    }

    #[test]
    fn test_find_untimed_windows_all_untimed() {
        let input = include_str!("../../../../../test-fixtures/fa_two_untimed_with_media.cha");
        let chat = parse_chat(input);
        let windows = super::find_untimed_windows(&chat, &recording(30000), 500)
            .expect("clamped windows lie inside the recording");
        assert_eq!(windows.len(), 1, "all untimed → one merged window");
        assert_eq!(windows[0].audio_start(), FileMs::new(0), "starts at 0");
        assert_eq!(
            windows[0].end(),
            FileMs::new(30000),
            "ends at total_audio_ms"
        );
    }
}
