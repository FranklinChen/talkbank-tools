//! Utterance Timing Recovery (UTR): inject ASR-derived timing into untimed CHAT utterances.
//!
//! When a CHAT file has a mix of timed and untimed utterances, UTR uses ASR
//! output to recover utterance-level bullets for the untimed ones. This is a
//! pre-pass before forced alignment, FA then operates on a fully-timed file.
//!
//! Algorithm:
//! 1. Flatten ALL utterance words (timed + untimed) into one reference sequence,
//!    each tagged with its source utterance index.
//! 2. Try a cheap O(n+m) fast path: if the full transcript words are a
//!    *uniquely embedded* exact monotonic subsequence of the ASR token stream,
//!    use that directly.
//! 3. If the exact subsequence is missing or ambiguous, fall back to a single
//!    global Hirschberg DP alignment of all words against all ASR tokens
//!    (`dp_align::align`).
//! 4. Collect per-utterance min/max matched ASR token indices from the chosen
//!    alignment.
//! 5. Set `utterance.main.content.bullet` from matched tokens' time span
//!    (untimed only). Already-timed utterances are left unchanged.
//!
//! This global monotonic alignment is the current correctness boundary for UTR:
//! it fixes 407-style hand-edited transcript failures where earlier utterances
//! consumed tokens that later utterances needed. It does not make UTR
//! non-monotonic, so dense overlap / text-audio reordering cases can still
//! remain unmatched.

use talkbank_model::model::{Bullet, ChatFile, Line};

use batchalign_transform::dp_align::{self, MatchMode};

use tracing::debug;

use crate::media::window::MediaWindow;

use super::coordinates::{FileMs, Recording};

use super::extraction::collect_fa_words;

mod evidence;
mod lexical;
pub mod overlap_markers;
mod two_pass;

#[cfg(test)]
use evidence::UtrAsrTokenOrdinal;
use evidence::UtrWordOrdinal;
pub use evidence::{
    NonEmptyUtrWordMatches, UtrAlignmentEvidence, UtrAlignmentPlan, UtrAlignmentStrategy,
    UtrAsrTokenAddress, UtrLexicalRelation, UtrOverlapRecovery, UtrResult, UtrTimingProposal,
    UtrUtteranceAlignmentEvidence, UtrUtteranceOrdinal, UtrWordAddress, UtrWordMatch,
};

/// Synthetic drift-class regression scenarios. Public entry point is
/// [`inject_utr_timing`]; the scenarios generate CHAT + ASR in-memory and
/// assert monotonicity / non-silent-strip invariants on the output. The
/// four drift-inducing scenarios are `#[ignore]` because they are currently
/// RED on `GlobalUtr` (the default dispatch), by design; they codify a
/// bug. Run explicitly with
/// `cargo test -p batchalign-chat-ops --lib drift_scenarios -- --ignored`.
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
    /// Whether the utterance already had a bullet before UTR.
    pub(super) has_bullet: bool,
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
    let utt_infos = collect_utr_utterance_info(chat_file);
    if asr_tokens.is_empty() {
        // Nothing can match. The plan still records every utterance as
        // unmatched (or excluded, or wordless) so the evidence is complete,
        // but no decision is written and no bullet is touched: the count is
        // the whole outcome.
        let plan = build_alignment_plan(
            UtrAlignmentStrategy::GlobalDp,
            &[],
            &lexical::UtrLexicalStream::from_tokens(asr_tokens),
            &utt_infos,
            std::iter::empty(),
            participation,
        );
        let skipped = utt_infos.iter().filter(|info| info.has_bullet).count();
        return UtrResult {
            injected: 0,
            skipped,
            unmatched: utt_infos.len() - skipped,
            alignment: UtrAlignmentEvidence::Global { plan },
            decisions: Vec::new(),
        };
    }
    let plan = plan_global_utr_alignment_for(&utt_infos, asr_tokens, dp_match_mode, participation);
    project_global_utr_plan(chat_file, &utt_infos, plan)
}

/// Project a global plan's typed evidence onto the transcript's bullets.
///
/// This consumes each utterance's [`UtrTimingProposal`] as the plan recorded
/// it. It takes nothing but the plan and the census it was built from, so it
/// cannot re-derive a span or a positive/non-positive verdict from raw tokens
/// and disagree with the evidence it also retains: the two used to be
/// computed twice, from the same tokens, in two places. `utt_infos` is the
/// census the plan was built from, so its population is the plan's by
/// construction.
pub(super) fn project_global_utr_plan(
    chat_file: &mut ChatFile,
    utt_infos: &[UtrUtteranceInfo],
    plan: UtrAlignmentPlan,
) -> UtrResult {
    let mut result = UtrResult {
        injected: 0,
        skipped: 0,
        unmatched: 0,
        alignment: UtrAlignmentEvidence::NotRunNoUntimed,
        decisions: Vec::new(),
    };

    // Build utterance ordinal → line index mapping for decision records.
    let utt_line_indices: Vec<usize> = chat_file
        .lines
        .iter()
        .enumerate()
        .filter_map(|(i, line)| {
            if matches!(line, Line::Utterance(_)) {
                Some(i)
            } else {
                None
            }
        })
        .collect();
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
    let mut bullets_to_set: Vec<Option<(u64, u64)>> = vec![None; utt_infos.len()];
    for (utt_idx, (info, evidence)) in utt_infos.iter().zip(&plan.utterances).enumerate() {
        if info.has_bullet {
            result.skipped += 1;
            continue;
        }
        match evidence {
            UtrUtteranceAlignmentEvidence::Matched {
                proposal: UtrTimingProposal::Positive { start_ms, end_ms },
                ..
            } => {
                bullets_to_set[utt_idx] = Some((*start_ms, *end_ms));
                result.injected += 1;
            }
            UtrUtteranceAlignmentEvidence::Matched {
                proposal: UtrTimingProposal::NonPositive { start_ms, end_ms },
                matches,
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

    // Post-pass: enforce strictly increasing start_ms for adjacent non-overlap
    // utterances among the bullets being set.
    //
    // The global DP can assign the same start_ms to two adjacent non-overlap
    // utterances when Whisper's 20ms DTW grid places consecutive short words
    // (e.g., "mhm" at 1000ms and "yeah" also at 1000ms) at the same token
    // boundary.  Without this fix, `enforce_monotonicity` pass 2 later clamps
    // prev.end → next.start = prev.start → zero-duration •T_T•, which fails
    // E362 validation and then perpetuates through every subsequent align re-run.
    //
    // Strategy: walk non-overlap utterances in document order, tracking
    // `floor_end_ms`: the end_ms of the last bullet we committed (either an
    // already-timed utterance's existing bullet, or a newly assigned one).  If a
    // newly assigned bullet starts before `floor_end_ms`, advance its start_ms
    // (and end_ms if necessary) so it begins strictly after the previous one ended.
    {
        // Pre-collect existing timing for already-timed utterances so we can
        // seed `floor_end_ms` when the first timed block precedes new ones.
        let existing_timing: Vec<Option<(u64, u64)>> = chat_file
            .lines
            .iter()
            .filter_map(|l| {
                if let Line::Utterance(u) = l {
                    Some(
                        u.main
                            .content
                            .bullet
                            .as_ref()
                            .map(|b| (b.timing.start_ms, b.timing.end_ms)),
                    )
                } else {
                    None
                }
            })
            .collect();

        let mut floor_end_ms: u64 = 0;
        for (utt_idx, info) in utt_infos.iter().enumerate() {
            // Overlap utterances legitimately share timing with the previous
            // utterance: do not advance their start or update the floor.
            if info.has_lazy_overlap || info.has_ca_overlap {
                continue;
            }
            if info.has_bullet {
                // Already-timed: update floor from existing bullet.
                if let Some(Some((_, end_ms))) = existing_timing.get(utt_idx) {
                    floor_end_ms = floor_end_ms.max(*end_ms);
                }
                continue;
            }
            if let Some((ref mut start_ms, ref mut end_ms)) = bullets_to_set[utt_idx] {
                if *start_ms < floor_end_ms {
                    // Advance start so this bullet begins after the previous one ended.
                    *start_ms = floor_end_ms;
                    if *end_ms <= *start_ms {
                        // Also extend end to preserve at least 1ms of duration.
                        *end_ms = *start_ms + 1;
                    }
                }
                floor_end_ms = *end_ms;
            }
        }
    }

    // Apply bullets to the actual ChatFile utterances
    let mut utt_idx = 0;
    for line in &mut chat_file.lines {
        if let Line::Utterance(utt) = line {
            if let Some((start_ms, end_ms)) = bullets_to_set[utt_idx] {
                // Mark as a provisional UTR hint so that update_utterance_bullet
                // (called after FA injection) overwrites this bullet with the
                // FA word span instead of union-expanding from it.
                utt.main.content.bullet = Some(Bullet::utr_hint(start_ms, end_ms));
            }
            utt_idx += 1;
        }
    }

    result
}

/// Extract alignable words, bullet presence, `+<` linker status, and CA
/// overlap marker info for every utterance in the order UTR sees them.
impl UtrUtteranceInfo {
    /// Whether a global pass with this participation leaves the utterance
    /// out of its flattened word sequence: the one owner of that rule, used
    /// both to build the payload and to label the evidence.
    fn excluded_from(&self, participation: GlobalUtrParticipation) -> bool {
        participation == GlobalUtrParticipation::ExcludeMarkedOverlap
            && (self.has_lazy_overlap || self.has_ca_overlap)
    }
}

pub(super) fn collect_utr_utterance_info(chat_file: &ChatFile) -> Vec<UtrUtteranceInfo> {
    let mut utt_infos = Vec::new();
    for line in &chat_file.lines {
        if let Line::Utterance(utt) = line {
            let mut words = Vec::new();
            collect_fa_words(&utt.main.content.content, &mut words);
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
                has_bullet: utt.main.content.bullet.is_some(),
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
fn plan_global_utr_alignment_for(
    utt_infos: &[UtrUtteranceInfo],
    asr_tokens: &[AsrTimingToken],
    dp_match_mode: MatchMode,
    participation: GlobalUtrParticipation,
) -> UtrAlignmentPlan {
    let mut payload = Vec::new();
    for (utterance_index, info) in utt_infos.iter().enumerate() {
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
        asr_tokens,
        utt_infos,
        dp_match_mode,
        participation,
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UtrPayloadWord {
    text: String,
    address: UtrWordAddress,
}

fn plan_utr_alignment(
    payload: &[UtrPayloadWord],
    asr_tokens: &[AsrTimingToken],
    utt_infos: &[UtrUtteranceInfo],
    dp_match_mode: MatchMode,
    participation: GlobalUtrParticipation,
) -> UtrAlignmentPlan {
    let all_words = payload
        .iter()
        .map(|word| word.text.clone())
        .collect::<Vec<_>>();
    let lexical = lexical::UtrLexicalStream::from_tokens(asr_tokens);
    let asr_texts = lexical.texts();

    // This fast path deliberately uses case-folded comparison, so it is only
    // valid for the case-insensitive DP policy. An exact caller must reach the
    // exact DP relation instead of being silently weakened here.
    if matches!(dp_match_mode, MatchMode::CaseInsensitive)
        && let Some(reference_indices) =
            try_unique_exact_subsequence_indices(&all_words, &asr_texts)
    {
        return build_alignment_plan(
            UtrAlignmentStrategy::UniqueExactSubsequence,
            payload,
            &lexical,
            utt_infos,
            reference_indices.into_iter().enumerate(),
            participation,
        );
    }

    let alignment = dp_align::align(&all_words, &asr_texts, dp_match_mode);
    let matched_indices = alignment.into_iter().filter_map(|item| match item {
        dp_align::AlignResult::Match {
            payload_idx,
            reference_idx,
            ..
        } => Some((payload_idx, reference_idx)),
        dp_align::AlignResult::ExtraPayload { .. }
        | dp_align::AlignResult::ExtraReference { .. } => None,
    });
    build_alignment_plan(
        UtrAlignmentStrategy::GlobalDp,
        payload,
        &lexical,
        utt_infos,
        matched_indices,
        participation,
    )
}

/// Attempt the exact-subsequence fast path for UTR.
///
/// The fast path is accepted only when the transcript words have exactly one
/// monotonic embedding into the ASR stream. Repeated-token ambiguity therefore
/// forces a DP fallback instead of silently accepting an arbitrary greedy path.
fn try_unique_exact_subsequence_indices(
    all_words: &[String],
    asr_texts: &[String],
) -> Option<Vec<usize>> {
    let earliest = greedy_forward_match_indices(all_words, asr_texts)?;
    let latest = greedy_reverse_match_indices(all_words, asr_texts)?;
    if earliest != latest {
        return None;
    }

    Some(earliest)
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

fn build_alignment_plan(
    strategy: UtrAlignmentStrategy,
    payload: &[UtrPayloadWord],
    asr_tokens: &lexical::UtrLexicalStream<'_>,
    utt_infos: &[UtrUtteranceInfo],
    matched_indices: impl IntoIterator<Item = (usize, usize)>,
    participation: GlobalUtrParticipation,
) -> UtrAlignmentPlan {
    let mut by_utterance = vec![Vec::new(); utt_infos.len()];
    for (payload_idx, reference_idx) in matched_indices {
        let payload_word = &payload[payload_idx];
        by_utterance[payload_word.address.utterance_index.index()].push(asr_tokens.matched_word(
            reference_idx,
            payload_word.address,
            &payload_word.text,
        ));
    }

    let utterances = utt_infos
        .iter()
        .enumerate()
        .map(|(utterance_index, info)| {
            let matches = std::mem::take(&mut by_utterance[utterance_index]);
            let Some(matches) = NonEmptyUtrWordMatches::from_vec(matches) else {
                if info.excluded_from(participation) {
                    return UtrUtteranceAlignmentEvidence::ExcludedMarkedOverlap {
                        utterance_index: UtrUtteranceOrdinal(utterance_index),
                        alignable_words: info.words.len(),
                    };
                }
                return if info.words.is_empty() {
                    UtrUtteranceAlignmentEvidence::NoAlignableWords {
                        utterance_index: UtrUtteranceOrdinal(utterance_index),
                    }
                } else {
                    UtrUtteranceAlignmentEvidence::Unmatched {
                        utterance_index: UtrUtteranceOrdinal(utterance_index),
                        alignable_words: info.words.len(),
                    }
                };
            };
            let proposal = asr_tokens.proposal(&matches);
            UtrUtteranceAlignmentEvidence::Matched {
                utterance_index: UtrUtteranceOrdinal(utterance_index),
                alignable_words: info.words.len(),
                matches,
                proposal,
            }
        })
        .collect();

    UtrAlignmentPlan {
        strategy,
        utterances,
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
/// Key = BLAKE3("utr_asr|{audio_identity}|{lang}").
pub fn utr_asr_cache_key(
    audio_identity: &super::AudioIdentity,
    lang: &str,
) -> crate::chat_ops::CacheKey {
    let input = format!("utr_asr|{}|{lang}", audio_identity.as_str());
    crate::chat_ops::CacheKey::from_content(&input)
}

/// Cache key for a segment-level UTR ASR result (partial-window mode).
///
/// Key = BLAKE3("utr_asr_segment|{audio_identity}|{start_ms}|{end_ms}|{lang}").
pub fn utr_asr_segment_cache_key(
    audio_identity: &super::AudioIdentity,
    start_ms: u64,
    end_ms: u64,
    lang: &str,
) -> crate::chat_ops::CacheKey {
    let input = format!(
        "utr_asr_segment|{}|{start_ms}|{end_ms}|{lang}",
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
/// Returns PROVEN windows. This used to hand back bare `(u64, u64)` pairs, and
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
pub fn find_untimed_windows(
    chat_file: &ChatFile,
    recording: &Recording,
    padding_ms: u64,
) -> Vec<MediaWindow> {
    // Collect bullet info for each utterance in order
    let mut utt_bullets: Vec<Option<(u64, u64)>> = Vec::new();
    for line in &chat_file.lines {
        if let Line::Utterance(utt) = line {
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
        return Vec::new();
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
        return Vec::new();
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
    // still pairs. A degenerate one cannot be transcribed, so it is dropped
    // and counted rather than passed on for three consumers to re-check.
    let total = merged.len();
    let windows: Vec<MediaWindow> = merged
        .into_iter()
        .filter_map(|(start, end)| MediaWindow::new(FileMs::new(start), FileMs::new(end)).ok())
        .collect();
    if windows.len() != total {
        debug!(
            dropped = total - windows.len(),
            kept = windows.len(),
            "Dropped degenerate untimed windows (zero-length after clamping)"
        );
    }
    windows
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
                has_bullet: false,
                has_lazy_overlap: false,
                has_ca_overlap: false,
                overlap_onset_fraction: None,
                speaker: "PAR".to_string(),
                bottom_indices: Vec::new(),
                top_onsets: Vec::new(),
            })
            .collect()
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

    /// Projection consumes the plan's typed proposal, never the raw token
    /// stream: the projection signature carries no tokens at all, so a plan
    /// whose proposal says `[100, 900]` projects exactly that, and a
    /// `NonPositive` proposal is refused with the zero-duration decision.
    #[test]
    fn projection_consumes_the_plans_proposal_not_the_raw_tokens() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello world .\n*CHI:\tmhm .\n@End\n";
        let mut chat = parse_chat(chat_text);
        let utt_infos = collect_utr_utterance_info(&chat);
        let matched = |utterance_index: usize, proposal: UtrTimingProposal| {
            UtrUtteranceAlignmentEvidence::Matched {
                utterance_index: UtrUtteranceOrdinal(utterance_index),
                alignable_words: utt_infos[utterance_index].words.len(),
                matches: NonEmptyUtrWordMatches {
                    first: UtrWordMatch {
                        word: test_word_address(utterance_index, 0),
                        token: test_token_address(utterance_index * 2),
                        chat_text: "x".to_string(),
                        asr_text: "x".to_string(),
                        relation: UtrLexicalRelation::Exact,
                    },
                    rest: Vec::new(),
                },
                proposal,
            }
        };
        let plan = UtrAlignmentPlan {
            strategy: UtrAlignmentStrategy::GlobalDp,
            utterances: vec![
                matched(
                    0,
                    UtrTimingProposal::Positive {
                        start_ms: 100,
                        end_ms: 900,
                    },
                ),
                matched(
                    1,
                    UtrTimingProposal::NonPositive {
                        start_ms: 1_000,
                        end_ms: 1_000,
                    },
                ),
            ],
        };
        let result = project_global_utr_plan(&mut chat, &utt_infos, plan);
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

        let plan = plan_utr_alignment(
            &payload,
            &asr_tokens,
            &utt_infos,
            MatchMode::CaseInsensitive,
            GlobalUtrParticipation::AllUtterances,
        );

        assert_eq!(plan.strategy, UtrAlignmentStrategy::UniqueExactSubsequence);
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
                proposal: UtrTimingProposal::Positive {
                    start_ms: 10,
                    end_ms: 30,
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

        let plan = plan_utr_alignment(
            &payload,
            &asr_tokens,
            &utt_infos,
            MatchMode::CaseInsensitive,
            GlobalUtrParticipation::AllUtterances,
        );

        assert_eq!(plan.strategy, UtrAlignmentStrategy::GlobalDp);
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

        let plan = plan_utr_alignment(
            &payload,
            &asr_tokens,
            &utt_infos,
            MatchMode::Exact,
            GlobalUtrParticipation::AllUtterances,
        );

        assert!(matches!(
            plan.utterances[0],
            UtrUtteranceAlignmentEvidence::Unmatched { .. }
        ));
    }

    #[test]
    fn global_plan_retains_a_proposal_for_an_already_timed_utterance() {
        let input = include_str!("../../../../../test-fixtures/fa_two_timed_utterances.cha");
        let chat = parse_chat(input);
        let tokens = make_asr_tokens(&[
            ("hello", 100, 300),
            ("world", 500, 900),
            ("I", 1_100, 1_200),
            ("want", 1_300, 1_500),
            ("cookie", 1_600, 1_900),
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
                    start_ms: 100,
                    end_ms: 900
                },
                ..
            }
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
    /// 1. coverage matches the known-good legacy output, and
    /// 2. recovered bullets still land in the same timing neighborhood.
    #[test]
    fn test_utr_real_world_trimmed_regression() {
        let chat_input =
            include_str!("../../../../../test-fixtures/utr_real_world_regression_input.cha");
        let tokens_json =
            include_str!("../../../../../test-fixtures/utr_real_world_regression_tokens.json");
        let expected_json =
            include_str!("../../../../../test-fixtures/utr_real_world_regression_expected.json");

        let mut chat = parse_chat(chat_input);
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

        let old_timed = expected.iter().filter(|e| e.start_ms.is_some()).count();

        // The goal: match or exceed old batchalign's coverage.
        // Old batchalign: 53/54 timed. We allow at most 1 regression.
        assert!(
            timed_count >= old_timed,
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

    #[test]
    fn test_utr_asr_cache_key_deterministic() {
        use super::super::AudioIdentity;
        let identity = AudioIdentity::from_metadata("/tmp/audio.wav", 1234, 5678);
        let a = super::utr_asr_cache_key(&identity, "eng");
        let b = super::utr_asr_cache_key(&identity, "eng");
        assert_eq!(a, b);
    }

    #[test]
    fn test_utr_asr_cache_key_differs_for_different_inputs() {
        use super::super::AudioIdentity;
        let id1 = AudioIdentity::from_metadata("/tmp/a.wav", 1234, 5678);
        let id2 = AudioIdentity::from_metadata("/tmp/b.wav", 1234, 5678);
        let key1 = super::utr_asr_cache_key(&id1, "eng");
        let key2 = super::utr_asr_cache_key(&id2, "eng");
        assert_ne!(key1, key2, "different audio should produce different keys");

        let key3 = super::utr_asr_cache_key(&id1, "spa");
        assert_ne!(key1, key3, "different lang should produce different keys");
    }

    #[test]
    fn test_utr_asr_segment_cache_key_differs_for_windows() {
        use super::super::AudioIdentity;
        let identity = AudioIdentity::from_metadata("/tmp/audio.wav", 1234, 5678);
        let a = super::utr_asr_segment_cache_key(&identity, 0, 5000, "eng");
        let b = super::utr_asr_segment_cache_key(&identity, 5000, 10000, "eng");
        assert_ne!(a, b, "different windows should produce different keys");
    }

    #[test]
    fn test_find_untimed_windows_all_timed() {
        let input = include_str!("../../../../../test-fixtures/fa_two_timed_utterances.cha");
        let chat = parse_chat(input);
        let windows = super::find_untimed_windows(&chat, &recording(60000), 500);
        assert!(windows.is_empty(), "all timed → no windows");
    }

    #[test]
    fn test_find_untimed_windows_mixed() {
        let input =
            include_str!("../../../../../test-fixtures/fa_mixed_timed_untimed_interleaved.cha");
        let chat = parse_chat(input);
        // This fixture has 3 timed and 3 untimed utterances interleaved
        let windows = super::find_untimed_windows(&chat, &recording(60000), 500);
        assert!(!windows.is_empty(), "should find untimed windows");
        // Windows should be non-overlapping and ordered
        for w in windows.windows(2) {
            assert!(
                w[0].end() <= w[1].start(),
                "windows should be non-overlapping"
            );
        }
    }

    #[test]
    fn test_find_untimed_windows_all_untimed() {
        let input = include_str!("../../../../../test-fixtures/fa_two_untimed_with_media.cha");
        let chat = parse_chat(input);
        let windows = super::find_untimed_windows(&chat, &recording(30000), 500);
        assert_eq!(windows.len(), 1, "all untimed → one merged window");
        assert_eq!(windows[0].start(), FileMs::new(0), "starts at 0");
        assert_eq!(
            windows[0].end(),
            FileMs::new(30000),
            "ends at total_audio_ms"
        );
    }
}
