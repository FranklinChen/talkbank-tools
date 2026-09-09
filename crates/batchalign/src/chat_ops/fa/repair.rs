//! Post-FA bullet repair: fix timing violations without destroying accuracy.
//!
//! Replaces CLAN's FIXBULLETS with three principled strategies, listed
//! below in the order [`repair_bullets`] applies them.
//!
//! The strategies used to carry NUMBERS as well as names, and the two
//! numberings disagreed: this list said 1/2/3 while every comment and
//! constant in the body said 3/1/2 for the same three strategies. A number
//! beside a name is a value proxying for a fact, with no owner and nothing
//! keeping the two in step, so the numbers are gone and the names are the
//! only handle. Nothing outside this file ever referred to them.
//!
//! - **Same-speaker gap filling**: snap small gaps (500-1000ms) between
//!   consecutive same-speaker utterances.
//! - **Boundary averaging**: for small overlaps (≤ threshold), classified
//!   from measured word hulls through the same `EndOverlapResolution` Pass
//!   2 of monotonicity uses (2026-09-01 review, item 7). A cross-speaker
//!   pair is left alone (ordinary conversational overlap) unless the
//!   policy says every adjacent pair clamps; a same-speaker pair whose
//!   hulls do not overlap splits the difference, but never past either
//!   side's own measured hull, so it can only eat inherited coverage,
//!   never a real word; a pair whose hulls DO overlap is a genuine
//!   conflict and is clamped, bullet and words together, through the same
//!   route Pass 2 uses.
//! - **Selective timing removal via LIS**: for large violations, find the
//!   longest increasing subsequence of start times and strip timing from
//!   utterances outside the LIS.
//!
//! Design principle: every bullet either points to the correct audio location
//! or doesn't exist. No lying bullets.

use std::collections::HashMap;

use talkbank_model::model::{ChatFile, Line, Utterance};

use super::EndOverlapPolicy;
use super::orchestrate::{
    ClampedWordCounts, DroppedWordTiming, E704_TOLERANCE_MS, EndOverlapResolution,
    clamp_words_past_bound, classify_end_overlap, earliest_word_timing_start,
    file_order_successor_start_ms, furthest_word_timing_end, strip_utterance_timing,
};

/// Maximum overlap (ms) eligible for boundary averaging.
/// Beyond this, the overlap is either genuine or a real alignment failure.
/// This IS chatter's own E704 tolerance (2026-09-01 review, item 15): not a
/// second, independently hand-typed 500, but the same shared constant Pass
/// 2 of monotonicity would check the overlap against, if it were
/// threshold-gated (it is not: monotonicity resolves every same-speaker
/// overlap regardless of size; only THIS repair strategy is threshold-gated,
/// on purpose, since a large overlap is repair's OWN signal to leave it for
/// monotonicity's blunter but always-correct resolution, or for LIS removal).
const BOUNDARY_AVERAGING_THRESHOLD_MS: u64 = E704_TOLERANCE_MS;

/// Gap range (ms) eligible for same-speaker gap filling.
const GAP_FILL_MAX_MS: u64 = 1000;

/// Whether the optional post-FA bullet-repair phase runs.
///
/// A closed policy keeps pipeline phase selection out of loosely coordinated
/// booleans. Stored command options remain backward-compatible booleans and
/// cross into this type once, at the projection boundary.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum BulletRepairPolicy {
    /// Skip experimental repair.
    #[default]
    Disabled,
    /// Apply repair before final monotonicity enforcement.
    Enabled,
}

impl From<bool> for BulletRepairPolicy {
    fn from(enabled: bool) -> Self {
        if enabled {
            Self::Enabled
        } else {
            Self::Disabled
        }
    }
}

/// Statistics from a bullet repair pass.
#[derive(Debug, Clone, Default)]
pub struct RepairStats {
    /// Utterances whose bullets were adjusted by boundary averaging.
    pub boundary_averaged: usize,
    /// Utterances whose start was snapped to previous same-speaker end.
    pub gaps_filled: usize,
    /// Utterances whose timing was stripped by LIS removal.
    pub timing_stripped: usize,
    /// Total utterances with bullets before repair.
    pub total_bulleted: usize,
}

impl std::fmt::Display for RepairStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "bullet repair: {} boundary-averaged, {} gaps-filled, {} timing-stripped \
             (of {} bulleted utterances)",
            self.boundary_averaged, self.gaps_filled, self.timing_stripped, self.total_bulleted
        )
    }
}

/// A single decision record from the repair pass.
///
/// Each record describes one action taken on one utterance, with enough
/// detail for structured evidence and the evaluation harness.
#[derive(Debug, Clone)]
pub struct RepairDecision {
    /// Index into `chat_file.lines`.
    pub line_idx: usize,
    /// Speaker code.
    pub speaker: String,
    /// Typed FA strategy that produced this decision. Narrower than
    /// the crate-wide [`DecisionStrategy`](batchalign_transform::decisions::DecisionStrategy):
    /// this struct is FA-specific, so the strategy is constrained to
    /// `FaStrategy` at the point of construction.
    pub strategy: batchalign_transform::decisions::FaStrategy,
    /// Human-readable reason string for evidence and tracing. Stays a
    /// SUMMARY (a count, via `words_dropped.len()`, when this decision cut
    /// any words at all): the full per-word records live in
    /// `words_dropped` instead, which is where a reviewer actually needs
    /// them, not duplicated into a string.
    pub reason: String,
    /// Words whose timing was thrown away entirely by THIS decision
    /// (empty for every strategy but `BoundaryAveraged`'s
    /// `InterleavedWords` resolution, which is the only one that ever
    /// clamps a word at all). Carries what `clamp_words_past_bound`
    /// already returns via `ClampedWordCounts::dropped`, so the measured
    /// span survives past this decision instead of being reduced to
    /// `words_clamped.dropped.len()` in the reason string and nothing
    /// else (2026-09-02 review).
    pub words_dropped: Vec<DroppedWordTiming>,
    /// Whether this decision is low-confidence and needs human review.
    pub needs_review: bool,
}

/// Combined result of a repair pass: aggregate stats + per-utterance decisions.
#[derive(Debug, Clone, Default)]
pub struct RepairResult {
    /// Aggregate statistics.
    pub stats: RepairStats,
    /// Per-utterance decision log (one entry per action taken).
    pub decisions: Vec<RepairDecision>,
}

/// Info about an utterance's bullet, extracted for gap-filling and LIS
/// analysis. Boundary averaging (2026-09-01 review, item 11) no longer uses
/// this snapshot at all -- it reads bullets and word hulls LIVE, from
/// `chat_file` directly, at the point each pair is resolved.
struct BulletEntry {
    /// Index into `chat_file.lines` (the `Line::Utterance` position).
    line_idx: usize,
    /// Speaker code.
    speaker: String,
    /// Current bullet start_ms.
    start_ms: u64,
    /// Current bullet end_ms.
    end_ms: u64,
}

/// Apply all three repair strategies to a parsed CHAT file.
///
/// Strategies are applied in order:
/// 1. Same-speaker gap filling (conservative, always safe)
/// 2. Boundary averaging (for small overlaps)
/// 3. Selective timing removal via LIS (for large violations)
///
/// Returns statistics describing what was changed.
///
/// `end_overlap_policy` governs boundary averaging exactly as
/// it governs Pass 2 of monotonicity enforcement: a cross-speaker pair is
/// ordinary conversational overlap and is left alone under the default
/// `PreserveCrossSpeaker`, averaged only under `ClampAllAdjacent`
/// (2026-09-01 review, item 7).
pub fn repair_bullets(
    chat_file: &mut ChatFile,
    dry_run: bool,
    end_overlap_policy: EndOverlapPolicy,
) -> RepairResult {
    let mut stats = RepairStats::default();
    let mut decisions = Vec::new();

    // Collect bullet entries in document order.
    let entries = collect_bullet_entries(chat_file);
    stats.total_bulleted = entries.len();

    if entries.is_empty() {
        return RepairResult { stats, decisions };
    }

    // Same-speaker gap filling.
    // Must run first because it only narrows gaps, never creates new violations.
    let gap_fills = find_gap_fills(&entries);
    stats.gaps_filled = gap_fills.len();
    for &(line_idx, new_start_ms) in &gap_fills {
        // SAFETY: `line_idx` came from `find_gap_fills(&entries)`, which only
        // produces indices present in `entries`.
        #[allow(clippy::unwrap_used)]
        let entry = entries.iter().find(|e| e.line_idx == line_idx).unwrap();
        let gap = entry.start_ms - new_start_ms;
        decisions.push(RepairDecision {
            line_idx,
            speaker: entry.speaker.clone(),
            strategy: batchalign_transform::decisions::FaStrategy::GapFilled,
            reason: format!(
                "gap_filled gap={}ms same_speaker machine={}_{} snapped_start={}",
                gap, entry.start_ms, entry.end_ms, new_start_ms
            ),
            // Gap filling only moves a bullet's START; it never touches a
            // word's timing.
            words_dropped: Vec::new(),
            needs_review: true,
        });
    }

    // Boundary averaging needs to see gap-filling's effect and
    // must itself run LIVE and INTERLEAVED (2026-09-01 review, item 11): it
    // operates on a scratch clone under `dry_run` (so the real file stays
    // untouched) or on `chat_file` directly otherwise, and either way gap
    // fills are applied to that target FIRST.
    let mut scratch;
    let working: &mut ChatFile = if dry_run {
        scratch = chat_file.clone();
        &mut scratch
    } else {
        chat_file
    };
    for &(line_idx, new_start_ms) in &gap_fills {
        if let Some(utt) = get_utterance_mut(working, line_idx)
            && let Some(ref mut bullet) = utt.main.content.bullet
        {
            bullet.timing.start_ms = new_start_ms;
        }
    }

    // Boundary averaging for small overlaps, classified from
    // measured word hulls exactly as Pass 2 of monotonicity classifies an
    // end overlap (2026-09-01 review, item 7): a cross-speaker pair is left
    // alone unless the policy says otherwise, a pair whose hulls do not
    // overlap is averaged with each edge clamped to stay inside its own
    // coverage (never past the measured hull, so averaging can never eat a
    // real word), and a pair whose hulls DO overlap is the same conflict
    // Pass 2 calls `InterleavedWords`: the bullet and every affected word
    // are clamped together through `clamp_words_past_bound`, the one route
    // by which either pass may cut a word.
    let (boundary_averaged, boundary_decisions) =
        resolve_boundary_averages(working, end_overlap_policy);
    stats.boundary_averaged = boundary_averaged;
    decisions.extend(boundary_decisions);

    // Selective timing removal via LIS. Re-collected from
    // `working`'s CURRENT state (post gap-fill, post boundary-averaging),
    // not the original snapshot: a boundary average can itself have moved a
    // start, and LIS must decide against what is actually there.
    let entries_after_averaging = collect_bullet_entries(working);
    let lis_removals = find_lis_removals(&entries_after_averaging);
    stats.timing_stripped = lis_removals.len();
    for &line_idx in &lis_removals {
        // SAFETY: `line_idx` came from `find_lis_removals`, which only
        // produces indices present in `entries_after_averaging`.
        #[allow(clippy::unwrap_used)]
        let entry = entries_after_averaging
            .iter()
            .find(|e| e.line_idx == line_idx)
            .unwrap();
        decisions.push(RepairDecision {
            line_idx,
            speaker: entry.speaker.clone(),
            strategy: batchalign_transform::decisions::FaStrategy::LisRemoval,
            // LIS removal strips a whole utterance's timing outright; it
            // does not clamp individual words, so there is nothing to
            // report here beyond the reason text below.
            words_dropped: Vec::new(),
            reason: format!(
                "lis_removal same_speaker_non_monotonic machine={}_{}",
                entry.start_ms, entry.end_ms
            ),
            needs_review: true,
        });
    }

    if dry_run {
        return RepairResult { stats, decisions };
    }

    // Gap fills and boundary averages are already applied to `working`,
    // which under `!dry_run` IS `chat_file`. Only LIS removal remains.
    for &line_idx in &lis_removals {
        if let Some(utt) = get_utterance_mut(chat_file, line_idx) {
            strip_utterance_timing(utt);
        }
    }

    RepairResult { stats, decisions }
}

/// Boundary averaging's live, interleaved sweep (2026-09-01 review, item 11):
/// resolve one eligible small-overlap pair at a time, APPLY it to
/// `chat_file` immediately, then move to the next pair, so a pair's
/// classification always sees what an EARLIER pair in this same sweep
/// already did -- exactly Pass 2's own approach
/// (`orchestrate::enforce_monotonicity_with_policy`), and for the same
/// reason: a two-phase "compute every pair from one snapshot, apply
/// afterward" design lets (A,B) move B's edge and then (B,C) classify
/// against B's STALE, pre-move state. The successor guard input is read
/// live too, via `orchestrate::file_order_successor_start_ms` (2026-09-01
/// review, item 16: the FILE-order successor, any speaker, shared with
/// Pass 2, not a per-sweep substitute), giving `classify_end_overlap`'s
/// `BoundaryFromWords` guard the same protection against creating a fresh
/// file-order start violation that Pass 2 has.
///
/// Two sweeps (2026-09-01 review, item 15), same order and proof as Pass
/// 2's: the per-speaker-stream sweep runs FIRST and UNCONDITIONALLY (E704
/// same-speaker overlap must be caught regardless of policy, and is defined
/// on the speaker's OWN sequence, not on file adjacency); the file-adjacent
/// sweep runs SECOND, honoring `end_overlap_policy`. Every resolution this
/// pass makes only SHRINKS the pair it touches, so a pair the per-speaker
/// sweep already resolved satisfies the file-adjacent sweep's own entry
/// guard (`later.start_ms >= earlier.end_ms`) and is a no-op there; see
/// `orchestrate::enforce_monotonicity_with_policy`'s own doc for the full
/// argument, identical here.
///
/// Returns the number of pairs resolved and their decision records.
fn resolve_boundary_averages(
    chat_file: &mut ChatFile,
    end_overlap_policy: EndOverlapPolicy,
) -> (usize, Vec<RepairDecision>) {
    let mut decisions = Vec::new();
    let mut resolved = 0usize;

    resolved += resolve_boundary_averages_same_speaker_stream(chat_file, &mut decisions);

    // IDENTITY ONLY (line index, speaker), never a timing value, and
    // captured AFTER the per-speaker sweep so it reflects anything that
    // sweep already moved. Every timing this sweep needs is read live, at
    // the point it is used.
    let identity: Vec<(usize, String)> = collect_bullet_entries(chat_file)
        .into_iter()
        .map(|entry| (entry.line_idx, entry.speaker))
        .collect();

    for window_idx in 0..identity.len().saturating_sub(1) {
        let (earlier_line_idx, earlier_speaker) = &identity[window_idx];
        let (later_line_idx, later_speaker) = &identity[window_idx + 1];
        let (earlier_line_idx, later_line_idx) = (*earlier_line_idx, *later_line_idx);

        if !end_overlap_policy.should_clamp(earlier_speaker, later_speaker) {
            continue;
        }

        if let Some(decision) = resolve_boundary_average_pair(
            chat_file,
            earlier_line_idx,
            earlier_speaker,
            later_line_idx,
            later_speaker,
        ) {
            decisions.push(decision);
            resolved += 1;
        }
    }

    (resolved, decisions)
}

/// The per-speaker-stream sweep (2026-09-01 review, item 15): each
/// speaker's own bulleted utterances, in file order, paired and resolved
/// consecutively WITHIN that speaker's own stream, skipping any
/// intervening other-speaker utterance -- exactly
/// `orchestrate::resolve_same_speaker_stream_overlaps`, for the same
/// reason (E704 is defined on the speaker's OWN sequence). Speakers are
/// visited in first-appearance order for deterministic decision ordering;
/// a pair is always same-speaker by construction, so the order across
/// speakers cannot change what either speaker's own resolution computes.
///
/// Returns the number of pairs resolved.
fn resolve_boundary_averages_same_speaker_stream(
    chat_file: &mut ChatFile,
    decisions: &mut Vec<RepairDecision>,
) -> usize {
    let mut speaker_order: Vec<String> = Vec::new();
    let mut streams: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    for entry in collect_bullet_entries(chat_file) {
        if !streams.contains_key(&entry.speaker) {
            speaker_order.push(entry.speaker.clone());
        }
        streams
            .entry(entry.speaker)
            .or_default()
            .push(entry.line_idx);
    }

    let mut resolved = 0usize;
    for speaker in &speaker_order {
        // SAFETY: `speaker` came from `speaker_order`, built from the same
        // insertion the `streams` entry itself was.
        #[allow(clippy::unwrap_used)]
        let stream = streams.get(speaker).unwrap().clone();
        for window_idx in 0..stream.len().saturating_sub(1) {
            let earlier_line_idx = stream[window_idx];
            let later_line_idx = stream[window_idx + 1];
            if let Some(decision) = resolve_boundary_average_pair(
                chat_file,
                earlier_line_idx,
                speaker,
                later_line_idx,
                speaker,
            ) {
                decisions.push(decision);
                resolved += 1;
            }
        }
    }
    resolved
}

/// Resolve one eligible small-overlap PAIR, live, against `chat_file`.
/// Shared (2026-09-01 review, item 15) by both sweeps above; only how the
/// pair and its successor are FOUND differs between them. Returns `None`
/// when there is no overlap, the overlap exceeds
/// [`BOUNDARY_AVERAGING_THRESHOLD_MS`], or either utterance has no bullet.
fn resolve_boundary_average_pair(
    chat_file: &mut ChatFile,
    earlier_line_idx: usize,
    earlier_speaker: &str,
    later_line_idx: usize,
    later_speaker: &str,
) -> Option<RepairDecision> {
    let earlier_bullet = get_utterance_mut(chat_file, earlier_line_idx)
        .and_then(|utt| utt.main.content.bullet.clone())?;
    let later_bullet = get_utterance_mut(chat_file, later_line_idx)
        .and_then(|utt| utt.main.content.bullet.clone())?;

    if later_bullet.timing.start_ms >= earlier_bullet.timing.end_ms {
        return None;
    }
    let overlap = earlier_bullet.timing.end_ms - later_bullet.timing.start_ms;
    if overlap > BOUNDARY_AVERAGING_THRESHOLD_MS {
        return None;
    }

    let earlier_hull_end_ms = get_utterance_mut(chat_file, earlier_line_idx)
        .as_deref()
        .and_then(furthest_word_timing_end);
    let later_hull_start_ms = get_utterance_mut(chat_file, later_line_idx)
        .as_deref()
        .and_then(earliest_word_timing_start);
    // The FILE-ORDER successor, live, any speaker (2026-09-01 review, item
    // 16): the only correct guard input for `BoundaryFromWords`, shared
    // with Pass 2 rather than a second, weaker per-caller substitute.
    let next_successor_start_ms = file_order_successor_start_ms(chat_file, later_line_idx);

    let resolution = classify_end_overlap(
        earlier_hull_end_ms,
        later_bullet.timing.start_ms,
        later_hull_start_ms,
        next_successor_start_ms,
    );

    let (earlier_new_end_ms, later_new_start_ms) = match resolution {
        EndOverlapResolution::InterleavedWords => {
            // A genuine word conflict: clamp fully to the later
            // utterance's own original start, exactly like Pass 2's
            // `InterleavedWords`. The later utterance is not moved.
            (later_bullet.timing.start_ms, later_bullet.timing.start_ms)
        }
        EndOverlapResolution::BoundaryFromWords {
            prev_hull_end_ms,
            next_hull_start_ms,
        } => {
            // Both hulls measured; assign them DIRECTLY (2026-09-01
            // review, item 10), never a midpoint (see that item's own
            // record for the fresh-overlap bug a midpoint produced
            // here).
            (prev_hull_end_ms, next_hull_start_ms)
        }
        EndOverlapResolution::CoverageOnly { hull_end_ms } => match hull_end_ms {
            Some(hull_end_ms) => (hull_end_ms, later_bullet.timing.start_ms),
            None => {
                let midpoint = later_bullet.timing.start_ms + overlap / 2;
                (midpoint, midpoint)
            }
        },
    };

    // Apply immediately, before the next pair in this sweep is read.
    if let Some(utt) = get_utterance_mut(chat_file, earlier_line_idx)
        && let Some(ref mut bullet) = utt.main.content.bullet
    {
        bullet.timing.end_ms = earlier_new_end_ms;
    }
    let mut words_clamped = ClampedWordCounts::default();
    match resolution {
        EndOverlapResolution::InterleavedWords => {
            if let Some(utt) = get_utterance_mut(chat_file, earlier_line_idx) {
                words_clamped = clamp_words_past_bound(utt, earlier_new_end_ms);
            }
        }
        EndOverlapResolution::CoverageOnly { .. }
        | EndOverlapResolution::BoundaryFromWords { .. } => {
            if let Some(utt) = get_utterance_mut(chat_file, later_line_idx)
                && let Some(ref mut bullet) = utt.main.content.bullet
            {
                bullet.timing.start_ms = later_new_start_ms;
            }
        }
    }

    let reason = match resolution {
        EndOverlapResolution::InterleavedWords => format!(
            "boundary_averaged overlap={overlap}ms resolution=interleaved_words clamped_to={earlier_new_end_ms} machine={}_{} adjacent={earlier_speaker}:{earlier_line_idx} words_trimmed={} words_dropped={}",
            later_bullet.timing.start_ms,
            later_bullet.timing.end_ms,
            words_clamped.trimmed,
            words_clamped.dropped.len(),
        ),
        EndOverlapResolution::CoverageOnly { .. }
        | EndOverlapResolution::BoundaryFromWords { .. } => format!(
            "boundary_averaged overlap={overlap}ms resolution=hull_respecting earlier_new_end={earlier_new_end_ms} later_new_start={later_new_start_ms} machine={}_{} adjacent={earlier_speaker}:{earlier_line_idx}",
            later_bullet.timing.start_ms, later_bullet.timing.end_ms,
        ),
    };

    Some(RepairDecision {
        line_idx: later_line_idx,
        speaker: later_speaker.to_string(),
        strategy: batchalign_transform::decisions::FaStrategy::BoundaryAveraged,
        reason,
        // The per-word records `clamp_words_past_bound` already collected
        // (2026-09-02 review): a `CoverageOnly`/`BoundaryFromWords`
        // resolution never calls it at all, so `words_clamped.dropped` is
        // still its `Default`, empty `Vec`, here.
        words_dropped: words_clamped.dropped,
        // A hull-respecting average never touches a measured word, same
        // as Pass 2's `CoverageOnly`/`BoundaryFromWords`; only a real
        // word conflict (`InterleavedWords`) needs a human's review.
        needs_review: matches!(resolution, EndOverlapResolution::InterleavedWords),
    })
}

/// Collect bullet entries from all main-tier utterances in document order.
fn collect_bullet_entries(chat_file: &ChatFile) -> Vec<BulletEntry> {
    let mut entries = Vec::new();

    for (line_idx, line) in chat_file.lines.iter().enumerate() {
        let Line::Utterance(utt) = line else {
            continue;
        };
        let Some(ref bullet) = utt.main.content.bullet else {
            continue;
        };
        entries.push(BulletEntry {
            line_idx,
            speaker: utt.main.speaker.to_string(),
            start_ms: bullet.timing.start_ms,
            end_ms: bullet.timing.end_ms,
        });
    }

    entries
}

/// Find same-speaker gaps eligible for filling.
///
/// Returns `(line_idx, new_start_ms)` pairs, the later utterance's start
/// should be snapped to the previous same-speaker utterance's end.
///
/// Never needs the measured-hull treatment `find_boundary_averages` and
/// Pass 2 both need (2026-09-01 review, item 7): it only fires on a
/// POSITIVE gap (`entry.start_ms > prev_end`) and only ever moves that
/// start EARLIER, toward `prev_end`, which WIDENS the later utterance's
/// bullet rather than narrowing it. It never touches an end, never touches
/// a word, and can never cut into anything, measured or not.
fn find_gap_fills(entries: &[BulletEntry]) -> Vec<(usize, u64)> {
    let mut fills = Vec::new();
    // Track per-speaker last end time.
    let mut speaker_last_end: HashMap<&str, u64> = HashMap::new();

    for entry in entries {
        if let Some(&prev_end) = speaker_last_end.get(entry.speaker.as_str())
            && entry.start_ms > prev_end
        {
            let gap = entry.start_ms - prev_end;
            if gap <= GAP_FILL_MAX_MS {
                fills.push((entry.line_idx, prev_end));
            }
        }
        speaker_last_end.insert(&entry.speaker, entry.end_ms);
    }

    fills
}

/// Find utterances to strip timing from using per-speaker LIS.
///
/// For each speaker, computes the Longest Increasing Subsequence of start
/// times. Utterances NOT in their speaker's LIS have same-speaker
/// non-monotonic timing. Their timing is stripped rather than mangled.
///
/// Cross-speaker non-monotonicity is intentionally left alone, it
/// represents normal conversational overlap, not a data error.
///
/// Returns `line_idx` values of utterances to strip.
fn find_lis_removals(entries: &[BulletEntry]) -> Vec<usize> {
    // Group entry indices by speaker.
    let mut speaker_entries: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, entry) in entries.iter().enumerate() {
        speaker_entries
            .entry(entry.speaker.as_str())
            .or_default()
            .push(i);
    }

    let mut removals = Vec::new();

    for indices in speaker_entries.values() {
        if indices.len() <= 1 {
            continue;
        }

        // Extract start times for this speaker's utterances (in document order).
        let starts: Vec<u64> = indices.iter().map(|&i| entries[i].start_ms).collect();
        let lis = longest_increasing_subsequence(&starts);

        // Build set of positions (within this speaker's list) that are in the LIS.
        let mut in_lis = vec![false; indices.len()];
        for &pos in &lis {
            in_lis[pos] = true;
        }

        // Entries NOT in this speaker's LIS are stripped. They are the
        // minimal set whose removal makes the speaker's timeline monotonic.
        for (pos, &entry_idx) in indices.iter().enumerate() {
            if !in_lis[pos] {
                removals.push(entries[entry_idx].line_idx);
            }
        }
    }

    removals
}

/// Compute the Longest Increasing Subsequence (non-strictly increasing).
///
/// Returns the indices of elements in the LIS.
/// Uses the patience sorting / binary search algorithm: O(n log n).
fn longest_increasing_subsequence(values: &[u64]) -> Vec<usize> {
    if values.is_empty() {
        return Vec::new();
    }

    let n = values.len();
    // tails[i] = index of smallest tail element for IS of length i+1.
    let mut tails: Vec<usize> = Vec::new();
    // prev[i] = index of previous element in LIS ending at i.
    let mut prev: Vec<Option<usize>> = vec![None; n];

    for i in 0..n {
        // Binary search: find first tail >= values[i].
        let pos = tails.partition_point(|&t| values[t] <= values[i]);

        if pos == tails.len() {
            tails.push(i);
        } else {
            tails[pos] = i;
        }

        if pos > 0 {
            prev[i] = Some(tails[pos - 1]);
        }
    }

    // Reconstruct LIS from prev pointers.
    let mut result = Vec::with_capacity(tails.len());
    // SAFETY: `tails` is non-empty because the loop above always pushes at least
    // one element (input `starts` is non-empty, checked by caller).
    #[allow(clippy::unwrap_used)]
    let mut idx = *tails.last().unwrap();
    result.push(idx);
    while let Some(p) = prev[idx] {
        result.push(p);
        idx = p;
    }
    result.reverse();
    result
}

/// Get a mutable reference to an utterance by its line index.
fn get_utterance_mut(chat_file: &mut ChatFile, line_idx: usize) -> Option<&mut Utterance> {
    if let Some(Line::Utterance(utt)) = chat_file.lines.as_mut_slice().get_mut(line_idx) {
        Some(utt)
    } else {
        None
    }
}

impl From<&RepairDecision> for batchalign_transform::decisions::DecisionRecord {
    fn from(d: &RepairDecision) -> Self {
        Self {
            line_idx: batchalign_transform::decisions::LineIdx::new(d.line_idx),
            speaker: d.speaker.clone(),
            strategy: batchalign_transform::decisions::DecisionStrategy::Fa(d.strategy),
            reason: d.reason.clone(),
            needs_review: d.needs_review,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Integration test: parse a real trimmed CHAT file with E704 (same-speaker
    /// overlap) and E701 (cross-speaker non-monotonicity), run bullet repair,
    /// verify that boundary averaging and LIS removal produce correct results.
    ///
    /// The fixture is `include_str!`d rather than read at run time, and that is
    /// load-bearing for more than tidiness. It was the ONLY filesystem call in
    /// `chat_ops`, a 16,000-line module that is otherwise 97% pure and imports
    /// nothing else in the crate, so this one line was what kept the largest
    /// movable block in the crate out of `batchalign-core`. `include_str!` is
    /// resolved by the compiler and the binary touches no disk, the same
    /// distinction the purity gate already draws between `env!` and
    /// `std::env::var`.
    ///
    /// It is also simply a better test: the old form resolved
    /// `../../test-fixtures/...` against the process working directory, so it
    /// failed at run time with a "regenerate the fixture" message whenever the
    /// cwd was not the package root. A missing fixture is now a compile error
    /// that names the path.
    ///
    /// Policy changed 2026-09-01 (review item 7): boundary averaging now
    /// takes an `EndOverlapPolicy` and classifies each pair from measured
    /// word hulls. This test passes `ClampAllAdjacent` deliberately, to keep
    /// exercising the fixture's cross-speaker overlaps (its own comment
    /// below already documents that cross-speaker non-monotonicity survives
    /// repair as normal conversational overlap); a dedicated test below
    /// covers the new default, `PreserveCrossSpeaker`, leaving a
    /// cross-speaker pair untouched. `needs_review` is no longer
    /// unconditionally `true`: a hull-respecting average (no word conflict)
    /// does not need review, matching Pass 2's own `CoverageOnly` /
    /// `BoundaryFromWords`; only a genuine word conflict does.
    #[test]
    fn test_repair_on_real_bre_fixture() {
        let chat_text = include_str!("../../../../../test-fixtures/bullet_repair_e704.cha");

        let parser = batchalign_transform::parse::TreeSitterParser::new().expect("parser init");
        let (mut chat_file, _errors) =
            batchalign_transform::parse::parse_lenient(&parser, chat_text);

        // Dry-run first: verify we detect issues without modifying the file.
        let dry_result = repair_bullets(&mut chat_file, true, EndOverlapPolicy::ClampAllAdjacent);
        assert!(
            dry_result.stats.total_bulleted > 0,
            "fixture should have bulleted utterances"
        );
        // The fixture has cross-speaker overlaps that trigger boundary averaging,
        // and at least one same-speaker overlap. The exact counts may shift if
        // the fixture is re-trimmed, so just verify we found something to repair.
        assert!(
            !dry_result.decisions.is_empty(),
            "expected at least one repair decision"
        );

        // Now apply for real.
        let result = repair_bullets(&mut chat_file, false, EndOverlapPolicy::ClampAllAdjacent);
        assert_eq!(result.stats.total_bulleted, dry_result.stats.total_bulleted);
        // Decisions should include per-utterance records.
        assert!(!result.decisions.is_empty());
        // Every decision should have a non-empty reason; whether it needs
        // review now depends on the resolution (see this test's docstring).
        for d in &result.decisions {
            assert!(!d.reason.is_empty(), "decision has empty reason");
        }

        // After repair: same-speaker bullets should be monotonically increasing.
        // Cross-speaker non-monotonicity is expected (normal conversational overlap).
        let entries_after = collect_bullet_entries(&chat_file);
        let mut speaker_last_start: HashMap<&str, u64> = HashMap::new();
        for entry in &entries_after {
            if let Some(&prev) = speaker_last_start.get(entry.speaker.as_str()) {
                assert!(
                    entry.start_ms >= prev,
                    "same-speaker non-monotonic after repair: {} starts at {}ms but \
                     previously started at {}ms",
                    entry.speaker,
                    entry.start_ms,
                    prev,
                );
            }
            speaker_last_start.insert(&entry.speaker, entry.start_ms);
        }
    }

    /// Verify that repair on a clean file (no violations) is a no-op.
    #[test]
    fn test_repair_noop_on_clean_file() {
        // Construct a minimal CHAT with two well-ordered bullets.
        // Gap > 1000ms so gap-filling doesn't trigger.
        let chat_text = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI|2;0.0||||Target_Child|||
*CHI:\thello . \u{0015}1000_2000\u{0015}
*CHI:\tworld . \u{0015}4000_5000\u{0015}
@End
";
        let parser = batchalign_transform::parse::TreeSitterParser::new().expect("parser init");
        let (mut chat_file, _errors) =
            batchalign_transform::parse::parse_lenient(&parser, chat_text);

        let result = repair_bullets(&mut chat_file, false, EndOverlapPolicy::ClampAllAdjacent);
        assert_eq!(result.stats.boundary_averaged, 0);
        assert_eq!(result.stats.gaps_filled, 0);
        assert_eq!(result.stats.timing_stripped, 0);
        assert_eq!(result.stats.total_bulleted, 2);
        assert!(result.decisions.is_empty());
    }

    fn parse(text: &str) -> ChatFile {
        let parser = batchalign_transform::parse::TreeSitterParser::new().expect("parser init");
        batchalign_transform::parse::parse_lenient(&parser, text).0
    }

    /// 2026-09-01 review, item 7: a cross-speaker overlap is ordinary
    /// conversational overlap and must not be averaged under the default
    /// policy, `PreserveCrossSpeaker`, exactly as Pass 2 of monotonicity
    /// leaves a cross-speaker end overlap alone under that policy.
    #[test]
    fn cross_speaker_overlap_is_untouched_under_preserve_cross_speaker() {
        let mut chat = parse(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child, MOT Mother\n\
             @ID:\teng|x|CHI|||||Child|||\n@ID:\teng|x|MOT|||||Mother|||\n\
             *CHI:\thello . \u{15}1000_3000\u{15}\n*MOT:\tworld . \u{15}2700_4000\u{15}\n@End\n",
        );

        let result = repair_bullets(&mut chat, false, EndOverlapPolicy::PreserveCrossSpeaker);

        assert_eq!(result.stats.boundary_averaged, 0);
        assert!(result.decisions.is_empty());
        assert_eq!(collect_bullet_entries(&chat)[0].end_ms, 3000);
        assert_eq!(collect_bullet_entries(&chat)[1].start_ms, 2700);
    }

    /// 2026-09-01 review, item 7: a same-speaker overlap whose measured word
    /// hulls do NOT themselves overlap resolves like Pass 2's
    /// `BoundaryFromWords`: both bullet edges move toward a boundary that
    /// respects both hulls (never past either), and neither word is
    /// touched.
    #[test]
    fn same_speaker_overlap_with_non_overlapping_hulls_averages_without_touching_words() {
        let mut chat = parse(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n\
             @ID:\teng|x|CHI|||||Child|||\n\
             *CHI:\thello . \u{15}1000_3000\u{15}\n%wor:\thello \u{15}1000_2800\u{15} .\n\
             *CHI:\tworld . \u{15}2700_4000\u{15}\n%wor:\tworld \u{15}2900_4000\u{15} .\n@End\n",
        );

        let result = repair_bullets(&mut chat, false, EndOverlapPolicy::ClampAllAdjacent);

        assert_eq!(result.stats.boundary_averaged, 1);
        assert!(!result.decisions[0].needs_review, "no word conflict");
        let entries = collect_bullet_entries(&chat);
        // 2026-09-01 review, item 10: assigned from the measured hulls
        // DIRECTLY, no midpoint, since both hulls are known.
        assert_eq!(
            entries[0].end_ms, 2800,
            "the earlier utterance's own hull end"
        );
        assert_eq!(
            entries[1].start_ms, 2900,
            "the next utterance's own hull start"
        );
        assert_eq!(
            get_utterance_mut(&mut chat, entries[0].line_idx)
                .as_deref()
                .and_then(furthest_word_timing_end),
            Some(2800),
            "earlier word untouched"
        );
        assert_eq!(
            get_utterance_mut(&mut chat, entries[1].line_idx)
                .as_deref()
                .and_then(earliest_word_timing_start),
            Some(2900),
            "later word untouched"
        );
    }

    /// 2026-09-01 review, item 10: the exact failing construction. Close
    /// hulls (2950 / 2960) independently clamped against a shared midpoint
    /// (2850) used to give (2950, 2850): the earlier utterance's new end
    /// STILL past the later utterance's new start, a FRESH 100 ms overlap
    /// the averaging step exists to remove, not create. `BoundaryFromWords`
    /// must assign both hull edges directly, with no midpoint arithmetic in
    /// that arm at all.
    #[test]
    fn boundary_from_words_assigns_hull_edges_directly_no_midpoint() {
        let mut chat = parse(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n\
             @ID:\teng|x|CHI|||||Child|||\n\
             *CHI:\thello . \u{15}1000_3000\u{15}\n%wor:\thello \u{15}1000_2950\u{15} .\n\
             *CHI:\tworld . \u{15}2700_4000\u{15}\n%wor:\tworld \u{15}2960_4000\u{15} .\n@End\n",
        );

        let result = repair_bullets(&mut chat, false, EndOverlapPolicy::ClampAllAdjacent);

        assert_eq!(result.stats.boundary_averaged, 1);
        let entries = collect_bullet_entries(&chat);
        assert_eq!(
            entries[0].end_ms, 2950,
            "the earlier utterance's own hull end"
        );
        assert_eq!(
            entries[1].start_ms, 2960,
            "the next utterance's own hull start"
        );
        assert!(
            entries[0].end_ms <= entries[1].start_ms,
            "must not create a fresh overlap: earlier ends {} but later starts {}",
            entries[0].end_ms,
            entries[1].start_ms,
        );
    }

    /// 2026-09-01 review, item 11: a chain of three overlapping same-speaker
    /// pairs (A,B) and (B,C). The two-phase design this replaced computed
    /// BOTH pairs from one snapshot before applying either; this asserts
    /// the live, interleaved sweep produces a globally self-consistent
    /// result instead: every bullet still has a positive duration, and no
    /// pair is left overlapping by more than the resolved amount. B
    /// participates in BOTH pairs (as `later` in the first, `earlier` in
    /// the second), which is exactly the shared state a snapshot-based
    /// design risks reading stale.
    #[test]
    fn chained_overlapping_pairs_resolve_against_live_state() {
        let mut chat = parse(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n\
             @ID:\teng|x|CHI|||||Child|||\n\
             *CHI:\ta . \u{15}1000_3000\u{15}\n%wor:\ta \u{15}1000_2900\u{15} .\n\
             *CHI:\tb . \u{15}2700_4200\u{15}\n%wor:\tb \u{15}2950_4100\u{15} .\n\
             *CHI:\tc . \u{15}4000_5000\u{15}\n%wor:\tc \u{15}4150_4900\u{15} .\n@End\n",
        );

        let result = repair_bullets(&mut chat, false, EndOverlapPolicy::ClampAllAdjacent);

        assert_eq!(
            result.stats.boundary_averaged, 2,
            "both adjacent pairs overlap and are within threshold"
        );
        let entries = collect_bullet_entries(&chat);
        assert_eq!(entries.len(), 3);
        for entry in &entries {
            assert!(
                entry.start_ms < entry.end_ms,
                "every bullet must keep a positive duration: {}..{}",
                entry.start_ms,
                entry.end_ms
            );
        }
        assert!(
            entries[0].end_ms <= entries[1].start_ms,
            "pair (a,b) must not be left overlapping: {} > {}",
            entries[0].end_ms,
            entries[1].start_ms
        );
        assert!(
            entries[1].end_ms <= entries[2].start_ms,
            "pair (b,c) must not be left overlapping: {} > {}",
            entries[1].end_ms,
            entries[2].start_ms
        );
    }

    /// 2026-09-01 review, item 15: the real-data shape, for repair's own
    /// resolver. A small (within-tolerance) same-speaker overlap separated
    /// by an intervening other-speaker utterance must still be resolved by
    /// the per-speaker-stream sweep; the intervening utterance is untouched.
    #[test]
    fn boundary_averaging_resolves_same_speaker_pair_across_an_intervening_speaker() {
        let mut chat = parse(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child, PAR Parent\n\
             @ID:\teng|x|CHI|||||Child|||\n@ID:\teng|x|PAR|||||Parent|||\n\
             *CHI:\tearlier . \u{15}1000_2000\u{15}\n%wor:\tearlier \u{15}1000_1900\u{15} .\n\
             *PAR:\tinterjection . \u{15}1500_1600\u{15}\n\
             *CHI:\tlater . \u{15}1950_3000\u{15}\n%wor:\tlater \u{15}1950_2900\u{15} .\n\
             @End\n",
        );

        let result = repair_bullets(&mut chat, false, EndOverlapPolicy::PreserveCrossSpeaker);

        assert_eq!(result.stats.boundary_averaged, 1);
        let entries = collect_bullet_entries(&chat);
        assert_eq!(
            (entries[1].start_ms, entries[1].end_ms),
            (1500, 1600),
            "the intervening PAR utterance must be untouched"
        );
        assert_eq!(
            entries[0].end_ms, 1900,
            "utterance 0's own measured hull end"
        );
        assert_eq!(
            entries[2].start_ms, 1950,
            "utterance 2's own measured hull start"
        );
    }

    /// 2026-09-01 review, item 16: repair's own boundary-averaging must
    /// carry the SAME file-order successor guard Pass 2 does. CHI's own
    /// speaker stream has no utterance after "later" at all, so a
    /// speaker-stream-only guard would see no successor and wrongly permit
    /// the move; PAR, immediately after "later" in FILE order, starts
    /// before that would-be new start.
    #[test]
    fn boundary_from_words_guard_reads_file_order_successor_in_repair_too() {
        let mut chat = parse(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child, PAR Parent\n\
             @ID:\teng|x|CHI|||||Child|||\n@ID:\teng|x|PAR|||||Parent|||\n\
             *CHI:\tearlier . \u{15}1000_2200\u{15}\n%wor:\tearlier \u{15}1000_2150\u{15} .\n\
             *CHI:\tlater . \u{15}2000_3000\u{15}\n%wor:\tlater \u{15}2150_2900\u{15} .\n\
             *PAR:\tresponse . \u{15}2100_2500\u{15}\n\
             @End\n",
        );

        let result = repair_bullets(&mut chat, false, EndOverlapPolicy::PreserveCrossSpeaker);

        assert_eq!(result.stats.boundary_averaged, 1);
        assert!(
            result.decisions[0].needs_review,
            "a real conflict, not hull-respecting"
        );
        let entries = collect_bullet_entries(&chat);
        assert_eq!(
            entries[1].start_ms, 2000,
            "CHI's later utterance must not be moved to 2150 (past PAR's 2100)"
        );
        assert_eq!(
            entries[0].end_ms, 2000,
            "CHI's earlier utterance is clamped to the later utterance's ORIGINAL start instead"
        );
        assert!(
            entries[0].start_ms <= entries[0].end_ms && entries[1].start_ms <= entries[2].start_ms,
            "file order stays monotonic"
        );
    }

    /// 2026-09-01 review, item 7: a same-speaker overlap whose measured word
    /// hulls DO overlap is the same conflict Pass 2 calls
    /// `InterleavedWords`: the earlier bullet and its words past the bound
    /// are clamped together, and the later utterance is not moved.
    #[test]
    fn same_speaker_overlap_with_overlapping_hulls_clamps_bullet_and_words_together() {
        let mut chat = parse(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n\
             @ID:\teng|x|CHI|||||Child|||\n\
             *CHI:\thello . \u{15}1000_3000\u{15}\n%wor:\thello \u{15}1000_2950\u{15} .\n\
             *CHI:\tworld . \u{15}2700_4000\u{15}\n%wor:\tworld \u{15}2850_4000\u{15} .\n@End\n",
        );

        let result = repair_bullets(&mut chat, false, EndOverlapPolicy::ClampAllAdjacent);

        assert_eq!(result.stats.boundary_averaged, 1);
        assert!(result.decisions[0].needs_review, "a genuine word conflict");
        let entries = collect_bullet_entries(&chat);
        assert_eq!(
            entries[0].end_ms, 2700,
            "clamped to the later utterance's own original start"
        );
        assert_eq!(
            get_utterance_mut(&mut chat, entries[0].line_idx)
                .as_deref()
                .and_then(furthest_word_timing_end),
            Some(2700),
            "the earlier word was clamped down, not left at 2950"
        );
        assert_eq!(
            entries[1].start_ms, 2700,
            "the later utterance is not moved"
        );
        assert_eq!(
            get_utterance_mut(&mut chat, entries[1].line_idx)
                .as_deref()
                .and_then(earliest_word_timing_start),
            Some(2850),
            "the later word is untouched"
        );
    }

    /// 2026-09-02 review: repair's `InterleavedWords` resolution goes
    /// through the same `clamp_words_past_bound` Pass 2 uses, so it must
    /// carry the same per-word records forward into `RepairDecision`
    /// rather than reducing them to `words_clamped.dropped.len()` in the
    /// reason string and nothing else. `third` (4100-4300) sits wholly past
    /// the clamp bound (4000: the later utterance's own start); `world`
    /// (3000-4200) straddles it and is merely trimmed.
    #[test]
    fn interleaved_words_repair_decision_carries_the_dropped_word_record() {
        use crate::chat_ops::fa::TimeSpan;
        use crate::chat_ops::fa::orchestrate::WordTier;

        // Overlap must stay within `BOUNDARY_AVERAGING_THRESHOLD_MS` (500ms)
        // or `resolve_boundary_average_pair` skips the pair entirely; the
        // earlier bullet's end (4400) is 400ms past the later bullet's
        // start (4000).
        let mut chat = parse(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n\
             @ID:\teng|x|CHI|||||Child|||\n\
             *CHI:\thello world third . \u{15}1000_4400\u{15}\n\
             %wor:\thello \u{15}1000_3000\u{15} world \u{15}3000_4200\u{15} third \u{15}4100_4300\u{15} .\n\
             *CHI:\tnext . \u{15}4000_6000\u{15}\n%wor:\tnext \u{15}4000_6000\u{15} .\n@End\n",
        );

        let result = repair_bullets(&mut chat, false, EndOverlapPolicy::ClampAllAdjacent);

        assert_eq!(result.stats.boundary_averaged, 1);
        assert_eq!(result.decisions.len(), 1);
        let decision = &result.decisions[0];
        assert!(decision.needs_review, "a genuine word conflict");
        assert!(
            decision.reason.contains("words_trimmed=1"),
            "reason stays a summary: {}",
            decision.reason
        );
        assert!(
            decision.reason.contains("words_dropped=1"),
            "reason stays a summary: {}",
            decision.reason
        );

        // The point of this test: the dropped side is not just a count in
        // the reason string. `RepairDecision` itself must carry which word
        // and what was lost.
        assert_eq!(decision.words_dropped.len(), 1);
        let dropped = &decision.words_dropped[0];
        assert_eq!(
            dropped.word_index, 2,
            "third is the third %wor word, index 2 (0-based)"
        );
        assert_eq!(dropped.tier, WordTier::Wor);
        assert_eq!(
            dropped.measured,
            TimeSpan::new(4_100, 4_300),
            "the measured extent that was lost must survive to the RepairDecision"
        );
    }

    #[test]
    fn test_lis_simple() {
        let values = vec![3, 1, 2, 4, 3, 5];
        let lis = longest_increasing_subsequence(&values);
        // LIS: 1, 2, 3, 5 (indices 1, 2, 4, 5) or 1, 2, 4, 5 (indices 1, 2, 3, 5)
        assert_eq!(lis.len(), 4);
        // Verify it's actually increasing.
        for pair in lis.windows(2) {
            assert!(values[pair[0]] <= values[pair[1]]);
        }
    }

    #[test]
    fn test_lis_already_sorted() {
        let values = vec![1, 2, 3, 4, 5];
        let lis = longest_increasing_subsequence(&values);
        assert_eq!(lis.len(), 5);
    }

    #[test]
    fn test_lis_reverse_sorted() {
        let values = vec![5, 4, 3, 2, 1];
        let lis = longest_increasing_subsequence(&values);
        assert_eq!(lis.len(), 1);
    }

    #[test]
    fn test_lis_empty() {
        let values: Vec<u64> = vec![];
        let lis = longest_increasing_subsequence(&values);
        assert!(lis.is_empty());
    }
}
