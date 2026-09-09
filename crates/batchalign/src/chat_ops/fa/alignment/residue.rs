//! The FALLBACK half of label attribution: what happens to the residue of a
//! broken stitch.
//!
//! [`super::token_map`] owns the exact in-order stitch and everything both
//! routes share (the slots, the label track, the spans). This module owns the
//! one thing that is a GUESS: when the stitch cannot identify a word,
//! everything from that word onwards goes through a character-level Hirschberg
//! alignment, and the labels are handed out by how well the characters fitted.
//!
//! Split out because the two halves answer different questions and fail in
//! different ways. The stitch either spells a word or does not; the remap can
//! be over budget, can contradict its own monotonicity assumption, and can
//! place a word on evidence weak enough that the resulting number must be
//! marked as our own answer rather than the engine's. Keeping them apart keeps
//! the exact path readable next to the approximate one.

use std::collections::BTreeMap;

use batchalign_transform::dp_align::{self, AlignResult, MatchMode};

use crate::chat_ops::fa::origin::CharEdits;

use super::token_map::{
    Attribution, LabelRun, LabelSlot, LabelTrack, UntimedReason, WordSlot, WordTrack,
};

/// The most characters, on either side, the residue alignment will consider.
///
/// # Why a budget exists at all
///
/// The character DP is O(transcript chars x label chars). Grouping caps a
/// MERGE at [`crate::chat_ops::fa::MAX_GROUP_LABEL_BYTES`], but that bounds
/// merging only: a single utterance whose own labels exceed the cap becomes
/// its own group and is never split, so nothing upstream bounds one group's
/// length. And the residue is whatever follows the first stitch failure, so a
/// group that fails on word 1 puts almost the whole group through the DP.
///
/// # Why this number
///
/// 4096 squared is about 17 million character comparisons, which is the most
/// arithmetic worth spending to attribute ONE group's labels; Hirschberg keeps
/// the memory linear, so time is the only thing being bounded.
///
/// It is a STANDALONE ceiling on this DP, not a figure derived from anything
/// upstream. Until 2026-09-07 this paragraph said it was roughly nine times
/// [`crate::chat_ops::fa::MAX_GROUP_LABEL_BYTES`] and concluded that "no group
/// that grouping actually built can reach it". That does not follow, and both
/// halves of it are wrong. The cap bounds the TRANSCRIPT BYTES a merge may
/// accumulate; the two streams measured here are transcript CHARACTERS on one
/// side and the ENGINE'S LABEL characters on the other, which no cap touches
/// at all, and the budget is checked against each side separately rather than
/// against their sum. Which inputs can reach this number is a question the
/// grouping cap does not answer, so the refusal below is a live path rather
/// than a formality.
///
/// Above it the residue is REFUSED, not truncated: aligning a prefix and
/// letting the rest fall off would attribute labels from a comparison that
/// never saw the words they belong to.
pub(crate) const MAX_RESIDUE_ALIGN_CHARS: usize = 4_096;
/// Give the residue's labels to the residue's words by character edit distance.
///
/// # The algorithm, and the two rules that keep it honest
///
/// The residue words' normalized characters are concatenated into one stream,
/// and so are the residue labels'. [`dp_align::align_chars`] aligns the two,
/// and because that alignment is monotone, the labels each word matched into
/// form a contiguous, non-overlapping run.
///
/// 1. **A label goes to ONE word**, the one that matched the most of its
///    characters (earliest word on a tie). A label is a single measured
///    interval; handing it to two words means splitting it, and the split point
///    would be a millisecond nobody observed.
/// 2. **A word that matched nothing may still be BRACKETED.** `1995` shares no
///    character with "nineteen ninety five", so no alignment can anchor it; but
///    its neighbours were placed, and the labels strictly between their claims
///    are the numeral's span. This only fires when exactly one unplaced word
///    sits in the gap, because two unplaced words in one gap have nothing to
///    say which label starts the second.
pub(crate) fn remap_residue(
    words: &WordTrack,
    labels: &LabelTrack,
    first_word: WordSlot,
    first_label: LabelSlot,
    attributions: &mut [Option<Attribution>],
) -> Result<(), UntimedReason> {
    // Character streams, each remembering which word or label it came from.
    let mut word_chars: Vec<char> = Vec::new();
    let mut word_of_char: Vec<WordSlot> = Vec::new();
    for slot in words.slots_from(first_word) {
        for ch in words.norm(slot).chars() {
            word_chars.push(ch);
            word_of_char.push(slot);
        }
    }
    let mut label_chars: Vec<char> = Vec::new();
    let mut label_of_char: Vec<LabelSlot> = Vec::new();
    for slot in first_label.0..labels.len() {
        for ch in labels.norm(LabelSlot(slot)).chars() {
            label_chars.push(ch);
            label_of_char.push(LabelSlot(slot));
        }
    }
    if word_chars.is_empty() || label_chars.is_empty() {
        return Ok(());
    }
    // The budget, checked BEFORE the alignment rather than after: the cost this
    // bounds is the alignment itself.
    if word_chars.len() > MAX_RESIDUE_ALIGN_CHARS || label_chars.len() > MAX_RESIDUE_ALIGN_CHARS {
        return Err(UntimedReason::ResidueTooLongToAlign {
            transcript_chars: word_chars.len(),
            label_chars: label_chars.len(),
            budget: MAX_RESIDUE_ALIGN_CHARS,
        });
    }

    // Exact matching: both streams are already lower-cased and stripped to
    // alphanumerics, so a case-insensitive mode would only cost time, and a
    // fuzzy mode is meaningless on single characters.
    let alignment = dp_align::align_chars(&word_chars, &label_chars, MatchMode::Exact);

    // How many characters each (word, label) pair shares, and what neither side
    // could account for.
    let mut shared: Vec<BTreeMap<WordSlot, usize>> = vec![BTreeMap::new(); labels.len()];
    let mut unmatched_label_chars: Vec<usize> = vec![0; labels.len()];
    let mut edits: BTreeMap<WordSlot, CharEdits> = BTreeMap::new();
    for item in &alignment {
        match item {
            AlignResult::Match {
                payload_idx,
                reference_idx,
                ..
            } => {
                let word = word_of_char[*payload_idx];
                let label = label_of_char[*reference_idx];
                *shared[label.0].entry(word).or_default() += 1;
            }
            AlignResult::ExtraPayload { payload_idx, .. } => {
                edits
                    .entry(word_of_char[*payload_idx])
                    .or_insert(CharEdits::ZERO)
                    .transcript_only += 1;
            }
            AlignResult::ExtraReference { reference_idx, .. } => {
                unmatched_label_chars[label_of_char[*reference_idx].0] += 1;
            }
        }
    }

    // Rule 1: one owner per label. `BTreeMap` iterates ascending and the
    // comparison is strict, so the earliest word wins a tie.
    let mut owner: Vec<Option<WordSlot>> = vec![None; labels.len()];
    for slot in first_label.0..labels.len() {
        let mut best: Option<(WordSlot, usize)> = None;
        for (word, count) in &shared[slot] {
            if best.is_none_or(|(_, best_count)| *count > best_count) {
                best = Some((*word, *count));
            }
        }
        owner[slot] = best.map(|(word, _)| word);
    }

    // Contiguity and non-overlap follow from the alignment being monotone: a
    // later word's characters can only match later label characters. That used
    // to be asserted in prose here, which is exactly the shape this codebase
    // treats as a bug with documentation, so it is CHECKED instead. If it ever
    // fails, the claims below could interleave and two words would be given
    // overlapping spans, so the residue is refused rather than timed from a
    // contradiction.
    if !owners_are_monotone(&owner) {
        return Err(UntimedReason::NonMonotoneAttribution);
    }

    // Each word's claim is the first and last label it owns.
    let mut claim: BTreeMap<WordSlot, (LabelSlot, LabelSlot)> = BTreeMap::new();
    for slot in first_label.0..labels.len() {
        let Some(word) = owner[slot] else {
            continue;
        };
        let label = LabelSlot(slot);
        claim
            .entry(word)
            .and_modify(|(_, last)| *last = label)
            .or_insert((label, label));
        edits.entry(word).or_insert(CharEdits::ZERO).label_only += unmatched_label_chars[slot];
    }

    // Rule 2: an unplaced word BRACKETED ON BOTH SIDES takes the gap between
    // the two claims.
    //
    // Both sides, and that is the whole of the rule. An open side means nothing
    // has settled where this word begins (or stops), so the labels beyond that
    // edge are as likely to be the engine's own preamble as this word's speech:
    // Whisper hallucinates "thanks for watching" over silence, and giving those
    // three labels to word 0 put 6.2 seconds of nothing into a transcript as if
    // measured. The trailing case was the same fabrication wearing the 500 ms
    // fallback, which made it look plausible instead of absurd.
    let mut cursor = Some(first_word);
    while let Some(word) = cursor {
        if claim.contains_key(&word) {
            cursor = words.after(word);
            continue;
        }
        // The first placed word at or after this one, or `None` when the
        // residue runs to the end of the track.
        let mut end = Some(word);
        while let Some(slot) = end {
            if claim.contains_key(&slot) {
                break;
            }
            end = words.after(slot);
        }
        // The left bracket is the label just after the previous word's claim.
        // The STITCHED PREFIX counts as a claim, and is the stronger one: it
        // proved its labels spell its words and consumed exactly
        // `0..first_label`. But only when there IS a prefix; an empty one
        // brackets nothing.
        let left = match word == first_word {
            true => words.before(first_word).map(|_| first_label.0),
            false => match words.before(word) {
                Some(previous) => claim.get(&previous).map(|(_, last)| last.0 + 1),
                None => None,
            },
        };
        // The right bracket is the first label the next placed word took. A
        // residue that runs to the end of the word list has none.
        let right = end
            .and_then(|slot| claim.get(&slot))
            .map(|(first, _)| first.0);
        if words.after(word) == end
            && let (Some(left), Some(right)) = (left, right)
            && left < right
            && let Some(run) = LabelRun::through(LabelSlot(left), LabelSlot(right - 1))
        {
            // None of these labels matched anything, so every one of their
            // characters is unaccounted for. Saying so is what separates "the
            // DP recognised this word one boundary over" from "the DP
            // recognised nothing and the neighbours boxed it in".
            let unaccounted: usize = unmatched_label_chars[left..right].iter().sum();
            edits.entry(word).or_insert(CharEdits::ZERO).label_only += unaccounted;
            claim.insert(word, (run.first, run.last));
        }
        cursor = end;
    }

    for (slot, (first, last)) in claim {
        let Some(run) = LabelRun::through(first, last) else {
            continue;
        };
        attributions[slot.0] = Some(Attribution {
            run,
            // A word absent from the tally reconciled perfectly: zero is what
            // the DP actually counted for it, not a value invented here, and
            // it says the fold is as exact as a stitch.
            edits: match edits.get(&slot) {
                Some(counted) => *counted,
                None => CharEdits::ZERO,
            },
        });
    }

    Ok(())
}

/// Whether label ownership runs FORWARD through the group.
///
/// The property `remap_residue` rests on, extracted so it can be checked and
/// tested rather than asserted in a comment: reading the owners in label order,
/// the word they name never goes backwards. A monotone character alignment
/// cannot produce anything else, which is precisely why a violation means an
/// assumption has stopped holding and the residue must not be timed.
///
/// Unowned labels are skipped: a gap in the middle of a word's run is fine
/// (the engine emitted something neither side matched), and only the ORDER of
/// the owners that do exist carries the property.
pub(crate) fn owners_are_monotone(owner: &[Option<WordSlot>]) -> bool {
    let mut previous: Option<WordSlot> = None;
    for word in owner.iter().flatten() {
        if let Some(previous) = previous
            && *word < previous
        {
            return false;
        }
        previous = Some(*word);
    }
    true
}
