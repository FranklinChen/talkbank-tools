//! Utterance segmentation helpers.
//!
//! Splits a single utterance into multiple utterances based on word-level
//! assignments from a segmentation callback.
//!
//! Also provides types and functions for the server-side utseg orchestrator:
//! payload collection, cache key computation, and result application.
//!
//! ## Outcome model
//!
//! Every utterance visited by `collect_utseg_payloads` + `apply_utseg_results`
//! produces exactly one [`UtsegOutcome`]. This is the sibling-task analog of
//! morphotag's [`MorOutcome`](crate::morphosyntax::outcome::MorOutcome) and
//! serves the same architectural purpose: making correct-by-design behavior
//! (e.g. single-word utterances that trivially need no segmentation) visible
//! as a typed `NotApplicable` outcome rather than invisible silent skip, and
//! making worker-response shape mismatches fail loudly as typed
//! `MisalignmentBug` diagnostics rather than being absorbed by defensive
//! index guards. Chatter owns structural partition admission and rebuilding.
//!
//! The utseg invariant is simpler than morphotag's: there is no tokenizer
//! realignment stage: the Python utseg worker is a per-word classifier
//! whose `assignments` return MUST have the same length as the input
//! `words`. A mismatch is always a worker-contract bug, not an expected
//! divergence class.

// Wildcard matches over closed enums are denied in this file, following
// chatter's per-file ratchet. The generic tier policy now belongs to Chatter's
// source-bound split API rather than a second BA3-specific owner.
#![deny(clippy::wildcard_enum_match_arm)]
// Test code is exempt, matching this crate's existing treatment of the panic
// lints: `other => panic!("unexpected {other:?}")` is how a test says a variant
// should be unreachable, and denying it there would push tests toward asserting
// less rather than more.
#![cfg_attr(test, allow(clippy::wildcard_enum_match_arm))]

use crate::decisions::LineIdx;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use talkbank_model::alignment::helpers::PositionalDomain;
use talkbank_model::model::ChatFileLines;
use talkbank_model::model::{ChatFile, Line};
#[cfg(test)]
use talkbank_model::model::{Retrace, Utterance, UtteranceContent};

use crate::extract;
use talkbank_model::SpeakerCode;

// ---------------------------------------------------------------------------
// Wire types (match Python's UtsegBatchItem / UtsegResponse)
// ---------------------------------------------------------------------------

/// Input payload for a single utterance segmentation request.
///
/// Matches the Python `UtsegBatchItem` Pydantic model.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct UtsegBatchItem {
    /// Tokenized words from the utterance.
    pub words: Vec<String>,
    /// Full utterance text (for constituency parsing).
    pub text: String,
}

/// Response from utterance segmentation inference.
///
/// Each element in `assignments` is a 0-based utterance group ID, parallel
/// to the `words` in the corresponding `UtsegBatchItem`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtsegResponse {
    /// 0-based utterance group ID per word, parallel to `UtsegBatchItem::words`.
    pub assignments: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Typed outcome model (Wave 5 of the morphotag reconciliation architecture)
// ---------------------------------------------------------------------------

/// One utterance segmentation outcome.
///
/// Carries `utt_ordinal` and `speaker` so it can be converted to a
/// [`DecisionRecord`](crate::decisions::DecisionRecord) without further
/// context. `line_idx` is also available when needed, utseg is indexed
/// by `utt_ordinal` rather than `line_idx` to align with the existing
/// `HashMap<utt_ordinal, assignments>` dispatch map, but the two are
/// trivially interconvertible.
#[derive(Debug, Clone)]
pub struct UtsegOutcome {
    /// 0-based index of the utterance among all `Utterance` lines in the file.
    pub utt_ordinal: usize,
    /// Speaker code for the affected utterance.
    pub speaker: SpeakerCode,
    /// What happened on this utterance.
    pub kind: UtsegOutcomeKind,
}

/// The three possible utseg outcomes per utterance.
///
/// Structurally parallel to
/// [`MorOutcomeKind`](crate::morphosyntax::outcome::MorOutcomeKind); see
/// the morphotag invariants architecture doc for rationale.
#[derive(Debug, Clone)]
pub enum UtsegOutcomeKind {
    /// The utterance did not require segmentation.
    ///
    /// Most commonly this is a single-word utterance, a one-word
    /// utterance trivially occupies one segment, so utseg skips the
    /// worker call entirely. It is CORRECT behavior, not a silent skip.
    NotApplicable {
        /// Why this utterance was not dispatched.
        reason: UtsegNotApplicableReason,
    },
    /// Worker returned exactly N assignments for N input words. This checks
    /// response shape only; source-bound structural admission is still required.
    Aligned {
        /// The agreed word count on both sides.
        n_words: usize,
        /// Number of segments the utterance was split into.
        /// `1` means "all words assigned the same group" (no split).
        n_segments: usize,
    },
    /// Worker returned a response whose `assignments` length does not
    /// match the dispatched `words` length. This is always a
    /// worker-contract bug: the Python classifier is supposed to emit
    /// one assignment per input word.
    MisalignmentBug(UtsegMisalignmentDiagnostic),
}

/// Why an utterance was not dispatched to the utseg worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UtsegNotApplicableReason {
    /// The utterance had a single alignable word. Segmentation into one
    /// segment is trivial; the worker call is skipped for efficiency.
    SingleWord,
    /// The utterance had zero alignable words (filler-only, empty, etc.).
    /// Nothing to segment.
    Empty,
}

impl UtsegNotApplicableReason {
    /// Stable label for tracing and structured decision evidence.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SingleWord => "single_word",
            Self::Empty => "empty",
        }
    }
}

/// Diagnostic for an utseg misalignment bug, the worker did not return
/// the contract-required number of assignments.
#[derive(Debug, Clone)]
pub struct UtsegMisalignmentDiagnostic {
    /// Number of words sent to the worker.
    pub expected_assignments: usize,
    /// Number of assignments the worker actually returned.
    pub actual_assignments: usize,
    /// The words that were sent, helps a developer reproduce the case.
    pub words: Vec<String>,
}

impl UtsegOutcome {
    /// Convert into a [`DecisionRecord`](crate::decisions::DecisionRecord)
    /// for tracing and structured evidence. Aligned outcomes return `None`:
    /// happy-path utterances should not flood the reporting surface.
    pub fn to_decision_record(&self, line_idx: usize) -> Option<crate::decisions::DecisionRecord> {
        use crate::decisions::{DecisionRecord, DecisionStrategy, UtsegStrategy};
        match &self.kind {
            UtsegOutcomeKind::Aligned { .. } => None,
            UtsegOutcomeKind::NotApplicable { reason } => Some(DecisionRecord {
                line_idx: LineIdx::new(line_idx),
                speaker: self.speaker.as_str().to_string(),
                strategy: DecisionStrategy::Utseg(UtsegStrategy::NotApplicable),
                reason: format!("reason={}", reason.as_str()),
                needs_review: false,
            }),
            UtsegOutcomeKind::MisalignmentBug(diag) => Some(DecisionRecord {
                line_idx: LineIdx::new(line_idx),
                speaker: self.speaker.as_str().to_string(),
                strategy: DecisionStrategy::Utseg(UtsegStrategy::MisalignmentBug),
                reason: format!(
                    "expected_assignments={} actual_assignments={} words={:?}",
                    diag.expected_assignments, diag.actual_assignments, diag.words,
                ),
                needs_review: true,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Payload collection
// ---------------------------------------------------------------------------

/// Result of [`collect_utseg_payloads`]: batch items to dispatch plus
/// typed outcomes for every utterance that was not dispatched.
///
/// Mirrors the shape of
/// [`PayloadCollection`](crate::morphosyntax::payloads::PayloadCollection)
/// from Wave 1. Utterances fall into one of two mutually-exclusive sets:
/// `batch_items` (will be sent to the worker) and `not_applicable`
/// (will not be dispatched; correct).
pub struct UtsegPayloadCollection {
    /// Utterances that will be sent to the utseg worker.
    pub batch_items: Vec<(usize, UtsegBatchItem)>,
    /// Utterances that were classified as NotApplicable and not dispatched,
    /// each carrying a structured reason.
    pub not_applicable: Vec<UtsegOutcome>,
}

/// Collect utseg payloads from all multi-word utterances in a ChatFile.
///
/// Single-word and empty utterances are classified as
/// [`UtsegNotApplicableReason::SingleWord`] / `Empty` and returned as
/// [`UtsegOutcome::NotApplicable`] entries, no worker call, no silent
/// skip.
pub fn collect_utseg_payloads(chat_file: &ChatFile) -> UtsegPayloadCollection {
    let mut batch_items = Vec::new();
    let mut not_applicable = Vec::new();
    let mut utt_idx = 0usize;

    for line in chat_file.lines.iter() {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        let mut words = Vec::new();
        extract::collect_utterance_content(
            &utt.main.content.content,
            PositionalDomain::Mor,
            &mut words,
        );

        let speaker = SpeakerCode::new(utt.main.speaker.as_str());
        match words.len() {
            0 => {
                not_applicable.push(UtsegOutcome {
                    utt_ordinal: utt_idx,
                    speaker,
                    kind: UtsegOutcomeKind::NotApplicable {
                        reason: UtsegNotApplicableReason::Empty,
                    },
                });
            }
            1 => {
                not_applicable.push(UtsegOutcome {
                    utt_ordinal: utt_idx,
                    speaker,
                    kind: UtsegOutcomeKind::NotApplicable {
                        reason: UtsegNotApplicableReason::SingleWord,
                    },
                });
            }
            _ => {
                // Single pass: build both `text` (space-joined) and `word_texts` together
                let mut text = String::new();
                let mut word_texts = Vec::with_capacity(words.len());
                for (i, w) in words.iter().enumerate() {
                    if i > 0 {
                        text.push(' ');
                    }
                    let s = w.text.as_str();
                    text.push_str(s);
                    word_texts.push(s.to_string());
                }

                batch_items.push((
                    utt_idx,
                    UtsegBatchItem {
                        words: word_texts,
                        text,
                    },
                ));
            }
        }

        utt_idx += 1;
    }

    UtsegPayloadCollection {
        batch_items,
        not_applicable,
    }
}

/// Validate one utseg worker response against the dispatched batch item.
///
/// Returns the classified outcome kind. [`UtsegOutcomeKind::MisalignmentBug`]
/// is emitted when the worker's `assignments` vector has a different
/// length than the dispatched `words`, always a worker-contract bug.
pub fn validate_utseg_response(
    batch_item: &UtsegBatchItem,
    response: &UtsegResponse,
) -> UtsegOutcomeKind {
    let expected = batch_item.words.len();
    let actual = response.assignments.len();
    if expected != actual {
        return UtsegOutcomeKind::MisalignmentBug(UtsegMisalignmentDiagnostic {
            expected_assignments: expected,
            actual_assignments: actual,
            words: batch_item.words.clone(),
        });
    }
    // Count distinct segment IDs in the assignments to report n_segments.
    let mut distinct = std::collections::BTreeSet::new();
    for &a in &response.assignments {
        distinct.insert(a);
    }
    UtsegOutcomeKind::Aligned {
        n_words: expected,
        n_segments: distinct.len(),
    }
}

// ---------------------------------------------------------------------------
// Result application
// ---------------------------------------------------------------------------

/// A dependent tier invalidated while applying an admitted utterance partition.
#[derive(Debug, Clone)]
pub struct UtsegTierInvalidation {
    utterance_ordinal: usize,
    tier_index: usize,
    tier_kind: String,
    reason: talkbank_transform::utterance_split::TierInvalidationReason,
}

impl UtsegTierInvalidation {
    /// Original utterance ordinal, before any split.
    pub const fn utterance_ordinal(&self) -> usize {
        self.utterance_ordinal
    }
    /// Position of the original dependent tier.
    pub const fn tier_index(&self) -> usize {
        self.tier_index
    }
    /// Label derived from the original typed tier.
    pub fn tier_kind(&self) -> &str {
        &self.tier_kind
    }
    /// Why the shared partition owner refused reuse.
    pub const fn reason(&self) -> talkbank_transform::utterance_split::TierInvalidationReason {
        self.reason
    }
}

/// What applying a segmentation did to a file: the dependent tiers its
/// splits could not carry, and where each input utterance went.
#[derive(Debug)]
pub struct UtsegApplied {
    /// Dependent tiers a split invalidated, in input order.
    pub invalidated: Vec<UtsegTierInvalidation>,
    /// Where each input utterance went.
    pub layout: SegmentationLayout,
}

/// Where each utterance of a segmented file went: for each input utterance,
/// in order, the output position of its first child and how many children it
/// has (one when it was left whole).
///
/// Built only by [`apply_utseg_results`], from the splits it performed, so it
/// describes exactly the file that function returned. A judgement of the
/// output that must find the utterances segmentation was not asked to touch
/// reads their output positions here, rather than inferring them from the
/// output's content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentationLayout {
    /// Per input utterance: (output position of its first child, children).
    placements: Vec<(usize, usize)>,
}

impl SegmentationLayout {
    /// The output position of input utterance `input` when it is still one
    /// utterance; `None` when it was split or the file had no such
    /// utterance.
    pub fn whole_output_ordinal(&self, input: usize) -> Option<usize> {
        match self.placements.get(input) {
            Some(&(first, 1)) => Some(first),
            Some(_) | None => None,
        }
    }

    /// How many utterances the input file had.
    pub fn input_utterances(&self) -> usize {
        self.placements.len()
    }
}

/// A proposed segmentation that cannot be applied without changing source structure.
#[derive(Debug, thiserror::Error)]
pub enum UtsegApplyRefusal {
    /// A source-bound split could not preserve this utterance.
    #[error("utterance {utterance_ordinal} partition refused: {source}")]
    Partition {
        /// The original source ordinal.
        utterance_ordinal: usize,
        /// The shared CHAT partition owner's typed refusal.
        #[source]
        source: talkbank_transform::utterance_split::SplitRefusal,
    },
    /// The supplied map refers to an utterance not present in the source.
    #[error("segmentation references absent utterance {0}")]
    UnknownUtterance(usize),
}

/// Apply all proposed splits atomically, retaining explicit tier invalidations
/// and the layout of the result.
///
/// No source line changes until every selected utterance has admitted its own
/// source-bound partition. A refusal therefore cannot leave a partially split
/// file for the output path to mistake for successful processing.
pub fn apply_utseg_results(
    chat_file: &mut ChatFile,
    assignment_map: &HashMap<usize, Vec<usize>>,
) -> Result<UtsegApplied, UtsegApplyRefusal> {
    let mut replacements = HashMap::new();
    let mut invalidated = Vec::new();
    // Each input utterance's first output position and child count, in
    // order: a split's children replace its line in place, so positions
    // are the running sum of the counts before it.
    let mut placements = Vec::new();
    let mut next_output = 0usize;
    let mut ordinal = 0usize;
    for (line_index, line) in chat_file.lines.iter().enumerate() {
        let Line::Utterance(source) = line else {
            continue;
        };
        if let Some(assignments) = assignment_map.get(&ordinal) {
            let outcome = talkbank_transform::utterance_split::UtteranceSplitPlan::for_morphology(
                source,
                assignments,
            )
            .map_err(|source| UtsegApplyRefusal::Partition {
                utterance_ordinal: ordinal,
                source,
            })?
            .execute();
            match outcome {
                // One child: the source stands exactly as it is, with every
                // dependent tier, so its line is not replaced.
                talkbank_transform::utterance_split::SplitOutcome::Unchanged => {
                    placements.push((next_output, 1));
                    next_output += 1;
                }
                talkbank_transform::utterance_split::SplitOutcome::Split(split) => {
                    let (children, losses) = split.into_parts();
                    invalidated.extend(losses.into_iter().map(|loss| UtsegTierInvalidation {
                        utterance_ordinal: ordinal,
                        tier_index: loss.index(),
                        tier_kind: loss.tier().kind().to_owned(),
                        reason: loss.reason(),
                    }));
                    placements.push((next_output, children.len()));
                    next_output += children.len();
                    replacements.insert(line_index, children);
                }
            }
        } else {
            placements.push((next_output, 1));
            next_output += 1;
        }
        ordinal += 1;
    }
    let layout = SegmentationLayout { placements };
    if let Some(&missing) = assignment_map
        .keys()
        .filter(|&&index| index >= ordinal)
        .min()
    {
        return Err(UtsegApplyRefusal::UnknownUtterance(missing));
    }
    if replacements.is_empty() {
        return Ok(UtsegApplied {
            invalidated,
            layout,
        });
    }
    let old_lines = chat_file.lines.take();
    let mut new_lines = Vec::with_capacity(old_lines.len());
    for (index, line) in old_lines.into_iter().enumerate() {
        match replacements.remove(&index) {
            Some(children) => new_lines.extend(
                children
                    .into_iter()
                    .map(|child| Line::Utterance(Box::new(child))),
            ),
            None => new_lines.push(line),
        }
    }
    chat_file.lines = ChatFileLines::new(new_lines);
    Ok(UtsegApplied {
        invalidated,
        layout,
    })
}

/// The shared CHAT owner also provides the morphology-domain content projection.
pub use talkbank_transform::utterance_split::build_word_to_content_map;

#[cfg(test)]
fn split_utterance(utterance: Utterance, assignments: &[usize]) -> Vec<Utterance> {
    let children = match talkbank_transform::utterance_split::UtteranceSplitPlan::for_morphology(
        &utterance,
        assignments,
    )
    .expect("admitted test partition")
    .execute()
    {
        talkbank_transform::utterance_split::SplitOutcome::Unchanged => None,
        talkbank_transform::utterance_split::SplitOutcome::Split(split) => {
            Some(split.into_parts().0)
        }
    };
    children.unwrap_or_else(|| vec![utterance])
}

#[cfg(test)]
fn as_retrace(item: &UtteranceContent) -> Option<&Retrace> {
    match item {
        UtteranceContent::Retrace(retrace) => Some(retrace),
        UtteranceContent::AnnotatedRetrace(annotated) => Some(&annotated.inner),
        UtteranceContent::Word(_)
        | UtteranceContent::AnnotatedWord(_)
        | UtteranceContent::ReplacedWord(_)
        | UtteranceContent::Event(_)
        | UtteranceContent::AnnotatedEvent(_)
        | UtteranceContent::Pause(_)
        | UtteranceContent::Group(_)
        | UtteranceContent::AnnotatedGroup(_)
        | UtteranceContent::Quotation(_)
        | UtteranceContent::AnnotatedQuotation(_)
        | UtteranceContent::PhoGroup(_)
        | UtteranceContent::SinGroup(_)
        | UtteranceContent::Action(_)
        | UtteranceContent::AnnotatedAction(_)
        | UtteranceContent::Freecode(_)
        | UtteranceContent::Separator(_)
        | UtteranceContent::OverlapPoint(_)
        | UtteranceContent::InternalBullet(_)
        | UtteranceContent::LongFeatureBegin(_)
        | UtteranceContent::LongFeatureEnd(_)
        | UtteranceContent::UnderlineBegin(_)
        | UtteranceContent::UnderlineEnd(_)
        | UtteranceContent::NonvocalBegin(_)
        | UtteranceContent::NonvocalEnd(_)
        | UtteranceContent::NonvocalSimple(_)
        | UtteranceContent::OtherSpokenEvent(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use talkbank_model::model::{DependentTier, Terminator, UtteranceContent, WorTier, WriteChat};
    use talkbank_parser::TreeSitterParser;

    fn parse_chat(text: &str) -> ChatFile {
        let parser = TreeSitterParser::new().unwrap();
        parser.parse_chat_file(text).expect_built()
    }

    fn get_utterance(chat: &ChatFile, idx: usize) -> &Utterance {
        let mut utt_idx = 0;
        for line in chat.lines.as_slice() {
            if let Line::Utterance(utt) = line {
                if utt_idx == idx {
                    return utt.as_ref();
                }
                utt_idx += 1;
            }
        }
        panic!("Utterance {idx} not found");
    }

    fn count_utterances(chat: &ChatFile) -> usize {
        chat.lines
            .iter()
            .filter(|l| matches!(l, Line::Utterance(_)))
            .count()
    }

    #[test]
    fn test_split_no_change() {
        let chat_text = include_str!("../../../test-fixtures/eng_i_eat_cookies.cha");
        let chat = parse_chat(chat_text);
        let utt = get_utterance(&chat, 0).clone();
        let assignments = vec![0; build_word_to_content_map(&utt.main.content.content).len()];
        let result = split_utterance(utt, &assignments);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_split_two_groups() {
        let chat_text =
            include_str!("../../../test-fixtures/eng_i_eat_cookies_and_he_likes_cake.cha");
        let chat = parse_chat(chat_text);
        let utt = get_utterance(&chat, 0).clone();
        let result = split_utterance(utt, &[0, 0, 0, 1, 1, 1, 1]);
        assert_eq!(result.len(), 2);

        let out0 = result[0].to_chat_string();
        let out1 = result[1].to_chat_string();
        assert!(out0.contains("I eat cookies"), "First split: {out0}");
        assert!(out1.contains("and he likes cake"), "Second split: {out1}");
    }

    /// True if a segment ENDS with a retrace, which means the split separated
    /// the retrace from the material it points at.
    ///
    /// A question about the MODEL, not about rendered text. This used to scan
    /// the segment's CHAT string for a marker followed by a terminator, and any
    /// annotation sitting between the two hid the dangle:
    /// `now <the red> [/] [* p:w] .` is stranded and the scan called it
    /// clean. So when chatter v0.10.0 introduced `AnnotatedRetrace` and the
    /// pre-assignment stopped matching it, the regression was invisible to the
    /// very test written to catch it.
    ///
    /// Asks `as_retrace`, the same owner the production path asks. It first
    /// spelled the predicate out again as a `matches!`, which was the exact
    /// construct that function's docstring condemns, so the guard against this
    /// bug class carried the bug class: a `matches!` is not exhaustive, and a
    /// third retrace spelling would have made this helper answer "not a
    /// retrace" while compiling cleanly.
    fn ends_with_dangling_retrace(utt: &Utterance) -> bool {
        utt.main
            .content
            .content
            .iter()
            .next_back()
            .and_then(as_retrace)
            .is_some()
    }

    /// utseg must never split an utterance between a retrace marker and the
    /// repeated/corrected material it points at. A split before the kept word
    /// of `big cat [/] cat [/] cat runs .` once produced a dangling
    /// `big cat [/] cat [/] .` plus `cat runs .`.
    #[test]
    fn utseg_split_does_not_strand_retrace() {
        // `big cat [/] cat [/] cat runs .`: a leading word, two retraced "cat",
        // then the kept run "cat runs". Retraced words are not counted in the
        // Mor word domain, so the three countable words are `big`, `cat` (kept),
        // `runs`. A stanza boundary before the kept "cat" assigns `big` to group
        // 0 and `cat runs` to group 1; the retrace nodes back-fill to the
        // preceding word's group (0), stranding them away from their material.
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tPAR0 Participant\n\
            @ID:\teng|test|PAR0|||||Participant|||\n\
            *PAR0:\tbig cat [/] cat [/] cat runs .\n@End\n";
        let chat = parse_chat(chat_text);
        let utt = get_utterance(&chat, 0).clone();
        let result = split_utterance(utt, &[0, 1, 1]);
        for (i, seg) in result.iter().enumerate() {
            let s = seg.to_chat_string();
            println!("segment {i}: {s}");
            assert!(
                !ends_with_dangling_retrace(seg),
                "utseg split stranded a retrace in segment {i}: {s}"
            );
        }
    }

    /// Annotated-retrace variant of `utseg_split_does_not_strand_retrace`.
    ///
    /// chatter v0.10.0 gave an annotated retrace its own content node,
    /// `AnnotatedRetrace`, so `<that first> [/] [* p:w]` stopped matching
    /// `UtteranceContent::Retrace(_)`. The pre-assignment above then skipped
    /// it, the generic back-fill attached it to the PRECEDING word, and the
    /// stranding this whole block exists to prevent came back for exactly the
    /// utterances that carry an error code. The crate still COMPILED, because
    /// the arm that lost the case was a `matches!`.
    #[test]
    fn utseg_split_does_not_strand_annotated_retrace() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tPAR0 Participant\n\
            @ID:\teng|test|PAR0|||||Participant|||\n\
            *PAR0:\tnow <the red> [/] [* p:w] the red ball .\n@End\n";
        let chat = parse_chat(chat_text);
        let utt = get_utterance(&chat, 0).clone();
        let result = split_utterance(utt, &[0, 1, 1, 1]);
        assert_eq!(result.len(), 2, "expected a split into two segments");
        for (i, seg) in result.iter().enumerate() {
            let s = seg.to_chat_string();
            println!("segment {i}: {s}");
            assert!(
                !ends_with_dangling_retrace(seg),
                "utseg split stranded an annotated retrace in segment {i}: {s}"
            );
        }
    }

    /// Group-form (`<...> [/]`) variant of `utseg_split_does_not_strand_retrace`.
    /// The failure shape was `now <the red> [/] .` followed by `the red ball .`
    /// (a split of `now <the red> [/] the red ball .` before the kept "the").
    /// A `<...> [/]` group is a single `Retrace` content
    /// node, so it must bind forward to its material exactly like a single-word
    /// retrace.
    #[test]
    fn utseg_split_does_not_strand_group_retrace() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tPAR0 Participant\n\
            @ID:\teng|test|PAR0|||||Participant|||\n\
            *PAR0:\tnow <the red> [/] the red ball .\n@End\n";
        let chat = parse_chat(chat_text);
        let utt = get_utterance(&chat, 0).clone();
        // Words in the Mor domain (the retrace group is skipped): now, the,
        // red, ball. A boundary before the kept "the" puts `now` in group 0 and
        // the kept material in group 1.
        let result = split_utterance(utt, &[0, 1, 1, 1]);
        assert_eq!(result.len(), 2, "expected a split into two segments");
        for (i, seg) in result.iter().enumerate() {
            let s = seg.to_chat_string();
            println!("segment {i}: {s}");
            assert!(
                !ends_with_dangling_retrace(seg),
                "utseg split stranded a group retrace in segment {i}: {s}"
            );
        }
    }

    #[test]
    fn test_collect_utseg_payloads() {
        // 3 utterances: 1 single-word, 2 multi-word
        let chat_text = include_str!("../../../test-fixtures/eng_three_utterances.cha");
        let chat = parse_chat(chat_text);
        let collected = collect_utseg_payloads(&chat);
        let payloads = &collected.batch_items;

        // Single-word utterance "hello" should be classified NotApplicable.
        assert_eq!(payloads.len(), 2);
        assert_eq!(collected.not_applicable.len(), 1);
        match &collected.not_applicable[0].kind {
            UtsegOutcomeKind::NotApplicable { reason } => {
                assert_eq!(*reason, UtsegNotApplicableReason::SingleWord);
            }
            other => panic!("expected NotApplicable(SingleWord), got {other:?}"),
        }
        assert_eq!(payloads[0].0, 1); // utt_ordinal of "I eat cookies"
        assert_eq!(payloads[0].1.words, vec!["I", "eat", "cookies"]);
        assert_eq!(payloads[0].1.text, "I eat cookies");
        assert_eq!(payloads[1].0, 2); // utt_ordinal of "he likes cake too"
        assert_eq!(payloads[1].1.words, vec!["he", "likes", "cake", "too"]);
    }

    #[test]
    fn test_apply_utseg_results() {
        let chat_text =
            include_str!("../../../test-fixtures/eng_i_eat_cookies_and_he_likes_cake.cha");
        let mut chat = parse_chat(chat_text);
        assert_eq!(count_utterances(&chat), 1);

        let mut assignment_map = HashMap::new();
        assignment_map.insert(0, vec![0, 0, 0, 1, 1, 1, 1]);

        let applied =
            apply_utseg_results(&mut chat, &assignment_map).expect("admitted segmentation");
        assert_eq!(count_utterances(&chat), 2);
        assert_eq!(applied.layout.input_utterances(), 1);
        assert_eq!(
            applied.layout.whole_output_ordinal(0),
            None,
            "a split utterance has no single output position"
        );

        let out0 = get_utterance(&chat, 0).to_chat_string();
        let out1 = get_utterance(&chat, 1).to_chat_string();
        assert!(out0.contains("I eat cookies"), "First: {out0}");
        assert!(out1.contains("and he likes cake"), "Second: {out1}");
    }

    #[test]
    fn test_apply_utseg_empty_map() {
        let chat_text = include_str!("../../../test-fixtures/eng_i_eat_cookies.cha");
        let mut chat = parse_chat(chat_text);
        let original_count = count_utterances(&chat);

        apply_utseg_results(&mut chat, &HashMap::new()).expect("unchanged file");
        assert_eq!(count_utterances(&chat), original_count);
    }

    #[test]
    fn a_later_refusal_cannot_leave_earlier_utterances_partially_split() {
        let source = include_str!("../../../test-fixtures/live_fixture/eng_multi_utt.cha");
        let mut chat = crate::parse_and_validate(
            source,
            talkbank_model::ParseValidateOptions::default().with_validation(),
        )
        .expect("valid existing multi-utterance CHAT");
        let original = chat.to_chat_string();
        let assignments = HashMap::from([(0, vec![0, 0, 1, 1]), (1, vec![0, 1])]);
        assert!(matches!(
            apply_utseg_results(&mut chat, &assignments),
            Err(UtsegApplyRefusal::Partition {
                utterance_ordinal: 1,
                ..
            })
        ));
        assert_eq!(chat.to_chat_string(), original);
    }

    /// The layout places every input utterance where the output has it: a
    /// split one spans its children, and each utterance after it moves down
    /// by the extra children before it. A whole utterance has one position.
    #[test]
    fn the_layout_places_each_input_utterance_after_the_splits_before_it() {
        let source = include_str!("../../../test-fixtures/live_fixture/eng_multi_utt.cha");
        let mut chat = crate::parse_and_validate(
            source,
            talkbank_model::ParseValidateOptions::default().with_validation(),
        )
        .expect("valid existing multi-utterance CHAT");
        let before = count_utterances(&chat);
        assert!(before >= 2, "the fixture has a second utterance");
        let applied = apply_utseg_results(&mut chat, &HashMap::from([(0, vec![0, 0, 1, 1])]))
            .expect("admitted segmentation");
        assert_eq!(count_utterances(&chat), before + 1);
        assert_eq!(applied.layout.input_utterances(), before);
        assert_eq!(applied.layout.whole_output_ordinal(0), None);
        assert_eq!(applied.layout.whole_output_ordinal(1), Some(2));
        assert_eq!(
            get_utterance(&chat, 2).to_chat_string(),
            crate::parse_and_validate(
                source,
                talkbank_model::ParseValidateOptions::default().with_validation(),
            )
            .map(|original| get_utterance(&original, 1).to_chat_string())
            .expect("valid existing multi-utterance CHAT"),
            "the second input utterance is the third output utterance, unchanged"
        );
        assert_eq!(applied.layout.whole_output_ordinal(before), None);
    }

    #[test]
    fn an_unknown_source_ordinal_cannot_partially_apply_a_known_one() {
        let source = include_str!("../../../test-fixtures/live_fixture/eng_multi_utt.cha");
        let mut chat = crate::parse_and_validate(
            source,
            talkbank_model::ParseValidateOptions::default().with_validation(),
        )
        .expect("valid existing CHAT");
        let original = chat.to_chat_string();
        let assignments = HashMap::from([(0, vec![0, 0, 1, 1]), (99, vec![0])]);
        assert!(matches!(
            apply_utseg_results(&mut chat, &assignments),
            Err(UtsegApplyRefusal::UnknownUtterance(99))
        ));
        assert_eq!(chat.to_chat_string(), original);
    }

    /// After utseg splits, no utterance should start with a Separator node.
    ///
    /// Rev.AI returns "dishes , or she didn't order them" and Stanza puts
    /// the boundary after "dishes". The comma is correctly modeled as
    /// `UtteranceContent::Separator(Separator::Comma)` by build_chat.rs.
    /// But after the split, it lands as the first content item of the second
    /// utterance, which is invalid CHAT. Leading separators must be stripped.
    ///
    /// Bug report: a user, 2026-04-02, 25-3.cha, `*INV: , or she didn't...`
    #[test]
    fn utseg_split_keeps_a_separator_with_its_preceding_child() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tINV Investigator\n\
            @ID:\teng|test|INV|||||Investigator|||\n\
            *INV:\tshe's washing dishes , or she didn't order them .\n@End\n";
        let mut chat = parse_chat(chat_text);

        // The extraction domain includes the comma slot. Keep it with the
        // preceding child rather than inventing a shorter assignment vector.
        let mut assignment_map = HashMap::new();
        assignment_map.insert(0, vec![0, 0, 0, 0, 1, 1, 1, 1, 1]);

        apply_utseg_results(&mut chat, &assignment_map).expect("admitted segmentation");

        // Verify the split produced two utterances
        let utt_count = count_utterances(&chat);
        assert!(
            utt_count >= 2,
            "expected at least 2 utterances, got {utt_count}"
        );

        // No utterance's first content item should be a Separator
        for (i, line) in chat.lines.iter().enumerate() {
            if let Line::Utterance(u) = line
                && let Some(first) = u.main.content.content.first()
            {
                assert!(
                    !matches!(first, UtteranceContent::Separator(_)),
                    "utterance at line {i} starts with a Separator node \
                         (must have retained source ownership): {}",
                    u.to_chat_string()
                );
            }
        }
    }

    /// A split never hands a child a span nobody measured.
    ///
    /// The parent bullet measures the WHOLE parent: `1000_5000` below covers
    /// all seven words. The split puts three words in one child and four in
    /// the other, and nothing observed where the first ended and the second
    /// began. Writing `1000_5000` onto the second child says that child began
    /// at 1000, while the first child's own three words are the evidence that
    /// it did not.
    ///
    /// This replaces `utseg_split_preserves_parent_bullet_on_last_child`,
    /// whose expectation was the defect rather than the fix. That test came
    /// from a real regression (2026-04-26): `split_utterance` built each
    /// child's `MainTier` with `MainTier::new(...)`, which sets
    /// `TierContent.bullet = None`, so every child lost the parent's timing,
    /// 854 of 885 MOST corpus files ended with none at all (223,277 to
    /// 152,192 bullets, -31.8%), and files whose only timing came from
    /// to-be-split utterances tripped E544. The repair restored a bullet by
    /// giving one child a span that was never that child's, which is a
    /// fabricated measurement. The original regression is pinned where the
    /// evidence actually exists, by
    /// `utseg_split_derives_each_child_main_bullet_from_partitioned_wor`:
    /// with `%wor` timing, every child keeps its own measured hull.
    ///
    /// Both halves are asserted here on purpose. An implementation that never
    /// writes a parent bullet passes the first; one that always writes it
    /// passes the second; only the rule itself passes both.
    #[test]
    fn utseg_split_gives_no_child_a_span_that_was_never_measured() {
        // Bullet syntax: NAK-delimited "start_end" appended after the
        // terminator. \u{15} is NAK (0x15). Real example from MOST:
        // `*PAR0: ... . 0_668430` (the 0_668430 is the bullet).
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI eat cookies and he likes cake . \u{15}1000_5000\u{15}\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        // Sanity-check the fixture: the parent utterance carries a bullet.
        assert!(
            parent.main.content.bullet.is_some(),
            "fixture pre-condition: parent must have a bullet"
        );

        // Seven words across two children: the parent's span describes the
        // pair, and neither one of them.
        let split = split_utterance(parent.clone(), &[0, 0, 0, 1, 1, 1, 1]);
        assert_eq!(split.len(), 2, "expected 2 children from the split");
        for (index, child) in split.iter().enumerate() {
            assert!(
                child.main.content.bullet.is_none(),
                "child {index} must carry no main-tier bullet: nothing measured its \
                 span. Output: {}",
                child.to_chat_string()
            );
        }

        // An eighth assignment cannot name a phantom child. A genuinely
        // unchanged seven-word plan retains the parent's actual measurement.
        assert!(matches!(
            talkbank_transform::utterance_split::UtteranceSplitPlan::for_morphology(
                &parent,
                &[0, 0, 0, 0, 0, 0, 0, 1],
            ),
            Err(
                talkbank_transform::utterance_split::SplitRefusal::SlotCount {
                    expected: 7,
                    actual: 8,
                }
            ),
        ));
        let sole = split_utterance(parent, &[0; 7]);
        assert_eq!(sole.len(), 1, "the unchanged parent is retained");
        let kept = sole[0]
            .main
            .content
            .bullet
            .as_ref()
            .expect("a sole child holds the parent's whole content, and its span");
        assert_eq!(kept.timing.start_ms, 1000);
        assert_eq!(kept.timing.end_ms, 5000);
    }

    /// %wor partitioning: when a parent has %wor with timing for every
    /// main-tier word, splitting must distribute the WorItems to the
    /// children matching their words. F1.5: BA2-equivalent per-word
    /// timing preservation across split.
    #[test]
    fn utseg_split_partitions_wor_tier_across_children() {
        // 4-word utterance with %wor giving each word its own timing.
        // Split 2/2: child 0 gets words "I eat", child 1 gets "the cookies".
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI eat the cookies . \u{15}0_4000\u{15}\n\
            %wor:\tI \u{15}0_500\u{15} eat \u{15}500_1500\u{15} the \u{15}1500_2200\u{15} \
            cookies \u{15}2200_4000\u{15} .\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        // Sanity: parent has 4 wor items
        let parent_wor = parent
            .dependent_tiers
            .iter()
            .find_map(|t| match &t.tier {
                DependentTier::Wor(w) => Some(w),
                _ => None,
            })
            .expect("fixture should parse a %wor tier");
        assert_eq!(parent_wor.word_count(), 4, "fixture sanity check");

        let result = split_utterance(parent, &[0, 0, 1, 1]);
        assert_eq!(result.len(), 2);

        let wor_of = |u: &Utterance| -> Option<WorTier> {
            u.dependent_tiers.iter().find_map(|t| match &t.tier {
                DependentTier::Wor(w) => Some(w.clone()),
                _ => None,
            })
        };
        let child0_wor = wor_of(&result[0]).expect("child 0 must carry %wor");
        let child1_wor = wor_of(&result[1]).expect("child 1 must carry %wor");
        assert_eq!(
            child0_wor.word_count(),
            2,
            "child 0 should carry 2 wor words (I, eat)"
        );
        assert_eq!(
            child1_wor.word_count(),
            2,
            "child 1 should carry 2 wor words (the, cookies)"
        );
    }

    /// A partitioned `%wor` tier describes its child main tier, including the
    /// terminator. Earlier children use the splitter's period; only the final
    /// child inherits the parent's question mark.
    #[test]
    fn utseg_split_keeps_child_wor_terminators_in_sync_with_main_tiers() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI eat the cookies ?\n\
            %wor:\tI eat the cookies ?\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        let result = split_utterance(parent, &[0, 0, 1, 1]);
        assert_eq!(result.len(), 2);

        for (i, child) in result.iter().enumerate() {
            let child_wor = child
                .dependent_tiers
                .iter()
                .find_map(|tier| match &tier.tier {
                    DependentTier::Wor(wor) => Some(wor),
                    _ => None,
                })
                .expect("each child must retain its corroborated %wor partition");
            assert_eq!(
                child_wor.terminator, child.main.content.terminator,
                "child {i} main and %wor terminators must describe the same utterance"
            );
        }
    }

    /// When a complete `%wor` tier is partitioned, its word bullets are
    /// stronger timing evidence than the enclosing parent main-tier bullet.
    /// Every child must receive the exact hull of its own timed words.
    #[test]
    fn utseg_split_derives_each_child_main_bullet_from_partitioned_wor() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tPAR Participant\n\
            @ID:\teng|test|PAR|||||Participant|||\n\
            *PAR:\tDepartment store almost anniversary . \u{15}24000_30000\u{15}\n\
            %wor:\tDepartment \u{15}24275_24945\u{15} store \u{15}24945_25265\u{15} \
            almost \u{15}25265_25585\u{15} anniversary \u{15}27455_28145\u{15} .\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        let result = split_utterance(parent, &[0, 0, 1, 1]);

        assert_eq!(result.len(), 2);
        let first = result[0]
            .main
            .content
            .bullet
            .as_ref()
            .expect("complete child %wor timing must produce a main bullet");
        assert_eq!(first.timing.start_ms, 24_275);
        assert_eq!(first.timing.end_ms, 25_265);
        let second = result[1]
            .main
            .content
            .bullet
            .as_ref()
            .expect("complete child %wor timing must produce a main bullet");
        assert_eq!(second.timing.start_ms, 25_265);
        assert_eq!(second.timing.end_ms, 28_145);
    }

    /// A single missing `%wor` timing keeps the split out of the complete
    /// per-child state, and then NEITHER child gets a main-tier bullet.
    ///
    /// This fixture is its own evidence that the old expectation was wrong.
    /// The parent bullet is `24000_30000`; the second child's words are
    /// `almost anniversary`, and the one timing `%wor` still carries for them
    /// starts at 27455. The parent-only fallback used to write `24000_30000`
    /// onto that child, so its main tier claimed it began at 24000 while the
    /// dependent tier beside it recorded 27455. A span contradicted by the
    /// timing evidence in its own utterance was never a measurement.
    #[test]
    fn utseg_split_drops_parent_timing_when_partitioned_wor_timing_is_incomplete() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tPAR Participant\n\
            @ID:\teng|test|PAR|||||Participant|||\n\
            *PAR:\tDepartment store almost anniversary . \u{15}24000_30000\u{15}\n\
            %wor:\tDepartment \u{15}24275_24945\u{15} store \u{15}24945_25265\u{15} \
            almost anniversary \u{15}27455_28145\u{15} .\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        let result = split_utterance(parent, &[0, 0, 1, 1]);

        assert_eq!(result.len(), 2);
        for (index, child) in result.iter().enumerate() {
            assert!(
                child.main.content.bullet.is_none(),
                "child {index} must carry no main-tier bullet: partial timing measures \
                 neither child's span. Output: {}",
                child.to_chat_string()
            );
        }
    }

    /// %wor partitioning falls back to dropping the tier when item counts
    /// don't match main-tier eligible-word counts (stale %wor). No panic,
    /// no validation error: silent drop matches the rename's intent that
    /// stale %wor is legal.
    #[test]
    fn utseg_split_drops_wor_on_count_mismatch() {
        // Parent has 4 main-tier words, but only 3 wor items (stale).
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI eat the cookies .\n\
            %wor:\tI \u{15}0_500\u{15} eat \u{15}500_1500\u{15} cookies \u{15}1500_2000\u{15} .\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        let result = split_utterance(parent, &[0, 0, 1, 1]);
        assert_eq!(result.len(), 2);

        for (i, child) in result.iter().enumerate() {
            let has_wor = child
                .dependent_tiers
                .iter()
                .any(|t| matches!(t.tier, DependentTier::Wor(_)));
            assert!(
                !has_wor,
                "child {i} should not carry %wor when parent counts mismatched (graceful drop)"
            );
        }
    }

    /// Equal word counts are not enough to prove that a `%wor` tier still
    /// describes its main tier. A same-count edit must invalidate the timing
    /// evidence rather than assigning another word's times to a child.
    #[test]
    fn utseg_split_drops_wor_on_lexical_mismatch() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI eat the cookies . \u{15}0_4000\u{15}\n\
            %wor:\tI \u{15}0_500\u{15} eat \u{15}500_1500\u{15} a \u{15}1500_2200\u{15} \
            cookie \u{15}2200_4000\u{15} .\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        let result = split_utterance(parent, &[0, 0, 1, 1]);
        assert_eq!(result.len(), 2);

        for (i, child) in result.iter().enumerate() {
            let has_wor = child
                .dependent_tiers
                .iter()
                .any(|tier| matches!(tier.tier, DependentTier::Wor(_)));
            assert!(
                !has_wor,
                "child {i} must not inherit timing from a lexically stale %wor tier"
            );
        }
        // Dropping the stale tier leaves nothing that measured either child, so
        // neither gets a main-tier bullet. This asserted that the last child
        // kept the parent's `0_4000` until 2026-09-16. That span covers all four
        // words; the split puts `I eat` in one child and `the cookies` in the
        // other, so writing it onto the second one claims that child began at 0,
        // when the first child's own words are the evidence that it did not.
        for (index, child) in result.iter().enumerate() {
            assert!(
                child.main.content.bullet.is_none(),
                "child {index} must carry no main-tier bullet: stale word timing \
                 measured neither child's span. Output: {}",
                child.to_chat_string()
            );
        }
    }

    /// %mor and %gra are dropped on split. Their analysis depends on
    /// utterance-scope context and is invalidated by re-segmentation;
    /// the user reruns morphotag to regenerate.
    #[test]
    fn utseg_split_drops_mor_and_gra() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI eat the cookies .\n\
            %mor:\tpron|I v|eat det|the n|cookies .\n\
            %gra:\t1|2|SUBJ 2|0|ROOT 3|4|DET 4|2|OBJ 5|2|PUNCT\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        let result = split_utterance(parent, &[0, 0, 1, 1]);
        assert_eq!(result.len(), 2);

        for (i, child) in result.iter().enumerate() {
            for tier in &child.dependent_tiers {
                assert!(
                    !matches!(tier.tier, DependentTier::Mor(_) | DependentTier::Gra(_)),
                    "child {i} should not carry %mor or %gra after split (dropped by policy)"
                );
            }
        }
    }

    /// %com and other free-form / utterance-level annotations attach to
    /// the first child. Strictly better than BA2's silent drop.
    #[test]
    fn utseg_split_attaches_com_to_first_child() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI eat the cookies .\n\
            %com:\tchild was excited\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        let result = split_utterance(parent, &[0, 0, 1, 1]);
        assert_eq!(result.len(), 2);

        let first_has_com = result[0]
            .dependent_tiers
            .iter()
            .any(|t| matches!(t.tier, DependentTier::Com(_)));
        let second_has_com = result[1]
            .dependent_tiers
            .iter()
            .any(|t| matches!(t.tier, DependentTier::Com(_)));
        assert!(first_has_com, "first child must inherit the %com");
        assert!(!second_has_com, "second child must not carry %com");
    }

    /// Terminator propagation: the LAST child inherits the parent's
    /// terminator; non-last children get the default `Period`. This
    /// preserves quote-introducer linkage (`+"/.` parent → next-utterance
    /// `+"` quoted speech), interruption markers, and any other
    /// non-default terminator. See spec/errors/E341_auto.md for the
    /// `+"/.` ↔ `+"` validation pairing.
    #[test]
    fn utseg_split_inherits_terminator_on_last_child() {
        // Parent ends with +"/. (quote-introducer). Real shape from
        // childes-eng-na-data/Eng-NA/Kuczaj/030115.cha.
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tand he says +\"/.\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        // Sanity: parent terminator is the quote-introducer, not Period.
        let parent_term = parent
            .main
            .content
            .terminator
            .as_ref()
            .expect("fixture must have a terminator");
        assert!(
            !matches!(parent_term, Terminator::Period { .. }),
            "fixture sanity: parent must end in +\"/. (non-default), got {parent_term:?}"
        );

        let result = split_utterance(parent, &[0, 0, 1]);
        assert_eq!(result.len(), 2, "expected 2 children");

        let first_term = result[0]
            .main
            .content
            .terminator
            .as_ref()
            .expect("child 0 must have a terminator");
        assert!(
            matches!(first_term, Terminator::Period { .. }),
            "non-last child must default to Period, got {first_term:?}"
        );

        let last_term = result[1]
            .main
            .content
            .terminator
            .as_ref()
            .expect("last child must have a terminator");
        assert!(
            !matches!(last_term, Terminator::Period { .. }),
            "LAST child must inherit the parent's non-default terminator, got {last_term:?}"
        );
    }

    /// Linker propagation: the FIRST child inherits the parent's
    /// linkers; non-first children get none. Linkers describe the
    /// utterance's relationship to the *prior* (different) utterance,
    /// so only the first split-piece is adjacent to that prior turn.
    #[test]
    fn utseg_split_inherits_linkers_on_first_child() {
        // Parent starts with `+,` (SelfCompletion linker). Real shape
        // from clan-info/examples/Adler/adler15a.cha.
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tINV Investigator\n\
            @ID:\teng|test|INV|||||Investigator|||\n\
            *INV:\t+, with that letter and one more thing .\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        // Sanity: parent has at least one linker.
        assert!(
            !parent.main.content.linkers.is_empty(),
            "fixture sanity: parent must carry at least one linker"
        );
        let parent_linkers_len = parent.main.content.linkers.len();
        assert!(parent_linkers_len > 0);

        let result = split_utterance(parent, &[0, 0, 0, 1, 1, 1, 1]);
        assert_eq!(result.len(), 2, "expected 2 children");

        assert_eq!(
            result[0].main.content.linkers.len(),
            parent_linkers_len,
            "FIRST child must inherit the parent's linkers"
        );
        assert!(
            result[1].main.content.linkers.is_empty(),
            "non-first child must have no linkers"
        );
    }

    /// Language code propagation: utterance-level `[- code]` applies
    /// to all of the utterance's words, so every child carries it.
    #[test]
    fn utseg_split_propagates_language_code_to_all_children() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng, spa\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\t[- spa] hola amigo y como estas .\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        let parent_lang = parent
            .main
            .content
            .language_code
            .clone()
            .expect("fixture must parse [- spa] language code");

        let result = split_utterance(parent, &[0, 0, 1, 1, 1]);
        assert!(result.len() >= 2);

        for (i, child) in result.iter().enumerate() {
            assert_eq!(
                child.main.content.language_code.as_ref(),
                Some(&parent_lang),
                "child {i} must carry the parent's [- spa] language code"
            );
        }
    }

    /// Postcode propagation: utterance-level `[+ exc]` and similar
    /// analysis tags attach to the LAST child only. They describe the
    /// original utterance as a unit; placing them on the last child
    /// (where they serialize after the terminator) keeps each tag
    /// attached exactly once and matches the conventional position.
    #[test]
    fn utseg_split_inherits_postcodes_on_last_child() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tone two three four . [+ exc]\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let parent = get_utterance(&chat, 0).clone();

        let parent_postcode_len = parent.main.content.postcodes.len();
        assert!(
            parent_postcode_len > 0,
            "fixture sanity: parent must carry at least one postcode"
        );

        let result = split_utterance(parent, &[0, 0, 1, 1]);
        assert_eq!(result.len(), 2, "expected 2 children");

        assert!(
            result[0].main.content.postcodes.is_empty(),
            "non-last child must have no postcodes"
        );
        assert_eq!(
            result[1].main.content.postcodes.len(),
            parent_postcode_len,
            "LAST child must inherit the parent's postcodes"
        );
    }

    /// Replaced-word handling: a `ReplacedWord(wanna [: want to])` is one
    /// main-tier slot but contributes N replacement words to TierDomain::Mor
    /// (the BERT classifier sees N words). `split_utterance` builds its
    /// word→content mapping with TierDomain::Mor too, so the assignment
    /// vector lengths match. The first-assignment-wins logic in
    /// `split_utterance` correctly attributes each ReplacedWord to ONE
    /// child group regardless of where the boundary lands relative to
    /// the replacement words.
    ///
    /// Coverage gap discovered 2026-04-27 while writing replacements
    /// docs (`book/src/batchalign/architecture/replacements-handling.md`); analogous
    /// to the FA bug shape from 2026-04-08. The current code is correct
    /// by construction (extract + split both use TierDomain::Mor); these
    /// tests pin that invariant against future drift.
    #[test]
    fn utseg_split_handles_replaced_word_boundary_before() {
        // Boundary BEFORE the ReplacedWord. Mor walks 4 words: I, want, to, go.
        // assignments=[0, 1, 1, 1] → "I" alone in group 0; "wanna" + "go" in group 1.
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI wanna [: want to] go .\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let utt = get_utterance(&chat, 0).clone();
        let result = split_utterance(utt, &[0, 1, 1, 1]);
        assert_eq!(result.len(), 2, "expected 2 children");
        let s0 = result[0].to_chat_string();
        let s1 = result[1].to_chat_string();
        // child 0: just "I", no fragment of the ReplacedWord or "go"
        assert!(
            !s0.contains("wanna") && !s0.contains("want") && !s0.contains("go"),
            "child 0 should be just I, got: {s0}"
        );
        // child 1: ReplacedWord preserved intact, plus "go"
        assert!(
            s1.contains("wanna [: want to]"),
            "child 1 should keep the ReplacedWord intact, got: {s1}"
        );
        assert!(s1.contains("go"), "child 1 should contain go, got: {s1}");
    }

    #[test]
    fn utseg_split_handles_replaced_word_boundary_after() {
        // Boundary AFTER the ReplacedWord. assignments=[0, 0, 0, 1] →
        // "I wanna" in group 0; "go" alone in group 1. The ReplacedWord
        // (one main-tier slot) lands fully in group 0.
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI wanna [: want to] go .\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let utt = get_utterance(&chat, 0).clone();
        let result = split_utterance(utt, &[0, 0, 0, 1]);
        assert_eq!(result.len(), 2);
        let s0 = result[0].to_chat_string();
        let s1 = result[1].to_chat_string();
        assert!(
            s0.contains("wanna [: want to]"),
            "child 0 should keep the ReplacedWord intact, got: {s0}"
        );
        assert!(
            !s1.contains("wanna") && !s1.contains("want"),
            "child 1 should not have any ReplacedWord fragment, got: {s1}"
        );
        assert!(s1.contains("go"), "child 1 should contain go, got: {s1}");
    }

    #[test]
    fn utseg_refuses_inconsistent_split_inside_replacement_targets() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI wanna [: want to] go .\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let utt = get_utterance(&chat, 0).clone();
        let refused = talkbank_transform::utterance_split::UtteranceSplitPlan::for_morphology(
            &utt,
            &[0, 0, 1, 1],
        );
        assert!(matches!(
            refused,
            Err(
                talkbank_transform::utterance_split::SplitRefusal::IndivisibleContent {
                    content_index: 1
                },
            )
        ));
    }

    #[test]
    fn snapshot_utseg_batch_item() {
        let item = UtsegBatchItem {
            words: vec!["I".into(), "eat".into(), "cookies".into()],
            text: "I eat cookies".into(),
        };
        insta::assert_json_snapshot!(item, @r#"
        {
          "words": [
            "I",
            "eat",
            "cookies"
          ],
          "text": "I eat cookies"
        }
        "#);
    }

    #[test]
    fn snapshot_utseg_response() {
        let resp = UtsegResponse {
            assignments: vec![0, 0, 0, 1, 1, 1, 1],
        };
        insta::assert_json_snapshot!(resp, @r#"
        {
          "assignments": [
            0,
            0,
            0,
            1,
            1,
            1,
            1
          ]
        }
        "#);
    }

    // ---------------------------------------------------------------------
    // Wave 5 outcome-classification tests
    // ---------------------------------------------------------------------

    #[test]
    fn validate_utseg_response_aligned_matching_counts() {
        let item = UtsegBatchItem {
            words: vec!["I".into(), "eat".into(), "cookies".into()],
            text: "I eat cookies".into(),
        };
        let resp = UtsegResponse {
            assignments: vec![0, 0, 0],
        };
        match validate_utseg_response(&item, &resp) {
            UtsegOutcomeKind::Aligned {
                n_words,
                n_segments,
            } => {
                assert_eq!(n_words, 3);
                assert_eq!(n_segments, 1, "all same group = 1 segment");
            }
            other => panic!("expected Aligned, got {other:?}"),
        }
    }

    #[test]
    fn validate_utseg_response_counts_distinct_segments() {
        let item = UtsegBatchItem {
            words: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            text: "a b c d".into(),
        };
        let resp = UtsegResponse {
            assignments: vec![0, 0, 1, 1],
        };
        match validate_utseg_response(&item, &resp) {
            UtsegOutcomeKind::Aligned { n_segments, .. } => assert_eq!(n_segments, 2),
            other => panic!("expected Aligned(2 segments), got {other:?}"),
        }
    }

    #[test]
    fn validate_utseg_response_length_mismatch_is_misalignment_bug() {
        let item = UtsegBatchItem {
            words: vec!["I".into(), "eat".into(), "cookies".into()],
            text: "I eat cookies".into(),
        };
        // Worker returned 2 assignments for 3 input words, contract violation.
        let resp = UtsegResponse {
            assignments: vec![0, 0],
        };
        match validate_utseg_response(&item, &resp) {
            UtsegOutcomeKind::MisalignmentBug(diag) => {
                assert_eq!(diag.expected_assignments, 3);
                assert_eq!(diag.actual_assignments, 2);
                assert_eq!(diag.words, vec!["I", "eat", "cookies"]);
            }
            other => panic!("expected MisalignmentBug, got {other:?}"),
        }
    }

    #[test]
    fn collect_utseg_emits_not_applicable_for_single_word() {
        let chat_text = include_str!("../../../test-fixtures/eng_three_utterances.cha");
        let chat = parse_chat(chat_text);
        let collected = collect_utseg_payloads(&chat);

        assert_eq!(
            collected.not_applicable.len(),
            1,
            "expected the single-word utterance (\"hello\") to be classified NotApplicable",
        );
        let outcome = &collected.not_applicable[0];
        match &outcome.kind {
            UtsegOutcomeKind::NotApplicable { reason } => {
                assert_eq!(*reason, UtsegNotApplicableReason::SingleWord);
            }
            other => panic!("expected NotApplicable(SingleWord), got {other:?}"),
        }
    }

    #[test]
    fn utseg_outcome_to_decision_record_aligned_is_none() {
        let outcome = UtsegOutcome {
            utt_ordinal: 0,
            speaker: SpeakerCode::new("CHI"),
            kind: UtsegOutcomeKind::Aligned {
                n_words: 3,
                n_segments: 1,
            },
        };
        assert!(outcome.to_decision_record(5).is_none());
    }

    #[test]
    fn utseg_outcome_to_decision_record_misalignment_bug_flags_review() {
        let outcome = UtsegOutcome {
            utt_ordinal: 0,
            speaker: SpeakerCode::new("CHI"),
            kind: UtsegOutcomeKind::MisalignmentBug(UtsegMisalignmentDiagnostic {
                expected_assignments: 3,
                actual_assignments: 2,
                words: vec!["hello".into(), "world".into(), "bye".into()],
            }),
        };
        let record = outcome.to_decision_record(5).expect("record for bug");
        assert!(matches!(
            record.strategy,
            crate::decisions::DecisionStrategy::Utseg(
                crate::decisions::UtsegStrategy::MisalignmentBug
            )
        ));
        assert!(record.needs_review);
        assert!(record.reason.contains("expected_assignments=3"));
        assert!(record.reason.contains("actual_assignments=2"));
    }
}
