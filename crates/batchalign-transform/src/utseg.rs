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
//! index guards in `split_utterance`.
//!
//! The utseg invariant is simpler than morphotag's: there is no tokenizer
//! realignment stage: the Python utseg worker is a per-word classifier
//! whose `assignments` return MUST have the same length as the input
//! `words`. A mismatch is always a worker-contract bug, not an expected
//! divergence class.

// Wildcard matches over closed enums are denied in this file, following
// chatter's per-file ratchet. The reason it is here rather than crate-wide is
// at `policy_for_tier`, which shipped a defect a `_` arm hid.
#![deny(clippy::wildcard_enum_match_arm)]
// Test code is exempt, matching this crate's existing treatment of the panic
// lints: `other => panic!("unexpected {other:?}")` is how a test says a variant
// should be unreachable, and denying it there would push tests toward asserting
// less rather than more.
#![cfg_attr(test, allow(clippy::wildcard_enum_match_arm))]

use crate::decisions::LineIdx;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use talkbank_model::Span;
use talkbank_model::alignment::helpers::PositionalDomain;
use talkbank_model::alignment::helpers::TierDomain;
use talkbank_model::alignment::helpers::{WordItem, counts_for_tier, walk_words};
use talkbank_model::alignment::{
    WorTimingBinding, WorTimingCorrespondence, WorTimingSequence, assess_wor_timing_sequence,
    bind_wor_timing, corroborate_wor_timing,
};
use talkbank_model::model::ChatFileLines;
use talkbank_model::model::dependent_tier::wor::WorItem;
use talkbank_model::model::{
    Bullet, ChatFile, DependentTier, Line, MainTier, Retrace, Terminator, Utterance,
    UtteranceContent, WorTier,
};

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
    /// Worker returned exactly N assignments for N input words;
    /// `split_utterance` applied cleanly. Happy path.
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

/// Apply utseg assignments to a ChatFile, splitting utterances as needed.
///
/// `assignment_map` maps `utt_ordinal` to assignments (parallel to extracted words).
/// Utterances whose ordinals are not in the map are left unchanged.
pub fn apply_utseg_results(chat_file: &mut ChatFile, assignment_map: &HashMap<usize, Vec<usize>>) {
    if assignment_map.is_empty() {
        return;
    }

    let old_lines = chat_file.lines.take();
    let mut new_lines: Vec<Line> = Vec::with_capacity(old_lines.len());
    let mut utt_ordinal = 0usize;

    for line in old_lines {
        let utt = match line {
            Line::Utterance(u) => u,
            other => {
                new_lines.push(other);
                continue;
            }
        };

        if let Some(assignments) = assignment_map.get(&utt_ordinal) {
            let split_utts = split_utterance(*utt, assignments);
            for split_utt in split_utts {
                new_lines.push(Line::Utterance(Box::new(split_utt)));
            }
        } else {
            new_lines.push(Line::Utterance(utt));
        }

        utt_ordinal += 1;
    }

    chat_file.lines = ChatFileLines::new(new_lines);
}

/// Build a mapping from extracted-word index to top-level content item index.
pub fn build_word_to_content_map(content: &[UtteranceContent]) -> Vec<usize> {
    let mut word_to_content = Vec::new();

    for (content_idx, item) in content.iter().enumerate() {
        let mut words = Vec::new();
        extract::collect_utterance_content(
            std::slice::from_ref(item),
            PositionalDomain::Mor,
            &mut words,
        );
        for _ in &words {
            word_to_content.push(content_idx);
        }
    }

    word_to_content
}

/// Per-tier behavior when an utterance is split into multiple children.
///
/// Splitting an utterance is a transformation that invalidates some
/// dependent-tier data and not others. This enum makes the per-tier
/// decision explicit and grep-able. `policy_for_tier` is the single
/// dispatch site; tests cover each variant.
///
/// History: BA3 deliberately removed parse-time `%wor` alignment in
/// commits `3c178f49` / `ca18388f` / `f7d86537` (2026-04-09) because
/// `chatter validate` was firing `%wor`-count errors on every shift in
/// token-classification semantics. That rename addressed *validation*
/// thrash. This policy addresses a separate concern: when `split_utterance`
/// repartitions words, the per-word data on `%wor` (and similar tiers)
/// should still travel with its words even though no validator demands
/// positional alignment. The rename made staleness *legal*; this policy
/// makes data preservation *useful*. They are independent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TierSplitPolicy {
    /// Walk the parent's items in lockstep with main-tier words and
    /// distribute them across children by the existing word→child
    /// mapping. Falls back to `Drop` if positional counts mismatch
    /// (stale `%wor` from prior edits, or tokenization drift). The
    /// fallback is silent: it emits `tracing::debug!`, never a
    /// validation error, preserving the stale-`%wor`-is-fine stance.
    Partition,
    /// Drop the tier from all children. The tier's data is
    /// semantically invalidated by the split: morphological analysis
    /// assumed the original utterance boundary; dependency arcs
    /// reference word indices that no longer match; coreference
    /// chains span document positions that the split changes. The
    /// user regenerates via `morphotag` / `coref`.
    Drop,
    /// Attach the tier (unchanged) to the first child only. The data
    /// is utterance-level free-form (`%com` comments, `%xtra`
    /// translations, user-defined `%x*` annotations) with no
    /// positional semantics to violate. Stale-on-first-child is
    /// strictly better than silent loss: the user can re-translate or
    /// correct manually, and improves on BA2 which dropped these
    /// unconditionally.
    AttachFirst,
}

/// Map a dependent tier to its split policy.
///
/// Word-positional, context-free tiers (`%wor`) get [`Partition`]. Word-positional
/// but context-dependent tiers (`%mor`, `%gra`) get [`Drop`], the data is
/// invalid in the new context. Document- or analysis-scoped tiers (`%xcoref`)
/// also `Drop`. Other word-positional tiers we don't yet have a partition
/// implementation for (`%pho`, `%mod`, `%sin`, etc.) `Drop` rather than
/// `AttachFirst`, because attaching the parent's full per-word data to one
/// child would falsely claim the data covers all the original words. Free-form
/// utterance-level tiers (`%com`, `%xtra`, `%add`, etc., text tiers, user-defined,
/// unsupported) default to `AttachFirst`, preserve the data on the first child.
///
/// [`Partition`]: TierSplitPolicy::Partition
/// [`Drop`]: TierSplitPolicy::Drop
/// [`AttachFirst`]: TierSplitPolicy::AttachFirst
/// The coreference tier `batchalign3 coref` injects.
///
/// Named here rather than spelled inline so the policy below and the injector
/// cannot disagree about which label they mean.
const XCOREF_LABEL: &str = "xcoref";

fn policy_for_tier(tier: &DependentTier) -> TierSplitPolicy {
    match tier {
        // Per-word timing: partitionable by word index.
        DependentTier::Wor(_) => TierSplitPolicy::Partition,

        // Context-dependent or reference-structured: drop, regenerate downstream.
        DependentTier::Mor(_) | DependentTier::Gra(_) => TierSplitPolicy::Drop,

        // Word-positional but no partition implementation yet. Dropping is
        // honest: attaching to first child would claim phonological / sign
        // data covering all original words, which is wrong. Add to Partition
        // explicitly when a partition impl lands for each shape.
        DependentTier::Pho(_)
        | DependentTier::Mod(_)
        | DependentTier::Sin(_)
        | DependentTier::Modsyl(_)
        | DependentTier::Phosyl(_)
        | DependentTier::Phoaln(_) => TierSplitPolicy::Drop,

        // `%xphoint` indexes INTO `%pho`: it is per-phone time intervals
        // segmenting each `%pho` word. `%pho` is dropped two arms above, so
        // attaching these to the first child leaves interval bullets for
        // phones that are no longer in that utterance, and nothing on the
        // rest. It reached `AttachFirst` through the wildcard this match used
        // to end with, which is the failure the arm above is written to avoid.
        DependentTier::Xphoint(_) => TierSplitPolicy::Drop,

        // Free-form / loosely-structured utterance-level annotations:
        // preserve on first child rather than silently lose.
        //
        // Enumerated so a tier added to chatter stops compiling here until
        // someone decides what splitting it does.
        DependentTier::Act(_)
        | DependentTier::Add(_)
        | DependentTier::Alt(_)
        | DependentTier::Cod(_)
        | DependentTier::Coh(_)
        | DependentTier::Com(_)
        | DependentTier::Def(_)
        | DependentTier::Eng(_)
        | DependentTier::Err(_)
        | DependentTier::Exp(_)
        | DependentTier::Fac(_)
        | DependentTier::Flo(_)
        | DependentTier::Gls(_)
        | DependentTier::Gpx(_)
        | DependentTier::Int(_)
        | DependentTier::Ort(_)
        | DependentTier::Par(_)
        | DependentTier::Sit(_)
        | DependentTier::Spa(_)
        | DependentTier::Tim(_)
        | DependentTier::Unsupported(_) => TierSplitPolicy::AttachFirst,

        // `UserDefined` is a FAMILY, not a policy class, and enumerating the
        // variants gave it one arm. This pipeline emits exactly two labels:
        // `%xtra` is free-form utterance-level, but `%xcoref` is coreference
        // chains, whose links span document positions the split changes, which
        // is the same reason `%mor` and `%gra` are dropped above. The doc for
        // this function has always said document-scoped tiers drop; before the
        // variants were enumerated, `%xcoref` reached `AttachFirst` through the
        // wildcard and contradicted it silently.
        DependentTier::UserDefined(tier) if tier.label.as_str() == XCOREF_LABEL => {
            TierSplitPolicy::Drop
        }
        DependentTier::UserDefined(_) => TierSplitPolicy::AttachFirst,
    }
}

/// Build a per-child `%wor` from the parent tier by walking main-tier words
/// in lockstep with `%wor` Word-items.
///
/// Returns `None` unless chatter proves both equal policy-selected counts and
/// canonical lexical correspondence. On `None`, the caller drops the tier
/// from all children, matching the existing stale-`%wor`-is-fine behavior,
/// never raising a validation error.
///
/// `main_word_groups` is the per-main-tier-word child-group assignment, in
/// main-tier word order, restricted to `%wor`-eligible words (the same
/// filtering `TierDomain::Wor` uses: untranscribed, fragments, and nonwords
/// are excluded; fillers are included).
fn partition_wor_tier(
    main: &MainTier,
    parent: &WorTier,
    main_word_groups: &[usize],
    num_groups: usize,
) -> Option<PartitionedWorTiers> {
    let count_matched = match bind_wor_timing(main, Some(parent)) {
        WorTimingBinding::CountMatched(count_matched) => count_matched,
        WorTimingBinding::Drifted(drift) => {
            tracing::debug!(
                parent_wor_words = drift.wor_count().get(),
                main_eligible_words = drift.main_count().get(),
                "%wor count mismatch on split, dropping tier (stale %wor expected after prior edits)"
            );
            return None;
        }
        WorTimingBinding::Missing(_) => {
            tracing::debug!("%wor binding unexpectedly reported a missing tier during split");
            return None;
        }
    };
    let corroborated = match corroborate_wor_timing(count_matched) {
        WorTimingCorrespondence::Corroborated(corroborated) => corroborated,
        WorTimingCorrespondence::Uncorroborated(uncorroborated) => {
            tracing::debug!(
                lexical_mismatches = uncorroborated.mismatches().len(),
                "%wor lexical mismatch on split, dropping stale timing tier"
            );
            return None;
        }
    };
    if corroborated.slots().len() != main_word_groups.len() {
        tracing::debug!(
            chatter_projection_words = corroborated.slots().len(),
            batchalign_projection_words = main_word_groups.len(),
            "%wor membership implementations disagree on split, dropping timing tier"
        );
        return None;
    }

    // Walk parent items, tracking which main-tier word index we're on for
    // Word items. Separators have no main-tier counterpart; we attach them
    // to the same child as the most recent Word, falling back to group 0
    // if we haven't seen any Word yet.
    let mut per_child: Vec<Vec<WorItem>> = vec![Vec::new(); num_groups];
    let mut next_word_idx = 0usize;
    let mut last_seen_group: Option<usize> = None;
    for item in &parent.items {
        match item {
            WorItem::Word(_) => {
                let group = main_word_groups[next_word_idx];
                last_seen_group = Some(group);
                per_child[group].push(item.clone());
                next_word_idx += 1;
            }
            WorItem::Separator { .. } => {
                let group = last_seen_group.unwrap_or(0);
                per_child[group].push(item.clone());
            }
        }
    }

    // Build a WorTier for each child. Children with empty item lists get an
    // empty WorTier; the caller filters those out (we don't emit empty
    // `%wor:` tiers).
    Some(PartitionedWorTiers(
        per_child
            .into_iter()
            .map(|items| {
                WorTier::new(items)
                    .with_language_code(parent.language_code.clone())
                    .with_span(Span::DUMMY)
            })
            .collect(),
    ))
}

/// Per-child `%wor` tiers admitted by count and lexical corroboration.
///
/// The inner vector stays private so split code cannot accidentally treat an
/// uncorroborated positional partition as reusable timing evidence.
struct PartitionedWorTiers(Vec<WorTier>);

impl PartitionedWorTiers {
    fn get(&self, group_idx: usize) -> Option<&WorTier> {
        self.0.get(group_idx)
    }
}

/// Complete child timing rederived from the partitioned `%wor` evidence.
///
/// Construction is private to [`split_main_timing_evidence`], which produces
/// exactly one bullet for every kept child or refuses this state entirely.
struct CompletePerChildMainTiming {
    bullets: Vec<Bullet>,
}

/// The conservative fallback when complete word timing is unavailable.
struct ParentOnlyMainTiming {
    bullet: Option<Bullet>,
}

/// Mutually exclusive timing evidence available after an utterance split.
enum SplitMainTimingEvidence {
    CompletePerChild(CompletePerChildMainTiming),
    ParentOnly(ParentOnlyMainTiming),
}

/// Derive one enclosing hull only when every `%wor` word is timed.
///
/// A partial tier cannot claim the full child span, so one missing word bullet
/// makes this return `None` and selects the parent-only fallback for the whole
/// split. Min/max is intentional: it encloses all admitted word spans even if
/// their serialized order contains a local timing inversion.
fn complete_wor_timing_hull(main: &MainTier, wor: &WorTier) -> Option<Bullet> {
    let count_matched = match bind_wor_timing(main, Some(wor)) {
        WorTimingBinding::CountMatched(count_matched) => count_matched,
        WorTimingBinding::Missing(_) | WorTimingBinding::Drifted(_) => return None,
    };
    let corroborated = match corroborate_wor_timing(count_matched) {
        WorTimingCorrespondence::Corroborated(corroborated) => corroborated,
        WorTimingCorrespondence::Uncorroborated(_) => return None,
    };
    let complete = match assess_wor_timing_sequence(corroborated) {
        WorTimingSequence::Complete(complete) => complete,
        WorTimingSequence::Empty(_) | WorTimingSequence::Rejected(_) => return None,
    };
    let hull = complete.hull();
    Some(Bullet::new(hull.start().get(), hull.end().get()))
}

fn split_main_timing_evidence(
    partitioned_wor: Option<&PartitionedWorTiers>,
    children: &[(usize, Utterance)],
    parent_bullet: Option<Bullet>,
) -> SplitMainTimingEvidence {
    let Some(per_group) = partitioned_wor else {
        return SplitMainTimingEvidence::ParentOnly(ParentOnlyMainTiming {
            bullet: parent_bullet,
        });
    };
    let Some(bullets) = children
        .iter()
        .map(|(group_idx, child)| {
            per_group
                .get(*group_idx)
                .and_then(|wor| complete_wor_timing_hull(&child.main, wor))
        })
        .collect::<Option<Vec<_>>>()
    else {
        return SplitMainTimingEvidence::ParentOnly(ParentOnlyMainTiming {
            bullet: parent_bullet,
        });
    };
    if bullets.is_empty() {
        return SplitMainTimingEvidence::ParentOnly(ParentOnlyMainTiming {
            bullet: parent_bullet,
        });
    }
    SplitMainTimingEvidence::CompletePerChild(CompletePerChildMainTiming { bullets })
}

/// The retrace this content node is, in EITHER spelling, or `None`.
///
/// One owner for the question, because the answer is what decides whether a
/// node binds forward to its material. It was written inline as
/// `matches!(item, UtteranceContent::Retrace(_))`, and when chatter v0.10.0
/// added `AnnotatedRetrace` that expression silently started answering "no"
/// for every retrace carrying an error code: the crate compiled, and the
/// stranding regression this module documents came back for exactly the
/// utterances a transcriber had annotated. A `matches!` is a catch-all wearing
/// a macro. The exhaustive match below makes a third spelling a compile error.
///
/// Returns the NODE rather than a bool: the two spellings differ only in
/// carrying the annotations that follow the marker, so a caller that later
/// needs the retraced material has it, and a predicate that answers `true`
/// and throws the answer away cannot be extended without being rewritten.
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

/// Compute the child-group assignment for each main-tier word that is
/// `%wor`-eligible.
///
/// "Eligible" is chatter's `%wor` slot rule (`WorSlotMembershipPolicy::
/// FilteredLexicalV1`): untranscribed words (`xxx`/`yyy`/`www`),
/// phonological fragments (`&+`), and nonwords (`&~`) are excluded; fillers
/// (`&-`) are included. The returned Vec has one entry per eligible word, in
/// main-tier order; entries are child-group indices.
///
/// Implementation: walk each content item on its own with the `%wor` walker
/// and admit exactly what `WorMainTierProjection::from_main` admits (a word,
/// or a replaced word's original, that `counts_for_tier` accepts for `%wor`).
/// The projection itself is only constructible from a whole `MainTier`, so
/// a per-item count has to restate its admission rule; the rule is named
/// here so the two cannot drift silently.
fn wor_eligible_word_groups(
    content_items: &[UtteranceContent],
    content_item_group: &[Option<usize>],
) -> Vec<usize> {
    let mut groups = Vec::new();
    for (content_idx, item) in content_items.iter().enumerate() {
        let group = content_item_group[content_idx].unwrap_or(0);
        walk_words(
            std::slice::from_ref(item),
            Some(TierDomain::Wor),
            &mut |word| {
                let admitted = match word {
                    WordItem::Word(word) => counts_for_tier(word, TierDomain::Wor),
                    WordItem::ReplacedWord(replaced) => {
                        counts_for_tier(&replaced.word, TierDomain::Wor)
                    }
                    WordItem::Separator(_) => false,
                };
                if admitted {
                    groups.push(group);
                }
            },
        );
    }
    groups
}

/// Split an utterance into multiple utterances based on word assignments.
///
/// `assignments` is a Vec parallel to the extracted words, where each element
/// is the 0-based utterance ID that word belongs to.
pub fn split_utterance(utt: Utterance, assignments: &[usize]) -> Vec<Utterance> {
    let content_items = &utt.main.content.content;
    let word_to_content = build_word_to_content_map(content_items);

    if assignments.is_empty() || word_to_content.is_empty() {
        return vec![utt];
    }

    let first = assignments[0];
    if assignments.iter().all(|&a| a == first) {
        return vec![utt];
    }

    let num_content_items = content_items.len();
    let mut content_item_group: Vec<Option<usize>> = vec![None; num_content_items];

    for (word_idx, &content_idx) in word_to_content.iter().enumerate() {
        if word_idx < assignments.len() && content_item_group[content_idx].is_none() {
            content_item_group[content_idx] = Some(assignments[word_idx]);
        }
    }

    // A retrace marker binds FORWARD to the repeated/corrected material it points
    // at: `<X> [/] Y` is one unit where X was abandoned and Y is the retry, so a
    // retrace content node must travel with the following kept word's group.
    // Retraced words are not counted in the Mor word domain, so retrace nodes get
    // no direct assignment above; the generic back-fill below would attach them to
    // the PRECEDING word, stranding them as a dangling `[/]` when the split
    // boundary falls between the retrace and its material (real BA3 utseg output
    // stranded retraces this way across the UMICH SNL corpus). Pre-assign each
    // un-grouped retrace node the group of the next already-grouped content item;
    // if none follows (a legitimately utterance-final retrace), leave it for the
    // generic back-fill. Regression: `utseg_split_does_not_strand_retrace`.
    for idx in 0..num_content_items {
        if content_item_group[idx].is_some() || as_retrace(&content_items[idx]).is_none() {
            continue;
        }
        if let Some(next_group) = content_item_group[idx + 1..].iter().find_map(|g| *g) {
            content_item_group[idx] = Some(next_group);
        }
    }

    // Back-fill unassigned items
    let mut last_group: Option<usize> = None;
    for group in content_item_group.iter_mut() {
        if group.is_some() {
            last_group = *group;
        } else {
            *group = last_group;
        }
    }
    // Forward-fill remaining None at the start
    let mut next_group: Option<usize> = None;
    for group in content_item_group.iter_mut().rev() {
        if group.is_some() {
            next_group = *group;
        } else {
            *group = next_group;
        }
    }

    let max_group = assignments.iter().copied().max().unwrap_or(0);

    let mut groups: Vec<Vec<UtteranceContent>> = vec![Vec::new(); max_group + 1];
    for (content_idx, item) in content_items.iter().enumerate() {
        if content_item_group[content_idx].is_none() {
            tracing::warn!(
                content_idx,
                "content item has no group assignment, defaulting to group 0"
            );
        }
        let group_id = content_item_group[content_idx].unwrap_or(0);
        if group_id <= max_group {
            groups[group_id].push(item.clone());
        }
    }

    let speaker = &utt.main.speaker;
    // Capture the parent's main-tier bullet before consuming `utt`. Complete
    // partitioned `%wor` evidence supersedes it with one exact hull per child;
    // otherwise the conservative fallback keeps this parent span on the last
    // child only.
    let parent_bullet = utt.main.content.bullet.clone();

    // Capture the rest of the parent's main-tier metadata so each child
    // can inherit the right slice of it. Per-field propagation policy
    // (linkers → first only, terminator/postcodes → last only, language
    // code/spans → all).
    let parent_linkers = utt.main.content.linkers.clone();
    let parent_terminator = utt.main.content.terminator.clone();
    let parent_language_code = utt.main.content.language_code.clone();
    let parent_postcodes = utt.main.content.postcodes.clone();
    let parent_main_span = utt.main.span;
    let parent_speaker_span = utt.main.speaker_span;

    // Compute per-child %wor item lists only when the parent tier is count-
    // matched and lexically corroborated against the typed main tier. None
    // means absent or stale evidence (graceful drop).
    let num_groups = max_group + 1;
    let partitioned_wor = utt
        .dependent_tiers
        .iter()
        // A search for one tier, not a policy over all of them, so the
        // "everything else" case is genuinely "not the tier I am looking for"
        // and a tier added later is correctly not it. Written as a `let-else`
        // rather than a match with a wildcard so that distinction is visible:
        // the file denies wildcard matches precisely because the OTHER one in
        // it was a policy decision wearing the same syntax.
        .find_map(|tier| {
            let DependentTier::Wor(wor) = &tier.tier else {
                return None;
            };
            Some(wor)
        })
        .and_then(|wor| {
            let main_groups = wor_eligible_word_groups(content_items, &content_item_group);
            partition_wor_tier(&utt.main, wor, &main_groups, num_groups)
        });

    // Track (original_group_idx, utterance) so we can later look up the
    // partitioned %wor for each kept child even after empty/all-separator
    // groups are skipped.
    let mut result: Vec<(usize, Utterance)> = Vec::new();

    for (group_idx, mut group_content) in groups.into_iter().enumerate() {
        if group_content.is_empty() {
            continue;
        }

        // Strip leading Separator nodes (comma, tag, vocative) that landed
        // at the start of this group after the split. A Separator at
        // utterance-initial position is invalid CHAT, it belongs with the
        // preceding content or should be dropped.
        let first_non_sep = group_content
            .iter()
            .position(|item| !matches!(item, UtteranceContent::Separator(_)))
            .unwrap_or(group_content.len());
        if first_non_sep > 0 {
            group_content.drain(..first_non_sep);
        }
        if group_content.is_empty() {
            continue;
        }

        let mut main = MainTier::new(
            speaker.clone(),
            group_content,
            Terminator::Period { span: Span::DUMMY },
        );
        // Language code applies to every child (utterance-scope), set
        // it at construction time. Linkers, terminator, postcodes, and
        // bullet are positional and applied to the right child after
        // the loop.
        if let Some(ref lang) = parent_language_code {
            main.content = main.content.with_language_code(lang.clone());
        }
        // Source spans: inherit the parent's so children retain a
        // useful (if coarse) source pointer instead of `Span::DUMMY`.
        main.span = parent_main_span;
        main.speaker_span = parent_speaker_span;
        let new_utt = Utterance::new(main);
        result.push((group_idx, new_utt));
    }

    if result.is_empty() {
        tracing::warn!("utterance segmentation produced no groups, returning original");
        return vec![utt];
    }

    let main_timing = split_main_timing_evidence(partitioned_wor.as_ref(), &result, parent_bullet);

    // Per-tier policy. Walk the parent's dependent tiers once, dispatching
    // each to its policy. `partitioned_wor` (if Some) is the precomputed
    // per-group payload; AttachFirst tiers go to result[0]; Drop tiers
    // produce no output.
    let parent_dep_tiers = utt.dependent_tiers.clone();
    if let Some((_, first_child)) = result.first_mut() {
        for tier in &parent_dep_tiers {
            if matches!(policy_for_tier(&tier.tier), TierSplitPolicy::AttachFirst) {
                first_child.dependent_tiers.push(tier.clone());
            }
        }
    }

    // Linkers go on the FIRST child only, they describe relation to the
    // *prior* (different) utterance, which only the first piece is adjacent
    // to. Use a non-empty check so we don't bother cloning the empty
    // SmallVec for the common case.
    if !parent_linkers.is_empty()
        && let Some((_, first_child)) = result.first_mut()
    {
        first_child.main.content.linkers = parent_linkers;
    }

    // Terminator and postcodes go on the LAST child only. Terminator
    // describes how the original utterance ended, that's the last child.
    // Postcodes are utterance-level analysis tags; placing them on the
    // last child matches the conventional after-terminator serialization.
    if let Some((_, last)) = result.last_mut() {
        if let Some(term) = parent_terminator {
            last.main.content.terminator = Some(term);
        }
        if !parent_postcodes.is_empty() {
            last.main.content.postcodes = parent_postcodes;
        }
    }

    // Attach partitioned %wor after child main-tier terminators are final, so
    // the dependent tier cannot retain the parent's terminator on an earlier
    // child or retain the default period on the last child.
    if let Some(per_group) = partitioned_wor {
        for (group_idx, child) in result.iter_mut() {
            if let Some(child_wor) = per_group.get(*group_idx)
                && !child_wor.items.is_empty()
            {
                let child_wor = child_wor
                    .clone()
                    .with_terminator(child.main.content.terminator.clone());
                child
                    .dependent_tiers
                    .push(DependentTier::Wor(child_wor).into());
            }
        }
    }

    match main_timing {
        SplitMainTimingEvidence::CompletePerChild(CompletePerChildMainTiming { bullets }) => {
            debug_assert_eq!(bullets.len(), result.len());
            for ((_, child), bullet) in result.iter_mut().zip(bullets) {
                child.main.content.bullet = Some(bullet);
            }
        }
        SplitMainTimingEvidence::ParentOnly(ParentOnlyMainTiming { bullet }) => {
            if let Some(bullet) = bullet
                && let Some((_, last)) = result.last_mut()
            {
                last.main.content.bullet = Some(bullet);
            }
        }
    }

    result.into_iter().map(|(_, u)| u).collect()
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
        let result = split_utterance(utt, &[0, 0, 0]);
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
    /// `Sis <that first> [/] [* p:w] .` is stranded and the scan called it
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
    /// repeated/corrected material it points at. The real BA3 utseg pass stranded
    /// retraces across the UMICH SNL corpus this way, e.g.
    /// `Dig up your mud dig [/] dig [/] dig [/] dig fire trucks coming .`
    /// split before the kept "dig" produced a dangling
    /// `... dig [/] dig [/] dig [/] .` plus `dig fire trucks coming .`.
    #[test]
    fn utseg_split_does_not_strand_retrace() {
        // `mud dig [/] dig [/] dig fire .`: a leading word, two retraced "dig",
        // then the kept run "dig fire". Retraced words are not counted in the
        // Mor word domain, so the three countable words are `mud`, `dig` (kept),
        // `fire`. A stanza boundary before the kept "dig" assigns `mud` to group
        // 0 and `dig fire` to group 1; the retrace nodes back-fill to the
        // preceding word's group (0), stranding them away from their material.
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tPAR0 Participant\n\
            @ID:\teng|test|PAR0|||||Participant|||\n\
            *PAR0:\tmud dig [/] dig [/] dig fire .\n@End\n";
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
    /// utterances that carry an error code, on the UMICH corpus the original
    /// bug was found in. The crate still COMPILED, because the arm that lost
    /// the case was a `matches!`.
    #[test]
    fn utseg_split_does_not_strand_annotated_retrace() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tPAR0 Participant\n\
            @ID:\teng|test|PAR0|||||Participant|||\n\
            *PAR0:\tSis <that first> [/] [* p:w] that first you .\n@End\n";
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
    /// The real headline case was `Sis <that first> [/] .` followed by
    /// `that first you .` (utseg split `Sis <that first> [/] that first you .`
    /// before the kept "that"). A `<...> [/]` group is a single `Retrace` content
    /// node, so it must bind forward to its material exactly like a single-word
    /// retrace.
    #[test]
    fn utseg_split_does_not_strand_group_retrace() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tPAR0 Participant\n\
            @ID:\teng|test|PAR0|||||Participant|||\n\
            *PAR0:\tSis <that first> [/] that first you .\n@End\n";
        let chat = parse_chat(chat_text);
        let utt = get_utterance(&chat, 0).clone();
        // Words in the Mor domain (the retrace group is skipped): Sis, that,
        // first, you. A boundary before the kept "that" puts Sis in group 0 and
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

        apply_utseg_results(&mut chat, &assignment_map);
        assert_eq!(count_utterances(&chat), 2);

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

        apply_utseg_results(&mut chat, &HashMap::new());
        assert_eq!(count_utterances(&chat), original_count);
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
    fn utseg_split_strips_leading_separator() {
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tINV Investigator\n\
            @ID:\teng|test|INV|||||Investigator|||\n\
            *INV:\tshe's washing dishes , or she didn't order them .\n@End\n";
        let mut chat = parse_chat(chat_text);

        // Stanza boundary: words 0-2 = group 0, words 3-7 = group 1.
        // The comma separator sits between groups.
        let mut assignment_map = HashMap::new();
        assignment_map.insert(0, vec![0, 0, 0, 1, 1, 1, 1, 1]);

        apply_utseg_results(&mut chat, &assignment_map);

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
                         (should have been stripped): {}",
                    u.to_chat_string()
                );
            }
        }
    }

    /// REGRESSION: the parent's main-tier timing bullet must not be silently
    /// dropped when an utterance is split.
    ///
    /// The utseg pipeline produced bullet-less output across 854/885 MOST
    /// corpus files on 2026-04-26 because `split_utterance` constructed each
    /// child's `MainTier` fresh via `MainTier::new(...)`, which sets
    /// `TierContent.bullet = None`, without copying `utt.main.content.bullet`
    /// from the parent. The aggregate signal: 223,277 → 152,192 bullets
    /// (−31.8%) corpus-wide. Files whose only timing came from to-be-split
    /// utterances ended up with no timing at all and tripped E544
    /// (@Media-linkage assertion).
    ///
    /// Conservative invariant tested here: at least one child of a split
    /// must carry the parent's bullet. We attach it to the LAST child
    /// the original utterance ended at the bullet's end timestamp, and
    /// the last child of the split contains the last words and ends at
    /// that same end timestamp.
    #[test]
    fn utseg_split_preserves_parent_bullet_on_last_child() {
        // Bullet syntax: NAK-delimited "start_end" appended after the
        // terminator. \u{15} is NAK (0x15). Real example from MOST:
        // `*PAR0: ... . 0_668430` (the 0_668430 is the bullet).
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI eat cookies and he likes cake . \u{15}1000_5000\u{15}\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let utt = get_utterance(&chat, 0).clone();

        // Sanity-check the fixture: the parent utterance carries a bullet.
        assert!(
            utt.main.content.bullet.is_some(),
            "fixture pre-condition: parent must have a bullet"
        );
        let parent_bullet = utt.main.content.bullet.as_ref().unwrap().clone();

        // Split into two children: words 0-2 → child 0, words 3-6 → child 1.
        let result = split_utterance(utt, &[0, 0, 0, 1, 1, 1, 1]);
        assert_eq!(result.len(), 2, "expected 2 children from the split");

        // The LAST child must carry the parent's bullet (same start_ms /
        // end_ms). The earlier children may have no bullet, we simply
        // don't know their per-child timing without realignment, and we
        // refuse to fabricate one.
        let last_child_bullet = &result.last().unwrap().main.content.bullet;
        assert!(
            last_child_bullet.is_some(),
            "last child of a split must inherit the parent's bullet \
             (got None: parent timing was dropped). Output: {}",
            result.last().unwrap().to_chat_string()
        );
        let last = last_child_bullet.as_ref().unwrap();
        assert_eq!(
            last.timing.start_ms, parent_bullet.timing.start_ms,
            "last child's bullet start must equal parent's"
        );
        assert_eq!(
            last.timing.end_ms, parent_bullet.timing.end_ms,
            "last child's bullet end must equal parent's"
        );
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
    /// per-child state. Do not mix exact child hulls with guessed spans.
    #[test]
    fn utseg_split_uses_parent_only_when_partitioned_wor_timing_is_incomplete() {
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
        assert!(
            result[0].main.content.bullet.is_none(),
            "partial timing must not create a first-child hull"
        );
        let fallback = result[1]
            .main
            .content
            .bullet
            .as_ref()
            .expect("parent-only fallback must preserve the enclosing anchor");
        assert_eq!(fallback.timing.start_ms, 24_000);
        assert_eq!(fallback.timing.end_ms, 30_000);
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
        assert!(
            result[0].main.content.bullet.is_none(),
            "stale word timing cannot establish a first-child hull"
        );
        let fallback = result[1]
            .main
            .content
            .bullet
            .as_ref()
            .expect("the last child must preserve the parent timing fallback");
        assert_eq!(fallback.timing.start_ms, 0);
        assert_eq!(fallback.timing.end_ms, 4_000);
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

        let result = split_utterance(parent, &[0, 0, 1, 1, 1, 1]);
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
    fn utseg_split_attributes_replaced_word_atomically_on_inconsistent_split() {
        // BERT puts the boundary BETWEEN the replacement words "want" and
        // "to" (assignments=[0, 0, 1, 1]). Structurally we cannot split
        // inside a single main-tier slot. The first-assignment-wins logic
        // attributes the entire ReplacedWord to "want"'s group (0). Net
        // result: "I wanna" in group 0, "go" in group 1, same as if the
        // boundary had been after the ReplacedWord. This pins the
        // "ReplacedWord is atomic to splits" invariant.
        let chat_text = "@UTF8\n@Begin\n@Languages:\teng\n\
            @Participants:\tCHI Child\n\
            @ID:\teng|test|CHI|||||Child|||\n\
            *CHI:\tI wanna [: want to] go .\n\
            @End\n";
        let chat = parse_chat(chat_text);
        let utt = get_utterance(&chat, 0).clone();
        let result = split_utterance(utt, &[0, 0, 1, 1]);
        assert_eq!(result.len(), 2);
        let s0 = result[0].to_chat_string();
        let s1 = result[1].to_chat_string();
        // ReplacedWord stayed atomic: went with "want"'s assignment (0).
        assert!(
            s0.contains("wanna [: want to]"),
            "ReplacedWord must be atomic on splits; got s0: {s0}"
        );
        assert!(
            !s1.contains("wanna") && !s1.contains("want"),
            "child 1 should not contain any fragment of the ReplacedWord, got: {s1}"
        );
        assert!(s1.contains("go"), "child 1 should contain go, got: {s1}");
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
