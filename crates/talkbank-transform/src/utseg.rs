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
//! realignment stage — the Python utseg worker is a per-word classifier
//! whose `assignments` return MUST have the same length as the input
//! `words`. A mismatch is always a worker-contract bug, not an expected
//! divergence class.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use talkbank_model::Span;
use talkbank_model::alignment::helpers::TierDomain;
use talkbank_model::model::dependent_tier::wor::WorItem;
use talkbank_model::model::{
    ChatFile, DependentTier, Line, MainTier, Terminator, Utterance, UtteranceContent, WorTier,
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
/// context. `line_idx` is also available when needed — utseg is indexed
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
    /// Most commonly this is a single-word utterance — a one-word
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
    /// worker-contract bug — the Python classifier is supposed to emit
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
    /// Short label for `%xalign` tier output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SingleWord => "single_word",
            Self::Empty => "empty",
        }
    }
}

/// Diagnostic for an utseg misalignment bug — the worker did not return
/// the contract-required number of assignments.
#[derive(Debug, Clone)]
pub struct UtsegMisalignmentDiagnostic {
    /// Number of words sent to the worker.
    pub expected_assignments: usize,
    /// Number of assignments the worker actually returned.
    pub actual_assignments: usize,
    /// The words that were sent — helps a developer reproduce the case.
    pub words: Vec<String>,
}

impl UtsegOutcome {
    /// Convert into a [`DecisionRecord`](crate::decisions::DecisionRecord)
    /// for surfacing via the `%xalign` tier. Aligned outcomes return
    /// `None` for the same reason as `MorOutcome`: happy-path
    /// utterances shouldn't flood the reporting tier.
    pub fn to_decision_record(&self, line_idx: usize) -> Option<crate::decisions::DecisionRecord> {
        use crate::decisions::{DecisionRecord, DecisionStrategy, UtsegStrategy};
        match &self.kind {
            UtsegOutcomeKind::Aligned { .. } => None,
            UtsegOutcomeKind::NotApplicable { reason } => Some(DecisionRecord {
                line_idx,
                speaker: self.speaker.as_str().to_string(),
                strategy: DecisionStrategy::Utseg(UtsegStrategy::NotApplicable),
                reason: format!("reason={}", reason.as_str()),
                needs_review: false,
            }),
            UtsegOutcomeKind::MisalignmentBug(diag) => Some(DecisionRecord {
                line_idx,
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
/// [`UtsegOutcome::NotApplicable`] entries — no worker call, no silent
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
        extract::collect_utterance_content(&utt.main.content.content, TierDomain::Mor, &mut words);

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
/// length than the dispatched `words` — always a worker-contract bug.
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

    let old_lines = std::mem::take(&mut chat_file.lines.0);
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

    chat_file.lines.0 = new_lines;
}

/// Build a mapping from extracted-word index to top-level content item index.
pub fn build_word_to_content_map(content: &[UtteranceContent]) -> Vec<usize> {
    let mut word_to_content = Vec::new();

    for (content_idx, item) in content.iter().enumerate() {
        let mut words = Vec::new();
        extract::collect_utterance_content(std::slice::from_ref(item), TierDomain::Mor, &mut words);
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
    /// fallback is silent — it emits `tracing::debug!`, never a
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
/// but context-dependent tiers (`%mor`, `%gra`) get [`Drop`] — the data is
/// invalid in the new context. Document- or analysis-scoped tiers (`%coref`)
/// also `Drop`. Other word-positional tiers we don't yet have a partition
/// implementation for (`%pho`, `%mod`, `%sin`, etc.) `Drop` rather than
/// `AttachFirst`, because attaching the parent's full per-word data to one
/// child would falsely claim the data covers all the original words. Free-form
/// utterance-level tiers (`%com`, `%xtra`, `%add`, etc., text tiers, user-defined,
/// unsupported) default to `AttachFirst` — preserve the data on the first child.
///
/// [`Partition`]: TierSplitPolicy::Partition
/// [`Drop`]: TierSplitPolicy::Drop
/// [`AttachFirst`]: TierSplitPolicy::AttachFirst
fn policy_for_tier(tier: &DependentTier) -> TierSplitPolicy {
    match tier {
        // Per-word timing — partitionable by word index.
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

        // Free-form / loosely-structured utterance-level annotations:
        // preserve on first child rather than silently lose.
        _ => TierSplitPolicy::AttachFirst,
    }
}

/// Build a per-child `%wor` from the parent tier by walking main-tier words
/// in lockstep with `%wor` Word-items.
///
/// Returns `None` if positional counts mismatch (stale `%wor` from prior
/// edits, or main-tier token policy drift). On `None`, the caller drops the
/// tier from all children — matching the existing stale-`%wor`-is-fine
/// behavior, never raising a validation error.
///
/// `main_word_groups` is the per-main-tier-word child-group assignment, in
/// main-tier word order, restricted to `%wor`-eligible words (the same
/// filtering `TierDomain::Wor` uses: untranscribed, fragments, and nonwords
/// are excluded; fillers are included).
fn partition_wor_tier(
    parent: &WorTier,
    main_word_groups: &[usize],
    num_groups: usize,
) -> Option<Vec<WorTier>> {
    let parent_word_count = parent.word_count();
    if parent_word_count != main_word_groups.len() {
        tracing::debug!(
            parent_wor_words = parent_word_count,
            main_eligible_words = main_word_groups.len(),
            "%wor count mismatch on split — dropping tier (stale %wor expected after prior edits)"
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
    Some(
        per_child
            .into_iter()
            .map(|items| WorTier {
                language_code: parent.language_code.clone(),
                items,
                terminator: parent.terminator.clone(),
                bullet: parent.bullet.clone(),
                span: Span::DUMMY,
            })
            .collect(),
    )
}

/// Compute the child-group assignment for each main-tier word that is
/// `%wor`-eligible.
///
/// "Eligible" matches `TierDomain::Wor`: untranscribed words (`xxx`/`yyy`/
/// `www`), phonological fragments (`&+`), and nonwords (`&~`) are excluded;
/// fillers (`&-`) are included. The returned Vec has one entry per eligible
/// word, in main-tier order; entries are child-group indices.
///
/// Implementation: walk content_items one at a time, count `%wor`-eligible
/// words inside each via `extract::collect_utterance_content` with
/// `TierDomain::Wor`. Each such word inherits its enclosing content item's
/// group.
fn wor_eligible_word_groups(
    content_items: &[UtteranceContent],
    content_item_group: &[Option<usize>],
) -> Vec<usize> {
    let mut groups = Vec::new();
    for (content_idx, item) in content_items.iter().enumerate() {
        let mut buf = Vec::new();
        extract::collect_utterance_content(std::slice::from_ref(item), TierDomain::Wor, &mut buf);
        let group = content_item_group[content_idx].unwrap_or(0);
        for _ in &buf {
            groups.push(group);
        }
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
    // Capture the parent's main-tier bullet before consuming `utt`. We
    // re-attach it to the LAST child below so the original utterance's
    // end-of-span timing anchor is preserved across the split — without
    // this, every split utterance loses its `_NNN` bullet and any
    // `@Media` linkage assertion the file relied on. See
    // `docs/postmortems/2026-04-26-utseg-split-bullet-loss.md`.
    let parent_bullet = utt.main.content.bullet.clone();

    // Capture the rest of the parent's main-tier metadata so each child
    // can inherit the right slice of it. Per-field propagation policy
    // (linkers → first only, terminator/postcodes → last only, language
    // code/spans → all) is documented in
    // `docs/postmortems/2026-04-26-utseg-split-bullet-loss.md` (F1.6).
    let parent_linkers = utt.main.content.linkers.clone();
    let parent_terminator = utt.main.content.terminator.clone();
    let parent_language_code = utt.main.content.language_code.clone();
    let parent_postcodes = utt.main.content.postcodes.clone();
    let parent_main_span = utt.main.span;
    let parent_speaker_span = utt.main.speaker_span;

    // Compute per-child %wor item lists if the parent has a Wor tier and
    // counts align with main-tier wor-eligible words. None means either
    // no Wor tier was present or counts mismatched (graceful drop).
    let num_groups = max_group + 1;
    let partitioned_wor: Option<Vec<WorTier>> = utt
        .dependent_tiers
        .iter()
        .find_map(|tier| match tier {
            DependentTier::Wor(wor) => {
                let main_groups = wor_eligible_word_groups(content_items, &content_item_group);
                Some(partition_wor_tier(wor, &main_groups, num_groups))
            }
            _ => None,
        })
        .flatten();

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
        // utterance-initial position is invalid CHAT — it belongs with the
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
        // Language code applies to every child (utterance-scope) — set
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

    // Per-tier policy. Walk the parent's dependent tiers once, dispatching
    // each to its policy. `partitioned_wor` (if Some) is the precomputed
    // per-group payload; AttachFirst tiers go to result[0]; Drop tiers
    // produce no output.
    let parent_dep_tiers = utt.dependent_tiers.clone();
    let mut wor_index: Option<usize> = None;
    for (i, tier) in parent_dep_tiers.iter().enumerate() {
        if matches!(tier, DependentTier::Wor(_)) {
            wor_index = Some(i);
            break;
        }
    }

    // Attach partitioned %wor to each child whose group has any items.
    if let Some(wor_idx) = wor_index
        && let Some(per_group) = partitioned_wor
    {
        let _ = wor_idx; // kept for symmetry / future re-ordering needs
        for (group_idx, child) in result.iter_mut() {
            if let Some(child_wor) = per_group.get(*group_idx)
                && !child_wor.items.is_empty()
            {
                child
                    .dependent_tiers
                    .push(DependentTier::Wor(child_wor.clone()));
            }
        }
    }

    if let Some((_, first_child)) = result.first_mut() {
        for tier in &parent_dep_tiers {
            if matches!(policy_for_tier(tier), TierSplitPolicy::AttachFirst) {
                first_child.dependent_tiers.push(tier.clone());
            }
        }
    }

    // Linkers go on the FIRST child only — they describe relation to the
    // *prior* (different) utterance, which only the first piece is adjacent
    // to. Use a non-empty check so we don't bother cloning the empty
    // SmallVec for the common case.
    if !parent_linkers.is_empty()
        && let Some((_, first_child)) = result.first_mut()
    {
        first_child.main.content.linkers = parent_linkers;
    }

    // Terminator and postcodes go on the LAST child only. Terminator
    // describes how the original utterance ended — that's the last child.
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

    // Re-attach the parent's main-tier bullet to the LAST child. The parent's
    // end_ms correctly describes the last child's end timestamp; we make no
    // fabricated claim about non-last children. F2 (proportional UTR hints
    // across all children) is a future refinement.
    if let Some(bullet) = parent_bullet
        && let Some((_, last)) = result.last_mut()
    {
        last.main.content.bullet = Some(bullet);
    }

    result.into_iter().map(|(_, u)| u).collect()
}
