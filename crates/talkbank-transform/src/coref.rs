//! Coreference resolution helpers for the server-side coref orchestrator.
//!
//! Extracts sentences from utterances, injects `%xcoref` dependent tiers,
//! and applies sparse coref annotations.
//!
//! Key difference from morphosyntax/utseg/translate: coref is **document-level**.
//! Each `CorefBatchItem` contains ALL sentences from one file (not one utterance).
//! No per-utterance caching — results depend on full document context.
//!
//! ## Outcome model
//!
//! Every utterance visited by `collect_coref_payloads` + `apply_coref_results`
//! produces exactly one [`CorefOutcome`]. Sibling to morphotag's
//! [`MorOutcome`](crate::morphosyntax::outcome::MorOutcome) and utseg's
//! [`UtsegOutcome`](crate::utseg::UtsegOutcome), but with a different
//! shape because coref is **sparse by design**: most utterances in a
//! document have no coreference chains at all, and that is correct — not
//! an anomaly.
//!
//! The five outcome variants distinguish:
//!
//! - Expected no-op (`NotApplicable`, `NoChainsForSentence`)
//! - Happy path (`ChainsInjected`)
//! - True anomalies (`SentenceIndexOutOfBounds`, `InjectionFailed`)

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use talkbank_model::Span;
use talkbank_model::alignment::helpers::TierDomain;
use talkbank_model::model::{
    ChatFile, DependentTier, Line, NonEmptyString, UserDefinedDependentTier,
};

use crate::extract;
use talkbank_model::SpeakerCode;

// ---------------------------------------------------------------------------
// Wire types (match Python's CorefBatchItem / CorefResponse)
// ---------------------------------------------------------------------------

/// Input payload for a single coref request — one complete document.
///
/// Unlike morphosyntax/translate where each item is one utterance,
/// each `CorefBatchItem` contains ALL sentences from one file because
/// coreference resolution needs cross-sentence context.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CorefBatchItem {
    /// List of sentences, each a list of words.
    pub sentences: Vec<Vec<String>>,
}

/// A single coref annotation for one sentence (bracket notation format).
///
/// Used for injection into `%xcoref` tiers and for backwards-compatible
/// wire format.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CorefAnnotation {
    /// Index into the `sentences` array of the corresponding `CorefBatchItem`.
    pub sentence_idx: usize,
    /// Coreference annotation string in bracket notation, e.g. `"(0, -, (1, 1)"`.
    pub annotation: String,
}

/// Response from coref inference — sparse annotations for sentences with chains.
///
/// Only sentences that contain actual coreference chains are included.
/// Sentences with all-`-` annotations are omitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorefResponse {
    /// Per-sentence coref annotations (only sentences with chains).
    pub annotations: Vec<CorefAnnotation>,
}

// ---------------------------------------------------------------------------
// Structured coref data model
// ---------------------------------------------------------------------------

/// A single coreference chain reference on a word.
///
/// Represents one chain that a word participates in. A word can
/// simultaneously start one chain and end another.
///
/// # Bracket notation mapping
///
/// | is_start | is_end | notation |
/// |----------|--------|----------|
/// | true     | false  | `(N`     |
/// | false    | true   | `N)`     |
/// | true     | true   | `(N)`    |
/// | false    | false  | `N`      |
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct ChainRef {
    /// Chain identifier (0-based, assigned by Stanza).
    pub chain_id: usize,
    /// Whether this word starts a mention in this chain.
    pub is_start: bool,
    /// Whether this word ends a mention in this chain.
    pub is_end: bool,
}

/// Raw per-sentence coref data from the Python worker.
///
/// Each element in `words` is parallel to the sentence's word list.
/// An empty vec means the word has no coreference chains (serialized as `-`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorefRawAnnotation {
    /// Index into the `sentences` array of the corresponding `CorefBatchItem`.
    pub sentence_idx: usize,
    /// Per-word chain references, parallel to the sentence's word list.
    /// Empty vec = no chains on this word.
    pub words: Vec<Vec<ChainRef>>,
}

/// Raw structured response from coref inference.
///
/// Python returns this when using the new wire format. Rust builds
/// bracket notation from it before injecting into `%xcoref`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorefRawResponse {
    /// Per-sentence structured coref data (only sentences with chains).
    pub annotations: Vec<CorefRawAnnotation>,
}

// ---------------------------------------------------------------------------
// Bracket notation serialization
// ---------------------------------------------------------------------------

/// Build bracket notation string from structured per-word chain data.
///
/// # Format
///
/// ```text
/// annotation = word_elem (", " word_elem)*
/// word_elem  = "-" | chain_ref (" " chain_ref)*
/// chain_ref  = "("? chain_id ")"?
/// ```
///
/// Positional: element `i` corresponds to word `i` in the sentence.
pub fn build_bracket_annotation(words: &[Vec<ChainRef>]) -> String {
    let mut parts = Vec::with_capacity(words.len());
    for word_chains in words {
        if word_chains.is_empty() {
            parts.push("-".to_string());
        } else {
            let refs: Vec<String> = word_chains
                .iter()
                .map(|cr| {
                    let mut s = String::new();
                    if cr.is_start {
                        s.push('(');
                    }
                    s.push_str(&cr.chain_id.to_string());
                    if cr.is_end {
                        s.push(')');
                    }
                    s
                })
                .collect();
            parts.push(refs.join(" "));
        }
    }
    parts.join(", ")
}

/// Convert a raw structured response to the bracket notation response.
///
/// This is the bridge between the typed data model and the serialized
/// `%xcoref` tier content.
pub fn raw_to_bracket_response(raw: &CorefRawResponse) -> CorefResponse {
    CorefResponse {
        annotations: raw
            .annotations
            .iter()
            .map(|ann| CorefAnnotation {
                sentence_idx: ann.sentence_idx,
                annotation: build_bracket_annotation(&ann.words),
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Typed outcome model (Wave 5 of the morphotag reconciliation architecture)
// ---------------------------------------------------------------------------

/// One coreference outcome for one utterance.
///
/// Carries `line_idx` and `speaker` so it can be surfaced as a
/// [`DecisionRecord`](crate::decisions::DecisionRecord) without
/// additional context.
#[derive(Debug, Clone)]
pub struct CorefOutcome {
    /// 0-based line index in `chat_file.lines`.
    pub line_idx: usize,
    /// Speaker code.
    pub speaker: SpeakerCode,
    /// What happened.
    pub kind: CorefOutcomeKind,
}

/// Per-utterance coref outcome classification.
///
/// Coref differs from morphotag and utseg in two ways that shape this
/// enum:
///
/// 1. **Sparse by design.** Most utterances in a document have no
///    coreference chains at all. That is correct output — not a bug,
///    not a skip. `NoChainsForSentence` names this outcome
///    explicitly so reports don't treat it as anomalous.
/// 2. **Document-level dispatch.** The worker receives all sentences
///    at once and can in principle return annotations referring to
///    sentence indices that don't exist. `SentenceIndexOutOfBounds`
///    captures that worker-contract violation as a typed anomaly.
#[derive(Debug, Clone)]
pub enum CorefOutcomeKind {
    /// Utterance had zero Mor-alignable words and was not dispatched.
    /// Correct behavior. Parallel to
    /// [`MorOutcomeKind::NotApplicable`](crate::morphosyntax::outcome::MorOutcomeKind::NotApplicable).
    NotApplicable,
    /// Utterance was dispatched but the worker returned no coreference
    /// chains for it. This is the **common expected case** — most
    /// sentences in natural text don't participate in coref chains.
    /// Not an anomaly.
    NoChainsForSentence,
    /// Coref annotation was injected into `%xcoref` successfully.
    ChainsInjected {
        /// The annotation text that was injected, for audit.
        annotation: String,
    },
    /// Worker response referred to a sentence_idx that maps to a
    /// `line_idx` outside `chat_file.lines`. This is always a
    /// worker-contract violation.
    SentenceIndexOutOfBounds {
        /// Worker-reported sentence_idx that could not be resolved.
        sentence_idx: usize,
        /// Line index the orchestrator mapped it to, but which is
        /// out of range.
        resolved_line_idx: usize,
    },
    /// `inject_coref` failed (e.g., `NonEmptyString` construction).
    InjectionFailed {
        /// Underlying error message.
        error: String,
    },
}

impl CorefOutcome {
    /// Convert this outcome into a [`DecisionRecord`](crate::decisions::DecisionRecord).
    ///
    /// Expected outcomes (`NotApplicable`, `NoChainsForSentence`,
    /// `ChainsInjected`) return `None` — surfacing them per-utterance
    /// would flood the reporting tier for every document. Only true
    /// anomalies return a record.
    pub fn to_decision_record(&self) -> Option<crate::decisions::DecisionRecord> {
        use crate::decisions::{CorefStrategy, DecisionRecord, DecisionStrategy};
        match &self.kind {
            CorefOutcomeKind::NotApplicable
            | CorefOutcomeKind::NoChainsForSentence
            | CorefOutcomeKind::ChainsInjected { .. } => None,
            CorefOutcomeKind::SentenceIndexOutOfBounds {
                sentence_idx,
                resolved_line_idx,
            } => Some(DecisionRecord {
                line_idx: self.line_idx,
                speaker: self.speaker.as_str().to_string(),
                strategy: DecisionStrategy::Coref(CorefStrategy::SentenceIndexOutOfBounds),
                reason: format!(
                    "sentence_idx={sentence_idx} resolved_line_idx={resolved_line_idx}"
                ),
                needs_review: true,
            }),
            CorefOutcomeKind::InjectionFailed { error } => Some(DecisionRecord {
                line_idx: self.line_idx,
                speaker: self.speaker.as_str().to_string(),
                strategy: DecisionStrategy::Coref(CorefStrategy::InjectionFailed),
                reason: format!("error={error}"),
                needs_review: true,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Payload collection
// ---------------------------------------------------------------------------

/// Result of [`collect_coref_payloads`]: the document-level batch item,
/// plus line_indices for result mapping, plus typed NotApplicable
/// outcomes for any utterance that was not dispatched.
pub struct CorefPayloadCollection {
    /// The document-level batch item (all non-empty sentences).
    pub batch_item: CorefBatchItem,
    /// `line_indices[i]` is the index into `chat_file.lines` for
    /// sentence `i` in the batch item.
    pub line_indices: Vec<usize>,
    /// Utterances that were classified NotApplicable (empty content).
    pub not_applicable: Vec<CorefOutcome>,
}

/// Collect coref payloads from all utterances in a ChatFile.
///
/// Empty utterances (no extractable words) are classified as
/// [`CorefOutcomeKind::NotApplicable`] — visible in reports rather than
/// silently dropped.
pub fn collect_coref_payloads(chat_file: &ChatFile) -> CorefPayloadCollection {
    let mut sentences = Vec::new();
    let mut line_indices = Vec::new();
    let mut not_applicable = Vec::new();

    for (line_idx, line) in chat_file.lines.iter().enumerate() {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        let mut words = Vec::new();
        extract::collect_utterance_content(&utt.main.content.content, TierDomain::Mor, &mut words);

        if words.is_empty() {
            not_applicable.push(CorefOutcome {
                line_idx,
                speaker: SpeakerCode::new(utt.main.speaker.as_str()),
                kind: CorefOutcomeKind::NotApplicable,
            });
        } else {
            let word_texts: Vec<String> = words.iter().map(|w| w.text.to_string()).collect();
            sentences.push(word_texts);
            line_indices.push(line_idx);
        }
    }

    CorefPayloadCollection {
        batch_item: CorefBatchItem { sentences },
        line_indices,
        not_applicable,
    }
}

// ---------------------------------------------------------------------------
// Injection
// ---------------------------------------------------------------------------

/// Inject a coref annotation as a `%xcoref` dependent tier on an utterance.
///
/// Creates a `DependentTier::UserDefined` with label "xcoref" and uses
/// `replace_or_add_tier` to inject it (replacing any existing `%xcoref`).
///
/// Empty `annotation_text` is a no-op (returns `Ok(())`).
///
/// # Errors
///
/// Returns `Err` if `NonEmptyString` construction fails for the tier label
/// or content (should only happen if `annotation_text` contains only
/// whitespace after the empty check).
pub fn inject_coref(
    utterance: &mut talkbank_model::model::Utterance,
    annotation_text: &str,
) -> Result<(), String> {
    if annotation_text.is_empty() {
        return Ok(());
    }

    let label = NonEmptyString::new("xcoref")
        .ok_or_else(|| "Failed to create NonEmptyString for 'xcoref'".to_string())?;
    let content = NonEmptyString::new(annotation_text)
        .ok_or_else(|| "Failed to create NonEmptyString for coref content".to_string())?;

    let new_tier = DependentTier::UserDefined(UserDefinedDependentTier {
        label,
        content,
        span: Span::DUMMY,
    });

    crate::inject::replace_or_add_tier(&mut utterance.dependent_tiers, new_tier);
    Ok(())
}

// ---------------------------------------------------------------------------
// Result application
// ---------------------------------------------------------------------------

/// Apply coref results to a ChatFile (sparse injection).
///
/// `results` maps `line_idx` to annotation text. Only lines whose indices
/// are in the map get a `%xcoref` tier — utterances without coreference
/// chains are left unchanged.
///
/// This is the legacy signature kept for existing callers; for the
/// typed-outcome variant see [`apply_coref_results_with_outcomes`].
pub fn apply_coref_results(chat_file: &mut ChatFile, results: &HashMap<usize, String>) {
    let _ = apply_coref_results_with_outcomes(chat_file, results, &[]);
}

/// Apply coref results and return a per-dispatched-utterance outcome stream.
///
/// `dispatched_line_indices` is the `line_indices` field from the
/// [`CorefPayloadCollection`] that produced this dispatch — i.e., the
/// line indices of every utterance that was sent to the worker. Any
/// dispatched line_idx that does NOT appear in `results` is classified
/// as [`CorefOutcomeKind::NoChainsForSentence`] (the common expected
/// case).
///
/// Return: a `Vec<CorefOutcome>` with one entry per dispatched
/// utterance. Caller may concatenate this with the `not_applicable`
/// outcomes from collection to get one outcome per utterance in the
/// document.
pub fn apply_coref_results_with_outcomes(
    chat_file: &mut ChatFile,
    results: &HashMap<usize, String>,
    dispatched_line_indices: &[usize],
) -> Vec<CorefOutcome> {
    let mut outcomes: Vec<CorefOutcome> = Vec::new();
    let mut handled_in_results: std::collections::BTreeSet<usize> =
        std::collections::BTreeSet::new();

    // First pass: walk the dispatched set in order, producing either
    // ChainsInjected (if an annotation exists for this line_idx) or
    // NoChainsForSentence (the common expected case).
    for (sentence_idx, &line_idx) in dispatched_line_indices.iter().enumerate() {
        let speaker_opt = match chat_file.lines.get(line_idx) {
            Some(Line::Utterance(u)) => Some(SpeakerCode::new(u.main.speaker.as_str())),
            _ => None,
        };
        let Some(speaker) = speaker_opt else {
            outcomes.push(CorefOutcome {
                line_idx,
                speaker: SpeakerCode::new(""),
                kind: CorefOutcomeKind::SentenceIndexOutOfBounds {
                    sentence_idx,
                    resolved_line_idx: line_idx,
                },
            });
            continue;
        };

        match results.get(&line_idx) {
            Some(annotation) => {
                handled_in_results.insert(line_idx);
                let outcome_kind = match chat_file.lines.get_mut(line_idx) {
                    Some(Line::Utterance(utt)) => match inject_coref(utt, annotation) {
                        Ok(()) => CorefOutcomeKind::ChainsInjected {
                            annotation: annotation.clone(),
                        },
                        Err(e) => CorefOutcomeKind::InjectionFailed { error: e },
                    },
                    _ => CorefOutcomeKind::SentenceIndexOutOfBounds {
                        sentence_idx,
                        resolved_line_idx: line_idx,
                    },
                };
                outcomes.push(CorefOutcome {
                    line_idx,
                    speaker,
                    kind: outcome_kind,
                });
            }
            None => {
                outcomes.push(CorefOutcome {
                    line_idx,
                    speaker,
                    kind: CorefOutcomeKind::NoChainsForSentence,
                });
            }
        }
    }

    // Second pass: any `results` entries whose line_idx was NOT in the
    // dispatched set are worker-contract violations — the worker
    // annotated something we didn't ask about.
    for (&line_idx, annotation) in results {
        if handled_in_results.contains(&line_idx) {
            continue;
        }
        let speaker = match chat_file.lines.get(line_idx) {
            Some(Line::Utterance(u)) => SpeakerCode::new(u.main.speaker.as_str()),
            _ => SpeakerCode::new(""),
        };
        // Try injection (the legacy behavior), but record the anomaly:
        // the worker returned an annotation for an undispatched line.
        if let Some(Line::Utterance(utt)) = chat_file.lines.get_mut(line_idx) {
            match inject_coref(utt, annotation) {
                Ok(()) => {
                    outcomes.push(CorefOutcome {
                        line_idx,
                        speaker,
                        kind: CorefOutcomeKind::ChainsInjected {
                            annotation: annotation.clone(),
                        },
                    });
                }
                Err(e) => {
                    tracing::warn!(
                        line_idx,
                        error = %e,
                        "Failed to inject coref annotation (undispatched)"
                    );
                    outcomes.push(CorefOutcome {
                        line_idx,
                        speaker,
                        kind: CorefOutcomeKind::InjectionFailed { error: e },
                    });
                }
            }
        } else {
            outcomes.push(CorefOutcome {
                line_idx,
                speaker,
                kind: CorefOutcomeKind::SentenceIndexOutOfBounds {
                    sentence_idx: usize::MAX, // undispatched; we have no sentence_idx
                    resolved_line_idx: line_idx,
                },
            });
        }
    }

    outcomes
}

// ---------------------------------------------------------------------------
// Clearing
// ---------------------------------------------------------------------------

/// Remove existing `%xcoref` tiers from all utterances.
///
/// Used for re-processing: clears stale coref annotations before
/// running a fresh coref pass.
pub fn clear_coref(chat_file: &mut ChatFile) {
    for line in &mut chat_file.lines {
        if let Line::Utterance(utt) = line {
            utt.dependent_tiers.retain(|tier| {
                !matches!(
                    tier,
                    DependentTier::UserDefined(ud) if ud.label.as_ref() == "xcoref"
                )
            });
        }
    }
}
