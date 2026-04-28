//! Contiguous span grouping for secondary dispatch.
//!
//! Groups @s words into per-utterance contiguous spans by target language,
//! preserving within-span context for Stanza (e.g., "los niños" stays
//! together rather than being sent as two isolated words).

use std::collections::HashMap;

use talkbank_model::model::LanguageCode;
use talkbank_model::validation::LanguageResolution;

use super::extract::L2DeferredPosition;

/// A contiguous span of @s words sharing the same target language.
///
/// Used by `group_l2_spans` for per-utterance analysis of @s patterns.
#[derive(Debug, Clone, PartialEq)]
pub struct L2Span {
    /// Indices into the utterance's word list (0-based).
    pub word_indices: Vec<usize>,
    /// Resolved target language for secondary dispatch.
    pub target_lang: LanguageCode,
    /// Word texts extracted for the secondary batch.
    pub words: Vec<String>,
}

/// A contiguous span of @s words ready for secondary dispatch, with
/// provenance tracking back to the deferred position array.
#[derive(Debug, Clone)]
pub struct DispatchSpan {
    /// Global indices into the `L2DeferredPosition` array.
    pub global_indices: Vec<usize>,
    /// Word texts for this span (sent as one sentence to Stanza).
    pub words: Vec<String>,
    /// Target language for dispatch.
    pub target_lang: LanguageCode,
}

/// Resolve the dispatch target language from a `LanguageResolution`.
fn resolve_dispatch_lang(resolution: &LanguageResolution) -> Option<LanguageCode> {
    resolution.languages().first().cloned()
}

/// Group @s words into contiguous spans by target language.
///
/// Consecutive @s words with the same resolved target language are merged
/// into a single [`L2Span`]. Non-contiguous @s words or words with
/// different target languages produce separate spans.
///
/// Words with `Unresolved` language resolution are skipped (they will
/// fall back to `L2|xxx`).
pub fn group_l2_spans(
    special_forms: &[(
        Option<talkbank_model::model::FormType>,
        Option<LanguageResolution>,
    )],
    word_texts: &[String],
) -> Vec<L2Span> {
    let mut spans: Vec<L2Span> = Vec::new();

    for (idx, (_form_type, lang_res)) in special_forms.iter().enumerate() {
        let target_lang = match lang_res {
            Some(res) => match resolve_dispatch_lang(res) {
                Some(lang) => lang,
                None => continue,
            },
            None => continue,
        };

        let extends = spans.last().is_some_and(|last: &L2Span| {
            last.target_lang == target_lang
                && last
                    .word_indices
                    .last()
                    .is_some_and(|&prev| prev + 1 == idx)
        });

        if extends {
            if let Some(last) = spans.last_mut() {
                last.word_indices.push(idx);
                last.words.push(word_texts[idx].clone());
            }
        } else {
            spans.push(L2Span {
                word_indices: vec![idx],
                target_lang,
                words: vec![word_texts[idx].clone()],
            });
        }
    }

    spans
}

/// Group deferred positions into per-utterance contiguous spans for dispatch.
///
/// Within each utterance, consecutive @s words with the same target language
/// are merged into a single span. Each span becomes one sentence in the
/// secondary Stanza batch.
///
/// `word_cache` provides word texts keyed by `(line_idx, word_idx)`,
/// pre-extracted from the ChatFile.
pub fn group_deferred_into_dispatch_spans(
    deferred: &[L2DeferredPosition],
    word_cache: &HashMap<(usize, usize), String>,
) -> Vec<DispatchSpan> {
    let mut spans: Vec<DispatchSpan> = Vec::new();

    for (global_idx, def) in deferred.iter().enumerate() {
        let word_text = word_cache
            .get(&(def.line_idx, def.word_idx))
            .cloned()
            .unwrap_or_default();

        let extends = spans.last().is_some_and(|last: &DispatchSpan| {
            last.target_lang == def.target_lang && !last.global_indices.is_empty() && {
                let prev_global = *last.global_indices.last().unwrap_or(&0);
                let prev_def = &deferred[prev_global];
                prev_def.line_idx == def.line_idx && prev_def.word_idx + 1 == def.word_idx
            }
        });

        if extends {
            if let Some(last) = spans.last_mut() {
                last.global_indices.push(global_idx);
                last.words.push(word_text);
            }
        } else {
            spans.push(DispatchSpan {
                global_indices: vec![global_idx],
                words: vec![word_text],
                target_lang: def.target_lang.clone(),
            });
        }
    }

    spans
}
