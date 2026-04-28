//! Translation helpers for the server-side translate orchestrator.
//!
//! Extracts text from utterances, computes cache keys, and injects `%xtra`
//! dependent tiers with translated text.
//!
//! Types and functions for the server-side translate orchestrator:
//! payload collection, cache key computation, injection, and extraction.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use talkbank_model::Span;
use talkbank_model::alignment::helpers::TierDomain;
use talkbank_model::model::{
    ChatFile, DependentTier, Line, NonEmptyString, UserDefinedDependentTier,
};

use crate::extract;

// ---------------------------------------------------------------------------
// Wire types (match Python's TranslateBatchItem / TranslateResponse)
// ---------------------------------------------------------------------------

/// Input payload for a single translation request.
///
/// Matches the Python `TranslateBatchItem` wire format.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TranslateBatchItem {
    /// Source-language text to translate.
    pub text: String,
}

/// Response from translation inference.
///
/// Contains the translated text for a single utterance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateResponse {
    /// Translated text in the target language.
    pub translation: String,
}

// ---------------------------------------------------------------------------
// Payload collection
// ---------------------------------------------------------------------------

/// Collect translate payloads from all utterances in a ChatFile.
///
/// Returns `(line_idx, TranslateBatchItem)` pairs. Empty utterances
/// (no extractable words) are skipped. `line_idx` is the index into
/// `chat_file.lines` (needed for injection).
pub fn collect_translate_payloads(chat_file: &ChatFile) -> Vec<(usize, TranslateBatchItem)> {
    let mut batch_items = Vec::new();

    for (line_idx, line) in chat_file.lines.iter().enumerate() {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        let mut words = Vec::new();
        extract::collect_utterance_content(&utt.main.content.content, TierDomain::Mor, &mut words);

        if !words.is_empty() {
            let text: String = words
                .iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            batch_items.push((line_idx, TranslateBatchItem { text }));
        }
    }

    batch_items
}

// ---------------------------------------------------------------------------
// Cache key
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Injection
// ---------------------------------------------------------------------------

/// Inject a translation as a `%xtra` dependent tier on an utterance.
///
/// Creates a `DependentTier::UserDefined` with label "xtra" and uses
/// `replace_or_add_tier` to inject it (replacing any existing `%xtra`).
pub fn inject_translation(
    utterance: &mut talkbank_model::model::Utterance,
    translation_text: &str,
) -> Result<(), String> {
    if translation_text.is_empty() {
        return Ok(());
    }

    let label = NonEmptyString::new("xtra")
        .ok_or_else(|| "Failed to create NonEmptyString for 'xtra'".to_string())?;
    let content = NonEmptyString::new(translation_text)
        .ok_or_else(|| "Failed to create NonEmptyString for translation content".to_string())?;

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

/// Apply translation results to a ChatFile.
///
/// `results` maps `line_idx` to translated text. Lines whose indices
/// are not in the map are left unchanged.
pub fn apply_translate_results(chat_file: &mut ChatFile, results: &HashMap<usize, String>) {
    if results.is_empty() {
        return;
    }

    for (&line_idx, translation) in results {
        if let Some(Line::Utterance(utt)) = chat_file.lines.get_mut(line_idx)
            && let Err(e) = inject_translation(utt, translation)
        {
            tracing::warn!(
                line_idx,
                error = %e,
                "Failed to inject translation"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Extraction (for caching after injection)
// ---------------------------------------------------------------------------

/// Entry for extracting `%xtra` tier content from a processed utterance.
pub struct TranslationStringsEntry {
    /// Index into `ChatFile.lines`.
    pub line_idx: usize,
    /// Extracted `%xtra` tier translation text.
    pub translation: String,
}

/// Extract `%xtra` tier content from specified utterances for caching.
pub fn extract_translation_strings(
    chat_file: &ChatFile,
    line_indices: &[usize],
) -> Vec<TranslationStringsEntry> {
    let mut results = Vec::with_capacity(line_indices.len());

    for &line_idx in line_indices {
        let Some(line) = chat_file.lines.get(line_idx) else {
            continue;
        };
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        for tier in &utt.dependent_tiers {
            if let DependentTier::UserDefined(ud) = tier
                && ud.label.as_ref() == "xtra"
            {
                results.push(TranslationStringsEntry {
                    line_idx,
                    translation: ud.content.as_ref().to_string(),
                });
                break;
            }
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Pre/post-processing (moved from Python translate.py)
// ---------------------------------------------------------------------------

/// Pre-process text before sending to a translation API.
///
/// For Chinese/Cantonese source languages, strips spaces and replaces
/// periods with ideographic full stops.
pub fn preprocess_for_translate(
    text: &str,
    src_lang: &talkbank_model::model::LanguageCode,
) -> String {
    let lang = src_lang.as_str();
    if lang == "yue" || lang == "zho" {
        let mut result = text.replace(' ', "");
        result = result.replace('.', "\u{3002}"); // ideographic full stop
        result
    } else {
        text.to_string()
    }
}

/// Returns the CHAT punctuation characters used for translation spacing.
///
/// Includes both MOR separators (vocative ‡, tag „, comma ,) and terminators
/// (. ? ! +... +/. +/? etc.).
pub fn chat_punct_chars() -> Vec<String> {
    use talkbank_model::Span;
    use talkbank_model::model::content::{Separator, Terminator};

    let separators: Vec<String> = vec![
        Separator::Vocative { span: Span::DUMMY },
        Separator::Tag { span: Span::DUMMY },
        Separator::Comma { span: Span::DUMMY },
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect();

    let terminators: Vec<String> = vec![
        Terminator::Period { span: Span::DUMMY },
        Terminator::Question { span: Span::DUMMY },
        Terminator::Exclamation { span: Span::DUMMY },
        Terminator::TrailingOff { span: Span::DUMMY },
        Terminator::Interruption { span: Span::DUMMY },
        Terminator::SelfInterruption { span: Span::DUMMY },
        Terminator::InterruptedQuestion { span: Span::DUMMY },
        Terminator::BrokenQuestion { span: Span::DUMMY },
        Terminator::QuotedNewLine { span: Span::DUMMY },
        Terminator::QuotedPeriodSimple { span: Span::DUMMY },
        Terminator::SelfInterruptedQuestion { span: Span::DUMMY },
        Terminator::TrailingOffQuestion { span: Span::DUMMY },
        Terminator::BreakForCoding { span: Span::DUMMY },
    ]
    .into_iter()
    .map(|t| t.to_string())
    .collect();

    [separators, terminators].concat()
}

/// Post-process raw translation output from the API.
///
/// Applies normalization:
/// - Ideographic full stop → period
/// - Curly quotes → straight quotes
/// - Zero-width spaces removed
/// - Tab → space
/// - Punctuation spacing (add space before each punct char)
pub fn postprocess_translation(raw: &str, punct_chars: &[&str]) -> String {
    // Single-pass normalization: char mapping + zero-width space removal
    let mut result: String = raw
        .chars()
        .filter(|&c| c != '\u{200b}') // zero-width space
        .map(|c| match c {
            '\u{3002}' => '.',               // ideographic full stop → period
            '\u{2018}' | '\u{2019}' => '\'', // curly quotes → straight
            '\t' => ' ',
            c => c,
        })
        .collect();

    for p in punct_chars {
        result = result.replace(p, &format!(" {p}"));
    }

    result
}
