//! Secondary L2 dispatch for @s words.

use std::collections::HashMap;

use super::identity::AppliedAnalyses;
use super::worker::infer_batch;
use crate::chat_ops::morphosyntax_ops::l2;
use crate::chat_ops::{ChatFile, LanguageCode};
use crate::pipeline::PipelineServices;

fn secondary_dispatch_supported(
    registry: Option<&crate::stanza_registry::StanzaRegistry>,
    target_lang: &LanguageCode,
) -> bool {
    registry.is_some_and(|reg| reg.supports_morphosyntax(target_lang.as_ref()))
}

// ---------------------------------------------------------------------------
// Experimental: secondary L2 dispatch for @s words
// ---------------------------------------------------------------------------

/// Dispatch @s words to secondary language workers and splice results back.
///
/// This function:
/// 1. Plans contiguous spans by target language, consuming the positions
/// 2. For each supported target language, builds one `MorphosyntaxBatchItem`
///    per span and dispatches them to a secondary Stanza worker via
///    `infer_batch`
/// 3. Merges each span with its secondary analysis (the secondary owns the
///    words' morphology, the primary the span's attachment)
/// 4. Splices the merged spans into the ChatFile, replacing `L2|xxx`
///
/// Words of unsupported languages, failed dispatches and spans whose
/// secondary analysis does not merge keep `L2|xxx`, each reported.
///
/// Returns what the secondary workers reported about the analysis merged in,
/// so a caller that stamps provenance names those models beside the
/// primary-language ones and counts their repairs with the rest.
pub(crate) async fn dispatch_secondary_l2(
    chat_file: &mut ChatFile,
    deferred: Vec<l2::L2DeferredPosition>,
    services: PipelineServices<'_>,
    filename: &str,
) -> AppliedAnalyses {
    use crate::chat_ops::morphosyntax_ops::{BatchWord, MorphosyntaxBatchItem};

    let deferred_words = deferred.len();
    let dispatch_plan = l2::plan_dispatch_spans(deferred);
    let planned_spans = dispatch_plan.spans.len();

    // Group spans by target language for batched dispatch.
    let mut by_lang: HashMap<LanguageCode, Vec<l2::L2SpanPlan>> = HashMap::new();
    for span in dispatch_plan.spans {
        by_lang
            .entry(span.target_lang().clone())
            .or_default()
            .push(span);
    }

    tracing::info!(
        filename = %filename,
        deferred = deferred_words,
        spans = planned_spans,
        languages = by_lang.len(),
        "L2 morphotag: dispatching @s words to secondary workers"
    );

    let mut merged: Vec<l2::MergedL2Span> = Vec::new();
    let mut applied = AppliedAnalyses::none();

    for (target_lang, lang_spans) in by_lang {
        let total_words: usize = lang_spans.iter().map(l2::L2SpanPlan::len).sum();
        let lang3 = match crate::api::LanguageCode3::try_new(target_lang.as_ref()) {
            Ok(l) => l,
            Err(_) => {
                tracing::warn!(
                    lang = %target_lang,
                    words = total_words,
                    "L2 morphotag: invalid language code; words stay L2|xxx"
                );
                continue;
            }
        };

        let supported = secondary_dispatch_supported(services.pool.stanza_registry(), &target_lang);

        if !supported {
            tracing::info!(
                lang = %lang3,
                words = total_words,
                registry_available = services.pool.stanza_registry().is_some(),
                "L2 morphotag: unsupported or unavailable language"
            );
            continue;
        }

        // Each span becomes one item (one Stanza "sentence"). It names no
        // utterance position: the span carries its own, and the merge pairs
        // each response with its span.
        let batch_items: Vec<MorphosyntaxBatchItem> = lang_spans
            .iter()
            .map(|span| {
                // Every span word is the secondary model's to analyse.
                let words = span.words().cloned().map(BatchWord::analysed).collect();
                MorphosyntaxBatchItem::new(
                    words,
                    // The span's own utterance terminator, not a period for
                    // everything. Stanza treats sentence-final punctuation as
                    // evidence, so a question dispatched as a statement comes
                    // back parsed as one.
                    span.terminator().clone(),
                    target_lang.clone(),
                )
            })
            .collect();

        let empty_mwt: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::new();

        match infer_batch(
            services.pool,
            &batch_items,
            &lang3,
            &empty_mwt,
            true,
            None,
            // Secondary L2 dispatch has no job or `MorphosyntaxParams` in
            // scope here (see the module doc: this splices @s-word results
            // back into an already-parsed `ChatFile`, several layers below
            // any job-carrying caller); genuinely NotWired.
            crate::infer_retry::Cancellation::NotWired {
                reason: "secondary L2 dispatch has no job or params in scope",
            },
        )
        .await
        {
            Ok(responses) => {
                let spans = lang_spans.len();
                if responses.len() != spans {
                    tracing::warn!(
                        lang = %lang3,
                        spans,
                        responses = responses.len(),
                        "L2 morphotag: secondary worker answered a different number of \
                         spans than were sent; unanswered spans stay L2|xxx"
                    );
                }
                for (span, admitted) in lang_spans.into_iter().zip(responses.iter()) {
                    let line_idx = span.line_idx();
                    let words = span.len();
                    let Some(sentence) = admitted.response().sentences.first() else {
                        tracing::warn!(
                            lang = %lang3,
                            line_idx,
                            words,
                            "L2 morphotag: secondary worker returned no sentence; \
                             words stay L2|xxx"
                        );
                        continue;
                    };
                    match l2::merge_planned_secondary_span(span, sentence) {
                        Ok(span) => {
                            applied.record(admitted.source());
                            merged.push(span);
                        }
                        Err(e) => {
                            tracing::warn!(
                                lang = %lang3,
                                line_idx,
                                words,
                                error = %e,
                                "L2 morphotag: secondary merge failed; words stay L2|xxx"
                            );
                        }
                    }
                }
                tracing::info!(
                    lang = %lang3,
                    spans,
                    words = total_words,
                    "L2 morphotag: secondary dispatch succeeded"
                );
            }
            Err(e) => {
                tracing::warn!(
                    lang = %lang3,
                    words = total_words,
                    error = %e,
                    "L2 morphotag: secondary dispatch failed; words stay L2|xxx"
                );
            }
        }
    }

    let merged_words: usize = merged.iter().map(|span| span.mors().len()).sum();
    let outcome = l2::splice_l2_into_chat(chat_file, merged);
    tracing::info!(
        filename = %filename,
        spliced = outcome.spliced,
        fallback = outcome.fallback,
        unmerged = deferred_words - merged_words,
        gra_upgraded = outcome.gra_upgraded,
        "L2 morphotag: splice complete"
    );
    applied
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::stanza_registry::StanzaRegistry;
    use crate::types::worker::StanzaLanguageProcessors;

    use super::secondary_dispatch_supported;

    fn registry_with_caps() -> StanzaRegistry {
        let mut caps = BTreeMap::new();
        caps.insert(
            "eng".to_string(),
            StanzaLanguageProcessors {
                alpha2: "en".to_string(),
                processors: vec![
                    "tokenize".to_string(),
                    "pos".to_string(),
                    "lemma".to_string(),
                    "depparse".to_string(),
                ],
            },
        );
        caps.insert(
            "pan".to_string(),
            StanzaLanguageProcessors {
                alpha2: "pa".to_string(),
                processors: vec!["tokenize".to_string()],
            },
        );
        StanzaRegistry::from_capabilities(&caps)
    }

    #[test]
    fn secondary_dispatch_requires_runtime_registry() {
        let lang = crate::chat_ops::LanguageCode::new("eng").expect("valid test language code");
        assert!(
            !secondary_dispatch_supported(None, &lang),
            "without runtime Stanza capabilities, L2 dispatch must skip conservatively"
        );
    }

    #[test]
    fn secondary_dispatch_rejects_partial_processor_language() {
        let registry = registry_with_caps();
        let lang = crate::chat_ops::LanguageCode::new("pan").expect("valid test language code");
        assert!(
            !secondary_dispatch_supported(Some(&registry), &lang),
            "tokenize-only languages must not reach L2 morphotag worker bootstrap"
        );
    }

    #[test]
    fn secondary_dispatch_accepts_full_morphosyntax_language() {
        let registry = registry_with_caps();
        let lang = crate::chat_ops::LanguageCode::new("eng").expect("valid test language code");
        assert!(secondary_dispatch_supported(Some(&registry), &lang));
    }
}
