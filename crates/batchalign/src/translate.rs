//! Server-side translation orchestrator.
//!
//! Owns the full CHAT lifecycle for translate jobs:
//! parse → collect payloads → infer → inject %xtra → serialize.
//!
//! Python workers receive only the rendered source text of each item and
//! return the engine's raw translation: pure inference, with zero CHAT
//! awareness.
//!
//! # What is sent
//!
//! What was spoken: a [`TranslationSource`] per utterance, rendered once at
//! the worker boundary (see `batchalign_transform::translate`). Every word the
//! speaker produced travels, retraced words and filled pauses included, and so
//! does the terminator, so a question reaches the engine as a question.
//!
//! # Engine identity
//!
//! Every translated item names the engine that translated it, and the
//! provenance comment names the engines behind the translations that were
//! applied. Nothing is read from the worker's capability report, so a
//! translate job never fails for want of a pre-dispatch identity.
//!
//! # Empty translations
//!
//! A translation with nothing to apply is refused where it is admitted, as a
//! typed per-item failure that names the engine, so the file fails visibly
//! instead of being written with a `%xtra` tier silently missing. The verdict
//! is terminal: the same request gets the same answer from the same engine, so
//! the remedy is a different engine or different options, not a retry.

use std::collections::HashMap;

use crate::api::{InvalidStampSafeText, LanguageCode3, ReportedEngineName};
use crate::chat_ops::{ChatFile, LanguageCode};
use crate::provenance::{ResultNamedCommand, TextStamp};
use crate::types::worker_v2::TranslationItemResultV2;
use crate::worker::artifacts_v2::PreparedArtifactRuntimeV2;
use crate::worker::pool::WorkerPool;
use crate::worker::text_request_v2::{PreparedTextRequestIdsV2, build_translate_request_v2};
use crate::worker::text_result_v2::parse_translate_result_v2;
use batchalign_transform::translate::{
    EmptyTranslation, TranslateBatchItem, TranslationSource, TranslationText, WritingSystem,
    apply_translate_results, chat_punct_chars, collect_translate_payloads, postprocess_translation,
};
use batchalign_transform::validate::ValidityLevel;
use tracing::info;

use crate::error::ServerError;
use crate::infer_retry::{Cancellation, dispatch_execute_v2_with_retry};
use crate::pipeline::text_infer::{TextBatchHooks, run_text_batch_pipeline};
use crate::text_batch::{ItemFailure, TextBatchFileInput, TextBatchFileResults};

/// The translate-specific per-item failure: the engine answered, with nothing
/// that can be written.
///
/// Translate's own type, so utseg, coref and morphotag cannot represent it:
/// their per-item failure is [`crate::text_batch::EngineItemFailure`], whose
/// command-specific variant is uninhabited.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmptyTranslationFailure {
    /// The engine that produced the empty translation.
    engine: ReportedEngineName,
}

impl std::fmt::Display for EmptyTranslationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} returned an empty translation; try a different --translate-engine \
             or options and run the file again",
            self.engine
        )
    }
}

/// What one translate item can fail with.
pub(crate) type TranslateItemFailure = ItemFailure<EmptyTranslationFailure>;

/// One translation item admitted at the worker boundary.
///
/// A translated item holds a [`TranslationText`], which exists only when there
/// is something to apply, so no later stage has to ask whether a translation
/// is usable.
#[derive(Debug, Clone)]
pub(crate) enum AdmittedTranslation {
    /// Translated, by the engine the worker named on the result.
    Translated {
        /// The postprocessed translation.
        text: TranslationText,
        /// The engine that produced it.
        engine: ReportedEngineName,
    },
    /// The input was blank, so nothing was translated and no engine ran.
    BlankInput,
}

impl AdmittedTranslation {
    /// The translation to apply, if there is one.
    fn translation(&self) -> Option<&TranslationText> {
        match self {
            Self::Translated { text, .. } => Some(text),
            Self::BlankInput => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-file batch translate processing
// ---------------------------------------------------------------------------

/// Process multiple CHAT files, pooling all payloads into a single
/// `batch_infer` call for maximum throughput.
///
/// Returns `(filename, Ok(output_text) | Err(error_msg))` for each file.
pub(crate) async fn process_translate_batch(
    files: &[TextBatchFileInput],
    lang: &LanguageCode3,
    pool: &WorkerPool,
    cancellation: Cancellation<'_>,
) -> TextBatchFileResults {
    run_text_batch_pipeline(
        files,
        lang,
        pool,
        TextBatchHooks {
            command: crate::api::ReleasedCommand::Translate,
            validity: ValidityLevel::StructurallyComplete,
            collect: collect_translate_payloads,
            apply: apply_translate_file,
            provenance: translation_provenance,
        },
        async move |pool, items, lang| infer_batch(pool, items, lang, cancellation).await,
    )
    .await
}

/// The translate stamp, named by the engines of the translations applied.
/// Never `Err`: each engine is a reported engine name, already stamp-safe; the
/// `Result` is the shape every text command's provenance hook shares.
fn translation_provenance(
    lang: &LanguageCode3,
    responses: &[AdmittedTranslation],
) -> Result<TextStamp, InvalidStampSafeText> {
    Ok(crate::provenance::result_named_provenance(
        ResultNamedCommand::Translate,
        lang,
        responses.iter().filter_map(|response| match response {
            AdmittedTranslation::Translated { engine, .. } => Some(engine),
            AdmittedTranslation::BlankInput => None,
        }),
    ))
}

/// Apply translate responses for one file. A blank input has nothing to apply;
/// a translation that had nothing to apply never reached this point, because
/// admission refused it.
fn apply_translate_file(
    chat_file: &mut ChatFile,
    items: &[(usize, TranslationSource)],
    responses: &[AdmittedTranslation],
) {
    let translation_map: HashMap<usize, TranslationText> = items
        .iter()
        .zip(responses)
        .filter_map(|((line_idx, _item), response)| {
            response
                .translation()
                .map(|translation| (*line_idx, translation.clone()))
        })
        .collect();
    if !translation_map.is_empty() {
        apply_translate_results(chat_file, &translation_map);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Send batch items to a worker for translation inference via batched
/// `execute_v2`.
///
/// Renders each [`TranslationSource`] for the source language's script (the
/// one place source text is built), and post-processes the raw response (punct
/// spacing, quote normalization) before admitting it.
async fn infer_batch(
    pool: &WorkerPool,
    items: &[(usize, TranslationSource)],
    lang: &LanguageCode3,
    cancellation: Cancellation<'_>,
) -> Result<Vec<Result<AdmittedTranslation, TranslateItemFailure>>, ServerError> {
    // Fallible in chatter 0.3.0; stringified error (`LanguageCodeError` not
    // re-exported upstream).
    let src_lang_code = LanguageCode::new(lang.as_ref()).map_err(|e| {
        ServerError::Validation(format!(
            "translate: invalid source language code {:?}: {e}",
            lang.as_ref()
        ))
    })?;

    // Render what was spoken, in the writing system this language uses.
    let script = WritingSystem::of_language(&src_lang_code);
    let preprocessed_items: Vec<TranslateBatchItem> = items
        .iter()
        .map(|(_, source)| source.render(script))
        .collect();
    let artifacts = PreparedArtifactRuntimeV2::new("translate_v2").map_err(|error| {
        ServerError::Validation(format!(
            "failed to create translate V2 artifact runtime: {error}"
        ))
    })?;
    let request_ids = PreparedTextRequestIdsV2::for_task("translate");
    let target_lang = LanguageCode3::eng();
    let request = build_translate_request_v2(
        artifacts.store(),
        &request_ids,
        lang,
        &target_lang,
        &preprocessed_items,
    )
    .map_err(|error| {
        ServerError::Validation(format!(
            "failed to build translate V2 worker request: {error}"
        ))
    })?;

    info!(
        num_items = items.len(),
        lang = %lang,
        "Dispatching translate execute_v2 batch"
    );

    let response = dispatch_execute_v2_with_retry(pool, lang, &request, cancellation).await?;
    let result = parse_translate_result_v2(response).map_err(|error| {
        ServerError::Validation(format!("invalid translate V2 result: {error}"))
    })?;

    let punct_strings = chat_punct_chars();
    let punct_refs: Vec<&str> = punct_strings.iter().map(|s| s.as_str()).collect();
    parse_translate_item_results(result.items, items.len(), &punct_refs)
}

/// Admit one batch of `TranslationItemResultV2` into per-item results.
///
/// A per-item failure is the inner `Err(TranslateItemFailure)`, so the driver
/// can attribute it to the source file and mark only that file as failed: an
/// engine's own failure (network error, rate-limit, model error), or a
/// translation with nothing to apply, which is refused here rather than
/// dropped at injection. A length mismatch is the outer `Err(ServerError)`,
/// because it is a batch-level protocol bug, not a per-item failure.
fn parse_translate_item_results(
    items: Vec<TranslationItemResultV2>,
    request_count: usize,
    punct_refs: &[&str],
) -> Result<Vec<Result<AdmittedTranslation, TranslateItemFailure>>, ServerError> {
    if items.len() != request_count {
        return Err(ServerError::Validation(format!(
            "translate V2 returned {} items for {request_count} requests",
            items.len(),
        )));
    }

    Ok(items
        .into_iter()
        .map(|item| match item {
            TranslationItemResultV2::Translated {
                raw_translation,
                engine,
            } => {
                let postprocessed = postprocess_translation(&raw_translation, punct_refs);
                match TranslationText::admit(&postprocessed) {
                    Ok(text) => Ok(AdmittedTranslation::Translated { text, engine }),
                    // The engine answered with nothing to write. This used to
                    // be skipped at injection, so the file was written with
                    // the `%xtra` tier missing and nothing said so.
                    Err(EmptyTranslation) => {
                        Err(ItemFailure::Command(EmptyTranslationFailure { engine }))
                    }
                }
            }
            TranslationItemResultV2::BlankInput => Ok(AdmittedTranslation::BlankInput),
            TranslationItemResultV2::Failed { error } => Err(ItemFailure::EngineReported(error)),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn punct_strings() -> Vec<String> {
        chat_punct_chars()
    }

    fn punct_refs(strs: &[String]) -> Vec<&str> {
        strs.iter().map(|s| s.as_str()).collect()
    }

    fn engine() -> ReportedEngineName {
        ReportedEngineName::try_from("googletrans-v1").expect("valid engine name")
    }

    fn translated(text: &str) -> TranslationItemResultV2 {
        TranslationItemResultV2::Translated {
            raw_translation: text.into(),
            engine: engine(),
        }
    }

    /// Helper to keep the test signature concise.
    fn parse_items(
        items: Vec<TranslationItemResultV2>,
        request_count: usize,
    ) -> Result<Vec<Result<AdmittedTranslation, TranslateItemFailure>>, ServerError> {
        let strs = punct_strings();
        let refs = punct_refs(&strs);
        parse_translate_item_results(items, request_count, &refs)
    }

    fn translation_of(result: &Result<AdmittedTranslation, TranslateItemFailure>) -> Option<&str> {
        result
            .as_ref()
            .ok()
            .and_then(AdmittedTranslation::translation)
            .map(TranslationText::as_str)
    }

    #[test]
    fn parse_items_all_success_returns_one_translation_per_item() {
        // Note: CHAT punctuation-spacing postprocessing inserts a
        // space before terminator punctuation; see chat_punct_chars.
        let parsed = parse_items(
            vec![translated("Hello world."), translated("How are you?")],
            2,
        )
        .expect("matching count");
        assert_eq!(parsed.len(), 2);
        assert_eq!(translation_of(&parsed[0]), Some("Hello world ."));
        assert_eq!(translation_of(&parsed[1]), Some("How are you ?"));
    }

    #[test]
    fn parse_items_per_item_error_propagates_as_inner_err() {
        // Google fails for one utterance and the worker reports it as that
        // item's failure: the Rust side must NOT silently emit an empty
        // translation, but surface an inner Err the driver attributes to the
        // source file.
        let parsed = parse_items(
            vec![
                translated("Hello world."),
                TranslationItemResultV2::Failed {
                    error: "Translation failed: ConnectionResetError".into(),
                },
            ],
            2,
        )
        .expect("matching count");
        assert!(parsed[0].is_ok());
        match &parsed[1] {
            Err(failure) => assert!(
                failure.to_string().contains("ConnectionResetError"),
                "expected the failure to carry the engine reason, got: {failure}"
            ),
            Ok(_) => panic!("per-item engine failure must propagate as inner Err"),
        }
    }

    /// RED FIRST (W2): an engine result with nothing to apply is refused as a
    /// typed per-item failure. Both paths used to accept it, the batch one by
    /// skipping the utterance and the single-file one by writing an empty
    /// `%xtra` tier, and neither said anything had gone wrong. The refusal
    /// covers the punctuation the renderer itself sends, so an engine echoing
    /// a terminator cannot produce a tier either.
    #[test]
    fn parse_items_refuses_a_translation_with_nothing_to_apply() {
        for empty in ["", "   ", ".", "?", "。"] {
            let parsed = parse_items(vec![translated(empty)], 1).expect("matching count");
            let failure = parsed[0]
                .as_ref()
                .expect_err("an empty translation must not be admitted");
            assert_eq!(
                failure,
                &ItemFailure::Command(EmptyTranslationFailure { engine: engine() }),
                "{empty:?} must be refused as an empty translation"
            );
            let rendered = failure.to_string();
            assert!(
                rendered.contains("googletrans-v1"),
                "the failure must name the engine, got: {rendered}"
            );
            assert!(
                rendered.contains("--translate-engine"),
                "the failure must name the remedy, got: {rendered}"
            );
        }
    }

    /// The file's verdict for an empty translation is terminal: the same
    /// request gets the same answer, so the control plane must not be told to
    /// expect a different one.
    #[test]
    fn an_empty_translation_fails_the_file_terminally() {
        let failure = crate::text_batch::TextWorkflowFileError::item_errors(
            "translate",
            crate::text_batch::ItemFailures::of_one(crate::text_batch::ItemError {
                item_index: 0,
                failure: ItemFailure::Command(EmptyTranslationFailure { engine: engine() }),
            }),
        );
        assert_eq!(
            failure.category(),
            crate::scheduling::FailureCategory::ProviderTerminal
        );
    }

    /// A blank input is admitted as its own outcome: nothing to apply, and no
    /// engine to name.
    #[test]
    fn parse_items_blank_input_translates_nothing_and_names_no_engine() {
        let parsed =
            parse_items(vec![TranslationItemResultV2::BlankInput], 1).expect("matching count");
        assert!(matches!(parsed[0], Ok(AdmittedTranslation::BlankInput)));
        let applied: Vec<AdmittedTranslation> = parsed
            .into_iter()
            .map(|item| item.expect("admitted"))
            .collect();
        assert!(matches!(
            translation_provenance(&LanguageCode3::eng(), &applied).expect("valid fields"),
            TextStamp::NotStamped(crate::provenance::NoStampReason::NothingApplied)
        ));
    }

    #[test]
    fn parse_items_count_mismatch_is_outer_err() {
        let err = parse_items(vec![translated("Hello.")], 2).unwrap_err();
        assert!(format!("{err}").contains("returned 1 items for 2 requests"));
    }

    /// The translate stamp names the engine the translations named.
    #[test]
    fn translation_provenance_names_the_engine_of_the_applied_results() {
        let applied: Vec<AdmittedTranslation> = parse_items(vec![translated("Hello.")], 1)
            .expect("matching count")
            .into_iter()
            .map(|item| item.expect("admitted"))
            .collect();
        let TextStamp::Stamped(comment) =
            translation_provenance(&LanguageCode3::eng(), &applied).expect("valid fields")
        else {
            panic!("a translation names its engine");
        };
        let stamp = comment.format();
        assert!(
            stamp.starts_with("[fc-ba3 translate | engine=googletrans-v1 ; lang=eng | "),
            "{stamp}"
        );
    }
}
