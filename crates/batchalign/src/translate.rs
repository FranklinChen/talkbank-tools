//! Server-side translation orchestrator.
//!
//! Owns the full CHAT lifecycle for translate jobs:
//! parse → collect payloads → infer → inject %xtra → serialize.
//!
//! Python workers receive only `(text) → TranslateResponse` via the infer protocol —
//! pure Google Translate / Seamless inference with zero CHAT awareness.

use std::collections::HashMap;

use async_trait::async_trait;

use crate::api::{ChatText, EngineVersion, LanguageCode3};
use crate::chat_ops::{ChatFile, LanguageCode};
use crate::worker::artifacts_v2::PreparedArtifactRuntimeV2;
use crate::worker::pool::WorkerPool;
use crate::worker::text_request_v2::{PreparedTextRequestIdsV2, build_translate_request_v2};
use crate::worker::text_result_v2::parse_translate_result_v2;
use talkbank_transform::translate::{
    TranslateBatchItem, TranslateResponse, apply_translate_results, chat_punct_chars,
    collect_translate_payloads, postprocess_translation, preprocess_for_translate,
};
use talkbank_transform::validate::ValidityLevel;
use tracing::info;

use crate::error::ServerError;
use crate::infer_retry::dispatch_execute_v2_with_retry;
use crate::pipeline::PipelineServices;
use crate::pipeline::text_infer::{
    TextBatchHooks, TextPipelineHooks, run_text_batch_pipeline, run_text_pipeline,
};
use crate::text_batch::{
    TextBatchFileInput, TextBatchFileResults, TextBatchOperation, TextBatchWorkflow,
    TextBatchWorkflowRequest, TextPerFileWorkflowRequest,
};

/// Command-specific parameters for the translate workflow family.
///
/// Retained as a zero-field struct so the `TextBatchOperation` shape stays
/// symmetric with morphotag/utseg; may grow again when translate acquires
/// real command-specific options.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TranslateWorkflowParams;

/// Typed workflow operation for translate.
pub(crate) struct TranslateOperation;

/// Trait-oriented workflow wrapper for translate.
pub(crate) type TranslateWorkflow = TextBatchWorkflow<TranslateOperation>;

#[async_trait]
impl TextBatchOperation for TranslateOperation {
    type Shared<'a>
        = PipelineServices<'a>
    where
        Self: 'a;

    type Params<'a>
        = TranslateWorkflowParams
    where
        Self: 'a;

    async fn run_single(
        chat_text: ChatText<'_>,
        lang: &LanguageCode3,
        shared: Self::Shared<'_>,
        _params: Self::Params<'_>,
    ) -> Result<String, ServerError> {
        run_translate_impl(
            chat_text.as_ref(),
            lang,
            shared.pool,
            shared.cache,
            shared.engine_version,
        )
        .await
    }

    async fn run_batch(
        files: &[TextBatchFileInput],
        lang: &LanguageCode3,
        shared: Self::Shared<'_>,
        _params: Self::Params<'_>,
    ) -> TextBatchFileResults {
        run_translate_batch_impl(files, lang, shared.pool).await
    }
}

// ---------------------------------------------------------------------------
// Per-file translate processing
// ---------------------------------------------------------------------------

/// Process a single CHAT file through the translation pipeline.
///
/// Returns the serialized CHAT text with `%xtra` tiers injected.
pub async fn process_translate(
    chat_text: &str,
    lang: &LanguageCode3,
    pool: &WorkerPool,
    cache: &crate::cache::UtteranceCache,
    engine_version: &EngineVersion,
) -> Result<String, ServerError> {
    TranslateWorkflow::new()
        .run_per_file(TextPerFileWorkflowRequest {
            chat_text: ChatText::from(chat_text),
            lang,
            shared: PipelineServices::new(pool, cache, engine_version),
            params: TranslateWorkflowParams,
        })
        .await
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
    cache: &crate::cache::UtteranceCache,
    engine_version: &EngineVersion,
) -> TextBatchFileResults {
    TranslateWorkflow::new()
        .run_batch_files(TextBatchWorkflowRequest {
            files,
            lang,
            shared: PipelineServices::new(pool, cache, engine_version),
            params: TranslateWorkflowParams,
        })
        .await
}

async fn run_translate_impl(
    chat_text: &str,
    lang: &LanguageCode3,
    pool: &WorkerPool,
    cache: &crate::cache::UtteranceCache,
    engine_version: &EngineVersion,
) -> Result<String, ServerError> {
    run_text_pipeline(
        chat_text,
        lang,
        PipelineServices::new(pool, cache, engine_version),
        TextPipelineHooks {
            command: "translate",
            validity: ValidityLevel::StructurallyComplete,
            collect: collect_translate_payloads,
            integrate: integrate_translations,
            apply: apply_translate_results,
        },
        infer_batch,
    )
    .await
}

async fn run_translate_batch_impl(
    files: &[TextBatchFileInput],
    lang: &LanguageCode3,
    pool: &WorkerPool,
) -> TextBatchFileResults {
    run_text_batch_pipeline(
        files,
        lang,
        pool,
        TextBatchHooks {
            command: "translate",
            validity: ValidityLevel::StructurallyComplete,
            collect: collect_translate_payloads,
            apply: apply_translate_file,
        },
        infer_batch,
    )
    .await
}

/// Apply translate responses for one file, skipping items with an
/// empty translation.
fn apply_translate_file(
    chat_file: &mut ChatFile,
    items: &[(usize, TranslateBatchItem)],
    responses: &[TranslateResponse],
) {
    let mut translation_map: HashMap<usize, String> = HashMap::new();
    for ((line_idx, _item), resp) in items.iter().zip(responses.iter()) {
        if !resp.translation.is_empty() {
            translation_map.insert(*line_idx, resp.translation.clone());
        }
    }
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
/// Applies pre-processing (Chinese space removal) before sending to Python
/// and post-processing (punct spacing, quote normalization) on the raw response.
async fn infer_batch(
    pool: &WorkerPool,
    items: &[(usize, TranslateBatchItem)],
    lang: &LanguageCode3,
) -> Result<Vec<Result<TranslateResponse, String>>, ServerError> {
    let src_lang_code = LanguageCode::new(lang.as_ref());

    // Pre-process text before sending to Python
    let preprocessed_items: Vec<TranslateBatchItem> = items
        .iter()
        .map(|(_, item)| TranslateBatchItem {
            text: preprocess_for_translate(&item.text, &src_lang_code),
        })
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

    let response = dispatch_execute_v2_with_retry(pool, lang, &request).await?;
    let result = parse_translate_result_v2(&response).map_err(|error| {
        ServerError::Validation(format!("invalid translate V2 result: {error}"))
    })?;

    let punct_strings = chat_punct_chars();
    let punct_refs: Vec<&str> = punct_strings.iter().map(|s| s.as_str()).collect();
    parse_translate_item_results(&result.items, items.len(), &punct_refs)
}

/// Convert one batch of `TranslationItemResultV2` into per-item
/// `Result<TranslateResponse, String>`.
///
/// Per-item engine failures (network error, rate-limit, model error)
/// and protocol violations (worker returned neither error nor raw
/// translation) are propagated as the inner `Err(String)` so the
/// driver can attribute them back to the source file and mark only
/// that file as failed. Length mismatches are surfaced as the outer
/// `Err(ServerError)` because they're a batch-level protocol bug,
/// not a per-item failure.
fn parse_translate_item_results(
    items: &[crate::types::worker_v2::TranslationItemResultV2],
    request_count: usize,
    punct_refs: &[&str],
) -> Result<Vec<Result<TranslateResponse, String>>, ServerError> {
    if items.len() != request_count {
        return Err(ServerError::Validation(format!(
            "translate V2 returned {} items for {request_count} requests",
            items.len(),
        )));
    }

    let mut translate_responses = Vec::with_capacity(items.len());
    for item_result in items.iter() {
        if let Some(error) = &item_result.error {
            translate_responses.push(Err(error.clone()));
            continue;
        }

        if let Some(raw_translation) = &item_result.raw_translation {
            let processed = postprocess_translation(raw_translation, punct_refs);
            translate_responses.push(Ok(TranslateResponse {
                translation: processed,
            }));
            continue;
        }

        // Protocol violation — Python worker returned neither error nor
        // raw_translation. Treat as a per-item failure so the affected
        // file fails rather than silently producing missing %xtra tiers.
        translate_responses.push(Err(
            "translate V2 returned neither error nor raw_translation".to_owned(),
        ));
    }

    Ok(translate_responses)
}

fn integrate_translations(
    translation_map: &mut HashMap<usize, String>,
    misses: &[(usize, TranslateBatchItem)],
    responses: &[TranslateResponse],
) {
    // ``infer_batch`` only emits successful translations now (per-item
    // engine failures are propagated up the call chain as typed errors,
    // not silently dropped as empty strings). Any caller that reaches
    // this point with an empty translation indicates a programmer error
    // in the success-path construction, so we simply insert as-is.
    for ((line_idx, _item), resp) in misses.iter().zip(responses.iter()) {
        translation_map.insert(*line_idx, resp.translation.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::worker_v2::TranslationItemResultV2;

    fn punct_strings() -> Vec<String> {
        chat_punct_chars()
    }

    fn punct_refs<'a>(strs: &'a [String]) -> Vec<&'a str> {
        strs.iter().map(|s| s.as_str()).collect()
    }

    /// Helper to keep the test signature concise.
    fn parse_items(
        items: &[TranslationItemResultV2],
        request_count: usize,
    ) -> Result<Vec<Result<TranslateResponse, String>>, ServerError> {
        let strs = punct_strings();
        let refs = punct_refs(&strs);
        parse_translate_item_results(items, request_count, &refs)
    }

    #[test]
    fn parse_items_all_success_returns_one_ok_per_item() {
        // Note: CHAT punctuation-spacing postprocessing inserts a
        // space before terminator punctuation; see chat_punct_chars
        // and the parse_items_applies_postprocessing test below.
        let items = vec![
            TranslationItemResultV2 {
                raw_translation: Some("Hello world.".into()),
                error: None,
            },
            TranslationItemResultV2 {
                raw_translation: Some("How are you?".into()),
                error: None,
            },
        ];
        let parsed = parse_items(&items, 2).unwrap();
        assert_eq!(parsed.len(), 2);
        assert!(parsed[0].is_ok());
        assert!(parsed[1].is_ok());
        assert_eq!(parsed[0].as_ref().unwrap().translation, "Hello world .");
        assert_eq!(parsed[1].as_ref().unwrap().translation, "How are you ?");
    }

    #[test]
    fn parse_items_per_item_error_propagates_as_inner_err() {
        // The bug the whole Option-C fix targets: Google fails for one
        // utterance, the worker reports it via item_result.error, and
        // the Rust side must NOT silently emit an empty translation.
        // It must surface the failure as an inner Err so the driver
        // can attribute it to the source file and mark that file
        // failed.
        let items = vec![
            TranslationItemResultV2 {
                raw_translation: Some("Hello world.".into()),
                error: None,
            },
            TranslationItemResultV2 {
                raw_translation: None,
                error: Some("Translation failed: ConnectionResetError".into()),
            },
        ];
        let parsed = parse_items(&items, 2).unwrap();
        assert!(parsed[0].is_ok());
        match &parsed[1] {
            Err(message) => assert!(
                message.contains("ConnectionResetError"),
                "expected error string to carry the engine reason, got: {message}"
            ),
            Ok(_) => panic!("per-item engine failure must propagate as inner Err"),
        }
    }

    #[test]
    fn parse_items_protocol_violation_propagates_as_inner_err() {
        // Worker returned neither error NOR raw_translation. That is
        // a protocol bug, not user input — but the orchestrator still
        // must surface it as a failure rather than emit an empty
        // translation that quietly drops out of the output.
        let items = vec![TranslationItemResultV2 {
            raw_translation: None,
            error: None,
        }];
        let parsed = parse_items(&items, 1).unwrap();
        match &parsed[0] {
            Err(message) => assert!(
                message.contains("neither")
                    || message.contains("no raw_translation")
                    || message.contains("protocol"),
                "expected protocol-violation error message, got: {message}"
            ),
            Ok(_) => panic!("protocol violation must propagate as inner Err"),
        }
    }

    #[test]
    fn parse_items_count_mismatch_is_outer_err() {
        let items = vec![TranslationItemResultV2 {
            raw_translation: Some("Hello.".into()),
            error: None,
        }];
        let err = parse_items(&items, 2).unwrap_err();
        assert!(format!("{err}").contains("returned 1 items for 2 requests"));
    }

    #[test]
    fn parse_items_applies_postprocessing_to_successful_translations() {
        let items = vec![TranslationItemResultV2 {
            raw_translation: Some("Hello world.".into()),
            error: None,
        }];
        let parsed = parse_items(&items, 1).unwrap();
        let translation = &parsed[0].as_ref().unwrap().translation;
        // Postprocessing inserts a space before the period (CHAT
        // punctuation-spacing convention).
        assert!(
            translation.ends_with(" ."),
            "expected postprocessed punctuation spacing, got: {translation:?}"
        );
    }
}
