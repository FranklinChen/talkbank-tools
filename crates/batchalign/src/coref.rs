//! Server-side coreference resolution orchestrator.
//!
//! Owns the full CHAT lifecycle for coref jobs:
//! parse → collect sentences → check language → infer → inject %xcoref → serialize.
//!
//! Key differences from morphosyntax/utseg/translate:
//! - **Document-level**: Each worker item is one complete document, not one utterance.
//! - **No caching**: Results depend on full document context, per-utterance caching is meaningless.
//! - **English-only**: Non-English files are passed through unchanged.
//! - **Sparse injection**: Only utterances with actual coref chains get `%xcoref`.

use std::collections::HashMap;

use crate::api::{LanguageCode3, ReportedEngineName};
use crate::chat_ops::LanguageCode;
use crate::chat_ops::morphosyntax_ops::declared_languages;
use crate::provenance::TextStamp;
use crate::types::worker_v2::{CorefAnnotationV2, CorefItemResultV2};
use crate::worker::artifacts_v2::PreparedArtifactRuntimeV2;
use crate::worker::pool::WorkerPool;
use crate::worker::text_request_v2::{PreparedTextRequestIdsV2, build_coref_request_v2};
use crate::worker::text_result_v2::parse_coref_result_v2;
use batchalign_transform::coref::{
    ChainRef, CorefBatchItem, CorefRawAnnotation, CorefRawResponse, CorefResponse,
    apply_coref_results, collect_coref_payloads, raw_to_bracket_response,
};
use batchalign_transform::parse::{is_dummy, parse_lenient};
use batchalign_transform::validate::{ValidityLevel, validate_to_level};
use tracing::{info, warn};

use crate::api::FileStampOutcome;
use crate::error::ServerError;
use crate::infer_retry::{Cancellation, dispatch_execute_v2_with_retry};
use crate::pipeline::post_validate::PostValidated;
use crate::text_batch::{
    EngineItemFailure, ItemError, ItemFailures, TextBatchFileInput, TextBatchFileResult,
    TextBatchFileResults,
};

/// Check whether a parsed CHAT file declares English as one of its languages.
///
/// Uses the per-file `@Languages` header (via `declared_languages()`); files
/// with no `@Languages` header fall back to `eng` (BA2 parity, coref is an
/// English-only command, so the fallback is fixed). The `--lang` flag was
/// removed for BA2 parity (`~/batchalign2-master/batchalign/cli/cli.py:276`
/// has no `--lang` for coref); see the 2026-05-03 incident for why a
/// job-level lang sentinel is unsafe.
///
/// Fallible only because `LanguageCode` construction became fallible in
/// chatter 0.3.0; the fallback here is the constant `eng`, so the error
/// arm is unreachable in practice but propagated rather than panicked on.
/// The error is stringified because chatter v0.3.0 does not re-export
/// `LanguageCodeError` (upstream defect, reported).
fn file_has_english(chat_file: &crate::chat_ops::ChatFile) -> Result<bool, ServerError> {
    let fallback = LanguageCode::new(LanguageCode3::eng().as_ref()).map_err(|e| {
        ServerError::Validation(format!(
            "coref: failed to construct the 'eng' fallback language code: {e}"
        ))
    })?;
    let langs = declared_languages(chat_file, &fallback);
    Ok(langs.iter().any(|l| l.as_str() == "eng"))
}

/// One document's coreference result, admitted at the worker boundary.
///
/// Each resolved document names the engine that resolved it, which is what
/// coref provenance names: nothing is read from the worker's capability
/// report.
#[derive(Debug, Clone)]
pub(crate) enum ResolvedCoref {
    /// Resolved, by the engine the worker named on the result.
    Resolved {
        /// The bracket annotations to apply.
        response: CorefResponse,
        /// The engine that produced them.
        engine: ReportedEngineName,
    },
    /// The worker found no sentences to resolve, so no engine ran.
    NoSentences,
}

// ---------------------------------------------------------------------------
// Cross-file batch coref processing
// ---------------------------------------------------------------------------
//
// There is no per-file entry point. Every coref job runs through the batch
// path below, which now stamps provenance the way the deleted per-file path
// did; keeping a second implementation of the same lifecycle meant the two
// could disagree about what a file records, and they did.

/// Process multiple CHAT files, sending one `CorefBatchItem` per eligible file
/// in a single batched `execute_v2` call.
///
/// Returns `(filename, Ok(output_text) | Err(error_msg))` for each file.
pub(crate) async fn process_coref_batch(
    files: &[TextBatchFileInput],
    pool: &WorkerPool,
    cancellation: Cancellation<'_>,
) -> TextBatchFileResults {
    run_coref_batch_impl(files, pool, cancellation).await
}

async fn run_coref_batch_impl(
    files: &[TextBatchFileInput],
    pool: &WorkerPool,
    cancellation: Cancellation<'_>,
) -> TextBatchFileResults {
    // No language parameter. Coref is English-only (BA2 parity): per-file
    // English-ness is read from each file's `@Languages:` header
    // (`file_has_english`) and the inference language is the constant
    // `LanguageCode3::eng()`. The parameter that used to sit here existed only
    // for shared-trait symmetry with utseg/translate; it was never read, and
    // the dispatch that filled it refused every coref job trying to produce a
    // value for it. See the 2026-05-03 morphotag incident for why a job-level
    // lang must not flow through.
    let parser = crate::chat_parser();
    let mut results: TextBatchFileResults = Vec::with_capacity(files.len());

    // 1. Parse all files
    let mut parsed_files: Vec<crate::chat_ops::ChatFile> = Vec::with_capacity(files.len());
    let mut parse_error_lists: Vec<Vec<crate::chat_ops::ParseError>> =
        Vec::with_capacity(files.len());
    for file in files {
        let filename = file.filename.as_ref();
        let (chat_file, parse_errors) = parse_lenient(&parser, file.chat_text.as_ref());
        if !parse_errors.is_empty() {
            warn!(
                filename = %filename,
                num_errors = parse_errors.len(),
                "Parse errors (continuing with recovery)"
            );
        }
        parse_error_lists.push(parse_errors);
        parsed_files.push(chat_file);
    }

    // 2. Collect payloads per file (per-file English gate)
    struct FileCorefInfo {
        line_indices: Vec<usize>,
        batch_idx: usize, // index into the execute_v2 batch array
    }

    let mut eligible_files: Vec<(usize, FileCorefInfo)> = Vec::new();
    let mut batch_items: Vec<CorefBatchItem> = Vec::new();
    let mut validation_errors: Vec<Option<String>> = vec![None; files.len()];

    for (file_idx, parsed_file) in parsed_files.iter().enumerate() {
        // Skip dummy files: they pass through unchanged
        if is_dummy(parsed_file) {
            continue;
        }

        // Pre-validation gate (L1: StructurallyComplete)
        if let Err(errors) = validate_to_level(
            parsed_file,
            &parse_error_lists[file_idx],
            ValidityLevel::StructurallyComplete,
        ) {
            let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            validation_errors[file_idx] =
                Some(format!("coref pre-validation failed: {}", msgs.join("; ")));
            continue;
        }

        // Per-file English-only gate: non-English files pass through unchanged.
        // A gate failure (unreachable in practice: the fallback code is the
        // constant `eng`) is recorded as a per-file error, mirroring the
        // pre-validation gate above, rather than silently skipping the file.
        match file_has_english(parsed_file) {
            Ok(true) => {}
            Ok(false) => continue,
            Err(e) => {
                validation_errors[file_idx] = Some(e.to_string());
                continue;
            }
        }

        let collected = collect_coref_payloads(parsed_file);
        let coref_item = collected.batch_item;
        let line_indices = collected.line_indices;

        if coref_item.sentences.is_empty() {
            continue;
        }

        let batch_idx = batch_items.len();
        batch_items.push(coref_item);
        eligible_files.push((
            file_idx,
            FileCorefInfo {
                line_indices,
                batch_idx,
            },
        ));
    }

    // 3. Single batched execute_v2 call across all files. Outer Err
    //    (worker spawn / IPC / schema) marks every eligible file as
    //    failed: silently emitting no-coref output would mask the
    //    failure as success.
    let all_responses = if batch_items.is_empty() {
        Vec::new()
    } else {
        info!(
            num_items = batch_items.len(),
            "Dispatching coref execute_v2 batch"
        );

        match infer_batch(pool, &batch_items, &LanguageCode3::eng(), cancellation).await {
            Ok(responses) => responses,
            Err(e) => {
                warn!(error = %e, "Batch coref execute_v2 failed for all files");
                for (file_idx, file) in files.iter().enumerate() {
                    if let Some(ref err) = validation_errors[file_idx] {
                        results.push(TextBatchFileResult::err(
                            file.filename.clone(),
                            crate::text_batch::TextWorkflowFileError::validation(err.clone()),
                        ));
                    } else if eligible_files.iter().any(|(idx, _)| *idx == file_idx) {
                        // The control plane's classifier owns the verdict; a
                        // bare string here reported every batch break, memory
                        // pressure included, as a terminal provider failure.
                        results.push(TextBatchFileResult::err(
                            file.filename.clone(),
                            crate::text_batch::TextWorkflowFileError::from_server_error(&e),
                        ));
                    } else {
                        // Non-eligible files (dummy / non-English) had
                        // no payload in the batch, so the batch
                        // failure does not affect them.
                        results.push(TextBatchFileResult::ok(
                            file.filename.clone(),
                            PostValidated::pass_through(
                                file.chat_text.as_ref(),
                                crate::api::ReleasedCommand::Coref,
                            ),
                        ));
                    }
                }
                return results;
            }
        }
    };

    // 4. Per-file outcome map driven by per-item engine errors.
    //    Files whose item came back Err are marked failed; files whose
    //    item came back Ok have annotations applied, and the engine that
    //    resolved them is kept for that file's provenance stamp, which the
    //    batch path used to drop on the floor. ``per_file_failures`` is
    //    indexed by file_idx so step 5 can take ownership of the failure via
    //    ``.take()`` without a HashMap lookup.
    let mut per_file_failures: Vec<Option<EngineItemFailure>> = vec![None; files.len()];
    let mut per_file_engines: Vec<Option<ReportedEngineName>> = vec![None; files.len()];
    // No bounds guard: `infer_batch` refuses a count mismatch, so every
    // `batch_idx` recorded above indexes a response. The guard that used to sit
    // here was unreachable, and had it ever run it would have dropped a file's
    // result silently, which is the shape this file's admission exists to stop.
    for &(file_idx, ref info) in &eligible_files {
        match &all_responses[info.batch_idx] {
            Err(failure) => {
                per_file_failures[file_idx] = Some(failure.clone());
            }
            Ok(ResolvedCoref::NoSentences) => {}
            Ok(ResolvedCoref::Resolved {
                response: coref_resp,
                engine,
            }) => {
                per_file_engines[file_idx] = Some(engine.clone());
                let mut annotation_map: HashMap<usize, String> = HashMap::new();
                for ann in &coref_resp.annotations {
                    if ann.sentence_idx < info.line_indices.len() {
                        let line_idx = info.line_indices[ann.sentence_idx];
                        annotation_map.insert(line_idx, ann.annotation.clone());
                    }
                }
                if !annotation_map.is_empty() {
                    apply_coref_results(&mut parsed_files[file_idx], &annotation_map);
                }
            }
        }
    }

    // 5. Serialize all files
    for (file_idx, file) in files.iter().enumerate() {
        let filename = file.filename.as_ref();
        // Skip files that failed pre-validation
        if let Some(ref err) = validation_errors[file_idx] {
            results.push(TextBatchFileResult::err(
                file.filename.clone(),
                crate::text_batch::TextWorkflowFileError::validation(err.clone()),
            ));
            continue;
        }

        // Per-item engine failure: file marked failed with typed
        // ItemErrors variant so the user sees the engine reason.
        if let Some(failure) = per_file_failures[file_idx].take() {
            results.push(TextBatchFileResult::err(
                file.filename.clone(),
                crate::text_batch::TextWorkflowFileError::item_errors(
                    "coref",
                    ItemFailures::of_one(ItemError {
                        item_index: 0,
                        failure,
                    }),
                ),
            ));
            continue;
        }

        // Provenance, before the gate so the proof covers the bytes that are
        // written: the engine THIS file's own result named. Coref is
        // English-only, so the stamp's language is the constant `eng` rather
        // than any job-level value (see the 2026-05-03 incident). A file that
        // was never eligible (dummy, or not English) had no coref run, so no
        // stamp question arises for it.
        let mut stamp = FileStampOutcome::Unrecorded;
        if eligible_files.iter().any(|(idx, _)| *idx == file_idx) {
            let command = crate::api::ReleasedCommand::Coref.to_string();
            match crate::provenance::result_named_provenance(
                crate::provenance::ResultNamedCommand::Coref,
                &LanguageCode3::eng(),
                per_file_engines[file_idx].as_ref(),
            ) {
                TextStamp::Stamped(comment) => {
                    crate::provenance::inject_provenance(&mut parsed_files[file_idx], &comment);
                    stamp = FileStampOutcome::Stamped { command };
                }
                TextStamp::NotStamped(reason) => {
                    info!(
                        filename = %filename,
                        reason = %reason,
                        "coref wrote no provenance stamp"
                    );
                    stamp = FileStampOutcome::NotStamped {
                        command,
                        reason: reason.to_string(),
                    };
                }
            }
        }

        // Fail-closed post-validation, per file: a file whose output fails
        // the gate is reported as a validation failure and never written.
        // The rest of the cross-file batch is unaffected.
        match PostValidated::gate(
            &parsed_files[file_idx],
            ValidityLevel::StructurallyComplete,
            crate::api::ReleasedCommand::Coref,
        ) {
            Ok(output) => results.push(TextBatchFileResult::ok_stamped(
                file.filename.clone(),
                output,
                stamp,
            )),
            Err(failure) => {
                warn!(filename = %filename, error = %failure, "coref output refused");
                results.push(TextBatchFileResult::err(file.filename.clone(), failure));
            }
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Send one or more documents to a worker for coref inference via batched
/// `execute_v2`.
///
/// Returns one `Result<ResolvedCoref, String>` per item. Per-item engine
/// failures are propagated as `Err(message)` so callers can attribute the
/// failure back to the affected file rather than silently emitting an
/// empty (no-coref) response that looks like success.
async fn infer_batch(
    pool: &WorkerPool,
    items: &[CorefBatchItem],
    lang: &LanguageCode3,
    cancellation: Cancellation<'_>,
) -> Result<Vec<Result<ResolvedCoref, EngineItemFailure>>, ServerError> {
    let artifacts = PreparedArtifactRuntimeV2::new("coref_v2").map_err(|error| {
        ServerError::Validation(format!(
            "failed to create coref V2 artifact runtime: {error}"
        ))
    })?;
    let request_ids = PreparedTextRequestIdsV2::for_task("coref");
    let request =
        build_coref_request_v2(artifacts.store(), &request_ids, lang, items).map_err(|error| {
            ServerError::Validation(format!("failed to build coref V2 worker request: {error}"))
        })?;

    let response = dispatch_execute_v2_with_retry(pool, lang, &request, cancellation).await?;
    let result = parse_coref_result_v2(response)
        .map_err(|error| ServerError::Validation(format!("invalid coref V2 result: {error}")))?;
    if result.items.len() != items.len() {
        return Err(ServerError::Validation(format!(
            "coref V2 returned {} items for {} requests",
            result.items.len(),
            items.len()
        )));
    }

    // Each item is one of three outcomes; a resolved one carries its engine.
    Ok(result
        .items
        .into_iter()
        .map(|item| match item {
            CorefItemResultV2::Resolved {
                annotations,
                engine,
            } => Ok(ResolvedCoref::Resolved {
                response: coref_response_from_v2_annotations(&annotations),
                engine,
            }),
            CorefItemResultV2::NoSentences => Ok(ResolvedCoref::NoSentences),
            CorefItemResultV2::Failed { error } => Err(EngineItemFailure::EngineReported(error)),
        })
        .collect())
}

/// Convert one resolved V2 document's annotations into the established Rust
/// response.
fn coref_response_from_v2_annotations(annotations: &[CorefAnnotationV2]) -> CorefResponse {
    let raw = CorefRawResponse {
        annotations: annotations
            .iter()
            .map(|annotation| CorefRawAnnotation {
                sentence_idx: annotation.sentence_idx,
                words: annotation
                    .words
                    .iter()
                    .map(|word_refs| {
                        word_refs
                            .iter()
                            .map(|chain_ref| ChainRef {
                                chain_id: chain_ref.chain_id,
                                is_start: chain_ref.is_start,
                                is_end: chain_ref.is_end,
                            })
                            .collect()
                    })
                    .collect(),
            })
            .collect(),
    };
    raw_to_bracket_response(&raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use batchalign_transform::parse::TreeSitterParser;

    #[test]
    fn test_file_has_english_with_eng_languages() {
        let parser = TreeSitterParser::new().unwrap();
        let chat = include_str!("../../../test-fixtures/eng_hello_world.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        assert!(file_has_english(&chat_file).expect("eng fallback code must construct"));
    }

    #[test]
    fn test_file_has_english_with_spa_languages() {
        let parser = TreeSitterParser::new().unwrap();
        let chat = include_str!("../../../test-fixtures/spa_chi_hola_mundo.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        assert!(!file_has_english(&chat_file).expect("eng fallback code must construct"));
    }

    #[test]
    fn test_file_has_english_no_languages_header_uses_eng_fallback() {
        // BA2 parity: coref's English-only fallback for missing @Languages is
        // hardcoded `eng` (--lang was removed from the CLI). A file without
        // an @Languages header is treated as English.
        let parser = TreeSitterParser::new().unwrap();
        let chat = include_str!("../../../test-fixtures/eng_hello_world_no_languages.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        assert!(file_has_english(&chat_file).expect("eng fallback code must construct"));
    }

    #[test]
    fn test_file_has_english_multilingual_with_eng() {
        let parser = TreeSitterParser::new().unwrap();
        // File declares both eng and spa, should be considered English
        let chat = include_str!("../../../test-fixtures/eng_spa_bilingual_hello_world.cha");
        let (chat_file, _) = parse_lenient(&parser, chat);
        assert!(file_has_english(&chat_file).expect("eng fallback code must construct"));
    }
}
