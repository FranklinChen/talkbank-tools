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
use batchalign_transform::parse::is_dummy;
use tracing::info;

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
        /// Raw sparse annotations, admitted against this request before application.
        annotations: Vec<CorefAnnotationV2>,
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
    run_coref_batch_impl(files, pool, async move |pool, items| {
        infer_batch(pool, items, &LanguageCode3::eng(), cancellation).await
    })
    .await
}

/// The source and its exact collected request stay paired until application.
struct PendingCoref {
    input: talkbank_model::validation::ValidChatFile,
    sentences: Vec<CorefSentenceBinding>,
    batch_idx: usize,
}

struct CorefSentenceBinding {
    line_idx: usize,
    word_count: usize,
}

fn sentence_bindings(
    payload: &batchalign_transform::coref::CorefPayloadCollection,
) -> Vec<CorefSentenceBinding> {
    // The collector produces both entries in the same utterance visit.
    payload
        .line_indices
        .iter()
        .zip(&payload.batch_item.sentences)
        .map(|(&line_idx, words)| CorefSentenceBinding {
            line_idx,
            word_count: words.len(),
        })
        .collect()
}

struct AppliedCoref {
    document: crate::chat_ops::ChatFile,
    engine: ReportedEngineName,
}

fn coref_protocol_error(message: impl Into<String>) -> ServerError {
    ServerError::Worker(crate::worker::error::WorkerError::Protocol(message.into()))
}

impl PendingCoref {
    fn complete(self, result: ResolvedCoref) -> Result<AppliedCoref, ServerError> {
        let ResolvedCoref::Resolved {
            annotations,
            engine,
        } = result
        else {
            return Err(coref_protocol_error(
                "coref returned no_sentences for a dispatched non-empty document",
            ));
        };
        let mut annotation_map = HashMap::new();
        for annotation in annotations {
            let Some(binding) = self.sentences.get(annotation.sentence_idx) else {
                return Err(coref_protocol_error(format!(
                    "coref annotation sentence {} is outside its request",
                    annotation.sentence_idx,
                )));
            };
            if annotation.words.len() != binding.word_count {
                return Err(coref_protocol_error(format!(
                    "coref annotation sentence {} has {} word positions for {} requested words",
                    annotation.sentence_idx,
                    annotation.words.len(),
                    binding.word_count,
                )));
            }
            // Sparse output is legal; duplicate ownership of one sentence is not.
            let response = coref_response_from_v2_annotations(&[annotation]);
            for bound in response.annotations {
                if annotation_map
                    .insert(binding.line_idx, bound.annotation)
                    .is_some()
                {
                    return Err(coref_protocol_error("duplicate coref annotation sentence"));
                }
            }
        }
        let mut document = self.input.into_unchecked();
        apply_coref_results(&mut document, &annotation_map);
        Ok(AppliedCoref { document, engine })
    }
}

async fn run_coref_batch_impl<Infer>(
    files: &[TextBatchFileInput],
    pool: &WorkerPool,
    infer: Infer,
) -> TextBatchFileResults
where
    Infer: AsyncFnOnce(
        &WorkerPool,
        &[CorefBatchItem],
    ) -> Result<Vec<Result<ResolvedCoref, EngineItemFailure>>, ServerError>,
{
    use crate::pipeline::text_infer::admit_retained_text;
    use crate::text_batch::TextWorkflowFileError;
    use talkbank_model::model::TranscriptName;

    enum Admission {
        Pending(PendingCoref),
        PassThrough(PostValidated),
        Refused(TextWorkflowFileError),
    }

    let parser = crate::chat_parser();
    let command = crate::api::ReleasedCommand::Coref;
    let mut admissions = Vec::with_capacity(files.len());
    let mut items = Vec::new();
    for file in files {
        let admitted = match admit_retained_text(
            &parser,
            file.chat_text.as_ref(),
            TranscriptName::for_path(std::path::Path::new(file.filename.as_ref())),
        ) {
            Ok(admitted) => admitted,
            Err(error) => {
                admissions.push(Admission::Refused(
                    TextWorkflowFileError::from_server_error(&error),
                ));
                continue;
            }
        };
        if is_dummy(admitted.document()) {
            admissions.push(Admission::PassThrough(PostValidated::pass_through(
                admitted, command,
            )));
            continue;
        }
        match file_has_english(admitted.document()) {
            Ok(true) => {}
            Ok(false) => {
                admissions.push(Admission::PassThrough(PostValidated::pass_through(
                    admitted, command,
                )));
                continue;
            }
            Err(error) => {
                admissions.push(Admission::Refused(
                    TextWorkflowFileError::from_server_error(&error),
                ));
                continue;
            }
        }
        let payload = collect_coref_payloads(admitted.document());
        if payload.batch_item.sentences.is_empty() {
            admissions.push(Admission::PassThrough(PostValidated::pass_through(
                admitted, command,
            )));
            continue;
        }
        let batch_idx = items.len();
        let sentences = sentence_bindings(&payload);
        items.push(payload.batch_item);
        admissions.push(Admission::Pending(PendingCoref {
            input: admitted.into_valid_file(),
            sentences,
            batch_idx,
        }));
    }

    let responses = if items.is_empty() {
        Ok(Vec::new())
    } else {
        info!(
            num_items = items.len(),
            "Dispatching coref execute_v2 batch"
        );
        infer(pool, &items).await
    };
    let mut results = Vec::with_capacity(files.len());
    for (file, admission) in files.iter().zip(admissions) {
        let pending = match admission {
            Admission::Refused(failure) => {
                results.push(TextBatchFileResult::err(file.filename.clone(), failure));
                continue;
            }
            Admission::PassThrough(output) => {
                results.push(TextBatchFileResult::ok(file.filename.clone(), output));
                continue;
            }
            Admission::Pending(pending) => pending,
        };
        let result = match &responses {
            Err(error) => {
                results.push(TextBatchFileResult::err(
                    file.filename.clone(),
                    TextWorkflowFileError::from_server_error(error),
                ));
                continue;
            }
            Ok(responses) => match responses.get(pending.batch_idx) {
                Some(Ok(result)) => result.clone(),
                Some(Err(failure)) => {
                    results.push(TextBatchFileResult::err(
                        file.filename.clone(),
                        TextWorkflowFileError::item_errors(
                            "coref",
                            ItemFailures::of_one(ItemError {
                                item_index: 0,
                                failure: failure.clone(),
                            }),
                        ),
                    ));
                    continue;
                }
                None => {
                    results.push(TextBatchFileResult::err(
                        file.filename.clone(),
                        TextWorkflowFileError::from_server_error(&coref_protocol_error(
                            "coref response batch does not cover its pending request",
                        )),
                    ));
                    continue;
                }
            },
        };
        let applied = match pending.complete(result) {
            Ok(applied) => applied,
            Err(error) => {
                results.push(TextBatchFileResult::err(
                    file.filename.clone(),
                    TextWorkflowFileError::from_server_error(&error),
                ));
                continue;
            }
        };
        let mut document = applied.document;
        let stamp = match crate::provenance::result_named_provenance(
            crate::provenance::ResultNamedCommand::Coref,
            &LanguageCode3::eng(),
            [&applied.engine],
        ) {
            TextStamp::Stamped(comment) => {
                crate::provenance::inject_provenance(&mut document, &comment);
                FileStampOutcome::Stamped {
                    command: command.to_string(),
                }
            }
            TextStamp::NotStamped(reason) => FileStampOutcome::NotStamped {
                command: command.to_string(),
                reason: reason.to_string(),
            },
        };
        match PostValidated::gate_owned(document, command) {
            Ok(output) => results.push(TextBatchFileResult::ok_stamped(
                file.filename.clone(),
                output,
                stamp,
            )),
            Err(failure) => results.push(TextBatchFileResult::err(file.filename.clone(), failure)),
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
        .map_err(|error| coref_protocol_error(format!("invalid coref V2 result: {error}")))?;
    if result.items.len() != items.len() {
        return Err(coref_protocol_error(format!(
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
                annotations,
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
    use batchalign_transform::parse::{TreeSitterParser, parse_lenient};

    const VALID: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Child\n\
        @ID:\teng|test|CHI|||||Child|||\n*CHI:\thello world .\n@End\n";

    fn pending() -> PendingCoref {
        let parser = crate::chat_parser();
        let input = crate::pipeline::text_infer::admit_retained_text(
            &parser,
            VALID,
            talkbank_model::model::TranscriptName::Anonymous,
        )
        .expect("source fixture is fully valid CHAT");
        let payload = collect_coref_payloads(input.document());
        PendingCoref {
            input: input.into_valid_file(),
            sentences: sentence_bindings(&payload),
            batch_idx: 0,
        }
    }

    fn resolved(annotations: Vec<CorefAnnotationV2>) -> ResolvedCoref {
        ResolvedCoref::Resolved {
            annotations,
            engine: ReportedEngineName::try_from("test-coref").expect("engine name"),
        }
    }

    #[test]
    fn coref_completion_refuses_unbound_or_incomplete_annotations() {
        let cases = [
            resolved(vec![CorefAnnotationV2 {
                sentence_idx: usize::MAX,
                words: vec![],
            }]),
            resolved(vec![CorefAnnotationV2 {
                sentence_idx: 0,
                words: vec![vec![]],
            }]),
            resolved(vec![
                CorefAnnotationV2 {
                    sentence_idx: 0,
                    words: vec![vec![], vec![]],
                },
                CorefAnnotationV2 {
                    sentence_idx: 0,
                    words: vec![vec![], vec![]],
                },
            ]),
            ResolvedCoref::NoSentences,
        ];
        for response in cases {
            let error = match pending().complete(response) {
                Ok(_) => panic!("unbound analysis must not produce applied coreference"),
                Err(error) => error,
            };
            assert_eq!(
                crate::runner::util::classify_server_error(&error),
                crate::scheduling::FailureCategory::WorkerProtocol
            );
        }
        assert!(
            pending().complete(resolved(vec![])).is_ok(),
            "a resolved document with no chains is legitimate sparse output"
        );
        assert!(
            pending()
                .complete(resolved(vec![CorefAnnotationV2 {
                    sentence_idx: 0,
                    words: vec![vec![], vec![]],
                }]))
                .is_ok(),
            "bound per-word annotations complete normally"
        );
    }

    #[tokio::test]
    async fn coref_mixed_batch_keeps_admission_verdicts_when_inference_fails() {
        use crate::api::DisplayPath;
        use crate::scheduling::FailureCategory;
        use crate::worker::pool::PoolConfig;
        let non_english = VALID.replace("eng", "spa");
        let invalid = VALID.replace("hello world .", "<the dog> [///] .");
        let invalid_non_english = non_english.replace("@End", "%mor:\tnoun| .\n@End");
        let files = [
            TextBatchFileInput::new(DisplayPath::from("english.cha"), VALID.to_owned()),
            TextBatchFileInput::new(DisplayPath::from("spanish.cha"), non_english.clone()),
            TextBatchFileInput::new(DisplayPath::from("invalid.cha"), invalid),
            TextBatchFileInput::new(
                DisplayPath::from("invalid-spanish.cha"),
                invalid_non_english,
            ),
        ];
        let pool = WorkerPool::new(PoolConfig::default());
        let results = run_coref_batch_impl(&files, &pool, async |_pool, items| {
            assert_eq!(
                items.len(),
                1,
                "only the valid English document reaches inference"
            );
            Err(coref_protocol_error("test provider failure"))
        })
        .await;
        assert_eq!(results.len(), 4);
        assert_eq!(
            results[0]
                .result
                .as_ref()
                .expect_err("eligible file failed")
                .category(),
            FailureCategory::WorkerProtocol
        );
        assert_eq!(
            results[1]
                .result
                .as_ref()
                .expect("valid non-English pass-through")
                .as_str(),
            non_english
        );
        for result in &results[2..] {
            assert_eq!(
                result
                    .result
                    .as_ref()
                    .expect_err("invalidity survives an unrelated provider failure")
                    .category(),
                FailureCategory::Validation
            );
        }
    }

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
