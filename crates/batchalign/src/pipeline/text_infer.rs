//! Shared text-infer pipeline skeleton: collect payloads, run worker
//! inference, apply results back to the CHAT AST. Supports both
//! single-file and cross-file-batch flows.

use std::collections::HashMap;

use crate::api::{LanguageCode3, ReleasedCommand};
use crate::chat_ops::ChatFile;
use crate::text_batch::{TextBatchFileInput, TextBatchFileResult, TextBatchFileResults};
use crate::worker::pool::WorkerPool;
use batchalign_transform::parse::{is_dummy, parse_lenient};
use batchalign_transform::validate::{ValidityLevel, validate_to_level};
use tracing::warn;

use crate::error::ServerError;
use crate::pipeline::PipelineServices;
use crate::pipeline::post_validate::PostValidated;

type IntegrateFn<Item, State, Response> =
    fn(&mut HashMap<usize, State>, &[(usize, Item)], &[Response]);

/// Hooks for a text-only single-file pipeline.
///
/// `collect` extracts payloads from the parsed CHAT; `integrate` merges
/// responses into the state map; `apply` writes the state back into
/// the CHAT AST. The inference function itself is a separate argument
/// to [`run_text_pipeline`] so callers can pass any `async fn` directly.
pub(crate) struct TextPipelineHooks<Item, State, Response> {
    /// The command this pipeline is running, for validation, error strings
    /// and the proof its gate produces.
    pub command: ReleasedCommand,
    /// Pre-validation gate required by the command.
    pub validity: ValidityLevel,
    /// Extract worker payloads from the parsed chat file.
    pub collect: fn(&ChatFile) -> Vec<(usize, Item)>,
    /// Merge inferred responses into the final application map.
    pub integrate: IntegrateFn<Item, State, Response>,
    /// Apply all results to the parsed chat file.
    pub apply: fn(&mut ChatFile, &HashMap<usize, State>),
}

/// Run the text-only pipeline for a single CHAT file.
///
/// `infer` runs worker inference for all collected payloads. It is an
/// `async` callable (stable `AsyncFnOnce` trait, Rust 2024), so native
/// `async fn` inference routines can be passed without a boxed-future
/// adapter at the call site.
pub(crate) async fn run_text_pipeline<Item, State, Response, Infer, Observe>(
    chat_text: &str,
    lang: &LanguageCode3,
    services: PipelineServices<'_>,
    hooks: TextPipelineHooks<Item, State, Response>,
    infer: Infer,
    observe: Observe,
) -> Result<PostValidated, ServerError>
where
    Infer: AsyncFnOnce(
        &WorkerPool,
        &[(usize, Item)],
        &LanguageCode3,
    ) -> Result<Vec<Result<Response, String>>, ServerError>,
    Observe: FnOnce(&[(usize, Item)], &[Response]) -> Result<(), ServerError>,
{
    let parser = crate::chat_parser();
    let (mut chat_file, parse_errors) = parse_lenient(&parser, chat_text);
    if !parse_errors.is_empty() {
        warn!(
            command = %hooks.command,
            num_errors = parse_errors.len(),
            "Parse errors in input (continuing with recovery)"
        );
    }

    if is_dummy(&chat_file) {
        // A dummy file is handed back untouched, so it is a pass-through
        // rather than gated output: see `PostValidated`'s module docs. The
        // INPUT bytes are what "untouched" means, so they are what it carries.
        return Ok(PostValidated::pass_through(chat_text, hooks.command));
    }

    if let Err(errors) = validate_to_level(&chat_file, &parse_errors, hooks.validity) {
        let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        return Err(ServerError::Validation(format!(
            "{} pre-validation failed: {}",
            hooks.command,
            msgs.join("; ")
        )));
    }

    let batch_items = (hooks.collect)(&chat_file);
    if batch_items.is_empty() {
        // Nothing was collected, so nothing was applied: another
        // pass-through, not a claim about the input's validity.
        return Ok(PostValidated::pass_through(chat_text, hooks.command));
    }

    let item_results = infer(services.pool, &batch_items, lang).await?;
    // The typed error's OWN category travels. Rendering it into
    // `ServerError::Validation` retyped every per-item PROVIDER failure as bad
    // input, contradicting `TextWorkflowFileError::ItemErrors`, which
    // classifies as `ProviderTerminal`, and killing the retry that category
    // exists to trigger.
    let responses =
        crate::text_batch::unwrap_per_item_results(hooks.command.as_str(), item_results)
            .map_err(|err| ServerError::from_classified_failure(&err))?;
    observe(&batch_items, &responses)?;
    let mut state_map: HashMap<usize, State> = HashMap::new();
    (hooks.integrate)(&mut state_map, &batch_items, &responses);

    (hooks.apply)(&mut chat_file, &state_map);

    // Inject processing provenance comment.
    let ev = services.engine_version.as_ref();
    let lang_str = lang.as_ref();
    let provenance = match hooks.command {
        ReleasedCommand::Utseg => Some(crate::provenance::utseg_provenance(lang_str, ev)),
        ReleasedCommand::Translate => Some(crate::provenance::translate_provenance(lang_str, ev)),
        // Every other command stamps its provenance in its own pipeline; this
        // shared skeleton runs only for the two above.
        _ => None,
    };
    if let Some(comment) = provenance {
        crate::provenance::inject_provenance(&mut chat_file, &comment);
    }

    // The gate runs LAST, on the model that is about to become the bytes we
    // return, so the proof covers what is written and not an earlier draft of
    // it. Failing it fails the command; nothing is serialized.
    PostValidated::gate_owned(chat_file, hooks.validity, hooks.command)
        .map_err(|failure| ServerError::Validation(failure.to_string()))
}

/// Hooks for the cross-file text-batch pipeline (pool all files'
/// payloads into one `batch_infer` call, then redistribute responses).
///
/// Shared between `utseg` and `translate`. Morphotag does not use this
/// because it has additional structure (language-group dispatch, L2
/// secondary dispatch, alignment validation) that doesn't fit the
/// generic shape.
/// Extract worker payloads from a parsed chat file. Each payload is tagged
/// with its utterance index so the injector can match responses back.
pub(crate) type TextBatchCollect<Item> = fn(&ChatFile) -> Vec<(usize, Item)>;

/// Apply one file's collected items + their worker responses to the chat
/// file's AST. Called once per file after global inference completes.
pub(crate) type TextBatchApply<Item, Response> = fn(&mut ChatFile, &[(usize, Item)], &[Response]);

pub(crate) struct TextBatchHooks<Item, Response> {
    /// The command this pipeline is running, for validation, log messages
    /// and the proof its gate produces.
    pub command: ReleasedCommand,
    /// Pre-validation gate required by the command.
    pub validity: ValidityLevel,
    /// Extract worker payloads from the parsed chat file.
    pub collect: TextBatchCollect<Item>,
    /// Apply one file's items + responses directly to that file's AST.
    /// Called once per file after global inference completes.
    pub apply: TextBatchApply<Item, Response>,
}

/// Run the cross-file text-batch pipeline for `files`.
///
/// The pipeline:
/// 1. Parses all files once.
/// 2. For each non-dummy file, validates and collects payloads,
///    recording a `{item_count, global_start}` slice into the pooled
///    payload vector.
/// 3. Calls `infer` once over every file's payloads together.
/// 4. Slices the pooled responses back to each file and invokes
///    `hooks.apply` to inject results into the AST.
/// 5. Runs the post-validation gate per file and serializes the ones that
///    pass; a file that fails is reported as a per-file validation failure
///    and never written.
///
/// On worker failure every file whose items went into the batch is
/// reported as an error; files with no payloads (empty/dummy) are
/// serialized unchanged.
pub(crate) async fn run_text_batch_pipeline<Item, Response, Infer>(
    files: &[TextBatchFileInput],
    lang: &LanguageCode3,
    pool: &WorkerPool,
    hooks: TextBatchHooks<Item, Response>,
    infer: Infer,
) -> TextBatchFileResults
where
    Response: Clone,
    Infer: AsyncFnOnce(
        &WorkerPool,
        &[(usize, Item)],
        &LanguageCode3,
    ) -> Result<Vec<Result<Response, String>>, ServerError>,
{
    let parser = crate::chat_parser();
    let mut results: TextBatchFileResults = Vec::with_capacity(files.len());

    // 1. Parse every input.
    let mut parsed_files: Vec<ChatFile> = Vec::with_capacity(files.len());
    let mut parse_error_lists: Vec<Vec<crate::chat_ops::ParseError>> =
        Vec::with_capacity(files.len());
    for file in files {
        let (chat_file, parse_errors) = parse_lenient(&parser, file.chat_text.as_ref());
        if !parse_errors.is_empty() {
            warn!(
                filename = %file.filename.as_ref(),
                num_errors = parse_errors.len(),
                "Parse errors (continuing with recovery)"
            );
        }
        parse_error_lists.push(parse_errors);
        parsed_files.push(chat_file);
    }

    // 2. Pool payloads across files, remembering each file's slice.
    struct PerFileBatch {
        item_count: usize,
        global_start: usize,
    }

    /// What admission decided about one input file.
    ///
    /// One value per file, replacing the two parallel `Option` vectors this
    /// loop used to fill (`per_file_info` and `validation_errors`). Those were
    /// the parallel-collections tell, and the defect they hid was exact: a
    /// file that failed PRE-validation set `per_file_info[idx] = None` AND
    /// `validation_errors[idx] = Some(..)`, so the batch-failure branch, which
    /// asked only `per_file_info`, took it for a file with no payloads and
    /// reported it as a successful pass-through. A refused file was written.
    /// With one value there is no second vector to disagree with, and every
    /// consumer matches exhaustively.
    enum FileAdmission {
        /// This file's payloads went into the pooled batch, at this slice.
        Admitted(PerFileBatch),
        /// Nothing was collected: a dummy file, or one with no payloads. The
        /// command applies nothing, so its own bytes are the output.
        NothingToDo,
        /// The file failed pre-validation. It must never be written, on any
        /// path, whatever the batch does.
        RefusedAtAdmission(String),
    }

    let mut all_items: Vec<(usize, Item)> = Vec::new();
    let mut admissions: Vec<FileAdmission> = Vec::with_capacity(files.len());

    for (file_idx, parsed_file) in parsed_files.iter().enumerate() {
        if is_dummy(parsed_file) {
            admissions.push(FileAdmission::NothingToDo);
            continue;
        }

        if let Err(errors) =
            validate_to_level(parsed_file, &parse_error_lists[file_idx], hooks.validity)
        {
            let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            let error_summary = format!(
                "{} pre-validation failed: {}",
                hooks.command,
                msgs.join("; ")
            );
            warn!(
                filename = %files[file_idx].filename,
                errors = %error_summary,
                chat_text = %files[file_idx].chat_text,
                command = %hooks.command,
                "pre-validation failed: dumping CHAT for diagnosis"
            );
            admissions.push(FileAdmission::RefusedAtAdmission(error_summary));
            continue;
        }

        let batch_items = (hooks.collect)(parsed_file);
        if batch_items.is_empty() {
            admissions.push(FileAdmission::NothingToDo);
            continue;
        }

        let global_start = all_items.len();
        let item_count = batch_items.len();
        admissions.push(FileAdmission::Admitted(PerFileBatch {
            item_count,
            global_start,
        }));
        all_items.extend(batch_items);
    }

    // 3. Single batch_infer across all files' pooled payloads.
    let all_item_results = if all_items.is_empty() {
        Vec::new()
    } else {
        match infer(pool, &all_items, lang).await {
            Ok(responses) => responses,
            Err(e) => {
                warn!(error = %e, command = %hooks.command, "Batch infer failed for all files");
                for (file_idx, file) in files.iter().enumerate() {
                    let outcome = match &admissions[file_idx] {
                        FileAdmission::Admitted(_) => TextBatchFileResult::err(
                            file.filename.clone(),
                            crate::text_batch::TextWorkflowFileError::from_server_error(&e),
                        ),
                        // Refused at admission, and a failed batch does not
                        // change that: this file is a validation failure here
                        // exactly as it is on the success path.
                        FileAdmission::RefusedAtAdmission(message) => TextBatchFileResult::err(
                            file.filename.clone(),
                            crate::text_batch::TextWorkflowFileError::validation(message.clone()),
                        ),
                        // No payloads collected (empty/dummy): the command
                        // applied nothing, so this is a pass-through of the
                        // file's own bytes.
                        FileAdmission::NothingToDo => TextBatchFileResult::ok(
                            file.filename.clone(),
                            PostValidated::pass_through(file.chat_text.as_ref(), hooks.command),
                        ),
                    };
                    results.push(outcome);
                }
                return results;
            }
        }
    };

    // 4. Redistribute responses per file, apply, post-validate, serialize.
    //
    // Per-item engine/network/model failures are attributed back to
    // the file they came from via ``per_file_info``. A file with any
    // failing item is marked as failed with a typed
    // ``TextWorkflowFileError::ItemErrors``; other files in the same
    // cross-file batch continue normally. This matches BA2's
    // per-file-isolation multi-file failure semantics.
    for (file_idx, file) in files.iter().enumerate() {
        let fm = match &admissions[file_idx] {
            FileAdmission::RefusedAtAdmission(message) => {
                results.push(TextBatchFileResult::err(
                    file.filename.clone(),
                    crate::text_batch::TextWorkflowFileError::validation(message.clone()),
                ));
                continue;
            }
            FileAdmission::NothingToDo => {
                // Nothing was applied, so the file's own bytes are the output
                // and there is no output to gate. This is the same verdict the
                // batch-failure branch above reaches for the same state.
                results.push(TextBatchFileResult::ok(
                    file.filename.clone(),
                    PostValidated::pass_through(file.chat_text.as_ref(), hooks.command),
                ));
                continue;
            }
            FileAdmission::Admitted(fm) => fm,
        };

        let chat_file = &mut parsed_files[file_idx];

        let end = fm.global_start + fm.item_count;
        let file_items = &all_items[fm.global_start..end];
        let file_item_results = &all_item_results[fm.global_start..end];

        // Collect any per-item failures for this file. If any
        // failed, mark the entire file as failed without writing
        // partial output: matches BA2 (one bad utterance abandons
        // the file).
        let item_errors: Vec<crate::text_batch::ItemError> = file_item_results
            .iter()
            .enumerate()
            .filter_map(|(local_idx, r)| match r {
                Err(message) => Some(crate::text_batch::ItemError {
                    item_index: local_idx,
                    message: message.clone(),
                }),
                Ok(_) => None,
            })
            .collect();
        if !item_errors.is_empty() {
            results.push(TextBatchFileResult::err(
                file.filename.clone(),
                crate::text_batch::TextWorkflowFileError::item_errors(
                    hooks.command.as_str(),
                    item_errors,
                ),
            ));
            continue;
        }

        // All items succeeded for this file, extract owned
        // responses and apply them. Any Err was already filtered
        // above (the loop `continue`d when `item_errors` was
        // non-empty), so `filter_map(.ok())` collects every response
        // here without panicking.
        let file_responses: Vec<Response> = file_item_results
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .cloned()
            .collect();
        (hooks.apply)(chat_file, file_items, &file_responses);

        // Fail-closed, per file: a file whose output fails the gate is
        // reported as a validation failure and never written; the rest of the
        // cross-file batch is unaffected, which is the same isolation the
        // per-item failure branch above already provides.
        match PostValidated::gate(chat_file, hooks.validity, hooks.command) {
            Ok(output) => results.push(TextBatchFileResult::ok(file.filename.clone(), output)),
            Err(failure) => {
                results.push(TextBatchFileResult::err(file.filename.clone(), failure));
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::DisplayPath;
    use crate::scheduling::FailureCategory;
    use crate::worker::pool::PoolConfig;

    /// A minimal file that satisfies L1: participants, languages, terminator.
    const VALID: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n\
@ID:\teng|test|CHI|3;|male|||Target_Child|||\n*CHI:\thello world .\n@End\n";

    /// One payload per file, so the pipeline reaches its apply-and-gate tail
    /// rather than short-circuiting as an empty pass-through.
    fn collect_one(_file: &ChatFile) -> Vec<(usize, ())> {
        vec![(0, ())]
    }

    /// A command that corrupts its own output by eating every terminator.
    /// This is the mutation the gate exists to catch; before the gate it
    /// produced a `warn!` and a written file.
    fn apply_dropping_terminators(file: &mut ChatFile, _items: &[(usize, ())], _r: &[()]) {
        use talkbank_model::model::Line;
        for line in &mut file.lines {
            if let Line::Utterance(utt) = line {
                utt.main.content.terminator = None;
            }
        }
    }

    /// A command that leaves the document alone: the positive control that
    /// proves the refusal below is caused by the mutation, not by the fixture.
    fn apply_nothing(_file: &mut ChatFile, _items: &[(usize, ())], _r: &[()]) {}

    async fn run_with(
        apply: TextBatchApply<(), ()>,
    ) -> Result<PostValidated, crate::text_batch::TextWorkflowFileError> {
        let pool = WorkerPool::new(PoolConfig::default());
        let files = vec![TextBatchFileInput::new(
            DisplayPath::from("a.cha"),
            VALID.to_owned(),
        )];
        let mut results = run_text_batch_pipeline(
            &files,
            &LanguageCode3::eng(),
            &pool,
            TextBatchHooks {
                command: ReleasedCommand::Utseg,
                validity: ValidityLevel::StructurallyComplete,
                collect: collect_one,
                apply,
            },
            // The pool is never touched: inference is stubbed out so the test
            // exercises the apply-and-gate tail, not the worker boundary.
            async move |_pool, items, _lang| Ok(items.iter().map(|_| Ok(())).collect()),
        )
        .await;
        assert_eq!(results.len(), 1, "one input file, one result");
        results.remove(0).result
    }

    /// Positive control: untouched output passes the gate and reaches the
    /// writer as a proof.
    #[tokio::test]
    async fn batch_pipeline_admits_output_that_still_validates() {
        let result = run_with(apply_nothing).await;
        let output = result.expect("an untouched document must pass its own gate");
        assert!(output.as_str().contains("*CHI:"));
    }

    /// A file that fails PRE-validation: no `@Participants`, so it cannot
    /// satisfy L1.
    const REFUSED_AT_ADMISSION: &str = "@UTF8\n@Begin\n*CHI:\thello world .\n@End\n";

    /// RED FIRST (review item 3): when the pooled batch fails, a file that was
    /// REFUSED AT ADMISSION must still be a validation failure. It used to be
    /// reported as a successful pass-through and written, because the
    /// branch asked only the `per_file_info` vector, and a pre-validation
    /// refusal set that entry to `None` in the same breath as it recorded the
    /// error in a SECOND vector nothing here read.
    #[tokio::test]
    async fn a_batch_failure_still_refuses_a_file_that_failed_pre_validation() {
        let pool = WorkerPool::new(PoolConfig::default());
        let files = vec![
            TextBatchFileInput::new(DisplayPath::from("good.cha"), VALID.to_owned()),
            TextBatchFileInput::new(
                DisplayPath::from("refused.cha"),
                REFUSED_AT_ADMISSION.to_owned(),
            ),
        ];
        let results = run_text_batch_pipeline(
            &files,
            &LanguageCode3::eng(),
            &pool,
            TextBatchHooks {
                command: ReleasedCommand::Utseg,
                validity: ValidityLevel::StructurallyComplete,
                collect: collect_one,
                apply: apply_nothing,
            },
            async move |_pool, _items, _lang| {
                Err(crate::error::ServerError::Validation("batch broke".into()))
            },
        )
        .await;

        let refused = results
            .iter()
            .find(|r| r.filename.as_ref() == "refused.cha")
            .expect("every input file gets a result");
        let failure = refused
            .result
            .as_ref()
            .expect_err("a file refused at admission must never be reported as success");
        assert_eq!(
            failure.category(),
            FailureCategory::Validation,
            "a pre-validation refusal is a validity failure, not a provider failure"
        );
        assert!(
            failure.to_string().contains("pre-validation failed"),
            "the failure must name the admission refusal, got: {failure}"
        );
    }

    /// The seam test: a command whose output drops a terminator FAILS that
    /// file, with `FailureCategory::Validation`, and produces no output for
    /// the writer to write.
    #[tokio::test]
    async fn batch_pipeline_refuses_output_that_dropped_a_terminator() {
        let result = run_with(apply_dropping_terminators).await;
        let failure = result.expect_err("corrupted output must fail the file, not be written");
        assert_eq!(
            failure.category(),
            FailureCategory::Validation,
            "a file refused on validity grounds must not be reported as a provider failure"
        );
        let rendered = failure.to_string();
        assert!(
            rendered.contains("lost its terminator"),
            "the failure must name what broke, got: {rendered}"
        );
    }
}
