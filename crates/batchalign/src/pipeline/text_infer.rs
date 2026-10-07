//! Shared text-infer pipeline skeleton: collect payloads, run worker
//! inference, apply results back to the CHAT AST. Supports both
//! single-file and cross-file-batch flows.

use std::collections::HashMap;

use crate::api::FileStampOutcome;
use crate::api::{InvalidStampSafeText, LanguageCode3, ReleasedCommand};
use crate::chat_ops::ChatFile;
use crate::provenance::TextStamp;
use crate::text_batch::{
    ItemError, ItemFailure, ItemFailures, TextBatchFileInput, TextBatchFileResult,
    TextBatchFileResults, TextWorkflowFileError,
};
use crate::worker::pool::WorkerPool;
use batchalign_transform::parse::is_dummy;
use batchalign_transform::{AdmittedSourceChat, parse_source_with_parser};
use talkbank_model::model::TranscriptName;
use talkbank_model::validation::ValidChatFile;
use tracing::{info, warn};

use crate::error::ServerError;
use crate::pipeline::PipelineServices;
use crate::pipeline::post_validate::PostValidated;

type IntegrateFn<Item, State, Response> =
    fn(&mut HashMap<usize, State>, &[(usize, Item)], &[Response]);

/// These workflows retain input content: no generated-tier recovery exemption
/// applies. Only Chatter's complete source admission can produce their input.
pub(crate) fn admit_retained_text<'source>(
    parser: &batchalign_transform::parse::TreeSitterParser,
    source: &'source str,
    name: TranscriptName<'_>,
) -> Result<AdmittedSourceChat<'source>, ServerError> {
    parse_source_with_parser(parser, source)
        .admit(name, &talkbank_model::NullErrorSink)
        .map_err(ServerError::ChatAdmission)
}

/// Where a text command's provenance comment comes from: the responses it
/// applied, which name the engines that produced them.
/// [`TextStamp::NotStamped`] carries the reason when no engine produced
/// anything that was applied. `Err` only when a name a response carries is not
/// stamp-safe text: translate and coref names are admitted as reported engine
/// names on arrival and never fail here, while a utseg boundary model's id and
/// revision are admitted when the stamp is built.
///
/// There is no pipeline-wide engine version and no capability report to fall
/// back on (see the book's provenance page), so a command cannot stamp an
/// engine its responses did not name.
pub(crate) type TextProvenance<Response> =
    fn(&LanguageCode3, &[Response]) -> Result<TextStamp, InvalidStampSafeText>;

/// Hooks for a text-only single-file pipeline.
///
/// `collect` extracts payloads from the parsed CHAT; `integrate` merges
/// responses into the state map; `apply` writes the state back into
/// the CHAT AST. The inference function itself is a separate argument
/// to [`run_admitted_text_pipeline`] so callers can pass any `async fn` directly.
pub(crate) struct TextPipelineHooks<Item, State, Response> {
    /// The command this pipeline is running, for validation, error strings
    /// and the proof its gate produces.
    pub command: ReleasedCommand,
    /// Extract worker payloads from the parsed chat file.
    pub collect: fn(&ChatFile) -> Vec<(usize, Item)>,
    /// Merge inferred responses into the final application map.
    pub integrate: IntegrateFn<Item, State, Response>,
    /// Apply all results to the parsed chat file, saying where its
    /// utterances went (a split moves the ones after it).
    pub apply: fn(
        &mut ChatFile,
        &HashMap<usize, State>,
    ) -> Result<crate::pipeline::post_validate::AppliedLayout, ServerError>,
    /// Source of the provenance comment stamped on the output.
    pub provenance: TextProvenance<Response>,
}

/// Test-only entry point exercising source refusal before the typed pipeline.
///
/// `infer` runs worker inference for all collected payloads. It is an
/// `async` callable (stable `AsyncFnOnce` trait, Rust 2024), so native
/// `async fn` inference routines can be passed without a boxed-future
/// adapter at the call site.
#[cfg(test)]
async fn run_text_pipeline<Item, State, Response, Failure, Infer, Observe>(
    chat_text: &str,
    lang: &LanguageCode3,
    services: PipelineServices<'_>,
    hooks: TextPipelineHooks<Item, State, Response>,
    infer: Infer,
    observe: Observe,
) -> Result<PostValidated, ServerError>
where
    Failure: std::fmt::Display,
    Infer: AsyncFnOnce(
        &WorkerPool,
        &[(usize, Item)],
        &LanguageCode3,
    ) -> Result<Vec<Result<Response, ItemFailure<Failure>>>, ServerError>,
    Observe: FnOnce(&[(usize, Item)], &[Response]) -> Result<(), ServerError>,
{
    let parser = crate::chat_parser();
    let admitted = admit_retained_text(&parser, chat_text, TranscriptName::Anonymous)?;
    run_admitted_text_pipeline(
        PostValidated::pass_through(admitted, hooks.command),
        lang,
        services,
        hooks,
        infer,
        observe,
    )
    .await
}

/// Continue from an already admitted typed document without a CHAT reparse.
///
/// Takes and returns a [`PostValidated`]: an admitted document is the
/// precondition (a diagnosed one, which only a generating producer makes, is
/// a different type and has no route in), and the strict gate this ends with
/// is the postcondition, so the result can continue into another stage that
/// requires admission.
pub(crate) async fn run_admitted_text_pipeline<Item, State, Response, Failure, Infer, Observe>(
    admitted: PostValidated,
    lang: &LanguageCode3,
    services: PipelineServices<'_>,
    hooks: TextPipelineHooks<Item, State, Response>,
    infer: Infer,
    observe: Observe,
) -> Result<PostValidated, ServerError>
where
    Failure: std::fmt::Display,
    Infer: AsyncFnOnce(
        &WorkerPool,
        &[(usize, Item)],
        &LanguageCode3,
    ) -> Result<Vec<Result<Response, ItemFailure<Failure>>>, ServerError>,
    Observe: FnOnce(&[(usize, Item)], &[Response]) -> Result<(), ServerError>,
{
    if is_dummy(admitted.document()) {
        // A dummy file is handed back untouched, so it is a pass-through
        // rather than gated output: see `PostValidated`'s module docs. The
        // INPUT bytes are what "untouched" means, so they are what it carries.
        return Ok(admitted);
    }

    let batch_items = (hooks.collect)(admitted.document());
    if batch_items.is_empty() {
        // Nothing was collected, so retain the already admitted input proof
        // without claiming that any analysis was applied.
        return Ok(admitted);
    }

    // Editing consumes the validity proof; the output must earn a new one.
    // The whole document is judged, so where its utterances went is not
    // needed.
    let mut chat_file = admitted.into_judged_document();
    let _layout = analyze_and_apply(
        &mut chat_file,
        &batch_items,
        lang,
        services,
        &hooks,
        infer,
        observe,
    )
    .await?;
    // The gate runs LAST, on the model that is about to become the bytes we
    // return, so the proof covers what is written and not an earlier draft of
    // it. Failing it fails the command; nothing is serialized.
    PostValidated::gate_owned(chat_file, hooks.command)
        .map_err(|failure| failure.into_server_error())
}

/// Continue from a generated document whose findings are confined to some
/// utterances: analyze and edit every OTHER utterance, then judge the result
/// with the producer transition. The held-out utterances are never sent to
/// the worker and keep their content, so the result is still diagnosed for
/// them (or admitted, if the edit happened to remove the finding's cause).
/// Unlike the admitted pipeline, an empty collection still re-judges the
/// model, because the input carried no proof to hand back.
pub(crate) async fn run_localized_text_pipeline<Item, State, Response, Failure, Infer, Observe>(
    localized: crate::pipeline::post_validate::LocalizedDiagnosis,
    lang: &LanguageCode3,
    services: PipelineServices<'_>,
    hooks: TextPipelineHooks<Item, State, Response>,
    infer: Infer,
    observe: Observe,
) -> Result<crate::pipeline::post_validate::ProducedOutput, ServerError>
where
    Failure: std::fmt::Display,
    Infer: AsyncFnOnce(
        &WorkerPool,
        &[(usize, Item)],
        &LanguageCode3,
    ) -> Result<Vec<Result<Response, ItemFailure<Failure>>>, ServerError>,
    Observe: FnOnce(&[(usize, Item)], &[Response]) -> Result<(), ServerError>,
{
    let (mut chat_file, held_out) = localized.into_model();
    let batch_items: Vec<(usize, Item)> = (hooks.collect)(&chat_file)
        .into_iter()
        .filter(|(utterance, _)| !held_out.contains(*utterance))
        .collect();
    let layout = if batch_items.is_empty() {
        crate::pipeline::post_validate::AppliedLayout::InPlace
    } else {
        analyze_and_apply(
            &mut chat_file,
            &batch_items,
            lang,
            services,
            &hooks,
            infer,
            observe,
        )
        .await?
    };
    // The held-out utterances, where the stage's output has them: a split
    // before one moves it down.
    let held_out = held_out
        .after(&layout)
        .ok_or_else(|| ServerError::OutputAdmission {
            command: hooks.command,
            details: crate::error::OutputAdmissionRefusal::unestablished(
                "the stage changed an utterance it was not given, so its output cannot be \
             judged against the utterances that carry the findings",
            ),
        })?;
    // Judged under the stage's command, as the admitted pipeline's gate is,
    // so both routes apply the same command checks. A result that adds a
    // finding outside the held-out utterances is refused, as the strict gate
    // refuses an admitted document's stage output; the caller keeps the
    // document from before the stage.
    PostValidated::produced_outside(chat_file, &held_out, hooks.command)
        .map_err(|failure| failure.into_server_error())
}

/// Infer the collected items, apply the responses and stamp provenance: the
/// part of a text stage that is the same whatever the document's standing.
async fn analyze_and_apply<Item, State, Response, Failure, Infer, Observe>(
    chat_file: &mut ChatFile,
    batch_items: &[(usize, Item)],
    lang: &LanguageCode3,
    services: PipelineServices<'_>,
    hooks: &TextPipelineHooks<Item, State, Response>,
    infer: Infer,
    observe: Observe,
) -> Result<crate::pipeline::post_validate::AppliedLayout, ServerError>
where
    Failure: std::fmt::Display,
    Infer: AsyncFnOnce(
        &WorkerPool,
        &[(usize, Item)],
        &LanguageCode3,
    ) -> Result<Vec<Result<Response, ItemFailure<Failure>>>, ServerError>,
    Observe: FnOnce(&[(usize, Item)], &[Response]) -> Result<(), ServerError>,
{
    let item_results = infer(services.pool, batch_items, lang).await?;
    // The typed error's OWN category travels. Rendering it into
    // `ServerError::Validation` retyped every per-item PROVIDER failure as bad
    // input, contradicting `TextWorkflowFileError::ItemErrors`, which
    // classifies as `ProviderTerminal`, and killing the retry that category
    // exists to trigger.
    let responses =
        crate::text_batch::unwrap_per_item_results(hooks.command.as_str(), item_results)
            .map_err(|err| ServerError::from_classified_failure(&err))?;
    observe(batch_items, &responses)?;
    let mut state_map: HashMap<usize, State> = HashMap::new();
    (hooks.integrate)(&mut state_map, batch_items, &responses);

    let layout = (hooks.apply)(chat_file, &state_map)?;

    // Inject the processing provenance comment, named by the applied
    // responses. A run that applied nothing says so rather than stamping a
    // command with no engine behind it.
    match (hooks.provenance)(lang, &responses)? {
        TextStamp::Stamped(comment) => {
            crate::provenance::inject_provenance(chat_file, &comment);
        }
        TextStamp::NotStamped(reason) => {
            info!(
                command = %hooks.command,
                reason = %reason,
                "no provenance stamp written"
            );
        }
    }
    Ok(layout)
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
pub(crate) type TextBatchApply<Item, Response> =
    fn(&mut ChatFile, &[(usize, Item)], &[Response]) -> Result<(), ServerError>;

pub(crate) struct TextBatchHooks<Item, Response> {
    /// The command this pipeline is running, for validation, log messages
    /// and the proof its gate produces.
    pub command: ReleasedCommand,
    /// Extract worker payloads from the parsed chat file.
    pub collect: TextBatchCollect<Item>,
    /// Apply one file's items + responses directly to that file's AST.
    /// Called once per file after global inference completes.
    pub apply: TextBatchApply<Item, Response>,
    /// Source of the provenance comment stamped on each file's output, from
    /// the responses that file applied. The same hook the single-file pipeline
    /// takes: a batch run records what produced a file exactly as a per-file
    /// run does.
    pub provenance: TextProvenance<Response>,
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
pub(crate) async fn run_text_batch_pipeline<Item, Response, Failure, Infer>(
    files: &[TextBatchFileInput],
    lang: &LanguageCode3,
    pool: &WorkerPool,
    hooks: TextBatchHooks<Item, Response>,
    infer: Infer,
) -> TextBatchFileResults
where
    Response: Clone,
    Failure: std::fmt::Display + Clone,
    Infer: AsyncFnOnce(
        &WorkerPool,
        &[(usize, Item)],
        &LanguageCode3,
    ) -> Result<Vec<Result<Response, ItemFailure<Failure>>>, ServerError>,
{
    let parser = crate::chat_parser();
    let mut results: TextBatchFileResults = Vec::with_capacity(files.len());

    // 1. Admit inputs and pool payloads, retaining the proof with its slice.
    struct PerFileBatch {
        input: ValidChatFile,
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
        NothingToDo(PostValidated),
        /// The file failed pre-validation. It must never be written, on any
        /// path, whatever the batch does.
        RefusedAtAdmission(TextWorkflowFileError),
    }

    let mut all_items: Vec<(usize, Item)> = Vec::new();
    let mut admissions: Vec<FileAdmission> = Vec::with_capacity(files.len());

    for file in files {
        let admitted = match admit_retained_text(
            &parser,
            file.chat_text.as_ref(),
            TranscriptName::for_path(std::path::Path::new(file.filename.as_ref())),
        ) {
            Ok(admitted) => admitted,
            Err(error) => {
                admissions.push(FileAdmission::RefusedAtAdmission(
                    TextWorkflowFileError::from_server_error(&error),
                ));
                continue;
            }
        };
        if is_dummy(admitted.document()) {
            admissions.push(FileAdmission::NothingToDo(PostValidated::pass_through(
                admitted,
                hooks.command,
            )));
            continue;
        }

        let batch_items = (hooks.collect)(admitted.document());
        if batch_items.is_empty() {
            admissions.push(FileAdmission::NothingToDo(PostValidated::pass_through(
                admitted,
                hooks.command,
            )));
            continue;
        }

        let global_start = all_items.len();
        let item_count = batch_items.len();
        admissions.push(FileAdmission::Admitted(PerFileBatch {
            input: admitted.into_valid_file(),
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
                        // A failed batch does not change the admission
                        // failure's own verdict (invalid CHAT or tool fault).
                        FileAdmission::RefusedAtAdmission(failure) => {
                            TextBatchFileResult::err(file.filename.clone(), failure.clone())
                        }
                        // No payloads collected (empty/dummy): the command
                        // applied nothing, so this is a pass-through of the
                        // file's own bytes.
                        FileAdmission::NothingToDo(output) => {
                            TextBatchFileResult::ok(file.filename.clone(), output.clone())
                        }
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
    for (file, admission) in files.iter().zip(admissions) {
        let fm = match admission {
            FileAdmission::RefusedAtAdmission(failure) => {
                results.push(TextBatchFileResult::err(file.filename.clone(), failure));
                continue;
            }
            FileAdmission::NothingToDo(output) => {
                // Nothing was applied, so the file's own bytes are the output
                // and there is no output to gate. This is the same verdict the
                // batch-failure branch above reaches for the same state.
                results.push(TextBatchFileResult::ok(file.filename.clone(), output));
                continue;
            }
            FileAdmission::Admitted(fm) => fm,
        };

        let mut destination = fm.input.into_unchecked();
        let chat_file = &mut destination;

        let end = fm.global_start + fm.item_count;
        let file_items = &all_items[fm.global_start..end];
        let file_item_results = &all_item_results[fm.global_start..end];

        // Collect any per-item failures for this file. If any
        // failed, mark the entire file as failed without writing
        // partial output: matches BA2 (one bad utterance abandons
        // the file).
        let item_errors: Vec<ItemError<Failure>> = file_item_results
            .iter()
            .enumerate()
            .filter_map(|(local_idx, r)| match r {
                Err(failure) => Some(ItemError {
                    item_index: local_idx,
                    failure: failure.clone(),
                }),
                Ok(_) => None,
            })
            .collect();
        if let Some(failures) = ItemFailures::new(item_errors) {
            results.push(TextBatchFileResult::err(
                file.filename.clone(),
                TextWorkflowFileError::item_errors(hooks.command.as_str(), failures),
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
        if let Err(error) = (hooks.apply)(chat_file, file_items, &file_responses) {
            results.push(TextBatchFileResult::err(
                file.filename.clone(),
                TextWorkflowFileError::from_server_error(&error),
            ));
            continue;
        }

        // Provenance for THIS file, from the responses this file applied, and
        // before the gate so the proof covers the bytes that are written. The
        // batch path used to write none at all, so a corpus processed as a
        // batch recorded nothing about what produced it.
        // Recorded on the file, not only logged: the job's per-file status
        // carries what this command decided, so "no stamp" is an answer with a
        // reason rather than an absence an operator has to interpret.
        let stamp = match (hooks.provenance)(lang, &file_responses) {
            Ok(TextStamp::Stamped(comment)) => {
                crate::provenance::inject_provenance(chat_file, &comment);
                FileStampOutcome::Stamped {
                    command: hooks.command.to_string(),
                }
            }
            Ok(TextStamp::NotStamped(reason)) => {
                info!(
                    filename = %file.filename,
                    command = %hooks.command,
                    reason = %reason,
                    "no provenance stamp written"
                );
                FileStampOutcome::NotStamped {
                    command: hooks.command.to_string(),
                    reason: reason.to_string(),
                }
            }
            // A name that cannot be written as a stamp fails THIS file, the
            // way its own gate would: the alternative is output whose
            // provenance the next run cannot recognize.
            Err(error) => {
                results.push(TextBatchFileResult::err(
                    file.filename.clone(),
                    TextWorkflowFileError::validation(error.to_string()),
                ));
                continue;
            }
        };

        // Fail-closed, per file: a file whose output fails the gate is
        // reported as a validation failure and never written; the rest of the
        // cross-file batch is unaffected, which is the same isolation the
        // per-item failure branch above already provides.
        match PostValidated::gate(chat_file, hooks.command) {
            Ok(output) => results.push(TextBatchFileResult::ok_stamped(
                file.filename.clone(),
                output,
                stamp,
            )),
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
    fn apply_dropping_terminators(
        file: &mut ChatFile,
        _items: &[(usize, ())],
        _r: &[()],
    ) -> Result<(), ServerError> {
        use talkbank_model::model::Line;
        for line in &mut file.lines {
            if let Line::Utterance(utt) = line {
                utt.main.content.terminator = None;
            }
        }
        Ok(())
    }

    /// A command that leaves the document alone: the positive control that
    /// proves the refusal below is caused by the mutation, not by the fixture.
    fn apply_nothing(
        _file: &mut ChatFile,
        _items: &[(usize, ())],
        _r: &[()],
    ) -> Result<(), ServerError> {
        Ok(())
    }

    /// Inference that succeeds for every item, with no payload.
    ///
    /// A named function rather than a closure: the pipeline is generic in the
    /// command's own failure type, and a closure that never constructs one
    /// leaves it for the compiler to guess.
    async fn stub_infer(
        _pool: &WorkerPool,
        items: &[(usize, ())],
        _lang: &LanguageCode3,
    ) -> Result<Vec<Result<(), crate::text_batch::EngineItemFailure>>, ServerError> {
        Ok(items.iter().map(|_| Ok(())).collect())
    }

    async fn must_not_infer(
        _pool: &WorkerPool,
        _items: &[(usize, ())],
        _lang: &LanguageCode3,
    ) -> Result<Vec<Result<(), crate::text_batch::EngineItemFailure>>, ServerError> {
        panic!("refused inputs and no-op files must not reach inference")
    }

    fn collect_nothing(_file: &ChatFile) -> Vec<(usize, ())> {
        Vec::new()
    }

    fn must_not_collect(_file: &ChatFile) -> Vec<(usize, ())> {
        panic!("invalid retained content must not reach payload collection")
    }

    /// Full retained-input admission is required even when collection would
    /// yield no work. In particular, the shared historical L1/L2 checks did
    /// not detect a retrace without corrected material or retained bad %mor.
    #[tokio::test]
    async fn retained_text_refuses_invalid_content_before_collection_or_inference() {
        let retrace = VALID.replace("hello world .", "<the dog> [///] .");
        let corrupt_mor = VALID.replace("@End", "%mor:\tnoun| .\n@End");
        let dummy_bad_header = "@UTF8\n@Begin\n@Options:\tdummy\n*CHI:\thello .\n@End\n";
        let sources = [retrace.as_str(), corrupt_mor.as_str(), dummy_bad_header];
        let pool = WorkerPool::new(PoolConfig::default());
        let files: Vec<_> = sources
            .iter()
            .enumerate()
            .map(|(index, source)| {
                TextBatchFileInput::new(
                    DisplayPath::from(format!("invalid-{index}.cha")),
                    (*source).to_owned(),
                )
            })
            .collect();
        let results = run_text_batch_pipeline(
            &files,
            &LanguageCode3::eng(),
            &pool,
            TextBatchHooks {
                command: ReleasedCommand::Utseg,
                collect: must_not_collect,
                apply: apply_nothing,
                provenance: test_stamp,
            },
            must_not_infer,
        )
        .await;
        assert_eq!(results.len(), sources.len());
        for result in results {
            assert_eq!(
                result
                    .result
                    .expect_err("invalid retained content must not pass through")
                    .category(),
                FailureCategory::Validation
            );
        }

        for source in sources {
            let cache = crate::cache::UtteranceCache::noop();
            let result = run_text_pipeline(
                source,
                &LanguageCode3::eng(),
                PipelineServices::new(&pool, &cache),
                TextPipelineHooks {
                    command: ReleasedCommand::Utseg,
                    collect: collect_nothing,
                    integrate: |_state: &mut HashMap<usize, ()>, _items, _responses| {},
                    apply: |_file, _state| {
                        Ok(crate::pipeline::post_validate::AppliedLayout::InPlace)
                    },
                    provenance: test_stamp,
                },
                must_not_infer,
                |_items, _responses| Ok(()),
            )
            .await;
            assert!(
                matches!(result, Err(ServerError::ChatAdmission(_))),
                "single-file retained input must be refused"
            );
        }
    }

    #[tokio::test]
    async fn fully_valid_no_work_file_preserves_its_original_bytes() {
        let pool = WorkerPool::new(PoolConfig::default());
        let files = [TextBatchFileInput::new(
            DisplayPath::from("valid.cha"),
            VALID.to_owned(),
        )];
        let results = run_text_batch_pipeline(
            &files,
            &LanguageCode3::eng(),
            &pool,
            TextBatchHooks {
                command: ReleasedCommand::Utseg,
                collect: collect_nothing,
                apply: apply_nothing,
                provenance: test_stamp,
            },
            must_not_infer,
        )
        .await;
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0]
                .result
                .as_ref()
                .expect("valid no-work input")
                .as_str(),
            VALID
        );
    }

    /// Inference that fails for the whole batch.
    async fn failing_infer(
        _pool: &WorkerPool,
        _items: &[(usize, ())],
        _lang: &LanguageCode3,
    ) -> Result<Vec<Result<(), crate::text_batch::EngineItemFailure>>, ServerError> {
        Err(ServerError::Validation("batch broke".into()))
    }

    /// A stamp naming one engine, so the batch path's injection is exercised.
    fn test_stamp(
        lang: &LanguageCode3,
        responses: &[()],
    ) -> Result<TextStamp, crate::api::InvalidStampSafeText> {
        let engine = crate::api::ReportedEngineName::try_from("test-engine")?;
        Ok(crate::provenance::result_named_provenance(
            crate::provenance::ResultNamedCommand::Translate,
            lang,
            responses.iter().map(|()| &engine),
        ))
    }

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
                // Translate, because the stub stamp source below is the
                // result-named builder translate uses: the command a pipeline
                // runs and the command its stamp records are the same thing.
                command: ReleasedCommand::Translate,
                collect: collect_one,
                apply,
                provenance: test_stamp,
            },
            // The pool is never touched: inference is stubbed out so the test
            // exercises the apply-and-gate tail, not the worker boundary.
            stub_infer,
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

    #[tokio::test]
    async fn a_refused_application_cannot_be_admitted_just_because_the_model_is_valid() {
        fn refuse(
            _file: &mut ChatFile,
            _items: &[(usize, ())],
            _responses: &[()],
        ) -> Result<(), ServerError> {
            Err(batchalign_transform::utseg::UtsegApplyRefusal::UnknownUtterance(99).into())
        }
        let error = run_with(refuse)
            .await
            .expect_err("application refusal is not success");
        assert_eq!(error.category(), crate::scheduling::FailureCategory::System);
        assert!(error.to_string().contains("absent utterance 99"));
    }

    #[tokio::test]
    async fn per_file_application_refusal_precedes_output_admission() {
        let pool = WorkerPool::new(PoolConfig::default());
        let cache = crate::cache::UtteranceCache::noop();
        let error = run_text_pipeline(
            VALID,
            &LanguageCode3::eng(),
            PipelineServices::new(&pool, &cache),
            TextPipelineHooks {
                command: ReleasedCommand::Utseg,
                collect: collect_one,
                integrate: |_state: &mut HashMap<usize, ()>, _items, _responses| {},
                apply: |_file, _state| {
                    Err(batchalign_transform::utseg::UtsegApplyRefusal::UnknownUtterance(99).into())
                },
                provenance: test_stamp,
            },
            stub_infer,
            |_items, _responses| Ok(()),
        )
        .await
        .expect_err("refused application must not issue output proof");
        assert!(matches!(error, ServerError::UtterancePartition(_)));
        assert_eq!(
            crate::runner::util::classify_server_error(&error),
            crate::scheduling::FailureCategory::System
        );
    }

    /// RED FIRST (W2): the batch path stamps the provenance of what it
    /// applied. It used to write none, so every file of a batch job recorded
    /// nothing about the engines behind it.
    #[tokio::test]
    async fn batch_pipeline_stamps_the_provenance_of_what_it_applied() {
        let result = run_with(apply_nothing).await;
        let output = result.expect("an untouched document must pass its own gate");
        assert!(
            output.as_str().contains(&format!(
                "{}engine=test-engine ; lang=eng | ",
                crate::provenance::written_stamp_opening("translate")
            )),
            "the batch output must carry its own stamp, got:\n{}",
            output.as_str()
        );
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
                command: ReleasedCommand::Translate,
                collect: collect_one,
                apply: apply_nothing,
                provenance: test_stamp,
            },
            failing_infer,
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
    /// file, with `FailureCategory::System`, and produces no output for
    /// the writer to write.
    #[tokio::test]
    async fn batch_pipeline_refuses_output_that_dropped_a_terminator() {
        let result = run_with(apply_dropping_terminators).await;
        let failure = result.expect_err("corrupted output must fail the file, not be written");
        assert_eq!(
            failure.category(),
            FailureCategory::System,
            "malformed produced output is a tool failure, not invalid client input"
        );
        let rendered = failure.to_string();
        assert!(
            rendered.contains("lost its terminator"),
            "the failure must name what broke, got: {rendered}"
        );
    }

    /// The diagnosed transcript of the localized-segmentation tests: three
    /// utterances, two speakers (so a rule about the participants cannot make
    /// every utterance fail on its own), the second holding a word CHAT
    /// cannot hold, localized to that utterance.
    fn localized_b2() -> crate::pipeline::post_validate::LocalizedDiagnosis {
        use talkbank_model::model::{Line, TierContentItems, UtteranceContent, Word};
        const GENERATED: &str = "@UTF8\n@Begin\n@Languages:\teng\n\
@Participants:\tPAR0 Participant, PAR1 Participant\n\
@ID:\teng|test|PAR0|||||Participant|||\n@ID:\teng|test|PAR1|||||Participant|||\n\
*PAR0:\tthe red ball rolled far away .\n\
*PAR1:\twe took it there .\n\
*PAR0:\tthen we went home again today .\n@End\n";
        let parser = crate::chat_parser();
        let (mut file, errors) = batchalign_transform::parse::parse_lenient(&parser, GENERATED);
        assert!(errors.is_empty(), "{errors:?}");
        // The generated word CHAT cannot hold, as a producer would have
        // built it before word forms were written (an invented utterance).
        let mut ordinal = 0;
        for line in &mut file.lines {
            if let Line::Utterance(utterance) = line {
                if ordinal == 1 {
                    utterance.main.content.content = TierContentItems::new(
                        ["we", "took", "b2", "there"]
                            .into_iter()
                            .map(|word| UtteranceContent::Word(Box::new(Word::simple(word))))
                            .collect(),
                    );
                }
                ordinal += 1;
            }
        }
        let crate::pipeline::post_validate::ProducedOutput::Diagnosed(diagnosed) =
            PostValidated::produced(file, ReleasedCommand::Transcribe)
        else {
            panic!("the invalid word must diagnose the transcript");
        };
        let localized = diagnosed
            .localize()
            .expect("the finding is the second utterance's");
        assert_eq!(localized.held_out().ordinals().collect::<Vec<_>>(), [1]);
        localized
    }

    /// A worker stand-in that splits every utterance it is given after its
    /// third word, recording which utterances it was given.
    fn split_after_third_word(
        items: &[(usize, batchalign_transform::utseg::UtsegBatchItem)],
    ) -> Vec<Result<Vec<usize>, crate::text_batch::EngineItemFailure>> {
        items
            .iter()
            .map(|(_, item)| {
                Ok((0..item.words.len())
                    .map(|word| usize::from(word >= 3))
                    .collect())
            })
            .collect()
    }

    /// Mirroring morphosyntax: segmentation of a
    /// diagnosed transcript that adds a finding of its own outside the
    /// held-out utterance is refused, as an admitted document's stage output
    /// is, instead of being written as though the finding were the
    /// transcript's. Here the application breaks the first utterance it split.
    /// The held-out utterance has moved down past that split, so the
    /// judgement finds it where the layout put it: the refusal carries the
    /// new finding alone, never the held-out one.
    #[tokio::test]
    async fn segmentation_that_adds_a_finding_outside_the_held_out_utterance_is_refused() {
        let pool = WorkerPool::new(PoolConfig::default());
        let cache = crate::cache::UtteranceCache::noop();
        let infer = async |_pool: &WorkerPool,
                           items: &[(usize, batchalign_transform::utseg::UtsegBatchItem)],
                           _lang: &LanguageCode3|
               -> Result<
            Vec<Result<Vec<usize>, crate::text_batch::EngineItemFailure>>,
            ServerError,
        > { Ok(split_after_third_word(items)) };
        let error = run_localized_text_pipeline(
            localized_b2(),
            &LanguageCode3::eng(),
            PipelineServices::new(&pool, &cache),
            TextPipelineHooks {
                command: ReleasedCommand::Utseg,
                collect: crate::utseg::collect_utseg_batch_items,
                integrate: |state: &mut HashMap<usize, Vec<usize>>,
                            items,
                            responses: &[Vec<usize>]| {
                    for ((utterance, _), assignment) in items.iter().zip(responses) {
                        state.insert(*utterance, assignment.clone());
                    }
                },
                apply: |file, state| {
                    use talkbank_model::model::{Line, TierContentItems, UtteranceContent, Word};
                    let layout = crate::utseg::apply_utseg_document(file, state)?;
                    // A defective application: the first child of the first
                    // split carries a word CHAT cannot hold.
                    for line in &mut file.lines {
                        if let Line::Utterance(utterance) = line {
                            utterance.main.content.content = TierContentItems::new(
                                ["the", "c3", "ball"]
                                    .into_iter()
                                    .map(|word| {
                                        UtteranceContent::Word(Box::new(Word::simple(word)))
                                    })
                                    .collect(),
                            );
                            break;
                        }
                    }
                    Ok(crate::pipeline::post_validate::AppliedLayout::Segmented(
                        layout,
                    ))
                },
                provenance: |_lang, _responses| {
                    Ok(TextStamp::NotStamped(
                        crate::provenance::NoStampReason::NothingApplied,
                    ))
                },
            },
            infer,
            |_items, _responses| Ok(()),
        )
        .await
        .expect_err("a stage that breaks an utterance it was given is refused");
        let ServerError::OutputAdmission {
            details: crate::error::OutputAdmissionRefusal::Judged { first, rest, .. },
            ..
        } = &error
        else {
            panic!("refused as the stage's judged output, got {error}");
        };
        let findings: Vec<&str> = std::iter::once(first)
            .chain(rest)
            .map(|finding| finding.message.as_str())
            .collect();
        // Never empty: the first finding is a field of its own.
        assert!(
            findings.iter().all(|message| message.contains("c3")),
            "only the stage's own finding: {findings:?}"
        );
    }

    /// Task 6 (a), at the stage boundary, with two speakers (so a rule about
    /// the participants cannot make every utterance fail on its own): a
    /// generated transcript diagnosed for
    /// one invalid word is segmented everywhere but in that word's utterance.
    /// Before, the whole file went unsegmented. The faulty utterance is never
    /// sent to the model, keeps its generated form, and the result is judged
    /// afresh: still diagnosed, for that word alone.
    #[tokio::test]
    async fn a_diagnosed_transcript_is_segmented_outside_its_faulty_utterance() {
        let localized = localized_b2();

        let pool = WorkerPool::new(PoolConfig::default());
        let cache = crate::cache::UtteranceCache::noop();
        let inferred = std::sync::Mutex::new(Vec::new());
        let infer = async |_pool: &WorkerPool,
                           items: &[(usize, batchalign_transform::utseg::UtsegBatchItem)],
                           _lang: &LanguageCode3|
               -> Result<
            Vec<Result<Vec<usize>, crate::text_batch::EngineItemFailure>>,
            ServerError,
        > {
            inferred
                .lock()
                .expect("test lock")
                .extend(items.iter().map(|(utterance, _)| *utterance));
            Ok(split_after_third_word(items))
        };
        let produced = run_localized_text_pipeline(
            localized,
            &LanguageCode3::eng(),
            PipelineServices::new(&pool, &cache),
            TextPipelineHooks {
                command: ReleasedCommand::Utseg,
                collect: crate::utseg::collect_utseg_batch_items,
                integrate: |state: &mut HashMap<usize, Vec<usize>>,
                            items,
                            responses: &[Vec<usize>]| {
                    for ((utterance, _), assignment) in items.iter().zip(responses) {
                        state.insert(*utterance, assignment.clone());
                    }
                },
                apply: |file, state| {
                    crate::utseg::apply_utseg_document(file, state)
                        .map(crate::pipeline::post_validate::AppliedLayout::Segmented)
                        .map_err(ServerError::from)
                },
                provenance: |_lang, _responses| {
                    Ok(TextStamp::NotStamped(
                        crate::provenance::NoStampReason::NothingApplied,
                    ))
                },
            },
            infer,
            |_items, _responses| Ok(()),
        )
        .await
        .expect("the localized stage runs");
        assert_eq!(
            *inferred.lock().expect("test lock"),
            [0, 2],
            "the faulty utterance is never inferred"
        );
        let crate::pipeline::post_validate::ProducedOutput::Diagnosed(still) = &produced else {
            panic!("the invalid word is still there, so the output is still diagnosed");
        };
        assert!(
            still
                .findings()
                .errors()
                .all(|finding| finding.message.contains("b2")),
            "{still:?}"
        );
        let text = produced.as_str();
        assert!(text.contains("*PAR0:\tthe red ball .\n"), "{text}");
        assert!(text.contains("*PAR1:\twe took b2 there .\n"), "{text}");
        assert!(text.contains("*PAR0:\tthen we went .\n"), "{text}");
    }
}
