//! Successful application owns the mutated source, layout and actual stamp.

use super::*;
use crate::api::{DisplayPath, FileStampOutcome};
use crate::pipeline::post_validate::{
    AppliedLayout, HeldOutUtterances, PostValidationFailure, ProducedOutput,
};
use crate::text_batch::{TextBatchFileResult, TextWorkflowFileError};
use talkbank_model::validation::ValidChatFile;
use tracing::info;

/// Produced only after the command's apply and provenance effects succeed.
pub(super) struct AppliedText {
    document: ChatFile,
    command: ReleasedCommand,
    layout: AppliedLayout,
    stamp: FileStampOutcome,
}

/// Final judgment and its already-computed provenance decision stay associated.
pub(super) struct StampedText {
    output: PostValidated,
    stamp: FileStampOutcome,
}

impl StampedText {
    pub(super) fn into_output(self) -> PostValidated {
        self.output
    }
    fn into_file(self, filename: DisplayPath) -> TextBatchFileResult {
        TextBatchFileResult::ok_stamped(filename, self.output, self.stamp)
    }
}

pub(super) fn complete_file<Item, Response>(
    matched: super::batch::MatchedFile<'_, Item, Response>,
    lang: &LanguageCode3,
    hooks: &TextBatchHooks<Item, Response>,
) -> TextBatchFileResult {
    let (filename, source, items, responses) = matched.into_parts();
    let output = AppliedText::batch(source, items, responses, lang, hooks)
        .and_then(|applied| applied.strict().map_err(TextWorkflowFileError::from));
    match output {
        Ok(output) => output.into_file(filename),
        Err(error) => TextBatchFileResult::err(filename, error),
    }
}

impl AppliedText {
    fn batch<Item, Response>(
        source: ValidChatFile,
        items: &[(usize, Item)],
        responses: Vec<Response>,
        lang: &LanguageCode3,
        hooks: &TextBatchHooks<Item, Response>,
    ) -> Result<Self, TextWorkflowFileError> {
        let mut document = source.into_unchecked();
        (hooks.apply)(&mut document, items, &responses)
            .map_err(|error| TextWorkflowFileError::from_server_error(&error))?;
        let provenance = (hooks.provenance)(lang, &responses)
            .map_err(|error| TextWorkflowFileError::validation(error.to_string()))?;
        let stamp = stamp(&mut document, hooks.command, provenance)
            .map_err(|error| TextWorkflowFileError::from_server_error(&error))?;
        Ok(Self {
            document,
            command: hooks.command,
            layout: AppliedLayout::InPlace,
            stamp,
        })
    }

    pub(super) fn strict(self) -> Result<StampedText, PostValidationFailure> {
        Ok(StampedText {
            output: PostValidated::gate_owned(self.document, self.command)?,
            stamp: self.stamp,
        })
    }

    pub(super) fn outside(
        self,
        held_out: HeldOutUtterances,
    ) -> Result<ProducedOutput, ServerError> {
        let held_out = held_out.after(&self.layout).ok_or_else(|| ServerError::OutputAdmission {
            command: self.command,
            details: crate::error::OutputAdmissionRefusal::unestablished(
                "the stage changed an utterance it was not given, so its output cannot be judged against the utterances that carry the findings",
            ),
        })?;
        PostValidated::produced_outside(self.document, &held_out, self.command)
            .map_err(|failure| failure.into_server_error())
    }
}

fn stamp(
    document: &mut ChatFile,
    command: ReleasedCommand,
    provenance: TextStamp,
) -> Result<FileStampOutcome, ServerError> {
    match provenance {
        TextStamp::Stamped(comment) => {
            crate::provenance::inject_provenance(document, &comment)?;
            Ok(FileStampOutcome::Stamped {
                command: command.to_string(),
            })
        }
        TextStamp::NotStamped(reason) => {
            info!(command = %command, reason = %reason, "no provenance stamp written");
            Ok(FileStampOutcome::NotStamped {
                command: command.to_string(),
                reason: reason.to_string(),
            })
        }
    }
}

/// Infer the collected items, apply the responses and stamp provenance: the
/// part of a text stage that is the same whatever the document's standing.
async fn analyze_and_apply<Item, State, Response, Failure, Infer, Observe>(
    mut chat_file: ChatFile,
    batch_items: Vec<(usize, Item)>,
    lang: &LanguageCode3,
    services: PipelineServices<'_>,
    hooks: &TextPipelineHooks<Item, State, Response>,
    infer: Infer,
    observe: Observe,
) -> Result<AppliedText, ServerError>
where
    Failure: std::fmt::Display,
    Infer: AsyncFnOnce(
        &WorkerPool,
        &[(usize, Item)],
        &LanguageCode3,
    ) -> Result<Vec<Result<Response, ItemFailure<Failure>>>, ServerError>,
    Observe: FnOnce(&[(usize, Item)], &[Response]) -> Result<(), ServerError>,
{
    let item_results = infer(services.pool, &batch_items, lang).await?;
    let matched = super::responses::MatchedPopulation::admit(batch_items, item_results)?;
    let (batch_items, item_results) = matched.into_parts();
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

    let layout = (hooks.apply)(&mut chat_file, &state_map)?;

    let stamp = stamp(
        &mut chat_file,
        hooks.command,
        (hooks.provenance)(lang, &responses)?,
    )?;
    Ok(AppliedText {
        document: chat_file,
        command: hooks.command,
        layout,
        stamp,
    })
}

/// Continue from an already admitted typed document without a CHAT reparse.
///
/// Takes and returns a [`PostValidated`]: an admitted document is the
/// precondition (a diagnosed one, which only a generating producer makes, is
/// a different type and has no route in), and the strict gate this ends with
/// is the postcondition, so the result can continue into another stage that
/// requires admission.
pub async fn run_admitted_text_pipeline<Item, State, Response, Failure, Infer, Observe>(
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
    let chat_file = admitted.into_judged_document();
    let applied = analyze_and_apply(
        chat_file,
        batch_items,
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
    applied
        .strict()
        .map(StampedText::into_output)
        .map_err(|failure| failure.into_server_error())
}

/// Continue from a generated document whose findings are confined to some
/// utterances: analyze and edit every OTHER utterance, then judge the result
/// with the producer transition. The held-out utterances are never sent to
/// the worker and keep their content, so the result is still diagnosed for
/// them (or admitted, if the edit happened to remove the finding's cause).
/// Unlike the admitted pipeline, an empty collection still re-judges the
/// model, because the input carried no proof to hand back.
pub async fn run_localized_text_pipeline<Item, State, Response, Failure, Infer, Observe>(
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
    let (chat_file, held_out) = localized.into_model();
    let batch_items: Vec<(usize, Item)> = (hooks.collect)(&chat_file)
        .into_iter()
        .filter(|(utterance, _)| !held_out.contains(*utterance))
        .collect();
    if batch_items.is_empty() {
        return PostValidated::produced_outside(chat_file, &held_out.in_place(), hooks.command)
            .map_err(|failure| failure.into_server_error());
    }
    analyze_and_apply(
        chat_file,
        batch_items,
        lang,
        services,
        &hooks,
        infer,
        observe,
    )
    .await?
    .outside(held_out)
}
