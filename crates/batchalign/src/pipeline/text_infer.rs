//! Shared text-infer pipeline skeleton: collect payloads, run worker
//! inference, apply results back to the CHAT AST. Supports both
//! single-file and cross-file-batch flows.

use std::collections::HashMap;

use crate::api::{DisplayPath, InvalidStampSafeText, LanguageCode3, ReleasedCommand};
use crate::chat_ops::ChatFile;
use crate::provenance::TextStamp;
use crate::text_batch::{ItemFailure, TextBatchFileInput, TextBatchFileResults};
use crate::worker::pool::WorkerPool;
use batchalign_transform::parse::is_dummy;
use batchalign_transform::{AdmittedSourceChat, parse_source_with_parser};
use talkbank_model::model::TranscriptName;

use crate::error::{ServerError, SourceRefusal};
use crate::pipeline::PipelineServices;
use crate::pipeline::post_validate::PostValidated;

mod application;
mod batch;
mod responses;

type IntegrateFn<Item, State, Response> =
    fn(&mut HashMap<usize, State>, &[(usize, Item)], &[Response]);

/// These workflows retain input content: no generated-tier recovery exemption
/// applies. Only Chatter's complete source admission can produce their input.
pub fn admit_retained_text<'source>(
    parser: &batchalign_transform::parse::TreeSitterParser,
    source: &'source str,
    name: TranscriptName<'_>,
) -> Result<AdmittedSourceChat<'source>, ServerError> {
    parse_source_with_parser(parser, source)
        .admit(name, &talkbank_model::NullErrorSink)
        .map_err(|error| ServerError::ChatAdmission(SourceRefusal::in_source(error, source)))
}

/// The staged name and the admitted parse of that file's exact bytes.
/// Only named source admission can produce this pair.
pub struct AdmittedTextSource<'source> {
    filename: &'source DisplayPath,
    source: AdmittedSourceChat<'source>,
}

impl<'source> AdmittedTextSource<'source> {
    pub(crate) fn admit(
        parser: &batchalign_transform::parse::TreeSitterParser,
        input: &'source TextBatchFileInput,
    ) -> Result<Self, ServerError> {
        let name = TranscriptName::for_path(std::path::Path::new(input.filename.as_ref()));
        let source = admit_retained_text(parser, &input.chat_text, name)?;
        Ok(Self {
            filename: &input.filename,
            source,
        })
    }

    pub(crate) fn document(&self) -> &ChatFile {
        self.source.document()
    }

    pub(crate) fn into_parts(self) -> (&'source DisplayPath, AdmittedSourceChat<'source>) {
        (self.filename, self.source)
    }
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
pub type TextProvenance<Response> =
    fn(&LanguageCode3, &[Response]) -> Result<TextStamp, InvalidStampSafeText>;

/// Hooks for a text-only single-file pipeline.
///
/// `collect` extracts payloads from the parsed CHAT; `integrate` merges
/// responses into the state map; `apply` writes the state back into
/// the CHAT AST. The inference function itself is a separate argument
/// to [`run_admitted_text_pipeline`] so callers can pass any `async fn` directly.
pub struct TextPipelineHooks<Item, State, Response> {
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

pub use application::{run_admitted_text_pipeline, run_localized_text_pipeline};

/// Hooks for the cross-file text-batch pipeline (pool all files'
/// payloads into one `batch_infer` call, then redistribute responses).
///
/// Shared between `utseg` and `translate`. Morphotag does not use this
/// because it has additional structure (language-group dispatch, L2
/// secondary dispatch, alignment validation) that doesn't fit the
/// generic shape.
/// Extract worker payloads from a parsed chat file. Each payload is tagged
/// with its utterance index so the injector can match responses back.
pub type TextBatchCollect<Item> = fn(&ChatFile) -> Vec<(usize, Item)>;

/// Apply one file's collected items + their worker responses to the chat
/// file's AST. Called once per file after global inference completes.
pub type TextBatchApply<Item, Response> =
    fn(&mut ChatFile, &[(usize, Item)], &[Response]) -> Result<(), ServerError>;

pub struct TextBatchHooks<Item, Response> {
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
/// Source admission parses and validates each source once. Collection retains
/// completed refusals and exact no-work outcomes, while pending files own their
/// valid source and private spans into the collected item population. Inference
/// runs once over that population. `MatchedPopulation` admits the actual response
/// count before consuming per-file partition, application, stamping and output
/// judgment. Worker failures refuse pending files while moving completed outcomes
/// through unchanged.
pub async fn run_text_batch_pipeline<Item, Response, Failure, Infer>(
    files: &[TextBatchFileInput],
    lang: &LanguageCode3,
    pool: &WorkerPool,
    hooks: TextBatchHooks<Item, Response>,
    infer: Infer,
) -> TextBatchFileResults
where
    Failure: std::fmt::Display,
    Infer: AsyncFnOnce(
        &WorkerPool,
        &[(usize, Item)],
        &LanguageCode3,
    ) -> Result<Vec<Result<Response, ItemFailure<Failure>>>, ServerError>,
{
    batch::CollectedTextBatch::collect(files, &hooks)
        .run_with(lang, &hooks, async |items| infer(pool, items, lang).await)
        .await
}

/// Run the shared text pipeline on the admitted source used to set its route.
/// Collection receives the model and consumes its proof without parsing bytes.
pub async fn run_admitted_text_source<Item, Response, Failure, Infer>(
    source: AdmittedTextSource<'_>,
    lang: &LanguageCode3,
    hooks: TextBatchHooks<Item, Response>,
    infer: Infer,
) -> TextBatchFileResults
where
    Failure: std::fmt::Display,
    Infer: AsyncFnOnce(
        &[(usize, Item)],
    ) -> Result<Vec<Result<Response, ItemFailure<Failure>>>, ServerError>,
{
    batch::CollectedTextBatch::one(source, &hooks)
        .run_with(lang, &hooks, infer)
        .await
}

#[cfg(test)]
mod tests;
