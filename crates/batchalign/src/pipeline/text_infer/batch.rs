//! Collection owns file identities and spans. Only admitted response populations
//! can consume those spans and reach successful application.

use super::*;
use crate::api::DisplayPath;
use crate::text_batch::{TextBatchFileResult, TextWorkflowFileError, unwrap_per_item_results};
use talkbank_model::validation::ValidChatFile;

pub(super) enum CollectedTextBatch<Item> {
    Complete(TextBatchFileResults),
    Pending(PendingTextBatch<Item>),
}

pub(super) struct PendingTextBatch<Item> {
    files: Vec<CollectedFile>,
    items: Vec<(usize, Item)>,
}

enum CollectedFile {
    Pending {
        filename: DisplayPath,
        input: ValidChatFile,
        span: FileSpan,
    },
    Complete(TextBatchFileResult),
}

/// Only collection constructs spans into its own immutable item population.
struct FileSpan(std::ops::Range<usize>);

/// Created at the collector edge only after it produced at least one item.
struct NonEmptyFileItems<Item>(Vec<(usize, Item)>);

struct AnsweredTextBatch<Item, Response, Failure> {
    files: Vec<CollectedFile>,
    matched: super::responses::MatchedPopulation<Item, Response, Failure>,
}

pub(super) struct MatchedFile<'items, Item, Response> {
    filename: DisplayPath,
    input: ValidChatFile,
    items: &'items [(usize, Item)],
    responses: Vec<Response>,
}
impl<'items, Item, Response> MatchedFile<'items, Item, Response> {
    pub(super) fn into_parts(
        self,
    ) -> (
        DisplayPath,
        ValidChatFile,
        &'items [(usize, Item)],
        Vec<Response>,
    ) {
        (self.filename, self.input, self.items, self.responses)
    }
}

impl<Item> CollectedTextBatch<Item> {
    pub(super) fn collect<Response>(
        files: &[TextBatchFileInput],
        hooks: &TextBatchHooks<Item, Response>,
    ) -> Self {
        let parser = crate::chat_parser();
        let mut batch = Self::Complete(Vec::with_capacity(files.len()));
        for file in files {
            let input = match admit_retained_text(
                &parser,
                &file.chat_text,
                TranscriptName::for_path(std::path::Path::new(file.filename.as_ref())),
            ) {
                Ok(input) => input,
                Err(error) => {
                    batch.push_complete(TextBatchFileResult::err(
                        file.filename.clone(),
                        TextWorkflowFileError::from_server_error(&error),
                    ));
                    continue;
                }
            };
            let work = if is_dummy(input.document()) {
                Vec::new()
            } else {
                (hooks.collect)(input.document())
            };
            if work.is_empty() {
                batch.push_complete(TextBatchFileResult::ok(
                    file.filename.clone(),
                    PostValidated::pass_through(input, hooks.command),
                ));
                continue;
            }
            batch = batch.attach_pending(
                file.filename.clone(),
                input.into_valid_file(),
                NonEmptyFileItems(work),
            );
        }
        batch
    }

    fn push_complete(&mut self, result: TextBatchFileResult) {
        match self {
            Self::Complete(results) => results.push(result),
            Self::Pending(pending) => pending.files.push(CollectedFile::Complete(result)),
        }
    }

    fn attach_pending(
        self,
        filename: DisplayPath,
        input: ValidChatFile,
        work: NonEmptyFileItems<Item>,
    ) -> Self {
        let mut pending = match self {
            Self::Complete(results) => PendingTextBatch {
                files: results.into_iter().map(CollectedFile::Complete).collect(),
                items: Vec::new(),
            },
            Self::Pending(pending) => pending,
        };
        let start = pending.items.len();
        pending.items.extend(work.0);
        pending.files.push(CollectedFile::Pending {
            filename,
            input,
            span: FileSpan(start..pending.items.len()),
        });
        Self::Pending(pending)
    }

    pub(super) async fn run_with<Response, Failure: std::fmt::Display, Infer>(
        self,
        lang: &LanguageCode3,
        hooks: &TextBatchHooks<Item, Response>,
        infer: Infer,
    ) -> TextBatchFileResults
    where
        Infer: AsyncFnOnce(
            &[(usize, Item)],
        )
            -> Result<Vec<Result<Response, ItemFailure<Failure>>>, ServerError>,
    {
        match self {
            Self::Complete(results) => results,
            Self::Pending(pending) => {
                let responses = match infer(&pending.items).await {
                    Ok(responses) => responses,
                    Err(error) => return pending.refuse(error),
                };
                match pending.admit(responses) {
                    Ok(answered) => answered.apply(lang, hooks),
                    Err(results) => results,
                }
            }
        }
    }
}

impl<Item> PendingTextBatch<Item> {
    fn refuse(self, error: ServerError) -> TextBatchFileResults {
        refuse_files(self.files, error)
    }

    fn admit<Response, Failure>(
        self,
        responses: Vec<Result<Response, ItemFailure<Failure>>>,
    ) -> Result<AnsweredTextBatch<Item, Response, Failure>, TextBatchFileResults> {
        let Self { files, items } = self;
        match super::responses::MatchedPopulation::admit(items, responses) {
            Ok(matched) => Ok(AnsweredTextBatch { files, matched }),
            Err(error) => Err(refuse_files(files, error)),
        }
    }
}

fn refuse_files(files: Vec<CollectedFile>, error: ServerError) -> TextBatchFileResults {
    files
        .into_iter()
        .map(|file| match file {
            CollectedFile::Complete(result) => result,
            CollectedFile::Pending { filename, .. } => {
                TextBatchFileResult::err(filename, TextWorkflowFileError::from_server_error(&error))
            }
        })
        .collect()
}

impl<Item, Response, Failure: std::fmt::Display> AnsweredTextBatch<Item, Response, Failure> {
    fn apply(
        self,
        lang: &LanguageCode3,
        hooks: &TextBatchHooks<Item, Response>,
    ) -> TextBatchFileResults {
        let (items, responses) = self.matched.into_parts();
        let mut responses = responses.into_iter();
        let files = self.files;
        files
            .into_iter()
            .map(|file| match file {
                CollectedFile::Complete(result) => result,
                CollectedFile::Pending {
                    filename,
                    input,
                    span,
                } => {
                    // Admission established the total population. Each private span
                    // belongs to this original collection, in order, exactly once.
                    let file_responses = responses.by_ref().take(span.0.len()).collect();
                    match unwrap_per_item_results(hooks.command.as_str(), file_responses) {
                        Ok(responses) => application::complete_file(
                            MatchedFile {
                                filename,
                                input,
                                items: &items[span.0],
                                responses,
                            },
                            lang,
                            hooks,
                        ),
                        Err(error) => TextBatchFileResult::err(filename, error),
                    }
                }
            })
            .collect()
    }
}
