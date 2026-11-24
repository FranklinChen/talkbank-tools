//! Only a read source can construct its non-cloneable command submission.

use super::publication::{publish_batch, publish_one, publish_unstamped};
use super::{PreparedTextFile, ReadTextBatch, ReadTextFile, SegmentationText, TranslationText};
use crate::execution::morphotag::progress::BackendProgressPort;
use crate::execution::worker_gateway::{
    CorefGateway, MorphotagGateway, MorphotagRuntimeOptions, TranslateGateway, UtsegGateway,
};
use crate::infer_retry::Cancellation;
use crate::pipeline::text_infer::AdmittedTextSource;
use crate::runner::util::{FileRunTracker, FileStage, FileTaskOutcome, observing_file_waits};
use crate::scheduling::WorkUnitKind;
use crate::text_batch::{TextBatchFileInput, TextWorkflowFileError};
use crate::translate::TranslationRoute;
use crate::utseg_route::ServableUtsegRoute;
use std::sync::Arc;

pub struct UtsegSubmission<'a> {
    input: TextBatchFileInput,
    route: &'a ServableUtsegRoute,
    cancellation: Cancellation<'a>,
}

impl<'a> UtsegSubmission<'a> {
    pub(crate) fn into_parts(
        self,
    ) -> (TextBatchFileInput, &'a ServableUtsegRoute, Cancellation<'a>) {
        (self.input, self.route, self.cancellation)
    }
}

pub struct TranslateSubmission<'a> {
    input: AdmittedTextSource<'a>,
    route: TranslationRoute,
    cancellation: Cancellation<'a>,
}

impl<'a> TranslateSubmission<'a> {
    fn admit(
        file: &'a TextBatchFileInput,
        options: &crate::options::TranslateOptions,
        cancellation: Cancellation<'a>,
    ) -> Result<Self, crate::error::ServerError> {
        let input = AdmittedTextSource::admit(&crate::chat_parser(), file)?;
        let language = crate::pipeline::morphosyntax::resolve_per_file_lang(input.document())?;
        let route = TranslationRoute::for_file(language, options);
        Ok(Self {
            input,
            route,
            cancellation,
        })
    }

    pub(crate) fn into_parts(self) -> (AdmittedTextSource<'a>, TranslationRoute, Cancellation<'a>) {
        (self.input, self.route, self.cancellation)
    }
}

pub struct CorefSubmission<'a> {
    inputs: Vec<TextBatchFileInput>,
    cancellation: Cancellation<'a>,
}

impl<'a> CorefSubmission<'a> {
    pub(crate) fn into_parts(self) -> (Vec<TextBatchFileInput>, Cancellation<'a>) {
        (self.inputs, self.cancellation)
    }
}

pub struct MorphotagSubmission<'a> {
    source: MorphotagSource,
    options: Arc<MorphotagRuntimeOptions>,
    progress: Option<BackendProgressPort>,
    cancellation: Cancellation<'a>,
}

pub enum MorphotagSource {
    Full(String),
    Incremental { before: String, after: String },
}

impl<'a> MorphotagSubmission<'a> {
    pub(crate) fn into_parts(
        self,
    ) -> (
        MorphotagSource,
        Arc<MorphotagRuntimeOptions>,
        Option<BackendProgressPort>,
        Cancellation<'a>,
    ) {
        (self.source, self.options, self.progress, self.cancellation)
    }
}

impl PreparedTextFile {
    async fn begin_submission(&self, stage: FileStage) {
        FileRunTracker::new(self.host().sink().as_ref(), self.filename().as_ref())
            .begin_first_attempt(WorkUnitKind::BatchInfer, stage)
            .await;
    }

    fn submission_cancellation(&self) -> Cancellation<'_> {
        Cancellation::Token(&self.run.job.cancel_token)
    }

    async fn observe_submission<T>(&self, call: crate::owned_future::OwnedFuture<'_, T>) -> T {
        observing_file_waits(self.host().sink(), [self.filename().to_string()], call).await
    }
}

impl ReadTextFile<SegmentationText> {
    pub(crate) async fn submit(
        self,
        gateway: &dyn UtsegGateway,
        route: &ServableUtsegRoute,
    ) -> FileTaskOutcome {
        let Self { owner, input, .. } = self;
        owner.begin_submission(FileStage::Segmenting).await;
        let results = owner
            .observe_submission(gateway.utseg_file(UtsegSubmission {
                input,
                route,
                cancellation: owner.submission_cancellation(),
            }))
            .await;
        publish_one(owner, results).await.into()
    }
}

impl ReadTextFile<TranslationText> {
    pub(crate) async fn submit(self, gateway: &dyn TranslateGateway) -> FileTaskOutcome {
        let Self {
            owner, input, mode, ..
        } = self;
        owner.begin_submission(FileStage::Translating).await;
        let submission = match TranslateSubmission::admit(
            &input,
            &mode.options,
            owner.submission_cancellation(),
        ) {
            Ok(submission) => submission,
            Err(error) => {
                drop(input);
                return publish_unstamped(
                    owner,
                    Err(TextWorkflowFileError::from_server_error(&error)),
                )
                .await
                .into();
            }
        };
        let results = owner
            .observe_submission(gateway.translate_file(submission))
            .await;
        drop(input);
        publish_one(owner, results).await.into()
    }
}

impl ReadTextFile<MorphotagRuntimeOptions> {
    pub(crate) async fn submit(
        self,
        gateway: &dyn MorphotagGateway,
        progress: Option<BackendProgressPort>,
    ) -> FileTaskOutcome {
        let Self {
            owner,
            input,
            before,
            mode,
        } = self;
        owner.begin_submission(FileStage::Parsing).await;
        let source = match before {
            Some(before) => MorphotagSource::Incremental {
                before,
                after: input.chat_text,
            },
            None => MorphotagSource::Full(input.chat_text),
        };
        let result = owner
            .observe_submission(gateway.morphotag_single(MorphotagSubmission {
                source,
                options: mode,
                progress,
                cancellation: owner.submission_cancellation(),
            }))
            .await;
        publish_unstamped(
            owner,
            result.map_err(|error| TextWorkflowFileError::from_server_error(&error)),
        )
        .await
        .into()
    }
}

impl ReadTextBatch {
    pub(crate) async fn submit(self, gateway: &dyn CorefGateway) {
        let Self {
            inputs,
            owners,
            run,
        } = self;
        for owner in &owners {
            owner
                .begin_submission(FileStage::ResolvingCoreference)
                .await;
        }
        let results = observing_file_waits(
            run.host.sink(),
            owners.iter().map(|owner| owner.filename().to_string()),
            gateway.coref_batch(CorefSubmission {
                inputs,
                cancellation: Cancellation::Token(&run.job.cancel_token),
            }),
        )
        .await;
        publish_batch(owners, results).await;
    }
}
