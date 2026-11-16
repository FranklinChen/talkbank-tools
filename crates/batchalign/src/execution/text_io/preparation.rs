//! Single-owner text preparation, admitted scheduling and publication.

use std::collections::VecDeque;
use std::future::Future;
use std::num::NonZeroUsize;
use std::sync::Arc;

use tracing::Instrument;

use crate::api::{DisplayPath, NumWorkers, ReleasedCommand};
use crate::error::ServerError;
use crate::planning::PlannedArtifactSet;
use crate::runner::util::{FileRunTracker, FileStage};
use crate::runner::{ChatDispatchInputs, DispatchHostContext};
use crate::scheduling::FailureCategory;
use crate::store::{PendingJobFile, RunnerFilesystemConfig, RunnerJobSnapshot};
use crate::text_batch::TextBatchFileInput;

mod publication;

/// Immutable run facts are shared. Preparation and publication authority move.
struct TextRun {
    job: Arc<RunnerJobSnapshot>,
    host: DispatchHostContext,
}

pub(crate) struct SegmentationText;
pub(crate) struct CoreferenceText;
pub(crate) struct TranslationText {
    options: Arc<crate::options::TranslateOptions>,
}

/// Only this producer implements command admission and prior-reading policy.
pub(crate) trait PreparationMode: Sized {
    fn admit(job: &RunnerJobSnapshot) -> Result<Self, ServerError>;
    fn prior(_: &PendingJobFile) -> Option<batchalign_types::paths::ServerPath> {
        None
    }
}
impl PreparationMode for SegmentationText {
    fn admit(job: &RunnerJobSnapshot) -> Result<Self, ServerError> {
        match (job.dispatch.command, &job.dispatch.options) {
            (ReleasedCommand::Utseg, crate::options::CommandOptions::Utseg(_)) => Ok(Self),
            (command, options) => Err(ServerError::Validation(format!(
                "segmentation preparation cannot pair {command} with {} options",
                options.command_name()
            ))),
        }
    }
}
impl PreparationMode for CoreferenceText {
    fn admit(job: &RunnerJobSnapshot) -> Result<Self, ServerError> {
        match (job.dispatch.command, &job.dispatch.options) {
            (ReleasedCommand::Coref, crate::options::CommandOptions::Coref(_)) => Ok(Self),
            (command, options) => Err(ServerError::Validation(format!(
                "coreference preparation cannot pair {command} with {} options",
                options.command_name()
            ))),
        }
    }
}
impl PreparationMode for TranslationText {
    fn admit(job: &RunnerJobSnapshot) -> Result<Self, ServerError> {
        match (job.dispatch.command, &job.dispatch.options) {
            (ReleasedCommand::Translate, crate::options::CommandOptions::Translate(options)) => {
                Ok(Self {
                    options: Arc::new(options.clone()),
                })
            }
            (command, options) => Err(ServerError::Validation(format!(
                "translation preparation cannot pair {command} with {} options",
                options.command_name()
            ))),
        }
    }
}
impl PreparationMode for super::super::worker_gateway::MorphotagRuntimeOptions {
    fn admit(job: &RunnerJobSnapshot) -> Result<Self, ServerError> {
        match (job.dispatch.command, &job.dispatch.options) {
            (ReleasedCommand::Morphotag, crate::options::CommandOptions::Morphotag(options)) => {
                Ok(Self::from_submitted(options))
            }
            (command, options) => Err(ServerError::Validation(format!(
                "morphotag preparation cannot pair {command} with {} options",
                options.command_name()
            ))),
        }
    }
    fn prior(file: &PendingJobFile) -> Option<batchalign_types::paths::ServerPath> {
        file.before_path()
    }
}

/// File fanout belongs to the two commands that process independent sources.
pub(crate) trait ConcurrentPreparation: PreparationMode {}
impl ConcurrentPreparation for SegmentationText {}
impl ConcurrentPreparation for super::super::worker_gateway::MorphotagRuntimeOptions {}

/// A queue can be scheduled or read as a batch exactly once.
pub(crate) struct PreparedTextJob<Mode> {
    mode: Mode,
    run: Arc<TextRun>,
    files: VecDeque<PreparedTextFile>,
    parallelism: NonZeroUsize,
}

/// Source membership and catalog layout were established by the producer.
/// This value is neither cloneable nor constructible by execution consumers.
pub(crate) struct PreparedTextFile {
    run: Arc<TextRun>,
    pending: PendingJobFile,
    artifacts: PlannedArtifactSet,
}

/// A successful read retains its source, prior content and publication owner.
pub(crate) struct ReadTextFile {
    owner: PreparedTextFile,
    input: TextBatchFileInput,
    before: Option<String>,
}

/// Contiguous gateway inputs and their unique publication owners travel together.
pub(crate) struct ReadTextBatch {
    inputs: Vec<TextBatchFileInput>,
    owners: Vec<PreparedTextFile>,
    run: Arc<TextRun>,
}

impl<Mode: PreparationMode> PreparedTextJob<Mode> {
    pub(crate) fn prepare(
        inputs: ChatDispatchInputs,
        workers: NumWorkers,
    ) -> Result<Self, ServerError> {
        let mode = Mode::admit(inputs.job())?;
        let parallelism = NonZeroUsize::new(workers.0).ok_or_else(|| {
            ServerError::Validation("text preparation requires a positive worker grant".into())
        })?;
        let crate::runner::DispatchInputParts { job, host, files } = inputs.into_parts();
        let policy = crate::command_model::command_spec(job.dispatch.command)
            .selected_output_policy(&job.dispatch.options)
            .map_err(|error| ServerError::Validation(error.to_string()))?;
        let run = Arc::new(TextRun { job, host });
        let files = files
            .into_iter()
            .map(|pending| {
                let artifacts = crate::recipe_runner::materialize::plan_materialized_files(
                    pending.filename(),
                    policy,
                );
                PreparedTextFile {
                    run: Arc::clone(&run),
                    pending,
                    artifacts,
                }
            })
            .collect();
        Ok(Self {
            run,
            files,
            parallelism,
            mode,
        })
    }

    pub(crate) fn host(&self) -> &DispatchHostContext {
        &self.run.host
    }

    /// Only active files are read. The runner retains its admitted job lease
    /// for dispatch. Normal queue return drains every submitted task.
    pub(crate) async fn run_files<F, Fut>(mut self, process: F)
    where
        Mode: ConcurrentPreparation,
        F: Fn(ReadTextFile, &Mode) -> Fut,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let mut tasks = tokio::task::JoinSet::new();
        while !self.run.job.cancel_token.is_cancelled() {
            if tasks.len() == self.parallelism.get() {
                Self::join_one(&mut tasks, &self.run).await;
            }
            // Cancellation may have arrived while waiting for an active file.
            if self.run.job.cancel_token.is_cancelled() {
                break;
            }
            let Some(file) = self.files.pop_front() else {
                break;
            };
            if let Some(file) = file.read::<Mode>().await {
                let span = crate::runner::current_file_span(
                    &file.input.filename,
                    self.run.job.dispatch.command.as_str(),
                );
                tasks.spawn(crate::worker::pool::job_tracker::inherit_job(
                    process(file, &self.mode).instrument(span),
                ));
            }
        }
        while !tasks.is_empty() {
            Self::join_one(&mut tasks, &self.run).await;
        }
    }

    async fn join_one(tasks: &mut tokio::task::JoinSet<()>, run: &TextRun) {
        if let Some(Err(error)) = tasks.join_next().await {
            tracing::warn!(job_id = %run.job.identity.job_id, %error, "Text file task failed");
        }
    }

    pub(crate) async fn next_file(&mut self) -> Option<ReadTextFile> {
        while !self.run.job.cancel_token.is_cancelled() {
            let file = self.files.pop_front()?;
            if let Some(file) = file.read::<Mode>().await {
                return Some(file);
            }
        }
        None
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        job: RunnerJobSnapshot,
        host: DispatchHostContext,
        workers: NumWorkers,
    ) -> Self {
        Self::prepare(
            crate::runner::DispatchInputs::for_test(job, host)
                .into_chat()
                .expect("CHAT-only fixture"),
            workers,
        )
        .expect("admitted text fixture")
    }
}

impl PreparedTextJob<TranslationText> {
    pub(crate) fn translation_options(&self) -> &Arc<crate::options::TranslateOptions> {
        &self.mode.options
    }
}

impl PreparedTextJob<CoreferenceText> {
    pub(crate) async fn read_batch(mut self) -> ReadTextBatch {
        let mut inputs = Vec::with_capacity(self.files.len());
        let mut owners = Vec::with_capacity(self.files.len());
        while let Some(ReadTextFile {
            owner,
            input,
            before: _,
        }) = self.next_file().await
        {
            inputs.push(input);
            owners.push(owner);
        }
        ReadTextBatch {
            inputs,
            owners,
            run: self.run,
        }
    }
}

impl PreparedTextFile {
    pub(crate) fn filename(&self) -> &DisplayPath {
        self.pending.filename()
    }
    pub(crate) fn filesystem(&self) -> &RunnerFilesystemConfig {
        &self.run.job.filesystem
    }
    pub(crate) fn pending(&self) -> &PendingJobFile {
        &self.pending
    }
    pub(crate) fn artifacts(&self) -> &PlannedArtifactSet {
        &self.artifacts
    }
    pub(crate) fn host(&self) -> &DispatchHostContext {
        &self.run.host
    }

    async fn read<Mode: PreparationMode>(self) -> Option<ReadTextFile> {
        let lifecycle =
            FileRunTracker::new(self.run.host.sink().as_ref(), self.filename().as_ref());
        lifecycle.stage(FileStage::Reading).await;
        let input = self
            .pending
            .input_path(&self.run.job.filesystem.staging_dir);
        let text = match tokio::fs::read_to_string(&input).await {
            Ok(text) => text,
            Err(error) => {
                lifecycle
                    .fail(
                        &format!("Failed to read input: {error}"),
                        FailureCategory::InputMissing,
                    )
                    .await;
                return None;
            }
        };
        let before = match Mode::prior(&self.pending) {
            Some(path) => match tokio::fs::read_to_string(&path).await {
                Ok(text) => Some(text),
                Err(error) => {
                    lifecycle
                        .fail(
                            &format!("Failed to read declared prior input: {error}"),
                            FailureCategory::InputMissing,
                        )
                        .await;
                    return None;
                }
            },
            None => None,
        };
        let input = TextBatchFileInput::new(self.filename().clone(), text);
        Some(ReadTextFile {
            owner: self,
            input,
            before,
        })
    }

    async fn fail(&self, error: &str) {
        FileRunTracker::new(self.host().sink().as_ref(), self.filename().as_ref())
            .fail(error, FailureCategory::System)
            .await;
    }
}

impl ReadTextFile {
    pub(crate) fn input(&self) -> &TextBatchFileInput {
        &self.input
    }
    pub(crate) fn before(&self) -> Option<&str> {
        self.before.as_deref()
    }
    pub(crate) fn host(&self) -> &DispatchHostContext {
        self.owner.host()
    }
    pub(crate) fn cancellation(&self) -> crate::infer_retry::Cancellation<'_> {
        crate::infer_retry::Cancellation::Token(&self.owner.run.job.cancel_token)
    }
}

impl ReadTextBatch {
    pub(crate) fn inputs(&self) -> &[TextBatchFileInput] {
        &self.inputs
    }
    pub(crate) fn host(&self) -> &DispatchHostContext {
        &self.run.host
    }
    pub(crate) fn cancellation(&self) -> crate::infer_retry::Cancellation<'_> {
        crate::infer_retry::Cancellation::Token(&self.run.job.cancel_token)
    }
}
