//! Single-owner text preparation, admitted scheduling and publication.

use std::collections::VecDeque;
use std::future::Future;
use std::num::NonZeroUsize;
use std::sync::Arc;

use crate::api::{DisplayPath, NumWorkers, ReleasedCommand};
use crate::error::ServerError;
use crate::planning::PlannedArtifactSet;
use crate::runner::util::{
    FileRunTracker, FileStage, FileTaskOutcome, SupervisedFileTasks, TerminalRecorded,
    spawn_supervised_file_task,
};
use crate::runner::{ChatDispatchInputs, DispatchHostContext};
use crate::scheduling::FailureCategory;
use crate::store::{PendingJobFile, RunnerFilesystemConfig, RunnerJobSnapshot};
use crate::text_batch::TextBatchFileInput;

mod publication;
mod submission;
pub use submission::{
    CorefSubmission, MorphotagSource, MorphotagSubmission, TranslateSubmission, UtsegSubmission,
};

/// Immutable run facts are shared. Preparation and publication authority move.
struct TextRun {
    job: Arc<RunnerJobSnapshot>,
    host: DispatchHostContext,
}

pub struct SegmentationText;
pub struct CoreferenceText;
pub struct TranslationText {
    options: Arc<crate::options::TranslateOptions>,
}

/// Only this producer implements command admission and prior-reading policy.
pub trait PreparationMode: Sized {
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
pub trait ConcurrentPreparation: PreparationMode {}
impl ConcurrentPreparation for SegmentationText {}
impl ConcurrentPreparation for super::super::worker_gateway::MorphotagRuntimeOptions {}

/// A queue can be scheduled or read as a batch exactly once.
pub struct PreparedTextJob<Mode> {
    mode: Arc<Mode>,
    run: Arc<TextRun>,
    files: VecDeque<PreparedTextFile>,
    parallelism: NonZeroUsize,
}

/// Source membership and catalog layout were established by the producer.
/// This value is neither cloneable nor constructible by execution consumers.
pub struct PreparedTextFile {
    run: Arc<TextRun>,
    pending: PendingJobFile,
    artifacts: PlannedArtifactSet,
}

/// A successful read retains its source, prior content and publication owner.
pub struct ReadTextFile<Mode> {
    owner: PreparedTextFile,
    input: TextBatchFileInput,
    before: Option<String>,
    mode: Arc<Mode>,
}

/// Contiguous gateway inputs and their unique publication owners travel together.
pub struct ReadTextBatch {
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
            mode: Arc::new(mode),
        })
    }

    pub(crate) fn host(&self) -> &DispatchHostContext {
        &self.run.host
    }

    /// Only active files are read. The runner retains its admitted job lease
    /// for dispatch. Each read file runs as a supervised file task, so a task
    /// that panics or is stopped by the job's cancellation still settles its
    /// own file; at most `parallelism` run at once.
    pub(crate) async fn run_files<F, Fut>(mut self, process: F)
    where
        Mode: ConcurrentPreparation,
        F: Fn(ReadTextFile<Mode>) -> Fut,
        Fut: Future<Output = FileTaskOutcome> + Send + 'static,
    {
        let sink = Arc::clone(self.run.host.sink());
        let cancel_token = self.run.job.cancel_token.clone();
        let role = self.run.job.dispatch.command.as_str();
        let mut tasks = SupervisedFileTasks::new();
        while !cancel_token.is_cancelled() {
            if tasks.len() == self.parallelism.get() {
                tasks.settle_next(sink.as_ref(), &cancel_token).await;
            }
            // Cancellation may have arrived while waiting for an active file.
            if cancel_token.is_cancelled() {
                break;
            }
            let Some(file) = self.files.pop_front() else {
                break;
            };
            if let Some(file) = file.read(Arc::clone(&self.mode)).await {
                let filename = file.filename().clone();
                tasks.push(spawn_supervised_file_task(filename, role, process(file)));
            }
        }
        tasks.drain(sink.as_ref(), &cancel_token).await;
    }

    async fn read_next(&mut self) -> Option<ReadTextFile<Mode>> {
        while !self.run.job.cancel_token.is_cancelled() {
            let file = self.files.pop_front()?;
            if let Some(file) = file.read(Arc::clone(&self.mode)).await {
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
    pub(crate) async fn next_file(&mut self) -> Option<ReadTextFile<TranslationText>> {
        self.read_next().await
    }
}

#[cfg(test)]
impl<Mode: ConcurrentPreparation> PreparedTextJob<Mode> {
    pub(crate) async fn next_file_for_test(&mut self) -> Option<ReadTextFile<Mode>> {
        self.read_next().await
    }
}

impl PreparedTextJob<CoreferenceText> {
    pub(crate) async fn read_batch(mut self) -> Option<ReadTextBatch> {
        let first = self.read_next().await?;
        let mut inputs = Vec::with_capacity(self.files.len() + 1);
        let mut owners = Vec::with_capacity(self.files.len() + 1);
        inputs.push(first.input);
        owners.push(first.owner);
        while let Some(file) = self.read_next().await {
            inputs.push(file.input);
            owners.push(file.owner);
        }
        Some(ReadTextBatch {
            inputs,
            owners,
            run: self.run,
        })
    }
}

impl PreparedTextFile {
    pub(crate) const fn filename(&self) -> &DisplayPath {
        self.pending.filename()
    }
    pub(crate) fn filesystem(&self) -> &RunnerFilesystemConfig {
        &self.run.job.filesystem
    }
    pub(crate) const fn pending(&self) -> &PendingJobFile {
        &self.pending
    }
    pub(crate) const fn artifacts(&self) -> &PlannedArtifactSet {
        &self.artifacts
    }
    pub(crate) fn host(&self) -> &DispatchHostContext {
        &self.run.host
    }

    async fn read<Mode: PreparationMode>(self, mode: Arc<Mode>) -> Option<ReadTextFile<Mode>> {
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
            mode,
        })
    }

    async fn fail(&self, error: &str) -> TerminalRecorded {
        FileRunTracker::new(self.host().sink().as_ref(), self.filename().as_ref())
            .fail(error, FailureCategory::System)
            .await
    }
}

impl<Mode> ReadTextFile<Mode> {
    pub(crate) fn filename(&self) -> &DisplayPath {
        self.owner.filename()
    }

    #[cfg(test)]
    pub(crate) const fn input(&self) -> &TextBatchFileInput {
        &self.input
    }
    #[cfg(test)]
    pub(crate) fn before(&self) -> Option<&str> {
        self.before.as_deref()
    }
}
