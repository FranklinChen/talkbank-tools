//! Managed native encoding: source-bound plans, ordinary file supervision,
//! and producer-owned publication, without a Python task or resident model.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use crate::api::NumWorkers;
use crate::media::export::{AudioExportError, AudioExportFormat, AudioExportPlan};
use crate::options::CommandOptions;
use crate::planning::{artifact_set_for_source, build_job_plan};
use crate::recipe_runner::materialize::{MaterializedArtifactRole, PlannedMaterializedFile};
use crate::recipe_runner::runtime::{discover_input_for_pending_file, output_write_path};
use crate::runner::DispatchHostContext;
use crate::runner::util::{
    FileRunTracker, FileStage, FileTaskOutcome, drain_supervised_file_tasks,
    spawn_supervised_file_task,
};
use crate::scheduling::{FailureCategory, WorkUnitKind};
use crate::store::{PendingJobFile, RunnerJobSnapshot};

#[derive(Debug, thiserror::Error)]
enum NativeExportAdmissionError {
    #[error("native export received options for another command")]
    Options,
    #[error(transparent)]
    Planning(#[from] crate::recipe_runner::planner::PlanningError),
    #[error("native export has no unique binary artifact for {0}")]
    Artifact(crate::api::DisplayPath),
    #[error("native export outputs collide at {0}")]
    Collision(PathBuf),
    #[error("native export destination already exists: {0}")]
    Existing(PathBuf),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// One submitted file owns its source, selected format and both output roles.
/// Consumers cannot substitute a filename or infer format from a suffix.
struct NativeExportWork {
    file: PendingJobFile,
    source: PathBuf,
    artifact: PlannedMaterializedFile,
    primary: PathBuf,
    staged: PathBuf,
    format: AudioExportFormat,
}

fn admit_work(
    job: &RunnerJobSnapshot,
    files: &[PendingJobFile],
) -> Result<Vec<NativeExportWork>, NativeExportAdmissionError> {
    let CommandOptions::Convert(options) = &job.dispatch.options else {
        return Err(NativeExportAdmissionError::Options);
    };
    let plan = build_job_plan(job)?;
    let mut destinations = HashSet::new();
    let mut work = Vec::with_capacity(files.len());
    for file in files {
        let source_file = job
            .pending_files
            .iter()
            .find(|candidate| {
                candidate.file_index == file.file_index
                    && candidate.filename == file.filename
                    && candidate.has_chat == file.has_chat
            })
            .ok_or_else(|| NativeExportAdmissionError::Artifact(file.filename.clone()))?;
        let set = artifact_set_for_source(&plan, &file.filename)
            .ok_or_else(|| NativeExportAdmissionError::Artifact(file.filename.clone()))?;
        let [artifact] = set.files.as_slice() else {
            return Err(NativeExportAdmissionError::Artifact(file.filename.clone()));
        };
        if artifact.role != MaterializedArtifactRole::Primary || !artifact.content_type.is_binary()
        {
            return Err(NativeExportAdmissionError::Artifact(file.filename.clone()));
        }
        let primary = output_write_path(&job.filesystem, file.file_index, &artifact.display_path);
        let staged = job
            .filesystem
            .staging_dir
            .join("output")
            .join(artifact.display_path.as_ref())
            .as_path()
            .to_owned();
        for destination in [&primary, &staged] {
            if destination == &staged && primary == staged {
                continue;
            }
            if !destinations.insert(destination.clone()) {
                return Err(NativeExportAdmissionError::Collision(destination.clone()));
            }
        }
        if primary == staged && !destinations.insert(primary.clone()) {
            return Err(NativeExportAdmissionError::Collision(primary));
        }
        for destination in [&primary, &staged] {
            match std::fs::symlink_metadata(destination) {
                Ok(_) => return Err(NativeExportAdmissionError::Existing(destination.clone())),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        work.push(NativeExportWork {
            file: source_file.clone(),
            source: discover_input_for_pending_file(&job.filesystem, source_file).source_path,
            artifact: artifact.clone(),
            primary,
            staged,
            format: options.format,
        });
    }
    Ok(work)
}

/// The managed runner has already reserved job capacity and memory. Native
/// file concurrency consumes that granted bound, never speculative ML workers.
pub(crate) async fn dispatch_audio_export(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    files: &[PendingJobFile],
    slots: NumWorkers,
) {
    let work = match admit_work(job, files) {
        Ok(work) => work,
        Err(error) => {
            host.sink()
                .fail_job(&job.identity.job_id, &error.to_string())
                .await;
            return;
        }
    };
    let limit = Arc::new(tokio::sync::Semaphore::new(slots.0.max(1)));
    let mut tasks = Vec::with_capacity(work.len());
    for work in work {
        let sink = host.sink().clone();
        let job_id = job.identity.job_id.clone();
        let limit = limit.clone();
        tasks.push(spawn_supervised_file_task(
            work.file.filename.clone(),
            "native export file task",
            async move {
                let filename = work.file.filename.clone();
                let lifecycle = FileRunTracker::new(sink.as_ref(), &job_id, filename.as_ref());
                let Ok(_slot) = limit.acquire_owned().await else {
                    lifecycle
                        .record_setup_failure(
                            "native execution capacity closed",
                            FailureCategory::System,
                        )
                        .await;
                    return FileTaskOutcome::TerminalStateRecorded;
                };
                lifecycle
                    .begin_first_attempt(WorkUnitKind::NativeMedia, FileStage::Processing)
                    .await;
                match work.encode_and_publish(&lifecycle).await {
                    Ok(artifact) => {
                        lifecycle
                            .complete_with_result(artifact.display_path, artifact.content_type)
                            .await
                    }
                    Err(error) => {
                        lifecycle
                            .fail(&error.to_string(), export_failure_category(&error))
                            .await
                    }
                }
                FileTaskOutcome::TerminalStateRecorded
            },
        ));
    }
    drain_supervised_file_tasks(
        host.sink().as_ref(),
        &job.identity.job_id,
        &job.cancel_token,
        tasks,
    )
    .await;
}

impl NativeExportWork {
    async fn encode_and_publish(
        self,
        lifecycle: &FileRunTracker<'_>,
    ) -> Result<PlannedMaterializedFile, AudioExportError> {
        for destination in [&self.primary, &self.staged] {
            let parent = destination.parent().ok_or_else(|| {
                AudioExportError::Stream("export destination has no directory".into())
            })?;
            tokio::fs::create_dir_all(parent).await?;
        }
        let mut plan = AudioExportPlan::admit(&self.source, &self.primary, self.format)?;
        if self.staged != self.primary {
            plan = plan.with_staged_copy(&self.staged)?;
        }
        let verified = plan.encode().await?;
        lifecycle.stage(FileStage::Writing).await;
        verified.publish()?;
        Ok(self.artifact)
    }
}

fn export_failure_category(error: &AudioExportError) -> FailureCategory {
    match error {
        AudioExportError::Stream(_)
        | AudioExportError::Refused { .. }
        | AudioExportError::Mp3Layout { .. }
        | AudioExportError::Duration(_)
        | AudioExportError::ExistingDestination(_) => FailureCategory::Validation,
        AudioExportError::Io(error) if error.kind() == std::io::ErrorKind::NotFound => {
            FailureCategory::InputMissing
        }
        AudioExportError::Io(_)
        | AudioExportError::Tool { .. }
        | AudioExportError::ProducerLayout => FailureCategory::System,
    }
}

#[cfg(test)]
mod tests;
