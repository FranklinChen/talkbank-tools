//! Consume source-bound inputs into completed preflight observations, record
//! their rejections, then release the accepted files for dispatch.

use std::future::Future;

use super::context::{ClaimedDispatch, DispatchHostContext};
use super::util::{FileRunTracker, RunnerEventSink};
use crate::media::MediaExtensions;
use crate::scheduling::FailureCategory;
use crate::store::{InPlaceMedia, PendingJobFile, PreflightInput};

/// The admitted files stay tied to the exact snapshot they came from.
/// Only completed rejection recording constructs the dispatch capability.
pub(super) struct PreflightedJob(DispatchInputs);

/// The kernel consumes both carried values together after preflight admission.
///
/// `job` is the ADMITTED snapshot: its `pending_files` is exactly the accepted
/// partition, because [`AdmittedJob::narrow`] replaced the claimed list before
/// this value could exist. Every dispatcher, planner and worker-count rule
/// reads `job.pending_files`, so none of them can reach a file preflight
/// rejected. (Before, the snapshot kept every claimed file beside a separate
/// accepted list, and the audio dispatchers walked the claimed one, so a
/// rejected recording could still be sent to ASR.)
pub struct DispatchInputs {
    job: std::sync::Arc<crate::store::RunnerJobSnapshot>,
    host: DispatchHostContext,
    population: Population,
}

/// Whether every admitted file is CHAT, decided once from the admitted list.
#[derive(Clone, Copy)]
enum Population {
    ChatOnly,
    WithMedia,
}

impl Population {
    fn of(files: &[PendingJobFile]) -> Self {
        let contains_media = files.iter().any(|file| match file.kind() {
            crate::store::InputKind::Chat => false,
            crate::store::InputKind::Media => true,
        });
        if contains_media {
            Self::WithMedia
        } else {
            Self::ChatOnly
        }
    }
}

/// The claimed snapshot with its pending list replaced by the accepted files.
/// The only route from a claimed job to [`DispatchInputs`].
struct AdmittedJob;

impl AdmittedJob {
    fn narrow(
        job: std::sync::Arc<crate::store::RunnerJobSnapshot>,
        accepted: Vec<PendingJobFile>,
    ) -> std::sync::Arc<crate::store::RunnerJobSnapshot> {
        let mut job = std::sync::Arc::unwrap_or_clone(job);
        job.pending_files = accepted;
        std::sync::Arc::new(job)
    }
}

/// Recorded preflight restricted to its CHAT-only source population.
pub struct ChatDispatchInputs(DispatchInputParts);

impl ChatDispatchInputs {
    pub(crate) fn job(&self) -> &crate::store::RunnerJobSnapshot {
        &self.0.job
    }
    pub(crate) fn into_parts(self) -> DispatchInputParts {
        self.0
    }
}

pub struct DispatchInputParts {
    pub(crate) job: std::sync::Arc<crate::store::RunnerJobSnapshot>,
    pub(crate) host: DispatchHostContext,
    pub(crate) files: Vec<PendingJobFile>,
}

impl DispatchInputs {
    pub(crate) const fn job(&self) -> &std::sync::Arc<crate::store::RunnerJobSnapshot> {
        &self.job
    }
    pub(crate) const fn host(&self) -> &DispatchHostContext {
        &self.host
    }
    pub(crate) fn files(&self) -> &[PendingJobFile] {
        &self.job.pending_files
    }

    pub(crate) const fn all_chat(&self) -> bool {
        matches!(self.population, Population::ChatOnly)
    }

    pub(crate) fn into_chat(self) -> Result<ChatDispatchInputs, crate::error::ServerError> {
        match self.population {
            Population::ChatOnly => Ok(ChatDispatchInputs(DispatchInputParts {
                files: self.job.pending_files.clone(),
                job: self.job,
                host: self.host,
            })),
            Population::WithMedia => Err(crate::error::ServerError::Validation(
                "text preparation requires CHAT input for every file".into(),
            )),
        }
    }

    /// An effect fixture derives its files from the same snapshot at admission.
    #[cfg(test)]
    pub(crate) fn for_test(
        job: crate::store::RunnerJobSnapshot,
        host: DispatchHostContext,
    ) -> Self {
        assert!(
            job.pending_files
                .iter()
                .all(|file| file.belongs_to(&job.filesystem.inputs))
        );
        Self {
            population: Population::of(&job.pending_files),
            job: std::sync::Arc::new(job),
            host,
        }
    }
}

impl PreflightedJob {
    pub(super) fn into_inputs(self) -> DispatchInputs {
        self.0
    }
}

#[must_use = "record rejected files before dispatching the admitted job"]
pub(super) struct ObservedPreflight {
    run: ClaimedDispatch,
    files: ObservedFiles,
}

impl ObservedPreflight {
    pub(super) async fn observe(run: ClaimedDispatch) -> Self {
        Self::observe_with(run, &FilesystemMedia).await
    }

    async fn observe_with(run: ClaimedDispatch, observer: &impl MediaObserver) -> Self {
        let files = ObservedFiles::observe_with(run.job().pending_files.clone(), observer).await;
        Self { run, files }
    }

    pub(super) async fn record(self) -> PreflightedJob {
        let files = self.files.record(self.run.host().sink().as_ref()).await;
        let super::context::ClaimedDispatchParts { job, host } = self.run.into_parts();
        let job = AdmittedJob::narrow(job, files);
        PreflightedJob(DispatchInputs {
            population: Population::of(&job.pending_files),
            job,
            host,
        })
    }
}

/// Owns each accepted or rejected input together with its completed observation.
/// No index-based reconstruction or second metadata observation is needed.
#[must_use = "record rejected files before dispatching the accepted partition"]
struct ObservedFiles {
    accepted: Vec<PendingJobFile>,
    rejected: Vec<RejectedMedia>,
}

impl ObservedFiles {
    #[cfg(test)]
    async fn observe(files: Vec<PendingJobFile>) -> Self {
        Self::observe_with(files, &FilesystemMedia).await
    }

    async fn observe_with(files: Vec<PendingJobFile>, observer: &impl MediaObserver) -> Self {
        let mut accepted = Vec::with_capacity(files.len());
        let mut rejected = Vec::new();
        for file in files {
            match file.into_preflight() {
                PreflightInput::Ready(file) => accepted.push(file),
                PreflightInput::Media(file) => {
                    match NonemptyMedia::observe(file, observer)
                        .await
                        .and_then(NonemptyMedia::admit_format)
                    {
                        Ok(file) => accepted.push(file),
                        Err(failure) => rejected.push(failure),
                    }
                }
            }
        }
        Self { accepted, rejected }
    }

    /// Complete the recording effects in input order, consuming every failure
    /// with its own filename. The sink still admits changing run generations.
    async fn record(self, sink: &dyn RunnerEventSink) -> Vec<PendingJobFile> {
        for failure in self.rejected {
            FileRunTracker::new(sink, failure.file.filename())
                .record_setup_failure(&failure.description(), FailureCategory::Validation)
                .await;
        }
        self.accepted
    }
}

/// The shared observation seam has one input capability and closed outcomes.
/// A double supplies the observed filesystem result, never an index or filename.
trait MediaObserver: Sync {
    fn size(&self, file: &InPlaceMedia) -> impl Future<Output = MediaSizeObservation> + Send;
}

enum MediaSizeObservation {
    Unavailable(std::io::Error),
    Empty,
    Nonempty,
}

struct FilesystemMedia;

impl MediaObserver for FilesystemMedia {
    async fn size(&self, file: &InPlaceMedia) -> MediaSizeObservation {
        match tokio::fs::metadata(file.source()).await {
            Err(error) => MediaSizeObservation::Unavailable(error),
            Ok(meta) if meta.len() == 0 => MediaSizeObservation::Empty,
            Ok(_) => MediaSizeObservation::Nonempty,
        }
    }
}

/// Only a successful nonempty metadata observation can reach format admission.
struct NonemptyMedia(InPlaceMedia);

impl NonemptyMedia {
    async fn observe(
        file: InPlaceMedia,
        observer: &impl MediaObserver,
    ) -> Result<Self, RejectedMedia> {
        match observer.size(&file).await {
            MediaSizeObservation::Nonempty => Ok(Self(file)),
            MediaSizeObservation::Empty => Err(RejectedMedia {
                file,
                reason: MediaRefusal::Empty,
            }),
            MediaSizeObservation::Unavailable(error) => Err(RejectedMedia {
                file,
                reason: MediaRefusal::Unavailable(error),
            }),
        }
    }

    fn admit_format(self) -> Result<PendingJobFile, RejectedMedia> {
        let Self(file) = self;
        let extension = file
            .source()
            .as_path()
            .extension()
            .and_then(|ext| ext.to_str())
            .map(str::to_lowercase);
        match extension {
            Some(extension) if MediaExtensions::is_known(&extension) => Ok(file.into_pending()),
            Some(extension) => Err(RejectedMedia {
                file,
                reason: MediaRefusal::UnknownFormat(extension),
            }),
            None => Err(RejectedMedia {
                file,
                reason: MediaRefusal::NoExtension,
            }),
        }
    }
}

struct RejectedMedia {
    file: InPlaceMedia,
    reason: MediaRefusal,
}

enum MediaRefusal {
    Unavailable(std::io::Error),
    Empty,
    UnknownFormat(String),
    NoExtension,
}

impl RejectedMedia {
    fn description(&self) -> String {
        let path = self.file.source();
        match &self.reason {
            MediaRefusal::Unavailable(error) => {
                if error.kind() == std::io::ErrorKind::NotFound {
                    format!("File not found: {path}")
                } else {
                    format!("Cannot inspect media file {path}: {error}")
                }
            }
            MediaRefusal::Empty => format!("File is empty: {path}"),
            MediaRefusal::UnknownFormat(extension) => {
                format!("Unknown media format '.{extension}': {path}")
            }
            MediaRefusal::NoExtension => format!("File has no extension: {path}"),
        }
    }
}

#[cfg(test)]
mod tests;
