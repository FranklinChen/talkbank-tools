//! Source admission and destination ownership for per-file output writes.

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use crate::api::{DisplayPath, ReleasedCommand};
use crate::options::CommandOptions;
use crate::recipe_runner::materialize::{
    ChatMetricsArtifacts, MetricsArtifact, PlannedArtifactSet, PrimaryArtifact,
};
use crate::recipe_runner::planner::PlanningError;
use crate::store::{InputLocation, PendingJobFile, RunnerFilesystemConfig};

/// Only the two catalog artifact roles provide their own output path.
trait ArtifactPath {
    fn display_path(&self) -> &DisplayPath;
}

impl ArtifactPath for PrimaryArtifact {
    fn display_path(&self) -> &DisplayPath {
        PrimaryArtifact::display_path(self)
    }
}

impl ArtifactPath for MetricsArtifact {
    fn display_path(&self) -> &DisplayPath {
        MetricsArtifact::display_path(self)
    }
}

/// Input membership is established once and retained with the selected layout.
struct OutputSource<'a> {
    filesystem: &'a RunnerFilesystemConfig,
    file: &'a PendingJobFile,
}

/// The admitted input identity carried by a successful writer's receipt.
#[derive(Debug)]
pub(crate) struct OutputSourceIdentity<'a> {
    file: &'a PendingJobFile,
}

impl<'a> OutputSourceIdentity<'a> {
    pub(crate) fn filename(&self) -> &'a DisplayPath {
        self.file.filename()
    }
}

impl<'a> OutputSource<'a> {
    fn identity(&self) -> OutputSourceIdentity<'a> {
        OutputSourceIdentity { file: self.file }
    }
    fn admit(
        filesystem: &'a RunnerFilesystemConfig,
        file: &'a PendingJobFile,
    ) -> Result<Self, PlanningError> {
        if !file.belongs_to(&filesystem.inputs) {
            return Err(PlanningError::OutputSourceMismatch(file.filename().clone()));
        }
        Ok(Self { filesystem, file })
    }

    fn target<'b, A: ArtifactPath>(
        &self,
        artifact: &'b A,
    ) -> Result<OutputTarget<'b, A>, PlanningError> {
        let display = artifact.display_path();
        let staged = self
            .filesystem
            .staging_dir
            .join("output")
            .join(display.as_ref())
            .as_path()
            .to_owned();
        let destinations = match self.file.location() {
            InputLocation::Staged => OutputDestinations::Single(staged),
            InputLocation::Paths(paths) => {
                let name = Path::new(display.as_ref())
                    .file_name()
                    .ok_or_else(|| PlanningError::MissingFileName(display.clone()))?;
                let requested = paths.output.server().as_path().with_file_name(name);
                if requested == staged {
                    OutputDestinations::Single(staged)
                } else {
                    OutputDestinations::Copies { requested, staged }
                }
            }
        };
        Ok(OutputTarget {
            artifact,
            destinations,
        })
    }
}

/// A catalog layout associated with its declared input and filesystem.
/// Borrowed plans retain their completed naming computation without cloning it.
pub(crate) struct SourceOutputs<'a> {
    source: OutputSource<'a>,
    artifacts: Cow<'a, PlannedArtifactSet>,
}

impl<'a> SourceOutputs<'a> {
    /// Preparation owns the exact preflight source and its derived layout.
    pub(crate) fn for_text_file(file: &'a crate::execution::text_io::PreparedTextFile) -> Self {
        Self {
            source: OutputSource {
                filesystem: file.filesystem(),
                file: file.pending(),
            },
            artifacts: Cow::Borrowed(file.artifacts()),
        }
    }

    pub(crate) fn plan(
        filesystem: &'a RunnerFilesystemConfig,
        file: &'a PendingJobFile,
        command: ReleasedCommand,
        options: &CommandOptions,
    ) -> Result<Self, PlanningError> {
        let source = OutputSource::admit(filesystem, file)?;
        let artifacts = super::planned_output_artifacts(command, options, file.filename())?;
        Ok(Self {
            source,
            artifacts: Cow::Owned(artifacts),
        })
    }

    pub(crate) fn admit_planned(
        filesystem: &'a RunnerFilesystemConfig,
        file: &'a PendingJobFile,
        artifacts: &'a PlannedArtifactSet,
    ) -> Result<Self, PlanningError> {
        let source = OutputSource::admit(filesystem, file)?;
        if artifacts.source() != file.filename() {
            return Err(PlanningError::OutputSourceMismatch(file.filename().clone()));
        }
        Ok(Self {
            source,
            artifacts: Cow::Borrowed(artifacts),
        })
    }

    pub(crate) fn primary(&self) -> &PrimaryArtifact {
        self.artifacts.primary()
    }

    pub(crate) fn source_identity(&self) -> OutputSourceIdentity<'a> {
        self.source.identity()
    }

    pub(crate) fn diagnostics_path(&self) -> PathBuf {
        self.source
            .filesystem
            .staging_dir
            .as_path()
            .join("diagnostics")
            .join(format!("{}.findings.json", self.primary().display_path()))
    }

    pub(crate) fn into_primary(self) -> crate::recipe_runner::materialize::PlannedMaterializedFile {
        match self.artifacts {
            Cow::Owned(layout) => layout.into_primary(),
            Cow::Borrowed(layout) => layout.primary().clone(),
        }
        .into_materialized()
    }

    pub(crate) fn primary_target(
        &self,
    ) -> Result<OutputTarget<'_, PrimaryArtifact>, PlanningError> {
        let artifact = self.primary();
        self.source.target(artifact)
    }

    pub(crate) fn into_chat_metrics(self) -> Result<ChatMetricsOutputs<'a>, PlanningError> {
        let artifacts =
            match self.artifacts {
                Cow::Borrowed(layout) => {
                    Cow::Borrowed(layout.chat_metrics().ok_or_else(|| {
                        PlanningError::OutputLayoutMismatch(layout.source().clone())
                    })?)
                }
                Cow::Owned(layout) => Cow::Owned(layout.into_chat_metrics().map_err(|layout| {
                    PlanningError::OutputLayoutMismatch(layout.source().clone())
                })?),
            };
        Ok(ChatMetricsOutputs {
            source: self.source,
            artifacts,
        })
    }
}

/// The named CHAT/CSV layout retains its admitted source through persistence.
pub(crate) struct ChatMetricsOutputs<'a> {
    source: OutputSource<'a>,
    artifacts: Cow<'a, ChatMetricsArtifacts>,
}

impl<'a> ChatMetricsOutputs<'a> {
    pub(crate) fn source_identity(&self) -> OutputSourceIdentity<'a> {
        self.source.identity()
    }
    pub(crate) fn artifacts(&self) -> &ChatMetricsArtifacts {
        &self.artifacts
    }

    pub(crate) fn into_primary(self) -> crate::recipe_runner::materialize::PlannedMaterializedFile {
        match self.artifacts {
            Cow::Owned(pair) => pair.into_primary(),
            Cow::Borrowed(pair) => pair.primary().clone(),
        }
        .into_materialized()
    }

    pub(crate) fn targets(&self) -> Result<ChatMetricsTargets<'_>, PlanningError> {
        Ok(ChatMetricsTargets {
            primary: self.source.target(self.artifacts.primary())?,
            metrics: self.source.target(self.artifacts.metrics())?,
        })
    }
}

/// Distinct artifact types keep the destination roles distinguishable.
pub(crate) struct ChatMetricsTargets<'a> {
    pub(crate) primary: OutputTarget<'a, PrimaryArtifact>,
    pub(crate) metrics: OutputTarget<'a, MetricsArtifact>,
}

/// One write capability. Writers consume it and cannot select another path.
pub(crate) struct OutputTarget<'a, A> {
    artifact: &'a A,
    destinations: OutputDestinations,
}

impl<A> OutputTarget<'_, A> {
    pub(crate) fn artifact(&self) -> &A {
        self.artifact
    }

    pub(crate) fn primary_path(&self) -> &Path {
        match &self.destinations {
            OutputDestinations::Single(path) => path,
            OutputDestinations::Copies { requested, .. } => requested,
        }
    }

    pub(crate) fn staged_path(&self) -> &Path {
        match &self.destinations {
            OutputDestinations::Single(path) => path,
            OutputDestinations::Copies { staged, .. } => staged,
        }
    }

    pub(super) fn into_paths(self) -> impl Iterator<Item = PathBuf> {
        let (first, second) = match self.destinations {
            OutputDestinations::Single(path) => (path, None),
            OutputDestinations::Copies { requested, staged } => (requested, Some(staged)),
        };
        std::iter::once(first).chain(second)
    }
}

enum OutputDestinations {
    Single(PathBuf),
    Copies { requested: PathBuf, staged: PathBuf },
}
