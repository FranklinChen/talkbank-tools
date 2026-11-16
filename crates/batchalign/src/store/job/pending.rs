//! Source-bound pending files and their media-preflight classification.
use super::{InputKind, JobInputs};
use crate::api::DisplayPath;
use batchalign_types::paths::{AdmittedPath, ServerPath};

/// One file that still requires runner work.
///
/// Made only by [`super::JobInputs::entries`]: the fields are private, so a pending
/// file's name, kind and location are always those of its own input, and no
/// caller (test fixtures included) can pair a file with another's paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingJobFile {
    /// Index into the job's inputs.
    file_index: usize,
    /// Logical filename of the work item.
    filename: DisplayPath,
    /// Transcript or recording.
    kind: InputKind,
    /// Where it is read and written.
    location: InputLocation,
}

/// Where a pending file is read and written: admitted paths on the server's
/// disk, or its staged copy. Taken from the file's own input entry when the
/// pending list is made, so no index lookup can miss.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputLocation {
    /// Paths mode: read, written and compared in place. Boxed: three
    /// admitted paths beside a unit variant.
    Paths(Box<PathsLocation>),
    /// Content mode: staged under the job's staging directory.
    Staged,
}

/// A paths-mode file's admitted paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathsLocation {
    /// The source the job reads.
    pub source: AdmittedPath,
    /// Where its output is written.
    pub output: AdmittedPath,
    /// The earlier version, for incremental processing.
    pub before: Option<AdmittedPath>,
}

impl PendingJobFile {
    /// Check the indexed declaration without rebuilding pending entries.
    pub(crate) fn belongs_to(&self, inputs: &JobInputs) -> bool {
        let (paths, content) = inputs.split();
        match &self.location {
            InputLocation::Staged => content
                .get(self.file_index)
                .is_some_and(|input| input.display == self.filename && input.kind == self.kind),
            InputLocation::Paths(location) => paths.get(self.file_index).is_some_and(|input| {
                input.display == self.filename
                    && input.kind == self.kind
                    && input.source == location.source
                    && input.output == location.output
                    && input.before == location.before
            }),
        }
    }

    /// Project the admitted input set in its own order. The constructors and
    /// input ordinals stay here, so callers cannot supply a different index.
    pub(super) fn entries<'a>(
        inputs: &'a JobInputs,
        keep: impl Fn(&DisplayPath) -> bool + Copy + 'a,
    ) -> impl Iterator<Item = PendingJobFile> + 'a {
        let (paths, content) = inputs.split();
        let paths = paths
            .iter()
            .enumerate()
            .filter(move |(_, file)| keep(&file.display))
            .map(|(file_index, file)| PendingJobFile {
                file_index,
                filename: file.display.clone(),
                kind: file.kind,
                location: InputLocation::Paths(Box::new(PathsLocation {
                    source: file.source.clone(),
                    output: file.output.clone(),
                    before: file.before.clone(),
                })),
            });
        let content = content
            .iter()
            .enumerate()
            .filter(move |(_, file)| keep(&file.display))
            .map(|(file_index, file)| PendingJobFile {
                file_index,
                filename: file.display.clone(),
                kind: file.kind,
                location: InputLocation::Staged,
            });
        paths.chain(content)
    }

    /// Consume the pending input into its exhaustive preflight route.
    pub(crate) fn into_preflight(self) -> PreflightInput {
        let Self {
            file_index,
            filename,
            kind,
            location,
        } = self;
        match (kind, location) {
            (InputKind::Media, InputLocation::Paths(paths)) => {
                PreflightInput::Media(InPlaceMedia {
                    file_index,
                    filename,
                    paths,
                })
            }
            (InputKind::Chat, location) => PreflightInput::Ready(Self {
                file_index,
                filename,
                kind: InputKind::Chat,
                location,
            }),
            (InputKind::Media, InputLocation::Staged) => PreflightInput::Ready(Self {
                file_index,
                filename,
                kind: InputKind::Media,
                location: InputLocation::Staged,
            }),
        }
    }

    /// Its index into the job's inputs, and into the job's file order.
    pub fn file_index(&self) -> usize {
        self.file_index
    }

    /// Its display name.
    pub fn filename(&self) -> &DisplayPath {
        &self.filename
    }

    /// Transcript or recording.
    pub fn kind(&self) -> InputKind {
        self.kind
    }

    /// Where it is read and written.
    pub fn location(&self) -> &InputLocation {
        &self.location
    }

    /// Where the input is read: its admitted source, or its staged copy
    /// under `input/`.
    pub(crate) fn input_path(&self, staging_dir: &ServerPath) -> ServerPath {
        match &self.location {
            InputLocation::Paths(paths) => paths.source.server().clone(),
            InputLocation::Staged => staging_dir.join("input").join(self.filename.as_ref()),
        }
    }

    /// Its earlier version, in paths mode, when one was given. Never a
    /// neighbouring file's.
    pub(crate) fn before_path(&self) -> Option<ServerPath> {
        match &self.location {
            InputLocation::Paths(paths) => paths.before.as_ref().map(|path| path.server().clone()),
            InputLocation::Staged => None,
        }
    }
}

/// Only recordings read in place carry the capability to inspect media metadata.
pub(crate) enum PreflightInput {
    Ready(PendingJobFile),
    Media(InPlaceMedia),
}

/// A pending recording with its own admitted paths. Fields and constructors
/// are private, so a staged input or CHAT file cannot enter the media observer.
pub(crate) struct InPlaceMedia {
    file_index: usize,
    filename: DisplayPath,
    paths: Box<PathsLocation>,
}

impl InPlaceMedia {
    pub(crate) fn filename(&self) -> &DisplayPath {
        &self.filename
    }
    pub(crate) fn source(&self) -> &ServerPath {
        self.paths.source.server()
    }
    pub(crate) fn into_pending(self) -> PendingJobFile {
        PendingJobFile {
            file_index: self.file_index,
            filename: self.filename,
            kind: InputKind::Media,
            location: InputLocation::Paths(self.paths),
        }
    }
}
