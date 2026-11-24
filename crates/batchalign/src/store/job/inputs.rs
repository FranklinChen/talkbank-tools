//! Admitted inputs, recorded history and their database boundary.
use super::PendingJobFile;
use crate::api::DisplayPath;
use batchalign_types::paths::{AdmittedPath, ClientPath, ServerPath};
use std::collections::HashMap;

/// What a job input is: a CHAT transcript, or a recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    /// A CHAT transcript.
    Chat,
    /// A media recording.
    Media,
}

impl InputKind {
    /// Whether it is a transcript: the database's `has_chat` column.
    pub(crate) const fn is_chat(self) -> bool {
        matches!(self, Self::Chat)
    }

    /// The kind the database's `has_chat` column records.
    pub(crate) const fn from_is_chat(chat: bool) -> Self {
        if chat { Self::Chat } else { Self::Media }
    }

    /// The kind of a submitted source, by its name.
    #[must_use]
    pub fn of_source_name(name: &str) -> Self {
        if crate::types::request::is_chat_source_name(name) {
            Self::Chat
        } else {
            Self::Media
        }
    }
}

/// One paths-mode input: where it is read, where its output is written, the
/// earlier version it may be compared with, and the name the job shows. Every
/// path was admitted by the server's allowed roots.
#[derive(Debug, Clone)]
pub struct PathsInput {
    /// The name the job shows for it, unique within the job.
    pub display: DisplayPath,
    /// Transcript or recording.
    pub kind: InputKind,
    /// The source the job reads.
    pub source: AdmittedPath,
    /// Where its output is written.
    pub output: AdmittedPath,
    /// The earlier version, for incremental processing.
    pub before: Option<AdmittedPath>,
}

/// One content-mode input, staged under the job's staging directory.
#[derive(Debug, Clone)]
pub struct ContentInput {
    /// The name the job shows for it, and its staged name.
    pub display: DisplayPath,
    /// Transcript or recording.
    pub kind: InputKind,
}

/// A paths-mode job's files, in file order, whose names and paths cannot
/// collide: built only by [`PathsFiles::admit`], which compares the files
/// the admitted paths name ([`batchalign_types::paths::FileIdentity`]), not
/// their spellings: a symbolic link, a hard link, or a case or Unicode
/// difference the volume ignores makes one file, and a case difference on a
/// case-sensitive volume makes two.
#[derive(Debug, Clone)]
pub struct PathsFiles(Vec<PathsInput>);

/// Why a paths-mode job's files cannot run together.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PathsOverlap {
    /// The job lists no file.
    #[error("a paths-mode job needs at least one file")]
    Empty,
    /// Two files would be shown, and tracked, under one name.
    #[error("two files would both be shown as {0}; give them distinct display names")]
    Display(DisplayPath),
    /// Two files would write one output.
    #[error("{first} and {second} would both write {output}; each needs its own output")]
    Output {
        /// The first file.
        first: DisplayPath,
        /// The second file.
        second: DisplayPath,
        /// The output they share, as the second named it.
        output: String,
    },
    /// One file's output is another file's source or earlier version, so
    /// one would be overwritten while the other reads it.
    #[error("{writer}'s output {output} is an input of {reader}")]
    OutputIsInput {
        /// The file whose output it is.
        writer: DisplayPath,
        /// The file that reads it.
        reader: DisplayPath,
        /// The path, as the writer named it.
        output: String,
    },
}

impl PathsFiles {
    /// The files, checked as a set. A file may write over its own source
    /// (in-place processing); never over another file's source or prior.
    pub(crate) fn admit(files: Vec<PathsInput>) -> Result<Self, PathsOverlap> {
        if files.is_empty() {
            return Err(PathsOverlap::Empty);
        }
        // The file a path names, as the disk tells files apart.
        let identity = |path: &AdmittedPath| path.identity().clone();
        let mut displays = std::collections::HashSet::with_capacity(files.len());
        let mut outputs: HashMap<batchalign_types::paths::FileIdentity, usize> =
            HashMap::with_capacity(files.len());
        // Every file that reads each identity: several files may read one
        // file (a shared prior), and every one of them must be checked.
        let mut readers: HashMap<batchalign_types::paths::FileIdentity, Vec<usize>> =
            HashMap::with_capacity(files.len());
        for (index, file) in files.iter().enumerate() {
            if !displays.insert(&file.display) {
                return Err(PathsOverlap::Display(file.display.clone()));
            }
            if let Some(&first) = outputs.get(&identity(&file.output)) {
                return Err(PathsOverlap::Output {
                    first: files[first].display.clone(),
                    second: file.display.clone(),
                    output: file.output.client().to_string(),
                });
            }
            outputs.insert(identity(&file.output), index);
            for input in std::iter::once(&file.source).chain(&file.before) {
                readers.entry(identity(input)).or_default().push(index);
            }
        }
        for (key, &writer) in &outputs {
            if let Some(&reader) = readers
                .get(key)
                .and_then(|readers| readers.iter().find(|&&reader| reader != writer))
            {
                return Err(PathsOverlap::OutputIsInput {
                    writer: files[writer].display.clone(),
                    reader: files[reader].display.clone(),
                    output: files[writer].output.client().to_string(),
                });
            }
        }
        Ok(Self(files))
    }
}

impl std::ops::Deref for PathsFiles {
    type Target = [PathsInput];
    fn deref(&self) -> &[PathsInput] {
        &self.0
    }
}

/// Why stored or submitted inputs cannot be admitted to run.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InputsRefusal {
    /// A path is not admitted by the authority.
    #[error(transparent)]
    Path(#[from] batchalign_types::paths::PathRefusal),
    /// The files collide.
    #[error(transparent)]
    Overlap(#[from] PathsOverlap),
}

/// A job's inputs, one entry per file, in file order: read and written in
/// place on the server's disk (paths mode), or staged from content the client
/// sent. One value per file, so a file's name, kind and paths cannot disagree
/// in length or position, and paths-mode paths exist only as admitted ones.
#[derive(Debug, Clone)]
pub enum JobInputs {
    /// Paths mode: every path admitted by the server's allowed roots.
    Paths {
        /// The files, in file order, checked as a set.
        files: PathsFiles,
        /// The folder the client submitted, admitted, when it sent one: the
        /// media search looks there.
        source_dir: Option<AdmittedPath>,
    },
    /// Content mode: the files the client sent.
    Content(Vec<ContentInput>),
}

impl JobInputs {
    /// How many files the job has.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Paths { files, .. } => files.len(),
            Self::Content(files) => files.len(),
        }
    }

    /// Whether the job has no files.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the selected mode's file list through two concrete slices.
    /// This lets projection iterate the selected inputs with static dispatch.
    pub(super) fn split(&self) -> (&[PathsInput], &[ContentInput]) {
        match self {
            Self::Paths { files, .. } => (&files[..], &[]),
            Self::Content(files) => (&[], files),
        }
    }

    /// The files whose display name `keep` accepts, as runner work, in file
    /// order: the one constructor of [`PendingJobFile`], so a pending file's
    /// name, kind and location are always its own input's. The name is
    /// tested before anything is cloned, so files already done cost nothing.
    pub(crate) fn entries<'a>(
        &'a self,
        keep: impl Fn(&DisplayPath) -> bool + Copy + 'a,
    ) -> impl Iterator<Item = PendingJobFile> + 'a {
        PendingJobFile::entries(self, keep)
    }

    /// Every display name, in file order.
    pub fn displays(&self) -> impl Iterator<Item = &DisplayPath> {
        let (paths, content) = self.split();
        paths
            .iter()
            .map(|file| &file.display)
            .chain(content.iter().map(|file| &file.display))
    }

    /// Where a job-level output named `name` (one file for the whole job,
    /// such as a consolidated `compare.csv`) is written: beside the first
    /// file's admitted output in paths mode, else in the staged output.
    #[must_use]
    pub fn job_level_output(&self, staging_dir: &ServerPath, name: &str) -> std::path::PathBuf {
        match self {
            Self::Paths { files, .. } => match &files[..] {
                [first, ..] => first.output.server().as_path().with_file_name(name),
                [] => staging_dir.join("output").join(name).as_path().to_owned(),
            },
            Self::Content(_) => staging_dir.join("output").join(name).as_path().to_owned(),
        }
    }
}

/// One paths-mode input as the database recorded it: the client's paths, not
/// admitted by this server process. History only: it becomes runnable input
/// solely through [`RecordedInputs::admit`].
#[derive(Debug, Clone)]
pub struct RecordedPathsInput {
    /// The name the job shows for it.
    pub display: DisplayPath,
    /// Transcript or recording.
    pub kind: InputKind,
    /// The source the client named.
    pub source: ClientPath,
    /// The output the client named.
    pub output: ClientPath,
    /// The earlier version the client named, for incremental processing.
    pub before: Option<ClientPath>,
}

/// A paths-mode job's inputs as stored, read back without touching the
/// filesystem. A finished job is shown from these; a job that runs again
/// (recovered mid-run, or restarted) is admitted first.
#[derive(Debug, Clone)]
pub struct RecordedInputs {
    files: Vec<RecordedPathsInput>,
    source_dir: Option<ClientPath>,
}

impl RecordedInputs {
    /// The files, in file order.
    #[must_use]
    pub fn files(&self) -> &[RecordedPathsInput] {
        &self.files
    }

    /// Admit every recorded path under `authority`: the one route from a
    /// stored paths-mode job to one that can run, its priors included.
    pub(crate) fn admit(
        &self,
        authority: &batchalign_types::paths::AllowedRoots,
    ) -> Result<JobInputs, InputsRefusal> {
        let files = self
            .files
            .iter()
            .map(|file| {
                Ok(PathsInput {
                    display: file.display.clone(),
                    kind: file.kind,
                    source: authority.admit(&file.source)?,
                    output: authority.admit(&file.output)?,
                    before: file
                        .before
                        .as_ref()
                        .map(|before| authority.admit(before))
                        .transpose()?,
                })
            })
            .collect::<Result<Vec<_>, batchalign_types::paths::PathRefusal>>()?;
        Ok(JobInputs::Paths {
            files: PathsFiles::admit(files)?,
            source_dir: self
                .source_dir
                .as_ref()
                .map(|dir| authority.admit_folder(dir))
                .transpose()?,
        })
    }
}

/// A job's inputs as this server process holds them: admitted, so the job
/// can run, or recorded, read back from the database for a job that is only
/// history here. Only [`StoredInputs::Admitted`] reaches a runner.
#[derive(Debug, Clone)]
pub enum StoredInputs {
    /// Admitted by this process: content-mode inputs, or paths admitted by
    /// the authority the job was submitted, recovered or restarted under.
    Admitted(JobInputs),
    /// Paths-mode inputs read back from the database, not admitted.
    Recorded(RecordedInputs),
}

impl StoredInputs {
    /// The inputs borrowed as stored.
    pub(crate) const fn as_ref(&self) -> StoredInputsRef<'_> {
        match self {
            Self::Admitted(inputs) => StoredInputsRef::Admitted(inputs),
            Self::Recorded(recorded) => StoredInputsRef::Recorded(recorded),
        }
    }
}

/// Infallible and invariant-preserving: inputs that were admitted stay so.
impl From<JobInputs> for StoredInputs {
    fn from(inputs: JobInputs) -> Self {
        Self::Admitted(inputs)
    }
}

impl<'a> StoredInputsRef<'a> {
    /// How many files the job has.
    pub(crate) fn len(self) -> usize {
        match self {
            Self::Admitted(inputs) => inputs.len(),
            Self::Recorded(recorded) => recorded.files.len(),
        }
    }

    /// The inputs a runner may use, when this process admitted them.
    pub(crate) const fn admitted(self) -> Option<&'a JobInputs> {
        match self {
            Self::Admitted(inputs) => Some(inputs),
            Self::Recorded(_) => None,
        }
    }

    /// Every display name, in file order.
    pub(crate) fn displays(self) -> Box<dyn Iterator<Item = &'a DisplayPath> + 'a> {
        match self {
            Self::Admitted(inputs) => Box::new(inputs.displays()),
            Self::Recorded(recorded) => Box::new(recorded.files.iter().map(|file| &file.display)),
        }
    }

    /// The inputs admitted afresh under `authority`, as a restart needs:
    /// every run is admitted under the rules in force when it starts, so a
    /// folder removed from the configuration since submission is honoured.
    /// Content-mode inputs name no server path and pass unchanged.
    pub(crate) fn readmit(
        self,
        authority: &batchalign_types::paths::AllowedRoots,
    ) -> Result<JobInputs, InputsRefusal> {
        match self {
            Self::Admitted(JobInputs::Content(files)) => Ok(JobInputs::Content(files.clone())),
            Self::Admitted(JobInputs::Paths { files, source_dir }) => RecordedInputs {
                files: files
                    .iter()
                    .map(|file| RecordedPathsInput {
                        display: file.display.clone(),
                        kind: file.kind,
                        source: file.source.client().clone(),
                        output: file.output.client().clone(),
                        before: file.before.as_ref().map(|before| before.client().clone()),
                    })
                    .collect(),
                source_dir: source_dir.as_ref().map(|dir| dir.client().clone()),
            }
            .admit(authority),
            Self::Recorded(recorded) => recorded.admit(authority),
        }
    }

    /// The columns the database stores.
    pub(crate) fn to_columns(self) -> InputColumns {
        match self {
            Self::Admitted(inputs) => inputs.to_columns(),
            Self::Recorded(recorded) => InputColumns {
                filenames: recorded
                    .files
                    .iter()
                    .map(|file| file.display.to_string())
                    .collect(),
                has_chat: recorded
                    .files
                    .iter()
                    .map(|file| file.kind.is_chat())
                    .collect(),
                paths_mode: true,
                source_paths: recorded
                    .files
                    .iter()
                    .map(|file| file.source.as_str().to_owned())
                    .collect(),
                output_paths: recorded
                    .files
                    .iter()
                    .map(|file| file.output.as_str().to_owned())
                    .collect(),
                before_paths: before_column(recorded.files.iter().map(|file| file.before.as_ref())),
            },
        }
    }
}

/// A job's inputs borrowed as stored: admitted or recorded.
#[derive(Debug, Clone, Copy)]
pub(crate) enum StoredInputsRef<'a> {
    /// Admitted by this process.
    Admitted(&'a JobInputs),
    /// Recorded history, not admitted.
    Recorded(&'a RecordedInputs),
}

/// A job's inputs as the database stores them: parallel columns. The storage
/// format only; [`StoredInputs::to_columns`] and [`InputColumns::parse`] are
/// the one serializer and the one parser, and nothing else reads them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InputColumns {
    pub(crate) filenames: Vec<String>,
    pub(crate) has_chat: Vec<bool>,
    pub(crate) paths_mode: bool,
    pub(crate) source_paths: Vec<String>,
    pub(crate) output_paths: Vec<String>,
    /// One entry per file, or empty when no file has a prior.
    pub(crate) before_paths: Vec<Option<String>>,
}

/// The `before_paths` column: empty when no file has a prior (as every
/// content-mode job), else one entry per file.
fn before_column<'a>(befores: impl Iterator<Item = Option<&'a ClientPath>>) -> Vec<Option<String>> {
    let befores: Vec<Option<String>> = befores
        .map(|before| before.map(|path| path.as_str().to_owned()))
        .collect();
    if befores.iter().any(Option::is_some) {
        befores
    } else {
        Vec::new()
    }
}

/// Stored input columns that disagree in length, so no job can be read from
/// them.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "stored input columns disagree in length (names {names}, kinds {kinds}, sources {sources}, outputs {outputs}, priors {befores})"
)]
pub(crate) struct InputColumnsRefusal {
    names: usize,
    kinds: usize,
    sources: usize,
    outputs: usize,
    befores: usize,
}

impl InputColumns {
    /// Read stored columns back, with no filesystem access: content-mode
    /// inputs are admitted as they are (they name no server path), and
    /// paths-mode inputs come back recorded, for the caller to admit only
    /// when the job is to run.
    pub(crate) fn parse(
        self,
        source_dir: Option<ClientPath>,
    ) -> Result<StoredInputs, InputColumnsRefusal> {
        let Self {
            filenames,
            has_chat,
            paths_mode,
            source_paths,
            output_paths,
            before_paths,
        } = self;
        let lengths_agree = has_chat.len() == filenames.len()
            && if paths_mode {
                source_paths.len() == filenames.len()
                    && output_paths.len() == filenames.len()
                    && (before_paths.is_empty() || before_paths.len() == filenames.len())
            } else {
                // A content-mode job names no server path: a row that does
                // is refused, never read with its paths silently dropped.
                source_paths.is_empty() && output_paths.is_empty() && before_paths.is_empty()
            };
        if !lengths_agree {
            return Err(InputColumnsRefusal {
                names: filenames.len(),
                kinds: has_chat.len(),
                sources: source_paths.len(),
                outputs: output_paths.len(),
                befores: before_paths.len(),
            });
        }
        let named = filenames
            .into_iter()
            .zip(has_chat)
            .map(|(name, chat)| (DisplayPath::from(name), InputKind::from_is_chat(chat)));
        Ok(if paths_mode {
            StoredInputs::Recorded(RecordedInputs {
                files: named
                    .zip(source_paths.into_iter().zip(output_paths))
                    // An empty column is no file with a prior.
                    .zip(
                        before_paths
                            .into_iter()
                            .map(Some)
                            .chain(std::iter::repeat(None)),
                    )
                    .map(
                        |(((display, kind), (source, output)), before)| RecordedPathsInput {
                            display,
                            kind,
                            source: ClientPath::from(source),
                            output: ClientPath::from(output),
                            before: before.flatten().map(ClientPath::from),
                        },
                    )
                    .collect(),
                source_dir,
            })
        } else {
            StoredInputs::Admitted(JobInputs::Content(
                named
                    .map(|(display, kind)| ContentInput { display, kind })
                    .collect(),
            ))
        })
    }
}

impl JobInputs {
    /// The columns the database stores.
    pub(crate) fn to_columns(&self) -> InputColumns {
        match self {
            Self::Paths { files, .. } => InputColumns {
                filenames: files.iter().map(|file| file.display.to_string()).collect(),
                has_chat: files.iter().map(|file| file.kind.is_chat()).collect(),
                paths_mode: true,
                source_paths: files
                    .iter()
                    .map(|file| file.source.client().as_str().to_owned())
                    .collect(),
                output_paths: files
                    .iter()
                    .map(|file| file.output.client().as_str().to_owned())
                    .collect(),
                before_paths: before_column(
                    files
                        .iter()
                        .map(|file| file.before.as_ref().map(AdmittedPath::client)),
                ),
            },
            Self::Content(files) => InputColumns {
                filenames: files.iter().map(|file| file.display.to_string()).collect(),
                has_chat: files.iter().map(|file| file.kind.is_chat()).collect(),
                paths_mode: false,
                source_paths: Vec::new(),
                output_paths: Vec::new(),
                before_paths: Vec::new(),
            },
        }
    }
}
