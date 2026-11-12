//! Job struct, associated types, and conflict detection.

use std::collections::{BTreeMap, HashMap};

use batchalign_types::paths::{
    AdmittedPath, ClientPath, MediaMappingKey, RepoRelativePath, ServerPath,
};

use crate::api::{
    ActiveStatus, ContentType, CorrelationId, DisplayPath, FileProgressStage, FinishedStatus,
    JobId, JobStatus, LanguageSpec, MachineTime, NumSpeakers, ReleasedCommand, StatusKind,
};
use crate::options::CommandOptions;
use crate::types::execution_plan::ExecutionPlan;
use tokio_util::sync::CancellationToken;

use crate::store::{FileResultEntry, FileStatus};

// ---------------------------------------------------------------------------
// Job
// ---------------------------------------------------------------------------

/// All metadata for a single processing job.
///
/// A job progresses through the state machine:
/// `Queued -> Running -> Completed | Failed | Cancelled`.
/// `Interrupted` is a transient state assigned during crash recovery when the
/// server exited while the job was `Running`.  On reload, interrupted jobs are
/// either re-queued (if resumable files remain) or promoted to a terminal state.
///
/// Jobs are created by `JobStore::submit()`, executed by `runner::run_job()`,
/// and persisted to SQLite for crash recovery.
pub struct Job {
    /// Stable identifiers for the job and its correlation context.
    pub identity: JobIdentity,
    /// Immutable dispatch-time command and option configuration.
    pub dispatch: JobDispatchConfig,
    /// Submitter-facing provenance for conflict detection and display.
    pub source: JobSourceContext,
    /// File lists, staging paths, and media-resolution configuration.
    pub filesystem: JobFilesystemConfig,
    /// Mutable execution state updated as files progress.
    pub execution: JobExecutionState,
    /// Scheduling, completion, and lease state for queue coordination.
    pub schedule: JobScheduleState,
    /// In-memory cancellation and runner-claim state.
    pub runtime: JobRuntimeControl,
    /// Optional execution plan describing where and how this job is processed.
    /// Present for staged-remote jobs; `None` for local and direct-mode jobs.
    pub execution_plan: Option<ExecutionPlan>,
}

/// Stable identifiers for one job.
#[derive(Debug, Clone)]
pub struct JobIdentity {
    /// UUID v4 uniquely identifying this job. Immutable after creation.
    pub job_id: JobId,
    /// Client-supplied correlation ID for tracing across services.
    pub correlation_id: CorrelationId,
}

/// Immutable dispatch-time configuration for one job.
#[derive(Debug, Clone)]
pub struct JobDispatchConfig {
    /// The batchalign command to run.
    pub command: ReleasedCommand,
    /// Language specification: may be `Auto` (for ASR auto-detection) or
    /// a resolved ISO 639-3 code.
    pub lang: LanguageSpec,
    /// Expected number of speakers in the audio workflow.
    pub num_speakers: NumSpeakers,
    /// Typed command options captured at submission time.
    pub options: CommandOptions,
    /// Server-internal runtime state for orchestration helpers.
    pub runtime_state: BTreeMap<String, serde_json::Value>,
    /// Whether detailed algorithm traces should be collected.
    pub debug_traces: bool,
}

/// Submitter-facing provenance for one job.
#[derive(Debug, Clone)]
pub struct JobSourceContext {
    /// Who submitted the job, or `None` when no submitter was recorded (a
    /// recovered row whose columns are empty). The empty string used to
    /// stand for that absence in two `String` fields, and only the API
    /// projection turned it back into an `Option`.
    pub submitter: Option<Submitter>,
    /// Client-visible source directory used for display and locality hints.
    pub source_dir: ClientPath,
}

/// Who submitted a job: the client's address as the server saw it, and the
/// name that address resolved to, when one was found.
///
/// Neither text is ever empty. The fields are private, so the constructors
/// below are the only routes in: [`Self::client`] where an HTTP submission is
/// born, [`Self::direct_cli`] for the in-process CLI, and
/// [`Self::from_columns`] at the database boundary, which is the one place
/// the empty string still means "absent" (the columns are
/// `TEXT NOT NULL DEFAULT ''`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Submitter {
    /// The client's address. Conflict detection keys on it.
    address: String,
    /// The name the address resolved to, for display.
    name: Option<String>,
}

impl Submitter {
    /// The address the in-process direct CLI is recorded under.
    const DIRECT_CLI_ADDRESS: std::net::Ipv4Addr = std::net::Ipv4Addr::LOCALHOST;
    /// The name the in-process direct CLI is recorded under.
    const DIRECT_CLI_NAME: &'static str = "direct-cli";
    /// How the database columns spell an absent submitter or name.
    const ABSENT_COLUMN: &'static str = "";

    /// An HTTP client, by its peer address and the name that address resolved
    /// to. A resolution that named nothing (a Tailscale peer reporting an
    /// empty host name) records no name rather than an empty one.
    pub fn client(address: std::net::IpAddr, resolved_name: String) -> Self {
        Self {
            address: address.to_string(),
            name: (!resolved_name.is_empty()).then_some(resolved_name),
        }
    }

    /// The in-process direct CLI, which has no network peer. It is recorded
    /// as a loopback client named `direct-cli`, the encoding it has always
    /// had, so its jobs still conflict with each other and display the same.
    pub fn direct_cli() -> Self {
        Self {
            address: Self::DIRECT_CLI_ADDRESS.to_string(),
            name: Some(Self::DIRECT_CLI_NAME.to_owned()),
        }
    }

    /// The client's address.
    pub fn address(&self) -> &str {
        &self.address
    }

    /// The name the address resolved to, when one was recorded.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Read the two database columns, where `''` encodes absence.
    ///
    /// An empty address is no submitter. A name with no address is not
    /// something this code writes ([`Self::columns`] cannot produce it), so it
    /// is not a submitter either; it is its own case, so the reader must say
    /// what it does with the name rather than lose it without a word.
    pub(crate) fn from_columns(address: String, name: String) -> StoredSubmitter {
        match (address.is_empty(), name.is_empty()) {
            (true, true) => StoredSubmitter::Absent,
            (true, false) => StoredSubmitter::NameWithoutAddress { name },
            (false, _) => StoredSubmitter::Recorded(Self {
                address,
                name: (!name.is_empty()).then_some(name),
            }),
        }
    }

    /// The two database columns `(submitted_by, submitted_by_name)` for an
    /// optional submitter, spelling absence as the columns' `''`.
    pub(crate) fn columns(submitter: Option<&Self>) -> (&str, &str) {
        match submitter {
            Some(submitter) => (
                submitter.address.as_str(),
                match &submitter.name {
                    Some(name) => name.as_str(),
                    None => Self::ABSENT_COLUMN,
                },
            ),
            None => (Self::ABSENT_COLUMN, Self::ABSENT_COLUMN),
        }
    }
}

/// What a job row's two submitter columns hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StoredSubmitter {
    /// An address, and the name it resolved to when there was one.
    Recorded(Submitter),
    /// Both columns empty: no submitter was recorded.
    Absent,
    /// A name with an empty address. Not written by this build; a row from
    /// another build or a hand edit. Not a submitter (conflict detection
    /// keys on the address), and the reader reports the name it drops.
    NameWithoutAddress {
        /// The name the row held.
        name: String,
    },
}

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
    pub(crate) fn is_chat(self) -> bool {
        matches!(self, Self::Chat)
    }

    /// The kind the database's `has_chat` column records.
    pub(crate) fn from_is_chat(chat: bool) -> Self {
        match chat {
            true => Self::Chat,
            false => Self::Media,
        }
    }

    /// The kind of a submitted source, by its name.
    pub fn of_source_name(name: &str) -> Self {
        match crate::types::request::is_chat_source_name(name) {
            true => Self::Chat,
            false => Self::Media,
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
        use std::collections::HashMap;
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
    pub fn len(&self) -> usize {
        match self {
            Self::Paths { files, .. } => files.len(),
            Self::Content(files) => files.len(),
        }
    }

    /// Whether the job has no files.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The two modes' file lists, exactly one of them non-empty: lets the
    /// iterators below borrow either without boxing or collecting.
    fn split(&self) -> (&[PathsInput], &[ContentInput]) {
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
        let (paths, content) = self.split();
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
    pub(crate) fn as_ref(&self) -> StoredInputsRef<'_> {
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
    pub(crate) fn admitted(self) -> Option<&'a JobInputs> {
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

/// Where a job is in its life, with the inputs that state needs: an active
/// job (queued or running) holds inputs this process admitted, so it can
/// run; a finished one holds its inputs as stored, admitted or recorded.
/// "Active with unadmitted inputs" is not a value this type has, so a job
/// that cannot run can never be queued.
///
/// The transitions are one method per event, each legal only from the
/// states that event leaves, so the state machine is these signatures and
/// their matches rather than a table beside them. Leaving finished for
/// active ([`Self::restart`], [`Self::resume`]) admits the inputs first,
/// under the rules in force now. A refused event changes nothing and says
/// which status refused it.
#[derive(Debug, Clone)]
pub enum JobLifecycle {
    /// Queued or running, on admitted inputs.
    Active {
        /// Which active status.
        status: ActiveStatus,
        /// Inputs this process admitted.
        inputs: JobInputs,
    },
    /// Completed, failed, cancelled, interrupted or failed at writeback.
    Finished {
        /// Which finished status.
        status: FinishedStatus,
        /// Its inputs as stored.
        inputs: StoredInputs,
    },
}

/// An event a job's lifecycle refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JobEvent {
    /// A runner picked a queued job up.
    Dispatch,
    /// The memory gate parked an active job.
    MemoryRequeue,
    /// A user cancel or a shutdown stopped an active job.
    Stop,
    /// An active job ended under its own power.
    Finish,
    /// An operator restarted a cancelled or failed job.
    Restart,
    /// Startup recovery resumed an interrupted job.
    Resume,
    /// Startup recovery settled an interrupted job whose files all ended.
    Settle,
}

impl std::fmt::Display for JobEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Dispatch => "be dispatched",
            Self::MemoryRequeue => "be requeued by the memory gate",
            Self::Stop => "be stopped",
            Self::Finish => "finish",
            Self::Restart => "be restarted",
            Self::Resume => "be resumed",
            Self::Settle => "be settled by recovery",
        })
    }
}

/// A lifecycle event refused in the job's current status. Most refusals are
/// ordinary races (a user cancels while a runner is mid-flight).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("a {from} job cannot {event}")]
pub(crate) struct Refused {
    /// The status the job was in, and stays in.
    pub(crate) from: JobStatus,
    /// What was refused.
    pub(crate) event: JobEvent,
}

/// Why a finished job did not become active again.
#[derive(Debug)]
pub(crate) enum Reactivation<E> {
    /// Its status does not allow the event; nothing was admitted.
    Refused(Refused),
    /// Its inputs were refused under the rules in force now.
    Admission(E),
}

/// How an active job finished under its own power.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// Every file ended, none failed.
    Completed,
    /// It failed, or a file did.
    Failed,
    /// Its cancellation took effect as it finished.
    Cancelled,
    /// Its outputs could not be written back.
    WritebackFailed,
}

impl From<Outcome> for FinishedStatus {
    fn from(outcome: Outcome) -> Self {
        match outcome {
            Outcome::Completed => Self::Completed,
            Outcome::Failed => Self::Failed,
            Outcome::Cancelled => Self::Cancelled,
            Outcome::WritebackFailed => Self::WritebackFailed,
        }
    }
}

/// What startup recovery concluded for an interrupted job whose files had
/// all ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Settlement {
    /// None failed.
    Completed,
    /// One failed, or its inputs were refused.
    Failed,
}

impl JobLifecycle {
    /// A job read back from the database. Nothing runs it in this process,
    /// so a stored active status means the server stopped while it ran: it
    /// is interrupted (startup marks such rows so first).
    pub(crate) fn recovered(status: JobStatus, inputs: StoredInputs) -> Self {
        let status = match status.kind() {
            StatusKind::Active(_) => FinishedStatus::Interrupted,
            StatusKind::Finished(status) => status,
        };
        Self::Finished { status, inputs }
    }

    /// The status.
    pub fn status(&self) -> JobStatus {
        match self {
            Self::Active { status, .. } => (*status).into(),
            Self::Finished { status, .. } => (*status).into(),
        }
    }

    /// The admitted inputs a runner may use: an active job's, or a
    /// finished one's that this process admitted (a cancelled run closing
    /// its files). `None` for recorded history.
    pub(crate) fn admitted(&self) -> Option<&JobInputs> {
        self.stored().admitted()
    }

    /// The inputs as stored, borrowed whichever state the job is in.
    pub(crate) fn stored(&self) -> StoredInputsRef<'_> {
        match self {
            Self::Active { inputs, .. } => StoredInputsRef::Admitted(inputs),
            Self::Finished { inputs, .. } => inputs.as_ref(),
        }
    }

    /// A refusal of `event` in the current status.
    fn refused(&self, event: JobEvent) -> Refused {
        Refused {
            from: self.status(),
            event,
        }
    }

    /// Queued to running.
    pub(crate) fn dispatch(&mut self) -> Result<(), Refused> {
        match self {
            Self::Active {
                status: status @ ActiveStatus::Queued,
                ..
            } => {
                *status = ActiveStatus::Running;
                Ok(())
            }
            Self::Active {
                status: ActiveStatus::Running,
                ..
            }
            | Self::Finished { .. } => Err(self.refused(JobEvent::Dispatch)),
        }
    }

    /// Active to queued, for the memory gate. A job that has left the
    /// active set (cancelled while waiting) stays gone.
    pub(crate) fn memory_requeue(&mut self) -> Result<(), Refused> {
        match self {
            Self::Active { status, .. } => {
                *status = ActiveStatus::Queued;
                Ok(())
            }
            Self::Finished { .. } => Err(self.refused(JobEvent::MemoryRequeue)),
        }
    }

    /// Active to cancelled or interrupted.
    pub(crate) fn stop(&mut self, stop: super::Stop) -> Result<(), Refused> {
        self.finish_active(stop.status(), JobEvent::Stop)
    }

    /// Active to its own `outcome`.
    pub(crate) fn finish(&mut self, outcome: Outcome) -> Result<(), Refused> {
        self.finish_active(outcome.into(), JobEvent::Finish)
    }

    /// Active to `next`, its admitted inputs becoming its stored ones.
    fn finish_active(&mut self, next: FinishedStatus, event: JobEvent) -> Result<(), Refused> {
        match self {
            Self::Active { inputs, .. } => {
                // Moved out from behind `&mut`; `Content(Vec::new())`
                // allocates nothing and is dropped as `self` is replaced on
                // the next line.
                let inputs = std::mem::replace(inputs, JobInputs::Content(Vec::new()));
                *self = Self::Finished {
                    status: next,
                    inputs: StoredInputs::Admitted(inputs),
                };
                Ok(())
            }
            Self::Finished { .. } => Err(self.refused(event)),
        }
    }

    /// Cancelled, failed or failed at writeback, back to queued on inputs
    /// `admit` admits now.
    pub(crate) fn restart<E>(
        &mut self,
        admit: impl FnOnce(StoredInputsRef<'_>) -> Result<JobInputs, E>,
    ) -> Result<(), Reactivation<E>> {
        let admitted = match self {
            Self::Finished {
                status:
                    FinishedStatus::Cancelled | FinishedStatus::Failed | FinishedStatus::WritebackFailed,
                inputs,
            } => admit(inputs.as_ref()),
            Self::Finished {
                status: FinishedStatus::Completed | FinishedStatus::Interrupted,
                ..
            }
            | Self::Active { .. } => {
                return Err(Reactivation::Refused(self.refused(JobEvent::Restart)));
            }
        };
        self.requeue_on(admitted)
    }

    /// Interrupted, back to queued on inputs `admit` admits now.
    pub(crate) fn resume<E>(
        &mut self,
        admit: impl FnOnce(StoredInputsRef<'_>) -> Result<JobInputs, E>,
    ) -> Result<(), Reactivation<E>> {
        let admitted = match self {
            Self::Finished {
                status: FinishedStatus::Interrupted,
                inputs,
            } => admit(inputs.as_ref()),
            Self::Finished {
                status:
                    FinishedStatus::Completed
                    | FinishedStatus::Failed
                    | FinishedStatus::Cancelled
                    | FinishedStatus::WritebackFailed,
                ..
            }
            | Self::Active { .. } => {
                return Err(Reactivation::Refused(self.refused(JobEvent::Resume)));
            }
        };
        self.requeue_on(admitted)
    }

    /// Queued on `admitted`, or unchanged when admission was refused.
    fn requeue_on<E>(&mut self, admitted: Result<JobInputs, E>) -> Result<(), Reactivation<E>> {
        let inputs = admitted.map_err(Reactivation::Admission)?;
        *self = Self::Active {
            status: ActiveStatus::Queued,
            inputs,
        };
        Ok(())
    }

    /// Interrupted to `settlement`, for a recovered job that will not run.
    pub(crate) fn settle(&mut self, settlement: Settlement) -> Result<(), Refused> {
        match self {
            Self::Finished {
                status: status @ FinishedStatus::Interrupted,
                ..
            } => {
                *status = match settlement {
                    Settlement::Completed => FinishedStatus::Completed,
                    Settlement::Failed => FinishedStatus::Failed,
                };
                Ok(())
            }
            Self::Finished {
                status:
                    FinishedStatus::Completed
                    | FinishedStatus::Failed
                    | FinishedStatus::Cancelled
                    | FinishedStatus::WritebackFailed,
                ..
            }
            | Self::Active { .. } => Err(self.refused(JobEvent::Settle)),
        }
    }
}

/// Fixture construction: tests set up a job in any state directly.
#[cfg(test)]
impl JobLifecycle {
    /// A fixture's lifecycle: `status` on `inputs`. An active status on
    /// recorded inputs is a state no job has; asking for one panics.
    pub(crate) fn for_test(status: JobStatus, inputs: impl Into<StoredInputs>) -> Self {
        match (status.kind(), inputs.into()) {
            (StatusKind::Active(status), StoredInputs::Admitted(inputs)) => {
                Self::Active { status, inputs }
            }
            (StatusKind::Active(_), StoredInputs::Recorded(_)) => {
                panic!("an active fixture needs admitted inputs")
            }
            (StatusKind::Finished(status), inputs) => Self::Finished { status, inputs },
        }
    }

    /// Put a fixture in `status`, keeping its inputs.
    pub(crate) fn force_status_for_test(&mut self, status: JobStatus) {
        let inputs = match self {
            Self::Active { inputs, .. } => StoredInputs::Admitted(inputs.clone()),
            Self::Finished { inputs, .. } => inputs.clone(),
        };
        *self = Self::for_test(status, inputs);
    }

    /// Give a fixture `inputs`, keeping its status.
    pub(crate) fn replace_inputs_for_test(&mut self, inputs: impl Into<StoredInputs>) {
        let status = self.status();
        *self = Self::for_test(status, inputs);
    }

    /// A fixture's admitted inputs, to change in place.
    pub(crate) fn admitted_mut_for_test(&mut self) -> &mut JobInputs {
        match self {
            Self::Active { inputs, .. }
            | Self::Finished {
                inputs: StoredInputs::Admitted(inputs),
                ..
            } => inputs,
            Self::Finished {
                inputs: StoredInputs::Recorded(_),
                ..
            } => panic!("the fixture's inputs are recorded"),
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
    match befores.iter().any(Option::is_some) {
        true => befores,
        false => Vec::new(),
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
            && match paths_mode {
                true => {
                    source_paths.len() == filenames.len()
                        && output_paths.len() == filenames.len()
                        && (before_paths.is_empty() || before_paths.len() == filenames.len())
                }
                // A content-mode job names no server path: a row that does
                // is refused, never read with its paths silently dropped.
                false => {
                    source_paths.is_empty() && output_paths.is_empty() && before_paths.is_empty()
                }
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
        Ok(match paths_mode {
            true => StoredInputs::Recorded(RecordedInputs {
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
            }),
            false => StoredInputs::Admitted(JobInputs::Content(
                named
                    .map(|(display, kind)| ContentInput { display, kind })
                    .collect(),
            )),
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

/// File lists and storage layout for one job.
#[derive(Debug, Clone)]
pub struct JobFilesystemConfig {
    /// Server-local temporary directory for staged input/output content.
    pub staging_dir: ServerPath,
    /// Key into the server's configured media-mapping roots.
    pub media_mapping: MediaMappingKey,
    /// Optional subdirectory within the selected media mapping.
    pub media_subdir: RepoRelativePath,
}

/// Mutable execution state for one job.
#[derive(Debug, Clone)]
pub struct JobExecutionState {
    /// Where the job is in its life, with the inputs that state needs. The
    /// status and the inputs are one value so an active job cannot hold
    /// inputs this process did not admit. Written only through
    /// the event methods on [`JobLifecycle`].
    pub(crate) lifecycle: JobLifecycle,
    /// Per-file processing state keyed by filename.
    pub file_statuses: HashMap<String, FileStatus>,
    /// Accumulated per-file result entries.
    pub results: Vec<FileResultEntry>,
    /// Job-level error message for terminal failures.
    pub error: Option<String>,
}

impl JobExecutionState {
    /// Files in a terminal phase. Derived, never stored: a stored counter
    /// beside the phases it summarises could drift (a repeated terminal
    /// write once counted a file twice) and needed resetting at every
    /// transition that changes phases.
    pub(crate) fn completed_files(&self) -> i64 {
        self.file_statuses
            .values()
            .filter(|file| file.status().is_terminal())
            .count() as i64
    }
}

/// Scheduling and completion state for one job.
#[derive(Debug, Clone)]
pub struct JobScheduleState {
    /// When the job was submitted.
    pub submitted_at: MachineTime,
    /// When the job reached a terminal state.
    pub completed_at: Option<MachineTime>,
    /// Earliest time a deferred queued job should be retried.
    pub next_eligible_at: Option<MachineTime>,
    /// Number of worker processes used for this job once running.
    pub num_workers: Option<i64>,
    /// The job's queue lease, if a node holds one. One value, so an owner
    /// without an expiry (or an expiry without an owner) cannot exist.
    pub lease: Option<crate::scheduling::LeaseRecord>,
    /// Most recent cancel attempt's metadata (denormalized from the
    /// `cancellations` audit table). `None` until a cancel arrives.
    /// Projected onto the `JobInfo`'s `last_cancelled_*` fields.
    pub last_cancel: Option<JobLastCancelInfo>,
}

/// Denormalized snapshot of the most recent cancel attempt.
///
/// Mirrors the relevant columns on `jobs.last_cancelled_*` so the
/// in-memory `Job` projection can fill `JobInfo` without a DB JOIN.
#[derive(Debug, Clone)]
pub struct JobLastCancelInfo {
    /// Wall-clock when the cancel arrived at the server.
    pub at: MachineTime,
    /// Wire-format source string (`"tui"`, `"api"`, `"signal"`, ...).
    pub source: String,
    /// Caller-reported host or peer-IP.
    pub host: Option<String>,
    /// Caller-reported reason text.
    pub reason: Option<String>,
}

/// Monotonic per-job run generation.
///
/// Every restart (`Job::prepare_for_restart`) increments it, so a runner
/// that began under an earlier generation can be recognized as STALE and
/// barred from finalizing the job or force-failing files that now belong
/// to the restarted run. Completion admits this generation and its file
/// cleanup in one registry transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunGeneration(u64);

impl RunGeneration {
    /// Generation of a job's first run, before any restart.
    pub(in crate::store) const FIRST: Self = Self(0);

    /// The next generation (used by restart).
    pub(crate) fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

impl std::fmt::Display for RunGeneration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "g{}", self.0)
    }
}

/// One run of one job: the identity every write from that run carries.
///
/// A job id alone cannot say whose write it is, because a restart reuses
/// the id under a new [`RunGeneration`]; a generation alone names no job.
/// The pair is minted only from a [`RunnerClaim`] (and, in tests, for a
/// job's first run), so no caller can assemble a binding from a job id and
/// a generation it picked. The store admits a write only while the bound
/// generation is still the job's current one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RunBinding {
    job_id: JobId,
    generation: RunGeneration,
}

impl RunBinding {
    /// The job this run belongs to.
    pub(crate) fn job_id(&self) -> &JobId {
        &self.job_id
    }

    /// Whether this run is still `job`'s current one. The only reading of
    /// the generation: the store asks this, and nothing compares raw
    /// generations.
    pub(in crate::store) fn is_current_for(&self, job: &super::Job) -> bool {
        job.runtime.run_generation == self.generation
    }

    /// The binding of a test job's first run, which has no claim to bind to.
    #[cfg(test)]
    pub(crate) fn first_run_of(job_id: JobId) -> Self {
        Self {
            job_id,
            generation: RunGeneration::FIRST,
        }
    }
}

impl std::fmt::Display for RunBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.job_id, self.generation)
    }
}

/// Outcome of a runner's attempt to claim exclusive execution of a job.
#[derive(Debug)]
pub(crate) enum BeginRunnerOutcome {
    /// The claim succeeded; the runner owns the job at this generation,
    /// under the lease the claim took (the store persists exactly that one).
    Started {
        /// Linear authority binding this run's generation and snapshot.
        claim: RunnerClaim,
        /// The lease taken with the claim.
        lease: crate::scheduling::LeaseRecord,
    },
    /// Another runner still owns the job (e.g. a restarted job whose
    /// previous runner is mid-teardown). The caller must wait and retry.
    RunnerStillLive,
    /// Cancellation arrived before a runner claimed the job. This owner
    /// can close unfinished files but cannot enter admission or dispatch.
    Stopped { completion: StoppedRun },
    /// A delayed or competing task arrived after this job stopped. It owns
    /// no claim, heartbeat, resources or release authority.
    NotRunnable { status: JobStatus },
}

/// Exclusive authority for one run, minted together with its immutable input.
///
/// Not cloneable: completion consumes it. A restart can supersede the claim,
/// but cannot give its cleanup authority over the restarted files.
#[derive(Debug)]
#[must_use = "a runner claim owns one attempt's completion"]
pub(crate) struct RunnerClaim {
    run: RunBinding,
    snapshot: std::sync::Arc<RunnerJobSnapshot>,
}

/// Completion-only authority for cancellation accepted before dispatch.
#[derive(Debug)]
#[must_use = "stopped files still require terminal bookkeeping"]
pub(crate) struct StoppedRun(RunnerClaim);

impl StoppedRun {
    pub(in crate::store) fn for_job(job: &Job) -> Option<Self> {
        RunnerClaim::for_job(job).map(Self)
    }

    pub(in crate::store) fn into_claim(self) -> RunnerClaim {
        self.0
    }
}

impl RunnerClaim {
    /// The claim for `job`'s current run: `None` when this process has not
    /// admitted the job's inputs, which then cannot run.
    pub(in crate::store) fn for_job(job: &Job) -> Option<Self> {
        Some(Self {
            run: RunBinding {
                job_id: job.identity.job_id.clone(),
                generation: job.runtime.run_generation,
            },
            snapshot: std::sync::Arc::new(job.to_runner_snapshot()?),
        })
    }

    /// Immutable dispatch input bound to the claimed generation.
    pub(crate) fn snapshot(&self) -> &std::sync::Arc<RunnerJobSnapshot> {
        &self.snapshot
    }

    /// The identity every write of this run carries.
    pub(crate) fn binding(&self) -> &RunBinding {
        &self.run
    }
}

/// Result of consuming a run's completion authority.
#[derive(Debug)]
pub(crate) enum RunCompletion {
    /// Restart or shutdown superseded this run; no old cleanup was applied.
    Superseded,
    /// Cleanup and final status were admitted together by the registry.
    Finished {
        /// Actual status, including a previously accepted stop.
        status: JobStatus,
        /// Aggregated failure reason for logging.
        error: Option<String>,
    },
}

/// In-memory runtime controls for one job.
pub struct JobRuntimeControl {
    /// Cancellation token checked between files by the runner.
    pub cancel_token: CancellationToken,
    /// Whether a runner task currently owns this job. Set by
    /// `JobRegistry::begin_runner` and cleared only when that runner
    /// releases its claim; restart does NOT clear it (the old runner is
    /// still alive during handoff).
    pub runner_active: bool,
    /// Current run generation (bumped by every restart). Only the store
    /// reads or advances it: everything outside names a run by the
    /// [`RunBinding`] its claim carries.
    pub(in crate::store) run_generation: RunGeneration,
}

impl JobRuntimeControl {
    /// A job's runtime state when it enters this process, newly submitted
    /// or recovered: not cancelled, no runner, its first run.
    pub fn unclaimed() -> Self {
        Self {
            cancel_token: CancellationToken::new(),
            runner_active: false,
            run_generation: RunGeneration::FIRST,
        }
    }
}

/// One file that still requires runner work.
///
/// Made only by [`JobInputs::entries`]: the fields are private, so a pending
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

/// Successful result metadata for one completed file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CompletedFileOutput {
    /// Logical result filename returned to clients.
    pub filename: DisplayPath,
    /// MIME-like content type stored with the result.
    pub content_type: ContentType,
    /// What the command decided about stamping this file with provenance.
    pub stamp: crate::api::FileStampOutcome,
    /// What the producer left out of its work on purpose; recorded on the
    /// file's terminal phase, whichever it is.
    pub exclusions: Vec<crate::api::OutputExclusionRecord>,
}

/// How a file finished without failing: what the store records as its
/// terminal phase and downloadable result.
///
/// A sum rather than an `Option<CompletedFileOutput>` beside a separate
/// "diagnostics" argument: a diagnosed file always has a written result, so
/// "diagnosed without a result" and "result with diagnostics nobody reads"
/// have no spelling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FileCompletion {
    /// Finished with nothing to download.
    WithoutResult,
    /// Finished with a result that passed its output admission, or whose kind
    /// has none. Recorded as `FilePhase::Done`.
    Clean(CompletedFileOutput),
    /// The command's own producer generated output that failed admission, and
    /// it was written together with what the admission found. Recorded as
    /// `FilePhase::Diagnosed`; terminal, never retried, not a failure.
    Diagnosed {
        /// The written result.
        result: CompletedFileOutput,
        /// Every finding, and every stage skipped because of them.
        diagnostics: crate::api::FileOutputDiagnostics,
    },
}

impl FileCompletion {
    /// The result to record, whichever way the file finished with one.
    pub(crate) fn result(&self) -> Option<&CompletedFileOutput> {
        match self {
            Self::WithoutResult => None,
            Self::Clean(result) | Self::Diagnosed { result, .. } => Some(result),
        }
    }
}

/// Failure details for one terminal file error.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FileFailureRecord {
    /// Human-readable error message.
    pub message: String,
    /// Broad failure category for grouping and retry policy.
    pub category: crate::scheduling::FailureCategory,
    /// When the file failed, by the store's clock.
    pub finished_at: crate::store::EventTime,
}

/// Retry metadata for one transient file failure.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FileRetryRecord {
    /// Human-readable retry message shown to clients.
    pub message: String,
    /// Broad failure category for the retryable attempt.
    pub category: crate::scheduling::FailureCategory,
    /// When the failed attempt finished, by the store's clock.
    pub finished_at: crate::store::EventTime,
    /// Earliest time when the next attempt may run.
    pub retry_at: MachineTime,
}

/// Ephemeral progress update for one in-flight file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FileProgressRecord {
    /// Stable machine-readable stage code.
    pub stage: FileProgressStage,
    /// Current progress counter when available.
    pub current: Option<i64>,
    /// Total progress counter when available.
    pub total: Option<i64>,
}

/// Stable identity fields for a running job.
#[derive(Debug, Clone)]
pub struct RunnerJobIdentity {
    /// Job identifier for logging and downstream APIs.
    pub job_id: JobId,
    /// Correlation identifier for structured logs.
    pub correlation_id: CorrelationId,
}

/// Immutable dispatch-time configuration for one job.
#[derive(Debug, Clone)]
pub struct RunnerDispatchConfig {
    /// Command being executed.
    pub command: ReleasedCommand,
    /// Language specification: may be `Auto` for ASR auto-detection.
    pub lang: LanguageSpec,
    /// Speaker-count hint for audio workflows.
    pub num_speakers: NumSpeakers,
    /// Typed command options captured at submission time.
    pub options: CommandOptions,
    /// Server-internal runtime state for orchestration helpers.
    pub runtime_state: BTreeMap<String, serde_json::Value>,
    /// Whether algorithm traces should be persisted for this job.
    pub debug_traces: bool,
}

/// Filesystem and media-resolution configuration for one job.
#[derive(Debug, Clone)]
pub struct RunnerFilesystemConfig {
    /// The job's inputs, indexed by [`PendingJobFile::file_index`].
    pub inputs: JobInputs,
    /// Staging directory for uploaded content mode.
    pub staging_dir: ServerPath,
    /// Media-mapping key for server-side audio lookup.
    pub media_mapping: MediaMappingKey,
    /// Subdirectory within the selected media mapping.
    pub media_subdir: RepoRelativePath,
    /// Client-provided source directory, used for media locality inference.
    ///
    /// The FA pipeline passes this to `infer_media_mapping()` to auto-detect
    /// which media volume and repo-relative subdir to search for audio files.
    pub source_dir: ClientPath,
}

/// Immutable runner-facing snapshot of job state.
///
/// The runner should read one of these projections instead of repeatedly
/// locking the raw job map and reconstructing the same static configuration.
#[derive(Debug, Clone)]
pub struct RunnerJobSnapshot {
    /// Stable identity values for this job.
    pub identity: RunnerJobIdentity,
    /// Dispatch-time configuration for orchestration and worker calls.
    pub dispatch: RunnerDispatchConfig,
    /// Filesystem and media-resolution layout for this job.
    pub filesystem: RunnerFilesystemConfig,
    /// Cancellation token cloned from the live job state.
    pub cancel_token: CancellationToken,
    /// Files that still need processing.
    pub pending_files: Vec<PendingJobFile>,
}

/// What startup recovery did to one loaded job, for the database.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Recovered {
    /// Nothing: the row stands as stored.
    Unchanged,
    /// The job's status columns changed, and so did each listed file's
    /// phase (requeued, or closed).
    Rewritten(Vec<(String, crate::store::FilePhase)>),
}

#[cfg(test)]
mod submitter_tests {
    use super::{StoredSubmitter, Submitter};
    use std::net::Ipv4Addr;

    /// The database boundary is the one place `''` means "absent", and the
    /// two directions agree on it.
    #[test]
    fn database_columns_round_trip_including_absence() {
        assert_eq!(Submitter::columns(None), ("", ""));
        assert_eq!(
            Submitter::from_columns(String::new(), String::new()),
            StoredSubmitter::Absent
        );

        let named = Submitter::client(Ipv4Addr::new(10, 0, 0, 7).into(), "lab-mac".into());
        let (address, name) = Submitter::columns(Some(&named));
        assert_eq!((address, name), ("10.0.0.7", "lab-mac"));
        assert_eq!(
            Submitter::from_columns(address.to_owned(), name.to_owned()),
            StoredSubmitter::Recorded(named)
        );

        let unnamed = Submitter::client(Ipv4Addr::new(10, 0, 0, 8).into(), String::new());
        assert_eq!(unnamed.name(), None, "an empty resolution records no name");
        let (address, name) = Submitter::columns(Some(&unnamed));
        assert_eq!(
            Submitter::from_columns(address.to_owned(), name.to_owned()),
            StoredSubmitter::Recorded(unnamed)
        );
    }

    /// A name with no address is not something the writer produces, so it
    /// is not read back as a submitter, and the name comes back to be
    /// reported rather than vanishing.
    #[test]
    fn a_name_without_an_address_is_no_submitter_and_is_handed_back() {
        assert_eq!(
            Submitter::from_columns(String::new(), "orphan".into()),
            StoredSubmitter::NameWithoutAddress {
                name: "orphan".into()
            }
        );
    }
}
