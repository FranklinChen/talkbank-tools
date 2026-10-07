//! Job struct, associated types, and conflict detection.

use std::collections::{BTreeMap, HashMap};

use batchalign_types::paths::{ClientPath, MediaMappingKey, RepoRelativePath, ServerPath};

use crate::api::{
    ContentType, CorrelationId, DisplayPath, FileProgressStage, JobId, JobStatus, LanguageSpec,
    MachineTime, NumSpeakers, ReleasedCommand,
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

/// File lists and storage layout for one job.
#[derive(Debug, Clone)]
pub struct JobFilesystemConfig {
    /// Ordered list of file basenames to process.
    pub filenames: Vec<DisplayPath>,
    /// Parallel CHAT/media markers for [`Self::filenames`].
    pub has_chat: Vec<bool>,
    /// Server-local temporary directory for staged input/output content.
    pub staging_dir: ServerPath,
    /// Whether the job reads and writes directly on the filesystem.
    pub paths_mode: bool,
    /// Absolute source paths parallel to [`Self::filenames`] in paths mode.
    pub source_paths: Vec<ClientPath>,
    /// Absolute output paths parallel to [`Self::filenames`] in paths mode.
    pub output_paths: Vec<ClientPath>,
    /// Optional "before" paths parallel to [`Self::source_paths`].
    pub before_paths: Vec<ClientPath>,
    /// Key into the server's configured media-mapping roots.
    pub media_mapping: MediaMappingKey,
    /// Optional subdirectory within the selected media mapping.
    pub media_subdir: RepoRelativePath,
    /// Client-provided source directory for media locality inference.
    ///
    /// In paths mode, the FA pipeline uses this to auto-detect the media
    /// mapping via [`batchalign_types::paths::infer_media_mapping()`].
    /// Without this field, `--server` jobs from remote clients cannot
    /// resolve media files.
    pub source_dir: ClientPath,
}

/// Mutable execution state for one job.
#[derive(Debug, Clone)]
pub struct JobExecutionState {
    /// Current lifecycle state of the job.
    pub status: JobStatus,
    /// Per-file processing state keyed by filename.
    pub file_statuses: HashMap<String, FileStatus>,
    /// Accumulated per-file result entries.
    pub results: Vec<FileResultEntry>,
    /// Job-level error message for terminal failures.
    pub error: Option<String>,
    /// Count of files that have reached a terminal status.
    pub completed_files: i64,
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
/// to the restarted run (2026-07-10 field failure: a stale runner
/// finalized a restarted 345-file job as `failed` with 317 bogus
/// "did not reach terminal status" file errors).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunGeneration(u64);

impl RunGeneration {
    /// Generation of a job's first run, before any restart.
    pub(crate) const FIRST: Self = Self(0);

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

/// Outcome of a runner's attempt to claim exclusive execution of a job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BeginRunnerOutcome {
    /// The claim succeeded; the runner owns the job at this generation,
    /// under the lease the claim took (the store persists exactly that one).
    Started {
        /// The run generation the runner owns.
        generation: RunGeneration,
        /// The lease taken with the claim.
        lease: crate::scheduling::LeaseRecord,
    },
    /// Another runner still owns the job (e.g. a restarted job whose
    /// previous runner is mid-teardown). The caller must wait and retry.
    RunnerStillLive,
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
    /// Current run generation (bumped by every restart).
    pub run_generation: RunGeneration,
}

/// One file that still requires runner work.
///
/// This is the stable runner-facing replacement for ad hoc
/// `(usize, DisplayPath, bool)` tuples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingJobFile {
    /// Index into the job's parallel path vectors.
    pub file_index: usize,
    /// Logical filename of the work item.
    pub filename: DisplayPath,
    /// Whether the input is CHAT text rather than media.
    pub has_chat: bool,
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
    /// Whether the job reads and writes directly on the filesystem.
    pub paths_mode: bool,
    /// Source paths parallel to [`PendingJobFile::file_index`].
    pub source_paths: Vec<ClientPath>,
    /// Output paths parallel to [`PendingJobFile::file_index`].
    pub output_paths: Vec<ClientPath>,
    /// Optional "before" paths parallel to [`PendingJobFile::file_index`].
    pub before_paths: Vec<ClientPath>,
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
    /// Run generation this snapshot belongs to; finalization is refused
    /// when the job has since moved to a newer generation.
    pub run_generation: RunGeneration,
}

/// Result of recovering a persisted interrupted/running job on startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecoveryDisposition {
    /// The job still has resumable work and was returned to the queue.
    Requeued,
    /// The job had only terminal failures and was promoted to failed.
    Failed,
    /// The job had completed output and was promoted to completed.
    Completed,
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
