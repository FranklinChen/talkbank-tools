//! Job metadata, execution records and immutable runner snapshots.
use super::{JobInputs, JobLifecycle, PendingJobFile, RunGeneration, Submitter};
use crate::api::{CorrelationId, JobId, LanguageSpec, MachineTime, NumSpeakers, ReleasedCommand};
use crate::options::CommandOptions;
use crate::store::{FileResultEntry, FileStatus};
use crate::types::execution_plan::ExecutionPlan;
use batchalign_types::paths::{ClientPath, MediaMappingKey, RepoRelativePath, ServerPath};
use std::collections::{BTreeMap, HashMap};
use tokio_util::sync::CancellationToken;

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
    #[must_use]
    pub fn unclaimed() -> Self {
        Self {
            cancel_token: CancellationToken::new(),
            runner_active: false,
            run_generation: RunGeneration::FIRST,
        }
    }
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
