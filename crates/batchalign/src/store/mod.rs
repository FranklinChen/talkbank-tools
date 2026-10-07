//! In-memory job store: port of `batchalign/serve/job_store.py`.
//!
//! `JobStore` is the top-level control-plane owner for in-memory job state.
//! It is composed from smaller owned runtime pieces:
//!
//! - [`registry::JobRegistry`] for the in-memory job map
//! - [`counters::OperationalCounterStore`] for health/metrics bookkeeping
//! - a store-owned `tokio::sync::Semaphore` for global job concurrency
//! - SQLite and WebSocket boundaries for durability and live notifications
//!
//! Each job still runs as a `tokio::spawn` task, but the store no longer
//! presents its synchronization primitives as just public fields to poke.

mod counters;
mod event_time;
mod job;
pub(crate) mod queries;
mod registry;

pub use event_time::EventTime;
pub use job::*;
pub(crate) use job::{
    CompletedFileOutput, FileCompletion, FileFailureRecord, FileProgressRecord, FileRetryRecord,
    JobStatusColumns, Stop,
};
pub(crate) use queries::LeaseRenewalOutcome;
pub(crate) use queries::{AttemptFinishRecord, AttemptStartRecord, PersistedFileUpdate};
pub(crate) use registry::JobCompletionSnapshot;

use std::sync::Arc;

use crate::api::{
    ContentType, DisplayPath, FileOutputDiagnostics, FileProgressStage, FileStatusEntry,
    FileStatusKind, JobStatus, MachineTime, NodeId, NonNegativeSeconds,
};
use crate::config::ServerConfig;
use crate::host_policy::HostExecutionPolicy;
#[cfg(test)]
use crate::host_policy::auto_max_concurrent_from as host_auto_max_concurrent_from;
use crate::scheduling::FailureCategory;
use tokio::sync::{AcquireError, Semaphore, SemaphorePermit, broadcast};
use tracing::info;

use crate::db::JobDB;
use crate::ws::WsEvent;
use counters::OperationalCounterStore;
use registry::JobRegistry;

// ---------------------------------------------------------------------------
// Per-file status
// ---------------------------------------------------------------------------

/// Why a file's attempt failed, as far as it is known.
///
/// This build records a message and a category for every failure
/// ([`FileFailure::recorded`]). A row another build wrote may hold one of the
/// two, or (on a failed row) neither; those shapes are read only at the
/// database boundary ([`FileFailure::from_columns`],
/// [`FileFailure::of_failed_row`]). The shape is private, so no other code
/// can make a failure this build did not record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileFailure(Failure);

/// The shapes a failure can have.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Failure {
    /// A message and a category, as this build records every failure.
    Recorded {
        message: String,
        category: FailureCategory,
    },
    /// A stored message with no category.
    MessageOnly(String),
    /// A stored category with no message.
    CategoryOnly(FailureCategory),
    /// A stored failed row that named neither: the file failed, and nothing
    /// says why.
    Unrecorded,
}

impl FileFailure {
    /// A failure this build records: always both a message and a category.
    pub fn recorded(message: String, category: FailureCategory) -> Self {
        Self(Failure::Recorded { message, category })
    }

    /// The failure stored columns hold; `None` when they hold neither. For
    /// a phase whose failure is optional (the attempt before an
    /// interruption).
    pub(crate) fn from_columns(
        message: Option<String>,
        category: Option<FailureCategory>,
    ) -> Option<Self> {
        match (message, category) {
            (None, None) => None,
            (message, category) => Some(Self::of_failed_row(message, category)),
        }
    }

    /// The failure of a stored row that says the file failed (an error, or
    /// a pending retry): its columns, or [`Failure::Unrecorded`] when they
    /// hold neither.
    pub(crate) fn of_failed_row(
        message: Option<String>,
        category: Option<FailureCategory>,
    ) -> Self {
        Self(match (message, category) {
            (Some(message), Some(category)) => Failure::Recorded { message, category },
            (Some(message), None) => Failure::MessageOnly(message),
            (None, Some(category)) => Failure::CategoryOnly(category),
            (None, None) => Failure::Unrecorded,
        })
    }

    /// The message, when one was recorded.
    pub fn message(&self) -> Option<&str> {
        match &self.0 {
            Failure::Recorded { message, .. } | Failure::MessageOnly(message) => Some(message),
            Failure::CategoryOnly(_) | Failure::Unrecorded => None,
        }
    }

    /// The category, when one was recorded.
    pub fn category(&self) -> Option<FailureCategory> {
        match &self.0 {
            Failure::Recorded { category, .. } | Failure::CategoryOnly(category) => Some(*category),
            Failure::MessageOnly(_) | Failure::Unrecorded => None,
        }
    }
}

/// Where a file is in its lifecycle, carrying only the times and the failure
/// that phase has.
///
/// A finish time exists only on `Done`, `Diagnosed` and `Error`, a retry
/// deadline only on `RetryPending`, a failure only where one happened,
/// admission diagnostics only on `Diagnosed`, and a queued file has no times
/// at all.
///
/// The times are `Option` only for rows recovered from storage written by a
/// build that did not record them; every transition this build makes records
/// them. A failed phase (`RetryPending`, `Error`) always has its failure; a
/// stored one that named none says so ([`FileFailure::of_failed_row`]).
#[derive(Debug, Clone, PartialEq)]
pub enum FilePhase {
    /// Waiting to be dispatched; nothing has run.
    Queued,
    /// An attempt is running.
    Processing {
        /// When the running attempt started.
        started_at: Option<MachineTime>,
    },
    /// The last attempt failed transiently; the next may run at `retry_at`.
    /// Still in flight, so it has no finish time and no duration.
    RetryPending {
        /// When the failed attempt started.
        started_at: Option<MachineTime>,
        /// When the failed attempt ended (stall-alarm activity, attempt
        /// history), deliberately not the file's finish time.
        failed_at: Option<MachineTime>,
        /// Earliest time the next attempt may run.
        retry_at: MachineTime,
        /// Why the attempt failed.
        failure: FileFailure,
    },
    /// Finished successfully. Terminal.
    Done {
        /// When the successful attempt started.
        started_at: Option<MachineTime>,
        /// When it finished.
        finished_at: Option<MachineTime>,
    },
    /// Finished, with the output written together with the diagnostics its
    /// admission found. Terminal, never retried, and not a failure: the
    /// command's own producer generated the document, so the written CHAT is
    /// the evidence a reviewer needs, not a reason to discard the file.
    Diagnosed {
        /// When the attempt started.
        started_at: Option<MachineTime>,
        /// When the output was written.
        finished_at: Option<MachineTime>,
        /// What admission found. `Option` only for a row recovered from
        /// storage that did not record it, like the times; every transition
        /// this build makes records it.
        diagnostics: Option<FileOutputDiagnostics>,
    },
    /// Finished with an error. Terminal.
    Error {
        /// When the failed attempt started (absent for a setup refusal that
        /// never started processing).
        started_at: Option<MachineTime>,
        /// When it failed.
        finished_at: Option<MachineTime>,
        /// Why.
        failure: FileFailure,
    },
    /// In flight when the server stopped; resumable on restart.
    Interrupted {
        /// When the interrupted attempt started.
        started_at: Option<MachineTime>,
        /// The failure of the attempt before it, if it was awaiting a retry.
        last_failure: Option<FileFailure>,
    },
}

impl FilePhase {
    /// The API's status vocabulary. A pending retry is still `Processing`.
    pub fn kind(&self) -> FileStatusKind {
        match self {
            Self::Queued => FileStatusKind::Queued,
            Self::Processing { .. } | Self::RetryPending { .. } => FileStatusKind::Processing,
            Self::Done { .. } => FileStatusKind::Done,
            Self::Diagnosed { .. } => FileStatusKind::Diagnosed,
            Self::Error { .. } => FileStatusKind::Error,
            Self::Interrupted { .. } => FileStatusKind::Interrupted,
        }
    }

    /// When the current or last attempt started.
    pub fn started_at(&self) -> Option<MachineTime> {
        match self {
            Self::Queued => None,
            Self::Processing { started_at }
            | Self::RetryPending { started_at, .. }
            | Self::Done { started_at, .. }
            | Self::Diagnosed { started_at, .. }
            | Self::Error { started_at, .. }
            | Self::Interrupted { started_at, .. } => *started_at,
        }
    }

    /// When the file finished: only a terminal file has finished.
    pub fn finished_at(&self) -> Option<MachineTime> {
        match self {
            Self::Done { finished_at, .. }
            | Self::Diagnosed { finished_at, .. }
            | Self::Error { finished_at, .. } => *finished_at,
            Self::Queued
            | Self::Processing { .. }
            | Self::RetryPending { .. }
            | Self::Interrupted { .. } => None,
        }
    }

    /// The failure this phase carries, if any.
    pub fn failure(&self) -> Option<&FileFailure> {
        match self {
            Self::RetryPending { failure, .. } | Self::Error { failure, .. } => Some(failure),
            Self::Interrupted { last_failure, .. } => last_failure.as_ref(),
            Self::Queued | Self::Processing { .. } | Self::Done { .. } | Self::Diagnosed { .. } => {
                None
            }
        }
    }

    /// The admission diagnostics of a file written with them.
    pub fn diagnostics(&self) -> Option<&FileOutputDiagnostics> {
        match self {
            Self::Diagnosed { diagnostics, .. } => diagnostics.as_ref(),
            Self::Queued
            | Self::Processing { .. }
            | Self::RetryPending { .. }
            | Self::Done { .. }
            | Self::Error { .. }
            | Self::Interrupted { .. } => None,
        }
    }

    /// The phase a file is in after the server stopped under it: an
    /// in-flight phase (queued, processing, awaiting a retry) becomes
    /// `Interrupted`, keeping its start and the failure it was retrying;
    /// a terminal or already interrupted phase is unchanged.
    pub(crate) fn interrupted(self) -> Self {
        match self {
            Self::Queued => Self::Interrupted {
                started_at: None,
                last_failure: None,
            },
            Self::Processing { started_at } => Self::Interrupted {
                started_at,
                last_failure: None,
            },
            Self::RetryPending {
                started_at,
                failure,
                ..
            } => Self::Interrupted {
                started_at,
                last_failure: Some(failure),
            },
            unchanged @ (Self::Done { .. }
            | Self::Diagnosed { .. }
            | Self::Error { .. }
            | Self::Interrupted { .. }) => unchanged,
        }
    }

    /// When a pending retry may run.
    pub fn next_eligible_at(&self) -> Option<MachineTime> {
        match self {
            Self::RetryPending { retry_at, .. } => Some(*retry_at),
            Self::Queued
            | Self::Processing { .. }
            | Self::Done { .. }
            | Self::Diagnosed { .. }
            | Self::Error { .. }
            | Self::Interrupted { .. } => None,
        }
    }

    /// The latest moment this file is known to have moved (stall alarm).
    pub fn last_activity_at(&self) -> Option<MachineTime> {
        let failed_at = match self {
            Self::RetryPending { failed_at, .. } => *failed_at,
            Self::Queued
            | Self::Processing { .. }
            | Self::Done { .. }
            | Self::Diagnosed { .. }
            | Self::Error { .. }
            | Self::Interrupted { .. } => None,
        };
        [self.started_at(), self.finished_at(), failed_at]
            .into_iter()
            .flatten()
            .max()
    }

    /// The `file_statuses` columns this phase owns, every one of them, with
    /// `None` written as NULL.
    ///
    /// The ONE route from a phase to a row: every write of a file's phase
    /// goes through it (the insert of a new job's rows as `Queued`,
    /// `JobDB::update_file_status` for every transition, startup
    /// interruption and the recovery requeue, both of which call it), all
    /// binding it with `db::update::bind_phase_columns`; [`Self::from_row`]
    /// is its inverse. A write therefore replaces the whole column set, so a
    /// column the new phase does not own cannot survive from an earlier one
    /// (a file that failed and then succeeded is stored `done` with no
    /// error).
    pub(crate) fn columns(&self) -> FilePhaseColumns<'_> {
        let failure = self.failure();
        let no_columns = FilePhaseColumns {
            status: self.kind(),
            error: failure.and_then(FileFailure::message),
            error_category: failure.and_then(FileFailure::category),
            diagnostics: self.diagnostics(),
            started_at: None,
            finished_at: None,
            next_eligible_at: None,
        };
        match self {
            Self::Queued => no_columns,
            Self::Processing { started_at } => FilePhaseColumns {
                started_at: *started_at,
                ..no_columns
            },
            Self::RetryPending {
                started_at,
                failed_at,
                retry_at,
                ..
            } => FilePhaseColumns {
                started_at: *started_at,
                // The failed attempt's end, stored in `finished_at`; the
                // deadline is what tells `from_row` it is not a finish time.
                finished_at: *failed_at,
                next_eligible_at: Some(*retry_at),
                ..no_columns
            },
            Self::Done {
                started_at,
                finished_at,
            }
            | Self::Diagnosed {
                started_at,
                finished_at,
                ..
            } => FilePhaseColumns {
                started_at: *started_at,
                finished_at: *finished_at,
                ..no_columns
            },
            Self::Error {
                started_at,
                finished_at,
                ..
            } => FilePhaseColumns {
                started_at: *started_at,
                finished_at: *finished_at,
                ..no_columns
            },
            Self::Interrupted { started_at, .. } => FilePhaseColumns {
                started_at: *started_at,
                ..no_columns
            },
        }
    }

    /// Rebuild a phase from a persisted `file_statuses` row: the database
    /// boundary, where the columns stay as they are. The inverse of
    /// [`Self::columns`].
    ///
    /// A `processing` row with a retry deadline is a pending retry (its
    /// `finished_at` column is the failed attempt's end). Columns a phase does
    /// not own are not carried into it, and every one that held a value is
    /// reported, by one rule for every phase: whatever [`Self::columns`] of
    /// the rebuilt phase would not write back. This build never writes such a
    /// column, so a report names a row written by another build or by hand.
    pub(crate) fn from_row(row: FilePhaseColumns<'_>) -> (Self, Option<String>) {
        let FilePhaseColumns {
            status: kind,
            error,
            error_category,
            diagnostics,
            started_at,
            finished_at,
            next_eligible_at,
        } = row;
        let message = error.map(str::to_owned);
        let held = RowHeld {
            started_at: started_at.is_some(),
            finished_at: finished_at.is_some(),
            next_eligible_at: next_eligible_at.is_some(),
            failure: message.is_some() || error_category.is_some(),
            diagnostics: diagnostics.is_some(),
        };
        let phase = match (kind, next_eligible_at) {
            (FileStatusKind::Processing, Some(retry_at)) => Self::RetryPending {
                started_at,
                failed_at: finished_at,
                retry_at,
                failure: FileFailure::of_failed_row(message, error_category),
            },
            (FileStatusKind::Processing, None) => Self::Processing { started_at },
            (FileStatusKind::Error, _) => Self::Error {
                started_at,
                finished_at,
                failure: FileFailure::of_failed_row(message, error_category),
            },
            (FileStatusKind::Interrupted, _) => Self::Interrupted {
                started_at,
                last_failure: FileFailure::from_columns(message, error_category),
            },
            (FileStatusKind::Done, _) => Self::Done {
                started_at,
                finished_at,
            },
            (FileStatusKind::Diagnosed, _) => Self::Diagnosed {
                started_at,
                finished_at,
                diagnostics: diagnostics.cloned(),
            },
            (FileStatusKind::Queued, _) => Self::Queued,
        };
        let dropped = held.not_kept_by(&phase.columns());
        let report = (!dropped.is_empty()).then(|| {
            format!(
                "the {kind} row held {}, which {} not kept",
                dropped.join(", "),
                if dropped.len() == 1 { "was" } else { "were" }
            )
        });
        (phase, report)
    }
}

/// Which phase-owned columns a persisted row held a value in.
struct RowHeld {
    started_at: bool,
    finished_at: bool,
    next_eligible_at: bool,
    failure: bool,
    diagnostics: bool,
}

impl RowHeld {
    /// The columns the row held that `kept` (the rebuilt phase's own column
    /// image) does not write back.
    fn not_kept_by(&self, kept: &FilePhaseColumns<'_>) -> Vec<&'static str> {
        let kept_failure = kept.error.is_some() || kept.error_category.is_some();
        [
            (self.started_at && kept.started_at.is_none(), "started_at"),
            (
                self.finished_at && kept.finished_at.is_none(),
                "finished_at",
            ),
            (
                self.next_eligible_at && kept.next_eligible_at.is_none(),
                "next_eligible_at",
            ),
            (self.failure && !kept_failure, "an error"),
            (
                self.diagnostics && kept.diagnostics.is_none(),
                "diagnostics",
            ),
        ]
        .into_iter()
        .filter_map(|(dropped, column)| dropped.then_some(column))
        .collect()
    }
}

/// The `file_statuses` columns a [`FilePhase`] owns, NULLs included: the
/// image [`FilePhase::columns`] writes and [`FilePhase::from_row`] reads.
///
/// The row writers take the [`FilePhase`] and derive this themselves, so a
/// write cannot name a column combination no phase has. A reader builds one
/// from a stored row and hands it to [`FilePhase::from_row`] whole.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FilePhaseColumns<'a> {
    /// The `status` column.
    pub status: FileStatusKind,
    /// The `error` column.
    pub error: Option<&'a str>,
    /// The `error_category` column.
    pub error_category: Option<FailureCategory>,
    /// The `diagnostics` column, decoded: a diagnosed file's admission
    /// findings. Its JSON text is written by `db::update::bind_phase_columns`
    /// and read back at the database boundary (`recover_file_phase`).
    pub diagnostics: Option<&'a FileOutputDiagnostics>,
    /// The `started_at` column.
    pub started_at: Option<MachineTime>,
    /// The `finished_at` column: the finish time, or a pending retry's
    /// failed-attempt end.
    pub finished_at: Option<MachineTime>,
    /// The `next_eligible_at` column: a pending retry's deadline.
    pub next_eligible_at: Option<MachineTime>,
}

/// Ephemeral progress of an in-flight file, for live display only (never
/// persisted).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FileProgress {
    /// Current step (e.g. utterance index).
    pub current: Option<i64>,
    /// Total expected steps.
    pub total: Option<i64>,
    /// Stable code for the current stage; the API derives the label.
    pub stage: Option<FileProgressStage>,
}

/// A change in how many of a file's worker checkouts wait on a saturated
/// pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorkerWaitChange {
    /// A checkout began waiting.
    Started,
    /// A checkout stopped waiting.
    Ended,
}

/// How many of a file's worker checkouts are waiting on a saturated pool.
///
/// Its own fact, beside the stage the pipeline reported ([`FileProgress`]):
/// two publishers (the pipeline's progress and the pool's wait observer)
/// each own one, so neither can overwrite the other, and what the file shows
/// is derived from both ([`FileStatus::to_entry`]). It used to be one shown
/// stage that each publisher overwrote, with a view in the runner trying to
/// restore whichever the other had replaced. Ephemeral, like progress.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpenWorkerWaits(u32);

impl OpenWorkerWaits {
    /// Apply one change. An end with no open wait is a pairing defect
    /// upstream (every end comes from a started wait's drop guard): it is
    /// logged and changes nothing.
    pub(crate) fn apply(&mut self, change: WorkerWaitChange) {
        match change {
            WorkerWaitChange::Started => self.0 = self.0.saturating_add(1),
            WorkerWaitChange::Ended => match self.0.checked_sub(1) {
                Some(remaining) => self.0 = remaining,
                None => tracing::warn!("a worker wait ended that never started; count unchanged"),
            },
        }
    }

    /// Whether any checkout of the file is waiting.
    pub(crate) fn any(self) -> bool {
        self.0 > 0
    }
}

/// Tracks the processing state of a single file within a job.
///
/// Each file begins `Queued` and moves through `Processing` (and possibly
/// `RetryPending`) to `Done`, `Diagnosed` or `Error`; see [`FilePhase`]. Progress is
/// ephemeral (not persisted to SQLite) and is cleared on job restart.
#[derive(Debug, Clone)]
pub struct FileStatus {
    /// Display path for this file, unique within the parent job. May be a
    /// bare basename (`"sample.cha"`) or a relative path (`"PWA/TYO_a1.cha"`)
    /// depending on whether the input had subdirectories.
    pub filename: DisplayPath,
    /// Where the file is, with the times and failure that phase has.
    pub phase: FilePhase,
    /// What the command decided about stamping this file with provenance,
    /// recorded when the file completed.
    /// Ephemeral -- not persisted to SQLite, so a status restored on restart
    /// reads `Unrecorded`.
    pub stamp: crate::api::FileStampOutcome,
    /// Durable identifier of the currently active attempt row for this file.
    /// Ephemeral -- restored as `None` on startup and recreated on the next
    /// dispatch when unfinished files are resumed.
    pub current_attempt_id: Option<String>,
    /// Live progress, as the pipeline reported it. Ephemeral.
    pub progress: FileProgress,
    /// Checkouts of this file waiting on a saturated pool. Ephemeral.
    pub worker_waits: OpenWorkerWaits,
}

impl FileStatus {
    /// Create a new `FileStatus` in the `Queued` state.
    pub fn new(filename: DisplayPath) -> Self {
        Self {
            filename,
            phase: FilePhase::Queued,
            stamp: crate::api::FileStampOutcome::Unrecorded,
            current_attempt_id: None,
            progress: FileProgress::default(),
            worker_waits: OpenWorkerWaits::default(),
        }
    }

    /// The progress the file shows: "waiting for a worker" while any of its
    /// checkouts waits on a saturated pool (an in-flight file only), else
    /// the stage its pipeline last reported, with its counts.
    fn shown_progress(&self) -> FileProgress {
        match (self.worker_waits.any(), self.status().is_terminal()) {
            (true, false) => FileProgress {
                stage: Some(FileProgressStage::WaitingForWorker),
                current: None,
                total: None,
            },
            (false, _) | (true, true) => self.progress.clone(),
        }
    }

    /// The API status of this file.
    pub fn status(&self) -> FileStatusKind {
        self.phase.kind()
    }

    /// Back to queued, as on a restart: no times, no failure, no attempt,
    /// no progress.
    pub(crate) fn requeue(&mut self) {
        self.phase = FilePhase::Queued;
        self.current_attempt_id = None;
        self.progress = FileProgress::default();
        self.worker_waits = OpenWorkerWaits::default();
    }

    /// Convert to the API response type.
    pub fn to_entry(&self) -> FileStatusEntry {
        let started_at = self.phase.started_at();
        let finished_at = self.phase.finished_at();
        let failure = self.phase.failure();
        let progress = self.shown_progress();
        FileStatusEntry {
            filename: self.filename.clone(),
            status: self.phase.kind(),
            error: failure.and_then(FileFailure::message).map(str::to_owned),
            error_category: failure.and_then(FileFailure::category),
            diagnostics: self.phase.diagnostics().cloned(),
            stamp: self.stamp.clone(),
            started_at,
            finished_at,
            duration_s: match (started_at, finished_at) {
                (Some(started), Some(finished)) => {
                    Some(NonNegativeSeconds::between(started, finished))
                }
                (None, _) | (_, None) => None,
            },
            next_eligible_at: self.phase.next_eligible_at(),
            progress_current: progress.current,
            progress_total: progress.total,
            progress_stage: progress.stage,
            progress_label: progress
                .stage
                .map(FileProgressStage::label)
                .map(str::to_string),
        }
    }
}

// ---------------------------------------------------------------------------
// Operational counters
// ---------------------------------------------------------------------------

/// Monotonically increasing counters for server health diagnostics.
///
/// Exposed via the `GET /health` endpoint so operators can detect systemic
/// problems (frequent crashes, memory exhaustion) without reading logs.
/// All counters start at zero and are never reset during a server's lifetime.
#[derive(Debug, Default)]
pub struct OperationalCounters {
    /// Number of times a Python worker process exited unexpectedly (non-zero
    /// exit code or broken pipe) while processing a file.  High values indicate
    /// model instability or OOM kills.
    pub worker_crashes: i64,
    /// Number of work-unit attempts started.
    pub attempts_started: i64,
    /// Number of attempts classified as retryable.
    pub attempts_retried: i64,
    /// Number of work units deferred for later scheduling.
    pub deferred_work_units: i64,
    /// Number of files that were forcibly marked as terminal errors by the
    /// runner (e.g. after exhausting retry attempts or encountering an
    /// unrecoverable worker failure).
    pub forced_terminal_errors: i64,
    /// Number of jobs that were rejected because available memory stayed below
    /// `memory_gate_mb` for the full `MEMORY_GATE_TIMEOUT_S` duration.
    pub memory_gate_aborts: i64,
}

// ---------------------------------------------------------------------------
// FileResultEntry
// ---------------------------------------------------------------------------

/// Minimal result entry for a single file, stored after it reaches a terminal
/// state.
///
/// Used by the results download endpoint to determine what to serve back to
/// the client.  Successful files have `error: None`; failed files carry the
/// error message so the client can report per-file failures.
#[derive(Debug, Clone)]
pub struct FileResultEntry {
    /// Display path (matches the corresponding `FileStatus::filename`).
    pub filename: DisplayPath,
    /// Content discriminator for the result file.
    pub content_type: ContentType,
    /// Error message if processing failed for this file; `None` on success.
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// JobDetail
// ---------------------------------------------------------------------------

/// Snapshot of job state needed by the results-download endpoint.
///
/// This is a lightweight projection of [`Job`] that avoids exposing the full
/// struct (and its `CancellationToken`) outside the store.  It carries just
/// enough information for the handler to locate result files on disk and decide
/// whether to stream content or return paths.
pub struct JobDetail {
    /// Submitted typed options, including the required native export format.
    pub options: crate::options::CommandOptions,
    /// Command whose output policy maps input names to result artifacts.
    pub command: crate::ReleasedCommand,
    /// Current lifecycle state -- the handler uses this to reject downloads for
    /// jobs that are still running.
    pub status: JobStatus,
    /// Whether the job used filesystem paths instead of staged content. When
    /// `true`, the runner still writes execution-host outputs to `output_paths`,
    /// but successful results are also mirrored into the staging dir for
    /// download/writeback.
    pub paths_mode: bool,
    /// Server-local directory containing staged result files.
    pub staging_dir: batchalign_types::paths::ServerPath,
    /// Per-file result entries for files that have reached a terminal state.
    pub results: Vec<FileResultEntry>,
    /// Current status of every file in the job, for returning alongside results.
    pub file_statuses: Vec<FileStatusEntry>,
}

// ---------------------------------------------------------------------------
// JobStore
// ---------------------------------------------------------------------------

/// In-memory job store with background execution tasks.
pub struct JobStore {
    registry: JobRegistry,
    semaphore: Semaphore,
    db: Option<Arc<JobDB>>,
    ws_tx: broadcast::Sender<WsEvent>,
    counters: OperationalCounterStore,
    config: ServerConfig,
    node_id: NodeId,
    trace_store: crate::trace_store::TraceStore,
    max_concurrent: usize,
    /// The one source of the current time for the store and its runners.
    clock: Arc<dyn crate::clock::Clock>,
}

impl JobStore {
    /// Create a `JobStore` with the given configuration, optional database,
    /// broadcast channel for WebSocket notifications, and the clock every
    /// time it records is read from.
    ///
    /// The clock is required: it is created once where the program is
    /// composed (the server, the direct host) and handed down, so there is
    /// no second clock for a store to fall back to. Tests pass a
    /// `ManualClock` where they need exact instants.
    pub fn new(
        config: ServerConfig,
        db: Option<Arc<JobDB>>,
        ws_tx: broadcast::Sender<WsEvent>,
        clock: Arc<dyn crate::clock::Clock>,
    ) -> Self {
        // `Some(n)` is an explicit operator override; `None` falls
        // through to the host-aware auto-tune. (Future host-facts
        // work will subsume `auto_max_concurrent_jobs` into
        // `EffectiveConfig::max_concurrent_jobs`.)
        let max_concurrent = match config.max_concurrent_jobs {
            Some(n) => n as usize,
            None => HostExecutionPolicy::from_server_config(&config).auto_max_concurrent_jobs(),
        };
        info!(
            max_concurrent_jobs = max_concurrent,
            config_value = ?config.max_concurrent_jobs,
            "JobStore initialized"
        );

        Self {
            registry: JobRegistry::new(),
            semaphore: Semaphore::new(max_concurrent),
            db,
            ws_tx,
            counters: OperationalCounterStore::new(),
            node_id: NodeId::from(format!("node-{}", uuid::Uuid::new_v4().simple())),
            trace_store: crate::trace_store::TraceStore::new(),
            config,
            max_concurrent,
            clock,
        }
    }

    /// Now, by the store's clock.
    pub(crate) fn now(&self) -> MachineTime {
        self.clock.now()
    }

    /// Now, by the store's clock, as the time of an event being recorded.
    /// The only production source of an [`EventTime`].
    pub(crate) fn event_time(&self) -> EventTime {
        EventTime::from_store_clock(self.clock.now())
    }

    /// Borrow the immutable server configuration owned by the store.
    pub(crate) fn config(&self) -> &ServerConfig {
        &self.config
    }

    /// Borrow the node identifier used for local queue-lease ownership.
    pub(crate) fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    /// Borrow the trace store used by trace-download routes.
    pub(crate) fn trace_store(&self) -> &crate::trace_store::TraceStore {
        &self.trace_store
    }

    /// Acquire one global job-concurrency slot from the store-owned semaphore.
    pub(crate) async fn acquire_job_slot(&self) -> Result<SemaphorePermit<'_>, AcquireError> {
        self.semaphore.acquire().await
    }
}

// ---------------------------------------------------------------------------
// Auto-tune concurrency
// ---------------------------------------------------------------------------

#[cfg(test)]
fn auto_max_concurrent_from(by_cpu: usize, by_memory: usize) -> usize {
    host_auto_max_concurrent_from(by_cpu, by_memory)
}

#[cfg(test)]
mod worker_wait_tests {
    use super::*;

    fn processing(stage: FileProgressStage, current: i64, total: i64) -> FileStatus {
        let mut file = FileStatus::new(DisplayPath::from("a.cha".to_owned()));
        file.phase = FilePhase::Processing { started_at: None };
        file.progress = FileProgress {
            stage: Some(stage),
            current: Some(current),
            total: Some(total),
        };
        file
    }

    fn shown(file: &FileStatus) -> (Option<FileProgressStage>, Option<i64>, Option<i64>) {
        let entry = file.to_entry();
        (
            entry.progress_stage,
            entry.progress_current,
            entry.progress_total,
        )
    }

    /// The reported stage and the open waits are two facts: a stage
    /// reported during a wait is kept, not shown, and shown with its counts
    /// when the last wait ends; neither publisher can overwrite the other.
    #[test]
    fn waiting_is_derived_and_never_overwrites_the_reported_stage() {
        let mut file = processing(FileProgressStage::Analyzing, 3, 10);
        file.worker_waits.apply(WorkerWaitChange::Started);
        file.worker_waits.apply(WorkerWaitChange::Started);
        assert_eq!(
            shown(&file),
            (Some(FileProgressStage::WaitingForWorker), None, None)
        );

        // The batch reporter publishes counts mid-wait: kept, not shown.
        file.progress = FileProgress {
            stage: Some(FileProgressStage::Analyzing),
            current: Some(5),
            total: Some(10),
        };
        assert_eq!(
            shown(&file),
            (Some(FileProgressStage::WaitingForWorker), None, None)
        );

        file.worker_waits.apply(WorkerWaitChange::Ended);
        assert_eq!(
            shown(&file),
            (Some(FileProgressStage::WaitingForWorker), None, None),
            "one wait still open"
        );
        file.worker_waits.apply(WorkerWaitChange::Ended);
        assert_eq!(
            shown(&file),
            (Some(FileProgressStage::Analyzing), Some(5), Some(10))
        );

        file.worker_waits.apply(WorkerWaitChange::Ended);
        assert!(!file.worker_waits.any(), "an unpaired end changes nothing");
    }

    /// A finished file never shows a wait, whatever the count.
    #[test]
    fn a_terminal_file_does_not_show_a_wait() {
        let mut file = processing(FileProgressStage::Writing, 1, 1);
        file.worker_waits.apply(WorkerWaitChange::Started);
        file.phase = FilePhase::Done {
            started_at: None,
            finished_at: None,
        };
        assert_eq!(
            shown(&file),
            (Some(FileProgressStage::Writing), Some(1), Some(1))
        );
    }
}
