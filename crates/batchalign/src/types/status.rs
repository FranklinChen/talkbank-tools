//! Job and file lifecycle status enums.
//!
//! These are re-exported from [`super::api`] for backward compatibility.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Job status
// ---------------------------------------------------------------------------

/// Lifecycle states for a job, mirrors `JobStatus` enum.
///
/// Jobs progress through a linear lifecycle: `Queued -> Running -> {terminal}`.
/// Terminal states (`Completed`, `Failed`, `Cancelled`, `Interrupted`) are
/// permanent and never transition further.  Only `Cancelled` and `Failed` jobs
/// can be restarted (which resets them to `Queued`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    /// Job is waiting for a worker to become available.  Transitions to
    /// `Running` once a worker is checked out from the pool.
    Queued,
    /// A worker is actively processing the job's files.  Transitions to a
    /// terminal state when all files finish, or when the job is cancelled.
    Running,
    /// Every file was processed and written, none failed: each is done
    /// (clean) or diagnosed (written with diagnostics, not certified
    /// complete; counted in `diagnosed_files`). A completed job with
    /// diagnosed files is not a clean success.  Terminal.
    Completed,
    /// One or more files encountered an unrecoverable error.  Terminal, but
    /// can be restarted, only files that did not complete will be re-queued.
    Failed,
    /// A user explicitly cancelled the job via `DELETE /jobs/{id}`.  Terminal,
    /// but can be restarted.
    Cancelled,
    /// The server shut down (or crashed) while the job was active.  Terminal.
    /// Detected during crash recovery when the SQLite journal contains
    /// non-terminal jobs from a previous run.
    Interrupted,
    /// Remote execution completed successfully but copying results back to
    /// the local output paths failed.  Terminal.  The remote scratch directory
    /// on the execution host still contains the results.
    #[serde(rename = "writeback_failed")]
    WritebackFailed,
}

impl JobStatus {
    /// A terminal job will never change status again.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed
                | Self::Failed
                | Self::Cancelled
                | Self::Interrupted
                | Self::WritebackFailed
        )
    }

    /// An active job is either waiting or currently running.
    pub fn is_active(self) -> bool {
        matches!(self, Self::Queued | Self::Running)
    }

    /// A job can be cancelled if it is still active.
    pub fn can_cancel(self) -> bool {
        self.is_active()
    }

    /// Whether startup recovery should reconcile a job left in this state.
    ///
    /// `Running` counts as well as `Interrupted`: the state diagram's
    /// `Running -> Interrupted -> Queued` is the CLEAN shutdown path, where
    /// `interrupt_for_shutdown` gets to run. A hard crash never reaches it, so
    /// the row is still `Running` when recovery finds it. Shared with the
    /// recovery scan so the two cannot drift.
    pub fn is_recoverable(self) -> bool {
        matches!(self, Self::Running | Self::Interrupted)
    }

    /// Only cancelled, failed, or writeback-failed jobs can be restarted.
    pub fn can_restart(self) -> bool {
        matches!(self, Self::Cancelled | Self::Failed | Self::WritebackFailed)
    }

    /// Whether this job may move to `next` for the given reason.
    ///
    /// The authority for the machine drawn in
    /// `book/src/batchalign/architecture/job-state-machine.md`. Every status
    /// write goes through `Job::set_status`, which consults this; before
    /// 2026-07-29 legality lived in whichever caller happened to check, and
    /// four writers checked nothing.
    ///
    /// The reason is part of the question, not decoration: `Cancelled ->
    /// Queued` is legal as an operator [`StatusChange::Restart`] and illegal
    /// as a [`StatusChange::MemoryRequeue`], and a plain from/to table cannot
    /// tell those apart. Conflating them is what let the memory gate
    /// resurrect a cancelled job.
    pub fn can_transition_to(self, next: Self, reason: StatusChange) -> bool {
        match reason {
            // A worker picked the job up. Only from the queue.
            StatusChange::Dispatch => self == Self::Queued && next == Self::Running,
            // The memory gate rejected an ACTIVE job; park it. A job that has
            // left the active set (cancelled while waiting) stays gone.
            StatusChange::MemoryRequeue => self.is_active() && next == Self::Queued,
            // User cancel, or shutdown interrupting whatever was live.
            //
            // Deliberately narrower than `Finalize`, which would also permit
            // these: keeping it separate means a cancel path can never write
            // `Completed` or `Failed` by passing the wrong target.
            StatusChange::Interrupt => {
                self.can_cancel() && matches!(next, Self::Cancelled | Self::Interrupted)
            }
            // The job reached a terminal state under its own power.
            StatusChange::Finalize => self.is_active() && next.is_terminal(),
            // Operator restart, only from the restartable terminals.
            StatusChange::Restart => self.can_restart() && next == Self::Queued,
            // Startup reconciliation of a job left mid-flight; see
            // [`JobStatus::is_recoverable`].
            StatusChange::Reconcile => {
                self.is_recoverable()
                    && matches!(next, Self::Queued | Self::Failed | Self::Completed)
            }
        }
    }
}

/// Why a job's status is changing.
///
/// Names the EVENT rather than the target state, because the same from/to pair
/// can be legal or illegal depending on what caused it (see
/// [`JobStatus::can_transition_to`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusChange {
    /// A worker was checked out and the job began running.
    Dispatch,
    /// The host memory gate refused the job; return it to the queue.
    MemoryRequeue,
    /// A user cancelled, or shutdown interrupted an active job.
    Interrupt,
    /// All file work stopped; record the outcome.
    Finalize,
    /// An operator restarted a terminal job.
    Restart,
    /// Startup recovery reconciled a job left `Interrupted` by a crash.
    Reconcile,
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Queued => write!(f, "queued"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::Interrupted => write!(f, "interrupted"),
            Self::WritebackFailed => write!(f, "writeback_failed"),
        }
    }
}

impl std::str::FromStr for JobStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "queued" => Ok(Self::Queued),
            "running" => Ok(Self::Running),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            "interrupted" => Ok(Self::Interrupted),
            "writeback_failed" => Ok(Self::WritebackFailed),
            other => Err(format!("unknown JobStatus: {other}")),
        }
    }
}

// ---------------------------------------------------------------------------
// File status
// ---------------------------------------------------------------------------

/// Per-file lifecycle states within a job.
///
/// Each file in a job tracks its own status independently.  On job restart,
/// only files in resumable states (`Queued`, `Processing`, `Interrupted`) are
/// re-queued; `Done`, `Diagnosed` and `Error` files are left as-is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "lowercase")]
pub enum FileStatusKind {
    /// File is waiting to be dispatched to a worker.
    Queued,
    /// File is currently being processed by a worker.
    Processing,
    /// File was processed successfully and its result is available.  Terminal.
    Done,
    /// The command's own producer generated output that did not pass output
    /// admission, and the output was written anyway together with every
    /// diagnostic the admission found (see `FileStatusEntry.diagnostics`).
    /// Terminal, not resumable, and not an error: the result is available,
    /// and the job completes. Only generating producers (transcription) can
    /// reach it; a command that transforms submitted CHAT is held to the
    /// strict gate and fails instead.
    Diagnosed,
    /// Processing failed for this file (see `FileStatusEntry.error`).  Terminal.
    Error,
    /// The job was interrupted (server shutdown/crash) while this file was
    /// in-flight.  Resumable on job restart.
    Interrupted,
}

impl FileStatusKind {
    /// A terminal file will not be processed further.
    pub fn is_terminal(self) -> bool {
        match self {
            Self::Done | Self::Diagnosed | Self::Error => true,
            Self::Queued | Self::Processing | Self::Interrupted => false,
        }
    }

    /// A resumable file can be reset to queued on job restart.
    pub fn is_resumable(self) -> bool {
        match self {
            Self::Interrupted | Self::Processing | Self::Queued => true,
            Self::Done | Self::Diagnosed | Self::Error => false,
        }
    }

    /// The file's output was written and is available as a result: a clean
    /// success, or output written with diagnostics. Neither is retried,
    /// neither is requeued by a restart, and neither counts as a failure.
    pub fn wrote_output(self) -> bool {
        match self {
            Self::Done | Self::Diagnosed => true,
            Self::Queued | Self::Processing | Self::Error | Self::Interrupted => false,
        }
    }
}

impl std::fmt::Display for FileStatusKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Queued => write!(f, "queued"),
            Self::Processing => write!(f, "processing"),
            Self::Done => write!(f, "done"),
            Self::Diagnosed => write!(f, "diagnosed"),
            Self::Error => write!(f, "error"),
            Self::Interrupted => write!(f, "interrupted"),
        }
    }
}

impl std::str::FromStr for FileStatusKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "queued" => Ok(Self::Queued),
            "processing" => Ok(Self::Processing),
            "done" => Ok(Self::Done),
            "diagnosed" => Ok(Self::Diagnosed),
            "error" => Ok(Self::Error),
            "interrupted" => Ok(Self::Interrupted),
            other => Err(format!("unknown FileStatusKind: {other}")),
        }
    }
}

// ---------------------------------------------------------------------------
// File progress stage
// ---------------------------------------------------------------------------

/// Machine-readable sub-stage within a file's processing lifecycle.
///
/// This is more specific than [`FileStatusKind`]. For example, a file may be
/// in the coarse `Processing` status while its current progress stage is
/// `Aligning`, `BuildingChat`, or `RetryScheduled`.
///
/// The API exposes this enum so clients can branch on stable stage codes
/// without parsing human-facing display labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum FileProgressStage {
    /// Generic processing fallback for commands without a narrower stage.
    Processing,
    /// Initial file read/setup work.
    Reading,
    /// Media discovery or normalization.
    ResolvingAudio,
    /// Utterance-timing recovery before main alignment.
    RecoveringUtteranceTiming,
    /// Fallback timing recovery after an FA error.
    RecoveringTimingFallback,
    /// Main forced-alignment work.
    Aligning,
    /// Main ASR transcription work.
    Transcribing,
    /// Benchmarking pipeline execution.
    Benchmarking,
    /// Cache inspection before a compute-heavy stage.
    CheckingCache,
    /// Applying computed results back into the document.
    ApplyingResults,
    /// Model-output cleanup before downstream document construction.
    PostProcessing,
    /// Building a CHAT document from intermediate utterance state.
    BuildingChat,
    /// Utterance segmentation in the transcribe pipeline.
    SegmentingUtterances,
    /// Morphosyntax enrichment in the transcribe pipeline.
    AnalyzingMorphosyntax,
    /// Final pipeline serialization/finalization.
    Finalizing,
    /// Final write-to-disk or write-to-result stage.
    Writing,
    /// Parsing CHAT files before dispatch.
    Parsing,
    /// Batched morphosyntax analysis.
    Analyzing,
    /// Batched utterance segmentation.
    Segmenting,
    /// Batched translation.
    Translating,
    /// Batched coreference resolution.
    ResolvingCoreference,
    /// Batched transcript/reference comparison.
    Comparing,
    /// File is deferred for a retry after a retryable failure.
    RetryScheduled,
    /// A request of the file is queued for a worker: the pool is saturated
    /// (every slot busy, nothing idle to evict). Congestion, not failure; the
    /// file keeps its place and resumes its previous stage when served.
    WaitingForWorker,
}

impl FileProgressStage {
    /// Return the operator-facing label for this stable stage code.
    ///
    /// The label is intentionally derived from the enum rather than stored as
    /// an independent source of truth. That keeps control-plane logic typed
    /// while still letting UIs render human-friendly text.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Processing => "Processing",
            Self::Reading => "Reading",
            Self::ResolvingAudio => "Resolving audio",
            Self::RecoveringUtteranceTiming => "Recovering utterance timing",
            Self::RecoveringTimingFallback => "Recovering timing (fallback)",
            Self::Aligning => "Aligning",
            Self::Transcribing => "Transcribing",
            Self::Benchmarking => "Benchmarking",
            Self::CheckingCache => "Checking cache",
            Self::ApplyingResults => "Applying results",
            Self::PostProcessing => "Post-processing",
            Self::BuildingChat => "Building CHAT",
            Self::SegmentingUtterances => "Segmenting utterances",
            Self::AnalyzingMorphosyntax => "Analyzing morphosyntax",
            Self::Finalizing => "Finalizing",
            Self::Writing => "Writing",
            Self::Parsing => "Parsing",
            Self::Analyzing => "Analyzing",
            Self::Segmenting => "Segmenting",
            Self::Translating => "Translating",
            Self::ResolvingCoreference => "Resolving coreference",
            Self::Comparing => "Comparing",
            Self::RetryScheduled => "Retry scheduled",
            Self::WaitingForWorker => "Waiting for a worker",
        }
    }
}

#[cfg(test)]
mod transition_tests {
    use super::{JobStatus, StatusChange};

    /// The pair that forced the table to be keyed by cause rather than by
    /// from/to alone. Both are `Cancelled -> Queued`.
    #[test]
    fn cancelled_to_queued_is_legal_only_as_a_restart() {
        assert!(
            JobStatus::Cancelled.can_transition_to(JobStatus::Queued, StatusChange::Restart),
            "an operator may restart a cancelled job"
        );
        assert!(
            !JobStatus::Cancelled.can_transition_to(JobStatus::Queued, StatusChange::MemoryRequeue),
            "the memory gate must never return a cancelled job to the active set"
        );
    }

    /// The originally-reported bug: a job cancelled while queued, then picked
    /// up by a runner.
    #[test]
    fn a_cancelled_job_cannot_be_dispatched() {
        assert!(
            !JobStatus::Cancelled.can_transition_to(JobStatus::Running, StatusChange::Dispatch)
        );
        assert!(JobStatus::Queued.can_transition_to(JobStatus::Running, StatusChange::Dispatch));
        assert!(
            !JobStatus::Running.can_transition_to(JobStatus::Running, StatusChange::Dispatch),
            "dispatch starts from the queue, not from Running"
        );
    }

    /// A late dispatch error must not overwrite the user's outcome.
    #[test]
    fn a_terminal_job_cannot_be_failed_again() {
        assert!(!JobStatus::Cancelled.can_transition_to(JobStatus::Failed, StatusChange::Finalize));
        assert!(!JobStatus::Completed.can_transition_to(JobStatus::Failed, StatusChange::Finalize));
        assert!(JobStatus::Running.can_transition_to(JobStatus::Failed, StatusChange::Finalize));
    }

    /// A finished job is not restartable. The positive cases are covered by
    /// `can_restart` itself and by `cancelled_to_queued_is_legal_only_as_a_restart`.
    #[test]
    fn completed_jobs_are_not_restartable() {
        assert!(!JobStatus::Completed.can_transition_to(JobStatus::Queued, StatusChange::Restart));
    }

    /// Recovery reconciles a hard crash (still `Running`) as well as a clean
    /// shutdown (`Interrupted`).
    #[test]
    fn reconcile_accepts_both_crash_and_clean_shutdown() {
        for from in [JobStatus::Running, JobStatus::Interrupted] {
            assert!(from.can_transition_to(JobStatus::Queued, StatusChange::Reconcile));
        }
        assert!(
            !JobStatus::Completed.can_transition_to(JobStatus::Queued, StatusChange::Reconcile),
            "a finished job is not mid-flight and must not be reconciled"
        );
    }
}
