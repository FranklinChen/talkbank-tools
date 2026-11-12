//! Per-file status mutation methods.
//!
//! These methods move individual file entries between [`FilePhase`]s:
//! `Queued`, `Processing`, `RetryPending`, `Done`, `Diagnosed`, `Error`. Each builds the
//! whole phase it moves to, so no field from the previous phase can survive
//! by being forgotten. Each says why it refused, as a [`FileWriteRefusal`]:
//! the file is not one of the job's, or it is already terminal. Within one
//! run a finished file stays finished, and [`open_file`] is the one place
//! that rule is checked.

use crate::api::{ContentType, DisplayPath, FileProgressStage, MachineTime};
use crate::store::{CompletedFileOutput, FileFailure, FilePhase, FileProgress, FileResultEntry};

use super::Job;
use super::types::{FileCompletion, FileFailureRecord, FileProgressRecord, FileRetryRecord};
use crate::store::FileStatus;

/// Why a [`Job`] did not apply a write to one of its files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FileWriteRefusal {
    /// The job has no file of that name.
    UnknownFile,
    /// The file already finished in this run. Only a restart, which starts
    /// a new run generation, may reopen it.
    AlreadyTerminal,
}

/// The file of that name, for any write.
fn file_mut<'a>(
    statuses: &'a mut std::collections::HashMap<String, FileStatus>,
    filename: &str,
) -> Result<&'a mut FileStatus, FileWriteRefusal> {
    statuses
        .get_mut(filename)
        .ok_or(FileWriteRefusal::UnknownFile)
}

/// The file of that name, for a write that moves or redraws it: refused
/// once the file is terminal. A free function over the status map rather
/// than a method on [`Job`], so a caller can still write the job's other
/// fields (its results) while it holds the file.
fn open_file<'a>(
    statuses: &'a mut std::collections::HashMap<String, FileStatus>,
    filename: &str,
) -> Result<&'a mut FileStatus, FileWriteRefusal> {
    let file_status = file_mut(statuses, filename)?;
    if file_status.status().is_terminal() {
        return Err(FileWriteRefusal::AlreadyTerminal);
    }
    Ok(file_status)
}

impl Job {
    /// Mark one file as actively processing.
    ///
    /// Entering processing drops any failure or retry deadline: a new attempt
    /// presents as "currently running", not "running but still errored from
    /// the last attempt".
    pub(crate) fn mark_file_processing(
        &mut self,
        filename: &str,
        started_at: MachineTime,
    ) -> Result<(), FileWriteRefusal> {
        let file_status = open_file(&mut self.execution.file_statuses, filename)?;
        file_status.phase = FilePhase::Processing {
            started_at: Some(started_at),
        };
        file_status.progress = FileProgress::default();
        Ok(())
    }

    /// Mark one file as finished without failing, `Done` or `Diagnosed` as
    /// the completion says, and attach its result record if it has one.
    pub(crate) fn mark_file_done(
        &mut self,
        filename: &str,
        finished_at: MachineTime,
        completion: FileCompletion,
    ) -> Result<(), FileWriteRefusal> {
        let file_status = open_file(&mut self.execution.file_statuses, filename)?;
        let started_at = file_status.phase.started_at();
        // A written result splits in two: its exclusions go onto the phase,
        // which is what is persisted and reported; the rest is the file's
        // downloadable result and stamp.
        let written = |finished: CompletedFileOutput| {
            let CompletedFileOutput {
                filename,
                content_type,
                stamp,
                exclusions,
            } = finished;
            let entry = FileResultEntry {
                filename,
                content_type,
                error: None,
            };
            (exclusions, Some((entry, stamp)))
        };
        let (phase, result) = match completion {
            FileCompletion::WithoutResult => (
                FilePhase::Done {
                    started_at,
                    finished_at: Some(finished_at),
                    exclusions: Vec::new(),
                },
                None,
            ),
            FileCompletion::Clean(result) => {
                let (exclusions, result) = written(result);
                (
                    FilePhase::Done {
                        started_at,
                        finished_at: Some(finished_at),
                        exclusions,
                    },
                    result,
                )
            }
            FileCompletion::Diagnosed {
                result,
                diagnostics,
            } => {
                let (exclusions, result) = written(result);
                (
                    FilePhase::Diagnosed {
                        started_at,
                        finished_at: Some(finished_at),
                        diagnostics: Some(diagnostics),
                        exclusions,
                    },
                    result,
                )
            }
        };
        file_status.phase = phase;
        file_status.progress = FileProgress::default();
        if let Some((entry, stamp)) = result {
            // The stamp decision belongs to the FILE, not to one of its
            // artifacts: it says what the command recorded about this run.
            file_status.stamp = stamp;
            self.execution.results.push(entry);
        }
        Ok(())
    }

    /// Mark one file as terminally failed and attach an error result.
    pub(crate) fn mark_file_error(
        &mut self,
        filename: &str,
        failure: &FileFailureRecord,
    ) -> Result<(), FileWriteRefusal> {
        let file_status = open_file(&mut self.execution.file_statuses, filename)?;
        file_status.phase = FilePhase::Error {
            started_at: file_status.phase.started_at(),
            finished_at: Some(failure.finished_at.instant()),
            failure: FileFailure::recorded(failure.message.clone(), failure.category),
        };
        file_status.progress = FileProgress::default();
        self.execution.results.push(FileResultEntry {
            filename: DisplayPath::from(filename),
            content_type: ContentType::Chat,
            error: Some(failure.message.clone()),
        });
        Ok(())
    }

    /// Record the start of a new file attempt: the file is processing from
    /// `started_at`, which is also what the database row says.
    pub(crate) fn start_file_attempt(
        &mut self,
        filename: &str,
        started_at: MachineTime,
    ) -> Result<(), FileWriteRefusal> {
        let file_status = open_file(&mut self.execution.file_statuses, filename)?;
        file_status.phase = FilePhase::Processing {
            started_at: Some(started_at),
        };
        file_status.progress.stage = None;
        Ok(())
    }

    /// Mark one file as waiting for a retry after a transient failure. It is
    /// still in flight: no finish time, no duration.
    pub(crate) fn mark_file_retry_pending(
        &mut self,
        filename: &str,
        retry: &FileRetryRecord,
    ) -> Result<(), FileWriteRefusal> {
        let file_status = open_file(&mut self.execution.file_statuses, filename)?;
        file_status.phase = FilePhase::RetryPending {
            started_at: file_status.phase.started_at(),
            failed_at: Some(retry.finished_at.instant()),
            retry_at: retry.retry_at,
            failure: FileFailure::recorded(retry.message.clone(), retry.category),
        };
        file_status.progress = FileProgress {
            stage: Some(FileProgressStage::RetryScheduled),
            ..FileProgress::default()
        };
        Ok(())
    }

    /// Clear transient retry state before a new attempt starts or succeeds.
    ///
    /// Retry scheduling keeps the last retryable error on the file so
    /// operators can see why the retry was queued. Once a new attempt starts,
    /// that stale error must disappear from the live file state or the
    /// dashboard/API will report a successful retry as still errored. Any
    /// other phase is left as it is.
    pub(crate) fn clear_file_retry_state(
        &mut self,
        filename: &str,
    ) -> Result<(), FileWriteRefusal> {
        let file_status = file_mut(&mut self.execution.file_statuses, filename)?;
        if let FilePhase::RetryPending { started_at, .. } = file_status.phase {
            file_status.phase = FilePhase::Processing { started_at };
        }
        file_status.progress.stage = None;
        Ok(())
    }

    /// Apply an ephemeral progress update to one file.
    ///
    /// Refused for a file in a terminal state. Progress is display state and a
    /// finished file's row must not be dragged back to "Analyzing 1200/1800" by
    /// a late-arriving update: with batch progress republishing on a timer
    /// (`execution::morphotag::progress`), events for a file can outlive the
    /// file's completion by design, and every other caller wants the same
    /// protection against an out-of-order write.
    pub(crate) fn set_file_progress(
        &mut self,
        filename: &str,
        progress: &FileProgressRecord,
    ) -> Result<(), FileWriteRefusal> {
        let file_status = open_file(&mut self.execution.file_statuses, filename)?;
        file_status.progress = FileProgress {
            stage: Some(progress.stage),
            current: progress.current,
            total: progress.total,
        };
        Ok(())
    }

    /// Apply a change in how many of one file's checkouts wait on a
    /// saturated pool. Applied whatever the file's phase, so the count stays
    /// paired within a run; a terminal file does not show it. Across runs
    /// the store refuses a superseded run's change, and that is safe because
    /// a restart resets the count (`FileStatus::requeue`) in the same
    /// mutation that starts the new run.
    pub(crate) fn set_file_worker_wait(
        &mut self,
        filename: &str,
        change: crate::store::WorkerWaitChange,
    ) -> Result<(), FileWriteRefusal> {
        let file_status = file_mut(&mut self.execution.file_statuses, filename)?;
        file_status.worker_waits.apply(change);
        Ok(())
    }
}
