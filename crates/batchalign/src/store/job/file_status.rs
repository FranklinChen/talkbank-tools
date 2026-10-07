//! Per-file status mutation methods.
//!
//! These methods move individual file entries between [`FilePhase`]s:
//! `Queued`, `Processing`, `RetryPending`, `Done`, `Diagnosed`, `Error`. Each builds the
//! whole phase it moves to, so no field from the previous phase can survive
//! by being forgotten. Each returns `false` if the filename is not found in
//! the job's file status map.

use crate::api::{ContentType, DisplayPath, FileProgressStage, MachineTime};
use crate::store::{CompletedFileOutput, FileFailure, FilePhase, FileProgress, FileResultEntry};

use super::Job;
use super::types::{FileCompletion, FileFailureRecord, FileProgressRecord, FileRetryRecord};

impl Job {
    /// Mark one file as actively processing.
    ///
    /// Entering processing drops any failure or retry deadline: a new attempt
    /// presents as "currently running", not "running but still errored from
    /// the last attempt".
    pub(crate) fn mark_file_processing(&mut self, filename: &str, started_at: MachineTime) -> bool {
        let Some(file_status) = self.execution.file_statuses.get_mut(filename) else {
            return false;
        };
        file_status.phase = FilePhase::Processing {
            started_at: Some(started_at),
        };
        file_status.progress = FileProgress::default();
        true
    }

    /// Mark one file as finished without failing, `Done` or `Diagnosed` as
    /// the completion says, and attach its result record if it has one.
    pub(crate) fn mark_file_done(
        &mut self,
        filename: &str,
        finished_at: MachineTime,
        completion: FileCompletion,
    ) -> bool {
        let Some(file_status) = self.execution.file_statuses.get_mut(filename) else {
            return false;
        };
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
        self.execution.completed_files += 1;
        true
    }

    /// Mark one file as terminally failed and attach an error result.
    pub(crate) fn mark_file_error(&mut self, filename: &str, failure: &FileFailureRecord) -> bool {
        let Some(file_status) = self.execution.file_statuses.get_mut(filename) else {
            return false;
        };
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
        self.execution.completed_files += 1;
        true
    }

    /// Record the start of a new file attempt: the file is processing from
    /// `started_at`, which is also what the database row says.
    pub(crate) fn start_file_attempt(&mut self, filename: &str, started_at: MachineTime) -> bool {
        let Some(file_status) = self.execution.file_statuses.get_mut(filename) else {
            return false;
        };
        file_status.phase = FilePhase::Processing {
            started_at: Some(started_at),
        };
        file_status.progress.stage = None;
        true
    }

    /// Mark one file as waiting for a retry after a transient failure. It is
    /// still in flight: no finish time, no duration.
    pub(crate) fn mark_file_retry_pending(
        &mut self,
        filename: &str,
        retry: &FileRetryRecord,
    ) -> bool {
        let Some(file_status) = self.execution.file_statuses.get_mut(filename) else {
            return false;
        };
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
        true
    }

    /// Clear transient retry state before a new attempt starts or succeeds.
    ///
    /// Retry scheduling keeps the last retryable error on the file so
    /// operators can see why the retry was queued. Once a new attempt starts,
    /// that stale error must disappear from the live file state or the
    /// dashboard/API will report a successful retry as still errored. Any
    /// other phase is left as it is.
    pub(crate) fn clear_file_retry_state(&mut self, filename: &str) -> bool {
        let Some(file_status) = self.execution.file_statuses.get_mut(filename) else {
            return false;
        };
        if let FilePhase::RetryPending { started_at, .. } = file_status.phase {
            file_status.phase = FilePhase::Processing { started_at };
        }
        file_status.progress.stage = None;
        true
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
    ) -> bool {
        let Some(file_status) = self.execution.file_statuses.get_mut(filename) else {
            return false;
        };
        if file_status.status().is_terminal() {
            return false;
        }
        file_status.progress = FileProgress {
            stage: Some(progress.stage),
            current: progress.current,
            total: progress.total,
        };
        true
    }

    /// Apply a change in how many of one file's checkouts wait on a
    /// saturated pool. Applied whatever the file's phase, so the count stays
    /// paired; a terminal file does not show it.
    pub(crate) fn set_file_worker_wait(
        &mut self,
        filename: &str,
        change: crate::store::WorkerWaitChange,
    ) -> bool {
        let Some(file_status) = self.execution.file_statuses.get_mut(filename) else {
            return false;
        };
        file_status.worker_waits.apply(change);
        true
    }

    /// Return the filenames of files that have not yet reached a terminal state.
    pub(crate) fn unfinished_files(&self) -> Vec<DisplayPath> {
        self.execution
            .file_statuses
            .values()
            .filter(|file_status| !file_status.status().is_terminal())
            .map(|file_status| file_status.filename.clone())
            .collect()
    }

    /// Return the current lifecycle label for one file.
    pub(crate) fn file_status_label(&self, filename: &str) -> Option<String> {
        self.execution
            .file_statuses
            .get(filename)
            .map(|file_status| file_status.status().to_string())
    }
}
