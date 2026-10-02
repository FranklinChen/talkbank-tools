//! API response projections and runner-facing snapshots.
//!
//! These methods convert a `Job` into the various read-only views consumed by
//! HTTP handlers (`JobInfo`, `JobListItem`) and the job runner
//! (`RunnerJobSnapshot`).  They also produce the `pending_files()` list that
//! drives file-level dispatch.

use crate::api::{FileStatusEntry, FileStatusKind, JobInfo, JobListItem, NonNegativeSeconds};

use super::Submitter;

use super::Job;
use super::types::{
    PendingJobFile, RunnerDispatchConfig, RunnerFilesystemConfig, RunnerJobIdentity,
    RunnerJobSnapshot,
};

/// The `jobs` columns a job's status owns, every one of them, NULLs included:
/// the image of the job as it IS, built only by [`Job::status_columns`].
///
/// The row writer (`JobDB::write_job_status`) takes this whole image, so a
/// write cannot keep a column from an earlier status (a requeued job does
/// not keep a failure's error) or persist a status the caller asked for
/// rather than the one the job reached.
///
/// Built only by its owners: [`Job::status_columns`] (the job as it is), the
/// stop transition ([`JobStatusColumns::stopped`]) and the database read of
/// a stored row ([`JobStatusColumns::read_stored`]). The fields are visible
/// to the job module (the live job adopts an image whole), and read
/// elsewhere through accessors, so no other code can mint an image to write.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct JobStatusColumns {
    /// The `status` column.
    pub(in crate::store::job) status: crate::api::JobStatus,
    /// The `error` column.
    pub(in crate::store::job) error: Option<String>,
    /// The `completed_at` column.
    pub(in crate::store::job) completed_at: Option<crate::api::MachineTime>,
    /// The `num_workers` column.
    pub(in crate::store::job) num_workers: Option<i64>,
    /// The `next_eligible_at` column.
    pub(in crate::store::job) next_eligible_at: Option<crate::api::MachineTime>,
}

/// How a job was stopped before it finished: the two statuses a stop
/// writes, and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stop {
    /// The server stopped (a shutdown, or a crash found at startup); the job
    /// is resumable.
    Interrupted,
    /// A user cancelled it.
    Cancelled,
}

impl Stop {
    /// The status the stop writes.
    pub(crate) fn status(self) -> crate::api::JobStatus {
        match self {
            Self::Interrupted => crate::api::JobStatus::Interrupted,
            Self::Cancelled => crate::api::JobStatus::Cancelled,
        }
    }
}

impl JobStatusColumns {
    /// A stored row's status image: the database boundary, reading the
    /// `status`, `error`, `completed_at`, `num_workers` and
    /// `next_eligible_at` columns of `row`. A status this build cannot read
    /// is refused.
    pub(crate) fn read_stored(
        job_id: &crate::api::JobId,
        row: &sqlx::sqlite::SqliteRow,
    ) -> Result<Self, crate::error::ServerError> {
        use sqlx::Row;
        let status: String = row.try_get("status")?;
        Ok(Self {
            status: status.parse().map_err(|error| {
                crate::error::ServerError::Persistence(format!(
                    "job {job_id}: status column: {error}"
                ))
            })?,
            error: row.try_get("error")?,
            completed_at: row.try_get("completed_at")?,
            num_workers: row.try_get("num_workers")?,
            next_eligible_at: row.try_get("next_eligible_at")?,
        })
    }

    /// Any image, for tests that write a row as a job would leave it.
    #[cfg(test)]
    pub(crate) fn for_test(
        status: crate::api::JobStatus,
        error: Option<String>,
        completed_at: Option<crate::api::MachineTime>,
        num_workers: Option<i64>,
        next_eligible_at: Option<crate::api::MachineTime>,
    ) -> Self {
        Self {
            status,
            error,
            completed_at,
            num_workers,
            next_eligible_at,
        }
    }

    /// The `status` column.
    pub(crate) fn status(&self) -> crate::api::JobStatus {
        self.status
    }

    /// The `error` column.
    pub(crate) fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// The `completed_at` column.
    pub(crate) fn completed_at(&self) -> Option<crate::api::MachineTime> {
        self.completed_at
    }

    /// The `num_workers` column.
    pub(crate) fn num_workers(&self) -> Option<i64> {
        self.num_workers
    }

    /// The `next_eligible_at` column.
    pub(crate) fn next_eligible_at(&self) -> Option<crate::api::MachineTime> {
        self.next_eligible_at
    }

    /// This image after `stop` at `at`: the stop's status, completed at `at`,
    /// no pending retry time; the error and the worker count are kept. The
    /// one statement of the transition: a live job applies it
    /// ([`Job::stop`]) and startup recovery writes it for each job a crash
    /// left queued or running.
    pub(crate) fn stopped(self, stop: Stop, at: crate::api::MachineTime) -> Self {
        Self {
            status: stop.status(),
            completed_at: Some(at),
            next_eligible_at: None,
            ..self
        }
    }
}

impl Job {
    /// The job's status columns as they are now.
    pub(crate) fn status_columns(&self) -> JobStatusColumns {
        JobStatusColumns {
            status: self.execution.status,
            error: self.execution.error.clone(),
            completed_at: self.schedule.completed_at,
            num_workers: self.schedule.num_workers,
            next_eligible_at: self.schedule.next_eligible_at,
        }
    }

    /// Return the stable job identifier.
    pub fn job_id(&self) -> &crate::api::JobId {
        &self.identity.job_id
    }

    /// Return the total number of logical files in the job.
    pub fn total_files(&self) -> usize {
        self.filesystem.filenames.len()
    }

    /// How long the job took from submission to completion; absent while it
    /// has not completed. Computed here once for every projection.
    fn duration(&self) -> Option<NonNegativeSeconds> {
        self.schedule.completed_at.map(|completed_at| {
            NonNegativeSeconds::between(self.schedule.submitted_at, completed_at)
        })
    }

    /// The submitter's address for the API, absent when none was recorded.
    fn submitted_by(&self) -> Option<String> {
        self.source
            .submitter
            .as_ref()
            .map(|submitter| submitter.address().to_owned())
    }

    /// The submitter's resolved name for the API, absent when none was
    /// recorded.
    fn submitted_by_name(&self) -> Option<String> {
        self.source
            .submitter
            .as_ref()
            .and_then(Submitter::name)
            .map(str::to_owned)
    }

    /// Convert to the API `JobInfo` response.
    pub fn to_info(&self) -> JobInfo {
        let file_statuses: Vec<FileStatusEntry> = self
            .execution
            .file_statuses
            .values()
            .map(|fs| fs.to_entry())
            .collect();
        let duration_s = self.duration();

        JobInfo {
            job_id: self.identity.job_id.clone(),
            status: self.execution.status,
            command: self.dispatch.command,
            options: self.dispatch.options.clone(),
            lang: self.dispatch.lang.clone(),
            source_dir: self.source.source_dir.as_str().to_owned(),
            total_files: self.total_files() as i64,
            completed_files: self.execution.completed_files,
            current_file: None,
            error: self.execution.error.clone(),
            file_statuses,
            submitted_at: self.schedule.submitted_at,
            submitted_by: self.submitted_by(),
            submitted_by_name: self.submitted_by_name(),
            completed_at: self.schedule.completed_at,
            duration_s,
            next_eligible_at: self.schedule.next_eligible_at,
            num_workers: self.schedule.num_workers,
            active_lease: self.active_lease(),
            control_plane: None,
            execution_plan: self.execution_plan.clone(),
            last_cancelled_at: self.schedule.last_cancel.as_ref().map(|c| c.at),
            last_cancelled_source: self.schedule.last_cancel.as_ref().map(|c| c.source.clone()),
            last_cancelled_host: self
                .schedule
                .last_cancel
                .as_ref()
                .and_then(|c| c.host.clone()),
            last_cancelled_reason: self
                .schedule
                .last_cancel
                .as_ref()
                .and_then(|c| c.reason.clone()),
        }
    }

    /// Convert to the API `JobListItem` summary.
    pub fn to_list_item(&self) -> JobListItem {
        let error_files = self
            .execution
            .file_statuses
            .values()
            .filter(|fs| fs.status() == FileStatusKind::Error)
            .count() as i64;
        let duration_s = self.duration();

        JobListItem {
            job_id: self.identity.job_id.clone(),
            status: self.execution.status,
            command: self.dispatch.command,
            lang: self.dispatch.lang.clone(),
            source_dir: self.source.source_dir.as_str().to_owned(),
            total_files: self.total_files() as i64,
            completed_files: self.execution.completed_files,
            error_files,
            error: self.execution.error.clone(),
            submitted_at: self.schedule.submitted_at,
            submitted_by: self.submitted_by(),
            submitted_by_name: self.submitted_by_name(),
            completed_at: self.schedule.completed_at,
            duration_s,
            next_eligible_at: self.schedule.next_eligible_at,
            num_workers: self.schedule.num_workers,
            active_lease: self.active_lease(),
            control_plane: None,
        }
    }

    /// Return the files that have not yet reached a terminal state.
    pub fn pending_files(&self) -> Vec<PendingJobFile> {
        self.filesystem
            .filenames
            .iter()
            .enumerate()
            .zip(self.filesystem.has_chat.iter().copied())
            .filter_map(|((file_index, filename), has_chat)| {
                let already_done = self
                    .execution
                    .file_statuses
                    .get(&**filename)
                    .is_some_and(|status| status.status().is_terminal());
                if already_done {
                    None
                } else {
                    Some(PendingJobFile {
                        file_index,
                        filename: filename.clone(),
                        has_chat,
                    })
                }
            })
            .collect()
    }

    /// Create the immutable runner-facing snapshot for this job.
    pub fn to_runner_snapshot(&self) -> RunnerJobSnapshot {
        RunnerJobSnapshot {
            identity: RunnerJobIdentity {
                job_id: self.identity.job_id.clone(),
                correlation_id: self.identity.correlation_id.clone(),
            },
            dispatch: RunnerDispatchConfig {
                command: self.dispatch.command,
                lang: self.dispatch.lang.clone(),
                num_speakers: self.dispatch.num_speakers,
                options: self.dispatch.options.clone(),
                runtime_state: self.dispatch.runtime_state.clone(),
                debug_traces: self.dispatch.debug_traces,
            },
            filesystem: RunnerFilesystemConfig {
                paths_mode: self.filesystem.paths_mode,
                source_paths: self.filesystem.source_paths.clone(),
                output_paths: self.filesystem.output_paths.clone(),
                before_paths: self.filesystem.before_paths.clone(),
                staging_dir: self.filesystem.staging_dir.clone(),
                media_mapping: self.filesystem.media_mapping.clone(),
                media_subdir: self.filesystem.media_subdir.clone(),
                source_dir: self.source.source_dir.clone(),
            },
            cancel_token: self.runtime.cancel_token.clone(),
            pending_files: self.pending_files(),
            run_generation: self.runtime.run_generation,
        }
    }
}
