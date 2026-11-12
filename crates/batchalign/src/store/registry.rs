//! Private in-memory job-registry implementation for [`JobStore`](super::JobStore).
//!
//! The rest of the server should think in terms of store operations and job
//! transitions, not in terms of "there is a `Mutex<HashMap<...>>` somewhere."
//! This type keeps the raw collection plus lock local to the store layer and
//! exposes only projection/mutation helpers.

use std::collections::HashMap;

use batchalign_types::paths::ServerPath;

use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio::sync::oneshot;

use crate::api::{FileStatusEntry, JobId, JobInfo, JobListItem, JobStatus, MachineTime, NodeId};
use crate::error::ServerError;
use crate::scheduling::LeaseRecord;

use super::JobDetail;
use super::job::{
    FileCompletion, FileFailureRecord, FileProgressRecord, FileRetryRecord, Job, RunnerJobSnapshot,
    find_conflicts,
};

/// One claimed lease that should be mirrored into durable storage.
///
/// Currently used only by the test-only queue-claim path.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct ClaimedLeaseRecord {
    /// Job whose runner claim was acquired.
    pub(crate) job_id: JobId,
    /// Lease details that should be written to SQLite.
    pub(crate) lease: LeaseRecord,
}

/// Result of polling a queue backend for currently eligible queued jobs.
///
/// Lived in a top-level `crate::queue` module until 2026-07-30, where the
/// step-5 purity measurement reported it as a 7-line module that would fit in a
/// pure core crate. It would compile there and it does not belong there: it is a
/// `#[cfg(test)]` type whose only two users are this file and
/// `store/queries/dispatch.rs`, so its home is beside the wrapper below.
#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct QueuePoll {
    /// Job IDs that are ready to run now and have been claimed by the backend.
    pub ready_job_ids: Vec<JobId>,
    /// Earliest future eligibility timestamp among still-queued jobs.
    pub next_wake_at: Option<MachineTime>,
}

/// Result of claiming all currently runnable queued jobs.
///
/// Currently used only by the test-only queue-claim path.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct ClaimedQueuePoll {
    /// Queue wakeup information for the dispatcher loop.
    pub(crate) poll: QueuePoll,
    /// Leases that must be mirrored into SQLite after the lock is released.
    pub(crate) claimed_leases: Vec<ClaimedLeaseRecord>,
}

/// The dispatch's actual exit, distinct from the derived job status.
pub(crate) enum RunExit {
    Dispatched,
    Failed(String),
}

/// Cleanup admitted by a run claim, ready for persistence and notification.
pub(crate) struct ForcedFileUpdate {
    pub update: FileUpdateProjection,
    pub failure: FileFailureRecord,
}

/// Effects of the registry's atomic completion transition.
pub(crate) struct FinishedRunProjection {
    pub job_update: JobListItem,
    pub files: Vec<ForcedFileUpdate>,
    pub cleanup: CompletionCleanup,
}

/// Attempt bookkeeping admitted with the file cleanup, never inferred again
/// from an error message or a possibly restarted job.
pub(crate) enum CompletionCleanup {
    Cancelled,
    SystemFailures(usize),
}

/// Superseded claims cannot produce file updates or a terminal projection.
pub(crate) enum RunFinishProjection {
    Superseded,
    Finished(Box<FinishedRunProjection>),
}

/// Admitted effects retain the old owner's claim until persistence ends.
pub(crate) struct FinishingRun {
    pub claim: crate::store::RunnerClaim,
    pub projection: RunFinishProjection,
}

/// Why the store did not apply one run's write to one file. Each reason is
/// a different fact: two are expected in a run's ordinary life, two say the
/// runner and the store disagree about the job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RefusedFileWrite {
    /// The job is no longer in the store.
    NoSuchJob,
    /// The run was cancelled or restarted; the job's files now belong to
    /// the run that replaced it.
    StaleRun,
    /// The job itself refused the write.
    File(crate::store::FileWriteRefusal),
}

impl RefusedFileWrite {
    /// Record a refused write at the level its reason deserves. A stale run
    /// or an already finished file is expected (late progress after a
    /// cancel, a task still finishing a step) and is debug; a missing job
    /// or file means the runner wrote about something the store never had,
    /// and is a warning.
    pub(crate) fn log(self, run: &crate::store::RunBinding, filename: &str, write: &'static str) {
        use crate::store::FileWriteRefusal;
        match self {
            Self::StaleRun | Self::File(FileWriteRefusal::AlreadyTerminal) => tracing::debug!(
                run = %run,
                filename,
                write,
                refusal = ?self,
                "file write refused"
            ),
            Self::NoSuchJob | Self::File(FileWriteRefusal::UnknownFile) => tracing::warn!(
                run = %run,
                filename,
                write,
                refusal = ?self,
                "file write refused: the store has no such job or file"
            ),
        }
    }
}

/// One file update projected into the WebSocket/dashboard shape.
#[derive(Debug, Clone)]
pub(crate) struct FileUpdateProjection {
    /// Parent job that owns the updated file.
    pub job_id: JobId,
    /// Current API-facing file status entry.
    pub file: FileStatusEntry,
    /// The file's phase after the update: what is written to its row.
    pub phase: crate::store::FilePhase,
    /// Running total of terminal files for the parent job.
    pub completed_files: i64,
}

/// Restart result that carries both REST and WebSocket projections.
#[derive(Debug, Clone)]
pub(crate) struct JobRestartProjection {
    /// REST response payload for the restart endpoint.
    pub info: JobInfo,
    /// Summary row that should be published to WebSocket listeners.
    pub job_update: JobListItem,
}

/// One read-only registry operation sent to the owned actor task.
type InspectOp = Box<dyn FnOnce(&HashMap<JobId, Job>) + Send + 'static>;

/// One mutable registry operation sent to the owned actor task.
type MutateOp = Box<dyn FnOnce(&mut HashMap<JobId, Job>) + Send + 'static>;

/// Command delivered to the job-registry actor.
enum RegistryCommand {
    /// Run one read-only projection against the current job map.
    Inspect {
        /// Closure that owns the full read-side work plus any reply handling.
        op: InspectOp,
    },
    /// Run one in-place mutation against the current job map.
    Mutate {
        /// Closure that owns the full write-side work plus any reply handling.
        op: MutateOp,
    },
}

/// In-memory registry of all known jobs.
///
/// `JobStore` owns one registry instance for the lifetime of the server. The
/// registry intentionally does not know about SQLite, WebSockets, or route
/// semantics; it is only the in-memory ownership boundary around the current
/// set of jobs. The collection itself now lives inside one owned actor task, so
/// callers talk to a message boundary rather than directly sharing a mutex.
pub(crate) struct JobRegistry {
    commands: UnboundedSender<RegistryCommand>,
}

impl Default for JobRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl JobRegistry {
    /// Create an empty registry.
    pub(crate) fn new() -> Self {
        let (commands, receiver) = unbounded_channel();
        tokio::spawn(run_registry(receiver));
        Self { commands }
    }

    fn send(&self, command: RegistryCommand) {
        // Actor-lifetime invariant: `run_registry` is spawned at
        // `JobRegistry::new` (line 125) and runs for the lifetime of
        // the daemon process. Its `mpsc::Receiver` is held alive by
        // that spawned task; an `mpsc::Sender::send` only fails when
        // every Receiver has been dropped, which can only happen when
        // the actor task has terminated. The actor never terminates
        // voluntarily: it loops on `receiver.recv().await` until the
        // process is killed. Reaching the expect therefore means the
        // actor has crashed, which is itself a panic-level bug.
        #[allow(clippy::expect_used)]
        self.commands
            .send(command)
            .expect("job registry actor dropped unexpectedly");
    }

    /// Execute a read-only projection over the current job map.
    ///
    /// This remains available for the few bulk operations that genuinely need
    /// the entire registry at once, such as crash recovery. New store code
    /// should prefer the named methods below so the coordinator logic remains
    /// readable in terms of jobs and transitions rather than map plumbing.
    pub(crate) async fn inspect_all<R>(
        &self,
        f: impl FnOnce(&HashMap<JobId, Job>) -> R + Send + 'static,
    ) -> R
    where
        R: Send + 'static,
    {
        let (reply, receiver) = oneshot::channel();
        self.send(RegistryCommand::Inspect {
            op: Box::new(move |jobs| {
                let _ = reply.send(f(jobs));
            }),
        });
        // oneshot reply invariant: `RegistryCommand::Inspect` was just
        // delivered to the actor (above); the actor's per-command
        // handler always sends on `reply` before continuing to the
        // next command. The expect therefore covers the same crash
        // case as `send` above.
        #[allow(clippy::expect_used)]
        receiver
            .await
            .expect("job registry actor dropped before read reply")
    }

    /// Execute one in-place mutation over the current job map.
    ///
    /// Like [`JobRegistry::inspect_all`], this is the low-level escape hatch
    /// for bulk operations that truly need collection-wide ownership.
    pub(crate) async fn mutate_all<R>(
        &self,
        f: impl FnOnce(&mut HashMap<JobId, Job>) -> R + Send + 'static,
    ) -> R
    where
        R: Send + 'static,
    {
        let (reply, receiver) = oneshot::channel();
        self.send(RegistryCommand::Mutate {
            op: Box::new(move |jobs| {
                let _ = reply.send(f(jobs));
            }),
        });
        // Same actor-lifetime invariant as `inspect_all` above.
        #[allow(clippy::expect_used)]
        receiver
            .await
            .expect("job registry actor dropped before write reply")
    }

    /// Project one job by ID.
    ///
    /// Store queries use this for genuinely job-local reads while keeping the
    /// underlying map private to the registry layer.
    async fn project_job<R>(
        &self,
        job_id: JobId,
        f: impl FnOnce(&Job) -> R + Send + 'static,
    ) -> Option<R>
    where
        R: Send + 'static,
    {
        self.inspect_all(move |jobs| jobs.get(&job_id).map(f)).await
    }

    /// Mutate one job by ID.
    ///
    /// This is the per-job mutable counterpart to
    /// [`JobRegistry::project_job`]. The caller describes the transition; the
    /// registry owns the lock.
    pub(crate) async fn update_job<R>(
        &self,
        job_id: JobId,
        f: impl FnOnce(&mut Job) -> R + Send + 'static,
    ) -> Option<R>
    where
        R: Send + 'static,
    {
        self.mutate_all(move |jobs| jobs.get_mut(&job_id).map(f))
            .await
    }

    /// Remove one job from the registry and hand ownership to the caller.
    async fn remove_job<R>(
        &self,
        job_id: JobId,
        f: impl FnOnce(Job) -> R + Send + 'static,
    ) -> Option<R>
    where
        R: Send + 'static,
    {
        self.mutate_all(move |jobs| jobs.remove(&job_id).map(f))
            .await
    }

    /// Build the current file-update projection for one file inside one job.
    ///
    /// This keeps the file notification shape close to the registry so query
    /// modules can talk in terms of `FileUpdateProjection` rather than holding a
    /// borrowed `Job` just to serialize one nested field.
    fn file_update_projection(job: &Job, filename: &str) -> Option<FileUpdateProjection> {
        let status = job.execution.file_statuses.get(filename)?;
        Some(FileUpdateProjection {
            job_id: job.identity.job_id.clone(),
            file: status.to_entry(),
            phase: status.phase.clone(),
            completed_files: job.execution.completed_files(),
        })
    }

    /// Insert a newly submitted job after conflict detection.
    pub(crate) async fn insert_checked(&self, job: Job) -> Result<(), ServerError> {
        let job_id = job.identity.job_id.clone();
        self.mutate_all(move |jobs| -> Result<(), ServerError> {
            let conflicts = find_conflicts(jobs, &job);
            if !conflicts.is_empty() {
                let filenames: Vec<&str> = conflicts
                    .iter()
                    .map(|conflict| conflict.filename.as_ref())
                    .collect();
                let message =
                    format!("Files already being processed by an active job: {filenames:?}");
                let details = conflicts
                    .into_iter()
                    .map(|conflict| crate::error::ConflictDetail {
                        filename: conflict.filename,
                        job_id: conflict.job_id,
                        command: conflict.command,
                        status: conflict.status,
                    })
                    .collect();
                return Err(ServerError::JobConflict {
                    message,
                    conflicts: details,
                });
            }

            jobs.insert(job_id, job);
            Ok(())
        })
        .await
    }

    /// Project one job into the API response type used by `GET /jobs/:id`.
    pub(crate) async fn job_info(&self, job_id: &JobId) -> Option<JobInfo> {
        self.project_job(job_id.clone(), |job| job.to_info()).await
    }

    /// Return all jobs in newest-first order for dashboard and CLI listing.
    pub(crate) async fn list_items(&self) -> Vec<JobListItem> {
        self.inspect_all(|jobs| {
            let mut items: Vec<JobListItem> = jobs.values().map(|job| job.to_list_item()).collect();
            // Newest first, comparing instants (a job whose time names no
            // instant sorts last), not their spellings.
            items.sort_by_key(|item| std::cmp::Reverse(item.submitted_at));
            items
        })
        .await
    }

    /// Request cancellation for one job and return any terminal timestamp that
    /// should be mirrored into SQLite.
    pub(crate) async fn request_cancellation(
        &self,
        job_id: &JobId,
        cancelled_at: MachineTime,
    ) -> Option<Option<MachineTime>> {
        self.update_job(job_id.clone(), move |job| {
            job.request_cancellation(cancelled_at)
        })
        .await
    }

    /// Project the most-recent cancel attempt's metadata onto the in-memory
    /// `Job` so `JobInfo::last_cancelled_*` reflects it without a DB round
    /// trip. Called by the store cancel pathway right after the audit-table
    /// row is persisted. Returns `true` if the job was found, `false` if it
    /// was already evicted.
    pub(crate) async fn set_last_cancel(
        &self,
        job_id: &JobId,
        info: crate::store::JobLastCancelInfo,
    ) -> bool {
        self.update_job(job_id.clone(), move |job| {
            job.schedule.last_cancel = Some(info);
        })
        .await
        .is_some()
    }

    /// Remove one job and return its staging directory for cleanup.
    pub(crate) async fn remove_staging_dir(&self, job_id: &JobId) -> Option<ServerPath> {
        self.remove_job(job_id.clone(), |job| job.filesystem.staging_dir)
            .await
    }

    /// Reset a failed or cancelled job back to queued state.
    ///
    /// The returned projection carries both the REST payload for the restart
    /// endpoint and the summary row that should be broadcast to live clients.
    pub(crate) async fn restart_job(
        &self,
        job_id: &JobId,
        authority: &batchalign_types::paths::AllowedRoots,
    ) -> Option<Result<JobRestartProjection, ServerError>> {
        let authority = authority.clone();
        self.update_job(
            job_id.clone(),
            move |job| -> Result<JobRestartProjection, ServerError> {
                job.prepare_for_restart(&authority)?;
                Ok(JobRestartProjection {
                    info: job.to_info(),
                    job_update: job.to_list_item(),
                })
            },
        )
        .await
    }

    /// One job's status columns as they are now: what the database row for
    /// the job's status is written from.
    pub(crate) async fn status_columns(
        &self,
        job_id: &JobId,
    ) -> Option<crate::store::JobStatusColumns> {
        self.project_job(job_id.clone(), |job| job.status_columns())
            .await
    }

    /// Return whether one job is currently running.
    pub(crate) async fn is_running(&self, job_id: &JobId) -> Option<bool> {
        self.project_job(job_id.clone(), |job| job.status() == JobStatus::Running)
            .await
    }

    /// Return whether the local dispatcher currently holds an active runner claim.
    #[cfg(test)]
    pub(crate) async fn runner_claim_active(&self, job_id: &JobId) -> Option<bool> {
        self.project_job(job_id.clone(), |job| job.runtime.runner_active)
            .await
    }

    /// Return a cloned snapshot of the current lease state for one job.
    #[cfg(test)]
    pub(crate) async fn lease_state(
        &self,
        job_id: &JobId,
    ) -> Option<Option<crate::scheduling::LeaseRecord>> {
        self.project_job(job_id.clone(), |job| job.schedule.lease.clone())
            .await
    }

    /// Return the current job status for one job.
    pub(crate) async fn job_status(&self, job_id: &JobId) -> Option<JobStatus> {
        self.project_job(job_id.clone(), |job| job.status()).await
    }

    /// Count how many jobs are currently running.
    pub(crate) async fn active_jobs(&self) -> i64 {
        self.inspect_all(|jobs| {
            jobs.values()
                .filter(|job| job.status() == JobStatus::Running)
                .count() as i64
        })
        .await
    }

    /// Re-queue one job after a memory-gate rejection and return the new
    /// summary-row projection for WebSocket listeners.
    ///
    /// `None` means no such job OR the job refused to be requeued because it
    /// is no longer active; callers must then persist nothing.
    pub(crate) async fn requeue_after_memory_gate(
        &self,
        run: &crate::store::RunBinding,
        retry_at: MachineTime,
    ) -> Option<JobListItem> {
        let binding = run.clone();
        self.update_job(run.job_id().clone(), move |job| {
            (binding.is_current_for(job) && job.requeue_after_memory_gate(retry_at))
                .then(|| job.to_list_item())
        })
        .await
        .flatten()
    }

    /// Mark one queued job as running and return its updated summary row.
    ///
    /// `None` means no such job, a run that is no longer the job's current
    /// one, or a job that refused the transition because it is no longer
    /// active (see [`Job::mark_running`]); callers must treat it as "do not
    /// persist Running".
    pub(crate) async fn mark_job_running(
        &self,
        run: &crate::store::RunBinding,
    ) -> Option<JobListItem> {
        let binding = run.clone();
        self.update_job(run.job_id().clone(), move |job| {
            (binding.is_current_for(job) && job.mark_running()).then(|| job.to_list_item())
        })
        .await
        .flatten()
    }

    /// Record the runner's worker-count choice for its run of one job;
    /// `false` when there is no such job or the run is no longer current.
    pub(crate) async fn record_job_worker_count(
        &self,
        run: &crate::store::RunBinding,
        num_workers: usize,
    ) -> bool {
        let binding = run.clone();
        self.update_job(run.job_id().clone(), move |job| {
            let current = binding.is_current_for(job);
            if current {
                job.record_worker_count(num_workers);
            }
            current
        })
        .await
        .unwrap_or(false)
    }

    /// Consume a run's authority in one actor mutation. Generation admission,
    /// unfinished-file cleanup and final status have no restart interleaving.
    pub(crate) async fn complete_run(
        &self,
        claim: crate::store::RunnerClaim,
        exit: RunExit,
        finished_at: crate::store::EventTime,
    ) -> Option<FinishingRun> {
        let job_id = claim.binding().job_id().clone();
        self.update_job(job_id, move |job| {
            if !claim.binding().is_current_for(job) {
                return FinishingRun { claim, projection: RunFinishProjection::Superseded };
            }
            let cancelled = job.is_cancelled();
            let files = job.execution.file_statuses.iter()
                .filter(|(_, file)| !file.status().is_terminal())
                .map(|(filename, file)| {
                    let (message, category) = if cancelled {
                        (
                            "File stopped after job cancellation before recording a terminal file state".to_owned(),
                            crate::scheduling::FailureCategory::Cancelled,
                        )
                    } else {
                        (
                            match &exit {
                                RunExit::Dispatched => format!("File did not reach terminal status (last status: {})", file.status()),
                                RunExit::Failed(message) => message.clone(),
                            },
                            crate::scheduling::FailureCategory::System,
                        )
                    };
                    (filename.clone(), FileFailureRecord { message, category, finished_at })
                })
                .collect::<Vec<_>>();
            let mut updates = Vec::with_capacity(files.len());
            for (filename, failure) in files {
                // This filename came from this same map, unfinished, and
                // cannot change during the actor mutation; the update is
                // published only for a write that applied.
                if job.mark_file_error(&filename, &failure).is_ok()
                    && let Some(update) = Self::file_update_projection(job, &filename)
                {
                    updates.push(ForcedFileUpdate { update, failure });
                }
            }
            let outcome = if cancelled {
                crate::store::Outcome::Cancelled
            } else if job.any_terminal_files_failed() {
                crate::store::Outcome::Failed
            } else {
                crate::store::Outcome::Completed
            };
            // Already accepted cancellation/shutdown retains its status/time.
            match exit {
                RunExit::Dispatched => { let _ = job.finalize(outcome, finished_at.instant()); }
                RunExit::Failed(message) => {
                    if cancelled {
                        let _ = job.finalize(outcome, finished_at.instant());
                    } else {
                        let _ = job.fail(&message, finished_at.instant());
                    }
                }
            }
            let projection = RunFinishProjection::Finished(Box::new(FinishedRunProjection {
                job_update: job.to_list_item(),
                cleanup: if cancelled {
                    CompletionCleanup::Cancelled
                } else {
                    CompletionCleanup::SystemFailures(updates.len())
                },
                files: updates,
            }));
            FinishingRun { claim, projection }
        }).await
    }

    /// Claim exclusive runner ownership of one job.
    ///
    /// Exactly one live runner may own a job at a time; a second claim
    /// (e.g. a restarted runner arriving while the cancelled one is still
    /// tearing down) reports [`BeginRunnerOutcome::RunnerStillLive`] and
    /// must wait. The successful claim owns its generation and snapshot;
    /// completion consumes it before admitting cleanup.
    pub(crate) async fn begin_runner(
        &self,
        job_id: &JobId,
        node_id: &NodeId,
        now: MachineTime,
        lease_ttl: crate::config::LeaseTtl,
    ) -> Option<crate::store::BeginRunnerOutcome> {
        use crate::store::BeginRunnerOutcome;
        let node_id = node_id.clone();
        self.update_job(job_id.clone(), move |job| {
            if job.runtime.runner_active {
                BeginRunnerOutcome::RunnerStillLive
            } else if job.status() == JobStatus::Cancelled
                && job
                    .execution
                    .file_statuses
                    .values()
                    .any(|file| !file.status().is_terminal())
            {
                // A job whose inputs this process has not admitted (recorded
                // history) has no runner work to finish; it is not runnable.
                match crate::store::StoppedRun::for_job(job) {
                    Some(completion) => {
                        job.runtime.runner_active = true;
                        BeginRunnerOutcome::Stopped { completion }
                    }
                    None => BeginRunnerOutcome::NotRunnable {
                        status: job.status(),
                    },
                }
            } else if job.status().is_terminal() {
                BeginRunnerOutcome::NotRunnable {
                    status: job.status(),
                }
            } else {
                // Recovery and restart admit every job they make runnable, so
                // unadmitted inputs here are recorded history: not runnable.
                let Some(claim) = crate::store::RunnerClaim::for_job(job) else {
                    return BeginRunnerOutcome::NotRunnable {
                        status: job.status(),
                    };
                };
                job.runtime.runner_active = true;
                // Establish the queue lease at claim time so the runner's
                // heartbeat loop has something to renew; before this,
                // production runners never held a lease and the heartbeat
                // loop silently stopped on its first tick.
                let lease = crate::scheduling::LeaseRecord::taken(node_id.clone(), now, lease_ttl);
                job.schedule.lease = Some(lease.clone());
                BeginRunnerOutcome::Started { claim, lease }
            }
        })
        .await
    }

    /// Return the most recent file-activity timestamp for one job: the
    /// latest per-file attempt start/finish, falling back to submission
    /// time when no file has moved yet. Drives the runner's stall alarm.
    pub(crate) async fn last_file_activity_at(&self, job_id: &JobId) -> Option<MachineTime> {
        self.project_job(job_id.clone(), |job| {
            job.execution
                .file_statuses
                .values()
                .filter_map(|file_status| file_status.phase.last_activity_at())
                .chain(std::iter::once(job.schedule.submitted_at))
                .max()
        })
        .await
        .flatten()
    }

    /// Project the download-facing detail view for one job.
    pub(crate) async fn job_detail(&self, job_id: &JobId) -> Option<JobDetail> {
        self.project_job(job_id.clone(), |job| {
            use crate::store::{JobInputs, StoredInputsRef};
            // The outputs the client named, whether this process admitted
            // them (a runnable job) or holds them as recorded history.
            let outputs: Option<Vec<batchalign_types::paths::ClientPath>> = match job.inputs() {
                StoredInputsRef::Admitted(JobInputs::Paths { files, .. }) => Some(
                    files
                        .iter()
                        .map(|file| file.output.client().clone())
                        .collect(),
                ),
                StoredInputsRef::Recorded(recorded) => Some(
                    recorded
                        .files()
                        .iter()
                        .map(|file| file.output.clone())
                        .collect(),
                ),
                StoredInputsRef::Admitted(JobInputs::Content(_)) => None,
            };
            JobDetail {
                output_folders: crate::api::JobOutputFolders {
                    job_id: job.identity.job_id.clone(),
                    access: crate::api::LocalOutputAccess::admit(match &outputs {
                        Some(outputs) => crate::types::output_access::OutputLocationSource::Paths {
                            count: outputs.len(),
                            outputs,
                        },
                        None => crate::types::output_access::OutputLocationSource::Content,
                    }),
                },
                options: job.dispatch.options.clone(),
                command: job.dispatch.command,
                status: job.status(),
                staging_dir: job.filesystem.staging_dir.clone(),
                results: job.execution.results.clone(),
                file_statuses: job
                    .execution
                    .file_statuses
                    .values()
                    .map(|file_status| file_status.to_entry())
                    .collect(),
            }
        })
        .await
    }

    /// Interrupt every active job for graceful server shutdown, returning how many
    /// were updated.
    ///
    /// Uses `Job::interrupt_for_shutdown` so jobs land in `JobStatus::Interrupted`
    /// (resumable via the startup recovery path) rather than `JobStatus::Cancelled`
    /// (terminal user-cancel state).  See `Job::interrupt_for_shutdown` doc for the
    /// full recovery-flow rationale.
    pub(crate) async fn interrupt_all_active(&self, interrupted_at: MachineTime) -> usize {
        self.mutate_all(move |jobs| {
            let mut count = 0;
            for job in jobs.values_mut() {
                if job.interrupt_for_shutdown(interrupted_at) {
                    count += 1;
                }
            }
            count
        })
        .await
    }

    /// Return IDs of jobs that remain queued after startup recovery.
    pub(crate) async fn queued_job_ids(&self) -> Vec<JobId> {
        self.inspect_all(|jobs| {
            jobs.values()
                .filter(|job| job.status() == JobStatus::Queued)
                .map(|job| job.identity.job_id.clone())
                .collect()
        })
        .await
    }

    /// Claim every queued job that is eligible to run on this node.
    ///
    /// Currently exercised only by the test-only local queue-claim path.
    #[cfg(test)]
    pub(crate) async fn claim_ready_queued_jobs(
        &self,
        now: MachineTime,
        node_id: &NodeId,
        lease_ttl: crate::config::LeaseTtl,
    ) -> ClaimedQueuePoll {
        let node_id = node_id.clone();
        self.mutate_all(move |jobs| {
            let mut ready: Vec<(MachineTime, JobId)> = jobs
                .values()
                .filter(|job| job.ready_for_local_dispatch(now))
                .map(|job| (job.schedule.submitted_at, job.identity.job_id.clone()))
                .collect();
            ready.sort_by_key(|(submitted_at, _)| *submitted_at);

            let ready_job_ids: Vec<JobId> = ready.into_iter().map(|(_, job_id)| job_id).collect();
            let mut claimed_leases: Vec<ClaimedLeaseRecord> =
                Vec::with_capacity(ready_job_ids.len());
            for job_id in &ready_job_ids {
                if let Some(job) = jobs.get_mut(job_id)
                    && let Some(lease) = job.claim_for_local_dispatch(&node_id, now, lease_ttl)
                {
                    claimed_leases.push(ClaimedLeaseRecord {
                        job_id: job_id.clone(),
                        lease,
                    });
                }
            }

            let next_wake_at = jobs
                .values()
                .filter(|job| job.status() == JobStatus::Queued && !job.runtime.runner_active)
                .filter_map(|job| job.next_local_dispatch_wake_at(now))
                .min();

            ClaimedQueuePoll {
                poll: QueuePoll {
                    ready_job_ids,
                    next_wake_at,
                },
                claimed_leases,
            }
        })
        .await
    }

    /// Deferral consumes the same exclusive owner as completion/failure.
    pub(crate) async fn release_run(&self, claim: crate::store::RunnerClaim) -> bool {
        let job_id = claim.snapshot().identity.job_id.clone();
        self.update_job(job_id, |job| job.release_local_dispatch_claim())
            .await
            .is_some()
    }

    /// Queue-lease controls exercise the legacy test-only claim operation.
    #[cfg(test)]
    pub(crate) async fn release_runner_claim(&self, job_id: &JobId) -> bool {
        self.update_job(job_id.clone(), |job| job.release_local_dispatch_claim())
            .await
            .is_some()
    }

    /// Renew the local queue lease for one claimed job.
    pub(crate) async fn renew_job_lease(
        &self,
        job_id: &JobId,
        node_id: &NodeId,
        now: MachineTime,
        lease_ttl: crate::config::LeaseTtl,
    ) -> Option<LeaseRecord> {
        let node_id = node_id.clone();
        self.update_job(job_id.clone(), move |job| {
            job.renew_local_dispatch_lease(&node_id, now, lease_ttl)
        })
        .await
        .flatten()
    }

    /// Return the immutable runner-facing projection for one job.
    pub(crate) async fn runner_snapshot(&self, job_id: &JobId) -> Option<RunnerJobSnapshot> {
        self.project_job(job_id.clone(), |job| job.to_runner_snapshot())
            .await
            .flatten()
    }

    /// One file write from one run, in one actor mutation, admitted only
    /// while `run` is still the job's current generation and the job's
    /// transition accepts it. Every per-file write goes through here, so a
    /// task of a cancelled or restarted run cannot write onto the files of
    /// the run that replaced it, whatever it was doing when it was stopped;
    /// and each refusal says which of those it was.
    async fn update_file_of_run(
        &self,
        run: &crate::store::RunBinding,
        filename: &str,
        transition: impl FnOnce(&mut Job, &str) -> Result<(), crate::store::FileWriteRefusal>
        + Send
        + 'static,
    ) -> Result<FileUpdateProjection, RefusedFileWrite> {
        let filename = filename.to_string();
        let binding = run.clone();
        self.update_job(run.job_id().clone(), move |job| {
            if !binding.is_current_for(job) {
                return Err(RefusedFileWrite::StaleRun);
            }
            transition(job, &filename).map_err(RefusedFileWrite::File)?;
            // The transition just found this file in the same mutation.
            Self::file_update_projection(job, &filename).ok_or(RefusedFileWrite::File(
                crate::store::FileWriteRefusal::UnknownFile,
            ))
        })
        .await
        .unwrap_or(Err(RefusedFileWrite::NoSuchJob))
    }

    /// Mark one file as processing and return its file-update projection.
    pub(crate) async fn mark_file_processing(
        &self,
        run: &crate::store::RunBinding,
        filename: &str,
        started_at: MachineTime,
    ) -> Result<FileUpdateProjection, RefusedFileWrite> {
        self.update_file_of_run(run, filename, move |job, file| {
            job.mark_file_processing(file, started_at)
        })
        .await
    }

    /// Mark one file as complete and return the new file-update projection.
    pub(crate) async fn mark_file_done(
        &self,
        run: &crate::store::RunBinding,
        filename: &str,
        finished_at: MachineTime,
        completion: FileCompletion,
    ) -> Result<FileUpdateProjection, RefusedFileWrite> {
        self.update_file_of_run(run, filename, move |job, file| {
            job.mark_file_done(file, finished_at, completion)
        })
        .await
    }

    /// Mark one file as terminally failed and return the projected update.
    pub(crate) async fn mark_file_error(
        &self,
        run: &crate::store::RunBinding,
        filename: &str,
        failure: &FileFailureRecord,
    ) -> Result<FileUpdateProjection, RefusedFileWrite> {
        let failure = failure.clone();
        self.update_file_of_run(run, filename, move |job, file| {
            job.mark_file_error(file, &failure)
        })
        .await
    }

    /// Record the start of one file attempt and return the projected update.
    pub(crate) async fn start_file_attempt(
        &self,
        run: &crate::store::RunBinding,
        filename: &str,
        started_at: MachineTime,
    ) -> Result<FileUpdateProjection, RefusedFileWrite> {
        self.update_file_of_run(run, filename, move |job, file| {
            job.start_file_attempt(file, started_at)
        })
        .await
    }

    /// Mark one file as waiting for retry and return the projected update.
    pub(crate) async fn mark_file_retry_pending(
        &self,
        run: &crate::store::RunBinding,
        filename: &str,
        retry: &FileRetryRecord,
    ) -> Result<FileUpdateProjection, RefusedFileWrite> {
        let retry = retry.clone();
        self.update_file_of_run(run, filename, move |job, file| {
            job.mark_file_retry_pending(file, &retry)
        })
        .await
    }

    /// Clear retry-only transient fields before a fresh attempt.
    pub(crate) async fn clear_file_retry_state(
        &self,
        run: &crate::store::RunBinding,
        filename: &str,
    ) -> Result<FileUpdateProjection, RefusedFileWrite> {
        self.update_file_of_run(run, filename, |job, file| job.clear_file_retry_state(file))
            .await
    }

    /// Apply one worker-wait change and return the projected file-update
    /// payload.
    pub(crate) async fn set_file_worker_wait(
        &self,
        run: &crate::store::RunBinding,
        filename: &str,
        change: crate::store::WorkerWaitChange,
    ) -> Result<FileUpdateProjection, RefusedFileWrite> {
        self.update_file_of_run(run, filename, move |job, file| {
            job.set_file_worker_wait(file, change)
        })
        .await
    }

    /// Apply one progress update and return the projected file-update payload.
    pub(crate) async fn set_file_progress(
        &self,
        run: &crate::store::RunBinding,
        filename: &str,
        progress: &FileProgressRecord,
    ) -> Result<FileUpdateProjection, RefusedFileWrite> {
        let progress = progress.clone();
        self.update_file_of_run(run, filename, move |job, file| {
            job.set_file_progress(file, &progress)
        })
        .await
    }

    /// Attach one newly created attempt identifier to a file of this run.
    pub(crate) async fn attach_attempt_id(
        &self,
        run: &crate::store::RunBinding,
        filename: &str,
        attempt_id: String,
    ) -> Result<(), RefusedFileWrite> {
        let filename = filename.to_string();
        let binding = run.clone();
        self.update_job(run.job_id().clone(), move |job| {
            if !binding.is_current_for(job) {
                return Err(RefusedFileWrite::StaleRun);
            }
            let file_status =
                job.execution
                    .file_statuses
                    .get_mut(&filename)
                    .ok_or(RefusedFileWrite::File(
                        crate::store::FileWriteRefusal::UnknownFile,
                    ))?;
            file_status.current_attempt_id = Some(attempt_id);
            Ok(())
        })
        .await
        .unwrap_or(Err(RefusedFileWrite::NoSuchJob))
    }

    /// Take and clear the active attempt id of a file of this run. A stale
    /// run gets nothing, so it cannot close the attempt of the run that
    /// replaced it.
    pub(crate) async fn take_attempt_id(
        &self,
        run: &crate::store::RunBinding,
        filename: &str,
    ) -> Option<String> {
        let filename = filename.to_string();
        let binding = run.clone();
        self.update_job(run.job_id().clone(), move |job| {
            if !binding.is_current_for(job) {
                return None;
            }
            job.execution
                .file_statuses
                .get_mut(&filename)
                .and_then(|file_status| file_status.current_attempt_id.take())
        })
        .await
        .flatten()
    }
}

/// Run the job-registry actor loop.
async fn run_registry(mut receiver: UnboundedReceiver<RegistryCommand>) {
    let mut jobs = HashMap::new();

    while let Some(command) = receiver.recv().await {
        match command {
            RegistryCommand::Inspect { op } => op(&jobs),
            RegistryCommand::Mutate { op } => op(&mut jobs),
        }
    }
}
