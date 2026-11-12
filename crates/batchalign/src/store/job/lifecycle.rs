//! Job-level lifecycle transitions.
//!
//! These methods move the job through its top-level state machine.
//! From an active state (`Queued | Running`), a job reaches one of three
//! terminal endpoints, or, on graceful shutdown, the recovery-eligible
//! `Interrupted` state:
//!
//! ```text
//! Queued → Running → { Completed | Failed | Cancelled | Interrupted }
//! ```
//!
//! `Cancelled` is reserved for user gestures (TUI cancel, HTTP DELETE/cancel)
//! and is permanent. `Interrupted` is the system-initiated counterpart written
//! by `interrupt_for_shutdown`: although `JobStatus::is_terminal()` returns
//! `true` for it, the recovery sequence at startup is special-cased to
//! transition resumable Interrupted rows back to `Queued` (see
//! `Job::recover` here and `load_from_db` in
//! `store/queries/recovery.rs`).
//!
//! Restart preparation (`prepare_for_restart`) resets unfinished file work
//! while preserving completed results, enabling partial retry after failures.

use tokio_util::sync::CancellationToken;

use std::collections::BTreeSet;

use crate::api::{FileStatusKind, JobStatus, MachineTime};

use super::Job;
use crate::store::FilePhase;

impl Job {
    /// Return whether the job's cancellation token has been triggered.
    pub(crate) fn is_cancelled(&self) -> bool {
        self.runtime.cancel_token.is_cancelled()
    }

    /// Return whether at least one terminal file currently recorded is
    /// an error.
    ///
    /// Used by registry completion after every unfinished file is closed.
    /// Any failed file makes the job failed, including partial success.
    ///
    /// Per the `JobStatus` docs at `types/status.rs`: `Completed` means
    /// "all files processed successfully", `Failed` means "one or more
    /// files encountered an unrecoverable error". This predicate is the
    /// distinguishing test between those two states.
    pub(crate) fn any_terminal_files_failed(&self) -> bool {
        self.execution
            .file_statuses
            .values()
            .filter(|file_status| file_status.status().is_terminal())
            .any(|file_status| file_status.status() == FileStatusKind::Error)
    }

    /// The job's status.
    pub(crate) fn status(&self) -> JobStatus {
        self.execution.lifecycle.status()
    }

    /// The job's inputs as stored, whichever state it is in.
    pub(crate) fn inputs(&self) -> crate::store::StoredInputsRef<'_> {
        self.execution.lifecycle.stored()
    }

    /// A queued or running job's inputs; `None` for a finished job.
    pub(crate) fn active_inputs(&self) -> Option<&crate::store::JobInputs> {
        match &self.execution.lifecycle {
            super::JobLifecycle::Active { inputs, .. } => Some(inputs),
            super::JobLifecycle::Finished { .. } => None,
        }
    }

    /// The inputs a runner may use: those this process admitted.
    pub(crate) fn admitted_inputs(&self) -> Option<&crate::store::JobInputs> {
        self.execution.lifecycle.admitted()
    }

    /// Whether a lifecycle event applied, logging a refusal: the one place
    /// a refused event is reported, so no transition has to remember. Most
    /// refusals are ordinary races (a user cancels while a runner is
    /// mid-flight); a refused `Finish` means work was completed and then
    /// discarded. A refused event changed nothing, and its caller must not
    /// persist a status either.
    fn accepted(&self, event: Result<(), super::Refused>) -> bool {
        match event {
            Ok(()) => true,
            Err(refused) => {
                tracing::debug!(job_id = %self.identity.job_id, %refused, "refusing job lifecycle event");
                false
            }
        }
    }

    /// Clear what the previous run concluded, for a job queued again: no
    /// carried error, completion time, retry time or lease.
    fn clear_run_outcome(&mut self) {
        self.execution.error = None;
        self.schedule.completed_at = None;
        self.schedule.next_eligible_at = None;
        self.clear_lease();
    }

    /// Stop the job at `at`, if its status allows: the status image's
    /// [`JobStatusColumns::stopped`] transition, applied to the live job.
    /// Returns whether it fired.
    ///
    /// [`JobStatusColumns::stopped`]: super::JobStatusColumns::stopped
    fn stop(&mut self, stop: super::Stop, at: MachineTime) -> bool {
        let event = self.execution.lifecycle.stop(stop);
        if !self.accepted(event) {
            return false;
        }
        let image = self.status_columns().stopped(stop, at);
        self.adopt_stopped_columns(image);
        true
    }

    /// Make the live job hold `image`, every column of it. Destructured
    /// whole, so a column added to the image cannot be left out here, and a
    /// transition stated on the image is the transition the live job takes.
    /// The status is the one column not written here: `stop` moved the
    /// lifecycle to `stop.status()`, which is the image's.
    fn adopt_stopped_columns(&mut self, image: super::JobStatusColumns) {
        let super::JobStatusColumns {
            status: _,
            error,
            completed_at,
            num_workers,
            next_eligible_at,
        } = image;
        self.execution.error = error;
        self.schedule.completed_at = completed_at;
        self.schedule.num_workers = num_workers;
        self.schedule.next_eligible_at = next_eligible_at;
    }

    /// Request cancellation and, when still active, transition to cancelled.
    pub(crate) fn request_cancellation(
        &mut self,
        completed_at: MachineTime,
    ) -> Option<MachineTime> {
        self.runtime.cancel_token.cancel();
        self.stop(super::Stop::Cancelled, completed_at)
            .then_some(completed_at)
    }

    /// Re-queue the job after memory pressure prevented dispatch.
    ///
    /// Refuses for a job that has left the active set: `Queued` is an ACTIVE
    /// status, so requeuing a cancelled job resurrects it. Reproduction:
    /// `cancelled_job_is_not_requeued_by_the_memory_gate`.
    #[must_use = "an ignored refusal requeues a job that is no longer active"]
    pub(crate) fn requeue_after_memory_gate(&mut self, retry_at: MachineTime) -> bool {
        let event = self.execution.lifecycle.memory_requeue();
        if !self.accepted(event) {
            return false;
        }
        self.schedule.completed_at = None;
        self.schedule.next_eligible_at = Some(retry_at);
        true
    }

    /// Mark the job as actively running, refusing if it is no longer active.
    ///
    /// Returns false when the job has already left the active set, which in
    /// practice means it was cancelled while sitting in the queue. The
    /// transition used to be unconditional, so a runner picking the job up
    /// moments after a cancel would resurrect it: `request_cancellation` set
    /// `Cancelled` and persisted it, then `mark_running` silently overwrote
    /// the in-memory status back to `Running` while the DB still read
    /// `Cancelled`.
    ///
    /// Observed 2026-07-29 in `cancel_twice_records_two_audit_rows`: cancel #1
    /// was accepted at t, the runner marked the job Running, and cancel #2 at
    /// t+5.9ms saw a live job and was accepted again, so the audit recorded two
    /// state-changing cancels instead of one accepted and one no-op. The user
    /// impact is worse than the audit row: a job the user cancelled reports
    /// itself Running.
    ///
    /// Refusing leaves the job `Cancelled`, and the database row is written
    /// from the job as it is (`Job::status_columns`), so registry and
    /// database agree. Every status change is an event on the job's
    /// [`JobLifecycle`](super::JobLifecycle), legal only from the states it
    /// leaves.
    #[must_use = "a dropped refusal silently resurrects a cancelled job"]
    pub(crate) fn mark_running(&mut self) -> bool {
        let event = self.execution.lifecycle.dispatch();
        if !self.accepted(event) {
            return false;
        }
        self.schedule.next_eligible_at = None;
        true
    }

    /// Record the per-job worker count chosen for this run.
    pub(crate) fn record_worker_count(&mut self, num_workers: usize) {
        self.schedule.num_workers = Some(num_workers as i64);
    }

    /// Fail the job immediately with a job-level error message.
    ///
    /// Refuses for a job that is no longer active, so a late dispatch error
    /// cannot overwrite a user's `Cancelled` with `Failed`.
    #[must_use = "an ignored refusal overwrites a terminal status"]
    pub(crate) fn fail(&mut self, error: &str, completed_at: MachineTime) -> bool {
        let event = self.execution.lifecycle.finish(super::Outcome::Failed);
        if !self.accepted(event) {
            return false;
        }
        self.execution.error = Some(error.to_string());
        self.schedule.completed_at = Some(completed_at);
        true
    }

    /// Close every unfinished file with `message`, for a recovered job that
    /// will not run: its inputs were refused, or it was cancelled with files
    /// still open (no runner is started for a cancelled job after a restart,
    /// and closing a file reads no path). Returns each closed file with its
    /// new phase, for the database.
    pub(crate) fn close_unfinished_files(
        &mut self,
        message: &str,
        category: crate::scheduling::FailureCategory,
        at: crate::store::EventTime,
    ) -> Vec<(String, crate::store::FilePhase)> {
        let unfinished: Vec<String> = self
            .execution
            .file_statuses
            .iter()
            .filter(|(_, file)| !file.status().is_terminal())
            .map(|(filename, _)| filename.clone())
            .collect();
        let failure = crate::store::FileFailureRecord {
            message: message.to_owned(),
            category,
            finished_at: at,
        };
        unfinished
            .into_iter()
            .filter_map(|filename| {
                // Taken from this map, unfinished, a moment ago.
                self.mark_file_error(&filename, &failure).ok()?;
                let phase = self.execution.file_statuses.get(&filename)?.phase.clone();
                Some((filename, phase))
            })
            .collect()
    }

    /// Mark the job interrupted by server shutdown and clear runtime claims.
    ///
    /// `Interrupted` is the correct lifecycle bit for shutdowns.  Although
    /// `JobStatus::is_terminal()` returns `true` for `Interrupted`, the
    /// recovery sequence special-cases it:
    ///
    /// 1. `db.recover_interrupted()` at startup moves `running|queued` rows
    ///    through `JobStatusColumns::stopped`, the transition this one
    ///    applies; it does **not** touch rows already written as
    ///    `interrupted` (such as the rows written here).
    /// 2. `load_from_db()` then reads each row (a stored active status
    ///    loads as `Interrupted`) and calls `Job::recover()`, which moves an
    ///    interrupted job (in memory, and written back to the DB) to
    ///    `Queued` if any file is resumable and its inputs are admitted.
    ///
    /// Writing `Cancelled` here (the previous behavior) skipped both steps:
    /// the row is never revisited, so an in-flight job that was running
    /// at shutdown stayed `Cancelled` forever and the user-visible dashboard
    /// showed it as a permanent user cancel even though no user pressed cancel.
    ///
    /// We deliberately set `completed_at = Some(_)` here because that is what
    /// `recover_interrupted()` itself writes for the rows it recovers, and
    /// downstream queries (e.g. dashboard "completed_at" projections) expect
    /// the field to be populated for any non-active row.  The recovery flow
    /// rewrites both `status` and `completed_at` when it transitions to
    /// `Queued`.
    ///
    /// Returns `true` if the transition fired; `false` if the job was already
    /// in a terminal state and could not be moved.
    pub(crate) fn interrupt_for_shutdown(&mut self, completed_at: MachineTime) -> bool {
        if self.stop(super::Stop::Interrupted, completed_at) {
            self.runtime.cancel_token.cancel();
            self.runtime.run_generation = self.runtime.run_generation.next();
            self.clear_lease();
            // The old run is still tearing down. It retains release authority
            // but cannot turn interrupted files into terminal cancellation.
            true
        } else {
            false
        }
    }

    /// Finalize the job after all file tasks have stopped mutating its state.
    ///
    /// A job that already reached a terminal state (cancelled mid-flight,
    /// interrupted by shutdown) keeps the status and the timestamp it has: the
    /// runner finished work the user had already stopped, and overwriting
    /// `completed_at` would lose when the cancel actually happened.
    /// File-level bookkeeping still runs either way, because those counters
    /// describe the files and are true regardless of the job's status.
    #[must_use = "a refused finalize must not be persisted either"]
    pub(crate) fn finalize(&mut self, outcome: super::Outcome, completed_at: MachineTime) -> bool {
        let event = self.execution.lifecycle.finish(outcome);
        let applied = self.accepted(event);
        // When finalizing as Failed, surface WHY: aggregate the per-file error
        // messages into the job-level error so the dashboard, jobs.db, and the
        // CLI all show a reason instead of a bare "failed". Without this the
        // job-level error stays `None` even though `file_statuses` carry the
        // real cause, the 2026-06 silent-failure bug, where a failed job
        // recorded an empty `error` column and no job-level reason.
        //
        // Gated on `applied`: a refused finalize must not hang a Failed-shaped
        // reason on a job that stayed Cancelled.
        if applied && outcome == super::Outcome::Failed {
            self.set_failure_reason_from_files();
        }
        if applied {
            self.schedule.completed_at = Some(completed_at);
            self.schedule.next_eligible_at = None;
        }
        applied
    }

    /// Build a job-level failure reason from the terminally-errored files.
    ///
    /// Distinct per-file messages are de-duplicated (the common case is many
    /// files hitting the identical worker error, e.g. a bad Stanza model) and
    /// sorted for a deterministic string. When errored files recorded no
    /// message, falls back to a count (`"N file(s) failed"`) rather than an
    /// empty reason. Returns `None` only when no file is terminally errored.
    fn aggregate_file_failure_reason(&self) -> Option<String> {
        let mut messages: BTreeSet<&str> = BTreeSet::new();
        let mut failed_files = 0usize;
        for file_status in self.execution.file_statuses.values() {
            if let FilePhase::Error { failure, .. } = &file_status.phase {
                failed_files += 1;
                if let Some(message) = failure.message() {
                    messages.insert(message);
                }
            }
        }
        if failed_files == 0 {
            return None;
        }
        let joined = messages.into_iter().collect::<Vec<_>>().join("; ");
        Some(if joined.is_empty() {
            format!("{failed_files} file(s) failed")
        } else if failed_files > 1 {
            format!("{failed_files} file(s) failed: {joined}")
        } else {
            joined
        })
    }

    /// Set the job-level error from the aggregated per-file failure reason,
    /// unless one was already set directly (e.g. by [`Job::fail`]).
    ///
    /// The single idempotent entry point that both [`Job::finalize`] and
    /// [`Job::recover`] use, so a Failed job always
    /// carries its cause regardless of which terminal path it took (normal
    /// finalization or recovery after a server bounce).
    fn set_failure_reason_from_files(&mut self) {
        if self.execution.error.is_none()
            && let Some(reason) = self.aggregate_file_failure_reason()
        {
            self.execution.error = Some(reason);
        }
    }

    /// Reset the job so unfinished files may run again from queued state.
    ///
    /// A file that wrote its output keeps it, clean or diagnosed: a
    /// diagnosed file's output is the producer's verdict on that recording,
    /// not a failure a rerun would repair, so it is never retried.
    pub(crate) fn prepare_for_restart(
        &mut self,
        authority: &batchalign_types::paths::AllowedRoots,
    ) -> Result<(), crate::error::ServerError> {
        // Validate before ANY mutation: recovered incompatible rows must stay
        // visible, non-runnable and unchanged, even through a manual restart.
        crate::command_model::command_spec(self.dispatch.command)
            .selected_output_policy(&self.dispatch.options)?;
        // Every run is admitted under the rules in force when it starts: a
        // job recorded as history, or one whose folder has since been removed
        // from the configuration, is admitted again here or not restarted.
        match self
            .execution
            .lifecycle
            .restart(|stored| stored.readmit(authority))
        {
            Ok(()) => {}
            Err(super::Reactivation::Refused(refused)) => {
                return Err(crate::error::ServerError::JobConflict {
                    message: format!(
                        "Job {} is {}, only cancelled or failed jobs can be restarted.",
                        self.identity.job_id, refused.from
                    ),
                    conflicts: Vec::new(),
                });
            }
            Err(super::Reactivation::Admission(refusal)) => return Err(refusal.into()),
        }
        for file_status in self.execution.file_statuses.values_mut() {
            if !file_status.status().wrote_output() {
                file_status.requeue();
            }
        }
        self.clear_run_outcome();
        self.runtime.cancel_token = CancellationToken::new();
        // A restarted job moves to a NEW run generation: the previous
        // runner (possibly still tearing down its in-flight file) becomes
        // stale and is refused finalization. `runner_active` is left
        // untouched here: it reflects the truth that the old runner is
        // still alive; the restarted runner waits on it via
        // `begin_runner` before taking over.
        self.runtime.run_generation = self.runtime.run_generation.next();
        self.execution
            .results
            .retain(|result| result.error.is_none());
        Ok(())
    }

    /// Recover a job read back from the database at startup: the one
    /// entry point, matched on the job's state.
    ///
    /// An interrupted job (the server stopped while it ran) with resumable
    /// files is queued again on inputs `authority` admits now, before
    /// anything changes; if they are refused, its unfinished files are
    /// closed with the reason (results already written are kept) and it
    /// fails visibly, so a restart can admit it once its folders are
    /// allowed. An interrupted job whose files all ended is settled: failed
    /// if any failed, otherwise completed. A cancelled job's open files are
    /// closed as cancelled: no runner is started for it after a restart, and
    /// closing reads no path. Every other job stands as stored.
    pub(crate) fn recover(
        &mut self,
        authority: &batchalign_types::paths::AllowedRoots,
        at: crate::store::EventTime,
    ) -> super::Recovered {
        match self.execution.lifecycle {
            super::JobLifecycle::Finished {
                status: crate::api::FinishedStatus::Interrupted,
                ..
            } => self.recover_interrupted(authority, at),
            super::JobLifecycle::Finished {
                status: crate::api::FinishedStatus::Cancelled,
                ..
            } => {
                let closed = self.close_unfinished_files(
                    "File stopped after job cancellation before the server restarted",
                    crate::scheduling::FailureCategory::Cancelled,
                    at,
                );
                if closed.is_empty() {
                    return super::Recovered::Unchanged;
                }
                self.clear_lease();
                super::Recovered::Rewritten(closed)
            }
            super::JobLifecycle::Finished {
                status:
                    crate::api::FinishedStatus::Completed
                    | crate::api::FinishedStatus::Failed
                    | crate::api::FinishedStatus::WritebackFailed,
                ..
            }
            | super::JobLifecycle::Active { .. } => super::Recovered::Unchanged,
        }
    }

    /// [`Job::recover`] for an interrupted job: resume, refuse or settle.
    fn recover_interrupted(
        &mut self,
        authority: &batchalign_types::paths::AllowedRoots,
        at: crate::store::EventTime,
    ) -> super::Recovered {
        let resumable = self
            .execution
            .file_statuses
            .values()
            .any(|file_status| file_status.status().is_resumable());
        if !resumable {
            let failed = self.any_terminal_files_failed();
            let settlement = match failed {
                true => super::Settlement::Failed,
                false => super::Settlement::Completed,
            };
            let event = self.execution.lifecycle.settle(settlement);
            if self.accepted(event) && failed {
                // Same as `finalize`: a recovered job that lands in Failed
                // must surface its cause, not an empty error (the 2026-06
                // silent-failure bug, on the startup-recovery path).
                self.set_failure_reason_from_files();
            }
            self.schedule.next_eligible_at = None;
            self.clear_lease();
            return super::Recovered::Rewritten(Vec::new());
        }

        let refusal = match self
            .execution
            .lifecycle
            .resume(|stored| stored.readmit(authority))
        {
            Ok(()) => {
                // A requeued job has not failed: an error from an earlier run
                // is not carried into the new one.
                self.clear_run_outcome();
                let requeued = self
                    .execution
                    .file_statuses
                    .iter_mut()
                    .filter(|(_, file_status)| file_status.status().is_resumable())
                    .map(|(filename, file_status)| {
                        file_status.requeue();
                        (filename.clone(), file_status.phase.clone())
                    })
                    .collect();
                return super::Recovered::Rewritten(requeued);
            }
            Err(super::Reactivation::Admission(refusal)) => refusal,
            // Matched as interrupted by the one caller.
            Err(super::Reactivation::Refused(refused)) => {
                tracing::error!(job_id = %self.identity.job_id, %refused, "recovery could not resume an interrupted job");
                return super::Recovered::Unchanged;
            }
        };
        tracing::warn!(job_id = %self.identity.job_id, %refusal,
            "Stored paths are not admitted by this server; recovering as a failed job");
        let reason = format!(
            "Not resumed after the server restarted: {refusal}. Results already written are kept; restart the job to run the rest once its folders are allowed."
        );
        let closed = self.close_unfinished_files(
            &reason,
            crate::scheduling::FailureCategory::Validation,
            at,
        );
        let event = self.execution.lifecycle.settle(super::Settlement::Failed);
        if self.accepted(event) {
            self.execution.error = Some(reason);
            self.schedule.completed_at = Some(at.instant());
        }
        self.schedule.next_eligible_at = None;
        self.clear_lease();
        super::Recovered::Rewritten(closed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduling::FailureCategory;
    use crate::store::job::test_support::{running_job_fixture, three_file_job_fixture};
    use crate::store::job::types::FileFailureRecord;

    /// Verify that `interrupt_for_shutdown` writes `JobStatus::Interrupted`
    /// (not `Cancelled`) so the recovery sequence can requeue unfinished work.
    ///
    /// This is the regression test for the 2026-04-27 investigation in which
    /// in-flight jobs were marked `Cancelled` when a server bounced; recovery
    /// never revisited `Cancelled` rows, so the jobs appeared permanently
    /// cancelled with no user action.
    #[test]
    fn interrupt_for_shutdown_writes_interrupted_not_cancelled() {
        let mut job = running_job_fixture();
        let now = crate::unix_time(1_777_300_000.0);

        let did_transition = job.interrupt_for_shutdown(now);

        assert!(did_transition, "running job must accept the interrupt");
        assert_eq!(
            job.status(),
            JobStatus::Interrupted,
            "shutdown must mark the job Interrupted (recovery-eligible), not Cancelled (final)"
        );
        assert!(
            job.runtime.cancel_token.is_cancelled(),
            "the cancel token still flips so in-flight workers stop"
        );
        assert_eq!(
            job.schedule.completed_at,
            Some(now),
            "completed_at is populated to match recover_interrupted's existing convention"
        );
        assert!(
            job.runtime.runner_active,
            "the live runner retains teardown ownership"
        );
    }

    /// A live stop leaves the job's whole status image as the shared
    /// transition states it (`JobStatusColumns::stopped`), the image startup
    /// recovery writes: every column, not only the ones the stop changes.
    #[test]
    fn a_live_stop_is_the_shared_transition_on_every_column() {
        for (stop, apply) in [
            (
                super::super::Stop::Interrupted,
                Job::interrupt_for_shutdown as fn(&mut Job, MachineTime) -> bool,
            ),
            (super::super::Stop::Cancelled, |job: &mut Job, at| {
                job.request_cancellation(at).is_some()
            }),
        ] {
            let mut job = running_job_fixture();
            job.execution.error = Some("an earlier failure".into());
            job.schedule.num_workers = Some(3);
            job.schedule.next_eligible_at = Some(crate::unix_time(1_777_200_000.0));
            let now = crate::unix_time(1_777_300_000.0);
            let expected = job.status_columns().stopped(stop, now);

            assert!(apply(&mut job, now), "{stop:?}");
            assert_eq!(job.status_columns(), expected, "{stop:?}");
        }
    }

    /// Regression: 2026-05-11 morphotag job `150c824a-48e`.
    ///
    /// The job submitted three files. Two terminal-errored (worker IPC
    /// protocol mismatches mid-run), one succeeded. `file_statuses` in
    /// jobs.db correctly recorded the per-file outcomes, but the overall
    /// job status was finalized as `Completed` because
    /// `all_terminal_files_failed()` reports only the all-or-nothing
    /// case. The dashboard and CLI therefore claimed the job succeeded,
    /// hiding the silent data loss for the two failed outputs.
    ///
    /// The fix predicate is `any_terminal_files_failed()`: true iff any
    /// terminal file is an error. The existing
    /// `all_terminal_files_failed()` predicate is preserved so any
    /// other caller that genuinely wants "every terminal file errored"
    /// semantics still has it. The two methods together match the
    /// `JobStatus` enum's documented contract at
    /// `crates/batchalign/src/types/status.rs` lines 27-31:
    ///
    /// > Completed = all files processed successfully
    /// > Failed = one or more files encountered an unrecoverable error
    #[test]
    fn any_terminal_files_failed_detects_partial_failure_shape() {
        let mut job = three_file_job_fixture();
        let started = crate::unix_time(1_778_517_977.7);
        let finished_done = crate::unix_time(1_778_518_056.4);
        let finished_error = crate::unix_time(1_778_518_053.6);

        // 60home-3.cha succeeds (the only file with an actual output
        // on disk after the incident).
        assert_eq!(job.mark_file_processing("60home-3.cha", started), Ok(()));
        assert_eq!(
            job.mark_file_done(
                "60home-3.cha",
                finished_done,
                super::super::FileCompletion::WithoutResult
            ),
            Ok(())
        );

        // 65-3.cha and 65home-3.cha both terminal-error with worker
        // protocol mismatches.
        for filename in ["65-3.cha", "65home-3.cha"] {
            assert_eq!(job.mark_file_processing(filename, started), Ok(()));
            assert_eq!(
                job.mark_file_error(
                    filename,
                    &FileFailureRecord {
                        message: "worker protocol error".into(),
                        category: FailureCategory::ProviderTerminal,
                        finished_at: crate::store::EventTime::fixed(finished_error),
                    },
                ),
                Ok(())
            );
        }

        assert!(
            job.any_terminal_files_failed(),
            "partial-failure must surface via any_terminal_files_failed()"
        );
    }

    /// A job that fails with terminally-errored files that recorded NO message
    /// still gets a job-level reason ("N file(s) failed"), not an empty error.
    #[test]
    fn finalize_failed_records_count_when_files_have_no_message() {
        let mut job = running_job_fixture();
        let file = job
            .execution
            .file_statuses
            .get_mut("job-file.cha")
            .expect("fixture file present");
        file.phase = FilePhase::Error {
            started_at: None,
            finished_at: None,
            failure: crate::store::FileFailure::of_failed_row(None, None),
        };

        // Assert the transition was APPLIED rather than discarding the result.
        // `finalize` is `#[must_use]` precisely because a refused transition
        // leaves the job in its prior status, and every assertion below would
        // then be checking the fixture's own state instead of finalize's work.
        assert!(
            job.finalize(
                crate::store::Outcome::Failed,
                crate::unix_time(1_778_518_060.0)
            ),
            "Running -> Failed by the finish event must be applied"
        );

        assert_eq!(job.status(), JobStatus::Failed);
        let reason = job.execution.error.as_deref().expect(
            "failed job with errored files must record a reason even without \
             per-file messages",
        );
        assert!(
            reason.contains("file(s) failed"),
            "expected a count-based reason, got: {reason}"
        );
    }

    /// Recovery of an interrupted job whose files all terminal-errored
    /// must surface the failure reason on the job-level error, the same as
    /// `finalize` does; a server bounce must not drop the cause.
    #[test]
    fn reconcile_recovered_failed_job_records_file_failure_reason() {
        let mut job = running_job_fixture();
        let started = crate::unix_time(1_778_517_900.0);
        let finished = crate::unix_time(1_778_518_000.0);
        assert_eq!(job.mark_file_processing("job-file.cha", started), Ok(()));
        assert_eq!(
            job.mark_file_error(
                "job-file.cha",
                &FileFailureRecord {
                    message: "worker bootstrap error: md5 mismatch".into(),
                    category: FailureCategory::WorkerBootstrap,
                    finished_at: crate::store::EventTime::fixed(finished),
                },
            ),
            Ok(())
        );
        assert!(
            job.execution.error.is_none(),
            "precondition: job-level error unset before recovery"
        );
        // The server stopped while it ran.
        assert!(job.interrupt_for_shutdown(finished));

        let recovered = job.recover(
            &batchalign_types::paths::AllowedRoots::none(),
            crate::store::EventTime::fixed(crate::unix_time(1_778_518_100.0)),
        );

        assert_eq!(recovered, crate::store::Recovered::Rewritten(Vec::new()));
        assert_eq!(job.status(), JobStatus::Failed);
        let reason = job
            .execution
            .error
            .as_deref()
            .expect("recovered failed job must record the aggregated file failure reason");
        assert!(reason.contains("md5 mismatch"), "got: {reason}");
    }
}
