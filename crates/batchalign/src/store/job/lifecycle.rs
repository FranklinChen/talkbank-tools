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
//! `reconcile_recovered_runtime_state` here and `load_from_db` in
//! `store/queries/recovery.rs`).
//!
//! Restart preparation (`prepare_for_restart`) resets unfinished file work
//! while preserving completed results, enabling partial retry after failures.

use tokio_util::sync::CancellationToken;

use std::collections::BTreeSet;

use crate::api::{FileStatusKind, JobStatus, MachineTime, StatusChange};

use super::Job;
use super::types::RecoveryDisposition;
use crate::store::FilePhase;

impl Job {
    /// Return whether the job's cancellation token has been triggered.
    pub(crate) fn is_cancelled(&self) -> bool {
        self.runtime.cancel_token.is_cancelled()
    }

    /// Return whether every terminal file currently recorded is an error.
    pub(crate) fn all_terminal_files_failed(&self) -> bool {
        let terminal: Vec<FileStatusKind> = self
            .execution
            .file_statuses
            .values()
            .filter(|file_status| file_status.status().is_terminal())
            .map(|file_status| file_status.status())
            .collect();
        !terminal.is_empty()
            && terminal
                .iter()
                .all(|status| *status == FileStatusKind::Error)
    }

    /// Return whether at least one terminal file currently recorded is
    /// an error.
    ///
    /// Pairs with [`all_terminal_files_failed`]: that predicate is the
    /// strict all-or-nothing test, this one is the partial-failure test.
    /// The job-finalization logic in `runner/execution.rs` uses this
    /// (the weaker predicate) to decide `JobStatus::Failed` so that a
    /// job with even one terminal-error file is honestly surfaced to
    /// dashboards and CLI rather than silently labelled `Completed`.
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

    /// Apply a status change if the state machine permits it.
    ///
    /// The ONLY place `execution.status` is written. Returns false when the
    /// transition is illegal for this reason, in which case nothing is
    /// mutated and the caller must not persist a status either. See
    /// [`JobStatus::can_transition_to`].
    #[must_use = "a refusal means the status did NOT change; status-derived \
                  bookkeeping and any persisted status must not assume it did"]
    pub(crate) fn set_status(&mut self, next: JobStatus, reason: StatusChange) -> bool {
        if !self.execution.status.can_transition_to(next, reason) {
            // Logged here because this is the single choke point, so no writer
            // has to remember. Most refusals are ordinary races (a user cancels
            // while a runner is mid-flight); the loud one is a refused
            // `Finalize`, which means work was completed and then discarded.
            tracing::debug!(
                job_id = %self.identity.job_id,
                from = %self.execution.status,
                to = %next,
                ?reason,
                "refusing job status transition"
            );
            return false;
        }
        self.execution.status = next;
        true
    }

    /// Stop the job at `at`, if its status allows: the status image's
    /// [`JobStatusColumns::stopped`] transition, applied to the live job.
    /// Returns whether it fired.
    ///
    /// [`JobStatusColumns::stopped`]: super::JobStatusColumns::stopped
    fn stop(&mut self, stop: super::Stop, at: MachineTime) -> bool {
        if !self.set_status(stop.status(), StatusChange::Interrupt) {
            return false;
        }
        let image = self.status_columns().stopped(stop, at);
        self.adopt_status_columns(image);
        true
    }

    /// Make the live job hold `image`, every column of it. Destructured
    /// whole, so a column added to the image cannot be left out here, and a
    /// transition stated on the image is the transition the live job takes.
    fn adopt_status_columns(&mut self, image: super::JobStatusColumns) {
        let super::JobStatusColumns {
            status,
            error,
            completed_at,
            num_workers,
            next_eligible_at,
        } = image;
        self.execution.status = status;
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
        if !self.set_status(JobStatus::Queued, StatusChange::MemoryRequeue) {
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
    /// database agree. Every status writer goes through [`Job::set_status`]
    /// and the [`JobStatus::can_transition_to`] table.
    #[must_use = "a dropped refusal silently resurrects a cancelled job"]
    pub(crate) fn mark_running(&mut self) -> bool {
        if !self.set_status(JobStatus::Running, StatusChange::Dispatch) {
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
        if !self.set_status(JobStatus::Failed, StatusChange::Finalize) {
            return false;
        }
        self.execution.error = Some(error.to_string());
        self.schedule.completed_at = Some(completed_at);
        true
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
    /// 2. `load_from_db()` then reads each row and, for any job with
    ///    `status ∈ {Interrupted, Running}`, calls
    ///    `Job::reconcile_recovered_runtime_state()` which transitions the
    ///    in-memory job (and writes back to the DB) to `Queued` if any
    ///    file is resumable.
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
        self.runtime.cancel_token.cancel();
        if self.stop(super::Stop::Interrupted, completed_at) {
            self.clear_lease();
            self.runtime.runner_active = false;
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
    pub(crate) fn finalize(&mut self, final_status: JobStatus, completed_at: MachineTime) -> bool {
        let applied = self.set_status(final_status, StatusChange::Finalize);
        // When finalizing as Failed, surface WHY: aggregate the per-file error
        // messages into the job-level error so the dashboard, jobs.db, and the
        // CLI all show a reason instead of a bare "failed". Without this the
        // job-level error stays `None` even though `file_statuses` carry the
        // real cause, the 2026-06 silent-failure bug, where a failed job
        // recorded an empty `error` column and no job-level reason.
        //
        // Gated on `applied`: a refused finalize must not hang a Failed-shaped
        // reason on a job that stayed Cancelled.
        if applied && final_status == JobStatus::Failed {
            self.set_failure_reason_from_files();
        }
        if applied {
            self.schedule.completed_at = Some(completed_at);
            self.schedule.next_eligible_at = None;
        }
        self.execution.completed_files = self
            .execution
            .file_statuses
            .values()
            .filter(|file_status| file_status.status().is_terminal())
            .count() as i64;
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
    /// [`Job::reconcile_recovered_runtime_state`] use, so a Failed job always
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
    pub(crate) fn prepare_for_restart(&mut self) {
        for file_status in self.execution.file_statuses.values_mut() {
            if file_status.status() != FileStatusKind::Done {
                file_status.requeue();
            }
        }

        // Cannot fail: `JobRegistry::restart_job` checks `can_restart()`
        // before calling, and the Restart arm asks the same question.
        debug_assert!(
            self.execution.status.can_restart(),
            "restart_job must pre-check can_restart"
        );
        let _ = self.set_status(JobStatus::Queued, StatusChange::Restart);
        self.execution.error = None;
        self.schedule.completed_at = None;
        self.schedule.next_eligible_at = None;
        self.clear_lease();
        self.runtime.cancel_token = CancellationToken::new();
        // A restarted job moves to a NEW run generation: the previous
        // runner (possibly still tearing down its in-flight file) becomes
        // stale and is refused finalization. `runner_active` is left
        // untouched here: it reflects the truth that the old runner is
        // still alive; the restarted runner waits on it via
        // `begin_runner` before taking over.
        self.runtime.run_generation = self.runtime.run_generation.next();
        self.execution.completed_files = self
            .execution
            .file_statuses
            .values()
            .filter(|file_status| file_status.status() == FileStatusKind::Done)
            .count() as i64;
        self.execution
            .results
            .retain(|result| result.error.is_none());
    }

    /// Reconcile a persisted interrupted/running job during startup recovery.
    pub(crate) fn reconcile_recovered_runtime_state(&mut self) -> RecoveryDisposition {
        let has_resumable = self
            .execution
            .file_statuses
            .values()
            .any(|file_status| file_status.status().is_resumable());

        if has_resumable {
            for file_status in self.execution.file_statuses.values_mut() {
                if file_status.status().is_resumable() {
                    file_status.requeue();
                }
            }
            let _ = self.set_status(JobStatus::Queued, StatusChange::Reconcile);
            // A requeued job has not failed: an error from an earlier run is
            // not carried into the new one.
            self.execution.error = None;
            self.schedule.completed_at = None;
            self.schedule.next_eligible_at = None;
            self.clear_lease();
            RecoveryDisposition::Requeued
        } else {
            let all_errored = self
                .execution
                .file_statuses
                .values()
                .all(|file_status| file_status.status() == FileStatusKind::Error);
            let reconciled = if all_errored {
                JobStatus::Failed
            } else {
                JobStatus::Completed
            };
            let _ = self.set_status(reconciled, StatusChange::Reconcile);
            // Same as `finalize`: a recovered job that lands in Failed must
            // surface its cause, not an empty error (the 2026-06 silent-failure
            // bug, on the startup-recovery path).
            if all_errored {
                self.set_failure_reason_from_files();
            }
            self.execution.completed_files = self.total_files() as i64;
            self.schedule.next_eligible_at = None;
            self.clear_lease();
            if all_errored {
                RecoveryDisposition::Failed
            } else {
                RecoveryDisposition::Completed
            }
        }
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
            job.execution.status,
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
        assert!(!job.runtime.runner_active, "runner claim is released");
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
        assert!(job.mark_file_processing("60home-3.cha", started));
        assert!(job.mark_file_done("60home-3.cha", finished_done, None));

        // 65-3.cha and 65home-3.cha both terminal-error with worker
        // protocol mismatches.
        for filename in ["65-3.cha", "65home-3.cha"] {
            assert!(job.mark_file_processing(filename, started));
            assert!(job.mark_file_error(
                filename,
                &FileFailureRecord {
                    message: "worker protocol error".into(),
                    category: FailureCategory::ProviderTerminal,
                    finished_at: crate::store::EventTime::fixed(finished_error),
                },
            ));
        }

        assert!(
            job.any_terminal_files_failed(),
            "partial-failure must surface via any_terminal_files_failed()"
        );
        assert!(
            !job.all_terminal_files_failed(),
            "all-failed semantics must remain narrow, one done file is enough \
             for the all-failed predicate to be false"
        );
    }

    /// Edge case: no terminal files yet (everything still queued or
    /// processing). Both predicates must return false, neither
    /// "all failed" nor "any failed" applies until there's at least
    /// one terminal file to evaluate.
    #[test]
    fn any_and_all_terminal_files_failed_both_false_when_no_terminal_files() {
        let job = three_file_job_fixture();
        assert!(!job.all_terminal_files_failed());
        assert!(!job.any_terminal_files_failed());
    }

    /// Edge case: every terminal file is an error. Both predicates
    /// must return true: `any_failed` is the weaker condition and
    /// must imply `all_failed` whenever all files are terminal-error.
    #[test]
    fn any_terminal_files_failed_true_when_all_files_error() {
        let mut job = three_file_job_fixture();
        let started = crate::unix_time(1_778_517_977.7);
        let finished = crate::unix_time(1_778_518_056.4);
        for filename in ["65-3.cha", "60home-3.cha", "65home-3.cha"] {
            assert!(job.mark_file_processing(filename, started));
            assert!(job.mark_file_error(
                filename,
                &FileFailureRecord {
                    message: "worker error".into(),
                    category: FailureCategory::ProviderTerminal,
                    finished_at: crate::store::EventTime::fixed(finished),
                },
            ));
        }
        assert!(job.all_terminal_files_failed());
        assert!(job.any_terminal_files_failed());
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
            job.finalize(JobStatus::Failed, crate::unix_time(1_778_518_060.0)),
            "Running -> Failed under StatusChange::Finalize must be applied"
        );

        assert_eq!(job.execution.status, JobStatus::Failed);
        let reason = job.execution.error.as_deref().expect(
            "failed job with errored files must record a reason even without \
             per-file messages",
        );
        assert!(
            reason.contains("file(s) failed"),
            "expected a count-based reason, got: {reason}"
        );
    }

    /// Recovery of an interrupted/running job whose files all terminal-errored
    /// must surface the failure reason on the job-level error, the same as
    /// `finalize` does; a server bounce must not drop the cause.
    #[test]
    fn reconcile_recovered_failed_job_records_file_failure_reason() {
        let mut job = running_job_fixture();
        let started = crate::unix_time(1_778_517_900.0);
        let finished = crate::unix_time(1_778_518_000.0);
        assert!(job.mark_file_processing("job-file.cha", started));
        assert!(job.mark_file_error(
            "job-file.cha",
            &FileFailureRecord {
                message: "worker bootstrap error: md5 mismatch".into(),
                category: FailureCategory::WorkerBootstrap,
                finished_at: crate::store::EventTime::fixed(finished),
            },
        ));
        assert!(
            job.execution.error.is_none(),
            "precondition: job-level error unset before recovery"
        );

        let disposition = job.reconcile_recovered_runtime_state();

        assert!(matches!(disposition, RecoveryDisposition::Failed));
        assert_eq!(job.execution.status, JobStatus::Failed);
        let reason = job
            .execution
            .error
            .as_deref()
            .expect("recovered failed job must record the aggregated file failure reason");
        assert!(reason.contains("md5 mismatch"), "got: {reason}");
    }
}
