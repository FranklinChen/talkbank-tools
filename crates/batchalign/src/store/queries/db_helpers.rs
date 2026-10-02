//! DB persistence helpers and WebSocket notification helpers.

use crate::api::{FileStatusEntry, JobId, JobListItem};
use crate::scheduling::{AttemptOutcome, FailureCategory, RetryDisposition, WorkUnitKind};
use tracing::{debug, warn};

use super::super::JobStore;
use crate::ws::WsEvent;

/// Persisted file-level status update written through to SQLite.
pub(crate) struct PersistedFileUpdate<'a> {
    /// Filename within the parent job.
    pub filename: &'a str,
    /// The file's phase after the transition; its whole column set is
    /// written (see `FilePhase::columns`).
    pub phase: &'a crate::store::FilePhase,
    /// Result content type, for a file that finished with output.
    pub content_type: Option<&'a str>,
}

/// Attempt-start facts persisted for one file work unit.
pub(crate) struct AttemptStartRecord<'a> {
    /// Filename for the attempt row.
    pub filename: &'a str,
    /// Kind of work unit being attempted.
    pub work_unit_kind: WorkUnitKind,
    /// When the attempt started, by the store's clock.
    pub started_at: crate::store::EventTime,
}

/// Attempt-finish facts persisted for one file work unit.
pub(crate) struct AttemptFinishRecord<'a> {
    /// Filename for the attempt row.
    pub filename: &'a str,
    /// Final attempt outcome.
    pub outcome: AttemptOutcome,
    /// Optional broad failure category.
    pub failure_category: Option<FailureCategory>,
    /// Retry/terminal disposition selected by the runner.
    pub disposition: RetryDisposition,
    /// When the attempt finished, by the store's clock.
    pub finished_at: crate::store::EventTime,
}

/// Which lease transition is being persisted, named in the warning when the
/// write fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LeaseWrite {
    /// The local queue claimed the job.
    #[cfg(test)]
    Claim,
    /// A runner took exclusive ownership.
    RunnerClaim,
    /// The heartbeat renewed the lease.
    Renew,
    /// The runner released its claim.
    Release,
    /// A restart cleared the lease.
    Restart,
}

impl std::fmt::Display for LeaseWrite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            #[cfg(test)]
            Self::Claim => "claim",
            Self::RunnerClaim => "runner claim",
            Self::Renew => "renew",
            Self::Release => "release",
            Self::Restart => "restart",
        })
    }
}

impl JobStore {
    // -------------------------------------------------------------------
    // DB helper methods (safe no-ops when db is None)
    // -------------------------------------------------------------------

    /// Persist a job's lease (or its absence) for one lease transition. The
    /// one place the lease columns are written from the store.
    pub(crate) async fn db_persist_lease(
        &self,
        job_id: &JobId,
        lease: Option<&crate::scheduling::LeaseRecord>,
        write: LeaseWrite,
    ) {
        if let Some(db) = &self.db
            && let Err(e) = db.update_job_lease(job_id, lease).await
        {
            warn!(job_id = %job_id, error = %e, "DB update_job_lease failed on {write}");
        }
    }

    /// Persist a job's status columns after a transition, as the job now IS
    /// (`Job::status_columns`), never as the caller asked: a transition the
    /// state machine declined leaves the row describing the status the job
    /// actually holds. A no-op without a database or for an unknown job.
    pub(crate) async fn db_persist_job_status(&self, job_id: &JobId) {
        let Some(db) = &self.db else {
            return;
        };
        let Some(columns) = self.registry.status_columns(job_id).await else {
            return;
        };
        if let Err(e) = db.write_job_status(job_id, &columns).await {
            warn!(job_id = %job_id, error = %e, "DB write_job_status failed");
        }
    }

    /// Persist one file-level status update to SQLite when the DB is enabled.
    pub(crate) async fn db_update_file(&self, job_id: &JobId, update: PersistedFileUpdate<'_>) {
        if let Some(db) = &self.db
            && let Err(e) = db
                .update_file_status(job_id, update.filename, update.phase, update.content_type)
                .await
        {
            warn!(
                job_id = %job_id,
                filename = %update.filename,
                error = %e,
                "DB update_file_status failed"
            );
        }
    }

    /// Persist and attach a new active attempt record for one file.
    pub(crate) async fn db_start_attempt(&self, job_id: &JobId, attempt: AttemptStartRecord<'_>) {
        let Some(db) = &self.db else {
            return;
        };

        match db
            .insert_attempt_start(
                job_id,
                attempt.filename,
                attempt.work_unit_kind,
                attempt.started_at.instant(),
                None,
                None,
            )
            .await
        {
            Ok((attempt_id, _attempt_number)) => {
                let _ = self
                    .registry
                    .attach_attempt_id(job_id, attempt.filename, attempt_id)
                    .await;
            }
            Err(e) => {
                warn!(
                    job_id = %job_id,
                    filename = %attempt.filename,
                    error = %e,
                    "DB insert_attempt_start failed"
                );
            }
        }
    }

    /// Finalize the currently active attempt for one file.
    pub(crate) async fn db_finish_attempt_for_file(
        &self,
        job_id: &JobId,
        attempt: AttemptFinishRecord<'_>,
    ) {
        let attempt_id = self
            .registry
            .take_attempt_id(job_id, attempt.filename)
            .await;

        let Some(attempt_id) = attempt_id else {
            return;
        };

        if let Some(db) = &self.db
            && let Err(e) = db
                .finish_attempt(
                    &attempt_id,
                    attempt.outcome,
                    attempt.failure_category,
                    attempt.disposition,
                    attempt.finished_at.instant(),
                )
                .await
        {
            warn!(
                job_id = %job_id,
                filename = %attempt.filename,
                attempt_id = %attempt_id,
                error = %e,
                "DB finish_attempt failed"
            );
        }
    }

    // -------------------------------------------------------------------
    // Notifications
    // -------------------------------------------------------------------

    /// Notify WS clients of one updated job summary row.
    pub(crate) fn notify_job_item(&self, item: JobListItem) {
        match serde_json::to_value(&item) {
            Ok(job) => self.broadcast_ws_event(
                "job_update",
                WsEvent::JobUpdate { job },
                Some(item.job_id.to_string()),
                None,
            ),
            Err(error) => {
                warn!(
                    job_id = %item.job_id,
                    error = %error,
                    "Failed to serialize job update for WS broadcast"
                );
            }
        }
    }

    /// Notify WS clients of one updated file-status row.
    pub(crate) fn notify_file_update(
        &self,
        job_id: &JobId,
        file: FileStatusEntry,
        completed_files: i64,
    ) {
        match serde_json::to_value(&file) {
            Ok(file_json) => self.broadcast_ws_event(
                "file_update",
                WsEvent::FileUpdate {
                    job_id: job_id.clone(),
                    file: file_json,
                    completed_files,
                },
                Some(job_id.to_string()),
                Some(file.filename.to_string()),
            ),
            Err(error) => {
                warn!(
                    job_id = %job_id,
                    filename = %file.filename,
                    error = %error,
                    "Failed to serialize file update for WS broadcast"
                );
            }
        }
    }

    fn broadcast_ws_event(
        &self,
        event_type: &'static str,
        event: WsEvent,
        job_id: Option<String>,
        filename: Option<String>,
    ) {
        if let Err(tokio::sync::broadcast::error::SendError(_event)) = self.ws_tx.send(event) {
            debug!(
                event_type,
                job_id = job_id.as_deref().unwrap_or(""),
                filename = filename.as_deref().unwrap_or(""),
                "Dropping WS broadcast because there are no subscribers"
            );
        }
    }
}
