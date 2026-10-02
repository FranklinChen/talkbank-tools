//! Startup recovery and TTL pruning operations.

use sqlx::Row;
use tracing::info;

use crate::error::ServerError;

use batchalign_types::paths::ServerPath;

use crate::api::{JobId, JobStatus, MachineTime};
use crate::config::JobTtlDays;
use crate::store::{JobStatusColumns, Stop};

/// A job deleted for being past its retention, and where its staging files
/// were.
#[derive(Debug, Clone, PartialEq)]
pub struct PrunedJob {
    /// The deleted job.
    pub job_id: JobId,
    /// Its staging directory on this host, for the caller to remove.
    pub staging_dir: ServerPath,
}

use super::JobDB;

impl JobDB {
    /// Mark queued/running jobs as interrupted on startup.
    ///
    /// Each job's status row is read as its typed image, moved through
    /// [`JobStatusColumns::stopped`] with [`Stop::Interrupted`] (the
    /// transition a shutdown applies to a live job) and written whole, and
    /// each of its files the interruption moves is written from its
    /// interrupted phase: one transaction per job, so a job is never left
    /// interrupted with its files still in flight. Returns the jobs that
    /// were marked interrupted.
    pub async fn recover_interrupted(&self, now: MachineTime) -> Result<Vec<JobId>, ServerError> {
        let rows = sqlx::query(
            "SELECT job_id, status, error, completed_at, num_workers, next_eligible_at
             FROM jobs WHERE status IN (?, ?)",
        )
        .bind(JobStatus::Queued.to_string())
        .bind(JobStatus::Running.to_string())
        .fetch_all(&self.pool)
        .await?;

        let mut ids = Vec::with_capacity(rows.len());
        for row in &rows {
            let id = JobId::from(row.try_get::<String, _>("job_id")?);
            let columns = JobStatusColumns::read_stored(&id, row)?;
            let file_rows = self.load_file_status_rows(id.as_ref()).await?;

            let mut tx = self.pool.begin().await?;
            super::update::write_job_status_on(
                &mut *tx,
                id.as_ref(),
                &columns.stopped(Stop::Interrupted, now),
            )
            .await?;
            for file in file_rows {
                // Only the rows the interruption moves are written (the
                // phase says which: `interrupted` leaves a terminal or
                // interrupted phase as it is). A row nothing moves is read,
                // reported and repaired once, when jobs load.
                let phase =
                    crate::store::queries::recover_file_phase(id.as_ref(), &file).into_phase();
                let interrupted = phase.clone().interrupted();
                if interrupted != phase {
                    super::update::write_file_phase_on(
                        &mut *tx,
                        id.as_ref(),
                        &file.filename,
                        &interrupted,
                        None,
                    )
                    .await?;
                }
            }
            tx.commit().await?;
            ids.push(id);
        }

        if !ids.is_empty() {
            info!("Marked {} interrupted jobs: {:?}", ids.len(), ids);
        }
        Ok(ids)
    }

    /// Delete jobs submitted before `ttl`'s cutoff at `now`.
    ///
    /// Returns each pruned job with its staging directory, as one value, so
    /// the caller can clean up the files on disk. It used to return the ids and
    /// the directories as two parallel `Vec<String>`s.
    pub async fn prune_expired(
        &self,
        ttl: JobTtlDays,
        now: MachineTime,
    ) -> Result<Vec<PrunedJob>, ServerError> {
        let cutoff = ttl.cutoff(now);

        let rows = sqlx::query("SELECT job_id, staging_dir FROM jobs WHERE submitted_at < ?")
            .bind(cutoff)
            .fetch_all(&self.pool)
            .await?;

        let mut pruned = Vec::with_capacity(rows.len());
        for row in &rows {
            pruned.push(PrunedJob {
                job_id: JobId::from(row.try_get::<String, _>("job_id")?),
                staging_dir: ServerPath::from(row.try_get::<String, _>("staging_dir")?),
            });
        }

        if !pruned.is_empty() {
            for job in &pruned {
                sqlx::query("DELETE FROM jobs WHERE job_id = ?")
                    .bind(job.job_id.as_ref())
                    .execute(&self.pool)
                    .await?;
            }
            info!("Pruned {} expired jobs", pruned.len());
        }

        Ok(pruned)
    }
}
