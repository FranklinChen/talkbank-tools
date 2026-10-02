//! Update and delete operations on the `jobs` and `file_statuses` tables.

use crate::api::MachineTime;
use crate::scheduling::{AttemptOutcome, FailureCategory, LeaseRecord, RetryDisposition};
use crate::store::{FilePhase, JobStatusColumns};

use crate::error::ServerError;

use super::JobDB;

/// [`JobDB::write_job_status`] on any connection, so a caller can make it
/// part of a transaction.
pub(super) async fn write_job_status_on<'e, E>(
    executor: E,
    job_id: &str,
    columns: &JobStatusColumns,
) -> Result<(), ServerError>
where
    E: sqlx::Executor<'e, Database = sqlx::Sqlite>,
{
    sqlx::query(
        "UPDATE jobs
         SET status = ?,
             error = ?,
             completed_at = ?,
             num_workers = ?,
             next_eligible_at = ?
         WHERE job_id = ?",
    )
    .bind(columns.status().to_string())
    .bind(columns.error())
    .bind(columns.completed_at())
    .bind(columns.num_workers())
    .bind(columns.next_eligible_at())
    .bind(job_id)
    .execute(executor)
    .await?;
    Ok(())
}

/// [`JobDB::update_file_status`] on any connection, so a caller can make it
/// part of a transaction.
pub(super) async fn write_file_phase_on<'e, E>(
    executor: E,
    job_id: &str,
    filename: &str,
    phase: &FilePhase,
    content_type: Option<&str>,
) -> Result<(), ServerError>
where
    E: sqlx::Executor<'e, Database = sqlx::Sqlite>,
{
    bind_phase_columns(
        sqlx::query(
            "UPDATE file_statuses
             SET status = ?,
                 error = ?,
                 error_category = ?,
                 started_at = ?,
                 finished_at = ?,
                 next_eligible_at = ?,
                 content_type = COALESCE(?, content_type)
             WHERE job_id = ? AND filename = ?",
        ),
        phase.columns(),
    )
    .bind(content_type)
    .bind(job_id)
    .bind(filename)
    .execute(executor)
    .await?;
    Ok(())
}

/// Bind a phase's six `file_statuses` columns, in the order every phase
/// writer's SQL names them: `status, error, error_category, started_at,
/// finished_at, next_eligible_at`.
pub(super) fn bind_phase_columns<'q>(
    query: sqlx::query::Query<'q, sqlx::Sqlite, sqlx::sqlite::SqliteArguments>,
    columns: crate::store::FilePhaseColumns<'q>,
) -> sqlx::query::Query<'q, sqlx::Sqlite, sqlx::sqlite::SqliteArguments> {
    query
        .bind(columns.status.to_string())
        .bind(columns.error)
        .bind(columns.error_category.map(|category| category.to_string()))
        .bind(columns.started_at)
        .bind(columns.finished_at)
        .bind(columns.next_eligible_at)
}

impl JobDB {
    /// Write a job's status columns, every one of them, NULLs included.
    ///
    /// Takes the job's whole status image ([`JobStatusColumns`], built only
    /// by its owners), so nothing an earlier status wrote survives a
    /// transition that does not own it.
    pub(crate) async fn write_job_status(
        &self,
        job_id: &str,
        columns: &JobStatusColumns,
    ) -> Result<(), ServerError> {
        write_job_status_on(&self.pool, job_id, columns).await
    }

    /// Write a job's submitter columns, `None` as the columns' empty
    /// spelling of no submitter.
    pub(crate) async fn write_job_submitter(
        &self,
        job_id: &str,
        submitter: Option<&crate::store::Submitter>,
    ) -> Result<(), ServerError> {
        let (address, name) = crate::store::Submitter::columns(submitter);
        sqlx::query("UPDATE jobs SET submitted_by = ?, submitted_by_name = ? WHERE job_id = ?")
            .bind(address)
            .bind(name)
            .bind(job_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Write a job's lease, or clear it with `None`. The three columns are
    /// written together, so a row never holds part of a lease.
    pub async fn update_job_lease(
        &self,
        job_id: &str,
        lease: Option<&LeaseRecord>,
    ) -> Result<(), ServerError> {
        sqlx::query(
            "UPDATE jobs
             SET leased_by_node = ?,
                 lease_expires_at = ?,
                 lease_heartbeat_at = ?
             WHERE job_id = ?",
        )
        .bind(lease.map(|lease| lease.leased_by_node().as_ref()))
        .bind(lease.map(LeaseRecord::expires_at))
        .bind(lease.map(LeaseRecord::heartbeat_at))
        .bind(job_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Write one file's phase to its `file_statuses` row.
    ///
    /// Every column the phase owns is written from [`FilePhase::columns`],
    /// NULLs included, so nothing an earlier phase wrote survives into a phase
    /// that does not own it. `content_type` is not a phase column (it is
    /// `NOT NULL` with a default) and is written only when given.
    pub async fn update_file_status(
        &self,
        job_id: &str,
        filename: &str,
        phase: &FilePhase,
        content_type: Option<&str>,
    ) -> Result<(), ServerError> {
        write_file_phase_on(&self.pool, job_id, filename, phase, content_type).await
    }

    /// Seed a job row's submitter columns with raw values, for tests that
    /// stand in for a row another build (or a hand edit) wrote.
    #[cfg(test)]
    pub(crate) async fn seed_job_submitter_columns(
        &self,
        job_id: &str,
        address: &str,
        name: &str,
    ) -> Result<(), ServerError> {
        sqlx::query("UPDATE jobs SET submitted_by = ?, submitted_by_name = ? WHERE job_id = ?")
            .bind(address)
            .bind(name)
            .bind(job_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Seed a `file_statuses` row with raw column values, for tests that
    /// stand in for a row another build (or a hand edit) wrote. Production
    /// writes go through [`Self::update_file_status`].
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn seed_file_status_row(
        &self,
        job_id: &str,
        filename: &str,
        status: &str,
        error: Option<&str>,
        error_category: Option<&str>,
        content_type: Option<&str>,
        started_at: Option<MachineTime>,
        finished_at: Option<MachineTime>,
        next_eligible_at: Option<MachineTime>,
    ) -> Result<(), ServerError> {
        sqlx::query(
            "UPDATE file_statuses
             SET status = ?,
                 error = ?,
                 error_category = ?,
                 content_type = COALESCE(?, content_type),
                 started_at = ?,
                 finished_at = ?,
                 next_eligible_at = ?
             WHERE job_id = ? AND filename = ?",
        )
        .bind(status)
        .bind(error)
        .bind(error_category)
        .bind(content_type)
        .bind(started_at)
        .bind(finished_at)
        .bind(next_eligible_at)
        .bind(job_id)
        .bind(filename)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Seed a job row's `status` column with a raw value, for tests that
    /// stand in for a row another build wrote.
    #[cfg(test)]
    pub(crate) async fn seed_job_status_column(
        &self,
        job_id: &str,
        status: &str,
    ) -> Result<(), ServerError> {
        sqlx::query("UPDATE jobs SET status = ? WHERE job_id = ?")
            .bind(status)
            .bind(job_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Make every later `file_statuses` update fail, for tests of a write
    /// that fails (a full disk, a locked database).
    #[cfg(test)]
    pub(crate) async fn fail_file_status_updates(&self) -> Result<(), ServerError> {
        sqlx::query(
            "CREATE TRIGGER fail_file_status_updates BEFORE UPDATE ON file_statuses
             BEGIN SELECT RAISE(ABORT, 'file_statuses updates fail in this test'); END",
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Delete a job row and its associated `file_statuses` rows.
    ///
    /// Relies on `ON DELETE CASCADE` in the `file_statuses` foreign key.
    pub async fn delete_job(&self, job_id: &crate::api::JobId) -> Result<(), ServerError> {
        sqlx::query("DELETE FROM jobs WHERE job_id = ?")
            .bind(job_id.as_ref())
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Finalize a previously inserted attempt row.
    pub async fn finish_attempt(
        &self,
        attempt_id: &str,
        outcome: AttemptOutcome,
        failure_category: Option<FailureCategory>,
        disposition: RetryDisposition,
        finished_at: MachineTime,
    ) -> Result<(), ServerError> {
        sqlx::query(
            "UPDATE attempts
             SET finished_at = ?,
                 outcome = ?,
                 failure_category = ?,
                 disposition = ?
             WHERE attempt_id = ?",
        )
        .bind(finished_at)
        .bind(outcome.to_string())
        .bind(failure_category.map(|category| category.to_string()))
        .bind(disposition.to_string())
        .bind(attempt_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}
