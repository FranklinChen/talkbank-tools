//! One file's attempts under the default retry policy, as a typestate.
//!
//! A file task runs an attempt, and a failed attempt either ends the file
//! (its terminal state recorded) or is retried after the policy's backoff.
//! Three dispatch paths (audio commands, benchmark, media analysis) used to
//! write that loop by hand, each a `for attempt in 1..=max_attempts` that
//! fell out of its last iteration into an outcome nothing could reach, and
//! each remembering for itself to record the retry, wait the backoff and
//! restart the attempt, in that order.
//!
//! Here the order is the types. [`FileAttempts::failed`] consumes the attempt
//! after a transient failure and yields either [`AfterFailure::Recorded`]
//! (past the budget: the terminal error is written) or a [`RetryGrant`];
//! [`FileAttempts::fail`] records a final failure, which no type offers a
//! retry. The only way back to an attempt is [`RetryGrant::wait`], which
//! records the retry, waits and restarts. A grant exists only when the policy
//! allows another attempt, so the budget cannot be overrun, and a caller with
//! work to do before retrying (the audio path's timing recovery) does it
//! while holding the grant. Every exit that ends the file carries a
//! [`TerminalRecorded`], the task's proof that its state was written.

use crate::error::ServerError;
use crate::scheduling::{DurationMs, FailureCategory, NextAttempt, RetryPolicy, WorkUnitKind};

use super::tracker::TerminalRecorded;
use super::{FileRunTracker, FileStage};

/// A file whose attempt is running, numbered from one.
pub(crate) struct FileAttempts<'t, 'a> {
    lifecycle: &'t FileRunTracker<'a>,
    number: u32,
    work_unit_kind: WorkUnitKind,
    running_stage: FileStage,
}

/// What a transient failure led to.
pub(crate) enum AfterFailure<'t, 'a> {
    /// The policy allows another attempt.
    Retry(RetryGrant<'t, 'a>),
    /// The file's terminal failure is recorded; nothing more runs.
    Recorded(TerminalRecorded),
}

/// Permission for one more attempt, held between a failure and its retry.
pub(crate) struct RetryGrant<'t, 'a> {
    attempts: FileAttempts<'t, 'a>,
    backoff: DurationMs,
    category: FailureCategory,
    message: String,
}

impl<'t, 'a> FileAttempts<'t, 'a> {
    /// The first attempt: shows `running_stage` while it runs.
    pub(crate) async fn begin(
        lifecycle: &'t FileRunTracker<'a>,
        work_unit_kind: WorkUnitKind,
        running_stage: FileStage,
    ) -> Self {
        lifecycle.stage(running_stage).await;
        Self {
            lifecycle,
            number: 1,
            work_unit_kind,
            running_stage,
        }
    }

    /// The running attempt failed in a way the policy may retry, by
    /// `category` and the budget: a [`RetryGrant`] within it, the file's
    /// recorded error past it. `message` is what the operator reads, on the
    /// retry notice or the final error.
    pub(crate) async fn failed(
        self,
        category: FailureCategory,
        message: String,
    ) -> AfterFailure<'t, 'a> {
        match RetryPolicy::default().after_failure(self.number, category) {
            NextAttempt::RetryAfter(backoff) => AfterFailure::Retry(RetryGrant {
                attempts: self,
                backoff,
                category,
                message,
            }),
            NextAttempt::Stop(_) => AfterFailure::Recorded(self.fail(category, &message).await),
        }
    }

    /// The running attempt failed in a way no retry can fix: record the
    /// file's error.
    pub(crate) async fn fail(self, category: FailureCategory, message: &str) -> TerminalRecorded {
        self.lifecycle.fail(message, category).await
    }

    /// The running attempt failed with `error`. Only a worker's failure is
    /// retried at file level (the policy then decides from its category and
    /// the budget); any other error ends the file. The one statement of that
    /// rule for every dispatch path that retries files.
    pub(crate) async fn after_error(
        self,
        error: &ServerError,
        category: FailureCategory,
        message: String,
    ) -> AfterFailure<'t, 'a> {
        if matches!(error, ServerError::Worker(_)) {
            self.failed(category, message).await
        } else {
            AfterFailure::Recorded(self.fail(category, &message).await)
        }
    }
}

impl<'t, 'a> RetryGrant<'t, 'a> {
    /// Record the retry, wait out the backoff, and open the next attempt.
    pub(crate) async fn wait(self) -> FileAttempts<'t, 'a> {
        let Self {
            attempts,
            backoff,
            category,
            message,
        } = self;
        attempts
            .lifecycle
            .retry_after(
                backoff.duration(),
                category,
                &format!("{message}; retrying in {backoff} ms"),
            )
            .await;
        tokio::time::sleep(backoff.duration()).await;
        attempts
            .lifecycle
            .restart_attempt(attempts.work_unit_kind, attempts.running_stage)
            .await;
        FileAttempts {
            number: attempts.number + 1,
            ..attempts
        }
    }

    /// Give the retry up and record the file's terminal failure instead (the
    /// work done while holding the grant decided against retrying).
    pub(crate) async fn abandon(
        self,
        message: &str,
        category: FailureCategory,
    ) -> TerminalRecorded {
        self.attempts.fail(category, message).await
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::runner::util::file_status::test_sink::{RecordedError, RecordingSink};

    #[tokio::test(start_paused = true)]
    async fn a_transient_failure_is_retried_until_the_budget_and_then_recorded() {
        let sink = RecordingSink::default();
        let lifecycle = FileRunTracker::new(&sink, "a.cha");
        let mut attempts =
            FileAttempts::begin(&lifecycle, WorkUnitKind::FileInfer, FileStage::Processing).await;
        let mut retries = 0;
        loop {
            attempts = match attempts
                .failed(FailureCategory::WorkerCrash, "worker crashed".to_owned())
                .await
            {
                AfterFailure::Retry(grant) => {
                    retries += 1;
                    grant.wait().await
                }
                AfterFailure::Recorded(_) => break,
            };
        }
        // Three attempts by the default policy: two retries, each opening a
        // new attempt, then the third failure is recorded.
        assert_eq!(retries, 2);
        assert_eq!(sink.attempts().len(), 2);
        assert_eq!(
            sink.errors(),
            vec![RecordedError {
                filename: "a.cha".to_owned(),
                error: "worker crashed".to_owned(),
                category: FailureCategory::WorkerCrash,
            }]
        );
    }

    #[tokio::test]
    async fn a_failure_other_than_a_workers_is_recorded_at_once() {
        let sink = RecordingSink::default();
        let lifecycle = FileRunTracker::new(&sink, "a.cha");
        let attempts =
            FileAttempts::begin(&lifecycle, WorkUnitKind::FileInfer, FileStage::Processing).await;
        let error = ServerError::Validation("not valid CHAT".to_owned());
        let after = attempts
            .after_error(
                &error,
                FailureCategory::Validation,
                "not valid CHAT".to_owned(),
            )
            .await;
        assert!(matches!(after, AfterFailure::Recorded(_)));
        assert!(sink.attempts().is_empty());
        assert_eq!(sink.errors()[0].category, FailureCategory::Validation);
    }

    #[tokio::test]
    async fn an_abandoned_grant_records_its_reason_and_opens_no_attempt() {
        let sink = RecordingSink::default();
        let lifecycle = FileRunTracker::new(&sink, "a.cha");
        let attempts =
            FileAttempts::begin(&lifecycle, WorkUnitKind::FileInfer, FileStage::Processing).await;
        let AfterFailure::Retry(grant) = attempts
            .failed(FailureCategory::WorkerCrash, "worker crashed".to_owned())
            .await
        else {
            panic!("a first transient failure is retryable");
        };
        let _recorded = grant
            .abandon("Timing recovery cancelled", FailureCategory::Cancelled)
            .await;
        assert!(sink.attempts().is_empty());
        assert_eq!(sink.errors()[0].error, "Timing recovery cancelled");
    }
}
