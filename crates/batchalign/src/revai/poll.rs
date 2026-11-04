//! Waiting for a submitted Rev.AI job, as a typestate.
//!
//! A Rev.AI job is asynchronous: the upload answers with a job id, and the
//! transcript (or language-id result) is ready some time later. Until
//! 2026-11-03 the wait was a bare `loop` in the client that polled the job's
//! status until Rev.AI said `transcribed` or `failed`. That loop permitted two
//! wrong values, and both were met in production:
//!
//! - **No deadline.** A job Rev.AI reported `in_progress` forever held the
//!   file, and the worker thread polling it, forever.
//! - **A transient poll failure ended the job.** The status request's `?`
//!   returned the first dropped connection, 5xx or 429 as the job's failure.
//!   The control plane classified it transient and retried the FILE, which
//!   re-submitted the media and paid Rev.AI a second time for a job that was
//!   still running.
//!
//! This module makes both unrepresentable. The wait is a graph of states with
//! one transition each, and the states are the signatures:
//!
//! ```text
//! SubmittedJob ──begin(policy)──► Polling ──step──► Polling (in progress, or
//!                                    │               a transient failure that
//!                                    │               the budget absorbed)
//!                                    └──► PollEnd::Ready(job, result)
//!                                         PollEnd::Unfinished(DeadlineExceeded)
//!                                         PollEnd::Unfinished(TransientBudgetExhausted)
//!                                         PollEnd::Rejected(TerminalFailure)
//!                                         PollEnd::Refused(AccountRefusal)
//! ```
//!
//! A `Polling` cannot exist without a [`PollPolicy`], so a wait without a
//! deadline has no route; and a transient failure reaches the end only through
//! [`TransientBudget`], so one cannot end the wait. An unfinished job is NOT
//! an error: it is a paid job that is still Rev.AI's, returned by name
//! ([`UnfinishedRevJob`]) so the evidence cache can record it and a later run
//! can resume the wait instead of uploading again.
//!
//! The verdict on a failure is [`RevAiError::classify`], the client's one
//! table; nothing here re-decides what is transient.

use std::marker::PhantomData;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::client::{
    AccountRefusal, ClassifiedFailure, Result, RevAiClient, TerminalFailure, TransientFailure,
};
use crate::api::RevAiJobId;

/// How long one wait may run in total, from the first poll.
///
/// A typed duration: a bare `Duration` could be any of the intervals below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct PollDeadline(Duration);

impl PollDeadline {
    /// A deadline of `duration`.
    pub(crate) const fn of(duration: Duration) -> Self {
        Self(duration)
    }

    /// The bound as a duration, for rendering and arithmetic.
    pub(crate) const fn duration(self) -> Duration {
        self.0
    }
}

/// How many transient poll failures one wait absorbs before it hands the job
/// back unfinished.
///
/// The budget bounds how long one worker keeps polling an endpoint that is
/// failing, which the deadline alone would allow for hours. Exhausting it does
/// not abandon the job: the wait ends in
/// [`UnfinishedCause::TransientBudgetExhausted`], job named.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct TransientBudget(u32);

impl TransientBudget {
    /// A budget of `failures` absorbed failures.
    pub(crate) const fn of(failures: u32) -> Self {
        Self(failures)
    }

    /// Whether `failures` transient failures exceed the budget.
    fn is_exhausted_by(self, failures: usize) -> bool {
        failures > self.0 as usize
    }
}

/// The schedule of waits between polls: start at `initial`, double every
/// three polls, never past `max`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PollInterval {
    pub(crate) initial: Duration,
    pub(crate) max: Duration,
}

impl PollInterval {
    /// Polls between each lengthening of the interval.
    const POLLS_PER_STEP: u32 = 3;

    /// The interval after `polls` polls have been made.
    fn after(self, polls: u32) -> Duration {
        let doublings = polls / Self::POLLS_PER_STEP;
        // Saturating: a wait that long would have met its deadline first,
        // but the arithmetic must not wrap into a short interval.
        let lengthened = self
            .initial
            .checked_mul(1u32.checked_shl(doublings).unwrap_or(u32::MAX))
            .unwrap_or(Duration::MAX);
        lengthened.min(self.max)
    }
}

/// The whole policy of one wait. A [`Polling`] cannot be built without one, so
/// no wait is unbounded, and a caller states which job kind's policy applies
/// rather than passing a bare number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PollPolicy {
    pub(crate) deadline: PollDeadline,
    pub(crate) transient_budget: TransientBudget,
    pub(crate) interval: PollInterval,
}

impl PollPolicy {
    /// The wait for a transcription job.
    ///
    /// Two hours is BA3's bound on how long one worker thread is held by one
    /// job, not a claim about Rev.AI's turnaround. Exceeding it is cheap by
    /// design: the job stays submitted and paid for, the evidence cache
    /// records it, and the next run of the same request resumes the wait
    /// instead of uploading again.
    pub(crate) const TRANSCRIPTION: Self = Self {
        deadline: PollDeadline::of(Duration::from_secs(2 * 60 * 60)),
        transient_budget: TransientBudget::of(5),
        interval: PollInterval {
            initial: Duration::from_secs(5),
            max: Duration::from_secs(30),
        },
    };

    /// The wait for a language-identification job, which precedes a
    /// transcription under detection. Shorter: an unfinished identification
    /// falls back to transcription-level detection, so the bound is on how
    /// long that fallback is delayed.
    pub(crate) const LANGUAGE_ID: Self = Self {
        deadline: PollDeadline::of(Duration::from_secs(20 * 60)),
        transient_budget: TransientBudget::of(5),
        interval: PollInterval {
            initial: Duration::from_secs(3),
            max: Duration::from_secs(30),
        },
    };
}

/// Where a wait's time comes from. Held by the client, so production has one
/// source and a test can run a deadline past in no wall time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PollTiming {
    /// Real time: waits sleep, and elapsed is measured on the monotonic clock.
    Real,
    /// Test only: waits do not sleep; each one advances a virtual clock by the
    /// interval it would have slept, so a deadline test takes no wall time.
    #[cfg(test)]
    Virtual,
}

/// The clock of one wait, started when polling begins.
#[derive(Debug)]
enum PollClock {
    Monotonic {
        started: Instant,
    },
    #[cfg(test)]
    Virtual {
        elapsed: Duration,
    },
}

impl PollClock {
    fn start(timing: PollTiming) -> Self {
        match timing {
            PollTiming::Real => Self::Monotonic {
                started: Instant::now(),
            },
            #[cfg(test)]
            PollTiming::Virtual => Self::Virtual {
                elapsed: Duration::ZERO,
            },
        }
    }

    /// Let `interval` pass.
    fn wait(&mut self, interval: Duration) {
        match self {
            Self::Monotonic { .. } => std::thread::sleep(interval),
            #[cfg(test)]
            Self::Virtual { elapsed } => *elapsed = elapsed.saturating_add(interval),
        }
    }

    /// How long this wait has run.
    fn elapsed(&self) -> Duration {
        match self {
            Self::Monotonic { started } => started.elapsed(),
            #[cfg(test)]
            Self::Virtual { elapsed } => *elapsed,
        }
    }
}

/// A job Rev.AI has accepted. The only way into [`Polling`].
///
/// Built from a submission's reply, or from the evidence cache's record of a
/// job an earlier run left unfinished; either way Rev.AI named the id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubmittedJob(RevAiJobId);

impl SubmittedJob {
    /// A job Rev.AI named, in a submission reply or an earlier run's record.
    pub(crate) fn named(job: RevAiJobId) -> Self {
        Self(job)
    }

    /// Start waiting under `policy`, on the client's clock.
    pub(crate) fn begin<K: PolledJob>(self, policy: PollPolicy, timing: PollTiming) -> Polling<K> {
        Polling {
            job: self.0,
            interval: policy.interval.after(0),
            policy,
            clock: PollClock::start(timing),
            polls: 0,
            transient_failures: Vec::new(),
            kind: PhantomData,
        }
    }
}

/// What one status poll says about a job that has not failed.
///
/// A job Rev.AI reports as failed is an `Err(RevAiError::JobFailed)` from
/// [`PolledJob::progress`], so the one verdict table classifies it (terminal)
/// with every other failure; nothing here decides that a failed job is final.
pub(crate) enum Progress<R> {
    /// Still being processed.
    InProgress,
    /// Done; `R` is what the status reply carries (a transcription's detected
    /// language), not the result itself, which is a separate download.
    Ready(R),
}

/// A kind of Rev.AI job that can be waited for: how its status is read.
pub(crate) trait PolledJob {
    /// What the ready status carries.
    type Ready;

    /// Ask Rev.AI how `job` is doing.
    fn progress(client: &RevAiClient, job: &RevAiJobId) -> Result<Progress<Self::Ready>>;
}

/// A wait in progress: the job, its policy, its clock, and what has happened
/// so far. Every field is private; the only way forward is [`Polling::run`].
#[derive(Debug)]
pub(crate) struct Polling<K: PolledJob> {
    job: RevAiJobId,
    policy: PollPolicy,
    clock: PollClock,
    /// The current wait between polls, lengthened by the schedule.
    interval: Duration,
    /// Polls made so far, which drive the schedule.
    polls: u32,
    /// Every transient poll failure so far, kept whole: an operator reading an
    /// exhausted budget can tell three resets from a 502 and two resets.
    transient_failures: Vec<TransientFailure>,
    kind: PhantomData<K>,
}

/// One step's result: the wait goes on, or it has ended.
enum PollStep<K: PolledJob> {
    Continue(Polling<K>),
    Done(PollEnd<K::Ready>),
}

/// How a wait ended. Exhaustive: every way out of [`Polling`] is named here.
pub(crate) enum PollEnd<R> {
    /// Rev.AI finished the job; `R` is what its status carried.
    Ready(RevAiJobId, R),
    /// The wait ended with the job still Rev.AI's, and paid for.
    Unfinished(UnfinishedRevJob),
    /// Rev.AI failed the job, or refused the poll in a way that will not
    /// change. The job is over.
    Rejected(TerminalFailure),
    /// Rev.AI refuses the account. No poll from it succeeds until an account
    /// holder acts; the job itself may well be fine.
    Refused(AccountRefusal),
}

impl<K: PolledJob> Polling<K> {
    /// Poll until the wait ends.
    pub(crate) fn run(mut self, client: &RevAiClient) -> PollEnd<K::Ready> {
        loop {
            match self.step(client) {
                PollStep::Continue(next) => self = next,
                PollStep::Done(end) => return end,
            }
        }
    }

    /// One poll and its consequence.
    fn step(mut self, client: &RevAiClient) -> PollStep<K> {
        self.polls = self.polls.saturating_add(1);
        match K::progress(client, &self.job) {
            Ok(Progress::Ready(ready)) => PollStep::Done(PollEnd::Ready(self.job, ready)),
            Ok(Progress::InProgress) => self.wait(),
            Err(error) => match error.classify() {
                // The socket or the service, not the job: absorbed by the
                // budget and polled again after the interval.
                ClassifiedFailure::Transient(failure) => {
                    self.transient_failures.push(failure);
                    if self
                        .policy
                        .transient_budget
                        .is_exhausted_by(self.transient_failures.len())
                    {
                        return PollStep::Done(PollEnd::Unfinished(UnfinishedRevJob {
                            job: self.job,
                            cause: UnfinishedCause::TransientBudgetExhausted {
                                budget: self.policy.transient_budget,
                                failures: self
                                    .transient_failures
                                    .iter()
                                    .map(ToString::to_string)
                                    .collect(),
                            },
                        }));
                    }
                    self.wait()
                }
                ClassifiedFailure::Terminal(failure) => PollStep::Done(PollEnd::Rejected(failure)),
                ClassifiedFailure::AccountRefused(refusal) => {
                    PollStep::Done(PollEnd::Refused(refusal))
                }
            },
        }
    }

    /// Let the interval pass, lengthen it per the schedule, and check the
    /// deadline. The deadline is checked here, after every wait, so a wait
    /// that has run out ends before it polls again.
    fn wait(mut self) -> PollStep<K> {
        self.clock.wait(self.interval);
        self.interval = self.policy.interval.after(self.polls);
        let elapsed = self.clock.elapsed();
        if elapsed >= self.policy.deadline.duration() {
            return PollStep::Done(PollEnd::Unfinished(UnfinishedRevJob {
                job: self.job,
                cause: UnfinishedCause::DeadlineExceeded {
                    deadline: self.policy.deadline,
                    elapsed,
                },
            }));
        }
        PollStep::Continue(self)
    }
}

/// What a caller gets back from a bounded wait: the result, or the job by
/// name because the wait ended first. Never a transient failure dressed as
/// the job's failure.
pub(crate) enum Awaited<R> {
    Ready(R),
    Unfinished(UnfinishedRevJob),
}

/// A paid Rev.AI job whose wait ended before Rev.AI finished it.
///
/// Carries the job id so the evidence cache can record it and a later run can
/// resume the wait rather than upload again. Serializable because that record
/// is durable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct UnfinishedRevJob {
    pub(crate) job: RevAiJobId,
    pub(crate) cause: UnfinishedCause,
}

/// Why a wait ended with the job unfinished.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum UnfinishedCause {
    /// The policy's deadline passed while Rev.AI still reported progress.
    DeadlineExceeded {
        deadline: PollDeadline,
        elapsed: Duration,
    },
    /// More transient poll failures than the policy absorbs, each rendered
    /// with its cause chain, in order.
    TransientBudgetExhausted {
        budget: TransientBudget,
        failures: Vec<String>,
    },
}

impl std::fmt::Display for UnfinishedCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeadlineExceeded { deadline, elapsed } => write!(
                f,
                "still in progress after {}s, past the {}s polling deadline",
                elapsed.as_secs(),
                deadline.duration().as_secs()
            ),
            Self::TransientBudgetExhausted { budget, failures } => write!(
                f,
                "{} transient status-poll failures exceeded the budget of {}: {}",
                failures.len(),
                budget.0,
                failures
                    .iter()
                    .enumerate()
                    .map(|(index, failure)| format!("poll failure {}: {failure}", index + 1))
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
        }
    }
}

impl std::fmt::Display for UnfinishedRevJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Rev.AI job {} is unfinished ({}); the job stays submitted and paid for",
            self.job, self.cause
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The schedule: the interval doubles every three polls and never passes
    /// its cap, so a long wait polls less often without going quiet.
    #[test]
    fn the_interval_doubles_every_three_polls_up_to_its_cap() {
        let schedule = PollInterval {
            initial: Duration::from_secs(5),
            max: Duration::from_secs(30),
        };
        assert_eq!(schedule.after(0), Duration::from_secs(5));
        assert_eq!(schedule.after(2), Duration::from_secs(5));
        assert_eq!(schedule.after(3), Duration::from_secs(10));
        assert_eq!(schedule.after(6), Duration::from_secs(20));
        assert_eq!(schedule.after(9), Duration::from_secs(30));
        assert_eq!(schedule.after(u32::MAX), Duration::from_secs(30));
    }

    /// The budget is the number of failures absorbed: one more exhausts it.
    #[test]
    fn a_budget_absorbs_exactly_its_count() {
        let budget = TransientBudget(2);
        assert!(!budget.is_exhausted_by(0));
        assert!(!budget.is_exhausted_by(2));
        assert!(budget.is_exhausted_by(3));
    }

    /// The durable record of an unfinished job reads back as written, so the
    /// cache can store the cause beside the job id.
    #[test]
    fn an_unfinished_job_round_trips_through_json() {
        let job = UnfinishedRevJob {
            job: RevAiJobId::try_new("job-7").expect("a job id"),
            cause: UnfinishedCause::TransientBudgetExhausted {
                budget: TransientBudget(1),
                failures: vec!["HTTP error: reset".to_owned(), "HTTP 503".to_owned()],
            },
        };
        let stored = serde_json::to_value(&job).expect("serializes");
        assert_eq!(stored["cause"]["kind"], "transient_budget_exhausted");
        let replayed: UnfinishedRevJob = serde_json::from_value(stored).expect("deserializes");
        assert_eq!(replayed, job);
        assert!(replayed.to_string().contains("job-7"), "{replayed}");
    }
}
