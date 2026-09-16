//! One owner for every deadline a live-harness test waits on.
//!
//! # The defect this exists to remove
//!
//! Every live suite used to spell its own wait: a bare
//! `Instant::now() + Duration::from_secs(60)`, an `assert!` that the deadline
//! had not passed, and a `sleep`. Fifteen such literals were counted on
//! 2026-09-16. A fixed deadline cannot tell "this job is hung" from "this job
//! has not started because another test's worker holds the process-global spawn
//! permit", so it fires on CONTENTION and reports it as failure. That happened
//! twice in consecutive change sets: `cli_morphotag_real_server` failed at its
//! deadline with `1/1 local startup slots in use`, then passed alone in about
//! 12.65 seconds. A suite result that can mean either thing means nothing.
//!
//! # What replaces it, and why it is not just a longer timeout
//!
//! A wait here carries TWO bounds, and the pair is the point:
//!
//! - an IDLE WINDOW, the longest the wait may go without observing any change
//!   in what it is watching. Observed progress resets it. A genuinely hung
//!   subject produces no change, so the idle window still fires, and fires
//!   at roughly the same time it always did.
//! - a CEILING, the longest the wait may run no matter how much progress it
//!   observes. No amount of progress can extend past it, so a subject that
//!   crawls forever is still bounded.
//!
//! So a test that is making progress is not killed for being slow, and a test
//! that is stuck is still killed for being stuck. Raising a single number
//! could not buy both.
//!
//! # Progress is observed, never assumed
//!
//! A [`ProgressSnapshot`] is built only from a real probe result through the
//! constructors below; there is no way to hand one a literal. Passing a value
//! that never changes degrades the wait to its idle window, which is the safe
//! direction. Passing a value that changes every probe (a clock, say) extends
//! the wait, which the ceiling bounds. Both wrong values are bounded by
//! construction rather than by a reviewer noticing.
//!
//! # Why a subject carries its own budget
//!
//! [`WaitSubject`] pairs what is being waited for with the budget that wait
//! gets, and [`WaitBudget`] is private. A caller therefore cannot pick a
//! job-completion budget for a control-plane flip, and cannot write a bare
//! `Duration` at a call site at all: the only way to get one is to name the
//! thing being waited for.
//!
//! # The refusal
//!
//! A spent budget names what the wait was for, which bound was hit, how long
//! it ran, what progress it last saw and how long ago, and how many probes and
//! changes it observed. The failures this replaces said only "did not finish
//! within 60s", which is what made two of them get diagnosed as the bound.
// Integration tests are exempt from the crate's deny-level panic lints,
// matching the src/lib.rs `#![cfg_attr(test, allow(...))]` pattern
// (see docs/panic-audit/).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented
)]
// Each suite compiles this module but uses only the subjects it waits on.
#![allow(dead_code)]

use std::fmt;
use std::time::{Duration, Instant};

use batchalign::api::JobInfo;

// ---------------------------------------------------------------------------
// Budgets
// ---------------------------------------------------------------------------

/// How long one wait may stand still, how long it may run at all, and how
/// often it probes.
///
/// Private on purpose: a budget is reached only through a [`WaitSubject`], so
/// the set of (subject, budget) pairings is closed and a call site cannot
/// invent a new one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WaitBudget {
    /// Longest the wait may go without observing any change.
    idle_window: Duration,
    /// Longest the wait may run regardless of observed progress.
    ceiling: Duration,
    /// Gap between probes.
    poll_interval: Duration,
}

/// A job running on test-echo or live workers until it reaches a terminal
/// state.
///
/// The idle window is the old fixed deadline: a job whose status and completed
/// count stop moving for a minute is stuck, exactly as before. The ceiling
/// allows a job that keeps completing files to take ten minutes on a loaded
/// box without being called a failure.
const JOB_COMPLETION: WaitBudget = WaitBudget {
    idle_window: Duration::from_secs(60),
    ceiling: Duration::from_secs(600),
    poll_interval: Duration::from_millis(200),
};

/// A job on the ML fixture, where one file can legitimately occupy a worker
/// for minutes while a model runs.
const ML_JOB_COMPLETION: WaitBudget = WaitBudget {
    idle_window: Duration::from_secs(300),
    ceiling: Duration::from_secs(1_800),
    poll_interval: Duration::from_millis(500),
};

/// A control-plane transition: a cancel flipping a status, a restart being
/// accepted, a job reaching its first completed file.
///
/// Short and tight, because these are server-local state changes that involve
/// no model and no spawn. A machine busy enough to delay one for twenty
/// seconds has a problem this test is not measuring.
const CONTROL_PLANE_TRANSITION: WaitBudget = WaitBudget {
    idle_window: Duration::from_secs(20),
    ceiling: Duration::from_secs(120),
    poll_interval: Duration::from_millis(25),
};

/// A worker becoming visible after a dispatch asked for one.
///
/// This is the wait that cannot be a fixed number. `memory_guard` holds ONE
/// process-global spawn permit until a worker reports ready, so a spawn here
/// queues behind every other test binary's cold model load. The idle window
/// is measured against the admission state (free permits, live workers), so
/// waiting while the queue ahead is moving is not counted against it, while a
/// dispatch that never registers and never changes the admission picture still
/// fails inside the window.
const WORKER_ADMISSION: WaitBudget = WaitBudget {
    idle_window: Duration::from_secs(30),
    ceiling: Duration::from_secs(300),
    poll_interval: Duration::from_millis(50),
};

/// A killed worker process leaving the process table.
const WORKER_PROCESS_EXIT: WaitBudget = WaitBudget {
    idle_window: Duration::from_secs(5),
    ceiling: Duration::from_secs(30),
    poll_interval: Duration::from_millis(100),
};

// ---------------------------------------------------------------------------
// Subjects
// ---------------------------------------------------------------------------

/// What one wait is for, and therefore what budget it gets.
///
/// One constructor per kind of wait. The budget is a consequence of the
/// subject rather than a second argument, so a subject and a budget cannot
/// disagree.
pub struct WaitSubject {
    /// Operator-facing description, printed in the refusal.
    what: String,
    /// The budget this kind of wait gets.
    budget: WaitBudget,
}

impl WaitSubject {
    /// A job reaching a terminal state on test-echo or fast live workers.
    ///
    /// The id is taken as `impl Display` rather than `&str` so a call site can
    /// pass whichever of `JobId` or `&String` it is holding without a
    /// conversion whose only purpose is to satisfy this signature.
    pub fn job_completion(job_id: impl fmt::Display) -> Self {
        Self {
            what: format!("job {job_id} to reach a terminal state"),
            budget: JOB_COMPLETION,
        }
    }

    /// A job reaching a terminal state on the ML fixture's real models.
    pub fn ml_job_completion(job_id: impl fmt::Display) -> Self {
        Self {
            what: format!("live-model job {job_id} to reach a terminal state"),
            budget: ML_JOB_COMPLETION,
        }
    }

    /// A job getting far enough into its run to open a race window.
    pub fn job_starts_making_progress(job_id: impl fmt::Display) -> Self {
        Self {
            what: format!("job {job_id} to complete its first file and still have more"),
            budget: CONTROL_PLANE_TRANSITION,
        }
    }

    /// A restart being accepted after an asynchronous cancel flips the status.
    pub fn restart_accepted(job_id: impl fmt::Display) -> Self {
        Self {
            what: format!("POST /jobs/{job_id}/restart to be accepted after cancel"),
            budget: CONTROL_PLANE_TRANSITION,
        }
    }

    /// A restarted job settling into a terminal state under its new runner.
    pub fn restarted_job_settles(job_id: impl fmt::Display) -> Self {
        Self {
            what: format!("restarted job {job_id} to settle"),
            budget: JOB_COMPLETION,
        }
    }

    /// A dispatch registering the worker it checked out for a job.
    pub fn worker_registered_for_job(job_id: impl fmt::Display) -> Self {
        Self {
            what: format!("a dispatch to register a worker for job {job_id}"),
            budget: WORKER_ADMISSION,
        }
    }

    /// A worker process actually leaving the process table after a kill.
    pub fn worker_process_exit(pid: u32) -> Self {
        Self {
            what: format!("worker pid {pid} to exit"),
            budget: WORKER_PROCESS_EXIT,
        }
    }
}

// ---------------------------------------------------------------------------
// Observed progress
// ---------------------------------------------------------------------------

/// One observation of whatever a wait is watching.
///
/// The inner value is private and there is no constructor from a literal: a
/// snapshot exists only because something real was probed. Two snapshots
/// comparing equal means the subject did not move between probes, which is
/// what the idle window measures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgressSnapshot(String);

impl ProgressSnapshot {
    /// What a job projection says about its own progress.
    ///
    /// Status, completed count and current file together: a job that is
    /// working its way through files changes this on every file even while its
    /// status stays `Running`, which is exactly the progress a fixed deadline
    /// could not see.
    pub fn job(info: &JobInfo) -> Self {
        Self(format!(
            "status={:?} completed={}/{} current={:?}",
            info.status, info.completed_files, info.total_files, info.current_file
        ))
    }

    /// The HTTP status a control-plane request answered with.
    pub fn http_status(status: u16) -> Self {
        Self(format!("http={status}"))
    }

    /// The pool's worker-admission picture.
    ///
    /// Free spawn permits and live worker count: while the machine is working
    /// through a spawn queue at least one of these moves, so a wait behind
    /// other binaries' model loads reads as progress rather than as a stall.
    pub fn spawn_admission(free_spawn_permits: usize, live_workers: usize) -> Self {
        Self(format!(
            "free_spawn_permits={free_spawn_permits} live_workers={live_workers}"
        ))
    }

    /// Whether a process is still in the process table.
    pub fn process_alive(alive: bool) -> Self {
        Self(format!("alive={alive}"))
    }
}

impl fmt::Display for ProgressSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// The deadline
// ---------------------------------------------------------------------------

/// Which bound a spent wait hit. Distinct variants because they mean
/// different things to whoever reads the failure: an idle window says the
/// subject stopped, a ceiling says it never finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpiryCause {
    /// No observed change for the whole idle window.
    StoodStill,
    /// Ran past the ceiling even though it kept moving.
    RanTooLong,
}

/// A wait whose budget is spent, carrying everything needed to diagnose it
/// without rerunning anything.
#[derive(Debug)]
pub struct WaitExpired {
    what: String,
    cause: ExpiryCause,
    elapsed: Duration,
    idle_window: Duration,
    ceiling: Duration,
    last_seen: ProgressSnapshot,
    since_last_change: Duration,
    probes: u32,
    changes: u32,
}

impl fmt::Display for WaitExpired {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let headline = match self.cause {
            ExpiryCause::StoodStill => format!(
                "stood still for {:?} (idle window {:?})",
                self.since_last_change, self.idle_window
            ),
            ExpiryCause::RanTooLong => {
                format!("ran past its {:?} ceiling", self.ceiling)
            }
        };
        write!(
            f,
            "waited for {what}: {headline}.\n\
             \x20 total elapsed: {elapsed:?}\n\
             \x20 last observed progress: {last_seen} ({since:?} ago)\n\
             \x20 probes: {probes}, observed changes: {changes}\n\
             This is the live-harness deadline owner (tests/live_deadline). It \
             extends while it can SEE progress and refuses when it cannot, so \
             this is a stalled subject rather than a bound that was too small. \
             Raising a number is not the fix; read what it was waiting for and \
             what it last saw.",
            what = self.what,
            headline = headline,
            elapsed = self.elapsed,
            last_seen = self.last_seen,
            since = self.since_last_change,
            probes = self.probes,
            changes = self.changes,
        )
    }
}

/// One live-harness wait: the deadline owner every polling test uses instead
/// of a bare literal.
pub struct ServerTestDeadline {
    subject: WaitSubject,
    started: Instant,
    last_change: Instant,
    last_seen: Option<ProgressSnapshot>,
    probes: u32,
    changes: u32,
}

impl ServerTestDeadline {
    /// Start waiting for `subject`. The budget comes with the subject.
    pub fn new(subject: WaitSubject) -> Self {
        let now = Instant::now();
        Self {
            subject,
            started: now,
            last_change: now,
            last_seen: None,
            probes: 0,
            changes: 0,
        }
    }

    /// How long to sleep between probes for this subject.
    pub fn poll_interval(&self) -> Duration {
        self.subject.budget.poll_interval
    }

    /// The pure core: record one probe taken at `now` and decide whether the
    /// wait may continue.
    ///
    /// Separated from the sleeping wrapper so the extend-and-ceiling algorithm
    /// is testable without a clock, which is what the unit tests below do.
    pub fn observe_at(
        &mut self,
        now: Instant,
        seen: ProgressSnapshot,
    ) -> Result<(), Box<WaitExpired>> {
        self.probes = self.probes.saturating_add(1);
        let changed = self.last_seen.as_ref() != Some(&seen);
        if changed {
            if self.last_seen.is_some() {
                self.changes = self.changes.saturating_add(1);
            }
            self.last_change = now;
        }
        self.last_seen = Some(seen);

        let elapsed = now.saturating_duration_since(self.started);
        let since_last_change = now.saturating_duration_since(self.last_change);
        let budget = self.subject.budget;

        // Ceiling first: it is the bound no amount of progress may cross, so
        // a subject that keeps moving forever is reported as running too long
        // rather than as standing still.
        let cause = if elapsed >= budget.ceiling {
            Some(ExpiryCause::RanTooLong)
        } else if since_last_change >= budget.idle_window {
            Some(ExpiryCause::StoodStill)
        } else {
            None
        };

        match cause {
            None => Ok(()),
            Some(cause) => Err(Box::new(WaitExpired {
                what: self.subject.what.clone(),
                cause,
                elapsed,
                idle_window: budget.idle_window,
                ceiling: budget.ceiling,
                #[allow(clippy::expect_used)]
                last_seen: self
                    .last_seen
                    .clone()
                    .expect("a snapshot was just recorded above"),
                since_last_change,
                probes: self.probes,
                changes: self.changes,
            })),
        }
    }

    /// Record one probe and wait for the next, or fail the test naming what
    /// the wait was for and what progress it last saw.
    ///
    /// This is the whole loop body the call sites used to spell out as an
    /// `assert!` on a bare deadline plus a `sleep` on a bare interval.
    pub async fn keep_waiting(&mut self, seen: ProgressSnapshot) {
        match self.observe_at(Instant::now(), seen) {
            Ok(()) => tokio::time::sleep(self.poll_interval()).await,
            Err(expired) => panic!("{expired}"),
        }
    }
}

// ---------------------------------------------------------------------------
// One-shot budgets
// ---------------------------------------------------------------------------

/// How much longer than the subject's own budget the harness waits before
/// killing it.
///
/// A killer that fires FIRST turns every diagnosable refusal into a signal
/// death, so this is the margin that keeps the subject's own error path
/// reachable: enough for the command to notice its wait expired, print its
/// reason and exit. It is a margin, never a budget in its own right, which is
/// why it is added to a real one rather than written at a call site.
const SUBJECT_REFUSAL_GRACE: Duration = Duration::from_secs(30);

/// How long one `batchalign3` subprocess may run before the harness kills it.
///
/// Deliberately NOT a [`ServerTestDeadline`]: `assert_cmd` hands the caller
/// nothing until the child exits, so there is no progress for a wait to
/// observe and nothing an extending deadline could key off. What this type
/// buys is the other half: no bare literal at a call site, and one place where
/// the budget for each kind of run is written down and can be re-judged.
///
/// # A subprocess that owns its own wait does not get a second number
///
/// [`Self::DaemonStart`] is DERIVED from the budget the command itself waits
/// on, not written here. When a subprocess already bounds itself, the harness
/// kill is a backstop for a process that stopped being a process at all, and it
/// is only a backstop while it is strictly later than the subject's own
/// refusal. Two independently written numbers cannot hold that; an addition
/// can, in the one direction that matters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliRunBudget {
    /// Argument parsing, `--help`, and validation failures that never reach a
    /// server.
    ArgumentValidation,
    /// A refusal that must happen without any network round trip at all.
    ImmediateRefusal,
    /// A command that talks to a local server and comes back.
    ServerRoundTrip,
    /// `serve start`, which spawns a daemon and waits for it to publish the
    /// port it bound.
    ///
    /// This is symptom two's number. The spawn used to run on
    /// [`Self::ServerRoundTrip`]'s 60 s, while `serve start` waits up to
    /// `daemon::startup_budget()` (90 s) for the handshake. The harness killer
    /// was therefore 30 s SHORTER than the wait it was supervising, so a
    /// contended machine produced a SIGKILL at 60 s with an empty `server.log`
    /// and wait status 9: indistinguishable from a daemon that never started,
    /// and reported at the helper's assert rather than at anything the test was
    /// about. Measured across four runs on byte-identical code the outcome went
    /// fail, pass, fail, fail, with two different tests failing at that one
    /// line, whenever three or four live tests ran past 60 s concurrently.
    ///
    /// Deriving it from `startup_budget()` is what makes the two cases
    /// distinguishable again: a daemon that is merely slow is no longer killed
    /// before its own wait ends, while a daemon that never comes up is refused
    /// BY `serve start` itself, at 90 s, with the phase it was stuck in named.
    /// The total stays bounded, and the kill stays in place for a CLI that
    /// stops answering at all.
    DaemonStart,
    /// `serve stop`, which signals a daemon and waits for it to go.
    ServeStop,
    /// A command that loads and runs a real model.
    LiveModelRun,
}

impl CliRunBudget {
    /// The wall-clock budget for this kind of run.
    pub fn as_duration(self) -> Duration {
        match self {
            Self::ArgumentValidation => Duration::from_secs(30),
            Self::ImmediateRefusal => Duration::from_secs(3),
            Self::ServerRoundTrip => Duration::from_secs(60),
            // Derived, never restated: the daemon's own startup budget plus the
            // margin it needs to report its own failure. A number written here
            // could silently fall under `startup_budget()` again.
            Self::DaemonStart => batchalign::cli::daemon::startup_budget() + SUBJECT_REFUSAL_GRACE,
            Self::ServeStop => Duration::from_secs(10),
            Self::LiveModelRun => Duration::from_secs(300),
        }
    }
}

/// Budgets for harness operations that are neither a poll loop nor a
/// subprocess: one-shot waits whose number still should not be a literal
/// scattered across fixtures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HarnessBudget {
    /// How long a fixture session's `shutdown_for_reuse` may take.
    SessionShutdown,
    /// How long one oversubscribed test-echo dispatch may take.
    ///
    /// Oversubscription is the point of the test that uses this: most callers
    /// are meant to queue. The budget bounds the queue, and the test asserts
    /// on how many completed, so this is a cap on the experiment rather than a
    /// deadline on one subject.
    OversubscribedEchoDispatch,
    /// How long a test fixture's worker may take to be admitted AND report
    /// ready.
    ///
    /// This is symptom one's number, and the decomposition matters.
    /// `memory_guard::acquire_spawn_permit` passes `ready_timeout_s` to the
    /// host-memory lease wait AND then to the readiness wait, so one number
    /// buys both: winning a startup slot from every other test binary on the
    /// machine, and this worker's own Python startup. The fixtures used to set
    /// 120s, which is ample for the second and not for the first: on a shared
    /// machine a single process-global spawn permit can be held for minutes by
    /// another suite's cold model load, and the failure surfaced as
    /// `timed out waiting for host-memory capacity ... 1/1 local startup slots
    /// in use`, which reads as a defect in the test that lost the race.
    ///
    /// The value below covers contention plus startup rather than startup
    /// alone. It is not a blanket "longer timeout": a worker that genuinely
    /// never comes up still fails here, and every wait the harness itself owns
    /// is bounded by [`ServerTestDeadline`]'s idle window instead.
    FixtureWorkerReady,
}

impl HarnessBudget {
    /// The wall-clock budget for this operation.
    pub fn as_duration(self) -> Duration {
        match self {
            Self::SessionShutdown => Duration::from_secs(5),
            Self::OversubscribedEchoDispatch => Duration::from_secs(60),
            Self::FixtureWorkerReady => Duration::from_secs(300),
        }
    }

    /// The same budget in whole seconds, for the config fields that take one.
    pub fn as_secs(self) -> u64 {
        self.as_duration().as_secs()
    }
}

// ---------------------------------------------------------------------------
// Tests for the algorithm itself
// ---------------------------------------------------------------------------
//
// The type cannot prove that its own arithmetic extends on change and stops at
// the ceiling, so that part is tested. Everything the type DOES prove (a
// subject cannot be paired with the wrong budget, a snapshot cannot be forged
// from a literal, a call site cannot write a bare Duration) is not tested,
// because a test of it would not compile.

#[cfg(test)]
mod tests {
    use super::*;

    /// A snapshot that differs from the default one, so a test can make the
    /// subject "move" without a real probe.
    fn moved(n: usize) -> ProgressSnapshot {
        ProgressSnapshot::spawn_admission(n, n)
    }

    #[test]
    fn a_subject_that_keeps_moving_survives_past_the_idle_window() {
        let mut deadline = ServerTestDeadline::new(WaitSubject::job_completion("j1"));
        let start = deadline.started;
        // Probe every 40 seconds, changing each time. Each probe is beyond the
        // 60s idle window measured from the START, but not from the previous
        // change, which is the whole property.
        for step in 1..=5u32 {
            let now = start + Duration::from_secs(u64::from(step) * 40);
            deadline
                .observe_at(now, moved(step as usize))
                .expect("observed progress must extend the wait");
        }
    }

    #[test]
    fn a_subject_that_stops_moving_fails_inside_the_idle_window() {
        let mut deadline = ServerTestDeadline::new(WaitSubject::job_completion("j2"));
        let start = deadline.started;
        deadline
            .observe_at(start, moved(1))
            .expect("first probe is always fine");
        deadline
            .observe_at(start + Duration::from_secs(59), moved(1))
            .expect("still inside the idle window");
        let expired = deadline
            .observe_at(start + Duration::from_secs(61), moved(1))
            .expect_err("an unchanged subject must fail once the idle window passes");
        assert_eq!(expired.cause, ExpiryCause::StoodStill);
        assert_eq!(expired.changes, 0);
    }

    #[test]
    fn progress_cannot_extend_a_wait_past_its_ceiling() {
        let mut deadline = ServerTestDeadline::new(WaitSubject::job_completion("j3"));
        let start = deadline.started;
        // Change on every probe, forever. Only the ceiling can stop this.
        let mut step = 0u32;
        let expired = loop {
            step += 1;
            let now = start + Duration::from_secs(u64::from(step) * 10);
            if let Err(expired) = deadline.observe_at(now, moved(step as usize)) {
                break expired;
            }
            assert!(
                step < 1_000,
                "the ceiling must stop a subject that never stops changing"
            );
        };
        assert_eq!(expired.cause, ExpiryCause::RanTooLong);
        assert!(expired.elapsed >= JOB_COMPLETION.ceiling);
    }

    /// The refusal has to be diagnosable on its own, because the failures this
    /// replaces were not: "did not finish within 60s" is what got two of them
    /// diagnosed as the bound rather than as contention.
    #[test]
    fn the_refusal_names_the_subject_and_the_last_observed_progress() {
        let mut deadline = ServerTestDeadline::new(WaitSubject::worker_registered_for_job("j4"));
        let start = deadline.started;
        deadline
            .observe_at(start, ProgressSnapshot::spawn_admission(0, 3))
            .expect("first probe");
        let expired = deadline
            .observe_at(
                start + WORKER_ADMISSION.idle_window,
                ProgressSnapshot::spawn_admission(0, 3),
            )
            .expect_err("unchanged admission state must expire");
        let rendered = expired.to_string();
        assert!(
            rendered.contains("register a worker for job j4"),
            "the refusal must name what it waited for: {rendered}"
        );
        assert!(
            rendered.contains("free_spawn_permits=0 live_workers=3"),
            "the refusal must name the last observed progress: {rendered}"
        );
    }

    /// Every subject's idle window must be strictly inside its ceiling,
    /// otherwise the ceiling is unreachable and the pair is decoration.
    #[test]
    fn every_budget_has_an_idle_window_strictly_inside_its_ceiling() {
        for budget in [
            JOB_COMPLETION,
            ML_JOB_COMPLETION,
            CONTROL_PLANE_TRANSITION,
            WORKER_ADMISSION,
            WORKER_PROCESS_EXIT,
        ] {
            assert!(
                budget.idle_window < budget.ceiling,
                "idle window must be smaller than the ceiling: {budget:?}"
            );
            assert!(
                budget.poll_interval < budget.idle_window,
                "a wait must be able to probe several times inside its idle window: {budget:?}"
            );
        }
    }
}
