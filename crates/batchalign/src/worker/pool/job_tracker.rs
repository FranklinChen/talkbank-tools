//! Per-job tracking of checked-out worker PIDs for cancel-driven shutdown.
//!
//! Without this, cancel waits for the in-flight worker call (Whisper,
//! Stanza, etc.) to complete naturally, minutes for long ASR passes.
//! `WorkerPool::detach_workers_for_job` drains the relevant PIDs
//! and `JobWorkersToTerminate::terminate` SIGTERMs them so the dispatch future returns BrokenPipe and
//! the runner unwinds at the next iteration.
//!
//! Registration is opt-in: callers under a `CURRENT_JOB_ID` scope
//! (set by the runner) get tracked; health / discovery
//! paths run without the scope and skip registration.

use std::time::Duration;

use crate::api::JobId;
use crate::worker::WorkerPid;
use tracing::{debug, info};

tokio::task_local! {
    /// Set at the entry of `run_server_job_attempt` so dispatch-site
    /// `TrackerGuard`s register against the correct job. The
    /// task-local propagates across awaits transparently, no
    /// signature changes at intermediate dispatch sites.
    pub(crate) static CURRENT_JOB_ID: JobId;
}

pub(crate) fn current_job_id() -> Option<JobId> {
    CURRENT_JOB_ID.try_with(|j| j.clone()).ok()
}

/// Carry the caller's job, if it has one, into work about to be spawned.
///
/// `tokio::spawn` does not inherit task-locals, so a dispatch inside a spawned
/// task would otherwise register against no job, and a cancel's worker kill
/// (`WorkerPool::detach_workers_for_job`) could not find it. Every spawn under
/// a job that can reach a worker wraps its future in this, or goes through the
/// runner's file-task scope, which does the same and also races the cancel.
/// Outside a job (direct CLI runs, health probes) the work runs unregistered,
/// as it would have in place.
pub(crate) fn inherit_job<F: std::future::Future>(
    work: F,
) -> impl std::future::Future<Output = F::Output> {
    let job = current_job_id();
    async move {
        match job {
            Some(job_id) => CURRENT_JOB_ID.scope(job_id, work).await,
            None => work.await,
        }
    }
}

mod registration;
pub(crate) use registration::{JobWorkerTracker, TrackerGuard};

/// Send SIGTERM to every PID, wait `grace`, SIGKILL any survivors.
/// Async-safe analog to `pool::reaper::kill_orphan`; both share the
/// `terminate_pgid` / `kill_pgid` / `process_alive` primitives in
/// `pool/reaper.rs`.
pub(crate) async fn signal_workers(pids: &[WorkerPid], grace: Duration) {
    if pids.is_empty() {
        return;
    }
    for pid in pids {
        super::reaper::terminate_pgid(pid.0);
    }
    tokio::time::sleep(grace).await;
    for pid in pids {
        if super::reaper::process_alive(pid.0) {
            info!(
                worker_pid = pid.0,
                "Worker survived SIGTERM, sending SIGKILL"
            );
            super::reaper::kill_pgid(pid.0);
        } else {
            debug!(worker_pid = pid.0, "Worker exited after SIGTERM");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pid(n: u32) -> WorkerPid {
        WorkerPid(n)
    }
    fn job(s: &str) -> JobId {
        JobId::from(s.to_string())
    }

    /// Guards: Overlapping invocations keep their worker cancellable until each owning registration ends, and detached registrations cannot remove a new registration for the same job and PID.
    /// Kind: contract
    /// Not a type: Private consuming registrations exclude raw removal and duplicate release. The shared map must still reflect actual guard drops, deduplicate cancellation PIDs and isolate job observations across detach and reuse.
    /// State: An isolated in-memory tracker records declared PIDs. Actual task-local scopes exercise the production guard constructor. The claim needs shared registration effects, with no process, socket, model or memory ledger.
    /// Concurrency: Each run owns its tracker. Overlapping guard lifetimes and detach order are explicit and deterministic, with no sleeps or background work.
    #[tokio::test]
    async fn owned_registrations_survive_overlap_and_detach() {
        async fn tracked(
            tracker: &JobWorkerTracker,
            name: &str,
            worker: WorkerPid,
        ) -> TrackerGuard {
            CURRENT_JOB_ID
                .scope(job(name), async { TrackerGuard::new(tracker, worker) })
                .await
        }
        fn ordered(mut workers: Vec<WorkerPid>) -> Vec<WorkerPid> {
            workers.sort_by_key(|worker| worker.0);
            workers
        }
        let tracker = JobWorkerTracker::new();
        let first = tracked(&tracker, "a", pid(1)).await;
        let remaining = tracked(&tracker, "a", pid(1)).await;
        let sibling = tracked(&tracker, "a", pid(2)).await;
        let other = tracked(&tracker.clone(), "b", pid(3)).await;
        drop(first);
        assert_eq!(ordered(tracker.snapshot(&job("a"))), [pid(1), pid(2)]);
        assert_eq!(ordered(tracker.drain(&job("a"))), [pid(1), pid(2)]);
        assert!(tracker.snapshot(&job("a")).is_empty());
        assert!(tracker.drain(&job("a")).is_empty());
        let fresh = tracked(&tracker, "a", pid(1)).await;
        drop(remaining);
        drop(sibling);
        assert_eq!(tracker.snapshot(&job("a")), [pid(1)]);
        assert_eq!(tracker.snapshot(&job("b")), [pid(3)]);
        drop(fresh);
        drop(other);
        assert!(tracker.snapshot(&job("a")).is_empty());
        assert!(tracker.snapshot(&job("b")).is_empty());
    }

    struct SignalChild(std::process::Child);

    impl SignalChild {
        fn spawn() -> Self {
            Self(
                std::process::Command::new("sleep")
                    .arg("60")
                    .spawn()
                    .expect("spawn sleep"),
            )
        }
    }

    impl Drop for SignalChild {
        fn drop(&mut self) {
            // Cleanup also owns the child when a signal assertion fails.
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    /// Guards: Cancellation terminates a real subprocess and tolerates a PID that has already disappeared.
    /// Kind: contract
    /// Not a type: Registration ownership retains the exact cancellation targets, while actual process exit and ESRCH behavior are changing operating-system effects.
    /// State: One private sleep subprocess exercises the real signal adapter and child reap. A declared absent PID exercises the missing-process path. A typed double cannot establish kernel signal delivery or child exit.
    /// Concurrency: The child belongs to this test and its RAII fixture reaps it on success or assertion failure. The declared absent PID exercises the existing missing-process fixture, with no shared worker, model or registry.
    #[cfg(unix)]
    #[tokio::test]
    async fn signal_workers_terminates_real_subprocess() {
        signal_workers(&[pid(999_999_999)], Duration::ZERO).await;
        let mut child = SignalChild::spawn();
        let real_pid = WorkerPid(child.0.id());
        assert!(super::super::reaper::process_alive(real_pid.0));

        signal_workers(&[real_pid], Duration::from_secs(2)).await;

        assert!(child.0.try_wait().expect("observe child exit").is_some());
        assert!(!super::super::reaper::process_alive(real_pid.0));
    }
}
