//! Waiting for a worker when the pool is saturated: congestion, not failure.
//!
//! A checkout that finds every slot for its key busy, the global cap reached
//! and no idle worker elsewhere to evict waits for one to come back. That is
//! the ordinary state of a busy server (six jobs with eight files in flight
//! each, all wanting the same few Stanza workers). The wait has no deadline
//! of its own. Until 2026-10-06 it did (300 s), and a file that queued behind
//! long requests failed with "pool saturated" although nothing was wrong with
//! it.
//!
//! Congestion is not the only way to find no worker, though: a group whose
//! live count (`total`) counts a worker that nothing holds would make the
//! wait endless. Every worker counted in `total` is in an idle queue or held
//! by an [`super::AwayFromQueue`] guard (a checkout, a spawn, a health
//! check, an eviction's shutdown), so the pool can reconcile the two. The
//! same group showing the same mismatch at two consecutive report intervals
//! is broken accounting, and the wait ends with the typed, terminal
//! [`WorkerError::PoolAccountingBroken`]. A mismatch seen once may be a
//! worker between a queue and its guard, and is only reported; two
//! unrelated mismatches never confirm each other. A checkout parked on its
//! own group's permits reconciles only that group, which is all it waits
//! on; one parked on the pool-wide return signal reconciles every group.
//! What this cannot detect is a worker that is genuinely checked out and
//! never comes back; that is held, so it is waited on.
//!
//! What replaced the deadline:
//!
//! - **A report interval.** Each [`SaturatedWait::park`] races the awaited
//!   event against the interval. When the interval passes first the wait is
//!   reported (a warning naming the key, how long it has waited and the
//!   group's counts) and the caller re-probes from the top (fast path, spawn,
//!   eviction), so a state that changed under it is re-read rather than
//!   trusted.
//! - **Reconciliation.** At each report interval the pool's counts are
//!   reconciled ([`SaturatedWait::reconcile`]); a persistent mismatch ends the
//!   wait with [`WorkerError::PoolAccountingBroken`].
//! - **The pool's own lifecycle.** Pool shutdown ends the wait with the typed
//!   [`WorkerError::PoolShuttingDown`].
//! - **The caller's.** A cancelled job drops its supervised file task, and
//!   with it this future; nothing here needs a token of its own.
//! - **Observability in progress.** A file task that installs a
//!   [`CheckoutWaitObserver`] (the audio-file shell does, for every attempt)
//!   is told when one of its checkouts starts and stops waiting, and shows
//!   "waiting for a worker" while it does. The observer is carried by a task
//!   local, as the job id for cancel-driven worker shutdown already is
//!   (`job_tracker::CURRENT_JOB_ID`), so no dispatch signature between the
//!   file task and the pool changes.

use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::api::{PositiveSeconds, WorkerLanguage};
use crate::worker::WorkerTarget;
use crate::worker::error::WorkerError;

use super::WorkerKey;

/// Told when a checkout under this task starts and stops waiting for a
/// saturated pool. Implemented by the runner, which owns file progress.
pub(crate) trait CheckoutWaitObserver: Send + Sync {
    /// A checkout began waiting for a worker for `target` / `lang`.
    fn waiting(&self, target: &WorkerTarget, lang: &WorkerLanguage);
    /// That checkout stopped waiting: it got a worker, failed, or was dropped.
    fn resumed(&self);
}

tokio::task_local! {
    /// The observer of the file task this checkout runs under, if any.
    static CHECKOUT_WAIT_OBSERVER: Arc<dyn CheckoutWaitObserver>;
}

/// Run `work` with `observer` told about every saturated checkout inside it.
///
/// Like every task local, it does not cross a `tokio::spawn`: a checkout in a
/// spawned task waits unobserved, which loses the progress label and nothing
/// else.
pub(crate) async fn observing_checkout_waits<F: Future>(
    observer: Arc<dyn CheckoutWaitObserver>,
    work: F,
) -> F::Output {
    CHECKOUT_WAIT_OBSERVER.scope(observer, work).await
}

/// Whether this wait is being shown to a file's progress.
enum Observation {
    /// The task has an observer; it was told the wait began and is told when
    /// it ends, by `Drop`, whichever way the wait ends.
    Reported(Arc<dyn CheckoutWaitObserver>),
    /// No file task observes this checkout (a health probe, a spawned task).
    Unobserved,
}

/// How one park on a saturated pool ended.
#[derive(Debug)]
pub(super) enum Parked<T> {
    /// The awaited event happened.
    Ready(T),
    /// The report interval passed first; the wait was reported and the caller
    /// re-probes.
    StillSaturated,
}

/// One checkout's wait for a saturated pool, from its first park to its end.
///
/// Created lazily, at the first park, so a checkout that never waits reports
/// nothing. Dropping it ends the observation.
pub(super) struct SaturatedWait {
    target: WorkerTarget,
    lang: WorkerLanguage,
    interval: PositiveSeconds,
    started: Instant,
    reports: u32,
    observation: Observation,
    /// The mismatches the previous report interval found, awaiting
    /// confirmation.
    suspected: Vec<UnaccountedGroup>,
}

/// A group whose live count exceeds what holds its workers.
#[derive(Debug, Clone)]
pub(super) struct UnaccountedGroup {
    pub(super) key: WorkerKey,
    pub(super) unaccounted: usize,
    pub(super) total: usize,
    pub(super) idle: usize,
    pub(super) away: usize,
}

impl UnaccountedGroup {
    /// The same group with the same number of workers nothing holds: what a
    /// lost worker looks like at two report intervals, and what a worker
    /// briefly between a queue and its guard does not.
    fn confirms(&self, earlier: &Self) -> bool {
        self.key == earlier.key && self.unaccounted == earlier.unaccounted
    }

    fn into_error(self) -> WorkerError {
        WorkerError::PoolAccountingBroken {
            target: self.key.target,
            lang: self.key.language,
            unaccounted: self.unaccounted,
            total: self.total,
            idle: self.idle,
            away: self.away,
        }
    }
}

impl SaturatedWait {
    /// Begin waiting for a worker for `target` / `lang`.
    pub(super) fn begin(
        target: &WorkerTarget,
        lang: &WorkerLanguage,
        interval: PositiveSeconds,
    ) -> Self {
        let observation = match CHECKOUT_WAIT_OBSERVER.try_with(Arc::clone) {
            Ok(observer) => {
                observer.waiting(target, lang);
                Observation::Reported(observer)
            }
            Err(_) => Observation::Unobserved,
        };
        Self {
            target: *target,
            lang: lang.clone(),
            interval,
            started: Instant::now(),
            reports: 0,
            observation,
            suspected: Vec::new(),
        }
    }

    /// After a report interval: act on the pool's reconciliation, `found`
    /// being every mismatch in the scope this park waits on. A mismatch the
    /// previous interval also found in the same group ends the wait; one
    /// seen for the first time is reported and remembered; any suspicion not
    /// seen again is cleared.
    pub(super) fn reconcile(&mut self, found: Vec<UnaccountedGroup>) -> Result<(), WorkerError> {
        let earlier = std::mem::take(&mut self.suspected);
        for group in found {
            if earlier.iter().any(|suspect| group.confirms(suspect)) {
                return Err(group.into_error());
            }
            warn!(
                target = %group.key.target.label(),
                lang = %group.key.language,
                unaccounted = group.unaccounted,
                total = group.total,
                idle = group.idle,
                away = group.away,
                "A worker group counts workers nothing holds; if this persists to the \
                 next report the wait ends as broken pool accounting"
            );
            self.suspected.push(group);
        }
        Ok(())
    }

    /// Wait for `event`, at most one report interval, unless the pool shuts
    /// down. `group_counts` describes the group for the report.
    pub(super) async fn park<F: Future>(
        &mut self,
        event: F,
        pool_cancel: &CancellationToken,
        group_counts: impl FnOnce() -> GroupCounts,
    ) -> Result<Parked<F::Output>, WorkerError> {
        tokio::select! {
            biased;
            () = pool_cancel.cancelled() => Err(WorkerError::PoolShuttingDown),
            output = event => Ok(Parked::Ready(output)),
            () = tokio::time::sleep(self.interval.duration()) => {
                self.reports += 1;
                let counts = group_counts();
                warn!(
                    target = %self.target.label(),
                    lang = %self.lang,
                    waited_s = self.started.elapsed().as_secs(),
                    reports = self.reports,
                    group_total = counts.total,
                    group_idle = counts.idle,
                    "Still waiting for a worker: the pool is saturated and no idle worker \
                     can be evicted. The request keeps its place; it fails only if the \
                     pool shuts down or its job is cancelled"
                );
                Ok(Parked::StillSaturated)
            }
        }
    }
}

impl Drop for SaturatedWait {
    fn drop(&mut self) {
        match &self.observation {
            Observation::Reported(observer) => observer.resumed(),
            Observation::Unobserved => {}
        }
    }
}

/// A group's worker counts, as a wait report states them.
#[derive(Debug, Clone, Copy)]
pub(super) struct GroupCounts {
    /// Live workers, idle and checked out (and spawns in flight).
    pub(super) total: usize,
    /// Idle workers.
    pub(super) idle: usize,
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// Records what it is told, in order.
    #[derive(Default)]
    pub(crate) struct RecordingObserver {
        pub(crate) events: Mutex<Vec<&'static str>>,
    }

    impl CheckoutWaitObserver for RecordingObserver {
        fn waiting(&self, _target: &WorkerTarget, _lang: &WorkerLanguage) {
            self.events.lock().unwrap().push("waiting");
        }
        fn resumed(&self) {
            self.events.lock().unwrap().push("resumed");
        }
    }

    fn key() -> (WorkerTarget, WorkerLanguage) {
        (
            WorkerTarget::infer_task(crate::worker::InferTask::Morphosyntax),
            WorkerLanguage::from(crate::api::LanguageCode3::eng()),
        )
    }

    /// The observer is told the wait began, and that it ended when the wait
    /// is dropped, including when the waiting future is cancelled mid-wait.
    #[tokio::test(flavor = "current_thread")]
    async fn an_observed_wait_is_reported_and_its_end_is_reported_on_drop() {
        let observer = Arc::new(RecordingObserver::default());
        let (target, lang) = key();
        let pool_cancel = CancellationToken::new();
        observing_checkout_waits(observer.clone(), async {
            let mut wait = SaturatedWait::begin(&target, &lang, PositiveSeconds::literal::<60>());
            let parked = tokio::time::timeout(
                std::time::Duration::from_millis(20),
                wait.park(std::future::pending::<()>(), &pool_cancel, || GroupCounts {
                    total: 1,
                    idle: 0,
                }),
            )
            .await;
            assert!(
                parked.is_err(),
                "nothing came back, so the wait is still parked"
            );
            assert_eq!(*observer.events.lock().unwrap(), vec!["waiting"]);
        })
        .await;
        assert_eq!(*observer.events.lock().unwrap(), vec!["waiting", "resumed"]);
    }

    /// The report interval ends one park, not the wait: the caller is told
    /// to re-probe, and the wait goes on.
    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn an_elapsed_interval_reports_and_keeps_waiting() {
        let (target, lang) = key();
        let pool_cancel = CancellationToken::new();
        let mut wait = SaturatedWait::begin(&target, &lang, PositiveSeconds::literal::<1>());
        for expected_reports in 1..=3 {
            let parked = wait
                .park(std::future::pending::<()>(), &pool_cancel, || GroupCounts {
                    total: 4,
                    idle: 0,
                })
                .await
                .expect("the pool is not shutting down");
            assert!(matches!(parked, Parked::StillSaturated));
            assert_eq!(wait.reports, expected_reports);
        }
        let parked = wait
            .park(std::future::ready(7), &pool_cancel, || GroupCounts {
                total: 4,
                idle: 1,
            })
            .await
            .expect("the event is ready");
        assert!(matches!(parked, Parked::Ready(7)));
    }

    /// Pool shutdown is the one thing that ends a wait with an error.
    #[tokio::test(flavor = "current_thread")]
    async fn pool_shutdown_ends_the_wait_with_a_typed_error() {
        let (target, lang) = key();
        let pool_cancel = CancellationToken::new();
        pool_cancel.cancel();
        let mut wait = SaturatedWait::begin(&target, &lang, PositiveSeconds::literal::<60>());
        let error = wait
            .park(std::future::pending::<()>(), &pool_cancel, || GroupCounts {
                total: 0,
                idle: 0,
            })
            .await
            .expect_err("a shutting-down pool ends the wait");
        assert!(matches!(error, WorkerError::PoolShuttingDown));
    }

    fn mismatch(task: crate::worker::InferTask, unaccounted: usize) -> UnaccountedGroup {
        UnaccountedGroup {
            key: WorkerKey::without_engine_selection(
                WorkerTarget::infer_task(task),
                WorkerLanguage::from(crate::api::LanguageCode3::eng()),
            ),
            unaccounted,
            total: unaccounted,
            idle: 0,
            away: 0,
        }
    }

    /// Two transient mismatches in different groups, or in one group with
    /// different counts, never confirm each other; the same group with the
    /// same count at the next interval does.
    #[test]
    fn only_the_same_mismatch_twice_confirms_broken_accounting() {
        use crate::worker::InferTask::{Morphosyntax, Utseg};
        let (target, lang) = key();
        let mut wait = SaturatedWait::begin(&target, &lang, PositiveSeconds::literal::<60>());
        wait.reconcile(vec![mismatch(Morphosyntax, 1)])
            .expect("first sighting is only reported");
        wait.reconcile(vec![mismatch(Utseg, 1)])
            .expect("another group's mismatch does not confirm it");
        wait.reconcile(vec![mismatch(Utseg, 2)])
            .expect("a different count in the same group does not confirm it");
        wait.reconcile(Vec::new())
            .expect("nothing found clears suspicion");
        wait.reconcile(vec![mismatch(Utseg, 2)])
            .expect("a cleared suspicion starts over");
        let error = wait
            .reconcile(vec![mismatch(Morphosyntax, 3), mismatch(Utseg, 2)])
            .expect_err("the same group's same mismatch confirms");
        assert!(
            matches!(
                error,
                WorkerError::PoolAccountingBroken {
                    target,
                    unaccounted: 2,
                    ..
                } if target == WorkerTarget::infer_task(Utseg)
            ),
            "{error}"
        );
    }
}
