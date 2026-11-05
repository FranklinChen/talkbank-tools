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
//!   [`PoolWaitObserver`] (the audio-file shell does, for every attempt)
//!   is told when one of its checkouts starts and stops waiting, and shows
//!   "waiting for a worker" while it does. The observer is carried by a task
//!   local, as the job id for cancel-driven worker shutdown already is
//!   (`job_tracker::CURRENT_JOB_ID`), so no dispatch signature between the
//!   file task and the pool changes.
//!
//! The same observer hears the pool's other congestion wait: a worker
//! startup or a model load queued for host memory
//! (`HostMemoryCoordinator::queue_for_lease`), told through
//! [`MemoryWaitObserver`], which the file shows as "waiting for host
//! memory". That wait reports itself to the log on its own schedule.

use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::api::{PositiveSeconds, WorkerLanguage};
use crate::store::PoolWait;
use crate::worker::WorkerTarget;
use crate::worker::error::WorkerError;

use super::WorkerKey;

/// Told when pool work under this task starts and stops waiting, and for
/// what: a saturated pool, or host memory. Implemented by the runner, which
/// owns file progress.
pub(crate) trait PoolWaitObserver: Send + Sync {
    /// A wait for `wait` began.
    fn waiting(&self, wait: PoolWait);
    /// That wait ended: it was served, failed, or was dropped.
    fn resumed(&self, wait: PoolWait);
    /// One scoped model preparation started, advanced, or ended.
    fn model_load(&self, change: crate::store::ModelLoadChange);
}

tokio::task_local! {
    /// The observer of the file task this work runs under, if any.
    static POOL_WAIT_OBSERVER: Observation;
}

/// Run `work` with `observer` told about every pool wait inside it.
///
/// Worker startup captures this observation before moving to the pool's
/// runtime, then reinstalls it for the lifetime of that startup.
pub(crate) async fn observing_pool_waits<F: Future>(
    observer: Arc<dyn PoolWaitObserver>,
    work: F,
) -> F::Output {
    Observation::Reported(observer).scope(work).await
}

/// Whether a wait is being shown to a file's progress.
#[derive(Clone)]
pub(crate) enum Observation {
    /// The task has an observer; it is told when the wait begins and when it
    /// ends.
    Reported(Arc<dyn PoolWaitObserver>),
    /// No file task observes this work (a health probe or background upkeep).
    Unobserved,
}

impl Observation {
    /// The observer of the task this runs in, if any.
    pub(crate) fn current() -> Self {
        match POOL_WAIT_OBSERVER.try_with(Clone::clone) {
            Ok(observation) => observation,
            Err(_) => Self::Unobserved,
        }
    }

    /// Carry an observation over a task boundary without binding the child
    /// process to the caller's runtime.
    pub(crate) async fn scope<F: Future>(self, work: F) -> F::Output {
        POOL_WAIT_OBSERVER.scope(self, work).await
    }

    fn report(&self, report: impl FnOnce(&dyn PoolWaitObserver)) {
        match self {
            Self::Reported(observer) => report(observer.as_ref()),
            Self::Unobserved => {}
        }
    }

    fn model_load(&self, change: crate::store::ModelLoadChange) {
        self.report(|observer| observer.model_load(change));
    }

    fn waiting(&self, wait: PoolWait) {
        self.report(|observer| observer.waiting(wait));
    }

    fn resumed(&self, wait: PoolWait) {
        self.report(|observer| observer.resumed(wait));
    }
}

/// Owns the visible lifetime of a loading operation. Dropping the operation,
/// including on cancellation or refusal, closes its observation. Request
/// progress cannot be passed to this port.
pub(crate) struct ObservedModelLoad(Observation);

impl ObservedModelLoad {
    pub(crate) fn begin() -> Self {
        let observation = Observation::current();
        observation.model_load(crate::store::ModelLoadChange::Started);
        Self(observation)
    }

    pub(crate) fn for_wait(wait: &crate::worker::ResponseWait) -> Option<Self> {
        match wait {
            crate::worker::ResponseWait::Loading { .. } => Some(Self::begin()),
            crate::worker::ResponseWait::Budget { .. } => None,
        }
    }

    pub(crate) fn report(&self, event: &crate::types::worker_v2::LoadProgressV2) {
        use crate::store::{ModelLoadChange, ModelLoadProgress};
        use crate::types::worker_v2::LoadMeasureV2;
        let progress = match event.measure {
            LoadMeasureV2::DownloadGrowth { grown_kib } => {
                ModelLoadProgress::Downloading { grown_kib }
            }
            LoadMeasureV2::Phase { .. } => ModelLoadProgress::Preparing,
        };
        self.0.model_load(ModelLoadChange::Advanced(progress));
    }
}

impl Drop for ObservedModelLoad {
    fn drop(&mut self) {
        self.0.model_load(crate::store::ModelLoadChange::Ended);
    }
}

/// A queued host-memory wait's observer: the file task it runs under, told
/// the wait is for host memory. Built where the wait begins, inside the
/// task, so the task local is read there.
pub(crate) struct MemoryWaitObserver(Observation);

impl MemoryWaitObserver {
    /// The observer of the task this runs in.
    pub(crate) fn current() -> Self {
        Self(Observation::current())
    }
}

impl crate::host_memory::QueuedLeaseObserver for MemoryWaitObserver {
    fn queued(&self) {
        self.0.waiting(PoolWait::HostMemory);
    }

    fn dequeued(&self) {
        self.0.resumed(PoolWait::HostMemory);
    }
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
        let observation = Observation::current();
        observation.waiting(PoolWait::Worker);
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
        self.observation.resumed(PoolWait::Worker);
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
pub(crate) mod tests {
    use std::sync::Mutex;

    use super::*;

    /// Records what it is told, in order.
    #[derive(Default)]
    pub(crate) struct RecordingObserver {
        pub(crate) events: Mutex<Vec<&'static str>>,
        pub(crate) loads: Mutex<Vec<crate::store::ModelLoadChange>>,
    }

    impl PoolWaitObserver for RecordingObserver {
        fn model_load(&self, change: crate::store::ModelLoadChange) {
            self.loads.lock().unwrap().push(change);
        }

        fn waiting(&self, wait: PoolWait) {
            self.events.lock().unwrap().push(match wait {
                PoolWait::Worker => "waiting",
                PoolWait::HostMemory => "waiting_memory",
            });
        }
        fn resumed(&self, wait: PoolWait) {
            self.events.lock().unwrap().push(match wait {
                PoolWait::Worker => "resumed",
                PoolWait::HostMemory => "resumed_memory",
            });
        }
    }

    fn key() -> (WorkerTarget, WorkerLanguage) {
        (
            WorkerTarget::infer_task(crate::worker::InferTask::Morphosyntax),
            WorkerLanguage::from(crate::api::LanguageCode3::eng()),
        )
    }

    /// An owned child receives the observation on its runtime, including
    /// memory waits, and closes its progress before releasing the publisher.
    #[tokio::test]
    async fn model_loading_progress_crosses_an_owned_spawn() {
        use crate::store::ModelLoadChange;
        let observer = Arc::new(RecordingObserver::default());
        let weak = Arc::downgrade(&observer);
        let observed = weak.clone();
        let (started, ready) = tokio::sync::oneshot::channel();
        let (child,) = observing_pool_waits(observer, async move {
            let captured = Observation::current();
            let child = tokio_util::task::AbortOnDropHandle::new(tokio::spawn(captured.scope(
                async move {
                    let _loading = ObservedModelLoad::begin();
                    let memory = MemoryWaitObserver::current();
                    crate::host_memory::QueuedLeaseObserver::queued(&memory);
                    crate::host_memory::QueuedLeaseObserver::dequeued(&memory);
                    started.send(()).unwrap();
                    std::future::pending::<()>().await;
                },
            )));
            ready.await.unwrap();
            assert_eq!(
                observed.upgrade().unwrap().loads.lock().unwrap().as_slice(),
                &[ModelLoadChange::Started]
            );
            assert_eq!(
                observed
                    .upgrade()
                    .unwrap()
                    .events
                    .lock()
                    .unwrap()
                    .as_slice(),
                &["waiting_memory", "resumed_memory"]
            );
            (child,)
        })
        .await;
        let observer = weak
            .upgrade()
            .expect("the child owns the observer until it stops");
        drop(child);
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while observer.loads.lock().unwrap().last() != Some(&ModelLoadChange::Ended) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("cancelling the child closes its progress");
        assert_eq!(
            observer.loads.lock().unwrap().as_slice(),
            &[ModelLoadChange::Started, ModelLoadChange::Ended]
        );
    }

    #[tokio::test]
    async fn model_loading_progress_ends_when_its_request_is_cancelled() {
        use crate::store::ModelLoadChange;
        let observer = Arc::new(RecordingObserver::default());
        let (started, ready) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(observing_pool_waits(observer.clone(), async move {
            let _loading = ObservedModelLoad::begin();
            started.send(()).unwrap();
            std::future::pending::<()>().await;
        }));
        ready.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert_eq!(
            observer.loads.lock().unwrap().as_slice(),
            &[ModelLoadChange::Started, ModelLoadChange::Ended]
        );
    }

    /// The observer is told the wait began, and that it ended when the wait
    /// is dropped, including when the waiting future is cancelled mid-wait.
    #[tokio::test(flavor = "current_thread")]
    async fn an_observed_wait_is_reported_and_its_end_is_reported_on_drop() {
        let observer = Arc::new(RecordingObserver::default());
        let (target, lang) = key();
        let pool_cancel = CancellationToken::new();
        observing_pool_waits(observer.clone(), async {
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
