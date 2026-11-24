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
//! Slot owners span reservation, actual queues, departed exchanges and retirement.
//! Queue contents are the only availability authority. A report re-observes
//! congestion without reconstructing hypothetical missing owners from separately
//! sampled counters.
//!
//! What replaced the deadline:
//!
//! - **A report interval.** Each [`SaturatedWait::park`] races the awaited
//!   event against the interval. When the interval passes first the wait is
//!   reported (a warning naming the key, how long it has waited and the
//!   group's counts) and the caller re-probes from the top (fast path, spawn,
//!   eviction), so a state that changed under it is re-read rather than
//!   trusted.
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

/// Told when pool work under this task starts and stops waiting, and for
/// what: a saturated pool, or host memory. Implemented by the runner, which
/// owns file progress.
pub trait PoolWaitObserver: Send + Sync {
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
pub async fn observing_pool_waits<F: Future>(
    observer: Arc<dyn PoolWaitObserver>,
    work: F,
) -> F::Output {
    Observation::Reported(observer).scope(work).await
}

/// Whether a wait is being shown to a file's progress.
#[derive(Clone)]
pub enum Observation {
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
pub struct ObservedModelLoad(Observation);

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
pub struct MemoryWaitObserver(Observation);

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
        }
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
pub mod tests {
    use std::sync::Mutex;

    use super::*;

    /// Records what it is told, in order.
    #[derive(Default)]
    pub struct RecordingObserver {
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
    /// Guards: an owned spawned task retains task-local model/memory observation
    /// and closes its model-load lifetime when cancellation completion is awaited.
    /// Kind: contract
    /// Not a type: task-local propagation, channel readiness and actual task abort
    /// are runtime effects beyond the captured observation's ownership guarantee.
    /// State: one owned Tokio task, recording observer and readiness channel exercise
    /// the real scope without a process, model, timer poll or host lease.
    /// Concurrency: readiness is channel-bound and abort completion is awaited.
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
        child.abort();
        assert!(child.await.unwrap_err().is_cancelled());
        assert_eq!(
            observer.loads.lock().unwrap().as_slice(),
            &[ModelLoadChange::Started, ModelLoadChange::Ended]
        );
    }

    /// Guards: wait begin, interval re-probe, ready event, shutdown and final
    /// observer release keep their order without turning a report into a deadline.
    /// Kind: contract
    /// Not a type: Actual Tokio readiness and controlled elapsed intervals exercise task-local
    /// observer callbacks are actual effects after the wait owns its lifecycle.
    /// State: one controlled runtime, cancellation token and recording observer
    /// exercise the real wait with declared group observations and no process.
    /// Concurrency: local paused time and owned futures need no sleep or shared pool.
    #[tokio::test(start_paused = true)]
    async fn wait_lifecycle_retains_events_reprobes_and_shutdown() {
        let observer = Arc::new(RecordingObserver::default());
        let (target, lang) = key();
        let cancel = CancellationToken::new();
        observing_pool_waits(observer.clone(), async {
            let mut wait = SaturatedWait::begin(&target, &lang, PositiveSeconds::literal::<1>());
            assert_eq!(*observer.events.lock().unwrap(), vec!["waiting"]);
            for expected in 1..=3 {
                assert!(matches!(
                    wait.park(std::future::pending::<()>(), &cancel, || GroupCounts {
                        total: 4,
                        idle: 0
                    })
                    .await
                    .unwrap(),
                    Parked::StillSaturated
                ));
                assert_eq!(wait.reports, expected);
            }
            assert!(matches!(
                wait.park(std::future::ready(7), &cancel, || GroupCounts {
                    total: 4,
                    idle: 1
                })
                .await
                .unwrap(),
                Parked::Ready(7)
            ));
            cancel.cancel();
            assert!(matches!(
                wait.park(std::future::pending::<()>(), &cancel, || GroupCounts {
                    total: 0,
                    idle: 0
                })
                .await,
                Err(WorkerError::PoolShuttingDown)
            ));
        })
        .await;
        assert_eq!(*observer.events.lock().unwrap(), vec!["waiting", "resumed"]);
    }
}
