//! Worker checkout and dispatch routing.
//!
//! Contains the core `checkout()` loop (observe queue → own departure →
//! consuming exchange), `dispatch_batch_infer`, `dispatch_execute_v2`, and TCP worker
//! checkout/return helpers. Routes keys whose workers serve concurrently
//! (`WorkerServing::SharedConcurrent`) to one shared worker; every other key
//! uses the exclusive-checkout model.

use std::sync::Arc;

use crate::api::{LanguageCode3, ReleasedCommand, WorkerLanguage};
use crate::types::worker_v2::{ExecuteRequestV2, ExecuteResponseV2};
use crate::worker::error::{WorkerAfterFailure, WorkerError};
use crate::worker::tcp_handle::TcpWorkerHandle;
use crate::worker::{BatchInferRequest, BatchInferResponse, WorkerTarget};
use tracing::{instrument, warn};

use super::checkout_wait::{GroupCounts, Parked, SaturatedWait};
use super::eviction::EvictionOutcome;
use super::execute_v2;
use super::job_tracker::TrackerGuard;
use super::worker_plan::{CommandWorkerPlan, WorkerPlan};
use super::{DispatchPreparation, ReadyRequest};
use super::{WorkerKey, WorkerPool, lock_recovered};

// The exchange order is identical on every transport. Both sends require the
// same lease-owning RunPermit; the FA preparation reply advances the owned
// candidate rather than reopening an arbitrary execution packet.
macro_rules! dispatch_prepared {
    ($prepared:expr, $ready:ident => $send:expr) => {
        'prepared_dispatch: {
            use $crate::worker::pool::{DispatchPreparation, FaPreparationCompletion};
            let ready = match $prepared {
                DispatchPreparation::Ready(ready) => ready,
                DispatchPreparation::ForcedAlignment(exchange) => {
                    let $ready = exchange.query();
                    let reply = $send.await?;
                    match exchange.bind(reply)? {
                        FaPreparationCompletion::Ready(ready) => ready.into_ready(),
                        FaPreparationCompletion::Refused(reply) => {
                            break 'prepared_dispatch Ok(reply);
                        }
                    }
                }
            };
            let $ready = ready;
            $send.await
        }
    };
}
pub(crate) use dispatch_prepared;

/// A completed TCP exchange retains its original queue departure and job use.
pub(super) struct TcpCheckout {
    handle: super::capacity::DepartedWorker<TcpWorkerHandle>,
    _registration: TrackerGuard,
}
impl TcpCheckout {
    pub(super) async fn run<T>(
        mut self,
        work: impl AsyncFnOnce(&mut TcpWorkerHandle) -> Result<T, WorkerError>,
    ) -> Result<T, WorkerError> {
        let outcome = work(&mut self.handle).await;
        match outcome
            .as_ref()
            .map(|_| WorkerAfterFailure::Reusable)
            .unwrap_or_else(|error| error.worker_after_failure())
        {
            WorkerAfterFailure::Reusable => self.handle.return_idle(),
            WorkerAfterFailure::Retire => drop(self),
        }
        outcome
    }
}

impl WorkerPool {
    /// Check out an idle worker or spawn a new one.
    ///
    /// 1. Register queue notifications and try to take an actual idle owner.
    /// 2. If none available, try to spawn a new worker (if under capacity).
    /// 3. If at capacity, wait for a worker to come back (async suspend).
    /// 4. Consume the tracked checkout through its exchange and completion.
    ///
    /// A saturated pool is congestion, not a failure of the request: the
    /// wait has no deadline, only a report interval after which it is logged
    /// and re-probed. It ends when a worker is available, when the pool
    /// shuts down, or when the caller drops it. See [`super::checkout_wait`].
    pub(super) async fn checkout(
        &self,
        key: &WorkerKey,
    ) -> Result<super::checkout::TrackedCheckout, WorkerError> {
        let group = self.get_or_create_group(key);
        let report_interval = self.config.checkout_wait_report_interval();
        let mut saturated: Option<SaturatedWait> = None;
        let counts = || GroupCounts {
            total: group.total.count(),
            idle: group.idle.view().len(),
        };
        loop {
            // Register before either observation or spawn. A burst wakes the
            // already-registered queue waiters once per actual publication.
            let queued = group.idle.changed();
            let changed = self.worker_returned.notified();
            tokio::pin!(queued, changed);
            queued.as_mut().enable();
            changed.as_mut().enable();
            if let Some(worker) = group.take_stdio() {
                return Ok(super::checkout::TrackedCheckout::from_departed(
                    worker,
                    &self.job_tracker,
                ));
            }
            let deferral = match self.try_spawn_into_group(&group, key).await? {
                super::lifecycle::SpawnOutcome::Published => continue,
                super::lifecycle::SpawnOutcome::Deferred(deferral) => deferral,
            };
            if group.is_empty()
                && matches!(
                    self.try_evict_idle_from_other_group(key),
                    EvictionOutcome::Evicted
                )
            {
                continue;
            }
            let wait = saturated.get_or_insert_with(|| {
                SaturatedWait::begin(&key.target, &key.language, report_interval)
            });
            let wake = async {
                tokio::select! {
                    _ = queued => {},
                    _ = changed => {},
                    _ = deferral.global_change(&self.spawn_permits) => {},
                }
            };
            match wait.park(wake, &self.cancel, counts).await? {
                Parked::Ready(()) | Parked::StillSaturated => {}
            }
        }
    }

    /// Dispatch a batch inference request to a single worker.
    ///
    /// Tries TCP workers first (from registry), then falls back to stdio
    /// workers. Checks out an idle worker (or spawns one), sends the batch
    /// infer request, and returns the response.
    pub async fn dispatch_batch_infer(
        &self,
        lang: &LanguageCode3,
        request: &BatchInferRequest,
    ) -> Result<BatchInferResponse, WorkerError> {
        let target =
            WorkerTarget::from_infer_task(request.task, self.config.runtime.bootstrap_mode);
        let worker_lang = WorkerLanguage::from(lang);
        let key = WorkerKey::without_engine_selection(target, worker_lang);

        if matches!(key.target, WorkerTarget::Profile(_))
            && let Some(checkout) = self.try_checkout_tcp(&key)
        {
            return checkout
                .run(async |worker| worker.batch_infer(request).await)
                .await;
        }
        let result = self
            .checkout(&key)
            .await?
            .run(async |worker| worker.batch_infer(request).await)
            .await;
        match result {
            Err(ref error @ (WorkerError::Io(_) | WorkerError::ProcessExited { .. })) => {
                warn!(%error, "worker crashed during batch_infer; retrying with a fresh worker");
                self.checkout(&key)
                    .await?
                    .run(async |worker| worker.batch_infer(request).await)
                    .await
            }
            other => other,
        }
    }

    /// Try to check out a TCP worker handle (non-blocking), as an owned
    /// [`TcpCheckout`] that gives it back or retires it.
    pub(super) fn try_checkout_tcp(&self, key: &WorkerKey) -> Option<TcpCheckout> {
        let group = lock_recovered(&self.groups).get(key)?.clone();
        let handle = group.take_tcp()?;
        Some(TcpCheckout {
            _registration: TrackerGuard::new(&self.job_tracker, handle.pid()),
            handle,
        })
    }

    /// Dispatch one typed worker-protocol V2 execute request.
    ///
    /// Shared-serving keys (`WorkerServing`) are routed to a shared concurrent worker (multiple
    /// requests in flight to one process). Non-GPU tasks try TCP workers first,
    /// then fall back to the traditional exclusive checkout model.
    pub async fn dispatch_execute_v2(
        &self,
        lang: impl Into<WorkerLanguage>,
        request: &ExecuteRequestV2,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        self.dispatch_execute_v2_with_progress(lang, request, None)
            .await
    }

    /// The registry GPU daemon connected for `key`, if its stream is still
    /// open. A connection whose stream has closed is removed here, so dispatch
    /// falls back to a spawned worker and the next discovery sweep can connect
    /// to the daemon afresh, instead of every request waiting out its timeout
    /// on a dead connection.
    pub(super) async fn live_gpu_tcp_worker(
        &self,
        key: &WorkerKey,
    ) -> Option<Arc<super::shared_gpu::SharedGpuTcpWorker>> {
        let mut workers = self.gpu_tcp_workers.lock().await;
        let worker = workers.get(key)?;
        match worker.check_available() {
            Ok(()) => Some(worker.clone()),
            Err(error) => {
                warn!(
                    target = %key.target.label(),
                    lang = %key.language,
                    pid = %worker.pid(),
                    %error,
                    "GPU TCP worker connection is closed; removing it"
                );
                workers.remove(key);
                None
            }
        }
    }

    /// Dispatch a V2 execute request, forwarding intermediate progress events
    /// through an optional async channel.
    ///
    /// Keys the worker by the request alone, which is right for text and IO
    /// tasks, whose workers select at most one engine per task. A GPU-profile
    /// request is refused here ([`WorkerError::UnroutedGpuRequest`]): it must
    /// travel on its job's route ([`Self::dispatch_routed_v2`]).
    #[instrument(
        skip_all,
        fields(request_id = %request.request_id),
    )]
    pub async fn dispatch_execute_v2_with_progress(
        &self,
        lang: impl Into<WorkerLanguage>,
        request: &ExecuteRequestV2,
        progress_tx: Option<&tokio::sync::mpsc::Sender<crate::types::worker_v2::ProgressEventV2>>,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        let plan = WorkerPlan::direct(
            lang.into(),
            DispatchPreparation::begin(request)?,
            self.config.runtime.bootstrap_mode,
        )?;
        self.dispatch_v2_on_plan(plan, progress_tx).await
    }

    /// The route for one GPU task of a job's model plan, in `language`.
    ///
    /// Every task of one plan gets the same worker key under the profile
    /// bootstrap modes, so one job reaches one GPU worker.
    pub(crate) fn gpu_route(
        &self,
        plan: &super::GpuModelPlan,
        task: super::GpuTask,
        language: impl Into<WorkerLanguage>,
    ) -> Result<super::GpuRoute, WorkerError> {
        Ok(super::GpuRoute::new(
            plan,
            task,
            language.into(),
            self.config.runtime.bootstrap_mode,
        )?)
    }

    /// Dispatch a GPU-profile request on its job's route.
    ///
    /// The request must be the route's task and name exactly the engines the
    /// route was planned with; otherwise it is refused
    /// ([`WorkerError::PlanMismatch`]) rather than sent to a worker spawned
    /// for other models.
    #[instrument(
        skip_all,
        fields(request_id = %request.request_id),
    )]
    pub(crate) async fn dispatch_routed_v2(
        &self,
        route: &super::GpuRoute,
        request: &ExecuteRequestV2,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        let plan = WorkerPlan::routed(route, DispatchPreparation::begin(request)?)?;
        self.dispatch_v2_on_plan(plan, None).await
    }

    /// Consume retained preparation on the ordinary route. The loaded worker
    /// still grants a fresh lease-owning run permit; a planning observation
    /// cannot substitute for execution admission on any transport.
    pub(crate) async fn dispatch_ready_routed(
        &self,
        route: &super::GpuRoute,
        ready: ReadyRequest,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        let plan = WorkerPlan::routed(route, DispatchPreparation::Ready(ready))?;
        self.dispatch_v2_on_plan(plan, None).await
    }

    /// Dispatch a GPU-profile request in integration tests, on the route
    /// `plan` gives its task. Production requests are routed by their job's
    /// plan at their own call sites ([`Self::dispatch_routed_v2`]).
    #[doc(hidden)]
    pub async fn dispatch_gpu_for_test(
        &self,
        plan: &super::GpuModelPlan,
        language: impl Into<WorkerLanguage>,
        request: &ExecuteRequestV2,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        let task = execute_v2::infer_task_for_execute_v2(request.task);
        let gpu_task = super::GpuTask::of(task).ok_or_else(|| {
            WorkerError::Protocol(format!("a {task:?} request is not a GPU-profile request"))
        })?;
        let route = self.gpu_route(plan, gpu_task, language)?;
        self.dispatch_routed_v2(&route, request).await
    }

    /// Dispatch one request to the worker `key` names.
    async fn dispatch_v2_on_plan(
        &self,
        plan: WorkerPlan,
        progress_tx: Option<&tokio::sync::mpsc::Sender<crate::types::worker_v2::ProgressEventV2>>,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        if self.serving(plan.key().target).is_shared() {
            return self.dispatch_gpu_execute_v2(plan).await;
        }
        let (key, prepared) = plan.into_parts();
        let key = &key;

        // The prepare step: every model loads here, before the request's
        // first dispatch to a worker, in every bootstrap mode (no worker
        // loads one at startup). `ensure_task` is the transition to the
        // `Loaded` proof the execute takes; a worker that already completed
        // this load returns its proof with no round trip. A download runs
        // under the loading stall window, never inside the request. The load
        // is read from the request once, before any worker is taken, so a
        // malformed request never retires a worker.
        let preparation = self.load_preparation();

        if matches!(key.target, WorkerTarget::Profile(_))
            && let Some(checkout) = self.try_checkout_tcp(key)
        {
            return checkout.run(async |worker| {
                dispatch_prepared!(prepared, ready => worker.execute_v2_with_progress(ready, &preparation, progress_tx))
            }).await;
        }
        self.checkout(key).await?.run(async |worker| {
            dispatch_prepared!(prepared, ready => worker.execute_v2_with_progress(ready, &preparation, progress_tx))
        }).await
    }

    /// Dispatch a V2 execute request to a GPU worker.
    ///
    /// Tries TCP workers first (discovered from registry), then falls back to
    /// stdio workers. For TCP workers, multiple callers share one worker via
    /// concurrent dispatch. For stdio workers, uses the existing
    /// `SharedGpuWorker` pattern.
    #[instrument(
        skip_all,
        fields(
            target = %plan.key().target.label(),
            lang = %plan.key().language,
        ),
    )]
    async fn dispatch_gpu_execute_v2(
        &self,
        plan: WorkerPlan,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        let (key, prepared) = plan.into_parts();
        let key = &key;
        let preparation = self.load_preparation();

        // Try TCP worker first (discovered from registry).
        if matches!(key.target, WorkerTarget::Profile(_))
            && let Some(tcp_worker) = self.live_gpu_tcp_worker(key).await
        {
            return dispatch_prepared!(prepared, ready => tcp_worker.execute_v2(ready, &preparation, TrackerGuard::new(&self.job_tracker, tcp_worker.pid())));
        }

        // Fall back to stdio worker.
        let gpu_worker = self.get_or_create_gpu_worker(key).await?;
        let result = dispatch_prepared!(prepared, ready => gpu_worker.execute_v2(ready, &preparation, TrackerGuard::new(&self.job_tracker, gpu_worker.pid())));
        if let Err(WorkerError::Timeout { .. }) = &result {
            // The worker thread serving the request may still be running it,
            // holding one of the worker's `gpu_thread_pool_size` slots for
            // good; enough of them and every request times out. Retire the
            // worker so the slot replaces it; the requests in flight on it
            // are answered `WorkerRetired` and retried.
            warn!(
                pid = %gpu_worker.pid(),
                "A shared GPU request timed out; retiring its worker"
            );
            let worker = Arc::clone(&gpu_worker);
            tokio::spawn(async move {
                worker
                    .shutdown(super::shared_gpu::Retirement::WorkerRetired)
                    .await;
            });
        }
        result
    }

    /// Load the command's task on the worker it selects, then probe that
    /// worker's capabilities.
    ///
    /// Startup may only have an optimistic command list with no infer-task
    /// metadata yet. Execution paths that need authoritative infer-task data
    /// call this to force one real worker bootstrap/probe before gating.
    ///
    /// The command's primary infer task is loaded FIRST (`ensure_task`, which
    /// is idempotent: a worker that already loaded the task answers at once),
    /// in every bootstrap mode, and only then is the report read. A worker
    /// supports a task before it has loaded it but names the engine only after,
    /// so the returned [`super::LoadedCapabilities`] is the post-load view: a
    /// lazily loading daemon probed before it loaded FA names its FA engine
    /// here. The report is admitted by [`WorkerPool::record_capabilities`], so
    /// the caller never sees (or re-admits) the raw report.
    ///
    /// A refused report retires the worker it came from on every branch (a
    /// registry GPU daemon is disconnected, a spawned GPU worker is shut down,
    /// a TCP handle is dropped and a checked-out worker is taken out of its
    /// group), so a worker whose report was refused is never used.
    pub async fn ensure_command_capabilities(
        &self,
        command: ReleasedCommand,
        lang: impl Into<WorkerLanguage>,
        options: &crate::options::CommandOptions,
        gpu: &super::GpuModelPlan,
    ) -> Result<super::LoadedCapabilities, WorkerError> {
        let requirement = crate::command_model::command_spec(command)
            .capabilities
            .require_inference()
            .map_err(|_| WorkerError::NativeCommand { command })?;
        let plan = CommandWorkerPlan::new(
            requirement,
            lang.into(),
            options,
            gpu,
            self.config.runtime.bootstrap_mode,
        );
        let key = plan.key();
        let load = plan.load();
        let task = load.task();
        let preparation = self.load_preparation();

        let reports = if self.serving(key.target).is_shared() {
            let registry_worker = if matches!(key.target, WorkerTarget::Profile(_)) {
                self.live_gpu_tcp_worker(key).await
            } else {
                None
            };
            match registry_worker {
                Some(worker) => {
                    let _registration = TrackerGuard::new(&self.job_tracker, worker.pid());
                    worker.ensure_task(load, &preparation).await?;
                    let caps = worker.capabilities().await?;
                    match self.record_capabilities(key, caps) {
                        Ok(reports) => reports,
                        Err(refusal) => {
                            self.gpu_tcp_workers.lock().await.remove(key);
                            worker
                                .shutdown(super::shared_gpu::Retirement::WorkerRetired)
                                .await;
                            return Err(refusal.into());
                        }
                    }
                }
                None => {
                    let worker = self.get_or_create_gpu_worker(key).await?;
                    let _registration = TrackerGuard::new(&self.job_tracker, worker.pid());
                    worker.ensure_task(load, &preparation).await?;
                    let caps = worker.capabilities().await?;
                    match self.record_capabilities(key, caps) {
                        Ok(reports) => reports,
                        Err(refusal) => {
                            self.gpu_workers.lock().await.remove(key);
                            worker
                                .shutdown(super::shared_gpu::Retirement::WorkerRetired)
                                .await;
                            return Err(refusal.into());
                        }
                    }
                }
            }
        } else if matches!(key.target, WorkerTarget::Profile(_))
            && let Some(checkout) = self.try_checkout_tcp(key)
        {
            checkout
                .run(async |worker| {
                    worker.ensure_task(load, &preparation).await?;
                    let caps = worker.capabilities().await?;
                    self.record_capabilities(key, caps)
                        .map_err(WorkerError::from)
                })
                .await?
        } else {
            self.checkout(key)
                .await?
                .run(async |worker| {
                    worker.ensure_task(load, &preparation).await?;
                    let caps = worker.capabilities().await?;
                    self.record_capabilities(key, caps)
                        .map_err(WorkerError::from)
                })
                .await?
        };
        Ok(super::LoadedCapabilities { task, reports })
    }
}

#[cfg(test)]
mod tcp_checkout_tests {
    use std::time::Duration;

    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

    use super::*;
    use crate::api::LanguageCode3;
    use crate::options::{CommandOptions, MorphotagOptions};
    use crate::worker::tcp_handle::TcpWorkerInfo;
    use crate::worker::{WorkerBootstrapMode, WorkerPid, WorkerProfile};

    /// A fake TCP worker daemon that reads one request and then either answers
    /// it with `reply` (and keeps the connection open), or, with no reply,
    /// closes the connection.
    async fn fake_daemon(reply: Option<&'static str>) -> u16 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind fake daemon");
        let port = listener.local_addr().expect("local addr").port();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept");
            let (read, mut write) = stream.into_split();
            let mut lines = tokio::io::BufReader::new(read).lines();
            let _request = lines.next_line().await;
            if let Some(reply) = reply {
                write.write_all(reply.as_bytes()).await.expect("reply");
                write.flush().await.expect("flush");
                tokio::time::sleep(Duration::from_secs(30)).await;
            }
        });
        port
    }

    /// A pool holding one TCP handle for morphotag's key, as registry
    /// discovery integrates one, plus that key and command options.
    async fn pool_with_one_tcp_handle(port: u16) -> (WorkerPool, WorkerKey, CommandOptions) {
        let mut config = super::super::PoolConfig::for_tests();
        config.runtime.bootstrap_mode = WorkerBootstrapMode::Profile;
        config.ensure_task_stall =
            crate::worker::LoadingStall::new(crate::api::PositiveSeconds::literal::<5>());
        let pool = WorkerPool::new(config);
        let options = CommandOptions::Morphotag(MorphotagOptions::default());
        let key = CommandWorkerPlan::new(
            crate::command_model::command_spec(ReleasedCommand::Morphotag)
                .capabilities
                .require_inference()
                .expect("morphotag requires inference"),
            WorkerLanguage::from(LanguageCode3::eng()),
            &options,
            &super::super::GpuModelPlan::without_engines(),
            pool.bootstrap_mode(),
        )
        .key()
        .clone();
        assert!(
            matches!(key.target, WorkerTarget::Profile(_)) && !pool.serving(key.target).is_shared(),
            "precondition: morphotag dispatches through the sequential TCP path"
        );
        let handle = TcpWorkerHandle::connect(TcpWorkerInfo {
            host: "127.0.0.1".into(),
            port,
            profile: WorkerProfile::Stanza,
            lang: WorkerLanguage::from(LanguageCode3::eng()),
            engine_overrides: String::new(),
            pid: WorkerPid(1),
            task_timeouts: crate::types::worker_v2::TaskTimeoutOverrides::NONE,
        })
        .await
        .expect("connect to fake daemon");
        let group = pool.get_or_create_group(&key);
        super::super::capacity::ReservedWorkerSlot::try_claim(&group, 1)
            .expect("admit the TCP fixture's actual slot")
            .publish_tcp(handle);
        (pool, key, options)
    }

    enum TaskReply {
        RuntimeRefusal,
        DownloadExit,
        WrongTask,
        ClosedConnection,
    }

    impl TaskReply {
        fn wire(&self) -> Option<&'static str> {
            match self {
                Self::RuntimeRefusal => Some(
                    "{\"op\":\"error\",\"error\":\"model load failed\",\"kind\":\"runtime\"}\n",
                ),
                Self::DownloadExit => Some(
                    "{\"op\":\"error\",\"error\":\"connection reset\",\"kind\":\"download\"}\n",
                ),
                Self::WrongTask => Some(
                    "{\"op\":\"ensure_task\",\"response\":{\"status\":\"loaded\",\"task\":\"fa\",\"elapsed_s\":0.1}}\n",
                ),
                Self::ClosedConnection => None,
            }
        }

        fn assert_error(&self, error: &WorkerError) {
            match self {
                Self::RuntimeRefusal => {
                    assert!(matches!(error, WorkerError::Bootstrap { .. }), "{error:?}")
                }
                Self::DownloadExit => assert!(
                    matches!(
                        error,
                        WorkerError::ModelDownload {
                            worker: crate::worker::error::WorkerAfterError::Exits,
                            ..
                        }
                    ),
                    "{error:?}"
                ),
                Self::WrongTask => assert!(
                    matches!(error, WorkerError::Protocol(message) if message.contains("answered about fa")),
                    "{error:?}"
                ),
                Self::ClosedConnection => assert!(
                    matches!(error, WorkerError::ProcessExited { .. }),
                    "{error:?}"
                ),
            }
        }

        fn retained_workers(&self) -> usize {
            match self {
                Self::RuntimeRefusal => 1,
                Self::DownloadExit | Self::WrongTask | Self::ClosedConnection => 0,
            }
        }
    }

    /// Guards: Exact task-preparation reply bytes and EOF classify errors truthfully and retain or retire the actual TCP worker with its original capacity slot.
    /// Kind: wire
    /// Not a type: Private slot publication prevents a bare TCP handle entering the pool and owns release. Runtime-refusal JSON, exit-promising download JSON, a loaded reply about fa instead of morphosyntax, and connection EOF still arrive as external bytes requiring transport and semantic admission.
    /// State: Each closed reply case owns a loopback daemon and isolated pool on the declared test host. The real TCP adapter must decode those bytes and close or retain its connection. A pure reply double cannot establish stream EOF or real connection lifecycle.
    /// Concurrency: Each daemon binds a unique ephemeral loopback port. Each case drops its pool/connection before the next case. No fixed port, models, shared runtime service or sleep establishes readiness.
    #[tokio::test]
    async fn task_preparation_replies_preserve_owned_tcp_capacity() {
        for reply in [
            TaskReply::RuntimeRefusal,
            TaskReply::DownloadExit,
            TaskReply::WrongTask,
            TaskReply::ClosedConnection,
        ] {
            let port = fake_daemon(reply.wire()).await;
            let (pool, key, options) = pool_with_one_tcp_handle(port).await;
            let permits_before = pool.spawn_permits.available_permits();
            let error = pool
                .ensure_command_capabilities(
                    ReleasedCommand::Morphotag,
                    WorkerLanguage::from(LanguageCode3::eng()),
                    &options,
                    &super::super::GpuModelPlan::without_engines(),
                )
                .await
                .expect_err("the declared wire reply refuses preparation");
            reply.assert_error(&error);
            let group = pool.get_or_create_group(&key);
            let retained = reply.retained_workers();
            assert_eq!(group.tcp_workers.view().len(), retained);
            assert_eq!(group.total.count(), retained);
            assert_eq!(
                pool.spawn_permits.available_permits(),
                permits_before + 1 - retained
            );
        }
    }
}

// The pool-spin fix (see `checkout`) is a structural invariant: every
// degenerate `permits > idle.len()` branch must have a `.await` before
// looping, so co-tenant tokio tasks (health check, HTTP server, other
// checkouts) can make progress. Meaningful runtime coverage of this
// requires the full `WorkerPool` with test-echo workers, a unit test
// over Semaphore + VecDeque alone only exercises the tiny state
// machine, not the real dispatch code path. That broader coverage
// lives in the test-echo integration tests alongside `WorkerPool`.

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::api::LanguageCode3;
    use crate::worker::{InferTask, WorkerTarget};

    use super::*;

    /// A pool whose one counted worker is checked out and held.
    fn saturated_pool() -> (
        std::sync::Arc<WorkerPool>,
        WorkerKey,
        super::super::capacity::ReservedWorkerSlot,
    ) {
        let pool = std::sync::Arc::new(WorkerPool::new(super::super::PoolConfig {
            max_workers_per_key: crate::host_facts::PerProfile::uniform(1),
            max_total_workers: 1,
            checkout_wait_report_interval: Some(crate::api::PositiveSeconds::literal::<1>()),
            ..super::super::PoolConfig::for_tests()
        }));
        let target = WorkerTarget::infer_task(InferTask::Morphosyntax);
        let lang = WorkerLanguage::from(LanguageCode3::eng());
        let key = WorkerKey::without_engine_selection(target, lang);
        let reservation = pool
            .try_claim_spawn_slot(&pool.get_or_create_group(&key))
            .expect("hold the actual admitted startup slot");
        (pool, key, reservation)
    }

    /// Guards: A legitimately held startup slot keeps a saturated checkout waiting across report intervals, and pool shutdown returns PoolShuttingDown.
    /// Kind: policy
    /// Not a type: The slot prevents fabricated saturation counts. Waiting indefinitely through congestion and selecting shutdown over further retry remain async scheduling policies observed through the actual checkout future.
    /// State: An isolated pool on the declared test host retains a real reservation. Its shared semaphore and cancellation token drive checkout; no worker process, socket, model or memory ledger is needed.
    /// Concurrency: One current-thread runtime uses paused Tokio time for ten report intervals. The task is joined after explicit cancellation and the original reservation is dropped.
    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_saturated_checkout_keeps_waiting_until_the_pool_shuts_down() {
        let (pool, key, held) = saturated_pool();

        let waiting = tokio::spawn({
            let pool = pool.clone();
            async move { pool.checkout(&key).await.map(drop) }
        });
        tokio::time::sleep(Duration::from_secs(10)).await;
        assert!(
            !waiting.is_finished(),
            "a saturated pool is congestion: the checkout must still be waiting"
        );

        pool.cancel.cancel();
        let outcome = waiting.await.expect("the checkout task does not panic");
        assert!(
            matches!(outcome, Err(WorkerError::PoolShuttingDown)),
            "only the pool's shutdown ends the wait, got {outcome:?}"
        );
        drop(held);
    }

    /// Guards: Actual profile limits, global capacity, per-profile metrics and refunds agree for the original admitted slot owners.
    /// Kind: policy
    /// Not a type: Unique slot ownership excludes forged counts, detached refunds and duplicate publication. The chosen 4/2/1 profile limits, global limit 8 and observable rejection/metrics projection still require semantic qualification against live reservations. Refunding speculative global admission must also wake its actual waiter.
    /// State: One isolated in-memory pool uses the declared test host. Reservations exercise the real semaphore and atomic projections without processes, models, sockets or a memory ledger lease.
    /// Concurrency: Reservations stay owned in a local vector. Drop and rejection ordering are explicit, with no spawned tasks or timing assumptions.
    #[tokio::test(flavor = "current_thread")]
    async fn try_claim_spawn_slot_uses_per_profile_cap() {
        let pool = WorkerPool::new(super::super::PoolConfig {
            max_workers_per_key: crate::host_facts::PerProfile {
                gpu: 2,
                stanza: 4,
                io: 1,
            },
            max_total_workers: 8,
            ..super::super::PoolConfig::for_tests()
        });
        let mut reservations = Vec::new();
        for (task, count) in [
            (InferTask::Morphosyntax, 4),
            (InferTask::Asr, 2),
            (InferTask::Translate, 1),
        ] {
            let key = WorkerKey::without_engine_selection(
                WorkerTarget::infer_task(task),
                WorkerLanguage::from(LanguageCode3::eng()),
            );
            let group = pool.get_or_create_group(&key);
            for _ in 0..count {
                reservations.push(
                    pool.try_claim_spawn_slot(&group)
                        .expect("admit up to the actual profile cap"),
                );
            }
            let changed = pool.spawn_permits.refund_event();
            tokio::pin!(changed);
            assert!(futures::poll!(&mut changed).is_pending());
            assert!(matches!(
                pool.try_claim_spawn_slot(&group),
                Err(super::super::lifecycle::AdmissionRejection::PerKeyCap)
            ));
            assert!(
                futures::poll!(&mut changed).is_ready(),
                "a speculative global refund wakes its waiter"
            );
            assert_eq!(group.total.count(), count);
        }
        let extra = pool.get_or_create_group(&WorkerKey::without_engine_selection(
            WorkerTarget::infer_task(InferTask::Coref),
            WorkerLanguage::from(LanguageCode3::eng()),
        ));
        reservations.push(
            pool.try_claim_spawn_slot(&extra)
                .expect("the eighth global slot fits"),
        );
        assert!(matches!(
            pool.try_claim_spawn_slot(&extra),
            Err(super::super::lifecycle::AdmissionRejection::GlobalCap)
        ));
        let metrics = pool.metrics_snapshot();
        assert_eq!(metrics.active_workers_total, 8);
        assert_eq!(
            (
                metrics.active_workers_by_profile.stanza,
                metrics.active_workers_by_profile.gpu,
                metrics.active_workers_by_profile.io
            ),
            (5, 2, 1)
        );
        assert_eq!(metrics.permits_available, 0);
        assert_eq!(metrics.spawn_rejections_total, 3);
        assert_eq!(metrics.permit_rejections_total, 1);
        drop(reservations);
        assert_eq!(pool.metrics_snapshot().active_workers_total, 0);
        assert_eq!(pool.spawn_permits.available_permits(), 8);
    }
}
