//! Worker checkout and dispatch routing.
//!
//! Contains the core `checkout()` loop (semaphore acquire → pop idle worker →
//! RAII guard), `dispatch_batch_infer`, `dispatch_execute_v2`, and TCP worker
//! checkout/return helpers. Routes keys whose workers serve concurrently
//! (`WorkerServing::SharedConcurrent`) to one shared worker; every other key
//! uses the exclusive-checkout model.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::api::{LanguageCode3, ReleasedCommand, WorkerLanguage};
use crate::types::worker_v2::{ExecuteRequestV2, ExecuteResponseV2};
use crate::worker::error::{WorkerAfterFailure, WorkerError};
use crate::worker::tcp_handle::TcpWorkerHandle;
use crate::worker::{BatchInferRequest, BatchInferResponse, WorkerBootstrapMode, WorkerTarget};
use tracing::{instrument, warn};

use super::checkout::CheckedOutWorker;
use super::checkout_wait::{GroupCounts, Parked, SaturatedWait};
use super::eviction::EvictionOutcome;
use super::execute_v2::{self, execute_v2_worker_key};
use super::job_tracker::TrackerGuard;
use super::{AwayFromQueue, WorkerGroup, WorkerKey, WorkerPool, lock_recovered};

/// A TCP worker handle checked out of its group for one exchange.
///
/// Owns the handle and its group slot together, so no exit path can lose
/// either. [`Self::finish`] reads what the exchange left: the handle goes back
/// to its group unless the error means the worker must be retired
/// ([`WorkerError::worker_after_failure`]), in which case the handle is
/// dropped and the slot released. Dropped without `finish` (the exchange
/// future was cancelled mid-request), the handle is dropped and the slot
/// released too: a request may be half-sent or a response half-read, and a
/// later exchange on that stream could read a stale reply as its own.
pub(super) struct TcpCheckout {
    handle: TcpWorkerHandle,
    lease: TcpSlotLease,
    /// Accounts for this worker while it is out of the TCP queue.
    _away: AwayFromQueue,
}

impl TcpCheckout {
    /// The checked-out handle, for the exchange.
    pub(super) fn handle(&mut self) -> &mut TcpWorkerHandle {
        &mut self.handle
    }

    /// End the exchange: return the handle to its group, or retire it.
    pub(super) fn finish<T>(self, outcome: &Result<T, WorkerError>) {
        // The away guard is held to the end, after the handle is back.
        let Self {
            handle,
            lease,
            _away,
        } = self;
        match outcome
            .as_ref()
            .map(|_| ())
            .map_err(WorkerError::worker_after_failure)
        {
            Ok(()) | Err(WorkerAfterFailure::Reusable) => lease.return_handle(handle),
            Err(WorkerAfterFailure::Retire) => {
                warn!(
                    pid = %handle.pid(),
                    "Retiring TCP worker handle after a failed exchange; the daemon is \
                     left running and is adopted again by the next registry sweep"
                );
                // Dropping the handle closes the connection; dropping the lease
                // releases the slot.
                drop(handle);
                drop(lease);
            }
        }
    }
}

/// The group slot a checked-out TCP handle occupies.
///
/// Which groups a saturated checkout's reconciliation looks at: the groups
/// whose accounting can keep what it is parked on from coming back.
#[derive(Debug, Clone, Copy)]
enum ReconcileScope<'a> {
    /// Parked on its own group's permits.
    OwnGroup(&'a WorkerKey),
    /// Parked on the pool-wide worker-returned signal.
    Pool,
}

/// Releases the slot (the group's live count and its global worker permit)
/// when dropped, unless [`Self::return_handle`] gave the handle back first.
struct TcpSlotLease {
    /// `Some` until the slot is either given back with its handle or released.
    group: Option<Arc<WorkerGroup>>,
}

impl TcpSlotLease {
    /// Give the handle back to its group; the slot stays occupied by it.
    fn return_handle(mut self, handle: TcpWorkerHandle) {
        if let Some(group) = self.group.take() {
            lock_recovered(&group.tcp_workers).push_back(handle);
            group.tcp_available.add_permits(1);
        }
    }
}

impl Drop for TcpSlotLease {
    fn drop(&mut self) {
        if let Some(group) = self.group.take() {
            group.total.fetch_sub(1, Ordering::Relaxed);
            group.spawn_permits.add_permits(1);
        }
    }
}

impl WorkerPool {
    /// Check out an idle worker or spawn a new one.
    ///
    /// 1. Try to acquire a semaphore permit immediately.
    /// 2. If none available, try to spawn a new worker (if under capacity).
    /// 3. If at capacity, wait for a worker to come back (async suspend).
    /// 4. Pop from the idle queue and wrap in `CheckedOutWorker` (RAII guard).
    ///
    /// A saturated pool is congestion, not a failure of the request: the
    /// wait has no deadline, only a report interval after which it is logged
    /// and re-probed. It ends when a worker is available, when the pool
    /// shuts down, or when the caller drops it. See [`super::checkout_wait`].
    pub(super) async fn checkout(&self, key: &WorkerKey) -> Result<CheckedOutWorker, WorkerError> {
        let group = self.get_or_create_group(key);
        let report_interval = self.config.checkout_wait_report_interval();
        // Begun at the first park, so a checkout that never waits reports
        // nothing; dropped on return, which ends its observation.
        let mut saturated: Option<SaturatedWait> = None;
        let group_counts = || GroupCounts {
            total: group.total.load(Ordering::Relaxed),
            idle: lock_recovered(&group.idle).len(),
        };

        loop {
            // Invariant (normal operation): `group.available.permits() ==
            // group.idle.len()`. When the invariant holds, a successful
            // `try_acquire` is matched by a non-empty idle queue and we
            // return immediately.
            //
            // Degenerate case: `permits > idle.len()`, a permit without a
            // matching worker. The health-check task drains idle before
            // re-adding permits for survivors, so this window is real.
            // Returning the permit and falling through is correct; the
            // `yield_now().await` is LOAD-BEARING. Without it, a tight
            // `try_acquire → pop-None → add_permits → continue` cycle
            // starves the tokio runtime, other tasks that might refill
            // idle never run, and one worker thread burns a core.
            if let Ok(permit) = group.available.try_acquire() {
                permit.forget();
                if let Some(handle) = lock_recovered(&group.idle).pop_front() {
                    return Ok(CheckedOutWorker::from_idle(&group, handle));
                }
                group.available.add_permits(1);
                tokio::task::yield_now().await;
            }

            // Slow path: try to spawn a new worker (if under cap).
            match self.try_spawn_into_group(&group, key).await {
                Ok(true) => {
                    // Spawned: loop back: the new permit is backed by
                    // a real enqueued worker, so the fast path will hit.
                    continue;
                }
                Ok(false) => {
                    // At capacity. If this group already has live
                    // workers they will eventually return permits
                    // fall through to the async wait below. Otherwise
                    // try to free a slot by evicting an idle worker
                    // from another group; if that fails, park on the
                    // pool-wide `worker_returned` Notify.
                    if group.is_empty() {
                        // Register on `worker_returned` BEFORE the
                        // eviction probe. `Notified::enable()` puts
                        // this task on the wait list without polling
                        // the future, so any `notify_one()` that
                        // fires during the probe is delivered here
                        // instead of being absorbed by the Notify's
                        // single-slot buffer (a burst of N>1 returns
                        // would otherwise lose N−1 wakeups, forcing
                        // late-comers to wait a whole report interval).
                        // checkout.rs uses `notify_one()` (not
                        // `notify_waiters()`) so each return wakes
                        // exactly one waiter: the BUG-028 herd fix.
                        let notified = self.worker_returned.notified();
                        tokio::pin!(notified);
                        notified.as_mut().enable();

                        if let EvictionOutcome::Evicted = self.try_evict_idle_from_other_group(key)
                        {
                            continue;
                        }

                        // A wakeup and an elapsed report interval both
                        // re-probe from the top; a shutdown, or accounting
                        // that stays broken, ends it.
                        let wait = saturated.get_or_insert_with(|| {
                            SaturatedWait::begin(&key.target, &key.language, report_interval)
                        });
                        match wait.park(notified, &self.cancel, group_counts).await? {
                            Parked::Ready(()) => {}
                            Parked::StillSaturated => {
                                // Parked on the pool-wide return signal: any
                                // group's broken count can be what blocks it.
                                wait.reconcile(self.unaccounted_groups(ReconcileScope::Pool))?;
                            }
                        }
                        continue;
                    }
                    // fall through to this-key async wait below
                }
                Err(e) => return Err(e),
            }

            // All workers busy and at capacity. Wait for a permit. An elapsed
            // report interval re-probes from the top, so a stale `total > 0`
            // count or a group that emptied meanwhile is re-read rather than
            // waited on forever without a word.
            let wait = saturated.get_or_insert_with(|| {
                SaturatedWait::begin(&key.target, &key.language, report_interval)
            });
            let permit = match wait
                .park(group.available.acquire(), &self.cancel, group_counts)
                .await?
            {
                Parked::Ready(acquired) => acquired
                    .map_err(|_| WorkerError::SpawnFailed("worker pool semaphore closed".into()))?,
                Parked::StillSaturated => {
                    // Parked on this group's own permits: only its own
                    // accounting can keep them from coming back.
                    wait.reconcile(self.unaccounted_groups(ReconcileScope::OwnGroup(key)))?;
                    continue;
                }
            };
            permit.forget();

            if let Some(handle) = lock_recovered(&group.idle).pop_front() {
                return Ok(CheckedOutWorker::from_idle(&group, handle));
            }
            // Rare: async-acquire returned a permit but idle is empty.
            // Same load-bearing yield as above, returning the permit
            // without yielding would reintroduce the runtime starvation.
            group.available.add_permits(1);
            tokio::task::yield_now().await;
        }
    }

    /// Every group in `scope` whose live count exceeds what holds its
    /// workers. Lock order is groups, then each group's queues, as
    /// elsewhere.
    fn unaccounted_groups(
        &self,
        scope: ReconcileScope<'_>,
    ) -> Vec<super::checkout_wait::UnaccountedGroup> {
        let groups = lock_recovered(&self.groups);
        groups
            .iter()
            .filter(|(key, _)| match scope {
                ReconcileScope::OwnGroup(own) => *key == own,
                ReconcileScope::Pool => true,
            })
            .filter_map(|(key, group)| {
                let unaccounted = group.unaccounted();
                (unaccounted > 0).then(|| super::checkout_wait::UnaccountedGroup {
                    key: key.clone(),
                    unaccounted,
                    total: group.total.load(Ordering::Relaxed),
                    idle: lock_recovered(&group.idle).len()
                        + lock_recovered(&group.tcp_workers).len(),
                    away: group.away.load(Ordering::Relaxed),
                })
            })
            .collect()
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

        // Try TCP worker first.
        if matches!(key.target, WorkerTarget::Profile(_))
            && let Some(mut checkout) = self.try_checkout_tcp(&key)
        {
            let result = checkout.handle().batch_infer(request).await;
            checkout.finish(&result);
            return result;
        }

        // Fall back to stdio worker. TrackerGuard registers this
        // worker against the current job for cancel-driven shutdown;
        // auto-unregisters on drop.
        let mut worker = self.checkout(&key).await?;
        let _job_guard = TrackerGuard::new(&self.job_tracker, worker.pid());
        let result = worker.batch_infer(request).await;

        // If the worker crashed (I/O error or process exit), it is dead and
        // must not be returned to the idle queue.  Discard it via `take()` so
        // the pool decrements `total` (freeing a slot), then retry once with a
        // freshly spawned replacement.
        //
        // Without this, `Drop` would silently return the corpse to the idle
        // queue, causing the *next* dispatch to also fail with BrokenPipe.
        //
        // Protocol errors also warrant discarding the worker (the stdio stream
        // may be desynchronized), but we do NOT retry them, the framing break
        // may be input-specific and a retry could hang.
        match result {
            Err(ref e @ (WorkerError::Io(_) | WorkerError::ProcessExited { .. })) => {
                warn!(
                    error = %e,
                    "worker crashed during batch_infer: discarding and retrying with a fresh worker"
                );
                worker.take(); // decrement total, do NOT return to idle queue
                drop(worker); // Drop now sees None handle, does nothing
                let mut fresh = self.checkout(&key).await?;
                fresh.batch_infer(request).await
            }
            Err(ref e @ WorkerError::Protocol(_)) => {
                // Desynchronized stream: discard without retry.
                warn!(error = %e, "worker protocol error, discarding worker");
                worker.take();
                result
            }
            other => other,
        }
    }

    /// Try to check out a TCP worker handle (non-blocking), as an owned
    /// [`TcpCheckout`] that gives it back or retires it.
    pub(super) fn try_checkout_tcp(&self, key: &WorkerKey) -> Option<TcpCheckout> {
        let group = lock_recovered(&self.groups).get(key)?.clone();
        let permit = group.tcp_available.try_acquire().ok()?;
        let Some(handle) = lock_recovered(&group.tcp_workers).pop_front() else {
            // A permit with no handle behind it: give the permit back rather
            // than lose it.
            drop(permit);
            return None;
        };
        permit.forget();
        Some(TcpCheckout {
            handle,
            _away: AwayFromQueue::new(&group, 1),
            lease: TcpSlotLease { group: Some(group) },
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

    /// Dispatch a V2 execute request, forwarding intermediate progress events
    /// through an optional async channel.
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
        let lang = lang.into();
        let key = execute_v2_worker_key(lang, request, self.config.runtime.bootstrap_mode)?;

        if self.serving(key.target).is_shared() {
            // Shared workers don't support progress forwarding yet.
            return self.dispatch_gpu_execute_v2(&key, request).await;
        }

        // In LazyProfile mode the worker started with no models, so the task's
        // models are loaded before dispatching (idempotent if already loaded).
        // The task is read from the request once, before any worker is taken,
        // so a malformed request never retires a worker.
        let lazy_task = match self.config.runtime.bootstrap_mode {
            WorkerBootstrapMode::LazyProfile => Some(execute_v2::ensure_task_params(request)?),
            WorkerBootstrapMode::Profile | WorkerBootstrapMode::Task => None,
        };
        let timeout = self.config.effective_ensure_task_timeout();

        // Try TCP worker first.
        if matches!(key.target, WorkerTarget::Profile(_))
            && let Some(mut checkout) = self.try_checkout_tcp(&key)
        {
            let result = async {
                if let Some((task, overrides)) = &lazy_task {
                    checkout
                        .handle()
                        .ensure_task(*task, overrides.as_ref(), timeout)
                        .await?;
                }
                checkout
                    .handle()
                    .execute_v2_with_progress(request, progress_tx)
                    .await
            }
            .await;
            checkout.finish(&result);
            return result;
        }

        // Fall back to stdio worker.
        let mut worker = self.checkout(&key).await?;
        let _job_guard = TrackerGuard::new(&self.job_tracker, worker.pid());
        if let Some((task, overrides)) = &lazy_task {
            worker
                .ensure_task(*task, overrides.as_ref(), timeout)
                .await?;
        }

        worker.execute_v2_with_progress(request, progress_tx).await
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

    /// Dispatch a V2 execute request to a GPU worker.
    ///
    /// Tries TCP workers first (discovered from registry), then falls back to
    /// stdio workers. For TCP workers, multiple callers share one worker via
    /// concurrent dispatch. For stdio workers, uses the existing
    /// `SharedGpuWorker` pattern.
    #[instrument(
        skip_all,
        fields(
            target = %key.target.label(),
            lang = %key.language,
            request_id = %request.request_id,
        ),
    )]
    async fn dispatch_gpu_execute_v2(
        &self,
        key: &WorkerKey,
        request: &ExecuteRequestV2,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        // Try TCP worker first (discovered from registry).
        if matches!(key.target, WorkerTarget::Profile(_))
            && let Some(tcp_worker) = self.live_gpu_tcp_worker(key).await
        {
            let _job_guard = TrackerGuard::new(&self.job_tracker, tcp_worker.pid());
            if self.config.runtime.bootstrap_mode == WorkerBootstrapMode::LazyProfile {
                let (task, overrides) = execute_v2::ensure_task_params(request)?;
                let timeout = self.config.effective_ensure_task_timeout();
                tcp_worker
                    .ensure_task(task, overrides.as_ref(), timeout)
                    .await?;
            }
            return tcp_worker.execute_v2(request).await;
        }

        // Fall back to stdio worker.
        let gpu_worker = self.get_or_create_gpu_worker(key).await?;
        let _job_guard = TrackerGuard::new(&self.job_tracker, gpu_worker.pid());

        if self.config.runtime.bootstrap_mode == WorkerBootstrapMode::LazyProfile {
            let (task, overrides) = execute_v2::ensure_task_params(request)?;
            let timeout = self.config.effective_ensure_task_timeout();
            gpu_worker
                .ensure_task(task, overrides.as_ref(), timeout)
                .await?;
        }

        let result = gpu_worker.execute_v2(request).await;
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
    ) -> Result<super::LoadedCapabilities, WorkerError> {
        let requirement = crate::command_model::command_spec(command)
            .capabilities
            .require_inference()
            .map_err(|_| WorkerError::NativeCommand { command })?;
        let key = WorkerKey::from_command_options(
            requirement,
            lang.into(),
            options,
            self.config.runtime.bootstrap_mode,
        );
        let loaded_task = requirement.task();
        let overrides = key.engine_selection.overrides().dispatch_overrides();
        let overrides = (!overrides.is_empty()).then_some(overrides);
        let timeout = self.config.effective_ensure_task_timeout();

        let reports = if self.serving(key.target).is_shared() {
            let registry_worker = if matches!(key.target, WorkerTarget::Profile(_)) {
                self.live_gpu_tcp_worker(&key).await
            } else {
                None
            };
            match registry_worker {
                Some(worker) => {
                    worker
                        .ensure_task(loaded_task, overrides.as_ref(), timeout)
                        .await?;
                    let caps = worker.capabilities().await?;
                    match self.record_capabilities(&key, caps) {
                        Ok(reports) => reports,
                        Err(refusal) => {
                            self.gpu_tcp_workers.lock().await.remove(&key);
                            worker
                                .shutdown(super::shared_gpu::Retirement::WorkerRetired)
                                .await;
                            return Err(refusal.into());
                        }
                    }
                }
                None => {
                    let worker = self.get_or_create_gpu_worker(&key).await?;
                    worker
                        .ensure_task(loaded_task, overrides.as_ref(), timeout)
                        .await?;
                    let caps = worker.capabilities().await?;
                    match self.record_capabilities(&key, caps) {
                        Ok(reports) => reports,
                        Err(refusal) => {
                            self.gpu_workers.lock().await.remove(&key);
                            worker
                                .shutdown(super::shared_gpu::Retirement::WorkerRetired)
                                .await;
                            return Err(refusal.into());
                        }
                    }
                }
            }
        } else if matches!(key.target, WorkerTarget::Profile(_))
            && let Some(mut checkout) = self.try_checkout_tcp(&key)
        {
            // Load, probe and admit as one exchange, so a refused report
            // retires the handle exactly as a broken connection would.
            let admitted = async {
                checkout
                    .handle()
                    .ensure_task(loaded_task, overrides.as_ref(), timeout)
                    .await?;
                let caps = checkout.handle().capabilities().await?;
                self.record_capabilities(&key, caps)
                    .map_err(WorkerError::from)
            }
            .await;
            checkout.finish(&admitted);
            admitted?
        } else {
            let mut worker = self.checkout(&key).await?;
            worker
                .ensure_task(loaded_task, overrides.as_ref(), timeout)
                .await?;
            let caps = worker.capabilities().await?;
            match self.record_capabilities(&key, caps) {
                Ok(reports) => reports,
                Err(refusal) => {
                    // `take` releases the slot; the handle drops here, which
                    // terminates the worker process.
                    drop(worker.take());
                    return Err(refusal.into());
                }
            }
        };
        Ok(super::LoadedCapabilities {
            task: loaded_task,
            reports,
        })
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
    use crate::worker::{WorkerPid, WorkerProfile};

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
        let mut config = super::super::PoolConfig::default();
        config.runtime.bootstrap_mode = WorkerBootstrapMode::Profile;
        config.ensure_task_timeout = Some(crate::api::PositiveSeconds::literal::<5>());
        let pool = WorkerPool::new(config);
        let options = CommandOptions::Morphotag(MorphotagOptions::default());
        let key = WorkerKey::from_command_options(
            crate::command_model::command_spec(ReleasedCommand::Morphotag)
                .capabilities
                .require_inference()
                .expect("morphotag requires inference"),
            WorkerLanguage::from(LanguageCode3::eng()),
            &options,
            pool.bootstrap_mode(),
        );
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
            gpu_thread_pool_size: 1,
        })
        .await
        .expect("connect to fake daemon");
        let group = pool.get_or_create_group(&key);
        group
            .spawn_permits
            .try_acquire()
            .expect("a global worker permit")
            .forget();
        lock_recovered(&group.tcp_workers).push_back(handle);
        group.tcp_available.add_permits(1);
        group.total.fetch_add(1, Ordering::Relaxed);
        (pool, key, options)
    }

    /// `ensure_task` failing with a complete error response leaves the
    /// connection in step: the handle goes back to its group. Before the owned
    /// checkout, the `?` returned early and the handle was lost for good.
    #[tokio::test]
    async fn a_worker_error_response_returns_the_handle_to_its_group() {
        let port = fake_daemon(Some(
            "{\"op\":\"error\",\"error\":\"model load failed\",\"kind\":\"bootstrap\"}\n",
        ))
        .await;
        let (pool, key, options) = pool_with_one_tcp_handle(port).await;
        let permits_before = pool.spawn_permits.available_permits();

        let error = pool
            .ensure_command_capabilities(
                ReleasedCommand::Morphotag,
                WorkerLanguage::from(LanguageCode3::eng()),
                &options,
            )
            .await
            .expect_err("the fake daemon refuses to load the task");
        assert!(matches!(error, WorkerError::Bootstrap(_)), "{error:?}");

        let group = pool.get_or_create_group(&key);
        assert_eq!(lock_recovered(&group.tcp_workers).len(), 1);
        assert_eq!(group.tcp_available.available_permits(), 1);
        assert_eq!(group.total.load(Ordering::Relaxed), 1);
        assert_eq!(pool.spawn_permits.available_permits(), permits_before);
    }

    /// An `ensure_task` reply about another task does not load the asked
    /// task: on a single-flight connection it is a protocol violation.
    #[tokio::test]
    async fn an_ensure_task_reply_about_another_task_is_a_protocol_error() {
        let port = fake_daemon(Some(
            "{\"op\":\"ensure_task\",\"response\":{\"status\":\"loaded\",\"task\":\"fa\",\"elapsed_s\":0.1}}\n",
        ))
        .await;
        let (pool, _key, options) = pool_with_one_tcp_handle(port).await;
        let error = pool
            .ensure_command_capabilities(
                ReleasedCommand::Morphotag,
                WorkerLanguage::from(LanguageCode3::eng()),
                &options,
            )
            .await
            .expect_err("a reply about fa does not answer ensure_task(morphosyntax)");
        assert!(
            matches!(&error, WorkerError::Protocol(message) if message.contains("answered about fa")),
            "{error:?}"
        );
    }

    /// A closed connection is the request losing its worker, as on the
    /// shared GPU paths (`ProcessExited`, retryable): the handle is retired
    /// and its slot released, not returned to serve another request.
    #[tokio::test]
    async fn a_closed_connection_retires_the_handle_and_releases_its_slot() {
        let port = fake_daemon(None).await;
        let (pool, key, options) = pool_with_one_tcp_handle(port).await;
        let permits_before = pool.spawn_permits.available_permits();

        let error = pool
            .ensure_command_capabilities(
                ReleasedCommand::Morphotag,
                WorkerLanguage::from(LanguageCode3::eng()),
                &options,
            )
            .await
            .expect_err("the fake daemon closed the connection");
        assert!(
            matches!(error, WorkerError::ProcessExited { .. }),
            "{error:?}"
        );

        let group = pool.get_or_create_group(&key);
        assert!(lock_recovered(&group.tcp_workers).is_empty());
        assert_eq!(group.tcp_available.available_permits(), 0);
        assert_eq!(group.total.load(Ordering::Relaxed), 0);
        assert_eq!(pool.spawn_permits.available_permits(), permits_before + 1);
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
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use crate::api::LanguageCode3;
    use crate::worker::{InferTask, WorkerTarget};

    use super::*;

    /// A pool whose one counted worker is checked out and held.
    fn saturated_pool() -> (std::sync::Arc<WorkerPool>, WorkerKey) {
        let pool = std::sync::Arc::new(WorkerPool::new(super::super::PoolConfig {
            max_workers_per_key: crate::host_facts::PerProfile::uniform(1),
            max_total_workers: 1,
            checkout_wait_report_interval: Some(crate::api::PositiveSeconds::literal::<1>()),
            ..Default::default()
        }));
        let target = WorkerTarget::infer_task(InferTask::Morphosyntax);
        let lang = WorkerLanguage::from(LanguageCode3::eng());
        let key = WorkerKey::without_engine_selection(target, lang);
        pool.get_or_create_group(&key)
            .total
            .store(1, Ordering::Relaxed);
        (pool, key)
    }

    /// RED FIRST (2026-10-06): a saturated checkout waits; it does not fail.
    ///
    /// The group's one counted worker is held (by a checkout that never
    /// returns it), so nothing can be checked out. This used to return
    /// `SpawnFailed("no worker available ... pool saturated")` once a fixed
    /// deadline passed, which failed the file that was waiting. Now ten
    /// report intervals pass and the checkout is still waiting (reported and
    /// reconciled each interval: the held worker is accounted for), and only
    /// the pool's shutdown ends it, with its own typed error.
    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_saturated_checkout_keeps_waiting_until_the_pool_shuts_down() {
        let (pool, key) = saturated_pool();
        let held = AwayFromQueue::new(&pool.get_or_create_group(&key), 1);

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

    /// A counted worker that nothing holds (a stale `total`) is not
    /// congestion: no worker will ever come back. Seen at two consecutive
    /// report intervals, the wait ends with the typed, terminal accounting
    /// failure instead of waiting forever.
    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_count_nothing_holds_ends_the_wait_as_broken_accounting() {
        let (pool, key) = saturated_pool();
        let waiting = tokio::spawn({
            let pool = pool.clone();
            async move { pool.checkout(&key).await.map(drop) }
        });
        tokio::time::sleep(Duration::from_millis(1_500)).await;
        assert!(!waiting.is_finished(), "one mismatch is only reported");
        tokio::time::sleep(Duration::from_secs(2)).await;
        let outcome = waiting.await.expect("the checkout task does not panic");
        let Err(error @ WorkerError::PoolAccountingBroken { .. }) = outcome else {
            panic!("a persistent mismatch is broken accounting, got {outcome:?}");
        };
        assert!(
            matches!(
                error,
                WorkerError::PoolAccountingBroken {
                    unaccounted: 1,
                    total: 1,
                    idle: 0,
                    away: 0,
                    ..
                }
            ),
            "{error}"
        );
        assert_eq!(
            crate::runner::util::classify_worker_error(&error),
            crate::scheduling::FailureCategory::System
        );
    }

    /// A checkout parked on its own group's permits waits on that group
    /// alone: another group's broken count is not what keeps its worker from
    /// coming back, and does not fail it.
    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn another_groups_broken_count_does_not_fail_an_own_group_wait() {
        let (pool, key) = saturated_pool();
        let held = AwayFromQueue::new(&pool.get_or_create_group(&key), 1);
        let other = WorkerKey::without_engine_selection(
            WorkerTarget::infer_task(InferTask::Utseg),
            WorkerLanguage::from(LanguageCode3::eng()),
        );
        pool.get_or_create_group(&other)
            .total
            .store(1, Ordering::Relaxed);

        let waiting = tokio::spawn({
            let pool = pool.clone();
            async move { pool.checkout(&key).await.map(drop) }
        });
        tokio::time::sleep(Duration::from_secs(10)).await;
        assert!(
            !waiting.is_finished(),
            "the own group's worker is held: this is congestion, whatever another group shows"
        );
        pool.cancel.cancel();
        let outcome = waiting.await.expect("the checkout task does not panic");
        assert!(
            matches!(outcome, Err(WorkerError::PoolShuttingDown)),
            "{outcome:?}"
        );
        drop(held);
    }

    /// Admission control reads the per-profile cap from
    /// `PoolConfig::max_workers_per_key` based on the requesting
    /// group's `WorkerProfile`. Different profiles can have different
    /// caps; one profile saturating must not affect another.
    #[tokio::test(flavor = "current_thread")]
    async fn try_claim_spawn_slot_uses_per_profile_cap() {
        let pool = WorkerPool::new(super::super::PoolConfig {
            max_workers_per_key: crate::host_facts::PerProfile {
                gpu: 2,
                stanza: 4,
                io: 1,
            },
            max_total_workers: 64, // not the binding constraint here
            // Disable the CPU-loadavg gate so this test isolates
            // per-profile-cap behavior. CI runners are CPU-saturated
            // by parallel cargo-test workers and would otherwise
            // reject every claim with CpuSaturated.
            cpu_gate_threshold_override: Some(f64::INFINITY),
            ..Default::default()
        });
        let lang = WorkerLanguage::from(LanguageCode3::eng());

        // Stanza profile: cap 4. Filling group.total to 3 still admits;
        // 4 rejects.
        let stanza_target = WorkerTarget::infer_task(InferTask::Morphosyntax);
        assert_eq!(
            stanza_target.profile_kind(),
            crate::worker::WorkerProfile::Stanza
        );
        let stanza_key = WorkerKey::without_engine_selection(stanza_target, lang.clone());
        let stanza_group = pool.get_or_create_group(&stanza_key);
        stanza_group.total.store(3, Ordering::Relaxed);
        assert!(
            pool.try_claim_spawn_slot(&stanza_group).is_ok(),
            "stanza cap=4 must admit when current=3"
        );
        // The successful claim incremented total to 4; the next probe
        // must reject.
        assert!(
            pool.try_claim_spawn_slot(&stanza_group).is_err(),
            "stanza cap=4 must reject when current=4"
        );

        // GPU profile: cap 2 (lower). Independent group; not affected
        // by stanza saturation.
        let gpu_target = WorkerTarget::infer_task(InferTask::Asr);
        assert_eq!(gpu_target.profile_kind(), crate::worker::WorkerProfile::Gpu);
        let gpu_key = WorkerKey::without_engine_selection(gpu_target, lang.clone());
        let gpu_group = pool.get_or_create_group(&gpu_key);
        gpu_group.total.store(1, Ordering::Relaxed);
        assert!(
            pool.try_claim_spawn_slot(&gpu_group).is_ok(),
            "gpu cap=2 must admit when current=1 (independent of stanza saturation)"
        );
        assert!(
            pool.try_claim_spawn_slot(&gpu_group).is_err(),
            "gpu cap=2 must reject when current=2"
        );

        // IO profile: cap 1 (smallest). At cap, rejects.
        let io_target = WorkerTarget::infer_task(InferTask::Translate);
        assert_eq!(io_target.profile_kind(), crate::worker::WorkerProfile::Io);
        let io_key = WorkerKey::without_engine_selection(io_target, lang);
        let io_group = pool.get_or_create_group(&io_key);
        io_group.total.store(1, Ordering::Relaxed);
        assert!(
            pool.try_claim_spawn_slot(&io_group).is_err(),
            "io cap=1 must reject when current=1"
        );
    }
}
