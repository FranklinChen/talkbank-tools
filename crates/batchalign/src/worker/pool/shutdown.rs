//! Pool shutdown and drop cleanup.
//!
//! `shutdown()` drains all worker groups gracefully: idle workers are shut down
//! immediately, checked-out workers are warned (they will be killed when their
//! RAII guard drops), GPU workers use their concurrent shutdown path, and
//! server-owned TCP daemons are killed via SIGTERM.
//!
//! The `Drop` impl (Layer 2) catches test code and panic unwinds where the pool
//! goes out of scope without a graceful `shutdown()` call.

use std::sync::atomic::Ordering;

use tracing::{info, warn};

use super::{WorkerPool, lock_recovered};

impl WorkerPool {
    /// Shut down all workers gracefully.
    ///
    /// Idle workers are shut down immediately.  Checked-out workers (currently
    /// processing a request) are logged as warnings -- they'll be killed when
    /// the `CheckedOutWorker` RAII guard drops. Shared GPU workers are shut
    /// down via their concurrent shutdown path. TCP daemon workers owned by the
    /// current server instance are killed via SIGTERM and removed from the
    /// registry. External daemons are preserved.
    pub async fn shutdown(&self) {
        self.cancel.cancel();

        // Retire only the TCP daemon workers owned by this server instance.
        let registry_path = self.registry_path();
        super::super::registry::kill_owned_daemons(
            &registry_path,
            self.current_server_instance_id(),
        );

        // Shut down shared GPU workers (stdio).
        //
        // A slot with no worker in it means a spawn for that key is in flight
        // right now. Since `self.cancel.cancel()` above happens BEFORE this
        // drain, that spawn's own completion path sees the cancelled token and
        // retires the worker it produces (see `get_or_create_gpu_worker`), so
        // dropping the empty slot here cannot orphan it. Waiting for it instead
        // would make shutdown block for a model load.
        {
            let mut gpu_workers = self.gpu_workers.lock().await;
            for (key, slot) in gpu_workers.drain() {
                match slot.retained_worker() {
                    Some(worker) => {
                        info!(
                            target = %key.target.label(),
                            lang = %key.language,
                            engine_selection = %key.engine_selection,
                            pid = %worker.pid(),
                            "Shutting down GPU worker"
                        );
                        worker.shutdown().await;
                    }
                    None => info!(
                        target = %key.target.label(),
                        lang = %key.language,
                        engine_selection = %key.engine_selection,
                        "GPU worker still spawning at shutdown; its spawner will retire it"
                    ),
                }
            }
        }

        // Disconnect shared TCP GPU workers (does not kill the daemon).
        {
            let mut tcp_gpu_workers = self.gpu_tcp_workers.lock().await;
            for (key, worker) in tcp_gpu_workers.drain() {
                info!(
                    target = %key.target.label(),
                    lang = %key.language,
                    engine_selection = %key.engine_selection,
                    pid = %worker.pid(),
                    "Disconnecting TCP GPU worker"
                );
                worker.shutdown().await;
            }
        }

        let all_groups: Vec<(super::WorkerKey, std::sync::Arc<super::WorkerGroup>)> = {
            let mut groups = lock_recovered(&self.groups);
            groups.drain().collect()
        };

        for (key, group) in all_groups {
            let workers: Vec<crate::worker::handle::WorkerHandle> =
                { lock_recovered(&group.idle).drain(..).collect() };
            let idle_count = workers.len();
            let total = group.total.load(Ordering::Relaxed);
            let checked_out = total.saturating_sub(idle_count);

            if checked_out > 0 {
                warn!(
                    target = %key.target.label(),
                    lang = %key.language,
                    engine_selection = %key.engine_selection,
                    checked_out,
                    "Workers still checked out during shutdown, \
                     they will be killed when their RAII guard drops"
                );
            }

            // Decrement total for drained workers and refund the
            // matching number of global-cap permits. (Checked-out
            // workers are not refunded here, their permits will
            // refund via CheckedOutWorker's drop/take paths once the
            // dispatch path returns. If shutdown races those returns,
            // the worst-case is a transient permit underflow that
            // resolves naturally.)
            group.total.fetch_sub(idle_count, Ordering::Relaxed);
            super::permit::SpawnPermitGuard::release_n(&group.spawn_permits, idle_count);

            for mut handle in workers {
                if let Err(e) = handle.shutdown_in_place().await {
                    warn!(
                        target = %key.target.label(),
                        lang = %key.language,
                        engine_selection = %key.engine_selection,
                        error = %e,
                        "Error shutting down worker"
                    );
                }
            }
        }
    }
}

/// Layer 2: kill idle workers when the pool is dropped without calling
/// `shutdown()`. This catches test code and panic unwinds where the pool
/// goes out of scope without graceful shutdown.
///
/// GPU workers behind `tokio::sync::Mutex` cannot be locked outside a
/// runtime, but their shared-worker owners are dropped when Arc refcounts
/// hit zero. The stdio variant's `Drop` impl kills the worker process; the
/// TCP variant only disconnects from the daemon it does not own.
impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.cancel.cancel();

        // Drain all groups and kill workers synchronously.
        // This works even outside a tokio context.
        let mut drained_groups = 0usize;
        let mut killed_handles = 0usize;
        if let Ok(mut groups) = self.groups.lock() {
            for (_, group) in groups.drain() {
                drained_groups += 1;
                if let Ok(mut idle) = group.idle.lock() {
                    for handle in idle.drain(..) {
                        // WorkerHandle::Drop sends SIGTERM+SIGKILL.
                        killed_handles += 1;
                        drop(handle);
                    }
                }
            }
        }

        // Retire the TCP daemons this pool spawned.
        //
        // Stdio workers die with their `WorkerHandle`, so the drain above (and
        // ordinary field drop, for the shared GPU maps) reaps them. A TCP daemon
        // is a DETACHED process with no handle to drop: it is retired only by
        // `kill_owned_daemons`, which `shutdown()` calls. Without the same call
        // here, this safety net did not cover the one worker kind that cannot
        // clean itself up, despite existing precisely for the case where
        // `shutdown()` was never reached.
        //
        // Read the instance id directly rather than through
        // `current_server_instance_id()`, which expects it to be present: this
        // runs during unwind, where a panic would abort the process. No id means
        // this pool never owned any daemon, so there is nothing to retire.
        let owns_daemons = self.config.runtime.server_instance_id.is_some();
        if let Some(instance_id) = self.config.runtime.server_instance_id.clone() {
            crate::worker::registry::kill_owned_daemons(&self.registry_path(), &instance_id);
        }

        // Evidence trail for the orphaned-worker question, which is still open:
        // real orphans (PPID 1, ~440 MB, one over six minutes old) were observed
        // after a full-suite run on 2026-07-30, while the binary that was blamed
        // leaked nothing when run alone. Three hypotheses survive, and this line
        // separates them at the next occurrence rather than requiring another
        // bisect:
        //
        //   - no line at all in the log  -> `Drop` never ran (leaked pool, or a
        //     process exit that skipped unwinding);
        //   - `drained_groups = 0`       -> `Drop` ran but the workers were not
        //     in this pool's map, so nothing here could ever have reaped them;
        //   - `owns_daemons = false`     -> no server instance id, so detached
        //     TCP daemons were never this pool's to retire.
        //
        // `debug` rather than `info`: this fires on every pool teardown,
        // including every test, and is wanted only when someone is hunting.
        // Enable with RUST_LOG=batchalign::worker::pool=debug.
        tracing::debug!(
            drained_groups,
            killed_handles,
            owns_daemons,
            "WorkerPool::drop completed"
        );
    }
}
