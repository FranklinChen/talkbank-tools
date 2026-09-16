//! Registry-based worker discovery.
//!
//! Reads `workers.json` to find pre-started TCP daemon workers and integrates
//! them into the pool. GPU workers become `SharedGpuTcpWorker` entries;
//! non-GPU workers join the sequential pool as `TcpWorkerHandle` entries.

use std::sync::atomic::Ordering;

use tracing::{info, warn};

use crate::worker::tcp_handle::{TcpWorkerHandle, TcpWorkerInfo};
use crate::worker::{WorkerPid, WorkerTarget, registry};

use super::{WorkerKey, WorkerPool, lock_recovered, shared_gpu};

impl WorkerPool {
    /// Discover pre-started TCP workers from the registry file.
    ///
    /// Reads `workers.json`, health-checks each entry, and integrates healthy
    /// workers into the pool. GPU workers become `SharedGpuWorker` entries;
    /// non-GPU workers are not integrated into the sequential pool (they use
    /// TCP handles directly via the dispatch path).
    ///
    /// Returns the number of workers discovered and integrated.
    pub async fn discover_from_registry(&self) -> usize {
        let registry_path = self.registry_path();

        let discovery = registry::discover_workers(
            &registry_path,
            self.config.audio_task_timeout_s,
            self.config.analysis_task_timeout_s,
            self.current_server_instance_id(),
        )
        .await;
        // Every daemon this sweep refused is recorded, replacing the previous
        // sweep's list, so `/health` shows daemons that are alive but not
        // adopted, and why.
        self.record_refused_registry_workers(
            discovery
                .refused
                .iter()
                .map(registry::ForeignBuildDaemon::to_refused_registry_worker)
                .collect(),
        );
        let discovered = discovery.workers;

        if discovered.is_empty() {
            return 0;
        }

        let count = discovered.len();
        info!(count, "Discovered worker(s) from registry");

        // Integrate GPU workers into the shared GPU worker map.
        // Non-GPU TCP workers are tracked in a separate TCP worker map.
        for mut worker in discovered {
            let target = WorkerTarget::profile(worker.profile);
            let key = match WorkerKey::from_registry_json(
                target,
                worker.lang.clone(),
                &worker.entry.engine_overrides,
            ) {
                Ok(key) => key,
                Err(error) => {
                    warn!(
                        profile = %worker.entry.profile,
                        lang = %worker.entry.lang,
                        engine_overrides = %worker.entry.engine_overrides,
                        %error,
                        "Skipping registry worker with invalid engine override JSON"
                    );
                    continue;
                }
            };
            // The one worker discovery probed carries its report; it is
            // admitted here, under the key it will be dispatched by. A refused
            // report means the worker is not used, so it is not integrated;
            // the refusal is that key's recorded admission, which `/health`
            // lists.
            if let Some(caps) = worker.capabilities.take()
                && let Err(error) = self.record_capabilities(&key, caps)
            {
                warn!(
                    profile = %worker.entry.profile,
                    lang = %worker.entry.lang,
                    %error,
                    "Registry worker's capability report was refused; not integrating it"
                );
                continue;
            }
            if target.is_concurrent() {
                let info = TcpWorkerInfo {
                    host: worker.entry.host.clone(),
                    port: worker.entry.port,
                    profile: worker.profile,
                    lang: worker.lang.clone(),
                    engine_overrides: worker.entry.engine_overrides.clone(),
                    pid: WorkerPid(worker.entry.pid),
                    audio_task_timeout_s: self.config.audio_task_timeout_s,
                    analysis_task_timeout_s: self.config.analysis_task_timeout_s,
                    gpu_thread_pool_size: self.config.runtime.gpu_thread_pool_size,
                };

                match shared_gpu::SharedGpuTcpWorker::connect(info).await {
                    Ok(shared) => {
                        self.gpu_tcp_workers
                            .lock()
                            .await
                            .entry(key)
                            .or_insert_with(|| std::sync::Arc::new(shared));
                        info!(
                            profile = %worker.entry.profile,
                            lang = %worker.entry.lang,
                            host = %worker.entry.host,
                            port = worker.entry.port,
                            "Integrated GPU TCP worker"
                        );
                    }
                    Err(e) => {
                        warn!(
                            host = %worker.entry.host,
                            port = worker.entry.port,
                            error = %e,
                            "Failed to integrate GPU TCP worker"
                        );
                    }
                }
            } else {
                // For non-GPU TCP workers, add them to the sequential pool.
                let info = TcpWorkerInfo {
                    host: worker.entry.host.clone(),
                    port: worker.entry.port,
                    profile: worker.profile,
                    lang: worker.lang.clone(),
                    engine_overrides: worker.entry.engine_overrides.clone(),
                    pid: WorkerPid(worker.entry.pid),
                    audio_task_timeout_s: self.config.audio_task_timeout_s,
                    analysis_task_timeout_s: self.config.analysis_task_timeout_s,
                    gpu_thread_pool_size: self.config.runtime.gpu_thread_pool_size,
                };

                match TcpWorkerHandle::connect(info).await {
                    Ok(handle) => {
                        let group = self.get_or_create_group(&key);
                        // Acquire a global-cap permit for the
                        // discovered worker. If the pool is full,
                        // skip rather than exceed the cap; a future
                        // discovery sweep can pick the daemon up.
                        let Some(permit_guard) =
                            super::permit::SpawnPermitGuard::try_acquire_or_skip(
                                &group.spawn_permits,
                                || {
                                    warn!(
                                        profile = %worker.entry.profile,
                                        lang = %worker.entry.lang,
                                        "Skipping TCP worker integration: global cap reached"
                                    );
                                },
                            )
                        else {
                            continue;
                        };
                        lock_recovered(&group.tcp_workers).push_back(handle);
                        group.tcp_available.add_permits(1);
                        group.total.fetch_add(1, Ordering::Relaxed);
                        permit_guard.forget();
                        info!(
                            profile = %worker.entry.profile,
                            lang = %worker.entry.lang,
                            host = %worker.entry.host,
                            port = worker.entry.port,
                            "Integrated non-GPU TCP worker into pool (key={:?})",
                            key
                        );
                    }
                    Err(e) => {
                        warn!(
                            host = %worker.entry.host,
                            port = worker.entry.port,
                            error = %e,
                            "Failed to integrate non-GPU TCP worker"
                        );
                    }
                }
            }
        }

        count
    }
}
