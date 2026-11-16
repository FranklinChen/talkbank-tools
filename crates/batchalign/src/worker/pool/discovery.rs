//! Integrate source-bound, lifetime-owned registry connections.
use super::WorkerPool;
use crate::worker::registry;
use crate::worker::serving::WorkerServing;
use std::sync::Arc;
use tracing::{info, warn};
impl WorkerPool {
    /// Observe unowned candidates and return the actual newly integrated count.
    pub async fn discover_from_registry(&self) -> usize {
        let discovery = registry::discover_workers(
            &self.registry_path(),
            self.config.task_timeouts,
            self.current_server_instance_id(),
            &self.registry_admissions,
        )
        .await;
        self.record_refused_registry_workers(
            discovery
                .refused
                .iter()
                .map(registry::ForeignBuildDaemon::to_refused_registry_worker)
                .collect(),
        );
        let mut integrated = 0;
        for candidate in discovery.workers {
            let worker = match candidate.admit(self) {
                Ok(worker) => worker,
                Err(error) => {
                    warn!(%error,"Registry worker report refused");
                    continue;
                }
            };
            info!(pid=worker.source().pid,host=%worker.source().host,port=worker.source().port,"Admitted original registry connection");
            match self.serving(worker.key().target) {
                WorkerServing::SharedConcurrent { in_flight } => {
                    let mut workers = self.gpu_tcp_workers.lock().await;
                    if workers
                        .get(worker.key())
                        .is_some_and(|existing| existing.check_available().is_ok())
                    {
                        continue;
                    }
                    let (key, shared) =
                        super::shared_gpu::SharedGpuTcpWorker::from_registry(worker, in_flight);
                    workers.insert(key, Arc::new(shared));
                    integrated += 1;
                }
                WorkerServing::OneRequestPerProcess => {
                    let group = self.get_or_create_group(worker.key());
                    let reservation = match super::capacity::ReservedWorkerSlot::try_adopt(&group) {
                        Ok(reservation) => reservation,
                        Err(_) => {
                            warn!(profile=%worker.info().profile.label(),"Registry adoption deferred at global capacity");
                            continue;
                        }
                    };
                    let (_key, handle) = worker.into_sequential();
                    reservation.publish_tcp(handle);
                    integrated += 1;
                }
            }
        }
        info!(integrated, "Integrated registry worker connections");
        integrated
    }
}
