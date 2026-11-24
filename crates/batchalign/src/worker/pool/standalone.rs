//! One worker launched exactly as a pool would launch it, owned by a single
//! caller outside any pool.
//!
//! # Why this exists
//!
//! `batchalign3 doctor` checks that a machine can run morphotag. It used to
//! do that with its own stdio client: its own command line, its own ready
//! loop, its own request (`batch_infer`, which production stopped sending),
//! and its own reading of the reply. Each of those could pass while the real
//! path failed, and two of them did (a missing `--serving` argument, and a
//! reply walked at a path the worker no longer used, which passed with "0
//! words").
//!
//! A [`StandaloneWorker`] is spawned from the same [`WorkerKey`] the pool
//! derives for the request ([`TaskLoad::worker_key`]), through the same
//! config builder ([`worker_config_for_key`]), and is asked through the same
//! `execute_v2` preparation, load lease and stall windows. The only
//! difference from a pooled worker is ownership: nothing else can check it
//! out, and there are no idle workers to retire when the host is full, so a
//! busy host makes it wait for memory as a job would.

use std::sync::Arc;

use crate::api::WorkerLanguage;
use crate::host_memory::NoIdleWorkers;
use crate::types::worker_v2::{ExecuteRequestV2, ExecuteResponseV2};
use crate::worker::WorkerPid;
use crate::worker::error::WorkerError;
use crate::worker::handle::{WorkerConfig, WorkerHandle};

use super::lifecycle::worker_config_for_key;
use super::{LoadAdmission, LoadPreparation, ModelHosting, PoolConfig, TaskLoad};

/// A ready worker spawned for one request family, with the load preparation
/// a pool on the same configuration would give it.
pub struct StandaloneWorker {
    handle: WorkerHandle,
    preparation: LoadPreparation,
}

impl PoolConfig {
    /// Spawn the worker a pool on this configuration would spawn to serve
    /// `request` in `language`, and wait until it is ready.
    ///
    /// # Errors
    ///
    /// The spawn plan cannot be resolved, the memory guard refuses the
    /// worker's baseline, or the worker fails to start.
    pub(crate) async fn spawn_standalone(
        &self,
        request: &ExecuteRequestV2,
        language: WorkerLanguage,
    ) -> Result<StandaloneWorker, WorkerError> {
        let key = TaskLoad::for_request(request).worker_key(language, self.runtime.bootstrap_mode);
        let config = worker_config_for_key(self, &key)?;
        let preparation = LoadPreparation::standalone(
            self.runtime.host.memory_coordinator(),
            self.model_hosting(),
            self.ensure_task_stall,
        );
        StandaloneWorker::spawn(config, preparation).await
    }
}

impl LoadPreparation {
    /// The preparation for a worker no pool owns: the host admits each
    /// load's lease as it would for a pooled worker, but a load that has to
    /// queue has no idle workers to retire, so it waits.
    pub(crate) fn standalone(
        coordinator: crate::host_memory::HostMemoryCoordinator,
        hosting: ModelHosting,
        stall: crate::worker::LoadingStall,
    ) -> Self {
        Self {
            admission: LoadAdmission::new(coordinator, hosting, Arc::new(NoIdleWorkers)),
            stall,
        }
    }
}

impl StandaloneWorker {
    /// Spawn the worker `config` describes and wait until it is ready; its
    /// loads are prepared by `preparation`. `PoolConfig::spawn_standalone`
    /// derives both as a pool would; `replay` derives the config from the
    /// launch a failure dump recorded.
    ///
    /// # Errors
    ///
    /// The memory guard refuses the worker's baseline, or it fails to start.
    pub(crate) async fn spawn(
        config: WorkerConfig,
        preparation: LoadPreparation,
    ) -> Result<Self, WorkerError> {
        let handle = WorkerHandle::spawn(config, Arc::new(NoIdleWorkers)).await?;
        Ok(Self {
            handle,
            preparation,
        })
    }

    /// The pid the worker reported in its ready handshake.
    pub(crate) const fn pid(&self) -> WorkerPid {
        self.handle.pid()
    }

    /// Send `request` the way a pool dispatch does: load its task first
    /// (taking the load's memory lease), then execute it.
    ///
    /// # Errors
    ///
    /// The request is not dispatchable, the load or the execute fails, or
    /// the worker stalls past its window.
    pub(crate) async fn execute(
        &mut self,
        request: &ExecuteRequestV2,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        let Self {
            handle,
            preparation,
        } = self;
        // The macro brings `DispatchPreparation` into scope for its argument.
        crate::worker::pool::dispatch_prepared!(
            DispatchPreparation::begin(request)?,
            ready => handle.execute_v2(ready, preparation)
        )
    }

    /// Ask the worker to exit and wait for it.
    ///
    /// # Errors
    ///
    /// The shutdown exchange failed; the handle's drop still kills it.
    pub(crate) async fn shutdown(self) -> Result<(), WorkerError> {
        self.handle.shutdown().await
    }
}
