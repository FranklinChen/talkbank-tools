//! Immutable preparation owns the key and task/model facts it routes by.

use super::*;

pub(super) struct WorkerPlan {
    key: WorkerKey,
    prepared: DispatchPreparation,
}
impl WorkerPlan {
    pub(super) fn direct(
        language: WorkerLanguage,
        prepared: DispatchPreparation,
        mode: WorkerBootstrapMode,
    ) -> Result<Self, WorkerError> {
        if GpuTask::of(prepared.load().task()).is_some() {
            return Err(WorkerError::UnroutedGpuRequest {
                task: prepared.load().task(),
            });
        }
        let key = prepared.load().worker_key(language, mode);
        Ok(Self { key, prepared })
    }
    pub(super) fn routed(
        route: &GpuRoute,
        prepared: DispatchPreparation,
    ) -> Result<Self, WorkerError> {
        route.admit(prepared.request())?;
        Ok(Self {
            key: route.key().clone(),
            prepared,
        })
    }
    pub(super) const fn key(&self) -> &WorkerKey {
        &self.key
    }
    pub(super) fn into_parts(self) -> (WorkerKey, DispatchPreparation) {
        (self.key, self.prepared)
    }
}

pub(super) struct CommandWorkerPlan {
    key: WorkerKey,
    load: TaskLoad,
}
impl CommandWorkerPlan {
    pub(super) fn new(
        requirement: crate::command_model::InferenceRequirement,
        language: WorkerLanguage,
        options: &crate::options::CommandOptions,
        gpu: &GpuModelPlan,
        mode: WorkerBootstrapMode,
    ) -> Self {
        let task = requirement.task();
        let (key, load) = match GpuTask::of(task) {
            Some(gpu_task) => (
                GpuRoute::worker_key(gpu, gpu_task, language, mode),
                TaskLoad::new(task, gpu.load_overrides(gpu_task)),
            ),
            None => {
                let load = TaskLoad::new(task, EngineSelection::overrides_for_command(options));
                (load.worker_key(language, mode), load)
            }
        };
        Self { key, load }
    }
    pub(super) const fn key(&self) -> &WorkerKey {
        &self.key
    }
    pub(super) const fn load(&self) -> &TaskLoad {
        &self.load
    }
}
