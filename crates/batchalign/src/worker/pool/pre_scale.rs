//! Worker pre-scaling.
//!
//! `pre_scale()` eagerly spawns workers for a specific command/lang before file
//! dispatch begins, so a batch does not pay one cold start per concurrent file.
//!
//! There used to be a sibling `warmup()` here that pre-spawned workers for a
//! configured command list at server startup. It was disabled on every real
//! server from 2026-03-26 onward (`HostExecutionPolicy::allows_command_warmup`
//! required test-echo mode), following the 2026-03-11 finding that warmed
//! models persist in memory for the server's whole lifetime, and was retired
//! in full on 2026-07-30. Pre-scale plus pressure-driven idle eviction is what
//! serves that need now: both act on demonstrated demand rather than on a
//! guess made at startup.

use std::sync::atomic::Ordering;

use crate::api::{ReleasedCommand, WorkerLanguage};
use crate::options::CommandOptions;
use crate::worker::WorkerTarget;
use tracing::{info, warn};

use super::{WorkerKey, WorkerPool, lock_recovered};

impl WorkerPool {
    /// Pre-scale a command with no engine selection.
    ///
    /// This exists for text-task test fixtures. The job runner uses
    /// [`Self::pre_scale_for_command`], and GPU integration tests use
    /// [`Self::pre_scale_gpu_for_test`].
    pub async fn pre_scale(
        &self,
        command: ReleasedCommand,
        lang: impl Into<WorkerLanguage>,
        target: usize,
    ) {
        let Ok(requirement) = crate::command_model::command_spec(command)
            .capabilities
            .require_inference()
        else {
            // Native work shares job scheduling, not Python worker capacity.
            return;
        };
        let key = WorkerKey::without_engine_selection(
            WorkerTarget::for_inference_with_mode(requirement, self.config.runtime.bootstrap_mode),
            lang.into(),
        );
        self.pre_scale_key(command, key, target).await;
    }

    /// Pre-scale the GPU worker `plan` selects for `command`, in
    /// integration tests. Production pre-scaling goes through
    /// [`Self::pre_scale_for_command`], which keys from the job's own plan.
    #[doc(hidden)]
    pub async fn pre_scale_gpu_for_test(
        &self,
        command: ReleasedCommand,
        lang: impl Into<WorkerLanguage>,
        target: usize,
        plan: &super::GpuModelPlan,
    ) {
        let Some(task) = crate::command_model::command_spec(command)
            .capabilities
            .require_inference()
            .ok()
            .and_then(|requirement| super::GpuTask::of(requirement.task()))
        else {
            return;
        };
        let key = super::GpuRoute::worker_key(
            plan,
            task,
            lang.into(),
            self.config.runtime.bootstrap_mode,
        );
        self.pre_scale_key(command, key, target).await;
    }

    /// Pre-scale the worker a job's command selects
    /// ([`WorkerKey::for_command`]): its plan's worker for a GPU command, its
    /// options' worker otherwise. This is the job-runner path, and the same
    /// key the capability probe uses, so pre-scaled workers are the ones the
    /// job's requests reach.
    pub async fn pre_scale_for_command(
        &self,
        command: ReleasedCommand,
        lang: impl Into<WorkerLanguage>,
        target: usize,
        options: &CommandOptions,
        gpu: &super::GpuModelPlan,
    ) {
        let Ok(requirement) = crate::command_model::command_spec(command)
            .capabilities
            .require_inference()
        else {
            return;
        };
        let key = WorkerKey::for_command(
            requirement,
            lang.into(),
            options,
            gpu,
            self.config.runtime.bootstrap_mode,
        );
        self.pre_scale_key(command, key, target).await;
    }

    /// Spawn workers eagerly so they're ready before file dispatch begins.
    ///
    /// `key` has already been derived from one typed source. Keeping this
    /// primitive private prevents a raw JSON override from becoming another
    /// identity construction path.
    async fn pre_scale_key(&self, command: ReleasedCommand, key: WorkerKey, target: usize) {
        let target = target.min(self.max_workers_per_key_for(key.target.profile_kind()));

        // TCP worker shortcut: if a TCP worker already exists for this
        // profile/lang, skip spawning: the worker is already running.
        if self.serving(key.target).is_shared() {
            if matches!(key.target, WorkerTarget::Profile(_))
                && self.live_gpu_tcp_worker(&key).await.is_some()
            {
                info!(
                    command = %command,
                    lang = %key.language,
                    "GPU TCP worker already discovered, skipping pre-scale"
                );
                return;
            }
        } else {
            let has_tcp = {
                let groups = lock_recovered(&self.groups);
                groups
                    .get(&key)
                    .is_some_and(|g| !lock_recovered(&g.tcp_workers).is_empty())
            };
            if has_tcp {
                info!(
                    command = %command,
                    lang = %key.language,
                    profile = %key.target.label(),
                    "TCP worker already discovered, skipping pre-scale"
                );
                return;
            }
        }

        // Shared-serving keys use the shared worker map. Pre-creating the
        // worker here ensures it's ready before file dispatch begins, avoiding
        // the TOCTOU race in `get_or_create_gpu_worker` where multiple tasks
        // would each try to spawn their own worker process.
        if self.serving(key.target).is_shared() {
            match self.get_or_create_gpu_worker(&key).await {
                Ok(_) => {
                    info!(
                        command = %command,
                        target = %key.target.label(),
                        lang = %key.language,
                        engine_selection = %key.engine_selection,
                        "GPU worker pre-scaled (ready for concurrent dispatch)"
                    );
                }
                Err(e) => {
                    warn!(
                        command = %command,
                        lang = %key.language,
                        target = %key.target.label(),
                        error = %e,
                        "GPU worker pre-scale failed"
                    );
                }
            }
            return;
        }

        let group = self.get_or_create_group(&key);

        loop {
            let current = group.total.load(Ordering::Relaxed);
            if current >= target {
                break;
            }

            match self.try_spawn_into_group(&group, &key).await {
                Ok(true) => {}      // Keep going
                Ok(false) => break, // At capacity
                Err(e) => {
                    warn!(
                        target = %key.target.label(),
                        lang = %key.language,
                        current = group.total.load(Ordering::Relaxed),
                        target = target,
                        error = %e,
                        "Pre-scale spawn failed, stopping early"
                    );
                    break;
                }
            }
        }
    }
}
