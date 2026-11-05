//! Typed worker bootstrap targets for the Rust control plane.
//!
//! Large hosts can still amortize model load across profile-shaped workers such
//! as `profile:gpu`. Constrained hosts instead launch task-shaped workers such
//! as `infer:asr` so one small machine does not speculatively hold unrelated
//! models in memory.

#[cfg(test)]
use crate::api::ReleasedCommand;
use crate::command_model::InferenceRequirement;

use super::InferTask;

// Re-export WorkerProfile from batchalign-types so all existing
// `crate::worker::WorkerProfile` import paths continue to resolve.
pub use batchalign_types::worker_profile::WorkerProfile;

/// How one host runs its Python workers.
///
/// In every mode a worker starts with no models and loads each task's models
/// in that task's `ensure_task`, before the task's first request; the modes
/// differ in how workers are grouped and pre-scaled, not in what loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerBootstrapMode {
    /// One worker per profile, pre-scaled for a batch. Large machines
    /// (>48 GB).
    Profile,
    /// One worker per profile, not pre-scaled: a worker is spawned when files
    /// need it. Medium machines (24-48 GB).
    LazyProfile,
    /// One worker per infer task. Small machines (<24 GB).
    Task,
}

/// Bootstrap target for one Python worker process.
///
/// Python workers are model hosts for one infer task such as ASR or forced
/// alignment. Top-level commands are mapped onto these infer-task workers by the
/// Rust control plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkerTarget {
    /// Launch a worker around one shared profile.
    Profile(WorkerProfile),
    /// Launch a worker around one pure inference task.
    InferTask(InferTask),
}

impl WorkerTarget {
    /// Build a profile bootstrap target.
    pub fn profile(profile: WorkerProfile) -> Self {
        Self::Profile(profile)
    }

    /// Build a pure inference worker target.
    pub fn infer_task(task: InferTask) -> Self {
        Self::InferTask(task)
    }

    /// Build one worker target from an infer task and host bootstrap mode.
    pub fn from_infer_task(task: InferTask, mode: WorkerBootstrapMode) -> Self {
        match mode {
            WorkerBootstrapMode::Profile | WorkerBootstrapMode::LazyProfile => {
                Self::Profile(WorkerProfile::for_task(task))
            }
            WorkerBootstrapMode::Task => Self::InferTask(task),
        }
    }

    /// Return the string label used in logs, health responses, and worker keys.
    pub fn label(&self) -> String {
        match self {
            Self::Profile(profile) => profile.label().to_string(),
            Self::InferTask(task) => format!("infer:{}", task_name(*task)),
        }
    }

    /// Return the shared profile that owns this target's task family.
    pub fn profile_kind(&self) -> WorkerProfile {
        match self {
            Self::Profile(profile) => *profile,
            Self::InferTask(task) => WorkerProfile::for_task(*task),
        }
    }

    /// Return the specific infer task if this is a task bootstrap target.
    pub fn task(&self) -> Option<InferTask> {
        match self {
            Self::Profile(_) => None,
            Self::InferTask(task) => Some(*task),
        }
    }

    /// Return the infer-task worker target used for one released command.
    #[cfg(test)]
    pub(crate) fn for_command(command: ReleasedCommand) -> Self {
        Self::InferTask(
            crate::command_model::command_spec(command)
                .capabilities
                .require_inference()
                .expect("test command requires inference")
                .task(),
        )
    }

    /// Derive a target only from admitted inference requirements. A native
    /// command cannot be passed to this constructor.
    pub(crate) fn for_inference_with_mode(
        requirement: InferenceRequirement,
        mode: WorkerBootstrapMode,
    ) -> Self {
        Self::from_infer_task(requirement.task(), mode)
    }
}

/// Convert one infer task into the stable snake_case label used across Rust and
/// Python bootstrap code.
pub(crate) fn task_name(task: InferTask) -> &'static str {
    task.wire_name()
}

#[cfg(test)]
mod tests {
    use super::{InferTask, WorkerBootstrapMode, WorkerProfile, WorkerTarget};
    use crate::api::ReleasedCommand;
    use crate::types::runtime;

    #[test]
    fn command_target_maps_transcribe_to_asr() {
        let target = WorkerTarget::for_command(ReleasedCommand::Transcribe);
        assert_eq!(target, WorkerTarget::InferTask(InferTask::Asr));
    }

    #[test]
    fn command_target_maps_compare_to_morphosyntax() {
        assert_eq!(
            WorkerTarget::for_command(ReleasedCommand::Compare),
            WorkerTarget::InferTask(InferTask::Morphosyntax)
        );
    }

    #[test]
    fn infer_target_label_is_prefixed() {
        assert_eq!(WorkerTarget::infer_task(InferTask::Fa).label(), "infer:fa");
    }

    #[test]
    fn profile_target_label_is_prefixed() {
        assert_eq!(
            WorkerTarget::profile(WorkerProfile::Gpu).label(),
            "profile:gpu"
        );
    }

    #[test]
    fn profile_mode_maps_task_to_profile_target() {
        assert_eq!(
            WorkerTarget::from_infer_task(InferTask::Asr, WorkerBootstrapMode::Profile),
            WorkerTarget::Profile(WorkerProfile::Gpu)
        );
    }

    #[test]
    fn command_target_respects_bootstrap_mode() {
        let requirement = crate::command_model::command_spec(ReleasedCommand::Morphotag)
            .capabilities
            .require_inference()
            .expect("morphotag requires inference");
        assert_eq!(
            WorkerTarget::for_inference_with_mode(requirement, WorkerBootstrapMode::Profile),
            WorkerTarget::Profile(WorkerProfile::Stanza)
        );
        assert_eq!(
            WorkerTarget::for_inference_with_mode(requirement, WorkerBootstrapMode::Task),
            WorkerTarget::InferTask(InferTask::Morphosyntax)
        );
    }

    // -- WorkerProfile tests --

    #[test]
    fn gpu_tasks_map_to_gpu_profile() {
        assert_eq!(WorkerProfile::for_task(InferTask::Asr), WorkerProfile::Gpu);
        assert_eq!(WorkerProfile::for_task(InferTask::Fa), WorkerProfile::Gpu);
        assert_eq!(
            WorkerProfile::for_task(InferTask::Speaker),
            WorkerProfile::Gpu
        );
    }

    #[test]
    fn stanza_tasks_map_to_stanza_profile() {
        assert_eq!(
            WorkerProfile::for_task(InferTask::Morphosyntax),
            WorkerProfile::Stanza
        );
        assert_eq!(
            WorkerProfile::for_task(InferTask::Utseg),
            WorkerProfile::Stanza
        );
        assert_eq!(
            WorkerProfile::for_task(InferTask::Coref),
            WorkerProfile::Stanza
        );
    }

    #[test]
    fn io_tasks_map_to_io_profile() {
        assert_eq!(
            WorkerProfile::for_task(InferTask::Translate),
            WorkerProfile::Io
        );
        assert_eq!(
            WorkerProfile::for_task(InferTask::Opensmile),
            WorkerProfile::Io
        );
        assert_eq!(WorkerProfile::for_task(InferTask::Avqi), WorkerProfile::Io);
    }

    #[test]
    fn profile_for_command_maps_align_to_gpu() {
        assert_eq!(
            WorkerProfile::for_command(ReleasedCommand::Align),
            Some(WorkerProfile::Gpu)
        );
    }

    #[test]
    fn profile_for_command_maps_morphotag_to_stanza() {
        assert_eq!(
            WorkerProfile::for_command(ReleasedCommand::Morphotag),
            Some(WorkerProfile::Stanza)
        );
    }

    #[test]
    fn gpu_default_max_workers_is_one() {
        assert_eq!(WorkerProfile::Gpu.default_max_workers(4), 1);
        assert_eq!(WorkerProfile::Stanza.default_max_workers(4), 4);
        assert_eq!(WorkerProfile::Io.default_max_workers(4), 1);
    }

    #[test]
    fn startup_reservations_for_large_tier_match_toml_constants() {
        let tier = runtime::MemoryTier::from_total_mb(64_000);
        let gpu = WorkerProfile::Gpu.startup_reservation_mb_for_tier(&tier);
        let stanza = WorkerProfile::Stanza.startup_reservation_mb_for_tier(&tier);
        let io = WorkerProfile::Io.startup_reservation_mb_for_tier(&tier);

        assert_eq!(gpu.0, 16_000, "GPU Large tier should match TOML constant");
        assert_eq!(
            stanza.0, 12_000,
            "Stanza Large tier should match TOML constant"
        );
        assert_eq!(io.0, 4_000, "IO Large tier should match TOML constant");
    }

    #[test]
    fn startup_reservations_for_small_tier_are_reduced() {
        let tier = runtime::MemoryTier::from_total_mb(16_000);
        let gpu = WorkerProfile::Gpu.startup_reservation_mb_for_tier(&tier);
        let stanza = WorkerProfile::Stanza.startup_reservation_mb_for_tier(&tier);

        assert!(gpu.0 < 16_000, "GPU Small tier must be less than Large");
        assert!(
            stanza.0 < 12_000,
            "Stanza Small tier must be less than Large"
        );
        assert!(gpu.0 > stanza.0, "GPU must still exceed Stanza");
    }
}
