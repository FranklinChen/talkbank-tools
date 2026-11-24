//! Typed worker bootstrap targets for the Rust control plane.
//!
//! Large hosts can still amortize model load across profile-shaped workers such
//! as `profile:gpu`. Constrained hosts instead launch task-shaped workers such
//! as `infer:asr` so one small machine does not speculatively hold unrelated
//! models in memory.

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
    #[must_use]
    pub const fn profile(profile: WorkerProfile) -> Self {
        Self::Profile(profile)
    }

    /// Build a pure inference worker target.
    #[must_use]
    pub const fn infer_task(task: InferTask) -> Self {
        Self::InferTask(task)
    }

    /// Build one worker target from an infer task and host bootstrap mode.
    #[must_use]
    pub const fn from_infer_task(task: InferTask, mode: WorkerBootstrapMode) -> Self {
        match mode {
            WorkerBootstrapMode::Profile | WorkerBootstrapMode::LazyProfile => {
                Self::Profile(WorkerProfile::for_task(task))
            }
            WorkerBootstrapMode::Task => Self::InferTask(task),
        }
    }

    /// Return the string label used in logs, health responses, and worker keys.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Profile(profile) => profile.label().to_string(),
            Self::InferTask(task) => format!("infer:{}", task_name(*task)),
        }
    }

    /// Return the shared profile that owns this target's task family.
    #[must_use]
    pub const fn profile_kind(&self) -> WorkerProfile {
        match self {
            Self::Profile(profile) => *profile,
            Self::InferTask(task) => WorkerProfile::for_task(*task),
        }
    }

    /// Return the specific infer task if this is a task bootstrap target.
    #[must_use]
    pub const fn task(&self) -> Option<InferTask> {
        match self {
            Self::Profile(_) => None,
            Self::InferTask(task) => Some(*task),
        }
    }

    /// Derive a target only from admitted inference requirements. A native
    /// command cannot be passed to this constructor.
    pub(crate) const fn for_inference_with_mode(
        requirement: InferenceRequirement,
        mode: WorkerBootstrapMode,
    ) -> Self {
        Self::from_infer_task(requirement.task(), mode)
    }
}

/// Convert one infer task into the stable `snake_case` label used across Rust and
/// Python bootstrap code.
pub const fn task_name(task: InferTask) -> &'static str {
    task.wire_name()
}

#[cfg(test)]
#[path = "target_tests.rs"]
mod tests;
