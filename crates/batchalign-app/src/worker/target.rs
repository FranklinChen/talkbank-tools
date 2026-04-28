//! Typed worker bootstrap targets for the Rust control plane.
//!
//! Large hosts can still amortize model load across profile-shaped workers such
//! as `profile:gpu`. Constrained hosts instead launch task-shaped workers such
//! as `infer:asr` so one small machine does not speculatively hold unrelated
//! models in memory.

use crate::api::{MemoryMb, ReleasedCommand};
use crate::commands::command_workflow_descriptor;
use crate::runtime;

use super::InferTask;

/// How one local Python worker should bootstrap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerBootstrapMode {
    /// Load the full shared profile for a task family. Used on large machines
    /// (>48 GB) where eager loading amortizes cold start across tasks.
    Profile,
    /// Start as a profile worker but load NO models at startup. Models are
    /// loaded on demand via `ensure_task` IPC before the first dispatch for
    /// each task. Used on medium-tier machines (24-48 GB) where eager loading
    /// would speculatively consume 10-15 GB. Peak memory grows incrementally
    /// as tasks are activated.
    LazyProfile,
    /// Load only the requested infer task. Used on small machines (<24 GB)
    /// where even one full profile would exhaust memory.
    Task,
}

// ---------------------------------------------------------------------------
// WorkerProfile
// ---------------------------------------------------------------------------

/// Worker profile grouping related [`InferTask`]s into fewer processes.
///
/// Instead of spawning one worker per `InferTask`, profiles group related tasks
/// so that loaded models are shared within a single process:
///
/// - **Gpu**: ASR, FA, Speaker — GPU-bound models, concurrent via Python
///   `ThreadPoolExecutor` (PyTorch releases the GIL during CUDA kernels).
///   Max 1 process per (lang, engine_overrides) key.
/// - **Stanza**: Morphosyntax, Utseg, Coref — Stanza NLP processors, sequential
///   per process. Multiple processes for CPU parallelism (auto-tuned).
/// - **Io**: Translate, OpenSMILE, AVQI — lightweight API/library calls.
///   Max 1 process per key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerProfile {
    /// GPU-bound models (ASR, FA, Speaker). Concurrent via threads inside one process.
    Gpu,
    /// Stanza NLP processors (Morphosyntax, Utseg, Coref). Multi-process for CPU parallelism.
    Stanza,
    /// Lightweight API/library calls (Translate, OpenSMILE, AVQI).
    Io,
}

impl WorkerProfile {
    /// Map one [`InferTask`] to its profile.
    pub fn for_task(task: InferTask) -> Self {
        match task {
            InferTask::Asr | InferTask::Fa | InferTask::Speaker => Self::Gpu,
            InferTask::Morphosyntax | InferTask::Utseg | InferTask::Coref => Self::Stanza,
            InferTask::Translate | InferTask::Opensmile | InferTask::Avqi => Self::Io,
        }
    }

    /// The string label used in logs and worker keys.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Gpu => "profile:gpu",
            Self::Stanza => "profile:stanza",
            Self::Io => "profile:io",
        }
    }

    /// The profile name used in the ``--profile`` CLI arg sent to Python.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::Stanza => "stanza",
            Self::Io => "io",
        }
    }

    /// Parse a profile name from a CLI argument or registry entry.
    ///
    /// Returns `None` for unrecognized names.
    pub fn try_from_name(name: &str) -> Option<Self> {
        match name {
            "gpu" => Some(Self::Gpu),
            "stanza" => Some(Self::Stanza),
            "io" => Some(Self::Io),
            _ => None,
        }
    }

    /// Whether this profile uses concurrent request handling inside one process.
    ///
    /// GPU workers always use concurrent serving (PyTorch releases the GIL during
    /// CUDA kernels). Stanza workers use concurrent serving only when running on
    /// free-threaded Python 3.14t, where OS threads share one model instance
    /// instead of each process holding a full private copy.
    pub fn is_concurrent(&self) -> bool {
        self.is_concurrent_for_runtime(crate::types::runtime::is_free_threaded_runtime())
    }

    /// Like `is_concurrent`, but takes an explicit free-threaded flag.
    ///
    /// Use this in tests or contexts where the runtime flag is supplied externally.
    pub fn is_concurrent_for_runtime(&self, free_threaded: bool) -> bool {
        match self {
            Self::Gpu => true,
            // Stanza workers share one model via ThreadPoolExecutor on 3.14t,
            // giving the same throughput as separate processes with 77% less
            // memory (see python-versioning.md benchmarks, 2026-02-19).
            Self::Stanza => free_threaded,
            Self::Io => false,
        }
    }

    /// Conservative host-wide startup reservation (MB) for spawning one worker
    /// of this profile.
    ///
    /// This is intentionally explicit and profile-shaped rather than derived
    /// from the smaller per-command execution budgets. The startup reservation
    /// protects the model-loading window where multiple local batchalign3
    /// processes could otherwise overcommit host RAM before the OS snapshot
    /// catches up.
    pub fn startup_reservation_mb(&self) -> MemoryMb {
        let tier = runtime::MemoryTier::detect();
        self.startup_reservation_mb_for_tier(&tier)
    }

    /// Startup reservation for a specific memory tier.
    ///
    /// Use this variant in tests and when the tier is already known.
    pub fn startup_reservation_mb_for_tier(&self, tier: &runtime::MemoryTier) -> MemoryMb {
        match self {
            Self::Gpu => tier.gpu_startup_mb,
            Self::Stanza => tier.stanza_startup_mb,
            Self::Io => tier.io_startup_mb,
        }
    }

    /// Default maximum worker processes per ``(profile, lang, engine_overrides)`` key.
    ///
    /// GPU: 1 process (concurrent via threads).
    /// Stanza: `auto_tune` (multiple processes for CPU parallelism).
    /// IO: 1 process (lightweight).
    pub fn default_max_workers(&self, auto_tune: usize) -> usize {
        match self {
            Self::Gpu => 1,
            Self::Stanza => auto_tune,
            Self::Io => 1,
        }
    }

    /// Map a command name to the profile needed for that command's infer-task worker.
    pub fn for_command(command: ReleasedCommand) -> Option<Self> {
        command_workflow_descriptor(command).map(|descriptor| Self::for_task(descriptor.infer_task))
    }
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

    /// Whether the target uses concurrent dispatch inside one process.
    pub fn is_concurrent(&self) -> bool {
        self.profile_kind().is_concurrent()
    }

    /// Return the infer-task worker target used for one released command.
    #[cfg(test)]
    pub(crate) fn for_command(command: ReleasedCommand) -> Option<Self> {
        let task = command_workflow_descriptor(command)?.infer_task;
        Some(Self::InferTask(task))
    }

    /// Return the actual bootstrap target used for one released command and host mode.
    pub(crate) fn for_command_with_mode(
        command: ReleasedCommand,
        mode: WorkerBootstrapMode,
    ) -> Option<Self> {
        let task = command_workflow_descriptor(command)?.infer_task;
        Some(Self::from_infer_task(task, mode))
    }
}

/// Convert one infer task into the stable snake_case label used across Rust and
/// Python bootstrap code.
pub(crate) fn task_name(task: InferTask) -> &'static str {
    match task {
        InferTask::Morphosyntax => "morphosyntax",
        InferTask::Utseg => "utseg",
        InferTask::Translate => "translate",
        InferTask::Coref => "coref",
        InferTask::Fa => "fa",
        InferTask::Asr => "asr",
        InferTask::Opensmile => "opensmile",
        InferTask::Avqi => "avqi",
        InferTask::Speaker => "speaker",
    }
}

#[cfg(test)]
mod tests {
    use super::{InferTask, WorkerBootstrapMode, WorkerProfile, WorkerTarget};
    use crate::api::ReleasedCommand;
    use crate::types::runtime;

    #[test]
    fn command_target_maps_transcribe_to_asr() {
        let target = WorkerTarget::for_command(ReleasedCommand::Transcribe);
        assert_eq!(target, Some(WorkerTarget::InferTask(InferTask::Asr)));
    }

    #[test]
    fn command_target_maps_compare_to_morphosyntax() {
        assert_eq!(
            WorkerTarget::for_command(ReleasedCommand::Compare),
            Some(WorkerTarget::InferTask(InferTask::Morphosyntax))
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
        assert_eq!(
            WorkerTarget::for_command_with_mode(
                ReleasedCommand::Morphotag,
                WorkerBootstrapMode::Profile
            ),
            Some(WorkerTarget::Profile(WorkerProfile::Stanza))
        );
        assert_eq!(
            WorkerTarget::for_command_with_mode(
                ReleasedCommand::Morphotag,
                WorkerBootstrapMode::Task
            ),
            Some(WorkerTarget::InferTask(InferTask::Morphosyntax))
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
    fn gpu_profile_is_concurrent() {
        // GPU is always concurrent.
        assert!(WorkerProfile::Gpu.is_concurrent_for_runtime(false));
        assert!(WorkerProfile::Gpu.is_concurrent_for_runtime(true));
        // Stanza is concurrent only on free-threaded Python.
        assert!(!WorkerProfile::Stanza.is_concurrent_for_runtime(false));
        assert!(WorkerProfile::Stanza.is_concurrent_for_runtime(true));
        // IO is never concurrent.
        assert!(!WorkerProfile::Io.is_concurrent_for_runtime(false));
        assert!(!WorkerProfile::Io.is_concurrent_for_runtime(true));
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
