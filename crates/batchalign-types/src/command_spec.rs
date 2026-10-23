//! Static per-command metadata. Sole canonical source of resource
//! shape and classification for every released batchalign command.
//!
//! Per the Phase β spec,
//! `CommandSpec` is orthogonal to `recipe_runner::CatalogEntry`
//! (planning/execution shape). They cross-reference by
//! `ReleasedCommand` identity.

use crate::api::{MemoryMb, ReleasedCommand};
use crate::worker::InferTask;
use crate::worker_profile::WorkerProfile;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Multiplier applied to the static memory budget to account for
/// transient allocation spikes during model loading. Validated > 1.0
/// at construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadingOverhead(pub f64);

impl LoadingOverhead {
    /// Construct without validation. Use only with literal values in
    /// `const` context. Runtime construction should validate via a
    /// (yet-to-be-added) `try_new` constructor.
    pub const fn new_unchecked(value: f64) -> Self {
        Self(value)
    }
}

/// Whether a command needs a process-isolated worker, which decides its
/// memory budget.
///
/// The process-isolated commands are the TOML's `process_commands` list. The
/// Stanza commands are among them: a Stanza worker serves one request per
/// process in every runtime (`WorkerServing::decide`), so each process loads
/// its own models whether or not Python is free-threaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GilProcessNeed {
    /// Needs a process worker regardless of runtime (in the TOML's
    /// `process_commands`): its budget is the process budget in every
    /// runtime.
    Always,
    /// Shares one process's models on threads in a free-threaded runtime
    /// (not in `process_commands`), where it has its own, smaller budget.
    Never {
        /// Base memory budget when running on a thread worker.
        threaded_base_mb: MemoryMb,
    },
}

impl GilProcessNeed {
    /// Whether the command needs a process worker; the same in every runtime.
    pub fn needs_process(self) -> bool {
        match self {
            Self::Always => true,
            Self::Never { .. } => false,
        }
    }
}

/// Static per-command metadata.
///
/// Sole canonical source for resource + classification. The
/// `recipe_runner::CatalogEntry` type holds the orthogonal
/// planning/execution shape; the two cross-reference by
/// `ReleasedCommand` identity.
#[derive(Debug, Clone, Copy)]
pub struct CommandSpec {
    /// Stable released command identity.
    pub name: ReleasedCommand,
    /// Worker infer tasks the command's recipe dispatches to.
    /// Slice rather than scalar for composite commands (benchmark);
    /// today only benchmark would carry a multi-task list, but
    /// every InferTask::Eval-class addition would land here.
    pub tasks: &'static [InferTask],
    /// Literal cmd2task TOML label. Verbatim string emitted to
    /// `runtime_constants.toml::cmd2task` for Python's import-time
    /// dict. Most commands' label equals their first `tasks` element's
    /// snake_case name; `utseg` (label `"utterance"`) and `benchmark`
    /// (label `"asr,eval"`) diverge: see the table in the Phase β
    /// plan, Step 4.1.
    pub task_label: &'static str,
    /// Worker profile classification; native work owns no worker profile.
    pub profile: Option<WorkerProfile>,
    /// Whether this command needs a process worker per runtime mode.
    pub gil_process_need: GilProcessNeed,
    /// Base memory budget when running on a process worker.
    pub base_mb_process: MemoryMb,
    /// Multiplier applied to the base budget for the model-loading spike.
    pub loading_overhead: LoadingOverhead,
}

/// Which Python runtime the workers run on, which decides a command's
/// memory budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PythonRuntime {
    /// GIL-enabled CPython: thread workers cannot share a model.
    Gil,
    /// Free-threaded CPython (`PYTHON_GIL=0`): a command that does not need
    /// a process may share one process's models on threads.
    FreeThreaded,
}

impl CommandSpec {
    /// Pick the base budget appropriate for the runtime: the one figure
    /// every budget table is generated from.
    pub fn base_mb_for_runtime(&self, runtime: PythonRuntime) -> MemoryMb {
        match (self.gil_process_need, runtime) {
            (GilProcessNeed::Never { threaded_base_mb }, PythonRuntime::FreeThreaded) => {
                threaded_base_mb
            }
            (GilProcessNeed::Never { .. }, PythonRuntime::Gil) | (GilProcessNeed::Always, _) => {
                self.base_mb_process
            }
        }
    }

    /// The command's execution reservation in `runtime`: its base budget
    /// with the model-loading spike. The one computation the execution plan
    /// and the host planner both read.
    pub fn execution_budget_mb(&self, runtime: PythonRuntime) -> MemoryMb {
        MemoryMb((self.base_mb_for_runtime(runtime).0 as f64 * self.loading_overhead.0) as u64)
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// The authoritative registry. Exhaustive over `ReleasedCommand`.
///
/// Order matches `ReleasedCommand::ALL` for stable codegen output.
/// All values verified from `batchalign/runtime_constants.toml` at plan-write time
/// (2026-05-10).
pub const COMMAND_SPECS: &[CommandSpec] = &[
    CommandSpec {
        name: ReleasedCommand::Align,
        tasks: &[InferTask::Fa],
        task_label: "fa",
        profile: Some(WorkerProfile::Gpu),
        // GPU workers handle process isolation; GIL does not apply.
        // Not in process_commands in the TOML.
        gil_process_need: GilProcessNeed::Never {
            threaded_base_mb: MemoryMb(4_000),
        },
        base_mb_process: MemoryMb(4_000),
        loading_overhead: LoadingOverhead::new_unchecked(1.5),
    },
    CommandSpec {
        name: ReleasedCommand::Transcribe,
        tasks: &[InferTask::Asr],
        task_label: "asr",
        profile: Some(WorkerProfile::Gpu),
        // GPU workers handle process isolation; GIL does not apply.
        // Not in process_commands in the TOML.
        gil_process_need: GilProcessNeed::Never {
            threaded_base_mb: MemoryMb(1_500),
        },
        base_mb_process: MemoryMb(1_500),
        loading_overhead: LoadingOverhead::new_unchecked(1.5),
    },
    CommandSpec {
        name: ReleasedCommand::TranscribeS,
        tasks: &[InferTask::Asr],
        task_label: "asr",
        profile: Some(WorkerProfile::Gpu),
        // GPU workers handle process isolation; GIL does not apply.
        // Not in process_commands in the TOML.
        gil_process_need: GilProcessNeed::Never {
            threaded_base_mb: MemoryMb(2_500),
        },
        base_mb_process: MemoryMb(2_500),
        loading_overhead: LoadingOverhead::new_unchecked(1.5),
    },
    CommandSpec {
        name: ReleasedCommand::Translate,
        tasks: &[InferTask::Translate],
        task_label: "translate",
        profile: Some(WorkerProfile::Io),
        // Io (API-call) worker; no subprocess isolation needed.
        // Not in process_commands in the TOML.
        gil_process_need: GilProcessNeed::Never {
            threaded_base_mb: MemoryMb(4_000),
        },
        base_mb_process: MemoryMb(4_000),
        loading_overhead: LoadingOverhead::new_unchecked(1.5),
    },
    CommandSpec {
        name: ReleasedCommand::Morphotag,
        tasks: &[InferTask::Morphosyntax],
        task_label: "morphosyntax",
        profile: Some(WorkerProfile::Stanza),
        gil_process_need: GilProcessNeed::Always,
        base_mb_process: MemoryMb(8_000),
        loading_overhead: LoadingOverhead::new_unchecked(1.5),
    },
    CommandSpec {
        name: ReleasedCommand::Coref,
        tasks: &[InferTask::Coref],
        task_label: "coref",
        profile: Some(WorkerProfile::Stanza),
        gil_process_need: GilProcessNeed::Always,
        base_mb_process: MemoryMb(4_000),
        loading_overhead: LoadingOverhead::new_unchecked(1.5),
    },
    CommandSpec {
        name: ReleasedCommand::Utseg,
        tasks: &[InferTask::Utseg],
        // Diverges from InferTask snake_case name: historical TOML label
        // is "utterance", not "utseg". Task 5 codegen emits this verbatim.
        task_label: "utterance",
        profile: Some(WorkerProfile::Stanza),
        gil_process_need: GilProcessNeed::Always,
        base_mb_process: MemoryMb(6_000),
        loading_overhead: LoadingOverhead::new_unchecked(1.5),
    },
    CommandSpec {
        name: ReleasedCommand::Benchmark,
        tasks: &[InferTask::Asr],
        // Composite label: benchmark runs ASR + eval. InferTask::Eval doesn't
        // exist (it's a string label in cmd2task only), so tasks carries only
        // the typed InferTask. The task_label is preserved verbatim for
        // Task 5 codegen's cmd2task projection.
        task_label: "asr,eval",
        profile: Some(WorkerProfile::Gpu),
        // GPU workers handle process isolation; GIL does not apply.
        // Not in process_commands in the TOML.
        gil_process_need: GilProcessNeed::Never {
            threaded_base_mb: MemoryMb(1_500),
        },
        base_mb_process: MemoryMb(1_500),
        loading_overhead: LoadingOverhead::new_unchecked(1.5),
    },
    CommandSpec {
        name: ReleasedCommand::Opensmile,
        tasks: &[InferTask::Opensmile],
        task_label: "opensmile",
        profile: Some(WorkerProfile::Io),
        // In the TOML's process_commands: needs process isolation in every
        // runtime.
        gil_process_need: GilProcessNeed::Always,
        base_mb_process: MemoryMb(500),
        loading_overhead: LoadingOverhead::new_unchecked(1.5),
    },
    CommandSpec {
        name: ReleasedCommand::Compare,
        tasks: &[InferTask::Morphosyntax],
        task_label: "morphosyntax",
        profile: Some(WorkerProfile::Stanza),
        // In the TOML's process_commands: needs process isolation in every
        // runtime.
        gil_process_need: GilProcessNeed::Always,
        base_mb_process: MemoryMb(8_000),
        loading_overhead: LoadingOverhead::new_unchecked(1.5),
    },
    CommandSpec {
        name: ReleasedCommand::Avqi,
        tasks: &[InferTask::Avqi],
        task_label: "avqi",
        profile: Some(WorkerProfile::Io),
        // In the TOML's process_commands: needs process isolation in every
        // runtime.
        gil_process_need: GilProcessNeed::Always,
        base_mb_process: MemoryMb(1_500),
        loading_overhead: LoadingOverhead::new_unchecked(1.5),
    },
    CommandSpec {
        name: ReleasedCommand::Diarize,
        tasks: &[InferTask::Speaker],
        task_label: "speaker",
        profile: Some(WorkerProfile::Gpu),
        // GPU workers handle process isolation; GIL does not apply
        // (same footing as the transcribe_s speaker stage).
        gil_process_need: GilProcessNeed::Never {
            threaded_base_mb: MemoryMb(2_500),
        },
        base_mb_process: MemoryMb(2_500),
        loading_overhead: LoadingOverhead::new_unchecked(1.5),
    },
    CommandSpec {
        name: ReleasedCommand::SpeakerIdentify,
        tasks: &[InferTask::Speaker],
        task_label: "speaker",
        profile: Some(WorkerProfile::Gpu),
        gil_process_need: GilProcessNeed::Never {
            threaded_base_mb: MemoryMb(1_000),
        },
        // Lower than diarization deliberately: the embedding graph is one
        // ~26 MB ONNX model run over a few seconds of audio at a time, where
        // diarization loads a segmentation network plus a clustering pipeline
        // and holds the whole recording. Reserving diarization's figure would
        // idle memory this command never uses and make the admission gate
        // refuse spawns it should have allowed.
        base_mb_process: MemoryMb(1_000),
        loading_overhead: LoadingOverhead::new_unchecked(1.5),
    },
    CommandSpec {
        name: ReleasedCommand::Convert,
        tasks: &[],
        task_label: "",
        profile: None,
        gil_process_need: GilProcessNeed::Never {
            threaded_base_mb: MemoryMb(256),
        },
        base_mb_process: MemoryMb(256),
        loading_overhead: LoadingOverhead::new_unchecked(1.5),
    },
];

// ---------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------

/// Lookup helper. Panics if the registry is missing an entry
/// the exhaustive-coverage test catches that at test time.
#[allow(clippy::expect_used)]
pub fn command_spec_for(name: ReleasedCommand) -> &'static CommandSpec {
    COMMAND_SPECS
        .iter()
        .find(|s| s.name == name)
        .expect("released command missing from COMMAND_SPECS, see exhaustive-coverage test")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::ReleasedCommand;
    use crate::worker::InferTask;
    use crate::worker_profile::WorkerProfile;

    /// Catalog invariant: every released command has an entry.
    /// Adding a new ReleasedCommand variant fails here.
    #[test]
    fn every_released_command_has_a_spec() {
        for command in &ReleasedCommand::ALL {
            let spec = command_spec_for(*command);
            assert_eq!(
                spec.name, *command,
                "registry entry for {command:?} has wrong name"
            );
        }
    }

    /// Spot-check a few commands to ensure the registry's classifications
    /// match the historical TOML.
    #[test]
    fn morphotag_classification_matches_toml() {
        let spec = command_spec_for(ReleasedCommand::Morphotag);
        assert_eq!(spec.profile, Some(WorkerProfile::Stanza));
        assert_eq!(spec.gil_process_need, GilProcessNeed::Always);
        assert_eq!(spec.base_mb_process.0, 8_000);
        // Process-isolated in every runtime, so its budget is the process one.
        assert_eq!(
            spec.base_mb_for_runtime(PythonRuntime::FreeThreaded),
            spec.base_mb_process
        );
        assert_eq!(spec.tasks, &[InferTask::Morphosyntax]);
        assert_eq!(spec.task_label, "morphosyntax");
    }

    #[test]
    fn opensmile_classification_matches_toml() {
        let spec = command_spec_for(ReleasedCommand::Opensmile);
        assert_eq!(spec.profile, Some(WorkerProfile::Io));
        assert_eq!(spec.gil_process_need, GilProcessNeed::Always);
    }

    #[test]
    fn benchmark_carries_composite_task_label() {
        let spec = command_spec_for(ReleasedCommand::Benchmark);
        assert_eq!(spec.task_label, "asr,eval");
        assert_eq!(spec.tasks, &[InferTask::Asr]);
    }

    #[test]
    fn utseg_task_label_diverges_from_infer_task_name() {
        let spec = command_spec_for(ReleasedCommand::Utseg);
        assert_eq!(
            spec.task_label, "utterance",
            "utseg's cmd2task TOML value is the historical 'utterance', \
             not the snake_case InferTask name"
        );
        assert_eq!(spec.tasks, &[InferTask::Utseg]);
    }
}
