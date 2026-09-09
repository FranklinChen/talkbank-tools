//! `CatalogEntry`: static metadata for one entry in the
//! `recipe_command_catalog()`.
//!
//! Renamed from `CommandSpec` in Phase β to free the `CommandSpec`
//! name for the public `batchalign-types::command_spec::CommandSpec`
//! (resource/classification spec). The two types are orthogonal,
//! cross-referenced by `ReleasedCommand` identity.

use crate::api::ReleasedCommand;
use crate::worker::InferTask;

use super::materialize::OutputPolicy;
use super::recipe::Recipe;

/// High-level command family in the replacement architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommandFamily {
    /// Main transcript plus reference transcript projection.
    ReferenceProjection,
    /// Audio-first sequential recipes such as transcribe and align.
    AudioSequential,
    /// Cross-unit text commands that still expose per-file results.
    BatchedText,
    /// Composite commands that reuse other recipes.
    Composite,
    /// Media-analysis commands that emit non-CHAT artifacts.
    MediaAnalysis,
}

/// High-level scheduling shape the command expects from the shared kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SchedulingPolicy {
    /// One audio/media file at a time, with bounded per-job parallelism.
    PerFileAudio,
    /// Many text files pooled into one or more shared infer batches.
    CrossFileBatch,
    /// One primary file plus one paired reference artifact.
    ReferenceProjection,
    /// The command is built by composing other command-owned flows.
    Composite,
    /// Per-file media analysis over non-CHAT inputs.
    PerFileMediaAnalysis,
}

/// How the command expects model state to be shared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelSharingPolicy {
    /// Reuse warm workers and shared model state whenever possible.
    SharedWarmWorkers,
    /// Let composed child commands own model sharing.
    DelegatedToSubcommands,
}

/// Whether the command benefits from cross-file or internal batching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchingPolicy {
    /// No profitable batching beyond ordinary per-file execution.
    None,
    /// Pool many files together into shared worker requests.
    CrossFileBatch,
    /// Keep the top-level unit per file, but allow internal stage batching.
    InternalStageBatching,
    /// One main file plus one paired reference artifact.
    PairedInputs,
}

/// How much per-command parallelism the shared kernel should expose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParallelismPolicy {
    /// Bound file-level concurrency and let the kernel auto-tune worker counts.
    BoundedFileWorkers,
    /// Keep one command-level dispatch at a time per job.
    SingleDispatchPerJob,
    /// Let composed child commands own their own parallelism.
    DelegatedToSubcommands,
}

/// How one command should behave on constrained-memory hosts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConstrainedHostPolicy {
    /// Allow the host to clamp execution to one worker and rely on lazy startup
    /// rather than speculative resident state.
    SequentialFallback,
    /// Let composed child commands own constrained-host behavior.
    DelegatedToSubcommands,
}

/// Dominant resource lane for the command's hot path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResourceLane {
    /// GPU-backed workloads where device memory is the main bottleneck.
    GpuHeavy,
    /// CPU-bound workloads that still reuse warm model workers.
    CpuBound,
    /// Mostly IO / media feature extraction.
    IoBound,
    /// Mixed pipelines touching both CPU and GPU stages.
    Mixed,
}

/// The runtime policy implied by a command's family.
///
/// These derivations lived on a second enum, `CommandExecutionShape`,
/// whose variants were identical to `CommandFamily`'s and which was reached
/// through an `execution_shape_for` that spelled out the identity mapping in
/// five arms. Both halves of the interrupted migration had invented the same
/// concept; the family is the one that is DECLARED per command, so it is the
/// one that survives. Collapsed 2026-07-29.
///
/// A THIRD naming existed too: `WorkflowFamily` in `command_family.rs`, a
/// 4-variant coarsening reached via `workflow_family()`. It merged
/// `AudioSequential` and `MediaAnalysis` into one `PerFileTransform`, losing the
/// distinction, and once the compatibility descriptor that carried it was
/// deleted nothing read it at all. Removed the same day.
impl CommandFamily {
    /// Every command family.
    ///
    /// Exists so tests can iterate the variants without hand-listing them. A
    /// literal list is not forced to grow when a variant is added: the test
    /// keeps compiling and simply never exercises the new family, which is the
    /// silent-default hazard this whole table exists to remove, one level up.
    #[allow(
        dead_code,
        reason = "a property of the enum, consumed by the catalog pin tests"
    )]
    pub const ALL: [Self; 5] = [
        Self::BatchedText,
        Self::ReferenceProjection,
        Self::AudioSequential,
        Self::MediaAnalysis,
        Self::Composite,
    ];

    /// High-level scheduling shape implied by this command family.
    pub const fn scheduling_policy(self) -> SchedulingPolicy {
        match self {
            Self::BatchedText => SchedulingPolicy::CrossFileBatch,
            Self::ReferenceProjection => SchedulingPolicy::ReferenceProjection,
            Self::AudioSequential => SchedulingPolicy::PerFileAudio,
            Self::MediaAnalysis => SchedulingPolicy::PerFileMediaAnalysis,
            Self::Composite => SchedulingPolicy::Composite,
        }
    }

    /// Model-sharing policy implied by this command family.
    pub const fn model_sharing_policy(self) -> ModelSharingPolicy {
        match self {
            Self::Composite => ModelSharingPolicy::DelegatedToSubcommands,
            Self::BatchedText
            | Self::ReferenceProjection
            | Self::AudioSequential
            | Self::MediaAnalysis => ModelSharingPolicy::SharedWarmWorkers,
        }
    }

    /// Batching policy implied by this command family.
    pub const fn batching_policy(self) -> BatchingPolicy {
        match self {
            Self::BatchedText => BatchingPolicy::CrossFileBatch,
            Self::ReferenceProjection => BatchingPolicy::PairedInputs,
            Self::AudioSequential => BatchingPolicy::InternalStageBatching,
            Self::MediaAnalysis | Self::Composite => BatchingPolicy::None,
        }
    }

    /// Parallelism policy implied by this command family.
    pub const fn parallelism_policy(self) -> ParallelismPolicy {
        match self {
            Self::AudioSequential | Self::MediaAnalysis => ParallelismPolicy::BoundedFileWorkers,
            Self::BatchedText | Self::ReferenceProjection => {
                ParallelismPolicy::SingleDispatchPerJob
            }
            Self::Composite => ParallelismPolicy::DelegatedToSubcommands,
        }
    }

    /// Dominant resource lane implied by this command family.
    pub const fn resource_lane(self) -> ResourceLane {
        match self {
            Self::BatchedText => ResourceLane::CpuBound,
            Self::ReferenceProjection | Self::Composite => ResourceLane::Mixed,
            Self::AudioSequential => ResourceLane::GpuHeavy,
            Self::MediaAnalysis => ResourceLane::IoBound,
        }
    }

    /// Constrained-host behavior implied by this command family.
    pub const fn constrained_host_policy(self) -> ConstrainedHostPolicy {
        match self {
            Self::Composite => ConstrainedHostPolicy::DelegatedToSubcommands,
            Self::BatchedText
            | Self::ReferenceProjection
            | Self::AudioSequential
            | Self::MediaAnalysis => ConstrainedHostPolicy::SequentialFallback,
        }
    }

    /// Whether host-memory admission should remain enabled for this shape.
    pub const fn uses_host_memory_gate(self) -> bool {
        true
    }
}

/// Which planner shape owns source discovery for a command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlannerKind {
    /// Plain CHAT inputs with optional `--before`.
    TextInputs,
    /// Plain audio inputs.
    AudioInputs,
    /// Main transcript + gold companion pairing.
    ComparePairs,
    /// Audio input + derived gold CHAT pairing.
    BenchmarkPairs,
    /// Media-analysis audio inputs.
    MediaAnalysisInputs,
}

/// Whether a released command is owned directly by one recipe or by recipe composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CapabilitySurface {
    /// One recipe owns the released command.
    RecipeOwned,
    /// The released command is defined by composing other recipes.
    Composite,
}

/// Worker-capability requirements for one released command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CapabilityPlan {
    /// The infer task the command is ADVERTISED from: the one whose presence in
    /// a worker's reported task set makes this command available at all.
    ///
    /// Named and separated from the rest so that "a command has at least one
    /// infer task" is a fact about the type rather than a runtime `expect` on
    /// `infer_tasks.first()`, which is what it was until 2026-07-29.
    pub primary_infer_task: InferTask,
    /// Further infer tasks the recipe reaches somewhere after the first.
    ///
    /// Declared but not yet read: capability advertisement keys off the primary
    /// task alone, exactly as it did when this was one `infer_tasks` slice and
    /// only `.first()` was consumed. Kept because it states a real fact about
    /// the recipe (transcribe_s also needs Speaker, benchmark also needs
    /// Morphosyntax) that a reader would otherwise have to reconstruct from the
    /// stage list. Widening advertisement to include these would change what
    /// `/health` reports, so it is a deliberate decision, not a cleanup.
    pub additional_infer_tasks: &'static [InferTask],
    /// Whether the released command is recipe-owned or composed.
    pub surface: CapabilitySurface,
}

/// How one released command is surfaced relative to the worker infer-task layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommandCapabilityKind {
    /// Command is advertised directly from one infer task.
    DirectInfer,
    /// Command is synthesized by Rust from lower-level infer capability.
    ServerComposed,
}

/// Which server-side runtime path currently owns one released command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunnerDispatchKind {
    /// Text-only commands pooled through the batched infer path.
    BatchedTextInfer,
    /// Forced alignment with per-file audio/media resolution.
    ForcedAlignment,
    /// Transcribe audio through the Rust-owned ASR orchestration path.
    TranscribeAudioInfer,
    /// Benchmark audio through the composite benchmark orchestrator.
    BenchmarkAudioInfer,
    /// Media-analysis V2 path for commands like openSMILE and AVQI.
    MediaAnalysisV2,
    /// Speaker identification against enrolled spans, CHAT in, evidence out.
    SpeakerIdentity,
}

/// How the CLI should ship inputs to the server for this command.
///
/// Makes the content/paths-mode superset structural: a command either
/// uploads file bodies over HTTP, sends paths for CHAT-only inputs, or
/// sends paths plus requires shared-filesystem audio access. The illegal
/// combination "needs local audio but cannot use paths mode" is
/// unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommandIoProfile {
    /// CLI uploads full file bodies over HTTP; server never reads client paths.
    ///
    /// Unconstructed today. Kept because it is the only way to express a
    /// command that cannot use paths mode at all, the correct shape for a
    /// REMOTE daemon (paths mode needs a shared filesystem). Its absence is
    /// why `supports_paths_mode()` is currently true for every released
    /// command.
    #[allow(dead_code, reason = "expresses remote-daemon-only commands; see above")]
    ContentOnly,
    /// CLI sends filesystem paths for text inputs; server reads CHAT directly.
    PathsModeText,
    /// CLI sends filesystem paths and the command also needs client-local
    /// audio on the shared filesystem (only valid for a local daemon).
    PathsModeAudio,
}

impl CommandIoProfile {
    /// Whether the server-side runner needs shared-filesystem audio access.
    pub const fn uses_local_audio(self) -> bool {
        matches!(self, Self::PathsModeAudio)
    }

    /// Whether the CLI may send paths instead of inlined content to a local daemon.
    pub const fn supports_paths_mode(self) -> bool {
        matches!(self, Self::PathsModeText | Self::PathsModeAudio)
    }
}

/// Static command metadata for the recipe-runner catalog.
///
/// # One thing it deliberately does NOT declare
///
/// The execution mode. It was a `pub execution_mode: ExecutionMode` field
/// written out by all thirteen entries beside the recipe that already carries
/// it, held equal by a drift test in `catalog.rs`, which is a standing
/// confession that one of the two should not exist. The recipe owns the
/// stages, so it owns their mode: ask `entry.recipe.mode`.
///
/// Every field is DECLARED per command in `catalog.rs`; nothing here is
/// inferred from the command's name or position. Three of these fields
/// (`capability_kind`, `io_profile`, `runner_dispatch_kind`) were until
/// 2026-07-29 computed by `match` arms with a catch-all `_ =>` default, so a
/// newly released command silently inherited whatever the default happened to
/// be. Stating them here means a new entry cannot compile until each question
/// has an answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CatalogEntry {
    /// Stable released command identity.
    pub command: ReleasedCommand,
    /// High-level family for contributor understanding.
    pub family: CommandFamily,
    /// Planner shape used to derive work units.
    pub planner: PlannerKind,
    /// Whether the command is advertised straight from one infer task or
    /// synthesized by the server from lower-level capability.
    pub capability_kind: CommandCapabilityKind,
    /// How the CLI ships inputs and whether paths mode is eligible.
    pub io_profile: CommandIoProfile,
    /// Which server-side runtime path currently owns this command.
    pub runner_dispatch_kind: RunnerDispatchKind,
    /// Worker capability requirements.
    pub capabilities: CapabilityPlan,
    /// Output naming and sidecar policy.
    pub output_policy: OutputPolicy,
    /// Ordered stage recipe for the command.
    pub recipe: &'static Recipe,
}
