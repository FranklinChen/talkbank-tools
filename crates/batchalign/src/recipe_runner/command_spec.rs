//! `CatalogEntry`: static metadata for one entry in the
//! `recipe_command_catalog()`.
//!
//! Renamed from `CommandSpec` in Phase β to free the `CommandSpec`
//! name for the public `batchalign-types::command_spec::CommandSpec`
//! (resource/classification spec). The two types are orthogonal,
//! cross-referenced by `ReleasedCommand` identity.

use crate::api::ReleasedCommand;
use crate::worker::InferTask;

use super::materialize::{OutputDeclaration, OutputPolicy};
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
    /// Native media work, with no resident model or inference task.
    NativeMedia,
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
    /// No ML model or inference worker participates.
    NoModels,
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
    pub const ALL: [Self; 6] = [
        Self::BatchedText,
        Self::ReferenceProjection,
        Self::AudioSequential,
        Self::MediaAnalysis,
        Self::Composite,
        Self::NativeMedia,
    ];

    /// High-level scheduling shape implied by this command family.
    pub const fn scheduling_policy(self) -> SchedulingPolicy {
        match self {
            Self::BatchedText => SchedulingPolicy::CrossFileBatch,
            Self::ReferenceProjection => SchedulingPolicy::ReferenceProjection,
            Self::AudioSequential => SchedulingPolicy::PerFileAudio,
            Self::MediaAnalysis | Self::NativeMedia => SchedulingPolicy::PerFileMediaAnalysis,
            Self::Composite => SchedulingPolicy::Composite,
        }
    }

    /// Model-sharing policy implied by this command family.
    pub const fn model_sharing_policy(self) -> ModelSharingPolicy {
        match self {
            Self::Composite => ModelSharingPolicy::DelegatedToSubcommands,
            Self::NativeMedia => ModelSharingPolicy::NoModels,
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
            Self::MediaAnalysis | Self::NativeMedia | Self::Composite => BatchingPolicy::None,
        }
    }

    /// Parallelism policy implied by this command family.
    pub const fn parallelism_policy(self) -> ParallelismPolicy {
        match self {
            Self::AudioSequential | Self::MediaAnalysis | Self::NativeMedia => {
                ParallelismPolicy::BoundedFileWorkers
            }
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
            Self::MediaAnalysis | Self::NativeMedia => ResourceLane::IoBound,
        }
    }

    /// Constrained-host behavior implied by this command family.
    pub const fn constrained_host_policy(self) -> ConstrainedHostPolicy {
        match self {
            Self::Composite => ConstrainedHostPolicy::DelegatedToSubcommands,
            Self::BatchedText
            | Self::ReferenceProjection
            | Self::AudioSequential
            | Self::MediaAnalysis
            | Self::NativeMedia => ConstrainedHostPolicy::SequentialFallback,
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

/// Execution requirements declared by the command catalog.
///
/// Inference owns exactly one initial task, not a list of downstream stages:
/// later stages select their own workers from their own typed requests. Native
/// operations own no inference task. Neither case can manufacture evidence
/// about a different worker's loaded models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CapabilityPlan {
    /// Worker-backed execution; possession of the payload permits deriving a
    /// worker key. Native execution has no such payload.
    Inference(InferenceRequirement),
    /// Native media execution, without an inference task or Python worker.
    NativeMedia(NativeMediaCapability),
}

/// A native operation supported by the Rust execution host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeMediaCapability {
    /// Checked WAV/MP3 encoding and no-clobber publication.
    AudioExport,
}

/// Catalog-owned evidence that an operation actually requires inference.
///
/// Its fields are private: worker consumers cannot invent a task for a native
/// command. Both targeting and engine selection consume this same requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InferenceRequirement {
    task: InferTask,
    surface: CapabilitySurface,
}

impl InferenceRequirement {
    pub(crate) const fn task(self) -> InferTask {
        self.task
    }
}

impl CapabilityPlan {
    pub(super) const fn inference(task: InferTask, surface: CapabilitySurface) -> Self {
        Self::Inference(InferenceRequirement { task, surface })
    }

    /// Admission precedes worker inspection, allocation or startup. Refusal
    /// carries the native operation rather than a fabricated inference task.
    pub(crate) const fn require_inference(
        self,
    ) -> Result<InferenceRequirement, NativeMediaCapability> {
        match self {
            Self::Inference(requirement) => Ok(requirement),
            Self::NativeMedia(capability) => Err(capability),
        }
    }
}

/// How one released command is surfaced relative to the worker infer-task layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommandCapabilityKind {
    /// Available natively, with no inference capability to probe.
    NativeMedia,
    /// Command is advertised directly from one infer task.
    DirectInfer,
    /// Command is synthesized by Rust from lower-level infer capability.
    ServerComposed,
}

/// Which server-side runtime path currently owns one released command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunnerDispatchKind {
    /// Checked native audio export, not worker protocol inference.
    NativeAudioExport,
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

/// What one command's inputs are, and therefore what the CLI has to put on
/// the wire to run it somewhere else.
///
/// The CLI's `--server` decision (`ServerTarget::parse_explicit`) reads this
/// and nothing else: a command whose inputs are transcripts can be shipped to
/// any server as content, and one whose inputs are recordings can run only
/// where the recordings are. Whether a CHAT source may be submitted to a
/// `MediaInput` command is a separate question, answered at
/// `JobSubmission::validate_source_kinds`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommandIoProfile {
    /// CHAT in, CHAT out. No recording is opened anywhere.
    Text,
    /// CHAT in; the execution host resolves each transcript's recording
    /// itself, from `media_roots`, `media_mappings` or `--media-dir`. The
    /// client ships only the transcript, so any server can be the execution
    /// host, provided it can see the media, and the CLI tells the operator so
    /// when it submits one of these to a remote server.
    ResolvedAudio,
    /// The recordings are the inputs. No transport carries a recording to
    /// another host, so only a server on this filesystem can run it.
    MediaInput,
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
    /// What the inputs are; decides what the CLI ships and where it may run.
    pub io_profile: CommandIoProfile,
    /// Which server-side runtime path currently owns this command.
    pub runner_dispatch_kind: RunnerDispatchKind,
    /// Worker capability requirements.
    pub capabilities: CapabilityPlan,
    /// Output naming and sidecar policy.
    pub output_policy: OutputDeclaration,
    /// Ordered stage recipe for the command.
    pub recipe: &'static Recipe,
}

impl CatalogEntry {
    /// Selection binds a catalog command to its submitted options once.
    pub(crate) fn selected_output_policy(
        &self,
        options: &crate::options::CommandOptions,
    ) -> Result<OutputPolicy, super::planner::PlanningError> {
        if self.command != options.command() {
            return Err(super::planner::PlanningError::OutputOptionsMismatch {
                expected: self.command,
                observed: options.command(),
            });
        }
        self.output_policy.select(options)
    }
}
