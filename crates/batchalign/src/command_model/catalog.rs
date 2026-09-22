//! Lookup over the one released-command catalog.

use crate::ReleasedCommand;
use crate::recipe_runner::catalog::recipe_command_catalog;

use super::CatalogEntry;

/// Return the authoritative catalog entry for one released command.
pub(crate) fn command_spec(command: ReleasedCommand) -> &'static CatalogEntry {
    // Catalog invariant: every `ReleasedCommand` variant has an entry in
    // `recipe_command_catalog()`, pinned by
    // `recipe_runner::catalog::tests::every_released_command_has_a_spec`. An
    // omission is a failing test, never a runtime surprise in the field.
    #[allow(clippy::expect_used)]
    command_specs()
        .iter()
        .find(|spec| spec.command == command)
        .expect("released command missing authoritative catalog entry")
}

/// Return the authoritative catalog entries for all released commands, in
/// capability-advertisement order.
pub(crate) fn command_specs() -> &'static [CatalogEntry] {
    recipe_command_catalog()
}

/// Every command whose provenance stamp a `command` job can write: its own,
/// and the stamp of each stage its recipe composes, following a composite
/// stage into the recipe it runs.
///
/// Derived from each stage's [`RecipeStageId::provenance`], so a recipe that
/// gains a stamping stage is covered with no second list. The no-op write gate
/// sets aside exactly these stamps (and still compares their fields). Small
/// and duplicate-free, in first-seen order.
///
/// [`RecipeStageId::provenance`]: crate::recipe_runner::recipe::RecipeStageId::provenance
pub(crate) fn commands_stamped_by(command: ReleasedCommand) -> Vec<ReleasedCommand> {
    use crate::recipe_runner::recipe::StageProvenance;

    let mut stamped = vec![command];
    for stage in command_spec(command).recipe.stages {
        let found = match stage.id.provenance() {
            StageProvenance::None => continue,
            StageProvenance::Stamps(stage_command) => vec![stage_command],
            StageProvenance::RunsRecipeOf(recipe_command) => commands_stamped_by(recipe_command),
        };
        for stage_command in found {
            if !stamped.contains(&stage_command) {
                stamped.push(stage_command);
            }
        }
    }
    stamped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command_model::{
        BatchingPolicy, CommandCapabilityKind, ConstrainedHostPolicy, ModelSharingPolicy,
        ParallelismPolicy, ResourceLane, RunnerDispatchKind, SchedulingPolicy,
    };
    // Straight from their real home: production reaches these two only through a
    // `CatalogEntry` field or a method on it, so `command_model` has no reason to
    // re-export them just for a test.
    use crate::recipe_runner::command_spec::{CommandFamily, CommandIoProfile};
    use crate::recipe_runner::materialize::{FileNamingPolicy, StemRewrite};
    use crate::worker::InferTask;

    /// Output filenames are a user-visible contract, and after the 2026-07-28
    /// cleanup there is exactly ONE representation of them: the `output_policy`
    /// declared per `CatalogEntry`.
    ///
    /// There used to be a second, `CommandOutputPathKind`, derived by matching
    /// on the command name with a catch-all `_ => PreserveInputName`. It was
    /// read by nothing in production and it contradicted the declared policy for
    /// four commands (benchmark, opensmile, avqi, diarize). Worse, it could not
    /// express `RewriteStem` at all, so for diarize (`X.turns.json`) it was
    /// structurally incapable of stating the truth. It was deleted rather than
    /// synchronised. This test pins the survivor so the contract cannot drift
    /// silently.
    #[test]
    fn declared_output_naming_is_stable() {
        // Matched on the ENUM with no catch-all arm: adding a `ReleasedCommand`
        // fails to compile here until its output naming is stated. A `_ =>`
        // default would reintroduce, in miniature, the silent-default hazard
        // the deleted `output_path_kind_for` embodied.
        for command in ReleasedCommand::ALL {
            let expected = match command {
                ReleasedCommand::Transcribe
                | ReleasedCommand::TranscribeS
                | ReleasedCommand::Benchmark => FileNamingPolicy::ReplaceExtension("cha"),
                ReleasedCommand::Opensmile => FileNamingPolicy::RewriteStem(StemRewrite {
                    strip_suffix: None,
                    append_suffix: ".opensmile",
                    extension: "csv",
                }),
                ReleasedCommand::Avqi => FileNamingPolicy::RewriteStem(StemRewrite {
                    strip_suffix: Some(".cs"),
                    append_suffix: ".avqi",
                    extension: "txt",
                }),
                ReleasedCommand::Diarize => FileNamingPolicy::RewriteStem(StemRewrite {
                    strip_suffix: None,
                    append_suffix: ".turns",
                    extension: "json",
                }),
                // Beside the transcript rather than over it: this command does
                // not rewrite CHAT, so its input is also an input to whatever
                // reads its verdicts next.
                ReleasedCommand::SpeakerIdentify => FileNamingPolicy::RewriteStem(StemRewrite {
                    strip_suffix: None,
                    append_suffix: "_speaker_identity",
                    extension: "json",
                }),
                // The CHAT transforms all rewrite their input in place.
                ReleasedCommand::Morphotag
                | ReleasedCommand::Utseg
                | ReleasedCommand::Translate
                | ReleasedCommand::Coref
                | ReleasedCommand::Align
                | ReleasedCommand::Compare => FileNamingPolicy::PreserveInput,
            };
            assert_eq!(
                command_spec(command).output_policy.primary,
                expected,
                "output naming changed for {command}"
            );
        }
    }

    /// Everything the rest of the app reads about ONE released command, in one
    /// named record.
    ///
    /// Written as a struct rather than a tuple deliberately: a tuple of five
    /// same-shaped enums is exactly the kind of positional seam that lets a
    /// transcription error pass review (charter rule: no tuple-packed domain
    /// seams).
    #[derive(Debug, PartialEq, Eq)]
    struct CommandMetadataPin {
        command: ReleasedCommand,
        family: CommandFamily,
        capability_kind: CommandCapabilityKind,
        io_profile: CommandIoProfile,
        runner_dispatch_kind: RunnerDispatchKind,
        primary_infer_task: InferTask,
    }

    /// The pinned metadata for every released command.
    ///
    /// Written as a CHARACTERIZATION table to make the 2026-07-29 collapse of
    /// three catalogs into one provably value-preserving, and kept as a
    /// standing gate. Three of these fields used to come from `match` arms with
    /// a catch-all default: the same hazard `declared_output_naming_is_stable`
    /// above pins for output naming. They are declared fields of `CatalogEntry`
    /// now, so the compiler shares the work, and this table is what stops a
    /// future edit from quietly changing an answer.
    fn command_metadata_pins() -> Vec<CommandMetadataPin> {
        // Matched on the enum with NO catch-all arm, so adding a released
        // command fails to compile here until its metadata is stated.
        ReleasedCommand::ALL
            .into_iter()
            .map(|command| match command {
                ReleasedCommand::Compare => CommandMetadataPin {
                    command,
                    family: CommandFamily::ReferenceProjection,
                    capability_kind: CommandCapabilityKind::DirectInfer,
                    io_profile: CommandIoProfile::Text,
                    runner_dispatch_kind: RunnerDispatchKind::BatchedTextInfer,
                    primary_infer_task: InferTask::Morphosyntax,
                },
                ReleasedCommand::Benchmark => CommandMetadataPin {
                    command,
                    family: CommandFamily::Composite,
                    capability_kind: CommandCapabilityKind::ServerComposed,
                    io_profile: CommandIoProfile::MediaInput,
                    runner_dispatch_kind: RunnerDispatchKind::BenchmarkAudioInfer,
                    primary_infer_task: InferTask::Asr,
                },
                ReleasedCommand::Transcribe => CommandMetadataPin {
                    command,
                    family: CommandFamily::AudioSequential,
                    capability_kind: CommandCapabilityKind::ServerComposed,
                    io_profile: CommandIoProfile::MediaInput,
                    runner_dispatch_kind: RunnerDispatchKind::TranscribeAudioInfer,
                    primary_infer_task: InferTask::Asr,
                },
                ReleasedCommand::TranscribeS => CommandMetadataPin {
                    command,
                    family: CommandFamily::AudioSequential,
                    capability_kind: CommandCapabilityKind::ServerComposed,
                    io_profile: CommandIoProfile::MediaInput,
                    runner_dispatch_kind: RunnerDispatchKind::TranscribeAudioInfer,
                    primary_infer_task: InferTask::Asr,
                },
                ReleasedCommand::Align => CommandMetadataPin {
                    command,
                    family: CommandFamily::AudioSequential,
                    capability_kind: CommandCapabilityKind::DirectInfer,
                    io_profile: CommandIoProfile::ResolvedAudio,
                    runner_dispatch_kind: RunnerDispatchKind::ForcedAlignment,
                    primary_infer_task: InferTask::Fa,
                },
                ReleasedCommand::Morphotag => CommandMetadataPin {
                    command,
                    family: CommandFamily::BatchedText,
                    capability_kind: CommandCapabilityKind::DirectInfer,
                    io_profile: CommandIoProfile::Text,
                    runner_dispatch_kind: RunnerDispatchKind::BatchedTextInfer,
                    primary_infer_task: InferTask::Morphosyntax,
                },
                ReleasedCommand::Utseg => CommandMetadataPin {
                    command,
                    family: CommandFamily::BatchedText,
                    capability_kind: CommandCapabilityKind::DirectInfer,
                    io_profile: CommandIoProfile::Text,
                    runner_dispatch_kind: RunnerDispatchKind::BatchedTextInfer,
                    primary_infer_task: InferTask::Utseg,
                },
                ReleasedCommand::Translate => CommandMetadataPin {
                    command,
                    family: CommandFamily::BatchedText,
                    capability_kind: CommandCapabilityKind::DirectInfer,
                    io_profile: CommandIoProfile::Text,
                    runner_dispatch_kind: RunnerDispatchKind::BatchedTextInfer,
                    primary_infer_task: InferTask::Translate,
                },
                ReleasedCommand::Coref => CommandMetadataPin {
                    command,
                    family: CommandFamily::BatchedText,
                    capability_kind: CommandCapabilityKind::DirectInfer,
                    io_profile: CommandIoProfile::Text,
                    runner_dispatch_kind: RunnerDispatchKind::BatchedTextInfer,
                    primary_infer_task: InferTask::Coref,
                },
                ReleasedCommand::Opensmile => CommandMetadataPin {
                    command,
                    family: CommandFamily::MediaAnalysis,
                    capability_kind: CommandCapabilityKind::DirectInfer,
                    io_profile: CommandIoProfile::MediaInput,
                    runner_dispatch_kind: RunnerDispatchKind::MediaAnalysisV2,
                    primary_infer_task: InferTask::Opensmile,
                },
                ReleasedCommand::Avqi => CommandMetadataPin {
                    command,
                    family: CommandFamily::MediaAnalysis,
                    capability_kind: CommandCapabilityKind::DirectInfer,
                    io_profile: CommandIoProfile::MediaInput,
                    runner_dispatch_kind: RunnerDispatchKind::MediaAnalysisV2,
                    primary_infer_task: InferTask::Avqi,
                },
                ReleasedCommand::Diarize => CommandMetadataPin {
                    command,
                    family: CommandFamily::MediaAnalysis,
                    capability_kind: CommandCapabilityKind::DirectInfer,
                    io_profile: CommandIoProfile::MediaInput,
                    runner_dispatch_kind: RunnerDispatchKind::MediaAnalysisV2,
                    primary_infer_task: InferTask::Speaker,
                },
                ReleasedCommand::SpeakerIdentify => CommandMetadataPin {
                    command,
                    family: CommandFamily::AudioSequential,
                    capability_kind: CommandCapabilityKind::DirectInfer,
                    io_profile: CommandIoProfile::ResolvedAudio,
                    runner_dispatch_kind: RunnerDispatchKind::SpeakerIdentity,
                    primary_infer_task: InferTask::Speaker,
                },
            })
            .collect()
    }

    /// The whole per-command metadata surface, pinned command by command.
    ///
    /// Every consumer in the crate (capability advertisement, worker targeting,
    /// runner routing, the CLI's paths-mode decision, kernel planning) reads one
    /// of these five values, so holding all five constant is what makes a
    /// restructuring of the catalog provably behaviour-preserving.
    #[test]
    fn per_command_metadata_is_stable() {
        for pin in command_metadata_pins() {
            let spec = command_spec(pin.command);
            let actual = CommandMetadataPin {
                command: spec.command,
                family: spec.family,
                capability_kind: spec.capability_kind,
                io_profile: spec.io_profile,
                runner_dispatch_kind: spec.runner_dispatch_kind,
                primary_infer_task: spec.capabilities.primary_infer_task,
            };
            assert_eq!(actual, pin, "declared metadata for {}", pin.command);
        }
    }

    /// Everything a command FAMILY implies about runtime policy, in one record.
    ///
    /// A `workflow` field pinning `WorkflowFamily` was dropped when that enum
    /// was deleted: it was a third name for this same concept, coarser than
    /// `CommandFamily` and read by nothing.
    #[derive(Debug, PartialEq, Eq)]
    struct FamilyPolicyPin {
        scheduling: SchedulingPolicy,
        model_sharing: ModelSharingPolicy,
        batching: BatchingPolicy,
        parallelism: ParallelismPolicy,
        resource_lane: ResourceLane,
        constrained_host: ConstrainedHostPolicy,
    }

    /// The policy vocabulary a command family implies, pinned family by family.
    ///
    /// The per-command table above pins WHICH family each command is in; this
    /// pins what being in that family MEANS. Together they cover the whole
    /// derivation, which lets the two identical five-variant enums
    /// (`CommandFamily` and `CommandExecutionShape`) be collapsed into one
    /// without silently re-mapping a family onto different runtime policy.
    #[test]
    fn family_policy_derivation_is_stable() {
        // Iterating `CommandFamily::ALL`, not a literal list: a hand-written
        // list is not forced to grow with the enum, so a sixth family would
        // compile and silently never be exercised here, while the `match` below
        // WOULD force an arm. The two must be driven by the same source.
        for family in CommandFamily::ALL {
            let expected = match family {
                CommandFamily::BatchedText => FamilyPolicyPin {
                    scheduling: SchedulingPolicy::CrossFileBatch,
                    model_sharing: ModelSharingPolicy::SharedWarmWorkers,
                    batching: BatchingPolicy::CrossFileBatch,
                    parallelism: ParallelismPolicy::SingleDispatchPerJob,
                    resource_lane: ResourceLane::CpuBound,
                    constrained_host: ConstrainedHostPolicy::SequentialFallback,
                },
                CommandFamily::ReferenceProjection => FamilyPolicyPin {
                    scheduling: SchedulingPolicy::ReferenceProjection,
                    model_sharing: ModelSharingPolicy::SharedWarmWorkers,
                    batching: BatchingPolicy::PairedInputs,
                    parallelism: ParallelismPolicy::SingleDispatchPerJob,
                    resource_lane: ResourceLane::Mixed,
                    constrained_host: ConstrainedHostPolicy::SequentialFallback,
                },
                CommandFamily::AudioSequential => FamilyPolicyPin {
                    scheduling: SchedulingPolicy::PerFileAudio,
                    model_sharing: ModelSharingPolicy::SharedWarmWorkers,
                    batching: BatchingPolicy::InternalStageBatching,
                    parallelism: ParallelismPolicy::BoundedFileWorkers,
                    resource_lane: ResourceLane::GpuHeavy,
                    constrained_host: ConstrainedHostPolicy::SequentialFallback,
                },
                CommandFamily::MediaAnalysis => FamilyPolicyPin {
                    scheduling: SchedulingPolicy::PerFileMediaAnalysis,
                    model_sharing: ModelSharingPolicy::SharedWarmWorkers,
                    batching: BatchingPolicy::None,
                    parallelism: ParallelismPolicy::BoundedFileWorkers,
                    resource_lane: ResourceLane::IoBound,
                    constrained_host: ConstrainedHostPolicy::SequentialFallback,
                },
                CommandFamily::Composite => FamilyPolicyPin {
                    scheduling: SchedulingPolicy::Composite,
                    model_sharing: ModelSharingPolicy::DelegatedToSubcommands,
                    batching: BatchingPolicy::None,
                    parallelism: ParallelismPolicy::DelegatedToSubcommands,
                    resource_lane: ResourceLane::Mixed,
                    constrained_host: ConstrainedHostPolicy::DelegatedToSubcommands,
                },
            };
            let actual = FamilyPolicyPin {
                scheduling: family.scheduling_policy(),
                model_sharing: family.model_sharing_policy(),
                batching: family.batching_policy(),
                parallelism: family.parallelism_policy(),
                resource_lane: family.resource_lane(),
                constrained_host: family.constrained_host_policy(),
            };
            assert_eq!(actual, expected, "policy derivation for {family:?}");
            assert!(
                family.uses_host_memory_gate(),
                "host memory gate must stay on for {family:?}"
            );
        }
    }

    #[test]
    fn command_specs_are_unique_and_stamped_commands_follow_the_recipes() {
        use super::commands_stamped_by;

        // Composition is read from the recipes: transcribe stamps its utseg
        // and morphotag stages too, and benchmark stamps whatever the
        // transcribe and compare recipes it runs stamp.
        let sorted = |command| {
            let mut names: Vec<&str> = commands_stamped_by(command)
                .into_iter()
                .map(ReleasedCommand::as_str)
                .collect();
            names.sort_unstable();
            names
        };
        assert_eq!(
            sorted(ReleasedCommand::Transcribe),
            ["morphotag", "transcribe", "utseg"]
        );
        assert_eq!(
            sorted(ReleasedCommand::TranscribeS),
            ["morphotag", "transcribe", "transcribe_s", "utseg"]
        );
        assert_eq!(
            sorted(ReleasedCommand::Benchmark),
            ["benchmark", "compare", "morphotag", "transcribe", "utseg"]
        );
        assert_eq!(sorted(ReleasedCommand::Align), ["align"]);
        assert_eq!(sorted(ReleasedCommand::Morphotag), ["morphotag"]);
        assert_eq!(sorted(ReleasedCommand::Translate), ["translate"]);

        command_specs_are_unique();
    }

    fn command_specs_are_unique() {
        let mut names: Vec<_> = command_specs().iter().map(|spec| spec.command).collect();
        let original_len = names.len();
        names.sort_unstable_by_key(|command| command.as_ref().to_owned());
        names.dedup();
        assert_eq!(names.len(), original_len);
    }
}
