//! Typed recipe metadata for the recipe-runner spike.

use std::fmt;

use crate::runner::util::FileStage;

/// How a command recipe owns execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExecutionMode {
    /// One work unit progresses through an ordered recipe.
    SequentialPerUnit,
    /// One or more stages explicitly batch work units together.
    BatchedStage,
    /// A main transcript is projected against a typed reference companion.
    ReferenceProjection,
    /// The recipe delegates to other recipes as sub-workflows.
    Composite,
}

/// Stable identifiers for recipe-runner stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum RecipeStageId {
    /// Turn discovered files into typed work units.
    PlanWorkUnits,
    /// Read main CHAT inputs.
    ReadChatInputs,
    /// Read paired reference inputs.
    ReadReferenceInputs,
    /// Resolve and normalize media inputs.
    ResolveAudio,
    /// Automatic speech recognition.
    AsrInfer,
    /// Optional or required speaker diarization.
    SpeakerDiarization,
    /// Rust-owned ASR post-processing.
    AsrPostprocess,
    /// Build CHAT from utterance state.
    BuildChat,
    /// Optional utterance segmentation.
    UtteranceSegmentation,
    /// Morphosyntax enrichment.
    Morphosyntax,
    /// Forced alignment.
    ForcedAlignment,
    /// Cross-unit worker batching.
    BatchInfer,
    /// Align the main transcript against a reference transcript.
    CompareAlign,
    /// Derive compare metrics and sidecar data.
    CompareMetrics,
    /// Dispatch one media-analysis request.
    MediaAnalysis,
    /// Reuse the transcribe recipe inside a composite command.
    RunTranscribeRecipe,
    /// Reuse the compare recipe inside a composite command.
    RunCompareRecipe,
    /// Final CHAT serialization before persistence.
    SerializeChat,
    /// Turn recipe outputs into persistent artifacts.
    MaterializeOutputs,
}

impl RecipeStageId {
    /// Equality usable inside a `const fn`.
    ///
    /// The derived `PartialEq` is not const, and [`Recipe::check`] must run at
    /// compile time. The enum is fieldless, so the discriminant IS the
    /// identity and the cast is exact.
    const fn same(self, other: Self) -> bool {
        self as u8 == other as u8
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::PlanWorkUnits => "plan_work_units",
            Self::ReadChatInputs => "read_chat_inputs",
            Self::ReadReferenceInputs => "read_reference_inputs",
            Self::ResolveAudio => "resolve_audio",
            Self::AsrInfer => "asr_infer",
            Self::SpeakerDiarization => "speaker_diarization",
            Self::AsrPostprocess => "asr_postprocess",
            Self::BuildChat => "build_chat",
            Self::UtteranceSegmentation => "utterance_segmentation",
            Self::Morphosyntax => "morphosyntax",
            Self::ForcedAlignment => "forced_alignment",
            Self::BatchInfer => "batch_infer",
            Self::CompareAlign => "compare_align",
            Self::CompareMetrics => "compare_metrics",
            Self::MediaAnalysis => "media_analysis",
            Self::RunTranscribeRecipe => "run_transcribe_recipe",
            Self::RunCompareRecipe => "run_compare_recipe",
            Self::SerializeChat => "serialize_chat",
            Self::MaterializeOutputs => "materialize_outputs",
        }
    }
}

impl fmt::Display for RecipeStageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Whether a stage is always present or controlled by command options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecipeStagePresence {
    /// The stage must always run.
    Required,
    /// The stage is part of the recipe but option-gated at runtime.
    Optional,
}

/// Whether a stage runs per work unit, across work units, or by delegating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StageExecutionKind {
    /// Each work unit runs the stage independently.
    PerWorkUnit,
    /// The stage pools multiple work units into one dispatch.
    BatchedAcrossWorkUnits,
    /// The stage delegates to another recipe.
    CompositeSubrecipe,
}

/// Static metadata for one recipe stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RecipeStage {
    /// Stable stage id.
    pub id: RecipeStageId,
    /// Whether the stage is always present.
    pub presence: RecipeStagePresence,
    /// Execution shape for this stage.
    pub execution: StageExecutionKind,
    /// Progress label surfaced to operators.
    pub progress_stage: FileStage,
    /// Other stages that must complete first.
    pub depends_on: &'static [RecipeStageId],
}

impl RecipeStage {
    /// Construct one static recipe stage.
    pub(crate) const fn new(
        id: RecipeStageId,
        presence: RecipeStagePresence,
        execution: StageExecutionKind,
        progress_stage: FileStage,
        depends_on: &'static [RecipeStageId],
    ) -> Self {
        Self {
            id,
            presence,
            execution,
            progress_stage,
            depends_on,
        }
    }
}

/// Why a stage list is not a legal recipe.
///
/// A sum type rather than a formatted string, for two reasons. It is returned
/// from a `const fn`, where formatting does not exist; and a test asserting
/// "the message contains 'declared after'" is a substring check standing in
/// for a fact the type can carry outright.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecipeCheck {
    /// The stage list is a legal recipe.
    Ok,
    /// A stage lists itself as its own prerequisite.
    SelfDependency(RecipeStageId),
    /// A prerequisite exists in the recipe but is declared AFTER its dependent.
    /// The runtime walks `stages` in declaration order, so it would run second.
    DependencyDeclaredLater {
        /// The dependent stage.
        stage: RecipeStageId,
        /// The prerequisite declared too late.
        dependency: RecipeStageId,
    },
    /// A prerequisite is not in the recipe at all.
    DependencyNotInRecipe {
        /// The dependent stage.
        stage: RecipeStageId,
        /// The prerequisite that is missing.
        dependency: RecipeStageId,
    },
    /// One stage id appears twice.
    DuplicateStage(RecipeStageId),
}

/// Proof that a [`Recipe`] went through [`Recipe::new`].
///
/// A private zero-sized field, which is the whole mechanism: `Recipe`'s own
/// fields stay readable everywhere, and no module outside this one can write
/// `Recipe { mode, stages }` and skip the check. Without it the constructor
/// would be a convention, and a convention is what the `#[cfg(test)]`
/// validator already was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Validated;

/// Static recipe metadata for one released command.
///
/// Every value of this type is a VALID recipe: dependencies exist, are
/// declared before their dependents, and no stage id repeats. That is checked
/// by [`Recipe::new`] at COMPILE time, because the catalog's recipes are
/// `const` items, so a malformed recipe is a build failure rather than a
/// runtime surprise on the one command nobody submitted this week.
///
/// Before 2026-09-07 the checker was `#[cfg(test)]` and returned a
/// `Result`, so the shipped recipes were validated only insofar as some test
/// remembered to walk them
/// (`rg -c 'const \w+_RECIPE: Recipe' recipes.rs` counts them).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Recipe {
    /// Top-level execution mode for the command.
    pub mode: ExecutionMode,
    /// Ordered recipe stages.
    pub stages: &'static [RecipeStage],
    /// Evidence that [`Recipe::new`] checked `stages`.
    _validated: Validated,
}

impl Recipe {
    /// Build a recipe, refusing an illegal stage list at compile time.
    ///
    /// THE only way to obtain a `Recipe`. The panic messages are static
    /// because a `const` panic cannot format; the variant naming the offending
    /// stage is available to a caller that wants detail, via [`Self::check`],
    /// and the compiler points at the offending `const` item either way.
    ///
    /// The `panic!`s below are the refusal mechanism of `const` evaluation,
    /// which has no other: every caller is a `const` item in the catalog, so
    /// they fire at COMPILE time and never in a running process. `expect`
    /// rather than `allow`, so that if the lint ever stops firing here the
    /// justification is re-examined instead of silently kept.
    #[expect(
        clippy::panic,
        reason = "compile-time refusal inside a const fn; the workspace ban is on runtime panics"
    )]
    pub(crate) const fn new(mode: ExecutionMode, stages: &'static [RecipeStage]) -> Self {
        match Self::check(stages) {
            RecipeCheck::Ok => Self {
                mode,
                stages,
                _validated: Validated,
            },
            RecipeCheck::SelfDependency(_) => {
                panic!("recipe stage depends on itself")
            }
            RecipeCheck::DependencyDeclaredLater { .. } => {
                panic!("recipe stage depends on a stage declared after it")
            }
            RecipeCheck::DependencyNotInRecipe { .. } => {
                panic!("recipe stage depends on a stage that is not in the recipe")
            }
            RecipeCheck::DuplicateStage(_) => panic!("recipe declares one stage twice"),
        }
    }

    /// Decide whether a stage list is a legal recipe.
    ///
    /// Checks three things in ONE pass, so that ORDER is checked too. The
    /// runtime executes `stages` in declaration order, so a prerequisite
    /// declared after its dependent would run after it; the dependency graph
    /// is only meaningful if the declared sequence is a topological order of
    /// it. A two-pass form cannot see that, because collecting every id first
    /// makes a forward reference indistinguishable from a backward one.
    ///
    /// Written with index loops rather than iterators because it must be
    /// callable from a `const` context, where `Iterator` is not available.
    pub(crate) const fn check(stages: &'static [RecipeStage]) -> RecipeCheck {
        let mut i = 0;
        while i < stages.len() {
            let stage = stages[i];

            // Every prerequisite must already have been DECLARED, i.e. appear
            // somewhere in stages[0..i].
            let dependencies = stage.depends_on;
            let mut d = 0;
            while d < dependencies.len() {
                let dependency = dependencies[d];
                if dependency.same(stage.id) {
                    return RecipeCheck::SelfDependency(stage.id);
                }

                let mut earlier = 0;
                let mut declared_before = false;
                while earlier < i {
                    if stages[earlier].id.same(dependency) {
                        declared_before = true;
                        break;
                    }
                    earlier += 1;
                }

                if !declared_before {
                    // Distinguish the two ways this happens: they need
                    // different fixes (reorder the stages, or add the missing
                    // one).
                    let mut later = i;
                    while later < stages.len() {
                        if stages[later].id.same(dependency) {
                            return RecipeCheck::DependencyDeclaredLater {
                                stage: stage.id,
                                dependency,
                            };
                        }
                        later += 1;
                    }
                    return RecipeCheck::DependencyNotInRecipe {
                        stage: stage.id,
                        dependency,
                    };
                }
                d += 1;
            }

            // No stage id may repeat.
            let mut earlier = 0;
            while earlier < i {
                if stages[earlier].id.same(stage.id) {
                    return RecipeCheck::DuplicateStage(stage.id);
                }
                earlier += 1;
            }

            i += 1;
        }

        RecipeCheck::Ok
    }

    /// Return the ordered stage ids for inspection tests and docs.
    #[cfg(test)]
    pub(crate) fn ordered_stage_ids(&self) -> Vec<RecipeStageId> {
        self.stages.iter().map(|stage| stage.id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_STAGES: &[RecipeStage] = &[
        RecipeStage::new(
            RecipeStageId::PlanWorkUnits,
            RecipeStagePresence::Required,
            StageExecutionKind::PerWorkUnit,
            FileStage::Reading,
            &[],
        ),
        RecipeStage::new(
            RecipeStageId::MaterializeOutputs,
            RecipeStagePresence::Required,
            StageExecutionKind::PerWorkUnit,
            FileStage::Writing,
            &[RecipeStageId::PlanWorkUnits],
        ),
    ];

    const DUPLICATE_STAGES: &[RecipeStage] = &[
        RecipeStage::new(
            RecipeStageId::PlanWorkUnits,
            RecipeStagePresence::Required,
            StageExecutionKind::PerWorkUnit,
            FileStage::Reading,
            &[],
        ),
        RecipeStage::new(
            RecipeStageId::PlanWorkUnits,
            RecipeStagePresence::Required,
            StageExecutionKind::PerWorkUnit,
            FileStage::Writing,
            &[],
        ),
    ];

    /// A stage whose prerequisite is declared AFTER it. Every id referenced
    /// exists and nothing is self-dependent, so the only thing wrong is the
    /// order, which is exactly what the runtime relies on.
    const FORWARD_DEPENDENCY_STAGES: &[RecipeStage] = &[
        RecipeStage::new(
            RecipeStageId::MaterializeOutputs,
            RecipeStagePresence::Required,
            StageExecutionKind::PerWorkUnit,
            FileStage::Writing,
            &[RecipeStageId::PlanWorkUnits],
        ),
        RecipeStage::new(
            RecipeStageId::PlanWorkUnits,
            RecipeStagePresence::Required,
            StageExecutionKind::PerWorkUnit,
            FileStage::Reading,
            &[],
        ),
    ];

    /// A stage depending on an id that is not in the recipe at all.
    const MISSING_DEPENDENCY_STAGES: &[RecipeStage] = &[RecipeStage::new(
        RecipeStageId::MaterializeOutputs,
        RecipeStagePresence::Required,
        StageExecutionKind::PerWorkUnit,
        FileStage::Writing,
        &[RecipeStageId::AsrInfer],
    )];

    /// The two failure modes need different fixes (reorder the stages versus add
    /// the missing one), so the verdict must tell them apart.
    #[test]
    fn recipe_check_rejects_dependency_outside_the_recipe() {
        assert!(matches!(
            Recipe::check(MISSING_DEPENDENCY_STAGES),
            RecipeCheck::DependencyNotInRecipe { .. }
        ));
    }

    /// The runtime executes `stages` in declaration order, so declaring a stage
    /// before its own prerequisite means running it before its prerequisite.
    /// An earlier two-pass check accepted this: it collected every id first and
    /// then checked dependencies against the complete set, which makes a
    /// forward reference indistinguishable from a backward one.
    #[test]
    fn recipe_check_rejects_dependency_declared_later() {
        assert!(matches!(
            Recipe::check(FORWARD_DEPENDENCY_STAGES),
            RecipeCheck::DependencyDeclaredLater { .. }
        ));
    }

    #[test]
    fn recipe_check_accepts_unique_known_dependencies() {
        assert!(matches!(Recipe::check(VALID_STAGES), RecipeCheck::Ok));
    }

    #[test]
    fn recipe_check_rejects_duplicate_stage_ids() {
        assert!(matches!(
            Recipe::check(DUPLICATE_STAGES),
            RecipeCheck::DuplicateStage(RecipeStageId::PlanWorkUnits)
        ));
    }

    /// A stage listing itself as its own prerequisite: the one failure mode
    /// with no fixture before, because the old test set never built one.
    const SELF_DEPENDENT_STAGES: &[RecipeStage] = &[RecipeStage::new(
        RecipeStageId::PlanWorkUnits,
        RecipeStagePresence::Required,
        StageExecutionKind::PerWorkUnit,
        FileStage::Reading,
        &[RecipeStageId::PlanWorkUnits],
    )];

    #[test]
    fn recipe_check_rejects_a_stage_that_depends_on_itself() {
        assert!(matches!(
            Recipe::check(SELF_DEPENDENT_STAGES),
            RecipeCheck::SelfDependency(RecipeStageId::PlanWorkUnits)
        ));
    }

    /// `check` runs in a const context, which is what makes `Recipe::new`
    /// reject a bad recipe at COMPILE time rather than at some later runtime
    /// moment. If this stops compiling, the catalog has lost its compile-time
    /// guarantee and validation has silently become a runtime concern again.
    #[test]
    fn the_check_is_usable_in_a_const_context() {
        const VERDICT: RecipeCheck = Recipe::check(VALID_STAGES);
        assert!(matches!(VERDICT, RecipeCheck::Ok));
    }
}
