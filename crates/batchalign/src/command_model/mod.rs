//! The authoritative command model for released processing commands.
//!
//! One catalog, one lookup, one import surface. Everything the rest of the crate
//! needs to know about a released command is a field of the `CatalogEntry`
//! declared in `recipe_runner::catalog`, or a `const fn` derived from its
//! family; this module is how the rest of the crate reaches it.
//!
//! Until 2026-07-29 there were three layers here: this module, a
//! `commands::catalog` of one-line delegations, and a pair of compatibility view
//! types (`CommandDefinition` / `CommandWorkflowDescriptor`) rebuilt on every
//! read from a mixture of declared fields and `match` arms. All three collapsed
//! into the declaration itself, which is what `command_model`'s original doc
//! comment set out to do: "so the rest of the app stops choosing between
//! parallel command catalogs."

mod catalog;

pub(crate) use crate::recipe_runner::command_spec::{
    BatchingPolicy, CapabilityPlan, CatalogEntry, CommandCapabilityKind, CommandIoProfile,
    ConstrainedHostPolicy, ModelSharingPolicy, ParallelismPolicy, ResourceLane, RunnerDispatchKind,
    SchedulingPolicy,
};
#[allow(unused_imports)]
pub(crate) use crate::recipe_runner::materialize::{
    FileNamingPolicy, MaterializedArtifactRole, OutputPolicy, PlannedMaterializedFile,
    SidecarPolicy, StemRewrite,
};
#[allow(unused_imports)]
pub(crate) use crate::recipe_runner::recipe::{
    ExecutionMode, Recipe, RecipeStage, RecipeStageId, RecipeStagePresence, StageExecutionKind,
};

pub(crate) use catalog::{command_spec, command_specs, commands_stamped_by};

use crate::ReleasedCommand;

/// Return the runner dispatch kind for one released command.
///
/// Total, not optional: `ReleasedCommand` is a closed enum and
/// `catalog::tests::every_released_command_has_a_spec` pins full coverage, so
/// there is no "unknown command" case for a caller to handle. The previous
/// `Option` return was a sentinel for a state the type system already excludes.
pub(crate) fn command_runner_dispatch_kind(command: ReleasedCommand) -> RunnerDispatchKind {
    command_spec(command).runner_dispatch_kind
}
