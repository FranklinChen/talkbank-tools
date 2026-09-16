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
    BatchingPolicy, CapabilityPlan, CatalogEntry, CommandCapabilityKind, ConstrainedHostPolicy,
    ModelSharingPolicy, ParallelismPolicy, ResourceLane, RunnerDispatchKind, SchedulingPolicy,
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

/// Return whether one closed released command requires shared-filesystem audio access.
pub fn released_command_uses_local_audio(command: ReleasedCommand) -> bool {
    command_spec(command).io_profile.uses_local_audio()
}

/// Return whether one released command requires shared-filesystem audio access.
///
/// An unrecognised name is not an audio command, which is the conservative
/// answer: it keeps the CLI from sending paths for something the server may not
/// be able to resolve locally.
pub fn command_uses_local_audio(command: &str) -> bool {
    match ReleasedCommand::try_from(command) {
        Ok(command) => released_command_uses_local_audio(command),
        Err(_) => false,
    }
}

/// Return whether a command may use `paths_mode`: that is, have the CLI
/// send filesystem paths instead of file content when submitting to a
/// local daemon. A superset of `released_command_uses_local_audio`:
/// every audio command supports paths_mode, and text commands
/// (morphotag, utseg, translate, coref, compare) also opt in because
/// the server-side runner already reads their input CHAT by path.
pub fn released_command_supports_paths_mode(command: ReleasedCommand) -> bool {
    command_spec(command).io_profile.supports_paths_mode()
}

/// Return the runner dispatch kind for one released command.
///
/// Total, not optional: `ReleasedCommand` is a closed enum and
/// `catalog::tests::every_released_command_has_a_spec` pins full coverage, so
/// there is no "unknown command" case for a caller to handle. The previous
/// `Option` return was a sentinel for a state the type system already excludes.
pub(crate) fn command_runner_dispatch_kind(command: ReleasedCommand) -> RunnerDispatchKind {
    command_spec(command).runner_dispatch_kind
}
