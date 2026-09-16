//! Which released commands a worker set can serve.
//!
//! The worker's report is admitted in [`crate::engine_reports`] (one typed
//! engine report per advertised task), where the pool records it. This module
//! owns the rule that turns an admitted report into command availability,
//! [`command_supported`], and applies it in both places that ask: the
//! advertised surface ([`WorkerCapabilitySnapshot::detected`], served by
//! `/health` and used to accept job submissions) and dispatch
//! (`runner::routing`).
//!
//! # Supporting a task is not knowing its engine
//!
//! A command is available when the worker advertises the command's primary
//! infer task. That alone decides advertisement and job acceptance, from any
//! admitted report, including one a lazily loading worker gave before it
//! loaded anything, so advertisement never withholds a command merely because
//! a model has not been loaded yet.
//!
//! Knowing which engine runs a task is a separate fact, read later. Only forced
//! alignment needs it before dispatch (its cache rows are namespaced by the FA
//! engine), and the forced-alignment dispatch arm reads it itself with
//! [`crate::engine_reports::FaCacheNamespace::from_loaded`], from the report
//! the pool took after loading FA on the selected worker. There is no second
//! declaration of which commands need an identity: the arm that reads the
//! namespace is the only place that says so.
//!
//! It does NOT depend on axum, sqlx, or any server-specific crate. Used by
//! both the direct-mode host ([`crate::worker_setup`]) and the HTTP server
//! ([`crate::state`]).

use tracing::warn;

use crate::api::ReleasedCommand;
use crate::command_model::{CapabilityPlan, CommandCapabilityKind, command_specs};
use crate::engine_reports::{EngineIdentityUnavailable, WorkerEngineReports};
use crate::worker::InferTask;

// ---------------------------------------------------------------------------
// The availability rule
// ---------------------------------------------------------------------------

/// The availability rule: the worker supports the command's primary infer
/// task.
///
/// Read from any admitted report, loaded or not, because support does not
/// depend on a model having been loaded.
pub(crate) fn command_supported(
    plan: &CapabilityPlan,
    reports: &WorkerEngineReports,
) -> Result<(), EngineIdentityUnavailable> {
    if reports.supports(plan.primary_infer_task) {
        Ok(())
    } else {
        Err(EngineIdentityUnavailable::NotSupported {
            task: plan.primary_infer_task,
        })
    }
}

/// The released commands a worker supports, in advertised order.
fn derive_command_capabilities(reports: &WorkerEngineReports) -> Vec<ReleasedCommand> {
    let mut derived = Vec::new();

    // Two passes, so server-composed commands are advertised after the
    // commands they are composed from. Within a pass the order is the catalog's
    // declaration order, which is itself the advertised order (see
    // `recipe_runner::catalog::COMMAND_SPECS`).
    for kind in [
        CommandCapabilityKind::DirectInfer,
        CommandCapabilityKind::ServerComposed,
    ] {
        for spec in command_specs()
            .iter()
            .filter(|spec| spec.capability_kind == kind)
        {
            if command_supported(&spec.capabilities, reports).is_ok()
                && !derived.contains(&spec.command)
            {
                derived.push(spec.command);
            }
        }
    }

    derived
}

// ---------------------------------------------------------------------------
// Capability snapshot
// ---------------------------------------------------------------------------

/// One resolved view of what a worker set can serve: the released commands
/// and the infer tasks behind them.
///
/// Both lists are derived at construction and private, so no caller can hand
/// in a command list that disagrees with the tasks, and there is exactly one
/// copy for the server's state, the direct host and `/health` to read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WorkerCapabilitySnapshot {
    commands: Vec<ReleasedCommand>,
    infer_tasks: Vec<InferTask>,
}

impl WorkerCapabilitySnapshot {
    /// Every released command, from every task a command is advertised from.
    ///
    /// The view for two situations that both lack a report to derive from:
    /// before any worker has answered (real capabilities are detected lazily
    /// on the first spawn, which avoids a slow probe worker at startup), and
    /// test-echo workers, which answer every task. Commands are listed by
    /// wire name, as `/health` has always listed this view.
    pub(crate) fn every_released_command() -> Self {
        let mut commands: Vec<ReleasedCommand> =
            command_specs().iter().map(|spec| spec.command).collect();
        commands.sort_by_key(|command| command.as_str());
        commands.dedup();
        let infer_tasks = command_specs()
            .iter()
            .map(|spec| spec.capabilities.primary_infer_task)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        Self {
            commands,
            infer_tasks,
        }
    }

    /// The view a live worker's admitted report supports.
    pub(crate) fn detected(reports: &WorkerEngineReports) -> Self {
        let commands = derive_command_capabilities(reports);
        let infer_tasks: Vec<InferTask> = reports.tasks().collect();
        if commands.is_empty() && !infer_tasks.is_empty() {
            warn!(
                infer_tasks = ?infer_tasks,
                "No released commands derived from the admitted infer-task set"
            );
        }
        Self {
            commands,
            infer_tasks,
        }
    }

    /// Choose the view: test-echo workers answer everything, a detected
    /// report is authoritative once one exists, and until then every released
    /// command is assumed.
    pub(crate) fn resolve(test_echo_mode: bool, detected: Option<&WorkerEngineReports>) -> Self {
        match (test_echo_mode, detected) {
            (true, _) | (false, None) => Self::every_released_command(),
            (false, Some(reports)) => Self::detected(reports),
        }
    }

    /// Released commands this worker set can serve.
    pub(crate) fn commands(&self) -> &[ReleasedCommand] {
        &self.commands
    }

    /// Whether this worker set can serve `command`.
    pub(crate) fn serves(&self, command: ReleasedCommand) -> bool {
        self.commands.contains(&command)
    }

    /// The served commands by wire name, for responses and error messages.
    pub(crate) fn command_names(&self) -> Vec<String> {
        self.commands
            .iter()
            .map(|command| command.as_str().to_owned())
            .collect()
    }

    /// Infer tasks behind those commands, in `InferTask` order.
    pub(crate) fn infer_tasks(&self) -> &[InferTask] {
        &self.infer_tasks
    }
}

#[cfg(test)]
mod tests {
    use super::{WorkerCapabilitySnapshot, command_supported};
    use crate::api::{ReleasedCommand, ReportedEngineName};
    use crate::command_model::command_spec;
    use crate::engine_reports::{EngineIdentityUnavailable, FaCacheNamespace, WorkerEngineReports};
    use crate::worker::InferTask;
    use crate::worker::pool::LoadedCapabilities;

    /// An admitted report: the advertised tasks plus one engine entry each.
    fn reports(entries: &[(InferTask, Option<&str>)]) -> WorkerEngineReports {
        WorkerEngineReports::admit(
            &entries.iter().map(|(task, _)| *task).collect::<Vec<_>>(),
            entries
                .iter()
                .map(|(task, name)| {
                    (
                        *task,
                        name.map(|name| ReportedEngineName::try_from(name).expect("valid name")),
                    )
                })
                .collect(),
        )
        .expect("complete report")
    }

    fn commands(snapshot: &WorkerCapabilitySnapshot) -> Vec<&str> {
        snapshot
            .commands()
            .iter()
            .map(|command| command.as_str())
            .collect()
    }

    #[test]
    fn no_infer_tasks_derive_no_commands() {
        assert!(
            WorkerCapabilitySnapshot::detected(&reports(&[]))
                .commands()
                .is_empty()
        );
    }

    #[test]
    fn released_commands_derive_from_infer_tasks_in_advertised_order() {
        let snapshot = WorkerCapabilitySnapshot::detected(&reports(&[
            (InferTask::Morphosyntax, None),
            (InferTask::Utseg, None),
            (InferTask::Translate, None),
            (InferTask::Coref, None),
            (InferTask::Fa, Some("whisper-fa-v1")),
            (InferTask::Opensmile, None),
            (InferTask::Avqi, None),
        ]));
        assert_eq!(
            commands(&snapshot),
            vec![
                "morphotag",
                "utseg",
                "translate",
                "coref",
                "align",
                "compare",
                "opensmile",
                "avqi",
            ]
        );
    }

    /// The lazy-daemon case. A registry daemon that loads models on demand is
    /// probed before it has loaded FA, so its first report SUPPORTS FA but
    /// names no FA engine. It still advertises and accepts `align`. At
    /// dispatch the pool loads FA and reads again: a named engine resolves the
    /// namespace, and only an engine still unnamed after that load refuses.
    #[test]
    fn a_lazy_daemon_probed_before_fa_loads_advertises_align_and_refuses_only_if_unnamed_after_load()
     {
        let align = &command_spec(ReleasedCommand::Align).capabilities;

        let before_load = reports(&[(InferTask::Fa, None), (InferTask::Asr, None)]);
        assert!(WorkerCapabilitySnapshot::detected(&before_load).serves(ReleasedCommand::Align));
        assert_eq!(command_supported(align, &before_load), Ok(()));

        let named_after_load = LoadedCapabilities::for_test(
            InferTask::Fa,
            reports(&[(InferTask::Fa, Some("wave2vec-fa-v1"))]),
        );
        assert_eq!(command_supported(align, named_after_load.reports()), Ok(()));
        assert_eq!(
            FaCacheNamespace::from_loaded(&named_after_load),
            Ok(FaCacheNamespace::for_test("wave2vec-fa-v1"))
        );

        let unnamed_after_load =
            LoadedCapabilities::for_test(InferTask::Fa, reports(&[(InferTask::Fa, None)]));
        assert_eq!(
            FaCacheNamespace::from_loaded(&unnamed_after_load),
            Err(EngineIdentityUnavailable::UnreportedAfterLoad {
                task: InferTask::Fa
            })
        );
    }

    /// A command whose task is not supported is neither advertised nor
    /// accepted, whatever else the worker names.
    #[test]
    fn an_unsupported_task_is_neither_advertised_nor_accepted() {
        let align = &command_spec(ReleasedCommand::Align).capabilities;
        let no_fa = reports(&[(InferTask::Morphosyntax, None)]);
        assert!(!WorkerCapabilitySnapshot::detected(&no_fa).serves(ReleasedCommand::Align));
        assert_eq!(
            command_supported(align, &no_fa),
            Err(EngineIdentityUnavailable::NotSupported {
                task: InferTask::Fa
            })
        );
    }

    #[test]
    fn server_owned_asr_commands_are_synthesized_after_direct_ones() {
        let snapshot = WorkerCapabilitySnapshot::detected(&reports(&[(InferTask::Asr, None)]));
        assert_eq!(
            commands(&snapshot),
            vec!["transcribe", "transcribe_s", "benchmark"]
        );
    }

    #[test]
    fn test_echo_and_an_unprobed_pool_assume_every_released_command() {
        let every = WorkerCapabilitySnapshot::every_released_command();
        assert_eq!(every.commands().len(), ReleasedCommand::ALL.len());
        let names = every.command_names();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(
            names, sorted,
            "the assumed view lists commands by wire name"
        );
        assert!(every.infer_tasks().contains(&InferTask::Fa));
        let detected = reports(&[(InferTask::Morphosyntax, None)]);
        assert_eq!(
            WorkerCapabilitySnapshot::resolve(true, Some(&detected)),
            every
        );
        assert_eq!(WorkerCapabilitySnapshot::resolve(false, None), every);
    }

    #[test]
    fn a_detected_report_replaces_the_assumed_view_even_when_empty() {
        let detected = reports(&[(InferTask::Morphosyntax, None), (InferTask::Utseg, None)]);
        let snapshot = WorkerCapabilitySnapshot::resolve(false, Some(&detected));
        assert_eq!(commands(&snapshot), vec!["morphotag", "utseg", "compare"]);
        assert_eq!(
            snapshot.infer_tasks(),
            &[InferTask::Morphosyntax, InferTask::Utseg]
        );

        let empty = WorkerCapabilitySnapshot::resolve(false, Some(&reports(&[])));
        assert!(empty.commands().is_empty());
        assert!(empty.infer_tasks().is_empty());
    }
}
