//! Typed planning layer from runner snapshots to immutable job plans.

use crate::api::DisplayPath;
use crate::command_model::{self, CatalogEntry};
use crate::recipe_runner::planner::{PlanningError, plan_work_units};
use crate::recipe_runner::runtime::discover_inputs_for_job;
use crate::recipe_runner::work_unit::{PlannedWorkUnit, TextWorkUnit};
use crate::store::RunnerJobSnapshot;

pub use crate::recipe_runner::materialize::PlannedArtifactSet;

/// Immutable execution input for one job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobPlan {
    /// Authoritative command model entry for the released command.
    pub spec: &'static CatalogEntry,
    /// Typed work units derived from the runner snapshot.
    pub work_units: Vec<PlannedWorkUnit>,
    /// Planned output artifacts for each work unit.
    pub artifacts: Vec<PlannedArtifactSet>,
}

/// Build one immutable job plan from the current runner snapshot.
pub fn build_job_plan(job: &RunnerJobSnapshot) -> Result<JobPlan, PlanningError> {
    let spec = command_model::command_spec(job.dispatch.command);
    let inputs = discover_inputs_for_job(job);
    let work_units = plan_work_units(spec.planner, &inputs)?;
    let output_policy = spec.selected_output_policy(&job.dispatch.options)?;
    let artifacts = work_units
        .iter()
        .map(|work_unit| {
            crate::recipe_runner::materialize::plan_materialized_files(
                primary_display_path_for_work_unit(work_unit),
                output_policy,
            )
        })
        .collect();

    Ok(JobPlan {
        spec,
        work_units,
        artifacts,
    })
}

/// Return the planned output artifacts for one source display path.
pub fn artifact_set_for_source<'a>(
    plan: &'a JobPlan,
    source_display_path: &DisplayPath,
) -> Option<&'a PlannedArtifactSet> {
    plan.artifacts
        .iter()
        .find(|artifacts| artifacts.source() == source_display_path)
}

const fn primary_display_path_for_work_unit(work_unit: &PlannedWorkUnit) -> &DisplayPath {
    match work_unit {
        PlannedWorkUnit::Text(TextWorkUnit { source })
        | PlannedWorkUnit::MediaAnalysis(
            crate::recipe_runner::work_unit::MediaAnalysisWorkUnit { source },
        ) => &source.display_path,
        PlannedWorkUnit::Audio(crate::recipe_runner::work_unit::AudioWorkUnit { audio }) => {
            &audio.display_path
        }
        // Read through the accessor: a benchmark unit's fields are private so
        // that only its planner can pair an audio file with a gold transcript.
        PlannedWorkUnit::Benchmark(benchmark) => &benchmark.audio().display_path,
        PlannedWorkUnit::Compare(crate::recipe_runner::work_unit::CompareWorkUnit {
            main, ..
        }) => &main.display_path,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::api::{CorrelationId, JobId, LanguageSpec, NumSpeakers, ReleasedCommand};
    use crate::options::{CommandOptions, CommonOptions, CompareOptions};
    use crate::store::{RunnerDispatchConfig, RunnerFilesystemConfig, RunnerJobIdentity};

    fn compare_snapshot(inputs: crate::store::JobInputs) -> RunnerJobSnapshot {
        crate::store::test_support::with_pending(RunnerJobSnapshot {
            identity: RunnerJobIdentity {
                job_id: JobId::from("job-plan"),
                correlation_id: CorrelationId::from("corr-plan"),
            },
            dispatch: RunnerDispatchConfig {
                command: ReleasedCommand::Compare,
                // Compare is a per-file command; no job-level language.
                lang: LanguageSpec::PerFile,
                num_speakers: NumSpeakers(1),
                options: CommandOptions::Compare(CompareOptions {
                    common: CommonOptions::default(),
                }),
                runtime_state: BTreeMap::new(),
                debug_traces: false,
            },
            filesystem: RunnerFilesystemConfig {
                inputs,
                staging_dir: batchalign_types::paths::ServerPath::new("/tmp/staging"),
                media_mapping: Default::default(),
                media_subdir: Default::default(),
                source_dir: batchalign_types::paths::ClientPath::new("/src"),
            },
            cancel_token: CancellationToken::new(),
            pending_files: Vec::new(),
        })
    }

    /// Guards: Compare planning selects reference projection, pairs main and gold sources, and derives the named CHAT and CSV outputs from that same main source.
    /// Kind: policy
    /// Not a type: Sealed layouts exclude missing or reversed roles, while reference-projection selection, nested suffixes and primary-first transport projection remain catalog and adapter semantics.
    /// State: None
    /// Concurrency: None
    #[test]
    fn compare_plan_uses_one_pair_and_compare_sidecar() {
        let plan = build_job_plan(&compare_snapshot(crate::store::JobInputs::Content(
            ["nested/sample.cha", "nested/sample.gold.cha"]
                .into_iter()
                .map(|name| crate::store::ContentInput {
                    display: name.into(),
                    kind: crate::store::InputKind::Chat,
                })
                .collect(),
        )))
        .expect("compare plan");
        assert_eq!(plan.spec.command, ReleasedCommand::Compare);
        assert!(
            matches!(plan.work_units.as_slice(), [PlannedWorkUnit::Compare(unit)]
            if unit.main.display_path.as_ref() == "nested/sample.cha"
                && unit.gold.display_path.as_ref() == "nested/sample.gold.cha")
        );
        assert_eq!(plan.artifacts.len(), 1);
        assert_eq!(
            *plan.artifacts[0].source(),
            DisplayPath::from("nested/sample.cha")
        );
        assert_eq!(
            plan.spec.family,
            crate::recipe_runner::command_spec::CommandFamily::ReferenceProjection
        );
        assert_eq!(
            plan.spec.recipe.mode,
            crate::recipe_runner::recipe::ExecutionMode::ReferenceProjection
        );
        let outputs = plan.artifacts[0]
            .chat_metrics()
            .expect("comparison output layout");
        assert_eq!(
            *outputs.primary().display_path(),
            DisplayPath::from("nested/sample.cha")
        );
        assert_eq!(
            outputs.primary().content_type(),
            crate::api::ContentType::Chat
        );
        assert_eq!(
            outputs.metrics().content_type(),
            crate::api::ContentType::Csv
        );
        assert_eq!(
            *outputs.metrics().display_path(),
            DisplayPath::from("nested/sample.compare.csv")
        );
        assert_eq!(
            plan.artifacts[0].clone().into_iter().collect::<Vec<_>>(),
            vec![
                crate::recipe_runner::materialize::PlannedMaterializedFile {
                    display_path: "nested/sample.cha".into(),
                    content_type: crate::api::ContentType::Chat,
                    role: crate::recipe_runner::materialize::MaterializedArtifactRole::Primary,
                },
                crate::recipe_runner::materialize::PlannedMaterializedFile {
                    display_path: "nested/sample.compare.csv".into(),
                    content_type: crate::api::ContentType::Csv,
                    role: crate::recipe_runner::materialize::MaterializedArtifactRole::Sidecar,
                },
            ]
        );
    }
}
