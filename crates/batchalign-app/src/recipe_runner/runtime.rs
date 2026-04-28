//! Live runtime adapters for the recipe-runner spike.
//!
//! These helpers let the current runner reuse recipe-runner planning and output
//! metadata without a flag-day dispatcher rewrite.

use std::path::{Path, PathBuf};

use crate::api::{DisplayPath, ReleasedCommand};
use crate::command_model;
use crate::store::{PendingJobFile, RunnerFilesystemConfig, RunnerJobSnapshot};

use super::materialize::{
    MaterializedArtifactRole, PlannedMaterializedFile, plan_materialized_files,
};
use super::planner::{PlanningError, plan_work_units};
use super::work_unit::{DiscoveredInput, PlannedWorkUnit};

/// Derive one discovered input from the runner's current filesystem layout.
pub(crate) fn discover_input_for_pending_file(
    filesystem: &RunnerFilesystemConfig,
    file: &PendingJobFile,
) -> DiscoveredInput {
    let source_path = if filesystem.paths_mode && file.file_index < filesystem.source_paths.len() {
        filesystem.source_paths[file.file_index]
            .assume_shared_filesystem()
            .as_path()
            .to_owned()
    } else {
        filesystem
            .staging_dir
            .join("input")
            .join(file.filename.as_ref())
            .as_path()
            .to_owned()
    };
    let before_path = (!filesystem.before_paths.is_empty()
        && file.file_index < filesystem.before_paths.len())
    .then(|| {
        filesystem.before_paths[file.file_index]
            .assume_shared_filesystem()
            .as_path()
            .to_owned()
    });
    DiscoveredInput {
        display_path: file.filename.clone(),
        source_path,
        before_path,
    }
}

/// Derive all discovered inputs for one runner job.
pub(crate) fn discover_inputs_for_job(job: &RunnerJobSnapshot) -> Vec<DiscoveredInput> {
    job.pending_files
        .iter()
        .map(|file| discover_input_for_pending_file(&job.filesystem, file))
        .collect()
}

/// Plan command-family-specific work units from the runner snapshot.
pub(crate) fn plan_work_units_for_job(
    command: ReleasedCommand,
    job: &RunnerJobSnapshot,
) -> Result<Vec<PlannedWorkUnit>, PlanningError> {
    let inputs = discover_inputs_for_job(job);
    plan_work_units(command_model::command_spec(command).planner, &inputs)
}

/// Plan all materialized artifacts for one released command and source path.
pub(crate) fn planned_output_artifacts(
    command: ReleasedCommand,
    source_display_path: &DisplayPath,
) -> Vec<PlannedMaterializedFile> {
    plan_materialized_files(
        source_display_path,
        command_model::command_spec(command).output_policy,
    )
}

/// Return the primary result artifact for one command/input pair.
pub(crate) fn primary_output_artifact(
    command: ReleasedCommand,
    source_display_path: &DisplayPath,
) -> PlannedMaterializedFile {
    planned_output_artifacts(command, source_display_path)
        .into_iter()
        .find(|artifact| artifact.role == MaterializedArtifactRole::Primary)
        .expect("recipe runner output policy must include a primary artifact")
}

/// Return all sidecar artifacts for one command/input pair.
pub(crate) fn sidecar_output_artifacts(
    command: ReleasedCommand,
    source_display_path: &DisplayPath,
) -> Vec<PlannedMaterializedFile> {
    planned_output_artifacts(command, source_display_path)
        .into_iter()
        .filter(|artifact| artifact.role == MaterializedArtifactRole::Sidecar)
        .collect()
}

/// Derive only the primary result display path for one command/input pair.
pub(crate) fn result_display_path_for_command(
    command: ReleasedCommand,
    filename: &str,
) -> DisplayPath {
    primary_output_artifact(command, &DisplayPath::from(filename)).display_path
}

/// Resolve the concrete write path for one planned output artifact.
pub(crate) fn output_write_path(
    filesystem: &RunnerFilesystemConfig,
    file_index: usize,
    result_display_path: &DisplayPath,
) -> PathBuf {
    if filesystem.paths_mode && file_index < filesystem.output_paths.len() {
        let result_name = Path::new(result_display_path.as_ref())
            .file_name()
            .unwrap_or_default();
        let output_server_path = filesystem.output_paths[file_index].assume_shared_filesystem();
        Path::new(output_server_path.as_str())
            .parent()
            .map(|parent| parent.join(result_name))
            .unwrap_or_else(|| result_name.into())
    } else {
        filesystem
            .staging_dir
            .join("output")
            .join(result_display_path.as_ref())
            .as_path()
            .to_owned()
    }
}

fn staged_output_path(
    filesystem: &RunnerFilesystemConfig,
    result_display_path: &DisplayPath,
) -> PathBuf {
    filesystem
        .staging_dir
        .join("output")
        .join(result_display_path.as_ref())
        .as_path()
        .to_owned()
}

async fn write_text_file(path: &Path, content: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(path, content).await
}

/// Write a text result to the command's primary target and to staged output.
///
/// Paths-mode jobs still write to the execution host's requested output path,
/// but they also preserve a staged copy so the submitting CLI can download and
/// write the same result back to its own local filesystem.
pub(crate) async fn write_text_output_artifact(
    filesystem: &RunnerFilesystemConfig,
    file_index: usize,
    result_display_path: &DisplayPath,
    content: &str,
) -> std::io::Result<()> {
    let write_path = output_write_path(filesystem, file_index, result_display_path);
    write_text_file(&write_path, content).await?;

    let staged_path = staged_output_path(filesystem, result_display_path);
    if staged_path != write_path {
        write_text_file(&staged_path, content).await?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::api::{CorrelationId, JobId, LanguageSpec, NumSpeakers};
    use crate::options::{CommandOptions, CommonOptions, CompareOptions};
    use crate::store::{
        PendingJobFile, RunnerDispatchConfig, RunnerFilesystemConfig, RunnerJobIdentity,
        RunnerJobSnapshot,
    };

    fn runner_snapshot(
        command: ReleasedCommand,
        pending_files: Vec<PendingJobFile>,
        source_paths: Vec<&str>,
        before_paths: Vec<&str>,
    ) -> RunnerJobSnapshot {
        RunnerJobSnapshot {
            identity: RunnerJobIdentity {
                job_id: JobId::from("job-runtime"),
                correlation_id: CorrelationId::from("corr-runtime"),
            },
            dispatch: RunnerDispatchConfig {
                command,
                lang: LanguageSpec::Resolved(crate::api::LanguageCode3::eng()),
                num_speakers: NumSpeakers(1),
                options: CommandOptions::Compare(CompareOptions {
                    common: CommonOptions::default(),
                    merge_abbrev: false.into(),
                }),
                runtime_state: BTreeMap::new(),
                debug_traces: false,
            },
            filesystem: RunnerFilesystemConfig {
                paths_mode: true,
                source_paths: source_paths
                    .into_iter()
                    .map(batchalign_types::paths::ClientPath::from)
                    .collect(),
                output_paths: vec![
                    batchalign_types::paths::ClientPath::new("/out/main.cha");
                    pending_files.len()
                ],
                before_paths: before_paths
                    .into_iter()
                    .map(batchalign_types::paths::ClientPath::from)
                    .collect(),
                staging_dir: batchalign_types::paths::ServerPath::new("/tmp/staging"),
                media_mapping: Default::default(),
                media_subdir: Default::default(),
                source_dir: batchalign_types::paths::ClientPath::new("/source"),
            },
            cancel_token: CancellationToken::new(),
            pending_files,
        }
    }

    #[test]
    fn discover_input_carries_before_path() {
        let file = PendingJobFile {
            file_index: 0,
            filename: DisplayPath::from("nested/sample.cha"),
            has_chat: true,
        };
        let job = runner_snapshot(
            ReleasedCommand::Morphotag,
            vec![file.clone()],
            vec!["/src/nested/sample.cha"],
            vec!["/before/nested/sample.cha"],
        );
        let input = discover_input_for_pending_file(&job.filesystem, &file);
        assert_eq!(input.source_path, PathBuf::from("/src/nested/sample.cha"));
        assert_eq!(
            input.before_path,
            Some(PathBuf::from("/before/nested/sample.cha"))
        );
    }

    #[test]
    fn compare_job_planning_uses_gold_companion_shape() {
        let job = runner_snapshot(
            ReleasedCommand::Compare,
            vec![
                PendingJobFile {
                    file_index: 0,
                    filename: DisplayPath::from("nested/sample.cha"),
                    has_chat: true,
                },
                PendingJobFile {
                    file_index: 1,
                    filename: DisplayPath::from("nested/sample.gold.cha"),
                    has_chat: true,
                },
            ],
            vec!["/src/nested/sample.cha", "/src/nested/sample.gold.cha"],
            vec![],
        );
        let planned = plan_work_units_for_job(ReleasedCommand::Compare, &job).expect("planned");
        assert_eq!(planned.len(), 1);
        let PlannedWorkUnit::Compare(unit) = &planned[0] else {
            panic!("expected compare unit");
        };
        assert_eq!(
            unit.main.display_path,
            DisplayPath::from("nested/sample.cha")
        );
        assert_eq!(
            unit.gold.display_path,
            DisplayPath::from("nested/sample.gold.cha")
        );
    }

    #[test]
    fn result_display_paths_follow_recipe_catalog() {
        assert_eq!(
            result_display_path_for_command(ReleasedCommand::Transcribe, "sub/nested.wav"),
            DisplayPath::from("sub/nested.cha")
        );
        assert_eq!(
            result_display_path_for_command(ReleasedCommand::Benchmark, "sub/nested.wav"),
            DisplayPath::from("sub/nested.cha")
        );
        assert_eq!(
            result_display_path_for_command(ReleasedCommand::Opensmile, "sub/nested.wav"),
            DisplayPath::from("sub/nested.opensmile.csv")
        );
        assert_eq!(
            result_display_path_for_command(ReleasedCommand::Avqi, "sub/nested.cs.wav"),
            DisplayPath::from("sub/nested.avqi.txt")
        );
    }

    #[test]
    fn compare_sidecars_follow_recipe_catalog() {
        let sidecars =
            sidecar_output_artifacts(ReleasedCommand::Compare, &DisplayPath::from("a/b.cha"));
        assert_eq!(sidecars.len(), 1);
        assert_eq!(
            sidecars[0].display_path,
            DisplayPath::from("a/b.compare.csv")
        );
    }

    #[test]
    fn output_write_path_preserves_staging_relative_dirs() {
        let filesystem = RunnerFilesystemConfig {
            paths_mode: false,
            source_paths: Vec::new(),
            output_paths: Vec::new(),
            before_paths: Vec::new(),
            staging_dir: batchalign_types::paths::ServerPath::new("/tmp/staging"),
            media_mapping: Default::default(),
            media_subdir: Default::default(),
            source_dir: batchalign_types::paths::ClientPath::new("/source"),
        };
        let write_path = output_write_path(
            &filesystem,
            0,
            &DisplayPath::from("nested/result.compare.csv"),
        );
        assert_eq!(
            write_path,
            PathBuf::from("/tmp/staging/output/nested/result.compare.csv")
        );
    }

    #[tokio::test]
    async fn write_text_output_artifact_keeps_staged_copy_for_paths_mode() {
        let root = tempfile::tempdir().expect("tempdir");
        let host_out = root.path().join("host/output/sample.cha");
        let filesystem = RunnerFilesystemConfig {
            paths_mode: true,
            source_paths: vec![batchalign_types::paths::ClientPath::new(
                root.path()
                    .join("input/sample.cha")
                    .to_string_lossy()
                    .to_string(),
            )],
            output_paths: vec![batchalign_types::paths::ClientPath::new(
                host_out.to_string_lossy().to_string(),
            )],
            before_paths: Vec::new(),
            staging_dir: batchalign_types::paths::ServerPath::from(root.path().join("jobs/job-1")),
            media_mapping: Default::default(),
            media_subdir: Default::default(),
            source_dir: batchalign_types::paths::ClientPath::new(
                root.path().join("input").to_string_lossy().to_string(),
            ),
        };

        write_text_output_artifact(
            &filesystem,
            0,
            &DisplayPath::from("nested/sample.cha"),
            "@Begin\n@End\n",
        )
        .await
        .expect("write output");

        assert_eq!(
            std::fs::read_to_string(&host_out).expect("host output"),
            "@Begin\n@End\n"
        );
        assert_eq!(
            std::fs::read_to_string(
                filesystem
                    .staging_dir
                    .join("output")
                    .join("nested/sample.cha")
            )
            .expect("staged output"),
            "@Begin\n@End\n"
        );
    }
}
