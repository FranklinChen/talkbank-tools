//! Source-bound benchmark output admission and successful-write completion.

use crate::api::ReleasedCommand;
use crate::benchmark::BenchmarkOutputs;
use crate::error::ServerError;
use crate::recipe_runner::materialize::PlannedMaterializedFile;
use crate::recipe_runner::runtime::{
    ChatMetricsOutputs, SourceOutputs, write_text_output_artifact,
};
use crate::recipe_runner::work_unit::BenchmarkWorkUnit;
use crate::runner::util::{
    FileRunTracker, FileTaskOutcome, RunnerEventSink, classify_server_error,
};
use crate::store::{PendingJobFile, RunnerFilesystemConfig};

/// Catalog-derived destinations, bound to the file whose completion they own.
pub(super) struct BenchmarkOutputPlan<'a> {
    file: &'a PendingJobFile,
    sink: &'a dyn RunnerEventSink,
    outputs: ChatMetricsOutputs<'a>,
}

impl<'a> BenchmarkOutputPlan<'a> {
    pub(super) fn admit(
        filesystem: &'a RunnerFilesystemConfig,
        file: &'a PendingJobFile,
        unit: &BenchmarkWorkUnit,
        options: &crate::options::CommandOptions,
        sink: &'a dyn RunnerEventSink,
    ) -> Result<Self, ServerError> {
        if *file.filename() != unit.audio().display_path {
            return Err(ServerError::Validation(
                "benchmark output plan belongs to a different source".into(),
            ));
        }
        let outputs = SourceOutputs::plan(filesystem, file, ReleasedCommand::Benchmark, options)?
            .into_chat_metrics()?;
        Ok(Self {
            file,
            sink,
            outputs,
        })
    }

    /// Persistence failures are terminal; they must not repeat paid inference.
    pub(super) async fn persist(self, outputs: BenchmarkOutputs) -> FileTaskOutcome {
        let lifecycle = FileRunTracker::new(self.sink, self.file.filename().as_ref());
        match self.write(outputs).await {
            Ok(written) => written.complete().await,
            Err(error) => lifecycle
                .fail(&error.to_string(), classify_server_error(&error))
                .await
                .into(),
        }
    }

    async fn write(self, outputs: BenchmarkOutputs) -> Result<WrittenBenchmark<'a>, ServerError> {
        let metrics_csv = outputs.metrics.to_csv_string().map_err(|error| {
            ServerError::Persistence(format!("benchmark CSV serialization failed: {error}"))
        })?;
        let artifacts = self.outputs.artifacts();
        let targets = self.outputs.targets()?;
        write_text_output_artifact(targets.primary, outputs.annotated_main_chat.as_str())
            .await
            .map_err(|error| {
                ServerError::Persistence(format!(
                    "failed to write benchmark CHAT output {}: {error}",
                    artifacts.primary().display_path()
                ))
            })?;
        write_text_output_artifact(targets.metrics, &metrics_csv)
            .await
            .map_err(|error| {
                ServerError::Persistence(format!(
                    "failed to write benchmark CSV output {}: {error}",
                    artifacts.metrics().display_path()
                ))
            })?;
        Ok(WrittenBenchmark {
            file: self.file,
            sink: self.sink,
            primary: self.outputs.into_primary(),
        })
    }
}

/// Only the successful writer creates this capability; no completion method
/// accepts a caller-selected result filename or an unchecked write claim.
struct WrittenBenchmark<'a> {
    file: &'a PendingJobFile,
    sink: &'a dyn RunnerEventSink,
    primary: PlannedMaterializedFile,
}

impl WrittenBenchmark<'_> {
    async fn complete(self) -> FileTaskOutcome {
        FileRunTracker::new(self.sink, self.file.filename().as_ref())
            .complete_with_result(self.primary.display_path, self.primary.content_type)
            .await
            .into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::post_validate::PostValidated;
    use crate::recipe_runner::command_spec::PlannerKind;
    use crate::recipe_runner::planner::plan_work_units;
    use crate::recipe_runner::work_unit::{DiscoveredInput, PlannedWorkUnit};
    use crate::runner::util::test_sink::RecordingSink;
    use batchalign_transform::compare::{
        CompareMetricName, CompareMetricValue, CompareMetricsCsvRow, CompareMetricsCsvTable,
    };
    use batchalign_types::paths::{ClientPath, ServerPath};

    const VALID: &str = "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\thello there .\n@End\n";

    fn outputs() -> BenchmarkOutputs {
        crate::compare::AdmittedComparisonReference::admit(VALID).expect("valid boundary fixture");
        BenchmarkOutputs {
            annotated_main_chat: PostValidated::for_test(VALID, ReleasedCommand::Benchmark),
            metrics: CompareMetricsCsvTable {
                rows: vec![CompareMetricsCsvRow {
                    metric: CompareMetricName::Wer,
                    value: CompareMetricValue::Decimal(0.25),
                }],
            },
        }
    }

    fn unit(source: &std::path::Path) -> BenchmarkWorkUnit {
        let units = plan_work_units(
            PlannerKind::BenchmarkPairs,
            &[DiscoveredInput::new("nested/clip.wav", source)],
        )
        .expect("benchmark planning");
        match units.into_iter().next().expect("one unit") {
            PlannedWorkUnit::Benchmark(unit) => unit,
            _ => panic!("expected benchmark unit"),
        }
    }

    /// The one-recording benchmark job: paths mode (read in place, written
    /// to `requested/`) or staged, with `display` as its name.
    enum FixtureMode {
        Paths,
        Staged,
    }

    fn filesystem_named(
        root: &std::path::Path,
        mode: &FixtureMode,
        display: &str,
    ) -> RunnerFilesystemConfig {
        let inputs = match mode {
            FixtureMode::Paths => {
                crate::store::test_support::paths_inputs(vec![crate::store::PathsInput {
                    display: display.into(),
                    kind: crate::store::InputKind::Media,
                    source: crate::store::test_support::admitted(
                        root.join("input/clip.wav").to_str().unwrap(),
                    ),
                    output: crate::store::test_support::admitted(
                        root.join("requested/clip.cha").to_str().unwrap(),
                    ),
                    before: None,
                }])
            }
            FixtureMode::Staged => {
                crate::store::JobInputs::Content(vec![crate::store::ContentInput {
                    display: display.into(),
                    kind: crate::store::InputKind::Media,
                }])
            }
        };
        RunnerFilesystemConfig {
            inputs,
            staging_dir: ServerPath::new(root.join("staging")),
            media_mapping: Default::default(),
            media_subdir: Default::default(),
            source_dir: ClientPath::new(root.join("input").to_string_lossy()),
        }
    }

    fn filesystem(root: &std::path::Path) -> RunnerFilesystemConfig {
        filesystem_named(root, &FixtureMode::Paths, "nested/clip.wav")
    }

    /// The job's one pending file, as its own inputs give it.
    fn file_of(filesystem: &RunnerFilesystemConfig) -> PendingJobFile {
        crate::store::test_support::pending_files(&filesystem.inputs).remove(0)
    }

    /// Guards: Wrong command options report the precise benchmark planning mismatch before publication.
    /// Kind: contract
    /// Not a type: A well-formed options enum cannot prove semantic agreement with a separately received command at this admission boundary.
    /// State: The isolated root supplies admitted source/destination identities and output-directory absence checks. The recording sink observes the absence of completion and attempts after refusal. No publication, worker or registry runs.
    /// Concurrency: The test owns its paths and recording sink. Constructor refusal finishes before directory-presence assertions and root cleanup.
    #[test]
    fn benchmark_option_mismatch_is_planning_failure_before_any_write() {
        let temp = tempfile::tempdir().expect("temporary directory");
        let filesystem = filesystem(temp.path());
        let unit = unit(&temp.path().join("input/clip.wav"));
        let file = file_of(&filesystem);
        let sink = RecordingSink::default();
        let options = crate::recipe_runner::runtime::test_options(ReleasedCommand::Align);
        let Err(error) = BenchmarkOutputPlan::admit(&filesystem, &file, &unit, &options, &sink)
        else {
            panic!("another command's options cannot admit benchmark output");
        };
        assert!(matches!(
            error,
            ServerError::Planning(crate::error::PlanningError::OutputOptionsMismatch {
                expected: ReleasedCommand::Benchmark,
                observed: ReleasedCommand::Align,
            })
        ));
        assert!(sink.completed_files().is_empty());
        assert!(sink.attempts().is_empty());
        assert!(!temp.path().join("requested").exists());
        assert!(!temp.path().join("staging").exists());
    }

    /// Guards: Every required primary and staged CHAT and CSV write must succeed before completion is recorded.
    /// Kind: contract
    /// Not a type: Admitted output plans cannot prevent later filesystem obstructions at any required artifact destination.
    /// State: Each isolated root supplies a directory obstructing one required artifact destination. Actual publication attempts exercise filesystem failure. The recording sink observes terminal error and the absence of completion or inference retry. No worker or registry runs.
    /// Concurrency: Each obstruction case owns its root and recording sink. Persistence returns before error observations and root cleanup.
    #[tokio::test]
    async fn benchmark_completion_requires_every_requested_and_staged_write() {
        for blocked in [
            "requested/clip.cha",
            "staging/output/nested/clip.cha",
            "requested/clip.compare.csv",
            "staging/output/nested/clip.compare.csv",
        ] {
            let temp = tempfile::tempdir().expect("temporary directory");
            std::fs::create_dir_all(temp.path().join(blocked))
                .expect("block destination with directory");
            let filesystem = filesystem(temp.path());
            let unit = unit(&temp.path().join("input/clip.wav"));
            let file = file_of(&filesystem);
            let sink = RecordingSink::default();
            let plan = BenchmarkOutputPlan::admit(
                &filesystem,
                &file,
                &unit,
                &crate::recipe_runner::runtime::test_options(ReleasedCommand::Benchmark),
                &sink,
            )
            .expect("admitted output plan");
            assert!(matches!(
                plan.persist(outputs()).await,
                FileTaskOutcome::TerminalStateRecorded(_)
            ));
            assert!(
                sink.completed_files().is_empty(),
                "failed write reported success: {blocked}"
            );
            let errors = sink.errors();
            assert_eq!(errors.len(), 1, "one terminal error: {blocked}");
            assert_eq!(errors[0].filename, "nested/clip.wav");
            assert_eq!(
                errors[0].category,
                crate::scheduling::FailureCategory::System
            );
            assert!(errors[0].error.contains("failed to write benchmark"));
            assert!(
                sink.attempts().is_empty(),
                "persistence must not retry inference"
            );
        }
    }

    /// Guards: Both input modes publish exact CHAT and CSV pairs and paths mode keeps requested copies.
    /// Kind: contract
    /// Not a type: Closed fixture modes prevent ignored path payloads but cannot establish actual serialization and publication at each destination.
    /// State: An isolated root supplies actual requested/staged CHAT and CSV writes. The source-bound plan and native synthetic output proof supply metadata and bytes. The recording sink observes completion without a worker or registry.
    /// Concurrency: Each input-mode case owns every destination and its recording sink, and awaits publication before reading bytes or dropping its root.
    #[tokio::test]
    async fn benchmark_writes_complete_pairs_in_paths_and_staged_modes() {
        for mode in [FixtureMode::Paths, FixtureMode::Staged] {
            let temp = tempfile::tempdir().expect("temporary directory");
            let filesystem = filesystem_named(temp.path(), &mode, "nested/clip.wav");
            let unit = unit(&temp.path().join("input/clip.wav"));
            let file = file_of(&filesystem);
            let sink = RecordingSink::default();
            let outputs = outputs();
            let expected_csv = outputs.metrics.to_csv_string().expect("CSV");
            let plan = BenchmarkOutputPlan::admit(
                &filesystem,
                &file,
                &unit,
                &crate::recipe_runner::runtime::test_options(ReleasedCommand::Benchmark),
                &sink,
            )
            .expect("admitted output plan");
            plan.persist(outputs).await;
            assert!(sink.errors().is_empty());
            assert_eq!(sink.completed_files(), vec!["nested/clip.wav"]);
            assert_eq!(
                std::fs::read_to_string(temp.path().join("staging/output/nested/clip.cha"))
                    .expect("staged CHAT"),
                VALID
            );
            assert_eq!(
                std::fs::read_to_string(temp.path().join("staging/output/nested/clip.compare.csv"))
                    .expect("staged CSV"),
                expected_csv
            );
            if matches!(mode, FixtureMode::Paths) {
                assert_eq!(
                    std::fs::read_to_string(temp.path().join("requested/clip.cha"))
                        .expect("requested CHAT"),
                    VALID
                );
                assert_eq!(
                    std::fs::read_to_string(temp.path().join("requested/clip.compare.csv"))
                        .expect("requested CSV"),
                    expected_csv
                );
            }
        }
    }

    /// Guards: An output plan refuses a well-formed pending file belonging to a different source name.
    /// Kind: contract
    /// Not a type: Per-file input admission cannot establish semantic agreement with an independently received benchmark work unit.
    /// State: The isolated root supplies admitted file identities for a different pending source. The recording sink observes no completion after the constructor refuses the independently planned work unit. No publication, worker or registry runs.
    /// Concurrency: The test owns all declaration values, paths and its recording sink. Refusal completes before the no-output assertions and root cleanup.
    #[test]
    fn benchmark_output_plan_rejects_a_different_source_file() {
        let temp = tempfile::tempdir().expect("temporary directory");
        let filesystem = filesystem_named(temp.path(), &FixtureMode::Paths, "other.wav");
        let unit = unit(&temp.path().join("input/clip.wav"));
        let file = file_of(&filesystem);
        let sink = RecordingSink::default();
        assert!(matches!(
            BenchmarkOutputPlan::admit(
                &filesystem,
                &file,
                &unit,
                &crate::recipe_runner::runtime::test_options(ReleasedCommand::Benchmark),
                &sink
            ),
            Err(ServerError::Validation(_))
        ));
        assert!(!temp.path().join("requested").exists());
        assert!(!temp.path().join("staging").exists());
        assert!(sink.completed_files().is_empty());
    }
}
