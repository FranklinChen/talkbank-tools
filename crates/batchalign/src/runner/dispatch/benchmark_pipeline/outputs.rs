//! Source-bound benchmark output admission and successful-write completion.

use crate::api::{JobId, ReleasedCommand};
use crate::benchmark::BenchmarkOutputs;
use crate::error::ServerError;
use crate::recipe_runner::materialize::{MaterializedArtifactRole, PlannedMaterializedFile};
use crate::recipe_runner::runtime::{
    ChatOutputTarget, planned_output_artifacts, write_text_output_artifact,
};
use crate::recipe_runner::work_unit::BenchmarkWorkUnit;
use crate::runner::util::{
    FileRunTracker, FileTaskOutcome, RunnerEventSink, classify_server_error,
};
use crate::store::{PendingJobFile, RunnerFilesystemConfig};

/// Catalog-derived destinations, bound to the file whose completion they own.
pub(super) struct BenchmarkOutputPlan<'a> {
    filesystem: &'a RunnerFilesystemConfig,
    file: &'a PendingJobFile,
    sink: &'a dyn RunnerEventSink,
    job_id: &'a JobId,
    primary: PlannedMaterializedFile,
    sidecar: PlannedMaterializedFile,
}

impl<'a> BenchmarkOutputPlan<'a> {
    pub(super) fn admit(
        filesystem: &'a RunnerFilesystemConfig,
        file: &'a PendingJobFile,
        unit: &BenchmarkWorkUnit,
        options: &crate::options::CommandOptions,
        sink: &'a dyn RunnerEventSink,
        job_id: &'a JobId,
    ) -> Result<Self, ServerError> {
        if file.filename != unit.audio().display_path {
            return Err(ServerError::Validation(
                "benchmark output plan belongs to a different source".into(),
            ));
        }
        let mut artifacts = planned_output_artifacts(
            ReleasedCommand::Benchmark,
            options,
            &unit.audio().display_path,
        )
        .map_err(|error| ServerError::Io(std::io::Error::other(error)))?
        .into_iter();
        let (Some(primary), Some(sidecar), None) =
            (artifacts.next(), artifacts.next(), artifacts.next())
        else {
            return Err(ServerError::Validation(
                "benchmark requires one CHAT output and one metrics sidecar".into(),
            ));
        };
        if primary.role != MaterializedArtifactRole::Primary
            || sidecar.role != MaterializedArtifactRole::Sidecar
        {
            return Err(ServerError::Validation(
                "benchmark output roles do not match its recipe".into(),
            ));
        }
        Ok(Self {
            filesystem,
            file,
            sink,
            job_id,
            primary,
            sidecar,
        })
    }

    /// Persistence failures are terminal; they must not repeat paid inference.
    pub(super) async fn persist(self, outputs: BenchmarkOutputs) -> FileTaskOutcome {
        let lifecycle = FileRunTracker::new(self.sink, self.job_id, self.file.filename.as_ref());
        match self.write(outputs).await {
            Ok(written) => written.complete().await,
            Err(error) => {
                lifecycle
                    .fail(&error.to_string(), classify_server_error(&error))
                    .await;
                FileTaskOutcome::TerminalStateRecorded
            }
        }
    }

    async fn write(self, outputs: BenchmarkOutputs) -> Result<WrittenBenchmark<'a>, ServerError> {
        let metrics_csv = outputs.metrics.to_csv_string().map_err(|error| {
            ServerError::Persistence(format!("benchmark CSV serialization failed: {error}"))
        })?;
        let chat_target = ChatOutputTarget::new(
            self.filesystem,
            self.file.file_index,
            &self.primary.display_path,
        );
        write_text_output_artifact(&chat_target, outputs.annotated_main_chat.as_str())
            .await
            .map_err(|error| {
                ServerError::Persistence(format!(
                    "failed to write benchmark CHAT output {}: {error}",
                    self.primary.display_path
                ))
            })?;
        let csv_target = ChatOutputTarget::new(
            self.filesystem,
            self.file.file_index,
            &self.sidecar.display_path,
        );
        write_text_output_artifact(&csv_target, &metrics_csv)
            .await
            .map_err(|error| {
                ServerError::Persistence(format!(
                    "failed to write benchmark CSV output {}: {error}",
                    self.sidecar.display_path
                ))
            })?;
        Ok(WrittenBenchmark {
            file: self.file,
            sink: self.sink,
            job_id: self.job_id,
            primary: self.primary,
        })
    }
}

/// Only the successful writer creates this capability; no completion method
/// accepts a caller-selected result filename or an unchecked write claim.
struct WrittenBenchmark<'a> {
    file: &'a PendingJobFile,
    sink: &'a dyn RunnerEventSink,
    job_id: &'a JobId,
    primary: PlannedMaterializedFile,
}

impl WrittenBenchmark<'_> {
    async fn complete(self) -> FileTaskOutcome {
        FileRunTracker::new(self.sink, self.job_id, self.file.filename.as_ref())
            .complete_with_result(self.primary.display_path, self.primary.content_type)
            .await;
        FileTaskOutcome::TerminalStateRecorded
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

    fn filesystem(root: &std::path::Path) -> RunnerFilesystemConfig {
        RunnerFilesystemConfig {
            paths_mode: true,
            source_paths: vec![ClientPath::new(
                root.join("input/clip.wav").to_str().expect("UTF-8"),
            )],
            output_paths: vec![ClientPath::new(
                root.join("requested/clip.cha").to_str().expect("UTF-8"),
            )],
            before_paths: vec![],
            staging_dir: ServerPath::new(root.join("staging").to_str().expect("UTF-8")),
            media_mapping: Default::default(),
            media_subdir: Default::default(),
            source_dir: ClientPath::new(root.join("input").to_str().expect("UTF-8")),
        }
    }

    fn file() -> PendingJobFile {
        PendingJobFile {
            file_index: 0,
            filename: "nested/clip.wav".into(),
            has_chat: false,
        }
    }

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
            let file = file();
            let sink = RecordingSink::default();
            let job_id = JobId::from("benchmark-output-test");
            let plan = BenchmarkOutputPlan::admit(
                &filesystem,
                &file,
                &unit,
                &crate::recipe_runner::runtime::test_options(ReleasedCommand::Benchmark),
                &sink,
                &job_id,
            )
            .expect("admitted output plan");
            assert!(matches!(
                plan.persist(outputs()).await,
                FileTaskOutcome::TerminalStateRecorded
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

    #[tokio::test]
    async fn benchmark_writes_complete_pairs_in_paths_and_staged_modes() {
        for paths_mode in [true, false] {
            let temp = tempfile::tempdir().expect("temporary directory");
            let mut filesystem = filesystem(temp.path());
            filesystem.paths_mode = paths_mode;
            let unit = unit(&temp.path().join("input/clip.wav"));
            let file = file();
            let sink = RecordingSink::default();
            let job_id = JobId::from("benchmark-output-test");
            let outputs = outputs();
            let expected_csv = outputs.metrics.to_csv_string().expect("CSV");
            let plan = BenchmarkOutputPlan::admit(
                &filesystem,
                &file,
                &unit,
                &crate::recipe_runner::runtime::test_options(ReleasedCommand::Benchmark),
                &sink,
                &job_id,
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
            if paths_mode {
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

    #[test]
    fn benchmark_output_plan_rejects_a_different_source_file() {
        let temp = tempfile::tempdir().expect("temporary directory");
        let filesystem = filesystem(temp.path());
        let unit = unit(&temp.path().join("input/clip.wav"));
        let mut file = file();
        file.filename = "other.wav".into();
        let sink = RecordingSink::default();
        let job_id = JobId::from("benchmark-output-test");
        assert!(matches!(
            BenchmarkOutputPlan::admit(
                &filesystem,
                &file,
                &unit,
                &crate::recipe_runner::runtime::test_options(ReleasedCommand::Benchmark),
                &sink,
                &job_id
            ),
            Err(ServerError::Validation(_))
        ));
        assert!(!temp.path().join("requested").exists());
        assert!(!temp.path().join("staging").exists());
        assert!(sink.completed_files().is_empty());
    }
}
