use std::collections::HashMap;

use crate::chat_ops::morphosyntax_ops::MwtDict;

use crate::api::{DisplayPath, ReleasedCommand};
use crate::compare::is_gold_file;
use crate::planning::{self, JobPlan};
use crate::recipe_runner::runtime::publish_text;
use crate::recipe_runner::work_unit::PlannedWorkUnit;
use crate::runner::DispatchHostContext;
use crate::runner::util::{FileRunTracker, FileStage, classify_server_error};

mod comparison;
mod outputs;
use crate::scheduling::{FailureCategory, WorkUnitKind};
use crate::store::RunnerJobSnapshot;
use comparison::PreparedComparison;
use outputs::{ConsolidatedCompareMetricsRow, format_consolidated_csv};

use super::worker_gateway::CompareMorphology;

/// Immutable execution inputs shared across stages for one job.
struct ExecutionContext<'a> {
    pub(crate) job: &'a RunnerJobSnapshot,
    pub(crate) host: &'a DispatchHostContext,
    pub(crate) gateway: &'a dyn CompareMorphology,
    pub(crate) mwt: &'a MwtDict,
}

/// Execute compare's concrete consuming graph over its planned work units.
async fn run_compare(
    plan: &JobPlan,
    ctx: &ExecutionContext<'_>,
) -> Result<(), crate::error::ServerError> {
    if plan.spec.command != ReleasedCommand::Compare {
        return Err(crate::error::ServerError::Validation(format!(
            "compare execution received command '{}'",
            plan.spec.command
        )));
    }
    let file_by_display: HashMap<&DisplayPath, &crate::store::PendingJobFile> = ctx
        .job
        .pending_files
        .iter()
        .map(|file| (file.filename(), file))
        .collect();
    let sink = ctx.host.sink().clone();
    let mut consolidated_rows = Vec::new();

    for file in &ctx.job.pending_files {
        let lifecycle = FileRunTracker::new(sink.as_ref(), file.filename().as_ref());
        lifecycle
            .begin_first_attempt(WorkUnitKind::BatchInfer, FileStage::Comparing)
            .await;
        if is_gold_file(file.filename().as_ref()) {
            lifecycle.complete_without_result().await;
        }
    }

    for work_unit in &plan.work_units {
        let PlannedWorkUnit::Compare(unit) = work_unit else {
            continue;
        };
        let Some(file) = file_by_display.get(&unit.main.display_path).copied() else {
            continue;
        };
        let comparison = match PreparedComparison::admit(unit, file, plan, ctx) {
            Ok(comparison) => comparison,
            Err(error) => {
                FileRunTracker::new(sink.as_ref(), file.filename().as_ref())
                    .fail(&error.to_string(), classify_server_error(&error))
                    .await;
                continue;
            }
        };
        match comparison.run().await {
            Ok(row) => consolidated_rows.push(row),
            Err(error) => {
                FileRunTracker::new(sink.as_ref(), file.filename().as_ref())
                    .fail(&error.to_string(), classify_server_error(&error))
                    .await;
            }
        }
    }

    if !consolidated_rows.is_empty() {
        write_consolidated_compare_csv(&ctx.job.filesystem, &consolidated_rows).await?;
    }

    Ok(())
}

/// Runner-owned dispatch entrypoint for the first migrated command family.
pub(crate) async fn dispatch_compare_job(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    gateway: &dyn CompareMorphology,
    mwt: &MwtDict,
) -> Result<(), crate::error::ServerError> {
    let plan = match planning::build_job_plan(job) {
        Ok(plan) => plan,
        Err(error) => {
            let sink = host.sink().clone();
            for file in &job.pending_files {
                let lifecycle = FileRunTracker::new(sink.as_ref(), file.filename().as_ref());
                if is_gold_file(file.filename().as_ref()) {
                    lifecycle.complete_without_result().await;
                    continue;
                }
                lifecycle
                    .fail(
                        &format!("Compare planning failed: {error}"),
                        FailureCategory::Validation,
                    )
                    .await;
            }
            return Ok(());
        }
    };

    let ctx = ExecutionContext {
        job,
        host,
        gateway,
        mwt,
    };
    run_compare(&plan, &ctx).await
}

async fn write_consolidated_compare_csv(
    filesystem: &crate::store::RunnerFilesystemConfig,
    rows: &[ConsolidatedCompareMetricsRow],
) -> Result<(), crate::error::ServerError> {
    let output = format_consolidated_csv(rows)?;

    let primary_path = filesystem
        .inputs
        .job_level_output(&filesystem.staging_dir, "compare.csv");
    publish_text(&primary_path, &output)
        .await
        .map_err(|error| {
            crate::error::ServerError::Persistence(format!(
                "failed to write consolidated compare.csv: {error}"
            ))
        })?;

    let staged_path = filesystem.staging_dir.join("output").join("compare.csv");
    let staged_path = staged_path.as_path().to_owned();
    if staged_path != primary_path {
        publish_text(&staged_path, &output).await.map_err(|error| {
            crate::error::ServerError::Persistence(format!(
                "failed to write staged consolidated compare.csv: {error}"
            ))
        })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use async_trait::async_trait;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::api::{CorrelationId, JobId, LanguageSpec, NumSpeakers};
    use crate::options::{CommandOptions, CommonOptions, CompareOptions};
    use crate::store::{RunnerDispatchConfig, RunnerFilesystemConfig, RunnerJobIdentity};

    #[derive(Default)]
    struct FakeGateway {
        calls: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl CompareMorphology for FakeGateway {
        async fn morphotag_for_compare(
            &self,
            chat_text: &str,
            _mwt: &MwtDict,
            _cancellation: crate::infer_retry::Cancellation<'_>,
        ) -> Result<crate::pipeline::post_validate::PostValidated, crate::error::ServerError>
        {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // A GATED proof over the double's own text, which is what the real
            // gateway returns: the production implementation runs the gate, so
            // a double standing in for it upstream must carry the same kind of
            // proof or the comparison stage would be exercising a seam that
            // does not exist.
            Ok(crate::pipeline::post_validate::PostValidated::for_test(
                chat_text,
                crate::api::ReleasedCommand::Morphotag,
            ))
        }
    }

    fn compare_snapshot(
        staging_dir: &std::path::Path,
        inputs: crate::store::JobInputs,
    ) -> RunnerJobSnapshot {
        crate::store::test_support::with_pending(RunnerJobSnapshot {
            identity: RunnerJobIdentity {
                job_id: JobId::from("job-compare-kernel"),
                correlation_id: CorrelationId::from("corr-compare-kernel"),
            },
            dispatch: RunnerDispatchConfig {
                command: ReleasedCommand::Compare,
                // Compare is a per-file command: the language is each
                // transcript's own, never a job-level code.
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
                staging_dir: batchalign_types::paths::ServerPath::new(staging_dir),
                media_mapping: Default::default(),
                media_subdir: Default::default(),
                source_dir: batchalign_types::paths::ClientPath::new("/source"),
            },
            cancel_token: CancellationToken::new(),
            pending_files: Vec::new(),
        })
    }

    const VALID_CHAT: &str = "@UTF8\n@Begin\n@Languages:\teng\n\
@Participants:\tPAR Participant\n@ID:\teng|test|PAR|||||Participant|||\n\
*PAR:\thello there .\n@End\n";

    enum ReferenceFixture<'a> {
        Companion(&'a str),
        TemplateOnly(&'a str),
        Both {
            companion: &'a str,
            template: &'a str,
        },
        UnreadableCompanion {
            template: &'a str,
        },
    }

    enum OutputFixture {
        Staged,
        BlockedChat,
        BlockedSidecar,
        Paths,
    }

    async fn run_compare_fixture(
        reference: ReferenceFixture<'_>,
        output: OutputFixture,
    ) -> (tempfile::TempDir, usize) {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let input_dir = tempdir.path().join("input");
        tokio::fs::create_dir_all(&input_dir)
            .await
            .expect("input dir");
        tokio::fs::write(input_dir.join("sample.cha"), VALID_CHAT)
            .await
            .expect("main");
        match reference {
            ReferenceFixture::Companion(text) => {
                tokio::fs::write(input_dir.join("sample.gold.cha"), text)
                    .await
                    .unwrap()
            }
            ReferenceFixture::TemplateOnly(text) => {
                tokio::fs::write(input_dir.join("template.gold.cha"), text)
                    .await
                    .unwrap()
            }
            ReferenceFixture::Both {
                companion,
                template,
            } => {
                tokio::fs::write(input_dir.join("sample.gold.cha"), companion)
                    .await
                    .unwrap();
                tokio::fs::write(input_dir.join("template.gold.cha"), template)
                    .await
                    .unwrap();
            }
            ReferenceFixture::UnreadableCompanion { template } => {
                tokio::fs::create_dir(input_dir.join("sample.gold.cha"))
                    .await
                    .unwrap();
                tokio::fs::write(input_dir.join("template.gold.cha"), template)
                    .await
                    .unwrap();
            }
        }
        let inputs = match &output {
            OutputFixture::Paths => {
                let destination = tempdir.path().join("requested").join("sample.cha");
                crate::store::test_support::paths_inputs(vec![crate::store::PathsInput {
                    display: "sample.cha".into(),
                    kind: crate::store::InputKind::Chat,
                    source: crate::store::test_support::admitted(
                        input_dir.join("sample.cha").to_str().unwrap(),
                    ),
                    output: crate::store::test_support::admitted(destination.to_str().unwrap()),
                    before: None,
                }])
            }
            OutputFixture::Staged | OutputFixture::BlockedChat | OutputFixture::BlockedSidecar => {
                crate::store::JobInputs::Content(vec![crate::store::ContentInput {
                    display: "sample.cha".into(),
                    kind: crate::store::InputKind::Chat,
                }])
            }
        };
        let blocked_name = match output {
            OutputFixture::BlockedChat => Some("sample.cha"),
            OutputFixture::BlockedSidecar => Some("sample.compare.csv"),
            OutputFixture::Staged | OutputFixture::Paths => None,
        };
        if let Some(name) = blocked_name {
            tokio::fs::create_dir_all(tempdir.path().join("output").join(name))
                .await
                .unwrap();
        }
        let snapshot = compare_snapshot(tempdir.path(), inputs);
        let host = DispatchHostContext::for_test_effects();
        let gateway = FakeGateway::default();
        dispatch_compare_job(&snapshot, &host, &gateway, &MwtDict::default())
            .await
            .expect("dispatch compare");

        (
            tempdir,
            gateway.calls.load(std::sync::atomic::Ordering::SeqCst),
        )
    }

    async fn run_compare_reference(reference: &str) -> (tempfile::TempDir, usize) {
        run_compare_fixture(
            ReferenceFixture::Companion(reference),
            OutputFixture::Staged,
        )
        .await
    }

    /// Guards: A valid reference produces the primary CHAT file and both required CSV artifacts.
    /// Kind: contract
    /// Not a type: Admitted plans cannot prove actual artifact creation through the compare reader and writer adapters.
    /// State: Temporary CHAT input, reference and outputs exercise real filesystem IO with a recording gateway. The shared sink removes registry effects.
    /// Concurrency: The test owns its directory, gateway counter and recording sink. No external worker or shared store runs.
    #[tokio::test]
    async fn compare_kernel_writes_primary_and_sidecar_outputs() {
        let (tempdir, calls) = run_compare_reference(VALID_CHAT).await;
        assert_eq!(calls, 1);

        let primary = tempdir.path().join("output").join("sample.cha");
        let csv = tempdir.path().join("output").join("sample.compare.csv");
        let consolidated = tempdir.path().join("output").join("compare.csv");
        assert!(primary.exists());
        assert!(csv.exists());
        assert!(consolidated.exists());
    }

    /// Guards: Malformed reference bytes produce neither inference requests nor output artifacts.
    /// Kind: contract
    /// Not a type: Typed plans cannot establish whether newly read external CHAT bytes pass reference validation.
    /// State: Each malformed reference is written under an isolated directory and read by production validation. A recording gateway checks absence of inference.
    /// Concurrency: Each case owns its directory, gateway counter and recording sink. There is no shared registry or worker.
    #[tokio::test]
    async fn compare_invalid_reference_cannot_dispatch_inference_or_write_outputs() {
        for invalid in [
            VALID_CHAT.replace("hello there .", "hello there"),
            VALID_CHAT.replace("@End", "%mor:\tnoun|hello\n@End"),
            VALID_CHAT.replace("@End", "%pho:\thəloʊ\n@End"),
        ] {
            let (tempdir, calls) = run_compare_reference(&invalid).await;
            assert_eq!(calls, 0, "invalid reference reached the inference gateway");
            for artifact in ["sample.cha", "sample.compare.csv", "compare.csv"] {
                assert!(!tempdir.path().join("output").join(artifact).exists());
            }
        }
    }

    /// Guards: Companion presence controls template fallback and unreadable companions refuse inference.
    /// Kind: contract
    /// Not a type: Closed fixture variants cannot establish actual filesystem presence, read errors or the chosen reference text.
    /// State: Distinct temporary layouts exercise companion selection, template fallback and a directory at the companion path as a real read failure.
    /// Concurrency: Each layout owns its directory, recording gateway and sink. No filesystem state is shared across cases.
    #[tokio::test]
    async fn compare_reference_selection_uses_template_only_when_companion_is_absent() {
        let different = VALID_CHAT.replace("hello there .", "goodbye there .");
        let (fallback, calls) = run_compare_fixture(
            ReferenceFixture::TemplateOnly(&different),
            OutputFixture::Staged,
        )
        .await;
        assert_eq!(calls, 1);
        assert!(
            tokio::fs::read_to_string(fallback.path().join("output/sample.cha"))
                .await
                .unwrap()
                .contains("*PAR:\tgoodbye there .")
        );
        let (preferred, calls) = run_compare_fixture(
            ReferenceFixture::Both {
                companion: VALID_CHAT,
                template: &different,
            },
            OutputFixture::Staged,
        )
        .await;
        assert_eq!(calls, 1);
        assert!(
            tokio::fs::read_to_string(preferred.path().join("output/sample.cha"))
                .await
                .unwrap()
                .contains("*PAR:\thello there .")
        );
        let (blocked, calls) = run_compare_fixture(
            ReferenceFixture::UnreadableCompanion {
                template: VALID_CHAT,
            },
            OutputFixture::Staged,
        )
        .await;
        assert_eq!(
            calls, 0,
            "an unreadable companion must not silently select a different reference"
        );
        assert!(!blocked.path().join("output/compare.csv").exists());
    }

    /// Guards: A failed required artifact write contributes no successful consolidated metric row.
    /// Kind: contract
    /// Not a type: An admitted output path cannot guarantee later filesystem writes succeed or prevent obstructing directory entries.
    /// State: Temporary directories placed at required output filenames induce real write failures. Actual consolidation must omit the row.
    /// Concurrency: Every obstruction case owns its directory, gateway counter and sink. No external process or store is used.
    #[tokio::test]
    async fn compare_failed_required_write_cannot_supply_a_successful_metric_row() {
        for output in [OutputFixture::BlockedChat, OutputFixture::BlockedSidecar] {
            let (directory, calls) =
                run_compare_fixture(ReferenceFixture::Companion(VALID_CHAT), output).await;
            assert_eq!(calls, 1);
            assert!(!directory.path().join("output/compare.csv").exists());
        }
    }

    /// Guards: Paths mode retains matching required artifacts in staging and requested destinations.
    /// Kind: contract
    /// Not a type: Admitted path pairs prove authority but cannot prove the bytes copied to both filesystem destinations.
    /// State: An isolated admitted paths fixture exercises real staging and requested-directory writes against a recording gateway and sink.
    /// Concurrency: The test owns both destinations under one temporary root and its gateway counter. No shared registry runs.
    #[tokio::test]
    async fn compare_paths_mode_keeps_both_required_artifacts_in_staging() {
        let (directory, calls) = run_compare_fixture(
            ReferenceFixture::Companion(VALID_CHAT),
            OutputFixture::Paths,
        )
        .await;
        assert_eq!(calls, 1);
        for file in ["sample.cha", "sample.compare.csv", "compare.csv"] {
            assert_eq!(
                tokio::fs::read(directory.path().join("requested").join(file))
                    .await
                    .unwrap(),
                tokio::fs::read(directory.path().join("output").join(file))
                    .await
                    .unwrap(),
            );
        }
    }
}
