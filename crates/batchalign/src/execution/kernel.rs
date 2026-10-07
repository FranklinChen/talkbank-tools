use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::chat_ops::morphosyntax_ops::MwtDict;
use async_trait::async_trait;

use crate::api::{DisplayPath, ReleasedCommand};
use crate::compare::{
    CompareMaterializedOutputs, is_gold_file, process_compare_morphotagged_main,
    template_gold_path_for,
};
use crate::dispatch_language::JobLanguage;
use crate::planning::{self, JobPlan};
use crate::recipe_runner::recipe::RecipeStageId;
use crate::recipe_runner::work_unit::{CompareWorkUnit, PlannedWorkUnit};
use crate::runner::DispatchHostContext;
use crate::runner::util::{FileRunTracker, FileStage, classify_server_error};

mod outputs;
use crate::scheduling::{FailureCategory, WorkUnitKind};
use crate::store::RunnerJobSnapshot;
use outputs::{ConsolidatedCompareMetricsRow, format_consolidated_csv, write_outputs};

use super::worker_gateway::WorkerGateway;

/// Immutable execution inputs shared across stages for one job.
struct ExecutionContext<'a> {
    pub(crate) job: &'a RunnerJobSnapshot,
    pub(crate) host: &'a DispatchHostContext,
    pub(crate) gateway: &'a dyn WorkerGateway,
    pub(crate) mwt: &'a MwtDict,
    pub(crate) should_merge_abbrev: bool,
    /// The job's resolved language, proven by the router's one resolution.
    pub(crate) job_language: &'a JobLanguage,
}

/// Stage executor interface used by the new execution kernel.
#[async_trait]
trait StageExecutor {
    /// Run one stage for the current work unit.
    async fn run_stage(
        &self,
        stage: RecipeStageId,
        state: &mut CompareExecutionState,
        plan: &JobPlan,
        ctx: &ExecutionContext<'_>,
    ) -> Result<(), crate::error::ServerError>;
}

/// Minimal execution kernel for recipe-owned commands.
struct ExecutionKernel {
    stage_executor: Box<dyn StageExecutor + Send + Sync>,
}

impl ExecutionKernel {
    /// Build a kernel with one stage executor implementation.
    pub(crate) fn new(stage_executor: Box<dyn StageExecutor + Send + Sync>) -> Self {
        Self { stage_executor }
    }

    /// Run one immutable job plan through the stage executor.
    pub(crate) async fn run(
        &self,
        plan: &JobPlan,
        ctx: &ExecutionContext<'_>,
    ) -> Result<(), crate::error::ServerError> {
        match plan.spec.command {
            ReleasedCommand::Compare => self.run_compare(plan, ctx).await,
            command => Err(crate::error::ServerError::Validation(format!(
                "execution kernel does not yet support command '{command}'"
            ))),
        }
    }

    async fn run_compare(
        &self,
        plan: &JobPlan,
        ctx: &ExecutionContext<'_>,
    ) -> Result<(), crate::error::ServerError> {
        let file_index_by_display: HashMap<DisplayPath, usize> = ctx
            .job
            .pending_files
            .iter()
            .map(|file| (file.filename.clone(), file.file_index))
            .collect();
        let sink = ctx.host.sink().clone();
        let mut consolidated_rows = Vec::new();

        for file in &ctx.job.pending_files {
            let lifecycle = FileRunTracker::new(
                sink.as_ref(),
                &ctx.job.identity.job_id,
                file.filename.as_ref(),
            );
            lifecycle
                .begin_first_attempt(WorkUnitKind::BatchInfer, FileStage::Comparing)
                .await;
            if is_gold_file(file.filename.as_ref()) {
                lifecycle.complete_without_result().await;
            }
        }

        for work_unit in &plan.work_units {
            let PlannedWorkUnit::Compare(unit) = work_unit else {
                continue;
            };
            let Some(file_index) = file_index_by_display.get(&unit.main.display_path).copied()
            else {
                continue;
            };
            let mut state = CompareExecutionState::new(unit.clone(), file_index);
            let stage_result = async {
                for stage in plan.spec.recipe.stages {
                    self.stage_executor
                        .run_stage(stage.id, &mut state, plan, ctx)
                        .await?;
                }
                Ok::<(), crate::error::ServerError>(())
            }
            .await;
            if let Err(error) = stage_result {
                state
                    .lifecycle(ctx)
                    .fail(&error.to_string(), classify_server_error(&error))
                    .await;
            } else if let Some(row) = state.consolidated_metrics.take() {
                consolidated_rows.push(row);
            }
        }

        if !consolidated_rows.is_empty() {
            write_consolidated_compare_csv(&ctx.job.filesystem, &consolidated_rows).await?;
        }

        Ok(())
    }
}

/// Runner-owned dispatch entrypoint for the first migrated command family.
pub(crate) async fn dispatch_compare_job(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    gateway: &dyn WorkerGateway,
    mwt: &MwtDict,
    should_merge_abbrev: bool,
    job_language: &JobLanguage,
) -> Result<(), crate::error::ServerError> {
    let plan = match planning::build_job_plan(job) {
        Ok(plan) => plan,
        Err(error) => {
            let sink = host.sink().clone();
            for file in &job.pending_files {
                let lifecycle = FileRunTracker::new(
                    sink.as_ref(),
                    &job.identity.job_id,
                    file.filename.as_ref(),
                );
                if is_gold_file(file.filename.as_ref()) {
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
        should_merge_abbrev,
        job_language,
    };
    ExecutionKernel::new(Box::new(CompareStageExecutor))
        .run(&plan, &ctx)
        .await
}

struct CompareExecutionState {
    unit: CompareWorkUnit,
    file_index: usize,
    main_text: Option<String>,
    reference: Option<crate::compare::AdmittedComparisonReference>,
    /// Morphotag's PROOF of the main transcript, carried between the two
    /// stages rather than its bytes, so the comparison stage continues in the
    /// document the gate judged and has no text to re-parse.
    morphotagged_main: Option<crate::pipeline::post_validate::PostValidated>,
    outputs: Option<CompareMaterializedOutputs>,
    consolidated_metrics: Option<ConsolidatedCompareMetricsRow>,
}

impl CompareExecutionState {
    fn new(unit: CompareWorkUnit, file_index: usize) -> Self {
        Self {
            unit,
            file_index,
            main_text: None,
            reference: None,
            morphotagged_main: None,
            outputs: None,
            consolidated_metrics: None,
        }
    }

    fn lifecycle<'a>(&'a self, ctx: &'a ExecutionContext<'_>) -> FileRunTracker<'a> {
        FileRunTracker::new(
            ctx.host.sink().as_ref(),
            &ctx.job.identity.job_id,
            self.unit.main.display_path.as_ref(),
        )
    }
}

struct CompareStageExecutor;

#[async_trait]
impl StageExecutor for CompareStageExecutor {
    async fn run_stage(
        &self,
        stage: RecipeStageId,
        state: &mut CompareExecutionState,
        plan: &JobPlan,
        ctx: &ExecutionContext<'_>,
    ) -> Result<(), crate::error::ServerError> {
        match stage {
            RecipeStageId::PlanWorkUnits => Ok(()),
            RecipeStageId::ReadChatInputs => {
                state.lifecycle(ctx).stage(FileStage::Reading).await;
                state.main_text = Some(
                    tokio::fs::read_to_string(&state.unit.main.source_path)
                        .await
                        .map_err(|error| {
                            crate::error::ServerError::Validation(format!(
                                "failed to read compare input {}: {error}",
                                state.unit.main.display_path
                            ))
                        })?,
                );
                Ok(())
            }
            RecipeStageId::ReadReferenceInputs => {
                state.lifecycle(ctx).stage(FileStage::Reading).await;
                let template_gold_source = state
                    .unit
                    .main
                    .source_path
                    .with_file_name("template.gold.cha");
                let gold_text = match tokio::fs::read_to_string(&state.unit.gold.source_path).await {
                    Ok(text) => text,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => tokio::fs::read_to_string(&template_gold_source)
                        .await
                        .map_err(|_| {
                            crate::error::ServerError::Validation(format!(
                                "No gold .cha file found for comparison. main: {}, expected: {} or {}",
                                state.unit.main.display_path,
                                state.unit.gold.display_path,
                                template_gold_path_for(state.unit.main.display_path.as_ref()),
                            ))
                        })?,
                    Err(error) => return Err(crate::error::ServerError::Validation(format!(
                        "failed to read compare reference {}: {error}", state.unit.gold.display_path
                    ))),
                };
                state.reference = Some(crate::compare::AdmittedComparisonReference::admit(
                    &gold_text,
                )?);
                Ok(())
            }
            RecipeStageId::Morphosyntax => {
                // A dynamic recipe cannot dispatch comparison inference until
                // its reference stage has produced complete admission.
                if state.reference.is_none() {
                    return Err(crate::error::ServerError::OutputAdmission {
                        command: ReleasedCommand::Compare,
                        details: crate::error::OutputAdmissionRefusal::unestablished(
                            "compare morphosyntax stage ran before reference admission",
                        ),
                    });
                }
                state
                    .lifecycle(ctx)
                    .stage(FileStage::AnalyzingMorphosyntax)
                    .await;
                let main_text = state.main_text.as_deref().ok_or_else(|| {
                    crate::error::ServerError::Validation(
                        "compare morphosyntax stage ran before input read".into(),
                    )
                })?;
                // The worker-pool / label key: the job's own resolved
                // language, handed to this dispatch rather than recovered from
                // the snapshot. Real per-file resolution still lives inside
                // `collect_payloads`.
                //
                // This was `as_resolved().unwrap_or_else(|| eng)`: a silent
                // English default behind a `warn!`, which is the same shape as
                // the check that made coref undispatchable, with the opposite
                // failure mode. Compare is a job-level command, so a
                // `JobLanguage` exists for it by construction and there is no
                // absent case left to default.
                let lang = ctx.job_language.code();
                state.morphotagged_main = Some(
                    ctx.gateway
                        .morphotag_for_compare(
                            main_text,
                            lang,
                            ctx.mwt,
                            crate::infer_retry::Cancellation::Token(&ctx.job.cancel_token),
                        )
                        .await?,
                );
                Ok(())
            }
            RecipeStageId::CompareAlign => {
                state.lifecycle(ctx).stage(FileStage::Comparing).await;
                // TAKEN, not borrowed: the proof is consumed by the comparison
                // it feeds, so a second CompareAlign on the same state reports
                // the stage-order error rather than comparing a document that
                // has already been compared.
                let morphotagged_main = state.morphotagged_main.take().ok_or_else(|| {
                    crate::error::ServerError::Validation(
                        "compare alignment stage ran before morphosyntax".into(),
                    )
                })?;
                let reference = state.reference.take().ok_or_else(|| {
                    crate::error::ServerError::Validation(
                        "compare alignment stage ran before reference read".into(),
                    )
                })?;
                state.outputs = Some(process_compare_morphotagged_main(
                    morphotagged_main,
                    reference,
                )?);
                Ok(())
            }
            RecipeStageId::CompareMetrics => Ok(()),
            RecipeStageId::SerializeChat => {
                state.lifecycle(ctx).stage(FileStage::Finalizing).await;
                Ok(())
            }
            RecipeStageId::MaterializeOutputs => {
                state.lifecycle(ctx).stage(FileStage::Writing).await;
                let CompareMaterializedOutputs {
                    chat_output,
                    metrics,
                } = state.outputs.take().ok_or_else(|| {
                    crate::error::ServerError::Validation(
                        "compare output materialization ran before compare outputs existed".into(),
                    )
                })?;
                // The merge is the LAST transition on the proof, so the bytes
                // written are bytes the gate judged. It ran on the finished
                // TEXT here until 2026-09-07 (`merge_abbreviations_in_chat_text`
                // over `chat_output`, then write the result), which put a
                // transform after the only thing that had looked at the
                // document. A refusal is returned, not warned about: this
                // stage's `Result` is what fails the file.
                //
                // The flag stays a `bool` here rather than the
                // `MergeAbbreviations` enum the two dispatch writers use:
                // `runner::dispatch` is private to `runner` and re-exports
                // nothing, so sharing the vocabulary would mean widening two
                // module boundaries for a two-variant enum.
                let chat_output = if ctx.should_merge_abbrev {
                    // POLICY: the refusal carries `unmerged`, an admissible
                    // compare output, and this stage declines to write it. A
                    // merge that breaks the gate its input passed is a defect
                    // in the merge, and writing past it would hide one.
                    chat_output.with_abbreviations_merged().map_err(|refused| {
                        crate::error::ServerError::Validation(refused.to_string())
                    })?
                } else {
                    chat_output
                };
                let Some(artifacts) =
                    planning::artifact_set_for_source(plan, &state.unit.main.display_path)
                else {
                    return Err(crate::error::ServerError::Validation(format!(
                        "compare job plan was missing artifacts for {}",
                        state.unit.main.display_path
                    )));
                };
                let written = write_outputs(
                    &ctx.job.filesystem,
                    state.file_index,
                    artifacts,
                    CompareMaterializedOutputs {
                        chat_output,
                        metrics,
                    },
                )
                .await?;
                state.consolidated_metrics = Some(written.complete(state.lifecycle(ctx)).await);
                Ok(())
            }
            other => Err(crate::error::ServerError::Validation(format!(
                "compare kernel does not yet support stage '{other}'"
            ))),
        }
    }
}

async fn write_compare_text_file(path: &Path, content: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(path, content).await
}

fn compare_output_root(filesystem: &crate::store::RunnerFilesystemConfig) -> PathBuf {
    if filesystem.paths_mode && !filesystem.output_paths.is_empty() {
        let first_output = filesystem.output_paths[0].assume_shared_filesystem();
        return Path::new(first_output.as_str())
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
    }

    filesystem.staging_dir.join("output").as_path().to_owned()
}

async fn write_consolidated_compare_csv(
    filesystem: &crate::store::RunnerFilesystemConfig,
    rows: &[ConsolidatedCompareMetricsRow],
) -> Result<(), crate::error::ServerError> {
    let output = format_consolidated_csv(rows)?;

    let primary_path = compare_output_root(filesystem).join("compare.csv");
    write_compare_text_file(&primary_path, &output)
        .await
        .map_err(|error| {
            crate::error::ServerError::Persistence(format!(
                "failed to write consolidated compare.csv: {error}"
            ))
        })?;

    let staged_path = filesystem.staging_dir.join("output").join("compare.csv");
    let staged_path = staged_path.as_path().to_owned();
    if staged_path != primary_path {
        write_compare_text_file(&staged_path, &output)
            .await
            .map_err(|error| {
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
    use crate::planning::build_job_plan;
    use crate::store::{
        PendingJobFile, RunnerDispatchConfig, RunnerFilesystemConfig, RunnerJobIdentity,
    };

    #[derive(Default)]
    struct FakeGateway {
        calls: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl WorkerGateway for FakeGateway {
        async fn morphotag_for_compare(
            &self,
            chat_text: &str,
            _lang: &crate::api::LanguageCode3,
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

        async fn morphotag_single(
            &self,
            _chat_text: &str,
            _before_text: Option<&str>,
            _options: crate::execution::MorphotagRuntimeOptions,
            _progress: Option<&crate::execution::morphotag::progress::BackendProgressPort>,
            _cancellation: crate::infer_retry::Cancellation<'_>,
        ) -> Result<crate::pipeline::post_validate::PostValidated, crate::error::ServerError>
        {
            unreachable!("compare tests do not call morphotag_single")
        }

        async fn utseg_batch(
            &self,
            _files: &[crate::text_batch::TextBatchFileInput],
            _lang: &crate::api::LanguageCode3,
            _allow_stanza_fallback: bool,
            _cancellation: crate::infer_retry::Cancellation<'_>,
        ) -> crate::text_batch::TextBatchFileResults {
            unreachable!("compare tests do not call utseg_batch")
        }

        async fn translate_file(
            &self,
            _file: &crate::text_batch::TextBatchFileInput,
            _route: &crate::translate::TranslationRoute,
            _cancellation: crate::infer_retry::Cancellation<'_>,
        ) -> crate::text_batch::TextBatchFileResults {
            unreachable!("compare tests do not call translate_file")
        }

        async fn coref_batch(
            &self,
            _files: &[crate::text_batch::TextBatchFileInput],
            _cancellation: crate::infer_retry::Cancellation<'_>,
        ) -> crate::text_batch::TextBatchFileResults {
            unreachable!("compare tests do not call coref_batch")
        }
    }

    fn compare_snapshot(staging_dir: &std::path::Path) -> RunnerJobSnapshot {
        RunnerJobSnapshot {
            run_generation: crate::store::RunGeneration::FIRST,
            identity: RunnerJobIdentity {
                job_id: JobId::from("job-compare-kernel"),
                correlation_id: CorrelationId::from("corr-compare-kernel"),
            },
            dispatch: RunnerDispatchConfig {
                command: ReleasedCommand::Compare,
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
                paths_mode: false,
                source_paths: Vec::new(),
                output_paths: Vec::new(),
                before_paths: Vec::new(),
                staging_dir: batchalign_types::paths::ServerPath::new(staging_dir),
                media_mapping: Default::default(),
                media_subdir: Default::default(),
                source_dir: batchalign_types::paths::ClientPath::new("/source"),
            },
            cancel_token: CancellationToken::new(),
            pending_files: vec![PendingJobFile {
                file_index: 0,
                filename: DisplayPath::from("sample.cha"),
                has_chat: true,
            }],
        }
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
        let mut snapshot = compare_snapshot(tempdir.path());
        match output {
            OutputFixture::Staged => {}
            OutputFixture::BlockedChat | OutputFixture::BlockedSidecar => {
                let name = match output {
                    OutputFixture::BlockedChat => "sample.cha",
                    _ => "sample.compare.csv",
                };
                tokio::fs::create_dir_all(tempdir.path().join("output").join(name))
                    .await
                    .unwrap();
            }
            OutputFixture::Paths => {
                let destination = tempdir.path().join("requested").join("sample.cha");
                snapshot.filesystem.paths_mode = true;
                snapshot.filesystem.source_paths = vec![batchalign_types::paths::ClientPath::new(
                    input_dir.join("sample.cha").to_str().unwrap(),
                )];
                snapshot.filesystem.output_paths = vec![batchalign_types::paths::ClientPath::new(
                    destination.to_str().unwrap(),
                )];
            }
        }
        let plan = build_job_plan(&snapshot).expect("plan");
        let (_tx, _rx) = tokio::sync::broadcast::channel(crate::ws::BROADCAST_CAPACITY);
        let store = std::sync::Arc::new(crate::store::JobStore::new(
            crate::config::ServerConfig::default(),
            None,
            _tx,
            std::sync::Arc::new(crate::clock::SystemClock),
        ));
        let host = DispatchHostContext::from_store(store);
        // Built through the one constructor rather than assembled here, so the
        // test exercises the real resolution: compare is a job-level command,
        // and this is the language its snapshot carries. Bound to a `let`
        // because the context borrows it.
        let resolved = crate::dispatch_language::DispatchLanguage::resolve(
            ReleasedCommand::Compare,
            &LanguageSpec::Resolved(crate::api::LanguageCode3::eng()),
        )
        .expect("compare submits a resolved language");
        let crate::dispatch_language::DispatchLanguage::Job(job_language) = resolved else {
            panic!("compare is a job-level command")
        };
        let gateway = FakeGateway::default();
        let ctx = ExecutionContext {
            job: &snapshot,
            host: &host,
            gateway: &gateway,
            mwt: &MwtDict::default(),
            should_merge_abbrev: false,
            job_language: &job_language,
        };

        ExecutionKernel::new(Box::new(CompareStageExecutor))
            .run(&plan, &ctx)
            .await
            .expect("run compare kernel");

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

    #[tokio::test]
    async fn compare_failed_required_write_cannot_supply_a_successful_metric_row() {
        for output in [OutputFixture::BlockedChat, OutputFixture::BlockedSidecar] {
            let (directory, calls) =
                run_compare_fixture(ReferenceFixture::Companion(VALID_CHAT), output).await;
            assert_eq!(calls, 1);
            assert!(!directory.path().join("output/compare.csv").exists());
        }
    }

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
