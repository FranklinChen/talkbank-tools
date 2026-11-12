//! Benchmark dispatch built on the Rust-owned transcribe and compare pipelines.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::chat_ops::morphosyntax_ops::MwtDict;
use tracing::warn;

use crate::api::NumWorkers;
use crate::benchmark::{BenchmarkRequest, process_benchmark};
use crate::cache::UtteranceCache;
use crate::pipeline::PipelineServices;
use crate::recipe_runner::runtime::plan_work_units_for_job;
use crate::recipe_runner::work_unit::{BenchmarkWorkUnit, PlannedWorkUnit};
use crate::runner::DispatchHostContext;
use crate::scheduling::{FailureCategory, WorkUnitKind};
use crate::store::RunnerJobSnapshot;
use crate::transcribe::TranscribeOptions;
use crate::worker::pool::WorkerPool;

use super::super::util::{
    AfterFailure, FileAttempts, FileRunTracker, FileStage, FileTaskOutcome, RunnerEventSink,
    classify_server_error, drain_supervised_file_tasks, spawn_observed_progress_forwarder,
    spawn_supervised_file_task, user_facing_error,
};
use super::BenchmarkDispatchPlan;
use super::asr_media::{prepare_asr_media_input, preserved_media_name_for_chat};

mod outputs;
use outputs::BenchmarkOutputPlan;

// Benchmark composes transcription and comparison. Keep that large state
// machine behind a pinned owner BEFORE the generic supervisor wraps/moves it;
// boxing only inside the supervisor would leave its entry stack unbounded.
type BenchmarkFileFuture = Pin<Box<dyn Future<Output = FileTaskOutcome> + Send>>;

/// Shared runtime dependencies for top-level benchmark dispatch.
///
/// Benchmark reuses the transcribe and compare stacks, so the runtime bundle is
/// the same worker/cache context plus the file-level concurrency cap.
pub(crate) struct BenchmarkDispatchRuntime {
    /// Worker pool used for the benchmark's ASR requests.
    pub pool: Arc<WorkerPool>,
    /// Shared utterance cache used by the compare-side morphosyntax phase.
    pub cache: Arc<UtteranceCache>,
    /// Maximum number of file tasks to run concurrently for this job.
    pub num_workers: NumWorkers,
}

/// Shared per-file benchmark dependencies.
///
/// Benchmark dispatch needs the same server/runtime state for every file in the
/// job. Grouping that state here keeps the per-file function focused on file
/// lifecycle rather than on a wide orchestration signature.
struct BenchmarkFileContext<'a> {
    /// Immutable runner snapshot for the current job.
    job: &'a RunnerJobSnapshot,
    /// File/job lifecycle sink for runner-side status updates.
    sink: Arc<dyn RunnerEventSink>,
    /// Shared cache/worker services for transcribe + compare.
    services: PipelineServices<'a>,
    /// MWT dictionary shared with the compare pipeline.
    mwt: &'a MwtDict,
    /// Planned benchmark pairs keyed by main audio display path.
    planned_units: &'a HashMap<String, BenchmarkWorkUnit>,
}

/// Dispatch benchmark through the Rust-owned benchmark pipeline.
pub(crate) async fn dispatch_benchmark_infer(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    runtime: BenchmarkDispatchRuntime,
    plan: BenchmarkDispatchPlan,
) {
    let BenchmarkDispatchPlan {
        kernel_plan,
        base_options,
        mwt,
    } = plan;
    let planned_units: Arc<HashMap<String, BenchmarkWorkUnit>> = {
        let sink = host.sink().clone();
        match plan_work_units_for_job(crate::api::ReleasedCommand::Benchmark, job) {
            Ok(units) => Arc::new(
                units
                    .into_iter()
                    .filter_map(|unit| match unit {
                        PlannedWorkUnit::Benchmark(benchmark) => {
                            Some((benchmark.audio().display_path.to_string(), benchmark))
                        }
                        _ => None,
                    })
                    .collect(),
            ),
            Err(error) => {
                for file in &job.pending_files {
                    FileRunTracker::new(sink.as_ref(), file.filename().as_ref())
                        .fail(
                            &format!("Benchmark planning failed: {error}"),
                            FailureCategory::Validation,
                        )
                        .await;
                }
                return;
            }
        }
    };
    let sink = host.sink().clone();

    let file_parallelism = runtime
        .num_workers
        .0
        .max(1)
        .min(kernel_plan.file_parallelism_hint.max(1));
    let file_sem = Arc::new(tokio::sync::Semaphore::new(file_parallelism));
    let mut tasks = Vec::new();

    for file in &job.pending_files {
        if job.cancel_token.is_cancelled() {
            break;
        }

        let Ok(permit) = file_sem.clone().acquire_owned().await else {
            tracing::warn!("file semaphore closed during shutdown");
            break;
        };
        let sink = sink.clone();
        let pool = runtime.pool.clone();
        let cache = runtime.cache.clone();
        let job = job.clone();
        let mut opts = base_options.clone();
        let file = file.clone();
        let mwt = mwt.clone();
        let planned_units = planned_units.clone();
        let filename = file.filename().clone();

        let task: BenchmarkFileFuture = Box::pin(async move {
            let _permit = permit;
            let services = PipelineServices::new(&pool, &cache);
            let context = BenchmarkFileContext {
                job: &job,
                sink: sink.clone(),
                services,
                mwt: &mwt,
                planned_units: planned_units.as_ref(),
            };
            process_one_benchmark_file(&file, &mut opts, context).await
        });
        tasks.push(spawn_supervised_file_task(
            filename,
            "benchmark file task",
            task,
        ));
    }

    let abnormal_exits = drain_supervised_file_tasks(sink.as_ref(), &job.cancel_token, tasks).await;
    if abnormal_exits > 0 {
        warn!(
            job_id = %job.identity.job_id,
            abnormal_exits,
            "Supervised benchmark file tasks exited abnormally"
        );
    }
}

async fn process_one_benchmark_file(
    file: &crate::store::PendingJobFile,
    opts: &mut TranscribeOptions,
    context: BenchmarkFileContext<'_>,
) -> FileTaskOutcome {
    // This dispatch stays fully in Rust once the raw ASR worker capability is
    // available: resolve the gold transcript, normalize media, run the Rust
    // transcribe+compare composition, then persist the hypothesis CHAT and CSV
    // metrics artifacts.
    let BenchmarkFileContext {
        job,
        sink,
        services,
        mwt,
        planned_units,
    } = context;
    let job_id = &job.identity.job_id;
    let filename = file.filename().as_ref();
    let lifecycle = FileRunTracker::new(sink.as_ref(), filename);

    lifecycle
        .begin_first_attempt(WorkUnitKind::FileInfer, FileStage::ResolvingAudio)
        .await;

    let original_audio_path = file
        .input_path(&job.filesystem.staging_dir)
        .as_path()
        .to_owned();

    let Some(planned_unit) = planned_units.get(filename) else {
        return lifecycle
            .fail(
                &format!("Benchmark planning produced no work unit for {filename}"),
                FailureCategory::Validation,
            )
            .await
            .into();
    };

    let output_plan = match BenchmarkOutputPlan::admit(
        &job.filesystem,
        file,
        planned_unit,
        &job.dispatch.options,
        sink.as_ref(),
    ) {
        Ok(plan) => plan,
        Err(error) => {
            return lifecycle
                .fail(&error.to_string(), classify_server_error(&error))
                .await
                .into();
        }
    };

    let gold_text = match tokio::fs::read_to_string(&planned_unit.gold_chat().source_path).await {
        Ok(text) => text,
        Err(err) => {
            let err_msg = format!(
                "Failed to read benchmark reference transcript {}: {err}",
                planned_unit.gold_chat().display_path
            );
            return lifecycle
                .fail(&err_msg, FailureCategory::InputMissing)
                .await
                .into();
        }
    };

    let reference = match crate::compare::AdmittedComparisonReference::admit(&gold_text) {
        Ok(reference) => reference,
        Err(error) => {
            return lifecycle
                .fail(
                    &error.to_string(),
                    crate::runner::util::classify_server_error(&error),
                )
                .await
                .into();
        }
    };

    let media_name = preserved_media_name_for_chat(&original_audio_path, &original_audio_path);
    let prepared_media =
        match prepare_asr_media_input(original_audio_path, media_name, filename).await {
            Ok(prepared) => prepared,
            Err(error) => {
                let err_msg = error.to_string();
                return lifecycle
                    .fail(&err_msg, FailureCategory::Validation)
                    .await
                    .into();
            }
        };
    let audio_path = prepared_media.inference_audio_path;
    opts.media_name = prepared_media.media_name;

    let mut attempts =
        FileAttempts::begin(&lifecycle, WorkUnitKind::FileInfer, FileStage::Benchmarking).await;
    loop {
        let (progress_tx, wait_observer, forwarder) =
            spawn_observed_progress_forwarder(sink.clone(), filename.to_string());

        let attempt = crate::worker::pool::checkout_wait::observing_pool_waits(
            wait_observer,
            process_benchmark(BenchmarkRequest {
                audio_path: &audio_path,
                reference: reference.clone(),
                services,
                transcribe_options: opts,
                mwt,
                progress: Some(&progress_tx),
            }),
        )
        .await;
        // Every update the attempt sent is published before its next stage.
        drop(progress_tx);
        forwarder.finished().await;
        match attempt {
            Ok(outputs) => {
                lifecycle.stage(FileStage::Writing).await;

                return output_plan
                    .persist(crate::benchmark::BenchmarkOutputs {
                        annotated_main_chat: outputs.annotated_main_chat,
                        metrics: outputs.metrics,
                    })
                    .await;
            }
            Err(err) => {
                let category = classify_server_error(&err);
                let raw_msg = format!("Benchmark failed: {err}");
                warn!(
                    job_id = %job_id,
                    filename,
                    category = %category,
                    raw_error = %raw_msg,
                    "Benchmark error (raw)"
                );
                let message = user_facing_error(category, "Benchmark", filename, &raw_msg);
                attempts = match attempts.after_error(&err, category, message).await {
                    AfterFailure::Retry(grant) => grant.wait().await,
                    AfterFailure::Recorded(recorded) => return recorded.into(),
                };
            }
        }
    }
}
