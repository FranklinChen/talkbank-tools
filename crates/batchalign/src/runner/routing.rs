//! Command dispatch routing: decides which dispatch family to invoke for a
//! given job, resolves runtime worker capabilities, and delegates to the
//! per-command dispatch wrappers.
//!
//! The central function is `dispatch_job_with_execution_context`, which is
//! called by `ExecutionEngine::dispatch_job` after all host-level concerns
//! (memory reservation, preflight, pre-scaling) have been handled.

use std::collections::BTreeMap;
use std::sync::Arc;

use tracing::{info, warn};

use crate::api::{EngineVersion, NumWorkers, ReleasedCommand};
use crate::cache::UtteranceCache;

use crate::capability::resolve_worker_capability_snapshot;
use crate::command_model::{RunnerDispatchKind, command_runner_dispatch_kind};
use crate::execution::{
    MorphotagRuntimeOptions, PooledWorkerGateway, dispatch_compare_job, dispatch_coref_job,
    dispatch_morphotag_job, dispatch_translate_job, dispatch_utseg_job,
};
use crate::store::{RunnerJobSnapshot, unix_now};
use crate::worker::InferTask;
use crate::worker::pool::WorkerPool;
use crate::worker::target::task_name as infer_task_name;

use super::context::{DispatchHostContext, JobDispatchRequest, RunnerExecutionContext};
use super::dispatch::{
    BatchedInferDispatchPlan, BenchmarkDispatchPlan, BenchmarkDispatchRuntime, FaDispatchPlan,
    FaDispatchRuntime, MediaAnalysisDispatchPlan, MediaAnalysisDispatchRuntime,
    TranscribeDispatchPlan, TranscribeDispatchRuntime, dispatch_benchmark_infer, dispatch_fa_infer,
    dispatch_media_analysis_v2, dispatch_transcribe_infer,
};
use super::policy::{command_requires_chat_infer, infer_task_for_command};
use super::test_echo::dispatch_test_echo_files;

/// Core dispatch router: resolves capabilities, selects the right dispatch
/// family (batched text, FA, transcribe, benchmark, media-analysis, or
/// test-echo), and delegates.
pub(super) async fn dispatch_job_with_execution_context(
    request: JobDispatchRequest,
    host: &DispatchHostContext,
    execution: &RunnerExecutionContext,
) -> Result<(), crate::error::ServerError> {
    let sink = host.sink().clone();
    let JobDispatchRequest {
        job,
        file_list,
        num_workers,
    } = request;
    let job_id = &job.identity.job_id;
    let correlation_id = job.identity.correlation_id.clone();
    let command = job.dispatch.command;
    let pool = &execution.pool;
    let cache = &execution.cache;
    let startup_infer_tasks = &execution.infer_tasks;
    let startup_engine_versions = &execution.engine_versions;
    let test_echo_mode = execution.test_echo_mode;
    // Capability discovery is language-agnostic, the worker reports its
    // resources.json which lists every supported language regardless of
    // which lang the worker boots with. The job-level `LanguageSpec`
    // (`Resolved(_)`, `Auto`, `PerFile`) is mapped to its `WorkerLanguage`
    // counterpart and forwarded as-is. The Python bootstrap recognises
    // `auto` and `per-file` as non-ISO sentinels and skips eager Stanza
    // model load for those, see
    // `batchalign/worker/_model_loading/bootstrap.py::_load_single_task`.
    // No English fallback here; the typed sum carries through.
    let capability_snapshot = match resolve_runtime_capability_snapshot(
        pool,
        startup_infer_tasks,
        startup_engine_versions,
        test_echo_mode,
        command,
        job.dispatch.lang.to_worker_language(),
        &job.dispatch.options,
    )
    .await
    {
        Ok(snapshot) => snapshot,
        Err(err_msg) => {
            warn!(job_id = %job_id, correlation_id = %correlation_id, "{}", err_msg);
            sink.fail_job(job_id, &err_msg, unix_now()).await;
            return Ok(());
        }
    };
    let infer_tasks = &capability_snapshot.infer_tasks;
    let engine_versions = &capability_snapshot.engine_versions;

    let all_chat = file_list.iter().all(|file| file.has_chat);
    let infer_task = infer_task_for_command(command);
    let infer_supported = infer_tasks.contains(&infer_task);
    let use_infer = all_chat && infer_supported;

    if command_requires_chat_infer(command) && !use_infer {
        let required_task = infer_task_name(infer_task);
        let err_msg = format!(
            "Rust-first dispatch requires infer task '{}' for '{}' (all_chat={}). \
             Worker advertises infer_tasks: {:?}",
            required_task, command, all_chat, infer_tasks
        );
        warn!(job_id = %job_id, correlation_id = %correlation_id, "{}", err_msg);
        let failed_at = unix_now();
        sink.fail_job(job_id, &err_msg, failed_at).await;
        return Ok(());
    }

    // Special case: transcribe/transcribe_s with server-side ASR orchestration.
    // These commands take audio input (not CHAT), so they do not go through the
    // standard `use_infer` path which requires all_chat=true.
    let runner_dispatch_kind = command_runner_dispatch_kind(command);
    let use_transcribe_infer = matches!(
        runner_dispatch_kind,
        RunnerDispatchKind::TranscribeAudioInfer
    ) && infer_tasks.contains(&InferTask::Asr);
    let use_benchmark_infer = matches!(
        runner_dispatch_kind,
        RunnerDispatchKind::BenchmarkAudioInfer
    ) && infer_tasks.contains(&InferTask::Asr);
    let use_media_analysis_infer =
        matches!(runner_dispatch_kind, RunnerDispatchKind::MediaAnalysisV2)
            && infer_tasks.contains(&infer_task);

    if test_echo_mode {
        dispatch_test_echo_files(&job, sink.as_ref(), &file_list, pool.test_delay_ms()).await;
    } else if use_transcribe_infer {
        let engine_version = EngineVersion::from(
            engine_versions
                .get("asr")
                .map(|s| s.as_str())
                .unwrap_or("unknown"),
        );

        info!(
            job_id = %job_id,
            correlation_id = %correlation_id,
            command = %command,
            engine_version = %engine_version,
            "Using server-side transcribe orchestrator"
        );

        dispatch_transcribe_command(&job, host, pool, cache, &engine_version, num_workers).await;
    } else if use_benchmark_infer {
        let engine_version = EngineVersion::from(
            engine_versions
                .get("asr")
                .map(|s| s.as_str())
                .unwrap_or("unknown"),
        );

        info!(
            job_id = %job_id,
            correlation_id = %correlation_id,
            command = %command,
            engine_version = %engine_version,
            "Using server-side benchmark orchestrator"
        );

        dispatch_benchmark_command(&job, host, pool, cache, &engine_version, num_workers).await;
    } else if use_media_analysis_infer {
        let engine_version = EngineVersion::from(
            engine_versions
                .get(infer_task_name(infer_task))
                .map(|s| s.as_str())
                .unwrap_or("unknown"),
        );
        info!(
            job_id = %job_id,
            correlation_id = %correlation_id,
            command = %command,
            engine_version = %engine_version,
            "Using server-side media-analysis V2 path"
        );

        dispatch_media_analysis_command(&job, host, pool, cache, num_workers).await;
    } else if use_infer && command == ReleasedCommand::Morphotag {
        let engine_version = EngineVersion::from(
            engine_versions
                .get(infer_task_name(InferTask::Morphosyntax))
                .map(|s| s.as_str())
                .unwrap_or("unknown"),
        );
        let plan = BatchedInferDispatchPlan::from_job(&job);
        let gateway = PooledWorkerGateway::new(pool.clone(), cache.clone(), engine_version.clone());
        info!(
            job_id = %job_id,
            correlation_id = %correlation_id,
            command = %command,
            engine_version = %engine_version,
            "Using recipe-owned morphotag execution path"
        );
        dispatch_morphotag_job(
            &job,
            host,
            Arc::new(gateway),
            MorphotagRuntimeOptions {
                tokenization_mode: plan.tokenization_mode,
                multilingual_policy: plan.multilingual_policy,
                mwt: Arc::new(plan.mwt),
                l2_policy: plan.l2_policy,
                pos_hint_policy: plan.pos_hint_policy,
                ca_policy: plan.ca_policy,
                should_merge_abbrev: plan.should_merge_abbrev,
                review_level: plan.review_level,
            },
            num_workers,
        )
        .await?;
    } else if use_infer && command == ReleasedCommand::Compare {
        let engine_version = EngineVersion::from(
            engine_versions
                .get(infer_task_name(InferTask::Morphosyntax))
                .map(|s| s.as_str())
                .unwrap_or("unknown"),
        );
        let plan = BatchedInferDispatchPlan::from_job(&job);
        let gateway = PooledWorkerGateway::new(pool.clone(), cache.clone(), engine_version.clone());
        info!(
            job_id = %job_id,
            correlation_id = %correlation_id,
            command = %command,
            engine_version = %engine_version,
            "Using recipe-owned compare execution kernel"
        );
        dispatch_compare_job(&job, host, &gateway, &plan.mwt, plan.should_merge_abbrev).await?;
    } else if use_infer && command == ReleasedCommand::Utseg {
        let engine_version = EngineVersion::from(
            engine_versions
                .get(infer_task_name(InferTask::Utseg))
                .map(|s| s.as_str())
                .unwrap_or("unknown"),
        );
        let plan = BatchedInferDispatchPlan::from_job(&job);
        let gateway: std::sync::Arc<dyn crate::execution::WorkerGateway> = std::sync::Arc::new(
            PooledWorkerGateway::new(pool.clone(), cache.clone(), engine_version.clone()),
        );
        info!(
            job_id = %job_id,
            correlation_id = %correlation_id,
            command = %command,
            engine_version = %engine_version,
            "Using recipe-owned utseg execution path"
        );
        dispatch_utseg_job(
            &job,
            host,
            gateway,
            plan.should_merge_abbrev,
            job.dispatch.options.utseg_fallback_policy().is_allowed(),
        )
        .await?;
    } else if use_infer && command == ReleasedCommand::Translate {
        let engine_version = EngineVersion::from(
            engine_versions
                .get(infer_task_name(InferTask::Translate))
                .map(|s| s.as_str())
                .unwrap_or("unknown"),
        );
        let plan = BatchedInferDispatchPlan::from_job(&job);
        let gateway = PooledWorkerGateway::new(pool.clone(), cache.clone(), engine_version.clone());
        info!(
            job_id = %job_id,
            correlation_id = %correlation_id,
            command = %command,
            engine_version = %engine_version,
            "Using recipe-owned translate execution path"
        );
        dispatch_translate_job(&job, host, &gateway, plan.should_merge_abbrev).await?;
    } else if use_infer && command == ReleasedCommand::Coref {
        let engine_version = EngineVersion::from(
            engine_versions
                .get(infer_task_name(InferTask::Coref))
                .map(|s| s.as_str())
                .unwrap_or("unknown"),
        );
        let plan = BatchedInferDispatchPlan::from_job(&job);
        let gateway = PooledWorkerGateway::new(pool.clone(), cache.clone(), engine_version.clone());
        info!(
            job_id = %job_id,
            correlation_id = %correlation_id,
            command = %command,
            engine_version = %engine_version,
            "Using recipe-owned coref execution path"
        );
        dispatch_coref_job(&job, host, &gateway, plan.should_merge_abbrev).await?;
    } else if use_infer {
        // --- Server-side infer path ---
        // The server owns CHAT parse/cache/inject/serialize.
        // Python workers provide pure Stanza inference only.
        let engine_version = EngineVersion::from(
            engine_versions
                .get(infer_task_name(infer_task))
                .map(|s| s.as_str())
                .unwrap_or("unknown"),
        );

        info!(
            job_id = %job_id,
            correlation_id = %correlation_id,
            command = %command,
            engine_version = %engine_version,
            "Using server-side infer path"
        );

        match runner_dispatch_kind {
            RunnerDispatchKind::SpeakerIdentity => {
                crate::runner::dispatch::speaker_identity_pipeline::dispatch_speaker_identity(
                    &job,
                    host,
                    pool.clone(),
                )
                .await;
            }
            RunnerDispatchKind::ForcedAlignment => {
                dispatch_forced_alignment_command(
                    &job,
                    host,
                    pool,
                    cache,
                    &engine_version,
                    num_workers,
                )
                .await;
            }
            // Every other dispatch kind is claimed by a name-matched arm
            // earlier in this chain, so arriving here means the catalog
            // declares a kind that no arm handles: a programming error, not a
            // user error.
            //
            // It FAILS the job. The previous code logged an error and returned
            // `Ok(())`, which left the job sitting in `Running` with nothing
            // dispatched and nothing that would ever reconcile it, since
            // recovery only revisits rows it can tell are abandoned. A loud
            // failure is the only honest outcome for a command the router
            // cannot route.
            //
            // Spelled out variant by variant with no catch-all, so a new
            // `RunnerDispatchKind` cannot be introduced without stating which
            // side of this it falls on. `BatchedTextInfer` belongs on this side
            // now: the five released batched-text commands are all intercepted
            // by name above (pinned by this module's test), and the legacy
            // batched-text dispatch they used to fall through to is gone.
            kind @ (RunnerDispatchKind::BatchedTextInfer
            | RunnerDispatchKind::TranscribeAudioInfer
            | RunnerDispatchKind::BenchmarkAudioInfer
            | RunnerDispatchKind::MediaAnalysisV2) => {
                let err_msg = format!(
                    "No dispatch arm handled command '{command}' on the infer path; it declares \
                     dispatch kind {kind:?}, which reaches this point only if the command has no \
                     name-matched arm in runner::routing."
                );
                tracing::error!(
                    job_id = %job_id,
                    correlation_id = %correlation_id,
                    command = %command,
                    runner_dispatch_kind = ?kind,
                    "{}", err_msg
                );
                sink.fail_job(job_id, &err_msg, unix_now()).await;
                return Ok(());
            }
        }
    } else {
        let err_msg = format!(
            "No released dispatch path remains for command '{}' (all_chat={}, infer_task={:?}, infer_supported={}). Legacy process-path fallback is retired.",
            command, all_chat, infer_task, infer_supported
        );
        warn!(job_id = %job_id, correlation_id = %correlation_id, "{}", err_msg);
        sink.fail_job(job_id, &err_msg, unix_now()).await;
        return Ok(());
    }

    Ok(())
}

/// Fail every pending file of a job whose plan was refused.
///
/// A `warn!` and a bare `return` used to be the whole response, so a job
/// naming an unimplemented engine vanished: no file failed, no error was
/// recorded, and the only trace was one log line nobody was reading. The
/// refusal is a property of the request, so its files fail with
/// `FailureCategory::Validation` and carry the refusal's own message.
pub(super) async fn fail_files_for_refused_plan(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    refusal: &super::dispatch::DispatchPlanRefusal,
) {
    let message = refusal.to_string();
    warn!(
        job_id = %job.identity.job_id,
        correlation_id = %job.identity.correlation_id,
        command = %job.dispatch.command,
        refusal = %message,
        "Command plan refused; failing this job's files"
    );
    let refused_at = unix_now();
    for file in &job.pending_files {
        record_plan_refusal_for_file(
            host.sink().as_ref(),
            &job.identity.job_id,
            file.filename.as_ref(),
            &message,
            refused_at,
        )
        .await;
    }
}

/// Record one file's plan refusal against a sink.
///
/// Split out from the loop above so the behaviour can be observed with a
/// recording sink: the whole point of this function is WHICH store writes it
/// makes, and a job snapshot plus a host context is not needed to check that.
///
/// It uses `record_setup_failure`, not a bare `fail`. This is a preflight
/// rejection, so no attempt has been opened for the file, and `fail` alone
/// records a terminal state with no attempt behind it: the operator gets an
/// error with no attempt history. `record_setup_failure` documents itself as
/// the route for exactly this case, and it opens a `FileSetup` attempt without
/// advertising the file as actively processing.
async fn record_plan_refusal_for_file(
    sink: &dyn super::util::RunnerEventSink,
    job_id: &crate::api::JobId,
    filename: &str,
    message: &str,
    refused_at: crate::api::UnixTimestamp,
) {
    super::util::FileRunTracker::new(sink, job_id, filename)
        .record_setup_failure(
            refused_at,
            message,
            crate::scheduling::FailureCategory::Validation,
            refused_at,
        )
        .await;
}

/// Build one command's dispatch plan, or fail this job's files and answer
/// `None`.
///
/// The four audio dispatchers below each repeated the same six lines: match the
/// plan constructor, and on a refusal call `fail_files_for_refused_plan` and
/// return. One of them getting that wrong is a job that vanishes with nothing
/// recorded, which is the exact failure `DispatchPlanRefusal` was introduced to
/// stop, so the response lives in one place and the call sites state only which
/// plan they want.
async fn plan_or_fail_files<Plan>(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    from_job: impl FnOnce(
        &RunnerJobSnapshot,
        &crate::config::ServerConfig,
    ) -> Result<Plan, super::dispatch::DispatchPlanRefusal>,
) -> Option<Plan> {
    match from_job(job, host.config()) {
        Ok(plan) => Some(plan),
        Err(refusal) => {
            fail_files_for_refused_plan(job, host, &refusal).await;
            None
        }
    }
}

async fn dispatch_forced_alignment_command(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    pool: &Arc<WorkerPool>,
    cache: &Arc<UtteranceCache>,
    engine_version: &EngineVersion,
    num_workers: NumWorkers,
) {
    let Some(plan) = plan_or_fail_files(job, host, FaDispatchPlan::from_job).await else {
        return;
    };

    dispatch_fa_infer(
        job,
        host,
        FaDispatchRuntime {
            pool: pool.clone(),
            cache: cache.clone(),
            engine_version: engine_version.clone(),
            num_workers,
        },
        plan,
    )
    .await;
}

async fn dispatch_transcribe_command(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    pool: &Arc<WorkerPool>,
    cache: &Arc<UtteranceCache>,
    engine_version: &EngineVersion,
    num_workers: NumWorkers,
) {
    let Some(plan) = plan_or_fail_files(job, host, TranscribeDispatchPlan::from_job).await else {
        return;
    };

    dispatch_transcribe_infer(
        job,
        host,
        TranscribeDispatchRuntime {
            pool: pool.clone(),
            cache: cache.clone(),
            engine_version: engine_version.clone(),
            num_workers,
        },
        plan,
    )
    .await;
}

async fn dispatch_benchmark_command(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    pool: &Arc<WorkerPool>,
    cache: &Arc<UtteranceCache>,
    engine_version: &EngineVersion,
    num_workers: NumWorkers,
) {
    let Some(plan) = plan_or_fail_files(job, host, BenchmarkDispatchPlan::from_job).await else {
        return;
    };

    dispatch_benchmark_infer(
        job,
        host,
        BenchmarkDispatchRuntime {
            pool: pool.clone(),
            cache: cache.clone(),
            engine_version: engine_version.clone(),
            num_workers,
        },
        plan,
    )
    .await;
}

async fn dispatch_media_analysis_command(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    pool: &Arc<WorkerPool>,
    cache: &Arc<UtteranceCache>,
    num_workers: NumWorkers,
) {
    let Some(plan) = plan_or_fail_files(job, host, MediaAnalysisDispatchPlan::from_job).await
    else {
        return;
    };

    dispatch_media_analysis_v2(
        job,
        host,
        MediaAnalysisDispatchRuntime {
            pool: pool.clone(),
            cache: cache.clone(),
            num_workers,
        },
        plan,
    )
    .await;
}

/// Resolve a runtime capability snapshot, bootstrapping live capabilities from
/// a worker if the pool has not yet detected them.
async fn resolve_runtime_capability_snapshot(
    pool: &WorkerPool,
    startup_infer_tasks: &[InferTask],
    startup_engine_versions: &BTreeMap<String, String>,
    test_echo_mode: bool,
    command: ReleasedCommand,
    lang: impl Into<crate::api::WorkerLanguage>,
    options: &crate::options::CommandOptions,
) -> Result<crate::capability::WorkerCapabilitySnapshot, String> {
    // Resolve capabilities from the exact worker key selected by this command.
    // A pool-wide first-worker snapshot can advertise task availability, but
    // it cannot identify the model behind another engine-specific key and must
    // never namespace that key's cache evidence.
    let selected_worker_capabilities = if test_echo_mode {
        None
    } else {
        Some(
            pool.ensure_command_capabilities(command, lang, options)
                .await
                .map_err(|error| {
                    format!(
                        "Failed to resolve selected worker capabilities for '{}': {}",
                        command, error
                    )
                })?,
        )
    };

    resolve_worker_capability_snapshot(
        &[],
        startup_infer_tasks,
        startup_engine_versions,
        test_echo_mode,
        selected_worker_capabilities.as_ref(),
    )
    .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::record_plan_refusal_for_file;
    use crate::ReleasedCommand;
    use crate::api::JobId;
    use crate::command_model::{RunnerDispatchKind, command_runner_dispatch_kind};
    use crate::runner::util::test_sink::RecordingSink;
    use crate::scheduling::{FailureCategory, WorkUnitKind};

    /// RED FIRST (review item 3): a refused plan used to call `fail` with no
    /// `start_file_attempt` before it, so the file went terminal with no
    /// attempt history at all. `record_setup_failure` is the documented route
    /// for a preflight rejection that still wants one.
    #[tokio::test]
    async fn a_refused_plan_records_a_setup_attempt_before_failing_the_file() {
        let sink = RecordingSink::default();
        record_plan_refusal_for_file(
            &sink,
            &JobId::from("job-refused"),
            "sample.cha",
            "command plan could not be built from job options",
            crate::api::UnixTimestamp(1_700_000_000.0),
        )
        .await;

        let attempts = sink.attempts();
        assert_eq!(
            attempts.len(),
            1,
            "a refused file must still get an attempt: {attempts:?}"
        );
        assert_eq!(attempts[0].filename, "sample.cha");
        assert_eq!(attempts[0].work_unit_kind, WorkUnitKind::FileSetup);

        let errors = sink.errors();
        assert_eq!(errors.len(), 1, "the file must also fail: {errors:?}");
        assert_eq!(errors[0].category, FailureCategory::Validation);
        assert!(
            errors[0].error.contains("could not be built"),
            "the failure must carry the refusal's own message, got: {}",
            errors[0].error
        );
    }

    /// The dispatch chain above intercepts each batched-text command by NAME
    /// (`else if use_infer && command == ReleasedCommand::Morphotag`, and four
    /// more) before reaching the generic `match runner_dispatch_kind` arm at the
    /// end. Every one of those arms runs the recipe-owned stack in
    /// `crate::execution`.
    ///
    /// This test is what allowed the legacy batched-text dispatch module
    /// (`runner/dispatch/infer_batched.rs`) to be deleted: it proved the
    /// module was unreachable, which is a property of two things AGREEING (the
    /// catalog's declared dispatch kinds, and the set of names the chain
    /// matches) that nothing was checking. It stays because that agreement can
    /// still be broken by adding a command.
    ///
    /// If it fails because a new command declares `BatchedTextInfer`, the fix
    /// is to give that command a name-matched arm on the recipe-owned path.
    /// Nothing silent happens if you forget: the generic arm now fails the job
    /// with a message naming the declared dispatch kind. This test exists so
    /// the problem is found at `cargo test` time instead of by a user whose
    /// job failed.
    #[test]
    fn every_batched_text_command_is_intercepted_before_the_legacy_arm() {
        // Matched on the enum with no catch-all, so a new released command
        // cannot be added without stating which side of this it falls on.
        for command in ReleasedCommand::ALL {
            let intercepted_by_name = match command {
                ReleasedCommand::Morphotag
                | ReleasedCommand::Utseg
                | ReleasedCommand::Translate
                | ReleasedCommand::Coref
                | ReleasedCommand::Compare => true,
                ReleasedCommand::Transcribe
                | ReleasedCommand::TranscribeS
                | ReleasedCommand::Benchmark
                | ReleasedCommand::Opensmile
                | ReleasedCommand::Avqi
                | ReleasedCommand::Diarize
                | ReleasedCommand::SpeakerIdentify
                | ReleasedCommand::Align => false,
            };

            if command_runner_dispatch_kind(command) == RunnerDispatchKind::BatchedTextInfer {
                assert!(
                    intercepted_by_name,
                    "{command} declares BatchedTextInfer but the dispatch chain has no \
                     name-matched arm for it, so every job for it would fail at the \
                     generic arm. Give it an arm on the recipe-owned execution path."
                );
            }
        }
    }
}
