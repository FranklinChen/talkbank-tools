//! Command dispatch routing: decides which dispatch family to invoke for a
//! given job, resolves the selected worker's admitted capabilities, and
//! delegates to the per-command dispatch wrappers.
//!
//! The central function is `dispatch_job_with_execution_context`, which is
//! called by `ExecutionEngine::dispatch_job` after all host-level concerns
//! (memory reservation, preflight, pre-scaling) have been handled.
//!
//! Whether a command can run at all is decided by
//! [`crate::capability::command_supported`], the same rule that decides what
//! `/health` advertises, so a command is never advertised and then refused
//! here for a reason advertisement did not check. The forced-alignment arm
//! additionally reads the FA engine the loaded worker reported
//! ([`FaCacheNamespace::from_loaded`]), because every FA cache row is
//! namespaced by it. Every refusal in this module goes through [`fail_job`].

use std::sync::Arc;

use tracing::{info, warn};

use crate::api::{NumWorkers, ReleasedCommand};
use crate::cache::UtteranceCache;
use crate::capability::command_supported;
use crate::command_model::{RunnerDispatchKind, command_runner_dispatch_kind, command_spec};
use crate::dispatch_language::DispatchLanguage;
use crate::engine_reports::FaCacheNamespace;
use crate::execution::{
    MorphotagRuntimeOptions, PooledWorkerGateway, dispatch_compare_job, dispatch_coref_job,
    dispatch_morphotag_job, dispatch_translate_job, dispatch_utseg_job,
};
use crate::store::{RunnerJobSnapshot, unix_now};
use crate::worker::pool::WorkerPool;

use super::context::{DispatchHostContext, JobDispatchRequest, RunnerExecutionContext};
use super::dispatch::{
    BatchedInferDispatchPlan, BenchmarkDispatchPlan, BenchmarkDispatchRuntime, FaDispatchPlan,
    FaDispatchRuntime, MediaAnalysisDispatchPlan, MediaAnalysisDispatchRuntime,
    TranscribeDispatchPlan, TranscribeDispatchRuntime, dispatch_benchmark_infer, dispatch_fa_infer,
    dispatch_media_analysis_v2, dispatch_transcribe_infer,
};
use super::policy::command_requires_chat_infer;
use super::test_echo::dispatch_test_echo_files;

/// Core dispatch router: resolves the selected worker's admitted capabilities,
/// applies the availability rule, selects the dispatch family (batched text,
/// FA, transcribe, benchmark, media-analysis, speaker identity, or test-echo),
/// and delegates.
pub(super) async fn dispatch_job_with_execution_context(
    request: JobDispatchRequest,
    host: &DispatchHostContext,
    execution: &RunnerExecutionContext,
) -> Result<(), crate::error::ServerError> {
    let JobDispatchRequest {
        job,
        file_list,
        num_workers,
    } = request;
    let command = job.dispatch.command;
    let pool = &execution.pool;
    let cache = &execution.cache;
    let all_chat = file_list.iter().all(|file| file.has_chat);

    if command_requires_chat_infer(command) && !all_chat {
        fail_job(
            &job,
            host,
            format!("'{command}' needs CHAT input for every file, and at least one file has none"),
        )
        .await;
        return Ok(());
    }

    // Test-echo workers answer every task and are never probed per command.
    if execution.test_echo_mode {
        dispatch_test_echo_files(&job, host.sink().as_ref(), &file_list, pool.test_delay_ms())
            .await;
        return Ok(());
    }

    // Resolve the admitted report of the exact worker key this command
    // selects, bootstrapping that worker if the pool has not probed it yet. A
    // pool-wide first-worker report can advertise task availability, but it
    // cannot identify the model behind another engine-specific key and must
    // never namespace that key's cache evidence.
    //
    // Capability discovery is language-agnostic: the worker reports its
    // resources.json, which lists every supported language regardless of which
    // lang the worker boots with. The job-level `LanguageSpec` is mapped to its
    // `WorkerLanguage` counterpart (a code-switched pair to its primary
    // language) and forwarded. The Python bootstrap recognises `auto` and `per-file` as
    // non-ISO sentinels and skips eager Stanza model load for those, see
    // `batchalign/worker/_model_loading/bootstrap.py::_load_single_task`.
    let loaded = match pool
        .ensure_command_capabilities(
            command,
            job.dispatch.lang.to_worker_language(),
            &job.dispatch.options,
        )
        .await
    {
        Ok(loaded) => loaded,
        Err(error) => {
            fail_job(
                &job,
                host,
                format!("Failed to resolve selected worker capabilities for '{command}': {error}"),
            )
            .await;
            return Ok(());
        }
    };
    if let Err(unavailable) =
        command_supported(&command_spec(command).capabilities, loaded.reports())
    {
        fail_job(&job, host, format!("cannot run '{command}': {unavailable}")).await;
        return Ok(());
    }

    let runner_dispatch_kind = command_runner_dispatch_kind(command);
    info!(
        job_id = %job.identity.job_id,
        correlation_id = %job.identity.correlation_id,
        command = %command,
        dispatch_kind = ?runner_dispatch_kind,
        "Dispatching job"
    );

    match runner_dispatch_kind {
        // Audio-first commands: they take audio input, not CHAT.
        RunnerDispatchKind::TranscribeAudioInfer => {
            dispatch_transcribe_command(&job, host, pool, cache, num_workers).await;
        }
        RunnerDispatchKind::BenchmarkAudioInfer => {
            dispatch_benchmark_command(&job, host, pool, cache, num_workers).await;
        }
        RunnerDispatchKind::MediaAnalysisV2 => {
            dispatch_media_analysis_command(&job, host, pool, cache, num_workers).await;
        }
        RunnerDispatchKind::SpeakerIdentity if all_chat => {
            crate::runner::dispatch::speaker_identity_pipeline::dispatch_speaker_identity(
                &job,
                host,
                pool.clone(),
            )
            .await;
        }
        RunnerDispatchKind::SpeakerIdentity => {
            fail_job(
                &job,
                host,
                format!(
                    "No released dispatch path remains for command '{command}' without CHAT \
                     input for every file. Legacy process-path fallback is retired."
                ),
            )
            .await;
        }
        // FA cache rows and evidence envelopes are namespaced by the engine the
        // selected worker reported after FA loaded on it. A worker that still
        // names no FA engine then is refused: there is nothing honest to
        // namespace the cache by.
        RunnerDispatchKind::ForcedAlignment => match FaCacheNamespace::from_loaded(&loaded) {
            Ok(cache_namespace) => {
                dispatch_forced_alignment_command(
                    &job,
                    host,
                    pool,
                    cache,
                    cache_namespace,
                    num_workers,
                )
                .await;
            }
            Err(unavailable) => {
                fail_job(&job, host, format!("cannot run '{command}': {unavailable}")).await;
            }
        },
        RunnerDispatchKind::BatchedTextInfer => {
            dispatch_batched_text_command(&job, host, execution, num_workers).await?;
        }
    }

    Ok(())
}

/// Fail a whole job with `message`, logged with the job's identity.
///
/// The one route every job-level refusal in this router takes, so each
/// refusal is logged and recorded the same way.
async fn fail_job(job: &RunnerJobSnapshot, host: &DispatchHostContext, message: String) {
    warn!(
        job_id = %job.identity.job_id,
        correlation_id = %job.identity.correlation_id,
        command = %job.dispatch.command,
        "{message}"
    );
    host.sink()
        .fail_job(&job.identity.job_id, &message, unix_now())
        .await;
}

/// Run one batched-text command on the recipe-owned execution path.
///
/// Matched on the (language shape, command) pair. The five batched-text
/// commands each name their arm; every other pairing falls to one arm that
/// FAILS the job.
///
/// That last arm IS a catch-all, and it has to be: the compiler cannot prove
/// that the five arms above cover every reachable pairing, because the pairing
/// is decided at runtime by `dispatch_language::language_source`. An earlier
/// version of this doc claimed there was no catch-all and that a new command
/// therefore could not compile without stating its side. That is not true, so
/// the guarantee is provided by a test instead:
/// `every_batched_text_command_has_a_named_arm` below fails if a command whose
/// catalog entry declares `BatchedTextInfer` has no arm here.
async fn dispatch_batched_text_command(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    execution: &RunnerExecutionContext,
    num_workers: NumWorkers,
) -> Result<(), crate::error::ServerError> {
    let command = job.dispatch.command;
    let plan = BatchedInferDispatchPlan::from_job(job);
    let gateway = PooledWorkerGateway::new(execution.pool.clone(), execution.cache.clone());

    // Where this command's language comes from, resolved ONCE from the command
    // itself. Each dispatcher used to ask its own version of this question of
    // the job's `LanguageSpec`, and the copies disagreed: coref is a per-file
    // command, so submission requires it to arrive as `PerFile`, while its
    // dispatch demanded a resolved code and therefore refused every coref job
    // ever submitted. A per-file command's dispatcher now takes no language at
    // all, and a job-level one takes a `JobLanguage` that exists only because
    // this resolution produced it.
    let dispatch_language = match DispatchLanguage::resolve(command, &job.dispatch.lang) {
        Ok(language) => language,
        Err(refusal) => {
            fail_job(job, host, format!("cannot run '{command}': {refusal}")).await;
            return Ok(());
        }
    };

    match (dispatch_language, command) {
        (DispatchLanguage::PerFile, ReleasedCommand::Morphotag) => {
            dispatch_morphotag_job(
                job,
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
            .await
        }
        (DispatchLanguage::Job(job_language), ReleasedCommand::Compare) => {
            dispatch_compare_job(
                job,
                host,
                &gateway,
                &plan.mwt,
                plan.should_merge_abbrev,
                &job_language,
            )
            .await
        }
        (DispatchLanguage::Job(job_language), ReleasedCommand::Utseg) => {
            // Refuse a language with no segmenter before dispatching, for the
            // same reason the transcribe plan does: whether utterance
            // segmentation can run is a property of the language and the
            // fallback policy, and both are known here. The standalone command
            // used to discover it at the worker, which named the wire format
            // rather than the missing model.
            let fallback = job.dispatch.options.utseg_fallback_policy();
            let route = match crate::utseg_route::UtsegRoute::resolve(job_language.code(), fallback)
            {
                Ok(route) => route,
                Err(unavailable) => {
                    fail_job(job, host, unavailable.to_string()).await;
                    return Ok(());
                }
            };
            dispatch_utseg_job(
                job,
                host,
                Arc::new(gateway),
                plan.should_merge_abbrev,
                &route,
            )
            .await
        }
        // Translation and coreference name their engines on every result
        // they apply, so nothing about the worker's report travels with them.
        // Neither takes a language: translate resolves one per file from that
        // file's `@Languages:` header, and coref is English-only.
        (DispatchLanguage::PerFile, ReleasedCommand::Translate) => {
            dispatch_translate_job(job, host, &gateway, plan.should_merge_abbrev).await
        }
        (DispatchLanguage::PerFile, ReleasedCommand::Coref) => {
            dispatch_coref_job(job, host, &gateway, plan.should_merge_abbrev).await
        }
        // A programming error, not a user error, in one of two ways: a command
        // whose catalog entry declares `BatchedTextInfer` has no arm above, or
        // its declared shape in `dispatch_language::language_source` disagrees
        // with the arm written for it. Changing one without the other is the
        // drift that made coref undispatchable, so both halves are named.
        //
        // It FAILS the job. Returning `Ok(())` would leave the job sitting in
        // `Running` with nothing dispatched and nothing to reconcile it.
        //
        // This replaced two adjacent arms, an eight-command enumeration and a
        // shape-mismatch arm, which reported the same thing twice: the
        // enumeration was absorbed by the arm that followed it, so it was
        // ceremony rather than the exhaustiveness guarantee it looked like.
        (language, command) => {
            let shape = match language {
                DispatchLanguage::PerFile => "per-file",
                DispatchLanguage::Job(_) => "job-level",
            };
            fail_job(
                job,
                host,
                format!(
                    "No dispatch arm handled command '{command}' with a {shape} language shape \
                     on the batched-text path. Either its catalog entry declares \
                     BatchedTextInfer without a text arm in runner::routing, or its declared \
                     shape in dispatch_language disagrees with that arm."
                ),
            )
            .await;
            Ok(())
        }
    }
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
    cache_namespace: FaCacheNamespace,
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
            cache_namespace,
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

    /// `dispatch_batched_text_command` gives each batched-text command a
    /// named arm and lists every other command as having none. This test is
    /// the other half of that agreement: every command the CATALOG declares
    /// as `BatchedTextInfer` must be one of the commands with an arm, because
    /// the two are separate facts (the catalog's declared kinds, and the set
    /// of names the router matches) that nothing else checks against each
    /// other.
    ///
    /// If it fails because a new command declares `BatchedTextInfer`, give
    /// that command an arm on the recipe-owned path. Nothing silent happens if
    /// you forget: the router fails the job with a message naming the
    /// declared dispatch kind. This test finds it at `cargo test` time instead
    /// of by a user whose job failed.
    #[test]
    fn every_batched_text_command_has_a_named_arm() {
        // Matched on the enum with no catch-all, so a new released command
        // cannot be added without stating which side of this it falls on.
        for command in ReleasedCommand::ALL {
            let has_text_arm = match command {
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
                    has_text_arm,
                    "{command} declares BatchedTextInfer but the router has no text arm for \
                     it, so every job for it would fail. Give it an arm on the recipe-owned \
                     execution path."
                );
            }
        }
    }
}
