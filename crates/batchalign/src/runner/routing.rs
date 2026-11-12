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
//! here for a reason advertisement did not check. Forced alignment is probed
//! per file language instead, inside its dispatch
//! ([`super::dispatch::FaWorkerAdmission`]): align is a per-file command, and
//! the worker a file reaches is keyed by the language that file declares, so
//! there is no job-level worker to read the FA engine (the cache namespace)
//! off. Every job-level refusal goes through [`refuse`] and returns to the
//! run scope, whose claim fails the run with that reason.

use std::sync::Arc;

use tracing::{info, warn};

use crate::api::{NumWorkers, ReleasedCommand};
use crate::command_model::{RunnerDispatchKind, command_runner_dispatch_kind};
use crate::dispatch_language::DispatchLanguage;
use crate::execution::{
    MorphotagRuntimeOptions, PooledWorkerGateway, dispatch_compare_job, dispatch_coref_job,
    dispatch_morphotag_job, dispatch_translate_job, dispatch_utseg_job,
};
use crate::store::RunnerJobSnapshot;
use crate::worker::pool::GpuModelPlan;

use super::context::{
    DispatchEnd, DispatchHostContext, JobDispatchRequest, JobRefusal, RunnerExecutionContext,
};
use super::dispatch::{
    BatchedInferDispatchPlan, BenchmarkDispatchPlan, BenchmarkDispatchRuntime, FaDispatchPlan,
    FaDispatchRuntime, MediaAnalysisDispatchPlan, MediaAnalysisDispatchRuntime,
    TranscribeDispatchPlan, TranscribeDispatchRuntime, dispatch_benchmark_infer, dispatch_fa_infer,
    dispatch_media_analysis_v2, dispatch_transcribe_infer, pre_scale_workers, probe_command_worker,
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
) -> Result<DispatchEnd, crate::error::ServerError> {
    let JobDispatchRequest {
        job,
        file_list,
        num_workers,
    } = request;
    let command = job.dispatch.command;
    let pool = &execution.pool;
    let cache = &execution.cache;
    let all_chat = file_list
        .iter()
        .all(|file| matches!(file.kind(), crate::store::InputKind::Chat));

    // Native work shares the managed runner's admission and supervision, but
    // cannot pass through test echo or probing an invented worker task.
    if command_runner_dispatch_kind(command) == RunnerDispatchKind::NativeAudioExport {
        return Ok(crate::execution::native_media::dispatch_audio_export(
            &job,
            host,
            &file_list,
            num_workers,
        )
        .await);
    }

    if command_requires_chat_infer(command) && !all_chat {
        return Ok(DispatchEnd::Refused(refuse(
            &job,
            format!("'{command}' needs CHAT input for every file, and at least one file has none"),
        )));
    }

    // Test-echo workers answer every task and are never probed per command.
    if execution.test_echo_mode {
        dispatch_test_echo_files(&job, host.sink().as_ref(), &file_list, pool.test_delay_ms())
            .await;
        return Ok(DispatchEnd::Dispatched);
    }

    // The job's family plan is built BEFORE any worker is probed. It owns the
    // engines the job will run, and the probe must spawn the worker those
    // engines select: probing on the command's main task alone used to give an
    // align job an FA-only worker beside the one its UTR requests needed. A
    // refused plan also fails the job before a model is loaded for it.
    let runner_dispatch_kind = command_runner_dispatch_kind(command);
    let family = match FamilyPlan::build(
        &job,
        host,
        runner_dispatch_kind,
        all_chat,
        pool.tencent_engine(),
    )
    .await
    {
        Ok(family) => family,
        // Each file already recorded its own refusal; the run finishes them.
        Err(PlanStop::FilesRefused) => return Ok(DispatchEnd::Dispatched),
        Err(PlanStop::JobRefused(refusal)) => return Ok(DispatchEnd::Refused(refusal)),
    };

    info!(
        job_id = %job.identity.job_id,
        correlation_id = %job.identity.correlation_id,
        command = %command,
        dispatch_kind = ?runner_dispatch_kind,
        "Dispatching job"
    );

    let family = match family {
        // Align probes per file language inside its dispatch: each file's
        // worker is keyed by the language that file declares, and the FA
        // engine that worker reports namespaces that file's cache rows. The
        // job itself carries no language to probe on.
        FamilyPlan::PerFileAudio(plan) => {
            dispatch_fa_infer(
                &job,
                host,
                FaDispatchRuntime {
                    pool: pool.clone(),
                    cache: cache.clone(),
                    num_workers,
                },
                plan,
            )
            .await;
            return Ok(DispatchEnd::Dispatched);
        }
        FamilyPlan::JobLevel(family) => family,
    };

    // Probe the exact worker key this command selects, bootstrapping that
    // worker if the pool has not probed it yet, and apply the availability
    // rule to its post-load report. A pool-wide first-worker report can
    // advertise task availability, but it cannot identify the model behind
    // another engine-specific key and must never namespace that key's cache
    // evidence.
    //
    // Capability discovery is language-agnostic: the worker reports its
    // resources.json, which lists every supported language regardless of which
    // lang the worker boots with. The job-level `LanguageSpec` is mapped to its
    // `WorkerLanguage` counterpart (a code-switched pair to its primary
    // language) and forwarded; a per-file text command forwards the `per-file`
    // sentinel, which the Python bootstrap recognises, with `auto`, as a
    // non-ISO value and skips eager Stanza model load for, see
    // `batchalign/worker/_model_loading/bootstrap.py::record_load`. A
    // command that owns its language (coref) is probed in that language
    // (`dispatch_language::probe_language`), because its requests are keyed
    // by it: the probe's `ensure_task` then prepares the worker they reach.
    let gpu_models = family.gpu_models();
    match probe_command_worker(
        pool,
        command,
        crate::dispatch_language::probe_language(command, &job.dispatch.lang),
        &job.dispatch.options,
        &gpu_models,
        crate::infer_retry::Cancellation::Token(&job.cancel_token),
    )
    .await
    {
        Ok(_) => {}
        // The run scope owns cancellation and records the job's end; there
        // is nothing to refuse.
        Err(crate::runner::dispatch::WorkerProbeRefusal::Cancelled) => {
            return Ok(DispatchEnd::Dispatched);
        }
        Err(refusal) => {
            return Ok(DispatchEnd::Refused(refuse(
                &job,
                format!("cannot run '{command}': {refusal}"),
            )));
        }
    }
    // Skipped for per-file text commands (morphotag, translate, coref): they
    // have no job-level language, and pre-scaling against a placeholder would
    // spawn a worker no file is dispatched to; their pipelines spawn
    // per-language workers as they walk the files.
    if !job.dispatch.lang.is_per_file() {
        pre_scale_workers(
            pool,
            command,
            job.dispatch.lang.to_worker_language(),
            num_workers,
            &job.dispatch.options,
            &gpu_models,
        )
        .await;
    }

    match family {
        // Audio-first commands: they take audio input, not CHAT.
        JobLevelFamily::Transcribe(plan) => {
            dispatch_transcribe_infer(
                &job,
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
        JobLevelFamily::Benchmark(plan) => {
            dispatch_benchmark_infer(
                &job,
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
        JobLevelFamily::MediaAnalysis(plan) => {
            dispatch_media_analysis_v2(
                &job,
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
        JobLevelFamily::SpeakerIdentity => {
            return Ok(
                crate::runner::dispatch::speaker_identity_pipeline::dispatch_speaker_identity(
                    &job,
                    host,
                    pool.clone(),
                )
                .await,
            );
        }
        JobLevelFamily::BatchedText => {
            return dispatch_batched_text_command(&job, host, execution, num_workers).await;
        }
    }

    Ok(DispatchEnd::Dispatched)
}

/// Refuse a whole job with `message`, logged with the job's identity.
///
/// The one route every job-level refusal in this router takes. It records
/// nothing: the refusal travels back to the run scope, whose claim fails the
/// run with this reason on the job and on every file left unfinished.
pub(crate) fn refuse(job: &RunnerJobSnapshot, message: String) -> JobRefusal {
    warn!(
        job_id = %job.identity.job_id,
        correlation_id = %job.identity.correlation_id,
        command = %job.dispatch.command,
        "{message}"
    );
    JobRefusal::new(message)
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
) -> Result<DispatchEnd, crate::error::ServerError> {
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
            return Ok(DispatchEnd::Refused(refuse(
                job,
                format!("cannot run '{command}': {refusal}"),
            )));
        }
    };

    let dispatched = match (dispatch_language, command) {
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
                    review_level: plan.review_level,
                },
                num_workers,
            )
            .await
        }
        // Compare takes no language either: its internal morphotag of each
        // main transcript reads that file's `@Languages:` header.
        (DispatchLanguage::PerFile, ReleasedCommand::Compare) => {
            dispatch_compare_job(job, host, &gateway, &plan.mwt).await
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
                    return Ok(DispatchEnd::Refused(refuse(job, unavailable.to_string())));
                }
            };
            // The Stanza route reads constituency trees, so a language whose
            // installed catalog entry has no constituency parser is refused
            // here, before any file is read, instead of failing every
            // multi-word item at the worker. The capability probe above has
            // reported the catalog this reads.
            let route = match route.admit(&execution.pool.stanza_registry()) {
                Ok(route) => route,
                Err(unservable) => {
                    return Ok(DispatchEnd::Refused(refuse(job, unservable.to_string())));
                }
            };
            dispatch_utseg_job(job, host, Arc::new(gateway), &route).await
        }
        // Translation and coreference name their engines on every result
        // they apply, so nothing about the worker's report travels with them.
        // Neither takes a language: translate resolves one per file from that
        // file's `@Languages:` header, and coref is English-only.
        (DispatchLanguage::PerFile, ReleasedCommand::Translate) => {
            dispatch_translate_job(job, host, &gateway).await
        }
        // Its English requests reach the English worker the probe prepared
        // (`dispatch_language::probe_language`).
        (DispatchLanguage::Fixed(_), ReleasedCommand::Coref) => {
            dispatch_coref_job(job, host, &gateway).await
        }
        // A programming error, not a user error, in one of two ways: a command
        // whose catalog entry declares `BatchedTextInfer` has no arm above, or
        // its declared shape in `dispatch_language::language_source` disagrees
        // with the arm written for it. Changing one without the other is the
        // drift that made coref undispatchable, so both halves are named.
        //
        // It REFUSES the job: the run fails with this reason. Treating it as
        // dispatched would end the job with no reason recorded.
        //
        // This replaced two adjacent arms, an eight-command enumeration and a
        // shape-mismatch arm, which reported the same thing twice: the
        // enumeration was absorbed by the arm that followed it, so it was
        // ceremony rather than the exhaustiveness guarantee it looked like.
        (language, command) => {
            let shape = match language {
                DispatchLanguage::PerFile => "per-file",
                DispatchLanguage::Fixed(_) => "fixed",
                DispatchLanguage::Job(_) => "job-level",
            };
            return Ok(DispatchEnd::Refused(refuse(
                job,
                format!(
                    "No dispatch arm handled command '{command}' with a {shape} language shape \
                     on the batched-text path. Either its catalog entry declares \
                     BatchedTextInfer without a text arm in runner::routing, or its declared \
                     shape in dispatch_language disagrees with that arm."
                ),
            )));
        }
    };
    dispatched.map(|()| DispatchEnd::Dispatched)
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
    for file in &job.pending_files {
        record_plan_refusal_for_file(host.sink().as_ref(), file.filename().as_ref(), &message)
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
    filename: &str,
    message: &str,
) {
    super::util::FileRunTracker::new(sink, filename)
        .record_setup_failure(message, crate::scheduling::FailureCategory::Validation)
        .await;
}

/// Build one command's dispatch plan, or fail this job's files and answer
/// [`PlanStop::FilesRefused`].
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
        &crate::host_policy::HostExecutionPolicy,
    ) -> Result<Plan, super::dispatch::DispatchPlanRefusal>,
) -> Result<Plan, PlanStop> {
    match from_job(job, &host.host_policy()) {
        Ok(plan) => Ok(plan),
        Err(refusal) => {
            fail_files_for_refused_plan(job, host, &refusal).await;
            Err(PlanStop::FilesRefused)
        }
    }
}

/// Why no dispatch plan was built. Two different outcomes, so not an
/// `Option`: refused files are already terminal and the run finishes them
/// normally, while a refused job fails the run with its reason.
enum PlanStop {
    /// Every file recorded its own plan refusal.
    FilesRefused,
    /// The job itself cannot run.
    JobRefused(JobRefusal),
}

/// One job's dispatch plan, built from its persisted snapshot before any
/// worker is probed.
///
/// The first level is WHERE the job's worker is probed, so the router cannot
/// probe a job-level worker for a command that has no job-level language: the
/// per-file arm carries no `LoadedCapabilities` and the job-level arm cannot
/// be reached without one.
enum FamilyPlan {
    /// Align: audio-bound and per-file in its language too. Each file's worker
    /// is probed by the language that file declares, inside its dispatch
    /// ([`super::dispatch::FaWorkerAdmission`]).
    PerFileAudio(FaDispatchPlan),
    /// Every other family: one worker probed before any file is dispatched,
    /// keyed by the job's language, or by the `per-file` sentinel for the
    /// batched text commands that resolve their language per file (their
    /// pipelines spawn per-language workers as they walk the files; the probe
    /// establishes task support).
    JobLevel(JobLevelFamily),
}

/// The dispatch families probed once per job.
///
/// The variant is the dispatch family; the payload is that family's typed
/// plan, which owns the engines the job runs and therefore its GPU models.
enum JobLevelFamily {
    Transcribe(TranscribeDispatchPlan),
    Benchmark(BenchmarkDispatchPlan),
    MediaAnalysis(MediaAnalysisDispatchPlan),
    SpeakerIdentity,
    BatchedText,
}

impl FamilyPlan {
    /// Build the plan for `kind`, or say why not. Every refusal is decided
    /// here, before any worker is probed, so a job that cannot run never has
    /// a model loaded for it.
    async fn build(
        job: &RunnerJobSnapshot,
        host: &DispatchHostContext,
        kind: RunnerDispatchKind,
        all_chat: bool,
        tencent: &crate::model_manifest::TencentEngineSetting,
    ) -> Result<Self, PlanStop> {
        Ok(match kind {
            RunnerDispatchKind::NativeAudioExport => {
                // Unreachable through `dispatch_job_with_execution_context`,
                // which runs native commands before planning; refused rather
                // than probed if a future caller gets here.
                return Err(PlanStop::JobRefused(refuse(
                    job,
                    "native command reached worker-backed dispatch".into(),
                )));
            }
            RunnerDispatchKind::TranscribeAudioInfer => Self::JobLevel(JobLevelFamily::Transcribe(
                plan_or_fail_files(job, host, |job, policy| {
                    TranscribeDispatchPlan::from_job(job, policy, tencent)
                })
                .await?,
            )),
            RunnerDispatchKind::BenchmarkAudioInfer => Self::JobLevel(JobLevelFamily::Benchmark(
                plan_or_fail_files(job, host, |job, policy| {
                    BenchmarkDispatchPlan::from_job(job, policy, tencent)
                })
                .await?,
            )),
            RunnerDispatchKind::MediaAnalysisV2 => Self::JobLevel(JobLevelFamily::MediaAnalysis(
                plan_or_fail_files(job, host, MediaAnalysisDispatchPlan::from_job).await?,
            )),
            RunnerDispatchKind::SpeakerIdentity if all_chat => {
                Self::JobLevel(JobLevelFamily::SpeakerIdentity)
            }
            RunnerDispatchKind::SpeakerIdentity => {
                return Err(PlanStop::JobRefused(refuse(
                    job,
                    format!(
                        "No released dispatch path remains for command '{}' without CHAT \
                         input for every file. Legacy process-path fallback is retired.",
                        job.dispatch.command
                    ),
                )));
            }
            RunnerDispatchKind::ForcedAlignment => {
                Self::PerFileAudio(plan_or_fail_files(job, host, FaDispatchPlan::from_job).await?)
            }
            RunnerDispatchKind::BatchedTextInfer => Self::JobLevel(JobLevelFamily::BatchedText),
        })
    }
}

impl JobLevelFamily {
    /// The GPU-profile engines the job's requests name, which key the one
    /// worker the router probes and pre-scales. The per-file family derives
    /// its own from its plan inside `dispatch_fa_infer`.
    fn gpu_models(&self) -> GpuModelPlan {
        match self {
            Self::Transcribe(plan) => plan.gpu_models(),
            Self::Benchmark(plan) => plan.gpu_models(),
            Self::MediaAnalysis(plan) => plan.gpu_models(),
            Self::SpeakerIdentity => GpuModelPlan::speaker_identity(),
            Self::BatchedText => GpuModelPlan::without_engines(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::record_plan_refusal_for_file;
    use crate::ReleasedCommand;
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
            "sample.cha",
            "command plan could not be built from job options",
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
                | ReleasedCommand::Align
                | ReleasedCommand::Convert => false,
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
