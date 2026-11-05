//! Dispatch functions for released infer processing.
//!
//! This module is the "traffic director" between the job runner and
//! command-specific orchestrators.
//!
//! # Three dispatch shapes, all audio-bound
//!
//! - [`dispatch_fa_infer`] handles forced alignment (`align`), which is audio-
//!   bound and therefore processed per file (with per-group batching inside each
//!   file).
//! - [`dispatch_transcribe_infer`] handles transcription (`transcribe`), which
//!   takes audio input and produces CHAT output through a multi-step pipeline
//!   (ASR -> post-processing -> CHAT assembly -> optional utseg/morphosyntax).
//! - [`dispatch_benchmark_infer`] handles benchmarking (`benchmark`), which
//!   composes the Rust transcribe and compare pipelines for audio-plus-gold
//!   evaluation.
//!
//! # Why this split exists
//!
//! FA, transcribe and benchmark all require per-file audio, so the top-level
//! loop stays per-file to keep file/audio provenance and failure handling
//! deterministic.
//!
//! # This module is legacy, and text no longer lives here
//!
//! Every command that operates on CHAT payloads only (`morphotag`, `utseg`,
//! `translate`, `coref`, `compare`) executes on the recipe-owned stack in
//! `crate::execution`, reached by a name-matched arm in
//! `crate::runner::routing`. The batched-text dispatch module that used to sit
//! here was retired once the last of those arms landed and nothing could reach
//! it; `routing`'s test module pins that no released batched-text command can
//! fall through to this side again.
//!
//! What remains here is the audio ML orchestration (FA, ASR, media analysis,
//! benchmark), which has no counterpart on the recipe stack yet. Porting it is
//! its own piece of work; do not add new commands to this module.
//!
//! # Related modules
//!
//! - `crate::runner::mod` decides whether infer mode is enabled.
//! - `plan` translates store-owned job snapshots into typed command-family
//!   plans before orchestration begins.
//! - `crate::fa` implements forced-alignment orchestration.
//! - `crate::transcribe` implements the multi-step transcribe orchestrator.
//! - `crate::execution` implements the recipe-owned text orchestrators.

mod asr_media;
mod audio_output;
mod audio_task;
mod benchmark_pipeline;
mod diarize_chat;
pub(crate) mod diarize_turns;
mod fa_admission;
mod fa_pipeline;
mod kernel_plan;
mod media_analysis_v2;
mod media_search;
mod options;
mod plan;
pub(crate) mod speaker_identity_pipeline;
mod transcribe_pipeline;
mod utr;

pub(crate) use benchmark_pipeline::{BenchmarkDispatchRuntime, dispatch_benchmark_infer};
pub(crate) use fa_admission::FaWorkerAdmission;

/// Why a command's worker could not be probed into a usable state.
///
/// Two refusals and not one, because they are different operator actions: a
/// probe failure is the worker not coming up or not loading (a download, a
/// crash, a timeout), while an unavailable identity is a worker that came up
/// and still does not support the command's task or did not name its engine.
/// The router fails the whole job with one of these; `align` fails every file
/// of one language with one (`FaWorkerAdmission`). Both say the same words.
#[derive(Debug, thiserror::Error)]
pub(crate) enum WorkerProbeRefusal {
    /// The worker keyed for this command could not be probed.
    #[error("failed to resolve selected worker capabilities: {0}")]
    Probe(#[source] crate::worker::error::WorkerError),
    /// The worker was probed and does not support the command's task, or
    /// named no engine where one is required.
    #[error("{0}")]
    Unavailable(#[source] crate::engine_reports::EngineIdentityUnavailable),
    /// The job was cancelled while its worker was being probed: not a
    /// refusal of the command, and nothing for the job to report.
    #[error("cancelled while probing the worker")]
    Cancelled,
}

impl WorkerProbeRefusal {
    /// How a file failed by this refusal is classified, and so whether it is
    /// retried: a probe failure by what the worker did (a crash is retried, a
    /// model that will not load is not), an unavailable identity as analysis
    /// the configured backend cannot provide.
    pub(crate) fn category(&self) -> crate::scheduling::FailureCategory {
        match self {
            Self::Probe(error) => crate::runner::util::classify_worker_error(error),
            Self::Unavailable(_) => crate::scheduling::FailureCategory::AnalysisUnavailable,
            Self::Cancelled => crate::scheduling::FailureCategory::Cancelled,
        }
    }
}

/// Probe the worker `command`'s requests will reach under `language`: load the
/// command's task on it, take its report, and apply the availability rule
/// ([`crate::capability::command_supported`], the same rule that decides what
/// `/health` advertises).
///
/// The one probe for a job-level command (the router, once per job) and for
/// `align` (`FaWorkerAdmission`, once per language), so the two cannot drift
/// in what they load, what they check or what they say when refusing.
///
/// A probe that fails the way a transient worker failure does (a crash, a
/// startup stall while models download) is retried by the same loop a
/// request's dispatch uses ([`crate::infer_retry::retry_worker_operation`]),
/// and `cancellation` stops it during an attempt or a backoff. Probing is
/// where a worker starts, and since models download before a worker serves,
/// it is where a stalled first download surfaces; it used to fail the whole
/// job at the first stall (2026-11-05), where the same failure inside a
/// request would have been retried.
pub(crate) async fn probe_command_worker(
    pool: &crate::worker::pool::WorkerPool,
    command: crate::api::ReleasedCommand,
    language: impl Into<crate::api::WorkerLanguage>,
    options: &crate::options::CommandOptions,
    gpu_models: &crate::worker::pool::GpuModelPlan,
    cancellation: crate::infer_retry::Cancellation<'_>,
) -> Result<crate::worker::pool::LoadedCapabilities, WorkerProbeRefusal> {
    let language = language.into();
    let loaded = crate::infer_retry::retry_worker_operation(cancellation, "worker probe", || {
        Box::pin(pool.ensure_command_capabilities(command, language.clone(), options, gpu_models))
    })
    .await
    .map_err(|end| match end {
        crate::infer_retry::RetryEnd::Failed(error) => WorkerProbeRefusal::Probe(error),
        crate::infer_retry::RetryEnd::Cancelled => WorkerProbeRefusal::Cancelled,
    })?;
    crate::capability::command_supported(
        &crate::command_model::command_spec(command).capabilities,
        loaded.reports(),
    )
    .map_err(WorkerProbeRefusal::Unavailable)?;
    Ok(loaded)
}
pub(crate) use fa_pipeline::*;
pub(crate) use media_analysis_v2::{MediaAnalysisDispatchRuntime, dispatch_media_analysis_v2};
pub(crate) use plan::*;
pub(crate) use transcribe_pipeline::{TranscribeDispatchRuntime, dispatch_transcribe_infer};

/// Pre-scale the workers one command's requests will reach, so a batch pays
/// one cold start rather than one per file.
///
/// Runs after a capability probe, on the same key the probe spawned
/// ([`crate::worker::pool::WorkerPool::pre_scale_for_command`] keys a GPU
/// command by its model plan), so the workers it warms are the ones the
/// requests reach. A job-level command calls it once from the router with the
/// job's language; `align` calls it once per language from
/// [`FaWorkerAdmission`], with the language the files declare.
///
/// Skipped for LazyProfile: on constrained machines workers start empty and
/// load models on demand, so a pre-scaled worker would sit idle until files
/// need it.
pub(crate) async fn pre_scale_workers(
    pool: &std::sync::Arc<crate::worker::pool::WorkerPool>,
    command: crate::api::ReleasedCommand,
    language: impl Into<crate::api::WorkerLanguage>,
    num_workers: crate::api::NumWorkers,
    options: &crate::options::CommandOptions,
    gpu_models: &crate::worker::pool::GpuModelPlan,
) {
    if *num_workers > 1 && pool.bootstrap_mode() != crate::worker::WorkerBootstrapMode::LazyProfile
    {
        pool.pre_scale_for_command(command, language, num_workers.0, options, gpu_models)
            .await;
    }
}
