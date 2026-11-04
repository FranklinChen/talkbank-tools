//! Forced alignment dispatch and per-file FA pipeline.

use std::path::PathBuf;
use std::sync::Arc;

use crate::api::{DisplayPath, LanguageCode3, NumWorkers};
use crate::cache::UtteranceCache;
use crate::engine_reports::FaCacheNamespace;
use crate::fa::{
    ActiveAlignment, AdmittedFaResult, AlignRoute, FaServices, NoRecordingRoute, RealignmentRequest,
};
use crate::options::EngineBackend as _;
use crate::params::{AudioContext, FaParams};
use crate::pipeline::PipelineServices;
use crate::provenance::AlignTiming;
use crate::provenance::UtrRecovery;
use crate::runner::DispatchHostContext;
use crate::runner::debug_dumper::DebugDumper;
use crate::scheduling::{FailureCategory, WorkUnitKind};
use crate::worker::pool::{GpuModelPlan, WorkerPool};
use async_trait::async_trait;
use tracing::{info, warn};

use crate::store::RunnerJobSnapshot;
use crate::types::request::{
    AdmittedFaParams, DeclaredLanguages, HeaderLanguageError, validate_fa_language_support,
    validate_utr_language_support,
};

use super::super::util::{
    FileRunTracker, FileStage, FileTaskOutcome, RunnerEventSink, compute_audio_identity,
    drain_supervised_file_tasks, probe_audio_duration, spawn_observed_progress_forwarder,
    spawn_supervised_file_task,
};
use super::audio_output::{ChatOutput, FileOutput, MergeAbbreviations};
use super::audio_task::{AudioFileTask, AudioTaskReporting, run_audio_file_task};
use super::fa_admission::FaWorkerRefusal;
use super::utr::{UtrPassContext, run_utr_pass};
use super::{FaDispatchPlan, FaWorkerAdmission};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AlignUtrDecision<'a> {
    SkipAllTimed,
    Run(&'a crate::options::UtrEngine),
    ProceedWithoutUtr,
}

/// Requested destinations for the structured FA timeline.
///
/// Durable debug artifacts and ephemeral dashboard traces are independent
/// options. Encoding all four combinations prevents one flag from
/// accidentally gating the other, which previously made `--debug-dir` lose
/// the evidence sidecar unless dashboard tracing was also enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaEvidenceRetention {
    None,
    DurableOnly,
    TraceOnly,
    DurableAndTrace,
}

impl FaEvidenceRetention {
    fn requested(durable: bool, trace: bool) -> Self {
        match (durable, trace) {
            (false, false) => Self::None,
            (true, false) => Self::DurableOnly,
            (false, true) => Self::TraceOnly,
            (true, true) => Self::DurableAndTrace,
        }
    }

    fn requires_timeline(self) -> bool {
        !matches!(self, Self::None)
    }

    fn stores_trace(self) -> bool {
        matches!(self, Self::TraceOnly | Self::DurableAndTrace)
    }
}

fn plan_align_utr_stage<'a>(
    untimed: usize,
    file_lang: &LanguageCode3,
    utr_engine: Option<&'a crate::options::UtrEngine>,
) -> Result<AlignUtrDecision<'a>, crate::types::request::ValidationError> {
    if untimed == 0 {
        return Ok(AlignUtrDecision::SkipAllTimed);
    }

    match utr_engine {
        Some(engine) => {
            validate_utr_language_support(file_lang, engine)?;
            Ok(AlignUtrDecision::Run(engine))
        }
        None => Ok(AlignUtrDecision::ProceedWithoutUtr),
    }
}

/// Shared runtime dependencies for top-level FA dispatch.
///
/// The runner always passes this set together when it hands an `align` job to
/// the server-owned FA pipeline, so the bundle is the real boundary rather
/// than eight separate parameters.
pub(crate) struct FaDispatchRuntime {
    /// Worker pool used for typed V2 FA requests and any worker-owned UTR work.
    pub pool: Arc<WorkerPool>,
    /// Cache used by FA group reuse and worker result persistence.
    pub cache: Arc<UtteranceCache>,
    /// Maximum number of file tasks to run concurrently for this job.
    pub num_workers: NumWorkers,
}

/// The job's route from a file's declared language to the FA services that
/// align it: the GPU model plan that routes FA requests and their fallbacks,
/// and the gate that admits the language and names the engine.
///
/// The two are used only together, at one step, so they are one value with
/// one transition: [`Self::admit`] takes the file's language and yields an
/// [`AdmittedFaEngine`], and only that can build [`FaServices`]. No stage of
/// the pipeline can therefore run under a cache namespace read off a worker
/// keyed by another language.
#[derive(Clone, Copy)]
struct FaEngineAdmission<'a> {
    gpu: &'a GpuModelPlan,
    gate: &'a FaWorkerAdmission,
}

/// A file language the gate admitted, carrying the namespace every FA cache
/// row and evidence envelope for this file is read and written under.
struct AdmittedFaEngine<'a> {
    gpu: &'a GpuModelPlan,
    cache_namespace: FaCacheNamespace,
}

impl<'a> FaEngineAdmission<'a> {
    /// Admit the file's language. Probed once per language per job by the
    /// gate; a refusal names the language and what the host could not supply.
    async fn admit(
        self,
        lang: &LanguageCode3,
    ) -> Result<AdmittedFaEngine<'a>, Arc<FaWorkerRefusal>> {
        let cache_namespace = self.gate.admit(lang).await?;
        Ok(AdmittedFaEngine {
            gpu: self.gpu,
            cache_namespace,
        })
    }
}

impl AdmittedFaEngine<'_> {
    /// The FA services for this file: the shared pool and cache, under the
    /// admitted namespace and the job's GPU plan.
    fn services<'s>(
        &'s self,
        pipeline: PipelineServices<'s>,
        attention_capture_dir: Option<&'s std::path::Path>,
    ) -> FaServices<'s> {
        FaServices {
            pipeline,
            cache_namespace: &self.cache_namespace,
            gpu: self.gpu,
            attention_capture_dir,
        }
    }
}

/// Shared per-file FA dependencies.
///
/// Grouping the job snapshot, services, and per-dispatch options here keeps
/// `process_one_fa_file` focused on the file lifecycle rather than repeating a
/// long parameter list for every call site.
struct FaFileContext<'a> {
    /// Immutable runner snapshot for the current job.
    job: &'a RunnerJobSnapshot,
    /// Read-only host/runtime context for media resolution and config access.
    host: DispatchHostContext,
    /// File/job lifecycle sink for runner-side status updates.
    sink: Arc<dyn RunnerEventSink>,
    /// Worker pool and cache shared with every stage; UTR uses only these.
    pipeline: PipelineServices<'a>,
    /// The job's route from a file's declared language to the FA engine that
    /// will align it; the FA services exist only after it admits the file's
    /// language.
    engine: FaEngineAdmission<'a>,
    /// Typed FA parameter bundle.
    fa_params: FaParams,
    /// Independently resolved policy for UTR ASR evidence.
    utr_cache_policy: crate::params::UtrCachePolicies,
    /// Whether merge-abbrev should run before writing the result.
    should_merge_abbrev: bool,
    /// Optional before-file path for incremental align reruns.
    before_path: Option<&'a std::path::Path>,
    /// Optional UTR engine for the pre-pass and fallback paths.
    utr_engine: Option<&'a crate::options::UtrEngine>,
    /// Overlap strategy for `+<` utterances during UTR.
    utr_strategy: super::options::ResolvedUtrStrategy,
    /// Debug artifact writer for offline replay.
    dumper: DebugDumper,
    /// Custom media directory from `--media-dir`.
    media_dir: Option<&'a crate::options::AbsoluteMediaRoot>,
}

#[cfg(test)]
mod tests {
    use super::super::media_search::media_search_subdir;
    use super::FaEvidenceRetention;

    #[test]
    fn media_search_subdir_preserves_mapping_subdir() {
        assert_eq!(
            media_search_subdir("d01oma12a.cha", "French/Newcastle/Discussion/12"),
            "French/Newcastle/Discussion/12"
        );
        assert_eq!(
            media_search_subdir("Discussion/12/d01oma12a.cha", "French/Newcastle"),
            "French/Newcastle/Discussion/12"
        );
    }

    #[test]
    fn durable_evidence_does_not_depend_on_dashboard_tracing() {
        let retention = FaEvidenceRetention::requested(true, false);

        assert_eq!(retention, FaEvidenceRetention::DurableOnly);
        assert!(retention.requires_timeline());
        assert!(!retention.stores_trace());
    }
}

/// The recording a file is aligned against, resolved and prepared once.
///
/// Built only by [`Self::resolve`], and held only by [`AlignAudioTask`]: the
/// route that reads audio. The two routes that do not
/// ([`AlignRoute::WithoutRecording`]) have no field of this type, so a
/// complete-`%wor` refresh or a declared no-op cannot be refused for a
/// recording nothing on its route would open. Media resolution used to run
/// for every file before the route was known.
pub(super) struct ResolvedMedia {
    /// The recording as found: the path the Rust-owned UTR engines read.
    original: PathBuf,
    /// The recording as 16 kHz mono WAV, converted when the container needs
    /// it: the path the Python workers read.
    wav: PathBuf,
    /// Content identity of the WAV for cache keying.
    identity: crate::chat_ops::fa::AudioIdentity,
    /// The probe's measurement of the WAV, taken once per file and handed to
    /// both the UTR pass and FA so neither probes again. Optional because a
    /// failed probe is logged and FA probes again for the recording it cannot
    /// run without, reporting that failure as the file's error.
    duration: Option<crate::media::probe::AudioDuration>,
}

/// Why a file's recording could not be made ready, naming the file.
#[derive(Debug, thiserror::Error)]
enum MediaUnavailable {
    /// The six-rung search found no recording; its message names every place
    /// it looked.
    #[error("{}", .0.message)]
    Unresolved(super::media_search::UnresolvedMedia),
    /// The recording was found and could not be converted to WAV.
    #[error("Media conversion failed for {filename}: {error}")]
    Conversion {
        /// The transcript whose recording it is.
        filename: String,
        /// The conversion's own account.
        error: crate::ensure_wav::EnsureWavError,
    },
}

impl ResolvedMedia {
    /// Find, convert, identify and measure the recording for one transcript.
    async fn resolve(
        job: &RunnerJobSnapshot,
        host: &DispatchHostContext,
        filename: &str,
        read_path: &std::path::Path,
        document: &crate::chat_ops::ChatFile,
        media_dir: Option<&crate::options::AbsoluteMediaRoot>,
    ) -> Result<Self, MediaUnavailable> {
        // The six-rung media search, shared with `speaker-identify` since
        // 2026-09-02. It lives in `media_search` because it states where a
        // corpus keeps its recordings, which is not an alignment fact.
        let original = super::media_search::resolve_transcript_media(
            job,
            host,
            filename,
            read_path,
            || crate::media::DeclaredMedia::from_document(document),
            media_dir,
        )
        .await
        .map_err(MediaUnavailable::Unresolved)?;

        // Convert non-WAV media (e.g. mp4) to WAV via ffmpeg if needed.
        // soundfile (Python) cannot read container formats like mp4 directly.
        let wav = crate::ensure_wav::ensure_wav(&original, None)
            .await
            .map_err(|error| MediaUnavailable::Conversion {
                filename: filename.to_owned(),
                error,
            })?;

        // Compute audio identity for cache keying: path|mtime|size
        let wav_str = wav.to_string_lossy();
        let identity = compute_audio_identity(&wav_str).await.unwrap_or_else(|| {
            // Fallback: use path with zeroed metadata
            crate::chat_ops::fa::AudioIdentity::from_metadata(&wav_str, 0, 0)
        });

        // How long the audio runs, measured once for UTR and FA alike.
        let duration = probe_audio_duration(&wav).await;
        Ok(Self {
            original,
            wav,
            identity,
            duration,
        })
    }

    /// The recording as FA and the worker-hosted UTR engines read it.
    fn context(&self) -> AudioContext<'_> {
        AudioContext {
            audio_path: &self.wav,
            audio_identity: &self.identity,
            audio_duration: self.duration,
        }
    }

    /// The path a UTR engine reads: a Rust-owned engine decodes the recording
    /// as found, a worker-hosted one the prepared WAV.
    fn utr_audio_path(&self, utr_engine: Option<&crate::options::UtrEngine>) -> &std::path::Path {
        if utr_engine.is_some_and(|engine| engine.is_rust_owned()) {
            self.original.as_path()
        } else {
            self.wav.as_path()
        }
    }
}

/// The file task for the route that reads the recording: alignment in full,
/// in part, or incrementally against a declared prior.
struct AlignAudioTask<'a> {
    /// Where the output and its evidence go.
    destination: FaOutputDestination<'a>,
    services: FaServices<'a>,
    /// The run's FA parameters and the languages they were admitted against.
    ///
    /// One proof rather than a `FaParams` beside a `LanguageCode3`: the pair
    /// used to be two fields nothing tied together, and the language was read
    /// off the header BEFORE the engine was checked against it, so the task
    /// could be assembled from an engine and a language that had never been
    /// compared. The only way to obtain this value is to pass the check.
    admitted: AdmittedFaParams,
    utr_cache_policy: crate::params::UtrCachePolicies,
    /// The recording, resolved for this route alone.
    media: ResolvedMedia,
    /// The model FA runs on, with the anchors UTR observed on it and the
    /// prior it is aligned against when the run is incremental.
    model: ActiveAlignment,
    /// This file's observed recovery outcomes and its single fallback permit.
    utr_recovery: UtrRecovery,
    utr_strategy: super::options::ResolvedUtrStrategy,
    /// Language stamped into the provenance comment: the file's own.
    provenance_lang: LanguageCode3,
}

#[async_trait]
impl AudioFileTask for AlignAudioTask<'_> {
    type AttemptOutput = AdmittedFaResult;

    async fn run_attempt(
        &mut self,
        progress_tx: crate::runner::util::ProgressSender,
    ) -> Result<Self::AttemptOutput, crate::error::ServerError> {
        let audio = self.media.context();

        // Both branches hand over the SAME document. The incremental one used
        // to serialize `chat_file` into a string that it then parsed back
        // twice, so this task paid a serialization and two parses to give the
        // callee a model it was already holding.
        let document = self.model.attempt();
        if let Some(before) = self.model.before() {
            crate::fa::process_fa_incremental(
                before,
                document,
                &audio,
                self.admitted.primary_language(),
                self.services,
                self.admitted.params(),
                Some(&progress_tx),
            )
            .await
        } else {
            crate::fa::run_fa_from_ast(
                document,
                &audio,
                self.admitted.primary_language(),
                self.services,
                self.admitted.params(),
                Some(&progress_tx),
            )
            .await
        }
    }

    async fn finalize_success(
        &mut self,
        fa_result: Self::AttemptOutput,
    ) -> Result<FileOutput, crate::error::ServerError> {
        // `incremental=` is a fact about this route: a prior was aligned
        // against. It used to be read off the request, which also said
        // `incremental` on a refresh that used no prior.
        let provenance = crate::provenance::align_provenance(
            &self.provenance_lang,
            self.services.cache_namespace,
            self.utr_recovery.contribution(),
            false,
            self.model.before().is_some(),
            self.admitted.params().main_bullets,
            AlignTiming::Aligned,
        );
        finalize_fa_output(
            &self.destination,
            fa_result,
            self.utr_recovery.contribution().shortfalls(),
            &provenance,
        )
        .await
    }

    async fn on_retryable_worker_failure(
        &mut self,
        lifecycle: &FileRunTracker<'_>,
        _error: &crate::error::ServerError,
    ) -> Result<(), crate::chat_ops::fa::utr::UtrCancelled> {
        if let Some(pending) = self.utr_recovery.fallback() {
            info!(
                filename = %self.destination.filename,
                "FA failed with untimed utterances; attempting fallback UTR"
            );
            lifecycle.stage(FileStage::RecoveringTimingFallback).await;

            let media = &self.media;
            let recovered = self
                .model
                .active_mut()
                .recover_timing(async |file| {
                    run_utr_pass(
                        file,
                        UtrPassContext {
                            audio_path: media.wav.as_path(),
                            lang: self.admitted.primary_language(),
                            services: self.services,
                            audio_identity: &media.identity,
                            cache_policy: self.utr_cache_policy,
                            audio_duration: media.duration,
                            max_group_ms: Some(self.admitted.params().max_group_ms()),
                            filename: &self.destination.filename,
                            engine: pending.engine(),
                            strategy: &self.utr_strategy,
                            dumper: self.destination.dumper,
                        },
                        None,
                    )
                    .await
                })
                .await;
            pending.record(recovered)?;
        }
        Ok(())
    }
}

/// The file task for the two routes that read no recording: a declared no-op
/// returned byte for byte, or a complete-`%wor` file whose bullets are
/// refreshed from it. It holds no [`ResolvedMedia`], by type, and makes no
/// worker request, so there is nothing for the shell to retry.
struct AlignWithoutRecording<'a> {
    /// Where the output and its evidence go.
    destination: FaOutputDestination<'a>,
    /// The FA engine the file's worker reported, which names the result's
    /// `fa=`; nothing else of the services is read on this route.
    cache_namespace: &'a crate::engine_reports::FaCacheNamespace,
    admitted: AdmittedFaParams,
    route: NoRecordingRoute,
    /// Language stamped into the provenance comment: the file's own.
    provenance_lang: LanguageCode3,
}

#[async_trait]
impl AudioFileTask for AlignWithoutRecording<'_> {
    type AttemptOutput = AdmittedFaResult;

    async fn run_attempt(
        &mut self,
        _progress_tx: crate::runner::util::ProgressSender,
    ) -> Result<Self::AttemptOutput, crate::error::ServerError> {
        self.route.run(self.admitted.params(), self.cache_namespace)
    }

    async fn finalize_success(
        &mut self,
        fa_result: Self::AttemptOutput,
    ) -> Result<FileOutput, crate::error::ServerError> {
        // No recovery ran and no prior was aligned against on this route, so
        // the stamp names no `utr=` and no `incremental`; the route says where
        // its timing came from.
        let provenance = crate::provenance::align_provenance(
            &self.provenance_lang,
            self.cache_namespace,
            UtrRecovery::not_run().contribution(),
            false,
            false,
            self.admitted.params().main_bullets,
            self.route.timing(),
        );
        finalize_fa_output(&self.destination, fa_result, Vec::new(), &provenance).await
    }
}

/// Where one file's aligned output and its evidence go, for the finalization
/// both align tasks share. Built once per file, held by whichever task the
/// route selects.
struct FaOutputDestination<'a> {
    host: DispatchHostContext,
    job_id: crate::api::JobId,
    file_index: usize,
    filename: String,
    dumper: &'a DebugDumper,
    debug_traces: bool,
    /// Whether to merge abbreviations before writing: a property of the CHAT
    /// this run produces, not of the alignment.
    merge_abbreviations: MergeAbbreviations,
}

/// Turn an admitted FA result into the file output the writer takes: retain
/// the evidence that was asked for, add the recovery's shortfalls, stamp the
/// provenance. One function for both align tasks, so the two cannot drift in
/// what they retain or how they stamp.
async fn finalize_fa_output(
    destination: &FaOutputDestination<'_>,
    fa_result: AdmittedFaResult,
    utr_shortfalls: Vec<crate::pipeline::post_validate::Shortfall>,
    provenance: &crate::provenance::ProvenanceComment,
) -> Result<FileOutput, crate::error::ServerError> {
    let FaOutputDestination {
        host,
        job_id,
        file_index,
        filename,
        dumper,
        debug_traces,
        merge_abbreviations,
    } = destination;
    let (file_index, debug_traces, merge_abbreviations) =
        (*file_index, *debug_traces, *merge_abbreviations);
    let retention = FaEvidenceRetention::requested(dumper.is_enabled(), debug_traces);
    // The document leaves as the PROOF `FaAdmission` produced, never as a
    // fresh serialization of the model beside it: that second
    // serialization is what made the L2-gated bytes and the written bytes
    // two different things.
    // The shortfalls (untimed words make the file diagnosed) and the
    // exclusions (notes left out on purpose, which do not) leave with the
    // bytes, whichever way they leave.
    let aligned =
        if retention.requires_timeline() {
            let (aligned, timeline) = fa_result.into_document_and_timeline();
            dumper.dump_fa_evidence(filename, &timeline).map_err(|error| {
            crate::error::ServerError::Persistence(format!(
                "could not retain requested forced-alignment evidence for {filename}: {error}"
            ))
        })?;
            if retention.stores_trace() {
                let file_traces = crate::types::traces::FileTraces {
                    filename: DisplayPath::from(filename.as_str()),
                    dp_alignments: Vec::new(),
                    asr_pipeline: None,
                    fa_timeline: Some(timeline),
                    retokenizations: Vec::new(),
                };
                host.trace_store()
                    .upsert_file(job_id, file_index, file_traces)
                    .await;
            }
            aligned
        } else {
            fa_result.into_document()
        };
    let crate::fa::AlignedOutput {
        document,
        mut shortfalls,
        exclusions,
    } = aligned;
    shortfalls.extend(utr_shortfalls);

    // Asked BEFORE the bytes are, because asking is what materializes
    // them: `PostValidated::as_str` serializes the judged model the first
    // time anyone reads it, and a switched-off dumper would otherwise pay
    // for a whole-document serialization it then discards. `dump_fa_output`
    // already no-ops on the same condition, so this changes what is
    // computed and not what is written.
    if dumper.is_enabled() {
        dumper.dump_fa_output(filename, document.as_str());
    }
    // A named transition on the proof, re-gated at align's own admitted
    // level. It used to be `inject_provenance_into_text` on a bare string
    // after the proof had been thrown away, and it stamped a `@Comment`
    // even onto the `NoAlign` documents the book promises are returned
    // with zero modifications; a pass-through is now returned untouched.
    let document = document.with_provenance_injected(provenance)?;
    Ok(FileOutput::Chat(ChatOutput {
        document: document.into(),
        shortfalls,
        exclusions,
        merge_abbreviations,
    }))
}

/// Dispatch FA (forced alignment) via the server-side infer path.
///
/// Unlike morphosyntax/utseg/translate/coref, FA is per-file: each file has
/// its own audio, so there is no cross-file batching. Files are processed
/// concurrently up to `num_workers` at a time, bounded by a semaphore.
/// Within each file, utterances are grouped into time windows and batched
/// to the worker.
pub(crate) async fn dispatch_fa_infer(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    runtime: FaDispatchRuntime,
    plan: FaDispatchPlan,
) {
    let job_id = &job.identity.job_id;
    let sink = host.sink().clone();
    // The job's GPU model plan, from the dispatch plan that owns its engines,
    // before the plan's fields are moved out; every file's FA and UTR
    // requests are routed by it.
    let gpu_models = Arc::new(plan.gpu_models());
    // Align has no job-level language: each file declares its own in
    // `@Languages:`, read inside `process_one_fa_file` after parsing, and the
    // FA worker a file reaches is keyed by it. This gate probes that worker
    // the first time the job reaches each language and hands every file of
    // the language the FA engine it reported.
    let admission = Arc::new(FaWorkerAdmission::new(
        runtime.pool.clone(),
        job.dispatch.options.clone(),
        gpu_models.clone(),
        runtime.num_workers,
    ));
    let fa_params = plan.options.fa_params;
    let utr_cache_policy = plan.utr_cache_policy;
    let should_merge_abbrev = plan.options.merge_abbrev.should_merge();
    let utr_engine = plan.options.utr_engine;
    let utr_strategy = plan.options.utr_strategy;
    let media_dir = plan.media_dir;
    let file_parallelism = runtime
        .num_workers
        .0
        .max(1)
        .min(plan.kernel_plan.file_parallelism_hint.max(1));

    // Process files concurrently, bounded by available workers.
    let file_sem = Arc::new(tokio::sync::Semaphore::new(file_parallelism));
    let mut tasks = Vec::new();

    for file in &job.pending_files {
        // Check cancellation before spawning
        if job.cancel_token.is_cancelled() {
            break;
        }

        let Ok(permit) = file_sem.clone().acquire_owned().await else {
            tracing::warn!("file semaphore closed during shutdown");
            break;
        };
        let host = host.clone();
        let sink = sink.clone();
        let pool = runtime.pool.clone();
        let cache = runtime.cache.clone();
        let job = job.clone();
        let media_dir = media_dir.clone();
        let admission = admission.clone();
        let gpu_models = gpu_models.clone();
        let file = file.clone();
        let file_index = file.file_index;
        let before_path = job.filesystem.before_path(file_index);
        let utr_engine = utr_engine.clone();
        let utr_strategy = utr_strategy.clone();
        let filename = file.filename.clone();

        tasks.push(spawn_supervised_file_task(
            filename,
            "align file task",
            async move {
                let _permit = permit;
                let dumper = DebugDumper::new(job.dispatch.options.common().debug_dir.as_deref());
                let context = FaFileContext {
                    job: &job,
                    host,
                    sink: sink.clone(),
                    pipeline: PipelineServices::new(&pool, &cache),
                    engine: FaEngineAdmission {
                        gpu: &gpu_models,
                        gate: &admission,
                    },
                    fa_params,
                    utr_cache_policy,
                    should_merge_abbrev,
                    before_path: before_path.as_ref().map(|p| p.as_path()),
                    utr_engine: utr_engine.as_ref(),
                    utr_strategy,
                    dumper,
                    media_dir: media_dir.as_ref(),
                };
                process_one_fa_file(&file, context).await
            },
        ));
    }

    let abnormal_exits =
        drain_supervised_file_tasks(sink.as_ref(), job_id, &job.cancel_token, tasks).await;
    if abnormal_exits > 0 {
        warn!(
            job_id = %job_id,
            abnormal_exits,
            "Supervised align file tasks exited abnormally"
        );
    }
}

/// Process a single CHAT file through the server-side FA pipeline.
async fn process_one_fa_file(
    file: &crate::store::PendingJobFile,
    context: FaFileContext<'_>,
) -> FileTaskOutcome {
    let FaFileContext {
        job,
        host,
        sink,
        pipeline,
        engine,
        fa_params,
        utr_cache_policy,
        should_merge_abbrev,
        before_path,
        utr_engine,
        utr_strategy: _,
        ref dumper,
        media_dir,
    } = context;
    let job_id = &job.identity.job_id;
    let file_index = file.file_index;
    let filename = file.filename.as_ref();
    let lifecycle = FileRunTracker::new(sink.as_ref(), job_id, filename);

    lifecycle
        .begin_first_attempt(WorkUnitKind::FileForcedAlignment, FileStage::Reading)
        .await;

    // `--main-bullets exact` keeps untimed utterances untimed; timing recovery
    // exists to time them. The UTR engine the rest of this file uses is the
    // one the policy admits, so a contradictory job cannot reach the pre-pass.
    let utr_engine = match fa_params.main_bullets.admit_utr(utr_engine) {
        Ok(admitted) => admitted,
        Err(refused) => {
            lifecycle
                .fail(&refused.to_string(), FailureCategory::Validation)
                .await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
    };

    // Read the CHAT file
    let read_path: PathBuf =
        if job.filesystem.paths_mode && file_index < job.filesystem.source_paths.len() {
            job.filesystem.source_paths[file_index]
                .assume_shared_filesystem()
                .as_path()
                .to_owned()
        } else {
            job.filesystem
                .staging_dir
                .join("input")
                .join(filename)
                .as_path()
                .to_owned()
        };
    let chat_text = match tokio::fs::read_to_string(&read_path).await {
        Ok(content) => content,
        Err(e) => {
            let err_msg = format!("Failed to read input: {e}");
            lifecycle
                .fail(&err_msg, FailureCategory::InputMissing)
                .await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
    };

    // Preserve one source parse and admit every reused prior tier before any
    // media preparation or timing-recovery inference. A declared no-op does
    // not consume the optional prior file.
    let model = match crate::fa::read_fa_source_named(
        &chat_text,
        talkbank_model::model::TranscriptName::for_path(std::path::Path::new(filename)),
        fa_params.main_bullets,
    ) {
        Ok(source) => source,
        Err(error) => {
            lifecycle
                .fail(
                    &error.to_string(),
                    crate::runner::util::classify_server_error(&error),
                )
                .await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
    };
    // Read the WHOLE `@Languages:` header in one step. The reading lives on
    // `DeclaredLanguages` rather than here because doing it here is what made
    // header ORDER decide admission: the primary was resolved locally (falling
    // back to the job's then `--lang` for an unreadable first entry) and the
    // secondaries were then taken as `skip(1)`, so the unreadable entry
    // disappeared from the declaration that admission was checked against.
    // There is no fallback now: align has no `--lang`, so the file's header
    // is the only place its language can come from.
    //
    // The whole header, not `first()`: a secondary language reaches words
    // through `[- deu]` precodes and `word@s:deu` markers, and those words
    // travel to the aligner in the same groups as the primary language's.
    let header_languages = match model.document().language_declaration() {
        Ok(languages) => Some(languages),
        Err(talkbank_model::LanguageDeclarationError::Absent) => None,
        Err(
            error @ (talkbank_model::LanguageDeclarationError::Empty
            | talkbank_model::LanguageDeclarationError::Rejected
            | talkbank_model::LanguageDeclarationError::Repeated),
        ) => {
            lifecycle.fail(&format!("align: unusable @Languages declaration: {error}; fix the header before aligning"), FailureCategory::Validation).await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
    };
    let declared_languages = match DeclaredLanguages::from_header(
        header_languages
            .into_iter()
            .flat_map(|languages| languages.iter())
            .map(|code| code.as_str()),
    ) {
        Ok(declared) => declared,
        Err(HeaderLanguageError::UnreadablePrimary { raw }) => {
            let msg = format!(
                "align: file '{filename}' declares `@Languages: {raw}`, which is not a \
                 parseable ISO 639-3 code. Align reads each file's language from its own \
                 header and has no `--lang` to stand in for it; fix the header and re-run."
            );
            lifecycle.fail(&msg, FailureCategory::Validation).await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
        Err(HeaderLanguageError::NoHeader) => {
            let msg = format!(
                "align: file '{filename}' has no `@Languages:` header. Align reads each \
                 file's language from its own header and has no `--lang` to stand in for \
                 it; add the header (for example `@Languages:\teng`) so the aligner, the \
                 timing-recovery engine and the provenance stamp all name the language \
                 the file actually declares."
            );
            lifecycle.fail(&msg, FailureCategory::Validation).await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
    };
    // The selected aligner has to be able to handle every language THIS file
    // declares, and this is the first moment both facts are known: the engine
    // came from the job's options, the languages from `@Languages:` just
    // above. Refusing here costs the operator a message; the alternative,
    // silently aligning with a language-general engine instead, costs them a
    // `%wor` tier produced by a model they did not choose and cannot tell
    // apart afterwards.
    //
    // Both values are MOVED into the check, and what comes back is the proof.
    // The engine that runs and the language this file is processed under are
    // read off it below, so neither can be the pre-check value: the language
    // used to be cloned off the header on the line ABOVE this call, which made
    // "was it validated" a question about statement order.
    let admitted = match validate_fa_language_support(declared_languages, fa_params) {
        Ok(admitted) => admitted,
        Err(error) => {
            lifecycle
                .fail(&error.to_string(), FailureCategory::Validation)
                .await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
    };
    let file_lang: LanguageCode3 = admitted.primary_language().clone();

    // The worker this file's requests reach is keyed by its language, and the
    // FA engine that worker reports is the namespace its cache rows are read
    // and written under. Probed once per language per job; a refusal names the
    // language and the fact the host could not supply.
    let engine = match engine.admit(&file_lang).await {
        Ok(engine) => engine,
        Err(refusal) => {
            lifecycle
                .fail(&refusal.to_string(), refusal.category())
                .await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
    };
    let services = engine.services(pipeline, job.dispatch.options.common().debug_dir.as_deref());

    // Provenance comments embed the language for auditability. Use the
    // per-file language we just resolved above (`file_lang`), not the
    // job-level placeholder. This matches what gets stamped into
    // `@Languages:` and avoids the silent eng substitution that the
    // 2026-05-03 incident punished.
    let provenance_lang = file_lang.clone();
    let destination = FaOutputDestination {
        host: host.clone(),
        job_id: job_id.clone(),
        file_index,
        filename: filename.to_string(),
        dumper,
        debug_traces: job.dispatch.debug_traces,
        merge_abbreviations: MergeAbbreviations::from_option(should_merge_abbrev),
    };
    let reporting = AudioTaskReporting {
        work_unit_kind: WorkUnitKind::FileForcedAlignment,
        running_stage: FileStage::Aligning,
        command_label: "Alignment",
    };

    // Where this file's alignment goes is decided once, here, from the
    // admitted document and the request. Only the route that reads audio
    // resolves the recording; a complete-`%wor` refresh and a declared no-op
    // finish without it.
    let mut alignment =
        match AlignRoute::decide(model, RealignmentRequest::from(fa_params.cache_policy)) {
            AlignRoute::WithoutRecording(route) => {
                let mut task = AlignWithoutRecording {
                    destination,
                    cache_namespace: &engine.cache_namespace,
                    admitted,
                    route,
                    provenance_lang,
                };
                return run_audio_file_task(
                    job,
                    sink.clone(),
                    file,
                    &lifecycle,
                    reporting,
                    &mut task,
                )
                .await;
            }
            AlignRoute::NeedsAudio(alignment) => alignment,
        };

    // A declared `--before` prior is read here, on the one route that aligns
    // against it; the no-recording routes never parse one they would drop.
    if let Some(path) = before_path {
        let prior_text = match tokio::fs::read_to_string(path).await {
            Ok(text) => text,
            Err(error) => {
                lifecycle
                    .fail(
                        &format!("Failed to read declared before-file: {error}"),
                        FailureCategory::InputMissing,
                    )
                    .await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
        };
        match crate::fa::RetainedFaPrior::admit(&prior_text) {
            Ok(prior) => alignment = alignment.with_prior(prior),
            Err(error) => {
                lifecycle
                    .fail(
                        &error.to_string(),
                        crate::runner::util::classify_server_error(&error),
                    )
                    .await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
        }
    }

    lifecycle.stage(FileStage::ResolvingAudio).await;
    let media = match ResolvedMedia::resolve(
        job,
        &host,
        filename,
        read_path.as_path(),
        alignment.document(),
        media_dir,
    )
    .await
    {
        Ok(media) => media,
        Err(unavailable) => {
            lifecycle
                .fail(&unavailable.to_string(), FailureCategory::Validation)
                .await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
    };
    let utr_audio_path = media.utr_audio_path(utr_engine);

    // UTR pre-pass: if untimed utterances exist and a UTR engine is configured,
    // run ASR to recover utterance-level timing before FA grouping. The model
    // and the anchors the pass observes on it travel together from here.
    let utr_recovery = {
        let active = alignment.active_mut();
        let (timed, untimed) = crate::chat_ops::fa::count_utterance_timing(active.document());

        match plan_align_utr_stage(untimed, &file_lang, utr_engine) {
            Ok(AlignUtrDecision::SkipAllTimed) => {
                info!(filename, timed, "All utterances timed, skipping UTR");
                UtrRecovery::not_run()
            }
            Ok(AlignUtrDecision::Run(utr_engine)) => {
                lifecycle.stage(FileStage::RecoveringUtteranceTiming).await;

                // Observed: the pre-pass runs before the audio-file shell, so
                // it installs its own observer, and the file shows "waiting
                // for a worker" while one of its checkouts waits.
                let (utr_progress, utr_waits, utr_forwarder) = spawn_observed_progress_forwarder(
                    sink.clone(),
                    job_id.clone(),
                    filename.to_string(),
                );

                let recovered = active.recover_timing(async |file| {
                    run_utr_pass(
                        file,
                        UtrPassContext {
                            audio_path: utr_audio_path,
                            lang: &file_lang,
                            services,
                            audio_identity: &media.identity,
                            cache_policy: utr_cache_policy,
                            audio_duration: media.duration,
                            max_group_ms: Some(admitted.params().max_group_ms()),
                            filename,
                            engine: utr_engine,
                            strategy: &context.utr_strategy,
                            dumper,
                        },
                        Some(&utr_progress),
                    )
                    .await
                });
                let recovered = crate::worker::pool::checkout_wait::observing_checkout_waits(
                    utr_waits, recovered,
                )
                .await;
                // Every update the pre-pass sent is published before the next
                // stage.
                drop(utr_progress);
                utr_forwarder.finished().await;
                match UtrRecovery::from_initial(utr_engine, recovered) {
                    Ok(recovery) => recovery,
                    Err(crate::chat_ops::fa::utr::UtrCancelled) => {
                        lifecycle
                            .fail("Timing recovery cancelled", FailureCategory::Cancelled)
                            .await;
                        return FileTaskOutcome::TerminalStateRecorded;
                    }
                }
            }
            Ok(AlignUtrDecision::ProceedWithoutUtr) => {
                warn!(
                    filename,
                    untimed,
                    "Untimed utterances detected but no UTR engine configured, using interpolation"
                );
                UtrRecovery::not_run()
            }
            Err(error) => {
                lifecycle
                    .fail(&error.to_string(), FailureCategory::Validation)
                    .await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
        }
    };

    let mut task = AlignAudioTask {
        destination,
        services,
        admitted,
        utr_cache_policy,
        media,
        model: alignment,
        utr_recovery,
        utr_strategy: context.utr_strategy.clone(),
        provenance_lang,
    };

    run_audio_file_task(job, sink.clone(), file, &lifecycle, reporting, &mut task).await
}

#[cfg(test)]
mod auto_detect_tests {
    use super::super::media_search::{MediaSearch, SearchedPlace};
    use batchalign_types::paths::{ClientPath, MediaMappingKey, ServerPath};
    use std::collections::BTreeMap;

    /// Regression test: when `--server` is used with a top-level data
    /// directory (e.g. `/Volumes/data-drive/talkbank-data`) the filenames
    /// carry a repo-key prefix (`aphasia-data/English/...`).  The old code
    /// passed only `source_dir` to `infer_media_mapping`, which had no repo
    /// key in its path and always returned `None`, every file failed with
    /// "Cannot find audio file".
    ///
    /// The fix: join `source_dir + parent(filename)` before inference so the
    /// repo key is visible, then use `repo_subdir` directly (no double-join).
    #[tokio::test]
    async fn auto_detect_media_mapping_content_mode_filename_has_repo_key_prefix() {
        let dir = tempfile::tempdir().unwrap();
        // Set up: /tmp/X/aphasia/English/Protocol/APROCSA/2256_T4.mp3
        let media_root = dir.path().join("aphasia");
        let nested = media_root.join("English/Protocol/APROCSA");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("2256_T4.mp3"), b"mp3").unwrap();

        let source_dir = ClientPath::new("/Volumes/data-drive/talkbank-data".to_string());
        let filename = "aphasia-data/English/Protocol/APROCSA/2256_T4.cha";
        let stem = "2256_T4";

        // Build mappings: aphasia-data → <tempdir>/aphasia
        let mut mappings: BTreeMap<MediaMappingKey, ServerPath> = BTreeMap::new();
        mappings.insert(
            MediaMappingKey::new("aphasia-data"),
            ServerPath::new(media_root.to_str().unwrap().to_string()),
        );

        let infer_client =
            match crate::runner::dispatch::media_search::filename_parent_dir(filename) {
                Some(parent) => source_dir.join(parent),
                None => source_dir.clone(),
            };

        let result = batchalign_types::paths::infer_media_mapping(&infer_client, &mappings);
        assert!(
            result.is_some(),
            "infer_media_mapping should find aphasia-data key in '{}'",
            infer_client.as_str()
        );
        let (_key, inferred_root, repo_subdir) = result.unwrap();

        // repo_subdir already contains the correct path within the volume;
        // do NOT also join mapped_subdir (that would double the path).
        let search_dir = repo_subdir.resolve_on_server(&inferred_root);
        let mut search = MediaSearch::for_stem(stem);
        let found = search
            .try_place(SearchedPlace::InferredMediaMapping {
                root: inferred_root.as_path().to_path_buf(),
                subdir: repo_subdir.as_str().to_owned(),
            })
            .await
            .unwrap();
        assert!(
            found.is_some(),
            "Should find 2256_T4.mp3 under {}",
            search_dir.as_path().display()
        );
    }

    /// An operator's scenario: `source_dir` contains "slabank-data" and
    /// `media_mappings` maps that repo to a media volume, so the auto-detect
    /// rung should find the audio under the inferred subdir.
    ///
    /// This test used to hand-rebuild `infer_media_mapping` with its own string
    /// splitting and then assert `"{repo_suffix}/{mapped_subdir}"`, which is
    /// exactly the double-count the production comment forbids ("Do NOT join
    /// repo_subdir with mapped_subdir afterwards ... Joining again would
    /// double-count it"). It passed, because it was checking its own
    /// reconstruction rather than the code. It now calls the real inference and
    /// the real search, which is possible only because the rung became a value
    /// something can call.
    #[tokio::test]
    async fn auto_detect_media_mapping_from_source_path() {
        let dir = tempfile::tempdir().unwrap();
        let media_root = dir.path().join("slabank");
        let nested = media_root.join("French/Newcastle/Photos/13");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("p08aul13.mp3"), b"mp3").unwrap();

        let client_path =
            ClientPath::new("/Users/operator/chat-data/slabank-data/French/Newcastle/Photos/13");
        let mut mappings: BTreeMap<MediaMappingKey, ServerPath> = BTreeMap::new();
        mappings.insert(
            MediaMappingKey::new("slabank-data"),
            ServerPath::new(media_root.to_string_lossy().into_owned()),
        );

        let (key, root, repo_subdir) =
            batchalign_types::paths::infer_media_mapping(&client_path, &mappings)
                .expect("the repo key is a component of the client path");
        assert_eq!(key, MediaMappingKey::new("slabank-data"));
        assert_eq!(
            repo_subdir.to_string(),
            "French/Newcastle/Photos/13",
            "the inferred subdir is COMPLETE; production must not join the \
             mapped subdir onto it"
        );

        let mut search = MediaSearch::for_stem("p08aul13");
        let found = search
            .try_place(SearchedPlace::InferredMediaMapping {
                root: root.as_path().to_path_buf(),
                subdir: repo_subdir.as_str().to_owned(),
            })
            .await
            .unwrap();

        assert_eq!(found, Some(nested.join("p08aul13.mp3")));
    }
}

#[cfg(test)]
mod utr_stage_tests {
    use super::{AlignUtrDecision, plan_align_utr_stage};
    use crate::api::LanguageCode3;
    use crate::options::UtrEngine;

    #[test]
    fn fully_timed_align_skips_utr_even_if_selected_engine_would_not_support_language() {
        let decision =
            plan_align_utr_stage(0, &LanguageCode3::yue(), Some(&UtrEngine::RevAi)).unwrap();
        assert_eq!(decision, AlignUtrDecision::SkipAllTimed);
    }

    #[test]
    fn untimed_align_with_no_utr_engine_uses_interpolation_path() {
        let decision = plan_align_utr_stage(3, &LanguageCode3::eng(), None).unwrap();
        assert_eq!(decision, AlignUtrDecision::ProceedWithoutUtr);
    }

    #[test]
    fn untimed_align_with_supported_utr_runs_selected_backend() {
        let decision =
            plan_align_utr_stage(2, &LanguageCode3::yue(), Some(&UtrEngine::Whisper)).unwrap();
        assert_eq!(decision, AlignUtrDecision::Run(&UtrEngine::Whisper));
    }

    #[test]
    fn untimed_align_with_unsupported_rev_utr_fails_stage_validation() {
        let err =
            plan_align_utr_stage(4, &LanguageCode3::yue(), Some(&UtrEngine::RevAi)).unwrap_err();
        assert!(
            err.to_string()
                .contains("requires utterance timing recovery")
        );
    }
}
