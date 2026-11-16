//! The GPU-profile models one job asks workers for, declared once per job.
//!
//! # Why this exists
//!
//! A GPU worker is identified by the engines it was spawned with, and the pool
//! reuses a worker only for an identical key. That key used to be derived
//! separately for every request, from that request's own task: an ASR request
//! was keyed `{asr, fa}` (the FA default filled in because a profile worker
//! preloads FA), an FA request `{fa}`. The two halves of ONE job therefore
//! never met. An `align --utr-engine whisper` run kept a Whisper worker for
//! its UTR requests (about 10 GB) and spawned a second worker for its FA
//! requests (about 2.8 GB), with wav2vec loaded in both.
//!
//! The defect's shape was one type carrying three meanings: `EngineOverrides`
//! held what the job's options asked for, what one request needed, and what a
//! worker was believed to hold, so a request's need could be compared with a
//! worker's identity as if they were the same thing.
//!
//! # The graph
//!
//! ```text
//! family dispatch plan --(gpu_models)--> GpuModelPlan
//! GpuModelPlan + task + language --(WorkerPool::gpu_route)--> GpuRoute
//! GpuRoute + request --(WorkerPool::dispatch_routed_v2)--> response
//! ```
//!
//! - [`GpuModelPlan`] is built only from a family plan's own typed engine
//!   choices ([`AsrBackend`], [`FaEngineName`]), so the engines a job declares
//!   are the engines its pipeline runs. Rust-hosted ASR (Rev.AI, whisper.cpp)
//!   is never a worker resource and cannot enter a plan: the constructors match
//!   [`AsrBackend`] exhaustively and keep only [`AsrBackend::Worker`].
//! - The job's worker key is a function of the plan alone
//!   ([`GpuRoute::worker_key`]), and the capability probe, pre-scaling and
//!   every primary route share it, so one job reaches one worker.
//! - [`GpuRoute`] is minted only by the pool, for a task the plan declared.
//!   Dispatch checks that the request names exactly the engines its route
//!   expects ([`PlanMismatch`]), so a request builder that drifted from the
//!   plan fails loudly instead of quietly spawning another worker.
//! - The request-keyed dispatch refuses GPU tasks
//!   ([`WorkerError::UnroutedGpuRequest`]), so there is no route around the
//!   plan in production code.
//!
//! - The speaker slot ([`PlannedSpeaker`]) names the one speaker model the
//!   job's speaker requests need. It is NOT worker identity: GPU workers are
//!   shared across speaker models, so splitting them by one would add workers
//!   under concurrency. It reaches the worker in the speaker task's
//!   `ensure_task` instead ([`GpuModelPlan::load_overrides`],
//!   `TaskLoad::for_request`), which builds exactly that model, once per
//!   worker; a second speaker model on the same worker is a second load. A
//!   job that declares no speaker model cannot route a speaker request.
//!
//! Later phases replace exact-key matching with monotone per-slot loadouts,
//! have request builders read their engines from the route, and move every
//! engine default out of the Python worker.

use std::collections::BTreeMap;

use batchalign_types::worker::InferTask;

use crate::api::WorkerLanguage;
use crate::transcribe::{AsrBackend, AsrWorkerMode};
use crate::types::engines::{AsrEngineName, EngineOverrides, FaEngineName, SpeakerEngineName};
use crate::types::worker_v2::{
    ExecuteRequestV2, SpeakerBackendV2, SpeakerEmbeddingBackendV2, TaskRequestV2,
};
use crate::worker::error::WorkerError;
use crate::worker::{WorkerBootstrapMode, WorkerTarget};

use super::execute_v2::{asr_backend_engine, execute_request_overrides, infer_task_for_execute_v2};
use super::{EngineSelection, WorkerKey};

/// The GPU-profile engines one job's requests will name.
///
/// Fields are private and every constructor takes a family plan's typed
/// choice, so a plan cannot be assembled from loose strings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuModelPlan {
    /// Worker-hosted ASR the job dispatches (transcription, or UTR in
    /// alignment). Absent when the job runs no ASR or runs it in Rust.
    asr: Option<PlannedWorkerAsr>,
    /// The forced-alignment engine, for alignment jobs. Its fallback is read
    /// from the engine's own row when needed, never stored beside it.
    fa: Option<FaEngineName>,
    /// The speaker model the job's speaker requests need: diarization on one
    /// engine, or the embedding model. Absent when the job runs no speaker
    /// requests, and then it cannot route one.
    speaker: Option<PlannedSpeaker>,
}

/// The one speaker model a job's speaker requests need.
///
/// The speaker task's load builds what this names, and nothing else: local
/// pyannote builds its pipeline, the embedding model builds the embedding
/// model, pyannoteAI (a hosted API) builds nothing. A request on the route
/// must ask for exactly this model ([`RouteEngines::named_by`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlannedSpeaker {
    /// Speaker diarization on this backend.
    Diarization(SpeakerBackendV2),
    /// Speaker embedding on this model (`speaker-identify`).
    Embedding(SpeakerEmbeddingBackendV2),
}

impl PlannedSpeaker {
    /// The engine overrides that select the model a speaker task's load
    /// builds: `speaker` for local pyannote, `speaker_embedding` for the
    /// embedding model, and nothing for pyannoteAI, a hosted API with no
    /// model. They travel only in `ensure_task`, never in a worker key.
    ///
    /// The one derivation, for the probe's load
    /// ([`GpuModelPlan::load_overrides`]) and for a request's
    /// (`execute_request_overrides`) alike.
    pub(super) fn loaded_model_overrides(self) -> EngineOverrides {
        match self {
            Self::Diarization(SpeakerBackendV2::Pyannote) => EngineOverrides {
                speaker: Some(SpeakerEngineName::Pyannote),
                ..EngineOverrides::default()
            },
            Self::Diarization(SpeakerBackendV2::PyannoteAi) => EngineOverrides::default(),
            Self::Embedding(backend) => EngineOverrides {
                speaker_embedding: Some(backend),
                ..EngineOverrides::default()
            },
        }
    }
}

/// One worker-hosted ASR engine and the configuration its requests carry.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedWorkerAsr {
    mode: AsrWorkerMode,
    /// Per-engine knobs (`qwen_model`, `funaudio_model`, ...). They configure
    /// the loaded model, so they are part of the worker's identity, and every
    /// ASR request of the job must carry exactly these.
    extras: BTreeMap<String, String>,
}

impl PlannedWorkerAsr {
    /// Keep the engine only when a Python worker runs it.
    fn worker_hosted(backend: AsrBackend, extras: BTreeMap<String, String>) -> Option<Self> {
        match backend {
            AsrBackend::Worker(mode) => Some(Self { mode, extras }),
            // Rust-hosted: no worker resource to plan.
            AsrBackend::RustRevAi | AsrBackend::RustWhisperRs => None,
        }
    }

    fn engine(&self) -> AsrEngineName {
        asr_backend_engine(self.mode.as_v2_backend())
    }
}

impl GpuModelPlan {
    /// A job with no GPU requests: openSMILE, AVQI and the batched text
    /// commands, whose workers are not GPU workers.
    pub(crate) const fn without_engines() -> Self {
        Self {
            asr: None,
            fa: None,
            speaker: None,
        }
    }

    /// A transcription or benchmark job: the ASR backend its admitted plan
    /// selected (`None` when nothing is recognized, as in a replay), with the
    /// job's engine extras, and the dedicated diarization engine when one
    /// runs after recognition.
    pub(crate) fn transcription(
        asr: Option<AsrBackend>,
        extras: &BTreeMap<String, String>,
        diarization: Option<SpeakerBackendV2>,
    ) -> Self {
        Self {
            asr: asr.and_then(|backend| PlannedWorkerAsr::worker_hosted(backend, extras.clone())),
            fa: None,
            speaker: diarization.map(PlannedSpeaker::Diarization),
        }
    }

    /// A standalone diarization job on `backend`.
    pub(crate) const fn diarization(backend: SpeakerBackendV2) -> Self {
        Self {
            asr: None,
            fa: None,
            speaker: Some(PlannedSpeaker::Diarization(backend)),
        }
    }

    /// A speaker-identification job: the embedding model, the one node of
    /// the released local model graph it reads.
    pub(crate) const fn speaker_identity() -> Self {
        Self {
            asr: None,
            fa: None,
            speaker: Some(PlannedSpeaker::Embedding(
                SpeakerEmbeddingBackendV2::Pyannote,
            )),
        }
    }

    /// An alignment job: its FA engine, and the backend of its UTR pass when
    /// it runs one. UTR runs each engine's default configuration, so its
    /// requests carry no engine extras; the route check holds the UTR request
    /// builder to that.
    pub(crate) fn alignment(fa: FaEngineName, utr: Option<AsrBackend>) -> Self {
        Self {
            asr: utr.and_then(|backend| PlannedWorkerAsr::worker_hosted(backend, BTreeMap::new())),
            fa: Some(fa),
            speaker: None,
        }
    }

    /// A plan for exercising the shared GPU worker from integration tests:
    /// stock Whisper ASR with no extras. Production plans come only from
    /// family dispatch plans.
    #[doc(hidden)]
    pub fn local_whisper_for_test() -> Self {
        Self::transcription(
            Some(AsrBackend::Worker(AsrWorkerMode::LocalWhisperV2)),
            &BTreeMap::new(),
            None,
        )
    }

    /// The engine the job's FA groups retry on, from its FA engine's row.
    fn fa_fallback(&self) -> Option<FaEngineName> {
        self.fa?.fallback_policy().retry_target()
    }

    /// The engines a request on `task`'s route must name, or `None` when the
    /// plan did not declare the task.
    fn expected(&self, task: GpuTask) -> Option<RouteEngines> {
        match task {
            GpuTask::Asr => self.asr.as_ref().map(|asr| RouteEngines::Asr {
                engine: asr.engine(),
                extras: asr.extras.clone(),
            }),
            GpuTask::Fa => self.fa.map(RouteEngines::Fa),
            GpuTask::FaFallback => self.fa_fallback().map(RouteEngines::Fa),
            GpuTask::Speaker => self.speaker.map(RouteEngines::Speaker),
        }
    }

    /// What `task`'s load reads on this plan's worker: the worker's identity
    /// for the task, and for the speaker task the speaker model too, which
    /// is not identity. The capability probe prepares its worker with it, so
    /// the probe loads the very model the job's requests then need.
    pub(super) fn load_overrides(&self, task: GpuTask) -> EngineOverrides {
        match (task, self.speaker) {
            (GpuTask::Speaker, Some(speaker)) => speaker.loaded_model_overrides(),
            (GpuTask::Speaker, None) | (GpuTask::Asr | GpuTask::Fa | GpuTask::FaFallback, _) => {
                self.identity(task)
            }
        }
    }

    /// The engines a worker serving `task` of this plan is identified by.
    ///
    /// Every primary task shares the whole plan. The FA fallback is the
    /// exception: its worker hosts the fallback engine, which the job's main
    /// worker does not, so it is identified by that engine alone, as the
    /// fallback request's own key always was.
    fn identity(&self, task: GpuTask) -> EngineOverrides {
        match task {
            GpuTask::Asr | GpuTask::Fa | GpuTask::Speaker => EngineOverrides {
                asr: self.asr.as_ref().map(PlannedWorkerAsr::engine),
                fa: self.fa,
                extras: self
                    .asr
                    .as_ref()
                    .map(|asr| asr.extras.clone())
                    .unwrap_or_default(),
                ..EngineOverrides::default()
            },
            GpuTask::FaFallback => EngineOverrides {
                fa: self.fa_fallback(),
                ..EngineOverrides::default()
            },
        }
    }
}

/// The GPU-profile tasks a job can route.
///
/// Speaker diarization and speaker embedding are one pool task (see
/// `infer_task_for_execute_v2`), so they are one variant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTask {
    /// Speech recognition, including UTR.
    Asr,
    /// Forced alignment on the job's FA engine.
    Fa,
    /// Forced alignment of a group the job's FA engine could not take,
    /// retried on that engine's fallback.
    FaFallback,
    /// Speaker diarization or speaker embedding.
    Speaker,
}

impl GpuTask {
    fn infer_task(self) -> InferTask {
        match self {
            Self::Asr => InferTask::Asr,
            Self::Fa | Self::FaFallback => InferTask::Fa,
            Self::Speaker => InferTask::Speaker,
        }
    }

    /// The primary GPU task an infer task is, or `None` for a task another
    /// profile serves. A test holds this to `WorkerProfile::for_task`, the
    /// owner of which tasks the GPU profile hosts.
    pub(crate) fn of(task: InferTask) -> Option<Self> {
        match task {
            InferTask::Asr => Some(Self::Asr),
            InferTask::Fa => Some(Self::Fa),
            InferTask::Speaker => Some(Self::Speaker),
            InferTask::Morphosyntax
            | InferTask::Utseg
            | InferTask::Translate
            | InferTask::Coref
            | InferTask::Opensmile
            | InferTask::Avqi
            | InferTask::WordSegmentation => None,
        }
    }
}

/// Why a route could not be issued.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GpuRouteRefusal {
    /// The job's plan did not declare this task, so its requests would reach a
    /// worker spawned for engines the job never chose.
    #[error("this job's model plan declares no {task:?} engine")]
    Undeclared {
        /// The task that was asked for.
        task: GpuTask,
    },
}

/// The engines one route's requests must name.
#[derive(Debug, Clone, PartialEq, Eq)]
enum RouteEngines {
    Asr {
        engine: AsrEngineName,
        extras: BTreeMap<String, String>,
    },
    Fa(FaEngineName),
    Speaker(PlannedSpeaker),
}

impl RouteEngines {
    /// Whether `payload` names exactly these engines. Compared by reference,
    /// so the check allocates nothing.
    fn named_by(&self, payload: &TaskRequestV2) -> bool {
        match (self, payload) {
            (Self::Asr { engine, extras }, TaskRequestV2::Asr(request)) => {
                asr_backend_engine(request.backend) == *engine && request.extras == *extras
            }
            (Self::Fa(engine), TaskRequestV2::ForcedAlignment(request)) => {
                FaEngineName::from_worker_backend(request.backend) == *engine
            }
            (
                Self::Speaker(PlannedSpeaker::Diarization(backend)),
                TaskRequestV2::Speaker(request),
            ) => request.backend == *backend,
            (
                Self::Speaker(PlannedSpeaker::Embedding(backend)),
                TaskRequestV2::SpeakerEmbedding(request),
            ) => request.backend == *backend,
            // The request's task was already matched against the route's, so
            // a payload of another shape is a request whose task and payload
            // disagree: it names nothing this route expects. A diarization
            // request on an embedding route (or the reverse) names a model
            // the worker was not prepared with.
            (Self::Asr { .. } | Self::Fa(_) | Self::Speaker(_), _) => false,
        }
    }
}

/// Permission to send one job's requests for one task to one worker key.
///
/// Minted only by [`super::WorkerPool::gpu_route`].
#[derive(Debug, Clone)]
pub(crate) struct GpuRoute {
    key: WorkerKey,
    task: GpuTask,
    expected: RouteEngines,
}

impl GpuRoute {
    pub(super) fn key(&self) -> &WorkerKey {
        &self.key
    }
    /// The worker that serves `task` of `plan` in `language` under `mode`.
    ///
    /// Under the profile bootstrap modes every primary task of a plan shares
    /// one key, the whole plan. Under task mode (small hosts) a worker holds
    /// one task's model, so the key is that task's engines alone. This is
    /// also the key the capability probe and pre-scaling use, for the
    /// command's own task, whether or not the plan declared it: a Rev.AI
    /// transcription declares no worker ASR, yet its probe still selects the
    /// worker its speaker requests would reach.
    pub(super) fn worker_key(
        plan: &GpuModelPlan,
        task: GpuTask,
        language: WorkerLanguage,
        mode: WorkerBootstrapMode,
    ) -> WorkerKey {
        let target = WorkerTarget::from_infer_task(task.infer_task(), mode);
        WorkerKey {
            target,
            language,
            engine_selection: EngineSelection::for_job_plan(plan.identity(task), target),
        }
    }

    /// Build the route for a task `plan` declared.
    pub(super) fn new(
        plan: &GpuModelPlan,
        task: GpuTask,
        language: WorkerLanguage,
        mode: WorkerBootstrapMode,
    ) -> Result<Self, GpuRouteRefusal> {
        let expected = plan
            .expected(task)
            .ok_or(GpuRouteRefusal::Undeclared { task })?;
        Ok(Self {
            key: Self::worker_key(plan, task, language, mode),
            task,
            expected,
        })
    }

    /// Whether two routes reach the same worker.
    #[cfg(test)]
    pub(crate) fn shares_worker_with(&self, other: &Self) -> bool {
        self.key == other.key
    }

    /// Refuse a request that is not this route's task or names other engines.
    pub(super) fn admit(&self, request: &ExecuteRequestV2) -> Result<(), WorkerError> {
        let requested = infer_task_for_execute_v2(request.task);
        if requested != self.task.infer_task() {
            return Err(PlanMismatch::Task {
                request: requested,
                route: self.task,
            }
            .into());
        }
        if self.expected.named_by(&request.payload) {
            Ok(())
        } else {
            // Rendered only on this defect path.
            Err(PlanMismatch::Engines {
                route: self.task,
                requested: format!("{:?}", execute_request_overrides(request)),
                planned: format!("{:?}", self.expected),
            }
            .into())
        }
    }
}

/// A request that disagrees with the route it was sent on.
///
/// Always a defect in this program: a request builder and the job's plan chose
/// engines separately. Refused rather than dispatched, because dispatching it
/// on the route would hand the worker a request for a model it was not spawned
/// with, and dispatching it on its own key is the duplicate worker this module
/// exists to prevent.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PlanMismatch {
    /// The request is for another task than the route.
    #[error("a {request:?} request was sent on a {route:?} route")]
    Task {
        /// The request's pool task.
        request: InferTask,
        /// The route's task.
        route: GpuTask,
    },
    /// The request names other engines than the job planned for the route.
    #[error("a {route:?} request names {requested}, but the job planned {planned}")]
    Engines {
        /// The route's task.
        route: GpuTask,
        /// What the request named, rendered for the operator.
        requested: String,
        /// What the plan expected, rendered for the operator.
        planned: String,
    },
}

/// Whether the GPU profile hosts `task`, from the profile map's own owner.
#[cfg(test)]
mod tests;
