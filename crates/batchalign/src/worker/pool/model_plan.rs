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
//! Later phases replace exact-key matching with monotone per-slot loadouts,
//! give speaker engines a slot of their own, have request builders read their
//! engines from the route, and move every engine default out of the Python
//! worker.

use std::collections::BTreeMap;

use batchalign_types::worker::InferTask;

use crate::api::WorkerLanguage;
use crate::transcribe::{AsrBackend, AsrWorkerMode};
use crate::types::engines::{AsrEngineName, EngineOverrides, FaEngineName};
use crate::types::worker_v2::{ExecuteRequestV2, TaskRequestV2};
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
    /// A job whose GPU requests select no engine: diarization, speaker
    /// identity, replayed transcription, and every command with no GPU
    /// requests at all. Speaker requests remain routable; they select nothing.
    pub(crate) const fn without_engines() -> Self {
        Self {
            asr: None,
            fa: None,
        }
    }

    /// A transcription or benchmark job: the ASR backend its admitted plan
    /// selected, with the job's engine extras.
    pub(crate) fn transcription(asr: AsrBackend, extras: &BTreeMap<String, String>) -> Self {
        Self {
            asr: PlannedWorkerAsr::worker_hosted(asr, extras.clone()),
            fa: None,
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
        }
    }

    /// A plan for exercising the shared GPU worker from integration tests:
    /// stock Whisper ASR with no extras. Production plans come only from
    /// family dispatch plans.
    #[doc(hidden)]
    pub fn local_whisper_for_test() -> Self {
        Self::transcription(
            AsrBackend::Worker(AsrWorkerMode::LocalWhisperV2),
            &BTreeMap::new(),
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
            // Speaker requests select no engine; any job may send them.
            GpuTask::Speaker => Some(RouteEngines::Speaker),
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
            | InferTask::Avqi => None,
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
    Speaker,
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
            (Self::Speaker, TaskRequestV2::Speaker(_) | TaskRequestV2::SpeakerEmbedding(_)) => true,
            // The request's task was already matched against the route's, so
            // a payload of another shape is a request whose task and payload
            // disagree: it names nothing this route expects.
            (Self::Asr { .. } | Self::Fa(_) | Self::Speaker, _) => false,
        }
    }
}

/// Permission to send one job's requests for one task to one worker key.
///
/// Minted only by [`super::WorkerPool::gpu_route`].
#[derive(Debug, Clone)]
pub(crate) struct GpuRoute {
    pub(super) key: WorkerKey,
    task: GpuTask,
    expected: RouteEngines,
}

impl GpuRoute {
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
            engine_selection: EngineSelection::for_job_plan(plan.identity(task), target, mode),
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
        let requested = infer_task_for_execute_v2(request.task)?;
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
fn gpu_profile_hosts(task: InferTask) -> bool {
    crate::worker::WorkerProfile::for_task(task) == crate::worker::WorkerProfile::Gpu
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::LanguageCode3;
    use crate::worker::WorkerProfile;

    fn english() -> WorkerLanguage {
        WorkerLanguage::from(LanguageCode3::eng())
    }

    fn whisper() -> AsrBackend {
        AsrBackend::Worker(AsrWorkerMode::LocalWhisperV2)
    }

    fn route(plan: &GpuModelPlan, task: GpuTask, mode: WorkerBootstrapMode) -> GpuRoute {
        GpuRoute::new(plan, task, english(), mode).expect("declared task")
    }

    /// The measured defect: one align job with Whisper UTR, under the profile
    /// bootstrap modes, must key its UTR and FA requests to ONE worker.
    #[test]
    fn an_alignment_jobs_utr_and_fa_requests_share_one_worker_key() {
        let plan = GpuModelPlan::alignment(FaEngineName::Wave2Vec, Some(whisper()));
        for mode in [
            WorkerBootstrapMode::Profile,
            WorkerBootstrapMode::LazyProfile,
        ] {
            let asr = route(&plan, GpuTask::Asr, mode);
            let fa = route(&plan, GpuTask::Fa, mode);
            assert_eq!(asr.key, fa.key, "{mode:?}");
            assert_eq!(asr.key.target, WorkerTarget::Profile(WorkerProfile::Gpu));
            assert_eq!(
                asr.key.engine_selection.overrides().fa,
                Some(FaEngineName::Wave2Vec)
            );
            assert!(asr.key.engine_selection.overrides().asr.is_some());
        }
    }

    /// The probe and pre-scaling key from the plan for the command's own
    /// task, so they select the worker the routes reach.
    #[test]
    fn the_probe_key_is_the_routes_key() {
        let plan = GpuModelPlan::alignment(FaEngineName::Wave2Vec, Some(whisper()));
        let probe =
            GpuRoute::worker_key(&plan, GpuTask::Fa, english(), WorkerBootstrapMode::Profile);
        assert_eq!(
            probe,
            route(&plan, GpuTask::Asr, WorkerBootstrapMode::Profile).key
        );
    }

    /// A Rev.AI transcription declares no worker ASR, yet its probe still has
    /// a worker to select: the one its speaker requests would reach, keyed
    /// exactly as before plans existed (`{fa}` from the profile preload).
    #[test]
    fn a_rust_hosted_transcription_still_has_a_probe_worker() {
        let plan = GpuModelPlan::transcription(AsrBackend::RustRevAi, &BTreeMap::new());
        let probe =
            GpuRoute::worker_key(&plan, GpuTask::Asr, english(), WorkerBootstrapMode::Profile);
        assert_eq!(probe.engine_selection.overrides().asr, None);
        assert_eq!(
            probe.engine_selection.overrides().fa,
            Some(<FaEngineName as crate::types::engines::SelectableEngine>::DEFAULT)
        );
        assert_eq!(
            probe,
            route(&plan, GpuTask::Speaker, WorkerBootstrapMode::Profile).key
        );
    }

    /// A transcription job's diarization requests reach its ASR worker.
    #[test]
    fn a_transcription_jobs_speaker_requests_share_its_asr_worker_key() {
        let plan = GpuModelPlan::transcription(whisper(), &BTreeMap::new());
        let asr = route(&plan, GpuTask::Asr, WorkerBootstrapMode::Profile);
        let speaker = route(&plan, GpuTask::Speaker, WorkerBootstrapMode::Profile);
        assert_eq!(asr.key, speaker.key);
    }

    /// Small hosts load one task per worker, so the keys stay per task there.
    #[test]
    fn task_mode_keys_each_task_by_its_own_engines() {
        let plan = GpuModelPlan::alignment(FaEngineName::Wave2Vec, Some(whisper()));
        let asr = route(&plan, GpuTask::Asr, WorkerBootstrapMode::Task);
        let fa = route(&plan, GpuTask::Fa, WorkerBootstrapMode::Task);
        assert_ne!(asr.key, fa.key);
        assert_eq!(fa.key.target, WorkerTarget::InferTask(InferTask::Fa));
        assert_eq!(fa.key.engine_selection.overrides().asr, None);
    }

    /// Rust-hosted ASR is not a worker resource: Rev.AI UTR plans FA alone,
    /// and neither Rev.AI nor whisper.cpp transcription can route ASR.
    #[test]
    fn rust_hosted_asr_never_enters_a_plan() {
        let rev_utr = GpuModelPlan::alignment(FaEngineName::Wave2Vec, Some(AsrBackend::RustRevAi));
        assert_eq!(
            rev_utr,
            GpuModelPlan::alignment(FaEngineName::Wave2Vec, None)
        );
        for plan in [
            rev_utr,
            GpuModelPlan::transcription(AsrBackend::RustRevAi, &BTreeMap::new()),
            GpuModelPlan::transcription(AsrBackend::RustWhisperRs, &BTreeMap::new()),
        ] {
            assert_eq!(
                GpuRoute::new(&plan, GpuTask::Asr, english(), WorkerBootstrapMode::Profile)
                    .map(|_| ()),
                Err(GpuRouteRefusal::Undeclared { task: GpuTask::Asr })
            );
        }
    }

    /// A transcription job never declared FA, so it cannot route FA requests.
    #[test]
    fn an_undeclared_task_is_refused() {
        let plan = GpuModelPlan::transcription(whisper(), &BTreeMap::new());
        assert_eq!(
            GpuRoute::new(&plan, GpuTask::Fa, english(), WorkerBootstrapMode::Profile).map(|_| ()),
            Err(GpuRouteRefusal::Undeclared { task: GpuTask::Fa })
        );
    }

    /// Engine extras configure the loaded model, so they are identity.
    #[test]
    fn asr_extras_are_part_of_the_worker_key() {
        let plain = GpuModelPlan::transcription(
            AsrBackend::Worker(AsrWorkerMode::HkQwenV2),
            &BTreeMap::new(),
        );
        let large = GpuModelPlan::transcription(
            AsrBackend::Worker(AsrWorkerMode::HkQwenV2),
            &BTreeMap::from([("qwen_model".to_owned(), "Qwen/Qwen3-ASR-1.7B".to_owned())]),
        );
        assert_ne!(
            route(&plain, GpuTask::Asr, WorkerBootstrapMode::Profile).key,
            route(&large, GpuTask::Asr, WorkerBootstrapMode::Profile).key
        );
    }

    /// Wav2vec's failing groups retry on its fallback engine, on a worker
    /// identified by that engine alone, exactly as the fallback request's own
    /// key was.
    #[test]
    fn the_fa_fallback_routes_to_a_worker_keyed_by_the_fallback_engine() {
        let plan = GpuModelPlan::alignment(FaEngineName::Wave2Vec, Some(whisper()));
        let fallback = route(&plan, GpuTask::FaFallback, WorkerBootstrapMode::Profile);
        let target = FaEngineName::Wave2Vec
            .fallback_policy()
            .retry_target()
            .expect("wav2vec has a fallback");
        assert_eq!(fallback.key.engine_selection.overrides().fa, Some(target));
        assert_eq!(fallback.key.engine_selection.overrides().asr, None);
        assert_ne!(
            fallback.key,
            route(&plan, GpuTask::Fa, WorkerBootstrapMode::Profile).key
        );
    }

    #[test]
    fn gpu_tasks_are_exactly_the_tasks_the_gpu_profile_hosts() {
        for task in InferTask::ALL.iter().copied() {
            assert_eq!(
                GpuTask::of(task).is_some(),
                gpu_profile_hosts(task),
                "{task:?}"
            );
        }
        for task in [GpuTask::Asr, GpuTask::Fa, GpuTask::Speaker] {
            assert_eq!(GpuTask::of(task.infer_task()), Some(task));
        }
        assert_eq!(
            GpuTask::of(GpuTask::FaFallback.infer_task()),
            Some(GpuTask::Fa)
        );
    }

    fn asr_request(
        backend: crate::types::worker_v2::AsrBackendV2,
        extras: BTreeMap<String, String>,
    ) -> ExecuteRequestV2 {
        use crate::types::worker_v2::{
            AsrInputV2, AsrRequestV2, AsrRequestedModelsV2, InferenceTaskV2, ModelIdV2,
            PreparedAudioInputV2, RequestedModelV2, RequestedRevisionV2, WorkerArtifactIdV2,
            WorkerRequestIdV2,
        };
        ExecuteRequestV2 {
            request_id: WorkerRequestIdV2::from("model-plan-test"),
            task: InferenceTaskV2::Asr,
            payload: TaskRequestV2::Asr(AsrRequestV2 {
                lang: english(),
                backend,
                models: AsrRequestedModelsV2::Whisper {
                    asr: RequestedModelV2 {
                        id: ModelIdV2::from_static("openai/whisper-large-v3"),
                        revision: RequestedRevisionV2::Unpinned,
                    },
                },
                input: AsrInputV2::PreparedAudio(PreparedAudioInputV2 {
                    audio_ref_id: WorkerArtifactIdV2::from("audio-1"),
                }),
                extras,
                decode_budget_seconds: None,
            }),
            attachments: Vec::new(),
        }
    }

    /// A request whose builder chose engines apart from the plan is refused
    /// on its route, rather than sent to a worker spawned for other models.
    #[test]
    fn a_request_naming_other_engines_than_its_route_is_refused() {
        use crate::types::worker_v2::AsrBackendV2;
        let plan = GpuModelPlan::transcription(whisper(), &BTreeMap::new());
        let asr = route(&plan, GpuTask::Asr, WorkerBootstrapMode::Profile);
        assert!(
            asr.admit(&asr_request(AsrBackendV2::LocalWhisper, BTreeMap::new()))
                .is_ok()
        );
        for request in [
            asr_request(AsrBackendV2::HkQwen, BTreeMap::new()),
            asr_request(
                AsrBackendV2::LocalWhisper,
                BTreeMap::from([("whisper_cpu_dtype".to_owned(), "float32".to_owned())]),
            ),
        ] {
            assert!(matches!(
                asr.admit(&request),
                Err(WorkerError::PlanMismatch(PlanMismatch::Engines {
                    route: GpuTask::Asr,
                    ..
                }))
            ));
        }

        let alignment = GpuModelPlan::alignment(FaEngineName::Wave2Vec, None);
        let fa = route(&alignment, GpuTask::Fa, WorkerBootstrapMode::Profile);
        assert!(matches!(
            fa.admit(&asr_request(AsrBackendV2::LocalWhisper, BTreeMap::new())),
            Err(WorkerError::PlanMismatch(PlanMismatch::Task {
                request: InferTask::Asr,
                route: GpuTask::Fa,
            }))
        ));
    }

    /// No GPU request reaches a worker except on a route: the request-keyed
    /// dispatch refuses it before deriving any key or spawning anything.
    #[tokio::test]
    async fn request_keyed_dispatch_refuses_gpu_requests() {
        let pool = super::super::WorkerPool::new(super::super::PoolConfig::default());
        let error = pool
            .dispatch_execute_v2(
                english(),
                &asr_request(
                    crate::types::worker_v2::AsrBackendV2::LocalWhisper,
                    BTreeMap::new(),
                ),
            )
            .await
            .expect_err("an unrouted ASR request is refused");
        assert!(
            matches!(
                error,
                WorkerError::UnroutedGpuRequest {
                    task: InferTask::Asr
                }
            ),
            "{error:?}"
        );
    }
}
