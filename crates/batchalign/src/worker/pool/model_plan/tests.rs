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
fn speaker_request(backend: SpeakerBackendV2) -> ExecuteRequestV2 {
    use crate::types::worker_v2::{
        InferenceTaskV2, SpeakerInputV2, SpeakerPreparedAudioInputV2, SpeakerRequestV2,
        WorkerArtifactIdV2, WorkerRequestIdV2,
    };
    ExecuteRequestV2 {
        request_id: WorkerRequestIdV2::from("speaker-plan-test"),
        task: InferenceTaskV2::Speaker,
        payload: TaskRequestV2::Speaker(SpeakerRequestV2 {
            backend,
            input: SpeakerInputV2::PreparedAudio(SpeakerPreparedAudioInputV2 {
                audio_ref_id: WorkerArtifactIdV2::from("audio-1"),
            }),
            expected_speakers: None,
        }),
        attachments: Vec::new(),
    }
}
enum AsrCase {
    Whisper,
    Qwen,
    WhisperDtype,
}
fn asr_request(case: AsrCase) -> ExecuteRequestV2 {
    use crate::types::worker_v2::*;
    let named = |id| RequestedModelV2 {
        id: ModelIdV2::from_static(id),
        revision: RequestedRevisionV2::Unpinned,
    };
    let (backend, models, extras) = match case {
        AsrCase::Whisper => (
            AsrBackendV2::LocalWhisper,
            AsrRequestedModelsV2::Whisper {
                asr: named("openai/whisper-large-v3"),
            },
            BTreeMap::new(),
        ),
        AsrCase::WhisperDtype => (
            AsrBackendV2::LocalWhisper,
            AsrRequestedModelsV2::Whisper {
                asr: named("openai/whisper-large-v3"),
            },
            BTreeMap::from([("whisper_cpu_dtype".into(), "float32".into())]),
        ),
        AsrCase::Qwen => (
            AsrBackendV2::HkQwen,
            AsrRequestedModelsV2::Qwen {
                asr: named("fixture/asr"),
                aligner: named("fixture/aligner"),
            },
            BTreeMap::new(),
        ),
    };
    let payload = TaskRequestV2::Asr(AsrRequestV2 {
        lang: english(),
        backend,
        models,
        input: AsrInputV2::PreparedAudio(PreparedAudioInputV2 {
            audio_ref_id: WorkerArtifactIdV2::from("audio-1"),
        }),
        extras,
        decode_budget_seconds: None,
    });
    ExecuteRequestV2 {
        request_id: WorkerRequestIdV2::from("model-plan-test"),
        task: payload.task(),
        payload,
        attachments: Vec::new(),
    }
}

/// Guards: plan families share their primary profile worker, separate task-mode
/// and fallback workers, exclude Rust-hosted ASR and retain actual configured extras.
/// Kind: policy
/// Not a type: The choices of worker sharing, fallback and engine identity remain semantic
/// policies after private route fields prevent retargeting an admitted declaration.
/// State: typed plans, requirements and requests require no worker, model or host probe.
/// Concurrency: none
#[test]
fn family_routes_preserve_bootstrap_probe_and_fallback_policy() {
    let alignment = GpuModelPlan::alignment(FaEngineName::Wave2Vec, Some(whisper()));
    let transcription = GpuModelPlan::transcription(
        Some(whisper()),
        &BTreeMap::new(),
        Some(SpeakerBackendV2::Pyannote),
    );
    for mode in [
        WorkerBootstrapMode::Profile,
        WorkerBootstrapMode::LazyProfile,
        WorkerBootstrapMode::Task,
    ] {
        let asr = route(&alignment, GpuTask::Asr, mode);
        let fa = route(&alignment, GpuTask::Fa, mode);
        let trans_asr = route(&transcription, GpuTask::Asr, mode);
        let speaker = route(&transcription, GpuTask::Speaker, mode);
        match mode {
            WorkerBootstrapMode::Profile | WorkerBootstrapMode::LazyProfile => {
                assert_eq!(asr.key(), fa.key());
                assert_eq!(trans_asr.key(), speaker.key());
                assert_eq!(asr.key().target, WorkerTarget::Profile(WorkerProfile::Gpu));
                assert!(asr.key().engine_selection.overrides().asr.is_some());
                assert_eq!(
                    asr.key().engine_selection.overrides().fa,
                    Some(FaEngineName::Wave2Vec)
                );
                assert_eq!(
                    GpuRoute::worker_key(&alignment, GpuTask::Fa, english(), mode),
                    *asr.key()
                );
            }
            WorkerBootstrapMode::Task => {
                assert_ne!(asr.key(), fa.key());
                assert_ne!(trans_asr.key(), speaker.key());
                assert_eq!(fa.key().target, WorkerTarget::InferTask(InferTask::Fa));
                assert_eq!(fa.key().engine_selection.overrides().asr, None);
            }
        }
        let fallback = route(&alignment, GpuTask::FaFallback, mode);
        assert_eq!(
            fallback.key().engine_selection.overrides().fa,
            FaEngineName::Wave2Vec.fallback_policy().retry_target()
        );
        assert_eq!(fallback.key().engine_selection.overrides().asr, None);
        assert_ne!(fallback.key(), fa.key());
    }
    let hosted = GpuModelPlan::transcription(
        Some(AsrBackend::RustRevAi),
        &BTreeMap::new(),
        Some(SpeakerBackendV2::PyannoteAi),
    );
    let probe = GpuRoute::worker_key(
        &hosted,
        GpuTask::Asr,
        english(),
        WorkerBootstrapMode::Profile,
    );
    assert_eq!(probe.engine_selection.overrides().asr, None);
    assert_eq!(probe.engine_selection.overrides().fa, None);
    assert_eq!(
        probe,
        *route(&hosted, GpuTask::Speaker, WorkerBootstrapMode::Profile).key()
    );
    let small = GpuModelPlan::transcription(
        Some(AsrBackend::Worker(AsrWorkerMode::HkQwenV2)),
        &BTreeMap::new(),
        None,
    );
    let large = GpuModelPlan::transcription(
        Some(AsrBackend::Worker(AsrWorkerMode::HkQwenV2)),
        &BTreeMap::from([("qwen_model".into(), "fixture/large".into())]),
        None,
    );
    assert_ne!(
        route(&small, GpuTask::Asr, WorkerBootstrapMode::Profile).key(),
        route(&large, GpuTask::Asr, WorkerBootstrapMode::Profile).key()
    );
}

/// Guards: speaker engine choices share the intended worker while their actual
/// load wire overrides retain diarization, embedding or hosted model absence.
/// Kind: wire
/// Not a type: model-sharing and named wire-key choices are semantic and byte
/// obligations after TaskLoad and route constructors retain their actual producers.
/// State: typed speaker plans and the normal TaskLoad producer need no model or process.
/// Concurrency: none
#[test]
fn the_speaker_model_is_a_load_not_a_worker_key() {
    let plans = [
        GpuModelPlan::diarization(SpeakerBackendV2::Pyannote),
        GpuModelPlan::diarization(SpeakerBackendV2::PyannoteAi),
        GpuModelPlan::speaker_identity(),
    ];
    for mode in [
        WorkerBootstrapMode::Profile,
        WorkerBootstrapMode::LazyProfile,
        WorkerBootstrapMode::Task,
    ] {
        let keys: Vec<WorkerKey> = plans
            .iter()
            .map(|plan| route(plan, GpuTask::Speaker, mode).key)
            .collect();
        assert_eq!(keys[0], keys[1], "{mode:?}");
        assert_eq!(keys[0], keys[2], "{mode:?}");
    }
    let load = |plan: &GpuModelPlan| {
        crate::worker::pool::TaskLoad::new(
            InferTask::Speaker,
            plan.load_overrides(GpuTask::Speaker),
        )
    };
    let [local, hosted, embedding] = plans.each_ref().map(load);
    assert_eq!(
        local
            .wire_overrides()
            .and_then(|o| o.get("speaker"))
            .map(String::as_str),
        Some("pyannote")
    );
    assert_eq!(
        embedding
            .wire_overrides()
            .and_then(|o| o.get("speaker_embedding"))
            .map(String::as_str),
        Some("pyannote")
    );
    assert_eq!(hosted.wire_overrides(), None);
    assert_ne!(local, embedding);
    assert_ne!(local, hosted);
    // A diarized transcription shares the worker of one that runs no
    // diarization.
    let plain = GpuModelPlan::transcription(Some(whisper()), &BTreeMap::new(), None);
    let diarized = GpuModelPlan::transcription(
        Some(whisper()),
        &BTreeMap::new(),
        Some(SpeakerBackendV2::Pyannote),
    );
    assert_eq!(
        route(&plain, GpuTask::Asr, WorkerBootstrapMode::Profile).key,
        route(&diarized, GpuTask::Asr, WorkerBootstrapMode::Profile).key
    );
}

/// Guards: undeclared tasks, foreign models, changed extras and wrong task packets
/// are refused at route admission, while the declared packet is admitted.
/// Kind: boundary
/// Not a type: an outside request can name a different task or engine than its
/// job declaration. Admission must observe those actual typed wire values once.
/// State: closed valid request fixtures preserve backend/model composition without IO.
/// Concurrency: none
#[test]
fn route_admission_refuses_foreign_task_and_engine_choices() {
    let plan = GpuModelPlan::transcription(Some(whisper()), &BTreeMap::new(), None);
    for task in [GpuTask::Fa, GpuTask::Speaker] {
        assert!(matches!(
            GpuRoute::new(&plan, task, english(), WorkerBootstrapMode::Profile),
            Err(GpuRouteRefusal::Undeclared { .. })
        ));
    }
    for backend in [AsrBackend::RustRevAi, AsrBackend::RustWhisperRs] {
        let rust = GpuModelPlan::transcription(Some(backend), &BTreeMap::new(), None);
        assert!(matches!(
            GpuRoute::new(&rust, GpuTask::Asr, english(), WorkerBootstrapMode::Profile),
            Err(GpuRouteRefusal::Undeclared { task: GpuTask::Asr })
        ));
    }
    assert_eq!(
        GpuModelPlan::alignment(FaEngineName::Wave2Vec, Some(AsrBackend::RustRevAi)),
        GpuModelPlan::alignment(FaEngineName::Wave2Vec, None)
    );
    let asr = route(&plan, GpuTask::Asr, WorkerBootstrapMode::Profile);
    assert!(asr.admit(&asr_request(AsrCase::Whisper)).is_ok());
    for case in [AsrCase::Qwen, AsrCase::WhisperDtype] {
        assert!(matches!(
            asr.admit(&asr_request(case)),
            Err(WorkerError::PlanMismatch(PlanMismatch::Engines {
                route: GpuTask::Asr,
                ..
            }))
        ));
    }
    let fa = route(
        &GpuModelPlan::alignment(FaEngineName::Wave2Vec, None),
        GpuTask::Fa,
        WorkerBootstrapMode::Profile,
    );
    assert!(matches!(
        fa.admit(&asr_request(AsrCase::Whisper)),
        Err(WorkerError::PlanMismatch(PlanMismatch::Task {
            request: InferTask::Asr,
            route: GpuTask::Fa
        }))
    ));
    let local = route(
        &GpuModelPlan::diarization(SpeakerBackendV2::Pyannote),
        GpuTask::Speaker,
        WorkerBootstrapMode::Profile,
    );
    assert!(
        local
            .admit(&speaker_request(SpeakerBackendV2::Pyannote))
            .is_ok()
    );
    assert!(matches!(
        local.admit(&speaker_request(SpeakerBackendV2::PyannoteAi)),
        Err(WorkerError::PlanMismatch(PlanMismatch::Engines { .. }))
    ));
    let embedding = route(
        &GpuModelPlan::speaker_identity(),
        GpuTask::Speaker,
        WorkerBootstrapMode::Profile,
    );
    assert!(matches!(
        embedding.admit(&speaker_request(SpeakerBackendV2::Pyannote)),
        Err(WorkerError::PlanMismatch(PlanMismatch::Engines { .. }))
    ));
}

/// Guards: only the GPU host's task family acquires GPU route vocabulary, with
/// the explicit FA-fallback alias retaining the same underlying inference task.
/// Kind: policy
/// Not a type: the infrastructure/wire task family relation is a semantic mapping
/// between separate closed vocabularies, including the many-to-one fallback alias.
/// State: immutable task values require no runtime or pool.
/// Concurrency: none
#[test]
fn gpu_tasks_are_exactly_the_tasks_the_gpu_profile_hosts() {
    for task in InferTask::ALL.iter().copied() {
        assert_eq!(
            GpuTask::of(task).is_some(),
            matches!(WorkerProfile::for_task(task), WorkerProfile::Gpu)
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

/// Guards: the normal request-keyed entry refuses GPU execution before spawning,
/// requiring the job's declared route for every GPU request.
/// Kind: boundary
/// Not a type: the public raw request entry retains a typed refusal for callers
/// that supply GPU wire packets, while admitted execution requires a private plan.
/// State: a local declared empty pool enters the actual API with no spawn or model.
/// Concurrency: the runtime and unused pool belong only to this test.
#[tokio::test]
async fn request_keyed_dispatch_refuses_gpu_requests() {
    let pool = super::super::WorkerPool::new(super::super::PoolConfig::for_tests());
    assert!(matches!(
        pool.dispatch_execute_v2(english(), &asr_request(AsrCase::Whisper))
            .await,
        Err(WorkerError::UnroutedGpuRequest {
            task: InferTask::Asr
        })
    ));
}
