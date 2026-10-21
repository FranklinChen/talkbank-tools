//! V2 execute dispatch helpers, task mapping and engine override resolution.
//!
//! These pure functions bridge the V2 execute protocol (typed per-backend
//! requests) to the pool's typed worker-key abstraction. Extracted from
//! `mod.rs` for browsability.

use crate::api::WorkerLanguage;
use crate::types::engines::{AsrEngineName, EngineOverrides, FaEngineName, TranslateEngineName};
use crate::types::worker_v2::{AsrBackendV2, ExecuteRequestV2, InferenceTaskV2, TaskRequestV2};
use crate::worker::error::WorkerError;
use crate::worker::{InferTask, WorkerBootstrapMode, WorkerTarget};

use super::{EngineSelection, WorkerKey};

/// Map a V2 inference task enum to the pool's infer-task vocabulary.
pub(super) fn infer_task_for_execute_v2(task: InferenceTaskV2) -> Result<InferTask, WorkerError> {
    match task {
        InferenceTaskV2::Morphosyntax => Ok(InferTask::Morphosyntax),
        InferenceTaskV2::Utseg => Ok(InferTask::Utseg),
        InferenceTaskV2::Translate => Ok(InferTask::Translate),
        InferenceTaskV2::Coref => Ok(InferTask::Coref),
        InferenceTaskV2::Asr => Ok(InferTask::Asr),
        InferenceTaskV2::ForcedAlignment => Ok(InferTask::Fa),
        InferenceTaskV2::Speaker => Ok(InferTask::Speaker),
        // Deliberately the SAME pool task as diarization, not a new one.
        //
        // A worker-pool key names which model HOST a request needs, and both
        // tasks are served by the speaker host in the same Python module
        // family. Splitting them would spawn a second worker process to hold a
        // model the first one could have loaded, for no isolation anybody
        // needs. What makes this safe is that neither model is loaded at
        // bootstrap: the diarization pipeline and the embedding model are both
        // lazy, so a worker that only ever sees embedding requests never
        // constructs the diarization pipeline, and therefore never reaches the
        // gated calibration artifact that pipeline pulls in.
        InferenceTaskV2::SpeakerEmbedding => Ok(InferTask::Speaker),
        InferenceTaskV2::Opensmile => Ok(InferTask::Opensmile),
        InferenceTaskV2::Avqi => Ok(InferTask::Avqi),
    }
}

/// Derive the worker-pool key for one text or IO V2 execute request.
///
/// A GPU-profile request is refused ([`WorkerError::UnroutedGpuRequest`]): a
/// GPU worker hosts every task of its job's model plan, so its key comes from
/// the plan (`WorkerPool::gpu_route`), never from one request.
pub(super) fn execute_v2_worker_key(
    lang: WorkerLanguage,
    request: &ExecuteRequestV2,
    bootstrap_mode: WorkerBootstrapMode,
) -> Result<WorkerKey, WorkerError> {
    let infer_task = infer_task_for_execute_v2(request.task)?;
    if super::GpuTask::of(infer_task).is_some() {
        return Err(WorkerError::UnroutedGpuRequest { task: infer_task });
    }
    let target = WorkerTarget::from_infer_task(infer_task, bootstrap_mode);

    // The selected engine is part of the worker identity in every bootstrap
    // mode. A lazy worker loads the recipe on demand, but it is not allowed to
    // change recipes afterward: sharing one task-only key between Wave2Vec and
    // Whisper made the Rust and Python "already loaded" caches return the
    // first engine for later requests. It also allowed a concurrent request to
    // switch process-global model state underneath in-flight inference.
    //
    // The same derivation the command route uses, from what this request
    // names; the two keys agree by construction.
    let engine_selection = EngineSelection::derive(
        infer_task,
        execute_request_overrides(request),
        target,
        bootstrap_mode,
    );

    Ok(WorkerKey {
        target,
        language: lang,
        engine_selection,
    })
}

/// The engines a V2 request itself names, before any target-driven filling.
pub(super) fn execute_request_overrides(request: &ExecuteRequestV2) -> EngineOverrides {
    match &request.payload {
        TaskRequestV2::Asr(request) => EngineOverrides {
            asr: Some(asr_backend_engine(request.backend)),
            extras: request.extras.clone(),
            ..EngineOverrides::default()
        },
        TaskRequestV2::ForcedAlignment(request) => EngineOverrides {
            fa: Some(FaEngineName::from_worker_backend(request.backend)),
            ..EngineOverrides::default()
        },
        TaskRequestV2::Translate(request) => EngineOverrides {
            translate: Some(TranslateEngineName::from_worker_backend(request.engine)),
            ..EngineOverrides::default()
        },
        // One implementation each: nothing to select, so nothing enters the
        // key. Listed rather than wildcarded so a new task has to decide.
        TaskRequestV2::Morphosyntax(_)
        | TaskRequestV2::Utseg(_)
        | TaskRequestV2::Coref(_)
        | TaskRequestV2::Speaker(_)
        | TaskRequestV2::SpeakerEmbedding(_)
        | TaskRequestV2::Opensmile(_)
        | TaskRequestV2::Avqi(_) => EngineOverrides::default(),
    }
}

/// Extract the ensure_task parameters (task name + engine overrides map) from a
/// V2 execute request, without the JSON round-trip.
///
/// Returns `None` for tasks that don't need model loading (e.g., text tasks in
/// eager-profile mode). Used by the LazyProfile dispatch path.
pub(super) fn ensure_task_params(
    request: &ExecuteRequestV2,
) -> Result<
    (
        InferTask,
        Option<std::collections::BTreeMap<String, String>>,
    ),
    WorkerError,
> {
    let task = infer_task_for_execute_v2(request.task)?;

    // A lazy worker's identity is the task's own keys, whatever its target
    // would preload eagerly.
    let selection = EngineSelection::derive(
        task,
        execute_request_overrides(request),
        WorkerTarget::from_infer_task(task, WorkerBootstrapMode::LazyProfile),
        WorkerBootstrapMode::LazyProfile,
    );
    let overrides = (!selection.is_none()).then(|| selection.overrides().dispatch_overrides());

    Ok((task, overrides))
}

pub(super) fn asr_backend_engine(backend: AsrBackendV2) -> AsrEngineName {
    match backend {
        AsrBackendV2::LocalWhisper => AsrEngineName::Whisper,
        AsrBackendV2::WhisperHub => AsrEngineName::WhisperHub,
        AsrBackendV2::HkTencent => AsrEngineName::HkTencent,
        AsrBackendV2::HkAliyun => AsrEngineName::HkAliyun,
        AsrBackendV2::HkFunaudio => AsrEngineName::HkFunaudio,
        AsrBackendV2::HkQwen => AsrEngineName::HkQwen,
        AsrBackendV2::Revai => AsrEngineName::RevAi,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::LanguageCode3;
    use crate::types::worker_v2::{MorphosyntaxRequestV2, WorkerArtifactIdV2, WorkerRequestIdV2};
    use crate::worker::{WorkerBootstrapMode, WorkerProfile, WorkerTarget};

    fn request_with_payload(task: InferenceTaskV2, payload: TaskRequestV2) -> ExecuteRequestV2 {
        ExecuteRequestV2 {
            request_id: WorkerRequestIdV2::from("req-1"),
            task,
            payload,
            attachments: Vec::new(),
        }
    }

    #[test]
    fn maps_forced_alignment_execute_v2_to_fa_worker_profile() {
        assert_eq!(
            infer_task_for_execute_v2(InferenceTaskV2::ForcedAlignment).unwrap(),
            InferTask::Fa
        );
    }

    /// The worker key a transcription job's plan gives its ASR requests for
    /// `mode` in `code`, under the lazy profile mode.
    fn asr_worker_key(
        mode: crate::transcribe::AsrWorkerMode,
        code: &str,
        extras: std::collections::BTreeMap<String, String>,
    ) -> WorkerKey {
        let language = LanguageCode3::try_new(code).expect("language code");
        let plan = super::super::GpuModelPlan::transcription(
            crate::transcribe::AsrBackend::Worker(mode),
            &extras,
        );
        super::super::GpuRoute::worker_key(
            &plan,
            super::super::GpuTask::Asr,
            WorkerLanguage::from(language),
            WorkerBootstrapMode::LazyProfile,
        )
    }

    /// The user's engine extras reach the worker's spawn configuration through
    /// the plan-derived key: every key the job passed via
    /// `--engine-overrides` is in the JSON the worker is spawned with.
    /// (Regression, 2026-05-27: a Qwen job's `qwen_model` was dropped and the
    /// worker silently loaded the default model.)
    #[test]
    fn asr_extras_reach_the_worker_spawn_configuration() {
        let mut extras = std::collections::BTreeMap::new();
        extras.insert("qwen_model".to_owned(), "Qwen/Qwen3-ASR-0.6B".to_owned());
        extras.insert("qwen_device".to_owned(), "cpu".to_owned());
        let json = asr_worker_key(
            crate::transcribe::AsrWorkerMode::HkQwenV2,
            "yue",
            extras.clone(),
        )
        .engine_selection
        .worker_config_json();
        let parsed: std::collections::BTreeMap<String, String> =
            serde_json::from_str(&json).expect("engine_overrides JSON must round-trip");
        assert_eq!(parsed.get("asr").map(String::as_str), Some("qwen"));
        for (k, v) in &extras {
            assert_eq!(parsed.get(k), Some(v), "extras key {k:?} was dropped");
        }
    }

    /// The spawn override `name` a worker for `key` receives, parsed.
    fn spawn_override<T: serde::de::DeserializeOwned>(key: &WorkerKey, name: &str) -> T {
        let json = super::super::lifecycle::spawn_engine_overrides(key).expect("spawn");
        let map: std::collections::BTreeMap<String, String> =
            serde_json::from_str(&json).expect("spawn overrides are a JSON map");
        serde_json::from_str(map.get(name).expect("policy key")).expect("policy parses")
    }

    #[test]
    fn a_qwen_worker_spawns_with_its_language_recognition_policy() {
        use crate::transcribe::segmentation::qwen_recognition;
        use crate::types::worker_v2::{QWEN_RECOGNITION_KEY, QwenRecognitionV2};
        // The wiring under test: each worker receives exactly what the table
        // decides for its language (the values are pinned by the table's own
        // tests in `transcribe::segmentation`).
        for code in ["nan", "yue"] {
            let language = LanguageCode3::try_new(code).expect("language code");
            let key = asr_worker_key(
                crate::transcribe::AsrWorkerMode::HkQwenV2,
                code,
                Default::default(),
            );
            assert_eq!(
                spawn_override::<QwenRecognitionV2>(&key, QWEN_RECOGNITION_KEY),
                qwen_recognition(Some(&language)),
                "{code}"
            );
        }
    }

    /// A stock Whisper worker is told its language by Whisper's own token,
    /// decided by the control plane (the decisions themselves are pinned by
    /// `WhisperDecodeLanguageV2`'s tests).
    #[test]
    fn a_stock_whisper_worker_spawns_with_its_decode_language() {
        use crate::types::worker_v2::{WHISPER_DECODE_LANGUAGE_KEY, WhisperDecodeLanguageV2};
        for code in ["ell", "jav", "yue"] {
            let language = LanguageCode3::try_new(code).expect("language code");
            let key = asr_worker_key(
                crate::transcribe::AsrWorkerMode::LocalWhisperV2,
                code,
                Default::default(),
            );
            assert_eq!(
                spawn_override::<WhisperDecodeLanguageV2>(&key, WHISPER_DECODE_LANGUAGE_KEY),
                WhisperDecodeLanguageV2::for_language(&language).expect("Whisper has a token"),
                "{code}"
            );
        }
    }

    /// A worker that preloads ASR with no engine named resolves its own and
    /// falls back to stock Whisper for most languages, so it receives the
    /// decode language whenever Whisper has a token; for a language with none,
    /// nothing is sent and the worker refuses only if it does fall back.
    #[test]
    fn a_preloading_worker_receives_the_decode_language_when_whisper_has_one() {
        use crate::types::worker_v2::{WHISPER_DECODE_LANGUAGE_KEY, WhisperDecodeLanguageV2};
        let preload_key = |code: &str| WorkerKey {
            target: WorkerTarget::profile(WorkerProfile::Gpu),
            language: WorkerLanguage::from(LanguageCode3::try_new(code).expect("language code")),
            engine_selection: EngineSelection::derive(
                InferTask::Fa,
                EngineOverrides::default(),
                WorkerTarget::profile(WorkerProfile::Gpu),
                WorkerBootstrapMode::Profile,
            ),
        };
        let english = preload_key("eng");
        assert_eq!(
            english.engine_selection.overrides().asr,
            None,
            "no engine named"
        );
        assert_eq!(
            spawn_override::<WhisperDecodeLanguageV2>(&english, WHISPER_DECODE_LANGUAGE_KEY),
            WhisperDecodeLanguageV2::for_language(&LanguageCode3::eng()).expect("English")
        );
        let taiwanese =
            super::super::lifecycle::spawn_engine_overrides(&preload_key("nan")).expect("spawn");
        assert!(
            !taiwanese.contains(WHISPER_DECODE_LANGUAGE_KEY),
            "{taiwanese}"
        );
    }

    /// A worker rediscovered from the registry must land under the key the
    /// request that spawned it computes. The registry records the worker's
    /// full spawn overrides, including the keys the control plane injected
    /// (pinned models, language policies); those are derived from the key, so
    /// reading them back as user extras would file the worker under a key no
    /// request ever asks for, and it would never be reused.
    #[test]
    fn a_rediscovered_worker_keys_like_the_request_that_spawned_it() {
        let mut qwen_extras = std::collections::BTreeMap::new();
        qwen_extras.insert("qwen_context".to_owned(), String::new());
        let mut written = std::collections::BTreeSet::new();
        for key in [
            asr_worker_key(
                crate::transcribe::AsrWorkerMode::HkQwenV2,
                "nan",
                qwen_extras,
            ),
            // English also has an utterance-boundary model to pin.
            asr_worker_key(
                crate::transcribe::AsrWorkerMode::LocalWhisperV2,
                "eng",
                Default::default(),
            ),
        ] {
            let registered = super::super::lifecycle::spawn_engine_overrides(&key).expect("spawn");
            let map: std::collections::BTreeMap<String, String> =
                serde_json::from_str(&registered).expect("spawn overrides are a JSON map");
            written.extend(map.into_keys());
            let rediscovered =
                WorkerKey::from_registry_json(key.target, key.language.clone(), &registered)
                    .expect("registry JSON parses");
            assert_eq!(rediscovered, key);
        }
        // The cases above write every injected key, so a key the writer can
        // inject but the reader does not strip fails this test.
        for spawn_key in super::super::lifecycle::SpawnKey::ALL {
            assert!(written.contains(spawn_key.as_str()), "{spawn_key:?}");
        }
    }

    #[test]
    fn execute_v2_worker_key_uses_task_target_when_requested() {
        let request = request_with_payload(
            InferenceTaskV2::Morphosyntax,
            TaskRequestV2::Morphosyntax(MorphosyntaxRequestV2 {
                lang: LanguageCode3::eng(),
                payload_ref_id: WorkerArtifactIdV2::from("payload-1"),
                item_count: 1,
                retokenize: false,
            }),
        );

        let key = execute_v2_worker_key(
            WorkerLanguage::from(LanguageCode3::eng()),
            &request,
            WorkerBootstrapMode::Task,
        )
        .unwrap();

        assert_eq!(
            key.target,
            WorkerTarget::infer_task(InferTask::Morphosyntax)
        );
    }

    /// Shared overrides that name every key a request could carry plus an
    /// extra: whatever a job's command is, the keys its request does not
    /// carry must not enter its spawn-time identity either.
    fn crowded_common_options() -> crate::options::CommonOptions {
        crate::options::CommonOptions {
            engine_overrides: EngineOverrides {
                asr: Some(AsrEngineName::Whisper),
                fa: Some(FaEngineName::Whisper),
                translate: Some(crate::types::engines::TranslateEngineName::Nllb),
                extras: [("qwen_model".to_owned(), "Qwen/Qwen3-ASR-0.6B".to_owned())].into(),
                ..EngineOverrides::default()
            },
            ..crate::options::CommonOptions::default()
        }
    }

    /// The capability probe's key (from the job's options) and the dispatch
    /// key (from the request) for one command, which must agree in every
    /// bootstrap mode or the probed worker idles while dispatch spawns a
    /// second one. Before 2026-09-22 they agreed only when the shared
    /// overrides were empty: the probe started from the whole map.
    fn assert_probe_and_dispatch_keys_agree(
        command: crate::api::ReleasedCommand,
        options: crate::options::CommandOptions,
        request: ExecuteRequestV2,
    ) {
        let language = WorkerLanguage::from(LanguageCode3::eng());
        for mode in [
            WorkerBootstrapMode::LazyProfile,
            WorkerBootstrapMode::Profile,
            WorkerBootstrapMode::Task,
        ] {
            let requirement = crate::command_model::command_spec(command)
                .capabilities
                .require_inference()
                .expect("test command requires inference");
            let probe =
                WorkerKey::from_command_options(requirement, language.clone(), &options, mode);
            let dispatch = execute_v2_worker_key(language.clone(), &request, mode)
                .expect("a typed request derives a worker key");
            assert_eq!(probe, dispatch, "{command} in {mode:?}");
        }
    }

    fn text_payload_id() -> WorkerArtifactIdV2 {
        WorkerArtifactIdV2::from("payload-1")
    }

    #[test]
    fn translate_probe_and_dispatch_keys_agree_under_crowded_overrides() {
        assert_probe_and_dispatch_keys_agree(
            crate::api::ReleasedCommand::Translate,
            crate::options::CommandOptions::Translate(crate::options::TranslateOptions {
                common: crowded_common_options(),
                ..crate::options::TranslateOptions::default()
            }),
            request_with_payload(
                InferenceTaskV2::Translate,
                TaskRequestV2::Translate(crate::types::worker_v2::TranslateRequestV2 {
                    source_lang: LanguageCode3::eng(),
                    target_lang: LanguageCode3::eng(),
                    // The shared map's `translate` beats the flag, so the
                    // request names what the job resolved.
                    engine: crate::types::worker_v2::TranslateBackendV2::Nllb,
                    payload_ref_id: text_payload_id(),
                    item_count: 1,
                }),
            ),
        );
    }

    #[test]
    fn morphotag_probe_and_dispatch_keys_agree_under_crowded_overrides() {
        assert_probe_and_dispatch_keys_agree(
            crate::api::ReleasedCommand::Morphotag,
            crate::options::CommandOptions::Morphotag(crate::options::MorphotagOptions {
                common: crowded_common_options(),
                ..crate::options::MorphotagOptions::default()
            }),
            request_with_payload(
                InferenceTaskV2::Morphosyntax,
                TaskRequestV2::Morphosyntax(MorphosyntaxRequestV2 {
                    lang: LanguageCode3::eng(),
                    payload_ref_id: text_payload_id(),
                    item_count: 1,
                    retokenize: false,
                }),
            ),
        );
    }

    #[test]
    fn utseg_probe_and_dispatch_keys_agree_under_crowded_overrides() {
        assert_probe_and_dispatch_keys_agree(
            crate::api::ReleasedCommand::Utseg,
            crate::options::CommandOptions::Utseg(crate::options::UtsegOptions {
                common: crowded_common_options(),
                merge_abbrev: false.into(),
                utseg_fallback: false.into(),
            }),
            request_with_payload(
                InferenceTaskV2::Utseg,
                TaskRequestV2::Utseg(crate::types::worker_v2::UtsegRequestV2 {
                    lang: LanguageCode3::eng(),
                    payload_ref_id: text_payload_id(),
                    item_count: 1,
                    allow_stanza_fallback: false,
                }),
            ),
        );
    }

    #[test]
    fn coref_probe_and_dispatch_keys_agree_under_crowded_overrides() {
        assert_probe_and_dispatch_keys_agree(
            crate::api::ReleasedCommand::Coref,
            crate::options::CommandOptions::Coref(crate::options::CorefOptions {
                common: crowded_common_options(),
                merge_abbrev: false.into(),
            }),
            request_with_payload(
                InferenceTaskV2::Coref,
                TaskRequestV2::Coref(crate::types::worker_v2::CorefRequestV2 {
                    lang: LanguageCode3::eng(),
                    payload_ref_id: text_payload_id(),
                    item_count: 1,
                }),
            ),
        );
    }
}
