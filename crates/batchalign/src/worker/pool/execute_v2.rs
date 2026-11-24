//! V2 execute dispatch helpers, task mapping and engine override resolution.
//!
//! These pure functions bridge the V2 execute protocol (typed per-backend
//! requests) to the pool's typed worker-key abstraction. Extracted from
//! `mod.rs` for browsability.

use crate::types::engines::{AsrEngineName, EngineOverrides, FaEngineName, TranslateEngineName};
use crate::types::worker_v2::{AsrBackendV2, ExecuteRequestV2, InferenceTaskV2, TaskRequestV2};
use crate::worker::InferTask;

/// Map a V2 inference task enum to the pool's infer-task vocabulary: the
/// task a worker serving it is bootstrapped for. Total: every V2 task has a
/// host.
pub(super) const fn infer_task_for_execute_v2(task: InferenceTaskV2) -> InferTask {
    match task {
        InferenceTaskV2::Morphosyntax => InferTask::Morphosyntax,
        InferenceTaskV2::Utseg => InferTask::Utseg,
        InferenceTaskV2::Translate => InferTask::Translate,
        InferenceTaskV2::Coref => InferTask::Coref,
        InferenceTaskV2::Asr => InferTask::Asr,
        InferenceTaskV2::ForcedAlignment => InferTask::Fa,
        InferenceTaskV2::Speaker => InferTask::Speaker,
        // The SAME pool task as diarization: one speaker host serves both.
        // Which model that task's load builds (local pyannote, the embedding
        // model, or nothing for pyannoteAI) is the job plan's speaker slot,
        // part of the worker key (`GpuModelPlan::speaker_identity`), so a
        // speaker-identification worker never builds the diarization pipeline
        // or reaches the gated calibration artifact that pipeline pulls in.
        InferenceTaskV2::SpeakerEmbedding => InferTask::Speaker,
        InferenceTaskV2::Opensmile => InferTask::Opensmile,
        InferenceTaskV2::Avqi => InferTask::Avqi,
        InferenceTaskV2::WordSegmentation => InferTask::WordSegmentation,
    }
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
        // The speaker model the request needs, which its task load builds:
        // the plan's own derivation, so the two keys agree by construction.
        TaskRequestV2::Speaker(request) => {
            super::model_plan::PlannedSpeaker::Diarization(request.backend).loaded_model_overrides()
        }
        TaskRequestV2::SpeakerEmbedding(request) => {
            super::model_plan::PlannedSpeaker::Embedding(request.backend).loaded_model_overrides()
        }
        // One implementation each: nothing to select, so nothing enters the
        // key. Listed rather than wildcarded so a new task has to decide.
        // Word segmentation names its segmenter per request, and the worker
        // applies the named one to that request alone: no model is bound at
        // spawn, so the segmenter is not worker identity.
        TaskRequestV2::Morphosyntax(_)
        | TaskRequestV2::Utseg(_)
        | TaskRequestV2::Coref(_)
        | TaskRequestV2::WordSegmentation(_)
        | TaskRequestV2::Opensmile(_)
        | TaskRequestV2::Avqi(_) => EngineOverrides::default(),
    }
}

pub(super) const fn asr_backend_engine(backend: AsrBackendV2) -> AsrEngineName {
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
    use super::super::WorkerKey;
    use super::*;
    use crate::api::{LanguageCode3, WorkerLanguage};
    use crate::types::worker_v2::{MorphosyntaxRequestV2, WorkerArtifactIdV2, WorkerRequestIdV2};
    use crate::worker::{WorkerBootstrapMode, WorkerTarget};

    fn request_with_payload(payload: TaskRequestV2) -> ExecuteRequestV2 {
        ExecuteRequestV2 {
            request_id: WorkerRequestIdV2::from("req-1"),
            task: payload.task(),
            payload,
            attachments: Vec::new(),
        }
    }

    /// Guards: the wire `forced_alignment` task selects the infrastructure FA host.
    /// Kind: policy
    /// Not a type: the wire and pool task vocabularies have different variants,
    /// and the many-to-one host mapping remains an explicit semantic choice.
    /// State: none
    /// Concurrency: none
    #[test]
    fn maps_forced_alignment_execute_v2_to_fa_worker_profile() {
        assert_eq!(
            infer_task_for_execute_v2(InferenceTaskV2::ForcedAlignment),
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
            Some(crate::transcribe::AsrBackend::Worker(mode)),
            &extras,
            None,
        );
        super::super::GpuRoute::worker_key(
            &plan,
            super::super::GpuTask::Asr,
            WorkerLanguage::from(language),
            WorkerBootstrapMode::LazyProfile,
        )
    }

    /// The spawn override `name` a worker for `key` receives, parsed.
    fn spawn_override<T: serde::de::DeserializeOwned>(key: &WorkerKey, name: &str) -> T {
        let json = super::super::lifecycle::spawn_engine_overrides(
            key,
            &crate::model_manifest::TencentEngineSetting::LARGE_MODEL_2,
        )
        .expect("spawn");
        let map: std::collections::BTreeMap<String, String> =
            serde_json::from_str(&json).expect("spawn overrides are a JSON map");
        serde_json::from_str(map.get(name).expect("policy key")).expect("policy parses")
    }

    /// A worker rediscovered from the registry must land under the key the
    /// request that spawned it computes. The registry records the worker's
    /// full spawn overrides, including the keys the control plane injected
    /// (pinned models, language policies); those are derived from the key, so
    /// reading them back as user extras would file the worker under a key no
    /// request ever asks for, and it would never be reused.
    /// Guards: serialized spawn keys normalize back to the original recipe key,
    /// including every injected model and recognition-policy field.
    /// Kind: wire
    /// Not a type: registry JSON is independently stored external bytes and must
    /// strip injected fields without losing the actual configured engine identity.
    /// State: local registry JSON exercises the real writer and reader adapters.
    /// Concurrency: none
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
            let registered = super::super::lifecycle::spawn_engine_overrides(
                &key,
                &crate::model_manifest::TencentEngineSetting::LARGE_MODEL_2,
            )
            .expect("spawn");
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

    enum TextCommand {
        Translate,
        Morphotag,
        Utseg,
        Coref,
    }

    fn assert_planned_worker_identity(
        kind: TextCommand,
        options: crate::options::CommandOptions,
        request: ExecuteRequestV2,
    ) {
        use super::super::worker_plan::{CommandWorkerPlan, WorkerPlan};
        use crate::api::ReleasedCommand as Command;
        use crate::worker::WorkerProfile;
        let (command, task, profile, expected_engines) = match kind {
            TextCommand::Translate => (
                Command::Translate,
                InferTask::Translate,
                WorkerProfile::Io,
                serde_json::json!({"translate":"nllb"}),
            ),
            TextCommand::Morphotag => (
                Command::Morphotag,
                InferTask::Morphosyntax,
                WorkerProfile::Stanza,
                serde_json::json!({}),
            ),
            TextCommand::Utseg => (
                Command::Utseg,
                InferTask::Utseg,
                WorkerProfile::Stanza,
                serde_json::json!({}),
            ),
            TextCommand::Coref => (
                Command::Coref,
                InferTask::Coref,
                WorkerProfile::Stanza,
                serde_json::json!({}),
            ),
        };
        for mode in [
            WorkerBootstrapMode::LazyProfile,
            WorkerBootstrapMode::Profile,
            WorkerBootstrapMode::Task,
        ] {
            let requirement = crate::command_model::command_spec(command)
                .capabilities
                .require_inference()
                .unwrap();
            let command_plan = CommandWorkerPlan::new(
                requirement,
                LanguageCode3::eng().into(),
                &options,
                &super::super::GpuModelPlan::without_engines(),
                mode,
            );
            let prepared = super::super::DispatchPreparation::begin(&request).unwrap();
            let request_plan =
                WorkerPlan::direct(LanguageCode3::eng().into(), prepared, mode).unwrap();
            let expected_target = match mode {
                WorkerBootstrapMode::Task => WorkerTarget::InferTask(task),
                WorkerBootstrapMode::Profile | WorkerBootstrapMode::LazyProfile => {
                    WorkerTarget::Profile(profile)
                }
            };
            assert_eq!(request_plan.key().target, expected_target);
            assert_eq!(request_plan.key(), command_plan.key());
            assert_eq!(command_plan.load().task(), task);
            assert_eq!(
                serde_json::to_value(
                    request_plan
                        .key()
                        .engine_selection
                        .overrides()
                        .dispatch_overrides()
                )
                .unwrap(),
                expected_engines
            );
        }
    }

    fn text_payload_id() -> WorkerArtifactIdV2 {
        WorkerArtifactIdV2::from("payload-1")
    }

    /// Guards: command probes and direct requests retain each task's actual key,
    /// ignoring foreign overrides and selecting explicit profile/task targets.
    /// Kind: policy
    /// Not a type: which recipe keys belong to translation, morphology, segmentation
    /// and coreference is a semantic mapping across external command and wire inputs.
    /// State: none
    /// Concurrency: none
    #[test]
    fn bound_worker_plans_preserve_task_recipe_and_bootstrap_policy() {
        {
            assert_planned_worker_identity(
                TextCommand::Translate,
                crate::options::CommandOptions::Translate(crate::options::TranslateOptions {
                    common: crowded_common_options(),
                    ..crate::options::TranslateOptions::default()
                }),
                request_with_payload(TaskRequestV2::Translate(
                    crate::types::worker_v2::TranslateRequestV2 {
                        source_lang: LanguageCode3::eng(),
                        target_lang: LanguageCode3::eng(),
                        // The shared map's `translate` beats the flag, so the
                        // request names what the job resolved.
                        engine: crate::types::worker_v2::TranslateBackendV2::Nllb,
                        payload_ref_id: text_payload_id(),
                        item_count: 1,
                    },
                )),
            );
        }
        {
            assert_planned_worker_identity(
                TextCommand::Morphotag,
                crate::options::CommandOptions::Morphotag(crate::options::MorphotagOptions {
                    common: crowded_common_options(),
                    ..crate::options::MorphotagOptions::default()
                }),
                request_with_payload(TaskRequestV2::Morphosyntax(MorphosyntaxRequestV2 {
                    lang: LanguageCode3::eng(),
                    payload_ref_id: text_payload_id(),
                    item_count: 1,
                    retokenize: false,
                })),
            );
        }
        {
            assert_planned_worker_identity(
                TextCommand::Utseg,
                crate::options::CommandOptions::Utseg(crate::options::UtsegOptions {
                    common: crowded_common_options(),
                    utseg_fallback: false.into(),
                }),
                request_with_payload(TaskRequestV2::Utseg(
                    crate::types::worker_v2::UtsegRequestV2 {
                        lang: LanguageCode3::eng(),
                        payload_ref_id: text_payload_id(),
                        item_count: 1,
                        segmenter: crate::types::worker_v2::UtsegSegmenterV2::BoundaryModel,
                    },
                )),
            );
        }
        {
            assert_planned_worker_identity(
                TextCommand::Coref,
                crate::options::CommandOptions::Coref(crate::options::CorefOptions {
                    common: crowded_common_options(),
                }),
                request_with_payload(TaskRequestV2::Coref(
                    crate::types::worker_v2::CorefRequestV2 {
                        lang: LanguageCode3::eng(),
                        payload_ref_id: text_payload_id(),
                        item_count: 1,
                    },
                )),
            );
        }
    }
    /// Guards: actual spawn JSON retains ASR extras and the language-specific
    /// Qwen and Whisper recognition policies supplied by their typed job plan.
    /// Kind: wire
    /// Not a type: Python receives serialized key names and policy values whose
    /// wire spelling and selection must survive the native spawn producer.
    /// State: local plan values and JSON serialization require no worker process.
    /// Concurrency: none
    #[test]
    fn planned_asr_spawn_bytes_retain_extras_and_language_policies() {
        {
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
        {
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
        {
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
    }
}
