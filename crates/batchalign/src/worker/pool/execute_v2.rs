//! V2 execute dispatch helpers, task mapping and engine override resolution.
//!
//! These pure functions bridge the V2 execute protocol (typed per-backend
//! requests) to the pool's typed worker-key abstraction. Extracted from
//! `mod.rs` for browsability.

use crate::api::WorkerLanguage;
use crate::types::engines::{AsrEngineName, EngineOverrides, FaEngineName};
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

/// Derive the worker-pool key for one V2 execute request.
pub(super) fn execute_v2_worker_key(
    lang: WorkerLanguage,
    request: &ExecuteRequestV2,
    bootstrap_mode: WorkerBootstrapMode,
) -> Result<WorkerKey, WorkerError> {
    let infer_task = infer_task_for_execute_v2(request.task)?;
    let target = WorkerTarget::from_infer_task(infer_task, bootstrap_mode);

    // The selected engine is part of the worker identity in every bootstrap
    // mode. A lazy worker loads the recipe on demand, but it is not allowed to
    // change recipes afterward: sharing one task-only key between Wave2Vec and
    // Whisper made the Rust and Python "already loaded" caches return the
    // first engine for later requests. It also allowed a concurrent request to
    // switch process-global model state underneath in-flight inference.
    //
    // LazyProfile pre-scaling is disabled, so retaining this selection cannot
    // recreate the old empty-key pre-scale mismatch. On constrained hosts the
    // global worker permit and idle eviction still bound resident processes.
    let engine_selection = if bootstrap_mode == WorkerBootstrapMode::LazyProfile {
        EngineSelection::from_execute_request(request)
    } else {
        // Eager mode: the worker bootstrap preloads the whole profile, so the
        // selection must name an engine for every preloaded task, not only the
        // one this request carries.
        EngineSelection::from_execute_request_for_target(target, request)
    };

    Ok(WorkerKey {
        target,
        language: lang,
        engine_selection,
    })
}

impl EngineSelection {
    /// Derive worker configuration from a typed V2 request.
    ///
    /// This is the only V2 route into a pool engine identity. The typed
    /// `EngineOverrides` retains every opaque ASR extra, including
    /// `qwen_model`, until it is serialized for Python worker startup.
    pub(super) fn from_execute_request(request: &ExecuteRequestV2) -> Self {
        Self::from_overrides(execute_request_overrides(request))
    }

    /// Like [`Self::from_execute_request`], for spawning an EAGER worker
    /// whose target may preload more tasks than the request names. The
    /// target-driven fill has one owner (`fill_for_preloaded_tasks`), shared
    /// with the command spawn route.
    pub(super) fn from_execute_request_for_target(
        target: WorkerTarget,
        request: &ExecuteRequestV2,
    ) -> Self {
        Self::from_overrides(Self::fill_for_preloaded_tasks(
            target,
            execute_request_overrides(request),
        ))
    }
}

/// The engines a V2 request itself names, before any target-driven filling.
fn execute_request_overrides(request: &ExecuteRequestV2) -> EngineOverrides {
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
        _ => EngineOverrides::default(),
    }
}

/// Extract the ensure_task parameters (task name + engine overrides map) from a
/// V2 execute request, without the JSON round-trip.
///
/// Returns `None` for tasks that don't need model loading (e.g., text tasks in
/// eager-profile mode). Used by the LazyProfile dispatch path.
pub(super) fn ensure_task_params(
    request: &ExecuteRequestV2,
) -> Result<(String, Option<std::collections::BTreeMap<String, String>>), WorkerError> {
    let task = infer_task_for_execute_v2(request.task)?;
    let task_name = crate::worker::target::task_name(task).to_string();

    let selection = EngineSelection::from_execute_request(request);
    let overrides = (!selection.is_none()).then(|| selection.overrides().dispatch_overrides());

    Ok((task_name, overrides))
}

fn asr_backend_engine(backend: AsrBackendV2) -> AsrEngineName {
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
    use crate::types::engines::SelectableEngine;
    use crate::types::worker_v2::{
        AsrInputV2, AsrRequestV2, FaBackendV2, FaTextModeV2, ForcedAlignmentRequestV2,
        MorphosyntaxRequestV2, PreparedAudioInputV2, WorkerArtifactIdV2, WorkerRequestIdV2,
    };
    use crate::worker::{WorkerBootstrapMode, WorkerProfile, WorkerTarget};

    fn request_with_payload(task: InferenceTaskV2, payload: TaskRequestV2) -> ExecuteRequestV2 {
        ExecuteRequestV2 {
            request_id: WorkerRequestIdV2::from("req-1"),
            task,
            payload,
            attachments: Vec::new(),
        }
    }

    /// Regression (2026-08-21): in eager Profile mode an ASR request spawns a
    /// profile worker whose bootstrap preloads FA too, and the Python side
    /// refuses a preloaded task with no named engine rather than defaulting.
    /// The V2 route used to name only the engines the request itself carried,
    /// so the worker died at bootstrap ("no 'fa' engine in overrides") and
    /// every eager-mode whisper transcription failed as an internal error.
    #[test]
    fn eager_profile_asr_key_names_an_fa_engine_for_the_preloaded_task() {
        let request = request_with_payload(
            InferenceTaskV2::Asr,
            TaskRequestV2::Asr(AsrRequestV2 {
                lang: WorkerLanguage::from(LanguageCode3::eng()),
                backend: AsrBackendV2::LocalWhisper,
                input: AsrInputV2::PreparedAudio(PreparedAudioInputV2 {
                    audio_ref_id: WorkerArtifactIdV2::from("audio-1"),
                }),
                extras: std::collections::BTreeMap::new(),
                decode_budget_seconds: None,
            }),
        );

        let key = execute_v2_worker_key(
            WorkerLanguage::from(LanguageCode3::eng()),
            &request,
            WorkerBootstrapMode::Profile,
        )
        .expect("eager ASR request must derive a worker key");

        assert_eq!(
            key.engine_selection.overrides().fa,
            Some(FaEngineName::DEFAULT),
            "an eager profile worker preloads FA, so its key must name an FA engine"
        );
    }

    #[test]
    fn maps_forced_alignment_execute_v2_to_fa_worker_profile() {
        assert_eq!(
            infer_task_for_execute_v2(InferenceTaskV2::ForcedAlignment).unwrap(),
            InferTask::Fa
        );
    }

    #[test]
    fn execute_v2_asr_worker_key_uses_request_backend_override() {
        // Policy change (2026-08-21): this test used to pin the key JSON as
        // {"asr":"whisper"} alone, which was the DEFECT: in eager Profile
        // mode the spawned worker preloads FA too, and the Python bootstrap
        // refuses a preloaded task with no named engine, so the pinned value
        // was a worker that dies at startup. The eager key now also names
        // the default FA engine.
        let request = request_with_payload(
            InferenceTaskV2::Asr,
            TaskRequestV2::Asr(AsrRequestV2 {
                lang: WorkerLanguage::from(LanguageCode3::fra()),
                backend: AsrBackendV2::LocalWhisper,
                input: AsrInputV2::PreparedAudio(PreparedAudioInputV2 {
                    audio_ref_id: WorkerArtifactIdV2::from("audio-1"),
                }),
                extras: std::collections::BTreeMap::new(),
                decode_budget_seconds: None,
            }),
        );

        let key = execute_v2_worker_key(
            WorkerLanguage::from(LanguageCode3::fra()),
            &request,
            WorkerBootstrapMode::Profile,
        )
        .unwrap();

        assert_eq!(key.target, WorkerTarget::profile(WorkerProfile::Gpu));
        assert_eq!(key.language, WorkerLanguage::from(LanguageCode3::fra()));
        assert_eq!(
            key.engine_selection.worker_config_json(),
            r#"{"asr":"whisper","fa":"wave2vec"}"#
        );
    }

    #[test]
    fn execute_v2_engine_overrides_preserves_asr_extras_through_dispatch() {
        // Regression pin for the 2026-05-27 root-cause bug: prior to
        // adding ``extras`` to ``AsrRequestV2`` and threading them
        // through here, the pool key + worker spawn argv silently
        // dropped every per-engine knob the user passed via
        // ``--engine-overrides``. A request with
        // ``qwen_model=Qwen/Qwen3-ASR-0.6B`` was serialized as bare
        // ``{"asr":"qwen"}`` and the worker defaulted to 1.7B,
        // costing hours of wasted compute.
        //
        // This test is the seam test the per-knob Fix 1 CLI-parse
        // test SHOULD have been: it asserts the user's extras reach
        // the worker spawn argv JSON. Adding any new per-engine knob
        // (funaudio_*, future engines) is automatically covered
        // because the assertion is "every key in input.extras
        // appears in output JSON", not a fixed allowlist.
        let mut extras = std::collections::BTreeMap::new();
        extras.insert("qwen_model".to_owned(), "Qwen/Qwen3-ASR-0.6B".to_owned());
        extras.insert("qwen_device".to_owned(), "cpu".to_owned());

        let request = request_with_payload(
            InferenceTaskV2::Asr,
            TaskRequestV2::Asr(AsrRequestV2 {
                lang: WorkerLanguage::from(LanguageCode3::yue()),
                backend: AsrBackendV2::HkQwen,
                input: AsrInputV2::ProviderMedia(crate::types::worker_v2::ProviderMediaInputV2 {
                    media_path: "/dev/null".into(),
                    num_speakers: crate::api::NumSpeakers(1),
                }),
                extras: extras.clone(),
                decode_budget_seconds: None,
            }),
        );

        let json = EngineSelection::from_execute_request(&request).worker_config_json();
        let parsed: std::collections::BTreeMap<String, String> =
            serde_json::from_str(&json).expect("engine_overrides JSON must round-trip");

        assert_eq!(parsed.get("asr").map(String::as_str), Some("qwen"));
        for (k, v) in &extras {
            assert_eq!(
                parsed.get(k),
                Some(v),
                "extras key {k:?} (value {v:?}) was dropped at the V2 dispatch boundary, \
                 pool key + worker spawn argv would lose it"
            );
        }
    }

    #[test]
    fn execute_v2_non_worker_asr_does_not_fragment_keys_but_keeps_extras() {
        let mut extras = std::collections::BTreeMap::new();
        extras.insert("provider_region".to_owned(), "us-east".to_owned());
        let request = request_with_payload(
            InferenceTaskV2::Asr,
            TaskRequestV2::Asr(AsrRequestV2 {
                lang: WorkerLanguage::from(LanguageCode3::eng()),
                backend: AsrBackendV2::Revai,
                input: AsrInputV2::PreparedAudio(PreparedAudioInputV2 {
                    audio_ref_id: WorkerArtifactIdV2::from("audio-1"),
                }),
                extras: extras.clone(),
                decode_budget_seconds: None,
            }),
        );

        let selection = EngineSelection::from_execute_request(&request);
        assert_eq!(selection.overrides().asr, None);
        assert_eq!(selection.overrides().extras, extras);
        assert_eq!(
            selection.worker_config_json(),
            r#"{"provider_region":"us-east"}"#
        );
    }

    #[test]
    fn execute_v2_fa_worker_key_uses_request_backend_override() {
        let request = request_with_payload(
            InferenceTaskV2::ForcedAlignment,
            TaskRequestV2::ForcedAlignment(ForcedAlignmentRequestV2 {
                backend: FaBackendV2::Wave2vec,
                payload_ref_id: WorkerArtifactIdV2::from("payload-1"),
                audio_ref_id: WorkerArtifactIdV2::from("audio-1"),
                text_mode: FaTextModeV2::SpaceJoined,
            }),
        );

        let key = execute_v2_worker_key(
            WorkerLanguage::from(LanguageCode3::eng()),
            &request,
            WorkerBootstrapMode::Profile,
        )
        .unwrap();

        assert_eq!(key.target, WorkerTarget::profile(WorkerProfile::Gpu));
        assert_eq!(key.language, WorkerLanguage::from(LanguageCode3::eng()));
        assert_eq!(
            key.engine_selection.worker_config_json(),
            r#"{"fa":"wave2vec"}"#
        );
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

    #[test]
    fn lazy_profile_gpu_key_retains_the_request_engine_recipe() {
        let fa_request = request_with_payload(
            InferenceTaskV2::ForcedAlignment,
            TaskRequestV2::ForcedAlignment(ForcedAlignmentRequestV2 {
                backend: FaBackendV2::Wave2vec,
                payload_ref_id: WorkerArtifactIdV2::from("payload-1"),
                audio_ref_id: WorkerArtifactIdV2::from("audio-1"),
                text_mode: FaTextModeV2::SpaceJoined,
            }),
        );

        let asr_request = request_with_payload(
            InferenceTaskV2::Asr,
            TaskRequestV2::Asr(AsrRequestV2 {
                lang: WorkerLanguage::from(LanguageCode3::eng()),
                backend: AsrBackendV2::LocalWhisper,
                input: AsrInputV2::PreparedAudio(PreparedAudioInputV2 {
                    audio_ref_id: WorkerArtifactIdV2::from("audio-1"),
                }),
                extras: std::collections::BTreeMap::new(),
                decode_budget_seconds: None,
            }),
        );

        let fa_key = execute_v2_worker_key(
            WorkerLanguage::from(LanguageCode3::eng()),
            &fa_request,
            WorkerBootstrapMode::LazyProfile,
        )
        .unwrap();

        let asr_key = execute_v2_worker_key(
            WorkerLanguage::from(LanguageCode3::eng()),
            &asr_request,
            WorkerBootstrapMode::LazyProfile,
        )
        .unwrap();

        assert_eq!(
            fa_key.engine_selection.worker_config_json(),
            r#"{"fa":"wave2vec"}"#
        );
        assert_eq!(
            asr_key.engine_selection.worker_config_json(),
            r#"{"asr":"whisper"}"#
        );
        assert_eq!(fa_key.target, asr_key.target);
        assert_eq!(fa_key.language, asr_key.language);
        assert_ne!(fa_key, asr_key);
    }

    /// The command-level capability probe and the actual V2 ASR dispatch are
    /// two wire boundaries for one model recipe. This test survives because a
    /// Rust type cannot prove that independently decoded command options and a
    /// V2 request describe the same external model selection.
    #[test]
    fn lazy_transcribe_capability_key_matches_its_asr_execute_key() {
        let language = WorkerLanguage::from(LanguageCode3::eng());
        let options =
            crate::options::CommandOptions::Transcribe(crate::options::TranscribeOptions {
                auto_speakers: false,
                common: crate::options::CommonOptions::default(),
                asr_engine: AsrEngineName::Whisper,
                diarize: false,
                wor: false.into(),
                merge_abbrev: false.into(),
                utseg_fallback: false.into(),
                batch_size: 8,
            });
        let request = request_with_payload(
            InferenceTaskV2::Asr,
            TaskRequestV2::Asr(AsrRequestV2 {
                lang: language.clone(),
                backend: AsrBackendV2::LocalWhisper,
                input: AsrInputV2::PreparedAudio(PreparedAudioInputV2 {
                    audio_ref_id: WorkerArtifactIdV2::from("audio-1"),
                }),
                extras: std::collections::BTreeMap::new(),
                decode_budget_seconds: None,
            }),
        );

        let capability_key = WorkerKey::from_command_options(
            crate::api::ReleasedCommand::Transcribe,
            language.clone(),
            &options,
            WorkerBootstrapMode::LazyProfile,
        );
        let execute_key =
            execute_v2_worker_key(language, &request, WorkerBootstrapMode::LazyProfile)
                .expect("typed ASR request must derive a worker key");

        assert_eq!(capability_key, execute_key);
    }
}
