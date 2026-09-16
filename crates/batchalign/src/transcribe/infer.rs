//! ASR and speaker inference dispatch to worker backends.

use std::path::Path;

use super::types::{AsrResponse, AsrWorkerMode, NonRevAsrBackend};
use super::{
    SpeakerEvidenceInference, SpeakerEvidenceModelRevision, SpeakerEvidenceRequest,
    SpeakerEvidenceResolution, SpeakerEvidenceResolutionError, SpeakerEvidenceSource,
    SpeakerEvidenceTrace, SpeakerProjectionRevision, VerifiedSpeakerEvidenceRun,
    resolve_speaker_evidence,
};
use crate::api::{LanguageCode3, LanguageSpec, NumSpeakers, WorkerLanguage};
use crate::cache::UtteranceCache;
use crate::chat_ops::CacheKey;
use crate::error::{MissingRequiredEvidence, MissingSpeakerEvidence, ServerError};
use crate::params::CachePolicy;
use crate::types::worker_v2::{
    ProtocolErrorCodeV2, ProviderDiarizationV2, SpeakerBackendV2, SpeakerInferenceEvidenceV2,
    SpeakerSegmentV2,
};
use crate::worker::artifacts_v2::PreparedArtifactRuntimeV2;
use crate::worker::asr_request_v2::{
    AsrBuildInputV2, AsrInputSourceV2, PreparedAsrRequestIdsV2, build_asr_request_v2,
};
use crate::worker::asr_result_v2::parse_asr_response_v2;
use crate::worker::pool::WorkerPool;
use crate::worker::speaker_request_v2::{
    PreparedSpeakerRequestIdsV2, SpeakerAudioBuildSourceV2, SpeakerBuildInputV2,
    build_speaker_request_v2,
};
use crate::worker::speaker_result_v2::{SpeakerResultParseError, parse_speaker_result_v2};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parameters for ASR worker inference.
pub(crate) struct AsrInferParams<'a> {
    /// Which runtime boundary owns raw ASR inference.
    pub backend: NonRevAsrBackend,
    /// Audio file to transcribe.
    pub audio_path: &'a Path,
    /// Language specification for ASR dispatch. May be `Auto`, the GPU
    /// worker and ASR engine handle auto-detect internally.
    pub lang: &'a LanguageSpec,
    /// Expected number of speakers for diarization.
    pub num_speakers: NumSpeakers,
    /// Per-engine configuration extras (e.g. `qwen_model`,
    /// `qwen_device`) drawn from `CommonOptions.engine_overrides.extras`.
    /// Plumbed through to the worker spawn argv so the engine's load
    /// function sees what the user actually requested, the `backend`
    /// enum only encodes WHICH engine, not its configuration.
    pub extras: &'a std::collections::BTreeMap<String, String>,
}

/// Compute the worker-runtime language and an "expected response
/// language" hint used by `parse_asr_response_v2` when the ASR response
/// does not carry a usable detected language of its own.
///
/// For `Resolved(code)` jobs, both values are derived from `code`, the
/// CHAT header will reflect what the user explicitly asked for. For
/// `Auto` jobs there is no concrete hint, and the parse helper must
/// drive the language from the response itself; we return `None` so
/// the caller can surface a typed error if the response is also empty.
/// `PerFile` is not legal at this point (transcribe-class commands are
/// rejected by submission validation if they carry it).
pub(super) fn asr_worker_languages(
    lang: &LanguageSpec,
) -> Result<(WorkerLanguage, Option<LanguageCode3>), ServerError> {
    match lang {
        LanguageSpec::Resolved(code) => {
            Ok((WorkerLanguage::Resolved(code.clone()), Some(code.clone())))
        }
        LanguageSpec::Auto => Ok((WorkerLanguage::Auto, None)),
        LanguageSpec::PerFile => Err(ServerError::Validation(
            "transcribe pipeline received LanguageSpec::PerFile, which is reserved for \
             morphotag/translate/coref. Submission validation should have rejected \
             this: please file a bug report."
                .into(),
        )),
    }
}

/// Call the Python worker for ASR inference on a single audio file.
pub(crate) async fn infer_asr(
    pool: &WorkerPool,
    params: &AsrInferParams<'_>,
) -> Result<AsrResponse, ServerError> {
    let (worker_lang, fallback_lang) = asr_worker_languages(params.lang)?;

    match params.backend {
        NonRevAsrBackend::RustWhisperRs => {
            infer_whisper_rs_asr(params.audio_path, params.lang, params.extras).await
        }
        NonRevAsrBackend::Worker(worker_mode) => {
            infer_asr_via_worker_v2(
                pool,
                params,
                worker_mode,
                &worker_lang,
                fallback_lang.as_ref(),
            )
            .await
        }
    }
}

/// Rust-native Whisper ASR (whisper.cpp via whisper-rs), run in-process,
/// bypassing the Python worker.
///
/// The model resolves via `WhisperNativeConfig::resolve()` (env override,
/// else the default ggml-large-v3 fetched once through hf-hub), language
/// `Auto` engages whisper.cpp's own auto-detection (the detected code
/// comes back on the chunk result), the sync whisper.cpp inference runs
/// on a `spawn_blocking` thread (mirroring the Rev.AI path), and the
/// chunk result lowers through the shared converter so its `AsrResponse`
/// is identical to the Python Whisper worker path's.
///
/// When the `whisper-rs-backend` feature is not compiled in,
/// `whisper_native::transcribe` returns `FeatureDisabled`, surfaced here as
/// a `WhisperEngine` (infrastructure) error.
async fn infer_whisper_rs_asr(
    audio_path: &Path,
    lang: &LanguageSpec,
    extras: &std::collections::BTreeMap<String, String>,
) -> Result<AsrResponse, ServerError> {
    let requested = lang.as_resolved().cloned();

    let cfg = whisper_rs_config_from(extras).map_err(whisper_error_to_server_error)?;

    let audio_path = audio_path.to_path_buf();
    let lang_for_call = requested.clone();
    let result = tokio::task::spawn_blocking(move || {
        crate::whisper_native::transcribe(&audio_path, lang_for_call, &cfg)
    })
    .await
    .map_err(|error| {
        ServerError::WhisperEngine(format!("whisper_rs ASR task join failed: {error}"))
    })?
    .map_err(whisper_error_to_server_error)?;

    // `result.lang` is authoritative (the caller's when resolved,
    // whisper.cpp's detection when `Auto`); the fallback covers only a
    // response whose own language field is unusable.
    crate::worker::asr_result_v2::whisper_chunk_result_to_asr_response(&result, requested.as_ref())
        .map_err(|error| {
            ServerError::WhisperEngine(format!("whisper_rs ASR response lowering failed: {error}"))
        })
}

/// Engine-override key selecting the ggml model file for a single job
/// (same mechanism as `qwen_model`): `--engine-overrides
/// '{"asr":"whisper_rs","whisper_rs_model":"/path/model.bin"}'`.
const WHISPER_RS_MODEL_EXTRA: &str = "whisper_rs_model";

/// Model-resolution precedence for the whisper_rs engine: the per-job
/// engine-override extra, then the machine-wide env var, then the
/// auto-fetched default (inside `resolve()`).
fn whisper_rs_config_from(
    extras: &std::collections::BTreeMap<String, String>,
) -> Result<crate::whisper_native::WhisperNativeConfig, crate::whisper_native::WhisperNativeError> {
    if let Some(path) = extras.get(WHISPER_RS_MODEL_EXTRA) {
        return Ok(crate::whisper_native::WhisperNativeConfig::for_model(
            std::path::PathBuf::from(path),
        ));
    }
    crate::whisper_native::WhisperNativeConfig::resolve()
}

/// Route the typed native-Whisper error taxonomy onto the server error
/// surface without flattening it: genuine input problems are validation
/// (HTTP 400); infrastructure, build, and invariant failures are worker
/// errors (HTTP 500), so an HF outage is never reported to a client as
/// "you submitted something invalid".
fn whisper_error_to_server_error(error: crate::whisper_native::WhisperNativeError) -> ServerError {
    use crate::whisper_native::WhisperNativeError as E;
    match &error {
        E::UnsupportedLanguage(_) | E::UnsupportedDetectedLanguage(_) => {
            ServerError::Validation(format!("whisper_rs ASR failed: {error}"))
        }
        _ => ServerError::WhisperEngine(format!("whisper_rs ASR failed: {error}")),
    }
}

/// Call the live V2 Python worker path for ASR inference on a single audio
/// file and normalize its typed result into the shared Rust ASR response
/// shape.
async fn infer_asr_via_worker_v2(
    pool: &WorkerPool,
    params: &AsrInferParams<'_>,
    worker_mode: AsrWorkerMode,
    worker_lang: &WorkerLanguage,
    fallback_lang: Option<&LanguageCode3>,
) -> Result<AsrResponse, ServerError> {
    let artifacts = PreparedArtifactRuntimeV2::new("asr_v2").map_err(|error| {
        ServerError::Validation(format!("failed to create ASR V2 artifact runtime: {error}"))
    })?;
    // The pin is resolved HERE rather than by each caller: this function
    // already holds the backend, the language and the overrides the manifest
    // reads, so deriving it once keeps one owner and stops two call sites from
    // pinning differently for the same job.
    let models = crate::model_manifest::resolve_asr_models(
        worker_mode.as_v2_backend(),
        params.lang.as_resolved(),
        params.extras,
    )
    .map_err(|error| ServerError::Validation(error.to_string()))?;
    let request = build_asr_request_v2(
        artifacts.store(),
        AsrBuildInputV2 {
            ids: &PreparedAsrRequestIdsV2::fresh(),
            models: &models,
            input: match worker_mode {
                // Fine-tune HF Whisper shares the prepared-audio wire shape
                // with stock Whisper: Rust owns media decoding, the worker
                // receives a resampled mono waveform. The only difference
                // is which checkpoint the worker's ``WhisperASRHandle`` was
                // constructed around at bootstrap.
                AsrWorkerMode::LocalWhisperV2 | AsrWorkerMode::WhisperHubV2 => {
                    AsrInputSourceV2::PreparedAudio {
                        audio_path: params.audio_path,
                    }
                }
                AsrWorkerMode::HkTencentV2
                | AsrWorkerMode::HkAliyunV2
                | AsrWorkerMode::HkFunaudioV2
                | AsrWorkerMode::HkQwenV2 => AsrInputSourceV2::ProviderMedia {
                    media_path: params.audio_path,
                    diarization: ProviderDiarizationV2::for_expected_speakers(params.num_speakers),
                },
            },
            lang: worker_lang,
            backend: worker_mode.as_v2_backend(),
            extras: params.extras,
        },
    )
    .await
    .map_err(|error| {
        ServerError::Validation(format!(
            "failed to build worker protocol V2 ASR request: {error}"
        ))
    })?;

    let response = pool
        .dispatch_execute_v2(worker_lang, &request)
        .await
        .map_err(ServerError::Worker)?;

    // The pin stays in scope across the dispatch on purpose: it is the half
    // this side is entitled to assert, and the admission inside the parse
    // compares it with the half the worker reported.
    parse_asr_response_v2(
        &response,
        fallback_lang,
        &models,
        worker_mode.as_v2_backend(),
    )
    .map_err(|error| ServerError::Validation(format!("ASR V2 response parse failed: {error}")))
}

/// Call the live V2 Python worker path for dedicated speaker diarization on a
/// single audio file.
async fn infer_speaker(
    pool: &WorkerPool,
    worker_lang: &WorkerLanguage,
    run: VerifiedSpeakerEvidenceRun,
) -> Result<SpeakerInferenceEvidenceV2, ServerError> {
    let (source_bytes, backend, expected_speakers) = run.into_worker_input();
    let artifacts = PreparedArtifactRuntimeV2::new("speaker_v2").map_err(|error| {
        ServerError::Validation(format!(
            "failed to create speaker V2 artifact runtime: {error}"
        ))
    })?;
    let request = build_speaker_request_v2(
        artifacts.store(),
        SpeakerBuildInputV2 {
            ids: &PreparedSpeakerRequestIdsV2::fresh(),
            audio: SpeakerAudioBuildSourceV2::Bytes(source_bytes),
            backend,
            expected_speakers,
        },
    )
    .await
    .map_err(|error| {
        ServerError::Validation(format!(
            "failed to build worker protocol V2 speaker request: {error}"
        ))
    })?;

    let response = pool
        .dispatch_execute_v2(worker_lang, &request)
        .await
        .map_err(ServerError::Worker)?;

    parse_speaker_result_v2(&response)
        .map(|result| result.evidence.clone())
        .map_err(speaker_parse_failure_to_server_error)
}

/// Route a speaker V2 parse failure to its category, instead of collapsing
/// every failure into `ServerError::Validation`.
///
/// A worker-reported PROTOCOL failure (`SpeakerResultParseError::Protocol`)
/// carries a [`ProtocolErrorCodeV2`] the worker itself classified; a Hugging
/// Face Hub access denial becomes `ServerError::ModelAccessDenied`, a
/// configuration/credential condition on the SERVER's machine, never
/// `Validation`, which the dashboard renders as "pipeline bug, filed
/// automatically" and which describes a malformed CLIENT request. Every
/// other protocol code, and a shape mismatch
/// (`SpeakerResultParseError::UnexpectedPayload`), stays `Validation`: those
/// really are "the response this server got does not make sense".
fn speaker_parse_failure_to_server_error(error: SpeakerResultParseError) -> ServerError {
    match error {
        SpeakerResultParseError::Protocol {
            code: ProtocolErrorCodeV2::ModelAccessDenied,
            message,
        } => ServerError::ModelAccessDenied(message),
        SpeakerResultParseError::Protocol { message, .. } => {
            ServerError::Validation(format!("speaker V2 response parse failed: {message}"))
        }
        SpeakerResultParseError::UnexpectedPayload(message) => {
            ServerError::Validation(format!("speaker V2 response parse failed: {message}"))
        }
    }
}

/// Production implementation of the typed speaker-evidence service boundary.
pub(crate) struct SpeakerWorkerInference<'a> {
    pool: &'a WorkerPool,
    worker_lang: WorkerLanguage,
}

impl<'a> SpeakerWorkerInference<'a> {
    pub(crate) fn new(pool: &'a WorkerPool, worker_lang: WorkerLanguage) -> Self {
        Self { pool, worker_lang }
    }
}

#[async_trait::async_trait]
impl SpeakerEvidenceInference for SpeakerWorkerInference<'_> {
    async fn infer(
        &self,
        run: VerifiedSpeakerEvidenceRun,
    ) -> Result<SpeakerInferenceEvidenceV2, ServerError> {
        infer_speaker(self.pool, &self.worker_lang, run).await
    }
}

/// Complete semantic inputs for one live-or-replayed speaker-evidence run.
///
/// The optional speaker count is deliberate: standalone `diarize` supports
/// backend auto-detection, while `transcribe` always supplies a known count.
/// Keeping that distinction typed prevents the cache from conflating an
/// auto-detected result with a request for an arbitrary sentinel count.
pub(crate) struct SpeakerEvidenceRunParams<'a> {
    pub audio_path: &'a Path,
    pub backend: SpeakerBackendV2,
    pub expected_speakers: Option<NumSpeakers>,
    pub cache_policy: CachePolicy,
}

/// Validated speaker evidence together with the exact request identity that
/// produced (or replayed) it.
///
/// Consumers cannot obtain segments without going through the shared
/// cache-or-infer state machine, and cannot log a cache key from a different
/// request by accident.
pub(crate) struct ResolvedSpeakerEvidence {
    request: SpeakerEvidenceRequest,
    resolution: SpeakerEvidenceResolution,
}

impl ResolvedSpeakerEvidence {
    pub(crate) fn cache_key(&self) -> &CacheKey {
        self.request.cache_key()
    }

    pub(crate) fn source(&self) -> SpeakerEvidenceSource {
        self.resolution.source()
    }

    pub(crate) fn segments(&self) -> &[SpeakerSegmentV2] {
        self.resolution.segments()
    }

    pub(crate) fn trace(&self) -> SpeakerEvidenceTrace {
        self.resolution.trace(SpeakerProjectionRevision::SegmentsV1)
    }

    pub(crate) fn into_segments(self) -> Vec<SpeakerSegmentV2> {
        self.resolution.into_segments()
    }
}

/// Resolve speaker evidence through one production operation shared by
/// integrated transcription and standalone diarization.
pub(crate) async fn resolve_speaker_evidence_for_audio(
    pool: &WorkerPool,
    cache: &UtteranceCache,
    worker_lang: WorkerLanguage,
    params: SpeakerEvidenceRunParams<'_>,
) -> Result<ResolvedSpeakerEvidence, ServerError> {
    let model_revision = SpeakerEvidenceModelRevision::for_backend(params.backend);
    let request = SpeakerEvidenceRequest::from_audio(
        params.audio_path,
        params.backend,
        params.expected_speakers,
        &model_revision,
    )
    .await
    .map_err(|error| ServerError::Persistence(error.to_string()))?;
    let inference = SpeakerWorkerInference::new(pool, worker_lang);
    let resolution = resolve_speaker_evidence(&request, cache, params.cache_policy, &inference)
        .await
        .map_err(speaker_resolution_error_to_server_error)?;
    Ok(ResolvedSpeakerEvidence {
        request,
        resolution,
    })
}

fn speaker_resolution_error_to_server_error(error: SpeakerEvidenceResolutionError) -> ServerError {
    match error {
        SpeakerEvidenceResolutionError::Evidence(
            super::SpeakerEvidenceCacheError::RequiredEvidenceMissing(cache_key),
        ) => ServerError::RequiredEvidenceUnavailable(MissingRequiredEvidence::Speaker(
            MissingSpeakerEvidence::new(cache_key),
        )),
        SpeakerEvidenceResolutionError::Evidence(error) => {
            ServerError::Persistence(error.to_string())
        }
        SpeakerEvidenceResolutionError::Inference(error) => error,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use std::path::Path;

    /// The regression this function exists for (2026-09-02): a worker
    /// protocol failure carrying `ModelAccessDenied` (the category set when
    /// pyannote's speaker-diarization load hits a gated Hugging Face Hub
    /// repository) must become `ServerError::ModelAccessDenied`, and NEVER
    /// `Validation`. Before this, every speaker V2 parse failure collapsed
    /// into `Validation`, which the dashboard renders as "pipeline bug, not
    /// your input" even when the true cause was the operator's own missing
    /// Hub credentials.
    #[test]
    fn model_access_denied_protocol_failure_never_becomes_validation() {
        let error = SpeakerResultParseError::Protocol {
            code: ProtocolErrorCodeV2::ModelAccessDenied,
            message: "could not download the Hugging Face model at \
                      pyannote/speaker-diarization-community-1: its repository is gated"
                .to_owned(),
        };

        let mapped = speaker_parse_failure_to_server_error(error);

        assert!(
            matches!(mapped, ServerError::ModelAccessDenied(_)),
            "expected ServerError::ModelAccessDenied, got {mapped:?}"
        );
        assert!(!matches!(mapped, ServerError::Validation(_)));
    }

    /// Every OTHER protocol code, and a shape mismatch, still becomes
    /// `Validation`: those genuinely are "the response this server got does
    /// not make sense", unlike a Hub access denial.
    #[test]
    fn other_protocol_failures_and_shape_mismatches_still_become_validation() {
        let protocol_runtime = SpeakerResultParseError::Protocol {
            code: ProtocolErrorCodeV2::RuntimeFailure,
            message: "boom".to_owned(),
        };
        assert!(matches!(
            speaker_parse_failure_to_server_error(protocol_runtime),
            ServerError::Validation(_)
        ));

        let shape_mismatch =
            SpeakerResultParseError::UnexpectedPayload("wrong payload kind".to_owned());
        assert!(matches!(
            speaker_parse_failure_to_server_error(shape_mismatch),
            ServerError::Validation(_)
        ));
    }

    /// `Auto` is ACCEPTED on the native whisper-rs path (whisper.cpp's
    /// own detection runs; 2026-07-28 "fully supported and default"). A
    /// failure on this dispatch must therefore come from infrastructure
    /// (here: a nonexistent audio file, or model resolution on a cold
    /// cache), and it must surface as the 500-class `WhisperEngine`
    /// variant, never as `Validation`: infra failures may not be
    /// reported to clients as bad input.
    #[test]
    fn whisper_rs_model_extra_takes_precedence() {
        let mut extras = std::collections::BTreeMap::new();
        extras.insert(
            WHISPER_RS_MODEL_EXTRA.to_string(),
            "/per/job/model.bin".to_string(),
        );
        let cfg = whisper_rs_config_from(&extras).expect("explicit path always resolves");
        assert_eq!(
            cfg.model_path,
            std::path::PathBuf::from("/per/job/model.bin")
        );
    }

    #[test]
    fn required_speaker_evidence_maps_to_typed_precondition_refusal() {
        let key = CacheKey::from_content("speaker evidence identity");
        let error = SpeakerEvidenceResolutionError::Evidence(
            crate::transcribe::SpeakerEvidenceCacheError::RequiredEvidenceMissing(key.clone()),
        );

        let mapped = speaker_resolution_error_to_server_error(error);
        let ServerError::RequiredEvidenceUnavailable(MissingRequiredEvidence::Speaker(missing)) =
            mapped
        else {
            panic!("required speaker evidence must not become a persistence error");
        };
        assert_eq!(missing.cache_key(), &key);
    }

    #[tokio::test]
    async fn whisper_rs_dispatch_accepts_auto_and_types_infra_failures() {
        let extras = std::collections::BTreeMap::new();
        let err = infer_whisper_rs_asr(Path::new("/nonexistent.wav"), &LanguageSpec::Auto, &extras)
            .await
            .expect_err("a nonexistent audio file must fail");
        assert!(
            matches!(err, ServerError::WhisperEngine(_)),
            "expected a WhisperEngine (infrastructure) error, got {err:?}"
        );
    }
}
