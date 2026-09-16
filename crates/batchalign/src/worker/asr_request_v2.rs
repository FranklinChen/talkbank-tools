//! Rust-side request builders for live worker-protocol V2 ASR.
//!
//! The live V2 ASR path now covers both local prepared-audio execution and the
//! transitional HK/cloud provider-media path. This module turns that
//! architectural intent into a typed request builder:
//!
//! - prepare local audio as Rust-owned PCM when the backend needs it
//! - keep provider-media references typed while HK/cloud engines still depend
//!   on Python-only SDKs
//! - choose the concrete V2 ASR backend in Rust
//! - return a typed `ExecuteRequestV2` for the worker

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use thiserror::Error;

use crate::api::WorkerLanguage;
use crate::media::probe::MediaProbe;
use crate::types::worker_v2::{
    ArtifactRefV2, AsrBackendV2, AsrInputV2, AsrRequestV2, DecodeBudgetSeconds, ExecuteRequestV2,
    InferenceTaskV2, PreparedAudioInputV2, ProviderDiarizationV2, ProviderMediaInputV2,
    TaskRequestV2, WorkerArtifactIdV2, WorkerRequestIdV2,
};

use super::artifacts_v2::{PreparedArtifactErrorV2, PreparedArtifactStoreV2};

/// Derive a request's decode budget from a media file's probed duration.
///
/// `None` when the duration could not be probed (ffprobe missing, refused
/// the file, or could not be started): this is the honest "unknown to
/// Rust" case, not a fabricated budget, and callers fall back to
/// [`crate::types::worker_v2::PROVIDER_MEDIA_ASR_TRANSPORT_CEILING_SECONDS`]
/// for it (see `TaskRequestV2::timeout_seconds_with_config`). A probe
/// failure never blocks request construction: ASR still runs, it just
/// gets the generous named fallback ceiling instead of a duration-derived
/// one.
async fn probe_decode_budget_seconds(media_path: &Path) -> Option<DecodeBudgetSeconds> {
    match MediaProbe::new(media_path).duration().await {
        Ok(duration_ms) => Some(DecodeBudgetSeconds::for_duration_ms(duration_ms.0)),
        Err(error) => {
            tracing::debug!(
                path = %media_path.display(),
                %error,
                "could not probe ASR provider-media duration; falling back to the named ceiling",
            );
            None
        }
    }
}

/// Stable ids for the prepared artifacts and envelope of one V2 ASR request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedAsrRequestIdsV2 {
    /// Top-level request id for the worker envelope.
    pub request_id: WorkerRequestIdV2,
    /// Artifact id for the prepared audio payload.
    pub audio_ref_id: WorkerArtifactIdV2,
}

/// Monotonic sequence used to make prepared-ASR request ids unique across
/// concurrent inference calls.
///
/// The shared-GPU-worker transport routes responses by matching `request_id`
/// in a pending-requests map; two concurrent callers with the same id would
/// silently overwrite each other's pending sender and produce "orphaned
/// execute_v2 response" in the reader. Production callers must always use
/// [`PreparedAsrRequestIdsV2::fresh`]; `new(...)` is retained only for tests
/// and fixture construction that assert against stable ids.
static ASR_REQUEST_SEQUENCE_V2: AtomicU64 = AtomicU64::new(1);

impl PreparedAsrRequestIdsV2 {
    /// Construct explicit stable ids for one V2 ASR request.
    ///
    /// Intended for tests and fixtures. Concurrent production code paths
    /// must call [`Self::fresh`] instead so every request gets a unique
    /// id across the process.
    pub fn new(
        request_id: impl Into<WorkerRequestIdV2>,
        audio_ref_id: impl Into<WorkerArtifactIdV2>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            audio_ref_id: audio_ref_id.into(),
        }
    }

    /// Construct unique ids for one ASR V2 request.
    ///
    /// Uses a process-global monotonic counter so concurrent callers never
    /// collide on the shared-GPU-worker pending-requests map.
    pub fn fresh() -> Self {
        let sequence = ASR_REQUEST_SEQUENCE_V2.fetch_add(1, Ordering::Relaxed);
        Self::new(
            format!("asr-v2-request-{sequence}"),
            format!("asr-v2-audio-{sequence}"),
        )
    }
}

/// Input bundle for the live ASR V2 request builder.
#[derive(Debug, Clone)]
pub struct AsrBuildInputV2<'a> {
    /// Stable ids for the request and prepared artifacts.
    pub ids: &'a PreparedAsrRequestIdsV2,
    /// Typed input transport selected by Rust.
    pub input: AsrInputSourceV2<'a>,
    /// Worker-runtime language requested by the Rust control plane.
    pub lang: &'a WorkerLanguage,
    /// Concrete V2 ASR backend selected by Rust.
    pub backend: AsrBackendV2,
    /// The exact models the worker must load for this request, resolved from
    /// the manifest by the caller that knows the language and the overrides.
    ///
    /// Borrowed rather than moved, because the caller needs the same value
    /// again when the response comes back: it is the requested half of the
    /// admission, and reading it back off the response instead would compare
    /// the answer with itself.
    pub models: &'a crate::types::worker_v2::AsrRequestedModelsV2,
    /// Per-engine configuration extras (e.g. `qwen_model`,
    /// `qwen_device`) drawn from `CommonOptions.engine_overrides.extras`.
    /// Carried through the V2 dispatch boundary so the worker spawn
    /// argv preserves what the user asked for, the typed `backend`
    /// enum only encodes WHICH engine to load, not its configuration.
    pub extras: &'a std::collections::BTreeMap<String, String>,
}

/// Concrete ASR input transport selected by the Rust control plane.
#[derive(Debug, Clone)]
pub enum AsrInputSourceV2<'a> {
    /// Rust-owned prepared audio for local model execution.
    PreparedAudio {
        /// Source audio file to transcribe.
        audio_path: &'a Path,
    },
    /// Provider-local media path retained during the migration away from the
    /// legacy batch-infer ASR route.
    ProviderMedia {
        /// Media file path readable by the worker host.
        media_path: &'a Path,
        /// Whether this provider must separate speakers, and into how many.
        diarization: ProviderDiarizationV2,
    },
}

/// Errors produced while building a live V2 ASR request.
#[derive(Debug, Error)]
pub enum AsrRequestBuildErrorV2 {
    /// The request referenced an empty audio path.
    #[error("worker protocol V2 ASR request is missing an audio path")]
    MissingAudioPath,

    /// Rust-owned prepared-artifact creation failed.
    #[error(transparent)]
    Artifact(#[from] PreparedArtifactErrorV2),
}

/// Build a live worker-protocol V2 ASR request from a typed input source.
pub async fn build_asr_request_v2(
    store: &PreparedArtifactStoreV2,
    input: AsrBuildInputV2<'_>,
) -> Result<ExecuteRequestV2, AsrRequestBuildErrorV2> {
    let (asr_input, attachments, decode_budget_seconds) = match input.input {
        AsrInputSourceV2::PreparedAudio { audio_path } => {
            if audio_path.as_os_str().is_empty() {
                return Err(AsrRequestBuildErrorV2::MissingAudioPath);
            }

            let audio_attachment = store
                .prepare_audio_file_f32le(&input.ids.audio_ref_id, audio_path)
                .await?;

            // Born from the very attachment Rust just prepared: the
            // sanctioned way to derive a decode budget for a PreparedAudio
            // request, no re-probing needed since the duration is already
            // known exactly.
            let decode_budget_seconds = Some(DecodeBudgetSeconds::for_audio(
                audio_attachment.frame_count,
                audio_attachment.sample_rate_hz,
            ));

            (
                AsrInputV2::PreparedAudio(PreparedAudioInputV2 {
                    audio_ref_id: audio_attachment.id.clone(),
                }),
                vec![ArtifactRefV2::PreparedAudio(audio_attachment)],
                decode_budget_seconds,
            )
        }
        AsrInputSourceV2::ProviderMedia {
            media_path,
            diarization,
        } => {
            if media_path.as_os_str().is_empty() {
                return Err(AsrRequestBuildErrorV2::MissingAudioPath);
            }

            // No prepared attachment carries this input's duration (Rust
            // never decodes provider media locally), so it is probed
            // directly from the file. `None` on probe failure, never a
            // fabricated budget; see `probe_decode_budget_seconds`.
            let decode_budget_seconds = probe_decode_budget_seconds(media_path).await;

            (
                AsrInputV2::ProviderMedia(ProviderMediaInputV2 {
                    media_path: media_path.to_string_lossy().as_ref().into(),
                    diarization,
                }),
                Vec::new(),
                decode_budget_seconds,
            )
        }
    };

    Ok(ExecuteRequestV2 {
        request_id: input.ids.request_id.clone(),
        task: InferenceTaskV2::Asr,
        payload: TaskRequestV2::Asr(AsrRequestV2 {
            lang: input.lang.clone(),
            backend: input.backend,
            models: input.models.clone(),
            input: asr_input,
            extras: input.extras.clone(),
            decode_budget_seconds,
        }),
        attachments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::worker_v2::{AsrInputV2, TaskRequestV2};

    #[test]
    fn builds_stable_asr_request_ids() {
        let ids = PreparedAsrRequestIdsV2::new("req-asr-v2-1", "audio-asr-v2-1");
        assert_eq!(&*ids.request_id, "req-asr-v2-1");
        assert_eq!(&*ids.audio_ref_id, "audio-asr-v2-1");
    }

    /// Two calls to `fresh()` must produce distinct request and artifact ids.
    ///
    /// Regression: the UTR+whisper code path used hardcoded literal ids
    /// (`asr-v2-request` / `asr-v2-audio`) for every concurrent ASR inference,
    /// so the shared-GPU-worker pending-request map (keyed by `request_id`)
    /// silently overwrote pending senders when two UTR ASR requests ran in
    /// parallel. The reader then reported "orphaned execute_v2 response" and
    /// the worker crashed handling the state corruption.
    #[test]
    fn fresh_produces_distinct_ids_across_calls() {
        let a = PreparedAsrRequestIdsV2::fresh();
        let b = PreparedAsrRequestIdsV2::fresh();
        assert_ne!(
            a.request_id, b.request_id,
            "request_id must be unique per call to avoid GPU-worker response collisions"
        );
        assert_ne!(
            a.audio_ref_id, b.audio_ref_id,
            "audio_ref_id must be unique per call so prepared-audio artifacts don't alias"
        );
    }

    #[tokio::test]
    async fn builds_provider_media_asr_request_without_prepared_attachments() {
        let tempdir = tempfile::tempdir().expect("tempdir should exist");
        let store = PreparedArtifactStoreV2::new(tempdir.path().join("artifacts"))
            .expect("artifact store should exist");
        let ids = PreparedAsrRequestIdsV2::new("req-asr-v2-provider", "audio-asr-v2-provider");
        let lang = WorkerLanguage::from(crate::api::LanguageCode3::yue());
        let media_path = tempdir.path().join("sample.wav");

        let empty_extras = std::collections::BTreeMap::new();
        let models = crate::types::worker_v2::AsrRequestedModelsV2::Tencent {
            engine_model_type: crate::types::worker_v2::RequestedModelV2 {
                id: crate::types::worker_v2::ModelIdV2::from_static("tencent-asr"),
                revision: crate::types::worker_v2::RequestedRevisionV2::ProviderParameter {
                    parameter: crate::types::worker_v2::ProviderParameterV2::from_static(
                        "16k_zh_large",
                    ),
                },
            },
        };
        let request = build_asr_request_v2(
            &store,
            AsrBuildInputV2 {
                ids: &ids,
                input: AsrInputSourceV2::ProviderMedia {
                    media_path: &media_path,
                    diarization: ProviderDiarizationV2::for_expected_speakers(
                        crate::api::NumSpeakers(2),
                    ),
                },
                lang: &lang,
                backend: AsrBackendV2::HkTencent,
                models: &models,
                extras: &empty_extras,
            },
        )
        .await
        .expect("provider-media request should build");

        assert!(request.attachments.is_empty());
        let TaskRequestV2::Asr(payload) = request.payload else {
            panic!("expected ASR payload");
        };
        let AsrInputV2::ProviderMedia(provider_media) = payload.input else {
            panic!("expected provider-media input");
        };
        assert_eq!(
            &*provider_media.media_path,
            media_path.to_string_lossy().as_ref()
        );
        assert_eq!(
            provider_media.diarization,
            ProviderDiarizationV2::for_expected_speakers(crate::api::NumSpeakers(2))
        );
    }
}
