//! Rust-side request builder and response reader for speaker embedding.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use thiserror::Error;

use crate::chat_ops::speaker_identity::{
    EmbeddingDimension, EmbeddingRequest, EmbeddingResponse, MinimumEmbeddingFrames,
    NotAPreparedDecode, NotAnEmbedding, PreparedPcm, SpanOutcome, SpeakerEmbedding,
    ZeroEmbeddingDimension, ZeroMinimumEmbeddingFrames,
};
use crate::types::worker_v2::{
    ArtifactRefV2, ExecuteRequestV2, ExecuteResponseV2, InferenceTaskV2, SpeakerEmbeddingBackendV2,
    SpeakerEmbeddingOutcomeV2, SpeakerEmbeddingRequestV2, SpeakerEmbeddingSpanIdV2,
    SpeakerEmbeddingSpanV2, TaskRequestV2, TaskResultV2, WorkerArtifactIdV2, WorkerRequestIdV2,
};

use super::artifacts_v2::{PreparedArtifactErrorV2, PreparedArtifactStoreV2};
use super::execute_result_v2::require_success_result;

/// Process-unique ids for one embedding execution.
///
/// Same reason as the ASR and speaker sequences: shared-GPU-worker response
/// routing is keyed by `request_id`, so a duplicate orphans a response.
static EMBEDDING_REQUEST_SEQUENCE_V2: AtomicU64 = AtomicU64::new(1);

/// Errors produced while building a live V2 embedding request.
#[derive(Debug, Error)]
pub enum SpeakerEmbeddingRequestBuildErrorV2 {
    /// The request referenced an empty audio path.
    #[error("worker protocol V2 speaker embedding request is missing an audio path")]
    MissingAudioPath,
    /// Rust-owned prepared-artifact creation failed.
    #[error(transparent)]
    Artifact(#[from] PreparedArtifactErrorV2),
    /// The prepared decode cannot index anything.
    #[error(transparent)]
    Decode(#[from] NotAPreparedDecode),
}

/// One decoded recording, and the description of it the worker will receive.
///
/// The two travel as one value because a frame span is meaningless against a
/// different decode. Returning them separately would let a caller locate spans
/// in one decode and send them against another.
#[derive(Debug, Clone)]
pub struct PreparedRecording {
    descriptor: crate::types::worker_v2::PreparedAudioRefV2,
    /// The decode every span must be located against.
    pub prepared: PreparedPcm,
    request_id: WorkerRequestIdV2,
}

/// Decode one recording to canonical mono PCM, ONCE per file.
///
/// Separate from [`PreparedRecording::request_for`] because span locating needs
/// the decode's shape before any request can exist, and decoding twice would
/// both cost a second ffmpeg pass and create two decodes that a reader could
/// no longer tell apart.
pub async fn prepare_recording_for_embedding(
    store: &PreparedArtifactStoreV2,
    audio_path: &Path,
) -> Result<PreparedRecording, SpeakerEmbeddingRequestBuildErrorV2> {
    if audio_path.as_os_str().is_empty() {
        return Err(SpeakerEmbeddingRequestBuildErrorV2::MissingAudioPath);
    }
    let sequence = EMBEDDING_REQUEST_SEQUENCE_V2.fetch_add(1, Ordering::Relaxed);
    let request_id = WorkerRequestIdV2::from(format!("speaker-embedding-v2-request-{sequence}"));
    let audio_ref_id = WorkerArtifactIdV2::from(format!("speaker-embedding-v2-audio-{sequence}"));

    let descriptor = store
        .prepare_audio_file_f32le(&audio_ref_id, audio_path)
        .await?;
    // Built from the descriptor that just described this decode, so its rate
    // and its length cannot arrive from two different places.
    let prepared = PreparedPcm::new(descriptor.sample_rate_hz.0, descriptor.frame_count.0)?;

    Ok(PreparedRecording {
        descriptor,
        prepared,
        request_id,
    })
}

impl PreparedRecording {
    /// Build the request that asks about `spans` of THIS decode.
    ///
    /// Takes the already-located [`EmbeddingRequest`], which only
    /// `chat_ops::speaker_identity::identify_speakers` builds and which only
    /// [`PreparedPcm::locate`] can put spans into, so a caller cannot ask the
    /// worker about frames nobody located.
    #[must_use]
    pub fn request_for(&self, spans: &EmbeddingRequest) -> ExecuteRequestV2 {
        let wire_spans = spans
            .spans()
            .iter()
            .map(|span| SpeakerEmbeddingSpanV2 {
                span_id: SpeakerEmbeddingSpanIdV2::from(span.span_id.as_str()),
                start_frame: crate::types::worker_v2::FrameCountV2(span.frames.start()),
                end_frame: crate::types::worker_v2::FrameCountV2(span.frames.end()),
            })
            .collect();

        ExecuteRequestV2 {
            request_id: self.request_id.clone(),
            task: InferenceTaskV2::SpeakerEmbedding,
            payload: TaskRequestV2::SpeakerEmbedding(SpeakerEmbeddingRequestV2 {
                backend: SpeakerEmbeddingBackendV2::Pyannote,
                audio_ref_id: WorkerArtifactIdV2::from(self.descriptor.id.as_ref()),
                spans: wire_spans,
            }),
            attachments: vec![ArtifactRefV2::PreparedAudio(self.descriptor.clone())],
        }
    }
}

/// Why a worker response is not a usable set of embeddings.
#[derive(Debug, Error)]
pub enum SpeakerEmbeddingResultParseError {
    /// The response was not a successful embedding result.
    #[error("{0}")]
    UnexpectedPayload(String),
    /// A returned vector is not an embedding.
    #[error("worker protocol V2 speaker embedding response carried an unusable vector: {0}")]
    Vector(#[from] NotAnEmbedding),
    /// The response claimed vectors with no components.
    #[error(transparent)]
    Dimension(#[from] ZeroEmbeddingDimension),
    /// The response claimed the model could measure a zero-frame span.
    #[error(transparent)]
    MinimumFrames(#[from] ZeroMinimumEmbeddingFrames),
    /// The response repeated one requested span id.
    #[error("worker protocol V2 speaker embedding response repeated span {span_id}")]
    DuplicateSpan {
        /// The repeated id.
        span_id: String,
    },
    /// The response did not answer exactly the set of requested spans.
    #[error(
        "worker protocol V2 speaker embedding response span mismatch \
         (missing: {missing:?}; unexpected: {unexpected:?})"
    )]
    SpanSetMismatch {
        /// Requested ids with no answer.
        missing: Vec<String>,
        /// Answered ids that were never requested.
        unexpected: Vec<String>,
    },
}

/// Read one V2 embedding response into the typed outcomes the control plane
/// reasons with.
///
/// The width every vector is checked against is the one the RESPONSE declares,
/// not a constant here: the worker reports the loaded model's own dimension, so
/// a constant on this side would go on agreeing with a model that had moved.
pub fn parse_speaker_embedding_response_v2(
    response: &ExecuteResponseV2,
    request: &EmbeddingRequest,
) -> Result<EmbeddingResponse, SpeakerEmbeddingResultParseError> {
    let result = require_success_result(response, "speaker embedding")
        .map_err(|failure| SpeakerEmbeddingResultParseError::UnexpectedPayload(failure.into()))?;
    let TaskResultV2::SpeakerEmbeddingResult(result) = result else {
        return Err(SpeakerEmbeddingResultParseError::UnexpectedPayload(
            format!(
                "worker protocol V2 speaker embedding response returned {} data",
                result_kind(result)
            ),
        ));
    };

    let dimension = EmbeddingDimension::try_from(result.dimension)?;
    let minimum_frames = MinimumEmbeddingFrames::try_from(result.minimum_frames.0)?;
    let declared_width = dimension.get() as usize;
    let mut outcomes = std::collections::BTreeMap::new();
    for span in &result.spans {
        let outcome = match &span.outcome {
            SpeakerEmbeddingOutcomeV2::Embedded { vector } => SpanOutcome::Embedded(
                SpeakerEmbedding::from_worker(vector.clone(), declared_width)?,
            ),
            SpeakerEmbeddingOutcomeV2::TooShort { frame_count } => SpanOutcome::TooShort {
                frames: frame_count.0,
            },
        };
        let span_id = span.span_id.as_ref().to_owned();
        if outcomes.insert(span_id.clone(), outcome).is_some() {
            return Err(SpeakerEmbeddingResultParseError::DuplicateSpan { span_id });
        }
    }

    let expected: std::collections::BTreeSet<&str> = request
        .spans()
        .iter()
        .map(|span| span.span_id.as_str())
        .collect();
    let answered: std::collections::BTreeSet<&str> = outcomes.keys().map(String::as_str).collect();
    if expected != answered {
        return Err(SpeakerEmbeddingResultParseError::SpanSetMismatch {
            missing: expected
                .difference(&answered)
                .map(|span_id| (*span_id).to_owned())
                .collect(),
            unexpected: answered
                .difference(&expected)
                .map(|span_id| (*span_id).to_owned())
                .collect(),
        });
    }

    Ok(EmbeddingResponse {
        outcomes,
        minimum_frames,
        dimension,
    })
}

/// A short label for a result variant, for the "returned X data" refusals.
fn result_kind(result: &TaskResultV2) -> &'static str {
    match result {
        TaskResultV2::WhisperChunkResult(_) => "ASR chunk",
        TaskResultV2::MonologueAsrResult(_) => "monologue ASR",
        TaskResultV2::WhisperTokenTimingResult(_) => "forced-alignment token",
        TaskResultV2::IndexedWordTimingResult(_) => "indexed timing",
        TaskResultV2::MorphosyntaxResult(_) => "morphosyntax",
        TaskResultV2::UtsegResult(_) => "utterance-segmentation",
        TaskResultV2::TranslationResult(_) => "translation",
        TaskResultV2::CorefResult(_) => "coreference",
        TaskResultV2::SpeakerResult(_) => "speaker diarization",
        TaskResultV2::SpeakerEmbeddingResult(_) => "speaker embedding",
        TaskResultV2::OpensmileResult(_) => "openSMILE feature",
        TaskResultV2::AvqiResult(_) => "AVQI feature",
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;
    use crate::api::NonNegativeSeconds;
    use crate::chat_ops::speaker_identity::RequestedSpan;
    use crate::media::window::MediaWindow;
    use crate::time::FileMs;
    use crate::types::worker_v2::{
        FrameCountV2, SpeakerEmbeddingResultV2, SpeakerEmbeddingSpanResultV2,
    };

    fn request() -> EmbeddingRequest {
        let prepared = match PreparedPcm::new(16_000, 160_000) {
            Ok(prepared) => prepared,
            Err(error) => panic!("test prepared audio is valid: {error}"),
        };
        let spans = [("first", 0, 100), ("second", 100, 200)]
            .into_iter()
            .map(|(span_id, start, end)| {
                let window = match MediaWindow::new(FileMs::new(start), FileMs::new(end)) {
                    Ok(window) => window,
                    Err(error) => panic!("test window is valid: {error}"),
                };
                let frames = match prepared.locate(window) {
                    Ok(frames) => frames,
                    Err(error) => panic!("test window lies in the prepared audio: {error}"),
                };
                RequestedSpan {
                    span_id: span_id.to_owned(),
                    frames,
                }
            })
            .collect();
        EmbeddingRequest::for_test(spans)
    }

    fn response(dimension: u32, minimum_frames: u64, span_ids: &[&str]) -> ExecuteResponseV2 {
        ExecuteResponseV2::success(
            WorkerRequestIdV2::from("speaker-response-test"),
            TaskResultV2::SpeakerEmbeddingResult(SpeakerEmbeddingResultV2 {
                dimension,
                minimum_frames: FrameCountV2(minimum_frames),
                spans: span_ids
                    .iter()
                    .map(|span_id| SpeakerEmbeddingSpanResultV2 {
                        span_id: SpeakerEmbeddingSpanIdV2::from(*span_id),
                        outcome: SpeakerEmbeddingOutcomeV2::TooShort {
                            frame_count: FrameCountV2(10),
                        },
                    })
                    .collect(),
            }),
            NonNegativeSeconds::try_from(0.01).expect("fixture elapsed"),
        )
    }

    #[test]
    fn refuses_zero_model_measurements() {
        assert!(matches!(
            parse_speaker_embedding_response_v2(&response(0, 10, &["first", "second"]), &request()),
            Err(SpeakerEmbeddingResultParseError::Dimension(_))
        ));
        assert!(matches!(
            parse_speaker_embedding_response_v2(&response(2, 0, &["first", "second"]), &request()),
            Err(SpeakerEmbeddingResultParseError::MinimumFrames(_))
        ));
    }

    #[test]
    fn refuses_missing_and_unexpected_span_answers() {
        match parse_speaker_embedding_response_v2(
            &response(2, 10, &["first", "unexpected"]),
            &request(),
        ) {
            Err(SpeakerEmbeddingResultParseError::SpanSetMismatch {
                missing,
                unexpected,
            }) => {
                assert_eq!(missing, ["second"]);
                assert_eq!(unexpected, ["unexpected"]);
            }
            other => panic!("expected a span-set mismatch, got {other:?}"),
        }
    }

    #[test]
    fn refuses_duplicate_span_answers() {
        assert!(matches!(
            parse_speaker_embedding_response_v2(
                &response(2, 10, &["first", "first", "second"]),
                &request(),
            ),
            Err(SpeakerEmbeddingResultParseError::DuplicateSpan { span_id })
                if span_id == "first"
        ));
    }
}
