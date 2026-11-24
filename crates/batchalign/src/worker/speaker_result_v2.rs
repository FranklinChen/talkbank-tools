//! Rust-side adapters for live worker-protocol V2 speaker results.
//!
//! Speaker diarization is still a raw-model concern at the worker boundary.
//! This adapter keeps Rust in charge of interpreting the typed response shape
//! instead of letting downstream callers pattern-match on generic JSON.

use serde::Deserialize;

use crate::api::AudioPositionSeconds;
use crate::types::worker_v2::{
    ExecuteResponseV2, SpeakerInferenceEvidenceV2, SpeakerResultV2, SpeakerSegmentV2, TaskResultV2,
};
use crate::worker::execute_result_v2::{ExecuteFailureRead, take_success_result};
use batchalign_types::interval::AdmittedInterval;

/// Why a V2 speaker execute response could not be read as speaker evidence.
///
/// Kept distinct from a bare `String` (every other V2 result adapter's error
/// shape) so the caller can route a worker-reported PROTOCOL failure to its
/// own category instead of collapsing every parse failure into one generic
/// validation error. Without this, `infer_speaker` in `transcribe/infer.rs`
/// had no way to tell "the worker's response was shaped wrong" apart from
/// "the worker told us it could not download a gated Hugging Face model",
/// and reported both as `ServerError::Validation`, which the dashboard then
/// labelled a pipeline bug, even when the true cause is a configuration
/// condition on the operator's own machine.
#[derive(Debug, Clone)]
pub enum SpeakerResultParseError {
    /// The worker's response itself reported failure, carried whole so the
    /// caller categorizes it by its protocol code, never its message.
    Protocol(ExecuteFailureRead),
    /// The worker reported success but the payload was not a speaker result.
    UnexpectedPayload(String),
}

impl std::fmt::Display for SpeakerResultParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Protocol(failure) => write!(f, "{failure}"),
            Self::UnexpectedPayload(message) => f.write_str(message),
        }
    }
}

impl From<ExecuteFailureRead> for SpeakerResultParseError {
    fn from(value: ExecuteFailureRead) -> Self {
        Self::Protocol(value)
    }
}

/// Consume one V2 speaker execute response and take its admitted speaker payload.
pub fn parse_speaker_result_v2(
    response: ExecuteResponseV2,
) -> Result<SpeakerResultV2, SpeakerResultParseError> {
    let result = take_success_result(response, "speaker")?;

    match result {
        TaskResultV2::FaPreparation(_) => Err(SpeakerResultParseError::UnexpectedPayload(
            "worker protocol V2 speaker response returned FA preparation data".into(),
        )),
        TaskResultV2::SpeakerResult(result) => Ok(result),
        TaskResultV2::MorphosyntaxResult(_) => Err(SpeakerResultParseError::UnexpectedPayload(
            "worker protocol V2 speaker response returned morphosyntax data".into(),
        )),
        TaskResultV2::UtsegResult(_) => Err(SpeakerResultParseError::UnexpectedPayload(
            "worker protocol V2 speaker response returned utterance-segmentation data".into(),
        )),
        TaskResultV2::WhisperChunkResult(_) => Err(SpeakerResultParseError::UnexpectedPayload(
            "worker protocol V2 speaker response returned ASR chunk data".into(),
        )),
        TaskResultV2::MonologueAsrResult(_) => Err(SpeakerResultParseError::UnexpectedPayload(
            "worker protocol V2 speaker response returned monologue ASR data".into(),
        )),
        TaskResultV2::WhisperTokenTimingResult(_) => {
            Err(SpeakerResultParseError::UnexpectedPayload(
                "worker protocol V2 speaker response returned forced-alignment token data".into(),
            ))
        }
        TaskResultV2::IndexedWordTimingResult(_) => {
            Err(SpeakerResultParseError::UnexpectedPayload(
                "worker protocol V2 speaker response returned indexed timing data".into(),
            ))
        }
        TaskResultV2::TranslationResult(_) => Err(SpeakerResultParseError::UnexpectedPayload(
            "worker protocol V2 speaker response returned translation data".into(),
        )),
        TaskResultV2::CorefResult(_) => Err(SpeakerResultParseError::UnexpectedPayload(
            "worker protocol V2 speaker response returned coreference data".into(),
        )),
        TaskResultV2::WordSegmentationResult(_) => Err(SpeakerResultParseError::UnexpectedPayload(
            "worker protocol V2 speaker response returned word-segmentation data".into(),
        )),
        TaskResultV2::SpeakerEmbeddingResult(_) => Err(SpeakerResultParseError::UnexpectedPayload(
            "worker protocol V2 speaker response returned speaker embedding data".into(),
        )),
        TaskResultV2::OpensmileResult(_) => Err(SpeakerResultParseError::UnexpectedPayload(
            "worker protocol V2 speaker response returned openSMILE feature data".into(),
        )),
        TaskResultV2::AvqiResult(_) => Err(SpeakerResultParseError::UnexpectedPayload(
            "worker protocol V2 speaker response returned AVQI feature data".into(),
        )),
    }
}

/// One segment of a pyannoteAI job's output, as the provider wrote it.
///
/// `start` and `end` are POSITIONS in the recording, in seconds: reading them
/// as [`AudioPositionSeconds`] refuses a non-finite or negative bound at
/// deserialization, where a hand-written check after the fact used to.
#[derive(Debug, Deserialize)]
struct PyannoteAiSegment {
    start: AudioPositionSeconds,
    end: AudioPositionSeconds,
    speaker: String,
}

/// Normalize backend-specific evidence into ordered millisecond speaker spans.
///
/// Keeping this projection in Rust means completed paid provider output can be
/// stored and replayed under a newer normalization revision without another
/// service call.
pub(crate) fn normalize_speaker_evidence_v2(
    evidence: &SpeakerInferenceEvidenceV2,
) -> Result<Vec<SpeakerSegmentV2>, String> {
    let mut segments = match evidence {
        SpeakerInferenceEvidenceV2::PyannoteAi {
            job_id,
            output,
            warning: _,
        } => {
            if job_id.as_str().trim().is_empty() {
                return Err("pyannoteAI evidence has an empty provider job id".to_owned());
            }
            let raw_segments = output
                .get("exclusiveDiarization")
                .or_else(|| output.get("diarization"))
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    "pyannoteAI completed job has no diarization segment array".to_owned()
                })?;
            raw_segments
                .iter()
                .enumerate()
                .map(|(index, raw)| {
                    let parsed: PyannoteAiSegment =
                        serde_json::from_value(raw.clone()).map_err(|error| {
                            format!("pyannoteAI segment {index} has invalid data: {error}")
                        })?;
                    if parsed.speaker.trim().is_empty() {
                        return Err(format!(
                            "pyannoteAI segment {index} has an empty speaker label"
                        ));
                    }
                    // Admitted like every provider interval: rounded once,
                    // refused when inverted or beyond the media range (an
                    // epoch timestamp in seconds is not a media offset).
                    let interval = AdmittedInterval::admit_positions(parsed.start, parsed.end)
                        .map_err(|refusal| {
                            format!("pyannoteAI segment {index} is not an interval: {refusal}")
                        })?;
                    Ok(SpeakerSegmentV2 {
                        interval,
                        speaker: parsed.speaker,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?
        }
        SpeakerInferenceEvidenceV2::Pyannote { segments, .. } => segments.clone(),
    };
    segments.sort_by(|left, right| {
        (left.interval, left.speaker.as_str()).cmp(&(right.interval, right.speaker.as_str()))
    });
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::NonNegativeSeconds;
    use crate::types::worker_v2::{
        ExecuteResponseV2, ProtocolErrorCodeV2, SpeakerInferenceEvidenceV2, SpeakerProviderJobIdV2,
        SpeakerResultV2, SpeakerSegmentV2, WorkerRequestIdV2,
    };

    /// Guards: Local speaker replies preserve exact segment bounds and the loaded graph on the worker wire.
    /// Kind: wire
    /// Not a type: Externally produced JSON can omit digest evidence or supply malformed SHA-256 values.
    /// State: Synthetic response bytes drive the actual worker payload decoder and speaker projection.
    /// Concurrency: none
    #[test]
    fn parses_speaker_segments_from_typed_v2_result() {
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("req-speaker-v2-1"),
            TaskResultV2::SpeakerResult(SpeakerResultV2 {
                evidence: SpeakerInferenceEvidenceV2::Pyannote {
                    loaded_graph: None,
                    segments: vec![SpeakerSegmentV2 {
                        interval: AdmittedInterval::admit_millis(0, 900).expect("ordered"),
                        speaker: "SPEAKER_1".into(),
                    }],
                },
            }),
            NonNegativeSeconds::try_from(0.01).expect("fixture elapsed"),
        );

        let mut wire = serde_json::to_value(response).expect("worker JSON");
        let loaded_graph = serde_json::json!({
            "pipeline_config_sha256": "1".repeat(64),
            "segmentation_sha256": "2".repeat(64),
            "embedding_sha256": "3".repeat(64),
        });
        assert!(wire["result"]["evidence"].get("loaded_graph").is_none());
        for reported in [None, Some(loaded_graph.clone())] {
            if let Some(graph) = &reported {
                wire["result"]["evidence"]["loaded_graph"] = graph.clone();
            }
            let decoded = serde_json::from_value(wire.clone()).expect("admitted worker JSON");
            let parsed = parse_speaker_result_v2(decoded).expect("speaker result should parse");
            let segments = normalize_speaker_evidence_v2(&parsed.evidence)
                .expect("speaker evidence should normalize");
            assert_eq!(segments[0].speaker, "SPEAKER_1");
            assert_eq!(segments[0].interval.end_ms(), 900);
            let projected = serde_json::to_value(parsed.evidence).expect("speaker evidence JSON");
            assert_eq!(projected.get("loaded_graph"), reported.as_ref());
        }
        for malformed in ["A".repeat(64), format!("{}x", "2".repeat(64))] {
            wire["result"]["evidence"]["loaded_graph"]["segmentation_sha256"] =
                serde_json::Value::String(malformed);
            let refusal = serde_json::from_value::<ExecuteResponseV2>(wire.clone())
                .expect_err("malformed graph digest");
            assert!(refusal.to_string().contains("SHA-256"), "{refusal}");
        }
    }

    /// Guards: Completed hosted evidence normalizes exact second offsets into milliseconds in Rust.
    /// Kind: contract
    /// Not a type: Provider JSON fields and time-unit conversion remain concrete numeric semantics.
    /// State: Synthetic provider job JSON feeds the real normalization adapter.
    /// Concurrency: none
    #[test]
    fn normalizes_completed_pyannote_ai_evidence_without_python() {
        let evidence = SpeakerInferenceEvidenceV2::PyannoteAi {
            job_id: SpeakerProviderJobIdV2::from("job-1"),
            output: serde_json::from_value(serde_json::json!({
                "exclusiveDiarization": [
                    {"start": 1.25, "end": 1.5, "speaker": "SPEAKER_01"},
                    {"start": 0.0, "end": 0.75, "speaker": "SPEAKER_00"}
                ]
            }))
            .expect("provider output object"),
            warning: None,
        };

        let segments = normalize_speaker_evidence_v2(&evidence).expect("normalize");
        assert_eq!(segments[0].interval.start_ms(), 0);
        assert_eq!(segments[1].interval.end_ms(), 1500);
    }

    /// A pyannoteAI bound far past any recording (an epoch time in seconds)
    /// is refused, not saturated into a millisecond count; an inverted pair
    /// is refused by the same admission.
    /// Guards: Provider epoch-like, reversed and negative offsets are refused with the correct offending segment index.
    /// Kind: boundary
    /// Not a type: Untrusted numeric JSON can fall outside the position domain or conflict at interval admission.
    /// State: Synthetic hosted job results cover external numeric failure cases directly.
    /// Concurrency: none
    #[test]
    fn a_pyannote_ai_bound_beyond_the_media_range_is_refused() {
        for (start, end) in [(1.7e9, 1.7e9 + 1.0), (2.0, 1.0)] {
            let evidence = SpeakerInferenceEvidenceV2::PyannoteAi {
                job_id: SpeakerProviderJobIdV2::from("job-3"),
                output: serde_json::from_value(serde_json::json!({
                    "diarization": [{"start": start, "end": end, "speaker": "SPEAKER_00"}]
                }))
                .expect("provider output object"),
                warning: None,
            };
            let error = normalize_speaker_evidence_v2(&evidence).expect_err("refused");
            assert!(error.contains("segment 0 is not an interval"), "{error}");
        }

        let evidence = SpeakerInferenceEvidenceV2::PyannoteAi {
            job_id: SpeakerProviderJobIdV2::from("job-2"),
            output: serde_json::from_value(serde_json::json!({
                "diarization": [
                    {"start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"},
                    {"start": -0.25, "end": 1.0, "speaker": "SPEAKER_01"}
                ]
            }))
            .expect("provider output object"),
            warning: None,
        };

        let error = normalize_speaker_evidence_v2(&evidence).expect_err("negative start");
        assert!(error.contains("segment 1 has invalid data"), "{error}");
    }

    /// The regression this type exists for: a worker-reported protocol
    /// failure (a Hugging Face Hub access denial, in the real 2026-09-02
    /// incident) must come back as `Protocol { code, .. }`, never collapsed
    /// into the same `UnexpectedPayload` shape as an ordinary shape mismatch.
    /// Guards: A failed response carries its worker code and message while a successful translation payload is a speaker shape refusal.
    /// Kind: policy
    /// Not a type: Closed response variants cannot enforce the adapter's choices for diagnostics and task-specific payloads.
    /// State: Synthetic success and failure responses carry exact alternative runtime values.
    /// Concurrency: none
    #[test]
    fn a_protocol_failure_preserves_its_code_distinct_from_a_shape_mismatch() {
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("req-speaker-v2-2"),
            TaskResultV2::TranslationResult(crate::types::worker_v2::TranslationResultV2 {
                items: vec![crate::types::worker_v2::TranslationItemResultV2::BlankInput],
            }),
            NonNegativeSeconds::try_from(0.01).expect("fixture elapsed"),
        );

        let error =
            parse_speaker_result_v2(response).expect_err("translation result should be rejected");
        assert!(matches!(
            error,
            SpeakerResultParseError::UnexpectedPayload(_)
        ));
        assert!(error.to_string().contains("translation data"));

        let response = ExecuteResponseV2::failure(
            WorkerRequestIdV2::from("req-speaker-v2-3"),
            ProtocolErrorCodeV2::ModelAccessDenied,
            "could not download the Hugging Face model at \
             pyannote/speaker-diarization-community-1: its repository is gated"
                .to_owned(),
            NonNegativeSeconds::try_from(0.01).expect("fixture elapsed"),
        );

        let error =
            parse_speaker_result_v2(response).expect_err("a failed response is never speaker data");

        match error {
            SpeakerResultParseError::Protocol(failure) => {
                assert_eq!(failure.code(), ProtocolErrorCodeV2::ModelAccessDenied);
                assert!(
                    failure
                        .message()
                        .contains("speaker-diarization-community-1")
                );
            }
            SpeakerResultParseError::UnexpectedPayload(message) => {
                panic!("a protocol failure must not read as a shape mismatch: {message}");
            }
        }
    }
}
