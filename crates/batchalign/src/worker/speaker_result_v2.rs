//! Rust-side adapters for live worker-protocol V2 speaker results.
//!
//! Speaker diarization is still a raw-model concern at the worker boundary.
//! This adapter keeps Rust in charge of interpreting the typed response shape
//! instead of letting downstream callers pattern-match on generic JSON.

use serde::Deserialize;

use crate::api::DurationMs;
use crate::types::worker_v2::{
    ExecuteResponseV2, ProtocolErrorCodeV2, SpeakerInferenceEvidenceV2, SpeakerResultV2,
    SpeakerSegmentV2, TaskResultV2,
};
use crate::worker::execute_result_v2::{ExecuteFailureRead, require_success_result};

/// Why a V2 speaker execute response could not be read as speaker evidence.
///
/// Kept distinct from a bare `String` (every other V2 result adapter's error
/// shape) so the caller can route a worker-reported PROTOCOL failure to its
/// own category instead of collapsing every parse failure into one generic
/// validation error. Without this, `infer_speaker` in `transcribe/infer.rs`
/// had no way to tell "the worker's response was shaped wrong" apart from
/// "the worker told us it could not download a gated Hugging Face model",
/// and reported both as `ServerError::Validation`, which the dashboard
/// renders as "pipeline bug, filed automatically" even when the true cause
/// is a configuration condition on the operator's own machine.
#[derive(Debug, Clone)]
pub enum SpeakerResultParseError {
    /// The worker's response itself reported failure; carries the protocol
    /// code so the caller can categorize it without re-parsing the message.
    Protocol {
        /// The failed request's protocol category.
        code: ProtocolErrorCodeV2,
        /// The fully formatted diagnostic.
        message: String,
    },
    /// The worker reported success but the payload was not a speaker result.
    UnexpectedPayload(String),
}

impl std::fmt::Display for SpeakerResultParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Protocol { message, .. } => f.write_str(message),
            Self::UnexpectedPayload(message) => f.write_str(message),
        }
    }
}

impl From<ExecuteFailureRead> for SpeakerResultParseError {
    fn from(value: ExecuteFailureRead) -> Self {
        Self::Protocol {
            code: value.code,
            message: value.message,
        }
    }
}

/// Parse one V2 speaker execute response into the typed segment list.
pub fn parse_speaker_result_v2(
    response: &ExecuteResponseV2,
) -> Result<&SpeakerResultV2, SpeakerResultParseError> {
    let result = require_success_result(response, "speaker")?;

    match result {
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

#[derive(Debug, Deserialize)]
struct PyannoteAiSegment {
    start: f64,
    end: f64,
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
                    let start_ms = seconds_to_ms(parsed.start, index, "start")?;
                    let end_ms = seconds_to_ms(parsed.end, index, "end")?;
                    if end_ms < start_ms {
                        return Err(format!(
                            "pyannoteAI segment {index} has an inverted interval"
                        ));
                    }
                    Ok(SpeakerSegmentV2 {
                        start_ms,
                        end_ms,
                        speaker: parsed.speaker,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?
        }
        SpeakerInferenceEvidenceV2::Pyannote { segments }
        | SpeakerInferenceEvidenceV2::Nemo { segments } => segments.clone(),
    };
    segments.sort_by(|left, right| {
        (left.start_ms.0, left.end_ms.0, left.speaker.as_str()).cmp(&(
            right.start_ms.0,
            right.end_ms.0,
            right.speaker.as_str(),
        ))
    });
    Ok(segments)
}

fn seconds_to_ms(value: f64, index: usize, field: &str) -> Result<DurationMs, String> {
    if !value.is_finite() || value < 0.0 || value > (u64::MAX as f64 / 1000.0) {
        return Err(format!(
            "pyannoteAI segment {index} has invalid {field} seconds"
        ));
    }
    Ok(DurationMs((value * 1000.0).round() as u64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::DurationSeconds;
    use crate::types::worker_v2::{
        ExecuteResponseV2, SpeakerInferenceEvidenceV2, SpeakerProviderJobIdV2, SpeakerResultV2,
        SpeakerSegmentV2, WorkerRequestIdV2,
    };

    #[test]
    fn parses_speaker_segments_from_typed_v2_result() {
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("req-speaker-v2-1"),
            TaskResultV2::SpeakerResult(SpeakerResultV2 {
                evidence: SpeakerInferenceEvidenceV2::Pyannote {
                    segments: vec![SpeakerSegmentV2 {
                        start_ms: DurationMs(0),
                        end_ms: DurationMs(900),
                        speaker: "SPEAKER_1".into(),
                    }],
                },
            }),
            DurationSeconds(0.01),
        );

        let parsed = parse_speaker_result_v2(&response).expect("speaker result should parse");
        let segments = normalize_speaker_evidence_v2(&parsed.evidence)
            .expect("speaker evidence should normalize");
        assert_eq!(segments[0].speaker, "SPEAKER_1");
        assert_eq!(segments[0].end_ms, DurationMs(900));
    }

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
        assert_eq!(segments[0].start_ms, DurationMs(0));
        assert_eq!(segments[1].end_ms, DurationMs(1500));
    }

    #[test]
    fn rejects_non_speaker_v2_payloads() {
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("req-speaker-v2-2"),
            TaskResultV2::TranslationResult(crate::types::worker_v2::TranslationResultV2 {
                items: vec![crate::types::worker_v2::TranslationItemResultV2::BlankInput],
            }),
            DurationSeconds(0.01),
        );

        let error =
            parse_speaker_result_v2(&response).expect_err("translation result should be rejected");
        assert!(matches!(
            error,
            SpeakerResultParseError::UnexpectedPayload(_)
        ));
        assert!(error.to_string().contains("translation data"));
    }

    /// The regression this type exists for: a worker-reported protocol
    /// failure (a Hugging Face Hub access denial, in the real 2026-09-02
    /// incident) must come back as `Protocol { code, .. }`, never collapsed
    /// into the same `UnexpectedPayload` shape as an ordinary shape mismatch.
    #[test]
    fn a_protocol_failure_preserves_its_code_distinct_from_a_shape_mismatch() {
        let response = ExecuteResponseV2::failure(
            WorkerRequestIdV2::from("req-speaker-v2-3"),
            ProtocolErrorCodeV2::ModelAccessDenied,
            "could not download the Hugging Face model at \
             pyannote/speaker-diarization-community-1: its repository is gated"
                .to_owned(),
            DurationSeconds(0.01),
        );

        let error = parse_speaker_result_v2(&response)
            .expect_err("a failed response is never speaker data");

        match error {
            SpeakerResultParseError::Protocol { code, message } => {
                assert_eq!(code, ProtocolErrorCodeV2::ModelAccessDenied);
                assert!(message.contains("speaker-diarization-community-1"));
            }
            SpeakerResultParseError::UnexpectedPayload(message) => {
                panic!("a protocol failure must not read as a shape mismatch: {message}");
            }
        }
    }
}
