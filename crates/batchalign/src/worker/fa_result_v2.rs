//! Rust-side adapters for staged worker-protocol V2 forced-alignment results.
//!
//! The staged Python V2 FA executor now returns typed `ExecuteResponseV2`
//! payloads, but the production FA pipeline still expects the existing
//! `batchalign-chat-ops` alignment domain. This module is the bridge:
//!
//! - accept a typed V2 worker response
//! - normalize it into the established FA raw-response shape
//! - reuse the existing deterministic Rust alignment and timing parser
//!
//! That keeps model-output interpretation on the Rust side while the transport
//! migration is still staged.

use crate::chat_ops::nlp::{FaIndexedTiming, FaRawResponse, FaRawToken};
use crate::worker::execute_result_v2::{ExecuteFailureRead, require_success_result};

use crate::types::worker_v2::{ExecuteResponseV2, ProtocolErrorCodeV2, TaskResultV2};

/// A recoverable input constraint a CTC forced-alignment engine reported.
///
/// Each is a property of THIS request's words and audio that the row's
/// fallback engine does not share (see `FaFallbackPolicy::RetryGroupOn`), so
/// it is what decides a retry. Read once, from the worker's failure, by
/// [`FaWorkerFailure::classify`]; every consumer matches this value.
///
/// # The one place engine exception text is read
///
/// The engine raises a plain PyTorch `RuntimeError`/`ValueError` with no class
/// or code that names the constraint, and the worker protocol forwards only
/// its text under `RuntimeFailure`. So the constraint is recognised by the
/// engine's own words, HERE, where the response is read, and nowhere else:
/// the transport used to search the rendered `ServerError::Validation` string
/// for the same phrases, which classified by whatever prose wrapped them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaEngineConstraint {
    /// The CTC target sequence is longer than the emissions can carry.
    CtcTargetsTooLong,
    /// The targets contain the CTC blank index.
    CtcBlankIndexInTargets,
    /// The window is shorter than the feature extractor's receptive field
    /// (Wave2Vec MMS_FA's seventh conv layer, kernel 2, sees one sample).
    InputShorterThanKernel,
}

impl FaEngineConstraint {
    const ALL: [Self; 3] = [
        Self::CtcTargetsTooLong,
        Self::CtcBlankIndexInTargets,
        Self::InputShorterThanKernel,
    ];

    /// The phrase the engine's exception carries for this constraint.
    const fn engine_text(self) -> &'static str {
        match self {
            Self::CtcTargetsTooLong => "targets length is too long for CTC",
            Self::CtcBlankIndexInTargets => "targets Tensor shouldn't contain blank index",
            Self::InputShorterThanKernel => "Kernel size can't be greater than actual input size",
        }
    }

    /// The reason recorded on the fallback event and in fallback evidence.
    ///
    /// Unchanged from the strings those records have always carried, so a
    /// reader of older evidence and newer evidence sees one vocabulary.
    pub const fn reason(self) -> &'static str {
        match self {
            Self::InputShorterThanKernel => {
                "audio segment too short for Wave2Vec feature extractor"
            }
            // The CTC constraints are recorded in the engine's own words.
            Self::CtcTargetsTooLong | Self::CtcBlankIndexInTargets => self.engine_text(),
        }
    }

    fn recognize(message: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|constraint| message.contains(constraint.engine_text()))
    }
}

/// A forced-alignment worker's failure report, classified once where it is
/// read.
///
/// Built only by [`Self::classify`], an exhaustive match over the protocol
/// code, so a new code must choose its class there; the fields are private,
/// so no caller can pair a report with a class of its own choosing. The
/// transport decides what a failed request does next (retry, leave
/// unaligned, fail the file) by matching [`Self::class`], never by reading
/// message text.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{failure}")]
pub struct FaWorkerFailure {
    class: FaFailureClass,
    failure: ExecuteFailureRead,
}

/// What a forced-alignment worker failure means for the request: the class
/// [`FaWorkerFailure::classify`] gave it. Data-free apart from the
/// constraint, so holding one does not let a caller forge a failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaFailureClass {
    /// The model failed on this request's input with a recognised constraint.
    EngineConstraint(FaEngineConstraint),
    /// The worker has no model for the requested engine loaded
    /// (`ModelUnavailable`).
    ModelUnavailable,
    /// A failure without an admitted group-local recovery: an unclassified
    /// runtime failure, a refusal, download failure or device memory pressure.
    /// It must reach the file retry policy with its producing category.
    FileFailure,
}

impl FaWorkerFailure {
    /// Classify one worker-reported failure by its protocol code.
    pub fn classify(failure: ExecuteFailureRead) -> Self {
        let class = match failure.code() {
            ProtocolErrorCodeV2::RuntimeFailure => {
                match FaEngineConstraint::recognize(failure.message()) {
                    Some(constraint) => FaFailureClass::EngineConstraint(constraint),
                    None => FaFailureClass::FileFailure,
                }
            }
            ProtocolErrorCodeV2::ModelUnavailable => FaFailureClass::ModelUnavailable,
            ProtocolErrorCodeV2::UnsupportedProtocol
            | ProtocolErrorCodeV2::InvalidPayload
            | ProtocolErrorCodeV2::MissingAttachment
            | ProtocolErrorCodeV2::AttachmentUnreadable
            | ProtocolErrorCodeV2::ProviderQuotaExhausted
            | ProtocolErrorCodeV2::ModelAccessDenied
            | ProtocolErrorCodeV2::UnsupportedLanguage
            | ProtocolErrorCodeV2::ModelDownloadFailed
            | ProtocolErrorCodeV2::DeviceOutOfMemory
            | ProtocolErrorCodeV2::ModelOutputInvalid => FaFailureClass::FileFailure,
        };
        Self { class, failure }
    }

    /// What the failure means for the request.
    pub fn class(&self) -> FaFailureClass {
        self.class
    }

    /// The worker's report this was classified from.
    pub fn into_report(self) -> ExecuteFailureRead {
        self.failure
    }
}

/// Why a forced-alignment response is not a timing payload.
#[derive(Debug, thiserror::Error)]
pub enum FaPayloadRefusal {
    /// The worker answered with a failure rather than a result.
    #[error(transparent)]
    Failed(#[from] FaWorkerFailure),
    /// A successful response carried another task's result: the worker broke
    /// the protocol.
    #[error("worker protocol V2 forced-alignment response returned {0}")]
    WrongResult(&'static str),
}

/// Read one forced-alignment execute response into its timing payload: the
/// one read of an FA response. The evidence admission
/// (`fa::raw_evidence::FaRawEvidence::admit_requested`) calls it once and
/// keeps the payload; alignment consumes that payload, never the response
/// again.
///
/// # Errors
///
/// [`FaPayloadRefusal`]: a worker failure classified by its protocol code,
/// or a result for another task.
pub fn read_fa_payload(response: &ExecuteResponseV2) -> Result<FaRawResponse, FaPayloadRefusal> {
    let result =
        require_success_result(response, "forced-alignment").map_err(FaWorkerFailure::classify)?;
    Ok(match result {
        TaskResultV2::WhisperTokenTimingResult(result) => FaRawResponse::TokenLevel {
            tokens: result
                .tokens
                .iter()
                .map(|token| FaRawToken {
                    text: token.text.clone(),
                    time_s: token.time_s,
                })
                .collect(),
        },
        TaskResultV2::IndexedWordTimingResult(result) => FaRawResponse::IndexedWordLevel {
            indexed_timings: result
                .indexed_timings
                .iter()
                .map(|timing| {
                    timing.as_ref().map(|timing| FaIndexedTiming {
                        start_ms: timing.start_ms.0,
                        end_ms: timing.end_ms.0,
                        confidence: timing.confidence,
                    })
                })
                .collect(),
        },
        TaskResultV2::MorphosyntaxResult(_) => {
            return Err(FaPayloadRefusal::WrongResult("morphosyntax data"));
        }
        TaskResultV2::UtsegResult(_) => {
            return Err(FaPayloadRefusal::WrongResult("utterance-segmentation data"));
        }
        TaskResultV2::WhisperChunkResult(_) => {
            return Err(FaPayloadRefusal::WrongResult("ASR chunk data"));
        }
        TaskResultV2::TranslationResult(_) => {
            return Err(FaPayloadRefusal::WrongResult("translation data"));
        }
        TaskResultV2::WordSegmentationResult(_) => {
            return Err(FaPayloadRefusal::WrongResult("word-segmentation data"));
        }
        TaskResultV2::CorefResult(_) => {
            return Err(FaPayloadRefusal::WrongResult("coreference data"));
        }
        TaskResultV2::MonologueAsrResult(_) => {
            return Err(FaPayloadRefusal::WrongResult("monologue ASR data"));
        }
        TaskResultV2::SpeakerResult(_) => {
            return Err(FaPayloadRefusal::WrongResult("speaker diarization data"));
        }
        TaskResultV2::SpeakerEmbeddingResult(_) => {
            return Err(FaPayloadRefusal::WrongResult("speaker embedding data"));
        }
        TaskResultV2::OpensmileResult(_) => {
            return Err(FaPayloadRefusal::WrongResult("openSMILE feature data"));
        }
        TaskResultV2::AvqiResult(_) => {
            return Err(FaPayloadRefusal::WrongResult("AVQI feature data"));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_ops::fa::coordinates::FaWindow;
    use crate::chat_ops::fa::origin::EngineId;
    use crate::chat_ops::fa::{FaWord, WordTiming, align_fa_response};
    use crate::chat_ops::{UtteranceIdx, WordIdx};

    use crate::api::{AudioPositionSeconds, DurationMs, NonNegativeSeconds};
    use crate::types::worker_v2::{
        ExecuteResponseV2, IndexedWordTimingResultV2, IndexedWordTimingV2, TaskResultV2,
        WhisperTokenTimingResultV2, WhisperTokenTimingV2, WorkerRequestIdV2,
    };

    use crate::chat_ops::fa::coordinates::{FileMs, Ms, Recording};

    /// A window starting at `start_ms`, inside a recording long enough that
    /// containment never fires in these adapter tests.
    fn window_at(start_ms: u64) -> (Recording, FaWindow) {
        let recording = Recording::of_duration(Ms(600_000)).expect("non-zero");
        let window = FaWindow::within(
            &recording,
            FileMs::new(start_ms),
            FileMs::new(start_ms + 60_000),
        )
        .expect("window inside the recording");
        (recording, window)
    }

    fn adapter_engine() -> EngineId {
        EngineId::new("test-fa")
    }

    #[test]
    fn invalid_generated_timing_stays_final_through_the_fa_boundary() {
        use crate::runner::util::classify_server_error;
        use crate::types::scheduling::{FailureCategory, NextAttempt, RetryPolicy, StopReason};
        use crate::worker::execute_result_v2::ModelLocality;

        let response = ExecuteResponseV2::failure(
            WorkerRequestIdV2::from("req-invalid-timing"),
            ProtocolErrorCodeV2::ModelOutputInvalid,
            "Qwen unit 2 starts before unit 1; not: targets length is too long for CTC".into(),
            NonNegativeSeconds::ZERO,
        );
        let Err(FaPayloadRefusal::Failed(failure)) = read_fa_payload(&response) else {
            panic!("the failed response must preserve its typed failure");
        };
        assert_eq!(failure.class(), FaFailureClass::FileFailure);
        let category = classify_server_error(
            &failure
                .into_report()
                .into_server_error(ModelLocality::Local),
        );
        assert_eq!(category, FailureCategory::ModelOutputInvalid);
        assert_eq!(
            RetryPolicy::default().after_failure(1, category),
            NextAttempt::Stop(StopReason::Final)
        );
    }

    /// The production path: the one read, then alignment of its payload.
    fn parse_forced_alignment_result_v2(
        response: &ExecuteResponseV2,
        words: &[FaWord],
        window: &FaWindow,
        engine: &EngineId,
    ) -> Result<Vec<Option<WordTiming>>, String> {
        let payload = read_fa_payload(response).map_err(|error| error.to_string())?;
        align_fa_response(payload, words, window, engine).map_err(|error| error.to_string())
    }

    /// Build `FaWord` values for small result-adapter tests.
    fn make_words(texts: &[&str]) -> Vec<FaWord> {
        texts
            .iter()
            .enumerate()
            .map(|(index, text)| FaWord {
                utterance_index: UtteranceIdx::new(0),
                utterance_word_index: WordIdx::new(index),
                text: (*text).into(),
            })
            .collect()
    }

    #[test]
    fn parses_whisper_token_result_into_established_alignment_domain() {
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("req-fa-v2-1"),
            TaskResultV2::WhisperTokenTimingResult(WhisperTokenTimingResultV2 {
                attention_capture: None,
                tokens: vec![
                    WhisperTokenTimingV2 {
                        text: "hello".into(),
                        time_s: AudioPositionSeconds::try_from(0.10).expect("fixture onset"),
                    },
                    WhisperTokenTimingV2 {
                        text: "world".into(),
                        time_s: AudioPositionSeconds::try_from(0.35).expect("fixture onset"),
                    },
                ],
            }),
            NonNegativeSeconds::try_from(0.01).expect("fixture elapsed"),
        );

        let timings = parse_forced_alignment_result_v2(
            &response,
            &make_words(&["hello", "world"]),
            &window_at(1_000).1,
            &adapter_engine(),
        )
        .expect("V2 whisper token result should parse");

        assert_eq!(timings.len(), 2);
        assert_eq!(timings[0].as_ref().expect("timing").start_ms, 1_100);
        assert_eq!(timings[1].as_ref().expect("timing").start_ms, 1_350);
    }

    /// Worker-wire boundary: an assumed end is bounded by the actual request,
    /// even though the recording has plenty of audio after that request.
    #[test]
    fn request_context_bounds_the_terminal_assumption() {
        let (recording, _) = window_at(2_000);
        let window = FaWindow::within(&recording, FileMs::new(2_000), FileMs::new(2_100))
            .expect("short request inside a longer recording");
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("request-context-bound"),
            TaskResultV2::WhisperTokenTimingResult(WhisperTokenTimingResultV2 {
                attention_capture: None,
                tokens: vec![WhisperTokenTimingV2 {
                    text: "finish".into(),
                    time_s: AudioPositionSeconds::try_from(0.04).expect("onset"),
                }],
            }),
            NonNegativeSeconds::try_from(0.01).expect("elapsed"),
        );
        let timings = parse_forced_alignment_result_v2(
            &response,
            &make_words(&["finish"]),
            &window,
            &adapter_engine(),
        )
        .expect("response");
        let timing = timings[0].as_ref().expect("positive bounded timing");
        assert_eq!((timing.start_ms, timing.end_ms), (2_040, 2_100));
        let origin =
            serde_json::to_value(crate::types::traces::OriginTrace::from(timing.end_origin()))
                .expect("origin wire");
        assert_eq!(origin["bound"], "request_end");
        assert_eq!(origin["was"]["kind"], "fallback_duration");
        assert_eq!(origin["overshoot_ms"], 440);
    }

    #[test]
    fn parses_indexed_timing_result_into_established_alignment_domain() {
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("req-fa-v2-2"),
            TaskResultV2::IndexedWordTimingResult(IndexedWordTimingResultV2 {
                indexed_timings: vec![
                    Some(IndexedWordTimingV2 {
                        start_ms: DurationMs(25),
                        end_ms: DurationMs(75),
                        confidence: Some(0.9),
                    }),
                    None,
                ],
            }),
            NonNegativeSeconds::try_from(0.01).expect("fixture elapsed"),
        );

        let timings = parse_forced_alignment_result_v2(
            &response,
            &make_words(&["hello", "world"]),
            &window_at(500).1,
            &adapter_engine(),
        )
        .expect("V2 indexed timing result should parse");

        let first = timings[0].as_ref().expect("timing");
        assert_eq!(first.start_ms, 525);
        assert_eq!(first.model_score().expect("score").millionths(), 900_000);
        assert!(timings[1].is_none());
    }

    #[test]
    fn rejects_non_fa_result_payloads() {
        let response = ExecuteResponseV2::success(
            WorkerRequestIdV2::from("req-fa-v2-3"),
            TaskResultV2::TranslationResult(crate::types::worker_v2::TranslationResultV2 {
                items: vec![crate::types::worker_v2::TranslationItemResultV2::BlankInput],
            }),
            NonNegativeSeconds::try_from(0.01).expect("fixture elapsed"),
        );

        let error = read_fa_payload(&response).expect_err("translation result should be rejected");

        assert!(matches!(
            error,
            FaPayloadRefusal::WrongResult("translation data")
        ));
    }
}
