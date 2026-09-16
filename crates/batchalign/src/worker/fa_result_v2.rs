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

use crate::chat_ops::fa::coordinates::FaWindow;
use crate::chat_ops::fa::origin::EngineId;
use crate::chat_ops::fa::{FaWord, WordTiming, parse_fa_response};
use crate::chat_ops::nlp::{FaIndexedTiming, FaRawResponse, FaRawToken};
use crate::worker::execute_result_v2::require_success_result;

use crate::types::worker_v2::{ExecuteResponseV2, TaskResultV2};

/// Parse one staged V2 FA execute response into the established FA timing
/// domain.
pub fn parse_forced_alignment_result_v2(
    response: &ExecuteResponseV2,
    original_words: &[FaWord],
    window: &FaWindow,
    engine: &EngineId,
) -> Result<Vec<Option<WordTiming>>, String> {
    let result = require_success_result(response, "forced-alignment")?;

    let raw_response = match result {
        TaskResultV2::WhisperTokenTimingResult(result) => FaRawResponse::TokenLevel {
            tokens: result
                .tokens
                .iter()
                .map(|token| FaRawToken {
                    text: token.text.clone(),
                    time_s: token.time_s.0,
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
            return Err(
                "worker protocol V2 forced-alignment response returned morphosyntax data".into(),
            );
        }
        TaskResultV2::UtsegResult(_) => {
            return Err(
                "worker protocol V2 forced-alignment response returned utterance-segmentation data"
                    .into(),
            );
        }
        TaskResultV2::WhisperChunkResult(_) => {
            return Err(
                "worker protocol V2 forced-alignment response returned ASR chunk data".into(),
            );
        }
        TaskResultV2::TranslationResult(_) => {
            return Err(
                "worker protocol V2 forced-alignment response returned translation data".into(),
            );
        }
        TaskResultV2::CorefResult(_) => {
            return Err(
                "worker protocol V2 forced-alignment response returned coreference data".into(),
            );
        }
        TaskResultV2::MonologueAsrResult(_) => {
            return Err(
                "worker protocol V2 forced-alignment response returned monologue ASR data".into(),
            );
        }
        TaskResultV2::SpeakerResult(_) => {
            return Err(
                "worker protocol V2 forced-alignment response returned speaker diarization data"
                    .into(),
            );
        }
        TaskResultV2::SpeakerEmbeddingResult(_) => {
            return Err(
                "worker protocol V2 forced-alignment response returned speaker embedding data"
                    .into(),
            );
        }
        TaskResultV2::OpensmileResult(_) => {
            return Err(
                "worker protocol V2 forced-alignment response returned openSMILE feature data"
                    .into(),
            );
        }
        TaskResultV2::AvqiResult(_) => {
            return Err(
                "worker protocol V2 forced-alignment response returned AVQI feature data".into(),
            );
        }
    };

    let raw_json = serde_json::to_string(&raw_response).map_err(|error| {
        format!("failed to serialize staged FA V2 response for parsing: {error}")
    })?;
    // Wave 5 consolidation: parse_fa_response returns a typed
    // FaAlignmentError. The surrounding V2-adapter layer still returns
    // `Result<_, String>`; convert at this boundary so the adapter
    // contract stays stable. A future follow-up can propagate the
    // typed error further up.
    parse_fa_response(&raw_json, original_words, window, engine).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_ops::fa::FaWord;
    use crate::chat_ops::{UtteranceIdx, WordIdx};

    use crate::api::{DurationMs, DurationSeconds};
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
                tokens: vec![
                    WhisperTokenTimingV2 {
                        text: "hello".into(),
                        time_s: DurationSeconds(0.10),
                    },
                    WhisperTokenTimingV2 {
                        text: "world".into(),
                        time_s: DurationSeconds(0.35),
                    },
                ],
            }),
            DurationSeconds(0.01),
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
            DurationSeconds(0.01),
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
            DurationSeconds(0.01),
        );

        let error = parse_forced_alignment_result_v2(
            &response,
            &make_words(&["hello"]),
            &window_at(0).1,
            &adapter_engine(),
        )
        .expect_err("translation result should be rejected");

        assert!(error.contains("translation data"));
    }
}
