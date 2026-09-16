//! Rust helpers for Hong Kong ASR provider projection.
//!
//! **See also:** [INTERFACE_MAP.md](../../../INTERFACE_MAP.md) section "9. HK/Cantonese ASR Bridges" for:
//! - Python callers: `batchalign/inference/languages/cantonese/`, `batchalign/worker/_asr_v2.py`
//! - Design: Projects provider-specific output into common MonologueAsrResultV2 shapes.
//!
//! The HK/Tencent/FunAudio Python adapters must still talk to Python-only SDKs
//! and model objects, but the shared result shaping should live in Rust. This
//! module owns:
//!
//! - FunASR entry points (segment admission lives in the
//!   [`funasr_projection`] child module, which builds these private shapes)
//! - Tencent result-detail projection into monologues and timed words
//! - Aliyun sentence-result projection and its per-character fallback
//!   tokenization, used when Aliyun sends a sentence with no per-word timing
//!
//! Python code now forwards raw provider output into these helpers instead of
//! reimplementing the projection loops itself.
//!
//! # What this module deliberately does NOT do
//!
//! It does not normalize Cantonese. Until 2026-09-16 it did, per token for
//! Tencent and Aliyun and per joined segment for FunASR, which made
//! normalization depend on which engine ran (Qwen and Whisper never reached
//! this module) and put a second owner on a transformation that is not
//! idempotent. Surfaces now leave here exactly as the provider wrote them, and
//! `batchalign_transform::asr_postprocess::AlignedNormalization` normalizes each
//! monologue once, in the server.
//!
//! # Absent provider fields are states, not zeros
//!
//! Tencent and Aliyun document every result field as nullable, and their SDKs
//! initialize each attribute to `None`. Until 2026-09-16 this module read them
//! with `.ok().and_then(...).unwrap_or(0)`, so "the provider did not send a
//! time" and "the provider said zero" arrived as the same number. Zero is a
//! legal time, so a word with no timing claimed to have been spoken at the
//! start of its segment, and a segment with no `StartMs` claimed to start at
//! the beginning of the recording. Those fabrications reached CHAT.
//!
//! Every field is now read through [`provider_admission::FieldRead`], which has
//! three states and no default, and every time is admitted by
//! [`batchalign_transform::asr_postprocess::AdmittedInterval`], the one owner of
//! rounding and range admission. An absent time produces
//! [`WordTiming::Untimed`] with a named cause; a value of the wrong type, or a
//! time that is not admissible, refuses the file with a message naming the
//! provider, the position and the fault.

use crate::error::BatchalignBoundaryError;
use crate::py_json_bridge::py_to_json_value;
use batchalign_transform::asr_postprocess::cantonese as cantonese_ops;
use batchalign_transform::asr_postprocess::{AdmittedInterval, UntimedCause, WordTiming};
use funasr_projection::{FunasrSegmentWire, project_funasr_segments};
use provider_admission::{
    FieldRead, ProviderAdmissionError, ProviderFault, ProviderId, ProviderLocus,
};
use pyo3::prelude::*;
use pyo3::types::PyList;
use serde::{Deserialize, Serialize};

mod funasr_projection;
mod provider_admission;

/// Speaker attribution as this projection can express it.
///
/// Serialized in the SAME tagged shape as the worker wire's
/// `SpeakerAttributionV2`, so the Python provider payload carries the
/// distinction rather than flattening it: `{"kind": "attributed", "label": ..}`
/// or `{"kind": "undiarized"}`.
///
/// Until 2026-09-16 this rendered as a bare track number, because the wire had
/// no spelling for "this provider separates nobody" and every such monologue
/// had to write `0`. Downstream that became a `PAR0` tier indistinguishable
/// from a provider's real first speaker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ProviderSpeakerAttribution {
    /// The provider named this speaker; the value is its own label.
    Attributed {
        /// The provider's own label, as it spelled it.
        label: String,
    },
    /// The provider named no speaker. Either it does not separate speakers at
    /// all (Aliyun, FunASR), or it sent no `SpeakerId` for this segment.
    Undiarized,
}

impl ProviderSpeakerAttribution {
    /// Attribute a segment to a provider's numeric speaker id.
    fn numeric(speaker: usize) -> Self {
        Self::Attributed {
            label: speaker.to_string(),
        }
    }
}

/// Speaker-attributed ASR projection shared by the HK provider bridges.
#[derive(Debug, Clone, Serialize, PartialEq)]
struct HkAsrProjection {
    /// Speaker monologues ready for the shared ASR worker contract.
    monologues: Vec<HkAsrMonologue>,
    /// Flat timed words for UTR-style timing injection paths.
    timed_words: Vec<HkTimedWord>,
}

/// One speaker monologue in the HK ASR projection.
#[derive(Debug, Clone, Serialize, PartialEq)]
struct HkAsrMonologue {
    /// Who the provider attributed this span to, if anyone.
    speaker: ProviderSpeakerAttribution,
    /// Ordered ASR elements in this speaker span.
    elements: Vec<HkAsrElement>,
}

/// One token entry in a projected HK ASR monologue.
#[derive(Debug, Clone, Serialize, PartialEq)]
struct HkAsrElement {
    /// Token kind for the shared worker contract.
    #[serde(rename = "type")]
    element_type: &'static str,
    /// Start time in seconds when known. `None` means the provider reported
    /// none, never "zero".
    ts: Option<f64>,
    /// End time in seconds when known.
    end_ts: Option<f64>,
    /// Surface token value after provider-boundary normalization.
    value: String,
}

impl HkAsrElement {
    /// Build one text element from an admitted timing.
    fn text(value: String, timing: WordTiming) -> Self {
        let (ts, end_ts) = match timing.interval() {
            Some(interval) => {
                let (start_s, end_s) = interval.as_seconds();
                (Some(start_s), Some(end_s))
            }
            None => (None, None),
        };
        Self {
            element_type: "text",
            ts,
            end_ts,
            value,
        }
    }
}

/// One timed word emitted by a provider projection.
#[derive(Debug, Clone, Serialize, PartialEq)]
struct HkTimedWord {
    /// Surface token value.
    word: String,
    /// Start time in milliseconds.
    start_ms: i64,
    /// End time in milliseconds.
    end_ms: i64,
}

/// Language behaviour of every HK provider projection, decided once at the
/// Python boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderLanguage {
    /// `yue`: Cantonese normalization (s2hk plus the domain replacement table).
    Cantonese,
    /// Every other language: provider tokens are used as they are.
    Other,
}

impl ProviderLanguage {
    /// Decide the projection behaviour from a batchalign3 language code.
    fn from_code(code: &str) -> Self {
        match code {
            "yue" => Self::Cantonese,
            _ => Self::Other,
        }
    }
}

/// One word record from Tencent `ResultDetail`, after admission.
///
/// The segment's own start is GONE by this point: it was consumed making each
/// word's timing absolute, so no later stage can add it a second time or forget
/// to add it at all.
#[derive(Debug, Clone, PartialEq)]
struct TencentWordInput {
    /// Surface word returned by Tencent.
    word: String,
    /// The word's absolute timing, or a named absence.
    timing: WordTiming,
}

/// One segment record from Tencent `ResultDetail`, after admission.
#[derive(Debug, Clone, PartialEq)]
struct TencentSegmentInput {
    /// Who Tencent attributed the segment to.
    speaker: ProviderSpeakerAttribution,
    /// Word entries inside the segment.
    words: Vec<TencentWordInput>,
}

/// One word record from an Aliyun websocket sentence result.
#[derive(Debug, Clone, Deserialize, PartialEq)]
struct AliyunWordInput {
    /// Surface token returned by Aliyun when per-word timing is available.
    #[serde(default)]
    text: String,
    /// Start time in milliseconds, when the payload carried one.
    #[serde(default, rename = "startTime")]
    start_time_ms: Option<i64>,
    /// End time in milliseconds, when the payload carried one.
    #[serde(default, rename = "endTime")]
    end_time_ms: Option<i64>,
}

/// One sentence result emitted by the Aliyun websocket transport.
#[derive(Debug, Clone, Deserialize, PartialEq)]
struct AliyunSentenceInput {
    /// Per-word timing entries when Aliyun emits them.
    #[serde(default)]
    words: Vec<AliyunWordInput>,
    /// Sentence text fallback used when no per-word entries are present.
    #[serde(default)]
    sentence_text: String,
}

/// Tokenize one sentence-only provider fallback according to the provider
/// language.
///
/// Cantonese is split one character per token, which is the granularity every
/// Cantonese engine reports and the granularity the rest of the pipeline
/// expects. The split does not normalize; that happens once per monologue in
/// the server.
fn tokenize_sentence_fallback(text: &str, language: ProviderLanguage) -> Vec<String> {
    match language {
        ProviderLanguage::Cantonese => cantonese_ops::cantonese_char_tokens(text),
        ProviderLanguage::Other => text.split_whitespace().map(ToOwned::to_owned).collect(),
    }
}

/// Collect one admitted word into a monologue's elements and the flat timed-word
/// list, which only ever receives spans that locate real audio.
fn push_admitted_word(
    value: String,
    timing: WordTiming,
    elements: &mut Vec<HkAsrElement>,
    timed_words: &mut Vec<HkTimedWord>,
) {
    if let Some(interval) = timing.interval() {
        timed_words.push(HkTimedWord {
            word: value.clone(),
            start_ms: interval.start_ms(),
            end_ms: interval.end_ms(),
        });
    }
    elements.push(HkAsrElement::text(value, timing));
}

/// Project admitted Tencent segments into the shared HK ASR projection.
///
/// Tencent reports one word per entry, so there is nothing to tokenize and no
/// language behaviour left on this path.
fn project_tencent_segments(segments: Vec<TencentSegmentInput>) -> HkAsrProjection {
    let mut monologues = Vec::new();
    let mut timed_words = Vec::new();

    for segment in segments {
        let mut elements = Vec::new();

        for word in segment.words {
            push_admitted_word(word.word, word.timing, &mut elements, &mut timed_words);
        }

        if !elements.is_empty() {
            monologues.push(HkAsrMonologue {
                speaker: segment.speaker,
                elements,
            });
        }
    }

    timed_words.sort_by_key(|item| item.start_ms);
    HkAsrProjection {
        monologues,
        timed_words,
    }
}

/// Project Aliyun sentence results into the shared HK ASR projection.
///
/// Fallible, because a time Aliyun sends that is not admissible (non-finite
/// through the JSON bridge, negative, inverted, or beyond the admitted range)
/// is refused by name rather than rounded into something plausible.
fn project_aliyun_sentences(
    sentences: Vec<AliyunSentenceInput>,
    language: ProviderLanguage,
) -> Result<HkAsrProjection, ProviderAdmissionError> {
    let mut monologues = Vec::new();
    let mut timed_words = Vec::new();

    for (sentence_index, sentence) in sentences.into_iter().enumerate() {
        let mut elements = Vec::new();

        if sentence.words.is_empty() {
            // Sentence-only fallback: Aliyun sent no per-word entries, so there
            // is no timing to attribute and every token is untimed.
            for token in tokenize_sentence_fallback(&sentence.sentence_text, language) {
                elements.push(HkAsrElement::text(
                    token,
                    WordTiming::Untimed(UntimedCause::ProviderReportedNoTiming),
                ));
            }
        } else {
            for (word_index, word) in sentence.words.into_iter().enumerate() {
                let raw = word.text.trim();
                if raw.is_empty() {
                    continue;
                }
                let locus = ProviderLocus::SentenceWord {
                    sentence: sentence_index,
                    word: word_index,
                };
                let timing = WordTiming::from_millis(word.start_time_ms, word.end_time_ms)
                    .map_err(|refusal| {
                        ProviderAdmissionError::new(
                            ProviderId::Aliyun,
                            locus,
                            ProviderFault::Interval {
                                field: "startTime/endTime",
                                refusal,
                            },
                        )
                    })?;
                push_admitted_word(raw.to_owned(), timing, &mut elements, &mut timed_words);
            }
        }

        if !elements.is_empty() {
            monologues.push(HkAsrMonologue {
                // Aliyun NLS performs no speaker separation at all, so there is
                // no label to attribute and none is invented.
                speaker: ProviderSpeakerAttribution::Undiarized,
                elements,
            });
        }
    }

    timed_words.sort_by_key(|item| item.start_ms);
    Ok(HkAsrProjection {
        monologues,
        timed_words,
    })
}

/// Admit one Tencent segment's words against the segment's own start.
///
/// The segment start is read ONCE and consumed here. When Tencent sent none,
/// every word in the segment is untimed with that cause: the offsets are
/// relative, so without a base they locate nothing, and treating the base as
/// zero is what used to place a whole segment at the start of the recording.
fn admit_tencent_words(
    segment: &Bound<'_, PyAny>,
    words: Vec<Bound<'_, PyAny>>,
    segment_index: usize,
) -> Result<Vec<TencentWordInput>, ProviderAdmissionError> {
    let segment_locus = ProviderLocus::Segment {
        index: segment_index,
    };
    let segment_start_ms = FieldRead::<i64>::attr(segment, "StartMs").optional(
        ProviderId::Tencent,
        segment_locus,
        "StartMs",
        "an integer number of milliseconds",
    )?;

    let mut admitted = Vec::with_capacity(words.len());
    for (word_index, word) in words.iter().enumerate() {
        let locus = ProviderLocus::SegmentWord {
            segment: segment_index,
            word: word_index,
        };
        // Refused rather than defaulted: an empty surface used to be produced
        // by `unwrap_or_default()` and then dropped by the blank filter, so a
        // word Tencent failed to send disappeared without trace.
        let text = FieldRead::<String>::attr(word, "Word").required(
            ProviderId::Tencent,
            locus,
            "Word",
            "a string",
        )?;
        let trimmed = text.trim();
        if trimmed.is_empty() {
            continue;
        }

        let offset_start_ms = FieldRead::<i64>::attr(word, "OffsetStartMs").optional(
            ProviderId::Tencent,
            locus,
            "OffsetStartMs",
            "an integer number of milliseconds",
        )?;
        let offset_end_ms = FieldRead::<i64>::attr(word, "OffsetEndMs").optional(
            ProviderId::Tencent,
            locus,
            "OffsetEndMs",
            "an integer number of milliseconds",
        )?;

        let timing = match (segment_start_ms, offset_start_ms, offset_end_ms) {
            (None, _, _) => WordTiming::Untimed(UntimedCause::SegmentStartAbsent),
            (Some(_), None, None) => WordTiming::Untimed(UntimedCause::ProviderReportedNoTiming),
            (Some(_), None, Some(_)) => WordTiming::Untimed(UntimedCause::ProviderReportedNoStart),
            (Some(_), Some(_), None) => WordTiming::Untimed(UntimedCause::ProviderReportedNoEnd),
            (Some(start_ms), Some(offset_start), Some(offset_end)) => {
                let interval =
                    AdmittedInterval::admit_offset_from(start_ms, offset_start, offset_end)
                        .map_err(|refusal| {
                            ProviderAdmissionError::new(
                                ProviderId::Tencent,
                                locus,
                                ProviderFault::Interval {
                                    field: "StartMs + OffsetStartMs/OffsetEndMs",
                                    refusal,
                                },
                            )
                        })?;
                WordTiming::from_admitted(interval)
            }
        };

        admitted.push(TencentWordInput {
            word: trimmed.to_owned(),
            timing,
        });
    }
    Ok(admitted)
}

/// Convert a Python list of Tencent result-detail objects into admitted Rust
/// data.
fn extract_tencent_segments(
    result_detail: &Bound<'_, PyAny>,
) -> Result<Vec<TencentSegmentInput>, ProviderAdmissionError> {
    let Ok(list) = result_detail.cast::<PyList>() else {
        return Err(ProviderAdmissionError::new(
            ProviderId::Tencent,
            ProviderLocus::Segment { index: 0 },
            ProviderFault::FieldWrongType {
                field: "ResultDetail",
                expected: "a list of segments",
            },
        ));
    };
    let mut segments = Vec::with_capacity(list.len());

    for (index, segment) in list.iter().enumerate() {
        let locus = ProviderLocus::Segment { index };
        // Read the words FIRST. A segment carrying none contributes nothing, so
        // its other fields are not required to be present: that is the shape a
        // Tencent response uses for a segment it recognized nothing in, and
        // demanding a start or a speaker for it would fail a file over a
        // segment that says nothing.
        let words = FieldRead::<Vec<Bound<'_, PyAny>>>::attr(&segment, "Words").optional(
            ProviderId::Tencent,
            locus,
            "Words",
            "a list of word objects",
        )?;
        let Some(words) = words.filter(|words| !words.is_empty()) else {
            continue;
        };

        let speaker = FieldRead::<usize>::attr(&segment, "SpeakerId")
            .optional(
                ProviderId::Tencent,
                locus,
                "SpeakerId",
                "a non-negative integer",
            )?
            .map_or(
                ProviderSpeakerAttribution::Undiarized,
                ProviderSpeakerAttribution::numeric,
            );

        let words = admit_tencent_words(&segment, words, index)?;
        if words.is_empty() {
            continue;
        }
        segments.push(TencentSegmentInput { speaker, words });
    }

    Ok(segments)
}

/// Project raw FunASR output into monologues and timed words.
///
/// Python passes the model output after only shallow parsing; Rust admits each
/// segment in [`funasr_projection`] and projects it into the shared shape.
#[pyfunction]
pub(crate) fn funaudio_segments_to_asr(
    py: Python<'_>,
    segments: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let value = py_to_json_value(segments)?;

    py.detach(move || {
        let segments = match value {
            serde_json::Value::Array(_) => {
                serde_json::from_value::<Vec<FunasrSegmentWire>>(value).map_err(|error| {
                    BatchalignBoundaryError::internal(error.to_string()).into_py_err()
                })?
            }
            serde_json::Value::Object(_) => {
                vec![
                    serde_json::from_value::<FunasrSegmentWire>(value).map_err(|error| {
                        BatchalignBoundaryError::internal(error.to_string()).into_py_err()
                    })?,
                ]
            }
            _ => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "FunASR output must be a dict or list of dicts",
                ));
            }
        };

        let projection = project_funasr_segments(segments)
            .map_err(|error| BatchalignBoundaryError::internal(error).into_py_err())?;
        serde_json::to_string(&projection)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    })
}

/// Project Tencent `ResultDetail` objects into monologues and timed words.
///
/// Python keeps only the transport/SDK responsibilities. Rust owns the field
/// admission, the timing math, and projection into the shared worker shape.
#[pyfunction]
pub(crate) fn tencent_result_detail_to_asr(
    py: Python<'_>,
    result_detail: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let segments =
        extract_tencent_segments(result_detail).map_err(ProviderAdmissionError::into_py_err)?;

    py.detach(move || {
        serde_json::to_string(&project_tencent_segments(segments))
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    })
}

/// Project Aliyun sentence results into monologues and timed words.
///
/// Python keeps only websocket transport, credential handling, and shallow
/// payload parsing. Rust owns the sentence fallback tokenization plus the
/// shared monologue/timed-word projection shape.
#[pyfunction]
pub(crate) fn aliyun_sentences_to_asr(
    py: Python<'_>,
    sentences: &Bound<'_, PyAny>,
    lang: &str,
) -> PyResult<String> {
    let value = py_to_json_value(sentences)?;
    let language = ProviderLanguage::from_code(lang);

    py.detach(move || {
        let sentences = match value {
            serde_json::Value::Array(_) => {
                serde_json::from_value::<Vec<AliyunSentenceInput>>(value).map_err(|error| {
                    BatchalignBoundaryError::internal(error.to_string()).into_py_err()
                })?
            }
            serde_json::Value::Object(_) => {
                vec![
                    serde_json::from_value::<AliyunSentenceInput>(value).map_err(|error| {
                        BatchalignBoundaryError::internal(error.to_string()).into_py_err()
                    })?,
                ]
            }
            _ => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Aliyun sentences must be a dict or list of dicts",
                ));
            }
        };

        let projection = project_aliyun_sentences(sentences, language)
            .map_err(ProviderAdmissionError::into_py_err)?;
        serde_json::to_string(&projection)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an admitted Tencent word whose timing came from a segment start
    /// plus offsets, the way `admit_tencent_words` produces one.
    fn tencent_word(word: &str, segment_start_ms: i64, offsets: (i64, i64)) -> TencentWordInput {
        let interval =
            AdmittedInterval::admit_offset_from(segment_start_ms, offsets.0, offsets.1)
                .expect("test offsets are admissible");
        TencentWordInput {
            word: word.to_owned(),
            timing: WordTiming::from_admitted(interval),
        }
    }

    /// Timed words are ordered by start time, and every surface is the
    /// provider's own. `系` and `呀` would once have arrived as `係` and `啊`;
    /// normalizing them here made the result depend on which engine ran.
    #[test]
    fn tencent_projection_sorts_timed_words_and_keeps_provider_surfaces() {
        let projection = project_tencent_segments(vec![
            TencentSegmentInput {
                speaker: ProviderSpeakerAttribution::numeric(2),
                words: vec![
                    tencent_word("系", 1000, (0, 200)),
                    tencent_word("你", 1000, (300, 500)),
                ],
            },
            TencentSegmentInput {
                speaker: ProviderSpeakerAttribution::numeric(1),
                words: vec![tencent_word("呀", 500, (0, 100))],
            },
        ]);

        assert_eq!(projection.monologues.len(), 2);
        assert_eq!(projection.monologues[0].elements[0].value, "系");
        assert_eq!(projection.monologues[1].elements[0].value, "呀");
        assert_eq!(
            projection
                .timed_words
                .iter()
                .map(|item| (item.word.as_str(), item.start_ms, item.end_ms))
                .collect::<Vec<_>>(),
            vec![("呀", 500, 600), ("系", 1000, 1200), ("你", 1300, 1500)]
        );
    }

    /// A word the provider did not time carries NO time, and contributes no
    /// timed word. It used to carry its segment's start as both bounds.
    #[test]
    fn an_untimed_tencent_word_reaches_chat_without_a_time() {
        let projection = project_tencent_segments(vec![TencentSegmentInput {
            speaker: ProviderSpeakerAttribution::numeric(1),
            words: vec![
                TencentWordInput {
                    word: "好".to_owned(),
                    timing: WordTiming::Untimed(UntimedCause::SegmentStartAbsent),
                },
                tencent_word("嗎", 4850, (0, 200)),
            ],
        }]);

        let elements = &projection.monologues[0].elements;
        assert_eq!(elements[0].ts, None);
        assert_eq!(elements[0].end_ts, None);
        assert_eq!(elements[1].ts, Some(4.85));
        assert_eq!(
            projection
                .timed_words
                .iter()
                .map(|word| word.word.as_str())
                .collect::<Vec<_>>(),
            vec!["嗎"],
            "an untimed word must not become a timed one"
        );
    }

    #[test]
    fn aliyun_projection_handles_sentence_fallback_and_timed_words()
    -> Result<(), ProviderAdmissionError> {
        let projection = project_aliyun_sentences(
            vec![
                AliyunSentenceInput {
                    words: vec![AliyunWordInput {
                        text: "系".to_string(),
                        start_time_ms: Some(100),
                        end_time_ms: Some(250),
                    }],
                    sentence_text: "系".to_string(),
                },
                AliyunSentenceInput {
                    words: Vec::new(),
                    sentence_text: "真系呀，".to_string(),
                },
            ],
            ProviderLanguage::Cantonese,
        )?;

        assert_eq!(projection.monologues.len(), 2);
        assert_eq!(projection.monologues[0].elements[0].value, "系");
        assert_eq!(
            projection.monologues[1]
                .elements
                .iter()
                .map(|element| element.value.as_str())
                .collect::<Vec<_>>(),
            vec!["真", "系", "呀"],
            "the fallback splits per character and normalizes nothing"
        );
        assert_eq!(
            projection
                .timed_words
                .iter()
                .map(|item| (item.word.as_str(), item.start_ms, item.end_ms))
                .collect::<Vec<_>>(),
            vec![("系", 100, 250)]
        );
        Ok(())
    }

    /// The Aliyun half of the same fabrication: a word with one bound missing
    /// is untimed, not a span starting at zero.
    #[test]
    fn an_aliyun_word_missing_a_bound_is_untimed_not_zero_based()
    -> Result<(), ProviderAdmissionError> {
        let projection = project_aliyun_sentences(
            vec![AliyunSentenceInput {
                words: vec![
                    AliyunWordInput {
                        text: "你".to_string(),
                        start_time_ms: None,
                        end_time_ms: Some(200),
                    },
                    AliyunWordInput {
                        text: "好".to_string(),
                        start_time_ms: Some(300),
                        end_time_ms: None,
                    },
                ],
                sentence_text: "你好".to_string(),
            }],
            ProviderLanguage::Other,
        )?;

        let elements = &projection.monologues[0].elements;
        assert!(elements.iter().all(|element| element.ts.is_none()));
        assert!(elements.iter().all(|element| element.end_ts.is_none()));
        assert!(projection.timed_words.is_empty());
        Ok(())
    }

    /// An inadmissible time refuses the file, naming the provider and the
    /// position, instead of being rounded into a plausible span.
    #[test]
    fn an_inverted_aliyun_span_is_refused_by_name() {
        let refusal = project_aliyun_sentences(
            vec![AliyunSentenceInput {
                words: vec![AliyunWordInput {
                    text: "你".to_string(),
                    start_time_ms: Some(900),
                    end_time_ms: Some(100),
                }],
                sentence_text: "你".to_string(),
            }],
            ProviderLanguage::Other,
        )
        .expect_err("an inverted span must be refused");

        assert_eq!(refusal.provider, ProviderId::Aliyun);
        assert_eq!(
            refusal.locus,
            ProviderLocus::SentenceWord {
                sentence: 0,
                word: 0
            }
        );
        let message = refusal.to_string();
        assert!(message.contains("before its start"), "{message}");
    }

    /// An engine that separates nobody says SO, rather than writing a track
    /// number a reader cannot tell from a real first speaker.
    #[test]
    fn an_undiarized_monologue_says_so_instead_of_claiming_track_zero() {
        let projection = HkAsrProjection {
            monologues: vec![HkAsrMonologue {
                speaker: ProviderSpeakerAttribution::Undiarized,
                elements: vec![HkAsrElement::text(
                    "好".to_owned(),
                    WordTiming::Untimed(UntimedCause::ProviderReportedNoTiming),
                )],
            }],
            timed_words: Vec::new(),
        };
        let json = serde_json::to_value(&projection).expect("projection serializes");
        assert_eq!(json["monologues"][0]["speaker"]["kind"], "undiarized");
        assert!(
            json["monologues"][0]["speaker"].get("label").is_none(),
            "an undiarized monologue has no label to carry"
        );
        assert_eq!(json["monologues"][0]["elements"][0]["ts"], serde_json::Value::Null);
    }

    /// A provider that DOES name speakers carries its own label through.
    #[test]
    fn an_attributed_monologue_carries_the_providers_own_label() {
        let projection = project_tencent_segments(vec![TencentSegmentInput {
            speaker: ProviderSpeakerAttribution::numeric(2),
            words: vec![tencent_word("好", 0, (0, 100))],
        }]);
        let json = serde_json::to_value(&projection).expect("projection serializes");
        assert_eq!(json["monologues"][0]["speaker"]["kind"], "attributed");
        assert_eq!(json["monologues"][0]["speaker"]["label"], "2");
    }
}
