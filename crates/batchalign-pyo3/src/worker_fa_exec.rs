//! Rust-owned worker-protocol V2 forced-alignment executor control plane.
//!
//! **See also:** [INTERFACE_MAP.md](../../../INTERFACE_MAP.md) section "3. Forced Alignment V2" for:
//! - Python caller: `batchalign/worker/_fa_v2.py::execute_forced_alignment_request_v2()`
//! - Full Rust/Python responsibility split and input/output contracts.

use batchalign_types::api::{AudioPositionSeconds, DurationMs};
use batchalign_types::worker_v2::{
    ExecuteRequestV2, FaBackendV2, FaTextModeV2, ForcedAlignmentRequestV2,
    IndexedWordTimingResultV2, IndexedWordTimingV2, TaskRequestV2, TaskResultV2,
    WhisperAttentionCaptureReceiptV2, WhisperTokenTimingResultV2, WhisperTokenTimingV2,
};
use numpy::IntoPyArray;
use pyo3::prelude::*;

use crate::worker_artifacts::{
    load_prepared_text_json_impl, require_mono_prepared_audio, require_prepared_text_attachment,
};
use crate::worker_execute::{
    ExecuteFailure, ValidatedRequestV2, execute_request_v2, extract_task_payload,
    parse_host_output, require_runner,
};

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct PreparedFaPayloadV2 {
    words: Vec<String>,
    word_ids: Vec<String>,
    word_utterance_indices: Vec<i64>,
    word_utterance_word_indices: Vec<i64>,
}

fn load_fa_payload(
    request: &ExecuteRequestV2,
    fa_request: &ForcedAlignmentRequestV2,
) -> Result<PreparedFaPayloadV2, ExecuteFailure> {
    let attachment =
        require_prepared_text_attachment(&request.attachments, fa_request.payload_ref_id.as_ref())?;
    let raw = load_prepared_text_json_impl(attachment)?;
    serde_json::from_str(&raw).map_err(|error| ExecuteFailure::InvalidPayload(error.to_string()))
}

fn join_fa_words(words: &[String], text_mode: FaTextModeV2) -> String {
    match text_mode {
        FaTextModeV2::CharJoined => words.join("").replace('_', " ").trim().to_owned(),
        FaTextModeV2::SpaceJoined => words.join(" ").replace('_', " ").trim().to_owned(),
        // Byte-for-byte what the Python host used to do with `pauses=True`:
        // it took the space-joined text and re-spelled it one character per
        // token. Doing it here keeps every shaping decision in one function.
        FaTextModeV2::CharSpaced => words
            .join(" ")
            .replace('_', " ")
            .trim()
            .chars()
            .map(|character| character.to_string())
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn parse_whisper_tokens(
    response: &Bound<'_, PyAny>,
) -> Result<WhisperTokenTimingResultV2, ExecuteFailure> {
    let tokens: Vec<(String, f64)> = parse_host_output(response, "forced-alignment")?;

    let mut normalized = Vec::with_capacity(tokens.len());
    for (text, time_s) in tokens {
        // The position type's admission is the whole check: it refuses a
        // negative onset, and also the NaN and infinite ones the old `< 0.0`
        // comparison let through.
        let time_s = AudioPositionSeconds::try_from(time_s).map_err(|refused| {
            ExecuteFailure::Runtime(format!(
                "invalid forced-alignment host output: Whisper token time_s: {refused}"
            ))
        })?;
        normalized.push(WhisperTokenTimingV2 { text, time_s });
    }
    Ok(WhisperTokenTimingResultV2 {
        tokens: normalized,
        attention_capture: None,
    })
}

/// One row of an indexed-timing host's answer, as it arrives from Python.
///
/// A tuple struct because the host returns a sequence of three-element rows
/// (`Wave2VecWordAlignment` and `IndexedWordTimingRow` are both NamedTuples),
/// so this is the wire's own shape given a name and a place to say what the
/// slots mean.
///
/// The INTERVAL is optional because a host can time some of a group's words
/// and not others. The Qwen3 aligner segments the transcript into its own
/// units, folds those back onto the requested words, and reports no interval
/// for a word whose characters its tokenizer keeps none of; that lands in the
/// same `indexed_timings[i] = None` slot the wire has always had for an
/// unaligned word. It is an honest UNKNOWN, never a default. A host that times
/// everything (wave2vec, Cantonese) simply never emits one.
#[derive(serde::Deserialize)]
struct IndexedTimingRow(
    /// The word text, echoed by the host and unused here: the row's POSITION
    /// is what identifies the word, and the count is checked below.
    #[allow(dead_code)]
    String,
    /// The word's window-relative interval in milliseconds, or `None`.
    Option<(u64, u64)>,
    /// Optional model confidence, validated below.
    Option<f64>,
);

fn parse_indexed_timings(
    response: &Bound<'_, PyAny>,
    expected_words: usize,
) -> Result<IndexedWordTimingResultV2, ExecuteFailure> {
    let spans: Vec<IndexedTimingRow> = parse_host_output(response, "forced-alignment")?;

    // The host must answer with exactly one ROW per word it was asked about,
    // timed or not. A short answer used to be padded with `None` and a long
    // one silently truncated by `.take(expected_words)`, so a miscounted
    // response was indistinguishable from a partial alignment. Refuse it
    // instead: a row count is a different fact from a timing's presence, and
    // only the row count can put a timing on the wrong word.
    if spans.len() != expected_words {
        return Err(ExecuteFailure::Runtime(format!(
            "invalid forced-alignment host output: expected {expected_words} word rows, got {}",
            spans.len()
        )));
    }

    let mut indexed_timings = vec![None; expected_words];
    for (index, IndexedTimingRow(_, interval, confidence)) in spans.into_iter().enumerate() {
        let Some((start_ms, end_ms)) = interval else {
            // The host said this word has no timing. Leave the slot `None`,
            // which is what every other unaligned-word path already produces.
            continue;
        };
        if end_ms < start_ms {
            return Err(ExecuteFailure::Runtime(
                "invalid forced-alignment host output: Indexed word timing end_ms must be >= start_ms"
                    .to_owned(),
            ));
        }
        if confidence.is_some_and(|score| !score.is_finite() || !(0.0..=1.0).contains(&score)) {
            return Err(ExecuteFailure::Runtime(
                "invalid forced-alignment host output: model confidence must be finite and between 0 and 1"
                    .to_owned(),
            ));
        }
        indexed_timings[index] = Some(IndexedWordTimingV2 {
            start_ms: DurationMs(start_ms),
            end_ms: DurationMs(end_ms),
            confidence,
        });
    }
    Ok(IndexedWordTimingResultV2 { indexed_timings })
}

fn run_whisper(
    py: Python<'_>,
    request: &ExecuteRequestV2,
    fa_request: &ForcedAlignmentRequestV2,
    whisper_runner: Option<Py<PyAny>>,
) -> Result<TaskResultV2, ExecuteFailure> {
    let payload = load_fa_payload(request, fa_request)?;
    let audio = require_mono_prepared_audio(
        &request.attachments,
        fa_request.audio_ref_id.as_ref(),
        "forced-alignment V2",
    )?;
    let runner = require_runner(
        whisper_runner,
        "no whisper FA host loaded for worker protocol V2",
    )?;
    let audio_array = audio.samples()?.into_pyarray(py);
    let text = join_fa_words(&payload.words, fa_request.text_mode);
    let response = if let Some(capture) = &fa_request.attention_capture {
        #[derive(serde::Serialize)]
        struct CaptureContext<'a> {
            request_id: &'a batchalign_types::worker_v2::WorkerRequestIdV2,
            capture: &'a batchalign_types::worker_v2::WhisperAttentionCaptureRequestV2,
        }
        let context = serde_json::to_string(&CaptureContext {
            request_id: &request.request_id,
            capture,
        })
        .map_err(|error| ExecuteFailure::Runtime(error.to_string()))?;
        runner.call1(py, (audio_array, text.as_str(), context))?
    } else {
        runner.call1(py, (audio_array, text.as_str()))?
    };
    let mut result = parse_whisper_tokens(&response)?;
    if let Some(capture) = &fa_request.attention_capture {
        // The callback returns only timings. The bridge creates the receipt
        // from the closed artifact and the actual request, not host-supplied IDs.
        let destination = capture.destination().as_ref().to_owned();
        let digest = py
            .detach(move || {
                std::fs::read(destination).map(|bytes| blake3::hash(&bytes).to_hex().to_string())
            })
            .map_err(|error| {
                ExecuteFailure::Runtime(format!("attention capture missing: {error}"))
            })?;
        result.attention_capture = Some(WhisperAttentionCaptureReceiptV2 {
            capture: capture.clone(),
            request_id: request.request_id.clone(),
            blake3: digest,
        });
    }
    Ok(TaskResultV2::WhisperTokenTimingResult(result))
}

/// Which indexed-timing FA host a request targets.
///
/// Replaces a `canto_mode: bool` that travelled beside a separately supplied
/// unavailable-message: the same fact arrived twice and only the untyped copy
/// decided the call shape. The flavor now owns both decisions, so a message
/// cannot be paired with the wrong call shape.
///
/// Named for what the hosts have in common (one timing per requested word)
/// rather than for wave2vec, which stopped being true when the Qwen3 aligner
/// joined them.
#[derive(Clone, Copy)]
enum IndexedFaFlavor {
    /// The standard wave2vec host: called with the word list.
    Wave2vec,
    /// The Cantonese host: called with the serialized payload and request,
    /// because it re-derives its own tokenization.
    Cantonese,
    /// The Qwen3 aligner host: called with the word list, like wave2vec. It
    /// rebuilds the transcript itself, aligns on its OWN units (CJK characters
    /// individually), and folds those units back onto the requested words. A
    /// word it can time nothing of comes back with no interval rather than a
    /// padded one, so unlike wave2vec its rows can be untimed.
    Qwen3,
}

impl IndexedFaFlavor {
    fn unavailable_message(self) -> &'static str {
        match self {
            Self::Wave2vec => "no wave2vec FA host loaded for worker protocol V2",
            Self::Cantonese => "no Cantonese FA host loaded for worker protocol V2",
            Self::Qwen3 => "no Qwen3 FA host loaded for worker protocol V2",
        }
    }
}

fn run_indexed_fa(
    py: Python<'_>,
    request: &ExecuteRequestV2,
    fa_request: &ForcedAlignmentRequestV2,
    runner: Option<Py<PyAny>>,
    flavor: IndexedFaFlavor,
) -> Result<TaskResultV2, ExecuteFailure> {
    let payload = load_fa_payload(request, fa_request)?;
    let audio = require_mono_prepared_audio(
        &request.attachments,
        fa_request.audio_ref_id.as_ref(),
        "forced-alignment V2",
    )?;
    let runner = require_runner(runner, flavor.unavailable_message())?;
    let audio_array = audio.samples()?.into_pyarray(py);
    let response = match flavor {
        IndexedFaFlavor::Cantonese => {
            let payload_json = serde_json::to_string(&payload)
                .map_err(|error| ExecuteFailure::Runtime(error.to_string()))?;
            let request_json = serde_json::to_string(fa_request)
                .map_err(|error| ExecuteFailure::Runtime(error.to_string()))?;
            runner.call1(
                py,
                (audio_array, payload_json.as_str(), request_json.as_str()),
            )?
        }
        IndexedFaFlavor::Wave2vec | IndexedFaFlavor::Qwen3 => {
            runner.call1(py, (audio_array, payload.words.clone()))?
        }
    };
    Ok(TaskResultV2::IndexedWordTimingResult(
        parse_indexed_timings(&response, payload.words.len())?,
    ))
}

fn run_fa(
    py: Python<'_>,
    request: ValidatedRequestV2<'_>,
    whisper_runner: Option<Py<PyAny>>,
    wave2vec_runner: Option<Py<PyAny>>,
    canto_runner: Option<Py<PyAny>>,
    qwen3_runner: Option<Py<PyAny>>,
) -> Result<TaskResultV2, ExecuteFailure> {
    let fa_request = extract_task_payload(
        &request,
        |payload| match payload {
            TaskRequestV2::ForcedAlignment(value) => Some(value),
            _ => None,
        },
        "forced-alignment",
    )?;
    if fa_request.attention_capture.is_some() && fa_request.backend != FaBackendV2::Whisper {
        return Err(ExecuteFailure::InvalidPayload(
            "attention capture requires Whisper FA".into(),
        ));
    }
    match fa_request.backend {
        FaBackendV2::Whisper => run_whisper(py, &request, fa_request, whisper_runner),
        FaBackendV2::Wave2vec => run_indexed_fa(
            py,
            &request,
            fa_request,
            wave2vec_runner,
            IndexedFaFlavor::Wave2vec,
        ),
        FaBackendV2::Wav2vecCanto => run_indexed_fa(
            py,
            &request,
            fa_request,
            canto_runner,
            IndexedFaFlavor::Cantonese,
        ),
        FaBackendV2::Qwen3 => run_indexed_fa(
            py,
            &request,
            fa_request,
            qwen3_runner,
            IndexedFaFlavor::Qwen3,
        ),
    }
}

#[pyfunction]
#[pyo3(signature = (request, whisper_runner=None, wave2vec_runner=None, canto_runner=None, qwen3_runner=None))]
pub(crate) fn execute_forced_alignment_request_v2(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
    whisper_runner: Option<Py<PyAny>>,
    wave2vec_runner: Option<Py<PyAny>>,
    canto_runner: Option<Py<PyAny>>,
    qwen3_runner: Option<Py<PyAny>>,
) -> PyResult<String> {
    execute_request_v2(request, |request| {
        run_fa(
            py,
            request,
            whisper_runner,
            wave2vec_runner,
            canto_runner,
            qwen3_runner,
        )
    })
}
