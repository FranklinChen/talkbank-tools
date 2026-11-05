//! Rust-owned worker-protocol V2 forced-alignment executor control plane.
//!
//! **See also:** [INTERFACE_MAP.md](../../../INTERFACE_MAP.md) section "3. Forced Alignment V2" for:
//! - Python caller: `batchalign/worker/_fa_v2.py::execute_forced_alignment_request_v2()`
//! - Full Rust/Python responsibility split and input/output contracts.

use batchalign_types::api::{AudioPositionSeconds, DurationMs, ReportedEngineName};
use batchalign_types::worker_v2::{
    ExecuteRequestV2, FaBackendV2, FaPreparationBindingV2, FaPreparationOutcomeV2,
    FaPreparationResultV2, FaRequestMemoryV2, FaRequestPhaseV2, FaSampleCountV2, FaTextModeV2,
    ForcedAlignmentRequestV2, IndexedWordTimingResultV2, IndexedWordTimingV2, TaskRequestV2,
    TaskResultV2, WhisperAttentionCaptureReceiptV2, WhisperTokenTimingResultV2,
    WhisperTokenTimingV2,
};
use numpy::IntoPyArray;
use pyo3::prelude::*;

use crate::worker_artifacts::{
    MonoPreparedAudio, load_prepared_text_json_impl, require_mono_prepared_audio,
    require_prepared_text_attachment,
};
use crate::worker_execute::{
    ExecuteFailure, LoadedRunner, ValidatedRequestV2, admit_loaded_host, execute_request_v2,
    extract_task_payload, parse_host_attribute, parse_host_output,
};

/// Once this edge has refused a candidate, the executor can match only the
/// two legal operations. No later branch rechecks an unprepared wire state.
enum FaOperation<'a> {
    Prepare(&'a batchalign_types::worker_v2::WorkerRequestIdV2),
    Execute {
        plan: &'a batchalign_types::worker_v2::FaPreparationPlanV2,
        request_id: &'a batchalign_types::worker_v2::WorkerRequestIdV2,
    },
}

impl<'a> FaOperation<'a> {
    fn admit(
        phase: &'a FaRequestPhaseV2,
        request_id: &'a batchalign_types::worker_v2::WorkerRequestIdV2,
    ) -> Result<Self, ExecuteFailure> {
        match phase {
            FaRequestPhaseV2::Prepare {
                execution_request_id,
            } => Ok(Self::Prepare(execution_request_id)),
            FaRequestPhaseV2::Execute { plan } => Ok(Self::Execute { plan, request_id }),
            FaRequestPhaseV2::Candidate => Err(ExecuteFailure::InvalidPayload(
                "FA candidate requires preparation and host-memory admission".into(),
            )),
        }
    }

    fn execution_id(&self) -> &batchalign_types::worker_v2::WorkerRequestIdV2 {
        match self {
            Self::Prepare(request_id) | Self::Execute { request_id, .. } => request_id,
        }
    }
}

/// The part of the prepared FA payload a worker-side aligner reads: the
/// words, in order.
///
/// The payload also carries each word's id and utterance indices; those are
/// the control plane's, for placing the timings it gets back, and no aligner
/// reads them, so they are not parsed here. The Cantonese host used to be
/// handed the whole payload re-serialized, and read only the words.
#[derive(Debug, Clone, serde::Deserialize)]
struct PreparedFaPayloadV2 {
    words: Vec<String>,
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
    payload: PreparedFaPayloadV2,
    audio: MonoPreparedAudio<'_>,
    runner: LoadedRunner,
) -> Result<TaskResultV2, ExecuteFailure> {
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

/// Run a word-list aligner: Wave2Vec, Cantonese Wave2Vec or Qwen3.
///
/// All three take the requested words and answer one row per word, so they
/// share one call shape. The Cantonese host used to take the serialized
/// payload and the whole request, and read only the words; the shape is now
/// the one the other two always had, and the flavor enum that chose between
/// call shapes and "not loaded" messages is gone with it. The Qwen3 aligner
/// may leave a word untimed (it aligns on its own units and folds them back),
/// which is a row with no interval, never a padded one.
fn run_word_fa(
    py: Python<'_>,
    payload: PreparedFaPayloadV2,
    audio: MonoPreparedAudio<'_>,
    runner: LoadedRunner,
) -> Result<TaskResultV2, ExecuteFailure> {
    let audio_array = audio.samples()?.into_pyarray(py);
    let response = runner.call1(py, (audio_array, &payload.words))?;
    Ok(TaskResultV2::IndexedWordTimingResult(
        parse_indexed_timings(&response, payload.words.len())?,
    ))
}

fn run_fa(
    py: Python<'_>,
    request: ValidatedRequestV2<'_>,
    host: &Bound<'_, PyAny>,
) -> Result<TaskResultV2, ExecuteFailure> {
    let fa_request = extract_task_payload(
        &request,
        |payload| match payload {
            TaskRequestV2::ForcedAlignment(value) => Some(value),
            _ => None,
        },
        "forced-alignment",
    )?;
    let operation = FaOperation::admit(&fa_request.phase, &request.request_id)?;
    if fa_request.attention_capture.is_some() && fa_request.backend != FaBackendV2::Whisper {
        return Err(ExecuteFailure::InvalidPayload(
            "attention capture requires Whisper FA".into(),
        ));
    }
    let runner = admit_loaded_host(host, fa_request.backend, "forced-alignment")?;
    let payload = load_fa_payload(&request, fa_request)?;
    let audio = require_mono_prepared_audio(
        &request.attachments,
        fa_request.audio_ref_id.as_ref(),
        "forced-alignment V2",
    )?;
    let engine: ReportedEngineName = parse_host_attribute(host, "engine", "forced-alignment")?;
    let descriptor = audio.descriptor();
    let binding = FaPreparationBindingV2 {
        execution_request_id: operation.execution_id().clone(),
        backend: fa_request.backend,
        engine,
        payload_ref_id: fa_request.payload_ref_id.clone(),
        audio_ref_id: fa_request.audio_ref_id.clone(),
        samples: FaSampleCountV2::try_from(descriptor.frame_count.0)
            .map_err(|error| ExecuteFailure::InvalidPayload(error.to_string()))?,
        sample_rate_hz: descriptor.sample_rate_hz,
        text_mode: fa_request.text_mode,
        attention_capture: fa_request.attention_capture.clone(),
    };
    match operation {
        FaOperation::Prepare(_) => {
            let prepare = host.getattr("prepare").map_err(|error| {
                ExecuteFailure::Runtime(format!("loaded FA host has no preparer: {error}"))
            })?;
            let prepare = LoadedRunner::of_host(prepare.unbind());
            let text = join_fa_words(&payload.words, fa_request.text_mode);
            let binding_json = serde_json::to_string(&binding)
                .map_err(|error| ExecuteFailure::Runtime(error.to_string()))?;
            let response = prepare.call1(py, (binding_json, &payload.words, text))?;
            let outcome: FaPreparationOutcomeV2 = parse_host_output(&response, "FA preparation")?;
            if let FaPreparationOutcomeV2::Prepared { plan } = &outcome
                && (plan.binding != binding || matches!(plan.memory, FaRequestMemoryV2::NoModel))
            {
                return Err(ExecuteFailure::ModelOutputInvalid(
                    "FA preparer returned unbound or model-free evidence for a loaded model".into(),
                ));
            }
            return Ok(TaskResultV2::FaPreparation(FaPreparationResultV2 {
                outcome,
            }));
        }
        FaOperation::Execute { plan, .. } => {
            if plan.binding != binding || matches!(plan.memory, FaRequestMemoryV2::NoModel) {
                return Err(ExecuteFailure::InvalidPayload(
                    "FA execution plan does not bind the loaded model and prepared inputs".into(),
                ));
            }
        }
    }
    match fa_request.backend {
        FaBackendV2::Whisper => run_whisper(py, &request, fa_request, payload, audio, runner),
        FaBackendV2::Wave2vec | FaBackendV2::Wav2vecCanto | FaBackendV2::Qwen3 => {
            run_word_fa(py, payload, audio, runner)
        }
    }
}

/// Execute one FA request on `loaded`, the host this worker's FA load built
/// (`LoadedWhisperFa` or `LoadedWordFa` on the Python side): its `backend`
/// and `runner` are read from it.
#[pyfunction]
pub(crate) fn execute_forced_alignment_request_v2(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
    loaded: &Bound<'_, PyAny>,
) -> PyResult<String> {
    execute_request_v2(request, |request| run_fa(py, request, loaded))
}
