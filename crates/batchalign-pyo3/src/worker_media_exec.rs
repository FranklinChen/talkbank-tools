//! Rust-owned worker-protocol V2 prepared-audio executor control plane.
//!
//! **See also:** [INTERFACE_MAP.md](../../../INTERFACE_MAP.md) for:
//! - Section "4. Media Analysis V2: OpenSMILE" → `batchalign/worker/_opensmile_v2.py`
//! - Section "5. Media Analysis V2: AVQI" → `batchalign/worker/_avqi_v2.py`
//! - Section "6. Media Analysis V2: Speaker Diarization" → `batchalign/worker/_speaker_v2.py`

use std::collections::BTreeSet;

use batchalign_types::worker_v2::{
    AvqiResultV2, OpenSmileResultV2, SpeakerBackendV2, SpeakerEmbeddingBackendV2,
    SpeakerEmbeddingRequestV2, SpeakerEmbeddingResultV2, SpeakerInferenceEvidenceV2,
    SpeakerInputV2, SpeakerResultV2, TaskRequestV2, TaskResultV2,
};
use numpy::IntoPyArray;
use pyo3::prelude::*;

use crate::worker_artifacts::require_mono_prepared_audio;
use crate::worker_execute::{
    ExecuteFailure, ValidatedRequestV2, classify_runner_error, execute_request_v2,
    extract_task_payload, parse_host_output, require_runner,
};

fn parse_opensmile_result(
    response: &Bound<'_, PyAny>,
) -> Result<OpenSmileResultV2, ExecuteFailure> {
    parse_host_output(response, "openSMILE")
}

fn parse_avqi_result(response: &Bound<'_, PyAny>) -> Result<AvqiResultV2, ExecuteFailure> {
    parse_host_output(response, "AVQI")
}

fn parse_speaker_result(
    response: &Bound<'_, PyAny>,
    expected_backend: SpeakerBackendV2,
) -> Result<SpeakerResultV2, ExecuteFailure> {
    let parsed: SpeakerResultV2 = parse_host_output(response, "speaker")?;
    // Each segment's bounds are an `AdmittedInterval`, refused at parse when
    // inverted, so only the evidence's backend is checked here.
    match (&parsed.evidence, expected_backend) {
        (SpeakerInferenceEvidenceV2::PyannoteAi { .. }, SpeakerBackendV2::PyannoteAi)
        | (SpeakerInferenceEvidenceV2::Pyannote { .. }, SpeakerBackendV2::Pyannote)
        | (SpeakerInferenceEvidenceV2::Nemo { .. }, SpeakerBackendV2::Nemo) => Ok(parsed),
        _ => Err(ExecuteFailure::Runtime(format!(
            "speaker host evidence does not match requested backend {expected_backend:?}"
        ))),
    }
}

fn run_opensmile(
    py: Python<'_>,
    request: ValidatedRequestV2<'_>,
    prepared_audio_runner: Option<Py<PyAny>>,
) -> Result<TaskResultV2, ExecuteFailure> {
    let opensmile_request = extract_task_payload(
        &request,
        |payload| match payload {
            TaskRequestV2::Opensmile(value) => Some(value),
            _ => None,
        },
        "openSMILE",
    )?;
    let audio = require_mono_prepared_audio(
        &request.attachments,
        opensmile_request.audio_ref_id.as_ref(),
        "openSMILE V2",
    )?;
    let runner = require_runner(
        prepared_audio_runner,
        "no openSMILE host loaded for worker protocol V2",
    )?;
    let audio_array = audio.samples()?.into_pyarray(py);
    let descriptor = audio.descriptor();
    let response = runner
        .bind(py)
        .call1((
            audio_array,
            descriptor.sample_rate_hz.0,
            opensmile_request.feature_set.as_str(),
            opensmile_request.feature_level.as_str(),
            descriptor.path.as_ref(),
        ))
        .map_err(|error| ExecuteFailure::Runtime(error.to_string()))?;
    Ok(TaskResultV2::OpensmileResult(parse_opensmile_result(
        &response,
    )?))
}

fn run_avqi(
    py: Python<'_>,
    request: ValidatedRequestV2<'_>,
    prepared_audio_runner: Option<Py<PyAny>>,
) -> Result<TaskResultV2, ExecuteFailure> {
    let avqi_request = extract_task_payload(
        &request,
        |payload| match payload {
            TaskRequestV2::Avqi(value) => Some(value),
            _ => None,
        },
        "AVQI",
    )?;
    let cs_audio = require_mono_prepared_audio(
        &request.attachments,
        avqi_request.cs_audio_ref_id.as_ref(),
        "AVQI V2",
    )?;
    let sv_audio = require_mono_prepared_audio(
        &request.attachments,
        avqi_request.sv_audio_ref_id.as_ref(),
        "AVQI V2",
    )?;
    let runner = require_runner(
        prepared_audio_runner,
        "no AVQI host loaded for worker protocol V2",
    )?;
    let cs_audio_array = cs_audio.samples()?.into_pyarray(py);
    let sv_audio_array = sv_audio.samples()?.into_pyarray(py);
    let cs_descriptor = cs_audio.descriptor();
    let sv_descriptor = sv_audio.descriptor();
    let response = runner
        .bind(py)
        .call1((
            cs_audio_array,
            cs_descriptor.sample_rate_hz.0,
            sv_audio_array,
            sv_descriptor.sample_rate_hz.0,
            cs_descriptor.path.as_ref(),
            sv_descriptor.path.as_ref(),
        ))
        .map_err(|error| ExecuteFailure::Runtime(error.to_string()))?;
    Ok(TaskResultV2::AvqiResult(parse_avqi_result(&response)?))
}

fn run_speaker(
    py: Python<'_>,
    request: ValidatedRequestV2<'_>,
    pyannote_ai_prepared_audio_runner: Option<Py<PyAny>>,
    pyannote_prepared_audio_runner: Option<Py<PyAny>>,
    nemo_prepared_audio_runner: Option<Py<PyAny>>,
) -> Result<TaskResultV2, ExecuteFailure> {
    let speaker_request = extract_task_payload(
        &request,
        |payload| match payload {
            TaskRequestV2::Speaker(value) => Some(value),
            _ => None,
        },
        "speaker",
    )?;
    let audio_ref_id = match &speaker_request.input {
        SpeakerInputV2::PreparedAudio(value) => value.audio_ref_id.as_ref(),
    };
    let audio = require_mono_prepared_audio(
        &request.attachments,
        audio_ref_id,
        "worker protocol V2 speaker",
    )?;
    // Carried as an Option, never collapsed to a number here.
    //
    // `None` is a REAL STATE with a documented meaning: the CLI records it as
    // "auto-detect" (see `cli/args/mod.rs`, which says so in as many words),
    // the protocol skips the field entirely when it is absent, and pyannote
    // estimates the speaker count when handed `num_speakers=None`.
    //
    // This used to be `.unwrap_or(2)`, which erased that state at the FFI: a
    // request that deliberately said "I do not know" arrived at the diarizer
    // indistinguishable from one that asked for exactly two speakers. Since
    // `diarize` hardcodes the pyannote backend, every `batchalign3 diarize`
    // without `--num-speakers` was silently forced to two speakers instead of
    // estimating, and no reader of the result could tell.
    let expected_speakers = speaker_request.expected_speakers.map(|value| value.0);
    let runner = match speaker_request.backend {
        SpeakerBackendV2::PyannoteAi => require_runner(
            pyannote_ai_prepared_audio_runner,
            "no pyannoteAI speaker host loaded for prepared-audio V2",
        )?,
        SpeakerBackendV2::Pyannote => require_runner(
            pyannote_prepared_audio_runner,
            "no pyannote speaker host loaded for prepared-audio V2",
        )?,
        SpeakerBackendV2::Nemo => require_runner(
            nemo_prepared_audio_runner,
            "no NeMo speaker host loaded for prepared-audio V2",
        )?,
    };
    let audio_array = audio.samples()?.into_pyarray(py);
    let response = runner
        .bind(py)
        .call1((
            audio_array,
            audio.descriptor().sample_rate_hz.0,
            expected_speakers,
        ))
        .map_err(|error| classify_runner_error(py, error))?;
    Ok(TaskResultV2::SpeakerResult(parse_speaker_result(
        &response,
        speaker_request.backend,
    )?))
}

/// Check that the host answered about EXACTLY the spans it was asked about.
///
/// The request and the response are two sequences of the same length carrying
/// the same ids, which is the shape that misattributes one speaker's acoustic
/// evidence to another utterance when the two drift apart. An arity check
/// alone does not catch a reordering or a substitution, so the ids are
/// compared as sets and any disagreement is refused rather than reconciled.
fn require_answers_for_exactly_the_requested_spans(
    request: &SpeakerEmbeddingRequestV2,
    result: &SpeakerEmbeddingResultV2,
) -> Result<(), ExecuteFailure> {
    let asked: BTreeSet<&str> = request
        .spans
        .iter()
        .map(|span| span.span_id.as_ref())
        .collect();
    let answered: BTreeSet<&str> = result
        .spans
        .iter()
        .map(|span| span.span_id.as_ref())
        .collect();
    if asked != answered {
        return Err(ExecuteFailure::Runtime(format!(
            "speaker embedding host answered about {} span(s) and was asked about {}",
            answered.len(),
            asked.len()
        )));
    }
    if answered.len() != result.spans.len() {
        return Err(ExecuteFailure::Runtime(
            "speaker embedding host answered about one span more than once".to_owned(),
        ));
    }
    Ok(())
}

fn run_speaker_embedding(
    py: Python<'_>,
    request: ValidatedRequestV2<'_>,
    pyannote_span_runner: Option<Py<PyAny>>,
) -> Result<TaskResultV2, ExecuteFailure> {
    let embedding_request = extract_task_payload(
        &request,
        |payload| match payload {
            TaskRequestV2::SpeakerEmbedding(value) => Some(value),
            _ => None,
        },
        "speaker embedding",
    )?;
    let audio = require_mono_prepared_audio(
        &request.attachments,
        embedding_request.audio_ref_id.as_ref(),
        "worker protocol V2 speaker embedding",
    )?;
    let runner = match embedding_request.backend {
        SpeakerEmbeddingBackendV2::Pyannote => require_runner(
            pyannote_span_runner,
            "no pyannote speaker-embedding host loaded for prepared-audio V2",
        )?,
    };

    // Preserve the V2 request schema across the Rust/Python boundary. Passing
    // JSON lets Python validate named fields instead of rebuilding typed
    // requests from positional tuples.
    let spans_json = serde_json::to_string(&embedding_request.spans)
        .map_err(|error| ExecuteFailure::Runtime(error.to_string()))?;

    let audio_array = audio.samples()?.into_pyarray(py);
    let response = runner
        .bind(py)
        .call1((audio_array, audio.descriptor().sample_rate_hz.0, spans_json))
        // The same reclassification the diarization executor uses: a gated
        // repository or a missing credential must reach the operator as
        // `model_access_denied`, not as an anonymous runtime failure.
        .map_err(|error| classify_runner_error(py, error))?;

    let parsed: SpeakerEmbeddingResultV2 = parse_host_output(&response, "speaker embedding")?;
    require_answers_for_exactly_the_requested_spans(embedding_request, &parsed)?;
    Ok(TaskResultV2::SpeakerEmbeddingResult(parsed))
}

#[pyfunction]
#[pyo3(signature = (request, prepared_audio_runner=None))]
pub(crate) fn execute_opensmile_request_v2(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
    prepared_audio_runner: Option<Py<PyAny>>,
) -> PyResult<String> {
    execute_request_v2(request, |request| {
        run_opensmile(py, request, prepared_audio_runner)
    })
}

#[pyfunction]
#[pyo3(signature = (request, prepared_audio_runner=None))]
pub(crate) fn execute_avqi_request_v2(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
    prepared_audio_runner: Option<Py<PyAny>>,
) -> PyResult<String> {
    execute_request_v2(request, |request| {
        run_avqi(py, request, prepared_audio_runner)
    })
}

#[pyfunction]
#[pyo3(signature = (request, pyannote_ai_prepared_audio_runner=None, pyannote_prepared_audio_runner=None, nemo_prepared_audio_runner=None))]
pub(crate) fn execute_speaker_request_v2(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
    pyannote_ai_prepared_audio_runner: Option<Py<PyAny>>,
    pyannote_prepared_audio_runner: Option<Py<PyAny>>,
    nemo_prepared_audio_runner: Option<Py<PyAny>>,
) -> PyResult<String> {
    execute_request_v2(request, |request| {
        run_speaker(
            py,
            request,
            pyannote_ai_prepared_audio_runner,
            pyannote_prepared_audio_runner,
            nemo_prepared_audio_runner,
        )
    })
}

#[pyfunction]
#[pyo3(signature = (request, pyannote_span_runner=None))]
pub(crate) fn execute_speaker_embedding_request_v2(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
    pyannote_span_runner: Option<Py<PyAny>>,
) -> PyResult<String> {
    execute_request_v2(request, |request| {
        run_speaker_embedding(py, request, pyannote_span_runner)
    })
}
