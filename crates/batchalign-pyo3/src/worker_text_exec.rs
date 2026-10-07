//! Rust-owned worker-protocol V2 text-task executor control plane.
//!
//! The four batched text tasks (morphosyntax, utseg, translate, coref) were
//! the odd executors out: five audio/media executors were Rust control plane
//! over a Python model runner, while these were Python control plane over a
//! Python model runner (`_text_v2.py`), carrying a private copy of everything
//! `worker_execute` consolidates: response building with hand-threaded
//! elapsed times, four task extractors, and a six-arm exception ladder
//! written at four call sites. This module is that control plane, ported.
//!
//! Python keeps exactly one thing per task: a runner adapter that builds the
//! host's `BatchInferRequest` from the typed arguments, manages the progress
//! callback where the host reads one, and calls the loaded model. See
//! `batchalign/worker/_text_v2.py`.
//!
//! **See also:** [INTERFACE_MAP.md](../../../INTERFACE_MAP.md) section
//! "7. Text Task Result Normalization" for the shared result contracts.

use batchalign_types::worker::BatchInferResponse;
use batchalign_types::worker_v2::{ExecuteRequestV2, TaskRequestV2, TaskResultV2};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use crate::worker_artifacts::{ArtifactFailure, load_json_attachment_text};
use crate::worker_execute::{
    ExecuteFailure, ValidatedRequestV2, execute_request_v2, extract_task_payload,
    parse_host_output, require_runner,
};
use crate::worker_text_results::{
    TextResultShapeError, normalize_coref_result, normalize_morphosyntax_result,
    normalize_translation_result, normalize_utseg_result,
};

/// A prepared text batch, loaded and counted against the request's metadata.
///
/// The only constructor checks the item count, so a runner adapter cannot be
/// handed a batch whose length disagrees with what the request declared.
struct CountedBatch {
    /// The batch JSON text: for a prepared-text artifact these are the frozen
    /// bytes exactly as Rust wrote them, for an inline value one
    /// serialization of it. The Python adapter parses this into its own
    /// typed models (pydantic validation is the model-side boundary).
    json: String,
}

/// Counting probe: `IgnoredAny` items make this a length check, not a parse
/// of the batch's content, so a large batch is walked once without building
/// values.
#[derive(serde::Deserialize)]
struct BatchItemsProbe {
    items: Vec<serde::de::IgnoredAny>,
}

impl CountedBatch {
    fn load(
        request: &ExecuteRequestV2,
        payload_ref_id: &str,
        expected_count: u32,
        task: &str,
    ) -> Result<Self, ExecuteFailure> {
        let json = load_json_attachment_text(&request.attachments, payload_ref_id)?;
        let actual_count = match serde_json::from_str::<BatchItemsProbe>(&json) {
            Ok(probe) => probe.items.len(),
            Err(probe_error) => {
                // Two different defects fail the probe, and they carry
                // different wire codes (adjudicated 2026-08-21): a frozen
                // artifact that is not valid JSON is corrupt STORAGE
                // (`attachment_unreadable`), while valid JSON with no items
                // list is a malformed PAYLOAD (`invalid_payload`).
                return Err(match serde_json::from_str::<serde::de::IgnoredAny>(&json) {
                    Ok(_) => ExecuteFailure::InvalidPayload(format!(
                        "worker protocol V2 {task} payload must carry an items list"
                    )),
                    Err(_) => ArtifactFailure::Unreadable(format!(
                        "worker protocol V2 attachment {payload_ref_id:?} held \
                         invalid JSON: {probe_error}"
                    ))
                    .into(),
                });
            }
        };
        if actual_count != expected_count as usize {
            return Err(ExecuteFailure::InvalidPayload(format!(
                "worker protocol V2 {task} payload had {actual_count} items, \
                 expected {expected_count}"
            )));
        }
        Ok(Self { json })
    }
}

/// Classify an exception the runner adapter raised, preserving the taxonomy
/// the Python control plane applied: `NotImplementedError` means the model is
/// not loaded, `ValueError` (which pydantic's `ValidationError` subclasses)
/// means the payload or batch shape was wrong, anything else is a runtime
/// failure.
fn classify_runner_error(py: Python<'_>, error: PyErr) -> ExecuteFailure {
    // Not a silent default: both arms RENDER the same exception, one via its
    // __str__ and one via its Debug-ish display when __str__ itself raises.
    let message = match error.value(py).str() {
        Ok(text) => text.to_string(),
        Err(_) => error.to_string(),
    };
    if error.is_instance_of::<PyNotImplementedError>(py) {
        ExecuteFailure::ModelUnavailable(message)
    } else if error.is_instance_of::<PyValueError>(py) {
        ExecuteFailure::InvalidPayload(message)
    } else {
        ExecuteFailure::Runtime(message)
    }
}

/// Wrap a normalization refusal in the established per-task message prefix
/// the Python test matrix asserts on.
fn host_output_error(task: &str, error: TextResultShapeError) -> ExecuteFailure {
    let TextResultShapeError(message) = error;
    ExecuteFailure::Runtime(format!("invalid {task} host output: {message}"))
}

fn run_morphosyntax(
    py: Python<'_>,
    request: ValidatedRequestV2<'_>,
    runner: Option<Py<PyAny>>,
) -> Result<TaskResultV2, ExecuteFailure> {
    let task_request = extract_task_payload(
        &request,
        |payload| match payload {
            TaskRequestV2::Morphosyntax(value) => Some(value),
            _ => None,
        },
        "morphosyntax",
    )?;
    let batch = CountedBatch::load(
        &request,
        task_request.payload_ref_id.as_ref(),
        task_request.item_count,
        "morphosyntax",
    )?;
    let runner = require_runner(runner, "no morphosyntax host loaded for worker protocol V2")?;
    let response = runner
        .bind(py)
        .call1((
            request.request_id.as_ref(),
            task_request.lang.as_ref(),
            batch.json.as_str(),
            task_request.retokenize,
        ))
        .map_err(|error| classify_runner_error(py, error))?;
    let response: BatchInferResponse = parse_host_output(&response, "morphosyntax")?;
    Ok(TaskResultV2::MorphosyntaxResult(
        normalize_morphosyntax_result(&response, task_request.item_count as usize)
            .map_err(|error| host_output_error("morphosyntax", error))?,
    ))
}

fn run_utseg(
    py: Python<'_>,
    request: ValidatedRequestV2<'_>,
    runner: Option<Py<PyAny>>,
) -> Result<TaskResultV2, ExecuteFailure> {
    let task_request = extract_task_payload(
        &request,
        |payload| match payload {
            TaskRequestV2::Utseg(value) => Some(value),
            _ => None,
        },
        "utseg",
    )?;
    let batch = CountedBatch::load(
        &request,
        task_request.payload_ref_id.as_ref(),
        task_request.item_count,
        "utseg",
    )?;
    let runner = require_runner(runner, "no utseg host loaded for worker protocol V2")?;
    let response = runner
        .bind(py)
        .call1((
            task_request.lang.as_ref(),
            batch.json.as_str(),
            task_request.allow_stanza_fallback,
        ))
        .map_err(|error| classify_runner_error(py, error))?;
    let response: BatchInferResponse = parse_host_output(&response, "utseg")?;
    Ok(TaskResultV2::UtsegResult(
        normalize_utseg_result(&response, task_request.item_count as usize)
            .map_err(|error| host_output_error("utseg", error))?,
    ))
}

fn run_translate(
    py: Python<'_>,
    request: ValidatedRequestV2<'_>,
    runner: Option<Py<PyAny>>,
) -> Result<TaskResultV2, ExecuteFailure> {
    let task_request = extract_task_payload(
        &request,
        |payload| match payload {
            TaskRequestV2::Translate(value) => Some(value),
            _ => None,
        },
        "translate",
    )?;
    let batch = CountedBatch::load(
        &request,
        task_request.payload_ref_id.as_ref(),
        task_request.item_count,
        "translate",
    )?;
    let runner = require_runner(runner, "no translate host loaded for worker protocol V2")?;
    let response = runner
        .bind(py)
        .call1((
            task_request.source_lang.as_ref(),
            task_request.target_lang.as_ref(),
            batch.json.as_str(),
        ))
        .map_err(|error| classify_runner_error(py, error))?;
    let response: BatchInferResponse = parse_host_output(&response, "translate")?;
    Ok(TaskResultV2::TranslationResult(
        normalize_translation_result(&response, task_request.item_count as usize)
            .map_err(|error| host_output_error("translate", error))?,
    ))
}

fn run_coref(
    py: Python<'_>,
    request: ValidatedRequestV2<'_>,
    runner: Option<Py<PyAny>>,
) -> Result<TaskResultV2, ExecuteFailure> {
    let task_request = extract_task_payload(
        &request,
        |payload| match payload {
            TaskRequestV2::Coref(value) => Some(value),
            _ => None,
        },
        "coref",
    )?;
    let batch = CountedBatch::load(
        &request,
        task_request.payload_ref_id.as_ref(),
        task_request.item_count,
        "coref",
    )?;
    let runner = require_runner(runner, "no coref host loaded for worker protocol V2")?;
    let response = runner
        .bind(py)
        .call1((task_request.lang.as_ref(), batch.json.as_str()))
        .map_err(|error| classify_runner_error(py, error))?;
    let response: BatchInferResponse = parse_host_output(&response, "coref")?;
    Ok(TaskResultV2::CorefResult(
        normalize_coref_result(&response, task_request.item_count as usize)
            .map_err(|error| host_output_error("coref", error))?,
    ))
}

#[pyfunction]
#[pyo3(signature = (request, runner=None))]
pub(crate) fn execute_morphosyntax_request_v2(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
    runner: Option<Py<PyAny>>,
) -> PyResult<String> {
    execute_request_v2(request, |request| run_morphosyntax(py, request, runner))
}

#[pyfunction]
#[pyo3(signature = (request, runner=None))]
pub(crate) fn execute_utseg_request_v2(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
    runner: Option<Py<PyAny>>,
) -> PyResult<String> {
    execute_request_v2(request, |request| run_utseg(py, request, runner))
}

#[pyfunction]
#[pyo3(signature = (request, runner=None))]
pub(crate) fn execute_translate_request_v2(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
    runner: Option<Py<PyAny>>,
) -> PyResult<String> {
    execute_request_v2(request, |request| run_translate(py, request, runner))
}

#[pyfunction]
#[pyo3(signature = (request, runner=None))]
pub(crate) fn execute_coref_request_v2(
    py: Python<'_>,
    request: &Bound<'_, PyAny>,
    runner: Option<Py<PyAny>>,
) -> PyResult<String> {
    execute_request_v2(request, |request| run_coref(py, request, runner))
}
