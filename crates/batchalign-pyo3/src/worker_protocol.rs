//! Rust-owned request validation and op dispatch for the Python worker stdio loop.
//!
//! **See also:** [INTERFACE_MAP.md](../../../INTERFACE_MAP.md) section "1. Worker Protocol Dispatch" for:
//! - Python implementation: `batchalign/worker/_protocol.py` + `batchalign/worker/_handlers.py`
//! - Shared schema: `ipc-schema/worker_v2/`
//! - Full Rust/Python responsibility split.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

fn repr_text(value: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(value.repr()?.to_str()?.to_string())
}

/// Build an `{"op":"error",...}` envelope. When `request_id` is
/// supplied, it is included as a top-level field; the Rust reader loop
/// matches on `request_id` to fail the matching V2 dispatch's pending
/// oneshot directly, rather than routing the error to the sequential
/// control channel where a runtime V2 caller has no consumer registered.
fn error_payload<'py>(
    py: Python<'py>,
    message: &str,
    request_id: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let payload = PyDict::new(py);
    payload.set_item("op", "error")?;
    payload.set_item("error", message)?;
    if let Some(rid) = request_id {
        payload.set_item("request_id", rid)?;
    }
    Ok(payload.into_any())
}

/// Extract a `request_id` field from a request payload dict, if present
/// and string-typed.
fn extract_request_id(payload: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let Ok(dict) = payload.cast::<PyDict>() else {
        return Ok(None);
    };
    let Some(value) = dict.get_item("request_id")? else {
        return Ok(None);
    };
    Ok(value.extract::<String>().ok())
}

fn shutdown_payload<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let payload = PyDict::new(py);
    payload.set_item("op", "shutdown")?;
    Ok(payload.into_any())
}

fn response_payload<'py>(
    py: Python<'py>,
    op: &str,
    response_model: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("mode", "json")?;
    let payload = PyDict::new(py);
    payload.set_item("op", op)?;
    payload.set_item(
        "response",
        response_model.call_method("model_dump", (), Some(&kwargs))?,
    )?;
    Ok(payload.into_any())
}

fn validate_request_model<'py>(
    py: Python<'py>,
    op: &str,
    request_model: &Bound<'py, PyAny>,
    req_payload: &Bound<'py, PyAny>,
    validation_error_type: &Bound<'py, PyAny>,
) -> PyResult<Result<Bound<'py, PyAny>, Bound<'py, PyAny>>> {
    match request_model.call_method1("model_validate", (req_payload,)) {
        Ok(request) => Ok(Ok(request)),
        Err(error) => {
            if error.matches(py, validation_error_type)? {
                // Preserve any `request_id` on the rejected payload so the
                // Rust reader loop can fail the matching pending oneshot
                // immediately. Without this, a V2 dispatch sits on the
                // per-request timeout (default 180s for audio tasks) and
                // the operator sees a generic timeout instead of the
                // validation error.
                let request_id = extract_request_id(req_payload)?;
                let payload = error_payload(
                    py,
                    &format!("invalid {op} request: {error}"),
                    request_id.as_deref(),
                )?;
                Ok(Err(payload))
            } else {
                Err(error)
            }
        }
    }
}

/// Operations admitted to a worker thread. Shutdown has no executable variant.
#[derive(Clone, Copy)]
enum ExecutableOperation {
    Health,
    Capabilities,
    Infer,
    BatchInfer,
    ExecuteV2,
    EnsureTask,
}

impl ExecutableOperation {
    fn wire_name(self) -> &'static str {
        match self {
            Self::Health => "health",
            Self::Capabilities => "capabilities",
            Self::Infer => "infer",
            Self::BatchInfer => "batch_infer",
            Self::ExecuteV2 => "execute_v2",
            Self::EnsureTask => "ensure_task",
        }
    }
}

/// A request whose operation can execute on a worker thread.
/// No Python constructor exists; admission owns its immutable operation.
#[pyclass(frozen, module = "batchalign_core")]
pub(crate) struct PendingProtocolRequest {
    operation: ExecutableOperation,
    message: Py<PyDict>,
}

#[pymethods]
impl PendingProtocolRequest {
    /// Copy the envelope for exception correlation without exposing its mapping.
    #[getter]
    fn message(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        Ok(self.message.bind(py).copy()?.unbind())
    }
}

enum ImmediateReply {
    Rejected(Py<PyAny>),
    Shutdown,
}

/// A reader-thread reply. Its action and payload cannot disagree.
#[pyclass(frozen, module = "batchalign_core")]
pub(crate) struct ImmediateProtocolReply {
    reply: ImmediateReply,
}

#[pymethods]
impl ImmediateProtocolReply {
    #[getter]
    fn payload(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.reply {
            ImmediateReply::Rejected(payload) => Ok(payload.clone_ref(py)),
            ImmediateReply::Shutdown => Ok(shutdown_payload(py)?.unbind()),
        }
    }

    #[getter]
    fn should_shutdown(&self) -> bool {
        matches!(self.reply, ImmediateReply::Shutdown)
    }
}

fn rejected_message(py: Python<'_>, message: &str) -> PyResult<Py<PyAny>> {
    Ok(Py::new(
        py,
        ImmediateProtocolReply {
            reply: ImmediateReply::Rejected(error_payload(py, message, None)?.unbind()),
        },
    )?
    .into_any())
}

/// Classify control messages before the reader can block or enqueue work.
#[pyfunction]
pub(crate) fn prepare_protocol_message(
    py: Python<'_>,
    message: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let Ok(message) = message.cast::<PyDict>() else {
        return rejected_message(py, "request must be a JSON object");
    };
    let Some(op) = message.get_item("op")? else {
        return rejected_message(py, "unknown op: None");
    };
    let Ok(op_text) = op.cast::<PyString>() else {
        return rejected_message(py, &format!("unknown op: {}", repr_text(&op)?));
    };
    let Ok(op_text) = op_text.to_str() else {
        return rejected_message(py, &format!("unknown op: {}", repr_text(&op)?));
    };
    let operation = match op_text {
        "shutdown" => {
            return Ok(Py::new(
                py,
                ImmediateProtocolReply {
                    reply: ImmediateReply::Shutdown,
                },
            )?
            .into_any());
        }
        "health" => ExecutableOperation::Health,
        "capabilities" => ExecutableOperation::Capabilities,
        "infer" => ExecutableOperation::Infer,
        "batch_infer" => ExecutableOperation::BatchInfer,
        "execute_v2" => ExecutableOperation::ExecuteV2,
        "ensure_task" => ExecutableOperation::EnsureTask,
        _ => return rejected_message(py, &format!("unknown op: {}", repr_text(&op)?)),
    };
    Ok(Py::new(
        py,
        PendingProtocolRequest {
            operation,
            message: message.copy()?.unbind(),
        },
    )?
    .into_any())
}

/// Execute an admitted non-control request; raw messages cannot cross this boundary.
#[pyfunction]
#[pyo3(signature = (
    request,
    *,
    health_fn,
    capabilities_fn,
    infer_fn,
    batch_infer_fn,
    execute_v2_fn,
    ensure_task_fn,
    infer_request_model,
    batch_infer_request_model,
    execute_v2_request_model,
    validation_error_type,
))]
#[expect(
    clippy::too_many_arguments,
    reason = "thin PyO3 IPC dispatch boundary: threads the worker's Python \
              callback and request-model handles through to per-task handlers"
)]
pub(crate) fn dispatch_protocol_message(
    py: Python<'_>,
    request: &PendingProtocolRequest,
    health_fn: &Bound<'_, PyAny>,
    capabilities_fn: &Bound<'_, PyAny>,
    infer_fn: &Bound<'_, PyAny>,
    batch_infer_fn: &Bound<'_, PyAny>,
    execute_v2_fn: &Bound<'_, PyAny>,
    ensure_task_fn: &Bound<'_, PyAny>,
    infer_request_model: &Bound<'_, PyAny>,
    batch_infer_request_model: &Bound<'_, PyAny>,
    execute_v2_request_model: &Bound<'_, PyAny>,
    validation_error_type: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let message = request.message.bind(py);
    let op = request.operation.wire_name();

    let payload = match request.operation {
        ExecutableOperation::Health => response_payload(py, op, &health_fn.call0()?)?,
        ExecutableOperation::Capabilities => response_payload(py, op, &capabilities_fn.call0()?)?,
        ExecutableOperation::Infer => {
            let Some(req_payload) = message.get_item("request")? else {
                return Ok(error_payload(
                    py,
                    "infer request must include mapping field 'request'",
                    None,
                )?
                .unbind());
            };
            let req_payload = match req_payload.cast::<PyDict>() {
                Ok(payload) => payload.as_any(),
                Err(_) => {
                    return Ok(error_payload(
                        py,
                        "infer request must include mapping field 'request'",
                        None,
                    )?
                    .unbind());
                }
            };
            let request_model = match validate_request_model(
                py,
                op,
                infer_request_model,
                req_payload,
                validation_error_type,
            )? {
                Ok(request_model) => request_model,
                Err(payload) => return Ok(payload.unbind()),
            };
            response_payload(py, op, &infer_fn.call1((request_model,))?)?
        }
        ExecutableOperation::BatchInfer => {
            let Some(req_payload) = message.get_item("request")? else {
                return Ok(error_payload(
                    py,
                    "batch_infer request must include mapping field 'request'",
                    None,
                )?
                .unbind());
            };
            let req_payload = match req_payload.cast::<PyDict>() {
                Ok(payload) => payload.as_any(),
                Err(_) => {
                    return Ok(error_payload(
                        py,
                        "batch_infer request must include mapping field 'request'",
                        None,
                    )?
                    .unbind());
                }
            };
            let request_model = match validate_request_model(
                py,
                op,
                batch_infer_request_model,
                req_payload,
                validation_error_type,
            )? {
                Ok(request_model) => request_model,
                Err(payload) => return Ok(payload.unbind()),
            };
            response_payload(py, op, &batch_infer_fn.call1((request_model,))?)?
        }
        ExecutableOperation::ExecuteV2 => {
            let Some(req_payload) = message.get_item("request")? else {
                return Ok(error_payload(
                    py,
                    "execute_v2 request must include mapping field 'request'",
                    None,
                )?
                .unbind());
            };
            let req_payload = match req_payload.cast::<PyDict>() {
                Ok(payload) => payload.as_any(),
                Err(_) => {
                    return Ok(error_payload(
                        py,
                        "execute_v2 request must include mapping field 'request'",
                        None,
                    )?
                    .unbind());
                }
            };
            let request_model = match validate_request_model(
                py,
                op,
                execute_v2_request_model,
                req_payload,
                validation_error_type,
            )? {
                Ok(request_model) => request_model,
                Err(payload) => return Ok(payload.unbind()),
            };
            response_payload(py, op, &execute_v2_fn.call1((request_model,))?)?
        }
        ExecutableOperation::EnsureTask => {
            // ensure_task is a lightweight op: extract task + engine_overrides
            // from the request dict and call the Python handler directly.
            let Some(req_payload) = message.get_item("request")? else {
                return Ok(error_payload(
                    py,
                    "ensure_task request must include mapping field 'request'",
                    None,
                )?
                .unbind());
            };
            let req_dict = match req_payload.cast::<PyDict>() {
                Ok(d) => d,
                Err(_) => {
                    return Ok(
                        error_payload(py, "ensure_task request must be a mapping", None)?.unbind(),
                    );
                }
            };
            let task = match req_dict.get_item("task")? {
                Some(v) => v,
                None => {
                    return Ok(
                        error_payload(py, "ensure_task request must include 'task'", None)?
                            .unbind(),
                    );
                }
            };
            let engine_overrides = req_dict.get_item("engine_overrides")?;
            let result = match engine_overrides {
                Some(eo) => ensure_task_fn.call1((task, eo))?,
                None => ensure_task_fn.call1((task, py.None()))?,
            };
            response_payload(py, op, &result)?
        }
    };

    Ok(payload.unbind())
}
