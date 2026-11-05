//! Wire-level IPC protocol types and constants for Python worker communication.
//!
//! Defines the JSON-lines request/response envelopes, ready signals, and
//! diagnostic dump utilities. These types are internal to the worker handle
//! machinery: callers use the higher-level [`WorkerHandle`](super::WorkerHandle)
//! methods.

use crate::types::worker_v2::{
    ExecuteRequestV2, ExecuteResponseV2, ProgressEventV2, WorkerErrorKind,
};
use crate::worker::{
    BatchInferRequest, BatchInferResponse, InferRequest, InferResponse, InferTask,
    WorkerCapabilities, WorkerHealthResponse, WorkerPid,
};
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::worker::error::WorkerError;

/// Maximum chars of startup stderr to include in error messages.
pub(super) const STARTUP_STDERR_TAIL_CHARS: usize = 2_000;

/// Maximum non-JSON preamble lines to tolerate before the ready signal.
pub(super) const MAX_READY_STDOUT_PREAMBLE_LINES: usize = 32;

/// Consecutive non-protocol lines a reader tolerates before it stops
/// trusting the stream.
pub(crate) const MAX_RESPONSE_STDOUT_NOISE_LINES: usize = 8;

/// How much of a refused line a log or error message carries.
const LINE_EXCERPT_CHARS: usize = 200;

/// The start of a line, for a log or an error message.
pub(crate) fn excerpt(line: &str) -> String {
    match line.char_indices().nth(LINE_EXCERPT_CHARS) {
        Some((cut, _)) => format!("{}...", &line[..cut]),
        None => line.to_owned(),
    }
}

/// One line read from a worker's protocol stream, by the one rule every
/// reader (sequential stdio, sequential TCP, shared GPU) applies: a JSON
/// object is a protocol message, a blank line is nothing, and anything else
/// (text, a JSON scalar or array) is noise.
pub(crate) enum WireLine {
    /// Whitespace only.
    Blank,
    /// A JSON object: a message the reader decodes, or refuses as a
    /// protocol violation.
    Message(serde_json::Value),
    /// Not a protocol message.
    Noise,
}

impl WireLine {
    /// Classify one line.
    pub(crate) fn classify(line: &str) -> Self {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Self::Blank;
        }
        match serde_json::from_str::<serde_json::Value>(trimmed) {
            Ok(value @ serde_json::Value::Object(_)) => Self::Message(value),
            Ok(
                serde_json::Value::Null
                | serde_json::Value::Bool(_)
                | serde_json::Value::Number(_)
                | serde_json::Value::String(_)
                | serde_json::Value::Array(_),
            )
            | Err(_) => Self::Noise,
        }
    }
}

/// A run of consecutive noise lines on one stream, ended by a message.
#[derive(Default)]
pub(crate) struct NoiseRun {
    consecutive: usize,
}

/// A stream that reached [`MAX_RESPONSE_STDOUT_NOISE_LINES`] consecutive
/// noise lines: it is not trusted further.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NoiseLimit {
    /// The last noise line, as an [`excerpt`].
    pub(crate) last_line: String,
}

impl NoiseLimit {
    /// The error a waiter on such a stream receives: retryable, since the
    /// worker is retired and the request can run on another.
    pub(crate) fn into_worker_error(self) -> WorkerError {
        WorkerError::OutputNoise {
            last_line: self.last_line,
        }
    }
}

impl NoiseRun {
    /// A message ends the run.
    pub(crate) fn message(&mut self) {
        self.consecutive = 0;
    }

    /// Count one noise line; the limit once the run reaches it.
    pub(crate) fn noise(&mut self, line: &str) -> Result<(), NoiseLimit> {
        self.consecutive += 1;
        let last_line = excerpt(line.trim());
        warn!(
            line = %last_line,
            consecutive = self.consecutive,
            "worker: ignoring a line that is not a protocol message"
        );
        if self.consecutive >= MAX_RESPONSE_STDOUT_NOISE_LINES {
            return Err(NoiseLimit { last_line });
        }
        Ok(())
    }
}

/// Maximum bytes of response/error text to include in a failure dump.
const FAILED_REQUEST_DUMP_MAX_RESPONSE_BYTES: usize = 1_024 * 1_024;

/// Ready signal emitted by the Python worker on stdout, after its startup
/// progress: a stdio worker's and a TCP daemon's alike, read by the one
/// reader (`WorkerHandle::read_ready_line`).
#[derive(Debug, Deserialize)]
pub(super) struct ReadySignal {
    pub ready: bool,
    pub pid: u32,
    /// How the worker serves from now on, and where.
    #[serde(flatten)]
    pub transport: ReadyTransport,
    pub runtime: crate::worker::runtime_identity::WorkerRuntimeIdentity,
}

/// The transport a ready worker serves on, named by its ready line.
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(tag = "transport", rename_all = "snake_case")]
pub(super) enum ReadyTransport {
    /// Requests on the worker's stdin, replies on the stdout just read.
    Stdio,
    /// A daemon listening on `host:port`; its stdout ends at the ready line.
    Tcp {
        /// The address it listens on.
        host: String,
        /// The port it listens on.
        port: u16,
    },
}

/// Wire-level request envelope sent to a Python worker, over stdio or TCP.
#[derive(Debug, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub(crate) enum WorkerRequest<'a> {
    Infer { request: &'a InferRequest },
    BatchInfer { request: &'a BatchInferRequest },
    ExecuteV2 { request: &'a ExecuteRequestV2 },
    EnsureTask { request: EnsureTaskRequest<'a> },
    Health,
    Capabilities { request: CapabilitiesRequest<'a> },
    Shutdown,
}

/// The id a control request (`capabilities`, `ensure_task`) carries.
///
/// The worker tags a failure line with the `request_id` of the request that
/// failed, so carrying one lets a reader tell which control op a failure
/// answers. The shared GPU reader depends on it: a control op that timed out
/// can still be answered late, and that late failure must not answer the
/// next op. Minted only by [`ControlRequestId::next`], so no two control
/// requests of one server share an id, and the `control-` prefix keeps the
/// ids apart from V2 dispatch ids.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub(crate) struct ControlRequestId(String);

impl ControlRequestId {
    /// A fresh id, unique within this server process.
    pub(crate) fn next() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self(format!("control-{n}"))
    }

    /// Whether a line's raw `request_id` names this request.
    pub(crate) fn names(&self, raw: &str) -> bool {
        self.0 == raw
    }
}

impl std::fmt::Display for ControlRequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Request payload for the `capabilities` IPC operation: only its id.
#[derive(Debug, Serialize)]
pub(crate) struct CapabilitiesRequest<'a> {
    pub(crate) request_id: &'a ControlRequestId,
}

/// Request payload for the `ensure_task` IPC operation.
#[derive(Debug, Serialize)]
pub(crate) struct EnsureTaskRequest<'a> {
    pub(crate) request_id: &'a ControlRequestId,
    pub(crate) task: InferTask,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) engine_overrides: Option<&'a std::collections::BTreeMap<String, String>>,
}

/// Wire-level response envelope read from a Python worker, over stdio or TCP.
///
/// The `progress_v2` variant carries intermediate progress events emitted by
/// long-running V2 tasks.  Workers emit zero or more progress lines before the
/// final `execute_v2` response.  See `execute_v2_with_progress` for the
/// multiplexed read loop.
#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub(crate) enum WorkerResponse {
    Infer {
        response: InferResponse,
    },
    BatchInfer {
        response: BatchInferResponse,
    },
    ExecuteV2 {
        response: ExecuteResponseV2,
    },
    ProgressV2 {
        event: ProgressEventV2,
    },
    EnsureTask {
        response: crate::worker::EnsureTaskResponse,
    },
    Health {
        response: WorkerHealthResponse,
    },
    Capabilities {
        response: WorkerCapabilities,
    },
    Shutdown,
    Error(ReportedFailure),
}

/// A failure the worker reported in an `{"op":"error"}` line: its own
/// diagnosis and the required [`WorkerErrorKind`].
///
/// Every reader, sequential or shared GPU, turns one into a [`WorkerError`]
/// through the methods here, so the same line means the same error on every
/// transport and for every op.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub(crate) struct ReportedFailure {
    /// The worker's own diagnosis.
    #[serde(rename = "error")]
    pub(crate) message: String,
    /// What a retry could change.
    pub(crate) kind: WorkerErrorKind,
}

impl ReportedFailure {
    /// The failure as the error of a request the worker had admitted (infer,
    /// batch_infer, execute_v2, capabilities).
    pub(crate) fn into_worker_error(self) -> WorkerError {
        let worker = self.kind.worker_after();
        match self.kind {
            WorkerErrorKind::Runtime => WorkerError::WorkerResponse(self.message),
            WorkerErrorKind::Bootstrap => WorkerError::Bootstrap {
                message: self.message,
                worker,
            },
            WorkerErrorKind::Download => WorkerError::ModelDownload {
                message: self.message,
                worker,
            },
            WorkerErrorKind::OutOfMemory => WorkerError::DeviceOutOfMemory {
                message: self.message,
                worker,
            },
            WorkerErrorKind::AccessDenied => WorkerError::ModelAccessDenied {
                message: self.message,
                worker,
            },
            WorkerErrorKind::InvalidRequest => WorkerError::RequestRefused(self.message),
        }
    }

    /// The failure as the error of an `ensure_task`: the model load, which is
    /// where every model loads (no worker loads one at startup).
    ///
    /// Classified as a load failure has always been: a download that failed
    /// in transit is a download (retried), an exhausted device is memory
    /// pressure, a refused gated repository is an access denial, and any
    /// other failure is bootstrap-class, since the same load fails the same
    /// way again, so a `runtime` kind is not trusted to make it retryable. A
    /// refused request stays a refusal.
    pub(crate) fn into_ensure_task_error(self) -> WorkerError {
        let worker = self.kind.worker_after();
        match self.kind {
            WorkerErrorKind::Runtime | WorkerErrorKind::Bootstrap => WorkerError::Bootstrap {
                message: format!("ensure_task failed: {}", self.message),
                worker,
            },
            // A download that failed in transit is the one load failure a
            // later attempt can get past, with a memory shortfall, which
            // later work may have relieved.
            WorkerErrorKind::Download => WorkerError::ModelDownload {
                message: format!("ensure_task failed: {}", self.message),
                worker,
            },
            WorkerErrorKind::OutOfMemory => WorkerError::DeviceOutOfMemory {
                message: format!("ensure_task failed: {}", self.message),
                worker,
            },
            WorkerErrorKind::AccessDenied => WorkerError::ModelAccessDenied {
                message: format!("ensure_task failed: {}", self.message),
                worker,
            },
            WorkerErrorKind::InvalidRequest => {
                WorkerError::RequestRefused(format!("ensure_task refused: {}", self.message))
            }
        }
    }

    /// The failure as the answer to a health probe: whatever the kind, the
    /// worker is not healthy.
    pub(crate) fn into_health_error(self) -> WorkerError {
        WorkerError::HealthCheckFailed(self.message)
    }
}

/// Dump a failed worker IPC request to the always-on debug directory.
///
/// Writes to `~/.batchalign3/debug/failed_ipc_{timestamp}.json` so the
/// operator can inspect exactly what was sent, what came back (or didn't),
/// and which worker handled it, without needing `--debug-dir`.
///
/// The response field is truncated to [`FAILED_REQUEST_DUMP_MAX_RESPONSE_BYTES`]
/// to avoid disk exhaustion from malformed worker output.
pub(super) fn dump_failed_ipc_request(
    worker_pid: WorkerPid,
    worker: &super::WorkerConfig,
    request_json: &str,
    error: &WorkerError,
    response_fragment: Option<&str>,
) {
    let fallback_dir = crate::config::RuntimeLayout::from_env()
        .state_dir()
        .join("debug");
    if std::fs::create_dir_all(&fallback_dir).is_err() {
        return;
    }

    // UTC (a `Timestamp` formats in UTC), milliseconds, no separators.
    let timestamp = jiff::Timestamp::now().strftime("%Y%m%d_%H%M%S%3f");
    let path = fallback_dir.join(format!("failed_ipc_{timestamp}.json"));

    let truncated_response = response_fragment.map(|r| {
        if r.len() > FAILED_REQUEST_DUMP_MAX_RESPONSE_BYTES {
            format!(
                "{}... [truncated, {} bytes total]",
                &r[..FAILED_REQUEST_DUMP_MAX_RESPONSE_BYTES],
                r.len()
            )
        } else {
            r.to_string()
        }
    });

    let dump = serde_json::json!({
        "timestamp": timestamp.to_string(),
        "worker_pid": *worker_pid,
        "worker_label": worker.bootstrap_label(),
        // What `batchalign3 replay` rebuilds the worker from.
        "worker_launch": super::WorkerLaunch::of(worker),
        "error_type": format!("{error:?}").split('(').next().unwrap_or("Unknown"),
        "error_message": error.to_string(),
        "request": request_json,
        "response_fragment": truncated_response,
    });

    match serde_json::to_string_pretty(&dump) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                debug!(%e, "failed to write IPC failure dump");
            } else {
                warn!(
                    path = %path.display(),
                    worker_pid = *worker_pid,
                    "Worker IPC failure dump written for post-mortem"
                );
            }
        }
        Err(e) => debug!(%e, "failed to serialize IPC failure dump"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::util::classify_worker_error;
    use crate::types::scheduling::{FailureCategory, Retryability};
    use crate::worker::error::WorkerAfterError;

    fn ensure_task_failure(kind: WorkerErrorKind) -> WorkerError {
        ReportedFailure {
            message: "the load failed".to_owned(),
            kind,
        }
        .into_ensure_task_error()
    }

    /// Every model now loads in `ensure_task`, so a load that fails there
    /// must reach the server as the failure it was when models loaded at
    /// startup: a download that failed in transit is retried on a fresh
    /// worker, a memory shortfall after a backoff, a refused gated repository
    /// is an access denial (not retried, its remedy named), and anything
    /// else is a deterministic bootstrap failure.
    #[test]
    fn a_failed_load_keeps_its_classification() {
        let download = ensure_task_failure(WorkerErrorKind::Download);
        assert!(matches!(
            download,
            WorkerError::ModelDownload {
                worker: WorkerAfterError::Exits,
                ..
            }
        ));
        assert_eq!(
            classify_worker_error(&download).retryability(),
            Retryability::Retryable
        );

        let memory = ensure_task_failure(WorkerErrorKind::OutOfMemory);
        assert_eq!(
            classify_worker_error(&memory),
            FailureCategory::MemoryPressure
        );

        let refused = ensure_task_failure(WorkerErrorKind::AccessDenied);
        assert!(matches!(
            refused,
            WorkerError::ModelAccessDenied {
                worker: WorkerAfterError::Exits,
                ..
            }
        ));
        assert_eq!(
            classify_worker_error(&refused),
            FailureCategory::ModelAccessDenied
        );
        assert_ne!(
            classify_worker_error(&refused).retryability(),
            Retryability::Retryable
        );

        for kind in [WorkerErrorKind::Runtime, WorkerErrorKind::Bootstrap] {
            let error = ensure_task_failure(kind);
            assert_eq!(
                classify_worker_error(&error),
                FailureCategory::WorkerBootstrap,
                "{kind:?}"
            );
        }
        assert!(matches!(
            ensure_task_failure(WorkerErrorKind::InvalidRequest),
            WorkerError::RequestRefused(_)
        ));
    }
}
