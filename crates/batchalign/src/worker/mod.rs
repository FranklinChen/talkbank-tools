//! Python worker process manager: spawn, health-check, dispatch.
//!
//! This crate manages Python worker child processes that do the actual ML
//! inference. The Rust binary is the control plane; Python workers are the
//! data plane.
//!
//! # Architecture
//!
//! ```text
//! WorkerPool
//!   ├── WorkerHandle("infer:morphosyntax", "eng") → morphosyntax model host
//!   ├── WorkerHandle("infer:fa", "eng")           → forced-alignment model host
//!   └── WorkerHandle("infer:asr", "eng")          → ASR model host
//! ```
//!
//! Workers are spawned lazily on first request, health-checked periodically,
//! restarted on failure, and idle-timed out after inactivity.

pub mod artifacts_v2;
pub mod asr_request_v2;
pub mod asr_result_v2;
pub mod avqi_request_v2;
pub(crate) mod chunk_spans;
pub mod error;
pub mod execute_result_v2;
pub mod fa_result_v2;
pub mod handle;
pub mod memory_guard;
pub mod opensmile_request_v2;
pub mod pool;
pub(crate) mod provider_credentials;
pub mod python;
pub mod registry;
pub mod request_builder_v2;
pub mod runtime_identity;
pub mod serving;
pub mod speaker_embedding_request_v2;
pub mod speaker_request_v2;
pub mod speaker_result_v2;
pub(crate) mod target;
pub mod tcp_handle;
pub mod text_request_v2;
pub mod text_result_v2;

// Re-export wire-format types from types::worker so that
// `crate::worker::InferTask` etc. continues to resolve.
pub use crate::types::worker::*;
pub use target::{WorkerBootstrapMode, WorkerProfile, WorkerTarget};

/// Capability probes perform cold ML imports after the ready handshake.
/// Give all transports the same bounded startup budget, not a health-check budget.
pub(crate) const CAPABILITY_TIMEOUT: crate::api::PositiveSeconds =
    crate::api::PositiveSeconds::literal::<300>();

/// How long a health probe waits for its reply.
pub(crate) const HEALTH_TIMEOUT: crate::api::PositiveSeconds =
    crate::api::PositiveSeconds::literal::<10>();

/// How long a V1 `infer` waits for its reply (first-touch model loads included).
pub(crate) const INFER_TIMEOUT: crate::api::PositiveSeconds =
    crate::api::PositiveSeconds::literal::<120>();

/// How long a TCP connection to a worker daemon may take.
pub(crate) const CONNECT_TIMEOUT: crate::api::PositiveSeconds =
    crate::api::PositiveSeconds::literal::<10>();

/// When a wait on a worker ends: a limit, started when the wait starts.
///
/// Built only from a [`crate::api::PositiveSeconds`]. A limit past what the
/// clock can represent is a wait with no end, where `Instant::now() + limit`
/// would panic.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Deadline(Option<tokio::time::Instant>);

impl Deadline {
    /// The end of a wait of `limit` that starts now.
    pub(crate) fn after(limit: crate::api::PositiveSeconds) -> Self {
        Self(tokio::time::Instant::now().checked_add(limit.duration()))
    }

    /// `work`'s output, or `None` when the deadline passed first.
    pub(crate) async fn within<F: std::future::Future>(self, work: F) -> Option<F::Output> {
        match self.0 {
            Some(at) => tokio::time::timeout_at(at, work).await.ok(),
            None => Some(work.await),
        }
    }
}

// ---------------------------------------------------------------------------
// ensure_task IPC types (shared between sequential and concurrent paths)
// ---------------------------------------------------------------------------

/// Status of an `ensure_task` model-loading IPC call, as the worker reported it.
///
/// The Python worker returns `loaded` or `already_loaded`. When the Rust-side
/// cache already knows a task is loaded, no IPC happens and no response
/// exists: `ensure_task` returns `Ok(())` on every path, so there is nothing
/// to fabricate.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnsureTaskStatus {
    /// Models were loaded on demand (first call for this task).
    Loaded,
    /// Python confirmed the task was already loaded (IPC round-trip happened).
    AlreadyLoaded,
}

impl std::fmt::Display for EnsureTaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Loaded => f.write_str("loaded"),
            Self::AlreadyLoaded => f.write_str("already_loaded"),
        }
    }
}

/// Response from the `ensure_task` IPC operation.
///
/// Used by both the sequential worker path (`WorkerHandle::ensure_task`)
/// and the concurrent GPU path (`SharedGpuWorker::ensure_task`).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct EnsureTaskResponse {
    /// Whether the task was already loaded or had to be switched.
    pub status: EnsureTaskStatus,
    /// The task this reply is about, as the worker echoed it. A reader
    /// compares it with the task it asked for before treating the task as
    /// loaded (a name outside [`InferTask`] is refused at the wire).
    pub task: InferTask,
    /// Seconds the worker spent answering, as it measured them: the model
    /// load for `loaded`, the check (and any wait for a concurrent load of
    /// the same task) for `already_loaded`. A negative or non-finite reading
    /// is refused at the wire.
    pub elapsed_s: crate::api::NonNegativeSeconds,
}

impl EnsureTaskResponse {
    /// This reply, when it is about `asked`; a protocol error when it names
    /// another task. For a strict request-reply reader (one op in flight,
    /// and the worker retired on a timeout), a reply about another task can
    /// only be a protocol violation.
    pub(crate) fn about(self, asked: InferTask) -> Result<Self, error::WorkerError> {
        if self.task == asked {
            Ok(self)
        } else {
            Err(error::WorkerError::Protocol(format!(
                "ensure_task({}) was answered about {}",
                target::task_name(asked),
                target::task_name(self.task)
            )))
        }
    }
}
