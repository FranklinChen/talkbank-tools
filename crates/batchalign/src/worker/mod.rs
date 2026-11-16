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
pub(crate) mod process_id;
pub(crate) mod provider_credentials;
pub mod python;
pub mod registry;
pub mod request_builder_v2;
pub mod residency;
pub mod runtime_identity;
pub mod serving;
pub mod speaker_embedding_request_v2;
pub mod speaker_request_v2;
pub mod speaker_result_v2;
pub(crate) mod target;
pub mod tcp_handle;
pub mod text_request_v2;
pub mod text_result_v2;
mod whisper_windows;

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

/// How long a model-loading wait may go without the worker reporting progress
/// that advanced.
///
/// Not a budget for the whole wait. A worker that is starting, or loading a
/// model in `ensure_task`, may first have to download it, and a first
/// download's duration depends on the network, not on anything the
/// supervisor knows: a 1.3 GB Paraformer composition took 14 minutes on
/// 2026-11-05, against a fixed 300 s startup limit and a 120 s `ensure_task`
/// limit. A fixed limit cannot tell that download from a hung worker. This
/// window can: the worker reports measured download growth
/// (`batchalign/worker/_download_liveness.py`), and only a worker that stops
/// advancing for this long is stalled.
///
/// The one kind of wait a starting or loading worker is given. A spawn's wait
/// for host memory has no limit at all (it queues, reported, see
/// `crate::host_memory`), and a TCP daemon reports its startup on the stream
/// its spawner reads, so it waits under this window like a stdio worker; the
/// fixed limits those two once had (`PermitWait`, `TcpStartLimit`) are gone.
/// A distinct type from the decode budget ([`ResponseWait::Budget`]) so a
/// loading wait cannot be handed a budget, nor a request a stall window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadingStall(crate::api::PositiveSeconds);

impl LoadingStall {
    /// The startup stall when `worker_startup_stall_s` is not configured.
    pub const DEFAULT_STARTUP: Self = Self(crate::api::PositiveSeconds::literal::<
        { crate::config::DEFAULT_WORKER_STARTUP_STALL_S },
    >());

    /// The `ensure_task` stall when `ensure_task_stall_s` is not configured.
    pub const DEFAULT_ENSURE_TASK: Self = Self(crate::api::PositiveSeconds::literal::<
        { crate::config::DEFAULT_ENSURE_TASK_STALL_S },
    >());

    /// A window of exactly `window`: the configuration boundary
    /// (`ServerConfig::worker_startup_stall`, `ServerConfig::ensure_task_stall`)
    /// and tests.
    pub const fn new(window: crate::api::PositiveSeconds) -> Self {
        Self(window)
    }

    /// The window, in seconds.
    pub const fn window(self) -> crate::api::PositiveSeconds {
        self.0
    }
}

impl std::fmt::Display for LoadingStall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// What a model-loading wait is for: the only waits a stall window governs.
///
/// A narrower set than [`error::WorkerWait`], so a [`error::WorkerError::Stalled`]
/// can only name a load, never a health probe or a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadingWait {
    /// A spawned worker's ready signal, while it loads (and may download) its
    /// models.
    Ready,
    /// An on-demand model load in a lazy worker.
    EnsureTask {
        /// The task being loaded.
        task: InferTask,
    },
}

impl std::fmt::Display for LoadingWait {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ready => f.write_str("the worker's ready signal"),
            Self::EnsureTask { task } => {
                write!(f, "the ensure_task({}) response", target::task_name(*task))
            }
        }
    }
}

/// How a read from a worker is allowed to wait, and what it waits for: chosen
/// together by the operation, so a load cannot be given a budget nor a
/// request a stall window, and the error names the right wait.
#[derive(Debug, Clone)]
pub(crate) enum ResponseWait {
    /// A fixed limit from the start of the wait: a request whose cost the
    /// request itself bounds (a decode budget derived from the audio, a
    /// health probe). Progress events do not extend it.
    Budget {
        waited_for: error::WorkerWait,
        limit: crate::api::PositiveSeconds,
    },
    /// A model load: a stall window that progress which advanced re-arms.
    Loading {
        waited_for: LoadingWait,
        stall: LoadingStall,
    },
}

mod wait;
pub(crate) use wait::RunningWait;

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
