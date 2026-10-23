//! Error types for Python worker process management.
//!
//! [`WorkerError`] covers every failure mode in the worker lifecycle: spawning
//! the child process, waiting for its ready signal, communicating over the
//! stdio JSON-lines protocol, health checking, and unexpected exits.
//!
//! # Retryability
//!
//! Some variants are retryable (the pool's health loop will automatically
//! restart dead workers), while others indicate a configuration or
//! environmental problem that requires operator intervention. Each variant's
//! doc comment notes which category it falls into.

pub use crate::types::worker_v2::WorkerAfterError;

/// Format the `ProcessExited` Display output, including stderr when available.
fn format_process_exited(code: Option<i32>, stderr: Option<&str>) -> String {
    let header = format!("worker process exited unexpectedly (exit code: {code:?})");
    match stderr {
        Some(text) if !text.is_empty() => {
            format!("{header}\n--- worker stderr ---\n{text}")
        }
        _ => header,
    }
}

/// What a wait on a worker was for, so a timeout names it as data rather
/// than in a message a classifier would have to parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerWait {
    /// A TCP connection to a worker daemon at this address.
    Connect {
        /// `host:port`.
        addr: String,
    },
    /// A health probe's reply.
    Health,
    /// A capabilities report.
    Capabilities,
    /// An on-demand model load (`ensure_task`).
    EnsureTask {
        /// The task being loaded.
        task: crate::worker::InferTask,
    },
    /// A V1 `infer` reply.
    Infer,
    /// A V1 `batch_infer` reply.
    BatchInfer {
        /// Items in the batch.
        items: usize,
    },
    /// A V2 `execute_v2` reply.
    ExecuteV2 {
        /// The task family requested.
        task: crate::types::worker_v2::InferenceTaskV2,
    },
}

impl std::fmt::Display for WorkerWait {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connect { addr } => write!(f, "a connection to the worker at {addr}"),
            Self::Health => f.write_str("the health response"),
            Self::Capabilities => f.write_str("the capabilities response"),
            Self::EnsureTask { task } => write!(
                f,
                "the ensure_task({}) response",
                crate::worker::target::task_name(*task)
            ),
            Self::Infer => f.write_str("the infer response"),
            Self::BatchInfer { items } => write!(f, "the batch_infer response ({items} items)"),
            Self::ExecuteV2 { task } => write!(f, "the execute_v2 response ({task:?})"),
        }
    }
}

/// Errors arising from Python worker process management.
#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    /// A native operation was incorrectly routed to Python inference. This
    /// is a deterministic orchestration defect, not a missing model or a
    /// transient worker failure. Admission refuses before touching the pool.
    #[error("native command '{command}' cannot request an inference worker")]
    NativeCommand {
        /// Command wrongly submitted to the worker boundary.
        command: crate::api::ReleasedCommand,
    },
    /// The Python child process could not be created.
    ///
    /// Common causes: `python_path` does not exist, the `batchalign.worker`
    /// module is not importable (missing install), or OS resource limits
    /// (file descriptors, process count) have been reached.
    ///
    /// **Terminal** -- retrying with the same configuration will fail again.
    /// Callers should surface the inner message to the operator so they can
    /// fix the environment (install batchalign, adjust `python_path`, etc.).
    #[error("worker failed to start: {0}")]
    SpawnFailed(String),

    /// The worker process started but did not emit its JSON ready signal
    /// within the configured timeout.
    ///
    /// This usually means the worker is stuck loading heavy ML models (Stanza,
    /// Whisper) or the Python import itself is hanging. Increasing
    /// `ready_timeout_s` in [`super::pool::PoolConfig`] may help for
    /// legitimately slow model loads.
    ///
    /// **Retryable** -- a transient resource stall (disk I/O, network for
    /// model download) can resolve on retry. The pool will try again on the
    /// next dispatch.
    #[error("worker not ready after {timeout_s}s")]
    ReadyTimeout {
        /// Number of seconds waited before giving up.
        timeout_s: crate::api::PositiveSeconds,
    },

    /// The worker emitted something on stdout before the ready signal, but it
    /// was not valid JSON or contained unexpected fields (e.g. `ready: false`,
    /// unsupported transport).
    ///
    /// Indicates a version mismatch between the Rust server and the Python
    /// `batchalign.worker` module, or stray print statements in Python code
    /// polluting stdout.
    ///
    /// **Terminal** -- the same worker code will produce the same bad output.
    /// Callers should log the raw line for debugging and check that the
    /// installed batchalign version matches the server.
    #[error("failed to parse ready signal: {0}")]
    ReadyParseFailed(String),

    /// A periodic health check (or an on-demand probe) failed.
    ///
    /// The worker may have become unresponsive (deadlocked, GIL contention)
    /// or returned an unhealthy status. The pool's background health loop
    /// automatically removes unhealthy workers and spawns replacements, so
    /// callers do not need to handle this directly.
    ///
    /// **Retryable** -- the pool replaces the dead worker transparently.
    #[error("worker health check failed: {0}")]
    HealthCheckFailed(String),

    /// The worker process exited while the server was waiting for a response.
    ///
    /// Typical causes: OOM kill, segfault in a native library (torch, stanza),
    /// or an unhandled Python exception that crashes the process. The exit
    /// code (if available) can distinguish signals from normal exits.
    ///
    /// **Retryable** -- the pool's health loop will detect the dead process
    /// and spawn a replacement. However, if the crash is deterministic (e.g.
    /// always triggered by a specific input), the replacement will crash too.
    #[error("{}", format_process_exited(*.code, stderr.as_deref()))]
    ProcessExited {
        /// Exit code of the worker process, if available.
        code: Option<i32>,
        /// Last lines of the worker's stderr, captured at crash time.
        ///
        /// Contains the Python traceback, OOM message, or other diagnostic
        /// output that explains WHY the worker died. `None` if stderr was
        /// empty or could not be read before the process fully exited.
        stderr: Option<String>,
    },

    /// A wait on the worker ran past its limit: a reply, a model load, or a
    /// connection.
    ///
    /// **Retryable** (`WorkerTimeout`) -- a stall can clear. What happens to
    /// the worker depends on how it serves:
    ///
    /// - A sequential worker (a spawned process or a TCP daemon handle) is
    ///   retired, since its late reply could still arrive on its
    ///   single-flight stream.
    /// - A spawned shared GPU worker is retired too: the thread serving the
    ///   request may still hold one of its `gpu_thread_pool_size` slots.
    /// - A shared GPU daemon connection is kept: its replies are routed by
    ///   request id, so a late one is dropped, and the daemon is not this
    ///   server's to restart. A request hung inside it holds a slot until it
    ///   ends.
    #[error("timeout ({limit}s) waiting for {waited_for}")]
    Timeout {
        /// What was awaited.
        waited_for: WorkerWait,
        /// The limit that passed.
        limit: crate::api::PositiveSeconds,
    },

    /// The stdio JSON-lines protocol was violated: a request could not be
    /// serialized, a response could not be deserialized, or the response
    /// had the wrong `op` tag for the request that was sent.
    ///
    /// This points to a version mismatch between Rust and Python, or a bug
    /// in one side's serialization.
    ///
    /// **Terminal for this request** -- the worker's stdio stream may be
    /// desynchronized after a framing error. The pool should discard the
    /// worker (via `CheckedOutWorker::take()`) rather than returning it.
    #[error("worker protocol error: {0}")]
    Protocol(String),

    /// The worker understood the request but returned an application-level
    /// error (the `{"op":"error","error":"..."}` response) for a
    /// **runtime / per-request** failure.
    ///
    /// Examples: a transient model-inference failure on a malformed input,
    /// an external-API hiccup, a worker-side validation that depends on
    /// per-request payload. The error string comes directly from the
    /// Python side.
    ///
    /// **Retryable** -- a fresh worker may succeed on retry, especially
    /// when the underlying cause is per-request (different inputs, transient
    /// resource state).
    ///
    /// For deterministic bootstrap failures (missing model files, an
    /// unsupported language), the worker emits the
    /// `{"op":"error", "kind":"bootstrap"}` wire shape, which decodes into
    /// the [`Bootstrap`] variant instead; those are non-retryable.
    ///
    /// [`Bootstrap`]: WorkerError::Bootstrap
    #[error("worker returned error: {0}")]
    WorkerResponse(String),

    /// The worker reported a **bootstrap-class** failure: model loading,
    /// package import, an unsupported language, or any deterministic failure
    /// that would recur identically across worker restarts.
    ///
    /// Two sources: an `{"op":"error", "kind":"bootstrap"}` line (and a
    /// `runtime` line answering `ensure_task`, since a load that failed fails
    /// again), and an execute response coded `model_unavailable` or
    /// `unsupported_language`. `worker` says which: the first stops serving
    /// (the kind's own [`WorkerErrorKind::worker_after`]), the second answered
    /// and goes on serving.
    ///
    /// **Terminal** -- retrying a deterministic bootstrap failure 3× with
    /// a full traceback per attempt is the bug shape this variant exists
    /// to prevent. The orchestrator classifies this as non-retryable and
    /// surfaces the typed message verbatim to the user.
    ///
    /// [`WorkerErrorKind::worker_after`]: crate::types::worker_v2::WorkerErrorKind::worker_after
    #[error("worker bootstrap error: {message}")]
    Bootstrap {
        /// The worker's own diagnosis.
        message: String,
        /// Whether the worker still serves after reporting it.
        worker: WorkerAfterError,
    },

    /// A model or catalog download failed in transit (the network, the
    /// remote host): an `{"op":"error", "kind":"download"}` line, or an
    /// execute response coded `model_download_failed`.
    ///
    /// **Retryable**, unlike [`Self::Bootstrap`]: the worker configuration is
    /// sound and a later attempt may get the download through. It is the one
    /// load failure that does not withhold a same-engine UTR fallback. A
    /// worker that stopped serving after it is retired, so the retry runs on
    /// a fresh one rather than on a dead handle.
    #[error("worker model download failed: {message}")]
    ModelDownload {
        /// The worker's own diagnosis.
        message: String,
        /// Whether the worker still serves after reporting it.
        worker: WorkerAfterError,
    },

    /// The accelerator ran out of memory (`torch.OutOfMemoryError`): an
    /// `{"op":"error", "kind":"out_of_memory"}` line or an execute response
    /// coded `device_out_of_memory`.
    ///
    /// **Retryable** (`MemoryPressure`) after a backoff: the input is not at
    /// fault, and the request may fit once concurrent work releases memory.
    #[error("worker ran out of device memory: {message}")]
    DeviceOutOfMemory {
        /// The worker's own diagnosis.
        message: String,
        /// Whether the worker still serves after reporting it.
        worker: WorkerAfterError,
    },

    /// A local model reported an unclassified `RuntimeFailure`.
    /// The code does not establish a cause or deterministic input defect.
    /// Retried within the existing budget; the healthy channel remains reusable.
    #[error("local model failed on this request: {0}")]
    ModelRuntime(String),

    /// The worker refused the request as sent: the line was not JSON, the op
    /// or its `request` mapping was missing, or the payload failed the
    /// worker's request model (`{"op":"error", "kind":"invalid_request"}`).
    ///
    /// **Terminal** -- the same request is refused the same way again. The
    /// worker answered with a complete line, so it stays in service.
    #[error("worker refused the request: {0}")]
    RequestRefused(String),

    /// A newly spawned worker reports different executable code from the
    /// runtime already pinned by this server.
    ///
    /// **Terminal** -- admitting it would make result provenance ambiguous;
    /// the server must be restarted into one coherent Python environment.
    #[error("worker runtime identity differs from the runtime pinned by this server")]
    RuntimeIdentityMismatch,

    /// The worker's capability report was refused where the pool admits it:
    /// an advertised task with no engine entry, or an engine entry for a task
    /// it did not advertise.
    ///
    /// **Terminal** -- the same worker code reports the same thing again; the
    /// report, not the connection, is wrong.
    #[error(transparent)]
    CapabilitiesRefused(#[from] crate::engine_reports::EngineReportAdmissionError),

    /// Low-level I/O failure on the stdin/stdout pipes to the worker process.
    ///
    /// Usually means the pipe was closed (worker crashed) or a system-level
    /// I/O error occurred. Often accompanied or preceded by [`ProcessExited`].
    ///
    /// **Retryable** -- the pool spawns a replacement worker.
    ///
    /// [`ProcessExited`]: WorkerError::ProcessExited
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The worker could not start because the host-memory guard
    /// refused the per-spawn reservation (insufficient available RAM,
    /// or the host-coordinator's lease acquisition failed).
    ///
    /// Distinguishes RAM-pressure rejections from generic spawn
    /// failures so observability surfaces (`PoolMetrics`, doctor)
    /// can count host-memory pressure independently. The pool's
    /// `lifecycle::try_spawn_into_group` increments
    /// `memory_gate_rejections_total` on this variant.
    ///
    /// **Retryable** -- once host RAM is freed (other workers exit,
    /// idle timeouts fire), a fresh spawn can succeed.
    #[error("memory guard refused worker spawn: {0}")]
    MemoryGuard(#[from] crate::worker::memory_guard::MemoryGuardError),

    /// No worker could be found or spawned for the requested
    /// `(command, lang)` pair.
    ///
    /// Currently unused -- the pool lazily spawns workers on demand and
    /// returns [`SpawnFailed`] if spawning fails. This variant is reserved
    /// for future use (e.g. capability-gated dispatch where no worker
    /// advertises the required infer task).
    ///
    /// **Terminal** -- the server does not support this command/lang
    /// combination.
    ///
    /// [`SpawnFailed`]: WorkerError::SpawnFailed
    #[error("no worker available for command={command} lang={lang}")]
    NoWorker {
        /// Processing command that was requested.
        command: crate::api::ReleasedCommand,
        /// Language code that was requested.
        lang: crate::api::WorkerLanguage,
    },

    /// A GPU-profile request reached the request-keyed dispatch. GPU requests
    /// are dispatched only on a route from their job's model plan
    /// (`WorkerPool::gpu_route`), so that every request of one job reaches the
    /// worker its plan selected. Keying one by its own task is how one align
    /// job came to spawn two GPU workers.
    ///
    /// **Terminal** (`System`): a deterministic orchestration defect.
    #[error(
        "a {task:?} request was dispatched without a route from its job's model plan; \
         request-keyed dispatch serves only text and IO tasks"
    )]
    UnroutedGpuRequest {
        /// The request's pool task.
        task: crate::worker::InferTask,
    },

    /// A request disagreed with the route it was sent on: its builder and its
    /// job's model plan chose engines separately.
    ///
    /// **Terminal** (`System`): a deterministic orchestration defect.
    #[error(transparent)]
    PlanMismatch(#[from] crate::worker::pool::PlanMismatch),

    /// A job asked to route a GPU task its model plan did not declare.
    ///
    /// **Terminal** (`System`): a deterministic orchestration defect.
    #[error(transparent)]
    UndeclaredGpuTask(#[from] crate::worker::pool::GpuRouteRefusal),

    /// The pool or shared worker has begun shutting down. Dispatch, health
    /// checks and task loading refuse admission with this lifecycle fact,
    /// rather than claiming that an IPC protocol was violated.
    ///
    /// A shared-GPU spawn can also still be running when
    /// `WorkerPool::shutdown` drains the map. The spawning task retires its own
    /// worker in that case rather than return one nothing would ever reap.
    ///
    /// **Terminal** -- the pool is going away; there is nothing to retry
    /// against. Callers should let the job unwind.
    #[error("worker pool is shutting down")]
    PoolShuttingDown,

    /// The pool's own accounting no longer describes its workers: a group's
    /// live count exceeds what its idle queues and its checkout, spawn and
    /// health-check guards hold, at two consecutive report intervals of a
    /// saturated wait. A checkout waiting on such a count would wait forever,
    /// so it ends with this instead. An internal defect, never congestion.
    ///
    /// **Terminal** (`System`, not retried): retrying would wait on the same
    /// count.
    #[error(
        "worker pool accounting is broken for {} lang={lang}: {unaccounted} counted \
         worker(s) are held by nothing (total={total}, idle={idle}, away={away})",
        .target.label()
    )]
    PoolAccountingBroken {
        /// The group whose count is broken.
        target: crate::worker::WorkerTarget,
        /// Its language.
        lang: crate::api::WorkerLanguage,
        /// Counted workers nothing holds.
        unaccounted: usize,
        /// The group's live count.
        total: usize,
        /// Idle workers in its queues.
        idle: usize,
        /// Workers its guards account for.
        away: usize,
    },

    /// The pool retired the shared worker a request was sent to while the
    /// pool keeps serving: its capability report was refused, or it is being
    /// replaced. The request was not refused; its worker went away under it.
    ///
    /// **Retryable** (`WorkerCrash`) -- a retry is dispatched to another
    /// worker.
    #[error("the worker serving this request was retired")]
    WorkerRetired,

    /// The worker wrote `MAX_RESPONSE_STDOUT_NOISE_LINES` (8) consecutive
    /// lines that are not protocol messages, so its stream is
    /// not trusted further. A stdio worker writes its protocol on a private
    /// descriptor and sends library output to stderr, so this means
    /// something wrote into the protocol stream anyway.
    ///
    /// **Retryable** (`WorkerCrash`) -- the worker is retired and the request
    /// runs again on another.
    #[error("worker wrote too many lines that are not protocol messages; last: {last_line:?}")]
    OutputNoise {
        /// The last such line, truncated.
        last_line: String,
    },
}

/// What a failed exchange leaves of the worker it ran on: whether the worker
/// may serve another request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorkerAfterFailure {
    /// The worker answered with a complete error response and its connection
    /// is in step with the protocol: it goes back to its pool.
    Reusable,
    /// The connection or process can no longer be trusted (a stream that may
    /// be half-read, a dead process, a timeout whose late reply could still
    /// arrive), or the worker is not admissible: it is retired, not reused.
    Retire,
}

impl WorkerError {
    /// Whether the worker an exchange failed on may be reused.
    ///
    /// Matched on every variant, so a new error kind must decide. Kinds that
    /// cannot arise from an exchange on a live worker (spawn, ready and
    /// admission failures) answer `Retire`, the conservative choice.
    pub(crate) fn worker_after_failure(&self) -> WorkerAfterFailure {
        match self {
            // A report the worker answered and went on from, or one after
            // which it stopped serving: the kind decided which.
            Self::Bootstrap { worker, .. }
            | Self::ModelDownload { worker, .. }
            | Self::DeviceOutOfMemory { worker, .. } => match worker {
                WorkerAfterError::Serves => WorkerAfterFailure::Reusable,
                WorkerAfterError::Exits => WorkerAfterFailure::Retire,
            },
            Self::WorkerResponse(_)
            | Self::ModelRuntime(_)
            | Self::RequestRefused(_)
            | Self::MemoryGuard(_)
            | Self::NoWorker { .. }
            | Self::NativeCommand { .. }
            | Self::UnroutedGpuRequest { .. }
            | Self::PlanMismatch(_)
            | Self::UndeclaredGpuTask(_)
            | Self::PoolShuttingDown
            | Self::PoolAccountingBroken { .. } => WorkerAfterFailure::Reusable,
            Self::SpawnFailed(_)
            | Self::ReadyTimeout { .. }
            | Self::ReadyParseFailed(_)
            | Self::HealthCheckFailed(_)
            | Self::ProcessExited { .. }
            | Self::Timeout { .. }
            | Self::Protocol(_)
            | Self::Io(_)
            | Self::WorkerRetired
            | Self::OutputNoise { .. }
            | Self::RuntimeIdentityMismatch
            // A worker whose capability report was refused is not used.
            | Self::CapabilitiesRefused(_) => WorkerAfterFailure::Retire,
        }
    }
}

impl From<super::runtime_identity::WorkerRuntimeMismatch> for WorkerError {
    fn from(_: super::runtime_identity::WorkerRuntimeMismatch) -> Self {
        Self::RuntimeIdentityMismatch
    }
}
