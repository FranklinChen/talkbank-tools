//! `WorkerHandle`: manages a single Python worker child process.
//!
//! Split into submodules:
//! - [`config`]: `WorkerRuntimeConfig`, `WorkerConfig`
//! - [`protocol`]: Wire-level request/response envelopes, ready signals, constants
//! - [`spawn`]: Command building and TCP daemon spawning
//! - [`ipc`]: Request/response IPC methods (infer, batch_infer, execute_v2, health, capabilities)
//! - [`lifecycle`]: Startup helpers, shutdown, Drop, accessors

pub mod config;
pub(crate) mod control;
mod ipc;
mod lifecycle;
mod progress_preamble;
pub(crate) use progress_preamble::{ResponseSource, read_skipping_progress};
mod protocol;
pub mod spawn;
#[cfg(test)]
pub(crate) mod test_support;

/// The account snapshot [`WorkerRuntimeConfig::provider_accounts`] holds and
/// the engine setting inside it, re-exported so the runtime's public fields
/// are nameable.
pub use crate::model_manifest::{TencentEngine, TencentEngineRefusal, TencentEngineSetting};
pub use crate::worker::provider_credentials::ProviderAccounts;
pub(crate) use config::WorkerLaunch;
pub use config::{WorkerConfig, WorkerRuntimeConfig};
pub use spawn::spawn_tcp_daemon;
// The wire envelopes and the reported-failure mapping are shared by every
// transport (`worker::tcp_handle` and the shared GPU reader read the same
// lines), without making the whole `protocol` submodule public.
#[cfg(test)]
pub(crate) use protocol::MAX_RESPONSE_STDOUT_NOISE_LINES;
pub(crate) use protocol::{
    CapabilitiesRequest, ControlRequestId, EnsureTaskRequest, NoiseLimit, NoiseRun,
    ReportedFailure, WireLine, WorkerRequest, WorkerResponse, excerpt,
};

use crate::worker::WorkerPid;
use crate::worker::error::WorkerError;
use crate::worker::runtime_identity::WorkerRuntimeIdentity;
use tokio::io::BufReader;
use tokio::process::{Child, ChildStdin, ChildStdout};
use tracing::{debug, info};

/// Whether a [`WorkerHandle`] has a request written to the worker's stdin
/// whose response has not yet been fully read back.
///
/// Set the instant a request is written (`write_request`), cleared only
/// once a COMPLETE terminal response for it has been read (the shared
/// [`WorkerHandle::read_response_skipping_progress_via_self`] loop and
/// `execute_v2_with_progress`'s own loop both clear it at their terminal
/// match arms, never at an intermediate `ProgressV2` line).
///
/// This exists because dropping the FUTURE of an in-flight dispatch
/// (a job cancellation racing `pool.dispatch_execute_v2_with_progress` in
/// `infer_retry::dispatch_execute_v2_with_retry_and_progress`'s `select!`)
/// does not stop the worker process: it may have already written a
/// response to stdout that nobody read, or may still be about to. A
/// `CheckedOutWorker` dropped while `InFlight` must therefore be
/// DISCARDED, never returned to the idle queue, exactly the way
/// `dispatch_batch_infer` already discards a worker after an Io/Protocol
/// error; see `CheckedOutWorker::drop`. Returning it would let the NEXT
/// dispatch on this worker read that stale response and misattribute it
/// to an unrelated request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RequestFlight {
    /// No request is outstanding; safe to reuse.
    Idle,
    /// A request has been written and its response has not been fully
    /// read; NOT safe to reuse if dropped in this state.
    InFlight,
}

/// Manages a single Python worker child process.
pub struct WorkerHandle {
    config: WorkerConfig,
    child: Child,
    pid: WorkerPid,
    runtime: WorkerRuntimeIdentity,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    /// Monotonic instant when the last request was dispatched.
    last_activity: tokio::time::Instant,
    /// This worker's on-demand load states (`NotLoaded` by absence,
    /// `Loaded` proofs); a repeated preparation reuses the proof.
    loads: crate::worker::pool::TaskLoads,
    /// Receiver for stderr lines captured by the background drain task.
    ///
    /// The drain task sends each non-empty stderr line through this channel.
    /// On worker crash, [`drain_stderr_tail`](Self::drain_stderr_tail) reads
    /// remaining lines to attach to the [`WorkerError::ProcessExited`] error.
    stderr_rx: tokio::sync::mpsc::UnboundedReceiver<String>,
    /// See [`RequestFlight`].
    request_flight: RequestFlight,
}

/// Raw parts extracted from a [`WorkerHandle`] via [`WorkerHandle::into_parts`].
///
/// Used by [`SharedGpuWorker`](super::pool::shared_gpu::SharedGpuWorker) to
/// take ownership of the child's stdio channels for concurrent dispatch.
#[allow(dead_code)]
pub(crate) struct WorkerHandleParts {
    /// Worker configuration.
    pub config: WorkerConfig,
    /// The child process (caller must manage lifecycle).
    pub child: Child,
    /// Worker process ID.
    pub pid: WorkerPid,
    /// Content-addressed runtime evidence from the ready handshake.
    pub runtime: WorkerRuntimeIdentity,
    /// Child's stdin for writing requests.
    pub stdin: ChildStdin,
    /// Child's stdout for reading responses.
    pub stdout: BufReader<ChildStdout>,
}

impl WorkerHandle {
    /// Spawn a new Python worker and wait for it to become ready. A startup
    /// that has to queue for host memory retires idle workers through
    /// `reclaim` (the pool's own, or
    /// [`crate::host_memory::NoIdleWorkers`] outside a pool).
    pub async fn spawn(
        config: WorkerConfig,
        reclaim: std::sync::Arc<dyn crate::host_memory::MemoryReclaim>,
    ) -> Result<Self, WorkerError> {
        // Memory guard: acquire a serialized spawn permit and the host's
        // startup lease for this worker's baseline (queued while the host
        // cannot fit it). Each model the worker later loads takes its own
        // lease. Crash history and design rationale:
        // `book/src/batchalign/developer/memory-safety.md`.
        let spawn_permit =
            crate::worker::memory_guard::acquire_spawn_permit(&config, reclaim.as_ref()).await?;

        let mut cmd: tokio::process::Command = spawn::build_worker_command(&config).into();
        // Before ready there is no WorkerHandle to own termination. The
        // startup task owns the child, including when its caller cancels.
        cmd.kill_on_drop(true);

        info!(
            target = %config.bootstrap_label(),
            lang = %config.lang,
            test_echo = config.test_echo,
            force_cpu = config.runtime.force_cpu,
            python = %config.python_path,
            startup_reservation_mb = crate::worker::residency::WORKER_BASELINE.reserved().0,
            available_memory_mb = ?config
                .runtime
                .host
                .readings()
                .memory()
                .ok()
                .map(|memory| memory.available_mb.0),
            "Spawning worker (memory guard passed)"
        );

        let spawn_start = std::time::Instant::now();

        let mut child = cmd.spawn().map_err(|e| {
            WorkerError::SpawnFailed(format!("failed to spawn {}: {}", config.python_path, e))
        })?;

        // Tag the host-memory lease with the worker's PID so that if
        // the worker dies (OOM, crash, external kill) the next
        // `prune_stale_leases` cycle reclaims its reservation slot.
        // Without this, the lease's `owner_pid` is the daemon's PID
        //, always alive, and the slot becomes a ghost (Bug 2,
        // 2026-05-01). `child.id()` returns `None` only if the child
        // has already been awaited, which has not happened here.
        if let Some(child_pid) = child.id() {
            spawn_permit.set_worker_pid(crate::worker::WorkerPid(child_pid));
        }
        // Move the permit into a `_` binding so it remains live for
        // the rest of the spawn window (RAII semantics for the
        // local-spawn semaphore + host lease are unchanged).
        let _spawn_permit = spawn_permit;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| WorkerError::SpawnFailed("child stdout not captured".into()))?;
        let mut stdout_reader = BufReader::new(stdout);
        let mut stderr_reader = BufReader::new(
            child
                .stderr
                .take()
                .ok_or_else(|| WorkerError::SpawnFailed("child stderr not captured".into()))?,
        );

        // The startup wait is a stall window over the worker's progress, not
        // a total limit: see `read_ready_line`.
        let ready = match Self::read_ready_line(&mut stdout_reader, config.startup_stall)
            .await
            .and_then(protocol::ReadySignal::into_stdio)
        {
            Ok(ready) => ready,
            Err(error) => {
                return Err(
                    Self::finalize_startup_failure(&mut child, &mut stderr_reader, error).await,
                );
            }
        };

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| WorkerError::SpawnFailed("child stdin not captured".into()))?;
        let (pid, runtime) = ready.into_parts();
        let startup_ms = spawn_start.elapsed().as_millis() as u64;

        info!(
            target = %config.bootstrap_label(),
            lang = %config.lang,
            pid = %pid,
            startup_ms,
            "Worker ready"
        );

        // Layer 3: record PID file for orphan reaping.
        super::pool::reaper::record_worker_pid(pid.0);

        let target_label = config.bootstrap_label();
        let (stderr_tx, stderr_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        tokio::spawn(async move {
            use tokio::io::AsyncBufReadExt;
            let mut line = String::new();
            loop {
                line.clear();
                match stderr_reader.read_line(&mut line).await {
                    Ok(0) => break,
                    Ok(_) => {
                        let trimmed = line.trim_end();
                        if !trimmed.is_empty() {
                            debug!(worker = %target_label, "{}", trimmed);
                            // Best-effort: if the receiver is dropped, we just stop sending.
                            let _ = stderr_tx.send(trimmed.to_owned());
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        Ok(Self {
            config,
            child,
            pid,
            runtime,
            stdin,
            stdout: stdout_reader,
            last_activity: tokio::time::Instant::now(),
            loads: crate::worker::pool::TaskLoads::default(),
            stderr_rx,
            request_flight: RequestFlight::Idle,
        })
    }

    /// See [`RequestFlight`]. Read by `CheckedOutWorker::drop` to decide
    /// whether this worker is safe to return to the idle queue.
    pub(crate) fn request_flight(&self) -> RequestFlight {
        self.request_flight
    }
}

#[cfg(test)]
impl WorkerHandle {
    /// Spawn a trivial `sh` stub in place of the real Python worker.
    ///
    /// For tests that need a REAL child process's stdin/stdout pipes to
    /// exercise `write_request`/`read_response`/[`RequestFlight`] honestly
    /// (a fake in-memory reader cannot reproduce "the write succeeded, the
    /// response never arrived, then the future was dropped"), without a
    /// Python interpreter or the `batchalign` package: neither is a
    /// dependency this crate's own unit tests should need.
    ///
    /// `script` runs under `sh -c`; it must print a valid ready line (see
    /// [`Self::read_ready_line`]) as its first line of stdout. Skips the
    /// memory-guard permit and orphan-PID recording `spawn()` does for a
    /// real worker: irrelevant to what these tests assert, and pulling
    /// them in would make a unit test depend on host memory state.
    pub(crate) async fn spawn_stub_for_test(script: &str) -> Self {
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg(script)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        #[allow(clippy::expect_used)]
        let mut child = cmd.spawn().expect("spawn sh stub for test");

        #[allow(clippy::expect_used)]
        let stdout = child.stdout.take().expect("stub stdout captured");
        let mut stdout_reader = BufReader::new(stdout);
        #[allow(clippy::expect_used)]
        let stderr_reader = BufReader::new(child.stderr.take().expect("stub stderr captured"));

        #[allow(clippy::expect_used)]
        let ready = Self::read_ready_line(
            &mut stdout_reader,
            crate::worker::LoadingStall::new(crate::api::PositiveSeconds::literal::<30>()),
        )
        .await
        .and_then(protocol::ReadySignal::into_stdio)
        .expect("stub must emit a valid stdio ready line as its first stdout line");
        let (pid, runtime) = ready.into_parts();

        #[allow(clippy::expect_used)]
        let stdin = child.stdin.take().expect("stub stdin captured");

        let (_stderr_tx, stderr_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        // Drain stderr in the background so a stub that writes anything to
        // it cannot block on a full pipe. Nothing in these tests inspects
        // stub stderr, so the sender is dropped immediately; the drain
        // loop just discards lines until EOF.
        tokio::spawn(async move {
            use tokio::io::AsyncBufReadExt;
            let mut reader = stderr_reader;
            let mut line = String::new();
            while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                line.clear();
            }
        });

        Self {
            config: WorkerConfig::for_tests(),
            child,
            pid,
            runtime,
            stdin,
            stdout: stdout_reader,
            last_activity: tokio::time::Instant::now(),
            loads: crate::worker::pool::TaskLoads::default(),
            stderr_rx,
            request_flight: RequestFlight::Idle,
        }
    }
}

#[cfg(test)]
mod tests;
