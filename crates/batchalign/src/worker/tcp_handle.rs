//! `TcpWorkerHandle`: connects to a pre-started TCP worker daemon.
//!
//! Unlike [`WorkerHandle`](super::handle::WorkerHandle) which spawns and owns a
//! child process, `TcpWorkerHandle` connects to a worker that is already running
//! as a persistent daemon listening on a TCP port. The same JSON-lines protocol
//! is used: the only difference is the transport layer.
//!
//! # Lifecycle
//!
//! - `TcpWorkerHandle` does **not** own the worker process. Dropping the handle
//!   disconnects the TCP stream but does not kill the worker.
//! - If the connection drops, [`reconnect()`](TcpWorkerHandle::reconnect) tries
//!   to re-establish it before failing the request.
//! - Shutdown sends the `{"op":"shutdown"}` message but does not SIGKILL, the
//!   worker daemon is managed by launchd/systemd, not Rust.

use std::time::Duration;

use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tracing::{debug, info, warn};

use crate::api::WorkerLanguage;
use crate::types::worker_v2::{ExecuteResponseV2, ProgressEventV2};
use crate::worker::error::{WorkerError, WorkerWait};
use crate::worker::handle::{
    NoiseLimit, NoiseRun, WireLine, WorkerRequest, WorkerResponse, excerpt,
};
use crate::worker::{
    BatchInferRequest, BatchInferResponse, InferRequest, InferResponse, WorkerCapabilities,
    WorkerHealthResponse, WorkerPid, WorkerProfile,
};

/// Connect to a worker daemon at `addr`, within [`crate::worker::CONNECT_TIMEOUT`].
///
/// Shared by the sequential handle and the shared GPU TCP worker.
pub(crate) async fn connect_within(addr: &str) -> Result<TcpStream, WorkerError> {
    let limit = crate::worker::CONNECT_TIMEOUT;
    match tokio::time::timeout(limit.duration(), TcpStream::connect(addr)).await {
        Ok(Ok(stream)) => Ok(stream),
        Ok(Err(error)) => Err(WorkerError::Protocol(format!(
            "failed to connect to TCP worker at {addr}: {error}"
        ))),
        Err(_) => Err(WorkerError::Timeout {
            waited_for: WorkerWait::Connect {
                addr: addr.to_owned(),
            },
            limit,
        }),
    }
}

/// Metadata about a discovered TCP worker (from registry).
#[derive(Debug, Clone)]
pub struct TcpWorkerInfo {
    /// Host address (usually 127.0.0.1).
    pub host: String,
    /// TCP port.
    pub port: u16,
    /// Worker profile.
    pub profile: WorkerProfile,
    /// Worker-runtime language string.
    pub lang: WorkerLanguage,
    /// Engine overrides JSON string.
    pub engine_overrides: String,
    /// Worker process ID (from registry, for display).
    pub pid: WorkerPid,
    /// The operator's transport-ceiling overrides; each absent one leaves the
    /// task's built-in ceiling.
    pub task_timeouts: crate::types::worker_v2::TaskTimeoutOverrides,
}

/// Manages a TCP connection to a pre-started Python worker daemon.
///
/// Uses the same JSON-lines protocol as [`WorkerHandle`] but over TCP instead
/// of stdio pipes. Does not own the worker process, dropping disconnects but
/// does not kill.
pub struct TcpWorkerHandle {
    info: TcpWorkerInfo,
    reader: BufReader<tokio::io::ReadHalf<TcpStream>>,
    writer: tokio::io::WriteHalf<TcpStream>,
    /// Monotonic instant when the last request was dispatched.
    last_activity: tokio::time::Instant,
    /// This connection's on-demand load states. The daemon answers a load it
    /// already completed at once; a fresh connection asks again.
    loads: crate::worker::pool::TaskLoads,
    /// Last field keeps adoption owned until this connection's resources drop.
    ownership: super::registry::ConnectionOwnership,
}

/// The original buffered reader retains adoption through its actual reader task.
/// Its reader drops before its claim on ordinary completion and cancellation.
pub(super) struct AdoptedTcpReader {
    reader: BufReader<tokio::io::ReadHalf<TcpStream>>,
    _ownership: super::registry::ConnectionOwnership,
}
impl tokio::io::AsyncRead for AdoptedTcpReader {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.get_mut().reader).poll_read(cx, buf)
    }
}
impl tokio::io::AsyncBufRead for AdoptedTcpReader {
    fn poll_fill_buf(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<&[u8]>> {
        std::pin::Pin::new(&mut self.get_mut().reader).poll_fill_buf(cx)
    }
    fn consume(self: std::pin::Pin<&mut Self>, amount: usize) {
        std::pin::Pin::new(&mut self.get_mut().reader).consume(amount);
    }
}
pub(super) struct TcpConnectionParts {
    pub(super) info: TcpWorkerInfo,
    pub(super) reader: AdoptedTcpReader,
    pub(super) writer: tokio::io::WriteHalf<TcpStream>,
}

impl TcpWorkerHandle {
    /// Connect to an existing TCP worker.
    pub async fn connect(info: TcpWorkerInfo) -> Result<Self, WorkerError> {
        Self::connect_owned(info, super::registry::ConnectionOwnership::independent())
            .await
            .map_err(|(error, _)| error)
    }
    pub(super) async fn connect_owned(
        info: TcpWorkerInfo,
        ownership: super::registry::ConnectionOwnership,
    ) -> Result<Self, (WorkerError, super::registry::ConnectionOwnership)> {
        let addr = format!("{}:{}", info.host, info.port);
        info!(
            host = %info.host,
            port = info.port,
            profile = %info.profile.label(),
            lang = %info.lang,
            pid = %info.pid,
            "Connecting to TCP worker"
        );

        let stream = match connect_within(&addr).await {
            Ok(stream) => stream,
            Err(error) => return Err((error, ownership)),
        };

        let (read_half, write_half) = tokio::io::split(stream);

        Ok(Self {
            info,
            ownership,
            reader: BufReader::new(read_half),
            writer: write_half,
            last_activity: tokio::time::Instant::now(),
            loads: crate::worker::pool::TaskLoads::default(),
        })
    }

    /// Close the native socket before handing its claim to failure handling.
    pub(super) fn close(self) -> super::registry::ConnectionOwnership {
        self.ownership
    }

    /// Only a report-admitted registry owner can promote its existing socket.
    pub(super) fn promote_registry(
        worker: super::registry::AdmittedRegistryWorker,
    ) -> (crate::worker::pool::WorkerKey, TcpConnectionParts) {
        let (key, handle) = worker.into_sequential();
        let Self {
            info,
            reader,
            writer,
            ownership,
            ..
        } = handle;
        (
            key,
            TcpConnectionParts {
                info,
                reader: AdoptedTcpReader {
                    reader,
                    _ownership: ownership,
                },
                writer,
            },
        )
    }

    /// Reconnect to the worker after a connection drop.
    pub async fn reconnect(&mut self) -> Result<(), WorkerError> {
        let addr = format!("{}:{}", self.info.host, self.info.port);
        debug!(addr = %addr, "Reconnecting to TCP worker");

        let stream = connect_within(&addr).await?;

        let (read_half, write_half) = tokio::io::split(stream);
        self.reader = BufReader::new(read_half);
        self.writer = write_half;
        Ok(())
    }

    async fn write_request(&mut self, request: &WorkerRequest<'_>) -> Result<(), WorkerError> {
        let mut line = serde_json::to_string(request)
            .map_err(|e| WorkerError::Protocol(format!("failed to encode request: {e}")))?;
        line.push('\n');

        match self.writer.write_all(line.as_bytes()).await {
            Ok(()) => {}
            Err(e) => {
                // Try reconnect once before failing.
                warn!(error = %e, "TCP write failed, attempting reconnect");
                self.reconnect().await?;
                self.writer.write_all(line.as_bytes()).await?;
            }
        }
        self.writer.flush().await?;
        Ok(())
    }

    async fn read_response(&mut self) -> Result<WorkerResponse, WorkerError> {
        // One response per call, so a run of noise ends with the call.
        let mut noise = NoiseRun::default();

        loop {
            let mut line = String::new();
            let bytes = self.reader.read_line(&mut line).await?;
            if bytes == 0 {
                // The daemon closed the connection: the request lost its
                // worker, as when a shared GPU stream ends (retryable).
                return Err(WorkerError::ProcessExited {
                    code: None,
                    stderr: Some("TCP worker closed the connection (EOF)".into()),
                });
            }

            match WireLine::classify(&line) {
                WireLine::Blank => {}
                WireLine::Message(message) => {
                    return WorkerResponse::deserialize(&message).map_err(|e| {
                        WorkerError::Protocol(format!(
                            "failed to decode TCP response: {e} (line: {})",
                            excerpt(line.trim())
                        ))
                    });
                }
                WireLine::Noise => noise.noise(&line).map_err(NoiseLimit::into_worker_error)?,
            }
        }
    }

    /// The shared progress loop
    /// (`crate::worker::handle::read_skipping_progress`) on this connection.
    async fn control<O: crate::worker::handle::control::ControlOperation>(
        &mut self,
        operation: O,
    ) -> Result<O::Reply, WorkerError> {
        self.write_request(&operation.request()).await?;
        crate::worker::handle::read_skipping_progress(self, operation, None)
            .await?
            .into_result()
    }

    /// Check if the worker is healthy.
    pub async fn health_check(&mut self) -> Result<WorkerHealthResponse, WorkerError> {
        self.control(crate::worker::handle::control::Health).await
    }

    /// Send a single inference request.
    pub async fn infer(&mut self, request: &InferRequest) -> Result<InferResponse, WorkerError> {
        self.last_activity = tokio::time::Instant::now();
        self.control(crate::worker::handle::control::Infer::new(request))
            .await
    }

    /// Send a batched inference request.
    pub async fn batch_infer(
        &mut self,
        request: &BatchInferRequest,
    ) -> Result<BatchInferResponse, WorkerError> {
        self.last_activity = tokio::time::Instant::now();
        self.control(crate::worker::handle::control::BatchInfer::new(request))
            .await
    }

    /// Send one typed V2 execute request.
    pub async fn execute_v2(
        &mut self,
        prepared: crate::worker::pool::ReadyRequest,
        preparation: &crate::worker::pool::LoadPreparation,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        self.execute_v2_with_progress(prepared, preparation, None)
            .await
    }

    /// Send an `execute_v2` request over TCP, forwarding intermediate
    /// progress events through an optional async channel.
    pub async fn execute_v2_with_progress(
        &mut self,
        prepared: crate::worker::pool::ReadyRequest,
        preparation: &crate::worker::pool::LoadPreparation,
        progress_tx: Option<&tokio::sync::mpsc::Sender<ProgressEventV2>>,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        let permit = preparation.prepare_on(prepared, self).await?;
        self.execute_admitted(permit, progress_tx).await
    }

    pub(in crate::worker) async fn execute_admitted(
        &mut self,
        permit: crate::worker::pool::RunPermit,
        progress_tx: Option<&tokio::sync::mpsc::Sender<ProgressEventV2>>,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        let request = permit.request();
        self.last_activity = tokio::time::Instant::now();
        self.write_request(&WorkerRequest::ExecuteV2 { request })
            .await?;

        let limit = request.transport_timeout(self.info.task_timeouts);
        let deadline = crate::worker::Deadline::after(limit);

        loop {
            let response = match deadline.within(self.read_response()).await {
                Some(read) => read?,
                None => {
                    return Err(WorkerError::Timeout {
                        waited_for: WorkerWait::ExecuteV2 { task: request.task },
                        limit,
                    });
                }
            };

            match response {
                WorkerResponse::ProgressV2 { event } => {
                    if let Some(tx) = progress_tx {
                        let _ = tx.try_send(event);
                    }
                    continue;
                }
                WorkerResponse::ExecuteV2 { response } => return permit.finish(Ok(response)),
                WorkerResponse::Error(failure) => return Err(failure.into_worker_error()),
                other => {
                    return Err(WorkerError::Protocol(format!(
                        "unexpected TCP response for execute_v2: {other:?}"
                    )));
                }
            }
        }
    }

    /// Query worker capabilities.
    pub async fn capabilities(&mut self) -> Result<WorkerCapabilities, WorkerError> {
        self.control(crate::worker::handle::control::Capabilities::new())
            .await
    }

    /// Prepare the daemon for `load` (`ensure_task`): the `NotLoaded` to
    /// `Loaded` transition, returning the proof every execute takes. The
    /// load's host-memory lease is granted first and owned by the proof, as
    /// for a stdio worker. The lease lives with this connection's proof, not
    /// with the daemon: a reconnection asks again, the daemon answers at once,
    /// and the new proof takes a new lease.
    pub async fn ensure_task(
        &mut self,
        load: &crate::worker::pool::TaskLoad,
        preparation: &crate::worker::pool::LoadPreparation,
    ) -> Result<std::sync::Arc<crate::worker::pool::Loaded>, WorkerError> {
        if let Some(loaded) = self.loads.get(load) {
            return Ok(loaded);
        }
        let admitted = preparation.admission.admit(load, self.info.pid).await?;
        let loaded = self
            .control(crate::worker::handle::control::Loading::new(
                admitted,
                preparation.stall,
            ))
            .await?;
        let response = loaded.response();
        tracing::info!(task = ?response.task, status = %response.status,
            elapsed_s = response.elapsed_s.get(), "ensure_task completed (TCP worker)");
        Ok(self.loads.record(loaded.confirm().await))
    }

    /// Send shutdown message (does not kill process, daemon manager handles that).
    pub async fn shutdown(&mut self) -> Result<(), WorkerError> {
        info!(
            host = %self.info.host,
            port = self.info.port,
            pid = %self.info.pid,
            "Sending shutdown to TCP worker"
        );

        let _ = self.write_request(&WorkerRequest::Shutdown).await;
        let _ = tokio::time::timeout(Duration::from_secs(2), self.read_response()).await;
        Ok(())
    }

    /// The PID of the worker process (from registry).
    pub fn pid(&self) -> WorkerPid {
        self.info.pid
    }

    /// The profile label this worker handles.
    pub fn profile_label(&self) -> &'static str {
        self.info.profile.label()
    }

    /// The language this worker handles.
    pub fn lang(&self) -> &str {
        self.info.lang.as_worker_arg()
    }

    /// The transport this worker uses.
    pub fn transport(&self) -> &'static str {
        "tcp"
    }

    /// Duration since the last request was dispatched.
    pub fn idle_duration(&self) -> Duration {
        self.last_activity.elapsed()
    }

    /// The TCP connection info.
    pub fn info(&self) -> &TcpWorkerInfo {
        &self.info
    }
}
impl crate::worker::handle::ResponseSource for TcpWorkerHandle {
    async fn read_response(&mut self) -> Result<WorkerResponse, WorkerError> {
        TcpWorkerHandle::read_response(self).await
    }
}
