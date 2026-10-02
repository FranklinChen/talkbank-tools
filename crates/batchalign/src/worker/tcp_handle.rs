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

use crate::api::{PositiveSeconds, WorkerLanguage};
use crate::types::worker_v2::{
    ExecuteRequestV2, ExecuteResponseV2, ProgressEventV2, batched_items_timeout,
};
use crate::worker::error::{WorkerError, WorkerWait};
use crate::worker::handle::{
    EnsureTaskRequest, NoiseLimit, NoiseRun, WireLine, WorkerRequest, WorkerResponse, excerpt,
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
    /// Python worker's `ThreadPoolExecutor(max_workers=...)` capacity for
    /// concurrent V2 dispatch. Used by `SharedGpuTcpWorker` to cap in-flight
    /// `execute_v2` calls so per-request timeouts never count queue-wait
    /// behind earlier requests. Ignored by `TcpWorkerHandle` (which serves
    /// one request at a time anyway). Daemons are spawned with
    /// `--gpu-thread-pool-size`; pool callers should pass the same value
    /// they used at spawn (or registry-discovered).
    pub gpu_thread_pool_size: u32,
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
}

impl TcpWorkerHandle {
    /// Connect to an existing TCP worker.
    pub async fn connect(info: TcpWorkerInfo) -> Result<Self, WorkerError> {
        let addr = format!("{}:{}", info.host, info.port);
        info!(
            host = %info.host,
            port = info.port,
            profile = %info.profile.label(),
            lang = %info.lang,
            pid = %info.pid,
            "Connecting to TCP worker"
        );

        let stream = connect_within(&addr).await?;

        let (read_half, write_half) = tokio::io::split(stream);

        Ok(Self {
            info,
            reader: BufReader::new(read_half),
            writer: write_half,
            last_activity: tokio::time::Instant::now(),
        })
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

    /// Twin of `WorkerHandle::read_response_skipping_progress_via_self`
    /// for the TCP transport. Skip every `ProgressV2` event, optionally
    /// forward to a channel, return the first non-progress response, or
    /// time out at the deadline. See `worker/handle/progress_preamble.rs`
    /// for the bug-class rationale and the unit-tested static twin.
    async fn read_response_skipping_progress(
        &mut self,
        waited_for: WorkerWait,
        limit: PositiveSeconds,
        progress_tx: Option<&tokio::sync::mpsc::Sender<ProgressEventV2>>,
    ) -> Result<WorkerResponse, WorkerError> {
        let deadline = crate::worker::Deadline::after(limit);
        loop {
            let response = match deadline.within(self.read_response()).await {
                Some(read) => read?,
                None => return Err(WorkerError::Timeout { waited_for, limit }),
            };

            match response {
                WorkerResponse::ProgressV2 { event } => {
                    if let Some(tx) = progress_tx {
                        let _ = tx.try_send(event);
                    }
                    continue;
                }
                other => return Ok(other),
            }
        }
    }

    /// Check if the worker is healthy.
    pub async fn health_check(&mut self) -> Result<WorkerHealthResponse, WorkerError> {
        self.write_request(&WorkerRequest::Health).await?;

        let response = self
            .read_response_skipping_progress(
                WorkerWait::Health,
                crate::worker::HEALTH_TIMEOUT,
                None,
            )
            .await?;

        match response {
            WorkerResponse::Health { response } => {
                if !response.status.is_ok() {
                    return Err(WorkerError::HealthCheckFailed(format!(
                        "status={}",
                        response.status
                    )));
                }
                Ok(response)
            }
            WorkerResponse::Error(failure) => Err(failure.into_health_error()),
            other => Err(WorkerError::HealthCheckFailed(format!(
                "unexpected TCP response for health: {other:?}"
            ))),
        }
    }

    /// Send a single inference request.
    pub async fn infer(&mut self, request: &InferRequest) -> Result<InferResponse, WorkerError> {
        self.last_activity = tokio::time::Instant::now();
        self.write_request(&WorkerRequest::Infer { request })
            .await?;

        let response = self
            .read_response_skipping_progress(WorkerWait::Infer, crate::worker::INFER_TIMEOUT, None)
            .await?;

        match response {
            WorkerResponse::Infer { response } => Ok(response),
            WorkerResponse::Error(failure) => Err(failure.into_worker_error()),
            other => Err(WorkerError::Protocol(format!(
                "unexpected TCP response for infer: {other:?}"
            ))),
        }
    }

    /// Send a batched inference request.
    pub async fn batch_infer(
        &mut self,
        request: &BatchInferRequest,
    ) -> Result<BatchInferResponse, WorkerError> {
        self.last_activity = tokio::time::Instant::now();
        self.write_request(&WorkerRequest::BatchInfer { request })
            .await?;

        let items = request.items.len();
        let response = self
            .read_response_skipping_progress(
                WorkerWait::BatchInfer { items },
                batched_items_timeout(items as u64),
                None,
            )
            .await?;

        match response {
            WorkerResponse::BatchInfer { response } => Ok(response),
            WorkerResponse::Error(failure) => Err(failure.into_worker_error()),
            other => Err(WorkerError::Protocol(format!(
                "unexpected TCP response for batch_infer: {other:?}"
            ))),
        }
    }

    /// Send one typed V2 execute request.
    pub async fn execute_v2(
        &mut self,
        request: &ExecuteRequestV2,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        self.execute_v2_with_progress(request, None).await
    }

    /// Send an `execute_v2` request over TCP, forwarding intermediate
    /// progress events through an optional async channel.
    pub async fn execute_v2_with_progress(
        &mut self,
        request: &ExecuteRequestV2,
        progress_tx: Option<&tokio::sync::mpsc::Sender<ProgressEventV2>>,
    ) -> Result<ExecuteResponseV2, WorkerError> {
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
                WorkerResponse::ExecuteV2 { response } => return Ok(response),
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
        self.write_request(&WorkerRequest::Capabilities {
            request: crate::worker::handle::CapabilitiesRequest {
                request_id: &crate::worker::handle::ControlRequestId::next(),
            },
        })
        .await?;

        // Same protocol contract as the stdio handle: tolerate
        // progress_v2 preamble during cold-cache catalog/model loads.
        let response = self
            .read_response_skipping_progress(
                WorkerWait::Capabilities,
                crate::worker::CAPABILITY_TIMEOUT,
                None,
            )
            .await?;

        match response {
            WorkerResponse::Capabilities { response } => Ok(response),
            WorkerResponse::Error(failure) => Err(failure.into_worker_error()),
            other => Err(WorkerError::Protocol(format!(
                "unexpected TCP response for capabilities: {other:?}"
            ))),
        }
    }

    /// Load one task in a lazy-profile TCP worker.
    pub async fn ensure_task(
        &mut self,
        task: crate::worker::InferTask,
        engine_overrides: Option<&std::collections::BTreeMap<String, String>>,
        timeout: crate::api::PositiveSeconds,
    ) -> Result<(), WorkerError> {
        self.write_request(&WorkerRequest::EnsureTask {
            request: EnsureTaskRequest {
                request_id: &crate::worker::handle::ControlRequestId::next(),
                task,
                engine_overrides,
            },
        })
        .await?;
        let response = self
            .read_response_skipping_progress(WorkerWait::EnsureTask { task }, timeout, None)
            .await?;
        match response {
            WorkerResponse::EnsureTask { response } => {
                let response = response.about(task)?;
                tracing::info!(
                    task = ?response.task,
                    status = %response.status,
                    elapsed_s = response.elapsed_s.get(),
                    "ensure_task completed (TCP worker)"
                );
                Ok(())
            }
            WorkerResponse::Error(failure) => Err(failure.into_ensure_task_error()),
            other => Err(WorkerError::Protocol(format!(
                "unexpected TCP response for ensure_task: {other:?}"
            ))),
        }
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
