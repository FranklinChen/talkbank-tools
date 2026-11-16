//! Request/response IPC methods for [`WorkerHandle`].
//!
//! All communication with the Python worker child process flows through
//! these methods: writing JSON-lines requests to stdin, reading JSON-lines
//! responses from stdout, and handling timeouts, noise lines, and crashes.

use crate::types::worker_v2::{ExecuteResponseV2, ProgressEventV2};
use crate::worker::error::{WorkerError, WorkerWait};
use crate::worker::{
    BatchInferRequest, BatchInferResponse, InferRequest, InferResponse, WorkerCapabilities,
    WorkerHealthResponse,
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use tracing::instrument;

use super::WorkerHandle;
use super::protocol::{
    NoiseLimit, NoiseRun, WireLine, WorkerRequest, WorkerResponse, dump_failed_ipc_request, excerpt,
};
use serde::Deserialize;

impl WorkerHandle {
    async fn control<O: super::control::ControlOperation>(
        &mut self,
        operation: O,
    ) -> Result<O::Reply, WorkerError> {
        self.write_request(&operation.request()).await?;
        let completed = super::read_skipping_progress(self, operation, None).await?;
        self.request_flight = super::RequestFlight::Idle;
        completed.into_result()
    }

    /// Serialize and write a JSON-lines request to the worker's stdin.
    #[instrument(skip_all, fields(pid = %self.pid))]
    pub(super) async fn write_request(
        &mut self,
        request: &WorkerRequest<'_>,
    ) -> Result<(), WorkerError> {
        let mut line = serde_json::to_string(request)
            .map_err(|e| WorkerError::Protocol(format!("failed to encode request: {e}")))?;
        line.push('\n');
        // Set BEFORE the write, not after: a write that itself fails or is
        // interrupted partway (dropped future) can leave the worker's
        // stdin parser desynchronized too, so it is not safe to reuse
        // either. Only a fully-read terminal response clears this (see
        // `RequestFlight`).
        self.request_flight = super::RequestFlight::InFlight;
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    /// Read and parse a single JSON-lines response from the worker's stdout.
    ///
    /// Skips blank lines and noise (up to
    /// [`MAX_RESPONSE_STDOUT_NOISE_LINES`] in a row), by the one line rule
    /// ([`WireLine`]). On pipe errors or EOF, drains stderr for diagnostic
    /// output before returning the error.
    #[instrument(skip_all, fields(pid = %self.pid))]
    pub(super) async fn read_response(&mut self) -> Result<WorkerResponse, WorkerError> {
        // One response per call, so a run of noise ends with the call.
        let mut noise = NoiseRun::default();

        loop {
            let mut line = String::new();
            let bytes = match self.stdout.read_line(&mut line).await {
                Ok(b) => b,
                Err(io_err) => {
                    // Pipe error (BrokenPipe, etc.), worker likely crashed.
                    // Drain stderr to capture the Python traceback before
                    // returning the error. Without this, BrokenPipe errors
                    // produce no diagnostic information.
                    let stderr = self.drain_stderr_tail(50);
                    let code = self.child.try_wait().ok().flatten().and_then(|s| s.code());
                    return Err(WorkerError::ProcessExited {
                        code,
                        stderr: stderr.or_else(|| Some(format!("I/O error: {io_err}"))),
                    });
                }
            };
            if bytes == 0 {
                let code = self.child.try_wait().ok().flatten().and_then(|s| s.code());
                let stderr = self.drain_stderr_tail(50);
                return Err(WorkerError::ProcessExited { code, stderr });
            }

            match WireLine::classify(&line) {
                WireLine::Blank => {}
                WireLine::Message(message) => {
                    return WorkerResponse::deserialize(&message).map_err(|e| {
                        WorkerError::Protocol(format!(
                            "failed to decode response: {e} (line: {})",
                            excerpt(line.trim())
                        ))
                    });
                }
                WireLine::Noise => noise.noise(&line).map_err(NoiseLimit::into_worker_error)?,
            }
        }
    }

    /// Check if the worker is healthy.
    pub async fn health_check(&mut self) -> Result<WorkerHealthResponse, WorkerError> {
        self.control(super::control::Health).await
    }

    /// Prepare this worker for `load`: the `NotLoaded` to `Loaded`
    /// transition, returning the proof every execute takes.
    ///
    /// Unless this worker already completed the load (its proof is returned
    /// with no round trip and no second lease), the host grants the load's
    /// memory lease first (queued while the host cannot fit it), then the
    /// `ensure_task` IPC message is sent, and the proof recorded from the
    /// reply owns the lease. A load that fails releases its lease. The reply
    /// wait is a [`crate::worker::LoadingStall`] window that the worker's
    /// advancing progress re-arms, so a model download takes as long as it
    /// takes and only a worker that goes quiet is stalled.
    pub async fn ensure_task(
        &mut self,
        load: &crate::worker::pool::TaskLoad,
        preparation: &crate::worker::pool::LoadPreparation,
    ) -> Result<std::sync::Arc<crate::worker::pool::Loaded>, WorkerError> {
        if let Some(loaded) = self.loads.get(load) {
            return Ok(loaded);
        }
        let admitted = preparation.admission.admit(load, self.pid).await?;
        let loaded = self
            .control(super::control::Loading::new(admitted, preparation.stall))
            .await?;
        let response = loaded.response();
        tracing::info!(pid = %self.pid, task = ?response.task, status = %response.status,
            elapsed_s = response.elapsed_s.get(), "ensure_task completed (sequential worker)");
        Ok(self.loads.record(loaded.confirm().await))
    }

    /// Send a single inference request (CHAT-divorced protocol).
    ///
    /// The server owns all CHAT operations; this sends only structured
    /// payloads (words, lang) and receives structured results (mor, gra).
    pub async fn infer(&mut self, request: &InferRequest) -> Result<InferResponse, WorkerError> {
        self.last_activity = tokio::time::Instant::now();
        self.control(super::control::Infer::new(request)).await
    }

    /// Send a batched inference request (multiple items, one model call).
    ///
    /// Pools multiple utterances into a single NLP call for efficiency.
    pub async fn batch_infer(
        &mut self,
        request: &BatchInferRequest,
    ) -> Result<BatchInferResponse, WorkerError> {
        self.last_activity = tokio::time::Instant::now();
        self.control(super::control::BatchInfer::new(request)).await
    }

    /// Send one typed worker-protocol V2 execute request.
    ///
    /// This keeps the live FA migration on the same long-lived worker process
    /// and stdio transport while replacing the request/response payload shape
    /// with the staged V2 contract.
    /// Send an `execute_v2` request and return the final response.
    ///
    /// Any `ProgressV2` events emitted by the worker before the final
    /// response are silently discarded.  Use [`execute_v2_with_progress`]
    /// to receive them.
    pub async fn execute_v2(
        &mut self,
        prepared: crate::worker::pool::ReadyRequest,
        preparation: &crate::worker::pool::LoadPreparation,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        self.execute_v2_with_progress(prepared, preparation, None)
            .await
    }

    /// Send an `execute_v2` request, forwarding intermediate progress events
    /// through an optional async channel.
    ///
    /// The worker may emit zero or more `ProgressV2` JSON lines before the
    /// final `ExecuteV2` response.  Each progress event is sent through
    /// `progress_tx` (if provided) without blocking the read loop.  If the
    /// channel is full or closed, progress events are dropped silently
    /// losing a progress update is not an error.
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

        // Capture serialized request for failure dumps (before sending).
        let request_json_for_dump = serde_json::to_string(&WorkerRequest::ExecuteV2 { request })
            .unwrap_or_else(|_| "<serialization failed>".into());

        self.write_request(&WorkerRequest::ExecuteV2 { request })
            .await?;

        let limit = request.transport_timeout(self.config.task_timeouts);
        let deadline = crate::worker::Deadline::after(limit);

        // Read loop: consume progress events until the final response arrives.
        loop {
            let response = match deadline.within(self.read_response()).await {
                Some(read) => read?,
                None => {
                    let err = WorkerError::Timeout {
                        waited_for: WorkerWait::ExecuteV2 { task: request.task },
                        limit,
                    };
                    dump_failed_ipc_request(
                        self.pid,
                        &self.config,
                        &request_json_for_dump,
                        &err,
                        None,
                    );
                    return Err(err);
                }
            };

            match response {
                WorkerResponse::ProgressV2 { event } => {
                    // Forward to the progress channel.  Drop silently if the
                    // receiver is gone or the channel is full.
                    if let Some(tx) = progress_tx {
                        let _ = tx.try_send(event);
                    }
                    continue;
                }
                WorkerResponse::ExecuteV2 { response } => {
                    // Only this request's admitted terminal reply permits reuse.
                    // A foreign or wrong-task reply leaves the checkout in flight,
                    // so its consuming Drop path retires the uncertain stream.
                    let response = permit.finish(Ok(response))?;
                    self.request_flight = super::RequestFlight::Idle;
                    return Ok(response);
                }
                WorkerResponse::Error(failure) => {
                    // Also a complete, well-formed terminal message: the
                    // stream is not desynchronized, only the WORK failed.
                    self.request_flight = super::RequestFlight::Idle;
                    let err = failure.into_worker_error();
                    dump_failed_ipc_request(
                        self.pid,
                        &self.config,
                        &request_json_for_dump,
                        &err,
                        None,
                    );
                    return Err(err);
                }
                other => {
                    // Same reasoning: a full line was read and parsed, it
                    // just didn't match an expected variant.
                    self.request_flight = super::RequestFlight::Idle;
                    let err = WorkerError::Protocol(format!(
                        "unexpected response for execute_v2: {other:?}"
                    ));
                    dump_failed_ipc_request(
                        self.pid,
                        &self.config,
                        &request_json_for_dump,
                        &err,
                        Some(&format!("{other:?}")),
                    );
                    return Err(err);
                }
            }
        }
    }

    /// Query the worker's capabilities.
    pub async fn capabilities(&mut self) -> Result<WorkerCapabilities, WorkerError> {
        self.control(super::control::Capabilities::new()).await
    }
}
