//! `SharedGpuChannel`: the one shared GPU transport, generic over the writer.
//!
//! A shared GPU worker serves many V2 requests at once (the Python side runs a
//! `ThreadPoolExecutor`, and GPU inference releases the GIL). Whether its lines
//! travel over a child's stdio or a daemon's TCP socket changes only who owns
//! the process; dispatch, sequential ops, routing and liveness are the same,
//! so they live here once. [`super::SharedGpuWorker`] (stdio) adds process
//! ownership; [`super::SharedGpuTcpWorker`] adds the connection.

use std::time::Duration;

use tokio::io::{AsyncBufRead, AsyncWrite, AsyncWriteExt};
use tokio::sync::{Semaphore, oneshot};
use tracing::{debug, info, warn};

use crate::types::worker_v2::{ExecuteResponseV2, TaskTimeoutOverrides};
use crate::worker::error::{WorkerError, WorkerWait};
use crate::worker::handle::{
    CapabilitiesRequest, EnsureTaskRequest, ReportedFailure, WorkerRequest,
};
use crate::worker::{WorkerCapabilities, WorkerPid};

use super::reader::{Routes, spawn_reader};
use super::routes::{ControlAnswer, Retirement, StreamClosed};

/// How long a shutdown waits for the worker's acknowledgement.
const SHUTDOWN_ACK_LIMIT: Duration = Duration::from_secs(2);

/// One worker's request stream and the routes its replies come back on.
pub(crate) struct SharedGpuChannel<W> {
    /// Serialized writes, so concurrent requests' JSON lines never interleave.
    writer: tokio::sync::Mutex<W>,

    /// Where the reader task delivers replies; also the liveness record.
    routes: Routes,

    /// Serializes sequential ops across their full round trip. The control
    /// slot itself cannot, because the reader must reach it to deliver.
    control_gate: tokio::sync::Mutex<()>,

    /// This worker's on-demand load states, shared by its concurrent callers:
    /// filled by successful `ensure_task` replies, so later calls skip the
    /// round trip (and the single-slot control channel's contention under
    /// concurrent dispatch) and take no second lease.
    loads: tokio::sync::Mutex<crate::worker::pool::TaskLoads>,

    /// The reader task; aborted when the channel is dropped.
    reader_task: tokio::task::JoinHandle<()>,

    /// Worker process id, for logs and job tracking.
    pid: WorkerPid,

    /// The operator's transport-ceiling overrides.
    task_timeouts: TaskTimeoutOverrides,

    /// Bounds in-flight `execute_v2` calls to the worker's
    /// `ThreadPoolExecutor` capacity (`gpu_thread_pool_size`).
    ///
    /// Without it more callers register than the worker can serve, and the
    /// late ones spend their per-request timeout queued inside the worker
    /// (pinned by
    /// `tests/gpu_concurrent_dispatch.rs::gpu_concurrent_dispatch_does_not_charge_queue_wait_against_per_request_timeout`).
    /// The permit is taken BEFORE registration and the timer, so the timer
    /// runs only for work issued to the worker.
    dispatch_semaphore: Semaphore,
}

impl<W: AsyncWrite + Unpin + Send> SharedGpuChannel<W> {
    /// Start routing `reader`'s lines and take `writer` for requests.
    pub(super) fn open<R>(
        reader: R,
        writer: W,
        pid: WorkerPid,
        task_timeouts: TaskTimeoutOverrides,
        gpu_thread_pool_size: u32,
    ) -> Self
    where
        R: AsyncBufRead + Unpin + Send + 'static,
    {
        let routes = Routes::default();
        let reader_task = spawn_reader(reader, routes.clone(), pid);
        Self {
            writer: tokio::sync::Mutex::new(writer),
            routes,
            control_gate: tokio::sync::Mutex::new(()),
            loads: tokio::sync::Mutex::new(crate::worker::pool::TaskLoads::default()),
            reader_task,
            pid,
            task_timeouts,
            dispatch_semaphore: Semaphore::new(super::dispatch_permits_from(gpu_thread_pool_size)),
        }
    }

    /// Whether the worker can take requests: refused once its output stream
    /// has closed (EOF, read error, reader stopped) or shutdown has begun.
    pub(super) fn check_available(&self) -> Result<(), WorkerError> {
        self.routes.pending.liveness()
    }

    /// The worker process ID.
    pub(super) fn pid(&self) -> WorkerPid {
        self.pid
    }

    /// Write one request line under the writer lock.
    async fn write_request(&self, request: &WorkerRequest<'_>) -> Result<(), WorkerError> {
        let mut line = serde_json::to_string(request)
            .map_err(|error| WorkerError::Protocol(format!("failed to encode request: {error}")))?;
        line.push('\n');
        let mut writer = self.writer.lock().await;
        writer.write_all(line.as_bytes()).await?;
        writer.flush().await?;
        Ok(())
    }

    /// Send one V2 request and await its reply.
    ///
    /// Up to `gpu_thread_pool_size` callers run at once; the rest wait for a
    /// permit before their timer starts.
    pub(super) async fn execute_v2(
        &self,
        permit: &crate::worker::pool::RunPermit<'_>,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        let request = permit.request();
        self.check_available()?;
        let _permit = self
            .dispatch_semaphore
            .acquire()
            .await
            .map_err(|_| WorkerError::PoolShuttingDown)?;

        // Registered before the write so the reader can route the reply as
        // soon as it arrives; refused if the stream closed while we waited.
        let receipt = self.routes.pending.register(request.request_id.clone())?;
        if let Err(error) = self
            .write_request(&WorkerRequest::ExecuteV2 { request })
            .await
        {
            self.routes.pending.withdraw(&request.request_id);
            return Err(error);
        }

        let limit = request.transport_timeout(self.task_timeouts);
        match receipt.reply_within(limit.duration()).await {
            Some(reply) => reply,
            None => Err(WorkerError::Timeout {
                waited_for: WorkerWait::ExecuteV2 { task: request.task },
                limit,
            }),
        }
    }

    /// One sequential op's round trip: write the request (armed by the
    /// caller, which put the armed id in it), then wait for the answer under
    /// `wait`. The caller holds the control gate.
    ///
    /// `progress` is the load's own progress (`ArmedLoad`), present for
    /// `ensure_task` only: each report that advanced re-arms a loading wait
    /// (`RunningWait::observe_load`), so a load that downloads its models takes as
    /// long as the download, and only one that goes quiet for its stall
    /// window is [`WorkerError::Stalled`]. A budget is extended by nothing.
    async fn round_trip<T>(
        &self,
        request: &WorkerRequest<'_>,
        mut answer: oneshot::Receiver<ControlAnswer<T>>,
        mut progress: Option<
            tokio::sync::mpsc::UnboundedReceiver<crate::types::worker_v2::LoadProgressV2>,
        >,
        wait: crate::worker::ResponseWait,
    ) -> Result<ControlAnswer<T>, WorkerError> {
        /// What one wait step produced.
        enum Step<T> {
            Answer(Result<ControlAnswer<T>, oneshot::error::RecvError>),
            Progress(crate::types::worker_v2::LoadProgressV2),
        }

        let loading = super::super::checkout_wait::ObservedModelLoad::for_wait(&wait);
        self.write_request(request).await?;
        let mut running = crate::worker::RunningWait::start(wait);
        loop {
            let next_progress = async {
                match progress.as_mut() {
                    Some(receiver) => receiver.recv().await,
                    None => std::future::pending().await,
                }
            };
            let step = running
                .within(async {
                    tokio::select! {
                        answered = &mut answer => Step::Answer(answered),
                        Some(event) = next_progress => Step::Progress(event),
                    }
                })
                .await;
            match step {
                Some(Step::Answer(Ok(answered))) => return Ok(answered),
                // Every route out of the slot answers its waiter (a reply, a
                // failure, or the close reason); a dropped sender means the
                // slot went away with the reader that owned it.
                Some(Step::Answer(Err(_))) => {
                    return Err(StreamClosed::ReaderStopped.to_worker_error());
                }
                Some(Step::Progress(event)) => {
                    running.observe_load(&event);
                    if let Some(loading) = &loading {
                        loading.report(&event);
                    }
                }
                None => return Err(running.expired()),
            }
        }
    }

    /// Query this worker's live capability and engine identity.
    pub(super) async fn capabilities(&self) -> Result<WorkerCapabilities, WorkerError> {
        let _control_guard = self.control_gate.lock().await;
        let armed = self.routes.control.arm_capabilities()?;
        let request = WorkerRequest::Capabilities {
            request: CapabilitiesRequest {
                request_id: &armed.request_id,
            },
        };
        self.round_trip(
            &request,
            armed.answer,
            None,
            crate::worker::ResponseWait::Budget {
                waited_for: WorkerWait::Capabilities,
                limit: crate::worker::CAPABILITY_TIMEOUT,
            },
        )
        .await?
        .map_err(|failure| failure.into_worker_error(ReportedFailure::into_worker_error))
    }

    /// Prepare this worker for `load` (`ensure_task`): the `NotLoaded` to
    /// `Loaded` transition, returning the proof every execute takes.
    ///
    /// Idempotent on the worker side; the channel also records each proof so
    /// a repeated preparation skips the round trip. A load is recorded only
    /// from a reply that names its task (the control slot delivers no other).
    /// The load's host-memory lease is granted under the control gate, after
    /// the second look, so concurrent first requests for one load take one
    /// lease, not one each; the proof owns it.
    pub(super) async fn ensure_task(
        &self,
        load: &crate::worker::pool::TaskLoad,
        preparation: &crate::worker::pool::LoadPreparation,
    ) -> Result<std::sync::Arc<crate::worker::pool::Loaded>, WorkerError> {
        if let Some(loaded) = self.loads.lock().await.get(load) {
            return Ok(loaded);
        }
        let _control_guard = self.control_gate.lock().await;
        // A peer may have completed the load while this call waited.
        if let Some(loaded) = self.loads.lock().await.get(load) {
            return Ok(loaded);
        }
        let admitted = preparation.admission.admit(load, self.pid).await?;
        let task = admitted.load().task();
        let engine_overrides = admitted.load().wire_overrides();
        let super::routes::ArmedLoad {
            request_id,
            answer,
            progress,
        } = self.routes.control.arm_ensure_task(task)?;
        let request = WorkerRequest::EnsureTask {
            request: EnsureTaskRequest {
                request_id: &request_id,
                task,
                engine_overrides,
            },
        };
        let response = self
            .round_trip(
                &request,
                answer,
                Some(progress),
                crate::worker::ResponseWait::Loading {
                    waited_for: crate::worker::LoadingWait::EnsureTask { task },
                    stall: preparation.stall,
                },
            )
            .await?
            .map_err(|failure| {
                failure.into_worker_error(ReportedFailure::into_ensure_task_error)
            })?;
        info!(
            pid = %self.pid,
            task = ?response.task,
            status = %response.status,
            elapsed_s = response.elapsed_s.get(),
            "ensure_task completed"
        );
        // Settled before the lock, so concurrent fast-path lookups never wait
        // behind the ledger write.
        let confirmed = admitted.confirm().await;
        Ok(self.loads.lock().await.record(confirmed))
    }

    /// Begin stopping the worker for `why`: refuse new requests, answer the
    /// in-flight ones with `why`'s error, and ask the worker to shut down.
    /// Returns the acknowledgement to await, or `None` when shutdown already
    /// began, the stream had closed, or the request could not be written.
    pub(super) async fn begin_shutdown(
        &self,
        why: Retirement,
    ) -> Option<oneshot::Receiver<ControlAnswer<()>>> {
        if !self.routes.pending.close(StreamClosed::Stopped(why)) {
            return None;
        }
        let ack = self.routes.control.begin_shutdown(why)?;
        match self.write_request(&WorkerRequest::Shutdown).await {
            Ok(()) => Some(ack),
            Err(error) => {
                warn!(pid = %self.pid, %error, "Failed to write shared GPU shutdown request");
                None
            }
        }
    }

    /// Wait for a shutdown acknowledgement, logging what came back.
    pub(super) async fn await_shutdown_ack(&self, ack: oneshot::Receiver<ControlAnswer<()>>) {
        match tokio::time::timeout(SHUTDOWN_ACK_LIMIT, ack).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(failure))) => warn!(
                pid = %self.pid,
                error = %failure.message(),
                "Shared GPU worker answered shutdown with a failure"
            ),
            Ok(Err(_)) => debug!(pid = %self.pid, "Shared GPU shutdown ack channel closed"),
            Err(_) => debug!(pid = %self.pid, "Timed out waiting for shared GPU shutdown ack"),
        }
    }

    /// Stop reading and close both routes for `why`; every waiter is
    /// answered.
    pub(super) fn close(&self, why: Retirement) {
        self.reader_task.abort();
        self.routes.pending.close(StreamClosed::Stopped(why));
        self.routes.control.close(StreamClosed::Stopped(why));
    }
}

impl<W> Drop for SharedGpuChannel<W> {
    fn drop(&mut self) {
        // Aborting the task drops its close-on-exit guard, which closes the
        // routes; the explicit close covers a task that already finished. A
        // channel drops only once nothing holds its worker, so no request is
        // waiting to hear the reason.
        self.reader_task.abort();
        let why = StreamClosed::Stopped(Retirement::WorkerRetired);
        self.routes.pending.close(why.clone());
        self.routes.control.close(why);
    }
}
