//! `SharedGpuWorker`: a shared GPU worker this server spawned and owns, over
//! its stdio.
//!
//! Unlike [`CheckedOutWorker`](super::super::CheckedOutWorker), which grants
//! exclusive access, many V2 requests run on one `SharedGpuWorker` at once.
//! Everything about the request stream is the shared
//! [`SharedGpuChannel`](super::channel::SharedGpuChannel); this wrapper adds
//! ownership of the child process: the worker is retired by asking it to shut
//! down, then terminating its process group.

use std::time::Duration;

use tokio::process::{Child, ChildStdin};
use tracing::{info, instrument, warn};

use crate::types::worker_v2::{ExecuteRequestV2, ExecuteResponseV2};
use crate::worker::WorkerPid;
use crate::worker::error::WorkerError;
use crate::worker::handle::{WorkerConfig, WorkerHandle};

use super::channel::SharedGpuChannel;

/// A GPU worker that serves V2 requests concurrently over its stdio.
///
/// Created from a [`WorkerHandle`] by taking its stdio; the worker process
/// runs Python's `_serve_stdio_concurrent()`.
pub(crate) struct SharedGpuWorker {
    /// Owned child process: this wrapper supervises shutdown and kill.
    child: tokio::sync::Mutex<Option<Child>>,

    /// The request stream and its replies.
    channel: SharedGpuChannel<ChildStdin>,

    /// Worker configuration (for logs).
    config: WorkerConfig,
}

impl SharedGpuWorker {
    /// Take over a spawned [`WorkerHandle`]: its stdio becomes the channel and
    /// this wrapper becomes the child's lifecycle owner (the handle's own
    /// `Drop`, which would kill the child, is bypassed).
    pub(in crate::worker::pool) async fn from_handle(handle: WorkerHandle) -> Self {
        let pid = handle.pid();
        let config = handle.config().clone();
        let parts = handle.into_parts();
        // The pool records this identity before converting the handle. Move it
        // out explicitly so `ManuallyDrop` ownership remains complete.
        let _runtime = parts.runtime;
        let channel = SharedGpuChannel::open(
            parts.stdout,
            parts.stdin,
            pid,
            config.task_timeouts,
            config.runtime.gpu_thread_pool_size,
        );
        Self {
            child: tokio::sync::Mutex::new(Some(parts.child)),
            channel,
            config,
        }
    }

    /// Send one typed V2 execute request and await the response.
    #[instrument(
        skip_all,
        fields(pid = %self.channel.pid(), request_id = %request.request_id),
    )]
    pub(in crate::worker::pool) async fn execute_v2(
        &self,
        request: &ExecuteRequestV2,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        self.channel.execute_v2(request).await
    }

    /// Query this exact worker's live capability and engine identity.
    pub(in crate::worker::pool) async fn capabilities(
        &self,
    ) -> Result<crate::worker::WorkerCapabilities, WorkerError> {
        self.channel.capabilities().await
    }

    /// Load one task's models on demand via the `ensure_task` IPC operation.
    pub(in crate::worker::pool) async fn ensure_task(
        &self,
        task: crate::worker::InferTask,
        engine_overrides: Option<&std::collections::BTreeMap<String, String>>,
        timeout: crate::api::PositiveSeconds,
    ) -> Result<(), WorkerError> {
        self.channel
            .ensure_task(task, engine_overrides, timeout)
            .await
    }

    /// Gracefully shut down the GPU worker, for `why`. Idempotent.
    pub(in crate::worker::pool) async fn shutdown(&self, why: super::Retirement) {
        info!(
            target = %self.config.bootstrap_label(),
            pid = %self.channel.pid(),
            "Shutting down shared GPU worker"
        );
        if let Some(ack) = self.channel.begin_shutdown(why).await {
            self.channel.await_shutdown_ack(ack).await;
        }
        self.finish_shutdown(why).await;
    }

    /// Observe whether this process can still accept requests. The slot and
    /// dispatch share this observation; a successful spawn is not permanent
    /// proof of liveness.
    pub(in crate::worker::pool) fn check_available(&self) -> Result<(), WorkerError> {
        self.channel.check_available()
    }

    /// The worker process ID.
    pub(in crate::worker::pool) fn pid(&self) -> WorkerPid {
        self.channel.pid()
    }

    /// The worker's profile label.
    pub(in crate::worker::pool) fn profile_label(&self) -> String {
        self.config.bootstrap_label()
    }

    /// The worker's language code.
    pub(in crate::worker::pool) fn lang(&self) -> &str {
        self.config.lang.as_worker_arg()
    }

    async fn finish_shutdown(&self, why: super::Retirement) {
        let pid = self.channel.pid();
        // Remove the PID file before killing.
        super::super::reaper::remove_worker_pid(pid.0);

        let child = self.child.lock().await.take();
        if let Some(mut child) = child {
            #[cfg(unix)]
            {
                let _ = child.id().map(|child_pid| {
                    // SAFETY: the worker was spawned as its own process group.
                    unsafe { libc::killpg(child_pid as libc::pid_t, libc::SIGTERM) };
                });
            }

            match tokio::time::timeout(Duration::from_secs(5), child.wait()).await {
                Ok(Ok(status)) => {
                    info!(pid = %pid, ?status, "Shared GPU worker exited gracefully");
                }
                Ok(Err(error)) => {
                    warn!(pid = %pid, error = %error, "Error waiting for shared GPU worker");
                }
                Err(_) => {
                    warn!(
                        pid = %pid,
                        "Shared GPU worker didn't exit in 5s, killing process group"
                    );
                    #[cfg(unix)]
                    {
                        let _ = child.id().map(|child_pid| {
                            // SAFETY: the worker was spawned as its own process group.
                            unsafe { libc::killpg(child_pid as libc::pid_t, libc::SIGKILL) };
                        });
                    }
                    let _ = child.kill().await;
                }
            }
        }

        self.channel.close(why);
    }
}

impl Drop for SharedGpuWorker {
    fn drop(&mut self) {
        super::super::reaper::remove_worker_pid(self.channel.pid().0);
        // Dropped only once nothing holds the worker, so no request is
        // waiting to hear the reason.
        self.channel.close(super::Retirement::WorkerRetired);

        if let Ok(mut child_slot) = self.child.try_lock()
            && let Some(child) = child_slot.as_mut()
        {
            #[cfg(unix)]
            {
                if let Some(pid) = child.id() {
                    let pgid = pid as libc::pid_t;
                    // SAFETY: the worker was spawned as its own process group.
                    unsafe {
                        libc::killpg(pgid, libc::SIGTERM);
                    }
                    // Brief pause then SIGKILL to prevent zombies holding GPU/RAM.
                    std::thread::sleep(std::time::Duration::from_millis(200));
                    if unsafe { libc::kill(pgid, 0) } == 0 {
                        unsafe {
                            libc::killpg(pgid, libc::SIGKILL);
                        }
                    }
                }
            }
            let _ = child.start_kill();
        }
    }
}
