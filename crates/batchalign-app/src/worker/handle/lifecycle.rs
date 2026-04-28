//! Worker process lifecycle management — startup helpers, shutdown, and cleanup.
//!
//! Contains the startup error handling (stderr draining, error augmentation),
//! graceful and forced shutdown sequences, the `Drop` impl for emergency
//! cleanup, and accessors for worker state.

use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncReadExt};
use tokio::process::{Child, ChildStderr};
use tracing::{info, warn};

use super::protocol::{MAX_READY_STDOUT_PREAMBLE_LINES, ReadySignal, STARTUP_STDERR_TAIL_CHARS};
use super::{WorkerHandle, WorkerHandleParts};
use crate::worker::WorkerPid;
use crate::worker::error::WorkerError;

use super::protocol::WorkerRequest;

impl WorkerHandle {
    /// Read and parse the JSON ready signal from the worker's stdout.
    pub(super) async fn read_ready_line<R: tokio::io::AsyncBufRead + Unpin>(
        reader: &mut R,
    ) -> Result<ReadySignal, WorkerError> {
        let mut line = String::new();
        let mut preamble = Vec::new();
        loop {
            line.clear();
            reader.read_line(&mut line).await.map_err(|e| {
                WorkerError::ReadyParseFailed(format!("failed to read stdout: {e}"))
            })?;

            if line.is_empty() {
                let mut detail = "worker closed stdout without emitting ready signal".to_string();
                if !preamble.is_empty() {
                    detail.push_str("; pre-ready stdout: ");
                    detail.push_str(&preamble.join(" | "));
                }
                return Err(WorkerError::ReadyParseFailed(detail));
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if !trimmed.starts_with('{') {
                preamble.push(trimmed.to_owned());
                if preamble.len() > MAX_READY_STDOUT_PREAMBLE_LINES {
                    let mut detail = format!(
                        "worker emitted more than {MAX_READY_STDOUT_PREAMBLE_LINES} non-JSON line(s) before ready signal"
                    );
                    detail.push_str("; pre-ready stdout: ");
                    detail.push_str(&preamble.join(" | "));
                    return Err(WorkerError::ReadyParseFailed(detail));
                }
                continue;
            }

            return serde_json::from_str::<ReadySignal>(&line).map_err(|e| {
                let mut detail = format!("invalid ready JSON: {e} (line: {line:?})");
                if !preamble.is_empty() {
                    detail.push_str("; pre-ready stdout: ");
                    detail.push_str(&preamble.join(" | "));
                }
                WorkerError::ReadyParseFailed(detail)
            });
        }
    }

    /// Handle a startup failure: terminate the child, drain stderr, and
    /// augment the original error with stderr context.
    pub(super) async fn finalize_startup_failure(
        child: &mut Child,
        stderr_reader: &mut tokio::io::BufReader<ChildStderr>,
        error: WorkerError,
    ) -> WorkerError {
        Self::terminate_startup_child(child).await;
        let stderr = Self::drain_startup_stderr(stderr_reader).await;
        Self::augment_startup_error(error, stderr)
    }

    /// Send SIGTERM to the startup child's process group and wait briefly.
    async fn terminate_startup_child(child: &mut Child) {
        #[cfg(unix)]
        {
            if let Some(pid) = child.id() {
                unsafe {
                    libc::killpg(pid as libc::pid_t, libc::SIGTERM);
                }
            }
        }

        let waited = tokio::time::timeout(Duration::from_millis(500), child.wait()).await;
        if waited.is_ok() {
            return;
        }

        let _ = child.start_kill();
        let _ = tokio::time::timeout(Duration::from_millis(500), child.wait()).await;
    }

    /// Read remaining stderr from a startup-phase child (before the background
    /// drain task is set up).
    async fn drain_startup_stderr(
        stderr_reader: &mut tokio::io::BufReader<ChildStderr>,
    ) -> Option<String> {
        let mut stderr = String::new();
        let _ = tokio::time::timeout(
            Duration::from_secs(1),
            stderr_reader.read_to_string(&mut stderr),
        )
        .await;
        Self::compact_stderr(&stderr)
    }

    /// Augment a startup error with stderr context (if any).
    fn augment_startup_error(error: WorkerError, stderr: Option<String>) -> WorkerError {
        let Some(stderr) = stderr else {
            return error;
        };

        match error {
            WorkerError::SpawnFailed(message) => {
                WorkerError::SpawnFailed(format!("{message}; worker stderr: {stderr}"))
            }
            WorkerError::ReadyParseFailed(message) => {
                WorkerError::ReadyParseFailed(format!("{message}; worker stderr: {stderr}"))
            }
            other => other,
        }
    }

    /// Compact stderr output into a single line with a character limit.
    pub(super) fn compact_stderr(stderr: &str) -> Option<String> {
        let mut compact = stderr
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" | ");
        if compact.is_empty() {
            return None;
        }

        let chars: Vec<char> = compact.chars().collect();
        if chars.len() > STARTUP_STDERR_TAIL_CHARS {
            let tail = chars[chars.len() - STARTUP_STDERR_TAIL_CHARS..]
                .iter()
                .collect::<String>();
            compact = format!("…{tail}");
        }

        Some(compact)
    }

    /// Drain buffered stderr lines from the background capture task.
    ///
    /// Returns the last `max_lines` lines joined by newline, or `None` if
    /// no stderr was captured. Called on worker crash to attach diagnostic
    /// output (Python tracebacks, OOM messages) to the error.
    pub(super) fn drain_stderr_tail(&mut self, max_lines: usize) -> Option<String> {
        use std::collections::VecDeque;
        let mut tail = VecDeque::with_capacity(max_lines);
        while let Ok(line) = self.stderr_rx.try_recv() {
            tail.push_back(line);
            if tail.len() > max_lines {
                tail.pop_front();
            }
        }
        if tail.is_empty() {
            None
        } else {
            Some(tail.into_iter().collect::<Vec<_>>().join("\n"))
        }
    }

    /// Check if the worker process is still running.
    pub fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Gracefully shut down the worker in place (shutdown message + SIGTERM to
    /// process group + wait).
    ///
    /// Uses `killpg` to kill the entire process group (the worker + any children
    /// it spawned, e.g. Stanza subprocesses), ensuring no orphans survive.
    pub async fn shutdown_in_place(&mut self) -> Result<(), WorkerError> {
        // Layer 3: remove PID file before killing.
        super::super::pool::reaper::remove_worker_pid(self.pid.0);

        info!(
            target = %self.config.bootstrap_label(),
            pid = %self.pid,
            "Shutting down worker"
        );

        let _ = self.write_request(&WorkerRequest::Shutdown).await;
        let _ = tokio::time::timeout(Duration::from_secs(2), self.read_response()).await;

        #[cfg(unix)]
        {
            let _ = self.child.id().map(|pid| {
                // SAFETY: sending SIGTERM to the worker's process group.
                // The worker was spawned with setpgid(0,0), so its PGID == PID.
                unsafe { libc::killpg(pid as libc::pid_t, libc::SIGTERM) };
            });
        }

        // On Windows, there is no direct equivalent of killpg(SIGTERM).
        // We rely on the graceful shutdown message sent above. The timeout
        // below will call child.kill() (TerminateProcess) if the worker
        // does not exit in time.
        // TODO(windows): For full parity, use a Job Object to group the
        // worker and its children, then TerminateJobObject() to kill the
        // entire tree. This requires the `windows-sys` crate.

        match tokio::time::timeout(Duration::from_secs(5), self.child.wait()).await {
            Ok(Ok(status)) => {
                info!(pid = %self.pid, ?status, "Worker exited gracefully");
            }
            Ok(Err(e)) => {
                warn!(pid = %self.pid, error = %e, "Error waiting for worker");
            }
            Err(_) => {
                warn!(
                    pid = %self.pid,
                    "Worker didn't exit in 5s, killing process group"
                );
                #[cfg(unix)]
                {
                    let _ = self.child.id().map(|pid| {
                        unsafe { libc::killpg(pid as libc::pid_t, libc::SIGKILL) };
                    });
                }
                // On all platforms (including Windows), child.kill() sends
                // TerminateProcess / SIGKILL to the direct child. On Windows
                // this does NOT kill grandchildren; a Job Object would be
                // needed for full process-tree cleanup.
                let _ = self.child.kill().await;
            }
        }

        Ok(())
    }

    /// Gracefully shut down the worker (consuming `self`).
    pub async fn shutdown(mut self) -> Result<(), WorkerError> {
        self.shutdown_in_place().await
    }

    /// The PID of the worker process.
    pub fn pid(&self) -> WorkerPid {
        self.pid
    }

    /// The logical bootstrap target label this worker handles.
    pub fn profile_label(&self) -> String {
        self.config.bootstrap_label()
    }

    /// The language this worker handles.
    pub fn lang(&self) -> &str {
        self.config.lang.as_worker_arg()
    }

    /// The transport this worker uses.
    pub fn transport(&self) -> &'static str {
        "stdio"
    }

    /// Duration since the last request was dispatched.
    pub fn idle_duration(&self) -> Duration {
        self.last_activity.elapsed()
    }

    /// Reference to this worker's configuration.
    pub(crate) fn config(&self) -> &super::config::WorkerConfig {
        &self.config
    }

    /// Consume the handle into its raw parts for concurrent mode setup.
    ///
    /// The returned [`WorkerHandleParts`] owns the child process, stdin, and
    /// stdout. The caller becomes responsible for the child process lifecycle
    /// — the `WorkerHandle::Drop` impl does **not** run.
    pub(crate) fn into_parts(self) -> WorkerHandleParts {
        // Use ManuallyDrop to prevent Drop::drop from killing the child.
        let md = std::mem::ManuallyDrop::new(self);

        // SAFETY: We're moving each field out of a ManuallyDrop wrapper.
        // ManuallyDrop prevents Drop from running. Each field is moved
        // exactly once, so no double-free can occur.
        unsafe {
            WorkerHandleParts {
                config: std::ptr::read(&md.config),
                child: std::ptr::read(&md.child),
                pid: std::ptr::read(&md.pid),
                stdin: std::ptr::read(&md.stdin),
                stdout: std::ptr::read(&md.stdout),
            }
        }
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        // Layer 3: remove PID file on drop (covers panic/unwind paths).
        super::super::pool::reaper::remove_worker_pid(self.pid.0);

        if self.is_alive() {
            #[cfg(unix)]
            if let Some(pid) = self.child.id() {
                use super::super::pool::reaper::{kill_pgid, process_alive, terminate_pgid};
                terminate_pgid(pid);
                // Brief pause to let Python handle SIGTERM. If the worker
                // is stuck in a C extension (PyTorch, NumPy), SIGTERM may
                // be ignored — follow up with SIGKILL to prevent zombies
                // that hold 2-15 GB of RAM.
                std::thread::sleep(std::time::Duration::from_millis(200));
                if process_alive(pid) {
                    kill_pgid(pid);
                }
            }
            let _ = self.child.start_kill();
        }
    }
}
