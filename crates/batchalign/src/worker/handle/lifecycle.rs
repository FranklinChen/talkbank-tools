//! Worker process lifecycle management, startup helpers, shutdown, and cleanup.
//!
//! Contains the startup error handling (stderr draining, error augmentation),
//! graceful and forced shutdown sequences, the `Drop` impl for emergency
//! cleanup, and accessors for worker state.

use std::time::Duration;

use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, AsyncReadExt};
use tokio::process::{Child, ChildStderr};
use tracing::{info, warn};

use super::protocol::{MAX_READY_STDOUT_PREAMBLE_LINES, ReadySignal, STARTUP_STDERR_TAIL_CHARS};
use super::{WorkerHandle, WorkerHandleParts};
use crate::worker::WorkerPid;
use crate::worker::error::WorkerError;

use super::protocol::WorkerRequest;

/// A JSON line a worker may write before its ready signal that is not the
/// signal itself, told apart by its `op` tag.
#[derive(serde::Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum PreReadyLine {
    /// A bootstrap-time progress event (model download, model load, etc.).
    /// Its stage and measure decide whether it re-arms the startup stall
    /// window ([`crate::worker::RunningWait::observe`]).
    ProgressV2 {
        event: crate::types::worker_v2::ProgressEventV2,
    },
}

/// Progress lines kept for a startup failure's diagnostics: the latest ones.
///
/// A bounded tail rather than a cap: a long first download reports its growth
/// every few seconds for as long as it runs (over a hundred lines for twenty
/// minutes), which is the worker being alive, not misbehaving.
const READY_PROGRESS_TAIL_LINES: usize = 8;

/// What a worker wrote before its ready signal, for diagnostics.
#[derive(Default)]
struct ReadyPreamble {
    /// Non-JSON lines, capped: past [`MAX_READY_STDOUT_PREAMBLE_LINES`] the
    /// worker is writing something other than the protocol to stdout.
    noise: Vec<String>,
    /// The latest pre-ready progress lines.
    progress: std::collections::VecDeque<String>,
}

impl ReadyPreamble {
    /// Keep a non-JSON line; `true` once there are more than the cap allows.
    fn noise(&mut self, line: &str) -> bool {
        self.noise.push(line.to_owned());
        self.noise.len() > MAX_READY_STDOUT_PREAMBLE_LINES
    }

    /// Keep a progress line, dropping the oldest past the tail length.
    fn progress(&mut self, line: &str) {
        if self.progress.len() == READY_PROGRESS_TAIL_LINES {
            self.progress.pop_front();
        }
        self.progress.push_back(line.to_owned());
    }

    /// `detail`, followed by whatever the worker wrote before failing.
    fn describe(&self, mut detail: String) -> String {
        if !self.noise.is_empty() || !self.progress.is_empty() {
            detail.push_str("; pre-ready stdout: ");
            let lines: Vec<&str> = self
                .noise
                .iter()
                .chain(self.progress.iter())
                .map(String::as_str)
                .collect();
            detail.push_str(&lines.join(" | "));
        }
        detail
    }
}

impl WorkerHandle {
    /// Read and parse the JSON ready signal from the worker's stdout.
    ///
    /// Workers may emit `progress_v2` lines before they have finished
    /// bootstrapping (e.g. `download_liveness` reports a Stanza language
    /// pack's growth while `stanza.Pipeline()` downloads it on first use,
    /// which happens BEFORE the worker writes its ready signal). Treat any pre-ready JSON line whose `"op"` field is
    /// `"progress_v2"` as a tolerable bootstrap-time progress event,
    /// preserve it on the preamble for diagnostics, and keep reading for
    /// the actual ready signal. Any other JSON shape that is not a valid
    /// `ReadySignal` is still a hard error.
    ///
    /// This matters because the 22 morphotag failures observed on
    /// 2026-05-06 (`provider_terminal` errors with line content
    /// `{"op": "progress_v2", ..., "stage": "downloading_stanza_lang_zh"}`)
    /// were the parent killing a worker that had emitted a perfectly valid
    /// download-progress event one line ahead of its ready signal, a
    /// protocol-ordering race, not a real worker failure.
    ///
    /// The wait is a [`crate::worker::LoadingStall`] window, not a total
    /// limit: each progress event that advanced re-arms it, so a worker
    /// downloading its models for twenty minutes is ready when it finishes,
    /// and one that reports nothing new for `stall` is
    /// [`WorkerError::Stalled`].
    pub(super) async fn read_ready_line<R: tokio::io::AsyncBufRead + Unpin>(
        reader: &mut R,
        stall: crate::worker::LoadingStall,
    ) -> Result<ReadySignal, WorkerError> {
        let loading = crate::worker::pool::checkout_wait::ObservedModelLoad::begin();
        let mut wait = crate::worker::RunningWait::start(crate::worker::ResponseWait::Loading {
            waited_for: crate::worker::LoadingWait::Ready,
            stall,
        });
        let mut line = String::new();
        let mut preamble = ReadyPreamble::default();
        loop {
            line.clear();
            let bytes = wait
                .within(reader.read_line(&mut line))
                .await?
                .map_err(|e| {
                    WorkerError::ReadyParseFailed(format!("failed to read stdout: {e}"))
                })?;

            if bytes == 0 {
                return Err(WorkerError::ReadyParseFailed(preamble.describe(
                    "worker closed stdout without emitting ready signal".to_string(),
                )));
            }

            // The one line rule every reader applies.
            let message = match super::protocol::WireLine::classify(&line) {
                super::protocol::WireLine::Blank => continue,
                super::protocol::WireLine::Noise => {
                    if preamble.noise(line.trim()) {
                        return Err(WorkerError::ReadyParseFailed(preamble.describe(format!(
                            "worker emitted more than {MAX_READY_STDOUT_PREAMBLE_LINES} non-JSON line(s) before ready signal"
                        ))));
                    }
                    continue;
                }
                super::protocol::WireLine::Message(message) => message,
            };

            // Every pre-ready line carries an `op` tag; the ready signal does
            // not. A tagged line is decoded as one, and one that does not
            // decode is a malformed protocol line, named as one rather than
            // misreported as a bad ready signal.
            if message.get("op").is_none() {
                return ReadySignal::deserialize(&message).map_err(|e| {
                    WorkerError::ReadyParseFailed(
                        preamble
                            .describe(format!("invalid ready JSON: {e} (line: {:?})", line.trim())),
                    )
                });
            }
            match PreReadyLine::deserialize(&message) {
                // Pre-ready progress: the worker telling us about a
                // bootstrap-time wait (model download, model load). Logged
                // as `info!` so bootstrap timing reaches the daemon log live
                // (the Python worker's stderr is buffered until it exits),
                // kept in a bounded tail for failure diagnostics, and shown
                // to the stall window, which an advance re-arms.
                Ok(PreReadyLine::ProgressV2 { event }) => {
                    tracing::info!(event = %line.trim(), "Pre-ready progress event");
                    wait.observe(&event);
                    match &event {
                        crate::types::worker_v2::ProgressEventV2::Load(load) => {
                            loading.report(load)
                        }
                        crate::types::worker_v2::ProgressEventV2::Request(_) => {}
                    }
                    preamble.progress(line.trim());
                }
                Err(error) => {
                    return Err(WorkerError::ReadyParseFailed(preamble.describe(format!(
                        "undecodable pre-ready line: {error} (line: {:?})",
                        line.trim()
                    ))));
                }
            }
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
            if let Some(target) = crate::worker::process_id::NativePid::of_child(child) {
                let _ = target.terminate_group();
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
            if let Some(target) = crate::worker::process_id::NativePid::of_child(&self.child) {
                let _ = target.terminate_group();
            }
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
                    if let Some(target) =
                        crate::worker::process_id::NativePid::of_child(&self.child)
                    {
                        let _ = target.kill_group();
                    }
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
    pub const fn pid(&self) -> WorkerPid {
        self.pid
    }

    /// Content-addressed Python runtime evidence required by the ready schema.
    pub const fn runtime_identity(
        &self,
    ) -> &crate::worker::runtime_identity::WorkerRuntimeIdentity {
        &self.runtime
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
    pub const fn transport(&self) -> &'static str {
        "stdio"
    }

    /// Duration since the last request was dispatched.
    pub fn idle_duration(&self) -> Duration {
        self.last_activity.elapsed()
    }

    /// Reference to this worker's configuration.
    pub(crate) const fn config(&self) -> &super::config::WorkerConfig {
        &self.config
    }

    /// Consume the handle into its raw parts for concurrent mode setup.
    ///
    /// The returned [`WorkerHandleParts`] owns the child process, stdin, and
    /// stdout. The caller becomes responsible for the child process lifecycle
    ///: the `WorkerHandle::Drop` impl does **not** run.
    pub(crate) fn into_parts(self) -> WorkerHandleParts {
        // Use ManuallyDrop to prevent Drop::drop from killing the child.
        let md = std::mem::ManuallyDrop::new(self);

        // Reason: moving fields out of a type with `Drop` needs either this
        // or an `Option` that every accessor would have to re-check; the
        // handle's kill-on-drop must not run once the caller owns the parts.
        #[allow(unsafe_code)]
        // SAFETY: We're moving each field out of a ManuallyDrop wrapper.
        // ManuallyDrop prevents Drop from running. Each field is moved
        // exactly once, so no double-free can occur.
        unsafe {
            WorkerHandleParts {
                config: std::ptr::read(&md.config),
                child: std::ptr::read(&md.child),
                pid: std::ptr::read(&md.pid),
                runtime: std::ptr::read(&md.runtime),
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
                // be ignored: follow up with SIGKILL to prevent zombies
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

#[cfg(test)]
mod ready_signal_tests {
    use super::*;
    use tokio::io::BufReader;

    /// A stall window no in-memory reader test comes near.
    const GENEROUS_STALL: crate::worker::LoadingStall =
        crate::worker::LoadingStall::new(crate::api::PositiveSeconds::literal::<30>());

    fn buf(s: &str) -> BufReader<&[u8]> {
        BufReader::new(s.as_bytes())
    }

    fn ready(pid: u32) -> String {
        format!(
            concat!(
                "{{\"ready\":true,\"pid\":{},\"transport\":\"stdio\",\"runtime\":{{",
                "\"schema_version\":1,\"python_version\":\"3.13.12\",",
                "\"python_executable_sha256\":\"{}\",",
                "\"batchalign_package_tree_sha256\":\"{}\",",
                "\"batchalign_core_extension_sha256\":\"{}\",",
                "\"distribution_inventory_sha256\":\"{}\"}}}}\n"
            ),
            pid,
            "a".repeat(64),
            "b".repeat(64),
            "c".repeat(64),
            "d".repeat(64),
        )
    }

    /// Guards: startup admits affirmative transport-specific bytes and retains runtime digests
    /// through empty, stacked and long advancing progress preambles.
    /// Kind: wire
    /// Not a type: external JSON, its ordering and digest bytes still require decoding
    /// before private ready proofs can carry the admitted identity.
    /// State: local byte readers exercise the actual startup reader without a process.
    /// Concurrency: none
    #[tokio::test]
    async fn startup_bytes_carry_identity_through_transport_admission() {
        for progress_count in [0, 1, 3, 4 * MAX_READY_STDOUT_PREAMBLE_LINES] {
            let mut prefix = String::new();
            for kib in 1..=progress_count {
                prefix.push_str(&format!(
                    "{{\"op\":\"progress_v2\",\"event\":{{\"scope\":\"load\",\"stage\":\"download\",\"measure\":{{\"kind\":\"download_growth\",\"grown_kib\":{kib}}}}}}}\n"
                ));
            }
            let stdio = format!("{prefix}{}", ready(4242));
            let mut reader = buf(&stdio);
            let signal = WorkerHandle::read_ready_line(&mut reader, GENEROUS_STALL)
                .await
                .unwrap();
            let (pid, runtime) = signal.into_stdio().unwrap().into_parts();
            assert_eq!(pid, WorkerPid(4242));
            assert_eq!(runtime.python_version(), "3.13.12");
            assert_eq!(runtime.python_executable_sha256().as_str(), "a".repeat(64));
            let expected: serde_json::Value = serde_json::from_str(&ready(4242)).unwrap();
            assert_eq!(serde_json::to_value(&runtime).unwrap(), expected["runtime"]);

            let tcp = stdio.replace(
                "\"transport\":\"stdio\"",
                "\"transport\":\"tcp\",\"host\":\"127.0.0.1\",\"port\":42001",
            );
            let mut reader = buf(&tcp);
            let (pid, runtime, host, port) =
                WorkerHandle::read_ready_line(&mut reader, GENEROUS_STALL)
                    .await
                    .unwrap()
                    .into_tcp()
                    .unwrap()
                    .into_parts();
            assert_eq!(pid, WorkerPid(4242));
            assert_eq!(runtime.python_executable_sha256().as_str(), "a".repeat(64));
            assert_eq!((host.as_str(), port), ("127.0.0.1", 42001));
        }
    }

    /// Guards: refused startup bytes cannot produce readiness or the wrong transport proof.
    /// Kind: wire
    /// Not a type: hostile JSON can omit runtime facts, deny readiness or name a transport
    /// different from the real spawning adapter, so admission must inspect those bytes.
    /// State: local byte readers preserve each actual malformed or mismatched handshake.
    /// Concurrency: none
    #[tokio::test]
    async fn startup_admission_refuses_false_missing_and_mismatched_wire_facts() {
        for (stdout, expected) in [
            (
                ready(1).replace("\"ready\":true", "\"ready\":false"),
                "ready=false",
            ),
            (
                "{\"ready\":true,\"pid\":1,\"transport\":\"stdio\"}\n".into(),
                "runtime",
            ),
            (
                format!("{{\"hello\":\"world\"}}\n{}", ready(1)),
                "invalid ready JSON",
            ),
            (
                "{\"op\":\"progress_v2\",\"event\":{\"stage\":\"x\"}}\n".into(),
                "undecodable pre-ready line",
            ),
            (
                ready(1).replace(&"a".repeat(64), "invalid-digest"),
                "invalid ready JSON",
            ),
        ] {
            let mut reader = buf(&stdout);
            match WorkerHandle::read_ready_line(&mut reader, GENEROUS_STALL).await {
                Err(WorkerError::ReadyParseFailed(detail)) => {
                    assert!(detail.contains(expected), "{detail}")
                }
                other => panic!("expected startup refusal containing {expected}, got {other:?}"),
            }
        }

        let stdio = ready(1);
        let mut reader = buf(&stdio);
        assert!(
            WorkerHandle::read_ready_line(&mut reader, GENEROUS_STALL)
                .await
                .unwrap()
                .into_tcp()
                .is_err()
        );
        let tcp = stdio.replace(
            "\"transport\":\"stdio\"",
            "\"transport\":\"tcp\",\"host\":\"127.0.0.1\",\"port\":42001",
        );
        let mut reader = buf(&tcp);
        assert!(
            WorkerHandle::read_ready_line(&mut reader, GENEROUS_STALL)
                .await
                .unwrap()
                .into_stdio()
                .is_err()
        );
    }

    /// Guards: an open startup stream with no advancing progress becomes a typed ready stall.
    /// Kind: contract
    /// Not a type: future silence depends on external IO and elapsed time after admission,
    /// which a retained handshake proof cannot predict.
    /// State: a duplex pipe stays open and silent under Tokio's controlled clock.
    /// Concurrency: both pipe halves and the paused clock belong to this test runtime.
    #[tokio::test(start_paused = true)]
    async fn a_silent_starting_worker_is_stalled() {
        let (reader, _writer) = tokio::io::duplex(64);
        let mut r = BufReader::new(reader);
        match WorkerHandle::read_ready_line(&mut r, GENEROUS_STALL).await {
            Err(WorkerError::Stalled { waited_for, .. }) => {
                assert_eq!(waited_for, crate::worker::LoadingWait::Ready);
            }
            other => panic!("expected a startup stall, got {other:?}"),
        }
    }
}
