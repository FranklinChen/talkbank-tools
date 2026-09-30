//! `WorkerHandle`: manages a single Python worker child process.
//!
//! Split into submodules:
//! - [`config`]: `WorkerRuntimeConfig`, `WorkerConfig`
//! - [`protocol`]: Wire-level request/response envelopes, ready signals, constants
//! - [`spawn`]: Command building and TCP daemon spawning
//! - [`ipc`]: Request/response IPC methods (infer, batch_infer, execute_v2, health, capabilities)
//! - [`lifecycle`]: Startup helpers, shutdown, Drop, accessors

pub mod config;
mod ipc;
mod lifecycle;
mod progress_preamble;
mod protocol;
pub mod spawn;

pub use config::{WorkerConfig, WorkerRuntimeConfig};
pub use spawn::spawn_tcp_daemon;
// Re-exported for use across the crate (notably ``worker::tcp_handle``)
// without making the whole ``protocol`` submodule public. ``WorkerErrorKind``
// is the shared on-the-wire discriminator for ``{"op":"error", ...}``
// responses; see the type's own doc.
pub(crate) use protocol::WorkerErrorKind;

use std::time::Duration;

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
    /// Rust-side cache of tasks loaded via `ensure_task`. Skips redundant IPC
    /// round-trips for already-loaded tasks in LazyProfile mode.
    loaded_tasks: std::collections::HashSet<String>,
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
    /// Spawn a new Python worker and wait for it to become ready.
    pub async fn spawn(config: WorkerConfig) -> Result<Self, WorkerError> {
        // Memory guard: acquire a serialized spawn permit and check available RAM.
        // This prevents the TOCTOU race where N concurrent spawns all see "enough"
        // memory before any model is loaded, then collectively exceed physical RAM.
        // Crash history and design rationale:
        // `book/src/batchalign/developer/memory-safety.md`.
        let startup_reservation = config.startup_reservation_mb();
        let spawn_permit = crate::worker::memory_guard::acquire_spawn_permit(&config).await?;

        let mut cmd: tokio::process::Command = spawn::build_worker_command(&config).into();

        info!(
            target = %config.bootstrap_label(),
            lang = %config.lang,
            test_echo = config.test_echo,
            force_cpu = config.runtime.force_cpu,
            python = %config.python_path,
            startup_reservation_mb = startup_reservation.0,
            available_memory_mb = crate::worker::memory_guard::available_memory_mb(),
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
            spawn_permit.set_worker_pid(child_pid);
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

        let ready = match tokio::time::timeout(
            Duration::from_secs(config.ready_timeout_s),
            Self::read_ready_line(&mut stdout_reader),
        )
        .await
        {
            Ok(Ok(ready)) => ready,
            Ok(Err(error)) => {
                return Err(
                    Self::finalize_startup_failure(&mut child, &mut stderr_reader, error).await,
                );
            }
            Err(_) => {
                return Err(Self::finalize_startup_failure(
                    &mut child,
                    &mut stderr_reader,
                    WorkerError::ReadyTimeout {
                        timeout_s: config.ready_timeout_s,
                    },
                )
                .await);
            }
        };

        if !ready.ready {
            return Err(Self::finalize_startup_failure(
                &mut child,
                &mut stderr_reader,
                WorkerError::ReadyParseFailed(
                    "worker emitted ready line with ready=false".to_string(),
                ),
            )
            .await);
        }

        if let Some(transport) = ready.transport.as_deref()
            && transport != "stdio"
        {
            return Err(Self::finalize_startup_failure(
                &mut child,
                &mut stderr_reader,
                WorkerError::ReadyParseFailed(format!("unexpected worker transport: {transport}")),
            )
            .await);
        }

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| WorkerError::SpawnFailed("child stdin not captured".into()))?;
        let pid = WorkerPid(ready.pid);
        let runtime = ready.runtime;
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
            loaded_tasks: std::collections::HashSet::new(),
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
        let ready = Self::read_ready_line(&mut stdout_reader)
            .await
            .expect("stub must emit a valid ready line as its first stdout line");

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
            config: WorkerConfig::default(),
            child,
            pid: WorkerPid(ready.pid),
            runtime: ready.runtime,
            stdin,
            stdout: stdout_reader,
            last_activity: tokio::time::Instant::now(),
            loaded_tasks: std::collections::HashSet::new(),
            stderr_rx,
            request_flight: RequestFlight::Idle,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::spawn::build_worker_command;
    use super::{WorkerConfig, WorkerRuntimeConfig};
    use crate::api::{LanguageCode3, NumSpeakers, WorkerLanguage};
    use crate::host_memory::HostMemoryRuntimeConfig;
    use crate::worker::provider_credentials::HkAsrCredentialSources;
    use crate::worker::{InferTask, WorkerProfile};

    fn command_args(config: &WorkerConfig) -> Vec<String> {
        build_worker_command(config)
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    fn command_envs(config: &WorkerConfig) -> BTreeMap<String, String> {
        build_worker_command(config)
            .get_envs()
            .filter_map(|(key, value)| {
                value.map(|value| {
                    (
                        key.to_string_lossy().into_owned(),
                        value.to_string_lossy().into_owned(),
                    )
                })
            })
            .collect()
    }

    #[test]
    fn worker_command_forwards_runtime_force_cpu() {
        let args = command_args(&WorkerConfig {
            python_path: "python3".to_string(),
            profile: WorkerProfile::Gpu,
            lang: WorkerLanguage::from(LanguageCode3::eng()),
            num_speakers: NumSpeakers(1),
            engine_overrides: String::new(),
            test_echo: false,
            ready_timeout_s: 300,
            verbose: 0,
            runtime: WorkerRuntimeConfig::from_sources(
                true,
                false,
                None,
                4,
                HostMemoryRuntimeConfig::default(),
                crate::types::runtime::MemoryTier::detect(),
            ),
            ..Default::default()
        });

        assert!(args.iter().any(|arg| arg == "--force-cpu"));
    }

    #[test]
    fn worker_command_forwards_runtime_allow_mps() {
        let args = command_args(&WorkerConfig {
            python_path: "python3".to_string(),
            profile: WorkerProfile::Gpu,
            lang: WorkerLanguage::from(LanguageCode3::eng()),
            num_speakers: NumSpeakers(1),
            engine_overrides: String::new(),
            test_echo: false,
            ready_timeout_s: 300,
            verbose: 0,
            runtime: WorkerRuntimeConfig::from_sources(
                false,
                true,
                None,
                4,
                HostMemoryRuntimeConfig::default(),
                crate::types::runtime::MemoryTier::detect(),
            ),
            ..Default::default()
        });

        assert!(args.iter().any(|arg| arg == "--allow-mps"));
        assert!(!args.iter().any(|arg| arg == "--force-cpu"));
    }

    #[test]
    fn worker_command_uses_task_arg_for_task_bootstrap() {
        let args = command_args(&WorkerConfig {
            python_path: "python3".to_string(),
            profile: WorkerProfile::Stanza,
            task: Some(InferTask::Morphosyntax),
            ..Default::default()
        });

        assert!(
            args.windows(2)
                .any(|window| window[0] == "--task" && window[1] == "morphosyntax")
        );
        assert!(!args.iter().any(|arg| arg == "--profile"));
    }

    #[test]
    fn worker_command_forwards_gpu_thread_pool_size() {
        let args = command_args(&WorkerConfig {
            python_path: "python3".to_string(),
            profile: WorkerProfile::Gpu,
            runtime: WorkerRuntimeConfig::from_sources(
                false,
                false,
                None,
                7,
                HostMemoryRuntimeConfig::default(),
                crate::types::runtime::MemoryTier::detect(),
            ),
            ..Default::default()
        });

        assert!(
            args.windows(2)
                .any(|window| { window[0] == "--gpu-thread-pool-size" && window[1] == "7" })
        );
        assert!(
            args.windows(2)
                .any(|window| { window[0] == "--serving" && window[1] == "concurrent" })
        );
    }

    /// Wire format: a stdio worker is told which process supervises it, so it
    /// exits with the server that launched it.
    #[test]
    fn worker_command_names_this_process_as_the_supervisor() {
        let args = command_args(&WorkerConfig {
            python_path: "python3".to_string(),
            ..Default::default()
        });
        let own_pid = std::process::id().to_string();

        assert!(
            args.windows(2)
                .any(|window| window[0] == "--supervisor-pid" && window[1] == own_pid)
        );
    }

    /// Wire format: a forced-CPU GPU worker is told to serve one request at a
    /// time, with one serving thread, whatever the configured pool size. The
    /// pool routes it by exclusive checkout from the same decision.
    #[test]
    fn worker_command_tells_a_forced_cpu_gpu_worker_to_serve_sequentially() {
        let args = command_args(&WorkerConfig {
            python_path: "python3".to_string(),
            profile: WorkerProfile::Gpu,
            runtime: WorkerRuntimeConfig::from_sources(
                true,
                false,
                None,
                7,
                HostMemoryRuntimeConfig::default(),
                crate::types::runtime::MemoryTier::detect(),
            ),
            ..Default::default()
        });

        assert!(
            args.windows(2)
                .any(|window| { window[0] == "--serving" && window[1] == "sequential" })
        );
        assert!(
            args.windows(2)
                .any(|window| { window[0] == "--gpu-thread-pool-size" && window[1] == "1" })
        );
    }

    #[test]
    fn worker_command_injects_resolved_revai_key() {
        let envs = command_envs(&WorkerConfig {
            python_path: "python3".to_string(),
            profile: WorkerProfile::Gpu,
            lang: WorkerLanguage::from(LanguageCode3::eng()),
            num_speakers: NumSpeakers(1),
            engine_overrides: String::new(),
            test_echo: false,
            ready_timeout_s: 300,
            verbose: 0,
            runtime: WorkerRuntimeConfig::from_sources(
                false,
                false,
                Some("  injected-key  ".to_string()),
                4,
                HostMemoryRuntimeConfig::default(),
                crate::types::runtime::MemoryTier::detect(),
            ),
            ..Default::default()
        });

        assert_eq!(
            envs.get("BATCHALIGN_REV_API_KEY").map(String::as_str),
            Some("injected-key")
        );
    }

    #[test]
    fn worker_provider_envs_only_inject_selected_hk_asr_backend() {
        let envs = super::spawn::worker_provider_envs(
            &WorkerConfig {
                python_path: "python3".to_string(),
                profile: WorkerProfile::Gpu,
                lang: WorkerLanguage::from(LanguageCode3::yue()),
                num_speakers: NumSpeakers(1),
                engine_overrides: r#"{"asr":"tencent"}"#.to_string(),
                test_echo: false,
                ready_timeout_s: 300,
                verbose: 0,
                runtime: WorkerRuntimeConfig::from_sources(
                    false,
                    false,
                    None,
                    4,
                    HostMemoryRuntimeConfig::default(),
                    crate::types::runtime::MemoryTier::detect(),
                ),
                ..Default::default()
            },
            &HkAsrCredentialSources::from_sources(
                Some("id"),
                Some("key"),
                Some("ap-guangzhou"),
                Some("bucket"),
                None,
                None,
                None,
                Some("/tmp/unused-home"),
            ),
        )
        .into_iter()
        .collect::<BTreeMap<_, _>>();

        assert_eq!(
            envs.get("BATCHALIGN_TENCENT_ID").map(String::as_str),
            Some("id")
        );
        assert_eq!(
            envs.get("BATCHALIGN_TENCENT_BUCKET").map(String::as_str),
            Some("bucket")
        );
    }

    /// Verify that a `ProgressV2` JSON line deserializes correctly into the
    /// `WorkerResponse` enum.  This is the wire format emitted by Python
    /// workers during long-running V2 tasks.
    #[test]
    fn deserialize_progress_v2_response() {
        let json = r#"{"op": "progress_v2", "event": {"request_id": "req-001", "completed": 42, "total": 100, "stage": "stanza_processing"}}"#;
        let resp: super::protocol::WorkerResponse = serde_json::from_str(json).unwrap();
        match resp {
            super::protocol::WorkerResponse::ProgressV2 { event } => {
                assert_eq!(&*event.request_id, "req-001");
                assert_eq!(event.completed, 42);
                assert_eq!(event.total, 100);
                assert_eq!(event.stage, "stanza_processing");
            }
            other => panic!("Expected ProgressV2, got {other:?}"),
        }
    }

    /// Verify backward compatibility: a response stream with zero progress
    /// events still deserializes the final `ExecuteV2` response correctly.
    ///
    /// The fixture used to claim success while carrying no result, which the
    /// structural `ExecuteResponseV2` (2026-08-21) now refuses at parse time;
    /// the pairing was incidental to this test's purpose (op-level routing),
    /// so the fixture became a legal error response instead.
    #[test]
    fn deserialize_execute_v2_without_progress() {
        let json = r#"{"op": "execute_v2", "response": {"request_id": "req-002", "outcome": {"kind": "error", "code": "runtime_failure", "message": "boom"}, "elapsed_s": 1.5}}"#;
        let resp: super::protocol::WorkerResponse = serde_json::from_str(json).unwrap();
        assert!(matches!(
            resp,
            super::protocol::WorkerResponse::ExecuteV2 { .. }
        ));
    }
}
