//! Worker process spawning: command construction and TCP daemon launch.
//!
//! Contains the free functions for building the Python worker command line
//! and spawning a detached TCP worker daemon. The stdio-based spawn is in
//! [`WorkerHandle::spawn`](super::WorkerHandle::spawn) since it constructs
//! the handle directly.

use std::process::{Command as StdCommand, Stdio};

use serde_json::Value;
use tokio::process::Command;
use tracing::info;

use super::config::WorkerConfig;
use super::protocol::ReadyTransport;
use crate::worker::error::WorkerError;
use crate::worker::python::worker_command;
use crate::worker::serving::WorkerServing;
use crate::worker::target::task_name;
use crate::worker::{WorkerProfile, WorkerTarget};

/// Build the `std::process::Command` for a stdio-transport Python worker.
///
/// The caller converts this to a `tokio::process::Command` and spawns it.
/// Sets up stdin/stdout/stderr pipes, language, profile/task flags, engine
/// overrides, and process-group isolation (`setpgid` on Unix,
/// `CREATE_NEW_PROCESS_GROUP` on Windows).
///
/// When you add a new flag or env var here that the Python child reads at
/// startup, mirror the underlying `WorkerConfig` field in
/// `tests/common/test_worker_pool.rs::ConfigKey`. The shared test fixture
/// keys workers by that struct; missing a field there would silently let
/// it share workers across configs that should be distinct.
pub(super) fn build_worker_command(config: &WorkerConfig) -> StdCommand {
    let mut cmd = worker_command(&config.python_path);
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if config.test_echo {
        cmd.arg("--test-echo");
    }
    match config.bootstrap_target() {
        WorkerTarget::Profile(profile) => {
            // Every worker starts with no models: each task's load is its
            // `ensure_task`, before the task's first request.
            cmd.arg("--profile").arg(profile.name());
        }
        WorkerTarget::InferTask(task) => {
            cmd.arg("--task").arg(task_name(task));
        }
    }

    cmd.arg("--lang").arg(config.lang.as_worker_arg());
    cmd.arg("--num-speakers")
        .arg(config.num_speakers.0.to_string());

    // The worker watches this process and exits with it, including while it
    // is in the middle of a request: a stdio worker's results can reach no
    // one once the server holding its pipes is gone, however the server
    // ended (a graceful stop, a SIGKILL after a stop timed out, a crash).
    cmd.arg("--supervisor-pid")
        .arg(std::process::id().to_string());

    if !config.engine_overrides.is_empty() {
        cmd.arg("--engine-overrides").arg(&config.engine_overrides);
    }

    if config.runtime.force_cpu {
        cmd.arg("--force-cpu");
    }

    if config.runtime.allow_mps {
        cmd.arg("--allow-mps");
    }

    if config.verbose > 0 {
        cmd.arg("--verbose").arg(config.verbose.to_string());
    }

    cmd.args(serving_args(config));

    if config.test_delay_ms > 0 {
        cmd.arg("--test-delay-ms")
            .arg(config.test_delay_ms.to_string());
    }

    if let Some(api_key) = config.runtime.provider_accounts.revai_api_key() {
        cmd.env("BATCHALIGN_REV_API_KEY", api_key.as_str());
    }
    // Hand the child its state directory explicitly, on the child's Command
    // and never on us. See `WorkerRuntimeConfig::state_dir`.
    if let Some(state_dir) = &config.runtime.state_dir {
        cmd.env("BATCHALIGN_STATE_DIR", state_dir);
    }
    for (key, value) in worker_provider_envs(config) {
        cmd.env(key, value);
    }

    // Each worker becomes its own process group leader so that
    // killpg() in shutdown/Drop kills the worker AND all its children
    // (e.g. Stanza subprocesses).
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;

        unsafe {
            cmd.pre_exec(|| {
                libc::setpgid(0, 0);
                Ok(())
            });
        }
    }

    // On Windows, CREATE_NEW_PROCESS_GROUP places the worker in its own
    // console process group. This is the closest equivalent to Unix
    // setpgid(0,0) and allows targeted shutdown via GenerateConsoleCtrlEvent
    // or TerminateProcess.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NEW_PROCESS_GROUP = 0x00000200
        cmd.creation_flags(0x00000200);
    }

    cmd
}

/// Spawn a **detached** TCP worker daemon.
///
/// Unlike [`WorkerHandle::spawn`](super::WorkerHandle::spawn) which creates a
/// child process tied directly to the current Rust process, this launches a
/// standalone Python process with `--transport tcp` that:
/// 1. Starts with no model, binds a TCP port
/// 2. Registers itself in `workers.json`
/// 3. Reports its startup on stdout, progress then its ready line, exactly
///    as a stdio worker does, and stops writing there
/// 4. Continues running until explicitly shut down
///
/// The startup is read by the stdio worker's own reader
/// ([`WorkerHandle::read_ready_line`](super::WorkerHandle), its preamble
/// included) under the configured startup stall window, so a daemon that
/// keeps reporting progress may take as long as it needs and one that goes
/// quiet is [`WorkerError::Stalled`]. It used to report only a ready line, on
/// stderr, which the spawner scanned under a fixed total limit, giving up
/// after 100 lines that were not it.
///
/// If [`WorkerRuntimeConfig::server_instance_id`](super::config::WorkerRuntimeConfig::server_instance_id)
/// is present, the daemon registers itself as **server-owned** and should be
/// retired by that same server instance on shutdown. Otherwise it registers as
/// **external** and may be reused across server restarts.
///
/// A startup that has to queue for host memory retires idle workers through
/// `reclaim`.
///
/// Returns `(pid, port)` on success after waiting for the ready signal.
pub async fn spawn_tcp_daemon(
    config: &WorkerConfig,
    port: u16,
    reclaim: &dyn crate::host_memory::MemoryReclaim,
) -> Result<(u32, u16), WorkerError> {
    if config.task.is_some() {
        return Err(WorkerError::SpawnFailed(
            "persistent TCP workers only support profile bootstrap targets".into(),
        ));
    }
    // Memory guard, same as WorkerHandle::spawn().
    let spawn_permit = crate::worker::memory_guard::acquire_spawn_permit(config, reclaim).await?;

    let mut cmd = worker_command(&config.python_path);
    cmd.arg("--transport")
        .arg("tcp")
        .arg("--profile")
        .arg(config.profile.name())
        .arg("--lang")
        .arg(config.lang.as_worker_arg())
        .arg("--num-speakers")
        .arg(config.num_speakers.0.to_string())
        .arg("--host")
        .arg("127.0.0.1");

    if port > 0 {
        cmd.arg("--port").arg(port.to_string());
    }

    if config.test_echo {
        cmd.arg("--test-echo");
    }

    if !config.engine_overrides.is_empty() {
        cmd.arg("--engine-overrides").arg(&config.engine_overrides);
    }

    if config.runtime.force_cpu {
        cmd.arg("--force-cpu");
    }

    if config.runtime.allow_mps {
        cmd.arg("--allow-mps");
    }

    if config.verbose > 0 {
        cmd.arg("--verbose").arg(config.verbose.to_string());
    }

    cmd.args(serving_args(config));

    if config.test_delay_ms > 0 {
        cmd.arg("--test-delay-ms")
            .arg(config.test_delay_ms.to_string());
    }

    if let Some(api_key) = config.runtime.provider_accounts.revai_api_key() {
        cmd.env("BATCHALIGN_REV_API_KEY", api_key.as_str());
    }
    // Same state-dir handoff as `build_worker_command`.
    if let Some(state_dir) = &config.runtime.state_dir {
        cmd.env("BATCHALIGN_STATE_DIR", state_dir);
    }
    if let Some(server_instance_id) = config.runtime.server_instance_id.as_deref() {
        cmd.env("BATCHALIGN_SERVER_INSTANCE_ID", server_instance_id);
    }
    if let Some(server_process_id) = config.runtime.server_process_id {
        cmd.env("BATCHALIGN_SERVER_PID", server_process_id.to_string());
    }
    // The daemon writes this into its registry entry, so a server of another
    // build can refuse to adopt it (see `worker::registry`).
    cmd.env(
        crate::worker::registry::BUILD_IDENTITY_ENV,
        crate::build_hash(),
    );
    for (key, value) in worker_provider_envs(config) {
        cmd.env(key, value);
    }

    // Detach stdin. Stdout carries the startup (progress, then the ready
    // line) and the daemon stops writing to it there; stderr is piped so a
    // failed startup's tail can be read, and drained to the log after.
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // On Unix, create a new session (setsid) so the worker is fully detached
    // from the server's process group and terminal.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
    }

    info!(
        profile = %config.bootstrap_label(),
        lang = %config.lang,
        port = port,
        "Spawning TCP worker daemon"
    );

    let mut child: Command = cmd.into();
    let mut child = child
        .spawn()
        .map_err(|e| WorkerError::SpawnFailed(format!("failed to spawn TCP worker daemon: {e}")))?;

    // Tag the host-memory lease with the TCP-worker's PID, Bug 2 fix
    // (see WorkerHandle::spawn for rationale). Without this, the
    // lease's owner_pid is the daemon and ghost slots accumulate when
    // the worker dies. `child.id()` is `None` only after `wait`/`kill`,
    // which haven't run yet.
    if let Some(child_pid) = child.id() {
        spawn_permit.set_worker_pid(crate::worker::WorkerPid(child_pid));
    }
    let _spawn_permit = spawn_permit;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| WorkerError::SpawnFailed("TCP daemon stdout not captured".into()))?;
    let mut stdout_reader = tokio::io::BufReader::new(stdout);
    let mut stderr_reader = tokio::io::BufReader::new(
        child
            .stderr
            .take()
            .ok_or_else(|| WorkerError::SpawnFailed("TCP daemon stderr not captured".into()))?,
    );

    // The stdio worker's reader, preamble and stall window included; any
    // way the startup goes wrong ends the child once, with its stderr tail.
    let started = async {
        let ready =
            super::WorkerHandle::read_ready_line(&mut stdout_reader, config.startup_stall).await?;
        match (ready.ready, ready.transport) {
            (true, ReadyTransport::Tcp { host, port }) => Ok((ready.pid, host, port)),
            (false, _) => Err(WorkerError::ReadyParseFailed(
                "TCP daemon emitted ready line with ready=false".to_string(),
            )),
            (true, ReadyTransport::Stdio) => Err(WorkerError::ReadyParseFailed(
                "a worker spawned as a TCP daemon reported a stdio ready line".to_string(),
            )),
        }
    }
    .await;
    let (pid, host, port) = match started {
        Ok(started) => started,
        Err(error) => {
            return Err(super::WorkerHandle::finalize_startup_failure(
                &mut child,
                &mut stderr_reader,
                error,
            )
            .await);
        }
    };

    // The daemon writes nothing more to stdout; its stderr is drained to the
    // log for as long as this process lives. We intentionally do NOT wait on
    // the child: it is now a standalone daemon managed by the OS.
    drop(stdout_reader);
    let label = config.bootstrap_label();
    tokio::spawn(async move {
        use tokio::io::AsyncBufReadExt;
        let mut line = String::new();
        while matches!(stderr_reader.read_line(&mut line).await, Ok(read) if read > 0) {
            let trimmed = line.trim_end();
            if !trimmed.is_empty() {
                tracing::debug!(worker = %label, "{}", trimmed);
            }
            line.clear();
        }
    });

    info!(
        profile = %config.bootstrap_label(),
        lang = %config.lang,
        pid,
        %host,
        port,
        "TCP worker daemon ready"
    );

    Ok((pid, port))
}

/// How this worker serves requests: the pool's decision, the same value the
/// pool routes by, which the worker obeys rather than deriving its own.
fn serving_args(config: &WorkerConfig) -> [String; 4] {
    WorkerServing::decide(config.bootstrap_target().profile_kind(), &config.runtime).worker_args()
}

/// Build environment variables for HK ASR provider credentials.
///
/// Only injects credentials for the GPU profile (which handles ASR requests)
/// and only for the selected ASR engine override (if any), from the runtime's
/// one snapshot of the provider accounts.
pub(super) fn worker_provider_envs(config: &WorkerConfig) -> Vec<(String, String)> {
    // GPU profile includes ASR, inject provider credentials when the profile
    // handles ASR requests or the engine overrides select an HK ASR backend.
    if config.profile != WorkerProfile::Gpu {
        return Vec::new();
    }
    config
        .runtime
        .provider_accounts
        .worker_envs(selected_asr_override(&config.engine_overrides).as_deref())
        .into_iter()
        .collect()
}

/// Extract the `"asr"` key from an engine overrides JSON string.
pub(super) fn selected_asr_override(engine_overrides: &str) -> Option<String> {
    if engine_overrides.trim().is_empty() {
        return None;
    }
    let parsed = serde_json::from_str::<Value>(engine_overrides)
        .map_err(|e| tracing::warn!(overrides = engine_overrides, error = %e, "malformed engine override JSON, ignoring"))
        .ok()?;
    parsed.get("asr")?.as_str().map(str::to_string)
}
