//! Local daemon lifecycle: mirrors `batchalign/cli/daemon.py`.
//!
//! Key difference from Python: the daemon starts the **Rust server binary**
//! (`batchalign3 serve start --foreground`), not the Python server.
//!
//! ## Port policy
//!
//! The daemon takes the port REQUEST from the server config (default 8000)
//! and passes it through to the child, which may be `0` for "let the OS
//! choose". Discovery stays deterministic without a fixed port because the
//! child PUBLISHES the port it actually bound, in its handshake, and every
//! reader takes it from there; the old rule against ephemeral ports existed
//! only because nothing recorded the answer.
//!
//! ## Stale-binary detection
//!
//! `DaemonInfo` carries a `build_hash` (set at write time from
//! [`crate::build_hash()`]).  `ensure_daemon_locked()` compares it against
//! the current binary's hash and auto-restarts on mismatch. A daemon.json
//! that lacks `build_hash`, or cannot be read at all, is an
//! [`UnreadableRecord`]: the start is refused and the file kept, since it may
//! name a live daemon.

use std::path::{Path, PathBuf};
use std::time::Duration;

use std::num::NonZeroU16;

use crate::config::{PortRequest, RuntimeLayout, ServerConfig};
use crate::server_handshake::{HandshakeError, HandshakeSlot, ServerHandshake};
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::cli::client::BatchalignClient;
use crate::cli::error::{CliError, ServerBuild};
use crate::cli::python::resolve_python_executable;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum seconds for a newly spawned daemon to become usable, covering BOTH
/// phases: reaching its bind, and then answering `/health`.
///
/// 90 seconds is generous enough for a cold start on a slow machine while
/// still failing within a human-tolerable window if something is genuinely
/// broken (port conflict, missing Python, bad config).
///
/// It is ONE budget on purpose. The two phases used to have separate ceilings,
/// 30 s to publish a port and 90 s to answer afterwards, which put the small
/// number on the slow phase: everything expensive (config validation,
/// host-facts detection, database migration, cache init, and on a
/// freshly-built binary the first-exec cost) happens before the bind, while
/// the second phase only confirms a socket that already exists. A cold start
/// slower than 30 s therefore failed on the wrong clock, and the worst case
/// was 120 s rather than the 90 the constant advertises.
const HEALTH_TIMEOUT: f64 = 90.0;

/// The whole budget for getting a spawned daemon to a usable state.
///
/// Shared by `serve start` and the auto-daemon path so both mean the same
/// thing by "the daemon did not come up".
///
/// Public because it is also the floor for any harness that runs `serve start`
/// as a subprocess and kills it on a timeout. A killer whose budget is smaller
/// than this one preempts the wait below, so the command dies by signal with no
/// diagnosis instead of reporting which phase it was still in. The CLI test
/// suite derives its kill budget from this function rather than restating a
/// number that could drift under it (`tests/live_deadline`,
/// `CliRunBudget::DaemonStart`).
pub fn startup_budget() -> Duration {
    Duration::from_secs_f64(HEALTH_TIMEOUT)
}

/// How often to poll the daemon's `/health` endpoint while waiting for
/// startup. 1 second balances responsiveness (the user sees the daemon come
/// up within a second of it being ready) against CPU/network cost (one
/// loopback HTTP request per second is negligible).
const HEALTH_POLL: f64 = 1.0;

fn runtime_layout() -> RuntimeLayout {
    RuntimeLayout::from_env()
}

/// The loopback URL a locally-running server can be reached at.
///
/// Answers "where is the local server" from the strongest evidence available,
/// in order: the port the server PUBLISHED after binding, then the port the
/// config asked for if that request named one. `None` means there is nothing
/// to try, which for an ephemeral request with no published handshake is the
/// honest answer rather than a URL built from a placeholder.
///
/// One owner for a question that had two callers building the URL from
/// `cfg.port` directly, both of which silently assumed the request had been
/// granted.
pub(crate) fn local_server_url(layout: &RuntimeLayout, configured: PortRequest) -> Option<String> {
    local_port(layout, configured).map(|port| format!("http://127.0.0.1:{port}"))
}

/// The port a local server can be reached on, from the strongest evidence.
///
/// Takes the configured request rather than loading `server.yaml` itself.
/// Every caller already holds a validated config, and reloading it here meant a
/// second parse plus a second `is_dir()` on every configured media root, and
/// printed every config warning a second time.
pub(crate) fn local_port(layout: &RuntimeLayout, configured: PortRequest) -> Option<NonZeroU16> {
    if let Ok(Some(handshake)) = ServerHandshake::read(layout.state_dir(), HandshakeSlot::Main)
        && let Some(port) = handshake.bound_port()
    {
        return NonZeroU16::new(port.get());
    }
    // No published port: either an older server, or none running. A fixed
    // request is a usable guess for the first case; an ephemeral one leaves
    // nothing to guess with, and guessing is what this change removed.
    configured.fixed()
}

// ---------------------------------------------------------------------------
// Profiles
// ---------------------------------------------------------------------------

/// Which daemon role to start or connect to.
///
/// The CLI can manage two independent daemon processes simultaneously.
/// Each profile gets its own state file, lock file, and log file so they
/// do not interfere with each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DaemonProfile {
    /// The primary managed daemon profile for explicit `serve` lifecycle
    /// commands, auto-daemon CLI routing, and compatibility helpers. Uses the
    /// system or venv Python resolved by [`resolve_python_executable()`].
    Main,
    /// A secondary daemon dedicated to transcribe workloads that require
    /// a different Python environment or model stack from the main daemon.
    /// Selected when
    /// the dispatch layer detects a transcribe command and the sidecar
    /// Python is available. Uses `BATCHALIGN_SIDECAR_PYTHON` or falls
    /// back to `~/.batchalign3/sidecar/.venv/bin/python`.
    Sidecar,
}

impl DaemonProfile {
    fn label(self) -> &'static str {
        match self {
            Self::Main => "local",
            Self::Sidecar => "sidecar",
        }
    }

    /// The handshake slot this profile's server publishes to.
    ///
    /// Every other per-profile artifact was already namespaced; this closes the
    /// last shared one, which mattered once discovery started reading the
    /// published port rather than re-deriving it from config.
    fn handshake_slot(self) -> HandshakeSlot {
        match self {
            Self::Main => HandshakeSlot::Main,
            Self::Sidecar => HandshakeSlot::Sidecar,
        }
    }

    fn state_file(self, dir: &Path) -> PathBuf {
        match self {
            Self::Main => dir.join("daemon.json"),
            Self::Sidecar => dir.join("sidecar-daemon.json"),
        }
    }

    fn lock_file(self, dir: &Path) -> PathBuf {
        match self {
            Self::Main => dir.join("daemon.lock"),
            Self::Sidecar => dir.join("sidecar-daemon.lock"),
        }
    }

    fn log_file(self, dir: &Path) -> PathBuf {
        match self {
            Self::Main => dir.join("daemon.log"),
            Self::Sidecar => dir.join("sidecar-daemon.log"),
        }
    }

    fn startup_message(self) -> &'static str {
        match self {
            Self::Main => "Starting local daemon...",
            Self::Sidecar => "Starting sidecar daemon for transcribe workloads...",
        }
    }

    fn check_manual_server(self) -> bool {
        matches!(self, Self::Main)
    }

    fn default_python(self, dir: &Path) -> String {
        match self {
            Self::Main => resolve_python_executable(),
            Self::Sidecar => std::env::var("BATCHALIGN_SIDECAR_PYTHON").unwrap_or_else(|_| {
                dir.join("sidecar")
                    .join(".venv")
                    .join("bin")
                    .join("python")
                    .to_string_lossy()
                    .into_owned()
            }),
        }
    }

    fn require_sidecar_python_file(self) -> bool {
        matches!(self, Self::Sidecar) && std::env::var("BATCHALIGN_SIDECAR_PYTHON").is_err()
    }
}

// ---------------------------------------------------------------------------
// DaemonInfo: state file
// ---------------------------------------------------------------------------

/// State persisted to `daemon.json` so the CLI can reconnect to a running daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonInfo {
    /// OS process ID of the daemon.
    pub pid: u32,
    /// The build identity of the binary that started the daemon, compared
    /// with this binary's to decide whether the daemon is stale. Required:
    /// a state file without one is unreadable (see [`StateFile`]), never
    /// compared by version number instead.
    pub build_hash: String,
    /// The daemon process's *resolved* `force_cpu` value (operator
    /// intent merged with the host-facts recommendation), captured at
    /// spawn time. Used by `runtime_mismatch` to decide whether the
    /// next CLI invocation needs a restart: if the next call's
    /// resolved value differs from this stored one, the daemon's
    /// view of the host has drifted (operator edited server.yaml, or
    /// the CLI's `--force-cpu` flag changed) and the daemon must
    /// restart to pick up the new value.
    ///
    /// Pre-C2.2 daemon.json files recorded the raw CLI `--force-cpu`
    /// flag here. The shape is unchanged (still a `bool`), but the
    /// semantic shifted: resolved-vs-resolved comparison now matches
    /// the daemon's actual runtime behavior. Existing daemon.json
    /// files trigger one self-correcting restart on first contact
    /// post-upgrade: Apple Silicon hosts go from raw=false to
    /// resolved=true and the next invocation kicks the daemon over.
    pub force_cpu: bool,
    /// The daemon's resolved `allow_mps` value (the explicit Apple-GPU
    /// opt-in), captured at spawn time, compared by `runtime_mismatch`
    /// the same way as `force_cpu`.
    pub allow_mps: bool,
    /// The `--workers` value the daemon was started with; `None` when the
    /// operator didn't pass `--workers` (the daemon resolved its own
    /// per-job parallelism from host facts). Used by the warm-reuse path to fire the `--workers` shadowing
    /// warning only when the requested value actually differs from
    /// the running daemon's: eliminating false positives when the
    /// operator re-passes a value that already matches.
    pub workers: Option<crate::api::NumWorkers>,
    /// The `--timeout` value the daemon was started with; `None` when it
    /// was not passed. Same false-positive elimination as `workers`,
    /// applied to the daemon's per-task ceiling. A `daemon.json` written
    /// before the type refused zero may hold `0`, read as "not passed".
    #[serde(default, deserialize_with = "crate::config::zero_as_no_override")]
    pub audio_task_timeout_s: Option<crate::api::PositiveSeconds>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// What the operator asked of a daemon on this invocation: the CLI's device
/// switches and the two values fixed at a daemon's startup.
///
/// One value instead of the `(bool, bool, Option<usize>, Option<u64>)` that
/// every function from `ensure_daemon` to `start_daemon` used to take
/// positionally, with the worker count cast to `u32` to persist it and back
/// to `usize` to compare it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DaemonRequest {
    /// `--force-cpu`.
    pub force_cpu: bool,
    /// `--allow-mps`.
    pub allow_mps: bool,
    /// `--workers`, when passed.
    pub workers: Option<crate::api::NumWorkers>,
    /// `--timeout`, when passed.
    pub timeout: Option<crate::api::PositiveSeconds>,
}

/// Return the main daemon URL or `None` if the daemon cannot be started.
///
/// `config` is the caller's loaded `server.yaml`: the daemon path reads the
/// port request, the verbosity and the device settings from it rather than
/// loading the file again (it used to load it up to four times). A state file
/// that cannot be read refuses the start (see [`UnreadableRecord`]) rather
/// than writing a new record over one that may name a live daemon.
pub async fn ensure_daemon(
    layout: &RuntimeLayout,
    config: &ServerConfig,
    request: DaemonRequest,
) -> Result<Option<String>, CliError> {
    ensure_daemon_for(DaemonProfile::Main, layout, config, request).await
}

/// Return the sidecar daemon URL or `None` if the daemon cannot be started.
pub async fn ensure_sidecar_daemon(
    layout: &RuntimeLayout,
    config: &ServerConfig,
    request: DaemonRequest,
) -> Result<Option<String>, CliError> {
    ensure_daemon_for(DaemonProfile::Sidecar, layout, config, request).await
}

/// What a stop found and did.
///
/// It replaced a `bool` that said only whether a signal was sent, so
/// "nothing recorded", "recorded but already gone" and "a record that cannot
/// be read, left in place" all printed "No server process found".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopOutcome {
    /// A recorded, live process was stopped and its record removed.
    Stopped {
        /// The process that was stopped.
        pid: u32,
    },
    /// Nothing was recorded.
    NotRunning,
    /// A record named a process that had already exited; the record was
    /// removed.
    AlreadyDead {
        /// The process the record named.
        pid: u32,
    },
    /// A record exists but cannot be read. It may name a live process, so
    /// it was left in place and nothing was signalled.
    Unreadable(UnreadableRecord),
}

/// Stop the main daemon.
pub async fn stop_daemon() -> Result<StopOutcome, CliError> {
    stop_profile(DaemonProfile::Main).await
}

/// Stop the sidecar daemon.
pub async fn stop_sidecar_daemon() -> Result<StopOutcome, CliError> {
    stop_profile(DaemonProfile::Sidecar).await
}

/// Read the main daemon state file. Public for status checks.
pub fn read_daemon_info() -> Result<Option<DaemonInfo>, UnreadableRecord> {
    let layout = runtime_layout();
    StateFile::read(DaemonProfile::Main, layout.state_dir()).recorded()
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

async fn ensure_daemon_for(
    profile: DaemonProfile,
    layout: &RuntimeLayout,
    config: &ServerConfig,
    request: DaemonRequest,
) -> Result<Option<String>, CliError> {
    let lock = DaemonStartLock::acquire_async(profile, layout.state_dir()).await?;
    let result = ensure_daemon_locked(&lock, layout, config, request).await;
    drop(lock);
    result
}

/// The start lock of one daemon profile, held for as long as this value
/// lives. Only [`DaemonStartLock::acquire_async`] makes one, and
/// [`ensure_daemon_locked`] and [`start_daemon`] take it instead of a bare
/// profile, so checking, starting or replacing a daemon without holding that
/// profile's lock does not compile, and neither does doing it under another
/// profile's lock. Stopping (`stop_profile`) takes it too, and writing or
/// removing the state file needs it.
struct DaemonStartLock {
    profile: DaemonProfile,
    /// The state directory the lock (and every file it guards) lives in.
    state_dir: PathBuf,
    /// The OS lock; holding this field is holding the lock, and dropping it
    /// releases the lock.
    #[expect(
        dead_code,
        reason = "held for its lifetime; dropping it releases the lock"
    )]
    held: crate::file_lock::HeldFileLock,
}

impl DaemonStartLock {
    /// [`Self::acquire_async`] on the calling thread, for tests.
    #[cfg(test)]
    fn acquire(profile: DaemonProfile, state_dir: &Path) -> Result<Self, std::io::Error> {
        Ok(Self {
            profile,
            state_dir: state_dir.to_path_buf(),
            held: crate::file_lock::HeldFileLock::acquire(profile.lock_file(state_dir))?,
        })
    }

    /// Take `profile`'s lock in `state_dir` off the async runtime, waiting
    /// while another `batchalign3` process holds it. Only "held elsewhere"
    /// means wait; any other failure is an error, never mistaken for
    /// contention.
    async fn acquire_async(
        profile: DaemonProfile,
        state_dir: &Path,
    ) -> Result<Self, std::io::Error> {
        let held =
            crate::file_lock::HeldFileLock::acquire_async(profile.lock_file(state_dir)).await?;
        Ok(Self {
            profile,
            state_dir: state_dir.to_path_buf(),
            held,
        })
    }

    /// The profile's state file, as it is now.
    fn read_state(&self) -> StateFile {
        StateFile::read(self.profile, &self.state_dir)
    }

    /// Record the daemon this lock's holder just started.
    fn write_info(&self, info: &DaemonInfo) -> Result<(), CliError> {
        let state_path = self.profile.state_file(&self.state_dir);
        crate::atomic_file::write_atomically(
            &state_path,
            serde_json::to_string(info)?.as_bytes(),
            crate::atomic_file::Existing::Replace,
            crate::atomic_file::Audience::Owner,
        )?;
        Ok(())
    }

    /// Remove the profile's state file. Already absent is success; any
    /// other failure is an error, since a state file left behind names a
    /// daemon that no longer runs.
    fn remove_state(&self) -> Result<(), std::io::Error> {
        match std::fs::remove_file(self.profile.state_file(&self.state_dir)) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }

    /// The profile's server handshake, as it is now. The one read both the
    /// manual-server probe and `serve stop` use.
    fn read_handshake(&self) -> Result<Option<ServerHandshake>, HandshakeError> {
        ServerHandshake::read(&self.state_dir, self.profile.handshake_slot())
    }

    /// Remove the profile's server handshake (`server.pid` for the main
    /// profile) for a process that is gone, only while the record still
    /// names that process.
    fn remove_handshake_of(&self, pid: u32) -> Result<(), CliError> {
        ServerHandshake::remove_if_names(&self.state_dir, self.profile.handshake_slot(), pid)
            .map(drop)
            .map_err(|error| CliError::Io(std::io::Error::other(error.to_string())))
    }
}

/// Stop `profile`'s daemon under its start lock, so a stop cannot race a
/// start (or another stop) of the same profile, and remove its state file.
/// A state file that cannot be read is left in place and said so: it may
/// still name a live daemon.
async fn stop_profile(profile: DaemonProfile) -> Result<StopOutcome, CliError> {
    let layout = runtime_layout();
    let lock = DaemonStartLock::acquire_async(profile, layout.state_dir()).await?;
    let info = match lock.read_state().recorded() {
        Ok(Some(info)) => info,
        Ok(None) => return Ok(StopOutcome::NotRunning),
        Err(unreadable) => return Ok(StopOutcome::Unreadable(unreadable)),
    };
    // A PID that is no longer alive is not signalled: it may have been reused.
    let outcome = if is_process_alive(info.pid) && stop_server_process(info.pid) {
        StopOutcome::Stopped { pid: info.pid }
    } else {
        StopOutcome::AlreadyDead { pid: info.pid }
    };
    lock.remove_state()?;
    Ok(outcome)
}

/// `serve stop`: stop the server recorded in the main handshake, under the
/// main profile's lock, so it cannot race a daemon start. An unreadable
/// handshake is left in place and said so: it may still name a live server.
pub(crate) async fn stop_manual_server(layout: &RuntimeLayout) -> Result<StopOutcome, CliError> {
    let lock = DaemonStartLock::acquire_async(DaemonProfile::Main, layout.state_dir()).await?;
    let handshake = match lock.read_handshake() {
        Ok(Some(handshake)) => handshake,
        Ok(None) => return Ok(StopOutcome::NotRunning),
        Err(error) => {
            return Ok(StopOutcome::Unreadable(UnreadableRecord {
                path: ServerHandshake::path_in(&lock.state_dir, HandshakeSlot::Main),
                reason: error.to_string(),
            }));
        }
    };
    // Both states carry a PID, and stopping is the same act either way: a
    // server that has spawned but not yet bound still needs killing. A PID
    // that is no longer alive is not signalled, since it may have been reused.
    let pid = handshake.pid();
    let outcome = if is_process_alive(pid) && stop_server_process(pid) {
        StopOutcome::Stopped { pid }
    } else {
        StopOutcome::AlreadyDead { pid }
    };
    lock.remove_handshake_of(pid)?;
    Ok(outcome)
}

/// Whether the daemon was started by a different build than this one:
/// build identity, never a version number.
fn is_stale(info: &DaemonInfo) -> bool {
    info.build_hash != crate::build_hash()
}

fn runtime_mismatch(info: &DaemonInfo, flags: DaemonDeviceFlags) -> bool {
    info.force_cpu != flags.resolved_force_cpu || info.allow_mps != flags.resolved_allow_mps
}

/// True when the operator passed a CLI flag whose value differs from
/// the running daemon's recorded value. Suppresses the warm-reuse
/// shadowing warning when the user re-passes a value that already
/// matches the daemon. A `None` on the daemon side (pre-upgrade
/// daemon.json that lacks the field) is treated as "unknown" and
/// triggers the warning so the user still gets a signal, falling
/// back to the pre-persisted-value behavior on first contact after
/// upgrade.
fn flag_shadows_daemon<T: PartialEq>(requested: Option<T>, persisted: Option<T>) -> bool {
    match (requested, persisted) {
        (None, _) => false,
        (Some(_), None) => true,
        (Some(r), Some(p)) => r != p,
    }
}

/// The operator's device flags as given on the CLI plus their
/// host-resolved values, computed once per invocation so the restart
/// decision and the persisted `DaemonInfo` stay consistent (and so the
/// spawn/persist plumbing threads one named value instead of a parade
/// of positional bools).
#[derive(Clone, Copy, Debug)]
struct DaemonDeviceFlags {
    /// Raw `--force-cpu` switch (drives the spawned subprocess arg).
    cli_force_cpu: bool,
    /// Raw `--allow-mps` switch (drives the spawned subprocess arg).
    cli_allow_mps: bool,
    /// `force_cpu` merged with the host-facts recommendation.
    resolved_force_cpu: bool,
    /// `allow_mps` resolved (no recommendation; operator opt-in only).
    resolved_allow_mps: bool,
}

/// Resolve the daemon's effective `force_cpu` value given the CLI
/// flag plus the host's deployed `server.yaml` and detected facts.
///
/// Mirrors the boundary conversion `serve_cmd::start` does for the
/// daemon process itself: the CLI's presence-only `--force-cpu`
/// switch becomes `Some(true)` in the override, the YAML-side value
/// stays as configured, and the host-facts pipeline merges the two.
/// Used by `ensure_daemon_locked` for restart decisions and by
/// `start_daemon` for the value persisted to `DaemonInfo`.
///
/// Reads the caller's loaded config, so the value resolved here is the one
/// the daemon process will boot with from the same file.
fn resolve_device_flags_for_daemon(
    config: &ServerConfig,
    cli_force_cpu: bool,
    cli_allow_mps: bool,
) -> DaemonDeviceFlags {
    let mut cfg = config.clone();
    if cli_force_cpu {
        cfg.force_cpu = Some(true);
    }
    if cli_allow_mps {
        cfg.allow_mps = Some(true);
    }
    let effective = crate::host_facts::EffectiveConfig::resolve_from_server_config(&cfg);
    DaemonDeviceFlags {
        cli_force_cpu,
        cli_allow_mps,
        resolved_force_cpu: effective.force_cpu,
        resolved_allow_mps: effective.allow_mps,
    }
}

async fn ensure_daemon_locked(
    lock: &DaemonStartLock,
    layout: &RuntimeLayout,
    config: &ServerConfig,
    request: DaemonRequest,
) -> Result<Option<String>, CliError> {
    let profile = lock.profile;
    let dir = layout.state_dir();
    if profile.check_manual_server()
        && let Some(url) = detect_manual_server(lock, config.port).await?
    {
        return Ok(Some(url));
    }

    // Compute the resolved device flags once: they drive both the
    // restart decision (vs. stored DaemonInfo) and the values persisted
    // when start_daemon writes a new DaemonInfo. Resolving here keeps
    // both consistent.
    let device_flags =
        resolve_device_flags_for_daemon(config, request.force_cpu, request.allow_mps);
    let start = DaemonStart {
        port: config.port,
        verbose: config.verbose,
        flags: device_flags,
        workers: request.workers,
        timeout: request.timeout,
    };

    // An unreadable state file refuses the start: it may name a live daemon,
    // and writing a new record over it would strand that process.
    if let Some(info) = lock.read_state().recorded()? {
        if is_process_alive(info.pid) {
            if is_stale(&info) {
                eprintln!(
                    "Restarting {} daemon (stale build: {} -> {})...",
                    profile.label(),
                    info.build_hash,
                    crate::build_hash(),
                );
                stop_server_process(info.pid);
                lock.remove_state()?;
                return start_daemon(lock, layout, start).await;
            }

            if runtime_mismatch(&info, device_flags) {
                eprintln!(
                    "Restarting {} daemon (resolved force_cpu {} -> {}, allow_mps {} -> {})...",
                    profile.label(),
                    info.force_cpu,
                    device_flags.resolved_force_cpu,
                    info.allow_mps,
                    device_flags.resolved_allow_mps,
                );
                stop_server_process(info.pid);
                lock.remove_state()?;
                return start_daemon(lock, layout, start).await;
            }

            // The port comes from the handshake, which is where a bound
            // server publishes it. `daemon.json` used to mirror it; two
            // records of one fact is what this change set removed.
            let published = ServerHandshake::published_port(dir, profile.handshake_slot());
            if let Some(published) = published
                && health_check(published.get()).await
            {
                // The daemon's per-task ceiling (`audio_task_timeout_s`)
                // and per-job parallelism (`max_workers_per_job`) are
                // both fixed at daemon startup. On the warm-reuse path
                // the user's `--timeout` / `--workers` are silently
                // discarded, without surfacing that, a request can
                // fail with a timeout below the requested value, or a
                // multi-file batch can run serially because the daemon
                // stayed at workers=1 (the host-facts auto-clamp on
                // hosts without a usable GPU). Auto-restart is not the
                // answer here: the running daemon may be processing
                // other operators' jobs, and killing it would discard
                // their in-flight work. Warning is honest signal.
                if flag_shadows_daemon(request.timeout, info.audio_task_timeout_s) {
                    let running = match info.audio_task_timeout_s {
                        Some(s) => format!("{s}s"),
                        None => "<not set>".to_owned(),
                    };
                    let requested = match request.timeout {
                        Some(s) => format!("{s}s"),
                        None => "<not set>".to_owned(),
                    };
                    eprintln!(
                        "warning: --timeout {requested} requested but the {} daemon was \
                         started with --timeout {running}. The per-task ceiling stays at \
                         the running value for this submission. To apply the new ceiling, \
                         run `batchalign3 serve stop` then `batchalign3 serve start --timeout \
                         <secs>`, or pass `--no-server` to bypass the daemon entirely.",
                        profile.label(),
                    );
                }
                if flag_shadows_daemon(request.workers, info.workers) {
                    let running = match info.workers {
                        Some(n) => n.to_string(),
                        None => "<not set>".to_owned(),
                    };
                    let requested = match request.workers {
                        Some(n) => n.to_string(),
                        None => "<not set>".to_owned(),
                    };
                    eprintln!(
                        "warning: --workers {requested} requested but the {} daemon was \
                         started with --workers {running}. Per-job parallelism stays at \
                         the running value for this submission. To apply the new value, \
                         run `batchalign3 serve stop` then `batchalign3 serve start --workers \
                         <N>`, or pass `--no-server` to bypass the daemon entirely.",
                        profile.label(),
                    );
                }
                return Ok(Some(format!("http://127.0.0.1:{published}")));
            }

            stop_server_process(info.pid);
            lock.remove_state()?;
            return start_daemon(lock, layout, start).await;
        }
        // Process is dead but state file exists -- stale PID file.
        debug!(
            profile = profile.label(),
            pid = info.pid,
            "Cleaning up stale daemon state file (process is dead)"
        );
        lock.remove_state()?;
    }

    start_daemon(lock, layout, start).await
}

/// Find a manually-started server, if one is running and reachable.
///
/// Reads the port the server PUBLISHED rather than re-deriving it from
/// `server.yaml`. The config's port is a request: it is what an operator asked
/// for, which for an ephemeral request names no port at all, and even for a
/// fixed request can differ from reality if the file was edited after the
/// server started.
async fn detect_manual_server(
    lock: &DaemonStartLock,
    configured: PortRequest,
) -> Result<Option<String>, CliError> {
    let handshake = match lock.read_handshake() {
        Ok(Some(handshake)) => handshake,
        Ok(None) => return Ok(None),
        Err(error) => {
            // Deliberately left in place rather than deleted. A file we cannot
            // read is the only evidence that a server may still be running,
            // and removing it invites a second server onto the same port.
            debug!(%error, "Ignoring unreadable server handshake");
            return Ok(None);
        }
    };

    let pid = handshake.pid();
    if !is_process_alive(pid) {
        debug!(pid, "Removing stale server handshake (process is dead)");
        lock.remove_handshake_of(pid)?;
        return Ok(None);
    }

    let port = match handshake.bound_port() {
        Some(port) => port.get(),
        None => {
            // The server has not published a port: either it is still starting,
            // or it predates the published handshake. A fixed configured port
            // is a usable guess for the older-server case; an ephemeral request
            // leaves nothing to guess with, and guessing is what this change
            // exists to stop.
            match configured.fixed() {
                Some(port) => {
                    debug!(
                        pid,
                        port = port.get(),
                        "Server published no port; falling back to the configured one"
                    );
                    port.get()
                }
                None => {
                    debug!(pid, "Server published no port and none is configured");
                    return Ok(None);
                }
            }
        }
    };

    match ManualServerDisposition::probe(port).await {
        ManualServerDisposition::Reusable { url } => {
            debug!(port, pid, "Reusing manual server");
            Ok(Some(url))
        }
        ManualServerDisposition::Unreachable => Ok(None),
        // Dispatch refuses this job on this same error a moment later, so the
        // warning that used to print here was noise before a refusal. The
        // refusal belongs where the reuse decision is made.
        ManualServerDisposition::ForeignBuild { url, theirs } => {
            Err(CliError::ServerBuildMismatch {
                server: url,
                server_build: theirs,
                client_build: crate::build_hash().to_owned(),
            })
        }
    }
}

/// What a manually started server on a published port turned out to be.
///
/// Three facts the caller must act on differently, so the probe's answer is a
/// type rather than a URL with a warning printed beside it. The check this
/// replaced returned `()`: it could say something on stderr but could not
/// hand the caller the answer, so the URL it came back with looked identical
/// whether the build had matched, differed, or never been compared. A
/// validator that returns nothing leaves no proof it ran.
enum ManualServerDisposition {
    /// A batchalign3 server on OUR OWN build answered. Only this variant
    /// carries a URL to submit to.
    Reusable {
        /// Where it answered.
        url: String,
    },
    /// Nothing answered a batchalign3 health check there: no response, a
    /// non-success status, or a body that is not a health response. The
    /// caller falls through to the auto-daemon path, which probes the
    /// configured port itself.
    Unreachable,
    /// A batchalign3 server answered on a build that is not ours, including
    /// reporting none at all (absence of evidence cannot prove a match).
    /// Reusing it would serve this invocation's requests from code this
    /// binary did not build and cannot vouch for.
    ForeignBuild {
        /// Where it answered.
        url: String,
        /// What it said about its build.
        theirs: ServerBuild,
    },
}

impl ManualServerDisposition {
    /// The one constructor, so a `Reusable` cannot exist without a build match
    /// proven by [`build_hash_matches_ours`].
    ///
    /// Probes with the same single-shot health check the auto-daemon path
    /// uses: one request, no retry budget, and a strict parse, so an
    /// unrelated service holding the port cannot be mistaken for a daemon.
    async fn probe(port: u16) -> Self {
        let url = format!("http://127.0.0.1:{port}");
        match fast_single_shot_health_check(port).await {
            None => Self::Unreachable,
            Some(health) if build_hash_matches_ours(&health.build_hash) => Self::Reusable { url },
            Some(health) => Self::ForeignBuild {
                url,
                theirs: match health.build_hash.as_str() {
                    "" => ServerBuild::Unreported,
                    reported => ServerBuild::Reported(reported.to_owned()),
                },
            },
        }
    }
}

/// Whether a reported build hash proves the daemon that reported it is
/// running the SAME build as this CLI invocation.
///
/// The single owner of this comparison: [`ManualServerDisposition::probe`]
/// (the manual-server path) and [`probe_fixed_port`]'s adoption decision
/// (the auto-daemon path) both call this rather than each spelling out
/// `!reported.is_empty() && reported == crate::build_hash()`
/// independently, which is how the auto-daemon path came to skip the
/// comparison entirely.
///
/// The empty string ("unknown", a pre-build-hash daemon, or a health check
/// that could not be parsed strictly) can never prove a match: absence of
/// evidence is not evidence of a match, and a caller that treated it as
/// one would adopt a daemon it cannot actually vouch for.
fn build_hash_matches_ours(reported: &str) -> bool {
    !reported.is_empty() && reported == crate::build_hash()
}

/// What a probe of the daemon's configured fixed port found, immediately
/// before a spawn attempt.
///
/// There are four facts a caller must react to differently, so a bare
/// `bool` (or a bind `Result` alone) is the wrong type here: whether the
/// port is free (safe to spawn into), already held by a healthy batchalign3
/// daemon on OUR OWN build (adopt it, never spawn a second one), held by a
/// batchalign3 daemon on a DIFFERENT build (refuse and say so, exactly the
/// posture [`ManualServerDisposition`] already takes for a manually started
/// server), or held by something else entirely (refuse rather than
/// spawn into a bind that is guaranteed to fail).
#[derive(Debug, Clone, PartialEq, Eq)]
enum PortOccupant {
    /// The port was free at probe time: a bind attempt succeeded and was
    /// released immediately so the daemon about to be spawned can bind it
    /// itself.
    ///
    /// This still leaves a TOCTOU window between the probe and the actual
    /// spawn (another process could bind in between); the probe's job is
    /// to avoid the COMMON case of blindly spawning into a port a healthy
    /// daemon or an unrelated process already holds, never to make that
    /// window zero. The spawned child's own bind failure, if the window is
    /// lost, is still handled where `cmd.spawn()` and the handshake wait
    /// report their own errors.
    Free,
    /// A batchalign3 daemon on OUR OWN build answered a health check on
    /// the port. Only this variant is safe to adopt: reusing a daemon
    /// means every subsequent request in this CLI invocation is answered
    /// by code this binary can vouch for.
    ExistingDaemon {
        /// The responding daemon's build identity (`HealthResponse::build_hash`),
        /// proven equal to `crate::build_hash()` by construction
        /// (see [`build_hash_matches_ours`]): never empty.
        build_hash: String,
    },
    /// A batchalign3 daemon answered a health check on the port, but its
    /// build does not match ours (including the empty "unknown" sentinel,
    /// which cannot prove a match either way). Adopting it silently would
    /// mean this invocation's requests are served by code this binary did
    /// not build and cannot vouch for; refuse and name the exact restart
    /// command, the same posture [`ManualServerDisposition`] already takes
    /// for a manually started server.
    StaleDaemon {
        /// The responding daemon's reported build (possibly empty).
        theirs: String,
        /// This CLI invocation's own build.
        ours: String,
    },
    /// The port is held by something that did not answer a batchalign3
    /// health check: a foreign process, or a batchalign3 process that is
    /// alive but not (yet, or any longer) healthy.
    Occupied,
}

/// Probe a fixed port immediately before spawning a daemon on it.
///
/// This is a SINGLE bind attempt, deliberately not a retry loop: retrying a
/// bind against a port a healthy daemon or an unrelated process already
/// holds cannot succeed no matter how many times it is tried, so the fix
/// for a failed bind is to stop retrying and branch on what is actually
/// there, not to try again. This is what closes the incident where a
/// cancelled/racing dispatch caused eight auto-spawn attempts to fail
/// identically against the same occupied port.
///
/// The follow-up health check is deliberately the SAME short, single-shot
/// client `startup_health_check` uses (3s request, 1s connect), never
/// [`BatchalignClient`]: that client's `request_with_retry` exists to
/// tolerate a slow-starting daemon and retries connect/timeout failures
/// with backoff, which is exactly the wrong shape for a probe whose whole
/// point is answering "what is here" in one shot before a spawn decision.
async fn probe_fixed_port(port: NonZeroU16) -> PortOccupant {
    match tokio::net::TcpListener::bind(("127.0.0.1", port.get())).await {
        Ok(listener) => {
            drop(listener);
            PortOccupant::Free
        }
        Err(_) => match fast_single_shot_health_check(port.get()).await {
            Some(health) => {
                if build_hash_matches_ours(&health.build_hash) {
                    PortOccupant::ExistingDaemon {
                        build_hash: health.build_hash,
                    }
                } else {
                    PortOccupant::StaleDaemon {
                        theirs: health.build_hash,
                        ours: crate::build_hash().to_owned(),
                    }
                }
            }
            None => PortOccupant::Occupied,
        },
    }
}

/// One-shot, short-timeout `/health` check that parses the full typed
/// response (unlike [`startup_health_check`], which only needs success/
/// failure). `None` covers every way this can fail to identify a daemon:
/// no response, a non-success status, or a body that does not parse as
/// [`crate::types::response::HealthResponse`] (which is REQUIRED-field
/// strict specifically so an unrelated service on the port cannot be
/// misidentified as a batchalign3 daemon).
async fn fast_single_shot_health_check(
    port: u16,
) -> Option<crate::types::response::HealthResponse> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .connect_timeout(Duration::from_secs(1))
        .build()
        .ok()?;
    let response = client
        .get(format!("http://127.0.0.1:{port}/health"))
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    response.json().await.ok()
}

/// Best-effort description of whatever holds a port, for an operator-facing
/// refusal message. `None` when nothing more specific than the port number
/// itself could be determined (no `lsof`, or it reported nothing usable).
#[cfg(unix)]
fn describe_port_holder(port: u16) -> Option<String> {
    let pid_output = std::process::Command::new("lsof")
        .args(["-i", &format!("tcp:{port}"), "-sTCP:LISTEN", "-t"])
        .output()
        .ok()?;
    if !pid_output.status.success() {
        return None;
    }
    let pid = String::from_utf8_lossy(&pid_output.stdout)
        .lines()
        .next()?
        .trim()
        .to_owned();
    if pid.is_empty() {
        return None;
    }
    let name = std::process::Command::new("ps")
        .args(["-p", &pid, "-o", "comm="])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|name| !name.is_empty());
    Some(match name {
        Some(name) => format!("process {pid} ({name})"),
        None => format!("process {pid}"),
    })
}

#[cfg(not(unix))]
fn describe_port_holder(_port: u16) -> Option<String> {
    None
}

/// What a daemon is started with, taken from the caller's config and request.
///
/// `flags` carries the raw CLI switches (which control the spawned
/// subprocess's arguments: operator intent verbatim) and the resolved
/// values (persisted to `DaemonInfo` so later `runtime_mismatch`
/// calls compare apples to apples).
#[derive(Debug, Clone, Copy)]
struct DaemonStart {
    /// The port REQUEST from `server.yaml`, which may be ephemeral.
    port: PortRequest,
    /// `server.yaml`'s `verbose`, passed on as `-v` flags.
    verbose: u8,
    flags: DaemonDeviceFlags,
    workers: Option<crate::api::NumWorkers>,
    timeout: Option<crate::api::PositiveSeconds>,
}

async fn start_daemon(
    lock: &DaemonStartLock,
    layout: &RuntimeLayout,
    start: DaemonStart,
) -> Result<Option<String>, CliError> {
    let DaemonStart {
        port,
        verbose,
        flags,
        workers,
        timeout,
    } = start;
    let profile = lock.profile;
    let dir = layout.state_dir();
    let python = profile.default_python(dir);
    if profile.require_sidecar_python_file() && !Path::new(&python).is_file() {
        eprintln!(
            "warning: sidecar python not found at {python}. \
             Set BATCHALIGN_SIDECAR_PYTHON to a Python with transcribe deps."
        );
        return Ok(None);
    }

    // A fixed port can already be held by a healthy daemon (a sibling
    // invocation that won the race to spawn one, or a manually started
    // server `detect_manual_server` did not catch) or by something
    // unrelated. An ephemeral request (`port.fixed() == None`) has nothing
    // to probe: the OS assigns a free port by construction, so there is no
    // collision to check for.
    if let Some(fixed_port) = port.fixed() {
        match probe_fixed_port(fixed_port).await {
            PortOccupant::Free => {}
            PortOccupant::ExistingDaemon { build_hash } => {
                eprintln!(
                    "{} daemon already running on port {fixed_port} ({build_hash}); reusing it.",
                    profile.label(),
                );
                return Ok(Some(format!("http://127.0.0.1:{fixed_port}")));
            }
            PortOccupant::StaleDaemon { theirs, ours } => {
                let theirs_label = if theirs.is_empty() {
                    "unknown build".to_owned()
                } else {
                    theirs
                };
                eprintln!(
                    "warning: a {} daemon is already running on port {fixed_port}, but on a \
                     different build ({theirs_label} != {ours}). Refusing to reuse it silently. \
                     Restart with `batchalign3 serve stop && batchalign3 serve start`.",
                    profile.label(),
                );
                return Ok(None);
            }
            PortOccupant::Occupied => {
                let holder = describe_port_holder(fixed_port.get())
                    .unwrap_or_else(|| "an unidentified process".to_owned());
                eprintln!(
                    "warning: port {fixed_port} is already in use by {holder}, which did not \
                     answer a batchalign3 health check. Refusing to start the {} daemon there. \
                     Configure a different port in server.yaml, or free the port and retry.",
                    profile.label(),
                );
                return Ok(None);
            }
        }
    }

    eprintln!("{}", profile.startup_message());

    let exe = crate::cli::self_exe::resolve_self_exe();
    let mut cmd = std::process::Command::new(&exe);

    // `verbose` from server.yaml, so fleet deployments can set `verbose: 1`
    // for INFO-level logging without hardcoding it in the binary.
    for _ in 0..verbose {
        cmd.arg("-v");
    }

    cmd.args([
        "serve",
        "start",
        "--foreground",
        "--handshake-slot",
        profile.handshake_slot().as_arg(),
        "--port",
        // The REQUEST, which may be 0 for "let the OS choose". The child
        // publishes what it actually bound and we read that back below, so
        // nothing here has to be a prediction.
        &port.bind_value().to_string(),
        "--host",
        "127.0.0.1",
        "--python",
        &python,
    ]);

    let config_path = layout.config_path();
    if config_path.exists() {
        cmd.arg("--config").arg(config_path);
    }
    if flags.cli_force_cpu {
        cmd.arg("--force-cpu");
    }
    if flags.cli_allow_mps {
        cmd.arg("--allow-mps");
    }
    if let Some(n) = workers {
        cmd.args(["--workers", &n.to_string()]);
    }
    if let Some(t) = timeout {
        cmd.args(["--timeout", &t.to_string()]);
    }

    let log_path = profile.log_file(dir);
    // Append mode: preserve previous daemon logs across restarts.
    // Previous behavior (File::create) truncated logs on every restart,
    // destroying crash diagnostics from the previous session.
    let log_file = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(&log_path)?;
    cmd.stdout(std::process::Stdio::null());
    cmd.stderr(log_file);

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

    // On Windows, CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS ensures the
    // daemon survives after the spawning CLI process exits, analogous to
    // Unix setsid(). The daemon gets its own console group and is not
    // attached to the parent's console.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NEW_PROCESS_GROUP (0x200) | DETACHED_PROCESS (0x08)
        cmd.creation_flags(0x00000200 | 0x00000008);
    }

    let proc = match cmd.spawn() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("warning: failed to spawn daemon process: {e}");
            debug!(profile = profile.label(), error = %e, "Failed to spawn daemon");
            eprintln!(
                "warning: could not start local daemon. Check {}\n\
                 hint: run `batchalign3 serve start --foreground` to see startup errors.",
                log_path.display()
            );
            return Ok(None);
        }
    };

    let pid = proc.id();

    // Wait for the child to publish the port it BOUND before recording
    // anything. `daemon.json` used to be written here with the port we asked
    // for, before the child had bound, which made it a prediction: with an
    // ephemeral request there was no number to write at all, and with a fixed
    // one it could disagree with reality if the bind failed.
    let deadline = std::time::Instant::now() + startup_budget();
    let bound = match ServerHandshake::await_published(dir, profile.handshake_slot(), pid, deadline)
        .await
    {
        Ok(port) => port,
        Err(failure) => {
            // Kill it. A child that has not reported a port is still booting,
            // and leaving it running orphans a daemon that will bind moments
            // later: the next invocation adopts it while this one reports no
            // server available, having thrown away the whole boot.
            let reason = failure.reason();
            eprintln!(
                "warning: {} daemon (PID {pid}) {reason}. Check {}",
                profile.label(),
                log_path.display()
            );
            stop_server_process(pid);
            lock.remove_state()?;
            return Ok(None);
        }
    };
    let port = bound.get();
    // Persist the resolved device values, not the CLI raw bools, so future
    // restart decisions compare against the same merged result the daemon
    // process is actually running with. workers/timeout are recorded as
    // passed so the warm-reuse warning can name what the daemon was started
    // with.
    lock.write_info(&daemon_info(pid, flags, workers, timeout))?;

    if wait_for_health_until(pid, port, deadline).await {
        eprintln!(
            "{} daemon ready on port {} (PID {})",
            profile.label(),
            port,
            pid
        );
        return Ok(Some(format!("http://127.0.0.1:{port}")));
    }

    debug!(
        profile = profile.label(),
        port, "Daemon failed to become healthy"
    );
    stop_server_process(pid);
    lock.remove_state()?;

    eprintln!(
        "warning: could not start local daemon. Check {}\n\
         hint: run `batchalign3 serve start --foreground` to see startup errors.",
        log_path.display()
    );
    Ok(None)
}

async fn wait_for_health_until(pid: u32, port: u16, deadline: std::time::Instant) -> bool {
    while std::time::Instant::now() < deadline {
        if !is_process_alive(pid) {
            return false;
        }

        if startup_health_check(port).await {
            return true;
        }

        tokio::time::sleep(Duration::from_secs_f64(HEALTH_POLL)).await;
    }

    false
}

/// Quick health probe for localhost daemon startup.
///
/// Uses a short timeout (3s request, 1s connect) instead of the full
/// `BatchalignClient` timeout (120s). A connection to 127.0.0.1 should
/// succeed in well under 1 second, so this gives fast failure detection
/// while still tolerating brief startup latency.
async fn startup_health_check(port: u16) -> bool {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .connect_timeout(Duration::from_secs(1))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            eprintln!("warning: failed to build localhost health-check client: {error}");
            return false;
        }
    };
    client
        .get(format!("http://127.0.0.1:{port}/health"))
        .send()
        .await
        .is_ok_and(|r| r.status().is_success())
}

/// Full health check via `BatchalignClient` (120s timeout).
/// Used for the reuse path when daemon is already running and we want
/// the richer `HealthResponse`.
async fn health_check(port: u16) -> bool {
    let Ok(client) = BatchalignClient::new() else {
        return false;
    };
    client
        .health_check(&format!("http://127.0.0.1:{port}"))
        .await
        .is_ok()
}

/// What a profile's state file says.
#[derive(Debug)]
enum StateFile {
    /// No state file: no daemon of this profile is recorded.
    Absent,
    /// A daemon is recorded.
    Recorded(DaemonInfo),
    /// A state file that cannot be read. It may still name a live daemon,
    /// so it is reported and left in place, never deleted: the same policy
    /// as an unreadable server handshake.
    Unreadable {
        /// The file.
        path: PathBuf,
        /// Why it could not be read.
        reason: String,
    },
}

impl StateFile {
    /// Read `profile`'s state file in `dir`. Takes no lock and changes
    /// nothing, so status checks may call it freely.
    fn read(profile: DaemonProfile, dir: &Path) -> Self {
        let path = profile.state_file(dir);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Self::Absent,
            Err(error) => {
                return Self::Unreadable {
                    path,
                    reason: error.to_string(),
                };
            }
        };
        match serde_json::from_str(&text) {
            Ok(info) => Self::Recorded(info),
            Err(error) => Self::Unreadable {
                path,
                reason: error.to_string(),
            },
        }
    }

    /// The recorded daemon, or why the record cannot be read. An unreadable
    /// record is an error for the caller to act on, never "no daemon": it
    /// may name one that is running.
    fn recorded(self) -> Result<Option<DaemonInfo>, UnreadableRecord> {
        match self {
            Self::Absent => Ok(None),
            Self::Recorded(info) => Ok(Some(info)),
            Self::Unreadable { path, reason } => Err(UnreadableRecord { path, reason }),
        }
    }
}

/// A daemon state file or server handshake that exists but cannot be read.
///
/// It may still name a running process, so it is never treated as "nothing
/// running": a start refuses rather than write a new record over it, and a
/// stop leaves it in place.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "cannot read {} ({reason}). It may name a running batchalign3 server, so nothing \
     was started or stopped over it. Check whether that process is still running \
     (stop it if so), then remove the file and retry",
    path.display()
)]
pub struct UnreadableRecord {
    /// The file.
    pub path: PathBuf,
    /// Why it could not be read.
    pub reason: String,
}

/// The record of the daemon this binary just started.
fn daemon_info(
    pid: u32,
    flags: DaemonDeviceFlags,
    workers: Option<crate::api::NumWorkers>,
    audio_task_timeout_s: Option<crate::api::PositiveSeconds>,
) -> DaemonInfo {
    DaemonInfo {
        pid,
        build_hash: crate::build_hash().to_string(),
        force_cpu: flags.resolved_force_cpu,
        allow_mps: flags.resolved_allow_mps,
        workers,
        audio_task_timeout_s,
    }
}

/// Whether a process exists, via `kill(pid, 0)`.
pub(crate) fn is_process_alive(pid: u32) -> bool {
    #[cfg(unix)]
    unsafe {
        libc::kill(pid as i32, 0) == 0
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        false
    }
}

/// How long a stop waits for a server's own graceful shutdown before SIGKILL.
///
/// Long enough for the server to record its running jobs as interrupted
/// (runners get fifteen seconds) and retire its workers (up to seven seconds
/// each for a shared one). The old three-second wait always ended in SIGKILL
/// when a job was running, skipping that teardown.
pub(crate) const GRACEFUL_STOP_DEADLINE: std::time::Duration = std::time::Duration::from_secs(60);

/// How long a stop waits silently before saying it is still waiting.
const QUIET_STOP_WAIT: std::time::Duration = std::time::Duration::from_secs(3);

/// Stop a server process: SIGTERM its process group and the process, wait up
/// to [`GRACEFUL_STOP_DEADLINE`] for it to finish shutting down, then SIGKILL.
/// Returns `true` if the process was signalled at all.
///
/// The one stop routine for every server this CLI stops, the main and sidecar
/// daemons and `serve stop` alike. Its workers exit with it however it ends
/// (they watch `--supervisor-pid`); the wait is for an orderly teardown.
pub(crate) fn stop_server_process(pid: u32) -> bool {
    #[cfg(unix)]
    {
        let pgid_ok = unsafe { libc::killpg(pid as libc::pid_t, libc::SIGTERM) == 0 };
        let pid_ok = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) == 0 };

        if !pgid_ok && !pid_ok {
            return false;
        }

        if wait_for_exit(pid, QUIET_STOP_WAIT) {
            return true;
        }
        eprintln!(
            "Waiting for the server (PID {pid}) to finish shutting down (up to {}s)...",
            GRACEFUL_STOP_DEADLINE.as_secs()
        );
        if wait_for_exit(pid, GRACEFUL_STOP_DEADLINE - QUIET_STOP_WAIT) {
            return true;
        }

        eprintln!(
            "warning: server (PID {pid}) did not finish shutting down in {}s; killing it",
            GRACEFUL_STOP_DEADLINE.as_secs()
        );
        unsafe {
            libc::killpg(pid as libc::pid_t, libc::SIGKILL);
            libc::kill(pid as libc::pid_t, libc::SIGKILL);
        }
        // Brief wait for SIGKILL to take effect.
        std::thread::sleep(std::time::Duration::from_millis(200));
        true
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        false
    }
}

/// Whether `pid` exits within `limit`. The server is not our child, so there
/// is no wait handle to block on; its liveness is checked every half second,
/// first immediately.
#[cfg(unix)]
fn wait_for_exit(pid: u32, limit: std::time::Duration) -> bool {
    const POLL: std::time::Duration = std::time::Duration::from_millis(500);
    let started = std::time::Instant::now();
    loop {
        if !is_process_alive(pid) {
            return true;
        }
        if started.elapsed() >= limit {
            return false;
        }
        std::thread::sleep(POLL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All-off device flags for tests that do not exercise them.
    const NO_DEVICE_FLAGS: DaemonDeviceFlags = DaemonDeviceFlags {
        cli_force_cpu: false,
        cli_allow_mps: false,
        resolved_force_cpu: false,
        resolved_allow_mps: false,
    };

    /// Resolved-only flags helper for mismatch tests.
    const fn resolved_flags(force_cpu: bool, allow_mps: bool) -> DaemonDeviceFlags {
        DaemonDeviceFlags {
            cli_force_cpu: false,
            cli_allow_mps: false,
            resolved_force_cpu: force_cpu,
            resolved_allow_mps: allow_mps,
        }
    }

    /// Build a `DaemonInfo` with all the unrelated fields filled in,
    /// so each test can name only what it actually exercises. Keeps the
    /// individual tests focused on the field they pin.
    fn info_with(
        build_hash: String,
        force_cpu: bool,
        workers: Option<crate::api::NumWorkers>,
        audio_task_timeout_s: Option<crate::api::PositiveSeconds>,
    ) -> DaemonInfo {
        DaemonInfo {
            pid: 1,
            build_hash,
            force_cpu,
            allow_mps: false,
            workers,
            audio_task_timeout_s,
        }
    }

    #[test]
    fn daemon_info_roundtrip() {
        let info = info_with(
            "1.0.0-abc1234-1700000000".to_string(),
            true,
            Some(crate::api::NumWorkers(4)),
            Some(crate::api::PositiveSeconds::literal::<3600>()),
        );
        let json = serde_json::to_string(&info).unwrap();
        let back: DaemonInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.pid, 1);
        assert_eq!(back.build_hash, "1.0.0-abc1234-1700000000");
        assert_eq!(back.workers, Some(crate::api::NumWorkers(4)));
        assert_eq!(
            back.audio_task_timeout_s
                .map(crate::api::PositiveSeconds::get),
            Some(3600)
        );
    }

    /// A state file without the build that wrote it cannot say whether its
    /// daemon is stale, so it is unreadable: reported and left in place,
    /// never compared by version number.
    #[test]
    fn a_state_file_without_a_build_hash_is_unreadable() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("daemon.json"),
            r#"{"pid": 999, "version": "1.0.0", "force_cpu": false, "allow_mps": false}"#,
        )
        .unwrap();
        assert!(matches!(
            StateFile::read(DaemonProfile::Main, dir.path()),
            StateFile::Unreadable { .. }
        ));
        assert!(dir.path().join("daemon.json").exists(), "left in place");
    }

    #[test]
    fn is_stale_detects_build_hash_mismatch() {
        let info = info_with("old-build-hash".to_string(), false, None, None);
        // Our build hash is different from "old-build-hash"
        assert!(is_stale(&info));
    }

    #[test]
    fn is_stale_same_build_hash() {
        let info = info_with(crate::build_hash().to_string(), false, None, None);
        assert!(!is_stale(&info));
    }

    #[test]
    fn an_absent_state_file_reads_as_absent() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            StateFile::read(DaemonProfile::Main, dir.path()),
            StateFile::Absent
        ));
    }

    #[test]
    fn malformed_state_reads_as_unreadable() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("daemon.json"), "not json at all").unwrap();
        assert!(matches!(
            StateFile::read(DaemonProfile::Main, dir.path()),
            StateFile::Unreadable { .. }
        ));
    }

    /// Reading an unreadable state file under the lock (what `stop_profile`
    /// and the ensure path do) is an error naming the file, never "no
    /// daemon", and the file is left in place: it may still name a live
    /// daemon, and deleting or overwriting it would strand that process.
    #[test]
    fn an_unreadable_state_file_is_left_in_place_when_read_under_the_lock() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("daemon.json"), "not json at all").unwrap();
        let lock = DaemonStartLock::acquire(DaemonProfile::Main, dir.path()).unwrap();
        let unreadable = lock.read_state().recorded().unwrap_err();
        assert_eq!(unreadable.path, dir.path().join("daemon.json"));
        assert!(dir.path().join("daemon.json").exists());
    }

    /// The ensure path refuses on a state file it cannot read, naming the
    /// file, instead of starting a daemon and writing a record over one that
    /// may name a live daemon.
    #[tokio::test]
    async fn ensure_refuses_over_an_unreadable_state_file() {
        let dir = tempfile::tempdir().unwrap();
        let layout = RuntimeLayout::from_state_dir(dir.path().to_path_buf());
        std::fs::write(dir.path().join("daemon.json"), "not json at all").unwrap();
        let lock = DaemonStartLock::acquire(DaemonProfile::Main, dir.path()).unwrap();
        let request = DaemonRequest {
            force_cpu: false,
            allow_mps: false,
            workers: None,
            timeout: None,
        };
        let refused = ensure_daemon_locked(&lock, &layout, &ServerConfig::default(), request)
            .await
            .unwrap_err();
        assert!(
            matches!(&refused, CliError::UnreadableRecord(record)
                if record.path == dir.path().join("daemon.json")),
            "{refused}"
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("daemon.json")).unwrap(),
            "not json at all",
            "the record is left exactly as it was"
        );
    }

    #[test]
    fn write_read_daemon_info_roundtrip() {
        for profile in [DaemonProfile::Main, DaemonProfile::Sidecar] {
            let dir = tempfile::tempdir().unwrap();
            let lock = DaemonStartLock::acquire(profile, dir.path()).unwrap();
            let flags = DaemonDeviceFlags {
                cli_force_cpu: true,
                cli_allow_mps: false,
                resolved_force_cpu: true,
                resolved_allow_mps: false,
            };
            lock.write_info(&daemon_info(
                42,
                flags,
                Some(crate::api::NumWorkers(6)),
                Some(crate::api::PositiveSeconds::literal::<1800>()),
            ))
            .unwrap();
            let StateFile::Recorded(info) = lock.read_state() else {
                panic!("the written record reads back");
            };
            assert_eq!(info.pid, 42);
            assert_eq!(info.build_hash, crate::build_hash());
            assert!(info.force_cpu);
            assert_eq!(info.workers, Some(crate::api::NumWorkers(6)));
            assert_eq!(
                info.audio_task_timeout_s
                    .map(crate::api::PositiveSeconds::get),
                Some(1800)
            );
        }
    }

    #[test]
    fn write_read_daemon_info_roundtrip_no_workers_no_timeout() {
        // Daemon started without --workers or --timeout (the common
        // server-mode case where host facts pick the parallelism).
        let dir = tempfile::tempdir().unwrap();
        let lock = DaemonStartLock::acquire(DaemonProfile::Main, dir.path()).unwrap();
        lock.write_info(&daemon_info(7, NO_DEVICE_FLAGS, None, None))
            .unwrap();
        let StateFile::Recorded(info) = lock.read_state() else {
            panic!("the written record reads back");
        };
        assert_eq!(info.workers, None);
        assert_eq!(info.audio_task_timeout_s, None);
    }

    #[test]
    fn remove_state_removes_the_record() {
        let dir = tempfile::tempdir().unwrap();
        let lock = DaemonStartLock::acquire(DaemonProfile::Main, dir.path()).unwrap();
        lock.write_info(&daemon_info(1, NO_DEVICE_FLAGS, None, None))
            .unwrap();
        assert!(matches!(lock.read_state(), StateFile::Recorded(_)));
        lock.remove_state().unwrap();
        assert!(matches!(lock.read_state(), StateFile::Absent));
    }

    #[test]
    fn runtime_mismatch_detects_force_cpu_changes() {
        let info = info_with(crate::build_hash().to_string(), false, None, None);
        assert!(runtime_mismatch(&info, resolved_flags(true, false)));
        assert!(!runtime_mismatch(&info, resolved_flags(false, false)));
    }

    #[test]
    fn flag_shadows_daemon_silent_when_user_did_not_pass_flag() {
        // User accepted the daemon's default, never warn, regardless
        // of what the daemon was started with.
        assert!(!flag_shadows_daemon::<u32>(None, None));
        assert!(!flag_shadows_daemon(None, Some(4)));
    }

    #[test]
    fn flag_shadows_daemon_silent_when_values_match() {
        // The whole point of this helper: re-passing a value that
        // already matches the running daemon must NOT warn.
        assert!(!flag_shadows_daemon(Some(4), Some(4)));
        assert!(!flag_shadows_daemon(Some(1800u64), Some(1800u64)));
    }

    #[test]
    fn flag_shadows_daemon_warns_on_value_mismatch() {
        assert!(flag_shadows_daemon(Some(4), Some(1)));
        assert!(flag_shadows_daemon(Some(3600u64), Some(1800u64)));
    }

    #[test]
    fn flag_shadows_daemon_warns_when_persisted_is_unknown() {
        // Pre-upgrade daemon.json files lack the field → persisted is
        // None. The user passed a value, so we cannot prove it matches.
        // Warn (the pre-persisted-value behavior, preserved on first
        // contact post-upgrade).
        assert!(flag_shadows_daemon(Some(4), None::<u32>));
        assert!(flag_shadows_daemon(Some(1800u64), None));
    }

    /// `--force-cpu` always wins over the host-facts recommendation,
    /// regardless of which host the test runs on. The resolved value
    /// must equal `true` whenever the CLI flag is `true`. This pins
    /// the operator-override-wins contract from `EffectiveConfig`.
    #[test]
    fn resolve_force_cpu_for_daemon_cli_override_always_resolves_true() {
        // The default config has force_cpu = None. The CLI flag = true should
        // still resolve to true on every host.
        let resolved = resolve_device_flags_for_daemon(&ServerConfig::default(), true, false);
        assert!(
            resolved.resolved_force_cpu,
            "CLI --force-cpu must resolve to true regardless of host facts"
        );
        assert!(
            !resolved.resolved_allow_mps,
            "allow_mps stays false unless the operator opts in"
        );
    }

    #[test]
    fn removing_an_absent_record_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        // Removing a state file that is already gone succeeds.
        let lock = DaemonStartLock::acquire(DaemonProfile::Main, dir.path()).unwrap();
        lock.remove_state().unwrap();
        lock.remove_state().unwrap();
    }

    #[test]
    fn is_process_alive_current_pid() {
        let pid = std::process::id();
        assert!(is_process_alive(pid));
    }

    #[test]
    fn the_default_port_request_is_a_fixed_port() {
        let request = ServerConfig::default().port;
        assert!(
            request.fixed().is_some(),
            "the default must be a fixed port, so the auto-daemon path works \
             with no server.yaml: got {}",
            request.describe()
        );
    }

    /// The adversarial case this whole change exists for: a listener is
    /// already bound on the chosen port (nothing answering `/health`, just
    /// a raw socket, the same shape a stray or unrelated process would
    /// present), so a bind attempt is guaranteed to fail. The probe must
    /// classify this as `Occupied`, never as `Free` (which would let the
    /// caller spawn straight into the guaranteed-failing bind) and never
    /// as `ExistingDaemon` (which would wrongly adopt a process that never
    /// answered a batchalign3 health check, the exact 2026-08-27
    /// misidentification `HealthResponse`'s strict parsing was built to
    /// prevent).
    #[tokio::test]
    async fn probe_fixed_port_reports_occupied_for_a_bound_non_daemon_listener() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind a free port");
        let port = NonZeroU16::new(listener.local_addr().expect("local addr").port())
            .expect("OS-assigned port is never 0");

        let outcome = probe_fixed_port(port).await;

        assert_eq!(
            outcome,
            PortOccupant::Occupied,
            "a port held by a listener that never answers /health must be Occupied"
        );
        // Keep the listener alive for the whole probe so the test actually
        // exercises the "bind fails" branch rather than a race where the
        // listener happened to be dropped first.
        drop(listener);
    }

    /// The ordinary case: nothing is listening, so the probe finds the port
    /// free and releases it immediately (never holding it open itself,
    /// which would turn the probe into the very collision it exists to
    /// avoid).
    #[tokio::test]
    async fn probe_fixed_port_reports_free_and_releases_the_port() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind a free port");
        let port = NonZeroU16::new(listener.local_addr().expect("local addr").port())
            .expect("OS-assigned port is never 0");
        drop(listener);

        let outcome = probe_fixed_port(port).await;
        assert_eq!(outcome, PortOccupant::Free);

        // The probe must have released the port: binding it again here
        // must succeed. Proves `Free` did not leak the listener it used
        // to test the port.
        let rebound = std::net::TcpListener::bind(("127.0.0.1", port.get()));
        assert!(
            rebound.is_ok(),
            "probe_fixed_port must release the port it found free, not hold it open"
        );
    }

    /// Spin up a minimal `/health` responder reporting the given build
    /// hash, bound to an OS-assigned port. Returns the port and a handle
    /// the caller must keep alive for the probe's duration (dropping it
    /// stops the server).
    async fn spawn_fake_health_server(
        build_hash: &'static str,
    ) -> (NonZeroU16, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind fake health server");
        let port = NonZeroU16::new(listener.local_addr().expect("local addr").port())
            .expect("OS-assigned port is never 0");
        let router = axum::Router::new().route(
            "/health",
            axum::routing::get(move || async move {
                axum::Json(serde_json::json!({
                    "status": "ok",
                    "version": "test",
                    "build_hash": build_hash,
                }))
            }),
        );
        let handle = tokio::spawn(async move {
            axum::serve(listener, router.into_make_service())
                .await
                .expect("serve fake health server");
        });
        (port, handle)
    }

    /// The adversarial case this rule exists for: a batchalign3 daemon
    /// really is listening and really does answer `/health`, but on a
    /// DIFFERENT build than this CLI invocation. Adopting it silently
    /// (the pre-fix behavior: any answering daemon was `ExistingDaemon`)
    /// would mean every request this invocation submits is served by code
    /// this binary did not build and cannot vouch for. The probe must
    /// refuse to adopt and report `StaleDaemon`, not `ExistingDaemon`.
    #[tokio::test]
    async fn probe_fixed_port_refuses_to_adopt_a_daemon_on_a_different_build() {
        let their_build = "definitely-not-our-build-hash";
        assert_ne!(
            their_build,
            crate::build_hash(),
            "test fixture must actually differ from our real build hash"
        );
        let (port, _server) = spawn_fake_health_server(their_build).await;

        let outcome = probe_fixed_port(port).await;

        assert_eq!(
            outcome,
            PortOccupant::StaleDaemon {
                theirs: their_build.to_owned(),
                ours: crate::build_hash().to_owned(),
            },
            "a daemon on a different build must never be reported as ExistingDaemon"
        );
    }

    /// The companion positive case: a daemon on OUR OWN build is safe to
    /// adopt, and is still reported as `ExistingDaemon`.
    #[tokio::test]
    async fn probe_fixed_port_adopts_a_daemon_on_our_own_build() {
        let (port, _server) = spawn_fake_health_server(crate::build_hash()).await;

        let outcome = probe_fixed_port(port).await;

        assert_eq!(
            outcome,
            PortOccupant::ExistingDaemon {
                build_hash: crate::build_hash().to_owned(),
            }
        );
    }

    /// A second acquirer of a profile's start lock waits until the first
    /// releases it. std's `try_lock` reports contention as `WouldBlock`, which
    /// `acquire` turns into a blocking wait and never into an error; the fs2
    /// code before it treated every failed attempt as contention.
    #[test]
    fn a_second_acquirer_waits_for_the_first_to_release() {
        let dir = tempfile::tempdir().unwrap();
        let first = DaemonStartLock::acquire(DaemonProfile::Main, dir.path()).unwrap();
        let (sender, receiver) = std::sync::mpsc::channel();
        let state_dir = dir.path().to_path_buf();
        let waiter = std::thread::spawn(move || {
            let second = DaemonStartLock::acquire(DaemonProfile::Main, &state_dir);
            sender.send(second.map(|lock| lock.profile)).unwrap();
        });
        assert!(
            receiver.recv_timeout(Duration::from_millis(300)).is_err(),
            "a held lock was acquired a second time"
        );
        drop(first);
        let second = receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("the released lock was never acquired");
        assert_eq!(second.unwrap(), DaemonProfile::Main);
        waiter.join().unwrap();
    }

    /// Each profile has its own lock file, so holding one never blocks the other.
    #[test]
    fn profiles_lock_independently() {
        let dir = tempfile::tempdir().unwrap();
        let main = DaemonStartLock::acquire(DaemonProfile::Main, dir.path()).unwrap();
        let sidecar = DaemonStartLock::acquire(DaemonProfile::Sidecar, dir.path()).unwrap();
        assert_eq!(
            (main.profile, sidecar.profile),
            (DaemonProfile::Main, DaemonProfile::Sidecar)
        );
    }
}
