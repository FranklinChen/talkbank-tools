//! Worker registry: discover pre-started TCP workers from `workers.json`.
//!
//! The registry file is the bridge between independently started worker daemons
//! and the Rust server. Python workers write their entries on startup (via
//! `_registry.py`); the server reads and health-checks them on startup and
//! periodically.
//!
//! Registry path: `~/.batchalign3/workers.json` (configurable via
//! [`ServerConfig::worker_registry_path`] or `BATCHALIGN_STATE_DIR`).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::api::WorkerLanguage;
use crate::worker::WorkerProfile;

// ---------------------------------------------------------------------------
// Registry entry (JSON schema matches Python `WorkerRegistryEntry`)
// ---------------------------------------------------------------------------

/// How a registry worker relates to the current Rust server lifecycle.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RegistryOwnership {
    /// Independently started worker daemon that may outlive any one server.
    #[default]
    External,
    /// TCP daemon spawned and owned by one Rust server instance.
    ServerOwned,
}

mod admission;
mod adoption;
use admission::{DiscoveryDisposition, discovery_disposition};
pub(super) use adoption::{AdmittedRegistryWorker, ConnectionOwnership, RegistryAdmissions};

/// One worker's entry in the `workers.json` registry file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RegistryEntry {
    /// Worker process ID.
    pub pid: u32,
    /// Bind address (usually `"127.0.0.1"`).
    pub host: String,
    /// TCP port.
    pub port: u16,
    /// Worker profile name (`"gpu"`, `"stanza"`, `"io"`).
    pub profile: String,
    /// 3-letter language code.
    pub lang: String,
    /// Engine overrides JSON string (empty = none).
    #[serde(default)]
    pub engine_overrides: String,
    /// Whether the worker is external/persistent or owned by one server instance.
    #[serde(default)]
    pub ownership: RegistryOwnership,
    /// Owning Rust server instance id for server-owned daemons.
    #[serde(default)]
    pub owner_server_instance_id: Option<String>,
    /// Owning Rust server PID for server-owned daemons.
    #[serde(default)]
    pub owner_server_pid: Option<u32>,
    /// ISO 8601 timestamp when the worker started.
    #[serde(default)]
    pub started_at: String,
    /// Build identity of the Rust binary that started the daemon, read by the
    /// daemon from [`BUILD_IDENTITY_ENV`]. Absent in entries written before
    /// the field existed; such an entry is refused like any other build.
    #[serde(default)]
    pub build_identity: Option<String>,
}

impl RegistryEntry {
    /// Whether two entries describe the same daemon: the same process at the
    /// same address.
    fn is_same_daemon(&self, other: &Self) -> bool {
        self.pid == other.pid && self.host == other.host && self.port == other.port
    }

    /// Parse the profile string into a [`WorkerProfile`].
    #[must_use]
    pub fn worker_profile(&self) -> Option<WorkerProfile> {
        WorkerProfile::try_from_name(&self.profile)
    }

    fn owner_server_instance_id(&self) -> Option<&str> {
        self.owner_server_instance_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }

    fn is_server_owned(&self) -> bool {
        self.ownership == RegistryOwnership::ServerOwned
    }

    fn is_owned_by_server_instance(&self, server_instance_id: &str) -> bool {
        self.is_server_owned() && self.owner_server_instance_id() == Some(server_instance_id)
    }
}

/// Result of one registry scan.
#[derive(Default)]
pub(super) struct RegistryDiscovery {
    /// Healthy registry workers that can be integrated into the pool.
    pub workers: Vec<adoption::ReportReadyRegistryWorker>,
    /// Daemons the scan refused to adopt because their entry names another
    /// build, or none. Not reaped: they may belong to a server of their own
    /// build. The pool reports them in `/health`.
    pub refused: Vec<ForeignBuildDaemon>,
}

/// The environment variable a daemon reads its build identity from, set by
/// every Rust route that starts one (the server's daemon spawner and
/// `batchalign3 worker start`). The daemon writes it into its registry entry.
pub const BUILD_IDENTITY_ENV: &str = "BATCHALIGN_BUILD_IDENTITY";

/// A registry daemon whose build identity is not this server's.
///
/// Refused rather than adopted: its engine identities, cache namespaces and
/// wire contract belong to another build, and staleness is judged by build
/// identity, never by semver. The remedy is to restart the daemon with this
/// build.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "registry worker pid {pid} (profile {profile}, lang {lang}) runs {reported}, but this \
     server is build {ours}; restart it with this build: `batchalign3 worker stop`, then \
     `batchalign3 worker start --profile {profile} --lang {lang}`"
)]
pub struct ForeignBuildDaemon {
    pid: u32,
    profile: String,
    lang: String,
    reported: ReportedBuild,
    ours: String,
}

impl ForeignBuildDaemon {
    /// The refusal as `/health` reports it: the daemon's worker key as its
    /// registry entry names it, its pid, and a typed reason.
    #[must_use]
    pub fn to_refused_registry_worker(&self) -> crate::api::RefusedRegistryWorker {
        crate::api::RefusedRegistryWorker {
            worker_key: format!("profile:{}:{}", self.profile, self.lang),
            pid: self.pid,
            reason: match &self.reported {
                ReportedBuild::Unreported => crate::api::RegistryWorkerRefusal::UnreportedBuild {
                    server_build: self.ours.clone(),
                },
                ReportedBuild::Build(build) => crate::api::RegistryWorkerRefusal::ForeignBuild {
                    reported_build: build.clone(),
                    server_build: self.ours.clone(),
                },
            },
        }
    }
}

/// What a registry entry says about the build that wrote it.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ReportedBuild {
    /// The entry names no build: written by a daemon that predates build
    /// identity in the registry, or started without the environment variable.
    Unreported,
    /// The entry names this build identity.
    Build(String),
}

impl std::fmt::Display for ReportedBuild {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unreported => f.write_str("an unreported build"),
            Self::Build(build) => write!(f, "build {build}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Registry file I/O
// ---------------------------------------------------------------------------

/// Default registry file path: `workers.json` in the runtime state
/// directory (`~/.batchalign3` unless `BATCHALIGN_STATE_DIR` or
/// `BATCHALIGN_HOME` says otherwise), resolved by the one layout resolver.
#[must_use]
pub fn default_registry_path() -> PathBuf {
    crate::config::RuntimeLayout::from_env()
        .state_dir()
        .join("workers.json")
}

/// Read all entries from the registry file.
pub fn read_registry(path: &Path) -> Vec<RegistryEntry> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            if e.kind() != std::io::ErrorKind::NotFound {
                warn!(path = %path.display(), error = %e, "Failed to read worker registry");
            }
            return Vec::new();
        }
    };

    match serde_json::from_str::<Vec<RegistryEntry>>(&content) {
        Ok(entries) => entries,
        Err(e) => {
            warn!(path = %path.display(), error = %e, "Failed to parse worker registry");
            Vec::new()
        }
    }
}

/// Remove every entry `stale` matches, as one read-modify-write under the
/// registry's lock (`workers.json.lock`, which the Python registry writers
/// take too), so an entry a daemon registers meanwhile is never overwritten
/// by a stale snapshot. Returns how many entries were removed.
fn remove_entries(
    path: &Path,
    stale: impl Fn(&RegistryEntry) -> bool,
) -> Result<usize, std::io::Error> {
    let _lock = crate::file_lock::HeldFileLock::acquire(crate::file_lock::lock_file_for(path))?;
    let entries = read_registry(path);
    let before = entries.len();
    let remaining: Vec<RegistryEntry> = entries.into_iter().filter(|e| !stale(e)).collect();
    let removed = before - remaining.len();
    if removed > 0 {
        write_registry(path, &remaining)?;
    }
    Ok(removed)
}

/// Write entries back to the registry file; only [`remove_entries`] calls
/// this, holding the registry's lock.
fn write_registry(path: &Path, entries: &[RegistryEntry]) -> Result<(), std::io::Error> {
    let data = serde_json::to_string_pretty(entries)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    crate::atomic_file::write_atomically(
        path,
        data.as_bytes(),
        crate::atomic_file::Existing::Replace,
        crate::atomic_file::Audience::Owner,
    )
}

#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    crate::worker::pool::reaper::process_alive(pid)
}

#[cfg(not(unix))]
fn process_alive(_pid: u32) -> bool {
    false
}

fn should_shutdown_entry(entry: &RegistryEntry, current_server_instance_id: &str) -> bool {
    entry.is_owned_by_server_instance(current_server_instance_id)
}

fn terminate_registered_daemon(pid: u32, profile: &str) {
    #[cfg(unix)]
    {
        let Some(target) = crate::worker::process_id::NativePid::admit(pid) else {
            warn!(pid, profile, "Refusing invalid registry process ID");
            return;
        };
        match target.terminate_process() {
            Ok(()) => info!(pid, profile, "Sent SIGTERM to TCP daemon worker"),
            Err(error) if error.raw_os_error() == Some(libc::ESRCH) => {}
            Err(error) => warn!(pid,profile,%error,"Failed to SIGTERM TCP daemon worker"),
        }
    }

    #[cfg(not(unix))]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F"])
            .output();
        info!(pid, profile, "Sent taskkill to TCP daemon worker");
    }
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Remove a stale entry by PID (for crash cleanup). Takes the registry's
/// lock, so callers on the async runtime run it with `spawn_blocking`.
pub fn remove_stale_entry(registry_path: &Path, pid: u32) -> bool {
    match remove_entries(registry_path, |entry| entry.pid == pid) {
        Ok(removed) => removed > 0,
        Err(e) => {
            warn!(error = %e, "Failed to write registry after stale removal");
            false
        }
    }
}

/// Kill all TCP daemon workers owned by the current server instance and remove
/// their registry entries. External daemons are preserved.
///
/// Kills and removes in one pass under the registry's lock, so a daemon that
/// registers meanwhile is never dropped from the registry without being
/// killed. Takes the lock, so callers on the async runtime run it with
/// `spawn_blocking`.
pub fn kill_owned_daemons(registry_path: &Path, current_server_instance_id: &str) {
    match remove_entries(registry_path, |entry| {
        let owned = should_shutdown_entry(entry, current_server_instance_id);
        if owned {
            terminate_registered_daemon(entry.pid, &entry.profile);
        }
        owned
    }) {
        Ok(0) => {}
        Ok(killed) => info!(killed, "Retired owned TCP daemon workers"),
        Err(e) => warn!(error = %e, "Failed to retire owned TCP daemon workers"),
    }
}

/// Discover pre-started workers from the registry file.
///
/// Reads `workers.json`, connects to each entry, runs a health check, and
/// removes stale entries (workers that crashed without cleanup). Returns
/// only healthy, connectable workers plus one capability snapshot probed on the
/// same TCP connection used during discovery.
pub(super) async fn discover_workers(
    registry_path: &Path,
    task_timeouts: crate::types::worker_v2::TaskTimeoutOverrides,
    current_server_instance_id: &str,
    admissions: &RegistryAdmissions,
) -> RegistryDiscovery {
    let entries = read_registry(registry_path);
    if entries.is_empty() {
        return RegistryDiscovery::default();
    }

    info!(
        count = entries.len(),
        path = %registry_path.display(),
        "Checking worker registry"
    );

    let mut discovered = Vec::new();
    let mut refused = Vec::new();
    enum CapabilityProbe {
        Needed,
        Observed,
    }
    let mut capability_probe = CapabilityProbe::Needed;
    let mut stale_indices = Vec::new();
    let mut rejected_workers = Vec::new();

    for (i, entry) in entries.iter().enumerate() {
        let compatible =
            match discovery_disposition(entry, current_server_instance_id, crate::build_hash()) {
                DiscoveryDisposition::Accept(compatible) => compatible,
                DiscoveryDisposition::RefuseForeignBuild(refusal) => {
                    // Not reaped and not removed from the registry: the daemon is
                    // alive and may belong to a server of its own build. It is
                    // simply never adopted by this one, and the refusal is
                    // returned for `/health`.
                    warn!(error = %refusal, "Refusing registry worker from another build");
                    refused.push(refusal);
                    continue;
                }
                DiscoveryDisposition::SkipForeignOwner => {
                    info!(
                        pid = entry.pid,
                        profile = %entry.profile,
                        owner_server_instance_id = ?entry.owner_server_instance_id(),
                        owner_server_pid = ?entry.owner_server_pid,
                        "Skipping registry worker owned by another live server"
                    );
                    continue;
                }
                DiscoveryDisposition::ReapStaleOwned => {
                    warn!(
                        pid = entry.pid,
                        profile = %entry.profile,
                        owner_server_instance_id = ?entry.owner_server_instance_id(),
                        owner_server_pid = ?entry.owner_server_pid,
                        "Reaping stale server-owned registry worker"
                    );
                    terminate_registered_daemon(entry.pid, &entry.profile);
                    stale_indices.push(i);
                    continue;
                }
            };

        let prepared = match compatible.prepare() {
            Ok(prepared) => prepared,
            Err(error) => {
                warn!(pid=entry.pid, %error, "Registry entry cannot be prepared");
                stale_indices.push(i);
                continue;
            }
        };
        let claimed = match admissions.claim(prepared) {
            Ok(adoption::ClaimDisposition::Claimed(claimed)) => claimed,
            Ok(adoption::ClaimDisposition::AlreadyClaimed) => continue,
            Err(error) => {
                warn!(pid=entry.pid,%error,"Registry daemon identity could not be admitted");
                continue;
            }
        };
        let connection = match claimed.prepare(task_timeouts) {
            Ok(connection) => connection,
            Err(error) => {
                warn!(pid=entry.pid,%error,"Registry key could not be admitted");
                continue;
            }
        };
        let probed = match connection.connect().await {
            Ok(connection) => connection.check().await,
            Err(rejected) => Err(rejected),
        };
        let worker = match probed {
            Ok(worker) => worker,
            Err(rejected) => {
                let source = rejected.source();
                warn!(
                    host = %source.host,
                    port = source.port,
                    pid = source.pid,
                    error = %rejected.error(),
                    "Registry worker probe failed, marking stale"
                );
                if source.is_server_owned() {
                    terminate_registered_daemon(source.pid, &source.profile);
                }
                rejected_workers.push(rejected);
                continue;
            }
        };

        info!(
            profile = %entry.profile,
            lang = %entry.lang,
            host = %entry.host,
            port = entry.port,
            pid = entry.pid,
            "Discovered healthy TCP worker"
        );
        let worker = match capability_probe {
            CapabilityProbe::Needed => match worker.observe_capabilities().await {
                Ok(worker) => {
                    if worker.observed_report() {
                        capability_probe = CapabilityProbe::Observed;
                    }
                    worker
                }
                Err(error) => {
                    warn!(pid=entry.pid,%error,"Registry capability exchange did not complete safely");
                    continue;
                }
            },
            CapabilityProbe::Observed => worker.defer_report(),
        };
        discovered.push(worker);
    }

    // Remove stale entries from the registry file, by identity under the
    // registry's lock, off the async runtime.
    if !stale_indices.is_empty() || !rejected_workers.is_empty() {
        let stale: Vec<RegistryEntry> = stale_indices.iter().map(|&i| entries[i].clone()).collect();
        let path = registry_path.to_path_buf();
        let removal = crate::blocking::spawn_in_span(move || {
            remove_entries(&path, |entry| {
                stale.iter().any(|stale| stale.is_same_daemon(entry))
                    || rejected_workers
                        .iter()
                        .any(|rejected| rejected.source() == entry)
            })
        })
        .await
        .map_err(std::io::Error::other)
        .and_then(|removed| removed);
        match removal {
            Ok(removed) => info!(removed, "Removed stale entries from worker registry"),
            Err(e) => warn!(error = %e, "Failed to update worker registry after stale removal"),
        }
    }

    RegistryDiscovery {
        workers: discovered,
        refused,
    }
}

#[cfg(test)]
mod tests;
