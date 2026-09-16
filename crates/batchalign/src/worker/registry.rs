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
use tracing::{debug, info, warn};

use crate::api::WorkerLanguage;
use crate::worker::tcp_handle::{TcpWorkerHandle, TcpWorkerInfo};
use crate::worker::{WorkerCapabilities, WorkerPid, WorkerProfile};

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

#[derive(Debug, Clone, PartialEq, Eq)]
enum DiscoveryDisposition {
    Accept,
    SkipForeignOwner,
    ReapStaleOwned,
    /// Owned by nobody else, or by this server, but written by another build.
    RefuseForeignBuild(ForeignBuildDaemon),
}

/// One worker's entry in the `workers.json` registry file.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Parse the profile string into a [`WorkerProfile`].
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

/// A discovered worker that has been health-checked and is ready for use.
#[derive(Debug, Clone)]
pub struct DiscoveredWorker {
    /// Registry entry data.
    pub entry: RegistryEntry,
    /// Parsed worker profile.
    pub profile: WorkerProfile,
    /// Parsed worker-runtime language string.
    pub lang: WorkerLanguage,
    /// The raw capability report probed on this worker's discovery
    /// connection, for the one worker a scan probes. It travels with the
    /// worker it came from, so the pool admits it under that worker's key
    /// rather than guessing which worker a detached report described.
    pub capabilities: Option<WorkerCapabilities>,
}

/// Result of one registry scan.
#[derive(Debug, Clone, Default)]
pub struct RegistryDiscovery {
    /// Healthy registry workers that can be integrated into the pool.
    pub workers: Vec<DiscoveredWorker>,
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

/// Default registry file path: `~/.batchalign3/workers.json`.
pub fn default_registry_path() -> PathBuf {
    if let Ok(state_dir) = std::env::var("BATCHALIGN_STATE_DIR") {
        let state_dir = state_dir.trim();
        if !state_dir.is_empty() {
            return PathBuf::from(state_dir).join("workers.json");
        }
    }
    dirs::home_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join(".batchalign3")
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

/// Write entries back to the registry file (for removing stale entries).
fn write_registry(path: &Path, entries: &[RegistryEntry]) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let data = serde_json::to_string_pretty(entries)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let tmp_path = path.with_extension("tmp");
    std::fs::write(&tmp_path, data)?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    // SAFETY: kill(pid, 0) only checks process existence/permission.
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

#[cfg(not(unix))]
fn process_alive(_pid: u32) -> bool {
    false
}

/// Decide what discovery does with one registry entry.
///
/// Ownership is settled first (a live foreign owner's daemon is skipped, a
/// dead owner's is reaped). An entry this server could adopt, external or its
/// own, is then adopted only if it was written by `our_build`.
fn discovery_disposition(
    entry: &RegistryEntry,
    current_server_instance_id: &str,
    our_build: &str,
) -> DiscoveryDisposition {
    if entry.is_server_owned() {
        let Some(owner_server_instance_id) = entry.owner_server_instance_id() else {
            return DiscoveryDisposition::ReapStaleOwned;
        };
        let Some(owner_server_pid) = entry.owner_server_pid else {
            return DiscoveryDisposition::ReapStaleOwned;
        };
        if owner_server_instance_id != current_server_instance_id {
            return if process_alive(owner_server_pid) {
                DiscoveryDisposition::SkipForeignOwner
            } else {
                DiscoveryDisposition::ReapStaleOwned
            };
        }
    }

    let reported = match entry.build_identity.as_deref() {
        Some(build) if build == our_build => return DiscoveryDisposition::Accept,
        Some(build) => ReportedBuild::Build(build.to_owned()),
        None => ReportedBuild::Unreported,
    };
    DiscoveryDisposition::RefuseForeignBuild(ForeignBuildDaemon {
        pid: entry.pid,
        profile: entry.profile.clone(),
        lang: entry.lang.clone(),
        reported,
        ours: our_build.to_owned(),
    })
}

fn should_shutdown_entry(entry: &RegistryEntry, current_server_instance_id: &str) -> bool {
    entry.is_owned_by_server_instance(current_server_instance_id)
}

fn terminate_registered_daemon(pid: u32, profile: &str) {
    #[cfg(unix)]
    {
        // SAFETY: sending SIGTERM to a known PID. If the process already
        // exited, `kill()` returns ESRCH which we ignore.
        let ret = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
        if ret == 0 {
            info!(pid, profile, "Sent SIGTERM to TCP daemon worker");
        } else {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::ESRCH) {
                warn!(pid, profile, error = %err, "Failed to SIGTERM TCP daemon worker");
            }
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

/// Remove a stale entry by PID (for crash cleanup).
pub fn remove_stale_entry(registry_path: &Path, pid: u32) -> bool {
    let entries = read_registry(registry_path);
    let before = entries.len();
    let remaining: Vec<RegistryEntry> = entries.into_iter().filter(|e| e.pid != pid).collect();
    if remaining.len() == before {
        return false;
    }
    if let Err(e) = write_registry(registry_path, &remaining) {
        warn!(error = %e, "Failed to write registry after stale removal");
    }
    true
}

/// Kill all TCP daemon workers owned by the current server instance and remove
/// their registry entries. External daemons are preserved.
pub fn kill_owned_daemons(registry_path: &Path, current_server_instance_id: &str) {
    let entries = read_registry(registry_path);
    if entries.is_empty() {
        return;
    }

    let mut killed = 0usize;
    let mut remaining = Vec::new();
    for entry in entries {
        if should_shutdown_entry(&entry, current_server_instance_id) {
            terminate_registered_daemon(entry.pid, &entry.profile);
            killed += 1;
        } else {
            remaining.push(entry);
        }
    }

    if let Err(e) = write_registry(registry_path, &remaining) {
        warn!(error = %e, "Failed to rewrite worker registry after shutdown");
    } else {
        info!(
            killed,
            remaining = remaining.len(),
            "Retired owned TCP daemon workers"
        );
    }
}

/// Discover pre-started workers from the registry file.
///
/// Reads `workers.json`, connects to each entry, runs a health check, and
/// removes stale entries (workers that crashed without cleanup). Returns
/// only healthy, connectable workers plus one capability snapshot probed on the
/// same TCP connection used during discovery.
pub async fn discover_workers(
    registry_path: &Path,
    audio_task_timeout_s: u64,
    analysis_task_timeout_s: u64,
    current_server_instance_id: &str,
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
    let mut probed_capabilities = false;
    let mut stale_indices = Vec::new();

    for (i, entry) in entries.iter().enumerate() {
        match discovery_disposition(entry, current_server_instance_id, crate::build_hash()) {
            DiscoveryDisposition::Accept => {}
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
        }

        let Some(profile) = entry.worker_profile() else {
            warn!(
                profile = %entry.profile,
                pid = entry.pid,
                "Unknown worker profile in registry, skipping"
            );
            stale_indices.push(i);
            continue;
        };

        let lang = match WorkerLanguage::parse_untrusted(&entry.lang) {
            Ok(code) => code,
            Err(e) => {
                warn!(
                    lang = %entry.lang,
                    pid = entry.pid,
                    error = %e,
                    "Registry entry has invalid worker language, skipping"
                );
                stale_indices.push(i);
                continue;
            }
        };
        // Health-check connect uses TcpWorkerHandle (one-request-at-a-time),
        // not SharedGpuTcpWorker. The dispatch semaphore is unused on this
        // path, so any non-zero default suffices; the pool will rebuild with
        // the correct value at integration time (pool/discovery.rs).
        let info = TcpWorkerInfo {
            host: entry.host.clone(),
            port: entry.port,
            profile,
            lang: lang.clone(),
            engine_overrides: entry.engine_overrides.clone(),
            pid: WorkerPid(entry.pid),
            audio_task_timeout_s,
            analysis_task_timeout_s,
            // Placeholder per the comment above, the registry walker
            // does not own the host-facts pipeline; the discovery /
            // pool integration step replaces this with the real
            // `EffectiveConfig.gpu_thread_pool_size`. The literal
            // matches the legacy static default so any code path
            // that *does* read it before integration sees the same
            // value as before the host-facts migration.
            gpu_thread_pool_size: 4,
        };

        match TcpWorkerHandle::connect(info).await {
            Ok(mut handle) => {
                match handle.health_check().await {
                    Ok(_) => {
                        info!(
                            profile = %entry.profile,
                            lang = %entry.lang,
                            host = %entry.host,
                            port = entry.port,
                            pid = entry.pid,
                            "Discovered healthy TCP worker"
                        );
                        // One probe per scan. The report stays raw here: the
                        // pool admits it, once, under this worker's key.
                        let capabilities = if probed_capabilities {
                            None
                        } else {
                            match handle.capabilities().await {
                                Ok(caps) => {
                                    probed_capabilities = true;
                                    Some(caps)
                                }
                                Err(e) => {
                                    warn!(
                                        host = %entry.host,
                                        port = entry.port,
                                        pid = entry.pid,
                                        error = %e,
                                        "Failed to detect capabilities from discovered TCP worker"
                                    );
                                    None
                                }
                            }
                        };
                        discovered.push(DiscoveredWorker {
                            entry: entry.clone(),
                            profile,
                            lang: lang.clone(),
                            capabilities,
                        });
                    }
                    Err(e) => {
                        warn!(
                            host = %entry.host,
                            port = entry.port,
                            pid = entry.pid,
                            error = %e,
                            "TCP worker health check failed, marking stale"
                        );
                        if entry.is_server_owned() {
                            terminate_registered_daemon(entry.pid, &entry.profile);
                        }
                        stale_indices.push(i);
                    }
                }
                // Drop the handle: the pool will create its own connection.
                drop(handle);
            }
            Err(e) => {
                debug!(
                    host = %entry.host,
                    port = entry.port,
                    pid = entry.pid,
                    error = %e,
                    "Cannot connect to registered worker, marking stale"
                );
                if entry.is_server_owned() {
                    terminate_registered_daemon(entry.pid, &entry.profile);
                }
                stale_indices.push(i);
            }
        }
    }

    // Remove stale entries from the registry file.
    if !stale_indices.is_empty() {
        let remaining: Vec<RegistryEntry> = entries
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !stale_indices.contains(i))
            .map(|(_, e)| e)
            .collect();

        info!(
            removed = stale_indices.len(),
            remaining = remaining.len(),
            "Removed stale entries from worker registry"
        );

        if let Err(e) = write_registry(registry_path, &remaining) {
            warn!(error = %e, "Failed to update worker registry after stale removal");
        }
    }

    RegistryDiscovery {
        workers: discovered,
        refused,
    }
}

#[cfg(test)]
mod refused_registry_worker_tests {
    use super::{ForeignBuildDaemon, ReportedBuild};
    use crate::api::{RefusedRegistryWorker, RegistryWorkerRefusal};

    fn refusal(reported: ReportedBuild) -> ForeignBuildDaemon {
        ForeignBuildDaemon {
            pid: 4242,
            profile: "stanza".into(),
            lang: "eng".into(),
            reported,
            ours: "0.8.5-this".into(),
        }
    }

    /// A refused daemon is returned as a typed entry naming its key and why,
    /// for both refusal kinds, not only logged.
    #[test]
    fn a_refused_daemon_is_a_typed_entry_with_its_key_and_reason() {
        assert_eq!(
            refusal(ReportedBuild::Build("0.8.4-other".into())).to_refused_registry_worker(),
            RefusedRegistryWorker {
                worker_key: "profile:stanza:eng".into(),
                pid: 4242,
                reason: RegistryWorkerRefusal::ForeignBuild {
                    reported_build: "0.8.4-other".into(),
                    server_build: "0.8.5-this".into(),
                },
            }
        );
        assert_eq!(
            refusal(ReportedBuild::Unreported)
                .to_refused_registry_worker()
                .reason,
            RegistryWorkerRefusal::UnreportedBuild {
                server_build: "0.8.5-this".into(),
            }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DiscoveryDisposition, RegistryEntry, RegistryOwnership, discovery_disposition,
        should_shutdown_entry,
    };

    fn external_entry() -> RegistryEntry {
        RegistryEntry {
            pid: 10,
            host: "127.0.0.1".to_string(),
            port: 1234,
            profile: "stanza".to_string(),
            lang: "eng".to_string(),
            engine_overrides: String::new(),
            ownership: RegistryOwnership::External,
            owner_server_instance_id: None,
            owner_server_pid: None,
            started_at: String::new(),
            build_identity: Some(OUR_BUILD.to_string()),
        }
    }

    const OUR_BUILD: &str = "our-build";

    #[test]
    fn discovery_accepts_external_entry() {
        let entry = external_entry();
        assert_eq!(
            discovery_disposition(&entry, "current-server", OUR_BUILD),
            DiscoveryDisposition::Accept
        );
    }

    /// An adoptable daemon written by another build, or by a build that did
    /// not record one, is refused with the restart remedy, and not reaped.
    #[test]
    fn discovery_refuses_a_daemon_from_another_or_an_unreported_build() {
        for (build_identity, reported) in [
            (Some("older-build".to_string()), "build older-build"),
            (None, "an unreported build"),
        ] {
            let mut entry = external_entry();
            entry.build_identity = build_identity;
            let DiscoveryDisposition::RefuseForeignBuild(refusal) =
                discovery_disposition(&entry, "current-server", OUR_BUILD)
            else {
                panic!("a foreign build must be refused");
            };
            let message = refusal.to_string();
            assert!(message.contains(reported), "{message}");
            assert!(
                message.contains("batchalign3 worker start --profile stanza --lang eng"),
                "{message}"
            );
        }
    }

    /// Ownership is settled before build identity: a live foreign owner's
    /// daemon is skipped whatever build wrote it.
    #[test]
    fn a_foreign_owner_is_skipped_before_its_build_is_judged() {
        let mut entry = external_entry();
        entry.ownership = RegistryOwnership::ServerOwned;
        entry.owner_server_instance_id = Some("other-server".to_string());
        entry.owner_server_pid = Some(std::process::id());
        entry.build_identity = Some("older-build".to_string());
        assert_eq!(
            discovery_disposition(&entry, "current-server", OUR_BUILD),
            DiscoveryDisposition::SkipForeignOwner
        );
    }

    #[test]
    fn discovery_accepts_current_server_owned_entry() {
        let mut entry = external_entry();
        entry.ownership = RegistryOwnership::ServerOwned;
        entry.owner_server_instance_id = Some("current-server".to_string());
        entry.owner_server_pid = Some(std::process::id());

        assert_eq!(
            discovery_disposition(&entry, "current-server", OUR_BUILD),
            DiscoveryDisposition::Accept
        );
    }

    #[test]
    fn discovery_skips_foreign_live_server_owned_entry() {
        let mut entry = external_entry();
        entry.ownership = RegistryOwnership::ServerOwned;
        entry.owner_server_instance_id = Some("other-server".to_string());
        entry.owner_server_pid = Some(std::process::id());

        assert_eq!(
            discovery_disposition(&entry, "current-server", OUR_BUILD),
            DiscoveryDisposition::SkipForeignOwner
        );
    }

    #[test]
    fn shutdown_only_targets_current_server_owned_entries() {
        let mut owned = external_entry();
        owned.ownership = RegistryOwnership::ServerOwned;
        owned.owner_server_instance_id = Some("current-server".to_string());
        owned.owner_server_pid = Some(std::process::id());

        let mut foreign = external_entry();
        foreign.ownership = RegistryOwnership::ServerOwned;
        foreign.owner_server_instance_id = Some("other-server".to_string());
        foreign.owner_server_pid = Some(std::process::id());

        assert!(should_shutdown_entry(&owned, "current-server"));
        assert!(!should_shutdown_entry(&foreign, "current-server"));
        assert!(!should_shutdown_entry(&external_entry(), "current-server"));
    }
}
