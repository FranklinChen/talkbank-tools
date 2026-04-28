//! Host-wide memory coordination across local batchalign3 processes.
//!
//! The Rust server can be only one memory consumer on a machine that also runs
//! other batchalign3 ports, CLI daemons, tests, or unrelated inference tools.
//! This module provides a small machine-local coordination ledger so
//! participating batchalign3 processes can serialize heavy worker startups and
//! conservatively reserve job execution memory on shared hosts.

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sysinfo::{Pid, ProcessesToUpdate, System};
use uuid::Uuid;

use crate::api::{MemoryMb, NumWorkers, ReleasedCommand, WorkerLanguage};
use crate::config::ServerConfig;
use crate::runtime;
use crate::worker::WorkerProfile;

const DEFAULT_LOCK_POLL: Duration = Duration::from_secs(1);
const DEFAULT_TEST_LOCK_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// Host-wide memory pressure level derived from the current memory snapshot and
/// reserved headroom.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
pub enum HostMemoryPressureLevel {
    /// Plenty of free headroom remains after the configured reserve.
    #[default]
    Healthy,
    /// Some headroom remains, but operators should expect reduced concurrency.
    Guarded,
    /// Very little headroom remains; only small new reservations should fit.
    Constrained,
    /// The configured reserve is exhausted or nearly exhausted.
    Critical,
}

/// Runtime-owned configuration for the machine-local memory ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostMemoryRuntimeConfig {
    /// Shared ledger path used by local batchalign3 processes on this host.
    pub coordinator_path: PathBuf,
    /// Minimum free memory to preserve after reservations are granted.
    pub reserve_mb: MemoryMb,
    /// Maximum concurrent worker/model startups allowed across the host.
    pub max_concurrent_worker_startups: usize,
}

impl HostMemoryRuntimeConfig {
    /// Build runtime config from server-level settings.
    pub fn from_server_config(config: &ServerConfig) -> Self {
        Self {
            coordinator_path: default_host_memory_ledger_path(),
            reserve_mb: config.resolved_memory_gate_mb(),
            max_concurrent_worker_startups: config.max_concurrent_worker_startups as usize,
        }
    }

    /// Build runtime config from explicit sources.
    pub fn from_sources(
        coordinator_path: PathBuf,
        reserve_mb: MemoryMb,
        max_concurrent_worker_startups: usize,
    ) -> Self {
        Self {
            coordinator_path,
            reserve_mb,
            max_concurrent_worker_startups: max_concurrent_worker_startups.max(1),
        }
    }
}

impl Default for HostMemoryRuntimeConfig {
    fn default() -> Self {
        Self {
            coordinator_path: default_host_memory_ledger_path(),
            // `Default::default()` returns the post-resolution value
            // so callers (mainly tests) get a sensible reserve without
            // building a `ServerConfig` themselves.
            reserve_mb: ServerConfig::default().resolved_memory_gate_mb(),
            max_concurrent_worker_startups: ServerConfig::default().max_concurrent_worker_startups
                as usize,
        }
    }
}

/// Default machine-local ledger path for host memory coordination.
pub fn default_host_memory_ledger_path() -> PathBuf {
    if let Some(explicit) = std::env::var_os("BATCHALIGN_HOST_MEMORY_LEDGER") {
        return PathBuf::from(explicit);
    }
    let suffix = host_ledger_suffix();
    std::env::temp_dir().join(format!("batchalign3-host-memory-{suffix}.json"))
}

/// Summary of the current host-memory coordination state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostMemorySnapshot {
    /// Total physical memory observed by the OS.
    pub total_mb: MemoryMb,
    /// Currently available memory observed by the OS.
    pub available_mb: MemoryMb,
    /// Reserved low-water-mark that the coordinator keeps free.
    pub reserve_mb: MemoryMb,
    /// Sum of active reservation amounts recorded in the ledger.
    pub active_reserved_mb: MemoryMb,
    /// Number of active startup leases.
    pub startup_leases: usize,
    /// Number of active job-execution leases.
    pub job_execution_leases: usize,
    /// Number of active machine-wide ML test locks.
    pub ml_test_locks: usize,
    /// Human-readable lease labels for operator debugging.
    pub active_lease_labels: Vec<String>,
    /// Pressure level derived from the snapshot.
    pub pressure_level: HostMemoryPressureLevel,
}

/// One acquired host-memory lease. Releasing the lease removes it from the
/// machine-local ledger.
pub struct HostMemoryLease {
    ledger_path: PathBuf,
    lease_id: String,
    released: bool,
}

impl HostMemoryLease {
    fn new(ledger_path: PathBuf, lease_id: String) -> Self {
        Self {
            ledger_path,
            lease_id,
            released: false,
        }
    }

    /// Release the lease immediately.
    pub fn release(mut self) {
        self.release_internal();
    }

    fn release_internal(&mut self) {
        if self.released {
            return;
        }
        let path = self.ledger_path.clone();
        let lease_id = self.lease_id.clone();
        let _ = with_locked_ledger(&path, |ledger| {
            ledger.leases.retain(|lease| lease.id != lease_id);
            Ok(())
        });
        self.released = true;
    }
}

impl Drop for HostMemoryLease {
    fn drop(&mut self) {
        self.release_internal();
    }
}

/// A machine-local exclusive lock for real-model ML tests.
pub struct MachineMlTestLock {
    _lease: HostMemoryLease,
}

impl MachineMlTestLock {
    /// Acquire the machine-wide ML test lock, waiting until other local test
    /// binaries release it.
    pub fn acquire(label: &str) -> Result<Self, HostMemoryError> {
        let coordinator = HostMemoryCoordinator::new(HostMemoryRuntimeConfig::default());
        let lease = coordinator.acquire_ml_test_lock(
            label,
            DEFAULT_TEST_LOCK_TIMEOUT,
            DEFAULT_LOCK_POLL,
        )?;
        Ok(Self { _lease: lease })
    }
}

/// Result of planning one job's host-memory execution reservation.
pub struct JobExecutionPlan {
    /// File-level worker count granted for this job under current host pressure.
    pub granted_workers: NumWorkers,
    /// Original requested worker count before host-memory clamping.
    pub requested_workers: NumWorkers,
    /// Reservation held for the duration of the job.
    pub lease: HostMemoryLease,
    /// Total reserved memory for the job execution window.
    pub reserved_mb: MemoryMb,
}

/// Errors raised by the host-memory coordinator.
#[derive(Debug, thiserror::Error)]
pub enum HostMemoryError {
    /// Failed to read or write the machine-local ledger.
    #[error("host-memory ledger I/O failed at {path}: {source}")]
    Io {
        /// Path to the shared ledger file.
        path: PathBuf,
        /// Underlying filesystem error.
        #[source]
        source: std::io::Error,
    },
    /// The ledger file exists but cannot be parsed.
    #[error("host-memory ledger is corrupt at {path}: {message}")]
    CorruptLedger {
        /// Path to the shared ledger file.
        path: PathBuf,
        /// Human-readable parse failure.
        message: String,
    },
    /// The host reserve would be exceeded by the requested reservation.
    #[error(
        "host-memory reserve would be exceeded for {label}: {available_mb} MB available, \
         {pending_reserved_mb} MB already reserved, {requested_mb} MB requested, \
         {reserve_mb} MB reserved for host headroom (total RAM: {total_mb} MB)"
    )]
    CapacityRejected {
        /// Human-readable request label.
        label: String,
        /// Current available memory from the OS snapshot.
        available_mb: u64,
        /// Sum of active reservation amounts in the ledger.
        pending_reserved_mb: u64,
        /// Requested additional reservation.
        requested_mb: u64,
        /// Configured reserve that must remain free.
        reserve_mb: u64,
        /// Total physical RAM observed on this machine.
        total_mb: u64,
    },
    /// Another local process is already using the exclusive ML test lock.
    #[error("machine-wide ML test lock is already held by {holders:?}")]
    MlTestLockBusy {
        /// Labels of active lock holders.
        holders: Vec<String>,
    },
    /// The host-wide startup limit is currently saturated.
    #[error(
        "worker startup slots busy for {label}: {active_slots}/{max_slots} local startup slots in use"
    )]
    StartupSlotsBusy {
        /// Human-readable request label.
        label: String,
        /// Number of active startup slots.
        active_slots: usize,
        /// Configured host-wide startup slot limit.
        max_slots: usize,
    },
    /// Waiting for capacity exceeded the configured timeout.
    #[error(
        "timed out waiting for host-memory capacity for {label} after {waited_s}s: {last_reason}"
    )]
    TimedOut {
        /// Human-readable request label.
        label: String,
        /// Timeout window in seconds.
        waited_s: u64,
        /// Last rejection reason observed while waiting.
        last_reason: String,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum MemoryLeaseKind {
    WorkerStartup,
    JobExecution,
    MlTestExclusive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryLeaseRecord {
    id: String,
    kind: MemoryLeaseKind,
    owner_pid: u32,
    reserved_mb: u64,
    startup_slot: bool,
    label: String,
    created_at_epoch_s: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct MemoryLedger {
    leases: Vec<MemoryLeaseRecord>,
}

#[derive(Debug, Clone)]
struct MemoryLeaseRequest {
    kind: MemoryLeaseKind,
    reserved_mb: MemoryMb,
    startup_slot: bool,
    label: String,
}

#[derive(Debug, Clone, Copy)]
struct SystemMemorySnapshot {
    total_mb: MemoryMb,
    available_mb: MemoryMb,
}

/// Machine-local coordinator that serializes access to the shared memory ledger.
#[derive(Debug, Clone)]
pub struct HostMemoryCoordinator {
    config: HostMemoryRuntimeConfig,
}

impl HostMemoryCoordinator {
    /// Create one host-memory coordinator from runtime config.
    pub fn new(config: HostMemoryRuntimeConfig) -> Self {
        Self { config }
    }

    /// Build one coordinator from server settings.
    pub fn from_server_config(config: &ServerConfig) -> Self {
        Self::new(HostMemoryRuntimeConfig::from_server_config(config))
    }

    /// Return the shared ledger path backing this coordinator.
    pub fn coordinator_path(&self) -> &Path {
        &self.config.coordinator_path
    }

    /// Return the current host-memory snapshot for health/status reporting.
    pub fn snapshot(&self) -> Result<HostMemorySnapshot, HostMemoryError> {
        with_locked_ledger(&self.config.coordinator_path, |ledger| {
            let system = system_memory_snapshot();
            let active_reserved_mb: u64 = ledger.leases.iter().map(|lease| lease.reserved_mb).sum();
            let startup_leases = ledger
                .leases
                .iter()
                .filter(|lease| lease.kind == MemoryLeaseKind::WorkerStartup)
                .count();
            let job_execution_leases = ledger
                .leases
                .iter()
                .filter(|lease| lease.kind == MemoryLeaseKind::JobExecution)
                .count();
            let ml_test_locks = ledger
                .leases
                .iter()
                .filter(|lease| lease.kind == MemoryLeaseKind::MlTestExclusive)
                .count();
            let active_lease_labels = ledger
                .leases
                .iter()
                .map(|lease| format!("{:?}:{}:{}MB", lease.kind, lease.label, lease.reserved_mb))
                .collect();
            Ok(HostMemorySnapshot {
                total_mb: system.total_mb,
                available_mb: system.available_mb,
                reserve_mb: self.config.reserve_mb,
                active_reserved_mb: MemoryMb(active_reserved_mb),
                startup_leases,
                job_execution_leases,
                ml_test_locks,
                active_lease_labels,
                pressure_level: pressure_level_for(system.available_mb, self.config.reserve_mb),
            })
        })
    }

    /// Wait for a host-memory startup reservation for one worker/model load.
    pub fn acquire_worker_startup_lease(
        &self,
        profile: WorkerProfile,
        startup_reservation_mb: MemoryMb,
        lang: &WorkerLanguage,
        engine_overrides: &str,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<HostMemoryLease, HostMemoryError> {
        let request = MemoryLeaseRequest {
            kind: MemoryLeaseKind::WorkerStartup,
            reserved_mb: startup_reservation_mb,
            startup_slot: true,
            label: format!(
                "worker-startup:{}:{}:{}",
                profile.label(),
                lang,
                engine_overrides
            ),
        };
        self.wait_for_lease(request, timeout, poll_interval)
    }

    /// Acquire a machine-wide exclusive lock for real-model ML tests.
    pub fn acquire_ml_test_lock(
        &self,
        label: &str,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<HostMemoryLease, HostMemoryError> {
        let request = MemoryLeaseRequest {
            kind: MemoryLeaseKind::MlTestExclusive,
            reserved_mb: MemoryMb(0),
            startup_slot: false,
            label: label.to_owned(),
        };
        self.wait_for_lease(request, timeout, poll_interval)
    }

    /// Plan and reserve one job's execution memory window.
    ///
    /// The coordinator may reduce the requested worker count when host pressure
    /// is high. The returned lease stays alive for the whole job so other local
    /// processes see the reservation and conservatively back off.
    pub fn wait_for_job_execution_plan(
        &self,
        command: ReleasedCommand,
        requested_workers: NumWorkers,
        label: &str,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<JobExecutionPlan, HostMemoryError> {
        let deadline = Instant::now() + timeout;
        let mut last_reason = String::from("no capacity decision yet");
        loop {
            match self.try_plan_job_execution(command, requested_workers, label) {
                Ok(plan) => return Ok(plan),
                Err(error)
                    if timeout > Duration::ZERO
                        && Instant::now() < deadline
                        && retryable_error(&error) =>
                {
                    last_reason = error.to_string();
                    std::thread::sleep(poll_interval);
                }
                Err(error) => {
                    if timeout > Duration::ZERO && Instant::now() >= deadline {
                        return Err(HostMemoryError::TimedOut {
                            label: label.to_owned(),
                            waited_s: timeout.as_secs(),
                            last_reason,
                        });
                    }
                    return Err(error);
                }
            }
        }
    }

    fn try_plan_job_execution(
        &self,
        command: ReleasedCommand,
        requested_workers: NumWorkers,
        label: &str,
    ) -> Result<JobExecutionPlan, HostMemoryError> {
        let per_worker_budget = runtime::command_execution_budget_mb(command.as_ref());
        with_locked_ledger(&self.config.coordinator_path, |ledger| {
            let system = system_memory_snapshot();
            let pending_reserved_mb: u64 =
                ledger.leases.iter().map(|lease| lease.reserved_mb).sum();
            let Some((granted_workers, reserved_mb)) = plan_job_reservation(
                requested_workers.0,
                per_worker_budget.0,
                system.available_mb.0,
                self.config.reserve_mb.0,
                pending_reserved_mb,
            ) else {
                return Err(HostMemoryError::CapacityRejected {
                    label: label.to_owned(),
                    available_mb: system.available_mb.0,
                    pending_reserved_mb,
                    requested_mb: per_worker_budget
                        .0
                        .saturating_mul(requested_workers.0 as u64),
                    reserve_mb: self.config.reserve_mb.0,
                    total_mb: system.total_mb.0,
                });
            };

            let record = MemoryLeaseRecord {
                id: Uuid::new_v4().to_string(),
                kind: MemoryLeaseKind::JobExecution,
                owner_pid: std::process::id(),
                reserved_mb,
                startup_slot: false,
                label: label.to_owned(),
                created_at_epoch_s: unix_epoch_s(),
            };
            let lease =
                HostMemoryLease::new(self.config.coordinator_path.clone(), record.id.clone());
            ledger.leases.push(record);
            Ok(JobExecutionPlan {
                granted_workers: NumWorkers(granted_workers),
                requested_workers,
                lease,
                reserved_mb: MemoryMb(reserved_mb),
            })
        })
    }

    fn wait_for_lease(
        &self,
        request: MemoryLeaseRequest,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<HostMemoryLease, HostMemoryError> {
        let deadline = Instant::now() + timeout;
        let mut last_reason = String::from("no capacity decision yet");
        loop {
            match self.try_acquire_lease(request.clone()) {
                Ok(lease) => return Ok(lease),
                Err(error)
                    if timeout > Duration::ZERO
                        && Instant::now() < deadline
                        && retryable_error(&error) =>
                {
                    last_reason = error.to_string();
                    std::thread::sleep(poll_interval);
                }
                Err(error) => {
                    if timeout > Duration::ZERO && Instant::now() >= deadline {
                        return Err(HostMemoryError::TimedOut {
                            label: request.label,
                            waited_s: timeout.as_secs(),
                            last_reason,
                        });
                    }
                    return Err(error);
                }
            }
        }
    }

    fn try_acquire_lease(
        &self,
        request: MemoryLeaseRequest,
    ) -> Result<HostMemoryLease, HostMemoryError> {
        with_locked_ledger(&self.config.coordinator_path, |ledger| {
            if request.kind == MemoryLeaseKind::MlTestExclusive {
                let holders: Vec<String> = ledger
                    .leases
                    .iter()
                    .filter(|lease| lease.kind == MemoryLeaseKind::MlTestExclusive)
                    .map(|lease| lease.label.clone())
                    .collect();
                if !holders.is_empty() {
                    return Err(HostMemoryError::MlTestLockBusy { holders });
                }
            }

            if request.startup_slot {
                let active_slots = ledger
                    .leases
                    .iter()
                    .filter(|lease| lease.startup_slot)
                    .count();
                if active_slots >= self.config.max_concurrent_worker_startups {
                    return Err(HostMemoryError::StartupSlotsBusy {
                        label: request.label.clone(),
                        active_slots,
                        max_slots: self.config.max_concurrent_worker_startups,
                    });
                }
            }

            let system = system_memory_snapshot();
            let pending_reserved_mb: u64 =
                ledger.leases.iter().map(|lease| lease.reserved_mb).sum();
            let projected_available_mb = system
                .available_mb
                .0
                .saturating_sub(pending_reserved_mb)
                .saturating_sub(request.reserved_mb.0);
            if projected_available_mb < self.config.reserve_mb.0 {
                return Err(HostMemoryError::CapacityRejected {
                    label: request.label.clone(),
                    available_mb: system.available_mb.0,
                    pending_reserved_mb,
                    requested_mb: request.reserved_mb.0,
                    reserve_mb: self.config.reserve_mb.0,
                    total_mb: system.total_mb.0,
                });
            }

            let record = MemoryLeaseRecord {
                id: Uuid::new_v4().to_string(),
                kind: request.kind,
                owner_pid: std::process::id(),
                reserved_mb: request.reserved_mb.0,
                startup_slot: request.startup_slot,
                label: request.label,
                created_at_epoch_s: unix_epoch_s(),
            };
            let lease =
                HostMemoryLease::new(self.config.coordinator_path.clone(), record.id.clone());
            ledger.leases.push(record);
            Ok(lease)
        })
    }
}

fn with_locked_ledger<T>(
    path: &Path,
    action: impl FnOnce(&mut MemoryLedger) -> Result<T, HostMemoryError>,
) -> Result<T, HostMemoryError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| HostMemoryError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    }

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|source| HostMemoryError::Io {
            path: path.to_path_buf(),
            source,
        })?;

    file.lock_exclusive()
        .map_err(|source| HostMemoryError::Io {
            path: path.to_path_buf(),
            source,
        })?;

    let mut raw = String::new();
    file.read_to_string(&mut raw)
        .map_err(|source| HostMemoryError::Io {
            path: path.to_path_buf(),
            source,
        })?;

    let mut ledger = if raw.trim().is_empty() {
        MemoryLedger::default()
    } else {
        serde_json::from_str::<MemoryLedger>(&raw).map_err(|error| {
            HostMemoryError::CorruptLedger {
                path: path.to_path_buf(),
                message: error.to_string(),
            }
        })?
    };

    prune_stale_leases(&mut ledger);

    let result = action(&mut ledger);

    file.set_len(0).map_err(|source| HostMemoryError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    file.seek(SeekFrom::Start(0))
        .map_err(|source| HostMemoryError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    serde_json::to_writer_pretty(&mut file, &ledger).map_err(|error| {
        HostMemoryError::CorruptLedger {
            path: path.to_path_buf(),
            message: error.to_string(),
        }
    })?;
    file.write_all(b"\n")
        .map_err(|source| HostMemoryError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    file.sync_all().map_err(|source| HostMemoryError::Io {
        path: path.to_path_buf(),
        source,
    })?;

    result
}

fn prune_stale_leases(ledger: &mut MemoryLedger) {
    ledger
        .leases
        .retain(|lease| process_is_alive(lease.owner_pid));
}

fn process_is_alive(pid: u32) -> bool {
    let mut system = System::new();
    let pid = Pid::from_u32(pid);
    system.refresh_processes(ProcessesToUpdate::Some(&[pid]), false);
    system.process(pid).is_some()
}

fn system_memory_snapshot() -> SystemMemorySnapshot {
    let mut system = System::new();
    system.refresh_memory();
    SystemMemorySnapshot {
        total_mb: MemoryMb(system.total_memory() / (1024 * 1024)),
        available_mb: MemoryMb(system.available_memory() / (1024 * 1024)),
    }
}

/// Total physical RAM on this host, in MB.
///
/// Public delegator over `system_memory_snapshot()` so the
/// `host_facts::RealHostFactsSource` can populate `HostFacts::ram_total_mb`
/// without depending on this module's internal `SystemMemorySnapshot`
/// shape. Held in this module so all sysinfo polling for the host's
/// RAM lives in one place.
pub fn detect_total_memory_mb() -> MemoryMb {
    system_memory_snapshot().total_mb
}

/// Currently-available RAM on this host, in MB.
///
/// "Available" follows the sysinfo definition (`free + reclaimable`),
/// which on macOS undercounts vs Activity Monitor's "memory pressure"
/// — see `worker::memory_guard` for the live polling helper that
/// applies tier-aware reserves on top of this raw figure.
pub fn detect_available_memory_mb() -> MemoryMb {
    system_memory_snapshot().available_mb
}

fn pressure_level_for(available_mb: MemoryMb, reserve_mb: MemoryMb) -> HostMemoryPressureLevel {
    if available_mb.0 <= reserve_mb.0 {
        return HostMemoryPressureLevel::Critical;
    }
    let extra_headroom = available_mb.0.saturating_sub(reserve_mb.0);
    if extra_headroom <= 2_048 {
        HostMemoryPressureLevel::Constrained
    } else if extra_headroom <= 8_192 {
        HostMemoryPressureLevel::Guarded
    } else {
        HostMemoryPressureLevel::Healthy
    }
}

fn plan_job_reservation(
    requested_workers: usize,
    per_worker_budget_mb: u64,
    available_mb: u64,
    reserve_mb: u64,
    pending_reserved_mb: u64,
) -> Option<(usize, u64)> {
    for workers in (1..=requested_workers).rev() {
        let requested_mb = per_worker_budget_mb.saturating_mul(workers as u64);
        let projected_available_mb = available_mb
            .saturating_sub(pending_reserved_mb)
            .saturating_sub(requested_mb);
        if projected_available_mb >= reserve_mb {
            return Some((workers, requested_mb));
        }
    }
    None
}

fn retryable_error(error: &HostMemoryError) -> bool {
    match error {
        HostMemoryError::CapacityRejected {
            total_mb,
            requested_mb,
            reserve_mb,
            ..
        } => capacity_rejection_is_retryable(*total_mb, *requested_mb, *reserve_mb),
        HostMemoryError::MlTestLockBusy { .. } | HostMemoryError::StartupSlotsBusy { .. } => true,
        HostMemoryError::Io { .. }
        | HostMemoryError::CorruptLedger { .. }
        | HostMemoryError::TimedOut { .. } => false,
    }
}

fn capacity_rejection_is_retryable(total_mb: u64, requested_mb: u64, reserve_mb: u64) -> bool {
    requested_mb.saturating_add(reserve_mb) <= total_mb
}

fn unix_epoch_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn host_ledger_suffix() -> String {
    let raw = std::env::var("USER")
        .ok()
        .or_else(|| std::env::var("USERNAME").ok())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| String::from("default"));
    raw.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        HostMemoryError, HostMemoryPressureLevel, plan_job_reservation, pressure_level_for,
        retryable_error,
    };
    use crate::api::MemoryMb;

    #[test]
    fn pressure_levels_follow_available_headroom() {
        assert_eq!(
            pressure_level_for(MemoryMb(64_000), MemoryMb(8_192)),
            HostMemoryPressureLevel::Healthy
        );
        assert_eq!(
            pressure_level_for(MemoryMb(12_000), MemoryMb(8_192)),
            HostMemoryPressureLevel::Guarded
        );
        assert_eq!(
            pressure_level_for(MemoryMb(9_000), MemoryMb(8_192)),
            HostMemoryPressureLevel::Constrained
        );
        assert_eq!(
            pressure_level_for(MemoryMb(8_192), MemoryMb(8_192)),
            HostMemoryPressureLevel::Critical
        );
    }

    #[test]
    fn job_planner_reduces_worker_count_to_fit_headroom() {
        let planned =
            plan_job_reservation(8, 6_000, 32_000, 8_192, 0).expect("some worker count should fit");
        assert_eq!(planned, (3, 18_000));
    }

    #[test]
    fn job_planner_accounts_for_pending_reservations() {
        let planned = plan_job_reservation(4, 4_000, 24_000, 8_192, 6_000)
            .expect("some worker count should fit");
        assert_eq!(planned, (2, 8_000));
    }

    #[test]
    fn job_planner_rejects_when_even_one_worker_breaks_reserve() {
        assert!(plan_job_reservation(2, 8_000, 12_000, 8_192, 1_000).is_none());
    }

    #[test]
    fn impossible_capacity_rejection_is_not_retryable() {
        let error = HostMemoryError::CapacityRejected {
            label: String::from("impossible"),
            available_mb: 32_000,
            pending_reserved_mb: 0,
            requested_mb: 80_000,
            reserve_mb: 8_192,
            total_mb: 64_000,
        };
        assert!(
            !retryable_error(&error),
            "impossible reservations should fail immediately"
        );
    }

    #[test]
    fn capacity_rejection_stays_retryable_when_it_can_fit_later() {
        let error = HostMemoryError::CapacityRejected {
            label: String::from("contended"),
            available_mb: 6_000,
            pending_reserved_mb: 20_000,
            requested_mb: 16_000,
            reserve_mb: 8_192,
            total_mb: 64_000,
        };
        assert!(
            retryable_error(&error),
            "requests that could fit later should keep retry semantics"
        );
    }
}
