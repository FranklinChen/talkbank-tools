//! Host-wide memory coordination across local batchalign3 processes.
//!
//! The Rust server can be only one memory consumer on a machine that also runs
//! other batchalign3 ports, CLI daemons, tests, or unrelated inference tools.
//! This module provides a small machine-local coordination ledger so
//! participating batchalign3 processes can serialize worker startups, reserve
//! each model a worker loads, and conservatively reserve job execution memory
//! on shared hosts.
//!
//! # What a lease counts
//!
//! Admission compares the reservations with the memory the operating system
//! reports AVAILABLE now. Memory a worker has already taken is gone from that
//! reading, so a lease counts only what its holder has not yet taken: a
//! worker-startup lease counts the process baseline until the worker is ready
//! and is then dropped; a model-load lease counts the model while it loads and,
//! once the worker confirms the load, only the activation margin the model
//! has yet to use (`HostMemoryLease::settler`). Counting a resident model in
//! full would count it twice, once in the reading and once in the ledger, and
//! a load waiting with no deadline behind that phantom would wait for memory
//! the host has.
//!
//! # Queued waits
//!
//! A worker startup or a model load that the host cannot fit now queues
//! ([`HostMemoryCoordinator::queue_for_lease`]): it has no deadline, says it is
//! waiting (a log line when it queues, a warning every
//! [`QUEUED_LEASE_REPORT_INTERVAL`], and its [`QueuedLeaseObserver`]), and ends
//! when the lease is granted, when the request is refused for good (one larger
//! than the machine, an unreadable host or ledger), or when its future is
//! dropped (a cancelled job). Real-model test locks and job execution plans
//! keep a bounded wait ([`BoundedWait`]); a job that cannot fit is
//! re-queued by the runner rather than waiting inside it.

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sysinfo::{Pid, ProcessesToUpdate, System};
use uuid::Uuid;

use crate::api::{MachineTime, MemoryMb, NumWorkers, ReleasedCommand, WorkerLanguage};
use crate::config::{ServerConfig, WorkerStartupLimit};
use crate::runtime;
use crate::worker::WorkerPid;
use crate::worker::WorkerProfile;

const DEFAULT_LOCK_POLL: Duration = Duration::from_secs(1);
const DEFAULT_TEST_LOCK_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// How often a queued lease wait retries the ledger.
const QUEUED_LEASE_POLL: Duration = Duration::from_secs(1);

/// How often a queued lease wait warns that it is still waiting.
///
/// Shorter than the saturated-pool checkout report (300 s): a pool queue
/// behind long requests is ordinary, while a host that cannot fit a model
/// is worth an operator's attention within a minute.
pub const QUEUED_LEASE_REPORT_INTERVAL: Duration = Duration::from_secs(60);

pub use crate::types::response::HostMemoryPressureLevel;

/// Where the memory and load figures behind a spawn decision come from.
///
/// Every admission, spawn, ledger and eviction decision reads the host
/// through this value, never through [`read_system_memory`] directly. It is
/// fixed where the host is composed ([`HostSnapshot`]): `read` takes the live
/// machine, `for_tests` declares one. A test that read the real machine had
/// an outcome that depended on it: a fixture that fit a 256 GB workstation
/// was refused on a 16 GB CI runner, and the gates that guarded against that
/// were switched off in tests rather than made deterministic.
///
/// [`HostSnapshot`]: crate::host_facts::HostSnapshot
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostReadings {
    /// The machine this process runs on, read at each decision.
    Live,
    /// A fixed machine, so the same decisions are made on every machine that
    /// runs them.
    Declared(DeclaredHost),
}

/// The most memory this host can ever offer a reservation, before its
/// reserve is kept free: the one bound that both "could this request ever
/// fit" decisions read (worker lease admission and FA request feasibility),
/// so they cannot disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostCeiling(MemoryMb);

impl HostCeiling {
    /// A stated ceiling, for tests of the decisions that read one.
    #[cfg(test)]
    pub(crate) const fn for_tests(mb: u64) -> Self {
        Self(MemoryMb(mb))
    }

    /// The bound in MB.
    pub fn mb(self) -> MemoryMb {
        self.0
    }

    /// Whether `needed` can ever be granted here with `reserve` kept free.
    pub fn admits(self, needed: MemoryMb, reserve: MemoryMb) -> bool {
        needed.0.saturating_add(reserve.0) <= self.0.0
    }
}

impl HostReadings {
    /// The most this host can ever offer: a live machine's total RAM, since
    /// other processes can free what they hold, and a declared machine's
    /// available RAM, since an idle declared host never frees more.
    ///
    /// Taken from the same reading the decision uses, so the ceiling and the
    /// available figure beside it describe one moment.
    pub fn ceiling_of(&self, memory: SystemMemory) -> HostCeiling {
        HostCeiling(match self {
            Self::Live => memory.total_mb,
            Self::Declared(_) => memory.available_mb,
        })
    }

    /// Total and available RAM, as this source reports them now.
    pub fn memory(&self) -> Result<SystemMemory, HostUnreadable> {
        match self {
            Self::Live => read_system_memory(),
            Self::Declared(host) => Ok(host.memory),
        }
    }

    /// The 1-minute load average. A declared host is idle.
    pub fn load_average_1m(&self) -> Result<f64, sysinfo::Error> {
        match self {
            Self::Live => Ok(System::load_average()?.one),
            Self::Declared(_) => Ok(0.0),
        }
    }

    /// Logical CPUs: the CPU gate's saturation threshold.
    pub fn logical_cpus(&self) -> u32 {
        match self {
            // One CPU when the count is unreadable: a daemon that cannot
            // read it has larger problems, and one keeps the gate meaningful.
            Self::Live => std::thread::available_parallelism()
                .map(|count| u32::try_from(count.get()).unwrap_or(u32::MAX))
                .unwrap_or(1),
            Self::Declared(host) => host.logical_cpus,
        }
    }
}

/// A machine stated rather than measured: the memory and CPU count of a
/// declared [`HostFacts`], idle.
///
/// [`HostFacts`]: crate::host_facts::HostFacts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeclaredHost {
    memory: SystemMemory,
    logical_cpus: u32,
}

impl DeclaredHost {
    /// The host the facts describe.
    pub fn from_facts(facts: &crate::host_facts::HostFacts) -> Self {
        Self {
            memory: SystemMemory {
                total_mb: MemoryMb(facts.ram_total_mb),
                available_mb: MemoryMb(facts.ram_available_mb),
            },
            logical_cpus: facts.cpu_logical_count,
        }
    }
}

/// Runtime-owned configuration for the machine-local memory ledger.
///
/// Which host the ledger's decisions read is not here: it is the
/// [`HostReadings`] the coordinator is built with, from the same host
/// snapshot as everything else a runtime decides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostMemoryRuntimeConfig {
    /// Shared ledger path used by local batchalign3 processes on this host.
    pub coordinator_path: PathBuf,
    /// Minimum free memory to preserve after reservations are granted.
    pub reserve_mb: MemoryMb,
    /// Maximum concurrent worker/model startups allowed across the host;
    /// at least one by type.
    pub max_concurrent_worker_startups: WorkerStartupLimit,
}

impl HostMemoryRuntimeConfig {
    /// Build runtime config from server-level settings.
    pub fn from_server_config(config: &ServerConfig) -> Self {
        Self {
            coordinator_path: default_host_memory_ledger_path(),
            reserve_mb: config.resolved_memory_gate_mb(),
            max_concurrent_worker_startups: config.max_concurrent_worker_startups,
        }
    }

    /// Build runtime config from explicit sources.
    pub fn from_sources(
        coordinator_path: PathBuf,
        reserve_mb: MemoryMb,
        max_concurrent_worker_startups: WorkerStartupLimit,
    ) -> Self {
        Self {
            coordinator_path,
            reserve_mb,
            max_concurrent_worker_startups,
        }
    }
}

/// The environment variable naming a process's host-memory ledger file.
///
/// Read once, by [`default_host_memory_ledger_path`]. A test harness that
/// starts `batchalign3` as a child hands it a private ledger by setting this
/// on that child's `Command`; nothing sets it on its own process.
pub const HOST_MEMORY_LEDGER_ENV: &str = "BATCHALIGN_HOST_MEMORY_LEDGER";

/// Default machine-local ledger path for host memory coordination: the
/// file every batchalign3 process on this machine shares, unless
/// [`HOST_MEMORY_LEDGER_ENV`] names another.
pub fn default_host_memory_ledger_path() -> PathBuf {
    if let Some(explicit) = std::env::var_os(HOST_MEMORY_LEDGER_ENV) {
        return PathBuf::from(explicit);
    }
    let suffix = host_ledger_suffix();
    std::env::temp_dir().join(format!("batchalign3-host-memory-{suffix}.json"))
}

/// Which process-private ledger a test process uses.
///
/// Two, never one: an in-process declared host decides on declared memory,
/// and child `batchalign3` processes decide on the live machine, so leases
/// taken against one must never be counted against the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivateLedger {
    /// The declared test host's ledger (`HostSnapshot::for_tests`).
    DeclaredHost,
    /// The ledger a test harness hands its child `batchalign3` processes.
    ChildProcesses,
}

/// This process's private ledger file for `which`, emptied the first time
/// it is named in this process.
///
/// Private to the process because concurrent test binaries must not see
/// each other's leases, and emptied once because a file left by an earlier
/// process with the same PID would carry leases that look like ours.
///
/// This replaced a `Once`-guarded `set_var` of [`HOST_MEMORY_LEDGER_ENV`]
/// on the test process itself, which every test entry point had to call
/// before anything built a ledger (about twenty call sites, one forgotten
/// call away from sharing the machine's ledger). The declared host now
/// names its own ledger, and a child is handed its ledger explicitly.
#[doc(hidden)]
pub fn private_test_ledger_path(which: PrivateLedger) -> PathBuf {
    static DECLARED_HOST: LazyLock<PathBuf> =
        LazyLock::new(|| fresh_private_ledger("declared-host"));
    static CHILD_PROCESSES: LazyLock<PathBuf> =
        LazyLock::new(|| fresh_private_ledger("child-processes"));
    match which {
        PrivateLedger::DeclaredHost => DECLARED_HOST.clone(),
        PrivateLedger::ChildProcesses => CHILD_PROCESSES.clone(),
    }
}

/// A ledger path private to this process, with any earlier file removed.
fn fresh_private_ledger(purpose: &str) -> PathBuf {
    let pid = std::process::id();
    let path = std::env::temp_dir().join(format!("batchalign3-host-memory-{purpose}-{pid}.json"));
    // Absent is the expected case; any other failure leaves a stale file
    // whose leases the ledger's own pruning will still reclaim.
    if let Err(error) = std::fs::remove_file(&path)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!(%error, path = %path.display(), "could not empty a private test ledger");
    }
    path
}

/// Summary of the current host-memory coordination state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostMemorySnapshot {
    /// Total physical memory observed by the OS.
    pub total_mb: MemoryMb,
    /// The most this host can ever offer ([`HostCeiling`]): what physical
    /// feasibility is judged against.
    pub ceiling: HostCeiling,
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
#[derive(Debug)]
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

    /// Tag this lease with the worker subprocess PID it represents.
    ///
    /// Why this exists: when the daemon acquires a worker-startup lease,
    /// it doesn't yet have the worker's PID, the spawn happens after
    /// the reservation is granted. Once `Child::id()` is available, the
    /// caller calls `set_worker_pid(child_pid)` so the lease record
    /// records the worker, not the daemon. From that point on,
    /// `prune_stale_leases` uses the worker's liveness to decide whether
    /// to retain the lease, closing the Bug 2 (ghost-slot) accounting
    /// drift that wedged a development host (the historical "Bug 2" incident).
    ///
    /// Returns `Ok(())` on successful update, or `HostMemoryError` if
    /// the ledger could not be locked or rewritten. The lease's
    /// `released` state is unchanged.
    pub fn set_worker_pid(&self, worker_pid: WorkerPid) -> Result<(), HostMemoryError> {
        let path = self.ledger_path.clone();
        let lease_id = self.lease_id.clone();
        with_locked_ledger(&path, move |ledger: &mut MemoryLedger| {
            for lease in ledger.leases.iter_mut() {
                if lease.id == lease_id {
                    lease.worker_pid = Some(worker_pid);
                    return Ok(());
                }
            }
            // Lease was already pruned (e.g., daemon-side bookkeeping
            // already dropped it). Treat as a no-op rather than an
            // error: the caller's intent (associate this lease with a
            // worker PID) is satisfied vacuously.
            Ok(())
        })
    }

    /// What changing this lease's reservation in place needs, detached from
    /// the lease ([`LeaseSettler::settle`]): the transition from a model
    /// loading (counted in full) to a model resident (counted at what it has
    /// yet to use), see the module docs. Detached so the blocking pool can
    /// run it while the lease stays with its owner, and a settle that never
    /// runs cannot take the lease with it.
    pub fn settler(&self) -> LeaseSettler {
        LeaseSettler {
            ledger_path: self.ledger_path.clone(),
            lease_id: self.lease_id.clone(),
        }
    }

    fn release_internal(&mut self) {
        if self.released {
            return;
        }
        let path = self.ledger_path.clone();
        let lease_id = self.lease_id.clone();
        let _ = with_locked_ledger(&path, |ledger| {
            ledger.leases.retain(|lease| lease.id != lease_id);
            Ok::<(), UnusableHost>(())
        });
        self.released = true;
    }
}

impl Drop for HostMemoryLease {
    fn drop(&mut self) {
        self.release_internal();
    }
}

/// Settles one lease's reservation ([`HostMemoryLease::settler`]) from a
/// thread other than its owner's. Best effort, like
/// [`HostMemoryLease::set_worker_pid`]: a ledger that cannot be rewritten
/// keeps the larger reservation, which over-reserves and never
/// under-reserves. Blocking file I/O.
#[derive(Debug)]
pub struct LeaseSettler {
    ledger_path: PathBuf,
    lease_id: String,
}

impl LeaseSettler {
    /// Rewrite the lease's reservation to `reserved`. A lease already gone
    /// (released, or pruned with its dead worker) has nothing to settle.
    pub fn settle(self, reserved: MemoryMb) -> Result<(), HostMemoryError> {
        let Self {
            ledger_path,
            lease_id,
        } = self;
        with_locked_ledger(&ledger_path, move |ledger: &mut MemoryLedger| {
            if let Some(lease) = ledger.leases.iter_mut().find(|lease| lease.id == lease_id) {
                lease.reserved_mb = reserved;
            }
            Ok(())
        })
    }
}

/// A machine-local exclusive lock for real-model ML tests.
pub struct MachineMlTestLock {
    _lease: HostMemoryLease,
}

impl MachineMlTestLock {
    /// Acquire the machine-wide ML test lock, waiting until other local test
    /// binaries release it.
    pub fn acquire(label: &str) -> Result<Self, BoundedRefusal> {
        // Live: ML tests load real models, so the lock is taken on the
        // machine that has to hold them.
        let coordinator = HostMemoryCoordinator::new(
            HostMemoryRuntimeConfig::from_server_config(&ServerConfig::default()),
            HostReadings::Live,
        );
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
    /// The host or the ledger cannot be used for any decision.
    #[error(transparent)]
    Unusable(#[from] UnusableHost),
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
    /// The request can never fit this host ([`NeverFits`]).
    #[error(transparent)]
    NeverFits(NeverFits),
}

/// A request that, with the models its own worker keeps resident, is more
/// than this host can ever offer with its reserve kept free. Those models
/// cannot be freed while it waits, so no wait fits it. One record, carried
/// by every type that reports it, so the numbers and their sentence exist
/// once.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "{label} needs {needed_mb} MB of host memory; this host can ever offer {ceiling_mb} MB \
     with {reserve_mb} MB kept free"
)]
pub struct NeverFits {
    /// Human-readable request label.
    pub label: String,
    /// The request plus the models its own worker keeps resident.
    pub needed_mb: u64,
    /// The host's ceiling ([`HostCeiling`]).
    pub ceiling_mb: u64,
    /// The reserve kept free.
    pub reserve_mb: u64,
}

/// Why a queued wait stopped without a grant: no waiting could change it.
/// A closed type of its own, so a congestion (a full host, a busy startup
/// slot, a held test lock) cannot be reported as a stop, and a queued wait,
/// which has no deadline, cannot report a timeout.
#[derive(Debug, thiserror::Error)]
pub enum FinalRefusal {
    /// The request can never fit this host.
    #[error(transparent)]
    NeverFits(NeverFits),
    /// The host or the ledger cannot be read or written.
    #[error(transparent)]
    Unusable(UnusableHost),
}

/// A bounded wait (a real-model test lock, a job execution plan) that ran out
/// of time while its refusal could still have cleared.
#[derive(Debug, thiserror::Error)]
#[error("timed out waiting for host-memory capacity for {label} after {waited_s}s: {last_reason}")]
pub struct WaitExpired {
    /// Human-readable request label.
    pub label: String,
    /// Timeout window in seconds.
    pub waited_s: u64,
    /// Last congestion observed while waiting.
    pub last_reason: String,
}

/// Why a bounded wait stopped without a grant.
#[derive(Debug, thiserror::Error)]
pub enum BoundedRefusal {
    /// A refusal no waiting could change.
    #[error(transparent)]
    Final(FinalRefusal),
    /// The wait's deadline passed while it was congested.
    #[error(transparent)]
    TimedOut(WaitExpired),
}

impl From<HostUnreadable> for HostMemoryError {
    fn from(unreadable: HostUnreadable) -> Self {
        Self::Unusable(UnusableHost::Unreadable(unreadable))
    }
}

/// A host or ledger the coordinator cannot use for any decision.
#[derive(Debug, thiserror::Error)]
pub enum UnusableHost {
    /// The operating system would not report the host's memory.
    #[error(transparent)]
    Unreadable(#[from] HostUnreadable),
    /// The ledger could not be read or written.
    #[error("host-memory ledger I/O failed at {path}: {source}")]
    Io {
        /// Path to the shared ledger file.
        path: PathBuf,
        /// Underlying filesystem error.
        #[source]
        source: std::io::Error,
    },
    /// The ledger exists but cannot be parsed.
    #[error("host-memory ledger is corrupt at {path}: {message}")]
    CorruptLedger {
        /// Path to the shared ledger file.
        path: PathBuf,
        /// Human-readable parse failure.
        message: String,
    },
}

/// What a lease reserves memory for.
///
/// A new kind breaks every exhaustive match below until its author says
/// whether it holds a startup slot and which side of an owner's demand it
/// counts on.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum MemoryLeaseKind {
    /// A worker process starting: its baseline, until it reports ready.
    WorkerStartup,
    /// A job's file parallelism, for the job's life.
    JobExecution,
    /// The machine-wide real-model test lock (no memory of its own).
    MlTestExclusive,
    /// One model a worker loads on demand (`ensure_task`), for as long as
    /// the worker holds it: in full while it loads, then the activation
    /// margin it has yet to use.
    ModelLoad,
    /// One batch of a model a worker builds per batch (coreference), for the
    /// batch.
    ModelBatch,
    /// Request-shaped workspace of an already loaded model, for the request.
    ModelRequest,
}

impl MemoryLeaseKind {
    /// Whether a lease of this kind occupies one of the host's worker-startup
    /// slots. Only a worker startup does; it is a fact about the kind, not a
    /// second field to keep in agreement with it. A model load does not: its
    /// memory is reserved by its own lease, and a slot held through a first
    /// download would stall every other startup on the host for its length.
    fn holds_startup_slot(self) -> bool {
        match self {
            Self::WorkerStartup => true,
            Self::JobExecution
            | Self::MlTestExclusive
            | Self::ModelLoad
            | Self::ModelBatch
            | Self::ModelRequest => false,
        }
    }

    /// Whether a lease of this kind stays held for as long as its worker
    /// waits for more: a loaded model is resident until the worker lets it
    /// go, while a startup ends at ready and a batch or request lease ends
    /// with its own work. Only these count against a request as memory that
    /// can never be freed for it ([`HostMemoryError::NeverFits`]).
    fn held_while_worker_waits(self) -> bool {
        match self {
            Self::ModelLoad => true,
            Self::WorkerStartup
            | Self::JobExecution
            | Self::MlTestExclusive
            | Self::ModelBatch
            | Self::ModelRequest => false,
        }
    }

    /// Which half of an owner's demand a lease of this kind counts on
    /// ([`effective_reserved_mb`]).
    fn demand_side(self) -> DemandSide {
        match self {
            Self::JobExecution | Self::MlTestExclusive => DemandSide::Job,
            Self::WorkerStartup | Self::ModelLoad | Self::ModelBatch | Self::ModelRequest => {
                DemandSide::Worker
            }
        }
    }
}

/// The two halves of one owner's demand: what its jobs reserved, and what its
/// workers reserved for themselves (their startups and the models they load).
/// A job's reservation covers the workers serving it, so a worker's own
/// leases are sub-allocations of it (Contract D) and the owner's demand is
/// the larger half, never the sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DemandSide {
    Job,
    Worker,
}

/// The ledger's time source: the host wall clock, read here and nowhere else
/// in this module.
///
/// DECISION (2026-10-01), deliberately NOT the store's injected `Clock`: the
/// ledger is one file shared by every batchalign3 process on the host
/// (daemons, test servers, `--no-server` runs), and a lease's age is judged
/// by whichever process next opens it. Each process has its own store clock,
/// so an injected clock would let one process's test clock decide whether
/// another process's lease is stale. Wall time is the one clock they share.
fn ledger_now() -> MachineTime {
    MachineTime::now()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "MemoryLeaseRecordWire", into = "MemoryLeaseRecordWire")]
struct MemoryLeaseRecord {
    id: String,
    kind: MemoryLeaseKind,
    /// The process that requested this lease (the daemon, in production).
    /// Liveness-checked when `worker_pid` is `None`.
    owner_pid: WorkerPid,
    /// Optional PID of the worker subprocess whose memory the lease
    /// represents. When `Some`, prune_stale_leases checks THIS PID's
    /// liveness instead of `owner_pid`. The daemon spawns workers in
    /// their own process group, so a worker can die independently of
    /// the daemon: Bug 2 (ghost slots) was the daemon's reservation
    /// surviving the worker's death because the lease was tagged with
    /// the (always-alive) daemon PID. Tagging with the worker's PID
    /// instead lets the reaper detect the dead worker and reclaim the
    /// slot. `#[serde(default)]` keeps backward compatibility with
    /// older ledger files that predate this field.
    worker_pid: Option<WorkerPid>,
    reserved_mb: MemoryMb,
    label: String,
    /// When the lease was taken, by [`ledger_now`].
    created_at: MachineTime,
}

/// The ledger's file form of one lease, unchanged across builds so daemons of
/// different builds on one host keep reading each other's ledger.
///
/// `startup_slot` is still WRITTEN, from the kind, because an older build
/// counts slots by reading it; on read it is ignored, since the kind already
/// says it (every build has written exactly `kind == worker_startup`). It
/// used to be a field of the record as well, so a job-execution lease holding
/// a startup slot was representable. `worker_pid` defaults for ledgers that
/// predate it; times are whole Unix seconds under `created_at_epoch_s`.
#[derive(Serialize, Deserialize)]
struct MemoryLeaseRecordWire {
    id: String,
    kind: MemoryLeaseKind,
    owner_pid: WorkerPid,
    #[serde(default)]
    worker_pid: Option<WorkerPid>,
    reserved_mb: MemoryMb,
    startup_slot: bool,
    label: String,
    #[serde(rename = "created_at_epoch_s", with = "ledger_seconds")]
    created_at: MachineTime,
}

impl From<MemoryLeaseRecordWire> for MemoryLeaseRecord {
    fn from(wire: MemoryLeaseRecordWire) -> Self {
        Self {
            id: wire.id,
            kind: wire.kind,
            owner_pid: wire.owner_pid,
            worker_pid: wire.worker_pid,
            reserved_mb: wire.reserved_mb,
            label: wire.label,
            created_at: wire.created_at,
        }
    }
}

impl From<MemoryLeaseRecord> for MemoryLeaseRecordWire {
    fn from(record: MemoryLeaseRecord) -> Self {
        Self {
            startup_slot: record.kind.holds_startup_slot(),
            id: record.id,
            kind: record.kind,
            owner_pid: record.owner_pid,
            worker_pid: record.worker_pid,
            reserved_mb: record.reserved_mb,
            label: record.label,
            created_at: record.created_at,
        }
    }
}

/// One lease's contribution to the reservation arithmetic.
#[derive(Debug, Clone, Copy)]
struct Reservation {
    owner: WorkerPid,
    kind: MemoryLeaseKind,
    reserved: MemoryMb,
}

impl MemoryLeaseRecord {
    fn reservation(&self) -> Reservation {
        Reservation {
            owner: self.owner_pid,
            kind: self.kind,
            reserved: self.reserved_mb,
        }
    }
}

/// The ledger's time form: whole Unix seconds, as every build has written
/// it. A value that names no instant fails the read, which reports the
/// ledger corrupt rather than inventing a time.
mod ledger_seconds {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use crate::api::MachineTime;

    pub(super) fn serialize<S: Serializer>(
        at: &MachineTime,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        at.timestamp().as_second().serialize(serializer)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<MachineTime, D::Error> {
        let seconds = i64::deserialize(deserializer)?;
        jiff::Timestamp::from_second(seconds)
            .map(MachineTime::from_timestamp)
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct MemoryLedger {
    leases: Vec<MemoryLeaseRecord>,
}

/// Per-owner `max(sum(job side), sum(worker side))`, summed over owners
/// ([`DemandSide`]). A worker's startup and model leases are sub-allocations
/// of its owner's job execution leases (Contract D), so the larger of the
/// two surfaces the owner's actual pending demand. `MlTestExclusive`
/// carries 0 MB and folds into the job half.
///
/// Takes the reservations rather than the records, so admission can project
/// the total WITH a requested lease by chaining it on, instead of building a
/// placeholder record with an empty id and label to push onto a copy.
fn effective_reserved_mb(reservations: impl IntoIterator<Item = Reservation>) -> u64 {
    use std::collections::HashMap;
    #[derive(Default)]
    struct OwnerDemand {
        job: u64,
        worker: u64,
    }
    let mut by_owner: HashMap<WorkerPid, OwnerDemand> = HashMap::new();
    for reservation in reservations {
        let demand = by_owner.entry(reservation.owner).or_default();
        let side = match reservation.kind.demand_side() {
            DemandSide::Job => &mut demand.job,
            DemandSide::Worker => &mut demand.worker,
        };
        *side = side.saturating_add(reservation.reserved.0);
    }
    by_owner
        .values()
        .map(|demand| demand.job.max(demand.worker))
        .sum()
}

#[derive(Debug, Clone)]
struct MemoryLeaseRequest {
    kind: MemoryLeaseKind,
    reserved_mb: MemoryMb,
    label: String,
    /// The worker process the reservation is for, when it already exists (a
    /// model lease names its worker from the start, so it is reclaimed when
    /// that worker dies and is never mistaken for a pre-spawn intent).
    worker_pid: Option<WorkerPid>,
}

/// One reservation a worker asks the host for, ready to queue
/// ([`HostMemoryCoordinator::queue_for_lease`]).
///
/// Built only by the constructors below, one per kind of worker reservation,
/// so a queued wait cannot be handed a job's or a test lock's request.
#[derive(Debug, Clone)]
pub struct WorkerLeaseRequest(MemoryLeaseRequest);

impl WorkerLeaseRequest {
    /// A worker process starting, before it exists: its baseline
    /// ([`crate::worker::residency::WORKER_BASELINE`]), the same for every
    /// worker since none loads a model at startup.
    pub fn worker_startup(
        profile: WorkerProfile,
        lang: &WorkerLanguage,
        engine_overrides: &str,
    ) -> Self {
        Self(MemoryLeaseRequest {
            kind: MemoryLeaseKind::WorkerStartup,
            reserved_mb: crate::worker::residency::WORKER_BASELINE.reserved(),
            label: format!(
                "worker-startup:{}:{}:{}",
                profile.label(),
                lang,
                engine_overrides
            ),
            worker_pid: None,
        })
    }

    /// One model `worker` is about to load, `what` naming it (task and model).
    pub fn model_load(worker: WorkerPid, what: &str, reserved: MemoryMb) -> Self {
        Self(MemoryLeaseRequest {
            kind: MemoryLeaseKind::ModelLoad,
            reserved_mb: reserved,
            label: format!("model-load:{what}:pid {worker}"),
            worker_pid: Some(worker),
        })
    }

    /// One batch of a per-batch model `worker` is about to build.
    pub fn model_batch(worker: WorkerPid, what: &str, reserved: MemoryMb) -> Self {
        Self(MemoryLeaseRequest {
            kind: MemoryLeaseKind::ModelBatch,
            reserved_mb: reserved,
            label: format!("model-batch:{what}:pid {worker}"),
            worker_pid: Some(worker),
        })
    }

    /// Additional workspace of one prepared request on a resident model.
    pub fn model_request(worker: WorkerPid, what: &str, reserved: MemoryMb) -> Self {
        Self(MemoryLeaseRequest {
            kind: MemoryLeaseKind::ModelRequest,
            reserved_mb: reserved,
            label: format!("model-request:{what}:pid {worker}"),
            worker_pid: Some(worker),
        })
    }

    /// The ledger label, which names what the lease is for.
    pub fn label(&self) -> &str {
        &self.0.label
    }
}

/// Told when a queued lease wait starts and stops waiting, so whoever the
/// wait serves (a file's progress) can say it is waiting for host memory.
pub trait QueuedLeaseObserver: Send + Sync {
    /// The request could not be granted now and is queued.
    fn queued(&self);
    /// The queued request was granted, refused or dropped.
    fn dequeued(&self);
}

/// What one attempt to make room for a queued request did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reclaimed {
    /// An idle worker was retired, releasing its leases and its memory: the
    /// request is tried again at once.
    Retired,
    /// Nothing idle could be retired: the request waits for a lease to be
    /// released some other way.
    NothingIdle,
}

/// Frees host memory a queued request waits for, by retiring an idle worker.
///
/// A queued request asks for room directly rather than waiting for idle
/// workers to leave on their own: the worker pool retires its largest idle
/// worker, and a process with no pool ([`NoIdleWorkers`]) has nothing to
/// retire. Called after each refusal, so a request that needs several
/// workers' memory retires them one by one, and stops retiring once granted.
#[async_trait::async_trait]
pub trait MemoryReclaim: Send + Sync {
    /// Retire one idle worker, if there is one.
    async fn reclaim_one(&self) -> Reclaimed;
}

/// The reclaim of a process that owns no worker pool (a one-off command):
/// it has no idle workers to retire.
#[derive(Debug, Clone, Copy)]
pub struct NoIdleWorkers;

#[async_trait::async_trait]
impl MemoryReclaim for NoIdleWorkers {
    async fn reclaim_one(&self) -> Reclaimed {
        Reclaimed::NothingIdle
    }
}

/// An open queued wait: tells its observer it ended, whichever way it ends.
struct QueuedWait<'a> {
    observer: &'a dyn QueuedLeaseObserver,
}

impl<'a> QueuedWait<'a> {
    fn begin(observer: &'a dyn QueuedLeaseObserver) -> Self {
        observer.queued();
        Self { observer }
    }
}

impl Drop for QueuedWait<'_> {
    fn drop(&mut self) {
        self.observer.dequeued();
    }
}

/// One reading of this host's RAM: both figures from the same refresh, so a
/// caller that needs both never pairs two readings taken apart.
///
/// "Available" follows the sysinfo definition (`free + reclaimable`), which
/// on macOS undercounts against Activity Monitor's "memory pressure".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemMemory {
    /// Total physical RAM.
    pub total_mb: MemoryMb,
    /// RAM the operating system reports as available now.
    pub available_mb: MemoryMb,
}

impl SystemMemory {
    /// RAM in use: total less available.
    pub fn used_mb(&self) -> MemoryMb {
        MemoryMb(self.total_mb.0.saturating_sub(self.available_mb.0))
    }
}

/// Machine-local coordinator that serializes access to the shared memory ledger.
#[derive(Debug, Clone)]
pub struct HostMemoryCoordinator {
    config: HostMemoryRuntimeConfig,
    readings: HostReadings,
}

impl HostMemoryCoordinator {
    /// A coordinator for the ledger `config` names, deciding on `readings`.
    ///
    /// A server builds it from its host snapshot
    /// (`HostSnapshot::memory_coordinator`), so the ledger reads the same
    /// host its tier and admission gates describe.
    pub fn new(config: HostMemoryRuntimeConfig, readings: HostReadings) -> Self {
        Self { config, readings }
    }

    /// Return the shared ledger path backing this coordinator.
    pub fn coordinator_path(&self) -> &Path {
        &self.config.coordinator_path
    }

    /// Return the current host-memory snapshot for health/status reporting.
    pub fn snapshot(&self) -> Result<HostMemorySnapshot, UnusableHost> {
        with_locked_ledger(&self.config.coordinator_path, |ledger| {
            let system = self.readings.memory()?;
            let active_reserved_mb =
                effective_reserved_mb(ledger.leases.iter().map(MemoryLeaseRecord::reservation));
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
                .map(|lease| format!("{:?}:{}:{}MB", lease.kind, lease.label, lease.reserved_mb.0))
                .collect();
            Ok(HostMemorySnapshot {
                total_mb: system.total_mb,
                ceiling: self.readings.ceiling_of(system),
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

    /// Wait for a worker reservation (a startup or a model), queued behind
    /// the leases the host holds, with no deadline.
    ///
    /// The wait ends when the lease is granted; when the request is refused
    /// for good (larger than the machine, an unreadable host, a ledger that
    /// cannot be used: [`QueuedLeaseFailure::Refused`]); or when the future
    /// is dropped, which is how a cancelled job stops waiting. While queued
    /// it says so: one log line when it queues, a warning every
    /// [`QUEUED_LEASE_REPORT_INTERVAL`] naming what it waits for, how long it
    /// has waited and the host's last refusal, and `observer` told when it
    /// starts and stops waiting. After each refusal it asks `reclaim` to
    /// retire one idle worker, and tries again at once when one was.
    ///
    /// Each attempt (which takes the ledger's file lock) runs alone on the
    /// blocking pool and the wait between attempts is a `tokio` sleep, so a
    /// dropped future ends the wait at once and never takes a lease nobody
    /// will use.
    pub async fn queue_for_lease(
        &self,
        request: WorkerLeaseRequest,
        observer: &dyn QueuedLeaseObserver,
        reclaim: &dyn MemoryReclaim,
    ) -> Result<HostMemoryLease, QueuedLeaseFailure> {
        self.queue_for_lease_every(
            request,
            observer,
            reclaim,
            QUEUED_LEASE_POLL,
            QUEUED_LEASE_REPORT_INTERVAL,
        )
        .await
    }

    /// [`Self::queue_for_lease`] with its poll and report intervals named, so
    /// a test can drive the wait in milliseconds.
    async fn queue_for_lease_every(
        &self,
        WorkerLeaseRequest(request): WorkerLeaseRequest,
        observer: &dyn QueuedLeaseObserver,
        reclaim: &dyn MemoryReclaim,
        poll: Duration,
        report_every: Duration,
    ) -> Result<HostMemoryLease, QueuedLeaseFailure> {
        let started = Instant::now();
        let mut queued: Option<QueuedWait<'_>> = None;
        let mut next_report = started + report_every;
        let mut last_reason = String::from("no capacity decision yet");
        loop {
            let coordinator = self.clone();
            let attempt = request.clone();
            let outcome =
                crate::blocking::spawn_in_span(move || coordinator.try_acquire_lease(attempt))
                    .await
                    .map_err(QueuedLeaseFailure::AttemptDidNotFinish)?;
            let error = match outcome {
                Ok(lease) => {
                    if queued.is_some() {
                        tracing::info!(
                            lease = %request.label,
                            waited_s = started.elapsed().as_secs(),
                            "Host memory granted to a queued request"
                        );
                    }
                    return Ok(lease);
                }
                Err(error) => error,
            };
            match after_queued_attempt(error, &mut last_reason) {
                LeaseStep::Retry => {
                    if queued.is_none() {
                        tracing::info!(
                            lease = %request.label,
                            reserved_mb = request.reserved_mb.0,
                            reason = %last_reason,
                            "Waiting for host memory: the request keeps its place and is \
                             granted when the host can fit it"
                        );
                        queued = Some(QueuedWait::begin(observer));
                    }
                    if Instant::now() >= next_report {
                        tracing::warn!(
                            lease = %request.label,
                            reserved_mb = request.reserved_mb.0,
                            waited_s = started.elapsed().as_secs(),
                            reason = %last_reason,
                            "Still waiting for host memory. The request keeps its place; it \
                             fails only if it can never fit or its job is cancelled"
                        );
                        next_report += report_every;
                    }
                    // A retired worker (the reclaim logs which) frees its
                    // leases at once, so the request is tried again
                    // without waiting out the poll.
                    match reclaim.reclaim_one().await {
                        Reclaimed::Retired => {}
                        Reclaimed::NothingIdle => tokio::time::sleep(poll).await,
                    }
                }
                LeaseStep::Stop(error) => return Err(QueuedLeaseFailure::Refused(error)),
            }
        }
    }

    /// Acquire a machine-wide exclusive lock for real-model ML tests.
    pub fn acquire_ml_test_lock(
        &self,
        label: &str,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<HostMemoryLease, BoundedRefusal> {
        let request = MemoryLeaseRequest {
            kind: MemoryLeaseKind::MlTestExclusive,
            reserved_mb: MemoryMb(0),
            label: label.to_owned(),
            worker_pid: None,
        };
        self.wait_for_lease(request, timeout, poll_interval)
    }

    /// Plan and reserve one job's execution memory window.
    ///
    /// The coordinator may reduce the requested worker count when host pressure
    /// is high. The returned lease stays alive for the whole job so other local
    /// processes see the reservation and conservatively back off.
    pub async fn wait_for_job_execution_plan(
        &self,
        command: ReleasedCommand,
        requested_workers: NumWorkers,
        label: &str,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<JobExecutionPlan, PlanFailure> {
        let wait = BoundedWait::new(timeout);
        let mut last_reason = String::from("no capacity decision yet");
        loop {
            let coordinator = self.clone();
            let attempt_label = label.to_owned();
            let attempt = crate::blocking::spawn_in_span(move || {
                coordinator.try_plan_job_execution(command, requested_workers, &attempt_label)
            })
            .await
            .map_err(LeaseFailure::AttemptDidNotFinish)?;
            match attempt {
                Ok(plan) => return Ok(plan),
                Err(error) => match after_bounded_attempt(error, label, &wait, &mut last_reason) {
                    // Dropping the admission future stops polling. At most
                    // one ledger transaction remains; it cannot start a
                    // worker, and an unreceived plan releases its lease.
                    LeaseStep::Retry => tokio::time::sleep(poll_interval).await,
                    LeaseStep::Stop(error) => return Err(LeaseFailure::Refused(error)),
                },
            }
        }
    }

    fn try_plan_job_execution(
        &self,
        command: ReleasedCommand,
        requested_workers: NumWorkers,
        label: &str,
    ) -> Result<JobExecutionPlan, HostMemoryError> {
        with_locked_ledger(&self.config.coordinator_path, |ledger| {
            let system = self.readings.memory()?;
            // Tier-aware envelope: clamp the fleet-conservative
            // worst-case to this host's per-profile startup
            // reservation. Without the clamp, a 16 GB host that
            // asks for one morphotag worker reserves a 12 GB
            // envelope, the worker-startup lease check refuses
            // every spawn, and the pool stays dead-on-arrival.
            // memory_gate.rs admission and memory_guard.rs
            // in-spawn remain the safety net for actual at-spawn
            // OOM. See `batchalign_types::command_spec::CommandSpec`
            // and `batchalign_types::memory::estimate_per_worker_peak_mb_with_profile`.
            let tier = runtime::MemoryTier::from_total_mb(system.total_mb.0);
            let spec = batchalign_types::command_spec::command_spec_for(command);
            let worst_case = spec.execution_budget_mb(runtime::python_runtime());
            let per_worker_budget = match spec.profile {
                Some(profile) => {
                    batchalign_types::memory::estimate_per_worker_peak_mb_with_profile(
                        worst_case, profile, &tier,
                    )
                }
                // Native encoder slots still reserve host memory, but there
                // is no fictitious model-startup envelope to clamp against.
                None => worst_case,
            };
            // One worker is the least a job can run with. It can never fit
            // exactly when the planner itself would refuse one worker on
            // this host idle at its ceiling: the planner's own reserve
            // policy, not a second formula that could refuse for good a job
            // the planner would grant.
            let ceiling = self.readings.ceiling_of(system).mb().0;
            if plan_job_reservation(1, per_worker_budget.0, ceiling, self.config.reserve_mb.0, 0)
                .is_none()
            {
                return Err(HostMemoryError::NeverFits(NeverFits {
                    label: label.to_owned(),
                    needed_mb: per_worker_budget.0,
                    ceiling_mb: ceiling,
                    reserve_mb: effective_reserve_mb(self.config.reserve_mb.0, ceiling),
                }));
            }
            let pending_reserved_mb =
                effective_reserved_mb(ledger.leases.iter().map(MemoryLeaseRecord::reservation));
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
                owner_pid: WorkerPid(std::process::id()),
                worker_pid: None,
                reserved_mb: MemoryMb(reserved_mb),
                label: label.to_owned(),
                created_at: ledger_now(),
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
    ) -> Result<HostMemoryLease, BoundedRefusal> {
        let wait = BoundedWait::new(timeout);
        let mut last_reason = String::from("no capacity decision yet");
        loop {
            match self.try_acquire_lease(request.clone()) {
                Ok(lease) => return Ok(lease),
                Err(error) => {
                    match after_bounded_attempt(error, &request.label, &wait, &mut last_reason) {
                        LeaseStep::Retry => std::thread::sleep(poll_interval),
                        LeaseStep::Stop(error) => return Err(error),
                    }
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

            if request.kind.holds_startup_slot() {
                let active_slots = ledger
                    .leases
                    .iter()
                    .filter(|lease| lease.kind.holds_startup_slot())
                    .count();
                if active_slots >= self.config.max_concurrent_worker_startups.get() {
                    return Err(HostMemoryError::StartupSlotsBusy {
                        label: request.label.clone(),
                        active_slots,
                        max_slots: self.config.max_concurrent_worker_startups.get(),
                    });
                }
            }

            let system = self.readings.memory()?;
            // What this request can never be granted without: itself and the
            // models its own worker keeps resident, which cannot be freed
            // while it waits. A wait for more than the host can ever offer
            // would never end, so it is refused here, with the numbers. A
            // request that reserves nothing (the test lock) asks for no
            // memory, so it cannot be too large.
            let needed_mb = request.reserved_mb.0.saturating_add(
                request
                    .worker_pid
                    .map(|worker| {
                        ledger
                            .leases
                            .iter()
                            .filter(|lease| {
                                lease.worker_pid == Some(worker)
                                    && lease.kind.held_while_worker_waits()
                            })
                            .map(|lease| lease.reserved_mb.0)
                            .sum::<u64>()
                    })
                    .unwrap_or(0),
            );
            let ceiling = self.readings.ceiling_of(system);
            if request.reserved_mb.0 > 0
                && !ceiling.admits(MemoryMb(needed_mb), self.config.reserve_mb)
            {
                return Err(HostMemoryError::NeverFits(NeverFits {
                    label: request.label.clone(),
                    needed_mb,
                    ceiling_mb: ceiling.mb().0,
                    reserve_mb: self.config.reserve_mb.0,
                }));
            }
            let pending_reserved_mb =
                effective_reserved_mb(ledger.leases.iter().map(MemoryLeaseRecord::reservation));
            // What `effective_reserved_mb` would be AFTER admitting the
            // request: its reservation is chained onto the ledger's, so
            // Contract D semantics (sub-allocation) apply to it uniformly.
            let projected_reserved_mb = effective_reserved_mb(
                ledger
                    .leases
                    .iter()
                    .map(MemoryLeaseRecord::reservation)
                    .chain(std::iter::once(Reservation {
                        owner: WorkerPid(std::process::id()),
                        kind: request.kind,
                        reserved: request.reserved_mb,
                    })),
            );
            // A request that adds nothing to the projected total is a
            // sub-allocation of a reservation this owner already holds (a
            // model load inside its job's lease, Contract D): the memory was
            // promised when that lease was granted, so it is admitted even if
            // the host now reads as over-committed. Refusing it would make a
            // job wait on its own reservation, which a queued wait (no
            // deadline) would do for ever.
            let adds_demand = projected_reserved_mb > pending_reserved_mb;
            let projected_available_mb =
                system.available_mb.0.saturating_sub(projected_reserved_mb);
            if adds_demand && projected_available_mb < self.config.reserve_mb.0 {
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
                owner_pid: WorkerPid(std::process::id()),
                worker_pid: request.worker_pid,
                reserved_mb: request.reserved_mb,
                label: request.label,
                created_at: ledger_now(),
            };
            let lease =
                HostMemoryLease::new(self.config.coordinator_path.clone(), record.id.clone());
            ledger.leases.push(record);
            Ok(lease)
        })
    }
}

fn with_locked_ledger<T, E: From<UnusableHost>>(
    path: &Path,
    action: impl FnOnce(&mut MemoryLedger) -> Result<T, E>,
) -> Result<T, E> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|source| UnusableHost::Io {
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
        .map_err(|source| UnusableHost::Io {
            path: path.to_path_buf(),
            source,
        })?;

    file.lock().map_err(|source| UnusableHost::Io {
        path: path.to_path_buf(),
        source,
    })?;

    let mut raw = String::new();
    file.read_to_string(&mut raw)
        .map_err(|source| UnusableHost::Io {
            path: path.to_path_buf(),
            source,
        })?;

    let mut ledger = if raw.trim().is_empty() {
        MemoryLedger::default()
    } else {
        serde_json::from_str::<MemoryLedger>(&raw).map_err(|error| UnusableHost::CorruptLedger {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?
    };

    let pruned = prune_stale_leases(&mut ledger).map_err(UnusableHost::from)?;

    let result = action(&mut ledger);
    // An action that refuses changes nothing (every refusal returns before
    // its action writes), so a refusal with nothing pruned leaves the file
    // as it was: no rewrite, no sync. A queued wait refused every second
    // would otherwise rewrite the shared ledger every second for as long as
    // it waits.
    if let (Err(_), Pruned::Nothing) = (&result, pruned) {
        return result;
    }

    file.set_len(0).map_err(|source| UnusableHost::Io {
        path: path.to_path_buf(),
        source,
    })?;
    file.seek(SeekFrom::Start(0))
        .map_err(|source| UnusableHost::Io {
            path: path.to_path_buf(),
            source,
        })?;
    serde_json::to_writer_pretty(&mut file, &ledger).map_err(|error| {
        UnusableHost::CorruptLedger {
            path: path.to_path_buf(),
            message: error.to_string(),
        }
    })?;
    file.write_all(b"\n").map_err(|source| UnusableHost::Io {
        path: path.to_path_buf(),
        source,
    })?;
    file.sync_all().map_err(|source| UnusableHost::Io {
        path: path.to_path_buf(),
        source,
    })?;

    result
}

/// Pre-spawn intent leases (those with `worker_pid: None` whose owner
/// is the live daemon) must auto-expire if the spawn never
/// materialises. Without this, a failed-spawn or daemon-side bookkeeping
/// glitch produces a phantom hold that consumes budget indefinitely.
///
/// 30 minutes is generous: a worker starts with no model (each model's
/// lease names its worker from the start, so none of them is an intent),
/// and a starting worker may go without advancing for
/// `worker_startup_stall_s` (default 300 s) before it is retired. A
/// pre-spawn intent older than this window is by construction stuck.
///
/// Operational evidence (2026-05-01 afternoon wedge): 3 admitted jobs
/// each held ~76 GB JobExecution leases with `worker_pid: None`; no
/// workers ever spawned for them; the 228 GB sat as phantom holds for
/// hours. Without a deadline, every "failed admission cycle" leaks
/// budget.
const PRE_SPAWN_INTENT_DEADLINE: Duration = Duration::from_secs(1_800);

/// The operating system would not report this host's memory or processes.
///
/// Until sysinfo 0.40 this was not an error at all: an unsupported platform
/// or a failed read gave zero total memory, zero available memory and no
/// processes, which admission, tiering and lease pruning then read as facts
/// (a "Small" host, no headroom, every lease owner dead). Every memory and
/// process reading now goes through [`open_system`], the one place it can fail.
#[derive(Debug, Clone, thiserror::Error)]
#[error("cannot read this host's memory or processes from the operating system: {0}")]
pub struct HostUnreadable(#[source] std::sync::Arc<sysinfo::Error>);

/// The one way to open a sysinfo handle, so no memory or process reading can
/// mistake an unreadable host for an empty one. (CPU count and GPU detection
/// in `host_facts` do not use sysinfo and report their own failures.)
pub(crate) fn open_system() -> Result<System, HostUnreadable> {
    System::new().map_err(|error| HostUnreadable(std::sync::Arc::new(error)))
}

/// Whether a lease's process still exists, as the operating system reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Liveness {
    Alive,
    Dead,
}

/// Drop leases whose owners are gone or whose pre-spawn intent expired.
///
/// One process table read for the whole pass. If the table cannot be read the
/// pass fails and no lease is reclaimed: "could not tell" is not "dead", and
/// reclaiming a live worker's reservation is how a host gets oversubscribed.
///
/// An empty ledger has nothing to judge and reads nothing. Otherwise only the
/// process each lease is bound to is refreshed (its worker when it names one,
/// else its owner, the same choice the retain below makes), once per PID.
fn prune_stale_leases(ledger: &mut MemoryLedger) -> Result<Pruned, HostUnreadable> {
    if ledger.leases.is_empty() {
        return Ok(Pruned::Nothing);
    }
    let before = ledger.leases.len();
    let mut system = open_system()?;
    let mut pids: Vec<Pid> = ledger
        .leases
        .iter()
        .map(|lease| Pid::from_u32(lease.worker_pid.unwrap_or(lease.owner_pid).0))
        .collect();
    pids.sort_unstable();
    pids.dedup();
    system.refresh_processes(ProcessesToUpdate::Some(&pids), false);
    let liveness = |pid: WorkerPid| match system.process(Pid::from_u32(pid.0)) {
        Some(_) => Liveness::Alive,
        None => Liveness::Dead,
    };
    let now = ledger_now();
    ledger.leases.retain(|lease| {
        // Bug 2 fix (Contract B): when a lease names a specific worker
        // subprocess, its survival is bound to that worker, NOT to
        // the daemon. Workers run in their own process group
        // (`setpgid(0, 0)`) and can die independently, OOM, crash,
        // external kill, without the daemon noticing. Tagging the
        // lease with the worker's PID and checking THAT PID's
        // liveness is what closes the ghost-slot accounting drift
        // that wedged a development host (the historical "Bug 2" incident).
        if let Some(worker_pid) = lease.worker_pid {
            return liveness(worker_pid) == Liveness::Alive;
        }

        // No worker_pid: this is a pre-spawn intent or daemon-side
        // bookkeeping lease. First check the owner's liveness, a dead
        // daemon's leases are reclaimed at first ledger access (Contract
        // F, restart hygiene).
        if liveness(lease.owner_pid) == Liveness::Dead {
            return false;
        }

        // Owner is alive but no worker_pid is bound. Apply the
        // pre-spawn intent deadline (Contract E): if the lease is
        // older than the deadline window, the spawn it was reserving
        // for didn't materialise: reclaim the budget. A live owner
        // that legitimately holds a long-lived bookkeeping lease can
        // refresh `created_at` on its own cadence; this prune
        // protects against the daemon-side leak path, not against
        // healthy long-lived state.
        now.saturating_duration_since(lease.created_at) < PRE_SPAWN_INTENT_DEADLINE
    });
    Ok(match ledger.leases.len() == before {
        true => Pruned::Nothing,
        false => Pruned::Some,
    })
}

/// Whether a prune pass removed any lease, so the ledger must be written back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pruned {
    Nothing,
    Some,
}

/// TTL the cached `SystemMemory` is good for. Admission gates
/// and the eviction pre-pass both poll memory; with a 1 s TTL the
/// kernel-side `/proc/meminfo` (Linux) / `host_statistics64` (macOS)
/// is read at most once per second across all callers, regardless
/// of how many spawn admissions or health-check ticks land in
/// between.
const SNAPSHOT_TTL: Duration = Duration::from_millis(1000);

/// What the process-global memory cache holds.
///
/// `Empty` until the first successful read, rather than a reading taken
/// eagerly at first touch: a first read that fails has nothing true to store.
#[derive(Clone, Copy)]
enum CachedSnapshot {
    Empty,
    Read(SystemMemory, Instant),
}

/// Process-global TTL-cached memory reading, served by
/// [`read_system_memory`].
static SNAPSHOT_CACHE: LazyLock<Mutex<CachedSnapshot>> =
    LazyLock::new(|| Mutex::new(CachedSnapshot::Empty));

fn refresh_read_system_memory() -> Result<SystemMemory, HostUnreadable> {
    let mut system = open_system()?;
    system.refresh_memory();
    Ok(SystemMemory {
        total_mb: MemoryMb(system.total_memory() / (1024 * 1024)),
        available_mb: MemoryMb(system.available_memory() / (1024 * 1024)),
    })
}

/// This host's RAM: the one place it is read. The reading is cached
/// process-wide for [`SNAPSHOT_TTL`], so the admission gate, pressure
/// eviction and the spawn guard decide from the same figures rather than
/// each constructing a `sysinfo::System`. A failed refresh is the caller's error: a reading older than the TTL
/// is not served in its place, because admission decides from it.
pub fn read_system_memory() -> Result<SystemMemory, HostUnreadable> {
    // Recover from a poisoned mutex, the snapshot has no invariant
    // that a panicking holder could leave broken; it's just a
    // stale-cache pair.
    let mut state = SNAPSHOT_CACHE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match *state {
        CachedSnapshot::Read(snapshot, read_at) if read_at.elapsed() < SNAPSHOT_TTL => Ok(snapshot),
        CachedSnapshot::Read(..) | CachedSnapshot::Empty => {
            let snapshot = refresh_read_system_memory()?;
            *state = CachedSnapshot::Read(snapshot, Instant::now());
            Ok(snapshot)
        }
    }
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

/// Cap the configured `reserve_mb` headroom at this fraction of the
/// host's available RAM so a static reserve sized for a 64 GB server
/// doesn't lock out admission on a 16 GB laptop.
const RESERVE_HEADROOM_FRACTION_DENOM: u64 = 4;

/// The headroom a job plan keeps free with `available_mb` available: the
/// configured reserve, capped at a quarter of what is available. The one
/// statement of the job reserve policy, for the planner and its never-fits
/// check alike.
fn effective_reserve_mb(reserve_mb: u64, available_mb: u64) -> u64 {
    reserve_mb.min(available_mb / RESERVE_HEADROOM_FRACTION_DENOM)
}

fn plan_job_reservation(
    requested_workers: usize,
    per_worker_budget_mb: u64,
    available_mb: u64,
    reserve_mb: u64,
    pending_reserved_mb: u64,
) -> Option<(usize, u64)> {
    let effective_reserve_mb = effective_reserve_mb(reserve_mb, available_mb);
    for workers in (1..=requested_workers).rev() {
        let requested_mb = per_worker_budget_mb.saturating_mul(workers as u64);
        let projected_available_mb = available_mb
            .saturating_sub(pending_reserved_mb)
            .saturating_sub(requested_mb);
        if projected_available_mb >= effective_reserve_mb {
            return Some((workers, requested_mb));
        }
    }
    None
}

/// What a failed lease attempt leads to: try again, or stop with `R`, the
/// stop type of the wait that is asking.
enum LeaseStep<R> {
    /// The refusal may clear (and a bounded wait is inside its window): wait
    /// and try again.
    Retry,
    /// Stop: no further waiting can change the outcome.
    Stop(R),
}

/// A wait with a deadline: real-model test locks and job execution plans. A
/// zero timeout is one attempt.
struct BoundedWait {
    timeout: Duration,
    deadline: Instant,
}

impl BoundedWait {
    /// A bounded wait of `timeout`, starting now.
    fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            deadline: Instant::now() + timeout,
        }
    }
}

/// The one classification of a failed attempt, by the refusal's own type:
/// final (no waiting changes it), or a congestion the caller may wait out.
fn final_or_congestion(error: HostMemoryError) -> Result<FinalRefusal, HostMemoryError> {
    match error {
        HostMemoryError::NeverFits(never) => Ok(FinalRefusal::NeverFits(never)),
        HostMemoryError::Unusable(unusable) => Ok(FinalRefusal::Unusable(unusable)),
        congestion @ (HostMemoryError::CapacityRejected { .. }
        | HostMemoryError::MlTestLockBusy { .. }
        | HostMemoryError::StartupSlotsBusy { .. }) => Err(congestion),
    }
}

/// A queued wait (worker startups and model loads) retries every congestion
/// for as long as it is asked to wait, and stops only on a final refusal.
fn after_queued_attempt(
    error: HostMemoryError,
    last_reason: &mut String,
) -> LeaseStep<FinalRefusal> {
    match final_or_congestion(error) {
        Ok(refusal) => LeaseStep::Stop(refusal),
        Err(congestion) => {
            *last_reason = congestion.to_string();
            LeaseStep::Retry
        }
    }
}

/// A bounded wait retries a congestion inside its window and then stops
/// with the last one it saw; a final refusal stops it at once.
fn after_bounded_attempt(
    error: HostMemoryError,
    label: &str,
    wait: &BoundedWait,
    last_reason: &mut String,
) -> LeaseStep<BoundedRefusal> {
    match final_or_congestion(error) {
        Ok(refusal) => LeaseStep::Stop(BoundedRefusal::Final(refusal)),
        Err(congestion) => {
            *last_reason = congestion.to_string();
            if !wait.timeout.is_zero() && Instant::now() < wait.deadline {
                LeaseStep::Retry
            } else {
                LeaseStep::Stop(BoundedRefusal::TimedOut(WaitExpired {
                    label: label.to_owned(),
                    waited_s: wait.timeout.as_secs(),
                    last_reason: std::mem::take(last_reason),
                }))
            }
        }
    }
}

/// Why an asynchronous wait returned no lease or plan: its wait's own
/// refusal `R`, or an attempt that never finished.
#[derive(Debug)]
pub enum LeaseFailure<R> {
    /// The host refused for good, as the wait's stop type says.
    Refused(R),
    /// The blocking task taking one attempt panicked or was cancelled.
    AttemptDidNotFinish(tokio::task::JoinError),
}

/// Why [`HostMemoryCoordinator::queue_for_lease`] returned no lease: a
/// queued wait has no deadline, so it never times out.
pub type QueuedLeaseFailure = LeaseFailure<FinalRefusal>;

/// Why a job execution plan was not granted.
pub type PlanFailure = LeaseFailure<BoundedRefusal>;

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
        HostMemoryCoordinator, HostMemoryError, HostMemoryLease, HostMemoryPressureLevel,
        HostMemoryRuntimeConfig, HostReadings, MemoryLeaseKind, MemoryLeaseRecord,
        MemoryLeaseRequest, MemoryLedger, NoIdleWorkers, plan_job_reservation, pressure_level_for,
        with_locked_ledger,
    };
    use crate::api::MachineTime;

    /// The ledger keeps its file form: a lease's time is whole Unix seconds
    /// under `created_at_epoch_s`, so an older daemon on the same host still
    /// reads a ledger this build wrote, and this build reads theirs.
    #[test]
    fn ledger_times_are_whole_unix_seconds_on_disk() {
        let old_form = r#"{"leases":[{"id":"a","kind":"job_execution","owner_pid":1,
            "reserved_mb":10,"startup_slot":false,"label":"x","created_at_epoch_s":1700000000}]}"#;
        let ledger: MemoryLedger = serde_json::from_str(old_form).unwrap();
        assert_eq!(
            ledger.leases[0].created_at,
            MachineTime::from_timestamp(jiff::Timestamp::constant(1_700_000_000, 0))
        );
        let written = serde_json::to_string(&ledger).unwrap();
        assert!(
            written.contains(r#""created_at_epoch_s":1700000000"#),
            "{written}"
        );
    }
    use crate::api::MemoryMb;
    use crate::worker::WorkerPid;
    use std::path::PathBuf;
    use tempfile::TempDir;
    use uuid::Uuid;

    /// `startup_slot` is derived from the kind: it is still written (older
    /// builds count slots by it) and ignored on read, so a job-execution lease
    /// can no longer claim a slot. A ledger file round trips unchanged.
    #[test]
    fn the_startup_slot_on_disk_follows_the_kind() {
        let lying = r#"{"leases":[{"id":"a","kind":"job_execution","owner_pid":1,
            "reserved_mb":10,"startup_slot":true,"label":"x","created_at_epoch_s":1700000000}]}"#;
        let ledger: MemoryLedger = serde_json::from_str(lying).unwrap();
        assert!(!ledger.leases[0].kind.holds_startup_slot());
        let written = serde_json::to_string(&ledger).unwrap();
        assert!(written.contains(r#""startup_slot":false"#), "{written}");

        let startup = r#"{"leases":[{"id":"b","kind":"worker_startup","owner_pid":1,
            "reserved_mb":10,"startup_slot":true,"label":"y","created_at_epoch_s":1700000000}]}"#;
        let ledger: MemoryLedger = serde_json::from_str(startup).unwrap();
        let written = serde_json::to_string(&ledger).unwrap();
        assert!(written.contains(r#""startup_slot":true"#), "{written}");
    }

    /// Records what a queued wait tells its observer, in order.
    #[derive(Default)]
    struct RecordingObserver {
        events: std::sync::Mutex<Vec<&'static str>>,
    }

    impl super::QueuedLeaseObserver for RecordingObserver {
        fn queued(&self) {
            self.events.lock().expect("events").push("queued");
        }
        fn dequeued(&self) {
            self.events.lock().expect("events").push("dequeued");
        }
    }

    impl RecordingObserver {
        fn seen(&self) -> Vec<&'static str> {
            self.events.lock().expect("events").clone()
        }
    }

    /// A model lease for a worker that is this (live) test process.
    fn model_load(reserved_mb: u64) -> super::WorkerLeaseRequest {
        super::WorkerLeaseRequest::model_load(
            WorkerPid(std::process::id()),
            "fa:wav2vec",
            MemoryMb(reserved_mb),
        )
    }

    /// A load for another live worker (this test's parent process, so the
    /// ledger's dead-owner pruning keeps it): memory held by a different
    /// worker is congestion that can clear, unlike the requester's own.
    fn other_worker_load(reserved_mb: u64) -> super::WorkerLeaseRequest {
        super::WorkerLeaseRequest::model_load(
            WorkerPid(std::os::unix::process::parent_id()),
            "fa:wav2vec",
            MemoryMb(reserved_mb),
        )
    }

    /// Poll and report intervals short enough for a test.
    const POLL: std::time::Duration = std::time::Duration::from_millis(20);

    /// A load the host cannot fit now queues behind the leases it holds, says
    /// so, and is granted once one of them is released. The declared host
    /// has 14,000 MB available against an 8,000 MB reserve: a held 5,000 MB
    /// lease leaves no room for 2,000 MB more until it goes.
    #[tokio::test]
    async fn a_queued_load_is_granted_when_a_held_lease_is_released() {
        let (_dir, config) = test_runtime_config();
        let coordinator = HostMemoryCoordinator::new(config, test_readings());
        let held = coordinator
            .try_acquire_lease(other_worker_load(5_000).0)
            .expect("the first lease fits");
        let observer = RecordingObserver::default();
        let waiting = coordinator.queue_for_lease_every(
            model_load(2_000),
            &observer,
            &NoIdleWorkers,
            POLL,
            POLL,
        );
        tokio::pin!(waiting);
        assert!(
            tokio::time::timeout(POLL * 10, &mut waiting).await.is_err(),
            "the load must wait while the held lease leaves no room"
        );
        assert_eq!(observer.seen(), vec!["queued"]);

        held.release();
        let granted = waiting.await.expect("granted once the held lease is gone");
        assert_eq!(observer.seen(), vec!["queued", "dequeued"]);
        assert_eq!(
            coordinator.snapshot().expect("snapshot").active_reserved_mb,
            MemoryMb(2_000)
        );
        granted.release();
    }

    /// Stands in for the pool: retiring its one idle worker releases that
    /// worker's lease; after that nothing is idle.
    struct OneIdleWorker(std::sync::Mutex<Option<HostMemoryLease>>);

    #[async_trait::async_trait]
    impl super::MemoryReclaim for OneIdleWorker {
        async fn reclaim_one(&self) -> super::Reclaimed {
            match self.0.lock().expect("lock").take() {
                Some(lease) => {
                    lease.release();
                    super::Reclaimed::Retired
                }
                None => super::Reclaimed::NothingIdle,
            }
        }
    }

    /// A queued load retires an idle worker to make room instead of waiting
    /// for it to leave: the idle worker's 5,000 MB lease is all that keeps
    /// 2,000 MB more from fitting, and the load is granted with no release
    /// from anywhere else. The poll is an hour, so a grant inside the test's
    /// timeout can only come from the reclaim's immediate retry.
    #[tokio::test]
    async fn a_queued_load_retires_an_idle_worker_to_make_room() {
        let (_dir, config) = test_runtime_config();
        let coordinator = HostMemoryCoordinator::new(config, test_readings());
        let idle = OneIdleWorker(std::sync::Mutex::new(Some(
            coordinator
                .try_acquire_lease(other_worker_load(5_000).0)
                .expect("the idle worker's lease fits"),
        )));
        let observer = RecordingObserver::default();
        let hour = std::time::Duration::from_secs(3_600);
        let granted = tokio::time::timeout(
            POLL * 50,
            coordinator.queue_for_lease_every(model_load(2_000), &observer, &idle, hour, hour),
        )
        .await
        .expect("granted at once after the idle worker was retired")
        .expect("granted");
        assert_eq!(observer.seen(), vec!["queued", "dequeued"]);
        assert_eq!(
            coordinator.snapshot().expect("snapshot").active_reserved_mb,
            MemoryMb(2_000)
        );
        granted.release();
    }

    /// A cancelled job drops its queued wait: the wait ends, says so, and
    /// takes no lease, even once the host could fit it.
    #[tokio::test]
    async fn a_dropped_queued_wait_ends_and_takes_no_lease() {
        let (_dir, config) = test_runtime_config();
        let coordinator = HostMemoryCoordinator::new(config, test_readings());
        let held = coordinator
            .try_acquire_lease(other_worker_load(5_000).0)
            .expect("the first lease fits");
        let observer = RecordingObserver::default();
        let waited = tokio::time::timeout(
            POLL * 10,
            coordinator.queue_for_lease_every(
                model_load(2_000),
                &observer,
                &NoIdleWorkers,
                POLL,
                POLL,
            ),
        )
        .await;
        assert!(
            waited.is_err(),
            "nothing was released, so it was still queued"
        );
        assert_eq!(observer.seen(), vec!["queued", "dequeued"]);

        held.release();
        tokio::time::sleep(POLL * 5).await;
        let snapshot = coordinator.snapshot().expect("snapshot");
        assert_eq!(snapshot.active_reserved_mb, MemoryMb(0));
        assert!(
            !snapshot
                .active_lease_labels
                .iter()
                .any(|label| label.contains("model-load")),
            "{:?}",
            snapshot.active_lease_labels
        );
    }

    /// A load larger than the machine can ever fit is refused at once, never
    /// queued: 20,000 MB on a 15,989 MB host.
    #[tokio::test]
    async fn an_impossible_load_is_refused_at_once() {
        let (_dir, config) = test_runtime_config();
        let coordinator = HostMemoryCoordinator::new(config, test_readings());
        let observer = RecordingObserver::default();
        let outcome = coordinator
            .queue_for_lease_every(model_load(20_000), &observer, &NoIdleWorkers, POLL, POLL)
            .await;
        match outcome {
            Err(super::QueuedLeaseFailure::Refused(super::FinalRefusal::NeverFits(
                super::NeverFits {
                    needed_mb,
                    ceiling_mb,
                    ..
                },
            ))) => assert_eq!((needed_mb, ceiling_mb), (20_000, 14_000)),
            Err(other) => panic!("expected a never-fits refusal, got {other:?}"),
            Ok(_) => panic!("a load larger than the machine was granted"),
        }
        assert!(
            observer.seen().is_empty(),
            "an impossible load never queues"
        );
    }

    /// A worker's load inside its owner's job lease adds nothing to the
    /// owner's demand (Contract D), so it is admitted even when the host
    /// reads as over-committed: a job never waits on its own reservation.
    #[test]
    fn a_load_inside_its_owners_job_lease_is_admitted_on_a_full_host() {
        let (_dir, config) = test_runtime_config();
        let coordinator = HostMemoryCoordinator::new(config.clone(), test_readings());
        inject_lease(
            &config,
            MemoryLeaseKind::JobExecution,
            std::process::id(),
            None,
            13_000,
            "job",
        );
        // 14,000 available less 13,000 reserved is under the 8,000 reserve.
        let inside = coordinator
            .try_acquire_lease(model_load(4_000).0)
            .expect("a sub-allocation of the job's lease is admitted");
        assert_eq!(
            coordinator.snapshot().expect("snapshot").active_reserved_mb,
            MemoryMb(13_000),
            "the load is counted inside the job's reservation, not on top"
        );
        let Err(refused) = coordinator.try_acquire_lease(model_load(10_000).0) else {
            panic!("a load that raises the owner's demand past the host was admitted");
        };
        assert!(
            matches!(refused, HostMemoryError::NeverFits { .. }),
            "a load the host can never offer is refused for good: {refused:?}"
        );
        inside.release();
    }

    /// Settling a lease changes what it counts: a model loading is counted
    /// in full, a model resident only at what it has yet to use.
    #[test]
    fn a_settled_lease_counts_its_new_reservation() {
        let (_dir, config) = test_runtime_config();
        let coordinator = HostMemoryCoordinator::new(config, test_readings());
        let lease = coordinator
            .try_acquire_lease(model_load(4_000).0)
            .expect("fits");
        lease.settler().settle(MemoryMb(1_000)).expect("settle");
        let snapshot = coordinator.snapshot().expect("snapshot");
        assert_eq!(snapshot.active_reserved_mb, MemoryMb(1_000));
        lease.release();
        assert_eq!(
            coordinator.snapshot().expect("snapshot").active_reserved_mb,
            MemoryMb(0)
        );
    }

    /// A model lease names its worker from the start, so it is reclaimed when
    /// that worker dies.
    #[test]
    fn a_model_lease_for_a_dead_worker_is_reclaimed() {
        let (_dir, config) = test_runtime_config();
        let coordinator = HostMemoryCoordinator::new(config, test_readings());
        let lease = coordinator
            .try_acquire_lease(
                super::WorkerLeaseRequest::model_load(WorkerPid(4_000_000), "coref", MemoryMb(500))
                    .0,
            )
            .expect("fits");
        std::mem::forget(lease);
        assert_eq!(
            coordinator.snapshot().expect("snapshot").active_reserved_mb,
            MemoryMb(0)
        );
    }

    fn test_readings() -> HostReadings {
        crate::host_facts::HostSnapshot::for_tests(&crate::config::ServerConfig::default())
            .readings()
    }

    fn test_runtime_config() -> (TempDir, HostMemoryRuntimeConfig) {
        let dir = TempDir::new().expect("tmp dir for host-memory test");
        let coordinator_path: PathBuf = dir.path().join("ledger.json");
        let config = HostMemoryRuntimeConfig::from_sources(
            coordinator_path,
            MemoryMb(8_000),
            crate::config::WorkerStartupLimit::literal::<4>(),
        );
        (dir, config)
    }

    /// Test helper: inject a synthetic lease into the ledger so we can
    /// drive prune_stale_leases without spawning real workers.
    fn inject_lease(
        config: &HostMemoryRuntimeConfig,
        kind: MemoryLeaseKind,
        owner_pid: u32,
        worker_pid: Option<u32>,
        reserved_mb: u64,
        label: &str,
    ) {
        with_locked_ledger(&config.coordinator_path, |ledger: &mut MemoryLedger| {
            ledger.leases.push(MemoryLeaseRecord {
                id: Uuid::new_v4().to_string(),
                kind,
                owner_pid: WorkerPid(owner_pid),
                worker_pid: worker_pid.map(WorkerPid),
                reserved_mb: MemoryMb(reserved_mb),
                label: label.to_owned(),
                created_at: MachineTime::now(),
            });
            Ok::<(), super::UnusableHost>(())
        })
        .expect("inject lease");
    }

    /// Bug 2 (ghost slots) regression: when a worker subprocess dies
    /// without the daemon explicitly releasing its lease, the ledger
    /// retains a phantom reservation that blocks future jobs from
    /// fitting. Pruning MUST detect a dead worker_pid and drop the
    /// lease so reserved memory accounting stays in sync with reality.
    ///
    /// Failure mode (pre-fix): the lease's `owner_pid` is the DAEMON's
    /// PID. The daemon stays alive, so `process_is_alive(owner_pid)`
    /// returns true and the lease is never pruned. The reservation
    /// accumulates as ghost slots, exactly the 2026-05-01 wedge.
    #[test]
    fn lease_for_dead_worker_pid_is_pruned_on_next_ledger_access() {
        let (_dir, config) = test_runtime_config();
        let coord = HostMemoryCoordinator::new(config.clone(), test_readings());

        // PID 4_000_000 is a known-dead PID per the reaper test convention.
        let dead_worker_pid: u32 = 4_000_000;
        let daemon_pid = std::process::id();

        // Inject a phantom lease tagged with the dead worker's PID.
        inject_lease(
            &config,
            MemoryLeaseKind::JobExecution,
            daemon_pid,
            Some(dead_worker_pid),
            12_000,
            "test-ghost-slot",
        );

        // Sanity-check: the lease was inserted.
        let pre_prune = coord
            .snapshot()
            .expect("snapshot before second prune cycle");
        // First snapshot call already runs prune_stale_leases once.
        // After the fix, the lease should already be gone here:
        assert_eq!(
            pre_prune.active_reserved_mb,
            MemoryMb(0),
            "lease whose worker_pid is dead must be pruned, but ledger still reports {} MB reserved",
            pre_prune.active_reserved_mb.0
        );
    }

    /// End-to-end regression: the daemon-side flow is
    ///   1. acquire a lease (no worker_pid yet, daemon's PID held)
    ///   2. spawn the worker subprocess (now `Child::id()` is known)
    ///   3. tag the lease with the worker's PID via `set_worker_pid`
    ///   4. when the worker dies later, prune drops the lease.
    ///
    /// This test exercises the API surface that the worker-spawn code
    /// must call. Without `set_worker_pid` plumbed at the spawn site,
    /// the field stays `None` and Bug 2 is unfixed in deployed binaries.
    #[test]
    fn set_worker_pid_makes_subsequent_prune_target_the_worker() {
        let (_dir, config) = test_runtime_config();
        let coord = HostMemoryCoordinator::new(config.clone(), test_readings());

        // 1. Acquire a synthetic lease the way the daemon would
        //    owner_pid = daemon (this process), worker_pid = None.
        let lease_id = Uuid::new_v4().to_string();
        with_locked_ledger(&config.coordinator_path, |ledger| {
            ledger.leases.push(MemoryLeaseRecord {
                id: lease_id.clone(),
                kind: MemoryLeaseKind::WorkerStartup,
                owner_pid: WorkerPid(std::process::id()),
                worker_pid: None,
                reserved_mb: MemoryMb(12_000),
                label: "test-startup-lease".to_owned(),
                created_at: MachineTime::now(),
            });
            Ok::<(), super::UnusableHost>(())
        })
        .expect("inject startup lease");

        // 2. Build the public-API HostMemoryLease handle for that
        //    record (mirrors what acquire_worker_startup_lease returns).
        let lease = HostMemoryLease::new(config.coordinator_path.clone(), lease_id.clone());

        // 3. Tag the lease with a known-dead worker PID, same
        //    convention the reaper test uses.
        let dead_worker_pid: u32 = 4_000_000;
        lease
            .set_worker_pid(WorkerPid(dead_worker_pid))
            .expect("set_worker_pid should rewrite the ledger");

        // Don't drop the lease yet, Drop calls release_internal which
        // would remove the record outright and obscure whether prune
        // did the job.
        std::mem::forget(lease);

        // 4. Trigger a prune cycle by reading the ledger via snapshot.
        let snap = coord.snapshot().expect("snapshot");
        assert_eq!(
            snap.active_reserved_mb,
            MemoryMb(0),
            "after set_worker_pid pointed at a dead PID, prune must drop the lease"
        );
    }

    // ============================================================
    // Memory-accounting contract tests (2026-05-01 Feynman-discipline
    // pass). These tests are written from FIRST PRINCIPLES about what
    // a memory-reservation subsystem should do, not from inspection of
    // the implementation. Tests that fail are evidence that either the
    // contract is wrong (correct it) or the system violates it (bug).
    // ============================================================

    /// The coordinator reads the declared host, so its capacity boundary is
    /// exact on every machine: 14,000 MB available less an 8,000 MB reserve
    /// admits a 6,000 MB lease and refuses 6,001 MB, with the declared
    /// figures in the refusal.
    #[test]
    fn a_lease_is_admitted_exactly_up_to_the_reserve_on_a_declared_host() {
        let (_dir, config) = test_runtime_config();
        let coordinator = HostMemoryCoordinator::new(config, test_readings());
        let request = |reserved_mb| MemoryLeaseRequest {
            kind: MemoryLeaseKind::WorkerStartup,
            reserved_mb: MemoryMb(reserved_mb),
            label: "boundary".to_owned(),
            worker_pid: None,
        };

        let refused = coordinator
            .try_acquire_lease(request(6_001))
            .expect_err("one megabyte past the reserve was admitted");
        let HostMemoryError::NeverFits(super::NeverFits {
            needed_mb,
            ceiling_mb,
            reserve_mb,
            ..
        }) = refused
        else {
            panic!("expected a never-fits refusal, got {refused:?}");
        };
        assert_eq!((needed_mb, ceiling_mb, reserve_mb), (6_001, 14_000, 8_000));

        let admitted = coordinator
            .try_acquire_lease(request(6_000))
            .expect("exactly at the reserve");
        admitted.release();
    }

    /// Contract A: Conservation:
    /// Σ(reserved across leases) must never exceed (total − headroom).
    /// The coordinator should refuse a request that would push the
    /// total over capacity, not allocate beyond capacity.
    ///
    /// Test: simulate a finite-capacity host. Acquire reservations
    /// that fit. The next request that would exceed capacity must be
    /// rejected with CapacityRejected. The ledger sum must never
    /// transiently exceed (total − headroom).
    #[test]
    fn contract_a_capacity_rejection_when_sum_would_exceed_budget() {
        // The planner's arithmetic, directly; the coordinator's own
        // boundary on a declared host is
        // `a_lease_is_admitted_exactly_up_to_the_reserve_on_a_declared_host`.
        //
        // The contract statement is enforced by `plan_job_reservation`:
        // when the worst-case requested allocation would push beyond
        // (available - reserve_mb), it must return None (rejection).
        //
        // For a worst-case-rejecting-correctly test:
        //   available = 32 GB
        //   pending = 20 GB already reserved
        //   reserve_mb = 8 GB headroom
        //   per_worker = 8 GB; requesting 1 worker
        //
        // Math:  remaining = 32 - 20 - 8 = 4 GB; 1 worker needs 8 GB.
        // Expected: rejection (None).
        let result = plan_job_reservation(1, 8_000, 32_000, 8_000, 20_000);
        assert!(
            result.is_none(),
            "Contract A violated: planner allowed 1 worker × 8 GB when only 4 GB free above headroom (got {result:?})"
        );

        // Allowed case: reduce pending to leave room for 1 worker.
        let allowed = plan_job_reservation(1, 8_000, 32_000, 8_000, 16_000);
        assert!(
            allowed.is_some(),
            "Contract A: planner refused a request that should fit (32 - 16 - 8 - 8 = 0, equal to threshold)"
        );
    }

    /// Contract B: Liveness:
    /// A reservation tagged with a process PID must be reclaimed when
    /// that process dies. Already covered by the prune tests above
    /// (`lease_for_dead_worker_pid_is_pruned_on_next_ledger_access`),
    /// reasserted here as the named contract.
    #[test]
    fn contract_b_lease_reclaimed_when_bound_process_dies() {
        let (_dir, config) = test_runtime_config();
        let coord = HostMemoryCoordinator::new(config.clone(), test_readings());
        let dead_pid: u32 = 4_000_000;
        inject_lease(
            &config,
            MemoryLeaseKind::WorkerStartup,
            std::process::id(),
            Some(dead_pid),
            8_000,
            "contract-b-test",
        );
        let snap = coord.snapshot().expect("snapshot");
        assert_eq!(
            snap.active_reserved_mb,
            MemoryMb(0),
            "Contract B violated: lease bound to dead PID {dead_pid} not reclaimed"
        );
    }

    /// Contract D, No double-counting:
    /// A single live worker subprocess must be represented by exactly
    /// one reservation, not multiple stacking ones. If a job has a
    /// JobExecution lease covering its workers' budget, individual
    /// WorkerStartup leases for those same workers must NOT add to
    /// the daemon-side total reserved figure; they should be
    /// sub-allocations, not additions.
    ///
    /// Operational evidence (2026-05-01): with 3 admitted jobs each
    /// reserving ~76 GB at JobExecution level, the daemon ALSO
    /// counts WorkerStartup leases on top, producing the 228 GB ghost
    /// even though only 3 jobs are active. This test pins what the
    /// contract should be, and SHOULD FAIL TODAY.
    #[test]
    fn contract_d_worker_startup_does_not_double_count_with_job_execution() {
        let (_dir, config) = test_runtime_config();
        let coord = HostMemoryCoordinator::new(config.clone(), test_readings());

        // Inject a JobExecution lease that already reserves 96 GB
        // (the worst-case 8-worker × 12 GB plan).
        inject_lease(
            &config,
            MemoryLeaseKind::JobExecution,
            std::process::id(),
            None,
            96_000,
            "contract-d-job-exec",
        );
        // Now acquire a WorkerStartup lease for one of those workers.
        // Under correct accounting, this should NOT add to the
        // daemon-side total because the JobExecution reservation
        // already covers it.
        inject_lease(
            &config,
            MemoryLeaseKind::WorkerStartup,
            std::process::id(),
            Some(std::process::id()),
            12_000,
            "contract-d-worker-startup",
        );

        let snap = coord.snapshot().expect("snapshot");
        assert_eq!(
            snap.active_reserved_mb,
            MemoryMb(96_000),
            "Contract D violated: WorkerStartup double-counts on top of JobExecution; total reserved is {} MB, expected 96000 MB",
            snap.active_reserved_mb.0
        );
    }

    /// Contract E: Pre-spawn intent must not phantom-hold capacity:
    /// A reservation made for a worker that has not yet spawned (i.e.,
    /// `worker_pid: None`, owner_pid = daemon) must either:
    ///   (a) be tied to a deadline, automatically reclaimed if the
    ///       spawn doesn't materialize within the window; OR
    ///   (b) be tagged with the worker's PID promptly after spawn,
    ///       so liveness binding kicks in.
    ///
    /// If neither holds, a failed spawn or runaway daemon code path
    /// produces a phantom hold that consumes budget forever.
    ///
    /// Test: simulate the "spawn never materialises" case. Acquire a
    /// startup-style reservation and never tag it with a worker PID.
    /// Wait the prune cycle. The reservation must be reclaimed.
    ///
    /// Expectation: SHOULD FAIL TODAY. The pre-spawn intent has no
    /// deadline mechanism in the current code; daemon-PID-bound
    /// reservations survive indefinitely.
    #[test]
    fn contract_e_pre_spawn_reservation_without_worker_pid_must_have_deadline() {
        let (_dir, config) = test_runtime_config();
        let coord = HostMemoryCoordinator::new(config.clone(), test_readings());

        // Inject a "pending spawn" reservation: owner = daemon (alive),
        // worker_pid = None (no worker exists yet), age beyond any
        // reasonable spawn deadline.
        inject_lease(
            &config,
            MemoryLeaseKind::WorkerStartup,
            std::process::id(),
            None,
            12_000,
            "contract-e-pending-spawn-aged",
        );

        // Backdate the lease's created_at so it's clearly past any
        // reasonable startup window.
        with_locked_ledger(&config.coordinator_path, |ledger| {
            for lease in ledger.leases.iter_mut() {
                lease.created_at = MachineTime::from_timestamp(jiff::Timestamp::constant(1, 0));
            }
            Ok::<(), super::UnusableHost>(())
        })
        .unwrap();

        let snap = coord.snapshot().expect("snapshot");
        assert_eq!(
            snap.active_reserved_mb,
            MemoryMb(0),
            "Contract E violated: a pre-spawn reservation that never materialised into a worker is held indefinitely (got {} MB still reserved)",
            snap.active_reserved_mb.0
        );
    }

    /// Contract C: A 64 GB host with clean state must admit at least
    /// a 1-worker morphotag job. Per-worker budget for morphotag is
    /// 12 GB (8000 base × 1.5 loading_overhead, process mode); reserve
    /// headroom is 8 GB. With 64 GB available and 0 pending, the
    /// projected leftover after granting 1 worker is 64 − 0 − 12 = 52
    /// GB ≥ 8 GB reserve. The planner MUST grant at least 1 worker.
    ///
    /// This is the "tragicomic" case from a historical incident:
    /// several 64 GB fleet hosts reportedly couldn't run morphotag
    /// with default config, even though "they sure as hell can."
    /// This test pins what the planner is supposed to do.
    #[test]
    fn contract_c_64gb_host_with_clean_state_admits_1_worker_morphotag() {
        // morphotag per-worker = 8000 × 1.5 = 12000 MB (process mode)
        let result = plan_job_reservation(
            1,      // requested_workers
            12_000, // per_worker_budget
            64_000, // available_mb (clean 64 GB host, no other load)
            8_000,  // reserve_mb (host headroom)
            0,      // pending_reserved_mb
        );
        assert!(
            result.is_some(),
            "Contract C violated: 64 GB host with clean state cannot admit 1-worker morphotag (per-worker=12 GB, reserve=8 GB). Math: 64 - 0 - 12 = 52 ≥ 8."
        );
        let (granted, reserved_mb) = result.unwrap();
        assert_eq!(granted, 1);
        assert_eq!(reserved_mb, 12_000);
    }

    /// Contract C (laptop variant), A 16 GB laptop must admit a
    /// 1-worker morphotag job per the design intent stated in
    /// `runtime.rs:218`: "This allows batchalign3 to run on 16 GB
    /// laptops through 256 GB servers without manual tuning."
    ///
    /// Math under current constants: 16 - 0 - 12 = 4 < 8 reserve → reject.
    /// Expected: SHOULD FAIL TODAY. The 8 GB reserve is more than
    /// half a 16 GB laptop's RAM, so the planner can't fit even a
    /// single worker. The fix is tier-derived reserve sizing (small
    /// hosts get a smaller reserve).
    #[test]
    fn contract_c_16gb_laptop_with_clean_state_admits_1_worker_morphotag() {
        let result = plan_job_reservation(1, 12_000, 16_000, 8_000, 0);
        assert!(
            result.is_some(),
            "Contract C violated: 16 GB laptop with clean state cannot admit 1-worker morphotag. Math: 16 - 0 - 12 = 4, reserve=8. Either the per-worker budget is too high or the reserve is too high for small hosts. Design intent (runtime.rs:218): 'batchalign3 runs on 16 GB laptops through 256 GB servers without manual tuning', that's not happening with current constants."
        );
    }

    /// Contract F: Atomicity across daemon crash:
    /// On daemon restart, leases owned by the (now-dead) prior daemon
    /// PID must be reclaimed before any new accounting decision.
    ///
    /// Test: write a ledger file in which all leases are owned by a
    /// dead PID and have `worker_pid: None`. Build a new coordinator
    /// pointing at that ledger. The first snapshot must show zero
    /// reservations.
    ///
    /// Expectation: SHOULD PASS TODAY (the existing
    /// `prune_stale_leases` does this for `worker_pid: None` leases
    /// via the owner_pid liveness check). This contract test is a
    /// regression guard.
    #[test]
    fn contract_f_dead_daemon_pid_leases_reclaimed_at_first_access() {
        let (_dir, config) = test_runtime_config();
        let coord = HostMemoryCoordinator::new(config.clone(), test_readings());

        let dead_daemon_pid: u32 = 4_000_001;
        inject_lease(
            &config,
            MemoryLeaseKind::JobExecution,
            dead_daemon_pid,
            None,
            96_000,
            "contract-f-prior-daemon",
        );

        let snap = coord.snapshot().expect("snapshot");
        assert_eq!(
            snap.active_reserved_mb,
            MemoryMb(0),
            "Contract F violated: lease owned by dead daemon PID {dead_daemon_pid} survived first ledger access ({} MB still reserved)",
            snap.active_reserved_mb.0
        );
    }

    /// Counterpart: a lease whose worker_pid IS alive must be retained.
    /// This guards against an over-eager prune that would drop healthy
    /// reservations and double-book memory.
    #[test]
    fn lease_for_live_worker_pid_is_retained() {
        let (_dir, config) = test_runtime_config();
        let coord = HostMemoryCoordinator::new(config.clone(), test_readings());

        // Use the test process's own PID, guaranteed alive.
        let live_pid = std::process::id();

        inject_lease(
            &config,
            MemoryLeaseKind::JobExecution,
            live_pid,
            Some(live_pid),
            12_000,
            "test-live-slot",
        );

        let snap = coord.snapshot().expect("snapshot");
        assert_eq!(
            snap.active_reserved_mb,
            MemoryMb(12_000),
            "lease for live worker_pid must be retained"
        );
    }

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

    /// On 32 GB available the 8192 MB reserve caps at 8000 MB; the
    /// planner clamps 8 requested workers down to 4.
    #[test]
    fn job_planner_reduces_worker_count_to_fit_headroom() {
        let planned =
            plan_job_reservation(8, 6_000, 32_000, 8_192, 0).expect("some worker count should fit");
        assert_eq!(planned, (4, 24_000));
    }

    /// Pending reservations are subtracted before the headroom check.
    #[test]
    fn job_planner_accounts_for_pending_reservations() {
        let planned = plan_job_reservation(4, 4_000, 24_000, 8_192, 6_000)
            .expect("some worker count should fit");
        assert_eq!(planned, (3, 12_000));
    }

    /// On a host where even one worker would overflow the headroom
    /// after pending demand, the planner rejects.
    #[test]
    fn job_planner_rejects_when_even_one_worker_breaks_reserve() {
        assert!(plan_job_reservation(2, 8_000, 6_000, 8_192, 1_000).is_none());
    }

    /// The declared test host can ever offer 14,000 MB, and keeps 8,000 MB
    /// free, so one worker can ever hold 6,000 MB.
    fn worker_request(worker: u32, reserved_mb: u64) -> MemoryLeaseRequest {
        MemoryLeaseRequest {
            kind: MemoryLeaseKind::ModelRequest,
            reserved_mb: MemoryMb(reserved_mb),
            label: format!("model-request:fa:pid {worker}"),
            worker_pid: Some(WorkerPid(worker)),
        }
    }

    /// A request that, with what its own worker already holds, exceeds what
    /// the host can ever offer is refused for good. Those leases cannot
    /// release while it waits on them: before, the request alone was judged
    /// against total RAM and it waited for ever on its own worker.
    #[test]
    fn a_request_blocked_only_by_its_own_worker_never_fits() {
        let (_dir, config) = test_runtime_config();
        let own = std::process::id();
        inject_lease(
            &config,
            MemoryLeaseKind::ModelLoad,
            own,
            Some(own),
            4_000,
            "load",
        );
        let coordinator = HostMemoryCoordinator::new(config, test_readings());
        match coordinator.try_acquire_lease(worker_request(own, 3_000)) {
            Err(HostMemoryError::NeverFits(super::NeverFits {
                needed_mb,
                ceiling_mb,
                reserve_mb,
                ..
            })) => assert_eq!((needed_mb, ceiling_mb, reserve_mb), (7_000, 14_000, 8_000)),
            other => panic!("expected never-fits, got {other:?}"),
        }
    }

    /// The same request is only congestion when another worker holds the
    /// memory: that lease can release, so the request waits rather than
    /// failing.
    #[test]
    fn a_request_blocked_by_another_worker_is_congestion() {
        let (_dir, config) = test_runtime_config();
        let other = std::os::unix::process::parent_id();
        inject_lease(
            &config,
            MemoryLeaseKind::ModelLoad,
            other,
            Some(other),
            4_000,
            "load",
        );
        let coordinator = HostMemoryCoordinator::new(config, test_readings());
        match coordinator.try_acquire_lease(worker_request(std::process::id(), 3_000)) {
            Err(HostMemoryError::CapacityRejected { requested_mb, .. }) => {
                assert_eq!(requested_mb, 3_000);
            }
            other => panic!("expected congestion, got {other:?}"),
        }
    }

    /// The ceiling is what the host can ever offer: a live machine's total
    /// RAM, a declared machine's available RAM.
    #[test]
    fn a_declared_host_can_never_offer_more_than_it_declares_available() {
        let readings = test_readings();
        let ceiling = readings.ceiling_of(readings.memory().expect("declared host"));
        assert_eq!(ceiling.mb(), MemoryMb(14_000));
        assert!(ceiling.admits(MemoryMb(6_000), MemoryMb(8_000)));
        assert!(!ceiling.admits(MemoryMb(6_001), MemoryMb(8_000)));
    }

    /// A request that reserves nothing (the machine-wide test lock) asks for
    /// no memory, so it is never refused as too large, even on a host whose
    /// reserve exceeds what it can offer.
    #[test]
    fn a_request_for_no_memory_is_never_refused_as_too_large() {
        let dir = TempDir::new().expect("tmp dir");
        let config = HostMemoryRuntimeConfig::from_sources(
            dir.path().join("ledger.json"),
            MemoryMb(20_000),
            crate::config::WorkerStartupLimit::literal::<4>(),
        );
        let coordinator = HostMemoryCoordinator::new(config, test_readings());
        let lock = MemoryLeaseRequest {
            kind: MemoryLeaseKind::MlTestExclusive,
            reserved_mb: MemoryMb(0),
            label: "test lock".to_owned(),
            worker_pid: None,
        };
        assert!(
            !matches!(
                coordinator.try_acquire_lease(lock),
                Err(HostMemoryError::NeverFits { .. })
            ),
            "a zero-MB lock cannot be too large for any host"
        );
    }

    /// Another request's workspace on the same worker ends with that request,
    /// so it is congestion: only the worker's resident models count as memory
    /// a waiting request can never get.
    #[test]
    fn a_sibling_requests_lease_on_the_same_worker_is_congestion() {
        let (_dir, config) = test_runtime_config();
        let own = std::process::id();
        inject_lease(
            &config,
            MemoryLeaseKind::ModelRequest,
            own,
            Some(own),
            4_000,
            "sibling",
        );
        let coordinator = HostMemoryCoordinator::new(config, test_readings());
        match coordinator.try_acquire_lease(worker_request(own, 3_000)) {
            Err(HostMemoryError::CapacityRejected { .. }) => {}
            other => panic!("expected congestion, got {other:?}"),
        }
    }

    /// A job whose single worker can never fit is refused for good: before,
    /// it waited out its bounded wait and was requeued, for ever.
    #[test]
    fn a_job_plan_that_can_never_fit_one_worker_is_refused() {
        // A machine smaller than one morphotag worker's budget: even idle,
        // the planner cannot grant one worker, so waiting cannot help.
        let dir = TempDir::new().expect("tmp dir");
        let config = HostMemoryRuntimeConfig::from_sources(
            dir.path().join("ledger.json"),
            MemoryMb(512),
            crate::config::WorkerStartupLimit::literal::<4>(),
        );
        let mut facts = crate::host_facts::test_helpers::laptop_16gb();
        facts.ram_total_mb = 256;
        facts.ram_available_mb = 256;
        let tiny = super::HostReadings::Declared(super::DeclaredHost::from_facts(&facts));
        let coordinator = HostMemoryCoordinator::new(config, tiny);
        match coordinator.try_plan_job_execution(
            crate::api::ReleasedCommand::Morphotag,
            crate::api::NumWorkers(1),
            "plan",
        ) {
            Err(HostMemoryError::NeverFits(super::NeverFits {
                ceiling_mb,
                reserve_mb,
                ..
            })) => assert_eq!((ceiling_mb, reserve_mb), (256, 64)),
            other => panic!(
                "expected never-fits, got {:?}",
                other.map(|plan| plan.reserved_mb)
            ),
        }
    }

    /// A large configured reserve does not refuse for good a job the
    /// planner would grant: the planner caps the reserve at a quarter of
    /// what is available, and the never-fits check uses the same policy.
    #[test]
    fn a_large_reserve_does_not_refuse_a_job_the_planner_would_grant() {
        let dir = TempDir::new().expect("tmp dir");
        let config = HostMemoryRuntimeConfig::from_sources(
            dir.path().join("ledger.json"),
            MemoryMb(13_900),
            crate::config::WorkerStartupLimit::literal::<4>(),
        );
        let coordinator = HostMemoryCoordinator::new(config, test_readings());
        let plan = coordinator
            .try_plan_job_execution(
                crate::api::ReleasedCommand::Morphotag,
                crate::api::NumWorkers(1),
                "plan",
            )
            .expect("the planner grants one worker on the idle declared host");
        assert_eq!(plan.granted_workers.0, 1);
    }
}
