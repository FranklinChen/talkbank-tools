//! `WorkerPool` — manages multiple Python worker processes.
//!
//! Workers are keyed by `(bootstrap target, lang, engine overrides)`.
//! Large hosts may use shared profile targets such as `profile:gpu`, while
//! constrained hosts use task targets such as `infer:asr` so one laptop does
//! not hold unrelated models in memory.
//!
//! Each key maps to a `WorkerGroup` containing up to `max_workers_per_key`
//! workers, spawned lazily on demand. Background tasks handle health checking
//! and idle timeouts.
//!
//! ## Concurrency model
//!
//! Workers are *owned values* in a `VecDeque`, not wrapped in `Arc<Mutex>`.
//! Availability is tracked by a `tokio::sync::Semaphore` (one permit per idle
//! worker). Callers *check out* a worker via `checkout()`, which acquires a
//! semaphore permit (async wait if all busy) then pops from the idle queue.
//! The returned `CheckedOutWorker` is an RAII guard that returns the worker
//! to the pool on drop.
//!
//! This eliminates the previous `Arc<tokio::sync::Mutex<WorkerHandle>>` pattern
//! where a tokio mutex was held for 10–300 seconds during dispatch.
//!
//! ## Module layout
//!
//! | File | Responsibility |
//! |------|----------------|
//! | `mod.rs` | Types, pool struct, construction, accessors, GPU worker creation |
//! | `checkout.rs` | `CheckedOutWorker` RAII guard |
//! | `dispatch.rs` | Checkout loop, batch infer, V2 execute, TCP routing |
//! | `discovery.rs` | Registry-based TCP worker discovery |
//! | `warmup.rs` | Pre-spawning workers and pre-scaling |
//! | `shutdown.rs` | Graceful shutdown and `Drop` cleanup |
//! | `lifecycle.rs` | Background health checking, idle timeout, spawn helpers |
//! | `execute_v2.rs` | V2 request key resolution helpers |
//! | `status.rs` | Status query methods (has_idle_workers, worker_count, etc.) |
//! | `shared_gpu/` | Shared concurrent GPU worker wrappers (stdio, TCP, reader) |
//! | `reaper.rs` | Orphaned worker process reaping |

mod checkout;
mod discovery;
mod dispatch;
mod eviction;
mod execute_v2;
pub(crate) mod job_tracker;
mod lifecycle;
pub(crate) mod reaper;
pub(crate) mod shared_gpu;
mod shutdown;
pub mod status;
mod warmup;

pub use checkout::CheckedOutWorker;
pub use status::{WorkerSummaryEntry, WorkerTransport};

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicUsize};

use crate::api::{NumSpeakers, WorkerLanguage};
use crate::worker::{WorkerBootstrapMode, WorkerCapabilities, WorkerTarget};
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use uuid::Uuid;

use crate::worker::error::WorkerError;
use crate::worker::handle::{WorkerConfig, WorkerHandle, WorkerRuntimeConfig};
use crate::worker::python::resolve_python_executable;
use crate::worker::tcp_handle::TcpWorkerHandle;

// ---------------------------------------------------------------------------
// Poison-recovery helper for std::sync::Mutex
// ---------------------------------------------------------------------------

/// Lock a `std::sync::Mutex`, recovering from poison if a previous thread
/// panicked while holding it.
///
/// All `std::sync::Mutex` instances in the worker pool guard `VecDeque` or
/// `HashMap` containers with short (microsecond) critical sections. If a
/// panic occurs during a push/pop, the data structure may have been partially
/// mutated, but it is still structurally valid -- the worst case is a
/// missing or double-counted worker, which the health checker will reconcile.
/// Recovering from poison keeps the server alive instead of cascading the
/// panic into every subsequent request.
pub(super) fn lock_recovered<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| {
        warn!("Recovering from poisoned std::sync::Mutex in worker pool");
        poisoned.into_inner()
    })
}

/// Key for looking up workers: (bootstrap target, lang, engine overrides).
pub(super) type WorkerKey = (WorkerTarget, WorkerLanguage, String);

/// Lifecycle state of background model warmup.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "server", derive(utoipa::ToSchema))]
#[serde(rename_all = "snake_case")]
/// Background warmup lifecycle state for the worker pool.
///
/// The server pre-spawns workers at startup ("warmup") so the first job
/// does not pay cold-start costs. This enum tracks that lifecycle for
/// the health endpoint.
pub enum WarmupStatus {
    /// No warmup has been requested yet (initial state).
    #[default]
    NotStarted,
    /// Warmup is running — workers are being spawned in the background.
    InProgress,
    /// All requested warmup spawns have finished (or none were requested).
    Complete,
}

impl WarmupStatus {
    const NOT_STARTED: u8 = 0;
    const IN_PROGRESS: u8 = 1;
    const COMPLETE: u8 = 2;

    pub(super) fn from_u8(v: u8) -> Self {
        match v {
            Self::IN_PROGRESS => Self::InProgress,
            Self::COMPLETE => Self::Complete,
            _ => Self::NotStarted,
        }
    }

    pub(super) fn as_u8(self) -> u8 {
        match self {
            Self::NotStarted => Self::NOT_STARTED,
            Self::InProgress => Self::IN_PROGRESS,
            Self::Complete => Self::COMPLETE,
        }
    }
}

/// Default maximum workers per `(profile, lang, engine_overrides)` key.
///
/// Lowered from 8 to 4 to prevent OOM on 64 GB developer machines where
/// GPU workers consume 13-15 GB each (8 × 15 GB = 120 GB → crash).
/// Override via `max_workers_per_key` in `server.yaml` for production
/// servers with more RAM (e.g., net with 256 GB).
const DEFAULT_MAX_WORKERS_PER_KEY: usize = 4;

/// Absolute ceiling on total workers. Even with unlimited RAM, never spawn
/// more than this many concurrent Python processes.
const ABSOLUTE_MAX_TOTAL_WORKERS: usize = 32;

/// RAM budget per worker for the global cap heuristic (6 GB).
///
/// This is a conservative median across all profiles. GPU workers actually
/// use 4-15 GB, Stanza workers 2-8 GB. Using 6 GB prevents the heuristic
/// from allowing more workers than physical RAM can support.
const RAM_PER_WORKER_BYTES: u64 = 6 * 1024 * 1024 * 1024;

/// Compute a default global worker cap from available system memory.
///
/// Uses `available_memory / 6GB`, clamped to `[2, 32]`. Falls back to 4
/// if sysinfo reports 0 (macOS undercounts).
fn default_max_total_workers() -> usize {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let available = sys.available_memory(); // bytes
    if available == 0 {
        return 4; // sysinfo couldn't read memory
    }
    let computed = (available / RAM_PER_WORKER_BYTES) as usize;
    computed.clamp(2, ABSOLUTE_MAX_TOTAL_WORKERS)
}

/// Configuration for the worker pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Path to the Python executable.
    pub python_path: String,
    /// Seconds between health checks.
    pub health_check_interval_s: u64,
    /// Seconds of inactivity before a worker is shut down.
    pub idle_timeout_s: u64,
    /// Maximum seconds to wait for a worker to become ready.
    pub ready_timeout_s: u64,
    /// Use test-echo mode for all workers (no ML models).
    pub test_echo: bool,
    /// Maximum workers per `(profile, lang)` key. Default: 8.
    /// The pool is the capacity ceiling; the runner controls per-job
    /// concurrency via a semaphore.
    pub max_workers_per_key: usize,
    /// Hard ceiling on total workers across all keys. Prevents OOM when
    /// many different `(profile, lang, engine_overrides)` keys are active
    /// simultaneously (e.g. multi-language test suites, concurrent jobs).
    /// Default: computed from available RAM / 4GB per worker, capped at 32.
    /// 0 = use computed default.
    pub max_total_workers: usize,
    /// Maximum seconds `checkout()` may park when the pool is saturated
    /// and no idle worker is available to evict. 0 = use built-in
    /// default (300s).
    pub checkout_wait_timeout_s: u64,
    /// Verbosity level forwarded to Python workers (0=warn, 1=info, 2=debug).
    pub verbose: u8,
    /// Engine overrides as a JSON string, passed to every spawned worker via
    /// `--engine-overrides`. Empty string means no overrides.
    pub engine_overrides: String,
    /// Runtime-owned worker launch inputs (device policy, injected creds).
    pub runtime: WorkerRuntimeConfig,
    /// Timeout override for audio-heavy tasks (ASR, FA, speaker).
    /// 0 = use built-in default (1800).
    pub audio_task_timeout_s: u64,
    /// Timeout override for lightweight analysis tasks (OpenSMILE, AVQI).
    /// 0 = use built-in default (120).
    pub analysis_task_timeout_s: u64,
    /// Timeout in seconds for on-demand model loading via `ensure_task`.
    /// 0 = use built-in default (120).
    pub ensure_task_timeout_s: u64,
    /// Path to the worker registry file. Empty = default
    /// (`~/.batchalign3/workers.json`).
    pub worker_registry_path: String,
    /// Test-only: artificial delay in milliseconds before each worker response.
    /// 0 = no delay. Only effective when `test_echo` is also true. Plumbed to
    /// every spawned worker's `WorkerConfig.test_delay_ms`. Used by the
    /// gpu-concurrent-dispatch tests to assert that the per-request timeout
    /// budget is not consumed by queue-wait when many callers contend for one
    /// shared GPU worker process.
    pub test_delay_ms: u64,
}

/// Built-in default for `ensure_task` timeout (seconds).
const DEFAULT_ENSURE_TASK_TIMEOUT_S: u64 = 120;

/// Built-in default for `checkout_wait_timeout` (seconds).
///
/// Matches the morphosyntax batch's `group_timeout`: a checkout that has
/// waited 5 minutes with the pool saturated and no idle worker to evict is
/// operationally a stall, and failing here lets the orchestrator surface
/// a per-file error instead of hanging indefinitely.
const DEFAULT_CHECKOUT_WAIT_TIMEOUT_S: u64 = 300;

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            python_path: resolve_python_executable(),
            health_check_interval_s: 30,
            idle_timeout_s: 600, // 10 minutes
            ready_timeout_s: 300,
            test_echo: false,
            max_workers_per_key: DEFAULT_MAX_WORKERS_PER_KEY,
            max_total_workers: 0,       // 0 = use computed default
            checkout_wait_timeout_s: 0, // 0 = use built-in default (300s)
            verbose: 0,
            engine_overrides: String::new(),
            runtime: WorkerRuntimeConfig::default(),
            audio_task_timeout_s: 0,
            analysis_task_timeout_s: 0,
            ensure_task_timeout_s: 0,
            worker_registry_path: String::new(),
            test_delay_ms: 0,
        }
    }
}

impl PoolConfig {
    /// Resolved global worker cap: uses `max_total_workers` if nonzero,
    /// otherwise computes from available system memory.
    pub fn effective_max_total_workers(&self) -> usize {
        if self.max_total_workers > 0 {
            self.max_total_workers
        } else {
            default_max_total_workers()
        }
    }

    /// Resolved `ensure_task` timeout: uses the configured value if nonzero,
    /// otherwise falls back to the built-in default (120s).
    pub fn effective_ensure_task_timeout_s(&self) -> u64 {
        if self.ensure_task_timeout_s > 0 {
            self.ensure_task_timeout_s
        } else {
            DEFAULT_ENSURE_TASK_TIMEOUT_S
        }
    }

    /// Resolved checkout wait timeout as a `Duration`.
    pub(super) fn checkout_wait_timeout(&self) -> std::time::Duration {
        let secs = if self.checkout_wait_timeout_s > 0 {
            self.checkout_wait_timeout_s
        } else {
            DEFAULT_CHECKOUT_WAIT_TIMEOUT_S
        };
        std::time::Duration::from_secs(secs)
    }
}

// ---------------------------------------------------------------------------
// WorkerGroup — per (profile, lang) key
// ---------------------------------------------------------------------------

/// A group of workers for a single `(profile, lang)` key.
///
/// Each group independently tracks its own pool of workers. Workers are
/// spawned lazily on first demand and capped at `max_workers_per_key`.
/// The group uses a split concurrency model: a semaphore for async
/// waiting and a mutex for the actual worker queue, so the mutex is
/// never held across an `.await` point.
pub(super) struct WorkerGroup {
    /// Owned worker handles that are currently idle (not checked out).
    ///
    /// Protected by a `std::sync::Mutex` (not `tokio::sync::Mutex`)
    /// because it is held only for the duration of a `push_back` or
    /// `pop_front` -- microseconds, never across an `.await`. This avoids
    /// the overhead of a tokio-aware mutex and is safe because the
    /// critical section cannot yield.
    pub(super) idle: std::sync::Mutex<VecDeque<WorkerHandle>>,

    /// Semaphore with one permit per idle worker.
    ///
    /// `checkout()` acquires a permit (blocking asynchronously if all
    /// workers are busy), then pops from `idle`. When a `CheckedOutWorker`
    /// is dropped, it pushes the worker back into `idle` and adds a
    /// permit, waking the next waiter. Permits are managed manually
    /// (`.forget()` after acquire, `.add_permits(1)` on return) rather
    /// than via RAII `SemaphorePermit` guards.
    pub(super) available: Semaphore,

    /// TCP worker handles discovered from the registry. These are
    /// persistent daemons that survive server restarts.
    pub(super) tcp_workers: std::sync::Mutex<VecDeque<TcpWorkerHandle>>,

    /// Semaphore with one permit per idle TCP worker.
    pub(super) tcp_available: Semaphore,

    /// Total number of live workers in this group: idle + checked-out
    /// (both stdio and TCP).
    ///
    /// `AtomicUsize` so that `worker_count()` and spawn-cap checks can
    /// read it without acquiring any mutex. Incremented in
    /// `try_claim_spawn_slot()` (via `compare_exchange`) before the
    /// worker is spawned, and decremented when a worker is removed
    /// (idle timeout, health failure, or `CheckedOutWorker::take()`).
    pub(super) total: AtomicUsize,

    /// Serialize worker bootstrap for one key.
    ///
    /// This prevents a burst of concurrent requests from launching multiple
    /// heavy Python workers for the same `(profile, lang, engine_overrides)`
    /// bucket at once, which smooths model-loading spikes without changing the
    /// eventual steady-state concurrency of the pool.
    pub(super) bootstrap: AsyncMutex<()>,

    /// Clone of [`WorkerPool::worker_returned`] so `CheckedOutWorker::drop`
    /// can wake saturated waiters without holding a pool reference. All
    /// groups share the same `Notify` instance.
    pub(super) worker_returned: Arc<tokio::sync::Notify>,
}

impl WorkerGroup {
    pub(super) fn new(worker_returned: Arc<tokio::sync::Notify>) -> Self {
        Self {
            idle: std::sync::Mutex::new(VecDeque::new()),
            available: Semaphore::new(0),
            tcp_workers: std::sync::Mutex::new(VecDeque::new()),
            tcp_available: Semaphore::new(0),
            total: AtomicUsize::new(0),
            bootstrap: AsyncMutex::new(()),
            worker_returned,
        }
    }
}

/// Shared map of worker groups, accessible from both the pool and background tasks.
pub(super) type GroupsMap = Arc<std::sync::Mutex<HashMap<WorkerKey, Arc<WorkerGroup>>>>;

// ---------------------------------------------------------------------------
// WorkerPool
// ---------------------------------------------------------------------------

/// Key for shared GPU workers: (target, lang, engine_overrides).
pub(super) type GpuWorkerKey = (WorkerTarget, WorkerLanguage, String);

/// Manages a pool of Python worker processes.
pub struct WorkerPool {
    pub(super) config: PoolConfig,
    /// Sequential worker groups (Stanza, IO profiles).
    pub(super) groups: GroupsMap,
    /// Shared GPU workers for concurrent V2 dispatch (GPU profile, stdio).
    pub(super) gpu_workers:
        Arc<tokio::sync::Mutex<HashMap<GpuWorkerKey, Arc<shared_gpu::SharedGpuWorker>>>>,
    /// Shared GPU workers discovered from registry (TCP transport).
    pub(super) gpu_tcp_workers:
        Arc<tokio::sync::Mutex<HashMap<GpuWorkerKey, Arc<shared_gpu::SharedGpuTcpWorker>>>>,
    /// Pulsed on every `CheckedOutWorker::drop`. Parks saturated
    /// checkouts that have no live worker for their key and no idle
    /// worker elsewhere to evict; wakes them to retry spawn + eviction
    /// when any worker anywhere in the pool becomes available.
    pub(super) worker_returned: Arc<tokio::sync::Notify>,
    pub(super) cancel: CancellationToken,
    /// Background warmup lifecycle state.
    pub(super) warmup_status: AtomicU8,
    /// Lazily detected worker capabilities (populated on first worker spawn).
    pub(super) lazy_capabilities: std::sync::OnceLock<WorkerCapabilities>,
    /// Per-language Stanza processor registry (populated from first worker's
    /// stanza_capabilities field). Used for submission validation and dispatch.
    stanza_registry: std::sync::OnceLock<Box<crate::stanza_registry::StanzaRegistry>>,
    /// Side-table mapping each active job to the worker PIDs currently
    /// dispatching for it. Populated by `TrackerGuard` instances created
    /// inside the dispatch path; consulted by `shutdown_workers_for_job`
    /// when a cancel arrives. See `pool/job_tracker.rs`.
    pub(crate) job_tracker: job_tracker::JobWorkerTracker,
}

impl WorkerPool {
    /// The host-chosen bootstrap mode for local workers.
    pub fn bootstrap_mode(&self) -> WorkerBootstrapMode {
        self.config.runtime.bootstrap_mode
    }

    /// Create a new worker pool. Call [`start_background_tasks`](Self::start_background_tasks)
    /// to begin health checking and idle timeout.
    ///
    /// On creation, reaps any orphaned worker processes left by crashed or
    /// killed servers (Layer 3 of the OOM defense).
    pub fn new(mut config: PoolConfig) -> Self {
        if config.runtime.server_instance_id.is_none() {
            config.runtime.server_instance_id = Some(Uuid::new_v4().simple().to_string());
        }
        if config.runtime.server_process_id.is_none() {
            config.runtime.server_process_id = Some(std::process::id());
        }

        let effective_cap = config.effective_max_total_workers();
        info!(
            max_total_workers = effective_cap,
            max_workers_per_key = config.max_workers_per_key,
            server_instance_id = ?config.runtime.server_instance_id,
            "Worker pool created"
        );

        // Layer 3: reap orphans from any previous server that crashed.
        let reaped = reaper::reap_orphaned_workers();
        if reaped > 0 {
            info!(reaped, "Cleaned up orphaned workers from previous server");
        }

        Self {
            config,
            groups: Arc::new(std::sync::Mutex::new(HashMap::new())),
            gpu_workers: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            gpu_tcp_workers: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            worker_returned: Arc::new(tokio::sync::Notify::new()),
            cancel: CancellationToken::new(),
            warmup_status: AtomicU8::new(WarmupStatus::NotStarted.as_u8()),
            lazy_capabilities: std::sync::OnceLock::new(),
            stanza_registry: std::sync::OnceLock::new(),
            job_tracker: job_tracker::JobWorkerTracker::new(),
        }
    }

    /// Cancel-driven worker reaper. Looks up every worker PID currently
    /// dispatching for `job_id` (registered via `TrackerGuard` inside
    /// the dispatch path), sends SIGTERM, waits a brief grace, and
    /// SIGKILLs any survivors. The worker process dies, its IPC future
    /// errors out (BrokenPipe / ProcessExited), the dispatch caller
    /// propagates the error, and the runner sees cancel at the next
    /// loop iteration — instead of waiting for the in-flight call
    /// (which can be 8-25 minutes for a Whisper-CPU pass on a long
    /// audio file) to complete naturally.
    ///
    /// Worker PIDs currently dispatching for `job_id`. Narrow
    /// read-only accessor for integration tests that need to wait
    /// for dispatch registration before firing a cancel — the
    /// alternative (sleeping a fixed duration) is flaky on slow
    /// hosts. Runtime callers use `shutdown_workers_for_job`.
    #[doc(hidden)]
    pub fn workers_for_job(&self, job_id: &crate::api::JobId) -> Vec<crate::worker::WorkerPid> {
        self.job_tracker.snapshot(job_id)
    }

    /// Run a future under the `CURRENT_JOB_ID` task-local scope so
    /// dispatch-site `TrackerGuard`s register against `job_id`.
    /// Production code uses `runner::execution::run_server_job_attempt`
    /// to set the scope; this is the integration-test entry point.
    #[doc(hidden)]
    pub async fn dispatch_under_job_for_test<Fut, T>(job_id: crate::api::JobId, fut: Fut) -> T
    where
        Fut: std::future::Future<Output = T>,
    {
        job_tracker::CURRENT_JOB_ID.scope(job_id, fut).await
    }

    /// SIGTERM every worker dispatching for `job_id`, escalating to
    /// SIGKILL after a 2s grace. Long enough for a well-behaved
    /// Python worker to handle the signal cleanly, short enough
    /// that interactive cancel feels responsive.
    pub async fn shutdown_workers_for_job(&self, job_id: &crate::api::JobId) {
        let pids = self.job_tracker.drain(job_id);
        if pids.is_empty() {
            return;
        }
        tracing::info!(
            job_id = %job_id,
            worker_count = pids.len(),
            "Cancel: terminating in-flight workers for job"
        );
        job_tracker::signal_workers(&pids, std::time::Duration::from_secs(2)).await;
    }

    /// Maximum workers allowed per `(target, lang, engine_overrides)` key.
    ///
    /// Used by the morphosyntax batch dispatcher to decide how many chunks
    /// to split a single-language batch into for concurrent inference.
    pub fn max_workers_per_key(&self) -> usize {
        self.config.max_workers_per_key
    }

    /// Resolved global worker cap (computed from RAM if not explicitly set).
    ///
    /// Used by the morphosyntax batch dispatcher to bound the number of
    /// concurrent language groups, preventing deadlock when the number of
    /// languages × workers_per_key exceeds the global cap.
    pub fn effective_max_total_workers(&self) -> usize {
        self.config.effective_max_total_workers()
    }

    /// Access the per-language Stanza processor registry.
    ///
    /// Returns `None` until the first worker reports its capabilities.
    pub fn stanza_registry(&self) -> Option<&crate::stanza_registry::StanzaRegistry> {
        self.stanza_registry.get().map(|b| b.as_ref())
    }

    /// Record worker capabilities and populate the Stanza registry.
    pub(super) fn record_capabilities(&self, caps: WorkerCapabilities) {
        if !caps.stanza_capabilities.is_empty() && self.stanza_registry.get().is_none() {
            let _ = self.stanza_registry.set(Box::new(
                crate::stanza_registry::StanzaRegistry::from_capabilities(
                    &caps.stanza_capabilities,
                ),
            ));
        }
        let _ = self.lazy_capabilities.set(caps);
    }

    /// Get or create a shared GPU worker for the given (lang, engine_overrides).
    ///
    /// Holds the lock across the spawn to prevent the TOCTOU race where
    /// multiple concurrent callers each spawn their own worker process.
    /// The spawn includes waiting for the `{"ready": true}` signal, so
    /// the lock is held for 10-30 seconds on first call. This is acceptable
    /// because GPU worker creation is rare (once per lang+overrides combo),
    /// and the `pre_scale` call in the runner ensures the worker exists
    /// before file dispatch begins.
    pub(super) async fn get_or_create_gpu_worker(
        &self,
        target: &WorkerTarget,
        lang: &WorkerLanguage,
        engine_overrides: &str,
    ) -> Result<Arc<shared_gpu::SharedGpuWorker>, WorkerError> {
        let key = (*target, lang.clone(), engine_overrides.to_owned());

        let mut gpu_workers = self.gpu_workers.lock().await;

        // Fast path: worker already exists.
        if let Some(worker) = gpu_workers.get(&key) {
            return Ok(worker.clone());
        }

        // Slow path: spawn while holding the lock to prevent duplicate spawns.
        let config = WorkerConfig {
            python_path: self.config.python_path.clone(),
            profile: target.profile_kind(),
            task: target.task(),
            lang: lang.clone(),
            num_speakers: NumSpeakers(1),
            engine_overrides: engine_overrides.to_owned(),
            test_echo: self.config.test_echo,
            ready_timeout_s: self.config.ready_timeout_s,
            verbose: self.config.verbose,
            runtime: self.config.runtime.clone(),
            audio_task_timeout_s: self.config.audio_task_timeout_s,
            analysis_task_timeout_s: self.config.analysis_task_timeout_s,
            test_delay_ms: self.config.test_delay_ms,
        };

        let mut handle = WorkerHandle::spawn(config).await?;
        if self.lazy_capabilities.get().is_none()
            && let Err(e) = self.detect_capabilities_from_worker(&mut handle).await
        {
            tracing::warn!(error = %e, "Failed to detect capabilities from first GPU worker (continuing)");
        }
        info!(
            target = %target.label(),
            lang = %lang,
            pid = %handle.pid(),
            "GPU worker spawned (concurrent mode)"
        );
        let shared = Arc::new(shared_gpu::SharedGpuWorker::from_handle(handle).await);

        gpu_workers.insert(key, shared.clone());
        Ok(shared)
    }

    /// Query capabilities from an already-spawned worker and cache the result.
    ///
    /// Called once after the first worker spawn. The `OnceLock` ensures this
    /// only runs once even under concurrent job dispatch.
    pub(crate) async fn detect_capabilities_from_worker(
        &self,
        handle: &mut WorkerHandle,
    ) -> Result<(), WorkerError> {
        if self.lazy_capabilities.get().is_some() {
            return Ok(()); // Already detected
        }

        let caps = handle.capabilities().await?;
        info!(
            source = "spawned-worker",
            infer_tasks = ?caps.infer_tasks,
            engine_versions = ?caps.engine_versions,
            "Recorded detected worker capabilities"
        );
        self.record_capabilities(caps);
        Ok(())
    }

    /// Return lazily detected capabilities, or `None` if no worker has
    /// spawned yet.
    pub fn detected_capabilities(&self) -> Option<&WorkerCapabilities> {
        self.lazy_capabilities.get()
    }

    /// The server instance ID assigned at pool creation.
    pub(super) fn current_server_instance_id(&self) -> &str {
        self.config
            .runtime
            .server_instance_id
            .as_deref()
            .expect("WorkerPool::new always assigns a server instance id")
    }
}

// V2 execute key resolution helpers live in execute_v2.rs, used by dispatch.rs.
