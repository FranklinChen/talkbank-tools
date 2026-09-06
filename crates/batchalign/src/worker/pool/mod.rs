//! `WorkerPool`: manages multiple Python worker processes.
//!
//! Workers are keyed by bootstrap target, language, and typed engine selection.
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
//! where a tokio mutex was held for 10-300 seconds during dispatch.
//!
//! ## Module layout
//!
//! | File | Responsibility |
//! |------|----------------|
//! | `mod.rs` | Types, pool struct, construction, accessors, GPU worker creation |
//! | `checkout.rs` | `CheckedOutWorker` RAII guard |
//! | `dispatch.rs` | Checkout loop, batch infer, V2 execute, TCP routing |
//! | `discovery.rs` | Registry-based TCP worker discovery |
//! | `pre_scale.rs` | Eager per-job worker pre-spawning |
//! | `shutdown.rs` | Graceful shutdown and `Drop` cleanup |
//! | `lifecycle.rs` | Background health checking, idle timeout, spawn helpers |
//! | `execute_v2.rs` | V2 request key resolution helpers |
//! | `status.rs` | Status query methods (has_idle_workers, worker_count, etc.) |
//! | `shared_gpu/` | Shared concurrent GPU worker wrappers (stdio, TCP, reader) |
//! | `reaper.rs` | Orphaned worker process reaping |

mod checkout;
mod cpu_gate;
mod discovery;
mod dispatch;
mod eviction;
mod execute_v2;
pub(super) mod gpu_slot;
mod idle_eviction;
pub(crate) mod job_tracker;
mod lifecycle;
pub(crate) mod memory_gate;
mod permit;
mod pre_scale;
pub(crate) mod reaper;
mod rss_observer;
pub(crate) mod shared_gpu;
mod shutdown;
pub mod status;

pub use checkout::CheckedOutWorker;
pub use status::{WorkerSummaryEntry, WorkerTransport};

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::api::{ReleasedCommand, WorkerLanguage};
use crate::host_facts::PerProfile;
use crate::options::CommandOptions;
use crate::types::engines::{EngineOverrides, FaEngineName, SelectableEngine};
use crate::worker::{WorkerBootstrapMode, WorkerCapabilities, WorkerProfile, WorkerTarget};
use batchalign_types::worker::InferTask;
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
use tokio_util::sync::CancellationToken;
use tracing::{info, instrument, warn};
use uuid::Uuid;

use self::gpu_slot::GpuWorkerSlot;
use crate::worker::error::WorkerError;
use crate::worker::handle::{WorkerHandle, WorkerRuntimeConfig};
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

/// A typed engine selection for one worker process.
///
/// The pool never accepts arbitrary JSON as an identity. It derives this value
/// from a typed execute request or command options, then serializes it only
/// when crossing to a Python worker. Registry JSON is parsed at that external
/// boundary before it reaches the pool map.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub(super) struct EngineSelection(EngineOverrides);

impl EngineSelection {
    /// No engine-specific worker configuration.
    pub(super) fn none() -> Self {
        Self::default()
    }

    /// The engines a command itself names, before any target-driven filling.
    ///
    /// Split out so `for_target` can add to it and normalize ONCE.
    /// Normalizing, unwrapping and re-normalizing left a reader unable to tell
    /// whether the second pass was required or incidental.
    fn overrides_for_command(options: &CommandOptions) -> EngineOverrides {
        let mut overrides = options.common().engine_overrides.clone();
        match options {
            CommandOptions::Align(options) => {
                overrides
                    .fa
                    .get_or_insert_with(|| options.effective_fa_engine());
            }
            CommandOptions::Transcribe(options) | CommandOptions::TranscribeS(options) => {
                overrides
                    .asr
                    .get_or_insert_with(|| options.effective_asr_engine());
            }
            CommandOptions::Benchmark(options) => {
                overrides
                    .asr
                    .get_or_insert_with(|| options.effective_asr_engine());
            }
            CommandOptions::Translate(options) => {
                overrides
                    .translate
                    .get_or_insert_with(|| options.effective_translate_engine());
            }
            CommandOptions::Morphotag(_)
            | CommandOptions::Coref(_)
            | CommandOptions::Utseg(_)
            | CommandOptions::Opensmile(_)
            | CommandOptions::Compare(_)
            | CommandOptions::Avqi(_)
            | CommandOptions::Diarize(_)
            // No engine override: the embedding model is the one node of the
            // released local model graph, and there is nothing to select
            // between.
            | CommandOptions::SpeakerIdentify(_) => {}
        }
        overrides
    }

    /// The selection a worker for `target` must carry.
    ///
    /// A worker loads models for every task in its TARGET's bundle, while the
    /// overrides describe only what its COMMAND selected. Those are different
    /// sets: a transcribe command on the GPU profile preloads forced alignment
    /// while naming only an ASR engine, so the worker reached the FA loader
    /// with no engine named and had to guess.
    ///
    /// Only forced alignment is filled in, and the asymmetry is the point.
    /// Rust owns the FA default: the Python side has no way to choose between
    /// two models that report different SHAPES of timing, which is why its
    /// guess there was deleted rather than corrected. ASR and translation are
    /// the opposite. `resolve_asr_engine` picks per LANGUAGE (`yue` takes
    /// FunAudio, because Whisper measured worst on it) and prefers Rev.AI only
    /// when a key is actually present, and it reads any `asr` key as an
    /// explicit operator instruction. Injecting a default here would therefore
    /// silently overrule a better-informed choice: an align job, which never
    /// asked for ASR at all, would pin the ASR model for the whole worker.
    ///
    /// Lazy-profile workers preload nothing but still retain the command's
    /// selection in their identity. The selected recipe is loaded on demand;
    /// a worker keyed for one engine cannot silently serve another.
    pub(super) fn for_target(target: WorkerTarget, options: &CommandOptions) -> Self {
        Self::from_overrides(Self::fill_for_preloaded_tasks(
            target,
            Self::overrides_for_command(options),
        ))
    }

    /// The single audio-engine recipe a lazy command capability probe loads.
    ///
    /// Lazy workers do not preload their profile siblings, so an ASR probe
    /// must not acquire an unrelated FA selection and vice versa. Non-audio
    /// primary tasks retain their existing command-specific recipe;
    /// their V2 requests do not yet carry a backend identity to project here.
    fn for_lazy_command(command: ReleasedCommand, options: &CommandOptions) -> Self {
        let selected = Self::overrides_for_command(options);
        match crate::command_model::command_spec(command)
            .capabilities
            .primary_infer_task
        {
            InferTask::Asr => Self::from_overrides(EngineOverrides {
                asr: selected.asr,
                extras: selected.extras,
                ..EngineOverrides::default()
            }),
            InferTask::Fa => Self::from_overrides(EngineOverrides {
                fa: selected.fa,
                ..EngineOverrides::default()
            }),
            InferTask::Morphosyntax
            | InferTask::Utseg
            | InferTask::Translate
            | InferTask::Coref
            | InferTask::Opensmile
            | InferTask::Avqi
            | InferTask::Speaker => Self::for_target(
                WorkerTarget::for_command_with_mode(command, WorkerBootstrapMode::LazyProfile),
                options,
            ),
        }
    }

    /// Name an engine for every task the target's bootstrap will PRELOAD,
    /// beyond what the command or request itself named.
    ///
    /// One owner for a guarantee two spawn routes rely on: the command route
    /// (`for_target`) and the V2 execute route
    /// (`from_execute_request_for_target`) both spawn eager profile workers,
    /// and the Python bootstrap REFUSES a preloaded task with no named engine
    /// rather than defaulting (`resolve_fa_engine`). Until 2026-08-21 only
    /// the command route filled this in, so an eager-mode V2 ASR request
    /// spawned a profile worker that preloaded FA with no `fa` override and
    /// died at bootstrap ("no 'fa' engine in overrides").
    pub(super) fn fill_for_preloaded_tasks(
        target: WorkerTarget,
        mut overrides: EngineOverrides,
    ) -> EngineOverrides {
        if target.preloaded_tasks().contains(&InferTask::Fa) {
            overrides.fa.get_or_insert(FaEngineName::DEFAULT);
        }
        overrides
    }

    /// Parse the JSON held by an externally managed registry entry.
    ///
    /// This is ingress from the Python registry boundary, not another source
    /// of pool-key derivation. Once admitted, equality is structural
    /// [`EngineOverrides`] equality rather than spelling-sensitive JSON.
    pub(super) fn from_registry_json(json: &str) -> Result<Self, serde_json::Error> {
        if json.trim().is_empty() {
            return Ok(Self::none());
        }
        serde_json::from_str(json).map(Self::from_overrides)
    }

    /// JSON for the Rust/Python worker configuration boundary.
    pub(super) fn worker_config_json(&self) -> String {
        self.0.to_dispatch_json_string()
    }

    /// Typed engine details consumed by Rust-side admission logic.
    pub(super) fn overrides(&self) -> &EngineOverrides {
        &self.0
    }

    pub(super) fn is_none(&self) -> bool {
        self.0.is_empty()
    }

    /// Collapse engine choices that never reach a Python worker while
    /// preserving extras, which may still configure another selected engine.
    fn from_overrides(mut overrides: EngineOverrides) -> Self {
        if overrides
            .asr
            .as_ref()
            .is_some_and(|engine| engine.dispatch_override_name().is_none())
        {
            overrides.asr = None;
        }
        Self(overrides)
    }
}

impl std::fmt::Display for EngineSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.worker_config_json())
    }
}

/// Identity of one worker group.
///
/// The three components are named so an engine selection cannot be confused
/// with an anonymous third tuple field. Constructors derive the selection from
/// typed inputs; no caller can key the pool with a raw JSON string.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct WorkerKey {
    pub(super) target: WorkerTarget,
    pub(super) language: WorkerLanguage,
    pub(super) engine_selection: EngineSelection,
}

impl WorkerKey {
    /// A key with no engine selection.
    ///
    /// This is the door `for_target` closes only halfway, and it cannot be
    /// shut from here. Nothing in production reaches it, but its two callers,
    /// `WorkerPool::pre_scale` and `dispatch_batch_infer`, are ordinary `pub`
    /// functions whose own callers are all tests, so `#[cfg(test)]` on this
    /// constructor does not compile. Retiring it means retiring the V1
    /// batch-infer path and the untyped pre-scale helper, which is a larger
    /// change than adding the type it undermines.
    ///
    /// Until then: every key a real job uses comes from
    /// `from_command_options` or `from_registry_json`.
    pub(super) fn without_engine_selection(target: WorkerTarget, language: WorkerLanguage) -> Self {
        Self {
            target,
            language,
            engine_selection: EngineSelection::none(),
        }
    }

    pub(super) fn from_command_options(
        command: ReleasedCommand,
        language: WorkerLanguage,
        options: &CommandOptions,
        bootstrap_mode: WorkerBootstrapMode,
    ) -> Self {
        let target = WorkerTarget::for_command_with_mode(command, bootstrap_mode);
        let engine_selection = match bootstrap_mode {
            WorkerBootstrapMode::LazyProfile => EngineSelection::for_lazy_command(command, options),
            WorkerBootstrapMode::Profile | WorkerBootstrapMode::Task => {
                EngineSelection::for_target(target, options)
            }
        };
        Self {
            target,
            language,
            engine_selection,
        }
    }

    pub(super) fn from_registry_json(
        target: WorkerTarget,
        language: WorkerLanguage,
        engine_overrides_json: &str,
    ) -> Result<Self, serde_json::Error> {
        Ok(Self {
            target,
            language,
            engine_selection: EngineSelection::from_registry_json(engine_overrides_json)?,
        })
    }
}

/// Default per-profile `max_workers_per_key` for `PoolConfig::default()`
/// test fixtures.
///
/// Production sets this from
/// `EffectiveConfig::max_workers_per_key_by_profile` via
/// `cli/serve_cmd.rs`. Test fixtures get a uniform value across all
/// three profiles; profile-aware tests construct their own
/// `PerProfile` literal.
const DEFAULT_MAX_WORKERS_PER_KEY: PerProfile<usize> = PerProfile {
    gpu: 4,
    stanza: 4,
    io: 4,
};

/// Default `max_total_workers` for `PoolConfig::default()` test fixtures.
///
/// Production sets this from `EffectiveConfig::max_total_workers` (the
/// host-facts recommendation, RAM-derived, clamped to `[2, 32]`). Tests
/// use a concrete constant rather than a sysinfo probe so behavior is
/// deterministic across machines.
const DEFAULT_MAX_TOTAL_WORKERS_FOR_TESTS: usize = 32;

/// Configuration for the worker pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Path to the Python executable.
    pub python_path: String,
    /// Seconds between health checks.
    pub health_check_interval_s: u64,
    /// Maximum seconds to wait for a worker to become ready.
    pub ready_timeout_s: u64,
    /// Use test-echo mode for all workers (no ML models).
    pub test_echo: bool,
    /// Maximum workers per `(profile, lang)` key, indexed by the
    /// requested key's `WorkerProfile`. The pool is the capacity
    /// ceiling; the runner controls per-job concurrency via a
    /// semaphore. Production sets this from
    /// `EffectiveConfig::max_workers_per_key_by_profile`; test
    /// fixtures get a uniform per-profile literal.
    pub max_workers_per_key: PerProfile<usize>,
    /// Hard ceiling on total workers across all keys. Prevents OOM when
    /// many different `(profile, lang, engine selection)` keys are active
    /// simultaneously (e.g. multi-language test suites, concurrent jobs).
    /// Production sets this from `EffectiveConfig::max_total_workers`
    /// (host-facts-derived); test fixtures get
    /// `DEFAULT_MAX_TOTAL_WORKERS_FOR_TESTS`. Must always be a concrete
    /// nonzero value.
    pub max_total_workers: usize,
    /// Maximum seconds `checkout()` may park when the pool is saturated
    /// and no idle worker is available to evict. 0 = use built-in
    /// default (300s).
    pub checkout_wait_timeout_s: u64,
    /// Verbosity level forwarded to Python workers (0=warn, 1=info, 2=debug).
    pub verbose: u8,
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
    /// Test-only override for the CPU-loadavg admission threshold.
    ///
    /// Production leaves this `None`; the CPU gate uses
    /// `cpu_gate::host_cpu_count_as_threshold()` (live
    /// `available_parallelism()`). Tests that need to exercise the
    /// permit / per-key-cap / metrics paths set this to a value
    /// far above any realistic load (e.g. `f64::INFINITY`) so
    /// `try_claim_spawn_slot` does not reject at Layer 0 on
    /// CPU-saturated CI runners. Setting a finite override lets the
    /// `cpu_gate` tests in [`super::cpu_gate`] cover both branches.
    pub cpu_gate_threshold_override: Option<f64>,
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
            ready_timeout_s: 300,
            test_echo: false,
            max_workers_per_key: DEFAULT_MAX_WORKERS_PER_KEY,
            max_total_workers: DEFAULT_MAX_TOTAL_WORKERS_FOR_TESTS,
            checkout_wait_timeout_s: 0, // 0 = use built-in default (300s)
            verbose: 0,
            runtime: WorkerRuntimeConfig::default(),
            audio_task_timeout_s: 0,
            analysis_task_timeout_s: 0,
            ensure_task_timeout_s: 0,
            worker_registry_path: String::new(),
            test_delay_ms: 0,
            cpu_gate_threshold_override: None,
        }
    }
}

impl PoolConfig {
    /// Resolved global worker cap. Returns `self.max_total_workers`
    /// directly; production fills it from `EffectiveConfig`, tests
    /// from `DEFAULT_MAX_TOTAL_WORKERS_FOR_TESTS` via
    /// `PoolConfig::default()`.
    pub fn effective_max_total_workers(&self) -> usize {
        self.max_total_workers
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
// WorkerGroup, per (profile, lang) key
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

    /// Clone of [`WorkerPool::spawn_permits`] so worker-exit paths
    /// (`CheckedOutWorker::take`, eviction, the health-check reaper,
    /// shutdown drain) can refund global-cap admission slots without
    /// reaching back to the owning [`WorkerPool`]. All groups share
    /// the same `Semaphore` instance, it represents the pool-wide
    /// `max_total_workers` budget, not per-key capacity. Released
    /// once per `group.total.fetch_sub(1)` increment.
    pub(super) spawn_permits: Arc<Semaphore>,

    /// Worker profile for this group's key. Recovered from
    /// `WorkerTarget::profile_kind()` at construction; stable for the
    /// lifetime of the group. Read by admission control to look up
    /// the per-profile slot in `PoolConfig::max_workers_per_key`.
    pub(super) profile: WorkerProfile,

    /// Typed engine selection for this group's key. Stored on the group so admission control can
    /// consult the engine selection without re-locating the key. Used
    /// by the engine-aware memory reservation
    /// ([`super::memory_gate::engine_aware_startup_reservation_mb`])
    /// to inflate the per-spawn reservation when the translate engine
    /// is a heavy local-model backend (NLLB, SeamlessM4T).
    pub(super) engine_selection: EngineSelection,
}

impl WorkerGroup {
    pub(super) fn new(
        worker_returned: Arc<tokio::sync::Notify>,
        spawn_permits: Arc<Semaphore>,
        profile: WorkerProfile,
        engine_selection: EngineSelection,
    ) -> Self {
        Self {
            idle: std::sync::Mutex::new(VecDeque::new()),
            available: Semaphore::new(0),
            tcp_workers: std::sync::Mutex::new(VecDeque::new()),
            tcp_available: Semaphore::new(0),
            total: AtomicUsize::new(0),
            bootstrap: AsyncMutex::new(()),
            worker_returned,
            spawn_permits,
            profile,
            engine_selection,
        }
    }

    /// Record that `n` workers were removed from this group: decrement
    /// `total` and refund `n` global-cap permits in lockstep, so
    /// steady-state holds (`sum(group.total) + permits_available ==
    /// max_total_workers`). Single source of truth for the
    /// removal-side accounting; called by every eviction path
    /// (time-based, pressure-driven, health-check).
    pub(super) fn record_worker_removed(&self, n: usize) {
        if n == 0 {
            return;
        }
        self.total.fetch_sub(n, Ordering::Relaxed);
        permit::SpawnPermitGuard::release_n(&self.spawn_permits, n);
    }

    /// True when this group has no workers (live, idle, or in-flight
    /// for spawn). Drives the cold-start vs warm distinction at the
    /// admission gates (see `memory_gate::PoolGateState`) and the
    /// eviction-vs-wait branch in `dispatch::checkout`. The atomic
    /// load is `Relaxed` because the gate decision is intentionally
    /// race-tolerant: a concurrent spawn that flips the answer
    /// between this load and the downstream `compare_exchange`
    /// re-races at the CAS, and the gates themselves are stateless
    /// so a stale ColdStart classification cannot cause incorrect
    /// admission.
    pub(super) fn is_empty(&self) -> bool {
        self.total.load(Ordering::Relaxed) == 0
    }
}

/// Shared map of worker groups, accessible from both the pool and background tasks.
pub(super) type GroupsMap = Arc<std::sync::Mutex<HashMap<WorkerKey, Arc<WorkerGroup>>>>;

// ---------------------------------------------------------------------------
// WorkerPool
// ---------------------------------------------------------------------------

/// Monotonic rejection counters owned as one operational subsystem.
///
/// These values share lifecycle, memory ordering, and their sole consumer
/// (`metrics_snapshot`). Keeping them together prevents every new admission
/// reason from widening `WorkerPool` itself.
#[derive(Default)]
pub(super) struct WorkerRejectionCounters {
    /// Per-key worker-cap refusals.
    per_key_cap: AtomicU64,
    /// Refusals by the older in-spawn memory guard.
    memory_guard: AtomicU64,
    /// Refusals to acquire a global worker permit.
    global_permit: AtomicU64,
    /// Refusals while the host CPU is saturated.
    cpu_saturation: AtomicU64,
    /// Refusals while available host memory is below the hard floor.
    memory_constrained: AtomicU64,
}

/// Closed reason set for mutating worker-rejection telemetry.
#[derive(Debug, Clone, Copy)]
pub(super) enum WorkerRejectionKind {
    PerKeyCap,
    MemoryGuard,
    GlobalPermit,
    CpuSaturation,
    MemoryConstrained,
}

impl WorkerRejectionCounters {
    fn counter(&self, kind: WorkerRejectionKind) -> &AtomicU64 {
        match kind {
            WorkerRejectionKind::PerKeyCap => &self.per_key_cap,
            WorkerRejectionKind::MemoryGuard => &self.memory_guard,
            WorkerRejectionKind::GlobalPermit => &self.global_permit,
            WorkerRejectionKind::CpuSaturation => &self.cpu_saturation,
            WorkerRejectionKind::MemoryConstrained => &self.memory_constrained,
        }
    }

    pub(super) fn record(&self, kind: WorkerRejectionKind) -> u64 {
        self.counter(kind).fetch_add(1, Ordering::Relaxed) + 1
    }

    fn count(&self, kind: WorkerRejectionKind) -> u64 {
        self.counter(kind).load(Ordering::Relaxed)
    }
}

/// Manages a pool of Python worker processes.
pub struct WorkerPool {
    pub(super) config: PoolConfig,
    /// Sequential worker groups (Stanza, IO profiles).
    pub(super) groups: GroupsMap,
    /// Shared GPU workers for concurrent V2 dispatch (GPU profile, stdio).
    ///
    /// The value is a [`GpuWorkerSlot`], not a worker: an entry appears when a
    /// key's spawn STARTS, and holds a worker once that spawn succeeds. The map
    /// lock is therefore never held across a spawn, so a cold key no longer
    /// stalls dispatches to warm keys or `/health`. Readers must state what an
    /// in-flight spawn means for them; see `gpu_slot.rs`.
    pub(super) gpu_workers: Arc<tokio::sync::Mutex<HashMap<WorkerKey, GpuWorkerSlot>>>,
    /// Shared GPU workers discovered from registry (TCP transport).
    pub(super) gpu_tcp_workers:
        Arc<tokio::sync::Mutex<HashMap<WorkerKey, Arc<shared_gpu::SharedGpuTcpWorker>>>>,
    /// Distinct content-addressed Python runtimes observed since server start.
    /// This history remains available while a sequential worker is checked out
    /// and after a crash, unlike an instantaneous idle-worker listing.
    observed_worker_runtimes: Arc<crate::worker::runtime_identity::ObservedWorkerRuntimes>,
    /// Pulsed on every `CheckedOutWorker::drop`. Parks saturated
    /// checkouts that have no live worker for their key and no idle
    /// worker elsewhere to evict; wakes them to retry spawn + eviction
    /// when any worker anywhere in the pool becomes available.
    pub(super) worker_returned: Arc<tokio::sync::Notify>,
    pub(super) cancel: CancellationToken,
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
    /// Monotonic counters for every worker-admission rejection reason.
    pub(super) rejection_counters: WorkerRejectionCounters,
    /// Global-cap admission semaphore. Sized at construction time to
    /// `max_total_workers`. Each live worker (across all groups) holds
    /// exactly one permit for its lifetime; spawn attempts park on
    /// `acquire()` rather than the legacy `worker_returned`
    /// `notify_waiters()` re-probe storm. FIFO-fair: a single permit
    /// release wakes exactly one waiter, eliminating the thundering
    /// herd documented in BUG-028. See `pool/permit.rs` for the RAII
    /// guard that wraps an [`tokio::sync::OwnedSemaphorePermit`].
    pub(super) spawn_permits: Arc<tokio::sync::Semaphore>,
}

/// Read-only snapshot of pool counters. Cheap to compute (one
/// groups-mutex acquisition + a few atomic reads). Consumed by
/// observability surfaces that need ground-truth pool state without
/// reaching into the pool's internals.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct PoolMetrics {
    /// Total number of worker processes alive across every group.
    /// Equals the sum of `WorkerGroup::total` over all groups; this
    /// is the value the global-cap check compares against
    /// `max_total_workers`.
    pub active_workers_total: usize,
    /// Active worker counts split by `WorkerProfile`. Useful for
    /// answering "is the GPU profile saturated even though stanza
    /// has headroom?" without a per-group walk.
    pub active_workers_by_profile: PerProfile<usize>,
    /// Total spawn rejections at the global cap since pool startup.
    /// Monotonic. A high value with low elapsed time signals that
    /// the cap is binding under the current workload.
    pub spawn_rejections_total: u64,
    /// Total host-memory-guard rejections since pool startup.
    /// Monotonic. A high value indicates host RAM pressure has been
    /// repeatedly preventing worker spawns.
    pub memory_gate_rejections_total: u64,
    /// Available permits in the global-cap admission semaphore. Equal
    /// to `max_total_workers - active_workers_total` in steady state
    /// (with brief windows during spawn/shutdown where they differ by
    /// at most one). Reading this is cheap, `Semaphore::available_permits()`.
    pub permits_available: usize,
    /// Total times a spawn attempt was rejected at the global-cap
    /// permit acquisition (post-refactor counterpart to
    /// `spawn_rejections_total`, which after the refactor counts only
    /// per-key cap rejections). Monotonic.
    pub permit_rejections_total: u64,
    /// Total times the live CPU-loadavg gate refused a spawn since
    /// pool startup. Monotonic. A nonzero value reports that the
    /// host itself was at or above CPU saturation when admission
    /// was attempted; no permit-pool unwinding will help. Replaces
    /// the static `max_concurrent_jobs` estimate's role in
    /// preventing CPU oversubscription with a runtime measurement.
    pub cpu_saturation_rejections_total: u64,
    /// Total times the live available-memory gate refused a spawn
    /// since pool startup. Monotonic. A nonzero value reports that
    /// the host's free memory was at or below the hardcoded
    /// minimum-free floor when admission was attempted. Replaces
    /// the role of the tier-derived `recommend_memory_gate_mb`
    /// formula with a single mechanical floor enforced at admission
    /// rather than reactively inside spawn.
    pub memory_constrained_rejections_total: u64,
}

impl WorkerPool {
    /// Test-only artificial per-response delay configured for this pool.
    ///
    /// Exposed so the test-echo dispatch path (which bypasses workers
    /// entirely) can honor the same pacing knob integration tests use to
    /// hold race windows open.
    pub(crate) fn test_delay_ms(&self) -> u64 {
        self.config.test_delay_ms
    }

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
            max_workers_per_key_gpu = config.max_workers_per_key.gpu,
            max_workers_per_key_stanza = config.max_workers_per_key.stanza,
            max_workers_per_key_io = config.max_workers_per_key.io,
            server_instance_id = ?config.runtime.server_instance_id,
            "Worker pool created"
        );

        // Layer 3: reap orphans from any previous server that crashed.
        let reaped = reaper::reap_orphaned_workers();
        if reaped > 0 {
            info!(reaped, "Cleaned up orphaned workers from previous server");
        }

        // Global-cap permit pool: one permit per allowed live worker.
        // Sized once at construction; never grows. Each successful
        // worker spawn consumes one permit (held in the worker's RAII
        // guard for its lifetime); each worker exit releases one.
        let spawn_permits = Arc::new(tokio::sync::Semaphore::new(effective_cap));

        Self {
            config,
            groups: Arc::new(std::sync::Mutex::new(HashMap::new())),
            gpu_workers: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            gpu_tcp_workers: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            observed_worker_runtimes: Arc::new(
                crate::worker::runtime_identity::ObservedWorkerRuntimes::default(),
            ),
            worker_returned: Arc::new(tokio::sync::Notify::new()),
            cancel: CancellationToken::new(),
            lazy_capabilities: std::sync::OnceLock::new(),
            stanza_registry: std::sync::OnceLock::new(),
            job_tracker: job_tracker::JobWorkerTracker::new(),
            rejection_counters: WorkerRejectionCounters::default(),
            spawn_permits,
        }
    }

    /// Snapshot the pool's runtime counters for observability.
    ///
    /// Cheap: acquires the groups mutex briefly to sum per-profile
    /// active counts, then reads the rejection atomics. Counters are
    /// monotonic since pool startup.
    pub fn metrics_snapshot(&self) -> PoolMetrics {
        let mut active_total: usize = 0;
        let mut by_profile = PerProfile {
            gpu: 0usize,
            stanza: 0usize,
            io: 0usize,
        };
        {
            let groups = lock_recovered(&self.groups);
            for group in groups.values() {
                let n = group.total.load(Ordering::Relaxed);
                active_total += n;
                match group.profile {
                    WorkerProfile::Gpu => by_profile.gpu += n,
                    WorkerProfile::Stanza => by_profile.stanza += n,
                    WorkerProfile::Io => by_profile.io += n,
                }
            }
        }
        PoolMetrics {
            active_workers_total: active_total,
            active_workers_by_profile: by_profile,
            spawn_rejections_total: self
                .rejection_counters
                .count(WorkerRejectionKind::PerKeyCap),
            memory_gate_rejections_total: self
                .rejection_counters
                .count(WorkerRejectionKind::MemoryGuard),
            permits_available: self.spawn_permits.available_permits(),
            permit_rejections_total: self
                .rejection_counters
                .count(WorkerRejectionKind::GlobalPermit),
            cpu_saturation_rejections_total: self
                .rejection_counters
                .count(WorkerRejectionKind::CpuSaturation),
            memory_constrained_rejections_total: self
                .rejection_counters
                .count(WorkerRejectionKind::MemoryConstrained),
        }
    }

    fn admit_worker_runtime(&self, worker: &WorkerHandle) -> Result<(), WorkerError> {
        self.observed_worker_runtimes
            .admit(worker.runtime_identity())?;
        Ok(())
    }

    /// Distinct content-addressed worker runtimes observed since server start.
    pub fn observed_worker_runtimes(
        &self,
    ) -> Vec<crate::worker::runtime_identity::WorkerRuntimeIdentity> {
        self.observed_worker_runtimes.receipts()
    }

    /// Cancel-driven worker reaper. Looks up every worker PID currently
    /// dispatching for `job_id` (registered via `TrackerGuard` inside
    /// the dispatch path), sends SIGTERM, waits a brief grace, and
    /// SIGKILLs any survivors. The worker process dies, its IPC future
    /// errors out (BrokenPipe / ProcessExited), the dispatch caller
    /// propagates the error, and the runner sees cancel at the next
    /// loop iteration: instead of waiting for the in-flight call
    /// (which can be 8-25 minutes for a Whisper-CPU pass on a long
    /// audio file) to complete naturally.
    ///
    /// Worker PIDs currently dispatching for `job_id`. Narrow
    /// read-only accessor for integration tests that need to wait
    /// for dispatch registration before firing a cancel, the
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

    /// Maximum workers allowed per `(target, lang, engine_overrides)` key
    /// for the requested profile.
    ///
    /// Used by the morphosyntax batch dispatcher to decide how many chunks
    /// to split a single-language batch into for concurrent inference.
    pub fn max_workers_per_key_for(&self, profile: WorkerProfile) -> usize {
        self.config.max_workers_per_key.get(profile)
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

    /// Get or create a shared GPU worker for a derived key.
    ///
    /// Coordination is PER KEY, not per map: the map lock is held only long
    /// enough to hand out this key's [`GpuWorkerSlot`], and the spawn runs with
    /// the lock released. Two callers for one key still produce exactly one
    /// worker process (they share the slot's lifecycle lock); a caller for a DIFFERENT
    /// key, including one whose worker is already warm, no longer waits behind
    /// the spawn.
    ///
    /// Spawns still serialize globally on `memory_guard`'s single-permit
    /// semaphore. That is deliberate and unchanged: what this stops is work
    /// needing no spawn from queuing behind one. See `gpu_slot.rs`.
    #[instrument(
        skip_all,
        fields(target = %key.target.label(), lang = %key.language),
    )]
    pub(super) async fn get_or_create_gpu_worker(
        &self,
        key: &WorkerKey,
    ) -> Result<Arc<shared_gpu::SharedGpuWorker>, WorkerError> {
        // The only time the map lock is held: long enough to hand out this
        // key's slot. Every caller for this key gets the SAME slot, which is
        // what still makes duplicate spawns impossible.
        let slot = {
            let mut gpu_workers = self.gpu_workers.lock().await;
            gpu_workers
                .entry(key.clone())
                .or_insert_with(GpuWorkerSlot::pending)
                .clone()
        };

        let config = self.worker_config(key);

        let worker = slot
            .worker_or_init(|| async move {
                let mut handle = WorkerHandle::spawn(config).await?;
                self.admit_worker_runtime(&handle)?;
                if self.lazy_capabilities.get().is_none()
                    && let Err(e) = self.detect_capabilities_from_worker(&mut handle).await
                {
                    tracing::warn!(error = %e, "Failed to detect capabilities from first GPU worker (continuing)");
                }
                info!(
                    target = %key.target.label(),
                    lang = %key.language,
                    pid = %handle.pid(),
                    "GPU worker spawned (concurrent mode)"
                );
                Ok(Arc::new(shared_gpu::SharedGpuWorker::from_handle(handle).await))
            })
            .await?;

        // A spawn can now finish AFTER `shutdown()` has drained the map, which
        // was impossible while the map lock was held across it. `shutdown()`
        // cancels before it drains, so a set token here means the drain either
        // has already passed this slot (and missed a worker that did not exist
        // yet) or is about to find it. Retire the worker ourselves rather than
        // hand back a worker nothing will ever reap: that is precisely the
        // orphaned-model-process class this pool keeps being bitten by.
        // `SharedGpuWorker::shutdown` is idempotent, so racing the drain and
        // both retiring it is harmless.
        if self.cancel.is_cancelled() {
            warn!(
                target = %key.target.label(),
                lang = %key.language,
                pid = %worker.pid(),
                "GPU worker finished spawning during pool shutdown; retiring it"
            );
            worker.shutdown().await;
            return Err(WorkerError::PoolShuttingDown);
        }

        Ok(worker)
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
    /// Resolve the worker-registry file this pool reads and writes.
    ///
    /// An empty configured path means "the ambient default"
    /// (`BATCHALIGN_STATE_DIR`, else `~/.batchalign3/workers.json`). Resolved in
    /// one place because three callers need it (discovery, shutdown, and the
    /// Drop safety net) and a pool that discovers from one file while retiring
    /// daemons from another would leak exactly the daemons it started.
    pub(super) fn registry_path(&self) -> std::path::PathBuf {
        if self.config.worker_registry_path.is_empty() {
            crate::worker::registry::default_registry_path()
        } else {
            std::path::PathBuf::from(&self.config.worker_registry_path)
        }
    }

    /// This pool's registry ownership identity.
    ///
    /// Public because a caller that pre-starts a TCP daemon and expects THIS
    /// pool to adopt and later reap it must stamp the daemon with this id: the
    /// registry records ownership by instance, and `kill_owned_daemons` reaps
    /// only its own. That is how `dropping_pool_reaps_owned_tcp_daemons`
    /// constructs its subject, now that startup warmup (the former in-crate
    /// spawner of server-owned daemons) is retired.
    pub fn current_server_instance_id(&self) -> &str {
        // Constructor invariant: `WorkerPool::new` populates
        // `config.runtime.server_instance_id` unconditionally on
        // creation; the field is `Option<...>` only because
        // `WorkerPoolConfig` is reused for tests that don't go through
        // `new()`.
        #[allow(clippy::expect_used)]
        self.config
            .runtime
            .server_instance_id
            .as_deref()
            .expect("WorkerPool::new always assigns a server instance id")
    }
}

// V2 execute key resolution helpers live in execute_v2.rs, used by dispatch.rs.

#[cfg(test)]
mod default_pool_config_tests {
    use super::*;

    /// `PoolConfig::default()` must produce concrete numeric values rather
    /// than a sentinel that routes to a sysinfo probe at runtime
    /// production fills the field from `EffectiveConfig` and tests must
    /// be deterministic across machines.
    #[test]
    fn pool_default_has_concrete_max_total_workers() {
        let cfg = PoolConfig::default();
        assert!(
            cfg.max_total_workers > 0,
            "Default::default() must use a concrete value, not the legacy 0-sentinel"
        );
    }

    #[test]
    fn pool_default_effective_cap_is_deterministic() {
        let a = PoolConfig::default().effective_max_total_workers();
        let b = PoolConfig::default().effective_max_total_workers();
        assert_eq!(
            a, b,
            "effective_max_total_workers must not depend on transient sysinfo state"
        );
        assert!(a > 0, "effective cap must be a real concrete value");
    }

    use super::PerProfile;
    use crate::api::{LanguageCode3, WorkerLanguage};
    use crate::types::engines::{AsrEngineName, EngineOverrides};
    use crate::types::options::{AlignOptions, CommandOptions};
    use crate::worker::{InferTask, WorkerTarget};

    #[test]
    fn command_options_selection_normalizes_non_worker_asr_and_keeps_extras() {
        let mut options = AlignOptions::default();
        options.common.engine_overrides = EngineOverrides {
            asr: Some(AsrEngineName::RevAi),
            extras: [("provider_region".to_owned(), "us-east".to_owned())].into(),
            ..EngineOverrides::default()
        };

        // Through the production constructor: `for_target` is what builds
        // every real key, and it is a superset of what the retired
        // `from_command_options` did.
        let selection = EngineSelection::for_target(
            WorkerTarget::for_command_with_mode(
                ReleasedCommand::Align,
                WorkerBootstrapMode::Profile,
            ),
            &CommandOptions::Align(options),
        );

        assert_eq!(selection.overrides().asr, None);
        assert_eq!(selection.overrides().extras["provider_region"], "us-east");
        // Derived: `fa` is whatever the default engine is, not a value this
        // test chooses. What it asserts is that a non-worker ASR override is
        // dropped while an unknown extra survives.
        assert_eq!(
            selection.worker_config_json(),
            format!(
                r#"{{"fa":"{}","provider_region":"us-east"}}"#,
                <crate::types::engines::FaEngineName as
                    crate::types::engines::SelectableEngine>::DEFAULT
                    .dispatch_override_name()
            )
        );
    }

    #[test]
    fn lazy_profile_command_options_key_retains_engine_selection() {
        let options = CommandOptions::Align(AlignOptions::default());
        let key = WorkerKey::from_command_options(
            ReleasedCommand::Align,
            WorkerLanguage::from(LanguageCode3::eng()),
            &options,
            WorkerBootstrapMode::LazyProfile,
        );

        assert!(key.target.is_concurrent());
        assert_eq!(
            key.engine_selection.worker_config_json(),
            r#"{"fa":"wave2vec"}"#
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn metrics_snapshot_reflects_per_profile_active_counts() {
        let pool = WorkerPool::new(PoolConfig {
            max_workers_per_key: PerProfile::uniform(4),
            max_total_workers: 16,
            // Disable the CPU-loadavg gate so the test isolates
            // permit / per-key-cap / metrics behavior. CI runners
            // are CPU-saturated by parallel cargo-test workers and
            // the gate would otherwise reject every claim with
            // CpuSaturated before the assertions run.
            cpu_gate_threshold_override: Some(f64::INFINITY),
            ..Default::default()
        });
        let lang = WorkerLanguage::from(LanguageCode3::eng());

        // Ground-truth shape on a fresh pool: zero workers, zero rejections.
        let m0 = pool.metrics_snapshot();
        assert_eq!(m0.active_workers_total, 0);
        assert_eq!(
            m0.active_workers_by_profile,
            PerProfile {
                gpu: 0,
                stanza: 0,
                io: 0
            }
        );
        assert_eq!(m0.spawn_rejections_total, 0);
        assert_eq!(m0.memory_gate_rejections_total, 0);
        assert_eq!(
            m0.permits_available, 16,
            "spawn_permits semaphore must be sized to max_total_workers at startup"
        );
        assert_eq!(m0.permit_rejections_total, 0);

        // Force a stanza-keyed group total to 3 and a gpu-keyed group
        // total to 2 (simulating spawned workers without going through
        // the real Python path).
        let stanza_target = WorkerTarget::infer_task(InferTask::Morphosyntax);
        let stanza_key = WorkerKey::without_engine_selection(stanza_target, lang.clone());
        let stanza_group = pool.get_or_create_group(&stanza_key);
        stanza_group.total.store(3, Ordering::Relaxed);
        let gpu_target = WorkerTarget::infer_task(InferTask::Asr);
        let gpu_key = WorkerKey::without_engine_selection(gpu_target, lang.clone());
        let gpu_group = pool.get_or_create_group(&gpu_key);
        gpu_group.total.store(2, Ordering::Relaxed);

        let m1 = pool.metrics_snapshot();
        assert_eq!(m1.active_workers_total, 5);
        assert_eq!(m1.active_workers_by_profile.stanza, 3);
        assert_eq!(m1.active_workers_by_profile.gpu, 2);
        assert_eq!(m1.active_workers_by_profile.io, 0);

        // Drive a global-cap rejection. Post-refactor the global cap is
        // enforced by the spawn_permits semaphore, not by summing
        // group.total values, so we must drain the semaphore directly.
        // Probing a fresh io-keyed group then hits PermitRejected and
        // increments permit_rejections_total. The legacy
        // spawn_rejections_total now counts only per-key cap rejections
        // (none fired in this scenario, so it stays at 0).
        let _drained = pool
            .spawn_permits
            .clone()
            .try_acquire_many_owned(16)
            .expect("drain all 16 global permits");
        let io_target = WorkerTarget::infer_task(InferTask::Translate);
        let io_key = WorkerKey::without_engine_selection(io_target, lang);
        let io_group = pool.get_or_create_group(&io_key);
        let _ = pool.try_claim_spawn_slot(&io_group);

        let m2 = pool.metrics_snapshot();
        assert_eq!(
            m2.permit_rejections_total, 1,
            "global-cap rejection must increment permit_rejections_total exactly once"
        );
        assert_eq!(
            m2.spawn_rejections_total, 0,
            "no per-key cap was hit in this scenario, so spawn_rejections_total stays at 0"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn spawn_failure_releases_permit() {
        // Force WorkerHandle::spawn to fail by pointing python_path
        // at an unrunnable binary. The Err arm in try_spawn_into_group
        // must refund the speculatively-acquired permit so other
        // groups don't see the slot as permanently consumed.
        let pool = WorkerPool::new(PoolConfig {
            max_total_workers: 4,
            max_workers_per_key: PerProfile::uniform(8),
            python_path: "/nonexistent/python-binary-that-will-fail".to_string(),
            test_echo: false,
            // Disable Layer 0 so the test exercises the spawn-failure
            // permit-refund path and is not gated out on saturated
            // CI runners.
            cpu_gate_threshold_override: Some(f64::INFINITY),
            ..Default::default()
        });
        let baseline = pool.metrics_snapshot().permits_available;
        let lang = WorkerLanguage::from(LanguageCode3::eng());
        let target = WorkerTarget::infer_task(InferTask::Morphosyntax);
        let key = WorkerKey::without_engine_selection(target, lang);
        let group = pool.get_or_create_group(&key);

        let result = pool.try_spawn_into_group(&group, &key).await;
        assert!(result.is_err(), "expected spawn failure");

        let after = pool.metrics_snapshot().permits_available;
        assert_eq!(
            after, baseline,
            "spawn failure must refund the global permit (baseline {baseline}, after {after})",
        );
        assert_eq!(
            group.total.load(Ordering::Relaxed),
            0,
            "spawn failure must roll back the per-key total"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn shutdown_refunds_permits_for_drained_idle_workers() {
        // Plant fake worker counts on a fresh pool, then call the
        // shutdown drain path indirectly via direct manipulation of
        // group.total and the matching permit accounting. This is the
        // unit-test analogue of "a normal shutdown returns
        // permits_available to max_total_workers."
        let pool = WorkerPool::new(PoolConfig {
            max_total_workers: 8,
            max_workers_per_key: PerProfile::uniform(4),
            ..Default::default()
        });
        // Acquire 5 permits to simulate 5 live workers across two
        // groups; track them in the group totals as well.
        let drained = pool
            .spawn_permits
            .clone()
            .try_acquire_many_owned(5)
            .expect("drain 5");
        drained.forget();
        assert_eq!(pool.metrics_snapshot().permits_available, 3);

        let lang = WorkerLanguage::from(LanguageCode3::eng());
        let key = WorkerKey::without_engine_selection(
            WorkerTarget::infer_task(InferTask::Morphosyntax),
            lang,
        );
        let group = pool.get_or_create_group(&key);
        group.total.store(5, Ordering::Relaxed);

        // Mirror what shutdown.rs's drain branch does: fetch_sub +
        // release_n. After this, both group.total and permits_available
        // should reflect 0 live workers.
        let drained_count = group.total.swap(0, Ordering::Relaxed);
        super::permit::SpawnPermitGuard::release_n(&group.spawn_permits, drained_count);
        assert_eq!(
            pool.metrics_snapshot().permits_available,
            8,
            "shutdown drain must restore permits_available to max_total_workers"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn admission_rejects_when_no_permit_available() {
        let pool = WorkerPool::new(PoolConfig {
            max_workers_per_key: PerProfile::uniform(8),
            max_total_workers: 1,
            // Disable Layer 0: see the parallel test fixture above.
            cpu_gate_threshold_override: Some(f64::INFINITY),
            ..Default::default()
        });
        let lang = WorkerLanguage::from(LanguageCode3::eng());
        let target = WorkerTarget::infer_task(InferTask::Morphosyntax);
        let key = WorkerKey::without_engine_selection(target, lang);
        let group = pool.get_or_create_group(&key);

        // Drain the only permit out-of-band to simulate "global cap
        // already saturated by other workers."
        let _hold = pool
            .spawn_permits
            .clone()
            .try_acquire_owned()
            .expect("first permit");

        let result = pool.try_claim_spawn_slot(&group);
        assert!(result.is_err(), "expected admission rejection");
        let m = pool.metrics_snapshot();
        assert_eq!(
            m.permit_rejections_total, 1,
            "permit-pool exhaustion must increment permit_rejections_total"
        );
        assert_eq!(
            group.total.load(Ordering::Relaxed),
            0,
            "rejected admission must not bump group.total"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn admission_releases_permit_on_per_key_cap_failure() {
        let pool = WorkerPool::new(PoolConfig {
            max_workers_per_key: PerProfile::uniform(1),
            max_total_workers: 8,
            // Disable Layer 0: see the parallel test fixture above.
            cpu_gate_threshold_override: Some(f64::INFINITY),
            ..Default::default()
        });
        let lang = WorkerLanguage::from(LanguageCode3::eng());
        let target = WorkerTarget::infer_task(InferTask::Morphosyntax);
        let key = WorkerKey::without_engine_selection(target, lang);
        let group = pool.get_or_create_group(&key);

        let permits_before = pool.metrics_snapshot().permits_available;
        // First claim succeeds. We immediately drop the returned guard
        // so the permit refunds back to the pool, what we want to
        // exercise next is that a per-key cap rejection ALSO refunds.
        let first = pool.try_claim_spawn_slot(&group).expect("first claim");
        drop(first);
        assert_eq!(
            pool.metrics_snapshot().permits_available,
            permits_before,
            "dropping the success guard must refund the permit"
        );

        // group.total is now 1, which equals max_workers_per_key. The
        // next probe must hit the per-key cap. The speculatively-acquired
        // permit must be released by the auto-drop.
        let result = pool.try_claim_spawn_slot(&group);
        assert!(
            result.is_err(),
            "per-key cap must reject when group.total == max"
        );
        assert_eq!(
            pool.metrics_snapshot().permits_available,
            permits_before,
            "per-key rejection must release the speculatively-acquired permit"
        );
        // Per-key rejection bumps the legacy spawn_rejections_total
        // counter (post-refactor it counts ONLY per-key rejections).
        assert_eq!(pool.metrics_snapshot().spawn_rejections_total, 1);
    }

    // ------------------------------------------------------------------
    // Property test: arbitrary admission/exit/shutdown op sequences must
    // preserve the BUG-028 invariant
    //   sum(group.total) + permits_available == max_total_workers.
    // The harness drives the same primitives the production paths use
    // (try_claim_spawn_slot, fetch_sub + add_permits) so any accounting
    // bug at any C5/C6 site surfaces here under a small composite trace.
    // ------------------------------------------------------------------

    use proptest::prelude::*;

    const PROPTEST_MAX_TOTAL: usize = 6;
    const PROPTEST_NUM_KEYS: usize = 3;

    #[derive(Debug, Clone)]
    enum LifecycleOp {
        SpawnAttempt(usize),
        WorkerExit(usize),
        ShutdownGroup(usize),
    }

    fn lifecycle_op_strategy() -> impl Strategy<Value = LifecycleOp> {
        prop_oneof![
            (0..PROPTEST_NUM_KEYS).prop_map(LifecycleOp::SpawnAttempt),
            (0..PROPTEST_NUM_KEYS).prop_map(LifecycleOp::WorkerExit),
            (0..PROPTEST_NUM_KEYS).prop_map(LifecycleOp::ShutdownGroup),
        ]
    }

    fn proptest_groups(pool: &WorkerPool) -> Vec<Arc<WorkerGroup>> {
        let lang = WorkerLanguage::from(LanguageCode3::eng());
        let tasks = [
            InferTask::Morphosyntax,
            InferTask::Translate,
            InferTask::Coref,
        ];
        tasks
            .iter()
            .map(|task| {
                let target = WorkerTarget::infer_task(*task);
                let key = WorkerKey::without_engine_selection(target, lang.clone());
                pool.get_or_create_group(&key)
            })
            .collect()
    }

    fn run_lifecycle_ops(ops: &[LifecycleOp]) -> Result<(), TestCaseError> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("rt");
        rt.block_on(async {
            let pool = WorkerPool::new(PoolConfig {
                max_total_workers: PROPTEST_MAX_TOTAL,
                max_workers_per_key: PerProfile::uniform(PROPTEST_MAX_TOTAL),
                test_echo: true,
                ..Default::default()
            });
            let groups = proptest_groups(&pool);
            let mut bookkept = [0usize; PROPTEST_NUM_KEYS];

            for (i, op) in ops.iter().enumerate() {
                match *op {
                    LifecycleOp::SpawnAttempt(g) => {
                        if let Ok((_, guard)) = pool.try_claim_spawn_slot(&groups[g]) {
                            guard.forget();
                            bookkept[g] += 1;
                        }
                    }
                    LifecycleOp::WorkerExit(g) => {
                        if bookkept[g] > 0 {
                            bookkept[g] -= 1;
                            groups[g].total.fetch_sub(1, Ordering::Relaxed);
                            pool.spawn_permits.add_permits(1);
                        }
                    }
                    LifecycleOp::ShutdownGroup(g) => {
                        if bookkept[g] > 0 {
                            let n = bookkept[g];
                            bookkept[g] = 0;
                            groups[g].total.fetch_sub(n, Ordering::Relaxed);
                            permit::SpawnPermitGuard::release_n(&groups[g].spawn_permits, n);
                        }
                    }
                }

                let active: usize = groups.iter().map(|g| g.total.load(Ordering::Relaxed)).sum();
                let permits = pool.metrics_snapshot().permits_available;
                let total_bookkept: usize = bookkept.iter().sum();
                prop_assert_eq!(
                    active,
                    total_bookkept,
                    "after op #{}: pool's group.total sum ({}) must equal harness ({}); op={:?}",
                    i,
                    active,
                    total_bookkept,
                    op,
                );
                prop_assert_eq!(
                    active + permits,
                    PROPTEST_MAX_TOTAL,
                    "after op #{}: invariant violated, active={} permits={} max={}; op={:?}",
                    i,
                    active,
                    permits,
                    PROPTEST_MAX_TOTAL,
                    op,
                );
            }

            // Final cleanup: drain any remaining workers and confirm
            // permits return to PROPTEST_MAX_TOTAL.
            for g in 0..PROPTEST_NUM_KEYS {
                let n = bookkept[g];
                if n > 0 {
                    bookkept[g] = 0;
                    groups[g].total.fetch_sub(n, Ordering::Relaxed);
                    permit::SpawnPermitGuard::release_n(&groups[g].spawn_permits, n);
                }
            }
            let m = pool.metrics_snapshot();
            prop_assert_eq!(m.active_workers_total, 0);
            prop_assert_eq!(m.permits_available, PROPTEST_MAX_TOTAL);
            Ok(())
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            ..ProptestConfig::default()
        })]

        #[test]
        fn invariant_holds_under_arbitrary_op_sequence(
            ops in proptest::collection::vec(lifecycle_op_strategy(), 0..64)
        ) {
            run_lifecycle_ops(&ops)?;
        }
    }
}
