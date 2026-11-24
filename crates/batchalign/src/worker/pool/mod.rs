//! `WorkerPool`: manages multiple Python worker processes.
//!
//! Workers are keyed by bootstrap target, language, and typed engine selection.
//! Large hosts may use shared profile targets such as `profile:gpu`, while
//! constrained hosts use task targets such as `infer:asr` so one laptop does
//! not hold unrelated models in memory.
//!
//! Each key maps to a `WorkerGroup` containing up to `max_workers_per_key`
//! workers, spawned lazily on demand. Background tasks handle health checking
//! and memory-pressure eviction.
//!
//! ## Concurrency model
//!
//! Idle queues hold complete native-worker and capacity owners. Read-only views
//! expose their actual contents. Checkout registers change notifications before
//! observing availability, then owns a departure from its original queue.
//! Consuming `TrackedCheckout::run` binds the actual exchange result to return.
//! Dropping unfinished work retires its handle and original slot.
//!
//! ## Module layout
//!
//! | File | Responsibility |
//! |------|----------------|
//! | `mod.rs` | Types, pool struct, construction, accessors, GPU worker creation |
//! | `checkout.rs` | Consuming exclusive exchange ownership |
//! | `dispatch.rs` | Checkout loop, batch infer, V2 execute, TCP routing |
//! | `discovery.rs` | Registry-based TCP worker discovery |
//! | `pre_scale.rs` | Eager per-job worker pre-spawning |
//! | `shutdown.rs` | Graceful shutdown and `Drop` cleanup |
//! | `lifecycle.rs` | Background health checking, pressure eviction, spawn helpers |
//! | `execute_v2.rs` | V2 request key resolution helpers |
//! | `status.rs` | Status query methods (has_idle_workers, worker_count, etc.) |
//! | `shared_gpu/` | Shared concurrent GPU worker wrappers (stdio, TCP, reader) |
//! | `reaper.rs` | Orphaned worker process reaping |

mod capacity;
mod checkout;
pub(crate) mod checkout_wait;
mod cpu_gate;
mod discovery;
mod dispatch;
mod eviction;
mod execute_v2;
mod fa_capacity;
mod fa_preparation;
pub(super) mod gpu_slot;
mod idle_eviction;
pub(crate) mod job_tracker;
mod lifecycle;
pub(crate) mod memory_gate;
mod model_plan;
mod pre_scale;
pub(crate) mod reaper;
mod rss_observer;
pub(crate) mod shared_gpu;
mod shutdown;
mod spawn_runtime;
mod standalone;
pub mod status;
mod task_load;
mod worker_plan;
pub(crate) use dispatch::dispatch_prepared;
pub(crate) use fa_preparation::FaProbeCompletion;
pub use fa_preparation::{
    BoundFaPreparation, DispatchPreparation, FaPreparationCompletion, FaPreparationExchange,
    ReadyRequest,
};

pub use checkout::CheckedOutWorker;
pub(crate) use model_plan::GpuRoute;
pub use model_plan::{GpuModelPlan, GpuRouteRefusal, GpuTask, PlanMismatch};
pub(crate) use standalone::StandaloneWorker;
pub use status::{WorkerSummaryEntry, WorkerTransport};
pub(crate) use task_load::TaskLoads;
pub use task_load::{
    AdmittedLoad, ConfirmedLoad, LoadAdmission, LoadPreparation, Loaded, ModelHosting, RunPermit,
    TaskLoad,
};

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(test)]
use crate::api::ReleasedCommand;
use crate::api::{PositiveSeconds, WorkerLanguage};
use crate::host_facts::PerProfile;
use crate::options::CommandOptions;
use crate::types::engines::EngineOverrides;
use crate::types::worker_v2::TaskTimeoutOverrides;
use crate::worker::{WorkerBootstrapMode, WorkerCapabilities, WorkerProfile, WorkerTarget};
use batchalign_types::worker::InferTask;
use tokio::sync::Mutex as AsyncMutex;
use tokio_util::sync::CancellationToken;
use tracing::{info, instrument, warn};
use uuid::Uuid;

use self::gpu_slot::GpuWorkerSlot;
use self::spawn_runtime::PoolSpawnRuntime;
use crate::engine_reports::WorkerEngineReports;
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
/// interrupted operation. The queue still owns every worker it contains, and
/// each worker retains its original capacity until retirement.
/// Recovering from poison keeps the server alive instead of cascading the
/// panic into every subsequent request.
pub(crate) fn lock_recovered<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
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

    /// The engines a text or IO command itself names, before any
    /// target-driven filling.
    ///
    /// Only translate selects an engine among those commands. The GPU
    /// commands (align, transcribe, benchmark) are absent on purpose: their
    /// workers are keyed by the job's model plan, never from options here.
    fn overrides_for_command(options: &CommandOptions) -> EngineOverrides {
        let mut overrides = options.common().engine_overrides.clone();
        match options {
            CommandOptions::Translate(options) => {
                overrides
                    .translate
                    .get_or_insert_with(|| options.effective_translate_engine());
            }
            // GPU commands: keyed by their model plan (`GpuRoute::worker_key`).
            CommandOptions::Align(_)
            | CommandOptions::Transcribe(_)
            | CommandOptions::TranscribeS(_)
            | CommandOptions::Benchmark(_)
            | CommandOptions::Morphotag(_)
            | CommandOptions::Coref(_)
            | CommandOptions::Utseg(_)
            | CommandOptions::Opensmile(_)
            | CommandOptions::Compare(_)
            | CommandOptions::Avqi(_)
            | CommandOptions::Diarize(_)
            // No engine override: the embedding model is the one node of the
            // released local model graph, and there is nothing to select
            // between.
            | CommandOptions::SpeakerIdentify(_)
            | CommandOptions::Convert(_) => {}
        }
        overrides
    }

    /// The identity of a worker serving a job's GPU model plan.
    ///
    /// A profile worker hosts every task of its profile, so its identity is
    /// the plan's whole engine set; a task-mode worker hosts one task, so its
    /// identity stays that task's own keys.
    pub(super) fn for_job_plan(plan: EngineOverrides, target: WorkerTarget) -> Self {
        let keys = match target {
            WorkerTarget::Profile(_) => plan,
            WorkerTarget::InferTask(task) => Self::task_keys(task, plan),
        };
        Self::from_overrides(keys)
    }

    /// The keys a task's load reads from its overrides, and nothing else:
    /// a worker's identity for the task, and the overrides its
    /// `ensure_task` carries ([`TaskLoad`]).
    ///
    /// ASR reads `asr` and the extras (`qwen_model`, `funaudio_model`,
    /// `whisper_cpu_dtype`, ...); FA reads `fa`; translate reads `translate`;
    /// speaker reads `speaker` (local pyannote) and `speaker_embedding` (the
    /// embedding model), which never enter a GPU worker's key (the plan's
    /// identity carries no speaker model) and so reach the worker only in the
    /// speaker task's `ensure_task`. Every other task has one implementation
    /// and reads nothing. Listed rather than wildcarded so a new task decides.
    pub(super) fn task_keys(task: InferTask, selected: EngineOverrides) -> EngineOverrides {
        match task {
            InferTask::Asr => EngineOverrides {
                asr: selected.asr,
                extras: selected.extras,
                ..EngineOverrides::default()
            },
            InferTask::Fa => EngineOverrides {
                fa: selected.fa,
                ..EngineOverrides::default()
            },
            InferTask::Translate => EngineOverrides {
                translate: selected.translate,
                ..EngineOverrides::default()
            },
            InferTask::Speaker => EngineOverrides {
                speaker: selected.speaker,
                speaker_embedding: selected.speaker_embedding,
                ..EngineOverrides::default()
            },
            InferTask::Morphosyntax
            | InferTask::Utseg
            | InferTask::Coref
            | InferTask::Opensmile
            | InferTask::Avqi
            | InferTask::WordSegmentation => EngineOverrides::default(),
        }
    }

    /// Parse the JSON held by an externally managed registry entry.
    ///
    /// This is ingress from the Python registry boundary, not another source
    /// of pool-key derivation. Once admitted, equality is structural
    /// [`EngineOverrides`] equality rather than spelling-sensitive JSON.
    ///
    /// The registry records everything the worker was spawned with, including
    /// the keys the control plane injected from the worker key
    /// (`lifecycle::SpawnKey`). Those are removed first: they describe the
    /// key, they do not select it, so the rebuilt selection equals the one the
    /// spawning request computed.
    pub(super) fn from_registry_json(json: &str) -> Result<Self, serde_json::Error> {
        if json.trim().is_empty() {
            return Ok(Self::none());
        }
        // Every injected key is outside the typed fields, so it lands in
        // `extras`, where it is removed.
        let mut overrides: EngineOverrides = serde_json::from_str(json)?;
        for key in lifecycle::SpawnKey::ALL {
            overrides.extras.remove(key.as_str());
        }
        Ok(Self::from_overrides(overrides))
    }

    /// JSON for the Rust/Python worker configuration boundary.
    pub(super) fn worker_config_json(&self) -> String {
        self.0.to_dispatch_json_string()
    }

    /// Typed engine details consumed by Rust-side admission logic.
    pub(super) const fn overrides(&self) -> &EngineOverrides {
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

/// Why probing a freshly spawned worker's capabilities admitted no report.
///
/// The two cases call for different handling, which is why they are not one
/// `WorkerError`: an unanswered probe leaves a worker that may still serve (the
/// report is taken again at dispatch), while a refused report means the worker
/// is not used at all and must not be pooled.
#[derive(Debug)]
pub(super) enum CapabilityProbeFailure {
    /// The worker did not answer the capabilities request.
    Unanswered(WorkerError),
    /// The worker answered, and admission refused its report.
    Refused(crate::engine_reports::EngineReportAdmissionError),
}

/// What the pool decided about one worker key's latest capability report.
#[derive(Debug, Clone)]
enum CapabilityAdmission {
    /// Admitted; dispatch reads this report.
    Admitted(Arc<WorkerEngineReports>),
    /// Refused; the key's workers are not used for dispatch while this stands.
    Refused(crate::engine_reports::EngineReportAdmissionError),
}

/// A worker key's admitted capability report, taken AFTER the command's
/// primary infer task was loaded on the selected worker.
///
/// Supporting a task and naming the engine behind it are different facts: a
/// lazily loading worker supports forced alignment before it has loaded an FA
/// model, and names the FA engine only after. Advertisement reads support from
/// any admitted report. An engine a command must name before dispatch is read
/// only from this type (`FaCacheNamespace::from_loaded`), and only
/// [`WorkerPool::ensure_command_capabilities`] builds one, after `ensure_task`:
/// the fields are private to this module tree. So a `null` read from it means
/// "still unnamed after loading", and that alone is a refusal. It records the
/// task that was loaded, so an identity is never read for a task other than
/// the one the report was taken after.
#[derive(Debug, Clone)]
pub struct LoadedCapabilities {
    /// The task `ensure_task` loaded before the report was taken.
    task: InferTask,
    /// The admitted report taken after that load.
    reports: Arc<WorkerEngineReports>,
}

impl LoadedCapabilities {
    /// The task that was loaded before this report was taken.
    #[must_use]
    pub const fn task(&self) -> InferTask {
        self.task
    }

    /// The admitted post-load report.
    #[must_use]
    pub fn reports(&self) -> &WorkerEngineReports {
        &self.reports
    }

    /// A post-load report for tests that exercise the availability rule
    /// without a worker.
    #[cfg(test)]
    pub(crate) fn for_test(task: InferTask, reports: WorkerEngineReports) -> Self {
        Self {
            task,
            reports: Arc::new(reports),
        }
    }
}

impl WorkerKey {
    /// Operator-facing label: `target:lang`, plus the engine selection when
    /// one is set.
    pub(super) fn label(&self) -> String {
        if self.engine_selection.is_none() {
            format!("{}:{}", self.target.label(), self.language)
        } else {
            format!(
                "{}:{}:{}",
                self.target.label(),
                self.language,
                self.engine_selection
            )
        }
    }

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

    /// The worker a job's command is probed on, pre-scaled for and checked
    /// for: its plan's worker for a GPU command, its options' worker for a
    /// text or IO command. The one place a command-level key is derived, so
    /// probe, pre-scale and dispatch cannot select different workers.
    pub(super) fn for_command(
        requirement: crate::command_model::InferenceRequirement,
        language: WorkerLanguage,
        options: &CommandOptions,
        gpu: &GpuModelPlan,
        bootstrap_mode: WorkerBootstrapMode,
    ) -> Self {
        worker_plan::CommandWorkerPlan::new(requirement, language, options, gpu, bootstrap_mode)
            .key()
            .clone()
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

/// Default per-profile `max_workers_per_key` for `PoolConfig::for_tests()`
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

/// Default `max_total_workers` for `PoolConfig::for_tests()` test fixtures.
///
/// Production sets this from `EffectiveConfig::max_total_workers` (the
/// host-facts recommendation, RAM-derived, clamped to `[2, 32]`). Tests
/// use a concrete constant rather than a sysinfo probe so behavior is
/// deterministic across machines.
const DEFAULT_MAX_TOTAL_WORKERS_FOR_TESTS: usize = 32;
const _: () = assert!(DEFAULT_MAX_TOTAL_WORKERS_FOR_TESTS > 0);

/// Configuration for the worker pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Path to the Python executable.
    pub python_path: String,
    /// Interval between health checks.
    pub health_check_interval_s: PositiveSeconds,
    /// How long a starting worker may go without advancing
    /// (`worker_startup_stall_s`), copied to each worker.
    pub startup_stall: crate::worker::LoadingStall,
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
    /// How often a checkout waiting on a saturated pool reports that it is
    /// still waiting and re-probes. Not a deadline: a saturated pool is
    /// congestion, and the checkout keeps waiting (see
    /// `checkout_wait`). `None`: the built-in default (300s).
    pub checkout_wait_report_interval: Option<PositiveSeconds>,
    /// Verbosity level forwarded to Python workers (0=warn, 1=info, 2=debug).
    pub verbose: u8,
    /// Runtime-owned worker launch inputs (device policy, injected creds).
    pub runtime: WorkerRuntimeConfig,
    /// The operator's transport-ceiling overrides for audio and analysis
    /// tasks; each absent one leaves the built-in ceiling.
    pub task_timeouts: TaskTimeoutOverrides,
    /// Stall window for on-demand model loading via `ensure_task`: how long
    /// the load may go without advancing progress (`ensure_task_stall_s`),
    /// resolved once at the configuration boundary.
    pub ensure_task_stall: crate::worker::LoadingStall,
    /// Path to the worker registry file. `None`: the ambient default
    /// (`BATCHALIGN_STATE_DIR`, else `~/.batchalign3/workers.json`).
    pub worker_registry_path: Option<std::path::PathBuf>,
    /// Test-only: artificial delay in milliseconds before each worker response.
    /// 0 = no delay. Only effective when `test_echo` is also true. Plumbed to
    /// every spawned worker's `WorkerConfig.test_delay_ms`. Used by the
    /// gpu-concurrent-dispatch tests to assert that the per-request timeout
    /// budget is not consumed by queue-wait when many callers contend for one
    /// shared GPU worker process.
    pub test_delay_ms: u64,
}

/// Built-in default for `checkout_wait_report_interval` (seconds).
///
/// Long enough that an ordinary queue behind a few long requests is not
/// reported, short enough that a wait an operator should look at is.
const DEFAULT_CHECKOUT_WAIT_REPORT_INTERVAL: PositiveSeconds = PositiveSeconds::literal::<300>();

impl PoolConfig {
    /// A pool with every field at its default except `runtime`, which
    /// carries the host decisions (tier, device) the caller resolved.
    #[must_use]
    pub fn with_runtime(runtime: WorkerRuntimeConfig) -> Self {
        Self::with_runtime_and_python(runtime, resolve_python_executable())
    }

    const fn with_runtime_and_python(runtime: WorkerRuntimeConfig, python_path: String) -> Self {
        Self {
            python_path,
            health_check_interval_s: PositiveSeconds::literal::<30>(),
            startup_stall: crate::worker::LoadingStall::DEFAULT_STARTUP,
            test_echo: false,
            max_workers_per_key: DEFAULT_MAX_WORKERS_PER_KEY,
            max_total_workers: DEFAULT_MAX_TOTAL_WORKERS_FOR_TESTS,
            checkout_wait_report_interval: None,
            verbose: 0,
            runtime,
            task_timeouts: TaskTimeoutOverrides::NONE,
            ensure_task_stall: crate::worker::LoadingStall::DEFAULT_ENSURE_TASK,
            worker_registry_path: None,
            test_delay_ms: 0,
        }
    }

    /// A test pool on the declared test host
    /// ([`HostSnapshot::for_tests`](crate::host_facts::HostSnapshot::for_tests)):
    /// its admission gates, spawn guard, ledger and eviction read that host,
    /// so they run in tests and decide the same way on every machine.
    #[must_use]
    pub fn for_tests() -> Self {
        Self::with_runtime(WorkerRuntimeConfig::for_tests())
    }

    /// Carry a fixture's interpreter into the pool without discovering it again.
    #[doc(hidden)]
    #[must_use]
    pub fn for_tests_with_python(python_path: String) -> Self {
        Self::with_runtime_and_python(WorkerRuntimeConfig::for_tests(), python_path)
    }
}

impl PoolConfig {
    /// Whether this pool's workers host models: an echo pool's workers build
    /// nothing, so their loads take no memory lease.
    pub(crate) const fn model_hosting(&self) -> ModelHosting {
        if self.test_echo {
            ModelHosting::TestEcho
        } else {
            ModelHosting::Models
        }
    }

    /// Resolved global worker cap. Returns `self.max_total_workers`
    /// directly; production fills it from `EffectiveConfig`, tests
    /// from `DEFAULT_MAX_TOTAL_WORKERS_FOR_TESTS` via
    /// `PoolConfig::for_tests()`.
    #[must_use]
    pub const fn effective_max_total_workers(&self) -> usize {
        self.max_total_workers
    }

    /// Resolved checkout wait report interval: the configured one, else the
    /// built-in default (300s).
    pub(super) fn checkout_wait_report_interval(&self) -> PositiveSeconds {
        self.checkout_wait_report_interval
            .unwrap_or(DEFAULT_CHECKOUT_WAIT_REPORT_INTERVAL)
    }
}

// ---------------------------------------------------------------------------
// WorkerGroup, per (profile, lang) key
// ---------------------------------------------------------------------------

/// A group of workers for a single `(profile, lang)` key.
///
/// Each group independently tracks its own pool of workers. Workers are
/// spawned lazily on first demand and capped at `max_workers_per_key`.
/// Queue contents are the sole availability authority. Register a change
/// notification before observing the queue, and keep its read-only mutex view
/// out of every `.await` point. Capacity travels with the actual worker.
pub(super) struct WorkerGroup {
    /// Read-only consumers observe the actual queued owners. Only capacity's
    /// consuming publication and departure transitions mutate these queues.
    idle: capacity::IdleQueue<WorkerHandle>,
    tcp_workers: capacity::IdleQueue<TcpWorkerHandle>,

    /// Count of actual reserved and live slot owners for this group. The
    /// capacity producer owns mutation, and observers read the carried count.
    total: capacity::SlotAccounting,

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

    /// Original pool-wide budget used by the capacity producer. Observers can
    /// inspect availability. Slot owners retain and refund actual permits.
    spawn_permits: capacity::GlobalCapacity,

    /// Worker profile for this group's key. Recovered from
    /// `WorkerTarget::profile_kind()` at construction; stable for the
    /// lifetime of the group. Read by admission control to look up
    /// the per-profile slot in `PoolConfig::max_workers_per_key`.
    pub(super) profile: WorkerProfile,

    /// Completed reservation derived once from this group's engines and host.
    startup_reservation: memory_gate::StartupReservation,
}

impl WorkerGroup {
    fn new(
        spawn_permits: capacity::GlobalCapacity,
        profile: WorkerProfile,
        engine_selection: EngineSelection,
    ) -> Self {
        let startup_reservation = memory_gate::StartupReservation::for_worker(
            profile,
            engine_selection.overrides(),
            spawn_permits.host(),
        );
        Self {
            startup_reservation,
            idle: capacity::IdleQueue::new(),
            tcp_workers: capacity::IdleQueue::new(),
            total: capacity::SlotAccounting::default(),
            bootstrap: AsyncMutex::new(()),
            worker_returned: spawn_permits.notifier(),
            spawn_permits,
            profile,
        }
    }

    /// True when this group has no workers (live, idle, or in-flight
    /// for spawn). Drives the cold-start vs warm distinction at the
    /// admission gates (see `memory_gate::PoolGateState`) and the
    /// eviction-vs-wait branch in `dispatch::checkout`. The atomic
    /// load is `Relaxed` because the gate decision is intentionally
    /// race-tolerant: a concurrent spawn that flips the answer
    /// between this load and the downstream `compare_exchange`
    /// re-races at the CAS, and the gates themselves are stateless
    /// so a stale `ColdStart` classification cannot cause incorrect
    /// admission.
    pub(super) fn is_empty(&self) -> bool {
        self.total.count() == 0
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
    const fn counter(&self, kind: WorkerRejectionKind) -> &AtomicU64 {
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
    /// Shared workers for concurrent V2 dispatch (keys whose
    /// `WorkerServing` is `SharedConcurrent`, stdio).
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
    registry_admissions: crate::worker::registry::RegistryAdmissions,
    /// Distinct content-addressed Python runtimes observed since server start.
    /// This history remains available while a sequential worker is checked out
    /// and after a crash, unlike an instantaneous idle-worker listing.
    observed_worker_runtimes: Arc<crate::worker::runtime_identity::ObservedWorkerRuntimes>,
    /// Pulsed by actual queue publication and original capacity release. Parks saturated
    /// checkouts that have no live worker for their key and no idle
    /// worker elsewhere to evict; wakes them to retry spawn + eviction
    /// when any worker anywhere in the pool becomes available.
    pub(super) worker_returned: Arc<tokio::sync::Notify>,
    pub(super) cancel: CancellationToken,
    /// The first capability report any worker gave, as admitted: the
    /// pool-wide view the startup snapshot reads. Set once, on the first spawn
    /// or registry probe.
    pub(super) first_admitted_capabilities: std::sync::OnceLock<Arc<WorkerEngineReports>>,
    /// The latest admission outcome per worker key, replaced on every probe of
    /// that key: the admitted report, or why the report was refused. A report
    /// is admitted exactly once, in [`Self::record_capabilities`]; nothing
    /// downstream reads a raw report. `/health` lists these outcomes
    /// ([`Self::capability_admissions`]), so an operator sees why a refused
    /// worker is not used.
    admitted_capabilities: std::sync::Mutex<HashMap<WorkerKey, CapabilityAdmission>>,
    /// Registry daemons the latest discovery sweep found alive and refused to
    /// adopt, with why (another build, or no build named). Replaced by every
    /// sweep; `/health` lists them, so an operator sees why a running daemon
    /// is unused rather than finding only a log line.
    refused_registry_workers: std::sync::Mutex<Vec<crate::api::RefusedRegistryWorker>>,
    /// The Stanza language registry, built from the first catalog a worker
    /// reports (and until then why there is none): the one authority for
    /// morphosyntax admission.
    stanza_registry: crate::stanza_registry::StanzaRegistrySlot,
    /// Side-table mapping each active job to the worker PIDs currently
    /// dispatching for it. Populated by `TrackerGuard` instances created
    /// inside the dispatch path; drained by `detach_workers_for_job`
    /// when a cancel arrives. See `pool/job_tracker.rs`.
    pub(crate) job_tracker: job_tracker::JobWorkerTracker,
    /// Monotonic counters for every worker-admission rejection reason.
    pub(super) rejection_counters: WorkerRejectionCounters,
    /// Pool-wide capacity and its global-refund event. The producer carries
    /// the original permit through speculative admission and worker lifetime.
    /// `SpawnOutcome` retains the actual deferral cause for selecting global
    /// refunds, worker changes or queue availability. See pool/capacity.rs.
    spawn_permits: capacity::GlobalCapacity,
    /// The runtime every worker process this pool creates is bound to.
    ///
    /// Captured at construction rather than taken from whichever runtime is
    /// current at a spawn site, because a `tokio::process::Child` stays
    /// registered with the reactor that created it. See `spawn_runtime.rs` for
    /// the harness failure that made this a property of the pool.
    /// Private, not `pub(super)`: the pool's runtime is reached only from this
    /// module tree (`lifecycle`, `eviction`, and the GPU path here), which
    /// already sees it as a descendant of the defining module. Widening it
    /// would let worker code outside the pool spawn onto it.
    spawn_runtime: PoolSpawnRuntime,
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
    /// The Tencent engine this server's account chose, which every Tencent
    /// plan the server resolves reads, so a request's pin and the pin its
    /// worker was spawned with cannot name different engines.
    pub(crate) const fn tencent_engine(&self) -> &crate::model_manifest::TencentEngineSetting {
        self.config.runtime.provider_accounts.tencent_engine()
    }

    /// Test-only artificial per-response delay configured for this pool.
    ///
    /// Exposed so the test-echo dispatch path (which bypasses workers
    /// entirely) can honor the same pacing knob integration tests use to
    /// hold race windows open.
    pub(crate) const fn test_delay_ms(&self) -> u64 {
        self.config.test_delay_ms
    }

    /// The host-chosen bootstrap mode for local workers.
    pub const fn bootstrap_mode(&self) -> WorkerBootstrapMode {
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
        let spawn_permits =
            capacity::GlobalCapacity::for_host(effective_cap, config.runtime.host.clone());

        Self {
            config,
            groups: Arc::new(std::sync::Mutex::new(HashMap::new())),
            gpu_workers: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            gpu_tcp_workers: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            registry_admissions: crate::worker::registry::RegistryAdmissions::default(),
            observed_worker_runtimes: Arc::new(
                crate::worker::runtime_identity::ObservedWorkerRuntimes::default(),
            ),
            worker_returned: spawn_permits.notifier(),
            cancel: CancellationToken::new(),
            first_admitted_capabilities: std::sync::OnceLock::new(),
            admitted_capabilities: std::sync::Mutex::new(HashMap::new()),
            refused_registry_workers: std::sync::Mutex::new(Vec::new()),
            stanza_registry: crate::stanza_registry::StanzaRegistrySlot::new(),
            job_tracker: job_tracker::JobWorkerTracker::new(),
            rejection_counters: WorkerRejectionCounters::default(),
            spawn_permits,
            // Bind the pool to the runtime it is being built on, before any
            // caller can spawn a worker onto a shorter-lived one.
            spawn_runtime: PoolSpawnRuntime::captured_at_construction(),
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
                let n = group.total.count();
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
    /// hosts. The cancel path uses `detach_workers_for_job`.
    #[doc(hidden)]
    pub fn workers_for_job(&self, job_id: &crate::api::JobId) -> Vec<crate::worker::WorkerPid> {
        self.job_tracker.snapshot(job_id)
    }

    /// Run a future under the `CURRENT_JOB_ID` task-local scope so
    /// dispatch-site `TrackerGuard`s register against `job_id`.
    /// Production code uses `runner::execution::run_server_job_attempt`
    /// to set the scope; this is the integration-test entry point.
    #[doc(hidden)]
    pub fn dispatch_under_job_for_test<Fut, T>(
        job_id: crate::api::JobId,
        fut: Fut,
    ) -> std::pin::Pin<Box<impl std::future::Future<Output = T>>>
    where
        Fut: std::future::Future<Output = T>,
    {
        Box::pin(job_tracker::CURRENT_JOB_ID.scope(job_id, fut))
    }

    /// Detach and terminate in one step; for callers with no cancellation to
    /// publish in between (tests). The cancel path splits the two, see
    /// [`Self::detach_workers_for_job`].
    ///
    /// SIGTERM every worker dispatching for `job_id`, escalating to
    /// SIGKILL after a 2s grace. Long enough for a well-behaved
    /// Python worker to handle the signal cleanly, short enough
    /// that interactive cancel feels responsive.
    pub async fn shutdown_workers_for_job(&self, job_id: &crate::api::JobId) {
        self.detach_workers_for_job(job_id).terminate().await;
    }

    /// Take the workers dispatching for `job_id` out of the tracker, to be
    /// terminated once the job's cancellation has been published.
    ///
    /// The cancel path needs the two steps apart, in this order. Publishing
    /// the cancellation drops the job's file tasks, and a dropped dispatch
    /// unregisters its worker from the tracker as it unwinds; a drain AFTER
    /// the publish could therefore find nothing and leave a shared worker
    /// computing a result nobody will read. Detaching first captures every
    /// worker that was in flight when the cancel arrived.
    pub fn detach_workers_for_job(&self, job_id: &crate::api::JobId) -> JobWorkersToTerminate {
        JobWorkersToTerminate {
            job_id: job_id.clone(),
            pids: self.job_tracker.drain(job_id),
        }
    }

    /// How workers for `target` serve requests, and therefore how this pool
    /// dispatches to them: into one shared process per key, or by exclusive
    /// checkout from up to `max_workers_per_key` processes. The same value
    /// the worker is launched with (`--serving`), so the two cannot disagree.
    pub(crate) const fn serving(
        &self,
        target: WorkerTarget,
    ) -> crate::worker::serving::WorkerServing {
        crate::worker::serving::WorkerServing::decide(target.profile_kind(), &self.config.runtime)
    }

    /// Maximum workers allowed per `(target, lang, engine_overrides)` key
    /// for the requested profile.
    ///
    /// Used by the morphosyntax batch dispatcher to decide how many chunks
    /// to split a single-language batch into for concurrent inference.
    pub const fn max_workers_per_key_for(&self, profile: WorkerProfile) -> usize {
        self.config.max_workers_per_key.get(profile)
    }

    /// Resolved global worker cap (computed from RAM if not explicitly set).
    ///
    /// Used by the morphosyntax batch dispatcher to bound the number of
    /// concurrent language groups, preventing deadlock when the number of
    /// languages × `workers_per_key` exceeds the global cap.
    pub const fn effective_max_total_workers(&self) -> usize {
        self.config.effective_max_total_workers()
    }

    /// The admitted Stanza language registry, or why there is none (no
    /// worker has reported, Stanza is not installed, or the report failed);
    /// every morphosyntax admission refuses with that reason.
    pub fn stanza_registry(&self) -> crate::stanza_registry::StanzaRegistryLookup {
        self.stanza_registry.current()
    }

    /// Capability double for deterministic tests; never spawns a model worker.
    #[cfg(test)]
    pub(crate) fn with_test_stanza_registry(
        registry: crate::stanza_registry::StanzaRegistry,
    ) -> Self {
        let pool = Self::new(PoolConfig::for_tests());
        pool.stanza_registry.record(
            "test".to_owned(),
            crate::stanza_registry::StanzaCatalogReport::Catalog {
                languages: registry.into_catalog(),
            },
        );
        pool
    }

    /// Admit one worker's capability report and record the outcome for `key`.
    ///
    /// The ONE place a report is admitted. The outcome is stored per worker
    /// key, replacing that key's previous one (a lazily loaded model names its
    /// engine only after it loads). An admitted report is returned, and the
    /// first one also becomes the pool-wide view. A refused report is recorded
    /// as that key's refusal, which `/health` reports, and is also the
    /// caller's error, typed so the caller can retire the worker it came from.
    /// The Stanza registry is populated from the first admitted report that
    /// carries it.
    pub(super) fn record_capabilities(
        &self,
        key: &WorkerKey,
        caps: WorkerCapabilities,
    ) -> Result<Arc<WorkerEngineReports>, crate::engine_reports::EngineReportAdmissionError> {
        let WorkerCapabilities {
            infer_tasks,
            engine_versions,
            stanza_catalog,
            ..
        } = caps;
        let admitted = match WorkerEngineReports::admit(&infer_tasks, engine_versions) {
            Ok(reports) => Arc::new(reports),
            Err(refusal) => {
                warn!(
                    worker_key = %key.label(),
                    %refusal,
                    "Refused worker capabilities; this worker key is not used until it reports again"
                );
                lock_recovered(&self.admitted_capabilities)
                    .insert(key.clone(), CapabilityAdmission::Refused(refusal.clone()));
                return Err(refusal);
            }
        };
        // A catalog becomes the registry; any other report is the reason
        // there is none, which every morphosyntax refusal then names.
        if let crate::stanza_registry::StanzaCatalogReport::Failed { reason } = &stanza_catalog {
            warn!(
                worker_key = %key.label(),
                %reason,
                "The worker could not build its Stanza catalog report"
            );
        }
        self.stanza_registry.record(key.label(), stanza_catalog);
        info!(
            target = %key.target.label(),
            lang = %key.language,
            engine_reports = ?admitted,
            "Admitted worker capabilities"
        );
        let _ = self.first_admitted_capabilities.set(Arc::clone(&admitted));
        lock_recovered(&self.admitted_capabilities).insert(
            key.clone(),
            CapabilityAdmission::Admitted(Arc::clone(&admitted)),
        );
        Ok(admitted)
    }

    /// The latest capability admission outcome of every worker key that has
    /// reported, sorted by key label: what `/health` returns, so an operator
    /// sees which workers were admitted with which tasks, and why any were
    /// refused.
    pub fn capability_admissions(&self) -> Vec<crate::api::WorkerCapabilityAdmission> {
        let mut admissions: Vec<crate::api::WorkerCapabilityAdmission> =
            lock_recovered(&self.admitted_capabilities)
                .iter()
                .map(|(key, admission)| crate::api::WorkerCapabilityAdmission {
                    worker_key: key.label(),
                    outcome: match admission {
                        CapabilityAdmission::Admitted(reports) => {
                            crate::api::CapabilityAdmissionOutcome::Admitted {
                                infer_tasks: reports.tasks().collect(),
                            }
                        }
                        CapabilityAdmission::Refused(refusal) => {
                            crate::api::CapabilityAdmissionOutcome::Refused {
                                reason: refusal.clone(),
                            }
                        }
                    },
                })
                .collect();
        admissions.sort_by(|left, right| left.worker_key.cmp(&right.worker_key));
        admissions
    }

    /// Record the registry daemons one discovery sweep refused to adopt,
    /// replacing the previous sweep's list.
    pub(super) fn record_refused_registry_workers(
        &self,
        refused: Vec<crate::api::RefusedRegistryWorker>,
    ) {
        *lock_recovered(&self.refused_registry_workers) = refused;
    }

    /// The registry daemons the latest discovery sweep found alive and refused
    /// to adopt, with why: what `/health` returns.
    pub fn refused_registry_workers(&self) -> Vec<crate::api::RefusedRegistryWorker> {
        lock_recovered(&self.refused_registry_workers).clone()
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

        let worker = slot
            .worker_or_init(|| async move {
                // Built only when a worker is actually spawned: a live slot
                // returns its worker without re-resolving models or
                // re-serializing spawn overrides on every dispatch.
                let config = self.worker_config(key)?;
                let mut handle = self
                    .spawn_runtime
                    .spawn_worker(config, idle_eviction::IdleWorkers(self.groups.clone()))
                    .await?;
                self.admit_worker_runtime(&handle)?;
                if self.first_admitted_capabilities.get().is_none() {
                    match self.detect_capabilities_from_worker(key, &mut handle).await {
                        Ok(()) => {}
                        // Not published into the slot, which stays empty;
                        // `handle` drops here and the process is terminated.
                        Err(CapabilityProbeFailure::Refused(refusal)) => {
                            return Err(refusal.into());
                        }
                        Err(CapabilityProbeFailure::Unanswered(error)) => {
                            tracing::warn!(error = %error, "Failed to detect capabilities from first GPU worker (continuing)");
                        }
                    }
                }
                info!(
                    target = %key.target.label(),
                    lang = %key.language,
                    pid = %handle.pid(),
                    "Shared GPU worker spawned (serves requests concurrently)"
                );
                self.spawn_runtime.adopt_shared_gpu_worker(handle).await
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
            worker.shutdown(shared_gpu::Retirement::PoolShutdown).await;
            return Err(WorkerError::PoolShuttingDown);
        }

        Ok(worker)
    }

    /// Query capabilities from an already-spawned worker and cache the result.
    ///
    /// Called once after the first worker spawn. The `OnceLock` ensures this
    /// only runs once even under concurrent job dispatch.
    pub(super) async fn detect_capabilities_from_worker(
        &self,
        key: &WorkerKey,
        handle: &mut WorkerHandle,
    ) -> Result<(), CapabilityProbeFailure> {
        if self.first_admitted_capabilities.get().is_some() {
            return Ok(()); // Already detected
        }

        let caps = handle
            .capabilities()
            .await
            .map_err(CapabilityProbeFailure::Unanswered)?;
        self.record_capabilities(key, caps)
            .map_err(CapabilityProbeFailure::Refused)?;
        Ok(())
    }

    /// The first admitted capability report, or `None` if no worker has
    /// answered yet.
    pub fn detected_capabilities(&self) -> Option<&Arc<WorkerEngineReports>> {
        self.first_admitted_capabilities.get()
    }

    /// The server instance ID assigned at pool creation.
    /// Resolve the worker-registry file this pool reads and writes.
    ///
    /// No configured path means "the ambient default"
    /// (`BATCHALIGN_STATE_DIR`, else `~/.batchalign3/workers.json`). Resolved in
    /// one place because three callers need it (discovery, shutdown, and the
    /// Drop safety net) and a pool that discovers from one file while retiring
    /// daemons from another would leak exactly the daemons it started.
    pub(super) fn registry_path(&self) -> std::path::PathBuf {
        match &self.config.worker_registry_path {
            Some(path) => path.clone(),
            None => crate::worker::registry::default_registry_path(),
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

/// Workers that were dispatching for a cancelled job, detached from the
/// tracker by [`WorkerPool::detach_workers_for_job`].
///
/// `#[must_use]`: detaching without terminating leaves exactly the in-flight
/// work the cancel exists to stop.
#[must_use = "detached workers keep computing until they are terminated"]
pub struct JobWorkersToTerminate {
    job_id: crate::api::JobId,
    pids: Vec<crate::worker::WorkerPid>,
}

impl JobWorkersToTerminate {
    /// SIGTERM every detached worker, escalating to SIGKILL after a 2s grace.
    pub async fn terminate(self) {
        if self.pids.is_empty() {
            return;
        }
        tracing::info!(
            job_id = %self.job_id,
            worker_count = self.pids.len(),
            "Cancel: terminating in-flight workers for job"
        );
        job_tracker::signal_workers(&self.pids, std::time::Duration::from_secs(2)).await;
    }
}

// V2 execute key resolution helpers live in execute_v2.rs, used by dispatch.rs.

#[cfg(test)]
mod capability_admission_tests {
    use super::*;
    use crate::api::{CapabilityAdmissionOutcome, LanguageCode3, ReportedEngineName};
    use crate::engine_reports::EngineReportAdmissionError;

    fn report(entries: &[(InferTask, Option<&str>)]) -> WorkerCapabilities {
        WorkerCapabilities {
            commands: Vec::new(),
            free_threaded: false,
            infer_tasks: entries.iter().map(|(task, _)| *task).collect(),
            engine_versions: entries
                .iter()
                .map(|(task, name)| {
                    (
                        *task,
                        name.map(|name| ReportedEngineName::try_from(name).expect("valid name")),
                    )
                })
                .collect(),
            stanza_catalog: crate::stanza_registry::StanzaCatalogReport::NotInstalled {
                import_error: crate::stanza_registry::reason("No module named 'stanza'"),
            },
        }
    }

    /// A refused report is not log-only: it is the key's recorded state,
    /// returned for `/health`, until the key reports again and is admitted.
    /// Guards: the keyed capability store publishes the exact refusal and replaces it only after an admitted report.
    /// Kind: contract
    /// Not a type: actual report bytes select admission and the observable replacement policy remains a mutable store effect.
    /// State: a local empty declared pool exercises its real in-memory report
    /// and registry stores. No worker, model, process or filesystem is created.
    /// Concurrency: this test owns every key and store, with no shared mutation.
    #[test]
    fn a_refused_report_is_returned_with_its_key_and_reason_until_replaced() {
        let pool = WorkerPool::new(PoolConfig::for_tests());
        let key = WorkerKey::without_engine_selection(
            WorkerTarget::profile(WorkerProfile::Gpu),
            WorkerLanguage::from(LanguageCode3::eng()),
        );
        let mut broken = report(&[(InferTask::Fa, None)]);
        broken.engine_versions.clear();

        let refusal = pool
            .record_capabilities(&key, broken)
            .expect_err("an advertised task with no engine entry is refused");
        assert_eq!(
            refusal,
            EngineReportAdmissionError::MissingEngineReport {
                task: InferTask::Fa
            }
        );
        assert!(pool.detected_capabilities().is_none());
        assert_eq!(
            pool.capability_admissions(),
            vec![crate::api::WorkerCapabilityAdmission {
                worker_key: key.label(),
                outcome: CapabilityAdmissionOutcome::Refused {
                    reason: EngineReportAdmissionError::MissingEngineReport {
                        task: InferTask::Fa
                    },
                },
            }]
        );

        pool.record_capabilities(&key, report(&[(InferTask::Fa, Some("wave2vec-fa-v1"))]))
            .expect("a complete report is admitted");
        assert_eq!(
            pool.capability_admissions(),
            vec![crate::api::WorkerCapabilityAdmission {
                worker_key: key.label(),
                outcome: CapabilityAdmissionOutcome::Admitted {
                    infer_tasks: vec![InferTask::Fa],
                },
            }]
        );
    }

    /// RED FIRST: a worker whose Stanza table build failed reported an empty
    /// map, which the pool read as "no Stanza", so morphotag refused every
    /// language without a reason. The failure now reaches the registry
    /// lookup, and so every morphotag refusal, with the worker's reason.
    /// Guards: the actual admitted worker key and Stanza failure reason reach registry lookup unchanged.
    /// Kind: contract
    /// Not a type: externally supplied Stanza failure text must survive the actual registry publication and lookup policy.
    /// State: a local empty declared pool exercises its real in-memory report
    /// and registry stores. No worker, model, process or filesystem is created.
    /// Concurrency: this test owns every key and store, with no shared mutation.
    #[test]
    fn a_failed_stanza_report_is_the_registrys_stated_reason() {
        let pool = WorkerPool::new(PoolConfig::for_tests());
        let key = WorkerKey::without_engine_selection(
            WorkerTarget::profile(WorkerProfile::Stanza),
            WorkerLanguage::from(LanguageCode3::eng()),
        );
        let mut failed = report(&[(InferTask::Morphosyntax, None)]);
        failed.stanza_catalog = crate::stanza_registry::StanzaCatalogReport::Failed {
            reason: crate::stanza_registry::reason(
                "ImportError: cannot import name 'ISO_639_1_PAIRS'",
            ),
        };
        pool.record_capabilities(&key, failed)
            .expect("the engine report itself is admissible");
        let unavailable = pool
            .stanza_registry()
            .map(|_| ())
            .expect_err("a failed report admits no registry");
        assert_eq!(
            unavailable,
            crate::stanza_registry::StanzaRegistryUnavailable::ReportFailed {
                worker: key.label(),
                reason: crate::stanza_registry::reason(
                    "ImportError: cannot import name 'ISO_639_1_PAIRS'"
                ),
            }
        );
    }
}

#[cfg(test)]
mod default_pool_config_tests {
    use super::*;

    use super::PerProfile;
    use crate::api::{LanguageCode3, WorkerLanguage};
    use crate::types::engines::{AsrEngineName, EngineOverrides};
    use crate::types::options::{AlignOptions, CommandOptions};
    use crate::worker::{InferTask, WorkerTarget};

    /// Guards: A command's model plan determines its GPU key despite conflicting engine options and unrelated provider extras.
    /// Kind: policy
    /// Not a type: The route's private key prevents retargeting an admitted plan. Choosing the command plan over crowded incoming options remains producer policy exercised at command admission.
    /// State: Pure typed command options and a model plan require no worker, host probe or model loading.
    /// Concurrency: none
    #[test]
    fn a_gpu_commands_key_comes_from_its_plan_not_its_options() {
        let mut options = AlignOptions::default();
        options.common.engine_overrides = EngineOverrides {
            asr: Some(AsrEngineName::RevAi),
            fa: Some(crate::types::engines::FaEngineName::Whisper),
            extras: [("provider_region".to_owned(), "us-east".to_owned())].into(),
            ..EngineOverrides::default()
        };
        let plan = GpuModelPlan::alignment(crate::types::engines::FaEngineName::Wave2Vec, None);
        for mode in [
            WorkerBootstrapMode::Profile,
            WorkerBootstrapMode::LazyProfile,
        ] {
            let key = WorkerKey::for_command(
                crate::command_model::command_spec(ReleasedCommand::Align)
                    .capabilities
                    .require_inference()
                    .expect("align requires inference"),
                WorkerLanguage::from(LanguageCode3::eng()),
                &CommandOptions::Align(options.clone()),
                &plan,
                mode,
            );
            assert_eq!(key.target.profile_kind(), WorkerProfile::Gpu);
            assert_eq!(
                key.engine_selection.worker_config_json(),
                r#"{"fa":"wave2vec"}"#,
                "{mode:?}"
            );
        }
    }

    /// Guards: An actual failed executable spawn releases original capacity, and releasing a held startup slot wakes its own-group waiter to retry without waiting for a report interval.
    /// Kind: boundary
    /// Not a type: Private slot ownership excludes forged refunds and owns rollback. The operating system can still refuse executable creation, and that real error must unwind through the normal spawn producer without retaining capacity.
    /// State: An isolated pool uses the declared test host and an intentionally missing executable path. The native process adapter must report the spawn failure. No subprocess succeeds and no model, socket or inference server is launched.
    /// Concurrency: One current-thread runtime owns the pool and observations. The failed spawn is awaited. One explicit poll establishes the blocked checkout before releasing its held slot. All capacity remains local, with no shared mutable environment or elapsed-time assumptions.
    #[tokio::test(flavor = "current_thread")]
    async fn spawn_failure_releases_permit() {
        // Force WorkerHandle::spawn to fail by pointing python_path
        // at an unrunnable binary. The Err arm in try_spawn_into_group
        // must refund the speculatively-acquired permit so other
        // groups don't see the slot as permanently consumed.
        let pool = WorkerPool::new(PoolConfig {
            max_total_workers: 4,
            max_workers_per_key: PerProfile::uniform(1),
            python_path: "/nonexistent/python-binary-that-will-fail".to_string(),
            test_echo: false,
            // Use the shared declared test-host admission policy.
            ..PoolConfig::for_tests()
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
            group.total.count(),
            0,
            "spawn failure must roll back the per-key total"
        );

        // A released startup slot wakes an own-group waiter immediately.
        // Its retry reaches the same real executable error and refunds again.
        let held = pool
            .try_claim_spawn_slot(&group)
            .expect("hold the group's one slot");
        let mut waiting = Box::pin(pool.checkout(&key));
        assert!(futures::poll!(&mut waiting).is_pending());
        drop(held);
        let retry = waiting.await;
        assert!(
            retry.is_err(),
            "the resumed checkout reaches the missing executable"
        );
        assert_eq!(group.total.count(), 0);
        assert_eq!(pool.spawn_permits.available_permits(), baseline);
    }
}
