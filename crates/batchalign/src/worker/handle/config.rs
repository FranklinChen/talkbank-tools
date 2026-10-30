//! Configuration types for spawning Python worker processes.

use crate::api::{LanguageCode3, MemoryMb, NumSpeakers, WorkerLanguage};
use crate::host_memory::HostMemoryRuntimeConfig;
use crate::revai::load_revai_api_key;
use crate::types::runtime::MemoryTier;
use crate::worker::python::resolve_python_executable;
use crate::worker::{InferTask, WorkerBootstrapMode, WorkerProfile, WorkerTarget};

/// Runtime-owned launch inputs for one worker subprocess.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerRuntimeConfig {
    /// Whether the worker should force CPU-only model/device selection.
    pub force_cpu: bool,
    /// Explicit Apple-GPU opt-in (`--allow-mps`). MPS failures proved
    /// rare but catastrophic (Apr 2026 kernel deadlock), so CPU stays
    /// the safe default and MPS engages only on this flag; the Python
    /// side additionally keeps fp32 dtypes on MPS and never gives MPS
    /// to the speaker stage (Pyannote wrong-timestamps, upstream
    /// wontfix).
    pub allow_mps: bool,
    /// Optional Rev.AI key already resolved by the Rust control plane.
    pub revai_api_key: Option<String>,
    /// Maximum concurrent requests served inside one GPU worker process
    /// when it serves concurrently; see `worker::serving::WorkerServing`.
    pub gpu_thread_pool_size: u32,
    /// Host-memory coordination settings shared with the worker spawn path.
    pub host_memory: HostMemoryRuntimeConfig,
    /// Resolved memory tier used for startup reservations and other
    /// tier-derived worker-launch decisions.
    pub memory_tier: MemoryTier,
    /// State directory the spawned child should use, when the caller wants
    /// one other than the ambient default.
    ///
    /// The worker registry lives at `<state_dir>/workers.json`, and a child
    /// resolves it through `registry::default_registry_path()`, which reads
    /// `BATCHALIGN_STATE_DIR`. That made the directory ambient PROCESS state:
    /// the only way to point a spawned daemon somewhere else was to mutate
    /// your own process's environment, which is exactly what three
    /// `worker_integration` tests did, racing each other under cargo's
    /// parallel threads and producing off-by-one daemon counts that read like
    /// registry bugs (2026-07-29).
    ///
    /// Threading it here makes the directory an explicit input. The env var
    /// survives only as the parent-to-child handoff, set on that child's
    /// `Command`, never on the parent process.
    pub state_dir: Option<std::path::PathBuf>,
    /// Host-chosen bootstrap policy for local workers.
    pub bootstrap_mode: WorkerBootstrapMode,
    /// Unique identity for the current server instance when it owns spawned
    /// TCP daemons. `None` means spawned daemons should register as external.
    pub server_instance_id: Option<String>,
    /// PID of the current Rust server process when it owns spawned TCP daemons.
    /// `None` means there is no owning server process to record.
    pub server_process_id: Option<u32>,
}

/// The memory tier test fixtures launch with: declared, never read from the
/// machine running the tests, so a test's outcome cannot depend on how much
/// RAM that machine has or on whether the operating system reports it.
pub fn test_memory_tier() -> MemoryTier {
    MemoryTier::from_total_mb(64_000)
}

impl WorkerRuntimeConfig {
    /// Local launch inputs with the legacy defaults and the caller's tier.
    ///
    /// This was `Default`, and it read the host to find the tier: a `Default`
    /// cannot fail, so an unreadable host became whatever zero total RAM
    /// tiers to. The tier is now an argument, taken from the one reading of
    /// the host ([`HostSnapshot::tier`], whose `read` reports an unreadable
    /// host) or declared by a test ([`test_memory_tier`]).
    ///
    /// [`HostSnapshot::tier`]: crate::host_facts::HostSnapshot::tier
    pub fn local(memory_tier: MemoryTier) -> Self {
        // `gpu_thread_pool_size` is the in-process default for callers that
        // have not consulted the host-facts pipeline (`EffectiveConfig`
        // overrides it in production). `4` matches the pre-migration static
        // default; changing it would silently shift any test relying on it.
        const LEGACY_DEFAULT_GPU_THREAD_POOL_SIZE: u32 = 4;
        Self::from_sources(
            false,
            false,
            load_revai_api_key()
                .ok()
                .map(|key| key.as_str().to_string()),
            LEGACY_DEFAULT_GPU_THREAD_POOL_SIZE,
            HostMemoryRuntimeConfig::default(),
            memory_tier,
        )
    }
}

impl WorkerRuntimeConfig {
    /// Build worker runtime inputs from explicit sources.
    pub fn from_sources(
        force_cpu: bool,
        allow_mps: bool,
        revai_api_key: Option<String>,
        gpu_thread_pool_size: u32,
        host_memory: HostMemoryRuntimeConfig,
        memory_tier: MemoryTier,
    ) -> Self {
        Self {
            force_cpu,
            allow_mps,
            revai_api_key: revai_api_key
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty()),
            gpu_thread_pool_size,
            host_memory,
            memory_tier,
            bootstrap_mode: WorkerBootstrapMode::Profile,
            server_instance_id: None,
            server_process_id: None,
            state_dir: None,
        }
    }

    /// Override the host bootstrap mode for worker spawns built from this runtime.
    pub fn with_bootstrap_mode(mut self, bootstrap_mode: WorkerBootstrapMode) -> Self {
        self.bootstrap_mode = bootstrap_mode;
        self
    }

    /// Direct spawned children at an explicit state directory.
    ///
    /// See [`WorkerRuntimeConfig::state_dir`]: this is how a caller points a
    /// worker's registry somewhere other than the ambient default without
    /// mutating its own process environment.
    pub fn with_state_dir(mut self, state_dir: std::path::PathBuf) -> Self {
        self.state_dir = Some(state_dir);
        self
    }
}

/// Configuration for spawning a worker.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Path to the Python executable (e.g. "python3", "/usr/bin/python3.14t").
    pub python_path: String,
    /// Worker profile describing which task group this worker owns.
    pub profile: WorkerProfile,
    /// Optional infer task for task-targeted bootstrap.
    pub task: Option<InferTask>,
    /// Worker-runtime language string.
    pub lang: WorkerLanguage,
    /// Number of speakers.
    pub num_speakers: NumSpeakers,
    /// Engine overrides as JSON string (empty = none).
    pub engine_overrides: String,
    /// Use test-echo mode (no ML models).
    pub test_echo: bool,
    /// Longest wait for the worker to become ready.
    pub ready_timeout_s: crate::api::PositiveSeconds,
    /// Verbosity level (0=warn, 1=info, 2=debug, 3+=trace).
    /// Forwarded to the Python worker via `--verbose N` to control its logging
    /// level, enabling end-to-end verbosity from a single CLI `-v` flag.
    pub verbose: u8,
    /// Runtime-owned launch inputs resolved before this spawn boundary.
    pub runtime: WorkerRuntimeConfig,
    /// The operator's transport-ceiling overrides; each absent one leaves the
    /// task's built-in ceiling.
    pub task_timeouts: crate::types::worker_v2::TaskTimeoutOverrides,
    /// Test-only: artificial delay in milliseconds before each response.
    /// 0 = no delay. Only effective when `test_echo` is true.
    pub test_delay_ms: u64,
}

impl WorkerConfig {
    /// A worker launch with every field at its default except `runtime`.
    /// Private: production launches are built field by field where the pool
    /// or a replay knows each value, so no default stands in for one of them.
    fn with_runtime(runtime: WorkerRuntimeConfig) -> Self {
        Self {
            python_path: resolve_python_executable(),
            profile: WorkerProfile::Stanza,
            task: None,
            lang: WorkerLanguage::from(LanguageCode3::eng()),
            num_speakers: NumSpeakers(1),
            engine_overrides: String::new(),
            test_echo: false,
            ready_timeout_s: crate::api::PositiveSeconds::literal::<300>(),
            verbose: 0,
            runtime,
            task_timeouts: crate::types::worker_v2::TaskTimeoutOverrides::NONE,
            test_delay_ms: 0,
        }
    }

    /// A test worker launch at the declared [`test_memory_tier`].
    pub fn for_tests() -> Self {
        Self::with_runtime(WorkerRuntimeConfig::local(test_memory_tier()))
    }
}

impl WorkerConfig {
    /// Return the actual bootstrap target for this worker spawn.
    pub fn bootstrap_target(&self) -> WorkerTarget {
        match self.task {
            Some(task) => WorkerTarget::infer_task(task),
            None => WorkerTarget::profile(self.profile),
        }
    }

    /// Return the human-readable bootstrap label for logs and status.
    pub fn bootstrap_label(&self) -> String {
        self.bootstrap_target().label()
    }

    /// Resolve the startup reservation for this worker spawn using the
    /// runtime-owned memory tier rather than raw host auto-detection.
    pub fn startup_reservation_mb(&self) -> MemoryMb {
        self.profile
            .startup_reservation_mb_for_tier(&self.runtime.memory_tier)
    }
}

/// Which worker a spawn made: every launch input that decides what the
/// worker serves and on what device, as a failed-request dump records it and
/// `batchalign3 replay` rebuilds the worker from.
///
/// Built only from a [`WorkerConfig`] ([`Self::of`]), so a dump records the
/// worker that actually ran rather than a label a reader re-parses; and turned
/// back into one only by [`Self::into_config`], so the replay spawns through
/// the pool's own command builder. A Qwen3 ASR or Cantonese FA dump replays on
/// a worker with the same engine overrides, profile and device flags, not on
/// a default-engine worker for the request's task.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct WorkerLaunch {
    /// The profile the worker was spawned for.
    profile: WorkerProfile,
    /// The infer task it was bootstrapped for, when task-targeted.
    task: Option<InferTask>,
    /// The worker language, in its worker-argument spelling.
    #[serde(with = "worker_language_arg")]
    lang: WorkerLanguage,
    /// Speakers it was told to expect.
    num_speakers: NumSpeakers,
    /// The `--engine-overrides` JSON (empty for none).
    engine_overrides: String,
    /// `--force-cpu`.
    force_cpu: bool,
    /// `--allow-mps`.
    allow_mps: bool,
    /// Whether it loaded its models eagerly or on demand.
    bootstrap_mode: WorkerBootstrapMode,
    /// Its concurrent-request capacity.
    gpu_thread_pool_size: u32,
}

impl WorkerLaunch {
    /// The launch inputs of a worker spawned from `config`.
    pub(crate) fn of(config: &WorkerConfig) -> Self {
        Self {
            profile: config.profile,
            task: config.task,
            lang: config.lang.clone(),
            num_speakers: config.num_speakers,
            engine_overrides: config.engine_overrides.clone(),
            force_cpu: config.runtime.force_cpu,
            allow_mps: config.runtime.allow_mps,
            bootstrap_mode: config.runtime.bootstrap_mode,
            gpu_thread_pool_size: config.runtime.gpu_thread_pool_size,
        }
    }

    /// Whether the worker loads a task's models only on `ensure_task`, as
    /// the pool asks before it dispatches.
    pub(crate) fn loads_on_demand(&self) -> bool {
        match self.bootstrap_mode {
            WorkerBootstrapMode::LazyProfile => true,
            WorkerBootstrapMode::Profile | WorkerBootstrapMode::Task => false,
        }
    }

    /// The worker's language.
    pub(crate) fn lang(&self) -> &WorkerLanguage {
        &self.lang
    }

    /// The same launch under another language: the operator's explicit
    /// choice on replay.
    pub(crate) fn with_lang(self, lang: WorkerLanguage) -> Self {
        Self { lang, ..self }
    }

    /// A worker config that spawns this worker again with `python_path`;
    /// everything not recorded (timeouts, verbosity, credentials, and the
    /// host's `memory_tier`) is this process's own.
    pub(crate) fn into_config(self, python_path: String, memory_tier: MemoryTier) -> WorkerConfig {
        let mut runtime =
            WorkerRuntimeConfig::local(memory_tier).with_bootstrap_mode(self.bootstrap_mode);
        runtime.force_cpu = self.force_cpu;
        runtime.allow_mps = self.allow_mps;
        runtime.gpu_thread_pool_size = self.gpu_thread_pool_size;
        // Spelled out in full: a replay is a diagnostic launch, so it never
        // runs the echo test path, keeps the default ready budget and quiet
        // logging, and applies no operator task-timeout overrides.
        WorkerConfig {
            python_path,
            profile: self.profile,
            task: self.task,
            lang: self.lang,
            num_speakers: self.num_speakers,
            engine_overrides: self.engine_overrides,
            test_echo: false,
            ready_timeout_s: crate::api::PositiveSeconds::literal::<300>(),
            verbose: 0,
            runtime,
            task_timeouts: crate::types::worker_v2::TaskTimeoutOverrides::NONE,
            test_delay_ms: 0,
        }
    }
}

/// [`WorkerLanguage`] as its worker-argument string, refused on the way in
/// when it is not one.
mod worker_language_arg {
    use crate::api::WorkerLanguage;

    pub(super) fn serialize<S: serde::Serializer>(
        lang: &WorkerLanguage,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(lang.as_worker_arg())
    }

    pub(super) fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<WorkerLanguage, D::Error> {
        let text = <String as serde::Deserialize>::deserialize(deserializer)?;
        WorkerLanguage::parse_untrusted(&text).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::{WorkerConfig, WorkerRuntimeConfig};
    use crate::host_memory::HostMemoryRuntimeConfig;
    use crate::types::runtime::MemoryTier;
    use crate::worker::{InferTask, WorkerProfile};

    #[test]
    fn startup_reservation_uses_runtime_memory_tier() {
        let runtime = WorkerRuntimeConfig::from_sources(
            false,
            false,
            None,
            4,
            HostMemoryRuntimeConfig::default(),
            MemoryTier::from_total_mb(16_000),
        );
        let config = WorkerConfig {
            profile: WorkerProfile::Stanza,
            runtime,
            ..WorkerConfig::for_tests()
        };
        assert_eq!(config.startup_reservation_mb().0, 3_000);
    }

    #[test]
    fn bootstrap_target_uses_task_when_present() {
        let config = WorkerConfig {
            profile: WorkerProfile::Stanza,
            task: Some(InferTask::Morphosyntax),
            ..WorkerConfig::for_tests()
        };
        assert_eq!(config.bootstrap_label(), "infer:morphosyntax");
    }
}
