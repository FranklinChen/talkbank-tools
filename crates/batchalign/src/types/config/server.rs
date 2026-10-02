//! `ServerConfig` struct and serde defaults.
//!
//! The main configuration type deserialized from `server.yaml`. All fields
//! have sensible defaults so an empty YAML file (or a missing file) produces
//! a working configuration. The `FleetTarget` sub-struct also lives here.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::serde_helpers::{
    empty_path_as_none, zero_as_default, zero_as_no_override, zero_as_none,
};
use super::{JobTtlDays, LeaseTtl, MemoryGatePollSeconds, WorkerStartupLimit};
use crate::api::{LanguageCode3, MemoryMb, PositiveSeconds};
use crate::types::worker_v2::TaskTimeoutOverrides;

/// Configuration for the Batchalign processing server.
///
/// Deserialized from the runtime-owned `server.yaml`. All fields have sensible
/// defaults so an empty YAML file (or a missing file) produces a working
/// configuration. A scalar below its floor (`job_ttl_days`,
/// `memory_gate_poll_s`, `max_concurrent_worker_startups`,
/// `local_lease_ttl_s`) is refused where the file is read, naming the field;
/// nothing is corrected silently.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// Filesystem directories the server searches when resolving media files
    /// for transcribe/align.  Availability is checked only when media is requested,
    /// never while loading configuration.
    #[serde(default)]
    pub media_roots: Vec<batchalign_types::paths::ServerPath>,
    /// Named media directory mappings (e.g. `{"childes-data": "/nfs/childes"}`).
    /// Clients reference the key in `JobSubmission.media_mapping`; the server
    /// resolves it to the filesystem root.  Allows stable logical names even
    /// when mount paths change.
    #[serde(default)]
    pub media_mappings:
        BTreeMap<batchalign_types::paths::MediaMappingKey, batchalign_types::paths::ServerPath>,
    /// Server log verbosity level: 0=warn (default), 1=info, 2=debug.
    /// Controls the tracing filter when the server starts in daemon mode.
    /// Equivalent to the number of `-v` flags on the CLI.
    /// Set to 1 in fleet server.yaml for production observability.
    #[serde(default)]
    pub verbose: u8,
    /// 3-letter ISO language code used when the client omits `lang`.
    /// Defaults to `"eng"`.
    #[serde(default = "default_lang")]
    pub default_lang: LanguageCode3,
    /// Operator override for the maximum number of jobs processed in
    /// parallel. `None` (the canonical post-migration form) means
    /// "let `JobStore::new` auto-tune from the resolved memory tier
    /// plus CPU availability via
    /// `HostExecutionPolicy::auto_max_concurrent_jobs`": Phase B4 of
    /// the host-facts migration replaces that helper with
    /// `EffectiveConfig::max_concurrent_jobs`. The `zero_as_none`
    /// shim collapses pre-migration `max_concurrent_jobs: 0` to
    /// `None`; the field is unsigned, so the legacy sentinel has
    /// migrated to the type system.
    #[serde(
        default,
        deserialize_with = "zero_as_none",
        skip_serializing_if = "Option::is_none"
    )]
    pub max_concurrent_jobs: Option<u32>,
    /// TCP port for the HTTP server.
    ///
    /// A REQUEST, not the port the server ends up on: `0` asks the OS to
    /// choose, and the granted port is published in the server handshake
    /// rather than predicted from here. Default: 8000.
    #[serde(default)]
    pub port: super::PortRequest,
    /// Bind address for the HTTP server.  Default: `"0.0.0.0"` (all interfaces).
    #[serde(default = "default_host")]
    pub host: String,

    /// Operator override for the worker-side "skip GPU detection,
    /// force CPU-only inference" mode. `None` (the canonical post-
    /// migration form) means "let `EffectiveConfig::resolve` derive
    /// the value from `HostFacts`": today
    /// `recommend_force_cpu` returns `true` when the GPU is
    /// non-functional for batchalign (Apple Silicon MPS, no GPU,
    /// CUDA with `device_count == 0`) and `false` on functional
    /// CUDA hosts. The CLI `--force-cpu` switch is converted to
    /// `Some(true)` at the builder boundary; an operator can also
    /// set `force_cpu: false` in `server.yaml` to assert "use GPU
    /// even on a host where the recommendation would force CPU"
    /// (the validator may warn).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force_cpu: Option<bool>,

    /// Explicit Apple-GPU opt-in. `Some(true)` lets engines use MPS
    /// (still CUDA > MPS > CPU, still fp32 dtypes, never the speaker
    /// stage). `None`/`Some(false)` keep the safe CPU default: MPS
    /// failures proved rare but catastrophic (Apr 2026 kernel
    /// deadlock), so bravery is opt-in per operator.
    pub allow_mps: Option<bool>,
    /// Operator override for the per-job worker-process cap. `None`
    /// (the canonical post-migration form) means "let
    /// `EffectiveConfig::max_workers_per_job(command)` derive the
    /// per-command value from `HostFacts`": today the recommendation
    /// formula matches the legacy `compute_job_workers` auto-tune
    /// logic. `Some(n)` applies uniformly across every command (the
    /// legacy single-knob shape). The `zero_as_none` shim collapses
    /// pre-migration `max_workers_per_job: 0` to `None`. Phase G2 of
    /// the host-facts migration removes the shim.
    #[serde(
        default,
        deserialize_with = "zero_as_none",
        skip_serializing_if = "Option::is_none"
    )]
    pub max_workers_per_job: Option<u32>,
    /// Number of days to retain completed/failed job metadata in SQLite
    /// before automatic purge. Must be >= 1; a smaller value is refused
    /// where `server.yaml` is read. Default: 7.
    #[serde(default = "default_job_ttl_days")]
    pub job_ttl_days: JobTtlDays,
    /// Whether the CLI should auto-spawn a local daemon when no explicit
    /// `--server` is configured. Default: `true`.
    #[serde(default = "default_true")]
    pub auto_daemon: bool,
    /// Operator override for the host-memory headroom (MB) the
    /// coordinator keeps free after worker-start and job-execution
    /// reservations. `None` (the canonical post-migration form) means
    /// "let `resolved_memory_gate_mb` derive the value from
    /// `resolved_memory_tier().headroom_mb`": that's tier-aware and
    /// honors any `memory_tier` override. The `zero_as_none` shim
    /// collapses pre-migration `memory_gate_mb: 0` (the legacy
    /// "disable host-memory checks" sentinel and the value used by
    /// `--sequential` mode) to `None`; setting an explicit small
    /// value like `MemoryMb(1)` still expresses "effectively disabled
    /// without skipping resolution".
    #[serde(
        default,
        deserialize_with = "zero_as_none",
        skip_serializing_if = "Option::is_none"
    )]
    pub memory_gate_mb: Option<MemoryMb>,
    /// Seconds between worker health checks. Default: 30. Never zero; a
    /// legacy `0`, which meant the default, is read as the default.
    #[serde(
        default = "default_worker_health_interval_s",
        deserialize_with = "zero_as_default::<_, DEFAULT_WORKER_HEALTH_INTERVAL_S>"
    )]
    pub worker_health_interval_s: PositiveSeconds,

    /// Maximum number of local worker/model startups allowed at once across all
    /// participating batchalign3 processes on the host. Default: 1.
    #[serde(default = "default_max_concurrent_worker_startups")]
    pub max_concurrent_worker_startups: WorkerStartupLimit,

    /// Operator override for the maximum Python worker processes per
    /// `(profile, lang, engine)` key. `None` (the canonical
    /// post-migration form) means "use the built-in pool default
    /// (DEFAULT_MAX_WORKERS_PER_KEY = 8)". `Some(n)` applies
    /// uniformly across every profile (the legacy single-knob shape);
    /// the host-facts pipeline carries this through
    /// `ConfigOverrides::max_workers_per_key_by_profile` (each of
    /// gpu/stanza/io receives the same `Some(n)`). The `zero_as_none`
    /// shim collapses pre-migration `max_workers_per_key: 0` to
    /// `None`. A future PR introducing a per-profile `ServerConfig`
    /// shape would let operators set differentiated values; until
    /// then, the recommendation's per-profile RAM-derived formula
    /// (gpu = ram/16GB, stanza = ram/12GB, io = 1) lives only in
    /// `EffectiveConfig::max_workers_per_key_by_profile`.
    #[serde(
        default,
        deserialize_with = "zero_as_none",
        skip_serializing_if = "Option::is_none"
    )]
    pub max_workers_per_key: Option<u32>,

    /// Operator override for the hard ceiling on total workers across
    /// all `(profile, lang, engine)` keys. `None` (the canonical
    /// post-migration form) means "let `EffectiveConfig::resolve`
    /// derive the value from `HostFacts`": today
    /// `recommend_max_total_workers` returns
    /// `clamp(ram_total_mb / 6GB, 2, 32)`, with a fallback to 4 when
    /// `ram_total_mb` is zero. The `zero_as_none` shim collapses
    /// legacy `max_total_workers: 0` from pre-migration `server.yaml`
    /// files to `None` so deployed configs continue to mean "auto".
    /// Phase G2 of the host-facts migration removes the shim once
    /// every fleet `server.yaml` has been re-rendered.
    #[serde(
        default,
        deserialize_with = "zero_as_none",
        skip_serializing_if = "Option::is_none"
    )]
    pub max_total_workers: Option<u32>,

    /// Seconds to wait for a Python worker to become ready after spawn.
    /// Default: 300. Never zero; a legacy `0`, which meant the default, is
    /// read as the default.
    #[serde(
        default = "default_worker_ready_timeout_s",
        deserialize_with = "zero_as_default::<_, DEFAULT_WORKER_READY_TIMEOUT_S>"
    )]
    pub worker_ready_timeout_s: PositiveSeconds,

    /// Maximum HTTP request body size in megabytes. Default: 100.
    #[serde(default = "default_max_body_bytes_mb")]
    pub max_body_bytes_mb: MemoryMb,

    /// Seconds to wait for host-memory reservations to fit before rejecting or
    /// deferring a job. Default: 120. Never zero: a configured `0` is refused
    /// when the configuration is read.
    #[serde(default = "default_memory_gate_timeout_s")]
    pub memory_gate_timeout_s: PositiveSeconds,

    /// Seconds between host-memory reservation polling checks. Default: 5.
    #[serde(default = "default_memory_gate_poll_s")]
    pub memory_gate_poll_s: MemoryGatePollSeconds,

    /// Low-memory warning threshold in MB. Default: 4096.
    #[serde(default = "default_memory_warning_mb")]
    pub memory_warning_mb: MemoryMb,

    /// Operator override for the GPU worker's in-process dispatch
    /// concurrency. `None` (the canonical post-migration form) means
    /// "let `EffectiveConfig::resolve` derive the value from
    /// `HostFacts`": today that yields `4` on CUDA-functional hosts
    /// and `1` on hosts where the GPU is non-functional (Apple Silicon
    /// MPS, no GPU). The `zero_as_none` shim collapses legacy
    /// `gpu_thread_pool_size: 0` from pre-migration `server.yaml`
    /// files to `None` so deployed configs continue to mean "auto".
    /// Phase G2 of the host-facts migration removes the shim once
    /// every fleet `server.yaml` has been re-rendered.
    #[serde(
        default,
        deserialize_with = "zero_as_none",
        skip_serializing_if = "Option::is_none"
    )]
    pub gpu_thread_pool_size: Option<u32>,

    /// How long a locally-dispatched lease lives without renewal before it
    /// is considered orphaned; written as `local_lease_ttl_s` (whole seconds,
    /// default 300) and refused unless longer than the lease heartbeat.
    #[serde(default = "default_local_lease_ttl", rename = "local_lease_ttl_s")]
    pub local_lease_ttl: LeaseTtl,

    /// Override, in seconds, of the transport ceiling for audio-heavy worker
    /// tasks (ASR, FA, speaker). Absent: the built-in ceiling (1800 for FA
    /// and speaker; ASR's scales with the audio and an override can only
    /// raise it). Increase for very long recordings. A legacy `0`, which
    /// meant "absent", is read as absent.
    #[serde(
        default,
        deserialize_with = "zero_as_no_override",
        skip_serializing_if = "Option::is_none"
    )]
    pub audio_task_timeout_s: Option<PositiveSeconds>,

    /// Override, in seconds, of the transport ceiling for lightweight
    /// analysis tasks (OpenSMILE, AVQI). Absent: 120. A legacy `0` is read as
    /// absent.
    #[serde(
        default,
        deserialize_with = "zero_as_no_override",
        skip_serializing_if = "Option::is_none"
    )]
    pub analysis_task_timeout_s: Option<PositiveSeconds>,

    /// Override, in seconds, of the timeout for on-demand model loading via
    /// `ensure_task` IPC. Absent: 120. Increase on slow networks where
    /// first-time model downloads (Stanza, Whisper) may take longer. A legacy
    /// `0` is read as absent.
    #[serde(
        default,
        deserialize_with = "zero_as_no_override",
        skip_serializing_if = "Option::is_none"
    )]
    pub ensure_task_timeout_s: Option<PositiveSeconds>,

    /// Path to the worker registry file for discovering pre-started TCP
    /// workers. Absent: `workers.json` in the state directory. A legacy empty
    /// string, which meant "absent", is read as absent.
    #[serde(
        default,
        deserialize_with = "empty_path_as_none",
        skip_serializing_if = "Option::is_none"
    )]
    pub worker_registry_path: Option<std::path::PathBuf>,

    /// Override the auto-detected memory tier. When absent, the tier is
    /// detected from total system RAM. This overrides all tier-derived
    /// defaults (headroom, startup reservations, idle timeout, max workers)
    /// unless those fields are also explicitly set.
    #[serde(default)]
    pub memory_tier: Option<crate::types::runtime::MemoryTierKind>,

    /// Override GPU worker startup reservation (MB).
    ///
    /// `None` (or YAML `0`, via the `zero_as_none` shim) → use the
    /// resolved tier's default. `Some(value)` overrides unconditionally.
    /// The shim preserves wire-format compatibility with operators who
    /// have `gpu_startup_mb: 0` in `server.yaml` from before the type
    /// migration; the `0`-sentinel pattern is no longer expressible
    /// in-code (per `~/.claude/CLAUDE.md` "no sentinel values").
    #[serde(default, deserialize_with = "zero_as_none")]
    pub gpu_startup_mb: Option<MemoryMb>,

    /// Override Stanza worker startup reservation (MB). See
    /// [`Self::gpu_startup_mb`] for the `None`/`Some` semantics.
    #[serde(default, deserialize_with = "zero_as_none")]
    pub stanza_startup_mb: Option<MemoryMb>,

    /// Override IO worker startup reservation (MB). See
    /// [`Self::gpu_startup_mb`] for the `None`/`Some` semantics.
    #[serde(default, deserialize_with = "zero_as_none")]
    pub io_startup_mb: Option<MemoryMb>,

    /// Fleet server target for automatic remote execution routing.
    ///
    /// When present and non-empty, the local daemon auto-routes jobs:
    /// - Media already visible on the fleet server (NFS) → submit directly
    /// - Media only local → stage via rsync, execute remotely, copy back
    /// - Fleet server unreachable → fall back to local execution
    ///
    /// When absent or empty (the default for external `uv tool install`
    /// users), all jobs execute locally. No network probing, no SSH
    /// attempts, no fleet behavior.
    ///
    /// Deployed by Ansible to fleet machines only.
    #[serde(default)]
    pub fleet_target: Option<FleetTarget>,
}

/// Fleet server connection details.
///
/// Tells the local daemon how to reach the fleet's primary compute
/// server for automatic job routing. Deployed by Ansible, external
/// users never see this.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FleetTarget {
    /// HTTP URL of the batchalign server (e.g. `"http://your-server:8001"`).
    pub url: String,
    /// SSH host for rsync file transfer (e.g. `"your-server"`).
    pub ssh_host: String,
    /// SSH user for rsync file transfer (e.g. `"operator"`).
    pub ssh_user: String,
    /// Base directory on the remote host for staged job scratch space.
    pub scratch_base: String,
}

// ---------------------------------------------------------------------------
// Serde default functions
// ---------------------------------------------------------------------------

pub(crate) fn default_lang() -> LanguageCode3 {
    LanguageCode3::eng()
}

pub(crate) fn default_host() -> String {
    "0.0.0.0".to_string()
}

pub(crate) fn default_true() -> bool {
    true
}

pub(crate) fn default_job_ttl_days() -> JobTtlDays {
    JobTtlDays::literal::<7>()
}

/// Default seconds between worker health checks.
pub(crate) const DEFAULT_WORKER_HEALTH_INTERVAL_S: u64 = 30;

/// Default seconds to wait for a spawned worker to become ready.
pub(crate) const DEFAULT_WORKER_READY_TIMEOUT_S: u64 = 300;

pub(crate) fn default_worker_health_interval_s() -> PositiveSeconds {
    PositiveSeconds::literal::<DEFAULT_WORKER_HEALTH_INTERVAL_S>()
}

pub(crate) fn default_max_concurrent_worker_startups() -> WorkerStartupLimit {
    WorkerStartupLimit::literal::<1>()
}

pub(crate) fn default_worker_ready_timeout_s() -> PositiveSeconds {
    PositiveSeconds::literal::<DEFAULT_WORKER_READY_TIMEOUT_S>()
}

pub(crate) fn default_max_body_bytes_mb() -> MemoryMb {
    // 500 files of CHAT text from large corpora (e.g. childes-eng-uk,
    // childes-other) routinely exceed 100 MB when the CLI ships content
    // (`paths_mode=false`). 512 MB is the observed ceiling for a
    // 500-file chunk plus headroom; `server.yaml`'s `max_body_bytes_mb`
    // still overrides if a deployment needs a different ceiling.
    MemoryMb(512)
}

pub(crate) fn default_memory_gate_timeout_s() -> PositiveSeconds {
    PositiveSeconds::literal::<120>()
}

pub(crate) fn default_memory_gate_poll_s() -> MemoryGatePollSeconds {
    MemoryGatePollSeconds::literal::<5>()
}

pub(crate) fn default_memory_warning_mb() -> MemoryMb {
    MemoryMb(4096)
}

pub(crate) fn default_local_lease_ttl() -> LeaseTtl {
    LeaseTtl::DEFAULT
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            verbose: 0,
            media_roots: Vec::new(),
            media_mappings: BTreeMap::new(),
            default_lang: LanguageCode3::eng(),
            max_concurrent_jobs: None,
            port: super::PortRequest::default(),
            host: "0.0.0.0".to_string(),
            force_cpu: None,
            allow_mps: None,
            max_workers_per_job: None,
            job_ttl_days: default_job_ttl_days(),
            auto_daemon: true,
            memory_gate_mb: None,
            worker_health_interval_s: default_worker_health_interval_s(),
            max_concurrent_worker_startups: default_max_concurrent_worker_startups(),
            max_workers_per_key: None,
            max_total_workers: None,
            worker_ready_timeout_s: default_worker_ready_timeout_s(),
            max_body_bytes_mb: default_max_body_bytes_mb(),
            memory_gate_timeout_s: default_memory_gate_timeout_s(),
            memory_gate_poll_s: default_memory_gate_poll_s(),
            memory_warning_mb: default_memory_warning_mb(),
            gpu_thread_pool_size: None,
            local_lease_ttl: default_local_lease_ttl(),
            audio_task_timeout_s: None,
            analysis_task_timeout_s: None,
            ensure_task_timeout_s: None,
            worker_registry_path: None,
            memory_tier: None,
            gpu_startup_mb: None,
            stanza_startup_mb: None,
            io_startup_mb: None,
            fleet_target: None,
        }
    }
}

impl ServerConfig {
    /// The operator's transport-ceiling overrides, as the worker layer reads
    /// them.
    pub fn task_timeouts(&self) -> TaskTimeoutOverrides {
        TaskTimeoutOverrides {
            audio: self.audio_task_timeout_s,
            analysis: self.analysis_task_timeout_s,
        }
    }
}
