//! Runtime constants — loaded from `runtime_constants.toml` at compile time.
//!
//! Command-to-task mapping, memory budgets, and command classification.
//! The TOML file is the single source of truth shared with Python.

use std::collections::HashMap;
use std::sync::LazyLock;

use serde::Deserialize;

use crate::api::MemoryMb;

/// Raw TOML content, embedded at compile time.
const TOML_SRC: &str = include_str!("../../../../batchalign/runtime_constants.toml");

/// Parsed TOML structure.
#[derive(Deserialize)]
struct RuntimeConstants {
    cmd2task: HashMap<String, String>,
    worker_caps: WorkerCaps,
    memory: MemoryConstants,
    worker_startup_mb: WorkerStartupMb,
    gpu_heavy_commands: GpuHeavy,
    command_base_mb: CommandBaseMb,
    known_engine_keys: KnownEngineKeys,
}

#[derive(Deserialize)]
struct WorkerCaps {
    max_gpu_workers: usize,
    max_process_workers: usize,
    max_thread_workers: usize,
}

#[derive(Deserialize)]
struct MemoryConstants {
    default_base_mb: u64,
    mb_per_file_mb: u64,
    loading_overhead: f64,
}

#[derive(Deserialize)]
struct WorkerStartupMb {
    gpu: u64,
    stanza: u64,
    io: u64,
}

#[derive(Deserialize)]
struct GpuHeavy {
    commands: Vec<String>,
}

// Note: [process_commands] section is intentionally not deserialized here.
// Python reads it from runtime_constants.toml directly for GIL-aware dispatch
// classification. Rust selects process vs. threaded budgets via
// `is_free_threaded_runtime()` — see `command_execution_budget_mb()`.

#[derive(Deserialize)]
struct CommandBaseMb {
    process: HashMap<String, u64>,
    threaded: HashMap<String, u64>,
}

#[derive(Deserialize)]
struct KnownEngineKeys {
    keys: Vec<String>,
}

// Compile-time-constant embedded TOML — structurally validated by the test suite.
#[allow(clippy::expect_used)]
static CONSTANTS: LazyLock<RuntimeConstants> =
    LazyLock::new(|| toml::from_str(TOML_SRC).expect("runtime_constants.toml must be valid TOML"));

/// Whether the server is running in a free-threaded Python environment.
///
/// Detected once at process startup from the `PYTHON_GIL` environment variable:
/// `PYTHON_GIL=0` means CPython's GIL is disabled (Python 3.14t+). This is the
/// same variable that controls Python worker startup and is inherited by the Rust
/// server binary when launched from a free-threaded context.
///
/// When `true`:
/// - `command_execution_budget_mb()` uses the `threaded` (lower) table
/// - Stanza workers use concurrent serving (shared model via `ThreadPoolExecutor`)
///
/// When `false` (default):
/// - `command_execution_budget_mb()` uses the `process` (higher, conservative) table
/// - Stanza workers use sequential exclusive checkout
static FREE_THREADED_RUNTIME: LazyLock<bool> =
    LazyLock::new(|| std::env::var("PYTHON_GIL").as_deref() == Ok("0"));

/// Whether the current Rust server process is running alongside free-threaded Python workers.
///
/// Set by `PYTHON_GIL=0` in the environment. Used to select memory budgets and
/// dispatch routing for Stanza (CPU-bound) workers.
pub fn is_free_threaded_runtime() -> bool {
    *FREE_THREADED_RUNTIME
}

/// Command name -> pipeline task string (e.g. "align" -> "fa").
pub fn cmd2task() -> HashMap<&'static str, &'static str> {
    CONSTANTS
        .cmd2task
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect()
}

/// GPU-bound commands where MPS/CUDA is the bottleneck.
pub fn gpu_heavy_commands() -> &'static [String] {
    &CONSTANTS.gpu_heavy_commands.commands
}

/// Known engine-override keys (passed via --engine-overrides).
pub fn known_engine_keys() -> &'static [String] {
    &CONSTANTS.known_engine_keys.keys
}

/// Hard cap on concurrent GPU-bound workers (transcribe, align, benchmark).
pub fn max_gpu_workers() -> usize {
    CONSTANTS.worker_caps.max_gpu_workers
}

/// Hard cap on concurrent process-isolated workers (non-free-threaded Python).
pub fn max_process_workers() -> usize {
    CONSTANTS.worker_caps.max_process_workers
}

/// Hard cap on concurrent thread workers (free-threaded Python 3.14t+).
pub fn max_thread_workers() -> usize {
    CONSTANTS.worker_caps.max_thread_workers
}

/// Per-command base memory (MB) — non-free-threaded (process workers).
pub fn command_base_mb_process() -> HashMap<&'static str, MemoryMb> {
    CONSTANTS
        .command_base_mb
        .process
        .iter()
        .map(|(k, &v)| (k.as_str(), MemoryMb(v)))
        .collect()
}

/// Per-command base memory (MB) — free-threaded (thread workers, shared models).
pub fn command_base_mb_threaded() -> HashMap<&'static str, MemoryMb> {
    CONSTANTS
        .command_base_mb
        .threaded
        .iter()
        .map(|(k, &v)| (k.as_str(), MemoryMb(v)))
        .collect()
}

/// Fallback per-worker memory budget (MB) when a command is not listed.
pub fn default_base_mb() -> MemoryMb {
    MemoryMb(CONSTANTS.memory.default_base_mb)
}

/// Conservative per-command execution reservation (MB) used for job-level host
/// memory planning.
///
/// Selects the `process` (conservative) memory table on GIL-enabled Python and
/// the `threaded` (shared-model) table when `PYTHON_GIL=0` is set. The process
/// table accounts for each concurrent Stanza worker holding a full private copy
/// of the model; the threaded table accounts for workers sharing one model via
/// OS threads on free-threaded Python 3.14t.
///
/// The loading-overhead factor applies to both tables — requests carry transient
/// tensor buffers even when the base model weights are shared.
pub fn command_execution_budget_mb(command: &str) -> MemoryMb {
    let table = if is_free_threaded_runtime() {
        &CONSTANTS.command_base_mb.threaded
    } else {
        &CONSTANTS.command_base_mb.process
    };
    let base = table
        .get(command)
        .copied()
        .unwrap_or(CONSTANTS.memory.default_base_mb);
    MemoryMb((base as f64 * CONSTANTS.memory.loading_overhead) as u64)
}

/// Additional memory budget (MB) allocated per file queued to a worker.
pub fn mb_per_file_mb() -> MemoryMb {
    MemoryMb(CONSTANTS.memory.mb_per_file_mb)
}

/// Multiplier applied to the static memory budget to account for transient
/// allocation spikes during model loading.
pub fn loading_overhead() -> f64 {
    CONSTANTS.memory.loading_overhead
}

/// Conservative cross-process startup reservation (MB) for one GPU worker.
pub fn gpu_worker_startup_mb() -> MemoryMb {
    MemoryMb(CONSTANTS.worker_startup_mb.gpu)
}

/// Conservative cross-process startup reservation (MB) for one Stanza worker.
pub fn stanza_worker_startup_mb() -> MemoryMb {
    MemoryMb(CONSTANTS.worker_startup_mb.stanza)
}

/// Conservative cross-process startup reservation (MB) for one IO worker.
pub fn io_worker_startup_mb() -> MemoryMb {
    MemoryMb(CONSTANTS.worker_startup_mb.io)
}

// ---------------------------------------------------------------------------
// MemoryTier — adaptive memory budgets based on total system RAM
// ---------------------------------------------------------------------------

/// RAM-tier classification for adaptive memory budgets.
///
/// Detected once at server startup from total system RAM. All memory guard
/// parameters (startup reservations, host headroom, max workers) are derived
/// from the tier rather than from fixed constants. This allows batchalign3
/// to run on 16 GB laptops through 256 GB servers without manual tuning.
///
/// The Large and Fleet tiers reproduce the existing fixed constants from
/// `runtime_constants.toml` exactly, so fleet machines see zero behavior
/// change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTierKind {
    /// < 24 GB total RAM (laptops, CI runners)
    Small,
    /// 24–48 GB (workstations, Frodo)
    Medium,
    /// 48–128 GB (development servers)
    Large,
    /// > 128 GB (fleet servers like net)
    Fleet,
}

impl std::str::FromStr for MemoryTierKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "small" => Ok(Self::Small),
            "medium" => Ok(Self::Medium),
            "large" => Ok(Self::Large),
            "fleet" => Ok(Self::Fleet),
            _ => Err(format!(
                "unknown memory tier {s:?}; valid values: small, medium, large, fleet"
            )),
        }
    }
}

/// Concrete memory budget parameters for a detected tier.
///
/// Constructed via [`MemoryTier::from_total_mb`] (pure, testable) or
/// [`MemoryTier::detect`] (reads system RAM via sysinfo).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryTier {
    /// Which tier was selected.
    pub kind: MemoryTierKind,
    /// Total system RAM in MB (as detected).
    pub total_mb: u64,
    /// Host headroom reserve — the coordinator refuses reservations that
    /// would leave available RAM below this threshold.
    pub headroom_mb: MemoryMb,
    /// Startup reservation for a GPU worker (Whisper, Wave2Vec, speaker).
    pub gpu_startup_mb: MemoryMb,
    /// Startup reservation for a Stanza worker (morphosyntax, utseg, coref).
    pub stanza_startup_mb: MemoryMb,
    /// Startup reservation for an IO worker (translate, opensmile, avqi).
    pub io_startup_mb: MemoryMb,
    /// Suggested maximum concurrent workers across all profiles.
    pub max_suggested_workers: usize,
    /// Worker idle timeout in seconds. Shorter on small machines to reclaim
    /// memory faster (a Stanza worker holds ~2-3 GB while idle).
    pub idle_timeout_s: u64,
}

impl MemoryTier {
    /// Select a tier from total system RAM (in MB). Pure function — no
    /// sysinfo dependency, fully testable with arbitrary values.
    pub fn from_total_mb(total_mb: u64) -> Self {
        //                  (kind, headroom, gpu, stanza, io, max_workers, idle_timeout_s)
        let (kind, headroom, gpu, stanza, io, max_workers, idle_s) = if total_mb < 24_000 {
            (MemoryTierKind::Small, 2_000, 6_000, 3_000, 2_000, 1, 60)
        } else if total_mb < 48_000 {
            // Medium: LazyProfile mode — GPU worker starts empty, models loaded
            // on demand. Startup reservation is just process overhead (3 GB),
            // not full model weight. Max 1 worker to prevent OOM on 32 GB.
            (MemoryTierKind::Medium, 4_000, 3_000, 6_000, 3_000, 1, 300)
        } else if total_mb < 128_000 {
            // Large — matches existing TOML constants exactly
            (MemoryTierKind::Large, 8_000, 16_000, 12_000, 4_000, 4, 600)
        } else {
            // Fleet — same budgets as Large, more workers
            (MemoryTierKind::Fleet, 8_000, 16_000, 12_000, 4_000, 8, 600)
        };
        Self {
            kind,
            total_mb,
            headroom_mb: MemoryMb(headroom),
            gpu_startup_mb: MemoryMb(gpu),
            stanza_startup_mb: MemoryMb(stanza),
            io_startup_mb: MemoryMb(io),
            max_suggested_workers: max_workers,
            idle_timeout_s: idle_s,
        }
    }

    /// Detect the tier from actual system RAM via sysinfo.
    pub fn detect() -> Self {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let total_mb = sys.total_memory() / (1024 * 1024);
        Self::from_total_mb(total_mb)
    }
}

impl std::fmt::Display for MemoryTierKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Small => write!(f, "Small (<24 GB)"),
            Self::Medium => write!(f, "Medium (24-48 GB)"),
            Self::Large => write!(f, "Large (48-128 GB)"),
            Self::Fleet => write!(f, "Fleet (>128 GB)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toml_parses_successfully() {
        // Force LazyLock initialization — panics if TOML is malformed.
        let _ = cmd2task();
    }

    #[test]
    fn cmd2task_contains_core_commands() {
        let map = cmd2task();
        assert_eq!(map["align"], "fa");
        assert_eq!(map["morphotag"], "morphosyntax");
        assert_eq!(map["transcribe"], "asr");
    }

    #[test]
    fn worker_caps_are_positive() {
        assert!(max_gpu_workers() > 0);
        assert!(max_process_workers() > 0);
        assert!(max_thread_workers() > 0);
    }

    #[test]
    fn memory_constants_are_sane() {
        assert!(default_base_mb().0 > 0);
        assert!(mb_per_file_mb().0 > 0);
        assert!(loading_overhead() > 1.0);
        assert!(gpu_worker_startup_mb().0 > stanza_worker_startup_mb().0);
        assert!(stanza_worker_startup_mb().0 > io_worker_startup_mb().0);
        // Budget uses process table on GIL=1 (default in test env).
        assert!(command_execution_budget_mb("align").0 >= command_base_mb_process()["align"].0);
    }

    #[test]
    fn gpu_heavy_non_empty() {
        assert!(!gpu_heavy_commands().is_empty());
    }

    #[test]
    fn command_base_mb_has_all_commands() {
        let proc = command_base_mb_process();
        let thread = command_base_mb_threaded();
        // Both maps should have the same keys
        let mut proc_keys: Vec<_> = proc.keys().collect();
        let mut thread_keys: Vec<_> = thread.keys().collect();
        proc_keys.sort();
        thread_keys.sort();
        assert_eq!(proc_keys, thread_keys);
    }

    // ---- MemoryTier ----

    #[test]
    fn tier_16gb_laptop() {
        let tier = MemoryTier::from_total_mb(16_000);
        assert_eq!(tier.kind, MemoryTierKind::Small);
        assert_eq!(tier.headroom_mb.0, 2_000);
        assert_eq!(tier.stanza_startup_mb.0, 3_000);
        assert_eq!(tier.gpu_startup_mb.0, 6_000);
        assert_eq!(tier.io_startup_mb.0, 2_000);
        assert_eq!(tier.max_suggested_workers, 1);
    }

    #[test]
    fn tier_32gb_workstation() {
        let tier = MemoryTier::from_total_mb(32_000);
        assert_eq!(tier.kind, MemoryTierKind::Medium);
        assert_eq!(tier.headroom_mb.0, 4_000);
        assert_eq!(tier.stanza_startup_mb.0, 6_000);
        // LazyProfile: GPU startup is just process overhead (3 GB), not full model.
        assert_eq!(tier.gpu_startup_mb.0, 3_000);
        // Max 1 worker to prevent OOM on 32 GB.
        assert_eq!(tier.max_suggested_workers, 1);
    }

    #[test]
    fn tier_64gb_fleet() {
        let tier = MemoryTier::from_total_mb(64_000);
        assert_eq!(tier.kind, MemoryTierKind::Large);
        assert_eq!(tier.headroom_mb.0, 8_000);
        assert_eq!(tier.stanza_startup_mb.0, 12_000);
        assert_eq!(tier.gpu_startup_mb.0, 16_000);
        assert_eq!(tier.max_suggested_workers, 4);
    }

    #[test]
    fn tier_256gb_server() {
        let tier = MemoryTier::from_total_mb(256_000);
        assert_eq!(tier.kind, MemoryTierKind::Fleet);
        assert_eq!(tier.headroom_mb.0, 8_000);
        assert_eq!(tier.stanza_startup_mb.0, 12_000);
        assert_eq!(tier.max_suggested_workers, 8);
    }

    #[test]
    fn large_tier_matches_toml_constants() {
        let tier = MemoryTier::from_total_mb(64_000);
        assert_eq!(tier.gpu_startup_mb, gpu_worker_startup_mb());
        assert_eq!(tier.stanza_startup_mb, stanza_worker_startup_mb());
        assert_eq!(tier.io_startup_mb, io_worker_startup_mb());
    }

    #[test]
    fn small_machine_stanza_probe_passes_gate() {
        // Simulate: 16 GB total, macOS reports ~9 GB available
        let tier = MemoryTier::from_total_mb(16_000);
        let available = 9_000u64;
        let requested = tier.stanza_startup_mb.0;
        let reserve = tier.headroom_mb.0;
        // Gate formula: available - pending - requested >= reserve
        assert!(
            available.saturating_sub(requested) >= reserve,
            "Stanza probe must pass on 16 GB: {available} - {requested} = {} >= {reserve}",
            available - requested
        );
    }

    #[test]
    fn small_machine_gpu_and_stanza_concurrent_blocked() {
        // Two heavy workers at once should NOT fit on 16 GB
        let tier = MemoryTier::from_total_mb(16_000);
        let available = 9_000u64;
        let remaining_after_gpu = available.saturating_sub(tier.gpu_startup_mb.0);
        // After GPU reserved, Stanza should not fit within headroom
        assert!(
            remaining_after_gpu.saturating_sub(tier.stanza_startup_mb.0) < tier.headroom_mb.0,
            "Concurrent GPU+Stanza must NOT fit on 16 GB"
        );
    }

    #[test]
    fn tier_detect_returns_valid_tier() {
        let tier = MemoryTier::detect();
        assert!(tier.total_mb > 0);
        assert!(tier.headroom_mb.0 > 0);
        assert!(tier.gpu_startup_mb.0 > tier.stanza_startup_mb.0);
        assert!(tier.stanza_startup_mb.0 > tier.io_startup_mb.0);
    }

    #[test]
    fn tier_boundary_24gb_is_medium() {
        assert_eq!(
            MemoryTier::from_total_mb(24_000).kind,
            MemoryTierKind::Medium
        );
        assert_eq!(
            MemoryTier::from_total_mb(23_999).kind,
            MemoryTierKind::Small
        );
    }

    #[test]
    fn tier_boundary_48gb_is_large() {
        assert_eq!(
            MemoryTier::from_total_mb(48_000).kind,
            MemoryTierKind::Large
        );
        assert_eq!(
            MemoryTier::from_total_mb(47_999).kind,
            MemoryTierKind::Medium
        );
    }

    #[test]
    fn tier_boundary_128gb_is_fleet() {
        assert_eq!(
            MemoryTier::from_total_mb(128_000).kind,
            MemoryTierKind::Fleet
        );
        assert_eq!(
            MemoryTier::from_total_mb(127_999).kind,
            MemoryTierKind::Large
        );
    }

    #[test]
    fn small_tier_idle_timeout_is_short() {
        let tier = MemoryTier::from_total_mb(16_000);
        assert_eq!(tier.idle_timeout_s, 60);
    }

    #[test]
    fn large_tier_idle_timeout_unchanged() {
        let tier = MemoryTier::from_total_mb(64_000);
        assert_eq!(tier.idle_timeout_s, 600);
    }

    #[test]
    fn medium_tier_idle_timeout_is_intermediate() {
        let tier = MemoryTier::from_total_mb(32_000);
        assert_eq!(tier.idle_timeout_s, 300);
    }
}
