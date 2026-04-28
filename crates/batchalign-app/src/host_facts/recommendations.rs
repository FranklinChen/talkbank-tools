//! Pure recommendation function: given `HostFacts`, return the
//! `RecommendedKnobs` the binary would set if no operator override
//! applied.
//!
//! This module is the single place the formulas that previously lived
//! "in operators' heads" or scattered across `host_policy::auto_*`,
//! `auto_tune::compute_job_workers`, `pool::effective_max_total_workers`,
//! and `resolve::resolved_memory_gate_mb` are codified. Each knob has
//! its own helper (`recommend_<knob>`) so it can be unit-tested in
//! isolation and operator tooling can ask "why this value?" per knob
//! (the future `batchalign3 doctor --explain <knob>` surface).
//!
//! Phase B1 adds `gpu_thread_pool_size` — the knob that motivated the
//! architecture (see
//! `talkbank/docs/postmortems/2026-04-25-whisper-hub-malayalam-queue-wait-timeout.md`).
//! Subsequent B-phases extend `RecommendedKnobs` with `force_cpu`,
//! `max_total_workers`, `max_concurrent_jobs`, `max_workers_per_job`,
//! `max_workers_per_key_by_profile`, and `memory_gate_mb`.

use super::HostFacts;
use crate::api::{MemoryMb, ReleasedCommand};

/// Bundle of recommended values produced by [`recommend`].
///
/// Each field is a "the binary would set this absent an explicit
/// override" value. The downstream `EffectiveConfig::resolve` (Phase
/// C1) takes a `RecommendedKnobs` together with a `ServerConfig`
/// (carrying `Option<u32>` overrides) and produces resolved values.
///
/// Fields are added knob-by-knob across Phase B; the struct is non-
/// exhaustive so adding fields in later PRs does not break existing
/// callers.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct RecommendedKnobs {
    /// In-flight `execute_v2` cap per shared GPU worker, mirroring the
    /// Python `ThreadPoolExecutor(max_workers=...)` capacity. The
    /// canonical reference is
    /// `book/src/developer/worker-protocol-v2.md` § "The dispatch
    /// semaphore contract".
    pub gpu_thread_pool_size: u32,

    /// Whether the binary should disable GPU/MPS/CUDA model loading
    /// and run inference on CPU only. Derived from `GpuPresence`:
    /// hosts whose GPU is not functional for batchalign3 (Apple
    /// Silicon with MPS excluded, hosts without CUDA, hosts with a
    /// failed `nvidia-smi` probe) recommend `true`.
    ///
    /// Operator overrides remain available — the Phase C `EffectiveConfig`
    /// merge will let `force_cpu = false` survive on a host whose
    /// recommendation is `true` (with a validation warning), and
    /// `force_cpu = true` survive on a CUDA host (silently — that's
    /// a legitimate "I want CPU for testing" choice).
    pub force_cpu: bool,

    /// Hard ceiling on total workers across all `(profile, lang,
    /// engine)` keys, derived from physical RAM. Subsumes
    /// `WorkerPool::effective_max_total_workers()` /
    /// `default_max_total_workers()`.
    pub max_total_workers: u32,

    /// How many distinct jobs may run concurrently on this host.
    /// Derived from `min(cpu_logical_count.clamp(1, 8),
    /// MemoryTier::from_total_mb(ram).max_suggested_workers)`. Subsumes
    /// `HostExecutionPolicy::auto_max_concurrent_jobs()`.
    pub max_concurrent_jobs: u32,

    /// Per-`(profile, lang, engine)` worker-process cap, split by
    /// worker profile. Subsumes the flat `DEFAULT_MAX_WORKERS_PER_KEY = 4`
    /// constant in `worker/pool/mod.rs`. Each profile has a different
    /// peak per-worker RAM cost (Whisper ≫ Stanza ≫ opensmile), so a
    /// flat number is wrong on both ends of the host-size spectrum:
    /// too aggressive for small hosts (OOM risk) and too conservative
    /// for large hosts (under-utilized capacity).
    pub max_workers_per_key_by_profile: PerProfile<u32>,

    /// Host-memory headroom reserve. The host-memory coordinator
    /// refuses worker reservations that would leave available RAM
    /// below this threshold. Subsumes
    /// `ServerConfig::resolved_memory_gate_mb` (tier-derived path).
    /// Equal to `MemoryTier::from_total_mb(ram).headroom_mb` —
    /// 2 GB on Small, 4 GB on Medium, 8 GB on Large/Fleet.
    pub memory_gate_mb: MemoryMb,
}

/// Values keyed by worker profile.
///
/// The three profiles map to the runtime concept of "what kind of
/// model is loaded?" The naming matches `WorkerProfile` in
/// `crate::worker`, just lowercase as struct fields.
///
/// Generic over `T` so the same shape can carry process counts
/// (`u32`), RAM budgets (`MemoryMb`), or other per-profile values
/// in future phases. Today only `PerProfile<u32>` is in use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerProfile<T> {
    /// GPU profile: ASR, FA, speaker — Whisper-class model loads,
    /// peak RAM ~12-15 GB per process.
    pub gpu: T,
    /// Stanza profile: morphotag, utseg, translate, coref —
    /// per-language Stanza pipelines, peak RAM ~6-8 GB per process.
    pub stanza: T,
    /// IO profile: opensmile, avqi — lightweight signal processing,
    /// peak RAM ~1-2 GB per process.
    pub io: T,
}

/// Compute the recommended knob values for the given host facts.
///
/// Every field of `RecommendedKnobs` is computed by its own helper so
/// the per-knob rationale stays close to the per-knob value. Adding a
/// new knob is a two-line change here plus a new helper plus tests.
pub fn recommend(facts: &HostFacts) -> RecommendedKnobs {
    RecommendedKnobs {
        gpu_thread_pool_size: recommend_gpu_thread_pool_size(facts),
        force_cpu: recommend_force_cpu(facts),
        max_total_workers: recommend_max_total_workers(facts),
        max_concurrent_jobs: recommend_max_concurrent_jobs(facts),
        max_workers_per_key_by_profile: recommend_max_workers_per_key(facts),
        memory_gate_mb: recommend_memory_gate_mb(facts),
    }
}

/// Recommend the in-flight `execute_v2` cap for a shared GPU worker.
///
/// Rule: match the Python `ThreadPoolExecutor` capacity to the
/// underlying compute device's real parallelism.
///
/// - **CUDA-capable host**: 4. PyTorch releases the GIL during native
///   CUDA calls, so four Python threads sharing one process can run
///   four inferences concurrently. Matches today's static default.
/// - **CPU-only host** (no functional GPU detected, including Apple
///   Silicon with MPS excluded for batchalign3): 1. There is no GPU
///   parallelism to exploit; multiple in-flight `execute_v2` calls
///   would contend for cores and slow each other down. The Rust-side
///   `dispatch_semaphore` reads the same value, so K = 1 means one
///   permit, which means one inference at a time on either side.
///
/// Why this diverges from today's static default of 4: the static
/// default was a CUDA-host assumption baked into a constant. On
/// Apple Silicon (the entire current fleet), 4 produces silent
/// contention without any throughput gain. The architectural fix
/// for the dispatch_semaphore (see
/// `book/src/developer/worker-protocol-v2.md` § "The dispatch
/// semaphore contract") removed the spurious-timeout failure mode;
/// matching K to the real device parallelism removes the
/// contention-without-gain failure mode.
pub fn recommend_gpu_thread_pool_size(facts: &HostFacts) -> u32 {
    if facts.gpu.is_functional_for_batchalign() {
        4
    } else {
        1
    }
}

/// Per-worker peak RAM estimate, in MB. Whisper models use 4-15 GB
/// loaded; Stanza pipelines 2-8 GB; using a 6 GB midpoint matches the
/// legacy `default_max_total_workers` formula and prevents the cap
/// from allowing more workers than physical RAM can support.
const RAM_PER_WORKER_MB: u64 = 6 * 1024;

/// Lower bound on the worker cap. Hosts smaller than ~16 GB still get
/// at least two workers so a single failed/stuck worker doesn't strand
/// the whole pool. Matches the legacy clamp.
const MIN_TOTAL_WORKERS: u32 = 2;

/// Upper bound on the worker cap. Beyond this, the per-key,
/// per-profile concurrency budgets become the binding constraint —
/// adding more total workers stops helping and starts hurting via
/// process scheduling overhead. Matches the legacy
/// `ABSOLUTE_MAX_TOTAL_WORKERS` constant in
/// `crates/batchalign-app/src/worker/pool/mod.rs`.
const MAX_TOTAL_WORKERS: u32 = 32;

/// Conservative fallback when `ram_total_mb` is zero — almost
/// certainly a sysinfo detection failure, since no host that can run
/// the binary has zero physical RAM. Matches the legacy fallback.
const TOTAL_WORKERS_FALLBACK_ON_ZERO_RAM: u32 = 4;

/// Recommend the global worker cap from physical RAM.
///
/// Formula: `ram_total_mb / 6 GB`, clamped to `[2, 32]`. Matches the
/// legacy `default_max_total_workers()` per-worker estimate and clamp
/// bounds, but **uses `ram_total_mb` instead of `available_memory`**
/// for two reasons:
///
/// 1. **Deterministic.** `ram_total_mb` does not jitter between
///    calls; `available_memory` does. The recommendation should be a
///    function of host capability, not transient pressure.
/// 2. **Correct scope.** This cap governs how many workers can ever
///    coexist on the host — a question of physical RAM, not load.
///    The runtime `worker::memory_guard` separately enforces dynamic
///    memory pressure at spawn time, so the cap does not need to
///    pre-bake a margin for currently-running processes.
///
/// On a host with 64 GB total RAM and other processes consuming
/// memory, this returns a slightly higher value (10) than the legacy
/// helper would have at the same moment (some smaller number based
/// on currently-available bytes). The runtime memory gate continues
/// to refuse spawns that would exceed real-time pressure thresholds,
/// so the higher cap does not cause over-spawn.
pub fn recommend_max_total_workers(facts: &HostFacts) -> u32 {
    if facts.ram_total_mb == 0 {
        return TOTAL_WORKERS_FALLBACK_ON_ZERO_RAM;
    }
    let computed = facts.ram_total_mb / RAM_PER_WORKER_MB;
    let computed_u32 = u32::try_from(computed).unwrap_or(MAX_TOTAL_WORKERS);
    computed_u32.clamp(MIN_TOTAL_WORKERS, MAX_TOTAL_WORKERS)
}

/// Recommend the per-job file-parallelism cap for one command on this host.
///
/// Subsumes `runner::util::auto_tune::compute_job_workers` minus the
/// per-job file-count clamp (`min(num_files)`), which stays at the
/// dispatch site because it's a per-job quantity, not a host quantity.
///
/// Formula:
/// `min(cpu_logical_count, category_cap)` where
/// `category_cap = if is_gpu_heavy { max_gpu_workers ⌒ recommended_gpu_thread_pool_size } else { max_thread_workers } ⌒ tier.max_suggested_workers`
/// (where `⌒` is `min`).
///
/// Why this is **not** in `RecommendedKnobs`: the bundle struct holds
/// host-level recommendations. This knob is per-command — different
/// commands on the same host get different values (transcribe is
/// GPU-heavy, morphotag is CPU-only). Phase C's `EffectiveConfig`
/// resolver calls this function on demand per command, with the
/// operator override (`max_workers_per_job`) taking precedence when
/// set. The dispatch path then clamps further to `min(num_files)`.
///
/// Subtle architectural point worth flagging: the legacy formula's
/// GPU branch read `config.gpu_thread_pool_size` (operator-set value).
/// This function uses `recommend_gpu_thread_pool_size(facts)` (the
/// recommendation, not the override). The difference matters when an
/// operator sets `gpu_thread_pool_size` independently of
/// `max_workers_per_job`; under the new architecture, those two knobs
/// are decoupled — overriding one no longer cascades into the other.
/// Both can be overridden independently. Documented in
/// `docs/investigations/2026-04-25-host-facts-architecture.md` §
/// Layer 2.
pub(super) fn recommend_max_workers_per_job(facts: &HostFacts, command: &ReleasedCommand) -> u32 {
    let tier = crate::runtime::MemoryTier::from_total_mb(facts.ram_total_mb);
    let by_cpu: usize = usize::try_from(facts.cpu_logical_count.max(1)).unwrap_or(usize::MAX);
    let is_gpu_heavy = crate::runtime::gpu_heavy_commands()
        .iter()
        .any(|c| c.as_str() == command.as_ref());
    let recommended_thread_pool: usize =
        usize::try_from(recommend_gpu_thread_pool_size(facts)).unwrap_or(usize::MAX);
    let category_cap = if is_gpu_heavy {
        crate::runtime::max_gpu_workers().min(recommended_thread_pool)
    } else {
        crate::runtime::max_thread_workers()
    }
    .min(tier.max_suggested_workers)
    .max(1);
    let raw = by_cpu.min(category_cap).max(1);
    u32::try_from(raw).unwrap_or(u32::MAX)
}

// Peak RAM estimates per worker process, by profile. These are the
// memory costs the recommendation function uses to compute "how many
// of this worker class fit in physical RAM?" — distinct from the
// `MemoryTier::*_startup_mb` reservation values, which describe the
// reservation strategy (eager vs lazy) at worker startup.
//
// Sources:
// - GPU: Whisper Large-v2 + Wave2Vec FA + speaker pipeline ≈ 10-15 GB
//   resident at peak; 16 GB is a conservative midpoint that matches
//   the `MemoryTier::gpu_startup_mb` for Profile-mode tiers.
// - Stanza: per-language Stanza pipeline + 8 KB chunk batches ≈
//   6-10 GB at peak for English/Spanish/Chinese; smaller for others.
//   12 GB matches `MemoryTier::stanza_startup_mb` for Profile-mode.
const PEAK_RAM_PER_GPU_WORKER_MB: u64 = 16_000;
const PEAK_RAM_PER_STANZA_WORKER_MB: u64 = 12_000;

/// Worst-case peak RAM in MB any single concurrent job might
/// consume. Used by `validate()` to detect deterministically-OOM
/// configurations (`max_concurrent_jobs * worst_case > ram_total`).
///
/// "Worst case" = the heaviest worker profile. GPU workers
/// (Whisper + Wave2Vec FA + speaker pipeline) peak at ~16 GB
/// resident; everything else (Stanza, IO) is lighter. A jobset
/// where every job is the heaviest profile is the worst-case
/// scheduling outcome; if that wouldn't fit in physical RAM, the
/// configuration is unsafe regardless of what jobs the operator
/// actually runs.
pub fn worst_case_per_job_peak_ram_mb() -> u64 {
    PEAK_RAM_PER_GPU_WORKER_MB
}

/// Recommend the host-memory headroom reserve in MB.
///
/// Subsumes the tier-derived path in
/// `ServerConfig::resolved_memory_gate_mb`. Reads
/// `MemoryTier::from_total_mb(ram_total_mb).headroom_mb`:
///
/// - **Small** (<24 GB): 2 GB.
/// - **Medium** (24-48 GB): 4 GB.
/// - **Large** (48-128 GB): 8 GB.
/// - **Fleet** (≥128 GB): 8 GB.
///
/// The host-memory coordinator refuses reservations that would leave
/// available RAM below this threshold; the value scales with host
/// size so smaller hosts reserve less absolute headroom (since they
/// have less RAM to give) but more proportional headroom (since loss
/// of even a few GB hurts more on a small host).
///
/// Operator overrides are honored at the Phase C `EffectiveConfig`
/// merge layer; the path that originally lived in
/// `resolved_memory_gate_mb` (override-honoring with detected-default
/// fallback) maps cleanly onto `EffectiveConfig::resolve` once Phase C
/// lands.
pub fn recommend_memory_gate_mb(facts: &HostFacts) -> MemoryMb {
    crate::runtime::MemoryTier::from_total_mb(facts.ram_total_mb).headroom_mb
}

/// Recommend per-profile worker-process counts for `max_workers_per_key`.
///
/// Replaces the flat `DEFAULT_MAX_WORKERS_PER_KEY = 4` with a
/// per-profile, RAM-derived cap:
///
/// - **GPU**: `(ram_total_mb / 16 GB).clamp(1, max_gpu_workers)` where
///   `max_gpu_workers = 8` from runtime constants.
/// - **Stanza**: `(ram_total_mb / 12 GB).clamp(1, max_thread_workers)` where
///   `max_thread_workers = 8`.
/// - **IO**: flat 1. opensmile/avqi are lightweight (~1-2 GB) and
///   per-key parallelism gives no meaningful throughput benefit at
///   their typical job sizes.
///
/// Divergence from today's flat-4 default is intentional and goes both
/// ways:
///
/// - On RAM-constrained hosts (16 GB), the flat 4 was unsafe — four
///   GPU workers would have wanted 60+ GB. The recommendation drops to 1.
/// - On large hosts (256 GB Fleet), the flat 4 left capacity unused.
///   The recommendation rises to 8 (the runtime hard cap).
///
/// On a typical Large-tier host (64 GB), the recommendation is 4 for
/// GPU and 5 for Stanza — close to today's flat 4, no surprises.
pub fn recommend_max_workers_per_key(facts: &HostFacts) -> PerProfile<u32> {
    let max_gpu = u32::try_from(crate::runtime::max_gpu_workers()).unwrap_or(u32::MAX);
    let max_thread = u32::try_from(crate::runtime::max_thread_workers()).unwrap_or(u32::MAX);
    let gpu = ram_divided(facts.ram_total_mb, PEAK_RAM_PER_GPU_WORKER_MB).clamp(1, max_gpu);
    let stanza =
        ram_divided(facts.ram_total_mb, PEAK_RAM_PER_STANZA_WORKER_MB).clamp(1, max_thread);
    PerProfile {
        gpu,
        stanza,
        // IO: a per-key cap of 1 is sufficient. Increasing per-key
        // gives no benefit because opensmile/avqi don't share state
        // across calls and the dispatcher already parallelizes across
        // jobs via `max_concurrent_jobs`. Documented as a deliberate
        // flat value in the migration plan.
        io: 1,
    }
}

/// Integer divide `ram_total_mb / divisor`, returning `1` when the
/// dividend is zero (sysinfo failure case) or the quotient overflows
/// `u32`. Centralized so the per-profile helpers share one safe path.
fn ram_divided(ram_total_mb: u64, divisor_mb: u64) -> u32 {
    if ram_total_mb == 0 || divisor_mb == 0 {
        return 1;
    }
    let q = ram_total_mb / divisor_mb;
    u32::try_from(q).unwrap_or(u32::MAX)
}

/// Recommend the number of concurrent job slots on this host.
///
/// Formula (subsumes `HostExecutionPolicy::auto_max_concurrent_jobs`):
/// `min(cpu_logical_count.clamp(1, 8), tier.max_suggested_workers.max(1))`
/// where `tier = MemoryTier::from_total_mb(ram_total_mb)`.
///
/// `MemoryTier::max_suggested_workers` per tier:
/// - **Small** (<24 GB): 1 — single-job hosts.
/// - **Medium** (24-48 GB): 1 — same.
/// - **Large** (48-128 GB): 4 — typical fleet workers.
/// - **Fleet** (≥128 GB): 8 — high-memory hosts (`net`, `ming`).
///
/// The `min` ensures both axes constrain the result: a 256 GB host with
/// only 4 CPU cores recommends 4, not 8.
///
/// Reuses the existing pure helper `host_policy::auto_max_concurrent_from`
/// to keep the formula in exactly one place — this function adapts the
/// `HostFacts` shape to the helper's `(usize, usize)` interface and
/// converts the result back to `u32`.
pub fn recommend_max_concurrent_jobs(facts: &HostFacts) -> u32 {
    let tier = crate::runtime::MemoryTier::from_total_mb(facts.ram_total_mb);
    let by_cpu: usize = usize::try_from(facts.cpu_logical_count.max(1)).unwrap_or(usize::MAX);
    let raw = crate::host_policy::auto_max_concurrent_from(by_cpu, tier.max_suggested_workers);
    u32::try_from(raw).unwrap_or(u32::MAX)
}

/// Recommend whether to disable GPU model loading on this host.
///
/// Derived: `force_cpu = !gpu.is_functional_for_batchalign()`. The
/// operator override flow lives at the Phase C `EffectiveConfig` merge
/// layer; this function only answers "what would the binary set if
/// no override applied?" — which is "force CPU exactly when the GPU
/// is unusable."
///
/// The tighter coupling than `gpu_thread_pool_size` (which has a
/// CUDA-vs-CPU branch) is intentional: `force_cpu` is a hard
/// device-availability fact, not a tuning choice. There is no host
/// where "GPU is functional and we recommend forcing CPU"; that
/// situation is exclusively an operator override.
pub fn recommend_force_cpu(facts: &HostFacts) -> bool {
    !facts.gpu.is_functional_for_batchalign()
}

#[cfg(test)]
mod tests {
    use super::super::{
        CpuArch, DetectionWarning, GpuPresence, MpsExclusionReason, OperatingSystem,
    };
    use super::*;
    use crate::api::UnixTimestamp;

    /// Synthesize a `HostFacts` for tests. Builder-style: callers tweak
    /// the fields they care about and accept the rest as defaults. The
    /// defaults represent a 64 GB Apple Silicon host because that's
    /// the most common shape in the current fleet; tests for other
    /// shapes overwrite `os`, `arch`, `ram_total_mb`, `gpu` as needed.
    fn facts(os: OperatingSystem, arch: CpuArch, gpu: GpuPresence, ram_gb: u64) -> HostFacts {
        HostFacts {
            os,
            arch,
            cpu_logical_count: 12,
            cpu_physical_count: 8,
            ram_total_mb: ram_gb * 1024,
            ram_available_mb: ram_gb * 1024 / 2,
            gpu,
            disk_free_mb_for_cache: Some(500_000),
            hostname: "test-host".to_owned(),
            detection_timestamp: UnixTimestamp::from(1_700_000_000.0),
            detection_warnings: Vec::<DetectionWarning>::new(),
        }
    }

    fn apple_silicon(ram_gb: u64) -> HostFacts {
        facts(
            OperatingSystem::MacOs,
            CpuArch::Arm64,
            GpuPresence::AppleMps {
                functional_for_batchalign: false,
                reason_excluded: Some(MpsExclusionReason::AppleSiliconKernelDeadlock),
            },
            ram_gb,
        )
    }

    fn linux_cuda(ram_gb: u64, device_count: u32, total_vram_mb: u64) -> HostFacts {
        facts(
            OperatingSystem::Linux,
            CpuArch::X86_64,
            GpuPresence::NvidiaCuda {
                device_count,
                total_vram_mb,
                driver_version: "555.42.06".into(),
            },
            ram_gb,
        )
    }

    fn linux_no_gpu(ram_gb: u64) -> HostFacts {
        facts(
            OperatingSystem::Linux,
            CpuArch::X86_64,
            GpuPresence::None,
            ram_gb,
        )
    }

    fn windows_no_gpu(ram_gb: u64) -> HostFacts {
        facts(
            OperatingSystem::Windows,
            CpuArch::X86_64,
            GpuPresence::None,
            ram_gb,
        )
    }

    // -------------------------------------------------------------------
    // Apple Silicon — the entire current fleet. Every fleet RAM size
    // gets its own row so a future formula change that accidentally
    // returns >1 on small or large hosts trips a test.
    // -------------------------------------------------------------------

    #[test]
    fn apple_silicon_16gb_recommends_one_thread() {
        assert_eq!(recommend_gpu_thread_pool_size(&apple_silicon(16)), 1);
    }

    #[test]
    fn apple_silicon_32gb_recommends_one_thread() {
        assert_eq!(recommend_gpu_thread_pool_size(&apple_silicon(32)), 1);
    }

    #[test]
    fn apple_silicon_64gb_recommends_one_thread() {
        assert_eq!(recommend_gpu_thread_pool_size(&apple_silicon(64)), 1);
    }

    #[test]
    fn apple_silicon_96gb_recommends_one_thread() {
        assert_eq!(recommend_gpu_thread_pool_size(&apple_silicon(96)), 1);
    }

    #[test]
    fn apple_silicon_256gb_recommends_one_thread() {
        assert_eq!(recommend_gpu_thread_pool_size(&apple_silicon(256)), 1);
    }

    // -------------------------------------------------------------------
    // CUDA — the value the legacy static default was implicitly tuned
    // for. Single-device and multi-device should both recommend 4.
    // -------------------------------------------------------------------

    #[test]
    fn linux_single_cuda_recommends_four_threads() {
        assert_eq!(
            recommend_gpu_thread_pool_size(&linux_cuda(64, 1, 24_000)),
            4
        );
    }

    #[test]
    fn linux_dual_cuda_recommends_four_threads() {
        assert_eq!(
            recommend_gpu_thread_pool_size(&linux_cuda(256, 2, 48_000)),
            4
        );
    }

    // -------------------------------------------------------------------
    // CPU-only fallthroughs — Linux without CUDA, Windows without a
    // characterized GPU. Both must conservatively recommend 1.
    // -------------------------------------------------------------------

    #[test]
    fn linux_without_gpu_recommends_one_thread() {
        assert_eq!(recommend_gpu_thread_pool_size(&linux_no_gpu(64)), 1);
    }

    #[test]
    fn windows_without_gpu_recommends_one_thread() {
        assert_eq!(recommend_gpu_thread_pool_size(&windows_no_gpu(32)), 1);
    }

    // -------------------------------------------------------------------
    // Edge cases for `GpuPresence`. The function inspects only
    // `is_functional_for_batchalign`; verifying that the wrapper
    // matches the field-level expectation guards against a future
    // refactor that breaks the contract.
    // -------------------------------------------------------------------

    #[test]
    fn cuda_with_zero_devices_recommends_one_thread() {
        // Zero device_count means the probe parsed but found nothing;
        // is_functional_for_batchalign returns false, so K = 1.
        assert_eq!(recommend_gpu_thread_pool_size(&linux_cuda(64, 0, 0)), 1);
    }

    #[test]
    fn other_gpu_marked_functional_recommends_four() {
        // A future CUDA-equivalent (e.g., AMD ROCm) can be made
        // functional by setting the flag; the recommendation function
        // honors the flag without needing to know the device kind.
        let mut other = facts(
            OperatingSystem::Linux,
            CpuArch::X86_64,
            GpuPresence::Other {
                device_kind: "rocm".into(),
                functional_for_batchalign: true,
            },
            64,
        );
        // The struct is fine as-is; just for clarity:
        let _ = &mut other;
        assert_eq!(recommend_gpu_thread_pool_size(&other), 4);
    }

    #[test]
    fn other_gpu_marked_nonfunctional_recommends_one() {
        let other = facts(
            OperatingSystem::Linux,
            CpuArch::X86_64,
            GpuPresence::Other {
                device_kind: "rocm".into(),
                functional_for_batchalign: false,
            },
            64,
        );
        assert_eq!(recommend_gpu_thread_pool_size(&other), 1);
    }

    // -------------------------------------------------------------------
    // force_cpu — derived from `GpuPresence::is_functional_for_batchalign`,
    // inverted. Apple Silicon and any non-functional GPU recommend
    // force_cpu = true; CUDA recommends force_cpu = false. Same fact
    // shapes as gpu_thread_pool_size so a future divergence in either
    // recommendation surfaces as a row mismatch.
    // -------------------------------------------------------------------

    #[test]
    fn apple_silicon_recommends_force_cpu_true() {
        for ram_gb in [16, 32, 64, 96, 256] {
            assert!(
                recommend_force_cpu(&apple_silicon(ram_gb)),
                "Apple Silicon at {ram_gb}GB must recommend force_cpu = true"
            );
        }
    }

    #[test]
    fn linux_cuda_recommends_force_cpu_false() {
        assert!(!recommend_force_cpu(&linux_cuda(64, 1, 24_000)));
        assert!(!recommend_force_cpu(&linux_cuda(256, 2, 48_000)));
    }

    #[test]
    fn linux_without_gpu_recommends_force_cpu_true() {
        assert!(recommend_force_cpu(&linux_no_gpu(64)));
    }

    #[test]
    fn windows_without_gpu_recommends_force_cpu_true() {
        assert!(recommend_force_cpu(&windows_no_gpu(32)));
    }

    #[test]
    fn cuda_with_zero_devices_recommends_force_cpu_true() {
        // is_functional_for_batchalign returns false when device_count
        // is zero, so the recommendation is force_cpu = true even
        // though the variant is technically NvidiaCuda.
        assert!(recommend_force_cpu(&linux_cuda(64, 0, 0)));
    }

    #[test]
    fn other_gpu_marked_functional_recommends_force_cpu_false() {
        let other = facts(
            OperatingSystem::Linux,
            CpuArch::X86_64,
            GpuPresence::Other {
                device_kind: "rocm".into(),
                functional_for_batchalign: true,
            },
            64,
        );
        assert!(!recommend_force_cpu(&other));
    }

    #[test]
    fn other_gpu_marked_nonfunctional_recommends_force_cpu_true() {
        let other = facts(
            OperatingSystem::Linux,
            CpuArch::X86_64,
            GpuPresence::Other {
                device_kind: "rocm".into(),
                functional_for_batchalign: false,
            },
            64,
        );
        assert!(recommend_force_cpu(&other));
    }

    // -------------------------------------------------------------------
    // Cross-knob consistency: on every fact shape, `force_cpu` and
    // `gpu_thread_pool_size > 1` must move together. Either both
    // signal "use the GPU" or both signal "stay on CPU." A future
    // formula change that breaks this invariant — e.g., a host with
    // force_cpu=true but gpu_thread_pool_size=4 — is incoherent and
    // should fail this test loudly.
    // -------------------------------------------------------------------

    #[test]
    fn force_cpu_and_gpu_thread_pool_size_agree_across_shapes() {
        let shapes: Vec<HostFacts> = vec![
            apple_silicon(64),
            apple_silicon(256),
            linux_cuda(64, 1, 24_000),
            linux_cuda(256, 2, 48_000),
            linux_no_gpu(64),
            windows_no_gpu(32),
            linux_cuda(64, 0, 0),
        ];
        for shape in shapes {
            let force = recommend_force_cpu(&shape);
            let threads = recommend_gpu_thread_pool_size(&shape);
            assert_eq!(
                force,
                threads == 1,
                "force_cpu and gpu_thread_pool_size disagree for {shape:?}: \
                 force_cpu = {force}, gpu_thread_pool_size = {threads}. \
                 The contract is force_cpu == (threads == 1)."
            );
        }
    }

    // -------------------------------------------------------------------
    // The bundling function returns a struct whose fields agree with
    // the per-knob helpers. As more knobs are added in B3..B7, this
    // test grows alongside.
    // -------------------------------------------------------------------

    // -------------------------------------------------------------------
    // max_total_workers — RAM-derived cap. One row per fleet RAM bucket
    // plus clamp boundaries plus the zero-RAM fallback.
    // -------------------------------------------------------------------

    #[test]
    fn max_total_workers_8gb_clamps_to_min() {
        // 8 / 6 = 1, clamped to MIN = 2.
        assert_eq!(recommend_max_total_workers(&apple_silicon(8)), 2);
    }

    #[test]
    fn max_total_workers_16gb_clamps_to_min() {
        // 16 / 6 = 2, exactly at MIN.
        assert_eq!(recommend_max_total_workers(&apple_silicon(16)), 2);
    }

    #[test]
    fn max_total_workers_32gb() {
        // 32 / 6 = 5; no clamp.
        assert_eq!(recommend_max_total_workers(&apple_silicon(32)), 5);
    }

    #[test]
    fn max_total_workers_64gb() {
        // 64 / 6 = 10; no clamp.
        assert_eq!(recommend_max_total_workers(&apple_silicon(64)), 10);
    }

    #[test]
    fn max_total_workers_96gb() {
        // 96 / 6 = 16; no clamp.
        assert_eq!(recommend_max_total_workers(&apple_silicon(96)), 16);
    }

    #[test]
    fn max_total_workers_192gb_at_clamp_boundary() {
        // 192 / 6 = 32, exactly at MAX.
        assert_eq!(recommend_max_total_workers(&apple_silicon(192)), 32);
    }

    #[test]
    fn max_total_workers_256gb_clamps_to_max() {
        // 256 / 6 = 42, clamped to MAX = 32.
        assert_eq!(recommend_max_total_workers(&apple_silicon(256)), 32);
    }

    #[test]
    fn max_total_workers_zero_ram_returns_fallback() {
        // Pathological: ram_total_mb = 0 indicates a sysinfo failure.
        // Conservative fallback prevents the recommendation from
        // returning the MIN clamp (which would be 2) and giving the
        // false impression of a tiny but real host.
        let mut shape = apple_silicon(8);
        shape.ram_total_mb = 0;
        assert_eq!(recommend_max_total_workers(&shape), 4);
    }

    #[test]
    fn max_total_workers_independent_of_gpu_presence() {
        // The cap is a function of RAM, not GPU, so identical RAM on
        // different GPU classes must produce the same recommendation.
        let apple = apple_silicon(64);
        let cuda = linux_cuda(64, 1, 24_000);
        let no_gpu = linux_no_gpu(64);
        let win = windows_no_gpu(64);
        let expected = recommend_max_total_workers(&apple);
        assert_eq!(recommend_max_total_workers(&cuda), expected);
        assert_eq!(recommend_max_total_workers(&no_gpu), expected);
        assert_eq!(recommend_max_total_workers(&win), expected);
    }

    #[test]
    fn max_total_workers_overflow_safe_on_huge_ram() {
        // u64::MAX / 6 GB exceeds u32::MAX; the saturating conversion
        // must not panic and the clamp must still cap at MAX.
        let mut shape = apple_silicon(64);
        shape.ram_total_mb = u64::MAX;
        assert_eq!(recommend_max_total_workers(&shape), 32);
    }

    // -------------------------------------------------------------------
    // max_concurrent_jobs — RAM-tier × CPU-count, with both axes
    // constraining the result.
    // -------------------------------------------------------------------

    /// Small tier (<24 GB) is memory-limited to 1 regardless of CPU count.
    #[test]
    fn max_concurrent_jobs_small_tier_is_one() {
        for cpu in [1, 4, 8, 16] {
            let mut shape = apple_silicon(16);
            shape.cpu_logical_count = cpu;
            assert_eq!(
                recommend_max_concurrent_jobs(&shape),
                1,
                "small-tier host with {cpu} CPUs must recommend 1 (memory-limited)"
            );
        }
    }

    /// Medium tier (24-48 GB) is also memory-limited to 1.
    #[test]
    fn max_concurrent_jobs_medium_tier_is_one() {
        for cpu in [1, 8, 16] {
            let mut shape = apple_silicon(32);
            shape.cpu_logical_count = cpu;
            assert_eq!(recommend_max_concurrent_jobs(&shape), 1);
        }
    }

    /// Large tier (48-128 GB) caps at 4; CPU-bound below that.
    #[test]
    fn max_concurrent_jobs_large_tier_memory_bound() {
        let mut shape = apple_silicon(64);
        shape.cpu_logical_count = 12;
        assert_eq!(recommend_max_concurrent_jobs(&shape), 4);
    }

    #[test]
    fn max_concurrent_jobs_large_tier_cpu_bound() {
        let mut shape = apple_silicon(64);
        shape.cpu_logical_count = 2;
        assert_eq!(
            recommend_max_concurrent_jobs(&shape),
            2,
            "Large-tier host with only 2 CPUs must be CPU-bound to 2"
        );
    }

    /// Fleet tier (≥128 GB) caps at 8 (CPU clamp also caps at 8).
    #[test]
    fn max_concurrent_jobs_fleet_tier_memory_bound() {
        let mut shape = apple_silicon(256);
        shape.cpu_logical_count = 24;
        assert_eq!(
            recommend_max_concurrent_jobs(&shape),
            8,
            "Fleet-tier host with 24 CPUs caps at min(8 cpu-clamp, 8 memory) = 8"
        );
    }

    #[test]
    fn max_concurrent_jobs_fleet_tier_cpu_bound() {
        let mut shape = apple_silicon(256);
        shape.cpu_logical_count = 4;
        assert_eq!(
            recommend_max_concurrent_jobs(&shape),
            4,
            "Fleet-tier host with only 4 CPUs is CPU-bound to 4"
        );
    }

    /// Cross-tier boundary: a host that crosses the 48 GB Large boundary
    /// goes from 1 (Medium) to 4 (Large). This pins the boundary so a
    /// future MemoryTier change does not silently shift it.
    #[test]
    fn max_concurrent_jobs_boundary_47gb_vs_48gb() {
        let mut just_below = apple_silicon(47);
        just_below.cpu_logical_count = 12;
        let mut just_at = apple_silicon(48);
        just_at.cpu_logical_count = 12;
        // 47 GB = 47 * 1024 = 48128 MB, which IS >= 48000, so this is Large.
        // Tighten the test by using 47 GB - 1 MB:
        just_below.ram_total_mb = 47_999;
        assert_eq!(recommend_max_concurrent_jobs(&just_below), 1);
        just_at.ram_total_mb = 48_000;
        assert_eq!(recommend_max_concurrent_jobs(&just_at), 4);
    }

    /// Zero CPU count is treated as 1; we never recommend 0 jobs.
    #[test]
    fn max_concurrent_jobs_zero_cpu_returns_one() {
        let mut shape = apple_silicon(64);
        shape.cpu_logical_count = 0;
        assert_eq!(recommend_max_concurrent_jobs(&shape), 1);
    }

    /// Independent of GPU presence: same RAM + same CPU = same value
    /// regardless of GPU class. Pins the architectural rule that this
    /// knob is RAM/CPU-derived only.
    #[test]
    fn max_concurrent_jobs_independent_of_gpu_presence() {
        let make = |gpu: GpuPresence| {
            let mut shape = facts(OperatingSystem::Linux, CpuArch::X86_64, gpu, 64);
            shape.cpu_logical_count = 12;
            shape
        };
        let with_cuda = make(GpuPresence::NvidiaCuda {
            device_count: 1,
            total_vram_mb: 24_000,
            driver_version: "555.42".into(),
        });
        let without = make(GpuPresence::None);
        assert_eq!(
            recommend_max_concurrent_jobs(&with_cuda),
            recommend_max_concurrent_jobs(&without)
        );
    }

    /// CPU clamp at 8: a host with 32 CPUs must not recommend 32.
    #[test]
    fn max_concurrent_jobs_cpu_clamps_at_eight() {
        let mut shape = apple_silicon(256);
        shape.cpu_logical_count = 32;
        assert_eq!(recommend_max_concurrent_jobs(&shape), 8);
    }

    // -------------------------------------------------------------------
    // max_workers_per_job per command — GPU-heavy vs CPU-only branches
    // across tier sizes. The category cap is the load-bearing piece.
    // -------------------------------------------------------------------

    fn cmd(name: &str) -> ReleasedCommand {
        ReleasedCommand::try_from(name).expect("test command literal must be a known command")
    }

    /// On Apple Silicon, transcribe (GPU-heavy) is capped at 1 because
    /// `recommend_gpu_thread_pool_size` returns 1 (no functional GPU)
    /// and the formula's GPU branch takes the min with that value.
    #[test]
    fn max_workers_per_job_transcribe_apple_silicon_64gb() {
        let mut shape = apple_silicon(64);
        shape.cpu_logical_count = 12;
        assert_eq!(recommend_max_workers_per_job(&shape, &cmd("transcribe")), 1);
    }

    /// On a CUDA host, transcribe gets 4 — the recommended GPU thread
    /// pool size on functional GPUs, capped by the Large-tier
    /// max_suggested_workers (also 4) and max_gpu_workers (8).
    #[test]
    fn max_workers_per_job_transcribe_linux_cuda_64gb() {
        let mut shape = linux_cuda(64, 1, 24_000);
        shape.cpu_logical_count = 12;
        assert_eq!(recommend_max_workers_per_job(&shape, &cmd("transcribe")), 4);
    }

    /// All four GPU-heavy commands behave identically — they share one
    /// classification branch.
    #[test]
    fn max_workers_per_job_all_gpu_heavy_commands_match() {
        let mut shape = linux_cuda(64, 1, 24_000);
        shape.cpu_logical_count = 12;
        let expected = recommend_max_workers_per_job(&shape, &cmd("transcribe"));
        for name in ["align", "transcribe", "transcribe_s", "benchmark"] {
            assert_eq!(
                recommend_max_workers_per_job(&shape, &cmd(name)),
                expected,
                "GPU-heavy command `{name}` must match `transcribe`"
            );
        }
    }

    /// CPU-only commands (morphotag, utseg, etc.) take the
    /// `max_thread_workers` branch instead of `max_gpu_workers ⌒
    /// recommended_thread_pool`. On a Large-tier host with 12 CPUs,
    /// the formula yields min(12, max_thread_workers ⌒ tier_cap = 8 ⌒ 4) = 4.
    #[test]
    fn max_workers_per_job_morphotag_apple_silicon_64gb() {
        let mut shape = apple_silicon(64);
        shape.cpu_logical_count = 12;
        assert_eq!(recommend_max_workers_per_job(&shape, &cmd("morphotag")), 4);
    }

    /// Small-tier hosts max_workers_per_job is 1 regardless of command
    /// (memory tier dominates).
    #[test]
    fn max_workers_per_job_small_tier_is_one_for_every_command() {
        let mut shape = apple_silicon(16);
        shape.cpu_logical_count = 12;
        for name in ["transcribe", "align", "morphotag", "utseg"] {
            assert_eq!(
                recommend_max_workers_per_job(&shape, &cmd(name)),
                1,
                "Small-tier host must cap at 1 for `{name}`"
            );
        }
    }

    /// Fleet-tier on Apple Silicon: GPU-heavy still capped at 1
    /// (no functional GPU); CPU-only capped at 8 (CPU clamp / max_thread).
    #[test]
    fn max_workers_per_job_fleet_apple_silicon() {
        let mut shape = apple_silicon(256);
        shape.cpu_logical_count = 24;
        assert_eq!(
            recommend_max_workers_per_job(&shape, &cmd("transcribe")),
            1,
            "Fleet Apple Silicon transcribe still capped at 1 (no functional GPU)"
        );
        assert_eq!(
            recommend_max_workers_per_job(&shape, &cmd("morphotag")),
            8,
            "Fleet Apple Silicon morphotag uses min(24 cpu, 8 max_thread, 8 tier_cap) = 8"
        );
    }

    /// CPU-bound case: a Large-tier host with only 2 CPUs caps at 2,
    /// not at the tier's 4.
    #[test]
    fn max_workers_per_job_cpu_bound_below_tier_cap() {
        let mut shape = linux_cuda(64, 1, 24_000);
        shape.cpu_logical_count = 2;
        assert_eq!(
            recommend_max_workers_per_job(&shape, &cmd("transcribe")),
            2,
            "2-CPU host with CUDA still caps at 2 by CPU"
        );
    }

    /// Zero CPU count is treated as 1; we never recommend 0 workers.
    #[test]
    fn max_workers_per_job_zero_cpu_returns_one() {
        let mut shape = linux_cuda(64, 1, 24_000);
        shape.cpu_logical_count = 0;
        assert_eq!(recommend_max_workers_per_job(&shape, &cmd("transcribe")), 1);
    }

    /// Architectural decoupling: on a CUDA host the GPU branch uses
    /// `recommend_gpu_thread_pool_size(facts)` rather than any
    /// operator-set value. Since `RecommendedKnobs` is the only input
    /// the recommendation can see, the cascading-override behavior of
    /// the legacy formula is gone by construction. This test verifies
    /// the value matches the recommended thread pool size for both
    /// CUDA and Apple Silicon classes.
    #[test]
    fn max_workers_per_job_uses_recommended_gpu_thread_pool_size() {
        let mut cuda = linux_cuda(64, 1, 24_000);
        cuda.cpu_logical_count = 12;
        let cuda_result = recommend_max_workers_per_job(&cuda, &cmd("transcribe"));
        assert_eq!(
            cuda_result,
            recommend_gpu_thread_pool_size(&cuda).min(4) // tier cap also 4
        );

        let mut apple = apple_silicon(64);
        apple.cpu_logical_count = 12;
        let apple_result = recommend_max_workers_per_job(&apple, &cmd("transcribe"));
        assert_eq!(
            apple_result,
            recommend_gpu_thread_pool_size(&apple).min(4) // recommended = 1, so result = 1
        );
    }

    // -------------------------------------------------------------------
    // max_workers_per_key_by_profile — per-profile RAM-derived caps.
    // Tests pin the formula at every fleet RAM size: 16/32/64/96/256.
    // -------------------------------------------------------------------

    /// 16 GB host: gpu = 16/16 = 1, stanza = 16/12 = 1, io = 1 flat.
    /// Smaller than today's flat 4 — and that's the point.
    #[test]
    fn max_workers_per_key_16gb() {
        let p = recommend_max_workers_per_key(&apple_silicon(16));
        assert_eq!(p.gpu, 1);
        assert_eq!(p.stanza, 1);
        assert_eq!(p.io, 1);
    }

    /// 32 GB host: gpu = 32/16 = 2, stanza = 32/12 = 2, io = 1.
    #[test]
    fn max_workers_per_key_32gb() {
        let p = recommend_max_workers_per_key(&apple_silicon(32));
        assert_eq!(p.gpu, 2);
        assert_eq!(p.stanza, 2);
        assert_eq!(p.io, 1);
    }

    /// 64 GB host: gpu = 64/16 = 4 (matches today's flat default),
    /// stanza = 64/12 = 5, io = 1.
    #[test]
    fn max_workers_per_key_64gb() {
        let p = recommend_max_workers_per_key(&apple_silicon(64));
        assert_eq!(p.gpu, 4);
        assert_eq!(p.stanza, 5);
        assert_eq!(p.io, 1);
    }

    /// 96 GB host: gpu = 96/16 = 6, stanza = 96/12 = 8, io = 1.
    #[test]
    fn max_workers_per_key_96gb() {
        let p = recommend_max_workers_per_key(&apple_silicon(96));
        assert_eq!(p.gpu, 6);
        assert_eq!(p.stanza, 8);
        assert_eq!(p.io, 1);
    }

    /// 256 GB Fleet host: gpu = 256/16 = 16 → clamped to 8,
    /// stanza = 256/12 = 21 → clamped to 8, io = 1.
    #[test]
    fn max_workers_per_key_256gb_clamps_at_eight() {
        let p = recommend_max_workers_per_key(&apple_silicon(256));
        assert_eq!(p.gpu, 8);
        assert_eq!(p.stanza, 8);
        assert_eq!(p.io, 1);
    }

    /// 8 GB host: gpu = 8/16 = 0 → clamped to 1 (floor), stanza = 0 → 1.
    #[test]
    fn max_workers_per_key_under_one_worker_clamps_to_one() {
        let p = recommend_max_workers_per_key(&apple_silicon(8));
        assert_eq!(p.gpu, 1);
        assert_eq!(p.stanza, 1);
        assert_eq!(p.io, 1);
    }

    /// Zero RAM (sysinfo failure): all fields fall back to 1.
    #[test]
    fn max_workers_per_key_zero_ram_returns_ones() {
        let mut shape = apple_silicon(8);
        shape.ram_total_mb = 0;
        let p = recommend_max_workers_per_key(&shape);
        assert_eq!(p.gpu, 1);
        assert_eq!(p.stanza, 1);
        assert_eq!(p.io, 1);
    }

    /// IO is always flat 1 regardless of RAM. If a future contributor
    /// changes the formula to derive IO from RAM, this test fires —
    /// the architectural decision (documented in the design doc) is
    /// that opensmile/avqi don't benefit from per-key parallelism.
    #[test]
    fn max_workers_per_key_io_is_always_one() {
        for ram_gb in [8, 16, 32, 64, 96, 192, 256, 512, 1024] {
            assert_eq!(
                recommend_max_workers_per_key(&apple_silicon(ram_gb)).io,
                1,
                "IO profile must be flat 1 even at {ram_gb} GB"
            );
        }
    }

    /// Independent of GPU presence: same RAM = same per-profile counts
    /// regardless of GPU class. Pins the architectural rule that this
    /// knob is RAM-derived only.
    #[test]
    fn max_workers_per_key_independent_of_gpu_presence() {
        let make = |gpu: GpuPresence| facts(OperatingSystem::Linux, CpuArch::X86_64, gpu, 64);
        let with_cuda = make(GpuPresence::NvidiaCuda {
            device_count: 1,
            total_vram_mb: 24_000,
            driver_version: "555.42".into(),
        });
        let without = make(GpuPresence::None);
        assert_eq!(
            recommend_max_workers_per_key(&with_cuda),
            recommend_max_workers_per_key(&without)
        );
    }

    /// u64::MAX overflow safety: ram_divided's u32 conversion must
    /// not panic; clamping enforces the upper bound.
    #[test]
    fn max_workers_per_key_overflow_safe_on_huge_ram() {
        let mut shape = apple_silicon(64);
        shape.ram_total_mb = u64::MAX;
        let p = recommend_max_workers_per_key(&shape);
        assert_eq!(p.gpu, 8);
        assert_eq!(p.stanza, 8);
        assert_eq!(p.io, 1);
    }

    // -------------------------------------------------------------------
    // memory_gate_mb — tier-derived headroom. Pure delegation to
    // MemoryTier::from_total_mb(ram).headroom_mb; tests pin the per-tier
    // values and the tier boundaries.
    // -------------------------------------------------------------------

    #[test]
    fn memory_gate_mb_small_tier_is_2gb() {
        assert_eq!(recommend_memory_gate_mb(&apple_silicon(16)).0, 2_000);
    }

    #[test]
    fn memory_gate_mb_medium_tier_is_4gb() {
        assert_eq!(recommend_memory_gate_mb(&apple_silicon(32)).0, 4_000);
    }

    #[test]
    fn memory_gate_mb_large_tier_is_8gb() {
        assert_eq!(recommend_memory_gate_mb(&apple_silicon(64)).0, 8_000);
        assert_eq!(recommend_memory_gate_mb(&apple_silicon(96)).0, 8_000);
    }

    #[test]
    fn memory_gate_mb_fleet_tier_is_8gb() {
        assert_eq!(recommend_memory_gate_mb(&apple_silicon(256)).0, 8_000);
    }

    /// Pin the tier boundaries so a future `MemoryTier::from_total_mb`
    /// adjustment (e.g., shifting Medium-vs-Large from 48 GB to a
    /// different threshold) trips a recommendation test loudly.
    #[test]
    fn memory_gate_mb_tier_boundaries() {
        // Small ↔ Medium boundary at 24 GB.
        let mut just_below = apple_silicon(16);
        just_below.ram_total_mb = 23_999;
        assert_eq!(recommend_memory_gate_mb(&just_below).0, 2_000);
        let mut at_medium = apple_silicon(16);
        at_medium.ram_total_mb = 24_000;
        assert_eq!(recommend_memory_gate_mb(&at_medium).0, 4_000);

        // Medium ↔ Large boundary at 48 GB.
        let mut just_below = apple_silicon(16);
        just_below.ram_total_mb = 47_999;
        assert_eq!(recommend_memory_gate_mb(&just_below).0, 4_000);
        let mut at_large = apple_silicon(16);
        at_large.ram_total_mb = 48_000;
        assert_eq!(recommend_memory_gate_mb(&at_large).0, 8_000);

        // Large ↔ Fleet boundary at 128 GB. headroom is the same
        // (8 GB on both sides); the boundary still matters for other
        // knobs (max_concurrent_jobs uses tier.max_suggested_workers
        // which differs across this boundary).
        let mut just_below = apple_silicon(16);
        just_below.ram_total_mb = 127_999;
        assert_eq!(recommend_memory_gate_mb(&just_below).0, 8_000);
        let mut at_fleet = apple_silicon(16);
        at_fleet.ram_total_mb = 128_000;
        assert_eq!(recommend_memory_gate_mb(&at_fleet).0, 8_000);
    }

    /// Independent of GPU presence — same RAM = same headroom.
    #[test]
    fn memory_gate_mb_independent_of_gpu_presence() {
        let make = |gpu: GpuPresence| facts(OperatingSystem::Linux, CpuArch::X86_64, gpu, 64);
        let cuda = make(GpuPresence::NvidiaCuda {
            device_count: 1,
            total_vram_mb: 24_000,
            driver_version: "555.42".into(),
        });
        let none = make(GpuPresence::None);
        assert_eq!(
            recommend_memory_gate_mb(&cuda),
            recommend_memory_gate_mb(&none)
        );
    }

    #[test]
    fn recommend_bundles_per_knob_helpers() {
        let host = apple_silicon(64);
        let bundle = recommend(&host);
        assert_eq!(
            bundle.gpu_thread_pool_size,
            recommend_gpu_thread_pool_size(&host)
        );
        assert_eq!(bundle.force_cpu, recommend_force_cpu(&host));
        assert_eq!(bundle.max_total_workers, recommend_max_total_workers(&host));
        assert_eq!(
            bundle.max_concurrent_jobs,
            recommend_max_concurrent_jobs(&host)
        );
        assert_eq!(
            bundle.max_workers_per_key_by_profile,
            recommend_max_workers_per_key(&host)
        );
        assert_eq!(bundle.memory_gate_mb, recommend_memory_gate_mb(&host));
    }
}
