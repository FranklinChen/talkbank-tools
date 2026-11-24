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
//! Phase B1 adds `gpu_thread_pool_size`, the knob that motivated the
//! architecture.
//! Subsequent B-phases extend `RecommendedKnobs` with `force_cpu`,
//! `max_total_workers`, `max_concurrent_jobs`, `max_workers_per_job`,
//! and `max_workers_per_key_by_profile`.

use super::HostFacts;
use crate::api::ReleasedCommand;
use crate::types::runtime;
use batchalign_types::domain::MemoryMb;

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
    /// `book/src/batchalign/developer/worker-protocol-v2.md` § "The dispatch
    /// semaphore contract".
    pub gpu_thread_pool_size: u32,

    /// Whether the binary should disable GPU/MPS/CUDA model loading
    /// and run inference on CPU only. Derived from `GpuPresence`:
    /// hosts whose GPU is not functional for batchalign3 (Apple
    /// Silicon with MPS excluded, hosts without CUDA, hosts with a
    /// failed `nvidia-smi` probe) recommend `true`.
    ///
    /// Operator overrides remain available, the Phase C `EffectiveConfig`
    /// merge will let `force_cpu = false` survive on a host whose
    /// recommendation is `true` (with a validation warning), and
    /// `force_cpu = true` survive on a CUDA host (silently, that's
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
}

/// Values keyed by worker profile.
///
/// The three profiles map to the runtime concept of "what kind of
/// model is loaded?" The naming matches `WorkerProfile` in
/// `crate::worker`, just lowercase as struct fields.
///
/// Generic over `T` so the same shape can carry process counts
/// (`u32`), RAM budgets (`MemoryMb`), or other per-profile values
/// in future phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct PerProfile<T: Copy> {
    /// GPU profile: ASR, FA, speaker, Whisper-class model loads,
    /// peak RAM ~12-15 GB per process.
    pub gpu: T,
    /// Stanza profile: morphotag, utseg, translate, coref
    /// per-language Stanza pipelines, peak RAM ~6-8 GB per process.
    pub stanza: T,
    /// IO profile: opensmile, avqi, lightweight signal processing,
    /// peak RAM ~1-2 GB per process.
    pub io: T,
}

impl<T: Copy> PerProfile<T> {
    /// Lookup by profile. Single source of truth for the
    /// `WorkerProfile -> field` mapping; replaces inline match arms
    /// at admission-control and metrics-snapshot callsites.
    pub const fn get(&self, profile: crate::worker::WorkerProfile) -> T {
        match profile {
            crate::worker::WorkerProfile::Gpu => self.gpu,
            crate::worker::WorkerProfile::Stanza => self.stanza,
            crate::worker::WorkerProfile::Io => self.io,
        }
    }

    /// Build a `PerProfile<T>` whose three fields share the same value.
    /// Used at the `EffectiveConfig → PoolConfig` boundary when an
    /// operator override carries a single value to apply to all
    /// profiles uniformly.
    pub const fn uniform(value: T) -> Self {
        Self {
            gpu: value,
            stanza: value,
            io: value,
        }
    }

    /// Apply `f` to each field, returning a new `PerProfile<U>`.
    /// Used at the `EffectiveConfig → PoolConfig` boundary to convert
    /// `PerProfile<u32>` → `PerProfile<usize>`.
    pub fn map<U: Copy, F: Fn(T) -> U>(&self, f: F) -> PerProfile<U> {
        PerProfile {
            gpu: f(self.gpu),
            stanza: f(self.stanza),
            io: f(self.io),
        }
    }
}

/// Compute the recommended knob values for the given host facts.
///
/// Every field of `RecommendedKnobs` is computed by its own helper so
/// the per-knob rationale stays close to the per-knob value. Adding a
/// new knob is a two-line change here plus a new helper plus tests.
#[must_use]
pub fn recommend(facts: &HostFacts) -> RecommendedKnobs {
    RecommendedKnobs {
        gpu_thread_pool_size: recommend_gpu_thread_pool_size(facts),
        force_cpu: recommend_force_cpu(facts),
        max_total_workers: recommend_max_total_workers(facts),
        max_concurrent_jobs: recommend_max_concurrent_jobs(facts),
        max_workers_per_key_by_profile: recommend_max_workers_per_key(facts),
    }
}

/// Recommend the in-flight `execute_v2` cap for a shared GPU worker.
///
/// Rule: match the Python `ThreadPoolExecutor` capacity to the
/// underlying compute device's real parallelism.
///
/// - **CUDA-capable host**: 4. `PyTorch` releases the GIL during native
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
/// for the `dispatch_semaphore` (see
/// `book/src/batchalign/developer/worker-protocol-v2.md` § "The dispatch
/// semaphore contract") removed the spurious-timeout failure mode;
/// matching K to the real device parallelism removes the
/// contention-without-gain failure mode.
#[must_use]
pub const fn recommend_gpu_thread_pool_size(facts: &HostFacts) -> u32 {
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

/// Upper bound on the worker cap.
///
/// 32 corresponds to ~1.14 workers/logical-core on the largest current
/// fleet shape (28 cores). Beyond that, the CPU-bound Stanza profile
/// oversubscribes cores enough that throughput drops from contention,
/// and per-key / per-profile budgets become the binding constraint (see
/// `recommend_max_workers_per_key`).
///
/// The formula is RAM-only. On a hypothetical future host with many
/// more cores than the current Fleet-tier reference machine (e.g.
/// 64 cores / 512 GB), `ram/6GB = 85` clamped to 32 would severely
/// under-utilize CPU. When such a host arrives, evolve the formula
/// to derive from both CPU and RAM
/// (e.g. `min(ram/6GB, cpu_logical_count × oversubscription_factor)`).
///
/// Empirical evidence that this cap is currently binding under
/// multi-language morphosyntax workloads, plus the separately-filed
/// busy-loop spawn-rejection bug that surfaces it, is tracked as BUG-028.
const MAX_TOTAL_WORKERS: u32 = 32;

/// Conservative fallback when `ram_total_mb` is zero, almost
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
///    coexist on the host, a question of physical RAM, not load.
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
#[must_use]
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
/// host-level recommendations. This knob is per-command, different
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
/// are decoupled: overriding one no longer cascades into the other.
/// Both can be overridden independently (host-facts Layer 2, described in
/// `book/src/batchalign/developer/host-facts.md`).
pub(super) fn recommend_max_workers_per_job(facts: &HostFacts, command: &ReleasedCommand) -> u32 {
    let tier = crate::runtime::MemoryTier::from_total_mb(facts.ram_total_mb);
    let by_cpu: usize = usize::try_from(facts.cpu_logical_count.max(1)).unwrap_or(usize::MAX);
    let is_gpu_heavy = batchalign_types::command_spec::command_spec_for(*command).profile
        == Some(batchalign_types::worker_profile::WorkerProfile::Gpu);
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
// of this worker class fit in physical RAM?", distinct from the
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

/// Worst-case startup reservation in MB any single concurrent job
/// might claim, given the host's memory tier. Used by `validate()`
/// to detect deterministically-OOM configurations
/// (`max_concurrent_jobs * worst_case > ram_total`).
///
/// **What changed (rearch follow-up).** Earlier this function returned
/// a hardcoded 16 GB GPU peak regardless of host tier or installed
/// capabilities. That blocked startup on 16 GB laptops doing
/// morphotag-only work, exactly the laptop-class scenario the
/// rearch was meant to support. The runtime gates
/// (`memory_gate.rs` admission, `idle_eviction.rs` eviction, RSS observer
/// Mode B) are tier-aware via [`MemoryTier`]'s
/// `{gpu,stanza,io}_startup_mb` fields; this function now consults the
/// same source of truth. The boot-time gate's job is to refuse
/// configurations that cannot satisfy startup reservations on this
/// tier, NOT to model long-tail peak RSS, that's the runtime gates'
/// job, and they have observed-RSS data this layer cannot.
///
/// "Worst case" = the heaviest startup reservation across worker
/// profiles for the detected tier. On Small (< 24 GB), GPU is 6 GB,
/// Stanza 3 GB, IO 2 GB → returns 6 GB. On Large/Fleet (≥ 48 GB),
/// returns 16 GB (the eager-load GPU profile). The runtime gates
/// still refuse actual at-spawn OOM via observed RSS.
#[must_use]
pub fn worst_case_per_job_peak_ram_mb(tier: &runtime::MemoryTier) -> u64 {
    let MemoryMb(gpu) = tier.gpu_startup_mb;
    let MemoryMb(stanza) = tier.stanza_startup_mb;
    let MemoryMb(io) = tier.io_startup_mb;
    gpu.max(stanza).max(io)
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
/// - On RAM-constrained hosts (16 GB), the flat 4 was unsafe, four
///   GPU workers would have wanted 60+ GB. The recommendation drops to 1.
/// - On large hosts (256 GB Fleet), the flat 4 left capacity unused.
///   The recommendation rises to 8 (the runtime hard cap).
///
/// On a typical Large-tier host (64 GB), the recommendation is 4 for
/// GPU and 5 for Stanza, close to today's flat 4, no surprises.
#[must_use]
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
/// CPU clamped to `[1, 8]` against memory tier's
/// `max_suggested_workers`, both inside
/// `host_policy::auto_max_concurrent_from`.
/// `MemoryTier::max_suggested_workers` per tier: Small/Medium → 1,
/// Large → 4, Fleet → 8.
///
/// The previous empirical-knee cap at 4 (introduced 2026-05-08 from
/// a single-host morphotag sweep) was retired alongside the live
/// CPU + memory admission gates: those gates measure
/// "is there room for one more?" at the worker-spawn seam directly,
/// so the static cap is no longer load-bearing. On Fleet-tier hosts
/// (≥128 GB) the recommendation now climbs to 8 instead of being
/// pinned at 4; admission control moves to the live gates in
/// `worker/pool/cpu_gate.rs` and `worker/pool/memory_gate.rs`.
#[must_use]
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
/// no override applied?", which is "force CPU exactly when the GPU
/// is unusable."
///
/// The tighter coupling than `gpu_thread_pool_size` (which has a
/// CUDA-vs-CPU branch) is intentional: `force_cpu` is a hard
/// device-availability fact, not a tuning choice. There is no host
/// where "GPU is functional and we recommend forcing CPU"; that
/// situation is exclusively an operator override.
#[must_use]
pub const fn recommend_force_cpu(facts: &HostFacts) -> bool {
    !facts.gpu.is_functional_for_batchalign()
}

#[cfg(test)]
mod tests {
    use super::super::{
        CpuArch, DetectionWarning, GpuPresence, MpsExclusionReason, OperatingSystem,
    };
    use super::*;
    // Test concurrency pure-recommendation: None. Every recommendation helper is a pure function of an in-memory `HostFacts` value, so these tests share no resource and can run in any order or in parallel.

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
            detection_timestamp: crate::unix_time(1_700_000_000.0),
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

    fn cmd(name: &str) -> ReleasedCommand {
        ReleasedCommand::try_from(name).expect("test command literal must be a known command")
    }

    /// Guards: Every detected GPU shape maps to the device-parallelism cap the dispatch semaphore reads: functional GPU 4, otherwise 1.
    /// Kind: policy
    /// Not a type: `GpuPresence` is a closed enum and `is_functional_for_batchalign` is a pure query, but the choice that functional means four in-flight calls and every excluded class means one is a policy the type cannot carry; each original shape stays a row so a formula change cannot silently alter one class.
    /// State: None. Twelve in-memory host shapes cover five fleet RAM sizes, single and dual CUDA, both no-GPU OS classes, zero-device CUDA and both Other-flag values.
    /// Concurrency: [pure-recommendation]
    #[test]
    fn gpu_thread_pool_size_matches_device_parallelism() {
        let other = |functional| {
            facts(
                OperatingSystem::Linux,
                CpuArch::X86_64,
                GpuPresence::Other {
                    device_kind: "rocm".into(),
                    functional_for_batchalign: functional,
                },
                64,
            )
        };
        let cases: Vec<(&str, HostFacts, u32)> = vec![
            ("apple 16gb", apple_silicon(16), 1),
            ("apple 32gb", apple_silicon(32), 1),
            ("apple 64gb", apple_silicon(64), 1),
            ("apple 96gb", apple_silicon(96), 1),
            ("apple 256gb", apple_silicon(256), 1),
            ("linux single cuda", linux_cuda(64, 1, 24_000), 4),
            ("linux dual cuda", linux_cuda(256, 2, 48_000), 4),
            ("linux no gpu", linux_no_gpu(64), 1),
            ("windows no gpu", windows_no_gpu(32), 1),
            ("cuda zero devices", linux_cuda(64, 0, 0), 1),
            ("other functional", other(true), 4),
            ("other nonfunctional", other(false), 1),
        ];
        for (label, shape, expected) in cases {
            assert_eq!(
                recommend_gpu_thread_pool_size(&shape),
                expected,
                "{label} must recommend {expected} threads"
            );
        }
    }

    /// Guards: Every detected GPU shape maps to `force_cpu` as the inverse of device functionality.
    /// Kind: policy
    /// Not a type: `force_cpu` is derived from the same closed enum query, but that an excluded device forces CPU and a functional one does not is a deployment policy; the original shape set stays as rows so the two knobs cannot drift apart silently.
    /// State: None. The same twelve declared shapes as the thread-pool matrix, all in memory.
    /// Concurrency: [pure-recommendation]
    #[test]
    fn force_cpu_matches_device_functionality() {
        let other = |functional| {
            facts(
                OperatingSystem::Linux,
                CpuArch::X86_64,
                GpuPresence::Other {
                    device_kind: "rocm".into(),
                    functional_for_batchalign: functional,
                },
                64,
            )
        };
        let cases: Vec<(&str, HostFacts, bool)> = vec![
            ("apple 16gb", apple_silicon(16), true),
            ("apple 32gb", apple_silicon(32), true),
            ("apple 64gb", apple_silicon(64), true),
            ("apple 96gb", apple_silicon(96), true),
            ("apple 256gb", apple_silicon(256), true),
            ("linux single cuda", linux_cuda(64, 1, 24_000), false),
            ("linux dual cuda", linux_cuda(256, 2, 48_000), false),
            ("linux no gpu", linux_no_gpu(64), true),
            ("windows no gpu", windows_no_gpu(32), true),
            ("cuda zero devices", linux_cuda(64, 0, 0), true),
            ("other functional", other(true), false),
            ("other nonfunctional", other(false), true),
        ];
        for (label, shape, expected) in cases {
            assert_eq!(
                recommend_force_cpu(&shape),
                expected,
                "{label} must recommend force_cpu = {expected}"
            );
        }
    }

    /// Guards: `force_cpu` and `gpu_thread_pool_size` agree across every host shape: `force_cpu` exactly when the thread cap is 1.
    /// Kind: contract
    /// Not a type: Both knobs derive from `is_functional_for_batchalign`, but the compiler cannot prove two independently written helpers stay consistent; the closed shape set is the remaining boundary.
    /// State: None. Seven declared host shapes built in memory.
    /// Concurrency: [pure-recommendation]
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

    /// Guards: The RAM-derived global worker cap matches the legacy formula at every fleet bucket, its two clamp boundaries, the zero-RAM sysinfo fallback and u64 overflow.
    /// Kind: measurement
    /// Not a type: The cap is a u32 clamp over a u64 RAM reading; the compiler cannot pin the 6 GB divisor, the [2, 32] bounds or the distinct zero-fallback branch, so each boundary stays a row.
    /// State: None. Nine declared in-memory shapes; zero and `u64::MAX` RAM override one shape's reading.
    /// Concurrency: [pure-recommendation]
    #[test]
    fn max_total_workers_matches_ram_formula_and_bounds() {
        let ram = |mb| {
            let mut shape = apple_silicon(8);
            shape.ram_total_mb = mb;
            shape
        };
        let cases: Vec<(&str, HostFacts, u32)> = vec![
            ("8gb clamps to min", apple_silicon(8), 2),
            ("16gb at min", apple_silicon(16), 2),
            ("32gb", apple_silicon(32), 5),
            ("64gb", apple_silicon(64), 10),
            ("96gb", apple_silicon(96), 16),
            ("192gb at max", apple_silicon(192), 32),
            ("256gb clamps to max", apple_silicon(256), 32),
            ("zero ram fallback", ram(0), 4),
            ("u64 overflow clamps to max", ram(u64::MAX), 32),
        ];
        for (label, shape, expected) in cases {
            assert_eq!(
                recommend_max_total_workers(&shape),
                expected,
                "{label} must recommend {expected} total workers"
            );
        }
    }

    /// Guards: The global worker cap depends on RAM, not GPU class.
    /// Kind: policy
    /// Not a type: A private RAM-only view could hide GPU fields inside the formula, but the supported public helper takes `&HostFacts`; preserving that interface means a future contributor can still consult GPU class at the boundary, so four equal-RAM shapes remain the oracle.
    /// State: None. Four declared in-memory shapes share 64 GB RAM.
    /// Concurrency: [pure-recommendation]
    #[test]
    fn max_total_workers_independent_of_gpu_presence() {
        let apple = apple_silicon(64);
        let cuda = linux_cuda(64, 1, 24_000);
        let no_gpu = linux_no_gpu(64);
        let win = windows_no_gpu(64);
        let expected = recommend_max_total_workers(&apple);
        assert_eq!(recommend_max_total_workers(&cuda), expected);
        assert_eq!(recommend_max_total_workers(&no_gpu), expected);
        assert_eq!(recommend_max_total_workers(&win), expected);
    }

    /// Guards: Concurrent job slots follow the RAM tier ceiling and CPU count together across Small, Medium, Large and Fleet tiers, their shared boundary, zero CPU and the high-CPU clamp.
    /// Kind: measurement
    /// Not a type: `max_concurrent_jobs` is a u32 clamped by two independent axes (CPU and tier ceiling) and an off-by-one RAM boundary at 48 GB; the compiler cannot pin either the boundary or which axis binds, so each original probe remains a row.
    /// State: None. Declared in-memory shapes; CPU count and RAM bytes are overridden per row without touching the host.
    /// Concurrency: [pure-recommendation]
    #[test]
    fn max_concurrent_jobs_follows_tier_and_cpu() {
        let shaped = |ram: u64, cpu| {
            let mut shape = apple_silicon(16);
            shape.ram_total_mb = ram;
            shape.cpu_logical_count = cpu;
            shape
        };
        // 47 GB * 1024 = 48128, so the exact boundary is expressed in MB.
        let cases: Vec<(&str, HostFacts, u32)> = vec![
            ("small tier 1 cpu", shaped(16 * 1024, 1), 1),
            ("small tier 4 cpu", shaped(16 * 1024, 4), 1),
            ("small tier 8 cpu", shaped(16 * 1024, 8), 1),
            ("small tier 16 cpu", shaped(16 * 1024, 16), 1),
            ("medium tier 1 cpu", shaped(32 * 1024, 1), 1),
            ("medium tier 8 cpu", shaped(32 * 1024, 8), 1),
            ("medium tier 16 cpu", shaped(32 * 1024, 16), 1),
            ("large tier memory bound", shaped(64 * 1024, 12), 4),
            ("large tier cpu bound", shaped(64 * 1024, 2), 2),
            ("fleet tier memory bound", shaped(256 * 1024, 24), 8),
            ("fleet tier cpu bound", shaped(256 * 1024, 4), 4),
            ("just below large boundary", shaped(47_999, 12), 1),
            ("at large boundary", shaped(48_000, 12), 4),
            ("zero cpu", shaped(64 * 1024, 0), 1),
            ("high cpu fleet clamps at eight", shaped(256 * 1024, 32), 8),
        ];
        for (label, shape, expected) in cases {
            assert_eq!(
                recommend_max_concurrent_jobs(&shape),
                expected,
                "{label} must recommend {expected} concurrent jobs"
            );
        }
    }

    /// Guards: Concurrent job slots depend on RAM and CPU, not GPU class.
    /// Kind: policy
    /// Not a type: A private CPU-and-RAM view could hide GPU fields inside the formula, but the supported public helper takes `&HostFacts`; narrowing that interface is outside this review, so two equal-shape hosts remain the oracle that GPU class is ignored.
    /// State: None. Two declared in-memory shapes with identical RAM and CPU.
    /// Concurrency: [pure-recommendation]
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

    /// Guards: The per-job cap matches the CPU-vs-GPU formula at each host class: GPU-heavy on excluded GPUs stays 1, on CUDA reaches the thread-pool and tier caps, CPU-only commands use the thread cap, and small tier or zero CPU clamp to 1.
    /// Kind: measurement
    /// Not a type: `recommend_max_workers_per_job` combines command classification, functional-GPU state, CPU count and tier ceiling; no type pins which branch binds, so every original host/command pair is a row.
    /// State: None. Declared in-memory shapes; CPU count is overridden per row. No command is dispatched.
    /// Concurrency: [pure-recommendation]
    #[test]
    fn max_workers_per_job_follows_command_and_host_caps() {
        let shaped = |shape: HostFacts, cpu| {
            let mut shape = shape;
            shape.cpu_logical_count = cpu;
            shape
        };
        let cases: Vec<(&str, HostFacts, &str, u32)> = vec![
            (
                "apple transcribe",
                shaped(apple_silicon(64), 12),
                "transcribe",
                1,
            ),
            (
                "cuda transcribe",
                shaped(linux_cuda(64, 1, 24_000), 12),
                "transcribe",
                4,
            ),
            (
                "apple morphotag",
                shaped(apple_silicon(64), 12),
                "morphotag",
                4,
            ),
            (
                "small tier transcribe",
                shaped(apple_silicon(16), 12),
                "transcribe",
                1,
            ),
            (
                "small tier align",
                shaped(apple_silicon(16), 12),
                "align",
                1,
            ),
            (
                "small tier morphotag",
                shaped(apple_silicon(16), 12),
                "morphotag",
                1,
            ),
            (
                "small tier utseg",
                shaped(apple_silicon(16), 12),
                "utseg",
                1,
            ),
            (
                "fleet apple transcribe",
                shaped(apple_silicon(256), 24),
                "transcribe",
                1,
            ),
            (
                "fleet apple morphotag",
                shaped(apple_silicon(256), 24),
                "morphotag",
                8,
            ),
            (
                "cuda cpu bound",
                shaped(linux_cuda(64, 1, 24_000), 2),
                "transcribe",
                2,
            ),
            (
                "cuda zero cpu",
                shaped(linux_cuda(64, 1, 24_000), 0),
                "transcribe",
                1,
            ),
        ];
        for (label, shape, command, expected) in cases {
            assert_eq!(
                recommend_max_workers_per_job(&shape, &cmd(command)),
                expected,
                "{label} must recommend {expected} per-job workers"
            );
        }
    }

    /// Guards: Every released command whose worker profile is GPU gets the GPU-branch per-job cap (bounded by the recommended thread pool), and every other command gets the thread-worker cap, on hosts where the two branches differ.
    /// Kind: policy
    /// Not a type: `command_spec_for` maps the closed `ReleasedCommand` set to profiles at runtime, and which branch a profile takes is the policy itself; iterating `ReleasedCommand::ALL` means a newly released command is covered without editing this test.
    /// State: None. Two declared in-memory host shapes and the full released command set.
    /// Concurrency: [pure-recommendation]
    #[test]
    fn max_workers_per_job_follows_worker_profile_for_every_command() {
        use batchalign_types::command_spec::command_spec_for;
        use batchalign_types::worker_profile::WorkerProfile;

        // Apple Silicon: the GPU is excluded, so the recommended thread
        // pool is 1 and GPU-profile commands are capped at 1.
        let mut apple = apple_silicon(64);
        apple.cpu_logical_count = 12;
        // CUDA: the thread pool is 4, so GPU-profile commands get 4 while
        // thread-profile commands may use more.
        let mut cuda = linux_cuda(128, 1, 24_000);
        cuda.cpu_logical_count = 12;

        for (label, shape, gpu_cap, thread_cap) in
            [("apple silicon", &apple, 1, 4), ("cuda", &cuda, 4, 8)]
        {
            for command in ReleasedCommand::ALL {
                let is_gpu = command_spec_for(command).profile == Some(WorkerProfile::Gpu);
                let expected = if is_gpu { gpu_cap } else { thread_cap };
                assert_eq!(
                    recommend_max_workers_per_job(shape, &command),
                    expected,
                    "{label}: `{command:?}` (gpu profile: {is_gpu})"
                );
            }
        }
    }

    /// Guards: Per-profile per-key caps follow RAM and clamp to [1, 8] with IO flat at 1, across every fleet size, sub-one RAM, zero-RAM sysinfo failure and u64 overflow.
    /// Kind: measurement
    /// Not a type: `PerProfile`<u32> holds three counts whose divisors (16 GB GPU, 12 GB Stanza), clamps and flat-IO rule are policy the type cannot carry; each original RAM case stays a row and IO keeps its own wide sweep.
    /// State: None. Declared in-memory shapes; zero and `u64::MAX` RAM override one reading. No workers are spawned.
    /// Concurrency: [pure-recommendation]
    #[test]
    fn max_workers_per_key_follows_ram_by_profile() {
        let ram = |mb| {
            let mut shape = apple_silicon(8);
            shape.ram_total_mb = mb;
            shape
        };
        let cases: Vec<(&str, HostFacts, (u32, u32, u32))> = vec![
            ("8gb", apple_silicon(8), (1, 1, 1)),
            ("16gb", apple_silicon(16), (1, 1, 1)),
            ("32gb", apple_silicon(32), (2, 2, 1)),
            ("64gb", apple_silicon(64), (4, 5, 1)),
            ("96gb", apple_silicon(96), (6, 8, 1)),
            ("256gb clamps", apple_silicon(256), (8, 8, 1)),
            ("zero ram", ram(0), (1, 1, 1)),
            ("u64 overflow", ram(u64::MAX), (8, 8, 1)),
        ];
        for (label, shape, (gpu, stanza, io)) in cases {
            let p = recommend_max_workers_per_key(&shape);
            assert_eq!(
                (p.gpu, p.stanza, p.io),
                (gpu, stanza, io),
                "{label} profile caps"
            );
        }
    }

    /// Guards: The IO profile stays flat at 1 across small through terabyte hosts.
    /// Kind: policy
    /// Not a type: A unit IO type would prove flatness internally, but `PerProfile<u32>` and the public helper are supported interfaces; keeping the io field means a future RAM-derived IO formula must still fail this oracle.
    /// State: None. Nine declared in-memory shapes.
    /// Concurrency: [pure-recommendation]
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

    /// Guards: Per-profile per-key caps depend on RAM, not GPU class.
    /// Kind: policy
    /// Not a type: A private RAM-only view could hide GPU fields inside the formula, but the supported public helper takes `&HostFacts` and its per-profile shape is public; preserving both interfaces leaves the RAM-only policy as a semantic choice, so two equal-RAM hosts remain the oracle.
    /// State: None. Two declared in-memory shapes with identical RAM.
    /// Concurrency: [pure-recommendation]
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

    /// Guards: The bundled recommendation equals each per-knob helper on the same host.
    /// Kind: contract
    /// Not a type: recommend assembles five independently written helpers; a type cannot prove the bundle wires each field to its own helper, so the field-by-field comparison remains the oracle.
    /// State: None. One declared in-memory host.
    /// Concurrency: [pure-recommendation]
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
    }
}
