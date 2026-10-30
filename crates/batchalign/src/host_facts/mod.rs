//! `HostFacts`: single source of truth for "what is this host?".
//!
//! The architectural goal: every piece of code that needs to know "does
//! this host have a GPU?", "how much RAM is available?", "what OS is
//! running?" reads from one struct populated once at server startup.
//! That struct is `HostFacts`. Pure functions downstream (`recommend()`,
//! `validate()`) consume it and produce derived knobs / configuration
//! warnings without re-querying the OS.
//!
//! This module is the **type scaffolding** for that architecture
//! (Phase A1 of the host-facts migration; the four layers are described in
//! `book/src/batchalign/developer/host-facts.md`).
//! No production code consumes `HostFacts` yet; `Real::detect()` is a
//! `todo!()` placeholder. The point of landing the types first is that
//! every later phase's tests can be written against the `Mock` source
//! without waiting for live detection to be implemented.
//!
//! Layering:
//! - `os.rs`: `OperatingSystem`, `CpuArch`
//! - `gpu.rs`: `GpuPresence`, `MpsExclusionReason`
//! - `warnings.rs`: `DetectionWarning`
//! - `mod.rs` (this file), `HostFacts`, `HostFactsSource` trait,
//!   `RealHostFactsSource` and `MockHostFactsSource` impls.

pub mod effective;
pub mod gpu;
pub mod os;
pub mod recommendations;
pub mod snapshot;
pub mod test_helpers;
pub mod validation;
pub mod warnings;

use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub use effective::{ConfigOverrides, EffectiveConfig, PerProfileOverrides};
pub use gpu::{
    GpuPresence, MpsExclusionReason, NvidiaSmiOutcome, NvidiaSmiProbe, RealNvidiaSmiProbe,
    detect_gpu_presence,
};
pub use os::{CpuArch, OperatingSystem};
pub use snapshot::HostSnapshot;
// `recommend_max_workers_per_job` is intentionally NOT re-exported.
// External callers go through `EffectiveConfig::max_workers_per_job`,
// which composes the operator override with the per-command
// recommendation. The free function is `pub(super)` and used only
// within this module (by `EffectiveConfig::resolve` and tests).
pub use recommendations::{
    PerProfile, RecommendedKnobs, recommend, recommend_force_cpu, recommend_gpu_thread_pool_size,
    recommend_max_concurrent_jobs, recommend_max_total_workers, recommend_max_workers_per_key,
};
pub use validation::{ConfigError, ConfigValidation, ConfigWarning, validate};
pub use warnings::DetectionWarning;

use crate::api::MachineTime;

/// Detected facts about the host on which batchalign3 is running.
///
/// Populated once at server startup by a `HostFactsSource` and held in
/// the application state for the process lifetime. Downstream pure
/// functions (`recommend()`, `validate()`) read this struct; nothing
/// else in the runtime polls the OS for facts that live here.
///
/// The split between this struct (static-ish facts) and
/// `host_memory::read_system_memory()` (live RAM pressure poll) is
/// intentional. `ram_total_mb` and `ram_available_mb` here are
/// snapshots at startup; the live coordinator continues to poll
/// available memory for runtime gating decisions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostFacts {
    /// The host operating system family.
    pub os: OperatingSystem,
    /// The host CPU architecture.
    pub arch: CpuArch,
    /// Logical CPU count (includes SMT siblings).
    pub cpu_logical_count: u32,
    /// Physical CPU core count (excludes SMT siblings, where detectable).
    pub cpu_physical_count: u32,
    /// Total physical RAM in MB.
    pub ram_total_mb: u64,
    /// Available RAM in MB at the moment of detection.
    pub ram_available_mb: u64,
    /// What GPU is detected, and whether batchalign3 will use it.
    pub gpu: GpuPresence,
    /// Free disk MB at the cache directory location, if known. `None`
    /// means the path was not resolvable at detection time.
    pub disk_free_mb_for_cache: Option<u64>,
    /// The host's name (Tailscale hostname or `hostname` output).
    /// Display-only: used for logging and `doctor` output, not as a
    /// domain identifier.
    pub hostname: String,
    /// When detection ran. Useful for "is this snapshot still fresh?"
    /// questions later, though for now the snapshot lives for the
    /// process lifetime.
    pub detection_timestamp: MachineTime,
    /// Non-fatal probe failures encountered during detection. The
    /// presence of warnings does not invalidate the rest of the
    /// struct; consumers may surface them in operator-facing output.
    pub detection_warnings: Vec<DetectionWarning>,
}

/// A source of `HostFacts`.
///
/// The trait exists so production code (which detects from the live
/// host) and tests (which want to assert against synthetic fact shapes
/// like "Apple Silicon 64 GB" or "Linux + CUDA 256 GB") can share one
/// interface. The two impls in this module, `RealHostFactsSource` and
/// `MockHostFactsSource`: cover those two needs.
///
/// `Arc<dyn HostFactsSource + Send + Sync>` is the expected handle type
/// in `AppState` once Phase C wires the source into the runtime; this
/// trait deliberately uses `&self` so it can be held behind a shared
/// reference without locking.
pub trait HostFactsSource: Send + Sync {
    /// Produce a `HostFacts` snapshot.
    ///
    /// Production impls perform live detection and return what they see;
    /// the `Mock` impl returns the struct it was constructed from. Either
    /// way, the snapshot is owned, callers may store, share via `Arc`,
    /// or pass to pure functions without ceremony. A live source fails
    /// when the operating system will not report the host's memory.
    fn detect(&self) -> Result<HostFacts, crate::host_memory::HostUnreadable>;
}

/// Production source: detects facts from the running host.
///
/// Each field is populated from the existing scattered helpers so the
/// new consolidator is **consistent** with the live runtime's view of
/// the host, not a parallel sysinfo poll with subtle drift.
///
/// Field-by-field provenance:
/// - `ram_total_mb` / `ram_available_mb`: one
///   `host_memory::read_system_memory` reading, the reader the host-memory
///   coordinator uses.
/// - `cpu_logical_count`: `std::thread::available_parallelism`.
/// - `cpu_physical_count`: same as logical for now; refined in a later
///   phase that adds physical-vs-logical separation if we need to
///   distinguish SMT siblings.
/// - `os` / `arch`: `std::env::consts::{OS, ARCH}`.
/// - `gpu`: `GpuPresence::None` placeholder. **Phase A4 replaces this**
///   with real probing (Apple Silicon match, `nvidia-smi`).
/// - `hostname`: `"unknown"` placeholder. Refined in a later phase that
///   wires up a real hostname probe; not load-bearing for any
///   recommendation logic.
/// - `disk_free_mb_for_cache`: `None` placeholder. Refined in a later
///   phase if a recommendation rule needs disk-aware cache sizing.
/// - `detection_timestamp`: `SystemTime::now()` epoch seconds.
/// - `detection_warnings`: empty until probes can fail (Phase A4
///   onwards populates `nvidia-smi` failure modes).
///
/// Calling `detect()` is cheap, sysinfo polls are millisecond-scale
///, but it is intended to run **once** at server startup; the result
/// is cached for the process lifetime in `AppState` (wired in Phase C).
#[derive(Debug, Default)]
pub struct RealHostFactsSource;

impl HostFactsSource for RealHostFactsSource {
    fn detect(&self) -> Result<HostFacts, crate::host_memory::HostUnreadable> {
        let memory = crate::host_memory::read_system_memory()?;
        let ram_total_mb = memory.total_mb.0;
        let ram_available_mb = memory.available_mb.0;

        let cpu_logical_count = std::thread::available_parallelism()
            .map(|p| u32::try_from(p.get()).unwrap_or(u32::MAX))
            .unwrap_or(1);
        // Until physical-vs-logical separation is wired in, treat
        // physical as logical. SMT siblings on x86_64 may double-count
        // here; on Apple Silicon arm64 the values agree natively. The
        // downstream recommendation function clamps based on logical
        // count regardless, so the worst case today is a slight
        // over-estimate of headroom on hosts with SMT enabled.
        let cpu_physical_count = cpu_logical_count;

        let os = OperatingSystem::from_consts(std::env::consts::OS);
        let arch = CpuArch::from_consts(std::env::consts::ARCH);
        // GPU detection delegates to the pure `detect_gpu_presence`
        // function so the same logic is exercised by both the live
        // production path (here, with `RealNvidiaSmiProbe`) and the
        // unit tests in `gpu.rs` (with `FixedProbe` / `NeverProbe`).
        let probe = RealNvidiaSmiProbe;
        let (gpu, detection_warnings) = detect_gpu_presence(&os, &arch, &probe);

        Ok(HostFacts {
            os,
            arch,
            cpu_logical_count,
            cpu_physical_count,
            ram_total_mb,
            ram_available_mb,
            gpu,
            disk_free_mb_for_cache: None,
            // Placeholder; refined when we wire up a real hostname
            // probe. Not used by `recommend()` or `validate()`.
            hostname: "unknown".to_owned(),
            detection_timestamp: MachineTime::now(),
            // GPU detection populates warnings (nvidia-smi NotFound /
            // Failed / Unparseable). Future phases will append additional
            // probe failures (Stanza resources, Python interpreter,
            // disk-free probe) into the same vector before constructing
            // this struct.
            detection_warnings,
        })
    }
}

/// Test source: returns a pre-constructed `HostFacts`.
///
/// The point is to let `recommend()` and `validate()` table tests
/// synthesize arbitrary host shapes (small CPU-only laptop, large CUDA
/// server, Apple Silicon dev machine, future Windows host) without
/// touching real hardware. Wrap in `Arc` to share across test harnesses.
#[derive(Debug, Clone)]
pub struct MockHostFactsSource {
    facts: Arc<HostFacts>,
}

impl MockHostFactsSource {
    /// Build a mock source that always returns `facts`.
    pub fn new(facts: HostFacts) -> Self {
        Self {
            facts: Arc::new(facts),
        }
    }
}

impl HostFactsSource for MockHostFactsSource {
    fn detect(&self) -> Result<HostFacts, crate::host_memory::HostUnreadable> {
        Ok((*self.facts).clone())
    }
}

#[cfg(test)]
mod tests {
    use super::test_helpers::apple_silicon_64gb;
    use super::*;

    #[test]
    fn mock_source_round_trips_facts() {
        let original = apple_silicon_64gb();
        let source = MockHostFactsSource::new(original.clone());
        let detected = source.detect().expect("a mock source always reads");
        assert_eq!(
            detected, original,
            "MockHostFactsSource::detect() must return the facts it was \
             constructed with, byte-for-byte. This is the test contract \
             that unblocks every downstream table test (Phase B onwards)."
        );
    }

    #[test]
    fn gpu_presence_apple_silicon_is_not_functional() {
        let facts = apple_silicon_64gb();
        assert!(
            !facts.gpu.is_functional_for_batchalign(),
            "Apple Silicon MPS must surface as non-functional for batchalign3 \
             until the kernel-deadlock policy is reversed; \
             recommend() depends on this contract."
        );
    }

    #[test]
    fn gpu_presence_cuda_is_functional() {
        let cuda = GpuPresence::NvidiaCuda {
            device_count: 1,
            total_vram_mb: 24_000,
            driver_version: "555.42".into(),
        };
        assert!(cuda.is_functional_for_batchalign());
    }

    #[test]
    fn gpu_presence_none_is_not_functional() {
        assert!(!GpuPresence::None.is_functional_for_batchalign());
    }

    /// On an Apple Silicon dev/fleet host, live detection must surface
    /// `AppleMps { functional_for_batchalign: false, reason:
    /// AppleSiliconKernelDeadlock }`: the policy decision that
    /// downstream `recommend()` and `validate()` both depend on.
    ///
    /// This is the integration test for Phase A4 of the migration:
    /// confirms that the wiring through `RealHostFactsSource::detect()`
    /// → `detect_gpu_presence` → Apple Silicon short-circuit produces
    /// the correct variant on the actual host this test runs on.
    ///
    /// Skipped on non-Apple-Silicon hosts so the suite stays portable
    /// (a CUDA host would fail the assertion); the unit tests in
    /// `gpu.rs` cover the Linux + nvidia-smi paths against synthetic
    /// probes.
    #[test]
    fn real_source_gpu_on_apple_silicon_is_mps_excluded() {
        if std::env::consts::OS != "macos" || std::env::consts::ARCH != "aarch64" {
            eprintln!(
                "SKIP: this test pins Apple Silicon detection; running OS={} ARCH={}",
                std::env::consts::OS,
                std::env::consts::ARCH
            );
            return;
        }
        let facts = RealHostFactsSource
            .detect()
            .expect("the test host reports its memory");
        assert_eq!(
            facts.gpu,
            GpuPresence::AppleMps {
                functional_for_batchalign: false,
                reason_excluded: Some(MpsExclusionReason::AppleSiliconKernelDeadlock),
            }
        );
        assert!(
            !facts.gpu.is_functional_for_batchalign(),
            "Apple Silicon must surface as non-functional for batchalign3"
        );
        assert!(
            facts.detection_warnings.is_empty(),
            "Apple Silicon detection must not produce nvidia-smi warnings; \
             observed: {:?}",
            facts.detection_warnings
        );
    }
}
