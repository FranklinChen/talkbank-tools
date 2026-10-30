//! Runtime constants: loaded from `runtime_constants.toml` at compile time.
//!
//! Command-to-task mapping, memory budgets, and command classification.
//! The TOML file is the single source of truth shared with Python.

use std::collections::HashMap;
use std::sync::LazyLock;

use serde::Deserialize;

use crate::api::{MemoryMb, ReleasedCommand};
use batchalign_types::command_spec::PythonRuntime;

pub use batchalign_types::memory::{
    MemoryTier, MemoryTierKind, estimate_per_worker_peak_mb,
    estimate_per_worker_peak_mb_with_profile,
};

/// Raw TOML content, embedded at compile time.
const TOML_SRC: &str = include_str!("../../../../batchalign/runtime_constants.toml");

/// Parsed TOML structure.
// The gpu_heavy_commands field is retained for TOML structural validation.
// The Rust accessor was deleted in Phase β Task 4; classification now
// comes from COMMAND_SPECS in batchalign-types.
#[derive(Deserialize)]
#[allow(dead_code)]
struct RuntimeConstants {
    cmd2task: HashMap<String, String>,
    worker_caps: WorkerCaps,
    memory: MemoryConstants,
    gpu_heavy_commands: GpuHeavy,
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
    mb_per_file_mb: u64,
}

// The gpu_heavy_commands TOML section is deserialized to validate the TOML at startup,
// but the Rust accessor is gone, classification now comes from COMMAND_SPECS in
// batchalign-types. Python still reads this section directly at import time.
// Task 5 codegen will project it from COMMAND_SPECS so the TOML stays in sync.
#[derive(Deserialize)]
#[allow(dead_code)]
struct GpuHeavy {
    commands: Vec<String>,
}

// Note: the [process_commands] section is intentionally not deserialized
// here. Python reads it from runtime_constants.toml directly; Rust reads the
// same classification from `CommandSpec::gil_process_need`, which generates
// it. Rust selects process vs. threaded budgets via `python_runtime()`: see
// `command_execution_budget_mb()`.

#[derive(Deserialize)]
struct KnownEngineKeys {
    keys: Vec<String>,
}

// Compile-time-constant embedded TOML: structurally validated by the test suite.
#[allow(clippy::expect_used)]
static CONSTANTS: LazyLock<RuntimeConstants> =
    LazyLock::new(|| toml::from_str(TOML_SRC).expect("runtime_constants.toml must be valid TOML"));

/// Which Python runtime the workers run on, detected once at process
/// startup from the `PYTHON_GIL` environment variable: `PYTHON_GIL=0` means
/// CPython's GIL is disabled (Python 3.14t+). This is the same variable that
/// controls Python worker startup and is inherited by the Rust server binary
/// when launched from a free-threaded context. It selects each command's
/// memory budget (`CommandSpec::execution_budget_mb`).
static PYTHON_RUNTIME: LazyLock<PythonRuntime> =
    LazyLock::new(|| match std::env::var("PYTHON_GIL").as_deref() {
        Ok("0") => PythonRuntime::FreeThreaded,
        Ok(_) | Err(_) => PythonRuntime::Gil,
    });

/// The Python runtime the workers run on.
pub fn python_runtime() -> PythonRuntime {
    *PYTHON_RUNTIME
}

/// Command name -> pipeline task string (e.g. "align" -> "fa").
pub fn cmd2task() -> HashMap<&'static str, &'static str> {
    CONSTANTS
        .cmd2task
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect()
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

/// Per-command execution reservation (MB) used for job-level memory planning
/// in this process's Python runtime: the command's own spec
/// (`CommandSpec::execution_budget_mb`), which the host planner reads too.
/// Every released command has a spec, so there is no fallback figure.
pub fn command_execution_budget_mb(command: ReleasedCommand) -> MemoryMb {
    batchalign_types::command_spec::command_spec_for(command).execution_budget_mb(python_runtime())
}

/// Additional memory budget (MB) allocated per file queued to a worker.
pub fn mb_per_file_mb() -> MemoryMb {
    MemoryMb(CONSTANTS.memory.mb_per_file_mb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toml_parses_successfully() {
        // Force LazyLock initialization: panics if TOML is malformed.
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
        assert!(mb_per_file_mb().0 > 0);
        // Per spec Principle 1, MemoryTier is the canonical source for
        // per-tier per-profile startup envelopes. Pin the Large-tier ordering.
        let tier = MemoryTier::from_total_mb(64_000);
        assert!(tier.gpu_startup_mb.0 > tier.stanza_startup_mb.0);
        assert!(tier.stanza_startup_mb.0 > tier.io_startup_mb.0);
        // The execution budget is the spec's base with its loading spike.
        let align = batchalign_types::command_spec::command_spec_for(ReleasedCommand::Align);
        assert!(
            command_execution_budget_mb(ReleasedCommand::Align).0
                >= align.base_mb_for_runtime(python_runtime()).0
        );
    }

    #[test]
    fn gpu_heavy_commands_exist_in_registry() {
        use batchalign_types::command_spec::COMMAND_SPECS;
        use batchalign_types::worker_profile::WorkerProfile;
        // Registry must have at least one GPU command (align, transcribe, benchmark, transcribe_s).
        assert!(
            COMMAND_SPECS
                .iter()
                .any(|s| s.profile == Some(WorkerProfile::Gpu))
        );
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
        let tier = MemoryTier::from_total_mb(
            crate::host_memory::read_system_memory()
                .expect("the test host reports its memory")
                .total_mb
                .0,
        );
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

    // -------------------------------------------------------------
    // tier_aware_command_execution_budget_mb: Layer 3 fix.
    //
    // The fleet-conservative `command_execution_budget_mb` returns
    // 12000 MB for morphotag (8000 base × 1.5 loading overhead).
    // On a 16 GB laptop that's 75% of physical RAM, too big a
    // single-job envelope, the host_memory coordinator refuses
    // worker spawn before any actual work happens. The tier-aware
    // variant clamps the worst-case to the tier's per-profile
    // startup reservation so small hosts get realistic envelopes.
    // -------------------------------------------------------------

    #[test]
    fn tier_aware_budget_clamps_morphotag_to_stanza_startup_on_small() {
        use batchalign_types::api::ReleasedCommand;
        use batchalign_types::command_spec::command_spec_for;
        use batchalign_types::memory::estimate_per_worker_peak_mb_with_profile;
        let tier = MemoryTier::from_total_mb(16_000);
        assert_eq!(tier.kind, MemoryTierKind::Small);
        let spec = command_spec_for(ReleasedCommand::Morphotag);
        let worst_case = batchalign_types::api::MemoryMb(
            (spec.base_mb_for_runtime(PythonRuntime::Gil).0 as f64 * spec.loading_overhead.0)
                as u64,
        );
        let budget = estimate_per_worker_peak_mb_with_profile(
            worst_case,
            spec.profile.expect("inference command profile"),
            &tier,
        );
        assert!(
            budget.0 <= tier.stanza_startup_mb.0,
            "Small-tier morphotag budget must clamp to stanza_startup_mb \
             ({} MB), got {} MB",
            tier.stanza_startup_mb.0,
            budget.0
        );
        assert!(
            budget.0 < 12_000,
            "Small-tier morphotag budget must be smaller than the \
             fleet worst-case (12000 MB), got {} MB",
            budget.0
        );
    }

    #[test]
    fn tier_aware_budget_keeps_fleet_morphotag_at_full_worst_case() {
        use batchalign_types::api::ReleasedCommand;
        use batchalign_types::command_spec::command_spec_for;
        use batchalign_types::memory::estimate_per_worker_peak_mb_with_profile;
        let tier = MemoryTier::from_total_mb(256_000);
        assert_eq!(tier.kind, MemoryTierKind::Fleet);
        let spec = command_spec_for(ReleasedCommand::Morphotag);
        let worst_case = batchalign_types::api::MemoryMb(
            (spec.base_mb_for_runtime(PythonRuntime::Gil).0 as f64 * spec.loading_overhead.0)
                as u64,
        );
        let budget = estimate_per_worker_peak_mb_with_profile(
            worst_case,
            spec.profile.expect("inference command profile"),
            &tier,
        );
        assert_eq!(
            budget,
            command_execution_budget_mb(ReleasedCommand::Morphotag),
            "Fleet-tier morphotag must equal the fleet worst-case; the \
             clamp must not weaken fleet behavior"
        );
    }

    #[test]
    fn tier_aware_budget_uses_gpu_envelope_for_align_on_small() {
        use batchalign_types::api::ReleasedCommand;
        use batchalign_types::command_spec::command_spec_for;
        use batchalign_types::memory::estimate_per_worker_peak_mb_with_profile;
        let tier = MemoryTier::from_total_mb(16_000);
        let spec = command_spec_for(ReleasedCommand::Align);
        let worst_case = batchalign_types::api::MemoryMb(
            (spec.base_mb_for_runtime(PythonRuntime::Gil).0 as f64 * spec.loading_overhead.0)
                as u64,
        );
        let budget = estimate_per_worker_peak_mb_with_profile(
            worst_case,
            spec.profile.expect("inference command profile"),
            &tier,
        );
        assert!(
            budget.0 <= tier.gpu_startup_mb.0,
            "Small-tier align budget must clamp to gpu_startup_mb \
             ({} MB), got {} MB",
            tier.gpu_startup_mb.0,
            budget.0
        );
    }

    /// End-to-end check: with the tier-aware budget, a 16 GB host
    /// admits 1-worker morphotag through `plan_job_reservation`,
    /// matching the realistic CI scenario (13902 MB available after
    /// kernel/agent overhead, 2048 MB reserve floor).
    #[test]
    fn tier_aware_budget_lets_16gb_host_admit_1_worker_morphotag() {
        use batchalign_types::api::ReleasedCommand;
        use batchalign_types::command_spec::command_spec_for;
        use batchalign_types::memory::estimate_per_worker_peak_mb_with_profile;
        let tier = MemoryTier::from_total_mb(16_000);
        let spec = command_spec_for(ReleasedCommand::Morphotag);
        let worst_case = batchalign_types::api::MemoryMb(
            (spec.base_mb_for_runtime(PythonRuntime::Gil).0 as f64 * spec.loading_overhead.0)
                as u64,
        );
        let budget = estimate_per_worker_peak_mb_with_profile(
            worst_case,
            spec.profile.expect("inference command profile"),
            &tier,
        );
        // plan_job_reservation lives in host_memory.rs; reproduce its
        // math here so the test is self-contained without
        // cross-module wiring. The bug we're fixing is upstream of
        // plan_job_reservation: the BUDGET it receives.
        let available_mb = 13_902u64;
        let reserve_mb = 2_048u64;
        let pending_reserved_mb = 0u64;
        let projected_after = available_mb
            .saturating_sub(pending_reserved_mb)
            .saturating_sub(budget.0);
        assert!(
            projected_after >= reserve_mb,
            "Tier-aware budget must let a 16 GB host admit 1-worker \
             morphotag: budget={budget_mb} MB, available={available_mb} \
             MB, projected_after={projected_after} MB, reserve={reserve_mb} MB",
            budget_mb = budget.0,
        );
    }

    /// Architectural invariant for spec Principle 1:
    /// `MemoryTier::{gpu,stanza,io}_startup_mb` is the SOLE canonical source of
    /// per-tier per-profile worker startup memory. No parallel constants,
    /// functions, or struct fields holding the same concept may exist anywhere
    /// else in the production code of the `batchalign` crate.
    ///
    /// This test is identifier-shape-anchored (not numeric-value-anchored)
    /// because the canonical Large-tier values (16000, 12000, 4000) appear
    /// in legitimate unrelated contexts (audio sample rates, byte buffers).
    ///
    /// Allowlisted definition sites:
    ///   - `crates/batchalign-types/src/memory.rs`: canonical `MemoryTier` struct
    ///     (moved from `crates/batchalign/src/types/runtime.rs` in Phase β Task 2).
    ///   - `crates/batchalign/src/types/config/server.rs`: operator-override
    ///     fields on `RuntimeOverridesConfig` (flow INTO `MemoryTier`, not parallel to it).
    ///
    /// The scan is scoped to `batchalign/src` so the canonical definition in
    /// `batchalign-types` is never in scope; only illicit parallel copies would appear.
    ///
    /// On regression the test fails with a message pointing at the spec.
    #[test]
    fn memory_tier_is_sole_source_for_per_profile_envelopes() {
        // (a) Pin the canonical Large-tier values. Any change here must be
        //     accompanied by a deliberate update of the spec.
        let tier = MemoryTier::from_total_mb(64_000);
        assert_eq!(tier.gpu_startup_mb.0, 16_000, "Large-tier GPU envelope");
        assert_eq!(
            tier.stanza_startup_mb.0, 12_000,
            "Large-tier Stanza envelope"
        );
        assert_eq!(tier.io_startup_mb.0, 4_000, "Large-tier IO envelope");

        // (b) Scan Rust sources in process. A source-architecture assertion
        // must not depend on an external search executable, and an unreadable
        // file must fail the scan rather than look like "no matches".
        let src_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let allowed = src_root.join("types/config/server.rs");
        let definitions = regex::Regex::new(
            r"^\s*(pub\s+)?(fn|const|static)\s+(gpu|stanza|io)_startup_mb\b|^\s*pub\s+(gpu|stanza|io)_startup_mb\s*:",
        )
        .expect("definition-shape regex must compile");
        let mut matches = Vec::new();
        for entry in walkdir::WalkDir::new(&src_root) {
            let entry = entry.expect("source scan must read every directory");
            let path = entry.path();
            if !entry.file_type().is_file()
                || path.extension().is_none_or(|extension| extension != "rs")
                || path == allowed
            {
                continue;
            }
            let source = std::fs::read_to_string(path)
                .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
            for (line, text) in source.lines().enumerate() {
                if definitions.is_match(text) {
                    matches.push(format!("{}:{}:{text}", path.display(), line + 1));
                }
            }
        }
        assert!(
            matches.is_empty(),
            "MemoryTier is the sole source for startup envelopes; parallel definitions:\n{}",
            matches.join("\n"),
        );

        // (c) Check the actual embedded TOML table, not comments containing a
        // retired key's spelling. No workspace search or second file read.
        let constants: toml::Value = toml::from_str(TOML_SRC).expect("valid runtime constants");
        assert!(
            constants.get("worker_startup_mb").is_none(),
            "the retired [worker_startup_mb] table must not replace MemoryTier",
        );
    }
}
