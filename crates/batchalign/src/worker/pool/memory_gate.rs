//! Live available-memory admission gate.
//!
//! # The architectural principle
//!
//! Admission gates are **back-pressure, not safety**. Safety against
//! actual at-spawn OOM is one layer down: `memory_guard` (per-worker
//! RSS observation + kill) and the OS OOM killer. The admission gate
//! exists to slow down N+1 spawns when existing workers are
//! pressured, not to refuse the FIRST spawn on an empty pool, which
//! has no contention to back-pressure against.
//!
//! Earlier versions of this module conflated the two: a single rule
//! `available - reservation > floor` decided both cases, and on
//! memory-tight hosts (laptops, CI runners) it refused the cold-start
//! spawn, leaving the pool dead-on-arrival with no path to recover
//! (eviction has nothing to evict). That violated the rearch's stated
//! goal of "accurate, dynamic capability checking that doesn't block
//! weaker machines from doing work." See [`PoolGateState`] for the
//! cold-start vs warm distinction the gate now honors.
//!
//! # The two layers
//!
//! 1. **Cold-start bypass**: `PoolGateState::ColdStart` admits
//!    unconditionally. No back-pressure has anything to push
//!    against on an empty pool; the right place to surface
//!    host-too-small failures is `memory_guard` after spawn.
//! 2. **Warm-pool projection**: `PoolGateState::Warm` runs the
//!    `available - reservation > floor` check. Refusing here
//!    relieves real contention rather than wedging the pool.
//!
//! # The floor
//!
//! Tier-scaled, not fixed. See
//! [`host_min_free_mb_threshold_for_tier`]. The previous fixed
//! 2048 MB was right for Medium-tier workstations and wrong on every
//! other tier: too tight on laptops (50% of a 4 GB machine), too
//! loose on fleet (0.78% of a 256 GB machine).
//!
//! No env var, no `PoolConfig` field, no operator override. The
//! daemon is deployed by `pyinfra` from a Claude-generated
//! `server.yaml` that no human reads or edits.
//!
//! # Where this plugs in
//!
//! Called from
//! [`crate::worker::pool::WorkerPool::try_claim_spawn_slot`] as
//! Layer 0.5: after the [`super::cpu_gate`] check, before the
//! global-permit acquisition. Memory and CPU rejections do not
//! consume permits; they refuse the spawn outright and let the
//! caller retry once the host frees up.

use crate::host_memory::HostReadings;

/// Whether the worker pool currently has any workers for the
/// (profile, lang, engine) class the admission check is for.
///
/// The principled architectural distinction underlying the runtime
/// admission gate's purpose: gates implement *back-pressure*, not
/// *safety*. Back-pressure has nothing to push against on an empty
/// pool, so a `ColdStart` admission must always be allowed
/// refusing it leaves the pool dead-on-arrival on memory-tight
/// hosts (laptops, CI runners) with no path to ever spawn a
/// worker. Safety against actual at-spawn OOM lives one layer
/// down in `memory_guard` (per-worker RSS observation + kill) and
/// the OS OOM killer.
///
/// `Warm` admissions are the back-pressure case: at least one
/// worker exists for this class, so refusing the next spawn under
/// memory pressure relieves contention rather than wedging the
/// pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PoolGateState {
    /// The pool has no workers (and no concurrent spawn attempts)
    /// for the requested (profile, lang, engine) class. The
    /// memory gate must admit, there's nothing to back-pressure
    /// against, and refusing here means the pool never starts.
    ColdStart,
    /// At least one worker exists or is being spawned for the
    /// class. The memory gate applies normally; refuse N+1 under
    /// pressure.
    Warm,
}

/// Information about the memory reading that triggered an admission
/// refusal. Carries the live reading, the reservation the projection
/// was made against, and the floor that was breached, so
/// observability can report
/// "available=10240 MB minus new-worker reservation=12000 MB would
/// leave -1760 MB; floor=2048 MB" rather than just a boolean. Not
/// surfaced beyond the admission seam today; the pool's
/// `memory_constrained_rejections_total` counter is the metrics-layer
/// surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MemoryConstrained {
    /// Available memory in MB at the time of the check.
    pub(super) available_mb: u64,
    /// The new worker's projected reservation in MB. Subtracted
    /// from `available_mb` to compute the post-spawn projection.
    /// `0` means "no reservation projection" (i.e., the gate is
    /// only enforcing the floor, equivalent to the original
    /// pre-Mode-A behavior).
    pub(super) reservation_mb: u64,
    /// Floor the projected post-spawn memory was compared against.
    pub(super) threshold_mb: u64,
}

/// Why the memory gate refused a spawn.
#[derive(Debug)]
pub(super) enum GateRefusal {
    /// The projection fell below the floor: wait for memory to free.
    Constrained(MemoryConstrained),
    /// The host would not report its memory: nothing to project from, and
    /// waiting will not change that.
    Unreadable(crate::host_memory::HostUnreadable),
}

/// Tier-scaled minimum-free floor in MB. The 2048-MB hardcode the
/// rearch shipped was wrong on every host: too tight on laptops
/// (50% of a 4 GB machine), too loose on fleet (0.78% of a 256 GB
/// machine). The principled shape: floor reflects the host's safe
/// headroom budget, just like the per-profile reservation already
/// does via [`MemoryTier::stanza_startup_mb`] and friends.
///
/// Numbers chosen so a Small laptop (16 GB) sees ~6% headroom and
/// a Fleet (256 GB) sees ~1.5%:
///
/// | Tier   | Range      | Floor   |
/// |--------|------------|---------|
/// | Small  | < 24 GB    | 1024 MB |
/// | Medium | 24-48 GB   | 2048 MB |
/// | Large  | 48-128 GB  | 4096 MB |
/// | Fleet  | > 128 GB   | 4096 MB |
///
/// Returned by a function (not inlined as a constant) so unit
/// tests can audit "what number is the production gate using"
/// without duplicating tier-arithmetic at every call site.
pub(super) fn host_min_free_mb_threshold_for_tier(tier: &crate::types::runtime::MemoryTier) -> u64 {
    use crate::types::runtime::MemoryTierKind;
    match tier.kind {
        MemoryTierKind::Small => 1024,
        MemoryTierKind::Medium => 2048,
        MemoryTierKind::Large | MemoryTierKind::Fleet => 4096,
    }
}

/// Compute the admission-gate startup reservation for one worker,
/// taking the engine choice into account.
///
/// Returns `MAX(profile_baseline, per_engine_resident_size)` across
/// every populated engine field (asr / fa / translate). The
/// profile-level baseline
/// ([`batchalign_types::worker_profile::WorkerProfile::startup_reservation_mb_for_tier`])
/// is correct when every worker in a profile has roughly the same
/// resident footprint, but breaks down for any worker that loads its
/// own large model in-process:
///
/// * **Translate (IO profile):** SeamlessM4T-medium ~2.4 GB,
///   NLLB-200-distilled-1.3B ~5 GB, both well over
///   `tier.io_startup_mb` (2 GB Small/Medium, 4 GB Large/Fleet).
/// * **ASR (GPU profile):** Whisper-large-v3 ~3.5 GB, over the
///   Medium-tier `tier.gpu_startup_mb` (3 GB).
/// * **FA (GPU profile):** Whisper-large-v2 FA ~3.5 GB, same shape.
///
/// Reserving only the profile baseline lets the admission gate keep
/// approving heavy-model workers under memory pressure until the OS
/// OOM killer fires. Per
/// [`crate::types::engines::TranslateEngineName::resident_memory_mb`]
/// (and the analogous methods on `AsrEngineName` / `FaEngineName`),
/// each engine declares its footprint and this helper inflates the
/// reservation accordingly. A single worker only ever loads ONE
/// engine of each kind, so taking the MAX across populated fields
/// gives the correct worst-case projection for that worker.
///
/// Production callers pass `group.engine_selection.overrides()` (the
/// typed [`crate::types::engines::EngineOverrides`] the pool stores on
/// the group.
pub(super) fn engine_aware_startup_reservation_mb(
    profile: batchalign_types::worker_profile::WorkerProfile,
    engine_overrides: &crate::types::engines::EngineOverrides,
    tier: &crate::types::runtime::MemoryTier,
) -> batchalign_types::api::MemoryMb {
    let base = profile.startup_reservation_mb_for_tier(tier);
    let engine_floor = [
        engine_overrides
            .translate
            .as_ref()
            .map(|e| e.resident_memory_mb()),
        engine_overrides
            .asr
            .as_ref()
            .map(|e| e.resident_memory_mb()),
        engine_overrides.fa.as_ref().map(|e| e.resident_memory_mb()),
    ]
    .into_iter()
    .flatten()
    .max()
    .unwrap_or(0);
    batchalign_types::api::MemoryMb(base.0.max(engine_floor))
}

/// State-aware admission check.
///
/// Implements the principled rearch follow-up: the memory gate is
/// back-pressure, not safety. On `PoolGateState::ColdStart` (no
/// existing worker for the class), always admit, the pool has
/// nothing to push back against, and refusing here leaves the
/// pool dead-on-arrival on memory-tight hosts. Safety against
/// actual at-spawn OOM is `memory_guard`'s job (per-worker RSS
/// observation + kill) and the OS OOM killer's; this function
/// must not double up.
///
/// On `PoolGateState::Warm`, applies the original projection:
/// `Err(MemoryConstrained)` when `available_mb < reservation_mb +
/// threshold_mb`; otherwise `Ok(())`. Production callers pass the
/// new worker's `startup_reservation_mb_for_tier` so the
/// projection accounts for the worker's anticipated load.
/// Reservation is a static per-profile estimate, a known
/// imperfection that the rearch's Mode B (RSS observation of
/// running same-profile peers) refines once at least one worker
/// exists. Cold-start by definition has no peers to observe.
pub(super) fn check_memory_saturation_with_state(
    state: PoolGateState,
    readings: &HostReadings,
    threshold_mb: u64,
    reservation_mb: u64,
) -> Result<(), GateRefusal> {
    if state == PoolGateState::ColdStart {
        return Ok(());
    }
    let available_mb = readings
        .memory()
        .map_err(GateRefusal::Unreadable)?
        .available_mb
        .0;
    let projected_after_spawn_mb = available_mb.saturating_sub(reservation_mb);
    if projected_after_spawn_mb <= threshold_mb {
        Err(GateRefusal::Constrained(MemoryConstrained {
            available_mb,
            reservation_mb,
            threshold_mb,
        }))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    /// The declared test host's readings: no gate test depends on the
    /// machine running it.
    fn declared() -> HostReadings {
        crate::host_facts::HostSnapshot::for_tests(&crate::config::ServerConfig::default())
            .readings()
    }

    /// The Medium-tier floor must equal exactly 2048 MB, the
    /// rearch's original number, preserved for the workstation case
    /// (Frodo / dev hosts). If anyone ever changes the Medium tier's
    /// floor, the change should be deliberate enough to update this
    /// test alongside it.
    #[test]
    fn medium_tier_floor_is_2048_mb() {
        let medium = crate::types::runtime::MemoryTier::from_total_mb(32_000);
        assert_eq!(
            batchalign_types::memory::MIN_FREE_MEMORY_MB,
            batchalign_types::domain::MemoryMb(2048)
        );
        assert_eq!(host_min_free_mb_threshold_for_tier(&medium), 2048);
    }

    // ---- engine_aware_startup_reservation_mb ----

    fn overrides(json: &str) -> crate::types::engines::EngineOverrides {
        serde_json::from_str(json).expect("test override JSON is valid")
    }

    /// A translate worker carrying no engine override falls back to
    /// the IO profile's static reservation. The new helper must not
    /// regress that baseline.
    #[test]
    fn engine_aware_reservation_falls_back_to_profile_baseline() {
        let medium = crate::types::runtime::MemoryTier::from_total_mb(32_000);
        let baseline = batchalign_types::worker_profile::WorkerProfile::Io
            .startup_reservation_mb_for_tier(&medium)
            .0;
        let reservation = engine_aware_startup_reservation_mb(
            batchalign_types::worker_profile::WorkerProfile::Io,
            &overrides("{}"),
            &medium,
        )
        .0;
        assert_eq!(reservation, baseline);
    }

    /// A translate worker carrying ``{"translate":"google"}`` does
    /// not inflate the IO baseline, googletrans is a thin HTTP
    /// client with no local model, so the IO reservation is already
    /// enough. Pinned to detect accidental over-reservation that
    /// would refuse Google workers on memory-tight hosts.
    #[test]
    fn engine_aware_reservation_for_google_matches_io_baseline() {
        let medium = crate::types::runtime::MemoryTier::from_total_mb(32_000);
        let baseline = batchalign_types::worker_profile::WorkerProfile::Io
            .startup_reservation_mb_for_tier(&medium)
            .0;
        let reservation = engine_aware_startup_reservation_mb(
            batchalign_types::worker_profile::WorkerProfile::Io,
            &overrides(r#"{"translate":"google"}"#),
            &medium,
        )
        .0;
        assert_eq!(reservation, baseline);
    }

    /// A translate worker carrying ``{"translate":"nllb"}`` must
    /// reserve at least NLLB-200-distilled-1.3B's resident footprint
    /// (~5500 MB) on every tier, well above the IO baseline of 2 GB
    /// (Small/Medium) or 4 GB (Large/Fleet). One case per
    /// ``MemoryTierKind`` so a per-tier regression names itself.
    #[rstest]
    fn engine_aware_reservation_for_nllb_meets_resident_footprint(
        #[values(16_000_u64, 32_000, 64_000, 256_000)] total_mb: u64,
    ) {
        let nllb_rss = crate::types::engines::TranslateEngineName::Nllb.resident_memory_mb();
        let tier = crate::types::runtime::MemoryTier::from_total_mb(total_mb);
        let reservation = engine_aware_startup_reservation_mb(
            batchalign_types::worker_profile::WorkerProfile::Io,
            &overrides(r#"{"translate":"nllb"}"#),
            &tier,
        )
        .0;
        assert!(
            reservation >= nllb_rss,
            "tier total={total_mb} MB: reservation {reservation} < NLLB \
             resident footprint {nllb_rss}"
        );
    }

    /// A translate worker carrying ``{"translate":"seamless"}`` must
    /// reserve at least Seamless's resident footprint (~2900 MB). On
    /// Small/Medium tiers the IO baseline of 2 GB is too low; on
    /// Large/Fleet the 4 GB IO baseline already covers Seamless and
    /// the helper returns the larger value (baseline).
    #[rstest]
    fn engine_aware_reservation_for_seamless_meets_resident_footprint(
        #[values(16_000_u64, 32_000, 64_000, 256_000)] total_mb: u64,
    ) {
        let seamless_rss =
            crate::types::engines::TranslateEngineName::Seamless.resident_memory_mb();
        let tier = crate::types::runtime::MemoryTier::from_total_mb(total_mb);
        let reservation = engine_aware_startup_reservation_mb(
            batchalign_types::worker_profile::WorkerProfile::Io,
            &overrides(r#"{"translate":"seamless"}"#),
            &tier,
        )
        .0;
        assert!(
            reservation >= seamless_rss,
            "tier total={total_mb} MB: reservation {reservation} < Seamless \
             resident footprint {seamless_rss}"
        );
    }

    /// A worker carrying ``{"asr":"whisper"}`` must reserve at least
    /// Whisper-large-v3's resident footprint (~3500 MB) on every tier.
    /// Regression test against under-reservation on Medium-tier GPU
    /// hosts (3 GB baseline vs. ~3.5 GB resident).
    #[rstest]
    fn engine_aware_reservation_for_whisper_asr_meets_resident_footprint(
        #[values(16_000_u64, 32_000, 64_000, 256_000)] total_mb: u64,
    ) {
        let whisper_rss = crate::types::engines::AsrEngineName::Whisper.resident_memory_mb();
        let tier = crate::types::runtime::MemoryTier::from_total_mb(total_mb);
        let reservation = engine_aware_startup_reservation_mb(
            batchalign_types::worker_profile::WorkerProfile::Gpu,
            &overrides(r#"{"asr":"whisper"}"#),
            &tier,
        )
        .0;
        assert!(
            reservation >= whisper_rss,
            "tier total={total_mb} MB: reservation {reservation} < Whisper \
             resident footprint {whisper_rss}"
        );
    }

    /// A worker carrying ``{"fa":"whisper_fa"}`` must reserve at least
    /// Whisper-large-v2 FA's resident footprint (~3500 MB) on every
    /// tier.
    #[rstest]
    fn engine_aware_reservation_for_whisper_fa_meets_resident_footprint(
        #[values(16_000_u64, 32_000, 64_000, 256_000)] total_mb: u64,
    ) {
        let whisper_fa_rss = crate::types::engines::FaEngineName::Whisper.resident_memory_mb();
        let tier = crate::types::runtime::MemoryTier::from_total_mb(total_mb);
        let reservation = engine_aware_startup_reservation_mb(
            batchalign_types::worker_profile::WorkerProfile::Gpu,
            &overrides(r#"{"fa":"whisper_fa"}"#),
            &tier,
        )
        .0;
        assert!(
            reservation >= whisper_fa_rss,
            "tier total={total_mb} MB: reservation {reservation} < Whisper \
             FA resident footprint {whisper_fa_rss}"
        );
    }

    /// When more than one engine field is populated, the reservation
    /// must be at least the MAX of the per-engine footprints. A
    /// single worker only loads one engine of each kind, so MAX is
    /// the correct worst-case projection, never SUM.
    #[test]
    fn engine_aware_reservation_takes_max_across_engines() {
        let nllb_rss = crate::types::engines::TranslateEngineName::Nllb.resident_memory_mb();
        let whisper_rss = crate::types::engines::AsrEngineName::Whisper.resident_memory_mb();
        let expected_floor = nllb_rss.max(whisper_rss);
        let tier = crate::types::runtime::MemoryTier::from_total_mb(32_000);
        let reservation = engine_aware_startup_reservation_mb(
            batchalign_types::worker_profile::WorkerProfile::Gpu,
            &overrides(r#"{"asr":"whisper","translate":"nllb"}"#),
            &tier,
        )
        .0;
        assert!(
            reservation >= expected_floor,
            "reservation {reservation} < MAX(NLLB {nllb_rss}, Whisper {whisper_rss}) \
             = {expected_floor}"
        );
        // Sanity check: must NOT be summed.
        assert!(
            reservation < nllb_rss + whisper_rss,
            "reservation {reservation} looks like NLLB + Whisper rather than MAX"
        );
    }

    /// A worker carrying a cloud ASR engine (Tencent, here) on a
    /// Medium-tier GPU host gets the GPU baseline (3 GB), not the
    /// cloud client's 200 MB. The MAX clamps to the profile baseline
    /// rather than letting a thin HTTP client under-reserve.
    #[test]
    fn engine_aware_reservation_for_cloud_asr_matches_gpu_baseline() {
        let medium = crate::types::runtime::MemoryTier::from_total_mb(32_000);
        let baseline = batchalign_types::worker_profile::WorkerProfile::Gpu
            .startup_reservation_mb_for_tier(&medium)
            .0;
        let reservation = engine_aware_startup_reservation_mb(
            batchalign_types::worker_profile::WorkerProfile::Gpu,
            &overrides(r#"{"asr":"tencent"}"#),
            &medium,
        )
        .0;
        assert_eq!(reservation, baseline);
    }

    // ---------------------------------------------------------------
    // Layer 2 (cold-start bootstrap rule + tier-scaled floor).
    //
    // The rearch's stated goal is "accurate, dynamic capability
    // checking that doesn't block weaker machines from doing work."
    // The runtime admission gate landed in the rearch refused the
    // FIRST worker on memory-constrained hosts because the
    // empty-pool case went through the same projection as the
    // populated-pool case. That left the pool dead-on-arrival on
    // hosts where `available - reservation < floor`, with no path to
    // ever spawn a worker (eviction has nothing to evict).
    //
    // The principled architecture: admission gates are
    // back-pressure, not safety. Safety lives in memory_guard
    // (per-worker RSS observation + kill) and the OS OOM killer.
    // The admission gate's job is to slow down N+1 spawns when
    // existing workers are pressured, not to refuse the FIRST
    // spawn on an empty pool, which has no contention to
    // back-pressure against.
    //
    // Plus the floor: 2048 MB is wrong on both extremes. On a 4 GB
    // host it's 50% of total RAM (silly). On a 256 GB fleet host
    // it's 0.78% (silly). The floor needs to scale with tier just
    // like the per-profile reservation does.
    // ---------------------------------------------------------------

    use crate::types::runtime::{MemoryTier, MemoryTierKind};

    /// Cold-start: the pool has no workers (and no concurrent spawn
    /// attempts) for this class. Even with a u64::MAX threshold that
    /// would otherwise force refusal, the admission gate must allow
    /// the spawn: back-pressure semantics don't apply when there's
    /// nothing to push back against.
    #[test]
    fn cold_start_pool_admits_first_worker_under_memory_pressure() {
        let result =
            check_memory_saturation_with_state(PoolGateState::ColdStart, &declared(), u64::MAX, 0);
        assert!(
            result.is_ok(),
            "cold-start admission must bypass the floor: a pool with no \
             workers cannot be 'saturated' and refusing the first spawn \
             leaves the pool dead-on-arrival. got {result:?}"
        );
    }

    /// Cold-start with a non-trivial reservation also admits. Even
    /// when the static reservation projection is hostile, the cold
    /// path bypasses: memory_guard handles actual at-spawn OOM.
    #[test]
    fn cold_start_pool_admits_under_reservation_pressure_too() {
        let result =
            check_memory_saturation_with_state(PoolGateState::ColdStart, &declared(), 0, u64::MAX);
        assert!(
            result.is_ok(),
            "cold-start must admit regardless of reservation projection. \
             got {result:?}"
        );
    }

    /// Warm pool: at least one worker exists for this class. The
    /// admission gate applies normally, refuse the N+1 spawn under
    /// memory pressure. Preserves the rearch's runtime-admission
    /// behavior for the legitimate saturation case.
    #[test]
    fn warm_pool_still_refuses_under_memory_pressure() {
        let result =
            check_memory_saturation_with_state(PoolGateState::Warm, &declared(), u64::MAX, 0);
        assert!(
            result.is_err(),
            "warm-pool admission must still refuse N+1 under memory \
             pressure: back-pressure is the gate's whole job once the \
             pool has workers to push back against. got {result:?}"
        );
    }

    /// Warm pool with reservation pressure: same back-pressure
    /// semantics. Preserves existing rejection behavior for callers
    /// that pass a real reservation.
    #[test]
    fn warm_pool_refuses_when_reservation_alone_exceeds_available() {
        let result =
            check_memory_saturation_with_state(PoolGateState::Warm, &declared(), 0, u64::MAX);
        assert!(
            result.is_err(),
            "warm-pool admission must refuse when reservation exceeds \
             available. got {result:?}"
        );
    }

    /// On a declared host the projection is exact: 14,000 MB available,
    /// the Small floor 1,024 MB, so a 12,975 MB estimate admits (1,025 MB
    /// left) and 12,976 MB refuses (1,024 MB left is at the floor). A live
    /// reading could only test the extremes.
    #[test]
    fn declared_host_admits_up_to_the_floor_and_refuses_at_it() {
        let declared = declared();
        assert!(
            check_memory_saturation_with_state(PoolGateState::Warm, &declared, 1_024, 12_975)
                .is_ok()
        );
        let Err(GateRefusal::Constrained(constrained)) =
            check_memory_saturation_with_state(PoolGateState::Warm, &declared, 1_024, 12_976)
        else {
            panic!("expected a refusal at the floor");
        };
        assert_eq!(constrained.available_mb, 14_000);
    }

    /// Warm pool with no pressure admits, the gate isn't there to
    /// refuse work that fits, only to hold back work that doesn't.
    #[test]
    fn warm_pool_admits_when_floor_and_reservation_are_zero() {
        let result = check_memory_saturation_with_state(PoolGateState::Warm, &declared(), 0, 0);
        assert!(
            result.is_ok(),
            "warm-pool admission with no pressure must admit. got {result:?}"
        );
    }

    /// Floor scales with memory tier. The 2048-MB hardcode was
    /// wrong on every host: too tight on laptops (50% of a 4 GB
    /// machine), too loose on fleet (0.78% of a 256 GB machine).
    /// The principled shape: tier-aware floors that reflect the
    /// host's safe-headroom budget. Numbers chosen so a Small
    /// laptop (16 GB) sees ~6% headroom and a Fleet (256 GB) sees
    /// ~1.5%.
    #[test]
    fn host_min_free_mb_threshold_scales_with_tier() {
        let small = MemoryTier::from_total_mb(16_000);
        let medium = MemoryTier::from_total_mb(32_000);
        let large = MemoryTier::from_total_mb(64_000);
        let fleet = MemoryTier::from_total_mb(256_000);

        assert_eq!(small.kind, MemoryTierKind::Small);
        assert_eq!(medium.kind, MemoryTierKind::Medium);
        assert_eq!(large.kind, MemoryTierKind::Large);
        assert_eq!(fleet.kind, MemoryTierKind::Fleet);

        assert_eq!(host_min_free_mb_threshold_for_tier(&small), 1024);
        assert_eq!(host_min_free_mb_threshold_for_tier(&medium), 2048);
        assert_eq!(host_min_free_mb_threshold_for_tier(&large), 4096);
        assert_eq!(host_min_free_mb_threshold_for_tier(&fleet), 4096);
    }
}
