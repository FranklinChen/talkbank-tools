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

/// A completed startup projection tied to the source host and selected engines.
/// Consumers cannot independently replace or rederive this retained result.
#[derive(Debug)]
pub(super) struct StartupReservation(crate::api::MemoryMb);
impl StartupReservation {
    pub(super) fn for_worker(
        profile: crate::worker::WorkerProfile,
        engines: &crate::types::engines::EngineOverrides,
        host: &crate::host_facts::RuntimeHost,
    ) -> Self {
        Self(engine_aware_startup_reservation_mb(
            profile,
            engines,
            &host.tier(),
        ))
    }
    pub(super) fn mb(&self) -> u64 {
        self.0.0
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
fn engine_aware_startup_reservation_mb(
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
mod tests;
