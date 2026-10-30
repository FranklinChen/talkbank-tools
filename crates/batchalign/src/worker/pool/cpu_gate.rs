//! Live CPU-loadavg admission gate.
//!
//! # Why this exists
//!
//! For four months admission has been governed by static estimates
//! tier-derived `max_concurrent_jobs`, recommender-output
//! `max_workers_per_key`, empirical caps on the recommender output
//! and every estimate has turned out wrong on at least one
//! command/host combination. A direct measurement on a development
//! machine put morphotag's contention knee at K=4 and align's at K=1:
//! a single static value cannot satisfy both.
//!
//! The 1-minute CPU load average answers the question the static
//! estimates were trying to approximate ("does this host have CPU
//! headroom for one more worker?") with a runtime measurement instead.
//! `getloadavg(3)` is exposed identically on Linux and macOS (verified
//! responsive on Apple Silicon, 2026-05-08); `sysinfo::System::load_average()`
//! wraps it. The gate refuses spawns when the 1-minute load is
//! already at or above the host's CPU count, the universal
//! "system is fully busy" boundary, no per-host tuning required.
//!
//! # Where this plugs in
//!
//! Called from
//! [`crate::worker::pool::WorkerPool::try_claim_spawn_slot`] before
//! the global-permit acquisition so a CPU-saturation refusal does
//! not consume (and immediately refund) a global-cap permit. This is
//! the first measurement-based admission predicate; future predicates
//! (per-job RSS observation, available-memory delta tracking) will
//! plug in alongside it.

use crate::host_memory::HostReadings;

/// Information about the load reading that triggered an admission
/// refusal. Carries both numbers so observability can report
/// "rejected at loadavg=18.4 against threshold=16" rather than just
/// a boolean. Not surfaced beyond the admission seam today; the
/// pool's `cpu_saturation_rejections_total` counter is the
/// metrics-layer surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct CpuSaturated {
    /// 1-minute load average at the time of the check.
    pub(super) loadavg_1m: f64,
    /// Threshold the load was compared against.
    pub(super) threshold: f64,
}

/// Loadavg threshold: the host's logical CPU count, full stop.
///
/// No env var, no config field, no operator override. The
/// rearch's premise is that resource policy must be mechanical
/// because there is no human reading or tuning configuration on
/// fleet hosts: "operator" is a fiction in this deployment. The
/// count comes from `readings` ([`HostReadings::logical_cpus`]), so a
/// declared host's threshold is its declared CPU count.
pub(super) fn cpu_count_threshold(readings: &HostReadings) -> f64 {
    f64::from(readings.logical_cpus())
}

/// Check whether the host is currently CPU-saturated against
/// `threshold`. Returns `Err(CpuSaturated)` when the 1-minute load
/// is at or above the threshold; otherwise `Ok(())`.
///
/// The threshold is passed in (rather than computed inside) so unit
/// tests can drive both branches deterministically without env-var
/// contention between parallel test cases.
///
/// An unreadable load average admits, logged. This gate is back-pressure,
/// not safety (see [`check_cpu_saturation_with_state`]); memory safety is
/// the memory gate's, which fails closed on an unreadable host. Refusing
/// here instead would wedge the pool for good on a host without a load
/// average, which is the case the old silent zero already admitted.
pub(super) fn check_cpu_saturation(
    readings: &HostReadings,
    threshold: f64,
) -> Result<(), CpuSaturated> {
    let loadavg_1m = match readings.load_average_1m() {
        Ok(loadavg_1m) => loadavg_1m,
        Err(error) => {
            tracing::warn!(%error, "load average unreadable; CPU back-pressure not applied");
            return Ok(());
        }
    };
    if loadavg_1m >= threshold {
        Err(CpuSaturated {
            loadavg_1m,
            threshold,
        })
    } else {
        Ok(())
    }
}

/// State-aware CPU admission check, mirroring
/// [`super::memory_gate::check_memory_saturation_with_state`].
///
/// `PoolGateState::ColdStart` admits unconditionally: the pool has
/// no workers to back-pressure against, so refusing the first spawn
/// just to relieve unrelated host load wedges the pool with no
/// recovery path. CPU pressure is back-pressure, not safety.
///
/// `PoolGateState::Warm` runs the loadavg check normally.
pub(super) fn check_cpu_saturation_with_state(
    state: super::memory_gate::PoolGateState,
    readings: &HostReadings,
    threshold: f64,
) -> Result<(), CpuSaturated> {
    if state == super::memory_gate::PoolGateState::ColdStart {
        return Ok(());
    }
    check_cpu_saturation(readings, threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real loadavg is always non-negative; threshold of 0.0 makes
    /// the gate certain to refuse on any live machine. (On a freshly
    /// booted CI host loadavg can be exactly 0.0; the comparison is
    /// `>=` so the gate still rejects at the boundary, which is what
    /// we want: refuse to admit when the load equals the ceiling.)
    #[test]
    fn rejects_when_loadavg_meets_or_exceeds_threshold() {
        let result = check_cpu_saturation(&HostReadings::Live, 0.0);
        let Err(saturated) = result else {
            panic!("expected CpuSaturated rejection at threshold=0.0");
        };
        assert!(
            saturated.loadavg_1m >= 0.0,
            "loadavg must be non-negative, got {}",
            saturated.loadavg_1m
        );
        assert_eq!(saturated.threshold, 0.0);
    }

    /// A threshold far above any realistic load means the gate must
    /// admit. The bound (1e9) is well past every credible
    /// kernel-reported load value, so this exercises the Ok branch
    /// without depending on host idleness.
    #[test]
    fn admits_when_threshold_far_above_load() {
        let result = check_cpu_saturation(&HostReadings::Live, 1.0e9);
        assert!(
            result.is_ok(),
            "expected Ok at threshold=1e9, got {result:?}"
        );
    }

    /// A declared host is idle: it admits at any positive threshold, and
    /// refuses only at a zero threshold, where load equals the ceiling.
    #[test]
    fn a_declared_host_is_idle() {
        let declared =
            crate::host_facts::HostSnapshot::for_tests(&crate::config::ServerConfig::default())
                .readings();
        assert!(check_cpu_saturation(&declared, 1.0).is_ok());
        let Err(saturated) = check_cpu_saturation(&declared, 0.0) else {
            panic!("load 0 meets a threshold of 0");
        };
        assert_eq!(saturated.loadavg_1m, 0.0);
    }

    /// The live threshold is a positive number on any machine the daemon
    /// runs on (the unreadable fallback is one CPU); a declared host's is
    /// its declared count.
    #[test]
    fn the_threshold_is_the_hosts_cpu_count() {
        assert!(cpu_count_threshold(&HostReadings::Live) >= 1.0);
        let declared =
            crate::host_facts::HostSnapshot::for_tests(&crate::config::ServerConfig::default())
                .readings();
        assert_eq!(cpu_count_threshold(&declared), 4.0);
    }
}
