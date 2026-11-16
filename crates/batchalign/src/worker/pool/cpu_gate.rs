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

/// A completed CPU observation retains the load and the count it came from.
struct CpuObservation {
    loadavg_1m: f64,
    logical_cpus: u32,
}
impl CpuObservation {
    fn read(readings: &HostReadings) -> Result<Self, sysinfo::Error> {
        Ok(Self {
            loadavg_1m: readings.load_average_1m()?,
            logical_cpus: readings.logical_cpus(),
        })
    }
    fn admit(self) -> Result<(), CpuSaturated> {
        let threshold = f64::from(self.logical_cpus);
        if self.loadavg_1m >= threshold {
            Err(CpuSaturated {
                loadavg_1m: self.loadavg_1m,
                threshold,
            })
        } else {
            Ok(())
        }
    }
}

pub(super) fn check_cpu_saturation_with_state(
    state: super::memory_gate::PoolGateState,
    readings: &HostReadings,
) -> Result<(), CpuSaturated> {
    match state {
        super::memory_gate::PoolGateState::ColdStart => Ok(()),
        super::memory_gate::PoolGateState::Warm => match CpuObservation::read(readings) {
            Ok(observation) => observation.admit(),
            Err(error) => {
                tracing::warn!(%error, "load average unreadable; CPU back-pressure not applied");
                Ok(())
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Guards: CPU pressure refuses at the actual source count and admits below
    /// it, while the declared cold and warm paths use their admitted host facts.
    /// Kind: policy
    /// Not a type: the >= saturation boundary and cold-start bypass are semantic
    /// choices. Normal admission can no longer pair a host with an arbitrary threshold.
    /// State: declared CPU observations and one declared host require no live load probe.
    /// Concurrency: none
    #[test]
    fn cpu_pressure_uses_its_observed_host_count() {
        for (loadavg_1m, refused) in [(3.5, false), (4.0, true), (4.5, true)] {
            let result = CpuObservation {
                logical_cpus: 4,
                loadavg_1m,
            }
            .admit();
            match (result, refused) {
                (Err(value), true) => {
                    assert_eq!(value.threshold, 4.0);
                    assert_eq!(value.loadavg_1m, loadavg_1m);
                }
                (Ok(()), false) => {}
                other => panic!("unexpected CPU policy {other:?}"),
            }
        }
        let host =
            crate::host_facts::HostSnapshot::for_tests(&crate::config::ServerConfig::default());
        for state in [
            super::super::memory_gate::PoolGateState::ColdStart,
            super::super::memory_gate::PoolGateState::Warm,
        ] {
            assert!(check_cpu_saturation_with_state(state, &host.readings()).is_ok());
        }
    }
}
