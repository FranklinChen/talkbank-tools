//! Physical request feasibility, separate from congestion and execution leases.

use super::task_load::ModelLeaseCoverage;
use crate::api::MemoryMb;
use crate::host_memory::HostMemorySnapshot;
use crate::types::worker_v2::{FaModelBytesV2, FaWorkspaceBytesV2};
use crate::worker::memory_guard::MemoryGuardError;

/// A physical-capacity observation deliberately cannot express current free
/// memory. Congestion belongs to the coordinator's subsequent lease queue.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FaPhysicalCapacity {
    total: MemoryMb,
    reserve: MemoryMb,
}

impl FaPhysicalCapacity {
    pub(crate) fn from_snapshot(snapshot: &HostMemorySnapshot) -> Self {
        Self {
            total: snapshot.total_mb,
            reserve: snapshot.reserve_mb,
        }
    }
}

/// Loaded-model allocation evidence in the coordinator's units. Conversion,
/// the stated allocator margin and worker baseline have one owner.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FaProjectedDemand {
    model: MemoryMb,
    workspace_with_margin: MemoryMb,
    minimum: MemoryMb,
}

impl FaProjectedDemand {
    pub(crate) fn from_projection(workspace: FaWorkspaceBytesV2, model: FaModelBytesV2) -> Self {
        let workspace = workspace.get().div_ceil(1024 * 1024);
        let model = model.get().div_ceil(1024 * 1024);
        let margin =
            (workspace * crate::worker::residency::ACTIVATION_MARGIN_PERCENT).div_ceil(100);
        let workspace_with_margin = workspace + margin;
        Self {
            model: MemoryMb(model),
            workspace_with_margin: MemoryMb(workspace_with_margin),
            minimum: MemoryMb(
                model
                    + workspace_with_margin
                    + crate::worker::residency::WORKER_BASELINE.reserved().0,
            ),
        }
    }

    pub(crate) fn assess(
        self,
        host: FaPhysicalCapacity,
    ) -> Result<HostFeasibleFaDemand, FaCapacityRefusal> {
        if self.minimum.0 > host.total.0.saturating_sub(host.reserve.0) {
            Err(FaCapacityRefusal {
                required: self.minimum,
                reserve: host.reserve,
                total: host.total,
            })
        } else {
            Ok(HostFeasibleFaDemand { demand: self })
        }
    }
}

/// A feasible observation, never a permit to execute or allocate memory.
#[derive(Debug)]
pub(crate) struct HostFeasibleFaDemand {
    demand: FaProjectedDemand,
}

/// Hosting agreement and physical feasibility, before any execution lease.
#[derive(Debug)]
pub(crate) enum FaHostAssessment {
    ModelFree,
    Fits(HostFeasibleFaDemand),
    Exceeds(FaCapacityRefusal),
}

impl HostFeasibleFaDemand {
    /// Do not assume mapped weights are physically resident. Subtract only
    /// the reservation the confirmed load's lease still covers, which can
    /// include unread in-use memory and its margin.
    pub(super) fn additional_reservation(self, covered: ModelLeaseCoverage) -> MemoryMb {
        let unread = self.demand.model.0.saturating_sub(covered.reserved().0);
        MemoryMb(self.demand.workspace_with_margin.0 + unread)
    }
}

/// Exact physical quantities that made this request impossible. No duration
/// proxy and no transient available-memory reading enters this refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FaCapacityRefusal {
    pub(crate) required: MemoryMb,
    pub(crate) reserve: MemoryMb,
    pub(crate) total: MemoryMb,
}

impl FaCapacityRefusal {
    pub(crate) fn into_guard_error(self, label: String) -> MemoryGuardError {
        MemoryGuardError::ExceedsHost {
            label,
            requested_mb: self.required.0,
            reserve_mb: self.reserve.0,
            total_mb: self.total.0,
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn demand(workspace: u64, model: u64) -> FaProjectedDemand {
        FaProjectedDemand::from_projection(
            FaWorkspaceBytesV2::try_from(workspace).expect("positive fixture workspace"),
            FaModelBytesV2::try_from(model).expect("positive fixture model"),
        )
    }

    #[test]
    fn the_exact_physical_boundary_fits_and_one_less_does_not() {
        let demand = demand(64 * 1024 * 1024, 1024 * 1024);
        let required = crate::worker::residency::WORKER_BASELINE.reserved().0 + 81;
        let exact = FaPhysicalCapacity {
            total: MemoryMb(required + 64),
            reserve: MemoryMb(64),
        };
        assert!(demand.assess(exact).is_ok());
        let refusal = demand
            .assess(FaPhysicalCapacity {
                total: MemoryMb(required + 63),
                ..exact
            })
            .expect_err("one MiB too small");
        assert_eq!(refusal.required, MemoryMb(required));
        assert_eq!(refusal.reserve, MemoryMb(64));
        assert_eq!(refusal.total, MemoryMb(required + 63));
    }

    #[test]
    fn reservations_round_up_and_cover_unread_weights_without_double_counting() {
        let demand = demand(1, 3 * 1024 * 1024);
        let feasible = demand
            .assess(FaPhysicalCapacity {
                total: MemoryMb(16_384),
                reserve: MemoryMb(512),
            })
            .expect("declared host fits");
        assert_eq!(
            feasible.additional_reservation(ModelLeaseCoverage::declared(MemoryMb(1))),
            MemoryMb(4)
        );
        let feasible = demand
            .assess(FaPhysicalCapacity {
                total: MemoryMb(16_384),
                reserve: MemoryMb(512),
            })
            .expect("same demand fits");
        assert_eq!(
            feasible.additional_reservation(ModelLeaseCoverage::declared(MemoryMb(8))),
            MemoryMb(2)
        );
    }

    #[test]
    fn transient_pressure_cannot_change_physical_feasibility() {
        let snapshot = HostMemorySnapshot {
            total_mb: MemoryMb(16_384),
            available_mb: MemoryMb(0),
            reserve_mb: MemoryMb(512),
            active_reserved_mb: MemoryMb(16_384),
            startup_leases: 1,
            job_execution_leases: 1,
            ml_test_locks: 0,
            active_lease_labels: vec!["declared busy owner".into()],
            pressure_level: crate::api::HostMemoryPressureLevel::Critical,
        };
        let idle = HostMemorySnapshot {
            available_mb: MemoryMb(16_384),
            active_reserved_mb: MemoryMb(0),
            startup_leases: 0,
            job_execution_leases: 0,
            active_lease_labels: Vec::new(),
            pressure_level: crate::api::HostMemoryPressureLevel::Healthy,
            ..snapshot.clone()
        };
        let request = demand(64 * 1024 * 1024, 1024 * 1024);
        assert!(
            request
                .assess(FaPhysicalCapacity::from_snapshot(&snapshot))
                .is_ok()
        );
        assert!(
            request
                .assess(FaPhysicalCapacity::from_snapshot(&idle))
                .is_ok()
        );
    }
}
