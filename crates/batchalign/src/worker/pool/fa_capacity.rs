//! Physical request feasibility, separate from congestion and execution leases.

use super::task_load::ModelLeaseCoverage;
use crate::api::MemoryMb;
use crate::host_memory::{HostCeiling, HostMemorySnapshot};
use crate::types::worker_v2::{FaModelBytesV2, FaWorkspaceBytesV2};
use crate::worker::memory_guard::MemoryGuardError;

/// A physical-capacity observation deliberately cannot express current free
/// memory. Congestion belongs to the coordinator's subsequent lease queue.
/// It reads the host's [`HostCeiling`], the same bound the lease queue
/// refuses a never-fitting request against, so a request judged feasible
/// here is one the queue can eventually grant.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FaPhysicalCapacity {
    ceiling: HostCeiling,
    reserve: MemoryMb,
}

impl FaPhysicalCapacity {
    pub(crate) fn from_snapshot(snapshot: &HostMemorySnapshot) -> Self {
        Self {
            ceiling: snapshot.ceiling,
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
    /// The model and the worker baseline: what any request of this model
    /// needs before its own workspace, so a lower bound on every request.
    floor: MemoryMb,
    minimum: MemoryMb,
}

impl FaProjectedDemand {
    pub(crate) fn from_projection(workspace: FaWorkspaceBytesV2, model: FaModelBytesV2) -> Self {
        let workspace = workspace.get().div_ceil(1024 * 1024);
        let model = model.get().div_ceil(1024 * 1024);
        let margin =
            (workspace * crate::worker::residency::ACTIVATION_MARGIN_PERCENT).div_ceil(100);
        let workspace_with_margin = workspace + margin;
        let floor = model + crate::worker::residency::WORKER_BASELINE.reserved().0;
        Self {
            model: MemoryMb(model),
            workspace_with_margin: MemoryMb(workspace_with_margin),
            floor: MemoryMb(floor),
            minimum: MemoryMb(floor + workspace_with_margin),
        }
    }

    /// Judge this request against what the host can ever offer. A refusal
    /// says whether a smaller request could change it: when the model and
    /// the worker baseline alone exceed the ceiling, no request of any shape
    /// fits, since every request needs at least that.
    pub(crate) fn assess(
        self,
        host: FaPhysicalCapacity,
    ) -> Result<HostFeasibleFaDemand, FaCapacityRefusal> {
        let numbers = |required| FaCapacityNumbers {
            required,
            reserve: host.reserve,
            ceiling: host.ceiling.mb(),
        };
        if !host.ceiling.admits(self.floor, host.reserve) {
            Err(FaCapacityRefusal::ModelNeverFits(numbers(self.floor)))
        } else if !host.ceiling.admits(self.minimum, host.reserve) {
            Err(FaCapacityRefusal::RequestTooLarge(numbers(self.minimum)))
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

/// Exact physical quantities that made a request impossible. No duration
/// proxy and no transient available-memory reading enters this refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FaCapacityNumbers {
    pub(crate) required: MemoryMb,
    pub(crate) reserve: MemoryMb,
    pub(crate) ceiling: MemoryMb,
}

impl FaCapacityNumbers {
    pub(crate) fn into_guard_error(self, label: String) -> MemoryGuardError {
        MemoryGuardError::ExceedsHost(crate::host_memory::NeverFits {
            label,
            needed_mb: self.required.0,
            ceiling_mb: self.ceiling.0,
            reserve_mb: self.reserve.0,
        })
    }
}

/// Why a request can never run on this host, split by what a smaller
/// request could change. Planning acts on the difference: no request of a
/// model that never fits is worth probing, while a request that is merely
/// too large can be cut, and at its smallest cut leaves only its own words
/// untimed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FaCapacityRefusal {
    /// The model and the worker baseline alone exceed what the host can
    /// ever offer. `required` is that floor.
    ModelNeverFits(FaCapacityNumbers),
    /// The model fits; this request's workspace takes the total over.
    /// `required` is the whole request.
    RequestTooLarge(FaCapacityNumbers),
}

impl FaCapacityRefusal {
    /// The final memory refusal of a request that reached execution. Either
    /// cause is final there: planning already judged the request, so the
    /// host cannot have more to offer it now.
    pub(crate) fn into_guard_error(self, label: String) -> MemoryGuardError {
        match self {
            Self::ModelNeverFits(numbers) | Self::RequestTooLarge(numbers) => {
                numbers.into_guard_error(label)
            }
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
            ceiling: HostCeiling::for_tests(required + 64),
            reserve: MemoryMb(64),
        };
        assert!(demand.assess(exact).is_ok());
        let refusal = demand
            .assess(FaPhysicalCapacity {
                ceiling: HostCeiling::for_tests(required + 63),
                ..exact
            })
            .expect_err("one MiB too small");
        assert_eq!(
            refusal,
            FaCapacityRefusal::RequestTooLarge(FaCapacityNumbers {
                required: MemoryMb(required),
                reserve: MemoryMb(64),
                ceiling: MemoryMb(required + 63),
            })
        );
    }

    /// A host that cannot hold the model and the worker baseline before any
    /// workspace refuses every request of that model, and says so; one that
    /// holds them refuses only a request whose workspace is too large.
    #[test]
    fn a_model_that_never_fits_is_told_apart_from_a_request_too_large() {
        let demand = demand(64 * 1024 * 1024, 1024 * 1024);
        let floor = crate::worker::residency::WORKER_BASELINE.reserved().0 + 1;
        let refusal = |ceiling| {
            demand
                .assess(FaPhysicalCapacity {
                    ceiling: HostCeiling::for_tests(ceiling),
                    reserve: MemoryMb(64),
                })
                .expect_err("too small")
        };
        assert_eq!(
            refusal(floor + 63),
            FaCapacityRefusal::ModelNeverFits(FaCapacityNumbers {
                required: MemoryMb(floor),
                reserve: MemoryMb(64),
                ceiling: MemoryMb(floor + 63),
            })
        );
        assert!(matches!(
            refusal(floor + 64),
            FaCapacityRefusal::RequestTooLarge(numbers) if numbers.required.0 > floor
        ));
    }

    #[test]
    fn reservations_round_up_and_cover_unread_weights_without_double_counting() {
        let demand = demand(1, 3 * 1024 * 1024);
        let feasible = demand
            .assess(FaPhysicalCapacity {
                ceiling: HostCeiling::for_tests(16_384),
                reserve: MemoryMb(512),
            })
            .expect("declared host fits");
        assert_eq!(
            feasible.additional_reservation(ModelLeaseCoverage::declared(MemoryMb(1))),
            MemoryMb(4)
        );
        let feasible = demand
            .assess(FaPhysicalCapacity {
                ceiling: HostCeiling::for_tests(16_384),
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
            ceiling: HostCeiling::for_tests(16_384),
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
