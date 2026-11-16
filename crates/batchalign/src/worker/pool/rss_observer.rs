//! Per-process RSS observation for worker-pool admission (Mode B).
//!
//! # Why this exists
//!
//! Mode A's memory gate (`memory_gate.rs`) uses each profile's
//! tier-derived `startup_reservation_mb` as the projection of "how
//! much will the new worker consume?" That estimate is conservative
//! by design: it covers the *startup peak* during model load, not
//! the steady-state RSS after the worker settles into idle. On
//! long-running daemons whose workers have settled, reservation
//! over-estimates by 2-3× (e.g., a Stanza worker's 12 GB reservation
//! vs ~4 GB steady RSS).
//!
//! Mode B replaces the static reservation with a *measured* estimate
//! drawn from same-profile idle workers' actual RSS. When idle peers
//! exist, we know what a warmed-up worker of this profile actually
//! consumes; that's a tighter projection and admits more workers
//! when the host has the headroom.
//!
//! # Why idle workers, not all live workers
//!
//! `WorkerGroup.idle` holds workers that have completed model load
//! and returned to the pool, settled steady-state. Checked-out
//! workers are mid-inference and may carry transient request data
//! that overstates steady-state RSS. Workers loading their model
//! aren't in `idle` yet (they're held inside `try_spawn_into_group`
//! until ready). Idle is the cleanest sample for "what does a
//! warmed-up worker of this profile cost?"
//!
//! # Bootstrap
//!
//! When no idle peers exist (first spawn of the profile after pool
//! startup, or all peers currently checked out), the caller falls
//! back to the static reservation, same as Mode A. This means Mode
//! B is strictly an *opt-in* improvement over Mode A: when peer
//! observation is unavailable, behavior matches Mode A exactly.
//!
//! # Sysinfo cost
//!
//! Each admission attempt opens one `sysinfo::System` and refreshes
//! the specific worker PIDs we're interested in. Order ~ms for ≤10
//! PIDs. The cost is in the cold admission path; if `try_claim_spawn_slot`
//! ever moves to the hot per-request path, this should shift to
//! background polling with a cached read.

use std::collections::HashMap;

use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate};

use crate::worker::{WorkerPid, WorkerProfile};

use super::{GroupsMap, lock_recovered};

/// Origin of the per-spawn memory estimate, surfaced via
/// `MemoryConstrained` for diagnostic logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EstimateSource {
    /// No same-profile idle peers were available. Fell back to the
    /// static `startup_reservation_mb_for_tier` value: Mode A
    /// behavior.
    Reservation,
    /// Same-profile idle peers were sampled; the average of their
    /// observed RSS is the estimate. Mode B behavior.
    ObservedAvgIdle,
}

/// The actual queue member and diagnostic key stay attached during sampling.
pub(super) struct IdleMember {
    pub(super) key: super::WorkerKey,
    source: super::capacity::ObservedIdleWorker,
}
impl IdleMember {
    pub(super) fn pid(&self) -> WorkerPid {
        self.source.pid()
    }
    pub(super) fn take(
        self,
    ) -> Option<super::capacity::DepartedWorker<crate::worker::handle::WorkerHandle>> {
        self.source.take()
    }
}

pub(super) struct IdlePopulation {
    members: Vec<IdleMember>,
}
impl IdlePopulation {
    pub(super) fn all(groups: &GroupsMap) -> Self {
        Self::capture(groups, |_: WorkerProfile| true)
    }
    fn for_profile(groups: &GroupsMap, profile: WorkerProfile) -> Self {
        Self::capture(groups, |found| found == profile)
    }
    fn capture(groups: &GroupsMap, selected: impl Fn(WorkerProfile) -> bool) -> Self {
        let homes: Vec<_> = lock_recovered(groups)
            .iter()
            .filter(|(_, group)| selected(group.profile))
            .map(|(key, home)| (key.clone(), home.clone()))
            .collect();
        let members = homes
            .into_iter()
            .flat_map(|(key, home)| {
                home.observe_idle()
                    .into_iter()
                    .map(move |source| IdleMember {
                        key: key.clone(),
                        source,
                    })
            })
            .collect();
        Self { members }
    }
    pub(super) fn measure(
        self,
    ) -> Result<Vec<Measured<IdleMember>>, crate::host_memory::HostUnreadable> {
        let pids: Vec<_> = self.members.iter().map(IdleMember::pid).collect();
        let readings = refresh_rss_mb_for_pids(&pids)?;
        Ok(self
            .members
            .into_iter()
            .filter_map(|item| {
                readings
                    .get(&item.pid())
                    .copied()
                    .map(|rss_mb| Measured { item, rss_mb })
            })
            .collect())
    }
}

/// A measurement carries the exact object it describes through selection.
pub(super) struct Measured<T> {
    item: T,
    rss_mb: u64,
}
impl<T> Measured<T> {
    pub(super) fn rss_mb(&self) -> u64 {
        self.rss_mb
    }
    pub(super) fn item(&self) -> &T {
        &self.item
    }
    pub(super) fn into_item(self) -> T {
        self.item
    }
    /// Pure policy observations claim no native worker or process identity.
    #[cfg(test)]
    pub(super) fn declared(item: T, rss_mb: u64) -> Self {
        Self { item, rss_mb }
    }
}

pub(super) fn observed_avg_rss_mb_for_profile(
    groups: &GroupsMap,
    profile: WorkerProfile,
) -> Option<u64> {
    let samples = match IdlePopulation::for_profile(groups, profile).measure() {
        Ok(samples) => samples,
        Err(error) => {
            tracing::warn!(%error, "worker RSS unreadable; using the reservation estimate");
            return None;
        }
    };
    average_rss(&samples)
}

fn average_rss<T>(samples: &[Measured<T>]) -> Option<u64> {
    let count = std::num::NonZeroUsize::new(samples.len())?;
    let total: u128 = samples.iter().map(|sample| u128::from(sample.rss_mb)).sum();
    Some((total / count.get() as u128) as u64)
}

/// Refresh sysinfo for the given PIDs and return per-PID RSS in MB
/// for the ones that are still alive. PIDs that have died (or were
/// never live) are silently absent from the result map; callers
/// distinguish "missing" from "live with zero RSS" by membership.
///
/// Shared by `observed_avg_rss_mb_for_profile` (averages across the
/// values) and `idle_eviction::snapshot_idle_workers_with_rss`
/// (joins with caller-provided metadata). Both sites previously
/// hand-rolled the same `Pid::from_u32 + System::new + refresh +
/// per-PID process lookup` triple; consolidating here keeps the
/// sysinfo refresh shape (use_kind = `with_memory()`,
/// remove_dead = `true`) in one place.
pub(super) fn refresh_rss_mb_for_pids(
    pids: &[WorkerPid],
) -> Result<HashMap<WorkerPid, u64>, crate::host_memory::HostUnreadable> {
    if pids.is_empty() {
        return Ok(HashMap::new());
    }
    let sysinfo_pids: Vec<Pid> = pids.iter().map(|p| Pid::from_u32(p.0)).collect();
    let mut sys = crate::host_memory::open_system()?;
    sys.refresh_processes_specifics(
        ProcessesToUpdate::Some(&sysinfo_pids),
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    let mut out = HashMap::with_capacity(pids.len());
    for pid in pids {
        let sysinfo_pid = Pid::from_u32(pid.0);
        if let Some(process) = sys.process(sysinfo_pid) {
            out.insert(*pid, process.memory() / (1024 * 1024));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Guards: the actual process-table adapter retains this live process's PID
    /// and positive RSS when refreshing an explicitly selected population.
    /// Kind: measurement
    /// Not a type: the operating system supplies actual resident bytes at runtime,
    /// beyond the static association between each queued owner and its observations.
    /// State: one real sysinfo process-table refresh observes only this test process.
    /// Concurrency: the process owns its PID and the local refresh shares no mutable state.
    #[test]
    fn the_kernel_refresh_retains_the_live_process_measurement() {
        let pid = WorkerPid(std::process::id());
        let readings = refresh_rss_mb_for_pids(&[pid]).unwrap();
        assert!(readings.get(&pid).is_some_and(|rss| *rss > 0));
    }
    /// Guards: empty observations remain absent and overflow-safe averaging uses
    /// every carried measurement exactly once, including zero and maximum values.
    /// Kind: policy
    /// Not a type: the arithmetic mean, integer rounding and genuine absence are
    /// semantic decisions over admitted numeric observations.
    /// State: declared logical values claim no real process or queued worker.
    /// Concurrency: none
    #[test]
    fn averaging_consumes_only_the_carried_measurements() {
        for (weights, expected) in [
            (vec![], None),
            (vec![0, 3], Some(1)),
            (vec![u64::MAX, u64::MAX], Some(u64::MAX)),
        ] {
            let samples = weights
                .into_iter()
                .enumerate()
                .map(|(id, rss)| Measured::declared(id, rss))
                .collect::<Vec<_>>();
            assert_eq!(average_rss(&samples), expected);
        }
    }
}
