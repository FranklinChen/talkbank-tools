//! Memory-pressure-driven idle-worker eviction. Sibling to the
//! cap-driven `eviction.rs` (which evicts to make room at checkout).
//!
//! Pressure relief uses twice the current host tier's admission floor. Actual
//! queue observations carry allocation identity through RSS sampling and pure
//! selection. Retirement re-admits the same allocation against the current queue.

use crate::host_memory::{MemoryReclaim, Reclaimed};
use tracing::warn;

use super::GroupsMap;
use super::rss_observer::{IdleMember, IdlePopulation, Measured};

/// Pressure relief follows the current host's admitted tier and headroom floor.
pub(super) fn pressure_threshold(host: &crate::host_facts::RuntimeHost) -> u64 {
    super::memory_gate::host_min_free_mb_threshold_for_tier(&host.tier()).saturating_mul(2)
}

/// Pure selection rule: given live samples, the host's current
/// available memory, and the eviction threshold, return the subset
/// of samples to evict (largest RSS first) such that simulated
/// post-eviction available memory rises above the threshold or all
/// samples are exhausted.
///
/// Returns an empty `Vec` when `available_mb >= threshold_mb` (no
/// pressure) or when `samples` is empty.
///
/// The simulation is greedy and one-pass: each candidate, in
/// descending RSS order, is added to the eviction set, and its RSS
/// is added to a running "would-be-freed" accumulator. The function
/// stops as soon as `available_mb + freed_mb > threshold_mb`. This
/// over-evicts by at most one worker (the one that crosses the
/// threshold): acceptable because the alternative (perfect
/// minimum-set) requires more bookkeeping for marginal benefit.
pub(super) fn select_pressure_evictions<T>(
    samples: Vec<Measured<T>>,
    available_mb: u64,
    threshold_mb: u64,
) -> Vec<Measured<T>> {
    if available_mb >= threshold_mb {
        return Vec::new();
    }
    let mut sorted = samples;
    sorted.sort_by_key(|w| std::cmp::Reverse(w.rss_mb()));

    let mut evictions = Vec::new();
    let mut freed_mb: u64 = 0;
    for sample in sorted {
        freed_mb = freed_mb.saturating_add(sample.rss_mb());
        evictions.push(sample);
        if available_mb.saturating_add(freed_mb) > threshold_mb {
            break;
        }
    }
    evictions
}

/// One consuming population binds queue identity to its actual sampled RSS.
pub(super) fn snapshot_idle_workers_with_rss(groups: &GroupsMap) -> Vec<Measured<IdleMember>> {
    match IdlePopulation::all(groups).measure() {
        Ok(samples) => samples,
        Err(error) => {
            tracing::warn!(%error, "worker RSS unreadable; skipping idle eviction");
            Vec::new()
        }
    }
}

/// Current queue admission consumes the original allocation observation. A
/// worker that left the queue or a replacement with the same PID cannot match.
pub(super) async fn retire_sampled(sample: Measured<IdleMember>) -> bool {
    let item = sample.into_item();
    let pid = item.pid();
    let Some(mut worker) = item.take() else {
        return false;
    };
    if let Err(error) = worker.shutdown_in_place().await {
        warn!(%pid, %error, "Idle worker did not shut down cleanly");
    }
    true
}

/// The pool's idle workers, as the room a queued host-memory request asks
/// for: one reclaim retires the largest idle worker, which releases its
/// leases and its memory.
#[derive(Clone)]
pub(crate) struct IdleWorkers(pub(super) GroupsMap);

#[async_trait::async_trait]
impl MemoryReclaim for IdleWorkers {
    async fn reclaim_one(&self) -> Reclaimed {
        let mut samples = snapshot_idle_workers_with_rss(&self.0);
        samples.sort_by_key(|sample| std::cmp::Reverse(sample.rss_mb()));
        for sample in samples {
            let pid = sample.item().pid();
            let key = sample.item().key.clone();
            let rss_mb = sample.rss_mb();
            if retire_sampled(sample).await {
                tracing::info!(
                    pid = %pid,
                    rss_mb,
                    target = %key.target.label(),
                    lang = %key.language,
                    "Retired an idle worker to make room for a queued host-memory request"
                );
                return Reclaimed::Retired;
            }
        }
        Reclaimed::NothingIdle
    }
}

#[cfg(test)]
mod tests {
    use super::super::WorkerGroup;
    use super::*;
    use crate::api::{LanguageCode3, WorkerLanguage};
    use crate::worker::pool::{EngineSelection, WorkerKey};
    use crate::worker::{WorkerProfile, WorkerTarget};
    use std::sync::Arc;

    /// Guards: Memory reclamation retires the actual idle subprocess, consumes its availability and original capacity, then reports NothingIdle.
    /// Kind: contract
    /// Not a type: Slot ownership excludes manual refunds. Actual resident-memory sampling, subprocess shutdown and the queue's reclaim selection remain operating-system and adapter effects requiring the real admitted worker.
    /// State: One owned shell stub reports its real PID so RSS sampling observes the actual process. Its slot enters through normal publication. The test uses real process teardown with no model, network service or live memory lease.
    /// Concurrency: The fixture owns its process, group and reclaim map. Awaited reclaim establishes completion, and native handle drop owns cleanup. There is no shared worker or sleep-based readiness.
    #[tokio::test]
    async fn a_reclaim_retires_an_idle_worker_and_then_finds_none() {
        // The stub names its own shell's PID in its ready line (the shared
        // script's fixed PID is no process), so the reclaim's sample reads a
        // live process's resident memory, as it does for a real worker.
        let script =
            super::super::checkout::tests::never_responds_script().replace("424242", "'\"$$\"'");
        let handle = crate::worker::handle::WorkerHandle::spawn_stub_for_test(&script).await;
        let group = Arc::new(WorkerGroup::new(
            super::super::capacity::GlobalCapacity::new(1),
            WorkerProfile::Stanza,
            EngineSelection::none(),
        ));
        super::super::capacity::ReservedWorkerSlot::try_claim(&group, 1)
            .expect("admit the actual reclaim worker")
            .publish_stdio(handle);
        let key = WorkerKey::without_engine_selection(
            WorkerTarget::Profile(WorkerProfile::Stanza),
            WorkerLanguage::from(LanguageCode3::eng()),
        );
        // Kernel sampling does not reserve the observed queue member. A real
        // concurrent checkout invalidates its later queue-effect admission.
        let observed = group.observe_idle().pop().expect("published allocation");
        let departed = group.take_stdio().expect("checkout actual allocation");
        assert!(observed.take().is_none());
        departed.return_idle();
        let idle = IdleWorkers(Arc::new(std::sync::Mutex::new(
            std::collections::HashMap::from([(key, group.clone())]),
        )));

        assert_eq!(idle.reclaim_one().await, Reclaimed::Retired);
        assert!(group.idle.view().is_empty());
        assert_eq!(group.total.count(), 0);
        assert_eq!(idle.reclaim_one().await, Reclaimed::NothingIdle);
    }

    /// Guards: greedy pressure selection preserves descending RSS order, exact
    /// threshold strictness and best-effort exhaustion over the carried candidates.
    /// Kind: policy
    /// Not a type: which candidates to reclaim for given measured weights and
    /// a headroom target is a semantic choice after source/measurement association.
    /// State: declared logical candidates require no worker, group, process or semaphore.
    /// Concurrency: none
    #[test]
    fn pressure_selection_consumes_the_measured_candidates() {
        for (weights, available, target, expected) in [
            (vec![8000, 4000], 4096, 4096, vec![]),
            (vec![], 100, 4096, vec![]),
            (vec![1000, 4000, 2000], 1000, 8000, vec![4000, 2000, 1000]),
            (vec![5000, 1000, 500], 1000, 4000, vec![5000]),
            (vec![100, 200, 50], 0, 4096, vec![200, 100, 50]),
        ] {
            let samples = weights
                .into_iter()
                .enumerate()
                .map(|(id, rss)| Measured::declared(id, rss))
                .collect();
            let selected = select_pressure_evictions(samples, available, target);
            assert_eq!(
                selected.iter().map(Measured::rss_mb).collect::<Vec<_>>(),
                expected
            );
            let ids: std::collections::HashSet<_> =
                selected.into_iter().map(Measured::into_item).collect();
            assert_eq!(ids.len(), expected.len());
        }
    }

    /// Guards: pressure relief derives twice the current host's admitted floor
    /// across every tier instead of retaining one workstation-only threshold.
    /// Kind: policy
    /// Not a type: the headroom multiplier and tier policy are explicit choices,
    /// while the target's association with its actual host is computed once.
    /// State: declared host snapshots exercise the same threshold producer without live probes.
    /// Concurrency: none
    #[test]
    fn pressure_relief_uses_the_current_hosts_floor() {
        use crate::types::runtime::MemoryTierKind;
        for (kind, expected) in [
            (MemoryTierKind::Small, 2048),
            (MemoryTierKind::Medium, 4096),
            (MemoryTierKind::Large, 8192),
            (MemoryTierKind::Fleet, 8192),
        ] {
            let host = crate::host_facts::HostSnapshot::for_tests(&crate::config::ServerConfig {
                memory_tier: Some(kind),
                ..Default::default()
            })
            .runtime_host();
            assert_eq!(pressure_threshold(&host), expected);
        }
    }
}
