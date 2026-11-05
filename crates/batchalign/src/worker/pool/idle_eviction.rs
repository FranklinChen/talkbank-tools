//! Memory-pressure-driven idle-worker eviction. Sibling to the
//! cap-driven `eviction.rs` (which evicts to make room at checkout).
//!
//! When `available_memory_mb` falls at or below
//! [`EVICTION_PRESSURE_THRESHOLD_MB`] (= 2× the admission-gate floor
//! at `memory_gate::MIN_FREE_MEMORY_MB`), idle workers are evicted
//! largest-RSS first so each shutdown maximizes pressure relief.
//! Above the threshold, this module is a no-op and the time-based
//! eviction in `run_health_check` is the sole eviction trigger.

use std::sync::Arc;

use crate::host_memory::{MemoryReclaim, Reclaimed};
use crate::worker::WorkerPid;
use tracing::warn;

use super::rss_observer::refresh_rss_mb_for_pids;
use super::{GroupsMap, WorkerGroup, WorkerKey, lock_recovered};

/// Available-memory threshold below which idle-worker eviction is
/// triggered. Hardcoded at 2× the admission-gate floor so eviction
/// kicks in *before* admission would refuse, not after.
pub(super) const EVICTION_PRESSURE_THRESHOLD_MB: u64 = 4096;

/// One idle worker considered for eviction, with its observed RSS
/// at sample time. Carries an owning `Arc<WorkerGroup>` so the
/// eviction commit doesn't need to re-acquire the groups mutex to
/// re-look-up the group; one snapshot covers both phases.
#[derive(Clone)]
pub(super) struct IdleWorkerSample {
    /// Group key the worker belongs to. Retained for diagnostic
    /// log fields (target / lang).
    pub(super) key: WorkerKey,
    /// Owning handle on the group; resolves the snapshot/commit
    /// race where the group might otherwise be reaped from the map
    /// between phases.
    pub(super) group: Arc<WorkerGroup>,
    /// Worker process PID: stable across the snapshot/commit
    /// boundary as long as the process is alive.
    pub(super) pid: WorkerPid,
    /// Resident memory in MB at sample time.
    pub(super) rss_mb: u64,
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
pub(super) fn select_pressure_evictions(
    samples: Vec<IdleWorkerSample>,
    available_mb: u64,
    threshold_mb: u64,
) -> Vec<IdleWorkerSample> {
    if available_mb >= threshold_mb {
        return Vec::new();
    }
    let mut sorted = samples;
    sorted.sort_by_key(|w| std::cmp::Reverse(w.rss_mb));

    let mut evictions = Vec::new();
    let mut freed_mb: u64 = 0;
    for sample in sorted {
        evictions.push(sample.clone());
        freed_mb = freed_mb.saturating_add(sample.rss_mb);
        if available_mb.saturating_add(freed_mb) > threshold_mb {
            break;
        }
    }
    evictions
}

/// Walk live groups, snapshot all idle workers as `IdleWorkerSample`s
/// with sysinfo-sampled RSS. Caller pairs the samples with
/// [`select_pressure_evictions`] to decide which workers to evict.
///
/// PIDs are collected under each group's `idle` mutex briefly; the
/// sysinfo refresh is performed outside the mutex (same pattern as
/// `rss_observer::observed_avg_rss_mb_for_profile`) so concurrent
/// checkout/push_back callers aren't blocked across the refresh.
pub(super) fn snapshot_idle_workers_with_rss(groups: &GroupsMap) -> Vec<IdleWorkerSample> {
    let group_arcs: Vec<(WorkerKey, Arc<WorkerGroup>)> = {
        let groups = lock_recovered(groups);
        groups.iter().map(|(k, g)| (k.clone(), g.clone())).collect()
    };

    let mut entries: Vec<(WorkerKey, Arc<WorkerGroup>, WorkerPid)> = Vec::new();
    for (key, group) in &group_arcs {
        let idle = lock_recovered(&group.idle);
        for handle in idle.iter() {
            entries.push((key.clone(), group.clone(), handle.pid()));
        }
    }
    if entries.is_empty() {
        return Vec::new();
    }

    let pids: Vec<WorkerPid> = entries.iter().map(|(_, _, pid)| *pid).collect();
    // An unreadable process table evicts nothing this pass: choosing victims
    // by RSS with no RSS would be a guess, and the next pass reads again.
    let rss_by_pid = match refresh_rss_mb_for_pids(&pids) {
        Ok(rss_by_pid) => rss_by_pid,
        Err(error) => {
            tracing::warn!(%error, "worker RSS unreadable; skipping idle eviction");
            return Vec::new();
        }
    };

    entries
        .into_iter()
        .filter_map(|(key, group, pid)| {
            // Worker died between snapshot and refresh → absent from
            // `rss_by_pid`. The health-check liveness loop will reap
            // the slot.
            rss_by_pid
                .get(&pid)
                .copied()
                .map(|rss_mb| IdleWorkerSample {
                    key,
                    group,
                    pid,
                    rss_mb,
                })
        })
        .collect()
}

/// Retire one sampled idle worker: take it off its group's idle queue with
/// its permit, shut it down, and account for it until it is gone (the
/// teardown every eviction uses). `false` when the worker was checked out
/// since the sample, so it is serving a request and stays.
pub(super) async fn retire_sampled(sample: &IdleWorkerSample) -> bool {
    let group = &sample.group;
    // Taken off the queue under its lock, held only that long; the shutdown
    // happens after it is released.
    let (mut worker, leaving) = {
        let mut idle = super::lock_recovered(&group.idle);
        let Some(worker) = idle
            .iter()
            .position(|handle| handle.pid() == sample.pid)
            .and_then(|position| idle.remove(position))
        else {
            return false;
        };
        // Consume the idle permit that matched the handle. A missing one is
        // an upstream permit-versus-queue defect, surfaced rather than
        // masked; the worker still goes.
        match group.available.try_acquire() {
            Ok(permit) => permit.forget(),
            Err(error) => warn!(
                pid = %sample.pid,
                %error,
                "Idle queue had a worker but no matching permit; retiring it without one"
            ),
        }
        // The worker is still counted in `total` through its shutdown,
        // which can take seconds; the guard accounts for it from the moment
        // it leaves the queue until it is retired, so a saturated checkout
        // never mistakes it for a lost worker.
        (worker, super::AwayFromQueue::new(group, 1))
    };
    if let Err(error) = worker.shutdown_in_place().await {
        warn!(pid = %sample.pid, %error, "Idle worker did not shut down cleanly");
    }
    leaving.retire();
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
        samples.sort_by_key(|sample| std::cmp::Reverse(sample.rss_mb));
        for sample in samples {
            if retire_sampled(&sample).await {
                tracing::info!(
                    pid = %sample.pid,
                    rss_mb = sample.rss_mb,
                    target = %sample.key.target.label(),
                    lang = %sample.key.language,
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
    use super::*;
    use crate::api::{LanguageCode3, WorkerLanguage};
    use crate::worker::pool::{EngineSelection, WorkerKey};
    use crate::worker::{WorkerProfile, WorkerTarget};
    use tokio::sync::Semaphore;

    /// A queued request's reclaim retires the pool's idle worker: off its
    /// queue with its permit consumed, shut down and no longer counted. The
    /// next reclaim finds nothing idle, so a waiting request falls back to
    /// polling instead of retiring in a loop.
    #[tokio::test]
    async fn a_reclaim_retires_an_idle_worker_and_then_finds_none() {
        use std::sync::atomic::Ordering;

        // The stub names its own shell's PID in its ready line (the shared
        // script's fixed PID is no process), so the reclaim's sample reads a
        // live process's resident memory, as it does for a real worker.
        let script =
            super::super::checkout::tests::never_responds_script().replace("424242", "'\"$$\"'");
        let handle = crate::worker::handle::WorkerHandle::spawn_stub_for_test(&script).await;
        let group = Arc::new(WorkerGroup::new(
            Arc::new(tokio::sync::Notify::new()),
            Arc::new(Semaphore::new(0)),
            WorkerProfile::Stanza,
            EngineSelection::none(),
        ));
        group.total.fetch_add(1, Ordering::Relaxed);
        lock_recovered(&group.idle).push_back(handle);
        group.available.add_permits(1);
        let key = WorkerKey::without_engine_selection(
            WorkerTarget::Profile(WorkerProfile::Stanza),
            WorkerLanguage::from(LanguageCode3::eng()),
        );
        let idle = IdleWorkers(Arc::new(std::sync::Mutex::new(
            std::collections::HashMap::from([(key, group.clone())]),
        )));

        assert_eq!(idle.reclaim_one().await, Reclaimed::Retired);
        assert!(lock_recovered(&group.idle).is_empty());
        assert_eq!(group.available.available_permits(), 0);
        assert_eq!(group.total.load(Ordering::Relaxed), 0);
        assert_eq!(idle.reclaim_one().await, Reclaimed::NothingIdle);
    }

    fn make_sample(pid: u32, rss_mb: u64) -> IdleWorkerSample {
        let spawn_permits = Arc::new(Semaphore::new(0));
        let worker_returned = Arc::new(tokio::sync::Notify::new());
        IdleWorkerSample {
            key: WorkerKey::without_engine_selection(
                WorkerTarget::Profile(WorkerProfile::Stanza),
                WorkerLanguage::from(LanguageCode3::eng()),
            ),
            group: Arc::new(WorkerGroup::new(
                worker_returned,
                spawn_permits,
                WorkerProfile::Stanza,
                EngineSelection::none(),
            )),
            pid: WorkerPid(pid),
            rss_mb,
        }
    }

    /// `available >= threshold` returns no evictions, regardless of
    /// how many samples are passed. The gate isn't fired.
    #[test]
    fn returns_empty_when_available_meets_threshold() {
        let samples = vec![make_sample(1, 8_000), make_sample(2, 4_000)];
        let result = select_pressure_evictions(samples, 4_096, 4_096);
        assert!(
            result.is_empty(),
            "expected no evictions at available=threshold, got {} samples",
            result.len()
        );
    }

    /// Empty sample list returns empty regardless of pressure level.
    /// Pins that the caller can trust no spurious evictions when no
    /// idle workers exist.
    #[test]
    fn returns_empty_when_no_samples() {
        let result = select_pressure_evictions(Vec::new(), 100, 4_096);
        assert!(result.is_empty());
    }

    /// Under pressure, evictions are returned in descending-RSS
    /// order. The largest sample is first; subsequent samples are
    /// each smaller.
    #[test]
    fn evicts_largest_rss_first() {
        let samples = vec![
            make_sample(1, 1_000),
            make_sample(2, 4_000),
            make_sample(3, 2_000),
        ];
        // available=1 GB, threshold=8 GB → need to free at least
        // 7 GB to exceed threshold. Largest is 4 GB; pick that
        // (still 5 GB total < 8); pick next 2 GB (now 7 GB total ==
        // 8 GB available: still not > threshold); pick last 1 GB
        // (now 8 GB total == 9 GB available > 8 threshold). Evict
        // all three, in descending order.
        let result = select_pressure_evictions(samples, 1_000, 8_000);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].rss_mb, 4_000);
        assert_eq!(result[1].rss_mb, 2_000);
        assert_eq!(result[2].rss_mb, 1_000);
    }

    /// One eviction is sufficient when the largest worker frees
    /// enough memory. Subsequent samples are not added once
    /// projected_available > threshold.
    #[test]
    fn stops_after_first_sufficient_eviction() {
        let samples = vec![
            make_sample(1, 5_000),
            make_sample(2, 1_000),
            make_sample(3, 500),
        ];
        // available=1 GB, threshold=4 GB. Pick 5 GB sample; now
        // projected = 6 GB > 4 GB threshold. Stop.
        let result = select_pressure_evictions(samples, 1_000, 4_000);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].rss_mb, 5_000);
    }

    /// When all samples together still leave projected memory at or
    /// below threshold, every sample is returned. The caller knows
    /// the eviction is best-effort but won't fully relieve the
    /// pressure.
    #[test]
    fn returns_all_samples_when_total_freeing_insufficient() {
        let samples = vec![make_sample(1, 100), make_sample(2, 200), make_sample(3, 50)];
        // available=0 MB, threshold=4_096 MB. Total freed = 350 MB.
        // 0 + 350 < 4_096 → still no relief. Evict everything anyway.
        let result = select_pressure_evictions(samples, 0, 4_096);
        assert_eq!(result.len(), 3);
        // Also confirm desc order is preserved across the
        // exhaust-everything path.
        assert_eq!(result[0].rss_mb, 200);
        assert_eq!(result[1].rss_mb, 100);
        assert_eq!(result[2].rss_mb, 50);
    }

    /// The threshold constant must equal exactly 2× the admission
    /// gate's floor. If anyone changes either number the relationship
    /// should be examined deliberately.
    #[test]
    fn threshold_is_twice_the_admission_floor() {
        assert_eq!(
            EVICTION_PRESSURE_THRESHOLD_MB,
            *batchalign_types::memory::MIN_FREE_MEMORY_MB * 2
        );
    }
}
