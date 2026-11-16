//! Idle-worker eviction for `WorkerPool::checkout` saturation handling.
//!
//! When the pool's global cap is saturated and the requesting key has zero
//! live workers, pop an idle worker from another group to free a slot for
//! the new key. Without this, a saturated pool that happens to hold idle
//! workers of unrelated keys rejects new work with a "would deadlock"
//! error even though a slot could be freed synchronously.
//!
//! [`select_eviction_target`] is the pure selection rule (tested in
//! isolation); [`WorkerPool::try_evict_idle_from_other_group`] is the
//! imperative wrapper that acquires the live state.

use tracing::info;

use super::{WorkerKey, WorkerPool, lock_recovered};

/// Selection consumes actual metadata with each observation. Empty and excluded
/// groups never acquire candidate status, so the effect receives its original home.
fn select_eviction_target<K: Eq, T>(
    groups: impl IntoIterator<Item = (K, T, usize)>,
    skip_key: &K,
) -> Option<(K, T)> {
    groups
        .into_iter()
        .filter_map(|(key, source, count)| {
            if &key == skip_key {
                return None;
            }
            std::num::NonZeroUsize::new(count).map(|count| (key, source, count))
        })
        .max_by_key(|(_, _, count)| *count)
        .map(|(key, source, _)| (key, source))
}

/// Result of a single eviction attempt. Named so callers don't have to
/// read `bool` and guess which way is "we freed a slot".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EvictionOutcome {
    /// A slot was freed; the caller should retry spawn.
    Evicted,
    /// No other group had an idle worker (or a race stole the one we saw).
    NoIdleElsewhere,
}

impl WorkerPool {
    /// Evict one idle worker from any group other than `skip_key`.
    ///
    /// Snapshots idle counts under the groups-map lock to avoid nested
    /// idle-then-groups lock acquisitions elsewhere (the established
    /// order in this crate is groups → idle). Runs the evicted handle's
    /// drop on a detached task so the on-request-path `checkout` never
    /// waits on worker shutdown.
    pub(super) fn try_evict_idle_from_other_group(&self, skip_key: &WorkerKey) -> EvictionOutcome {
        let victim = {
            let groups = lock_recovered(&self.groups);
            select_eviction_target(
                groups
                    .iter()
                    .map(|(key, home)| (key, home, home.idle.view().len())),
                &skip_key,
            )
            .map(|(key, home)| (key.clone(), home.clone()))
        };
        let Some((victim_key, victim_group)) = victim else {
            return EvictionOutcome::NoIdleElsewhere;
        };

        let Some(handle) = victim_group.take_stdio() else {
            return EvictionOutcome::NoIdleElsewhere;
        };

        info!(
            victim = ?victim_key,
            requesting = ?skip_key,
            "Evicting idle worker to free a global-cap slot"
        );
        // Detach the handle drop (WorkerHandle::Drop sends SIGTERM+SIGKILL).
        // At runtime-shutdown the reaper in lifecycle.rs cleans up any
        // process this task would have killed.
        //
        // On the POOL's runtime, not the caller's: the drop signals the child
        // process, so it has to run where that child's reactor lives. An
        // eviction is triggered by whichever checkout happens to be saturated,
        // which in the harness is an arbitrary test's runtime. The owner also
        // decides what to do when the pool has no runtime at all.
        self.spawn_runtime.retire_worker_handle(handle.retiring());

        EvictionOutcome::Evicted
    }
}

#[cfg(test)]
mod tests;
