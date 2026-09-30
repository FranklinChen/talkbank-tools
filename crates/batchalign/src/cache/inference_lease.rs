//! Process-local single-flight ownership for expensive inference cache keys.

use std::sync::{Arc, OnceLock, Weak};

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use tokio::sync::{Mutex, OwnedMutexGuard};

use crate::chat_ops::CacheKey;

use super::{CacheInstanceId, UtteranceCache};

#[derive(Debug)]
struct InferenceLeaseCell {
    mutex: Arc<Mutex<()>>,
    committed_generations: DashMap<CacheInstanceId, u64>,
}

impl InferenceLeaseCell {
    fn new() -> Self {
        Self {
            mutex: Arc::new(Mutex::new(())),
            committed_generations: DashMap::new(),
        }
    }
}

static INFERENCE_LOCKS: OnceLock<DashMap<String, Weak<InferenceLeaseCell>>> = OnceLock::new();

fn inference_locks() -> &'static DashMap<String, Weak<InferenceLeaseCell>> {
    INFERENCE_LOCKS.get_or_init(DashMap::new)
}

/// Cancellation-safe ownership while `acquire()` is waiting for the mutex.
///
/// `InferenceLease` does not exist until the await completes. Without this
/// intermediate state, canceling a ready waiter can drop the final strong
/// reference while leaving a dead weak entry in the global registry.
struct PendingInferenceLease {
    cache_key: String,
    cell: Arc<InferenceLeaseCell>,
    cleanup_on_drop: bool,
}

impl PendingInferenceLease {
    fn finish(
        mut self,
        guard: OwnedMutexGuard<()>,
        cache_instance: CacheInstanceId,
        generation_before_wait: u64,
    ) -> InferenceLease {
        self.cleanup_on_drop = false;
        InferenceLease {
            cache_key: self.cache_key.clone(),
            cell: self.cell.clone(),
            guard: Some(guard),
            cache_instance,
            generation_before_wait,
        }
    }
}

impl Drop for PendingInferenceLease {
    fn drop(&mut self) {
        if self.cleanup_on_drop {
            remove_dead_registry_entry(&self.cache_key, &self.cell);
        }
    }
}

fn remove_dead_registry_entry(cache_key: &str, cell: &Arc<InferenceLeaseCell>) {
    inference_locks().remove_if(cache_key, |_, registered| {
        Weak::ptr_eq(registered, &Arc::downgrade(cell)) && Arc::strong_count(cell) == 1
    });
}

/// Ownership of one evidence key's lookup/inference/commit path.
///
/// Keep this value inside a task-specific miss/authorization typestate until
/// inference either commits or fails. A concurrent identical request then
/// waits and re-checks the durable cache instead of issuing duplicate work.
#[derive(Debug)]
pub(crate) struct InferenceLease {
    cache_key: String,
    cell: Arc<InferenceLeaseCell>,
    guard: Option<OwnedMutexGuard<()>>,
    cache_instance: CacheInstanceId,
    generation_before_wait: u64,
}

impl InferenceLease {
    pub(crate) async fn acquire(cache_key: &CacheKey, cache: &UtteranceCache) -> Self {
        let cache_key = cache_key.to_string();
        let cell = match inference_locks().entry(cache_key.clone()) {
            Entry::Occupied(mut entry) => match entry.get().upgrade() {
                Some(cell) => cell,
                None => {
                    let cell = Arc::new(InferenceLeaseCell::new());
                    entry.insert(Arc::downgrade(&cell));
                    cell
                }
            },
            Entry::Vacant(entry) => {
                let cell = Arc::new(InferenceLeaseCell::new());
                entry.insert(Arc::downgrade(&cell));
                cell
            }
        };
        let cache_instance = cache.instance_id();
        let generation_before_wait = cell
            .committed_generations
            .get(&cache_instance)
            .map_or(0, |generation| *generation);
        let mutex = cell.mutex.clone();
        let pending = PendingInferenceLease {
            cache_key,
            cell,
            cleanup_on_drop: true,
        };
        let guard = mutex.lock_owned().await;
        pending.finish(guard, cache_instance, generation_before_wait)
    }

    /// Whether another owner durably committed fresh evidence while this
    /// request waited. A forced refresh may replay that exact fresh result
    /// instead of issuing a second sequential paid call.
    pub(crate) fn observed_commit_while_waiting(&self) -> bool {
        self.cell
            .committed_generations
            .get(&self.cache_instance)
            .is_some_and(|generation| *generation > self.generation_before_wait)
    }

    /// How many callers currently hold or wait for `cache_key`'s lease.
    ///
    /// Test observation only: it lets a test tell, without a clock, that a
    /// second identical request has parked behind the first rather than
    /// running its own inference. The registry keeps only weak references, so
    /// every strong reference is one holder or one waiter.
    #[cfg(test)]
    pub(crate) fn contenders_for_test(cache_key: &CacheKey) -> usize {
        inference_locks()
            .get(cache_key.as_str())
            .and_then(|entry| entry.value().upgrade())
            // The upgrade itself is one more strong reference; do not count it.
            .map_or(0, |cell| Arc::strong_count(&cell) - 1)
    }

    /// Record a successful durable commit before releasing the lease.
    pub(crate) fn mark_committed(&self) {
        self.cell
            .committed_generations
            .entry(self.cache_instance)
            .and_modify(|generation| *generation += 1)
            .or_insert(1);
    }
}

impl Drop for InferenceLease {
    fn drop(&mut self) {
        drop(self.guard.take());
        remove_dead_registry_entry(&self.cache_key, &self.cell);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_commit_does_not_coalesce_a_waiter_using_another_cache_instance() {
        let key = CacheKey::from_content("cache-scoped-inference-lease-commit");
        let committed_cache = UtteranceCache::noop();
        let unrelated_cache = UtteranceCache::noop();
        let first = InferenceLease::acquire(&key, &committed_cache).await;
        let waiter_key = key.clone();
        let waiter =
            tokio::spawn(
                async move { InferenceLease::acquire(&waiter_key, &unrelated_cache).await },
            );

        tokio::task::yield_now().await;
        first.mark_committed();
        drop(first);
        let follower = waiter.await.expect("waiter task");

        assert!(!follower.observed_commit_while_waiting());
    }

    #[tokio::test]
    async fn canceling_a_ready_waiter_does_not_leave_a_dead_registry_entry() {
        let key = CacheKey::from_content("canceled-inference-lease-waiter");
        let first_cache = UtteranceCache::noop();
        let first = InferenceLease::acquire(&key, &first_cache).await;
        let waiter_key = key.clone();
        let waiter_cache = UtteranceCache::noop();
        let waiter =
            tokio::spawn(async move { InferenceLease::acquire(&waiter_key, &waiter_cache).await });

        tokio::task::yield_now().await;
        drop(first);
        waiter.abort();
        let _ = waiter.await;

        assert!(
            !inference_locks().contains_key(key.as_str()),
            "a waiter canceled after the mutex becomes ready must clean up its weak registry entry"
        );
    }
}
