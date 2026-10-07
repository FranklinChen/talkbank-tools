//! `CheckedOutWorker`: RAII guard for dispatched workers.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use tracing::warn;

use crate::worker::handle::{RequestFlight, WorkerHandle};

use super::WorkerGroup;

/// RAII guard that owns a [`WorkerHandle`] for the duration of a dispatch.
///
/// Created by `WorkerPool::checkout()` after acquiring a semaphore permit
/// and popping a worker from the idle queue. Dereferences to `WorkerHandle`
/// so callers can call `process()`, `batch_infer()`, etc. directly.
///
/// # Drop semantics
///
/// When this guard is dropped (whether the dispatch succeeded or failed),
/// the worker is returned to the group's idle queue and a semaphore permit
/// is released, unblocking the next caller waiting in `checkout()`.
///
/// If the worker was *taken* via [`take()`](Self::take) (e.g. because it
/// died mid-dispatch and should not be reused), `total` is decremented
/// instead and no permit is released -- the worker slot is permanently
/// freed so a fresh worker can be spawned later.
pub struct CheckedOutWorker {
    /// The worker and the guard that accounts for it, wrapped in `Option`
    /// so [`take()`](Self::take) can extract the worker. `None` only after
    /// `take()`, the `Deref` impl panics if accessed in this state. Holding
    /// the two together means a worker leaves this guard either back into
    /// the idle queue or retired from the group, never with its accounting
    /// left behind.
    held: Option<HeldWorker>,
    /// Back-reference to the group this worker belongs to, used by `Drop`
    /// to return the worker to the correct idle queue and semaphore.
    group: Arc<WorkerGroup>,
}

/// A worker out of its group's idle queue, with the guard counting it in
/// the group's `away` while it is out.
struct HeldWorker {
    handle: WorkerHandle,
    away: super::AwayFromQueue,
}

impl CheckedOutWorker {
    /// Check out `handle`, just popped from `group`'s idle queue (its
    /// permit already consumed): the only way to build this guard, so every
    /// checked-out worker is accounted for from the moment it leaves the
    /// queue.
    pub(super) fn from_idle(group: &Arc<WorkerGroup>, handle: WorkerHandle) -> Self {
        Self {
            held: Some(HeldWorker {
                handle,
                away: super::AwayFromQueue::new(group, 1),
            }),
            group: Arc::clone(group),
        }
    }

    /// Take the worker out of this guard (e.g., because it died).
    ///
    /// The taken worker will be dropped normally (triggering `WorkerHandle::Drop`
    /// which sends SIGTERM+SIGKILL). The slot is retired from the group:
    /// `total` is decremented and the global permit refunded, and no idle
    /// permit is released (the worker slot is gone).
    #[allow(dead_code)]
    pub fn take(&mut self) -> Option<WorkerHandle> {
        let HeldWorker { handle, away } = self.held.take()?;
        away.retire();
        Some(handle)
    }

    fn handle(&self) -> &WorkerHandle {
        // Caller-contract invariant (see the `Deref` impl): callers must not
        // dereference a guard after `.take()`. Reaching this expect
        // indicates a bug in the calling code, not a recoverable runtime
        // condition.
        #[allow(clippy::expect_used)]
        let held = self.held.as_ref().expect(
            "BUG: CheckedOutWorker dereferenced after take() -- \
             the worker handle has been consumed and is no longer available",
        );
        &held.handle
    }
}

/// # Panics
///
/// Panics if the handle has been removed via [`CheckedOutWorker::take()`].
/// This is a programming error -- callers must not dereference a guard
/// after taking the worker out. The `Deref` trait cannot return `Result`,
/// so a panic is the only signal available.
impl Deref for CheckedOutWorker {
    type Target = WorkerHandle;
    fn deref(&self) -> &WorkerHandle {
        // The `Deref` trait cannot return `Result`, so a panic is the only
        // signal available for the caller-contract violation.
        self.handle()
    }
}

/// # Panics
///
/// Panics if the handle has been removed via [`CheckedOutWorker::take()`].
/// See [`Deref`] impl for rationale.
impl DerefMut for CheckedOutWorker {
    fn deref_mut(&mut self) -> &mut WorkerHandle {
        // Same caller-contract invariant as `Deref::deref` above.
        #[allow(clippy::expect_used)]
        let held = self.held.as_mut().expect(
            "BUG: CheckedOutWorker dereferenced after take() -- \
             the worker handle has been consumed and is no longer available",
        );
        &mut held.handle
    }
}

impl Drop for CheckedOutWorker {
    fn drop(&mut self) {
        if let Some(HeldWorker { handle, away }) = self.held.take() {
            match handle.request_flight() {
                RequestFlight::Idle => {
                    // Return the worker to the idle queue and release a
                    // permit; only then does its guard stop counting it.
                    super::lock_recovered(&self.group.idle).push_back(handle);
                    self.group.available.add_permits(1);
                    drop(away);
                }
                RequestFlight::InFlight => {
                    // A request was written and its response was never
                    // fully read (typically: this dispatch's future was
                    // dropped by a cancellation racing the attempt in
                    // `infer_retry::dispatch_execute_v2_with_retry_and_progress`).
                    // The worker's stdout may still hold bytes belonging
                    // to that unread response, or the worker may still be
                    // about to emit one. Returning it to idle would let
                    // the NEXT dispatch on this worker read that stale
                    // response and misattribute it to an unrelated
                    // request. Discard exactly like `dispatch_batch_infer`
                    // already does for an Io/Protocol-broken worker: the
                    // slot is retired through its guard, and `handle`
                    // drops here, triggering `WorkerHandle::Drop`'s
                    // SIGTERM+SIGKILL teardown of the process. Its late
                    // response, if any, can never reach a later request.
                    warn!(
                        pid = %handle.pid(),
                        "discarding worker with a cancelled in-flight request \
                         rather than returning it to the idle queue"
                    );
                    away.retire();
                }
            }
        }
        // If the worker was taken via `take()`, its slot was already
        // retired: nothing to do.

        // Wake ONE task parked on `WorkerPool::worker_returned`
        // typically a cross-key spawn attempt waiting for an eviction
        // opportunity. FIFO-fair: each worker return wakes exactly
        // one waiter, eliminating the thundering-herd re-probe storm
        // documented in BUG-028. If the woken waiter's key turns out
        // to be uneviable for this particular return, it re-parks on
        // the same Notify and the next return wakes the next-in-line
        // waiter. The waiter re-probes at least once per report
        // interval as well (see `checkout_wait`), so a lost wakeup costs
        // one interval, never the request.
        self.group.worker_returned.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use tokio::sync::Semaphore;

    use crate::api::LanguageCode3;
    use crate::types::worker_v2::{
        AsrBackendV2, AsrInputV2, AsrRequestV2, ExecuteRequestV2, InferenceTaskV2,
        PreparedAudioInputV2, TaskRequestV2, WorkerArtifactIdV2, WorkerRequestIdV2,
    };
    use crate::worker::WorkerProfile;
    use crate::worker::handle::WorkerHandle;
    use crate::worker::pool::{EngineSelection, lock_recovered};

    use super::*;

    /// A minimal, well-formed `ready` line, the same shape
    /// `worker::handle::lifecycle`'s own tests use, PID hardcoded since
    /// nothing here asserts on it.
    fn ready_line() -> String {
        format!(
            concat!(
                "{{\"ready\":true,\"pid\":424242,\"transport\":\"stdio\",\"runtime\":{{",
                "\"schema_version\":1,\"python_version\":\"3.13.12\",",
                "\"python_executable_sha256\":\"{}\",",
                "\"batchalign_package_tree_sha256\":\"{}\",",
                "\"batchalign_core_extension_sha256\":\"{}\",",
                "\"distribution_inventory_sha256\":\"{}\"}}}}\n"
            ),
            "a".repeat(64),
            "b".repeat(64),
            "c".repeat(64),
            "d".repeat(64),
        )
    }

    /// Shell one-liner for a stub worker: print the ready line, then never
    /// respond to anything written to stdin. `execute_v2_with_progress`'s
    /// write therefore always succeeds (the OS pipe buffer absorbs the
    /// small request) while its read never completes, exactly the shape
    /// of a real dispatch cancelled mid-attempt.
    fn never_responds_script() -> String {
        let escaped = ready_line().replace('\'', "'\\''");
        format!("printf '{escaped}'; sleep 300")
    }

    fn minimal_asr_request() -> ExecuteRequestV2 {
        ExecuteRequestV2 {
            request_id: WorkerRequestIdV2::from("checkout-drop-test"),
            task: InferenceTaskV2::Asr,
            payload: TaskRequestV2::Asr(AsrRequestV2 {
                lang: crate::api::WorkerLanguage::from(LanguageCode3::eng()),
                backend: AsrBackendV2::LocalWhisper,
                models: crate::types::worker_v2::AsrRequestedModelsV2::Whisper {
                    asr: crate::types::worker_v2::RequestedModelV2 {
                        id: crate::types::worker_v2::ModelIdV2::from_static(
                            "openai/whisper-large-v3",
                        ),
                        revision: crate::types::worker_v2::RequestedRevisionV2::Unpinned,
                    },
                },
                input: AsrInputV2::PreparedAudio(PreparedAudioInputV2 {
                    audio_ref_id: WorkerArtifactIdV2::from("audio-1"),
                }),
                extras: std::collections::BTreeMap::new(),
                decode_budget_seconds: None,
            }),
            attachments: Vec::new(),
        }
    }

    fn make_group() -> Arc<super::super::WorkerGroup> {
        Arc::new(super::super::WorkerGroup::new(
            Arc::new(tokio::sync::Notify::new()),
            Arc::new(Semaphore::new(1)),
            WorkerProfile::Stanza,
            EngineSelection::none(),
        ))
    }

    /// SERIOUS: the bug this whole change exists to close. A request is
    /// written to a stub worker (so it is genuinely `InFlight`, not merely
    /// checked out), the in-flight `execute_v2_with_progress` future is
    /// dropped mid-wait, mirroring exactly what `select!` does in
    /// `infer_retry::dispatch_execute_v2_with_retry_and_progress` when a
    /// job is cancelled mid-attempt, and then the `CheckedOutWorker` guard
    /// itself is dropped. The worker must NOT be back in the idle pool:
    /// its stdout may still hold (or later receive) a response nobody is
    /// listening for, and the next checkout must not risk reading it.
    #[tokio::test]
    async fn dropping_an_in_flight_checkout_discards_rather_than_requeues() {
        let handle = WorkerHandle::spawn_stub_for_test(&never_responds_script()).await;
        let group = make_group();
        group.total.fetch_add(1, Ordering::Relaxed);

        let mut worker = CheckedOutWorker::from_idle(&group, handle);

        let request = minimal_asr_request();
        {
            let mut attempt = Box::pin(worker.execute_v2_with_progress(&request, None));
            // Deterministic gate, not a sleep-based race: a single `poll`
            // either returns the attempt's outcome or `Pending`. The stub
            // never responds, so `Pending` here means the write already
            // went through (it happens synchronously, before the first
            // await point inside `write_request`) and the future is now
            // genuinely parked waiting on a response that will never
            // arrive -- proven by the poll itself, not by elapsed time.
            let first_poll = futures::poll!(&mut attempt);
            assert!(
                matches!(first_poll, std::task::Poll::Pending),
                "stub must never respond on the very first poll, got {first_poll:?}"
            );
            // `attempt` (and the mutable borrow of `worker` it holds) is
            // dropped here at scope exit, exactly what `select!` does to
            // the losing branch's future in the real retry loop.
        }
        assert_eq!(
            worker
                .held
                .as_ref()
                .map(|held| held.handle.request_flight()),
            Some(RequestFlight::InFlight),
            "the write must have gone through (request genuinely in flight) for this test to \
             prove anything"
        );

        drop(worker);

        let idle_len = lock_recovered(&group.idle).len();
        assert_eq!(
            idle_len, 0,
            "a worker dropped mid-in-flight-request must be discarded, not returned to idle"
        );
        assert_eq!(
            group.total.load(Ordering::Relaxed),
            0,
            "the discarded slot must be freed from `total`, matching take()'s contract"
        );
    }

    /// Companion positive case: a checkout dropped while genuinely Idle
    /// (the ordinary, uncancelled path) is still returned to idle.
    #[tokio::test]
    async fn dropping_an_idle_checkout_returns_it_to_idle() {
        let handle = WorkerHandle::spawn_stub_for_test(&never_responds_script()).await;
        let group = make_group();
        group.total.fetch_add(1, Ordering::Relaxed);

        let worker = CheckedOutWorker::from_idle(&group, handle);
        assert_eq!(
            worker
                .held
                .as_ref()
                .map(|held| held.handle.request_flight()),
            Some(RequestFlight::Idle)
        );

        drop(worker);

        assert_eq!(lock_recovered(&group.idle).len(), 1);
        assert_eq!(group.total.load(Ordering::Relaxed), 1);
    }

    /// A taken worker is retired with its accounting at once: while the
    /// emptied guard is still alive it counts nothing, so it cannot mask a
    /// worker lost elsewhere in the group.
    #[tokio::test]
    async fn a_taken_worker_leaves_no_guard_counting_it() {
        let handle = WorkerHandle::spawn_stub_for_test(&never_responds_script()).await;
        let group = make_group();
        group.total.fetch_add(1, Ordering::Relaxed);
        let mut worker = CheckedOutWorker::from_idle(&group, handle);
        assert_eq!(group.away.load(Ordering::Relaxed), 1);

        let taken = worker.take();
        assert!(taken.is_some());
        assert_eq!(group.total.load(Ordering::Relaxed), 0);
        assert_eq!(
            group.away.load(Ordering::Relaxed),
            0,
            "the guard must not outlive the slot it accounted for"
        );
        // A worker lost from the same group is now visible, not masked.
        group.total.fetch_add(1, Ordering::Relaxed);
        assert_eq!(group.unaccounted(), 1);
        drop(worker);
        assert_eq!(group.unaccounted(), 1);
    }
}
