//! Exclusive exchanges consume their original departed worker exactly once.
use super::capacity::DepartedWorker;
use crate::worker::error::{WorkerAfterFailure, WorkerError};
use crate::worker::handle::{RequestFlight, WorkerHandle};
use std::ops::{Deref, DerefMut};
#[cfg(test)]
use std::sync::Arc;

pub(super) struct TrackedCheckout {
    worker: CheckedOutWorker,
    _registration: super::job_tracker::TrackerGuard,
}
impl TrackedCheckout {
    pub(super) fn from_departed(
        worker: DepartedWorker<WorkerHandle>,
        tracker: &super::job_tracker::JobWorkerTracker,
    ) -> Self {
        let registration = super::job_tracker::TrackerGuard::new(tracker, worker.pid());
        Self {
            worker: CheckedOutWorker::from_departed(worker),
            _registration: registration,
        }
    }
    pub(super) async fn run<T>(
        mut self,
        work: impl AsyncFnOnce(&mut WorkerHandle) -> Result<T, WorkerError>,
    ) -> Result<T, WorkerError> {
        let outcome = work(&mut self.worker).await;
        self.worker.finish(&outcome);
        outcome
    }
}

/// An owned exchange. Completion consumes return authority. Dropping unfinished
/// work retires its actual worker and original capacity, including cancellation.
pub struct CheckedOutWorker {
    worker: DepartedWorker<WorkerHandle>,
}
impl CheckedOutWorker {
    fn from_departed(worker: DepartedWorker<WorkerHandle>) -> Self {
        Self { worker }
    }
    fn finish<T>(self, outcome: &Result<T, WorkerError>) {
        let disposition = match self.worker.request_flight() {
            RequestFlight::InFlight => WorkerAfterFailure::Retire,
            RequestFlight::Idle => outcome
                .as_ref()
                .map(|_| WorkerAfterFailure::Reusable)
                .unwrap_or_else(|error| error.worker_after_failure()),
        };
        match disposition {
            WorkerAfterFailure::Reusable => self.worker.return_idle(),
            WorkerAfterFailure::Retire => drop(self),
        }
    }
}
impl Deref for CheckedOutWorker {
    type Target = WorkerHandle;
    fn deref(&self) -> &Self::Target {
        &self.worker
    }
}
impl DerefMut for CheckedOutWorker {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.worker
    }
}

#[cfg(test)]
pub(super) mod tests {

    use crate::api::LanguageCode3;
    use crate::types::worker_v2::{
        AsrBackendV2, AsrInputV2, AsrRequestV2, ExecuteRequestV2, InferenceTaskV2,
        PreparedAudioInputV2, TaskRequestV2, WorkerArtifactIdV2, WorkerRequestIdV2,
    };
    use crate::worker::WorkerProfile;
    use crate::worker::handle::WorkerHandle;
    use crate::worker::pool::EngineSelection;

    use super::*;

    use crate::worker::handle::test_support::ready_line;

    /// Shell one-liner for a stub worker: print the ready line, then never
    /// respond to anything written to stdin. `execute_v2_with_progress`'s
    /// write therefore always succeeds (the OS pipe buffer absorbs the
    /// small request) while its read never completes, exactly the shape
    /// of a real dispatch cancelled mid-attempt.
    pub(in crate::worker::pool) fn never_responds_script() -> String {
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
            super::super::capacity::GlobalCapacity::new(1),
            WorkerProfile::Stanza,
            EngineSelection::none(),
        ))
    }

    async fn admitted_checkout() -> (Arc<super::super::WorkerGroup>, CheckedOutWorker) {
        let group = make_group();
        let reservation = super::super::capacity::ReservedWorkerSlot::try_claim(&group, 1)
            .expect("admit the stub's actual slot");
        let handle = WorkerHandle::spawn_stub_for_test(&never_responds_script()).await;
        reservation.publish_stdio(handle);
        let worker = group.take_stdio().expect("published worker");
        let checkout = CheckedOutWorker::from_departed(worker);
        (group, checkout)
    }

    /// Guards: Cancelling a real pending pipe exchange discards its original worker and capacity, preventing unread response bytes from reaching a later checkout.
    /// Kind: contract
    /// Not a type: Consuming retirement prevents reuse of an emptied checkout. Whether bytes were written and a response remains unread is actual transport state, so the real in-flight adapter and its Drop disposition must still agree.
    /// State: A private shell stub emits admitted ready bytes and never answers the request. The real pipe establishes an in-flight request. Typed slot publication replaces invented counts; the shared declared-effect producer owns the admitted request and model lease without loading a model.
    /// Concurrency: The fixture owns its subprocess and pool. One explicit poll establishes Pending without elapsed-time assumptions. WorkerHandle owns signal teardown on cancellation and assertion failure.
    #[tokio::test]
    async fn dropping_an_in_flight_checkout_discards_rather_than_requeues() {
        let (group, mut worker) = admitted_checkout().await;

        let request = minimal_asr_request();
        let (_ledger, admission) = super::super::task_load::tests::test_admission();
        let permit = super::super::task_load::tests::declared_invocation(
            super::super::task_load::tests::ready(&request),
            &admission,
        )
        .await;
        {
            let mut attempt = Box::pin(worker.execute_admitted(permit, None));
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
            worker.request_flight(),
            RequestFlight::InFlight,
            "the write must have gone through (request genuinely in flight) for this test to \
             prove anything"
        );

        drop(worker);

        let idle_len = group.idle.view().len();
        assert_eq!(
            idle_len, 0,
            "a worker dropped mid-in-flight-request must be discarded, not returned to idle"
        );
        assert_eq!(
            group.total.count(),
            0,
            "the discarded slot must be freed from `total`, matching consuming retirement"
        );
    }

    /// Guards: Completing an actual health exchange returns the same worker to its queue while retaining its original capacity slot.
    /// Kind: contract
    /// Not a type: Unique slot ownership excludes detached refunds, but the completion adapter must still publish the real idle handle to its original queue. This is the positive adapter branch paired with cancelled in-flight retirement.
    /// State: The admitted fixture owns one shell subprocess and actual ready/health pipe bytes. It enters through the capacity publication producer. No model, socket or inference service is needed.
    /// Concurrency: Each test owns its group and subprocess. Explicit return is synchronous, and the final group/handle drop owns process cleanup without shared ports or timing waits.
    #[tokio::test]
    async fn a_completed_exchange_returns_to_its_original_queue() {
        let ready = ready_line();
        let response = r#"{"op":"health","response":{"status":"ok","command":"infer:morphosyntax","lang":"eng","pid":4242,"uptime_s":0.0}}"#;
        let script = format!(
            "printf '%s\\n' '{}'; read request; printf '%s\\n' '{}'; sleep 300",
            ready.trim(),
            response
        );
        let group = make_group();
        let worker = WorkerHandle::spawn_stub_for_test(&script).await;
        super::super::capacity::ReservedWorkerSlot::try_claim(&group, 1)
            .expect("admit actual process capacity")
            .publish_stdio(worker);
        let tracker = super::super::job_tracker::JobWorkerTracker::new();
        let checkout = TrackedCheckout::from_departed(group.take_stdio().unwrap(), &tracker);
        checkout
            .run(async |worker| worker.health_check().await)
            .await
            .unwrap();

        assert_eq!(group.idle.view().len(), 1);
        assert_eq!(group.total.count(), 1);
    }
}
