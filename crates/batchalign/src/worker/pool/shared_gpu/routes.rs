//! Where a shared GPU worker's replies go: the V2 dispatches waiting by
//! request id, and the one sequential control op waiting in its slot.
//!
//! Both ends are state machines that CLOSE when the worker's output stream
//! ends. Closing answers every waiter with the reason, through its own reply,
//! and refuses every later registration at once. So a dispatch can never wait
//! out its timeout on a stream that has already ended, and liveness is one
//! observation ([`PendingDispatches::liveness`]) for both transports.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::oneshot;

use crate::types::worker_v2::{ExecuteResponseV2, WorkerRequestIdV2};
use crate::worker::InferTask;
use crate::worker::error::WorkerError;
use crate::worker::handle::{ControlRequestId, NoiseLimit, ReportedFailure};

use super::super::lock_recovered;
use super::EnsureTaskResponse;

/// Why a worker's output stream stopped being read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StreamClosed {
    /// The worker closed its output (EOF): the process exited or the daemon
    /// dropped the connection.
    Eof,
    /// Reading the worker's output failed.
    ReadFailed(String),
    /// The worker wrote `MAX_RESPONSE_STDOUT_NOISE_LINES` consecutive lines that are not protocol messages; the stream is not
    /// trusted further.
    Noise(NoiseLimit),
    /// The reader task ended without reaching EOF or a read error (it was
    /// aborted, or it panicked).
    ReaderStopped,
    /// The pool stopped this worker, for the reason given.
    Stopped(Retirement),
}

/// Why the pool stops a shared worker, which decides what its waiters hear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Retirement {
    /// The pool is shutting down: nothing will serve the request
    /// ([`WorkerError::PoolShuttingDown`], not retried).
    PoolShutdown,
    /// The pool retired this one worker (its capability report was refused,
    /// or it is being replaced) and keeps serving, so a retry reaches another
    /// worker ([`WorkerError::WorkerRetired`], retried).
    WorkerRetired,
}

impl Retirement {
    /// The error a request refused or stranded by this stop receives.
    pub(crate) fn to_worker_error(self) -> WorkerError {
        match self {
            Self::PoolShutdown => WorkerError::PoolShuttingDown,
            Self::WorkerRetired => WorkerError::WorkerRetired,
        }
    }
}

impl StreamClosed {
    /// The error every waiter on a closed stream receives.
    pub(crate) fn to_worker_error(&self) -> WorkerError {
        let lost = |detail: String| WorkerError::ProcessExited {
            code: None,
            stderr: Some(detail),
        };
        match self {
            Self::Eof => lost("GPU worker closed its output stream (EOF)".into()),
            Self::ReadFailed(error) => lost(format!("reading GPU worker output failed: {error}")),
            Self::ReaderStopped => lost("GPU worker response reader stopped".into()),
            Self::Noise(limit) => limit.clone().into_worker_error(),
            Self::Stopped(why) => why.to_worker_error(),
        }
    }
}

/// Why a waiter got no response of its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReplyFailure {
    /// The worker reported a failure in an `op=error` line.
    Reported(ReportedFailure),
    /// The worker sent a line the protocol refuses: an envelope that does not
    /// decode, or an op this protocol does not name.
    Refused(String),
    /// The stream ended before an answer arrived.
    StreamClosed(StreamClosed),
}

impl ReplyFailure {
    /// The waiter's error. `reported` is the op's own reading of a reported
    /// failure (one of `ReportedFailure`'s methods); a refused line and a
    /// closed stream mean the same thing to every op.
    pub(crate) fn into_worker_error(
        self,
        reported: fn(ReportedFailure) -> WorkerError,
    ) -> WorkerError {
        match self {
            Self::Reported(failure) => reported(failure),
            Self::Refused(detail) => WorkerError::Protocol(detail),
            Self::StreamClosed(reason) => reason.to_worker_error(),
        }
    }

    /// The text, for logs.
    pub(crate) fn message(&self) -> String {
        match self {
            Self::Reported(failure) => failure.message.clone(),
            Self::Refused(detail) => detail.clone(),
            Self::StreamClosed(reason) => reason.to_worker_error().to_string(),
        }
    }
}

/// What a waiting V2 dispatch is answered with: the worker's own response
/// (success, or a failure the worker reported inside an `execute_v2` line),
/// or the error the sequential path returns for the same line.
pub(crate) type DispatchReply = Result<ExecuteResponseV2, WorkerError>;

/// The V2 dispatches waiting for a reply, keyed by the typed request id.
#[derive(Clone, Default)]
pub(crate) struct PendingDispatches(Arc<Mutex<PendingState>>);

enum PendingState {
    Open(HashMap<WorkerRequestIdV2, oneshot::Sender<DispatchReply>>),
    Closed(StreamClosed),
}

impl Default for PendingState {
    fn default() -> Self {
        Self::Open(HashMap::new())
    }
}

/// A registered dispatch's claim on its reply.
pub(crate) struct DispatchReceipt {
    request_id: WorkerRequestIdV2,
    reply: oneshot::Receiver<DispatchReply>,
    pending: PendingDispatches,
}

impl DispatchReceipt {
    /// Wait for the reply. `None` when the time limit passed first; the
    /// registration is withdrawn then, so a late reply is logged as orphaned
    /// rather than delivered to nobody.
    pub(crate) async fn reply_within(self, limit: std::time::Duration) -> Option<DispatchReply> {
        match tokio::time::timeout(limit, self.reply).await {
            Ok(Ok(reply)) => Some(reply),
            // Every route out of the pending map answers its waiter (a reply,
            // a routed failure, or the close reason), so a dropped sender
            // means the map itself is gone with the reader that owned it.
            Ok(Err(_)) => Some(Err(StreamClosed::ReaderStopped.to_worker_error())),
            Err(_) => {
                self.pending.withdraw(&self.request_id);
                None
            }
        }
    }
}

impl PendingDispatches {
    /// Register a dispatch before its request is written, so the reader can
    /// route the reply as soon as it arrives. Refused at once when the stream
    /// has closed, and refused for an id already waiting (a second sender
    /// would silently orphan the first).
    pub(crate) fn register(
        &self,
        request_id: WorkerRequestIdV2,
    ) -> Result<DispatchReceipt, WorkerError> {
        let (tx, rx) = oneshot::channel();
        match &mut *lock_recovered(&self.0) {
            PendingState::Closed(reason) => return Err(reason.to_worker_error()),
            PendingState::Open(waiting) => {
                if waiting.contains_key(&request_id) {
                    return Err(WorkerError::Protocol(format!(
                        "request id {request_id} is already waiting for a GPU worker reply"
                    )));
                }
                waiting.insert(request_id.clone(), tx);
            }
        }
        Ok(DispatchReceipt {
            request_id,
            reply: rx,
            pending: self.clone(),
        })
    }

    /// Withdraw a registration whose request was never written, or whose
    /// caller stopped waiting.
    pub(crate) fn withdraw(&self, request_id: &WorkerRequestIdV2) {
        if let PendingState::Open(waiting) = &mut *lock_recovered(&self.0) {
            waiting.remove(request_id);
        }
    }

    /// Answer the dispatch `request_id`; hand the reply back when nobody is
    /// waiting for that id.
    pub(crate) fn answer(
        &self,
        request_id: &str,
        reply: DispatchReply,
    ) -> Result<(), DispatchReply> {
        let waiter = match &mut *lock_recovered(&self.0) {
            PendingState::Open(waiting) => waiting.remove(request_id),
            PendingState::Closed(_) => None,
        };
        match waiter {
            Some(tx) => {
                // A receiver dropped since its registration is a caller that
                // stopped waiting; nothing is owed to it.
                let _ = tx.send(reply);
                Ok(())
            }
            None => Err(reply),
        }
    }

    /// Close: answer every waiter with `reason` and refuse every later
    /// registration. The first reason stands; returns whether this call
    /// closed the map.
    pub(crate) fn close(&self, reason: StreamClosed) -> bool {
        let stranded = {
            let mut state = lock_recovered(&self.0);
            match std::mem::replace(&mut *state, PendingState::Closed(reason.clone())) {
                PendingState::Open(waiting) => waiting,
                previous @ PendingState::Closed(_) => {
                    *state = previous;
                    return false;
                }
            }
        };
        for (_, tx) in stranded {
            let _ = tx.send(Err(reason.to_worker_error()));
        }
        true
    }

    /// Whether the worker can still take requests: `Ok` while the stream is
    /// open, the close reason's error after.
    pub(crate) fn liveness(&self) -> Result<(), WorkerError> {
        match &*lock_recovered(&self.0) {
            PendingState::Open(_) => Ok(()),
            PendingState::Closed(reason) => Err(reason.to_worker_error()),
        }
    }

    /// How many dispatches are waiting (tests and logs).
    #[cfg(test)]
    pub(crate) fn waiting(&self) -> usize {
        match &*lock_recovered(&self.0) {
            PendingState::Open(waiting) => waiting.len(),
            PendingState::Closed(_) => 0,
        }
    }
}

/// A sequential op's decoded reply, as the reader hands it to the slot.
#[derive(Debug)]
pub(crate) enum ControlReply {
    Capabilities(crate::worker::WorkerCapabilities),
    EnsureTask(EnsureTaskResponse),
    Shutdown,
}

/// Which sequential op a reply line's `op` names, for a line that does not
/// decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ControlOpName {
    Capabilities,
    EnsureTask,
}

impl ControlOpName {
    /// The op's name on the wire.
    pub(crate) fn wire_name(self) -> &'static str {
        match self {
            Self::Capabilities => "capabilities",
            Self::EnsureTask => "ensure_task",
        }
    }
}

/// One line the reader offers the control slot.
#[derive(Debug)]
pub(crate) enum ControlLine {
    /// A decoded reply.
    Reply(ControlReply),
    /// A reply line of this op that does not decode.
    Undecodable { op: ControlOpName, detail: String },
    /// A failure that names no owner: an `op=error` line without a
    /// `request_id`, a `health` reply, or an op the protocol does not name.
    /// Every control request carries an id, so such a line is a protocol
    /// fault; it answers whichever op waits rather than leaving it on its
    /// timeout.
    Unowned(ReplyFailure),
}

/// What a waiting op receives: its own reply, or why none came.
pub(crate) type ControlAnswer<T> = Result<T, ReplyFailure>;

/// An armed op: the id its request must carry, and where its answer arrives.
pub(crate) struct ArmedControl<T> {
    pub(crate) request_id: ControlRequestId,
    pub(crate) answer: oneshot::Receiver<ControlAnswer<T>>,
}

/// The op waiting in the slot. Each variant holds a sender for its own reply
/// type, so a reply of one op can never be delivered to another, and a
/// waiter knows what it asked for: the id its request carried and, for
/// `ensure_task`, the task.
enum ControlWaiter {
    Capabilities {
        request_id: ControlRequestId,
        reply: oneshot::Sender<ControlAnswer<crate::worker::WorkerCapabilities>>,
    },
    EnsureTask {
        request_id: ControlRequestId,
        task: InferTask,
        reply: oneshot::Sender<ControlAnswer<EnsureTaskResponse>>,
    },
}

impl ControlWaiter {
    fn request_id(&self) -> &ControlRequestId {
        match self {
            Self::Capabilities { request_id, .. } | Self::EnsureTask { request_id, .. } => {
                request_id
            }
        }
    }

    fn op(&self) -> ControlOpName {
        match self {
            Self::Capabilities { .. } => ControlOpName::Capabilities,
            Self::EnsureTask { .. } => ControlOpName::EnsureTask,
        }
    }

    /// Answer with a failure. A receiver gone since arming is an op that
    /// stopped waiting; nothing is owed to it.
    fn fail(self, failure: ReplyFailure) {
        match self {
            Self::Capabilities { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::EnsureTask { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
        }
    }

    /// Take `line` if it answers this op; hand both back if it does not.
    ///
    /// A reply answers the op it names (and, for `ensure_task`, the task it
    /// names); a tagged failure answers the op whose id it carries. Anything
    /// else is a late answer to an op that already timed out, or a reply
    /// for another op, and leaves this op waiting for its own.
    fn offer(self, line: ControlLine) -> Result<(), Box<(Self, ControlLine)>> {
        match (self, line) {
            (
                Self::Capabilities { reply, .. },
                ControlLine::Reply(ControlReply::Capabilities(response)),
            ) => {
                let _ = reply.send(Ok(response));
                Ok(())
            }
            (
                Self::EnsureTask { task, reply, .. },
                ControlLine::Reply(ControlReply::EnsureTask(response)),
            ) if response.task == task => {
                let _ = reply.send(Ok(response));
                Ok(())
            }
            (waiter, ControlLine::Undecodable { op, detail }) if op == waiter.op() => {
                waiter.fail(ReplyFailure::Refused(detail));
                Ok(())
            }
            (waiter, ControlLine::Unowned(failure)) => {
                waiter.fail(failure);
                Ok(())
            }
            (waiter, unclaimed) => Err(Box::new((waiter, unclaimed))),
        }
    }
}

/// The single slot a sequential op (capabilities, ensure_task, shutdown)
/// waits in. The ops are serialized by the worker's control gate, so at most
/// one live waiter exists; shutdown, which does not take the gate, supersedes
/// any waiter with [`StreamClosed::Stopped`].
///
/// A waiter is answered only by a line that names it (see
/// [`ControlWaiter::offer`]), so an op that timed out cannot have its late
/// reply, or its late failure, taken as the answer to the next op.
#[derive(Clone, Default)]
pub(crate) struct ControlSlot(Arc<Mutex<ControlState>>);

#[derive(Default)]
enum ControlState {
    #[default]
    Idle,
    Waiting(ControlWaiter),
    /// Shutdown has begun: only the shutdown acknowledgement is awaited.
    ShuttingDown {
        ack: oneshot::Sender<ControlAnswer<()>>,
        why: Retirement,
    },
    Closed(StreamClosed),
}

impl ControlSlot {
    /// Arm the slot with `waiter`. A waiter left by an op that timed out is
    /// replaced (its receiver is gone). Refused once shutdown has begun or
    /// the stream has closed.
    fn arm(&self, waiter: ControlWaiter) -> Result<(), WorkerError> {
        let mut state = lock_recovered(&self.0);
        match &*state {
            ControlState::Idle | ControlState::Waiting(_) => {
                *state = ControlState::Waiting(waiter);
                Ok(())
            }
            ControlState::ShuttingDown { why, .. } => Err(why.to_worker_error()),
            ControlState::Closed(reason) => Err(reason.to_worker_error()),
        }
    }

    /// Arm the slot for a `capabilities` op under a fresh request id.
    pub(crate) fn arm_capabilities(
        &self,
    ) -> Result<ArmedControl<crate::worker::WorkerCapabilities>, WorkerError> {
        let (reply, answer) = oneshot::channel();
        let request_id = ControlRequestId::next();
        self.arm(ControlWaiter::Capabilities {
            request_id: request_id.clone(),
            reply,
        })?;
        Ok(ArmedControl { request_id, answer })
    }

    /// Arm the slot for an `ensure_task(task)` op under a fresh request id.
    pub(crate) fn arm_ensure_task(
        &self,
        task: InferTask,
    ) -> Result<ArmedControl<EnsureTaskResponse>, WorkerError> {
        let (reply, answer) = oneshot::channel();
        let request_id = ControlRequestId::next();
        self.arm(ControlWaiter::EnsureTask {
            request_id: request_id.clone(),
            task,
            reply,
        })?;
        Ok(ArmedControl { request_id, answer })
    }

    /// Begin shutdown for `why`: a live waiter is answered with
    /// [`StreamClosed::Stopped`] and the slot waits only for the shutdown
    /// acknowledgement. `None` when shutdown already began or the stream
    /// closed.
    pub(crate) fn begin_shutdown(
        &self,
        why: Retirement,
    ) -> Option<oneshot::Receiver<ControlAnswer<()>>> {
        let (ack, rx) = oneshot::channel();
        let mut state = lock_recovered(&self.0);
        match std::mem::replace(&mut *state, ControlState::ShuttingDown { ack, why }) {
            ControlState::Idle => Some(rx),
            ControlState::Waiting(superseded) => {
                superseded.fail(ReplyFailure::StreamClosed(StreamClosed::Stopped(why)));
                Some(rx)
            }
            previous @ (ControlState::ShuttingDown { .. } | ControlState::Closed(_)) => {
                *state = previous;
                None
            }
        }
    }

    /// Offer a line to the waiting op; hand it back when no op it names is
    /// waiting.
    pub(crate) fn deliver(&self, line: ControlLine) -> Result<(), ControlLine> {
        let mut state = lock_recovered(&self.0);
        match std::mem::take(&mut *state) {
            ControlState::Waiting(waiter) => match waiter.offer(line) {
                Ok(()) => Ok(()),
                Err(unclaimed) => {
                    let (waiter, unclaimed) = *unclaimed;
                    *state = ControlState::Waiting(waiter);
                    Err(unclaimed)
                }
            },
            ControlState::ShuttingDown { ack, why } => match line {
                ControlLine::Reply(ControlReply::Shutdown) => {
                    *state = ControlState::Closed(StreamClosed::Stopped(why));
                    let _ = ack.send(Ok(()));
                    Ok(())
                }
                ControlLine::Unowned(failure) => {
                    *state = ControlState::Closed(StreamClosed::Stopped(why));
                    let _ = ack.send(Err(failure));
                    Ok(())
                }
                unclaimed @ (ControlLine::Reply(
                    ControlReply::Capabilities(_) | ControlReply::EnsureTask(_),
                )
                | ControlLine::Undecodable { .. }) => {
                    *state = ControlState::ShuttingDown { ack, why };
                    Err(unclaimed)
                }
            },
            previous @ (ControlState::Idle | ControlState::Closed(_)) => {
                *state = previous;
                Err(line)
            }
        }
    }

    /// Offer a failure tagged `request_id` to the waiting op; hand it back
    /// when the waiting op's request did not carry that id (the reader then
    /// routes it to the dispatch of that id).
    pub(crate) fn deliver_tagged_failure(
        &self,
        request_id: &str,
        failure: ReplyFailure,
    ) -> Result<(), ReplyFailure> {
        let mut state = lock_recovered(&self.0);
        match std::mem::take(&mut *state) {
            ControlState::Waiting(waiter) if waiter.request_id().names(request_id) => {
                waiter.fail(failure);
                Ok(())
            }
            previous => {
                *state = previous;
                Err(failure)
            }
        }
    }

    /// Close: answer any waiter with `reason` and refuse every later op. The
    /// first reason stands.
    pub(crate) fn close(&self, reason: StreamClosed) {
        let mut state = lock_recovered(&self.0);
        match std::mem::replace(&mut *state, ControlState::Closed(reason.clone())) {
            ControlState::Waiting(waiter) => waiter.fail(ReplyFailure::StreamClosed(reason)),
            ControlState::ShuttingDown { ack, .. } => {
                let _ = ack.send(Err(ReplyFailure::StreamClosed(reason)));
            }
            ControlState::Idle => {}
            previous @ ControlState::Closed(_) => *state = previous,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Closing answers every waiter through its own reply and refuses every
    /// later registration at once, instead of leaving it on its timeout.
    #[tokio::test]
    async fn closing_answers_waiters_and_refuses_new_registrations() {
        let pending = PendingDispatches::default();
        let receipt = pending
            .register(WorkerRequestIdV2::from("r-1"))
            .expect("an open map registers");
        assert!(pending.close(StreamClosed::Eof));
        assert!(
            !pending.close(StreamClosed::Stopped(Retirement::PoolShutdown)),
            "the first reason stands"
        );

        let reply = receipt
            .reply_within(std::time::Duration::from_secs(5))
            .await
            .expect("answered, not timed out");
        assert!(
            matches!(reply, Err(WorkerError::ProcessExited { .. })),
            "{reply:?}"
        );

        assert!(matches!(
            pending.register(WorkerRequestIdV2::from("r-2")),
            Err(WorkerError::ProcessExited { .. })
        ));
        assert!(pending.liveness().is_err());
    }

    /// A dispatch in flight on a worker the pool retires (a refused
    /// capability report, a replaced worker) while the pool keeps running is
    /// answered with a retryable error: the work was not refused, its worker
    /// went away.
    #[tokio::test]
    async fn retiring_a_live_worker_answers_its_dispatches_retryably() {
        use crate::runner::util::{classify_worker_error, is_retryable_worker_failure};
        let pending = PendingDispatches::default();
        let receipt = pending
            .register(WorkerRequestIdV2::from("r-1"))
            .expect("an open map registers");
        pending.close(StreamClosed::Stopped(Retirement::WorkerRetired));
        let error = receipt
            .reply_within(std::time::Duration::from_secs(5))
            .await
            .expect("answered")
            .expect_err("a retired worker answers no response");
        assert!(
            is_retryable_worker_failure(classify_worker_error(&error)),
            "{error:?}"
        );
    }

    /// A second registration under an id already waiting is refused rather
    /// than orphaning the first waiter.
    #[test]
    fn a_duplicate_request_id_is_refused() {
        let pending = PendingDispatches::default();
        let _first = pending
            .register(WorkerRequestIdV2::from("r-1"))
            .expect("first registration");
        assert!(matches!(
            pending.register(WorkerRequestIdV2::from("r-1")),
            Err(WorkerError::Protocol(_))
        ));
    }

    /// Shutdown supersedes a live control waiter, receives the ack itself,
    /// and refuses later ops.
    #[tokio::test]
    async fn shutdown_supersedes_the_control_waiter_and_closes_after_its_ack() {
        let slot = ControlSlot::default();
        let waiting = slot.arm_capabilities().expect("an idle slot arms");
        let ack = slot
            .begin_shutdown(Retirement::PoolShutdown)
            .expect("first shutdown");
        assert!(
            slot.begin_shutdown(Retirement::PoolShutdown).is_none(),
            "shutdown begins once"
        );
        assert!(matches!(
            waiting.answer.await.expect("answered"),
            Err(ReplyFailure::StreamClosed(StreamClosed::Stopped(
                Retirement::PoolShutdown
            )))
        ));
        assert!(matches!(
            slot.arm_capabilities(),
            Err(WorkerError::PoolShuttingDown)
        ));

        slot.deliver(ControlLine::Reply(ControlReply::Shutdown))
            .expect("the ack has a waiter");
        assert!(matches!(ack.await.expect("ack"), Ok(())));
        assert!(
            slot.deliver(ControlLine::Reply(ControlReply::Shutdown))
                .is_err()
        );
    }

    /// Retiring one worker refuses its later ops retryably, unlike a pool
    /// shutdown.
    #[test]
    fn a_retired_worker_refuses_later_ops_retryably() {
        let slot = ControlSlot::default();
        let _ack = slot
            .begin_shutdown(Retirement::WorkerRetired)
            .expect("first shutdown");
        assert!(matches!(
            slot.arm_ensure_task(InferTask::Fa),
            Err(WorkerError::WorkerRetired)
        ));
    }
}
