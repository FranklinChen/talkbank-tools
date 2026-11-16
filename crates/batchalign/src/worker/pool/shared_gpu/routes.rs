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

use tokio::sync::{mpsc, oneshot};

use crate::types::worker_v2::{ExecuteResponseV2, LoadProgressV2, WorkerRequestIdV2};
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

/// A pending invocation owns its resources until the terminal effect consumes it.
/// The reader's policy contracts use the same graph with an observation-only double.
pub(crate) trait DispatchOwner: Send + 'static {
    fn request_id(&self) -> &WorkerRequestIdV2;
    fn finish(self, reply: DispatchReply) -> DispatchReply;
}

/// Shared routing retains execution ownership independently of caller interest.
pub(crate) struct PendingDispatches<O: DispatchOwner = super::channel::ScheduledInvocation>(
    Arc<Mutex<PendingState<O>>>,
);

impl<O: DispatchOwner> Clone for PendingDispatches<O> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<O: DispatchOwner> Default for PendingDispatches<O> {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(PendingState::Open(HashMap::new()))))
    }
}

struct PendingInvocation<O> {
    owner: O,
    reply: oneshot::Sender<DispatchReply>,
}

enum PendingState<O> {
    Open(HashMap<WorkerRequestIdV2, PendingInvocation<O>>),
    Closed(StreamClosed),
}

/// Caller interest in the reply carries no authority over submitted work.
pub(crate) struct DispatchReceipt {
    reply: oneshot::Receiver<DispatchReply>,
}

impl DispatchReceipt {
    /// A timeout ends reply interest. The pending owner survives until its own
    /// terminal line or stream retirement, including when this future is dropped.
    pub(crate) async fn reply_within(self, limit: std::time::Duration) -> Option<DispatchReply> {
        match tokio::time::timeout(limit, self.reply).await {
            Ok(Ok(reply)) => Some(reply),
            Ok(Err(_)) => Some(Err(StreamClosed::ReaderStopped.to_worker_error())),
            Err(_) => None,
        }
    }
}

impl<O: DispatchOwner> PendingDispatches<O> {
    /// Transfer execution ownership before the first write. The channel's
    /// consuming write guard retires uncertain writes, including cancellation.
    pub(crate) fn register(&self, owner: O) -> Result<DispatchReceipt, WorkerError> {
        let (tx, rx) = oneshot::channel();
        let request_id = owner.request_id().clone();
        match &mut *lock_recovered(&self.0) {
            PendingState::Closed(reason) => return Err(reason.to_worker_error()),
            PendingState::Open(waiting) => match waiting.entry(request_id) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(PendingInvocation { owner, reply: tx });
                }
                std::collections::hash_map::Entry::Occupied(entry) => {
                    return Err(WorkerError::Protocol(format!(
                        "request id {} is already waiting for a GPU worker reply",
                        entry.key()
                    )));
                }
            },
        }
        Ok(DispatchReceipt { reply: rx })
    }

    /// Consume the original owner's terminal binding and resources once.
    pub(crate) fn answer(
        &self,
        request_id: &str,
        reply: DispatchReply,
    ) -> Result<(), DispatchReply> {
        let invocation = match &mut *lock_recovered(&self.0) {
            PendingState::Open(waiting) => waiting.remove(request_id),
            PendingState::Closed(_) => None,
        };
        match invocation {
            Some(PendingInvocation {
                owner,
                reply: sender,
            }) => {
                let _ = sender.send(owner.finish(reply));
                Ok(())
            }
            None => Err(reply),
        }
    }

    /// Close once with the original reason and settle every remaining owner.
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
        for (_, PendingInvocation { owner, reply }) in stranded {
            let _ = reply.send(owner.finish(Err(reason.to_worker_error())));
        }
        true
    }

    pub(crate) fn liveness(&self) -> Result<(), WorkerError> {
        match &*lock_recovered(&self.0) {
            PendingState::Open(_) => Ok(()),
            PendingState::Closed(reason) => Err(reason.to_worker_error()),
        }
    }

    #[cfg(test)]
    pub(crate) fn waiting(&self) -> usize {
        match &*lock_recovered(&self.0) {
            PendingState::Open(waiting) => waiting.len(),
            PendingState::Closed(_) => 0,
        }
    }
}

impl PendingDispatches<super::channel::ScheduledInvocation> {
    /// A retry may reclaim a detached reply only for the exact packet still
    /// running. Active duplicate IDs and changed input remain protocol errors.
    /// This observation precedes new memory admission and executor acquisition.
    pub(super) fn rejoin(
        &self,
        request: &crate::types::worker_v2::ExecuteRequestV2,
    ) -> Result<Option<DispatchReceipt>, WorkerError> {
        match &mut *lock_recovered(&self.0) {
            PendingState::Closed(reason) => Err(reason.to_worker_error()),
            PendingState::Open(waiting) => match waiting.get_mut(&request.request_id) {
                None => Ok(None),
                Some(invocation)
                    if invocation.reply.is_closed() && invocation.owner.request() == request =>
                {
                    let (tx, rx) = oneshot::channel();
                    invocation.reply = tx;
                    Ok(Some(DispatchReceipt { reply: rx }))
                }
                Some(_) => Err(WorkerError::Protocol(format!(
                    "request id {} already owns different work or an active caller",
                    request.request_id
                ))),
            },
        }
    }
}

/// The observation-only owner for route decisions. This double exercises the
/// same consuming registration/answer/retirement graph without inventing a
/// model load, kernel process or memory lease for byte-routing assertions.
#[cfg(test)]
pub(super) mod observations {
    use super::*;
    pub(crate) struct ObservedDispatch(WorkerRequestIdV2);
    impl ObservedDispatch {
        pub(crate) fn named(id: impl Into<WorkerRequestIdV2>) -> Self {
            Self(id.into())
        }
    }
    impl DispatchOwner for ObservedDispatch {
        fn request_id(&self) -> &WorkerRequestIdV2 {
            &self.0
        }
        fn finish(self, reply: DispatchReply) -> DispatchReply {
            reply.and_then(|reply| {
                if reply.request_id() == &self.0 {
                    Ok(reply)
                } else {
                    Err(WorkerError::Protocol("observed reply id mismatch".into()))
                }
            })
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

/// An armed `ensure_task`: like [`ArmedControl`], plus the progress the
/// worker reports while it loads, which keeps the load's stall window open
/// (`SharedChannel::ensure_task`).
pub(crate) struct ArmedLoad {
    pub(crate) request_id: ControlRequestId,
    pub(crate) answer: oneshot::Receiver<ControlAnswer<EnsureTaskResponse>>,
    pub(crate) progress: mpsc::UnboundedReceiver<LoadProgressV2>,
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
        /// Where the worker's load progress goes while it loads. Typed to
        /// carry a load's progress only, so a request's work cannot reach a
        /// load's stall window.
        progress: mpsc::UnboundedSender<LoadProgressV2>,
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
    pub(crate) fn arm_ensure_task(&self, task: InferTask) -> Result<ArmedLoad, WorkerError> {
        let (reply, answer) = oneshot::channel();
        let (progress_tx, progress) = mpsc::unbounded_channel();
        let request_id = ControlRequestId::next();
        self.arm(ControlWaiter::EnsureTask {
            request_id: request_id.clone(),
            task,
            reply,
            progress: progress_tx,
        })?;
        Ok(ArmedLoad {
            request_id,
            answer,
            progress,
        })
    }

    /// Hand a load's progress to the op waiting in the slot when it is a
    /// model load; `false` when no load is waiting to take it.
    ///
    /// Progress is observability, never an answer: it does not end or fail
    /// any op. A load's receiver gone since arming is a load that stopped
    /// waiting, and the event is simply not taken.
    pub(crate) fn progress(&self, event: LoadProgressV2) -> bool {
        let state = lock_recovered(&self.0);
        match &*state {
            ControlState::Waiting(ControlWaiter::EnsureTask { progress, .. }) => {
                progress.send(event).is_ok()
            }
            ControlState::Waiting(ControlWaiter::Capabilities { .. })
            | ControlState::Idle
            | ControlState::ShuttingDown { .. }
            | ControlState::Closed(_) => false,
        }
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
    use super::observations::ObservedDispatch;
    use super::*;
    type PendingDispatches = super::PendingDispatches<ObservedDispatch>;

    /// Guards: Closing answers every waiter through its own reply and refuses every later registration at once, instead of leaving it on its timeout.
    /// Kind: policy
    /// Not a type: the first externally observed EOF must settle existing reply owners and remain authoritative when a later shutdown reason arrives.
    /// State: local route owners, byte values and reply channels isolate the routing policy.
    /// Concurrency: all maps, channels and request identities belong to this test runtime.
    #[tokio::test]
    async fn closing_answers_waiters_and_refuses_new_registrations() {
        let pending = PendingDispatches::default();
        let receipt = pending
            .register(ObservedDispatch::named("r-1"))
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
            pending.register(ObservedDispatch::named("r-2")),
            Err(WorkerError::ProcessExited { .. })
        ));
        assert!(pending.liveness().is_err());
    }

    /// Guards: A dispatch in flight on a worker the pool retires (a refused capability report, a replaced worker) while the pool keeps running is answered with a retryable error: the work was not refused, its worker went away.
    /// Kind: policy
    /// Not a type: worker retirement during live routing selects retryable failure policy, whose classification remains a semantic decision after ownership is consumed.
    /// State: local route owners, byte values and reply channels isolate the routing policy.
    /// Concurrency: all maps, channels and request identities belong to this test runtime.
    #[tokio::test]
    async fn retiring_a_live_worker_answers_its_dispatches_retryably() {
        use crate::runner::util::classify_worker_error;
        let pending = PendingDispatches::default();
        let receipt = pending
            .register(ObservedDispatch::named("r-1"))
            .expect("an open map registers");
        pending.close(StreamClosed::Stopped(Retirement::WorkerRetired));
        let error = receipt
            .reply_within(std::time::Duration::from_secs(5))
            .await
            .expect("answered")
            .expect_err("a retired worker answers no response");
        assert!(
            ((classify_worker_error(&error)).retryability()
                == crate::types::scheduling::Retryability::Retryable),
            "{error:?}"
        );
    }

    /// Guards: A second registration under an id already waiting is refused rather than orphaning the first waiter.
    /// Kind: policy
    /// Not a type: distinct admitted external requests can repeat the same correlation string, requiring map admission to refuse the second live owner.
    /// State: local route owners, byte values and reply channels isolate the routing policy.
    /// Concurrency: all maps, channels and request identities belong to this test runtime.
    #[test]
    fn a_duplicate_request_id_is_refused() {
        let pending = PendingDispatches::default();
        let _first = pending
            .register(ObservedDispatch::named("r-1"))
            .expect("first registration");
        assert!(matches!(
            pending.register(ObservedDispatch::named("r-1")),
            Err(WorkerError::Protocol(_))
        ));
    }

    /// Guards: Shutdown supersedes a live control waiter, receives the ack itself, and refuses later ops.
    /// Kind: policy
    /// Not a type: shutdown overtakes a current control operation and its actual acknowledgement must settle only the shutdown owner before future operations are refused.
    /// State: local route owners, byte values and reply channels isolate the routing policy.
    /// Concurrency: all maps, channels and request identities belong to this test runtime.
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

    /// Guards: Retiring one worker refuses its later ops retryably, unlike a pool shutdown.
    /// Kind: policy
    /// Not a type: the WorkerRetired reason selects retryable refusal for later load attempts, independently of the different PoolShutdown policy.
    /// State: local route owners, byte values and reply channels isolate the routing policy.
    /// Concurrency: all maps, channels and request identities belong to this test runtime.
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
