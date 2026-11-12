//! Unique completion rights, minted with their receiving slot and progress.

use super::resolution::{UnitAnswer, bind_timings};
use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio::sync::oneshot;

/// Read-only progress for one ledger. Only its reply tickets can advance it.
#[derive(Clone, Debug)]
pub(super) struct ReplyProgress(Arc<ReplyCounts>);

#[derive(Debug)]
struct ReplyCounts {
    completed: AtomicUsize,
    total: usize,
}

impl ReplyProgress {
    pub(super) fn for_plan(plan: &DispatchPlan<'_>) -> Self {
        Self(Arc::new(ReplyCounts {
            completed: AtomicUsize::new(0),
            total: plan.units().count(),
        }))
    }

    pub(super) fn reused(&self, group: &PlannedGroup<'_>) {
        self.0
            .completed
            .fetch_add(group.units().count(), Ordering::Relaxed);
    }

    pub(super) fn completed(&self) -> usize {
        self.0.completed.load(Ordering::Relaxed)
    }

    pub(super) fn total(&self) -> usize {
        self.0.total
    }
}

/// One request's timings, bound to that request (one slot per request word,
/// every timing inside the request window), and where they came from.
///
/// Its fields are private to this module and only a [`ReplyTicket`] builds
/// one, against its own unit, so the ledger can assemble every resolution it
/// receives without a refusal path: an answer bound for another request, or
/// not bound at all, cannot reach it.
#[derive(Debug)]
pub(in crate::fa) struct UnitResolution {
    timings: Vec<Option<WordTiming>>,
    source: FaEvidenceSourceTrace,
}

impl UnitResolution {
    /// The timings, one per request word, and their source.
    pub(super) fn into_parts(self) -> (Vec<Option<WordTiming>>, FaEvidenceSourceTrace) {
        (self.timings, self.source)
    }
}

/// What became of a cached answer offered to a ticket.
#[must_use = "a refused cache answer leaves its request to be aligned"]
pub(super) enum CacheDelivery<'unit> {
    /// The answer fit its request and was delivered.
    Delivered,
    /// The answer does not answer its request: corrupt evidence for it, so
    /// the ticket comes back and the request is aligned again.
    Refused(ReplyTicket<'unit>),
}

/// A receiving slot cannot issue a new completion right.
struct ReplyReceiver(oneshot::Receiver<UnitResolution>);

pub(super) struct ReplySlot<'p, 'g> {
    unit: &'p DispatchUnit<'g>,
    receiver: ReplyReceiver,
}

/// At most one result can complete this exact slot. Never cloneable.
#[derive(Debug)]
#[must_use = "a reply ticket must be resolved or abandoned with its attempt"]
pub(in crate::fa) struct ReplyTicket<'unit> {
    unit: &'unit DispatchUnit<'unit>,
    sender: oneshot::Sender<UnitResolution>,
    progress: ReplyProgress,
}

impl<'p, 'g> ReplySlot<'p, 'g> {
    pub(super) fn open(
        unit: &'p DispatchUnit<'g>,
        progress: ReplyProgress,
    ) -> (Self, ReplyTicket<'p>) {
        let (sender, receiver) = oneshot::channel();
        (
            Self {
                unit,
                receiver: ReplyReceiver(receiver),
            },
            ReplyTicket {
                unit,
                sender,
                progress,
            },
        )
    }

    pub(super) fn receive(mut self) -> (&'p DispatchUnit<'g>, Option<UnitResolution>) {
        // Empty and Closed both mean no completed reply at assembly. Neither
        // a pending nor abandoned sender fabricates a timing population.
        let answer = self.receiver.0.try_recv().ok();
        (self.unit, answer)
    }
}

impl<'unit> ReplyTicket<'unit> {
    pub(in crate::fa) fn unit(&self) -> &'unit DispatchUnit<'unit> {
        self.unit
    }

    /// Offer a cached answer, bound to this ticket's own request.
    pub(super) fn cached(self, answer: UnitAnswer) -> Result<CacheDelivery<'unit>, ServerError> {
        let (timings, source) = answer.into_parts();
        match bind_timings(self.unit.input(), timings) {
            Ok(timings) => {
                self.deliver(UnitResolution { timings, source })?;
                Ok(CacheDelivery::Delivered)
            }
            Err(refusal) => {
                tracing::warn!(
                    %refusal,
                    request = self.unit.ordinal.0,
                    group = self.unit.group.0,
                    "Cached FA evidence does not answer its request; aligning it again"
                );
                Ok(CacheDelivery::Refused(self))
            }
        }
    }

    /// Deliver a worker's answer, bound to this ticket's own request. A
    /// reply that does not answer for it leaves only its words untimed,
    /// recorded as refused: a defect local to one window never fails the
    /// file.
    pub(in crate::fa) fn observed(
        self,
        timings: Vec<Option<WordTiming>>,
        source: FaEvidenceSourceTrace,
    ) -> Result<(), ServerError> {
        let request = self.unit.input();
        let resolution = match bind_timings(request, timings) {
            Ok(timings) => UnitResolution { timings, source },
            Err(refusal) => {
                tracing::warn!(
                    %refusal,
                    request = self.unit.ordinal.0,
                    group = self.unit.group.0,
                    "FA reply does not answer its request; that request's words stay untimed"
                );
                UnitResolution {
                    timings: vec![None; request.words().len()],
                    source: FaEvidenceSourceTrace::Refused,
                }
            }
        };
        self.deliver(resolution)
    }

    fn deliver(self, answer: UnitResolution) -> Result<(), ServerError> {
        self.sender.send(answer).map_err(|_| {
            crate::error::AlignmentReplyDeliveryFailure::OwnerReleased {
                group: self.unit.group.0,
                request: self.unit.ordinal.0,
            }
        })?;
        self.progress.0.completed.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}
