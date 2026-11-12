//! Unique completion rights, minted with their receiving slot and progress.

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

    pub(super) fn cached(self, answer: UnitResolution) -> Result<(), ServerError> {
        self.deliver(answer)
    }

    pub(in crate::fa) fn observed(
        self,
        timings: Vec<Option<WordTiming>>,
        source: FaEvidenceSourceTrace,
    ) -> Result<(), ServerError> {
        self.deliver(UnitResolution { timings, source })
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
