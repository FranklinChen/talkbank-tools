//! Owned UTR computation: only a completed candidate can replace its source.

use batchalign_transform::dp_align::control::WorkControl;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::chat_ops::ChatFile;
#[cfg(test)]
use crate::chat_ops::fa::utr::AsrTimingToken;
use crate::chat_ops::fa::utr::{PreparedUtrTokens, UtrCancelled, UtrResult, UtrStrategy};
use crate::error::ServerError;

#[derive(Clone)]
struct Stop(Arc<AtomicBool>);

struct CancelOnDrop(Stop);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.0.store(true, Ordering::Relaxed);
    }
}

impl WorkControl for Stop {
    type Error = UtrCancelled;
    fn check(&self) -> Result<(), UtrCancelled> {
        if self.0.load(Ordering::Relaxed) {
            Err(UtrCancelled)
        } else {
            Ok(())
        }
    }
}

/// The mutable borrow prevents concurrent source edits while work is pending.
/// The blocking worker receives a private candidate, never this borrowed source.
pub(super) struct PendingUtr<'source> {
    source: &'source mut ChatFile,
    strategy: Box<dyn UtrStrategy>,
    tokens: PreparedUtrTokens,
}

struct CompletedUtr {
    candidate: ChatFile,
    result: UtrResult,
}

impl<'source> PendingUtr<'source> {
    pub(super) fn new(
        source: &'source mut ChatFile,
        strategy: Box<dyn UtrStrategy>,
        tokens: PreparedUtrTokens,
    ) -> Self {
        Self {
            source,
            strategy,
            tokens,
        }
    }

    pub(super) async fn run(self) -> Result<UtrResult, ServerError> {
        let cancellation = Stop(Arc::new(AtomicBool::new(false)));
        // Dropping the awaiting future (job cancellation or task abort) stops
        // queued work at entry and started work at its next checkpoint.
        let _cancel_on_drop = CancelOnDrop(cancellation.clone());
        let mut candidate = self.source.clone();
        let task =
            tokio_util::task::AbortOnDropHandle::new(crate::blocking::spawn_in_span(move || {
                let control = cancellation;
                control.check()?;
                let result =
                    self.strategy
                        .inject_controlled(&mut candidate, &self.tokens, &control)?;
                control.check()?;
                Ok::<_, UtrCancelled>(CompletedUtr { candidate, result })
            }));
        let completed = task
            .await
            .map_err(ServerError::UtrTask)?
            .map_err(|UtrCancelled| ServerError::Cancelled)?;
        // No await between completion admission and installation. An abandoned
        // future has no access to this transition; its candidate is discarded.
        *self.source = completed.candidate;
        Ok(completed.result)
    }
}

#[cfg(test)]
#[path = "utr_work/tests.rs"]
mod tests;
