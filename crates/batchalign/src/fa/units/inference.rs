//! Owned reply retention and source-derived progress.

use super::*;

/// A view of the ledger's counts; only its unique reply tickets advance them.
pub(super) struct RetainUnitResults<'sink> {
    fallback_events: &'sink mut Vec<FaFallbackEventTrace>,
    services: crate::fa::FaServices<'sink>,
    sender: Option<&'sink ProgressSender>,
    progress: ReplyProgress,
}

impl<'sink> RetainUnitResults<'sink> {
    pub(super) fn new(
        progress: ReplyProgress,
        services: crate::fa::FaServices<'sink>,
        fallback_events: &'sink mut Vec<FaFallbackEventTrace>,
        sender: Option<&'sink ProgressSender>,
    ) -> Self {
        Self {
            progress,
            services,
            fallback_events,
            sender,
        }
    }

    pub(super) fn completed(&self) -> usize {
        self.progress.completed()
    }
    pub(super) fn total(&self) -> usize {
        self.progress.total()
    }
    pub(super) fn report(&self) {
        if let Some(tx) = self.sender {
            let _ = tx.send(ProgressUpdate::new(
                FileStage::Aligning,
                Some(self.completed() as i64),
                Some(self.total() as i64),
            ));
        }
    }
}

impl<'unit> crate::fa::transport::FaResultSink<'unit> for RetainUnitResults<'_> {
    fn retain<'step>(
        &'step mut self,
        result: crate::fa::transport::FaWorkerReply<'unit>,
    ) -> futures::future::BoxFuture<'step, Result<(), ServerError>>
    where
        'unit: 'step,
    {
        Box::pin(async move {
            if let Some(event) = result.retain(self.services).await? {
                self.fallback_events.push(event);
            }
            self.report();
            Ok(())
        })
    }
}
