//! Ledger-owned result retention and source-derived progress.

use super::*;

impl<'p, 'g> UnitLedger<'p, 'g> {
    /// The ledger derives progress from its actual resolved and awaited slots.
    /// Callers cannot supply an independent initial count or population size.
    pub(super) fn retaining<'sink>(
        &'sink mut self,
        services: crate::fa::FaServices<'sink>,
        fallback_events: &'sink mut Vec<FaFallbackEventTrace>,
        sender: Option<&'sink ProgressSender>,
    ) -> RetainUnitResults<'sink, 'p, 'g> {
        let mut progress = UnitProgress {
            completed: 0,
            total: 0,
        };
        for (planned, resolution) in &self.groups {
            match resolution {
                GroupResolution::Reused(_) => {
                    let count = planned.units().count();
                    progress.completed += count;
                    progress.total += count;
                }
                GroupResolution::Whole((_, resolution)) => {
                    progress.total += 1;
                    progress.completed += usize::from(resolution.is_some());
                }
                GroupResolution::Pieces(pieces) => {
                    for piece in pieces {
                        match piece {
                            PartResolution::Request {
                                slot: (_, resolution),
                                ..
                            } => {
                                progress.total += 1;
                                progress.completed += usize::from(resolution.is_some());
                            }
                            PartResolution::Unavailable(_) => {}
                        }
                    }
                }
            }
        }
        RetainUnitResults {
            ledger: self,
            fallback_events,
            services,
            sender,
            progress,
        }
    }
}

/// Counts of actual ledger slots, derived once after cache resolution.
struct UnitProgress {
    completed: usize,
    total: usize,
}

impl UnitProgress {
    fn report(&self, sender: Option<&ProgressSender>) {
        if let Some(tx) = sender {
            let _ = tx.send(ProgressUpdate::new(
                FileStage::Aligning,
                Some(self.completed as i64),
                Some(self.total as i64),
            ));
        }
    }
}

/// Owns result retention and progress together. A consumed worker outcome
/// advances completion only after its timing projection enters the ledger.
pub(super) struct RetainUnitResults<'sink, 'plan, 'groups> {
    ledger: &'sink mut UnitLedger<'plan, 'groups>,
    fallback_events: &'sink mut Vec<FaFallbackEventTrace>,
    services: crate::fa::FaServices<'sink>,
    sender: Option<&'sink ProgressSender>,
    progress: UnitProgress,
}

impl<'unit> crate::fa::transport::FaResultSink<'unit> for RetainUnitResults<'_, '_, '_> {
    fn retain<'step>(
        &'step mut self,
        result: crate::fa::transport::FaWorkerUnitResult<'unit>,
    ) -> futures::future::BoxFuture<'step, Result<(), ServerError>>
    where
        'unit: 'step,
    {
        Box::pin(async move {
            let projection = result.into_projection();
            let source = projection.source();
            if let Some(event) = projection.fallback_event {
                self.fallback_events.push(event);
            }
            // Canonical direct evidence alone owns a cache write. Shared aliases
            // contain timing projections, never a second raw evidence capability.
            if let Some(raw_evidence) = projection.raw_evidence {
                store_unit_evidence(
                    self.services,
                    projection.unit.cache_key.as_str(),
                    &projection.timings,
                    raw_evidence,
                )
                .await;
            }
            self.ledger.record(
                projection.unit,
                UnitResolution {
                    timings: projection.timings,
                    source,
                },
            )?;
            self.progress.completed += 1;
            self.progress.report(self.sender);
            Ok(())
        })
    }
}

impl RetainUnitResults<'_, '_, '_> {
    pub(super) fn completed(&self) -> usize {
        self.progress.completed
    }

    pub(super) fn total(&self) -> usize {
        self.progress.total
    }

    pub(super) fn report(&self) {
        self.progress.report(self.sender);
    }
}
