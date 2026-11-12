//! One consuming dispatch epoch; only applied outcomes release its continuation.

use super::*;
use crate::fa::preparation::PreparedFaDispatch;

/// Heap ownership is reused across transitions, not allocated per request.
struct ReadyRequests<'services, 'unit> {
    services: crate::fa::FaServices<'services>,
    language: WorkerLanguage,
    engine: crate::types::engines::FaEngineName,
    remaining: std::vec::IntoIter<PreparedFaDispatch<'unit>>,
}

/// The following request is inaccessible until this outcome has been retained.
struct CompletedDispatch<'services, 'unit> {
    result: FaWorkerReply<'unit>,
    aliases: Vec<ReplyTicket<'unit>>,
    continuation: Box<ReadyRequests<'services, 'unit>>,
}

impl<'services, 'unit> ReadyRequests<'services, 'unit> {
    async fn advance(
        mut self: Box<Self>,
    ) -> Result<Option<CompletedDispatch<'services, 'unit>>, ServerError> {
        let Some(dispatch) = self.remaining.next() else {
            return Ok(None);
        };
        let result = infer_prepared_unit(
            self.services,
            &self.language,
            self.engine,
            dispatch.reply.unit(),
            dispatch.prepared,
        )
        .await?;
        Ok(Some(CompletedDispatch {
            result: FaWorkerReply {
                reply: dispatch.reply,
                result,
            },
            aliases: dispatch.aliases,
            continuation: self,
        }))
    }
}

impl<'services, 'unit> CompletedDispatch<'services, 'unit> {
    async fn retain<S: FaResultSink<'unit>>(
        self,
        sink: &mut S,
    ) -> Result<Box<ReadyRequests<'services, 'unit>>, ServerError> {
        // Share only timing projections while the canonical result still owns
        // its raw evidence. No whole-batch result collection survives here.
        let aliases: Vec<_> = self
            .aliases
            .into_iter()
            .map(|reply| self.result.share_with(reply))
            .collect();
        sink.retain(self.result).await?;
        for alias in aliases {
            sink.retain(alias).await?;
        }
        Ok(self.continuation)
    }
}

pub(super) async fn infer<'unit, S: FaResultSink<'unit>>(
    services: crate::fa::FaServices<'_>,
    batch: FaWorkerBatch<'unit>,
    prepared: crate::fa::preparation::PreparedFaRequests,
    mut sink: S,
) -> Result<CompletedFaInference<S>, ServerError> {
    let dispatches = prepared.bind(batch.authorization)?;
    let mut ready = Box::new(ReadyRequests {
        services,
        language: batch.worker_lang,
        engine: batch.engine,
        remaining: dispatches.into_iter(),
    });
    loop {
        match ready.advance().await? {
            Some(completed) => ready = completed.retain(&mut sink).await?,
            None => return Ok(CompletedFaInference(sink)),
        }
    }
}
