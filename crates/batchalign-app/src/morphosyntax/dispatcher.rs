//! Typed seam between `run_morphosyntax_batch_impl` and the worker pool.
//!
//! Abstracts only the per-language-group dispatch call — the one
//! operation that can return an error the orchestrator has to route
//! through the per-file classifier. Capacity queries stay on
//! `PipelineServices::pool` because they are read-only metadata.
//! Letting tests inject canned per-language failures is the main
//! motivation; production code goes through the zero-cost
//! [`PoolDispatcher`] wrapper.

use std::future::Future;
use std::pin::Pin;

use batchalign_chat_ops::morphosyntax::{BatchItemWithPosition, MwtDict};
use batchalign_chat_ops::nlp::UdResponse;

use crate::api::LanguageCode3;
use crate::error::ServerError;
use crate::types::worker_v2::ProgressEventV2;
use crate::worker::pool::WorkerPool;

/// One language group's inference call.
///
/// Returns `Ok(responses)` where each response is positionally aligned
/// with `items`, or `Err` when the underlying pool rejected the dispatch
/// (saturation bailout, worker crash, timeout, worker-pool shutdown).
///
/// Object-safe: the `dispatch` method returns a boxed future so
/// `&dyn LanguageGroupDispatcher` can cross `async move` boundaries
/// inside the orchestrator's per-language futures loop.
pub(crate) trait LanguageGroupDispatcher: Send + Sync {
    /// Run inference for one language group and return one UD response
    /// per input item, in order.
    fn dispatch<'a>(
        &'a self,
        lang: &'a LanguageCode3,
        items: &'a [BatchItemWithPosition],
        mwt: &'a MwtDict,
        retokenize: bool,
        progress_tx: Option<&'a tokio::sync::mpsc::Sender<ProgressEventV2>>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<UdResponse>, ServerError>> + Send + 'a>>;
}

/// Production dispatcher — delegates to the real worker pool via
/// [`super::worker::infer_batch`]. Zero-cost wrapper.
pub(crate) struct PoolDispatcher<'p> {
    pool: &'p WorkerPool,
}

impl<'p> PoolDispatcher<'p> {
    pub(crate) fn new(pool: &'p WorkerPool) -> Self {
        Self { pool }
    }
}

impl LanguageGroupDispatcher for PoolDispatcher<'_> {
    fn dispatch<'a>(
        &'a self,
        lang: &'a LanguageCode3,
        items: &'a [BatchItemWithPosition],
        mwt: &'a MwtDict,
        retokenize: bool,
        progress_tx: Option<&'a tokio::sync::mpsc::Sender<ProgressEventV2>>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<UdResponse>, ServerError>> + Send + 'a>> {
        Box::pin(super::worker::infer_batch(
            self.pool,
            items,
            lang,
            mwt,
            retokenize,
            progress_tx,
        ))
    }
}
