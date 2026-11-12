//! Owned UTR computation: recognition observes a private candidate and never
//! replaces the source document.

use batchalign_transform::dp_align::control::WorkControl;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::chat_ops::fa::SourceBoundUtrInput;
#[cfg(test)]
use crate::chat_ops::fa::utr::AsrTimingToken;
use crate::chat_ops::fa::utr::{PreparedUtrTokens, UtrCancelled, UtrObservation, UtrStrategy};
use crate::error::ServerError;
use crate::provenance::{UtrModels, UtrProducer, UtrRecognitionEvidence};
use crate::runner::util::{FileStage, ProgressSender, ProgressUpdate};

/// Tokens and retained responses prepared together for one correspondence search.
pub(crate) struct PreparedRecognition {
    tokens: PreparedUtrTokens,
    producer: UtrProducer,
    inputs: UtrRecognitionEvidence,
}

/// A completed search and the recognition inputs it actually consumed.
#[derive(Debug)]
pub(crate) struct CompletedRecognition {
    observation: UtrObservation,
    producer: UtrProducer,
    inputs: UtrRecognitionEvidence,
}

impl PreparedRecognition {
    pub(super) fn full(recognition: super::full::FullRecognition<'_>) -> Result<Self, ServerError> {
        let (conversion, context, route) = recognition.into_parts();
        Self::prepare(
            conversion,
            Vec::new(),
            context.engine,
            context.lang,
            RecognitionOrder::Provider,
            route.into(),
        )
    }

    #[cfg(test)]
    pub(super) fn full_for_test(
        conversion: super::UtrTokenConversion,
        engine: &crate::options::UtrEngine,
        language: &crate::api::LanguageCode3,
        route: super::UtrFullRoute,
    ) -> Result<Self, ServerError> {
        Self::prepare(
            conversion,
            Vec::new(),
            engine,
            language,
            RecognitionOrder::Provider,
            route.into(),
        )
    }

    pub(super) fn partial(
        completed: super::partial::CompletedPartialRecognition<'_>,
    ) -> Result<Self, ServerError> {
        let (first, remaining, context) = completed.into_parts();
        Self::prepare(
            first,
            remaining,
            context.engine,
            context.lang,
            RecognitionOrder::Timeline,
            super::UtrRecognitionRoute::Partial,
        )
    }

    #[cfg(test)]
    pub(crate) fn partial_for_test(
        first: super::UtrTokenConversion,
        remaining: Vec<super::UtrTokenConversion>,
        engine: &crate::options::UtrEngine,
        language: &crate::api::LanguageCode3,
    ) -> Result<Self, ServerError> {
        Self::prepare(
            first,
            remaining,
            engine,
            language,
            RecognitionOrder::Timeline,
            super::UtrRecognitionRoute::Partial,
        )
    }

    fn prepare(
        first: super::UtrTokenConversion,
        remaining: Vec<super::UtrTokenConversion>,
        engine: &crate::options::UtrEngine,
        language: &crate::api::LanguageCode3,
        order: RecognitionOrder,
        route: super::UtrRecognitionRoute,
    ) -> Result<Self, ServerError> {
        let (mut tokens, first) = first.into_evidence();
        let (models, remaining) = remaining.into_iter().fold(
            (UtrModels::of(first.response()), Vec::new()),
            |(models, mut windows), conversion| {
                let (window_tokens, window) = conversion.into_evidence();
                tokens.extend(window_tokens);
                let models = models.and(window.response());
                windows.push(window);
                (models, windows)
            },
        );
        if matches!(order, RecognitionOrder::Timeline) {
            tokens.sort_by_key(|token| token.start_ms);
        }
        Ok(Self {
            tokens: super::normalize_utr_tokens(tokens, language)?,
            producer: UtrProducer::new(engine, models),
            inputs: UtrRecognitionEvidence::Observed {
                first: Box::new(first),
                remaining,
                route,
            },
        })
    }

    pub(super) fn tokens(&self) -> &PreparedUtrTokens {
        &self.tokens
    }

    pub(crate) async fn run(
        self,
        source: &SourceBoundUtrInput<'_>,
        strategy: Box<dyn UtrStrategy>,
        progress: Option<&ProgressSender>,
    ) -> Result<CompletedRecognition, ServerError> {
        let observation = PendingUtr::new(source, strategy, self.tokens)
            .run(progress)
            .await?;
        Ok(CompletedRecognition {
            observation,
            producer: self.producer,
            inputs: self.inputs,
        })
    }
}

enum RecognitionOrder {
    Provider,
    Timeline,
}

impl CompletedRecognition {
    pub(super) fn observation(&self) -> &UtrObservation {
        &self.observation
    }

    pub(crate) fn into_parts(self) -> (UtrObservation, UtrProducer, UtrRecognitionEvidence) {
        (self.observation, self.producer, self.inputs)
    }

    #[cfg(test)]
    pub(crate) fn synthetic(observation: UtrObservation, producer: UtrProducer) -> Self {
        Self {
            observation,
            producer,
            inputs: UtrRecognitionEvidence::Synthetic,
        }
    }
}

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
    fn packed_search_cost(
        &self,
        cost: &batchalign_transform::dp_align::interleaving::PackedSearchCost,
    ) {
        tracing::debug!(
            status = cost.status().name(),
            last_stage = cost.last_stage().name(),
            source_words = cost.source_words(),
            reference_words = cost.reference_words(),
            graph_nodes = cost.graph_nodes(),
            work = ?cost.used(),
            score_units = cost.reserved_score_units(),
            "UTR packed correspondence search finished"
        );
    }
}

/// The shared borrow prevents source edits while work is pending. The blocking
/// worker receives a private candidate, never a document it can publish.
pub(super) struct PendingUtr<'source, 'document> {
    source: &'source SourceBoundUtrInput<'document>,
    strategy: Box<dyn UtrStrategy>,
    tokens: PreparedUtrTokens,
}

impl<'source, 'document> PendingUtr<'source, 'document> {
    pub(super) fn new(
        source: &'source SourceBoundUtrInput<'document>,
        strategy: Box<dyn UtrStrategy>,
        tokens: PreparedUtrTokens,
    ) -> Self {
        Self {
            source,
            strategy,
            tokens,
        }
    }

    pub(super) async fn run(
        self,
        progress: Option<&ProgressSender>,
    ) -> Result<UtrObservation, ServerError> {
        // ASR-window counts describe completed provider work, not progress
        // through this CPU proof. Clear them at the consuming search boundary.
        if let Some(progress) = progress {
            let _ = progress.send(ProgressUpdate::new(
                FileStage::RecoveringUtteranceTiming,
                None,
                None,
            ));
        }
        let cancellation = Stop(Arc::new(AtomicBool::new(false)));
        // Dropping the awaiting future (job cancellation or task abort) stops
        // queued work at entry and started work at its next checkpoint.
        let _cancel_on_drop = CancelOnDrop(cancellation.clone());
        let candidate = crate::chat_ops::fa::utr::UtrCandidate::from_source(self.source);
        let task =
            tokio_util::task::AbortOnDropHandle::new(crate::blocking::spawn_in_span(move || {
                let control = cancellation;
                control.check()?;
                let observation = crate::chat_ops::fa::utr::observe_controlled(
                    self.strategy.as_ref(),
                    candidate,
                    self.tokens,
                    &control,
                )?;
                control.check()?;
                Ok::<_, UtrCancelled>(observation)
            }));
        let completed = task
            .await
            .map_err(ServerError::UtrTask)?
            .map_err(|UtrCancelled| ServerError::Cancelled)?;
        Ok(completed)
    }
}

#[cfg(test)]
#[path = "utr_work/tests.rs"]
mod tests;
