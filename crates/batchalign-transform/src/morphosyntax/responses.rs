//! Admission of utterance-level worker responses before any CHAT mutation.

use super::{BatchItemWithPosition, UdResponse};

/// A worker returned a different number of utterance responses than requested.
#[derive(Debug, thiserror::Error)]
#[error("morphotag: {expected} payloads received {actual} worker responses")]
pub struct ResponseCountMismatch {
    /// Number of submitted utterance payloads.
    pub expected: usize,
    /// Number of returned utterance responses.
    pub actual: usize,
}

/// Payloads and worker responses admitted together with equal cardinality.
///
/// Their vectors are private and only borrowed immutably. Injection consumes
/// this value, so the matching result cannot be detached and reused elsewhere.
/// Cardinality does not prove linguistic correctness or response ordering;
/// those remain worker-contract and per-utterance validation responsibilities.
pub struct MatchedMorphosyntaxResponses {
    items: Vec<BatchItemWithPosition>,
    responses: Vec<UdResponse>,
}

impl MatchedMorphosyntaxResponses {
    /// Admit the complete batch before an injector can alter any CHAT data.
    pub fn new(
        items: Vec<BatchItemWithPosition>,
        responses: Vec<UdResponse>,
    ) -> Result<Self, ResponseCountMismatch> {
        if items.len() != responses.len() {
            return Err(ResponseCountMismatch {
                expected: items.len(),
                actual: responses.len(),
            });
        }
        Ok(Self { items, responses })
    }

    /// Submitted items, for downstream planning such as secondary-language work.
    pub fn items(&self) -> &[BatchItemWithPosition] {
        &self.items
    }

    /// Responses in worker-contract order, available only as an immutable slice.
    pub fn responses(&self) -> &[UdResponse] {
        &self.responses
    }

    /// Consume the paired data without truncation: equal lengths were admitted
    /// once and cannot change through this type's public surface.
    pub(super) fn into_pairs(
        self,
    ) -> impl ExactSizeIterator<Item = (UdResponse, BatchItemWithPosition)> {
        self.responses.into_iter().zip(self.items)
    }
}

/// A matched batch whose destination positions were admitted while exclusively
/// borrowing the document. Only the injection implementation can consume it.
pub(super) struct BoundMorphosyntaxResponses<'chat> {
    chat: &'chat mut talkbank_model::ChatFile,
    batch: MatchedMorphosyntaxResponses,
}

#[derive(Debug, thiserror::Error)]
#[error("Line at index {index} is no longer an utterance")]
pub(super) struct InvalidInjectionPosition {
    index: usize,
}

impl MatchedMorphosyntaxResponses {
    pub(super) fn bind(
        self,
        chat: &mut talkbank_model::ChatFile,
    ) -> Result<BoundMorphosyntaxResponses<'_>, InvalidInjectionPosition> {
        for (index, ..) in &self.items {
            if !matches!(
                chat.lines.get(*index),
                Some(talkbank_model::Line::Utterance(_))
            ) {
                return Err(InvalidInjectionPosition { index: *index });
            }
        }
        Ok(BoundMorphosyntaxResponses { chat, batch: self })
    }
}

impl<'chat> BoundMorphosyntaxResponses<'chat> {
    pub(super) fn into_parts(
        self,
    ) -> (
        &'chat mut talkbank_model::ChatFile,
        MatchedMorphosyntaxResponses,
    ) {
        (self.chat, self.batch)
    }
}
