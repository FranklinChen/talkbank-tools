//! Admission of utterance-level worker responses before any CHAT mutation.

use super::{CollectedUtterance, UdResponse, UdSentence};

/// A model returned multiple sentences for one submitted utterance or L2 span.
#[derive(Debug, thiserror::Error)]
#[error("morphotag: one payload received {actual} sentences; expected zero or one")]
pub struct UnexpectedSentenceCount {
    /// Number of sentences returned by the model.
    pub actual: usize,
}

/// One admitted analysis: explicitly empty, or exactly one complete sentence.
///
/// This proves response cardinality, not lexical coverage or linguistic
/// correctness. No raw sentence vector survives admission for consumers to
/// truncate. Empty analyses remain available for typed special-form synthesis.
#[derive(Debug, Clone)]
pub struct AdmittedUdResponse {
    sentence: Option<UdSentence>,
}

impl TryFrom<UdResponse> for AdmittedUdResponse {
    type Error = UnexpectedSentenceCount;

    fn try_from(mut response: UdResponse) -> Result<Self, Self::Error> {
        match response.sentences.len() {
            0 | 1 => Ok(Self {
                sentence: response.sentences.pop(),
            }),
            actual => Err(UnexpectedSentenceCount { actual }),
        }
    }
}

impl AdmittedUdResponse {
    /// An explicit absence of model analysis, not completed morphology.
    pub fn empty() -> Self {
        Self { sentence: None }
    }

    /// The entire admitted sentence, if the model supplied an analysis.
    pub fn sentence(&self) -> Option<&UdSentence> {
        self.sentence.as_ref()
    }
}

/// A worker returned a different number of utterance responses than requested.
#[derive(Debug, thiserror::Error)]
#[error("morphotag: {expected} payloads received {actual} worker responses")]
pub struct ResponseCountMismatch {
    /// Number of submitted utterance payloads.
    pub expected: usize,
    /// Number of returned utterance responses.
    pub actual: usize,
}

/// Admission failed before any document mutation.
#[derive(Debug, thiserror::Error)]
pub enum ResponseAdmissionError {
    /// Utterance payload and response counts differ.
    #[error(transparent)]
    ResponseCount(#[from] ResponseCountMismatch),
    /// A particular utterance response contains more than one sentence.
    #[error("worker response {index}: {error}")]
    SentenceCount {
        /// Zero-based submitted payload position.
        index: usize,
        /// Retained model response cardinality.
        #[source]
        error: UnexpectedSentenceCount,
    },
}

/// Payloads and worker responses admitted together with equal cardinality.
///
/// Their vectors are private and only borrowed immutably. Injection consumes
/// this value, so the matching result cannot be detached and reused elsewhere.
/// Cardinality does not prove linguistic correctness or response ordering;
/// those remain worker-contract and per-utterance validation responsibilities.
pub struct MatchedMorphosyntaxResponses {
    items: Vec<CollectedUtterance>,
    responses: Vec<AdmittedUdResponse>,
}

impl MatchedMorphosyntaxResponses {
    /// Admit the complete batch before an injector can alter any CHAT data.
    pub fn new(
        items: Vec<CollectedUtterance>,
        responses: Vec<UdResponse>,
    ) -> Result<Self, ResponseAdmissionError> {
        // Check the batch before admitting any item, preserving count errors
        // even when an unpaired response is also malformed.
        check_response_count(items.len(), responses.len())?;
        let responses = responses
            .into_iter()
            .enumerate()
            .map(|(index, response)| {
                AdmittedUdResponse::try_from(response)
                    .map_err(|error| ResponseAdmissionError::SentenceCount { index, error })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { items, responses })
    }

    /// Pair analyses already admitted by their producer, without rechecking or
    /// reconstructing their single-sentence proof at each consumer.
    pub fn from_admitted(
        items: Vec<CollectedUtterance>,
        responses: Vec<AdmittedUdResponse>,
    ) -> Result<Self, ResponseCountMismatch> {
        check_response_count(items.len(), responses.len())?;
        Ok(Self { items, responses })
    }

    /// Submitted items, for downstream planning such as secondary-language work.
    pub fn items(&self) -> &[CollectedUtterance] {
        &self.items
    }

    /// Entire admitted analyses in worker-contract order, borrowed immutably.
    pub fn responses(&self) -> &[AdmittedUdResponse] {
        &self.responses
    }

    /// Consume the paired data without truncation: equal lengths were admitted
    /// once and cannot change through this type's public surface.
    pub(super) fn into_pairs(
        self,
    ) -> impl ExactSizeIterator<Item = (AdmittedUdResponse, CollectedUtterance)> {
        self.responses.into_iter().zip(self.items)
    }
}

fn check_response_count(expected: usize, actual: usize) -> Result<(), ResponseCountMismatch> {
    if expected != actual {
        return Err(ResponseCountMismatch { expected, actual });
    }
    Ok(())
}

/// A matched batch bound to its destinations: every response is paired with
/// the utterance it annotates, borrowed from the document. Binding is the one
/// check that a collected line still names an utterance, and the only route to
/// that utterance, so injection never looks a line up again.
pub(super) struct BoundMorphosyntaxResponses<'chat> {
    pairs: Vec<BoundUtterance<'chat>>,
}

/// One response, its collected item, and the utterance both describe.
pub(super) struct BoundUtterance<'chat> {
    pub(super) utterance: &'chat mut talkbank_model::model::Utterance,
    pub(super) response: AdmittedUdResponse,
    pub(super) collected: CollectedUtterance,
}

/// Why a collected line could not be bound.
#[derive(Debug, thiserror::Error)]
#[error("Line at index {index} is no longer an utterance, or is claimed twice")]
pub(super) struct InvalidInjectionPosition {
    pub(super) index: usize,
}

impl MatchedMorphosyntaxResponses {
    pub(super) fn bind(
        self,
        chat: &mut talkbank_model::ChatFile,
    ) -> Result<BoundMorphosyntaxResponses<'_>, InvalidInjectionPosition> {
        // One pass hands out a disjoint borrow of every utterance line; a
        // line is taken by the first response that names it, so a second
        // response for the same line finds it gone.
        let mut utterances: std::collections::HashMap<
            usize,
            &mut talkbank_model::model::Utterance,
        > = chat
            .lines
            .as_mut_slice()
            .iter_mut()
            .enumerate()
            .filter_map(|(index, line)| match line {
                talkbank_model::Line::Utterance(utterance) => Some((index, &mut **utterance)),
                _ => None,
            })
            .collect();
        let pairs = self
            .into_pairs()
            .map(|(response, collected)| {
                let index = collected.line().raw();
                utterances
                    .remove(&index)
                    .map(|utterance| BoundUtterance {
                        utterance,
                        response,
                        collected,
                    })
                    .ok_or(InvalidInjectionPosition { index })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(BoundMorphosyntaxResponses { pairs })
    }
}

impl<'chat> BoundMorphosyntaxResponses<'chat> {
    pub(super) fn into_pairs(self) -> impl Iterator<Item = BoundUtterance<'chat>> {
        self.pairs.into_iter()
    }
}
