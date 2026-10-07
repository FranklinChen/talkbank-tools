//! Complete lexical embeddings bound a search, never a selected timing.

use super::lexical::UtrLexicalStream;
use super::{UtrPayloadWord, greedy_forward_match_indices, greedy_reverse_match_indices};
use crate::chat_ops::fa::{FaWord, TimeSpan};

/// Both extremal embeddings of the entire participating lexical population.
/// Missing even one word prevents constructing this observation.
pub(super) struct CompleteExactEmbeddings {
    pub(super) earliest: Vec<usize>,
    pub(super) latest: Vec<usize>,
}

impl CompleteExactEmbeddings {
    pub(super) fn observe(words: &[String], reference: &[String]) -> Option<Self> {
        if words.is_empty() {
            return None;
        }
        Some(Self {
            earliest: greedy_forward_match_indices(words, reference)?,
            latest: greedy_reverse_match_indices(words, reference)?,
        })
    }

    pub(super) fn envelopes<'c>(
        &self,
        payload: &[UtrPayloadWord],
        searched: super::regions::SearchedCensus<'c>,
        lexical: &UtrLexicalStream<'_>,
    ) -> super::regions::PerSearched<'c, Option<FaSearchEnvelope>> {
        let Some(ordered) = lexical.ordered_search() else {
            return searched.map(|_| None);
        };
        let mut population = payload.iter().enumerate().peekable();
        searched.map(|(utterance, info)| {
            let first = population.peek().map(|(position, _)| *position);
            let mut count = 0;
            let mut same_source = true;
            let mut last = None;
            while population
                .peek()
                .is_some_and(|(_, word)| word.address.utterance_index.index() == utterance)
            {
                let (position, word) = population.next()?;
                same_source &= info.words.get(count) == Some(&word.text)
                    && word.address.word_index.fa_word() == talkbank_model::WordIdx::new(count);
                count += 1;
                last = Some(position);
            }
            // The census is exact, not a same-length claim. Excluded overlap
            // turns cannot acquire this capability from another turn's words.
            if info.retained_timing.is_some()
                || info.words.is_empty()
                || count != info.words.len()
                || !same_source
            {
                return None;
            }
            let first = first?;
            let last = last?;
            let (start_ms, end_ms) = ordered.interval(self.earliest[first], self.latest[last])?;
            Some(FaSearchEnvelope::Candidates(CandidateSearchEnvelope {
                words: info.words.clone(),
                start_ms,
                end_ms,
                scope: SearchScope::Document,
            }))
        })
    }
}

/// A source-bound search capability, deliberately not a word anchor, UTR
/// proposal, main-tier bullet or output-completion proof.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(untagged)]
pub(super) enum FaSearchEnvelope {
    Candidates(CandidateSearchEnvelope),
    OrderCorridor(FaOrderCorridor),
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(super) struct CandidateSearchEnvelope {
    words: Vec<String>,
    start_ms: u64,
    end_ms: u64,
    #[serde(skip_serializing_if = "SearchScope::is_document")]
    scope: SearchScope,
}

/// Missing words retain a search obligation, never fabricated timing. The
/// recording supplies an absent ceiling only when the request is grouped.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(super) struct FaOrderCorridor {
    words: Vec<String>,
    floor_ms: u64,
    ceiling_ms: Option<u64>,
    scope: OrderCorridorScope,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum OrderCorridorScope {
    OrderCorridor,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SearchScope {
    Document,
    /// Whole-source joint proof preserves within-speaker order; another
    /// speaker's provisional crop is not a retained bound for this turn.
    Interleaved {
        floor_ms: u64,
        ceiling_ms: Option<u64>,
    },
}

impl SearchScope {
    fn is_document(&self) -> bool {
        matches!(self, Self::Document)
    }
}

impl FaSearchEnvelope {
    /// Narrow only when ALL possible exact embeddings remain in the admitted
    /// corridor. A conflict leaves ordinary recovery/refusal intact; it never
    /// chooses whichever ambiguous occurrence happens to fit.
    pub(super) fn within(
        &self,
        words: &[FaWord],
        corridor: TimeSpan,
        recording_end_ms: u64,
    ) -> Option<TimeSpan> {
        let expected = match self {
            Self::Candidates(envelope) => &envelope.words,
            Self::OrderCorridor(envelope) => &envelope.words,
        };
        if words.len() != expected.len()
            || words
                .iter()
                .zip(expected)
                .any(|(actual, expected)| actual.text != *expected)
        {
            return None;
        }
        let envelope = match self {
            Self::Candidates(envelope) => envelope,
            Self::OrderCorridor(envelope) => {
                // This remains a proposal. GroupWindow alone admits containment
                // and the engine budget; contradictory bounds must not fall back
                // to a convenient but unsupported interpolation.
                return Some(TimeSpan::new(
                    envelope.floor_ms,
                    envelope.ceiling_ms.unwrap_or(recording_end_ms),
                ));
            }
        };
        match envelope.scope {
            SearchScope::Document
                if envelope.start_ms < corridor.start_ms || envelope.end_ms > corridor.end_ms =>
            {
                return None;
            }
            SearchScope::Interleaved {
                floor_ms,
                ceiling_ms,
            } if envelope.start_ms < floor_ms
                || ceiling_ms.is_some_and(|ceiling| envelope.end_ms > ceiling) =>
            {
                return None;
            }
            _ => {}
        }
        Some(TimeSpan::new(envelope.start_ms, envelope.end_ms))
    }

    pub(super) fn interleaved(
        words: Vec<String>,
        start_ms: u64,
        end_ms: u64,
        floor_ms: u64,
        ceiling_ms: Option<u64>,
    ) -> Option<Self> {
        (start_ms < end_ms
            && start_ms >= floor_ms
            && ceiling_ms.is_none_or(|ceiling| end_ms <= ceiling))
        .then_some(Self::Candidates(CandidateSearchEnvelope {
            words,
            start_ms,
            end_ms,
            scope: SearchScope::Interleaved {
                floor_ms,
                ceiling_ms,
            },
        }))
    }

    pub(super) fn order_corridor(
        words: Vec<String>,
        floor_ms: u64,
        ceiling_ms: Option<u64>,
    ) -> Self {
        Self::OrderCorridor(FaOrderCorridor {
            words,
            floor_ms,
            ceiling_ms,
            scope: OrderCorridorScope::OrderCorridor,
        })
    }
}
