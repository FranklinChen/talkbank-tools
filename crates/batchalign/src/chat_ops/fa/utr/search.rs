//! Complete lexical embeddings bound a search, never a selected timing.

use super::evidence::UtrUtteranceOrdinal;
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
                let (position, unit) = population.next()?;
                same_source &= info.units.texts().get(count) == Some(&unit.text)
                    && unit.address.unit_index.index() == count;
                count += 1;
                last = Some(position);
            }
            // The census is exact, not a same-length claim. Excluded overlap
            // turns cannot acquire this capability from another turn's units.
            // The envelope names the utterance's WORDS: it bounds forced
            // alignment, which aligns words.
            if info.retained_timing.is_some()
                || info.units.is_empty()
                || count != info.units.len()
                || !same_source
            {
                return None;
            }
            let first = first?;
            let last = last?;
            let (start_ms, end_ms) = ordered.interval(self.earliest[first], self.latest[last])?;
            Some(FaSearchEnvelope::Candidates(CandidateSearchEnvelope {
                words: info.words().to_vec(),
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

/// Missing words retain a search obligation, never fabricated timing. Each
/// edge of the corridor carries where it came from ([`CorridorFloor`],
/// [`CorridorCeiling`]): a measured neighbour, the recording's own edge, or
/// an overlap marker that withholds a bound. A corridor therefore cannot
/// start at the recording's start beside a measured neighbour and leave no
/// trace of why (2026-11-03; before this the floor was a bare `u64`, and an
/// unheard acknowledgement next to a measured turn searched from 0, exceeded
/// the engine budget in a long recording, and was refused before any
/// request).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(super) struct FaOrderCorridor {
    words: Vec<String>,
    floor: CorridorFloor,
    ceiling: CorridorCeiling,
    scope: OrderCorridorScope,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum OrderCorridorScope {
    OrderCorridor,
}

/// One bound a measured utterance supplies to a corridor: the edge value, in
/// file milliseconds, and the utterance it was read from. The ordinal is
/// region-local while a region's planner builds it and is placed into the
/// file's census by plan assembly ([`FaSearchEnvelope::placed_after`]), as
/// every other address in the plan is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub(in crate::chat_ops::fa) struct MeasuredBound {
    /// The bound, in file milliseconds.
    pub(in crate::chat_ops::fa) ms: u64,
    /// The utterance whose measurement supplied it.
    pub(in crate::chat_ops::fa) utterance: UtrUtteranceOrdinal,
}

impl MeasuredBound {
    /// The later of two bounds, `self` on a tie: the one rule for a floor,
    /// which only ever rises.
    pub(super) fn later(self, other: Self) -> Self {
        match self.ms >= other.ms {
            true => self,
            false => other,
        }
    }

    /// The earlier of two bounds, `self` on a tie: the one rule for a
    /// ceiling, which only ever falls.
    pub(super) fn earlier(self, other: Self) -> Self {
        match self.ms <= other.ms {
            true => self,
            false => other,
        }
    }

    fn placed_after(self, region_start: UtrUtteranceOrdinal) -> Self {
        Self {
            ms: self.ms,
            utterance: self.utterance.placed_after(region_start),
        }
    }
}

/// The lower edge of an order corridor, with its provenance.
///
/// These are the only ways a corridor can begin. `FileStart` is the value
/// exactly when no measured utterance precedes the searched one in its
/// region, so a floor at the recording's start beside a measured neighbour
/// is unrepresentable; a marked overlap says so by name instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(in crate::chat_ops::fa) enum CorridorFloor {
    /// No measured utterance precedes it in its region: the search starts
    /// at the recording's start, by absence.
    FileStart,
    /// The searched utterance is marked as overlapping the turn before it
    /// (`+<`, or a CA bottom overlap): the marker says the turn begins while
    /// the previous one is still going, so that turn's end is no floor, and
    /// a listened case showed its START is none either (a `+<` turn measured
    /// to begin half a second before the turn it overlaps; the written start
    /// is raised to it afterwards, but the search must not be cut there).
    /// No preceding turn's measurement and no same-speaker retained bullet is
    /// applied; the search starts at the recording's start by that evidence,
    /// not by absence. A required preceding word's earliest candidate onset
    /// may still raise it.
    MarkedOverlapUnbounded,
    /// The highest bound a measured utterance before it supplies: a retained
    /// bullet's end; the start of the latest provider token proved to one of
    /// a preceding utterance's words (the token's outer edge, never a
    /// boundary invented inside it); a required preceding word's earliest
    /// candidate onset; or, carried through an unmeasured turn that is marked
    /// as overlapping, an earlier turn's bullet start, which that marked turn
    /// can begin no earlier than.
    PrecedingMeasured(MeasuredBound),
}

impl CorridorFloor {
    /// The floor in file milliseconds: the recording's start where no
    /// measurement bounds it.
    pub(in crate::chat_ops::fa) fn ms(self) -> u64 {
        match self {
            Self::FileStart | Self::MarkedOverlapUnbounded => 0,
            Self::PrecedingMeasured(bound) => bound.ms,
        }
    }

    /// The floor raised to `bound` when `bound` is higher. A measured bound
    /// always replaces an unmeasured floor; two measured bounds keep the
    /// higher, the current one on a tie.
    pub(super) fn raised_to(self, bound: MeasuredBound) -> Self {
        match self {
            Self::PrecedingMeasured(current) => Self::PrecedingMeasured(current.later(bound)),
            Self::FileStart | Self::MarkedOverlapUnbounded => Self::PrecedingMeasured(bound),
        }
    }

    fn placed_after(self, region_start: UtrUtteranceOrdinal) -> Self {
        match self {
            Self::FileStart => Self::FileStart,
            Self::MarkedOverlapUnbounded => Self::MarkedOverlapUnbounded,
            Self::PrecedingMeasured(bound) => {
                Self::PrecedingMeasured(bound.placed_after(region_start))
            }
        }
    }
}

/// The upper edge of an order corridor, with its provenance: the mirror of
/// [`CorridorFloor`], except that the marker which withholds a following
/// bound sits on the FOLLOWING utterance, not on the searched one. The
/// searched utterance's own marker says nothing about its end: an unmarked
/// turn after it begins after it ends whatever it is marked with, so a
/// marked utterance takes a following measurement as its ceiling like any
/// other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(in crate::chat_ops::fa) enum CorridorCeiling {
    /// No measured utterance follows it in its region: the search runs to
    /// the recording's end, by absence.
    FileEnd,
    /// The next utterance is marked as overlapping the one before it, so its
    /// measurement bounds nothing here, and neither does any later one: the
    /// chain of following measurements stops at the named utterance.
    FollowingOverlaps {
        /// The marked utterance the chain stopped at.
        utterance: UtrUtteranceOrdinal,
    },
    /// The lowest bound a measured utterance after it supplies: a retained
    /// bullet's start; the end of the earliest provider token proved to one
    /// of a following utterance's words (its outer edge); or a required
    /// following word's latest candidate end.
    FollowingMeasured(MeasuredBound),
}

impl CorridorCeiling {
    /// The ceiling in file milliseconds: the recording's end where no
    /// measurement bounds it.
    pub(in crate::chat_ops::fa) fn ms(self, recording_end_ms: u64) -> u64 {
        match self {
            Self::FileEnd | Self::FollowingOverlaps { .. } => recording_end_ms,
            Self::FollowingMeasured(bound) => bound.ms,
        }
    }

    /// The ceiling lowered to `bound` when `bound` is lower; the mirror of
    /// [`CorridorFloor::raised_to`].
    pub(super) fn lowered_to(self, bound: MeasuredBound) -> Self {
        match self {
            Self::FollowingMeasured(current) => Self::FollowingMeasured(current.earlier(bound)),
            Self::FileEnd | Self::FollowingOverlaps { .. } => Self::FollowingMeasured(bound),
        }
    }

    fn placed_after(self, region_start: UtrUtteranceOrdinal) -> Self {
        match self {
            Self::FileEnd => Self::FileEnd,
            Self::FollowingOverlaps { utterance } => Self::FollowingOverlaps {
                utterance: utterance.placed_after(region_start),
            },
            Self::FollowingMeasured(bound) => {
                Self::FollowingMeasured(bound.placed_after(region_start))
            }
        }
    }
}

/// Where forced alignment's search window for an utterance came from, kept
/// beside the window so the request evidence can say so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::chat_ops::fa) enum SearchProvenance {
    /// Every exact embedding of the utterance's words in the provider
    /// stream: the window is their hull.
    Candidates,
    /// An order corridor between the utterance's measured neighbours, with
    /// each edge's provenance.
    OrderCorridor {
        /// Where the corridor begins.
        floor: CorridorFloor,
        /// Where it ends.
        ceiling: CorridorCeiling,
    },
}

/// A search window an envelope admitted for an utterance's words, with its
/// provenance. Grouping may still narrow the span (an admitted endpoint
/// anchor bounds its own side); the provenance names the envelope's edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::chat_ops::fa) struct SearchWindow {
    /// The admitted window.
    pub(in crate::chat_ops::fa) span: TimeSpan,
    /// Where it came from.
    pub(in crate::chat_ops::fa) provenance: SearchProvenance,
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
    ) -> Option<SearchWindow> {
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
                return Some(SearchWindow {
                    span: TimeSpan::new(envelope.floor.ms(), envelope.ceiling.ms(recording_end_ms)),
                    provenance: SearchProvenance::OrderCorridor {
                        floor: envelope.floor,
                        ceiling: envelope.ceiling,
                    },
                });
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
        Some(SearchWindow {
            span: TimeSpan::new(envelope.start_ms, envelope.end_ms),
            provenance: SearchProvenance::Candidates,
        })
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
        floor: CorridorFloor,
        ceiling: CorridorCeiling,
    ) -> Self {
        Self::OrderCorridor(FaOrderCorridor {
            words,
            floor,
            ceiling,
            scope: OrderCorridorScope::OrderCorridor,
        })
    }

    /// The same envelope with the utterances its edges name addressed in the
    /// file's census instead of a region's: plan assembly's one crossing for
    /// envelopes, beside the one it makes for evidence.
    pub(super) fn placed_after(self, region_start: UtrUtteranceOrdinal) -> Self {
        match self {
            Self::Candidates(envelope) => Self::Candidates(envelope),
            Self::OrderCorridor(FaOrderCorridor {
                words,
                floor,
                ceiling,
                scope,
            }) => Self::OrderCorridor(FaOrderCorridor {
                words,
                floor: floor.placed_after(region_start),
                ceiling: ceiling.placed_after(region_start),
                scope,
            }),
        }
    }
}
