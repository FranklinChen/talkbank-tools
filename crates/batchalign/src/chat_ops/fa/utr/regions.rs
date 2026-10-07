//! Anchored regions: the utterance bullets a file already has split UTR's
//! correspondence search into independent, separately bounded problems.
//!
//! # Why regions exist
//!
//! One whole-file correspondence search has a fixed work budget. A long
//! recording with a partly timed transcript used to exceed it as ONE problem,
//! and every utterance in the file was then refused, including the hundreds
//! whose recovery needed only their timed neighbours. The bullets the file
//! already has are exactly the information that makes the problem small:
//! each one says which stretch of the recording its utterance occupies, so the
//! untimed utterances between two timed ones can only have been spoken
//! between them.
//!
//! # The partition
//!
//! An ANCHOR is a timed utterance carrying no overlap marker. Anchors whose
//! starts decrease in document order contradict each other, so the anchors
//! used are a longest chain whose starts never decrease ([`anchor_chain`]);
//! a timed utterance left out of that chain keeps its timing and is simply a
//! member of the region around it.
//!
//! Regions partition the utterances: a region begins at an anchor (or at the
//! first utterance) and owns every utterance up to the next anchor. Its search
//! problem also includes that next anchor as trailing CONTEXT, never owned:
//! the anchor's own words absorb its own ASR tokens there, and a turn just
//! before it may legally interleave with it. Its ASR window is every token
//! whose onset lies between the leading anchor's start and the trailing
//! anchor's end; the first region starts at the stream's edge and the last
//! one ends at it. Every token therefore lies in some region's window.
//!
//! Each region is solved with the same planners and the same proof types the
//! whole-file search used, within the same budgets. A region whose search
//! still exceeds its budget is refused as that region: its untimed utterances
//! are refused with the region and the budget named, its timed utterances
//! keep their timing, and every other region is unaffected.
//!
//! ```mermaid
//! flowchart LR
//!     A0["region 0: start .. anchor A1 end"] --> A1["region 1: A1 start .. anchor A2 end"]
//!     A1 --> A2["region 2: A2 start .. stream end"]
//! ```

use super::evidence::UtrUtteranceOrdinal;
use super::lexical::ObservedWordTiming;
use super::{AsrTimingToken, UtrUtteranceInfo};

/// One side of a region's ASR onset window.
///
/// A region edge is either a retained utterance bullet or the edge of the
/// token stream itself; there is no sentinel time standing for "unbounded".
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UtrRegionEdge {
    /// No anchor bounds this side: the window runs to the token stream's edge.
    StreamEdge,
    /// A retained utterance bullet bounds this side: its start for a floor,
    /// its end for a ceiling.
    Anchor {
        /// The anchoring utterance.
        utterance_index: UtrUtteranceOrdinal,
        /// The bullet edge used, in recording milliseconds.
        ms: u64,
    },
}

/// Which utterances one region owns and which ASR tokens it may match.
///
/// Constructed only by [`partition`], so a span always describes a real
/// region of one census: a nonempty, in-order utterance range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct UtrRegionSpan {
    /// First utterance the region owns.
    first_utterance: UtrUtteranceOrdinal,
    /// Last utterance the region owns (inclusive).
    last_utterance: UtrUtteranceOrdinal,
    /// Tokens with an onset before this are outside the region.
    onset_floor: UtrRegionEdge,
    /// Tokens with an onset at or after this are outside the region.
    onset_ceiling: UtrRegionEdge,
}

impl UtrRegionSpan {
    /// First utterance the region owns.
    pub fn first_utterance(&self) -> UtrUtteranceOrdinal {
        self.first_utterance
    }

    /// Last utterance the region owns (inclusive).
    pub fn last_utterance(&self) -> UtrUtteranceOrdinal {
        self.last_utterance
    }

    /// How many utterances the region owns.
    pub(super) fn owned(&self) -> usize {
        self.last_utterance.index() + 1 - self.first_utterance.index()
    }

    /// Whether a provider token with this onset belongs to the region's window.
    pub(super) fn admits_onset(&self, onset_ms: u64) -> bool {
        let above_floor = match self.onset_floor {
            UtrRegionEdge::StreamEdge => true,
            UtrRegionEdge::Anchor { ms, .. } => onset_ms >= ms,
        };
        let below_ceiling = match self.onset_ceiling {
            UtrRegionEdge::StreamEdge => true,
            UtrRegionEdge::Anchor { ms, .. } => onset_ms < ms,
        };
        above_floor && below_ceiling
    }

    /// Compact form for review decisions and logs.
    pub(super) fn describe(&self) -> String {
        let edge = |edge: UtrRegionEdge| match edge {
            UtrRegionEdge::StreamEdge => "stream_edge".to_owned(),
            UtrRegionEdge::Anchor {
                utterance_index,
                ms,
            } => format!("u{}@{ms}", utterance_index.index()),
        };
        format!(
            "utterances={}..={} onsets=[{},{})",
            self.first_utterance.index(),
            self.last_utterance.index(),
            edge(self.onset_floor),
            edge(self.onset_ceiling)
        )
    }
}

/// One region of a census: the span it owns plus the census range its search
/// reads (the owned utterances and, when one follows, the trailing anchor).
pub(super) struct UtrRegion {
    span: UtrRegionSpan,
    searched_end: usize,
}

impl UtrRegion {
    /// The owned span, as published in evidence. Production reads it from
    /// the region's [`SearchedCensus`], which carries it into the plan.
    #[cfg(test)]
    pub(super) fn span(&self) -> UtrRegionSpan {
        self.span
    }

    /// The census slice the region's search reads: owned utterances first,
    /// then the trailing anchor as unowned context when one exists, bound to
    /// the span it belongs to.
    pub(super) fn searched<'c>(&self, census: &'c [UtrUtteranceInfo]) -> SearchedCensus<'c> {
        SearchedCensus {
            span: self.span,
            infos: &census[self.span.first_utterance.index()..self.searched_end],
        }
    }

    /// The region's tokens, in stream order, with their original ordinals.
    pub(super) fn admits(&self, token: &AsrTimingToken) -> bool {
        self.span.admits_onset(token.start_ms)
    }
}

/// One region's searched census: its owned utterances then any trailing
/// context, bound to the region's span. Built only by [`UtrRegion::searched`],
/// so a region plan built from it cannot be published under another span.
#[derive(Debug, Clone, Copy)]
pub(super) struct SearchedCensus<'c> {
    span: UtrRegionSpan,
    infos: &'c [UtrUtteranceInfo],
}

impl<'c> SearchedCensus<'c> {
    /// The span whose search this is.
    pub(super) fn span(&self) -> UtrRegionSpan {
        self.span
    }

    /// The searched utterances, in the region's local numbering.
    pub(super) fn infos(&self) -> &'c [UtrUtteranceInfo] {
        self.infos
    }

    /// One value per searched utterance, in order. The only constructor of
    /// [`PerSearched`], so its values and these utterances cannot disagree
    /// in number.
    pub(super) fn map<T>(
        self,
        f: impl FnMut((usize, &'c UtrUtteranceInfo)) -> T,
    ) -> PerSearched<'c, T> {
        PerSearched {
            searched: self,
            values: self.infos.iter().enumerate().map(f).collect(),
        }
    }
}

/// One value per utterance of a [`SearchedCensus`], carried with it: the
/// pairing replaces parallel lists whose lengths nothing tied together.
pub(super) struct PerSearched<'c, T> {
    searched: SearchedCensus<'c>,
    values: Vec<T>,
}

impl<'c, T> PerSearched<'c, T> {
    /// The census these values belong to.
    pub(super) fn searched(&self) -> SearchedCensus<'c> {
        self.searched
    }

    /// The same utterances with each value transformed.
    pub(super) fn map_values<U>(self, f: impl FnMut(T) -> U) -> PerSearched<'c, U> {
        PerSearched {
            searched: self.searched,
            values: self.values.into_iter().map(f).collect(),
        }
    }

    /// Each searched utterance with its value, in order.
    pub(super) fn into_pairs(self) -> impl Iterator<Item = (&'c UtrUtteranceInfo, T)> {
        self.searched.infos.iter().zip(self.values)
    }
}

/// A timed, non-overlap utterance usable as a region boundary, with the
/// bullet that makes it one.
struct Anchor {
    index: usize,
    timing: ObservedWordTiming,
}

/// The bullet that lets an utterance bound a region: timed and not
/// overlap-marked. An overlap-marked utterance may legitimately start before
/// its predecessor ends, so its bullet does not separate what precedes it
/// from what follows.
fn anchor_timing(info: &UtrUtteranceInfo) -> Option<ObservedWordTiming> {
    if info.has_lazy_overlap || info.has_ca_overlap {
        return None;
    }
    info.retained_timing
}

/// A longest chain of anchor candidates whose starts never decrease in
/// document order, in increasing census order.
///
/// Patience sorting with predecessor links: O(n log n). Among chains of the
/// maximum length it returns the one patience sorting builds, which is
/// deterministic for a given census.
fn anchor_chain(census: &[UtrUtteranceInfo]) -> Vec<Anchor> {
    let candidates: Vec<Anchor> = census
        .iter()
        .enumerate()
        .filter_map(|(index, info)| anchor_timing(info).map(|timing| Anchor { index, timing }))
        .collect();
    // `tails[l]` is the candidate ending the best chain of length l + 1 found
    // so far (the one with the smallest final start).
    let mut tails: Vec<usize> = Vec::new();
    let mut previous: Vec<Option<usize>> = vec![None; candidates.len()];
    for (position, candidate) in candidates.iter().enumerate() {
        let start = candidate.timing.start_ms;
        // First tail whose start is strictly greater: non-decreasing chains.
        let length = tails.partition_point(|&tail| candidates[tail].timing.start_ms <= start);
        previous[position] = length.checked_sub(1).map(|before| tails[before]);
        if length == tails.len() {
            tails.push(position);
        } else {
            tails[length] = position;
        }
    }
    let mut chain = Vec::with_capacity(tails.len());
    let mut cursor = tails.last().copied();
    while let Some(position) = cursor {
        chain.push(position);
        cursor = previous[position];
    }
    chain.reverse();
    let mut candidates: Vec<Option<Anchor>> = candidates.into_iter().map(Some).collect();
    chain
        .into_iter()
        .filter_map(|position| candidates[position].take())
        .collect()
}

/// Partition a census into anchored regions, in document order.
///
/// Every utterance is owned by exactly one region; an empty census has none.
pub(super) fn partition(census: &[UtrUtteranceInfo]) -> Vec<UtrRegion> {
    let Some(last_utterance) = census.len().checked_sub(1) else {
        return Vec::new();
    };
    // An anchor at utterance 0 bounds nothing before it: the first region
    // always opens at the stream edge, so tokens before the first bullet
    // still belong to a region.
    let chain = anchor_chain(census);
    // The end of the anchor that opens the current region, if it is one: a
    // region's window never closes before its own leading anchor's words.
    let mut leading_end = chain
        .first()
        .filter(|anchor| anchor.index == 0)
        .map(|anchor| anchor.timing.end_ms);
    let boundaries: Vec<Anchor> = chain
        .into_iter()
        .filter(|anchor| anchor.index > 0)
        .collect();
    let mut regions = Vec::with_capacity(boundaries.len() + 1);
    let mut first = 0;
    let mut onset_floor = UtrRegionEdge::StreamEdge;
    for position in 0..=boundaries.len() {
        let next = boundaries.get(position);
        let (last, onset_ceiling, searched_end) = match next {
            Some(anchor) => (
                anchor.index - 1,
                UtrRegionEdge::Anchor {
                    utterance_index: UtrUtteranceOrdinal(anchor.index),
                    // Never below the trailing anchor's start (a degenerate
                    // bullet must not invert the window), nor below the
                    // leading anchor's end: anchors share a start, and a
                    // zero-length trailing bullet would otherwise close the
                    // window before the region's own leading words.
                    ms: anchor
                        .timing
                        .end_ms
                        .max(anchor.timing.start_ms)
                        .max(leading_end.unwrap_or(0)),
                },
                anchor.index + 1,
            ),
            None => (last_utterance, UtrRegionEdge::StreamEdge, census.len()),
        };
        regions.push(UtrRegion {
            span: UtrRegionSpan {
                first_utterance: UtrUtteranceOrdinal(first),
                last_utterance: UtrUtteranceOrdinal(last),
                onset_floor,
                onset_ceiling,
            },
            searched_end,
        });
        if let Some(anchor) = next {
            first = anchor.index;
            onset_floor = UtrRegionEdge::Anchor {
                utterance_index: UtrUtteranceOrdinal(anchor.index),
                ms: anchor.timing.start_ms,
            };
            leading_end = Some(anchor.timing.end_ms);
        }
    }
    regions
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(timing: Option<(u64, u64)>, overlap: bool) -> UtrUtteranceInfo {
        UtrUtteranceInfo {
            words: vec!["word".to_owned()],
            retained_timing: timing
                .map(|(start_ms, end_ms)| ObservedWordTiming { start_ms, end_ms }),
            has_lazy_overlap: overlap,
            has_ca_overlap: false,
            overlap_onset_fraction: None,
            speaker: "PAR".to_owned(),
            presence: crate::chat_ops::fa::RecordingPresence::InRecording,
            bottom_indices: Vec::new(),
            top_onsets: Vec::new(),
        }
    }

    fn owned(regions: &[UtrRegion]) -> Vec<(usize, usize)> {
        regions
            .iter()
            .map(|region| {
                (
                    region.span().first_utterance().index(),
                    region.span().last_utterance().index(),
                )
            })
            .collect()
    }

    #[test]
    fn regions_partition_every_utterance_at_anchors() {
        let census = [
            info(None, false),
            info(Some((1_000, 2_000)), false),
            info(None, false),
            info(None, false),
            info(Some((5_000, 6_000)), false),
            info(None, false),
        ];
        let regions = partition(&census);
        assert_eq!(owned(&regions), vec![(0, 0), (1, 3), (4, 5)]);
        // Owned utterances plus the trailing anchor as context.
        assert_eq!(regions[0].searched(&census).infos().len(), 2);
        assert_eq!(regions[1].searched(&census).infos().len(), 4);
        assert_eq!(regions[2].searched(&census).infos().len(), 2);
        // Windows: stream edge to the trailing anchor's end, then from each
        // leading anchor's start.
        assert!(regions[0].span().admits_onset(0));
        assert!(regions[0].span().admits_onset(1_999));
        assert!(!regions[0].span().admits_onset(2_000));
        assert!(!regions[1].span().admits_onset(999));
        assert!(regions[1].span().admits_onset(1_000));
        assert!(regions[1].span().admits_onset(5_999));
        assert!(regions[2].span().admits_onset(u64::MAX));
    }

    /// Two anchors may share a start; a zero-length trailing bullet at that
    /// instant must not close the window before the leading anchor's own
    /// span, which would leave the region's untimed utterances no tokens.
    #[test]
    fn a_degenerate_trailing_anchor_keeps_the_leading_anchors_span_in_the_window() {
        let census = [
            info(None, false),
            info(Some((1_000, 2_000)), false),
            info(None, false),
            info(Some((1_000, 1_000)), false),
            info(None, false),
        ];
        let regions = partition(&census);
        assert_eq!(owned(&regions), vec![(0, 0), (1, 2), (3, 4)]);
        assert!(regions[1].span().admits_onset(1_000));
        assert!(
            regions[1].span().admits_onset(1_999),
            "the leading anchor's span stays in its region's window"
        );
        assert!(!regions[1].span().admits_onset(2_000));
    }

    #[test]
    fn a_timed_first_utterance_still_opens_at_the_stream_edge() {
        let census = [info(Some((1_000, 2_000)), false), info(None, false)];
        let regions = partition(&census);
        assert_eq!(owned(&regions), vec![(0, 1)]);
        assert!(
            regions[0].span().admits_onset(0),
            "speech before the first bullet"
        );
    }

    #[test]
    fn contradictory_and_overlap_bullets_are_members_not_boundaries() {
        // u1 starts after u3 although it precedes it: the longest
        // non-decreasing chain keeps u2 and u3 and leaves u1 a member.
        // u4 is overlap-marked, so it never bounds a region.
        let census = [
            info(None, false),
            info(Some((9_000, 9_500)), false),
            info(Some((2_000, 3_000)), false),
            info(Some((4_000, 5_000)), false),
            info(Some((4_500, 4_800)), true),
            info(None, false),
        ];
        let chain: Vec<usize> = anchor_chain(&census)
            .iter()
            .map(|anchor| anchor.index)
            .collect();
        assert_eq!(chain, vec![2, 3]);
        assert_eq!(owned(&partition(&census)), vec![(0, 1), (2, 2), (3, 5)]);
    }

    #[test]
    fn an_untimed_census_is_one_region_over_the_whole_stream() {
        let census = [info(None, false), info(None, false)];
        let regions = partition(&census);
        assert_eq!(owned(&regions), vec![(0, 1)]);
        assert!(regions[0].span().admits_onset(0) && regions[0].span().admits_onset(u64::MAX));
        assert!(partition(&[]).is_empty());
    }
}
