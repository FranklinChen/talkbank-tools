//! What a segmenter decided for one request: where its words divide into
//! utterances, and how each utterance it made ends.
//!
//! A bare `Vec<usize>` of group ids says where the cuts fall and nothing
//! about the cut itself. The boundary model predicts a *kind* of sentence end
//! at every cut (period, question, exclamation), and that kind used to be
//! reduced to "a boundary" the moment assignments were derived from it, so
//! every utterance a split produced was written with a `.` whatever the model
//! said. [`UtteranceSegmentation`] keeps the two together: a segmentation
//! cannot be built from boundary evidence without its ends, and a segmenter
//! that states no end (Stanza's constituency boundaries, a worker that returns
//! bare assignments) says so with [`SegmentEnds::Unstated`] rather than by a
//! period nobody chose.

use serde::{Deserialize, Serialize};

/// A sentence-end kind the boundary model can predict: the closed set of
/// CHAT terminators its action vocabulary names.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryTerminator {
    /// `.`
    Period,
    /// `?`
    Question,
    /// `!`
    Exclamation,
}

/// How the segments of a segmentation end: what the segmenter said at its
/// cuts, and after its last word.
///
/// One statement for the whole segmentation rather than one per segment: a
/// boundary model predicts a terminator at every cut it makes, and a
/// segmenter that predicts none (a constituency parse, a worker's bare
/// assignments) predicts none at any cut, so there is no mixed case to
/// represent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentEnds {
    /// The boundary model's predictions.
    Predicted {
        /// The terminator predicted at each cut between consecutive
        /// segments, in order: one fewer than the segments.
        cuts: Vec<BoundaryTerminator>,
        /// The terminator predicted after the request's last word, when the
        /// model predicted one there; otherwise the last segment is open and
        /// keeps what its source gives it (the recognizer's punctuation or a
        /// synthesized period before CHAT, the parent's own terminator after).
        tail: Option<BoundaryTerminator>,
    },
    /// The segmenter stated boundaries and no terminator for them. Every
    /// segment keeps what its source gives it: after CHAT, the parent's own
    /// terminator on the last child and a period on any other; before CHAT,
    /// the recognizer's punctuation or a synthesized period.
    Unstated,
}

/// One request's segmentation: a group id per word, and how the segments
/// those ids make end.
///
/// Fields are private: the ends describe these assignments and are built
/// with them by the constructors below, so a consumer cannot be handed cuts
/// that do not belong to the groups it is cutting. Lengths are the request's
/// business: [`Self::from_boundaries`] takes one decision per request word,
/// and `unstated` is given assignments admission has already matched to the
/// request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UtteranceSegmentation {
    /// Group id per request word, parallel to the request words.
    assignments: Vec<usize>,
    ends: SegmentEnds,
}

impl UtteranceSegmentation {
    /// A segmentation whose segmenter stated no terminator.
    pub fn unstated(assignments: Vec<usize>) -> Self {
        Self {
            assignments,
            ends: SegmentEnds::Unstated,
        }
    }

    /// The segmentation a boundary model's per-word decisions state: word `i`
    /// ends its segment with `Some(kind)`, or continues it with `None`.
    ///
    /// Groups advance after every predicted end, exactly as the worker derives
    /// its assignments, so a segmentation built here agrees with the
    /// assignments the worker declared whenever its evidence does. A
    /// prediction after the last word is the tail's; the cuts are the
    /// predictions before it.
    pub fn from_boundaries(
        boundaries: impl IntoIterator<Item = Option<BoundaryTerminator>>,
    ) -> Self {
        let mut assignments = Vec::new();
        let mut cuts = Vec::new();
        let mut group = 0;
        for boundary in boundaries {
            assignments.push(group);
            if let Some(kind) = boundary {
                cuts.push(kind);
                group += 1;
            }
        }
        // A cut after the last word opens no segment: it is the tail's end.
        let tail = match assignments.last() {
            Some(&last_group) if last_group == group => None,
            Some(_) => cuts.pop(),
            None => None,
        };
        Self {
            assignments,
            ends: SegmentEnds::Predicted { cuts, tail },
        }
    }

    /// Group id per request word.
    pub fn assignments(&self) -> &[usize] {
        &self.assignments
    }

    /// The group ids alone, for a receipt that records what a worker
    /// declared; the ends are dropped with the value.
    pub fn into_assignments(self) -> Vec<usize> {
        self.assignments
    }

    /// How the segments end.
    pub fn ends(&self) -> &SegmentEnds {
        &self.ends
    }

    /// The terminator predicted at cut `cut` (between segments `cut` and
    /// `cut + 1`), or `None` when the segmenter stated none there.
    ///
    /// `None` also for a cut the segmentation does not have, which a caller
    /// iterating its own groups never asks for; the pre-CHAT split indexes
    /// cuts by the group changes of these same assignments.
    pub fn cut_end(&self, cut: usize) -> Option<BoundaryTerminator> {
        match &self.ends {
            SegmentEnds::Predicted { cuts, .. } => cuts.get(cut).copied(),
            SegmentEnds::Unstated => None,
        }
    }

    /// The terminator predicted after the last word, if any.
    pub fn tail_end(&self) -> Option<BoundaryTerminator> {
        match &self.ends {
            SegmentEnds::Predicted { tail, .. } => *tail,
            SegmentEnds::Unstated => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicted_ends_follow_their_cuts_and_an_open_tail_has_none() {
        let segmentation = UtteranceSegmentation::from_boundaries([
            None,
            Some(BoundaryTerminator::Question),
            None,
            None,
        ]);
        assert_eq!(segmentation.assignments(), &[0, 0, 1, 1]);
        assert_eq!(
            segmentation.ends(),
            &SegmentEnds::Predicted {
                cuts: vec![BoundaryTerminator::Question],
                tail: None,
            }
        );
        assert_eq!(segmentation.cut_end(0), Some(BoundaryTerminator::Question));
        assert_eq!(segmentation.cut_end(1), None);
        assert_eq!(segmentation.tail_end(), None);
    }

    #[test]
    fn a_prediction_after_the_last_word_is_the_tails_end() {
        let segmentation =
            UtteranceSegmentation::from_boundaries([None, Some(BoundaryTerminator::Exclamation)]);
        assert_eq!(segmentation.assignments(), &[0, 0]);
        assert_eq!(
            segmentation.ends(),
            &SegmentEnds::Predicted {
                cuts: Vec::new(),
                tail: Some(BoundaryTerminator::Exclamation),
            }
        );
    }

    #[test]
    fn an_unstated_segmentation_predicts_nothing_anywhere() {
        let segmentation = UtteranceSegmentation::unstated(vec![0, 0, 1, 1, 1]);
        assert_eq!(segmentation.ends(), &SegmentEnds::Unstated);
        assert_eq!(segmentation.cut_end(0), None);
        assert_eq!(segmentation.tail_end(), None);
        assert_eq!(
            UtteranceSegmentation::from_boundaries([]).ends(),
            &SegmentEnds::Predicted {
                cuts: Vec::new(),
                tail: None
            }
        );
    }
}
