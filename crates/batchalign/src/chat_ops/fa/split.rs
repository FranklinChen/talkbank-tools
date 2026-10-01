//! Splitting an over-budget utterance window at recovered word anchors.
//!
//! # The problem
//!
//! An aligner engine accepts windows up to a fixed budget (15 s for wav2vec:
//! one model forward pass covers the whole window). An utterance whose own
//! window exceeds it cannot be aligned in ONE request; most such utterances
//! are not too long to align, only too long for one request.
//!
//! # The cure
//!
//! Utterance timing recovery (UTR) has usually heard many of the utterance's
//! words already: each `WordAnchor` says "this word ended at this instant".
//! [`AnchoredSplit::plan`] cuts the utterance at the END of anchored words so
//! every piece fits the budget, and each piece is aligned as its own request.
//! Every cut point is therefore an acoustic observation; there is no path that
//! cuts at a time nobody heard.
//!
//! ```text
//!  window   |--------------------------- 40 s ---------------------------|
//!  anchors      a0   a1        a2      a3          a4        a5     a6
//!  cuts                         ^ (end of a2)        ^ (end of a4)
//!  pieces   |---- piece 0 -----|------ piece 1 ------|----- piece 2 -----|
//! ```
//!
//! The utterance stays ONE group (injection walks a group's words with one
//! cursor, so a group holding part of an utterance would desynchronize it);
//! the pieces are how that group is EXECUTED. See `crate::fa::units`.
//!
//! # When it refuses, and why that is evidence (A4)
//!
//! If two consecutive cut points (or a window edge and its nearest cut) are
//! farther apart than the budget, no cut at an anchor can keep both sides in
//! budget. On real data that stretch is minutes long and means UTR placed the
//! utterance across audio it does not belong to, so the refusal names the
//! widest such stretch and the utterance is recorded for review, never
//! aligned.
//!
//! # Tricky parts for a newcomer
//!
//! * The input is an [`OverBudgetWindow`], which only an over-budget window
//!   can become, so every split has at least two pieces.
//! * Each piece OWNS its words ([`NonEmptyWords`]), split off the utterance's
//!   list in order, and its window, split off the utterance window in order
//!   ([`FaWindow::split_at`]). Neither split can leave an empty side, so the
//!   pieces partition both the words and the window by construction.
//! * Only anchors whose end lies strictly inside the window, on a word that
//!   is not the utterance's last, can be cuts.
//! * Equal cut instants (zero-length tokens) collapse to the later word, so
//!   no piece has zero duration.

use batchalign_transform::decisions::{RefusedWindow, SplitWindow, UnusableAnchors};
use talkbank_model::{UtteranceIdx, WordIdx};

use super::FaWord;
use super::coordinates::{FaWindow, FileMs, Ms};
use super::utr::{AlignableWords, AnchorDisorder, AnchorLookup};

/// A window inside its recording that is longer than the engine budget.
///
/// Minted only by [`OverBudgetWindow::exceeding`], so holding one is the
/// proof that the window cannot be aligned in one request, which is what makes
/// a split of it at least two pieces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct OverBudgetWindow {
    window: FaWindow,
    budget: Ms,
}

impl OverBudgetWindow {
    /// The window, when it is longer than `budget`.
    pub(super) fn exceeding(window: FaWindow, budget: Ms) -> Option<Self> {
        (window.len() > budget).then_some(Self { window, budget })
    }

    /// The `over_budget` refusal of this window.
    fn over_budget(self) -> RefusedWindow {
        RefusedWindow::OverBudget {
            start_ms: self.window.audio_start().get(),
            end_ms: self.window.end().get(),
            budget_ms: self.budget.0,
        }
    }

    /// The `anchors_unusable` refusal of this window, for `cause`.
    fn anchors_unusable(self, cause: UnusableAnchors) -> RefusedWindow {
        RefusedWindow::AnchorsUnusable {
            start_ms: self.window.audio_start().get(),
            end_ms: self.window.end().get(),
            budget_ms: self.budget.0,
            cause,
        }
    }
}

/// A non-empty run of consecutive words of one utterance, in order.
///
/// The first and last word are held beside the list, set where the list is
/// built, so they are answered without a fallible lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NonEmptyWords {
    first: WordIdx,
    last: WordIdx,
    words: Vec<FaWord>,
}

impl NonEmptyWords {
    /// The list, when it holds at least one word.
    pub(super) fn from_vec(words: Vec<FaWord>) -> Option<Self> {
        let (first, last) = match (words.first(), words.last()) {
            (Some(first), Some(last)) => (first.utterance_word_index, last.utterance_word_index),
            (None, _) | (Some(_), None) => return None,
        };
        Some(Self { first, last, words })
    }

    /// The words, in order.
    pub(super) fn as_slice(&self) -> &[FaWord] {
        &self.words
    }

    /// How many words.
    pub(super) fn count(&self) -> AlignableWords {
        AlignableWords::of(&self.words)
    }

    /// The list, for a group that holds its words as one vector.
    pub(super) fn into_vec(self) -> Vec<FaWord> {
        self.words
    }

    /// Split after `word`: the words up to and including it, and the rest.
    /// When either side would be empty the list comes back unchanged as the
    /// error, so the caller does not cut there.
    fn split_after(mut self, word: WordIdx) -> Result<(Self, Self), Self> {
        let at = self
            .words
            .iter()
            .position(|candidate| candidate.utterance_word_index > word)
            .unwrap_or(self.words.len());
        let head_last = at
            .checked_sub(1)
            .and_then(|position| self.words.get(position))
            .map(|found| found.utterance_word_index);
        let tail_first = self.words.get(at).map(|found| found.utterance_word_index);
        match (head_last, tail_first) {
            (Some(head_last), Some(tail_first)) => {
                // `at` names an element (`tail_first` exists), so it is within
                // the list and `split_off` cannot panic.
                let tail = self.words.split_off(at);
                Ok((
                    Self {
                        first: self.first,
                        last: head_last,
                        words: self.words,
                    },
                    Self {
                        first: tail_first,
                        last: self.last,
                        words: tail,
                    },
                ))
            }
            (None, _) | (_, None) => Err(self),
        }
    }
}

/// One request's worth of an anchored split: its words and the window they
/// are aligned against. Constructible only by [`AnchoredSplit::plan`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnchoredPiece {
    words: NonEmptyWords,
    window: FaWindow,
}

impl AnchoredPiece {
    /// The piece's words, in order.
    pub fn words(&self) -> &[FaWord] {
        self.words.as_slice()
    }

    /// The piece's first word, as its index among the utterance's words.
    pub fn first_word(&self) -> WordIdx {
        self.words.first
    }

    /// The piece's last word, likewise.
    pub fn last_word(&self) -> WordIdx {
        self.words.last
    }

    /// The piece's audio window: within the engine budget, inside the
    /// utterance's window and so inside the recording.
    pub fn window(&self) -> FaWindow {
        self.window
    }
}

/// An over-budget utterance window, partitioned at anchored words into
/// pieces that each fit the budget.
///
/// The pieces' windows partition the utterance window (the first starts at
/// its start, each later one at the previous one's end, the last ends at its
/// end) and their words partition the utterance's words in order. `last` is a
/// field of its own, so there is always a last piece to pad.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnchoredSplit {
    utterance: UtteranceIdx,
    window: FaWindow,
    budget: Ms,
    earlier: Vec<AnchoredPiece>,
    last: AnchoredPiece,
}

/// The pieces of a split, in order: a nameable type so a group can iterate
/// them without boxing.
pub type Pieces<'s> =
    std::iter::Chain<std::slice::Iter<'s, AnchoredPiece>, std::iter::Once<&'s AnchoredPiece>>;

/// Why an over-budget window could not be split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum SplitRefusal {
    /// UTR has nothing to say about this utterance.
    NotRecovered,
    /// UTR matched words, but none reliably enough to anchor.
    NoReliableMatch,
    /// UTR's anchors for this utterance were refused as a set.
    AnchorsRefused(AnchorDisorder),
    /// The anchors were read off a different word list than the one grouping
    /// holds. Cannot happen while UTR and grouping share `collect_fa_words`
    /// and nothing edits words between them; refused rather than trusted.
    AnchorsDescribeOtherWords {
        /// Alignable words when UTR matched the utterance.
        anchored: AlignableWords,
        /// Alignable words grouping extracted.
        utterance: AlignableWords,
    },
    /// Anchors exist, but none can be a cut: each lies on the last word or at
    /// or outside the window's edges.
    NoInteriorCut,
    /// A stretch with no usable cut is longer than the budget.
    Gap {
        /// Where the stretch starts.
        from: FileMs,
        /// Where it ends.
        to: FileMs,
    },
}

/// A candidate cut: after `word`, at the instant that word was heard ending.
#[derive(Debug, Clone, Copy)]
struct Cut {
    word: WordIdx,
    at: FileMs,
}

impl AnchoredSplit {
    /// Plan the pieces for one over-budget utterance, or say why there are
    /// none.
    ///
    /// Consumes the utterance's words: on success the pieces own them. Greedy:
    /// each piece is extended to the furthest cut that keeps it within the
    /// budget.
    pub(super) fn plan(
        over: OverBudgetWindow,
        utterance: UtteranceIdx,
        words: NonEmptyWords,
        anchors: AnchorLookup<'_>,
    ) -> Result<Self, SplitRefusal> {
        let OverBudgetWindow { window, budget } = over;
        let anchors = match anchors {
            AnchorLookup::NotRecovered => return Err(SplitRefusal::NotRecovered),
            AnchorLookup::NoReliableMatch => return Err(SplitRefusal::NoReliableMatch),
            AnchorLookup::Refused(disorder) => {
                return Err(SplitRefusal::AnchorsRefused(*disorder));
            }
            AnchorLookup::Anchored(anchors) => anchors,
        };
        let count = words.count();
        if anchors.alignable_words() != count {
            return Err(SplitRefusal::AnchorsDescribeOtherWords {
                anchored: anchors.alignable_words(),
                utterance: count,
            });
        }

        // Interior cuts, strictly increasing in time. Anchors are admitted
        // monotone, so their ends never decrease; an equal end replaces the
        // previous cut with the later word, so no piece is zero-length.
        let mut cuts: Vec<Cut> = Vec::new();
        for anchor in anchors.anchors() {
            let inside = window.audio_start() < anchor.end() && anchor.end() < window.end();
            if count.is_last(anchor.word()) || !inside {
                continue;
            }
            let cut = Cut {
                word: anchor.word(),
                at: anchor.end(),
            };
            match cuts.last_mut() {
                Some(previous) if previous.at == cut.at => *previous = cut,
                Some(_) | None => cuts.push(cut),
            }
        }
        if cuts.is_empty() {
            return Err(SplitRefusal::NoInteriorCut);
        }

        // A4 before A2: if any stretch between neighbouring cut points is
        // longer than the budget, no split exists, and the widest such
        // stretch is what a reviewer needs to see.
        let boundaries = std::iter::once(window.audio_start())
            .chain(cuts.iter().map(|cut| cut.at))
            .chain(std::iter::once(window.end()))
            .collect::<Vec<_>>();
        let widest = boundaries.iter().zip(boundaries.iter().skip(1)).fold(
            None::<(FileMs, FileMs)>,
            |widest, (&from, &to)| match widest {
                Some((wf, wt)) if wt.since(wf) >= to.since(from) => Some((wf, wt)),
                Some(_) | None => Some((from, to)),
            },
        );
        if let Some((from, to)) = widest
            && to.since(from) > budget
        {
            return Err(SplitRefusal::Gap { from, to });
        }

        // Greedy: cut at a candidate exactly when the NEXT boundary would put
        // the current piece over budget. A cut is taken only when both the
        // window and the words split there with something on each side; a cut
        // that cannot is skipped, and the check after the loop catches any
        // piece that skipping left over budget.
        let mut earlier = Vec::new();
        let mut rest_window = window;
        let mut rest_words = words;
        for (position, cut) in cuts.iter().enumerate() {
            let next_boundary = cuts.get(position + 1).map_or(window.end(), |next| next.at);
            if next_boundary.since(rest_window.audio_start()) <= budget {
                continue;
            }
            let Ok((piece_window, after_window)) = rest_window.split_at(cut.at) else {
                continue;
            };
            match rest_words.split_after(cut.word) {
                Ok((piece_words, after_words)) => {
                    earlier.push(AnchoredPiece {
                        words: piece_words,
                        window: piece_window,
                    });
                    rest_window = after_window;
                    rest_words = after_words;
                }
                Err(unsplit) => rest_words = unsplit,
            }
        }
        let last = AnchoredPiece {
            words: rest_words,
            window: rest_window,
        };
        if let Some(over) = earlier
            .iter()
            .chain(std::iter::once(&last))
            .find(|piece| piece.window.len() > budget)
        {
            return Err(SplitRefusal::Gap {
                from: over.window.audio_start(),
                to: over.window.end(),
            });
        }

        Ok(Self {
            utterance,
            window,
            budget,
            earlier,
            last,
        })
    }

    /// [`AnchoredSplit::plan`] for a test of a consumer (dispatch, injection)
    /// that needs a real split; panics if the fixture does not split.
    #[cfg(test)]
    pub(crate) fn plan_for_test(
        window: FaWindow,
        budget: Ms,
        words: Vec<FaWord>,
        anchors: AnchorLookup<'_>,
    ) -> Self {
        let over =
            OverBudgetWindow::exceeding(window, budget).expect("fixture window is over budget");
        let words = NonEmptyWords::from_vec(words).expect("fixture words");
        let utterance = words.as_slice()[0].utterance_index;
        Self::plan(over, utterance, words, anchors).expect("fixture anchors split the window")
    }

    /// The utterance the pieces belong to.
    pub fn utterance(&self) -> &UtteranceIdx {
        &self.utterance
    }

    /// The whole utterance window the pieces partition.
    pub fn window(&self) -> FaWindow {
        self.window
    }

    /// The pieces, in word and time order.
    pub fn pieces(&self) -> Pieces<'_> {
        self.earlier.iter().chain(std::iter::once(&self.last))
    }

    /// How many pieces: at least two.
    pub fn piece_count(&self) -> usize {
        self.earlier.len() + 1
    }

    /// Pad the last piece into the silence before the next utterance, as a
    /// single group's window is padded, but never past the budget: half the
    /// gap, at most `max_extension`, at most what keeps the last piece within
    /// the budget, and never past the recording.
    pub(super) fn extend_into_trailing_gap(&mut self, next_start: FileMs, max_extension: Ms) {
        let gap = next_start.since(self.last.window.end());
        let headroom = self.budget.0.saturating_sub(self.last.window.len().0);
        let extension = Ms((gap.0 / 2).min(max_extension.0).min(headroom));
        self.last.window = self.last.window.extend_by(extension);
        self.window = self.window.extend_by(extension);
    }

    /// The decision recording that this window was aligned in pieces.
    pub(super) fn decision(&self) -> SplitWindow {
        SplitWindow {
            start_ms: self.window.audio_start().get(),
            end_ms: self.window.end().get(),
            budget_ms: self.budget.0,
            pieces: self.piece_count(),
        }
    }
}

impl SplitRefusal {
    /// The one conversion from a failed split to its evidence form.
    ///
    /// The mapping, all of it durable data:
    /// - `NotRecovered` -> `over_budget`: recovery has nothing to say about
    ///   this utterance, so the window is simply too long.
    /// - `NoReliableMatch`, `AnchorsRefused`, `AnchorsDescribeOtherWords`,
    ///   `NoInteriorCut` -> `anchors_unusable` with the matching cause:
    ///   recovery matched this utterance and its evidence could not be used.
    /// - `Gap` -> `anchor_gap`, the A4 review signal.
    pub(super) fn into_refused_window(self, over: OverBudgetWindow) -> RefusedWindow {
        match self {
            Self::NotRecovered => over.over_budget(),
            Self::NoReliableMatch => over.anchors_unusable(UnusableAnchors::NoReliableAnchors),
            Self::AnchorsRefused(_) => over.anchors_unusable(UnusableAnchors::AnchorsRefused),
            Self::AnchorsDescribeOtherWords { .. } => {
                over.anchors_unusable(UnusableAnchors::AnchorsDescribeOtherWords)
            }
            Self::NoInteriorCut => over.anchors_unusable(UnusableAnchors::NoInteriorCut),
            Self::Gap { from, to } => RefusedWindow::AnchorGap {
                start_ms: over.window.audio_start().get(),
                end_ms: over.window.end().get(),
                budget_ms: over.budget.0,
                gap_start_ms: from.get(),
                gap_end_ms: to.get(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_ops::fa::coordinates::Recording;
    use crate::chat_ops::fa::utr::{UtteranceAnchors, WordAnchor};

    const BUDGET: Ms = Ms(15_000);

    fn recording() -> Recording {
        Recording::of_duration(Ms(1_000_000)).expect("non-empty")
    }

    fn over(start: u64, end: u64) -> OverBudgetWindow {
        let window =
            FaWindow::within(&recording(), FileMs::new(start), FileMs::new(end)).expect("inside");
        OverBudgetWindow::exceeding(window, BUDGET).expect("fixture windows are over budget")
    }

    fn words(count: usize) -> NonEmptyWords {
        NonEmptyWords::from_vec(
            (0..count)
                .map(|index| FaWord {
                    utterance_index: UtteranceIdx::new(0),
                    utterance_word_index: WordIdx::new(index),
                    text: format!("w{index}"),
                })
                .collect(),
        )
        .expect("fixture words are non-empty")
    }

    fn anchored(word_count: usize, anchors: &[(usize, u64, u64)]) -> UtteranceAnchors {
        UtteranceAnchors::fixture(
            word_count,
            anchors
                .iter()
                .map(|&(word, start, end)| WordAnchor::fixture(word, start, end))
                .collect(),
        )
        .expect("fixture anchors are monotone")
    }

    fn plan(
        window: OverBudgetWindow,
        words: NonEmptyWords,
        anchors: AnchorLookup<'_>,
    ) -> Result<AnchoredSplit, SplitRefusal> {
        AnchoredSplit::plan(window, UtteranceIdx::new(0), words, anchors)
    }

    /// The pieces as (word count, start, end), for comparison.
    fn shape(split: &AnchoredSplit) -> Vec<(usize, u64, u64)> {
        split
            .pieces()
            .map(|piece| {
                (
                    piece.words().len(),
                    piece.window().audio_start().get(),
                    piece.window().end().get(),
                )
            })
            .collect()
    }

    /// An in-budget window cannot be offered for splitting at all.
    #[test]
    fn only_an_over_budget_window_can_be_split() {
        let window =
            FaWindow::within(&recording(), FileMs::new(0), FileMs::new(15_000)).expect("inside");
        assert_eq!(OverBudgetWindow::exceeding(window, BUDGET), None);
    }

    /// Dense anchors: cuts land at anchored word ends, every piece is within
    /// budget, and the pieces partition both the words and the window.
    #[test]
    fn cuts_at_anchor_ends_and_partitions_words_and_window() {
        // 10 words over 0..40 s, one anchor per word, each 4 s long.
        let anchors = anchored(
            10,
            &(0..10)
                .map(|w| (w, w as u64 * 4_000, w as u64 * 4_000 + 3_500))
                .collect::<Vec<_>>(),
        );
        let split = plan(over(0, 40_000), words(10), AnchorLookup::Anchored(&anchors))
            .expect("dense anchors split");

        // Greedy: each piece runs to the furthest anchor end within 15 s of
        // its start (15.5 s is one anchor too far for the first piece).
        assert_eq!(
            shape(&split),
            vec![
                (3, 0, 11_500),
                (3, 11_500, 23_500),
                (3, 23_500, 35_500),
                (1, 35_500, 40_000),
            ]
        );
        let pieces: Vec<&AnchoredPiece> = split.pieces().collect();
        assert!(pieces.iter().all(|piece| piece.window().len() <= BUDGET));
        let words_in_order: Vec<usize> = pieces
            .iter()
            .flat_map(|piece| piece.words())
            .map(|word| word.utterance_word_index.raw())
            .collect();
        assert_eq!(words_in_order, (0..10).collect::<Vec<_>>());
        for pair in pieces.windows(2) {
            assert_eq!(pair[0].window().end(), pair[1].window().audio_start());
            assert_eq!(pair[0].last_word().raw() + 1, pair[1].first_word().raw());
        }
        assert_eq!(pieces[0].window().audio_start(), FileMs::new(0));
        assert_eq!(pieces[3].window().end(), FileMs::new(40_000));
        assert_eq!(
            split.decision(),
            SplitWindow {
                start_ms: 0,
                end_ms: 40_000,
                budget_ms: 15_000,
                pieces: 4,
            }
        );
    }

    /// A4: a stretch with no anchor longer than the budget is refused with
    /// the widest such stretch, which is where a reviewer must look.
    #[test]
    fn a_gap_over_budget_is_refused_with_its_position() {
        let anchors = anchored(
            6,
            &[
                (0, 1_000, 2_000),
                (1, 2_000, 3_000),
                // Four minutes with nothing heard.
                (2, 243_000, 244_000),
                (3, 244_000, 245_000),
                (4, 245_000, 246_000),
            ],
        );
        let refusal = plan(over(0, 250_000), words(6), AnchorLookup::Anchored(&anchors))
            .expect_err("a four-minute stretch cannot be cut within budget");
        assert_eq!(
            refusal,
            SplitRefusal::Gap {
                from: FileMs::new(3_000),
                to: FileMs::new(244_000),
            }
        );
        assert_eq!(
            refusal.into_refused_window(over(0, 250_000)),
            RefusedWindow::AnchorGap {
                start_ms: 0,
                end_ms: 250_000,
                budget_ms: 15_000,
                gap_start_ms: 3_000,
                gap_end_ms: 244_000,
            }
        );
    }

    /// Recovery saying nothing about the utterance is `over_budget`; recovery
    /// whose matches are unusable is `anchors_unusable` with its cause.
    #[test]
    fn unrecovered_and_unusable_anchors_are_told_apart() {
        let not_recovered = plan(over(0, 20_000), words(3), AnchorLookup::NotRecovered)
            .expect_err("nothing to cut at");
        assert_eq!(
            not_recovered.into_refused_window(over(0, 20_000)),
            RefusedWindow::OverBudget {
                start_ms: 0,
                end_ms: 20_000,
                budget_ms: 15_000,
            }
        );
        let fuzzy_only = plan(over(0, 20_000), words(3), AnchorLookup::NoReliableMatch)
            .expect_err("fuzzy matches are not anchors");
        assert_eq!(
            fuzzy_only.into_refused_window(over(0, 20_000)),
            RefusedWindow::AnchorsUnusable {
                start_ms: 0,
                end_ms: 20_000,
                budget_ms: 15_000,
                cause: UnusableAnchors::NoReliableAnchors,
            }
        );
    }

    /// An anchor on the last word, or at the window's edge, cannot be a cut:
    /// either would leave a piece with no words or no audio.
    #[test]
    fn anchors_on_the_last_word_or_window_edge_are_not_cuts() {
        let anchors = anchored(3, &[(0, 0, 0), (2, 19_000, 20_000)]);
        let refusal = plan(over(0, 20_000), words(3), AnchorLookup::Anchored(&anchors))
            .expect_err("no interior cut");
        assert_eq!(refusal, SplitRefusal::NoInteriorCut);
        assert_eq!(
            refusal.into_refused_window(over(0, 20_000)),
            RefusedWindow::AnchorsUnusable {
                start_ms: 0,
                end_ms: 20_000,
                budget_ms: 15_000,
                cause: UnusableAnchors::NoInteriorCut,
            }
        );
    }

    #[test]
    fn anchors_read_off_another_word_list_are_refused() {
        let anchors = anchored(4, &[(1, 5_000, 6_000)]);
        assert_eq!(
            plan(over(0, 20_000), words(3), AnchorLookup::Anchored(&anchors)),
            Err(SplitRefusal::AnchorsDescribeOtherWords {
                anchored: AlignableWords::of(&[(); 4]),
                utterance: AlignableWords::of(&[(); 3]),
            })
        );
    }

    /// Two anchors ending at the same instant make one cut, after the later
    /// word, so no piece has zero duration.
    #[test]
    fn equal_cut_instants_collapse_to_the_later_word() {
        let anchors = anchored(4, &[(0, 9_000, 10_000), (1, 10_000, 10_000)]);
        let split = plan(over(0, 20_000), words(4), AnchorLookup::Anchored(&anchors))
            .expect("one cut at 10 s");
        assert_eq!(shape(&split), vec![(2, 0, 10_000), (2, 10_000, 20_000)]);
    }

    /// The last piece is padded into the following silence like a single
    /// group, but only as far as the budget allows.
    #[test]
    fn trailing_padding_stops_at_the_budget() {
        let anchors = anchored(4, &[(1, 9_000, 10_000)]);
        let mut split = plan(over(0, 24_000), words(4), AnchorLookup::Anchored(&anchors))
            .expect("one cut at 10 s");
        // Last piece is 14 s: one second of headroom, though the gap allows 2.
        split.extend_into_trailing_gap(FileMs::new(28_000), Ms(1_500));
        assert_eq!(shape(&split), vec![(2, 0, 10_000), (2, 10_000, 25_000)]);
        assert_eq!(split.window().end(), FileMs::new(25_000));
    }

    #[test]
    fn non_empty_words_split_only_with_words_on_both_sides() {
        let (head, tail) = words(3)
            .split_after(WordIdx::new(0))
            .expect("one word, then two");
        assert_eq!((head.first, head.last), (WordIdx::new(0), WordIdx::new(0)));
        assert_eq!((tail.first, tail.last), (WordIdx::new(1), WordIdx::new(2)));
        let whole = words(3)
            .split_after(WordIdx::new(2))
            .expect_err("nothing after the last word");
        assert_eq!(whole.as_slice().len(), 3);
    }
}
