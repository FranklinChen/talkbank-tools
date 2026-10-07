//! Word anchors: transcript words utterance timing recovery heard.
//!
//! # What an anchor is
//!
//! UTR matches every transcript word it can to a token of the ASR stream, and
//! the token carries the time the recognizer heard it. A correspondence common
//! to EVERY optimal lexical alignment, EXACT or CASE-INSENSITIVE, to a token
//! holding exactly one word, is an admitted lexical observation. It is not an
//! independent proof of acoustic truth or transcript accuracy.
//! That is a [`WordAnchor`]. Forced-alignment grouping uses anchors to cut an
//! utterance whose window is longer than the engine's budget into pieces that
//! each fit (see `chat_ops::fa::split`), so every cut point is something the
//! recording was heard to contain rather than a time we chose.
//!
//! These kinds of match are deliberately NOT anchors:
//!
//! * an ambiguous selected match: optimal tie breaking is not correspondence;
//! * a FUZZY match, admitted by string similarity: it says the words look
//!   alike, not that this word was heard at that time;
//! * a match to a multi-word token (a provider SEGMENT rather than a word):
//!   the token's interval covers all its words, so it does not say when THIS
//!   word ended, and a cut at the token's end could fall after the next
//!   transcript word had already begun.
//!
//! Anchors come from the GLOBAL pass only. The two-pass strategy's second
//! pass matches excluded overlap utterances inside local windows to recover
//! their bullets; those local matches are not read for anchors.
//!
//! # Who can build what
//!
//! [`WordAnchor`], [`UtteranceAnchors`] and an observed [`AnchorIndex`] are
//! constructible only inside the UTR module, and the only production route is
//! [`AnchorIndex::from_plan`], called where the plan and the token stream that
//! produced it are both in hand (`run_global_utr`). Test fixtures exist under
//! `cfg(test)`. Grouping can read anchors and cannot make one, so a cut point
//! always traces back to a recognizer observation.

use std::collections::BTreeMap;

use talkbank_model::{UtteranceIdx, WordIdx};

use super::evidence::{UtrAlignmentPlan, UtrLexicalRelation, UtrUtteranceAlignmentEvidence};
use super::lexical::AdmittedUtrWordMatch;
use crate::chat_ops::fa::coordinates::FileMs;

/// How many alignable words an utterance has: the length of the word list
/// `collect_fa_words` extracts, which UTR matches and FA aligns.
///
/// A count, not an index; [`AlignableWords::holds`] is the one comparison
/// against a [`WordIdx`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AlignableWords(usize);

impl AlignableWords {
    /// The count of an extracted word list.
    pub fn of<T>(words: &[T]) -> Self {
        Self(words.len())
    }

    /// A count UTR recorded in its plan for the same extraction.
    pub(super) fn recorded(count: usize) -> Self {
        Self(count)
    }

    /// Whether `word` is one of these words.
    pub fn holds(self, word: WordIdx) -> bool {
        word.raw() < self.0
    }

    /// Whether `word` is the last of these words.
    pub fn is_last(self, word: WordIdx) -> bool {
        word.raw() + 1 == self.0
    }
}

impl std::fmt::Display for AlignableWords {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One transcript word the recognizer was heard to say, and when.
///
/// The word is addressed in forced alignment's own word space (`WordIdx`,
/// converted once through `UtrWordOrdinal::fa_word`), and the interval is the
/// ASR token's, in file coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WordAnchor {
    word: WordIdx,
    start: FileMs,
    end: FileMs,
}

impl WordAnchor {
    /// The anchored word's position among its utterance's alignable words.
    pub fn word(self) -> WordIdx {
        self.word
    }

    /// When the recognizer heard the word begin.
    pub fn start(self) -> FileMs {
        self.start
    }

    /// When the recognizer heard the word end: the instant an over-budget
    /// utterance may be cut at.
    pub fn end(self) -> FileMs {
        self.end
    }

    /// The anchor one UTR match provides: `None` for a fuzzy match or a
    /// multi-word token (see the module docs). Original timing is owned by
    /// the correspondence proof; no independent stream needs checking.
    fn from_match(admitted: &AdmittedUtrWordMatch) -> Option<Self> {
        let matched = admitted.matched();
        match matched.relation {
            UtrLexicalRelation::Exact
            | UtrLexicalRelation::CaseInsensitive
            | UtrLexicalRelation::TerminalPunctuation => {}
            UtrLexicalRelation::Fuzzy { .. } => return None,
        }
        let word = matched.word.word_index.fa_word();
        // The producer bound original single-word timing into the proof.
        admitted.word_timing().map(|timing| Self {
            word,
            start: FileMs::new(timing.start_ms),
            end: FileMs::new(timing.end_ms),
        })
    }

    /// An anchor for a unit test of a consumer. Test-only, so production
    /// anchors all come from a UTR plan.
    #[cfg(test)]
    pub(crate) fn fixture(word: usize, start_ms: u64, end_ms: u64) -> Self {
        Self {
            word: WordIdx::new(word),
            start: FileMs::new(start_ms),
            end: FileMs::new(end_ms),
        }
    }
}

/// Why one utterance's anchors were refused rather than used.
///
/// Refused, never reordered: anchors that are out of order in words or in
/// time say the match itself is suspect, and sorting them would turn that
/// evidence into a plausible-looking cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AnchorDisorder {
    /// A token whose end precedes its start.
    #[error("anchor for word {word} ends ({end}) before it starts ({start})")]
    Inverted {
        /// The anchored word.
        word: WordIdx,
        /// The token's start.
        start: FileMs,
        /// The token's end, which precedes it.
        end: FileMs,
    },
    /// An anchor names a word the utterance does not have.
    #[error("anchor for word {word} in an utterance of {alignable_words} alignable words")]
    BeyondWords {
        /// The anchored word.
        word: WordIdx,
        /// How many alignable words the utterance has.
        alignable_words: AlignableWords,
    },
    /// Two anchors not strictly increasing in word order.
    #[error("anchor for word {later} follows the anchor for word {earlier}")]
    WordOrder {
        /// The word anchored first.
        earlier: WordIdx,
        /// The word anchored next, at or before it.
        later: WordIdx,
    },
    /// A later word heard starting before an earlier word was heard ending.
    #[error(
        "word {later} was heard starting at {later_start}, before word {earlier} ended at {earlier_end}"
    )]
    TimeOrder {
        /// The earlier word.
        earlier: WordIdx,
        /// When the earlier word ended.
        earlier_end: FileMs,
        /// The later word.
        later: WordIdx,
        /// When the later word started.
        later_start: FileMs,
    },
}

/// One utterance's anchors, admitted as strictly increasing in word order and
/// monotone in time (each anchor ends no later than the next one starts).
///
/// Carries the utterance's alignable-word count as UTR saw it, so a consumer
/// can tell whether these anchors describe the words it holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UtteranceAnchors {
    anchors: Vec<WordAnchor>,
    alignable_words: AlignableWords,
}

impl UtteranceAnchors {
    /// The validating constructor: refuse, never reorder. Private to UTR.
    fn admit(
        alignable_words: AlignableWords,
        anchors: Vec<WordAnchor>,
    ) -> Result<Self, AnchorDisorder> {
        for anchor in &anchors {
            if anchor.start > anchor.end {
                return Err(AnchorDisorder::Inverted {
                    word: anchor.word,
                    start: anchor.start,
                    end: anchor.end,
                });
            }
            if !alignable_words.holds(anchor.word) {
                return Err(AnchorDisorder::BeyondWords {
                    word: anchor.word,
                    alignable_words,
                });
            }
        }
        for (earlier, later) in anchors.iter().zip(anchors.iter().skip(1)) {
            if earlier.word >= later.word {
                return Err(AnchorDisorder::WordOrder {
                    earlier: earlier.word,
                    later: later.word,
                });
            }
            if earlier.end > later.start {
                return Err(AnchorDisorder::TimeOrder {
                    earlier: earlier.word,
                    earlier_end: earlier.end,
                    later: later.word,
                    later_start: later.start,
                });
            }
        }
        Ok(Self {
            anchors,
            alignable_words,
        })
    }

    /// [`UtteranceAnchors::admit`] for tests of consumers, through the same
    /// validation.
    #[cfg(test)]
    pub(crate) fn fixture(
        alignable_words: usize,
        anchors: Vec<WordAnchor>,
    ) -> Result<Self, AnchorDisorder> {
        Self::admit(AlignableWords(alignable_words), anchors)
    }

    /// The anchors, in word order (which is also time order).
    pub fn anchors(&self) -> &[WordAnchor] {
        &self.anchors
    }

    /// How many alignable words the utterance had when UTR matched it.
    pub fn alignable_words(&self) -> AlignableWords {
        self.alignable_words
    }
}

/// What UTR observed about one utterance's anchors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorLookup<'a> {
    /// Recovery has nothing to say about this utterance: it did not run, had
    /// no tokens, or matched none of its words.
    NotRecovered,
    /// Recovery matched words of this utterance, but none is an anchor:
    /// matches were ambiguous, fuzzy or to multi-word tokens.
    NoReliableMatch,
    /// Reliable matches existed and were refused as a set.
    Refused(&'a AnchorDisorder),
    /// Admitted anchors.
    Anchored(&'a UtteranceAnchors),
}

/// What UTR concluded about one matched utterance.
#[derive(Debug, Clone, PartialEq, Eq)]
enum UtteranceAnchorState {
    NoReliableMatch,
    Refused(AnchorDisorder),
    Anchored(UtteranceAnchors),
}

/// Every utterance's anchors from one UTR pass, keyed by forced alignment's
/// utterance ordinal.
///
/// [`AnchorIndex::not_observed`] is what forced alignment uses when no UTR
/// pass with tokens ran: then no utterance can be split and grouping refuses
/// over-budget utterances exactly as `over_budget`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnchorIndex {
    state: Observation,
    search_envelopes: BTreeMap<UtteranceIdx, super::search::FaSearchEnvelope>,
}

/// Whether a token stream was ever matched.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Observation {
    /// No pass matched any token stream (none ran, or the stream was empty).
    NotObserved,
    /// A pass matched a non-empty stream; utterances it matched are present.
    Observed(BTreeMap<UtteranceIdx, UtteranceAnchorState>),
}

impl AnchorIndex {
    /// No anchors: UTR did not run, or had no tokens to match.
    pub fn not_observed() -> Self {
        Self {
            state: Observation::NotObserved,
            search_envelopes: BTreeMap::new(),
        }
    }

    /// Read the anchors one UTR plan observed off the non-empty token stream
    /// it was built from. The only production constructor of an observed
    /// index.
    ///
    /// An utterance UTR matched appears with its anchors, with the reason
    /// they were refused, or as having no reliable match; one it did not
    /// match is absent.
    pub(in crate::chat_ops::fa::utr) fn from_plan(plan: &UtrAlignmentPlan) -> Self {
        let mut by_utterance = BTreeMap::new();
        for utterance in &plan.utterances {
            let (utterance_index, alignable_words, anchors) = match utterance {
                UtrUtteranceAlignmentEvidence::Matched {
                    utterance_index,
                    alignable_words,
                    admitted_matches,
                    ..
                } => (
                    utterance_index,
                    alignable_words,
                    admitted_matches
                        .iter()
                        .filter_map(WordAnchor::from_match)
                        .collect::<Vec<_>>(),
                ),
                UtrUtteranceAlignmentEvidence::InteriorOnly {
                    utterance_index,
                    alignable_words,
                    admitted_matches,
                    ..
                } => (
                    utterance_index,
                    alignable_words,
                    admitted_matches
                        .iter()
                        .filter_map(WordAnchor::from_match)
                        .collect::<Vec<_>>(),
                ),
                // No match, so no observation to anchor at.
                UtrUtteranceAlignmentEvidence::SelectedOnly {
                    utterance_index, ..
                }
                | UtrUtteranceAlignmentEvidence::Refused {
                    utterance_index, ..
                } => {
                    by_utterance.insert(
                        utterance_index.fa_utterance(),
                        UtteranceAnchorState::NoReliableMatch,
                    );
                    continue;
                }
                // A retained utterance in a refused region was never
                // searched: recovery has nothing to say about its words.
                UtrUtteranceAlignmentEvidence::RetainedUnsearched { .. }
                | UtrUtteranceAlignmentEvidence::Unmatched { .. }
                | UtrUtteranceAlignmentEvidence::ExcludedMarkedOverlap { .. }
                | UtrUtteranceAlignmentEvidence::NoAlignableWords { .. } => continue,
            };
            let state = if anchors.is_empty() {
                UtteranceAnchorState::NoReliableMatch
            } else {
                match UtteranceAnchors::admit(AlignableWords::recorded(*alignable_words), anchors) {
                    Ok(admitted) => UtteranceAnchorState::Anchored(admitted),
                    Err(disorder) => UtteranceAnchorState::Refused(disorder),
                }
            };
            by_utterance.insert(utterance_index.fa_utterance(), state);
        }
        Self {
            state: Observation::Observed(by_utterance),
            search_envelopes: plan
                .search_envelopes
                .iter()
                .enumerate()
                .filter_map(|(index, envelope)| {
                    envelope
                        .as_ref()
                        .map(|envelope| (UtteranceIdx::new(index), envelope.clone()))
                })
                .collect(),
        }
    }

    /// An observed index holding the given anchor sets, for unit tests of
    /// grouping and splitting. Test-only, like [`WordAnchor::fixture`].
    #[cfg(test)]
    pub(crate) fn fixture(
        entries: impl IntoIterator<Item = (UtteranceIdx, Result<UtteranceAnchors, AnchorDisorder>)>,
    ) -> Self {
        Self {
            search_envelopes: BTreeMap::new(),
            state: Observation::Observed(
                entries
                    .into_iter()
                    .map(|(utterance, entry)| {
                        let state = match entry {
                            Ok(anchors) => UtteranceAnchorState::Anchored(anchors),
                            Err(disorder) => UtteranceAnchorState::Refused(disorder),
                        };
                        (utterance, state)
                    })
                    .collect(),
            ),
        }
    }

    /// Keep these anchors unless `newer` observed a token stream: a later
    /// pass that ran with no tokens (or did not run) says nothing about the
    /// words, so the earlier pass's anchors still describe them.
    pub fn superseded_by(&mut self, newer: AnchorIndex) {
        match newer.state {
            Observation::NotObserved => {}
            Observation::Observed(_) => *self = newer,
        }
    }

    /// What UTR observed about one utterance.
    pub fn lookup(&self, utterance: UtteranceIdx) -> AnchorLookup<'_> {
        match &self.state {
            Observation::NotObserved => AnchorLookup::NotRecovered,
            Observation::Observed(by_utterance) => match by_utterance.get(&utterance) {
                None => AnchorLookup::NotRecovered,
                Some(UtteranceAnchorState::NoReliableMatch) => AnchorLookup::NoReliableMatch,
                Some(UtteranceAnchorState::Refused(disorder)) => AnchorLookup::Refused(disorder),
                Some(UtteranceAnchorState::Anchored(anchors)) => AnchorLookup::Anchored(anchors),
            },
        }
    }

    /// Preserve all candidate occurrences and verify the exact source census.
    /// This grants only an FA search window, never a timing hint or cut point.
    pub(in crate::chat_ops::fa) fn search_window(
        &self,
        utterance: UtteranceIdx,
        words: &[crate::chat_ops::fa::FaWord],
        corridor: crate::chat_ops::fa::TimeSpan,
        recording: &crate::chat_ops::fa::coordinates::Recording,
    ) -> Option<crate::chat_ops::fa::TimeSpan> {
        if words.iter().enumerate().any(|(index, word)| {
            word.utterance_index != utterance || word.utterance_word_index != WordIdx::new(index)
        }) {
            return None;
        }
        self.search_envelopes
            .get(&utterance)?
            .within(words, corridor, recording.duration().get())
    }

    /// How many utterances have admitted anchors, for logging.
    pub fn anchored_utterances(&self) -> usize {
        match &self.state {
            Observation::NotObserved => 0,
            Observation::Observed(by_utterance) => by_utterance
                .values()
                .filter(|state| matches!(state, UtteranceAnchorState::Anchored(_)))
                .count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anchor(word: usize, start: u64, end: u64) -> WordAnchor {
        WordAnchor::fixture(word, start, end)
    }

    #[test]
    fn monotone_anchors_are_admitted_in_the_order_given() {
        let admitted = UtteranceAnchors::fixture(4, vec![anchor(0, 100, 300), anchor(2, 300, 500)])
            .expect("strictly increasing words, touching but not overlapping times");
        assert_eq!(
            admitted
                .anchors()
                .iter()
                .map(|a| a.word().raw())
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(admitted.alignable_words(), AlignableWords(4));
    }

    /// Anchors out of time order are refused with the pair that disagrees,
    /// never sorted into a plausible-looking set.
    #[test]
    fn anchors_out_of_time_order_are_refused_not_reordered() {
        let refused =
            UtteranceAnchors::fixture(3, vec![anchor(0, 1_000, 2_000), anchor(1, 500, 900)])
                .expect_err("the second word was heard before the first ended");
        assert_eq!(
            refused,
            AnchorDisorder::TimeOrder {
                earlier: WordIdx::new(0),
                earlier_end: FileMs::new(2_000),
                later: WordIdx::new(1),
                later_start: FileMs::new(500),
            }
        );
    }

    #[test]
    fn anchors_out_of_word_order_are_refused() {
        let refused = UtteranceAnchors::fixture(3, vec![anchor(1, 0, 10), anchor(1, 10, 20)])
            .expect_err("two anchors on one word");
        assert_eq!(
            refused,
            AnchorDisorder::WordOrder {
                earlier: WordIdx::new(1),
                later: WordIdx::new(1),
            }
        );
    }

    #[test]
    fn an_inverted_token_or_a_word_past_the_utterance_is_refused() {
        assert!(matches!(
            UtteranceAnchors::fixture(3, vec![anchor(0, 20, 10)]),
            Err(AnchorDisorder::Inverted { .. })
        ));
        assert_eq!(
            UtteranceAnchors::fixture(2, vec![anchor(2, 0, 10)]),
            Err(AnchorDisorder::BeyondWords {
                word: WordIdx::new(2),
                alignable_words: AlignableWords(2),
            })
        );
    }

    #[test]
    fn an_unobserved_index_recovers_nothing() {
        assert_eq!(
            AnchorIndex::not_observed().lookup(UtteranceIdx::new(0)),
            AnchorLookup::NotRecovered
        );
    }

    /// A later pass with no tokens keeps the earlier pass's anchors; one
    /// that observed a stream replaces them.
    #[test]
    fn only_an_observing_pass_supersedes_earlier_anchors() {
        let earlier = || {
            AnchorIndex::fixture([(
                UtteranceIdx::new(0),
                UtteranceAnchors::fixture(2, vec![anchor(0, 0, 10)]),
            )])
        };
        let mut kept = earlier();
        kept.superseded_by(AnchorIndex::not_observed());
        assert!(matches!(
            kept.lookup(UtteranceIdx::new(0)),
            AnchorLookup::Anchored(_)
        ));

        let mut replaced = earlier();
        replaced.superseded_by(AnchorIndex::fixture([]));
        assert_eq!(
            replaced.lookup(UtteranceIdx::new(0)),
            AnchorLookup::NotRecovered
        );
    }
}
