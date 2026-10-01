//! Word anchors: transcript words utterance timing recovery heard.
//!
//! # What an anchor is
//!
//! UTR matches every transcript word it can to a token of the ASR stream, and
//! the token carries the time the recognizer heard it. A match that is EXACT
//! or CASE-INSENSITIVE, to a token holding exactly one word, is an acoustic
//! observation of that word: "this word was said between these two instants".
//! That is a [`WordAnchor`]. Forced-alignment grouping uses anchors to cut an
//! utterance whose window is longer than the engine's budget into pieces that
//! each fit (see `chat_ops::fa::split`), so every cut point is something the
//! recording was heard to contain rather than a time we chose.
//!
//! Two kinds of match are deliberately NOT anchors:
//!
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

use super::AsrTimingToken;
use super::evidence::{
    UtrAlignmentPlan, UtrLexicalRelation, UtrUtteranceAlignmentEvidence, UtrWordMatch,
};
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

    /// The anchor one UTR match provides: `Ok(None)` for a fuzzy match or a
    /// multi-word token (see the module docs), and an error for a token
    /// address the stream does not hold, which would mean the plan and the
    /// stream disagree.
    fn from_match(
        matched: &UtrWordMatch,
        tokens: &[AsrTimingToken],
    ) -> Result<Option<Self>, AnchorDisorder> {
        match matched.relation {
            UtrLexicalRelation::Exact | UtrLexicalRelation::CaseInsensitive => {}
            UtrLexicalRelation::Fuzzy { .. } => return Ok(None),
        }
        let word = matched.word.word_index.fa_word();
        let token_index = matched.token.token_index();
        let Some(token) = tokens.get(token_index) else {
            return Err(AnchorDisorder::TokenNotInStream {
                word,
                token: token_index,
                stream_len: tokens.len(),
            });
        };
        // Exactly one word in the token, so its interval is this word's.
        let mut words = token.text.split_whitespace();
        match (words.next(), words.next()) {
            (Some(_), None) => Ok(Some(Self {
                word,
                start: FileMs::new(token.start_ms),
                end: FileMs::new(token.end_ms),
            })),
            (None, _) | (Some(_), Some(_)) => Ok(None),
        }
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
    /// A match names a token the stream does not hold: the plan and the
    /// stream it was read against disagree.
    #[error("anchor for word {word} names token {token} of a {stream_len}-token stream")]
    TokenNotInStream {
        /// The matched word.
        word: WordIdx,
        /// The token ordinal the match named.
        token: usize,
        /// How many tokens the stream holds.
        stream_len: usize,
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
    /// every match was fuzzy or to a multi-word token.
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
        }
    }

    /// Read the anchors one UTR plan observed off the non-empty token stream
    /// it was built from. The only production constructor of an observed
    /// index.
    ///
    /// An utterance UTR matched appears with its anchors, with the reason
    /// they were refused, or as having no reliable match; one it did not
    /// match is absent.
    pub(in crate::chat_ops::fa::utr) fn from_plan(
        plan: &UtrAlignmentPlan,
        tokens: &[AsrTimingToken],
    ) -> Self {
        let mut by_utterance = BTreeMap::new();
        for utterance in &plan.utterances {
            match utterance {
                UtrUtteranceAlignmentEvidence::Matched {
                    utterance_index,
                    alignable_words,
                    matches,
                    ..
                } => {
                    let anchors: Result<Vec<WordAnchor>, AnchorDisorder> =
                        std::iter::once(&matches.first)
                            .chain(matches.rest.iter())
                            .filter_map(|matched| {
                                WordAnchor::from_match(matched, tokens).transpose()
                            })
                            .collect();
                    let state = match anchors {
                        Err(disorder) => UtteranceAnchorState::Refused(disorder),
                        Ok(anchors) if anchors.is_empty() => UtteranceAnchorState::NoReliableMatch,
                        Ok(anchors) => {
                            match UtteranceAnchors::admit(
                                AlignableWords::recorded(*alignable_words),
                                anchors,
                            ) {
                                Ok(admitted) => UtteranceAnchorState::Anchored(admitted),
                                Err(disorder) => UtteranceAnchorState::Refused(disorder),
                            }
                        }
                    };
                    by_utterance.insert(utterance_index.fa_utterance(), state);
                }
                // No match, so no observation to anchor at.
                UtrUtteranceAlignmentEvidence::Unmatched { .. }
                | UtrUtteranceAlignmentEvidence::ExcludedMarkedOverlap { .. }
                | UtrUtteranceAlignmentEvidence::NoAlignableWords { .. } => {}
            }
        }
        Self {
            state: Observation::Observed(by_utterance),
        }
    }

    /// An observed index holding the given anchor sets, for unit tests of
    /// grouping and splitting. Test-only, like [`WordAnchor::fixture`].
    #[cfg(test)]
    pub(crate) fn fixture(
        entries: impl IntoIterator<Item = (UtteranceIdx, Result<UtteranceAnchors, AnchorDisorder>)>,
    ) -> Self {
        Self {
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

    /// A match naming a token the stream does not hold is refused with that
    /// fact, never mistaken for a fuzzy match and silently skipped.
    #[test]
    fn a_match_to_a_token_outside_the_stream_is_refused() {
        use super::super::evidence::{
            NonEmptyUtrWordMatches, UtrAlignmentStrategy, UtrAsrTokenAddress, UtrAsrTokenOrdinal,
            UtrAsrWordOrdinal, UtrTimingProposal, UtrUtteranceOrdinal, UtrWordAddress,
            UtrWordMatch, UtrWordOrdinal,
        };
        let plan = UtrAlignmentPlan {
            strategy: UtrAlignmentStrategy::GlobalDp,
            utterances: vec![UtrUtteranceAlignmentEvidence::Matched {
                utterance_index: UtrUtteranceOrdinal(0),
                alignable_words: 2,
                matches: NonEmptyUtrWordMatches {
                    first: UtrWordMatch {
                        word: UtrWordAddress {
                            utterance_index: UtrUtteranceOrdinal(0),
                            word_index: UtrWordOrdinal(1),
                        },
                        token: UtrAsrTokenAddress {
                            token_index: UtrAsrTokenOrdinal(5),
                            word_index: UtrAsrWordOrdinal(0),
                        },
                        chat_text: "hello".to_owned(),
                        asr_text: "hello".to_owned(),
                        relation: UtrLexicalRelation::Exact,
                    },
                    rest: Vec::new(),
                },
                proposal: UtrTimingProposal::Positive {
                    start_ms: 0,
                    end_ms: 10,
                },
            }],
        };
        let tokens = [AsrTimingToken {
            text: "hello".to_owned(),
            start_ms: 0,
            end_ms: 10,
        }];
        let index = AnchorIndex::from_plan(&plan, &tokens);
        assert_eq!(
            index.lookup(UtteranceIdx::new(0)),
            AnchorLookup::Refused(&AnchorDisorder::TokenNotInStream {
                word: WordIdx::new(1),
                token: 5,
                stream_len: 1,
            })
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
