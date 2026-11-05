//! Source ownership retained while candidate windows are merged and padded.

use std::ops::Range;

use talkbank_model::{UtteranceIdx, WordIdx};

use super::super::FaWord;
use super::super::coordinates::FaWindow;
use super::super::split::NonEmptyWords;

/// A localized utterance after its complete window was admitted to a recording.
/// Only the consuming localization transition constructs this owner.
pub(super) struct RecordedUtterance {
    utterance: UtteranceIdx,
    words: NonEmptyWords,
    window: FaWindow,
}

impl RecordedUtterance {
    pub(super) fn new(utterance: UtteranceIdx, words: NonEmptyWords, window: FaWindow) -> Self {
        Self {
            utterance,
            words,
            window,
        }
    }

    pub(super) fn window(&self) -> FaWindow {
        self.window
    }

    #[cfg(test)]
    pub(super) fn into_words(self) -> (UtteranceIdx, NonEmptyWords) {
        (self.utterance, self.words)
    }
}

/// The word range is created only when its source's words enter the population.
/// Appending populations translates ranges along with the words they select.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceSlice {
    utterance: UtteranceIdx,
    window: FaWindow,
    words: Range<usize>,
    first_word: WordIdx,
    last_word: WordIdx,
}

/// Canonical words and the retained source boundaries that own them.
/// There is no independently editable utterance list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GroupPopulation {
    words: Vec<FaWord>,
    sources: Vec<SourceSlice>,
}

impl GroupPopulation {
    pub(super) fn one(source: RecordedUtterance) -> Self {
        let first_word = source.words.first_word();
        let last_word = source.words.last_word();
        let words = source.words.into_vec();
        let range = 0..words.len();
        Self {
            words,
            sources: vec![SourceSlice {
                utterance: source.utterance,
                window: source.window,
                words: range,
                first_word,
                last_word,
            }],
        }
    }

    pub(super) fn append(&mut self, mut next: Self) {
        let offset = self.words.len();
        self.sources
            .extend(next.sources.into_iter().map(|source| SourceSlice {
                words: (source.words.start + offset)..(source.words.end + offset),
                ..source
            }));
        self.words.append(&mut next.words);
    }

    pub(super) fn words(&self) -> &[FaWord] {
        &self.words
    }

    pub(super) fn utterance_indices(&self) -> Vec<UtteranceIdx> {
        self.sources.iter().map(|source| source.utterance).collect()
    }

    pub(super) fn sources(&self) -> impl ExactSizeIterator<Item = SourceUtterance<'_>> {
        self.sources.iter().map(|source| SourceUtterance {
            utterance: source.utterance,
            window: source.window,
            words: &self.words[source.words.clone()],
            first_word: source.first_word,
            last_word: source.last_word,
        })
    }

    /// Downstream fixtures explicitly declare the same synthetic window for
    /// every utterance. Parsed-source planner tests must use real grouping.
    #[cfg(test)]
    pub(super) fn declared_fixture(
        window: FaWindow,
        words: Vec<FaWord>,
        utterances: Vec<UtteranceIdx>,
    ) -> Self {
        let mut cursor = 0;
        let sources = utterances
            .into_iter()
            .map(|utterance| {
                let start = cursor;
                while cursor < words.len() && words[cursor].utterance_index == utterance {
                    cursor += 1;
                }
                assert!(cursor > start, "fixture utterance must own words");
                SourceSlice {
                    utterance,
                    window,
                    words: start..cursor,
                    first_word: words[start].utterance_word_index,
                    last_word: words[cursor - 1].utterance_word_index,
                }
            })
            .collect();
        assert_eq!(cursor, words.len(), "fixture must account for every word");
        Self { words, sources }
    }
}

/// A view produced only by a population that owns both the words and window.
pub(crate) struct SourceUtterance<'a> {
    utterance: UtteranceIdx,
    window: FaWindow,
    words: &'a [FaWord],
    first_word: WordIdx,
    last_word: WordIdx,
}

impl<'a> SourceUtterance<'a> {
    pub(crate) fn first_word(&self) -> WordIdx {
        self.first_word
    }
    pub(crate) fn last_word(&self) -> WordIdx {
        self.last_word
    }
    pub(crate) fn utterance(&self) -> UtteranceIdx {
        self.utterance
    }
    pub(crate) fn window(&self) -> FaWindow {
        self.window
    }
    pub(crate) fn words(&self) -> &'a [FaWord] {
        self.words
    }
}
