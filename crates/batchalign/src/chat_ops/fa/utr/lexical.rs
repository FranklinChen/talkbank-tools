//! Word matching retains the provider token that owns each timing interval.

use super::evidence::{UtrAsrTokenOrdinal, UtrAsrWordOrdinal};
use super::{
    AsrTimingToken, NonEmptyUtrWordMatches, UtrAsrTokenAddress, UtrTimingProposal, UtrWordAddress,
    UtrWordMatch, lexical_relation,
};

/// A lexical projection of one retained ASR stream.
/// Raw provider segments cannot be passed directly to word alignment.
pub(super) struct UtrLexicalStream<'source> {
    tokens: &'source [AsrTimingToken],
    words: Vec<UtrLexicalWord<'source>>,
}

/// A nonempty whitespace-delimited word borrowed from its provider token.
/// Only projection constructs the word and its original-stream address.
struct UtrLexicalWord<'source> {
    text: &'source str,
    address: UtrAsrTokenAddress,
}

impl<'source> UtrLexicalStream<'source> {
    pub(super) fn from_tokens(tokens: &'source [AsrTimingToken]) -> Self {
        Self::project(tokens, |_| true)
    }

    /// Filter in provider coordinates before projecting words, preserving
    /// original token ordinals even in a local overlap-recovery window.
    pub(super) fn within_window(
        tokens: &'source [AsrTimingToken],
        start_ms: u64,
        end_ms: u64,
    ) -> Self {
        Self::project(tokens, |token| {
            token.start_ms < end_ms && token.end_ms > start_ms
        })
    }

    fn project(
        tokens: &'source [AsrTimingToken],
        include: impl Fn(&AsrTimingToken) -> bool,
    ) -> Self {
        let words = tokens
            .iter()
            .enumerate()
            .filter(|(_, token)| include(token))
            .flat_map(|(token_index, token)| {
                token
                    .text
                    .split_whitespace()
                    .enumerate()
                    .map(move |(word_index, text)| UtrLexicalWord {
                        text,
                        address: UtrAsrTokenAddress {
                            token_index: UtrAsrTokenOrdinal(token_index),
                            word_index: UtrAsrWordOrdinal(word_index),
                        },
                    })
            })
            .collect();
        Self { tokens, words }
    }

    pub(super) fn texts(&self) -> Vec<String> {
        self.words.iter().map(|word| word.text.to_owned()).collect()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Convert a matched lexical position into original provider evidence.
    pub(super) fn matched_word(
        &self,
        lexical_index: usize,
        word: UtrWordAddress,
        chat_text: &str,
    ) -> UtrWordMatch {
        let lexical = &self.words[lexical_index];
        UtrWordMatch {
            word,
            token: lexical.address,
            chat_text: chat_text.to_owned(),
            asr_text: lexical.text.to_owned(),
            relation: lexical_relation(chat_text, lexical.text),
        }
    }

    /// Timing belongs to provider tokens, never interpolated lexical words.
    pub(super) fn proposal(&self, matches: &NonEmptyUtrWordMatches) -> UtrTimingProposal {
        let (first, last) = matches.token_extent();
        UtrTimingProposal::spanning(&self.tokens[first.index()], &self.tokens[last.index()])
    }

    /// Original measured/coarse interval for a local lexical match.
    pub(super) fn timing_token(&self, lexical_index: usize) -> &AsrTimingToken {
        &self.tokens[self.words[lexical_index].address.token_index()]
    }
}
