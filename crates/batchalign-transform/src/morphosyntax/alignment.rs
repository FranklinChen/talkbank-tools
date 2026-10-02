//! The one walk of a UD sentence, and its alignment to the CHAT words it was
//! requested for.
//!
//! Five integer spaces meet in the morphosyntax path, and they used to travel
//! as bare `usize`:
//!
//! | Space | Base | Sequence |
//! |-------|------|----------|
//! | CHAT word index ([`MorItemIndex`]) | 0 | alignable words of the host utterance, one `%mor` item each |
//! | span word position ([`SpanWordPosition`]) | 0 | words of one dispatched secondary span |
//! | token position ([`UdTokenIndex`]) | 0 | top-level tokens of a walked sentence: a word or a complete multi-word token |
//! | UD word id ([`UdWordId`]) | 1 | syntactic words of a UD sentence (`ID` column) |
//! | position in `UdSentence::words` | 0 | UD rows, where multi-word-token RANGE rows sit beside their components |
//!
//! They coincide only on an utterance with no contraction, which is why
//! code that confused them passed every test written on English. An `@s`
//! word after `al` (`a` + `el`) is CHAT word 3, UD word 5, and row 5;
//! reading row 3 reads the `a` of `al`.
//!
//! [`UdTokens::walk`] is the ONE place a UD sentence is walked: a single word
//! or a complete multi-word token per top-level token, empty nodes skipped, a
//! terminator-punctuation word skipped. The `%mor` mapper
//! (`sentence_mapping::map_tokens`) consumes the walk, and so does every
//! alignment, so the two cannot drift. The walk validates what every later
//! lookup relies on, so that nothing downstream re-checks it:
//!
//! - every multi-word token's components follow its range row with the
//!   range's ids;
//! - no UD word id appears twice, and none is 0;
//! - every head is the root or a word of the walk;
//! - every multi-word token has a component whose head lies outside the
//!   token, its REPRESENTATIVE (see [`AlignedWord::representative`]).
//!
//! [`UdTokens::align`] then pairs the tokens with CHAT words, one each
//! ([`UdAlignment`]). After that, lookups go by UD id through a private
//! table, never by row position, and the word spaces cannot be mixed: the
//! alignment is generic over its CHAT-side space, and a value of one space
//! cannot be passed where another is expected.
//!
//! ```mermaid
//! flowchart LR
//!     S["UdSentence"] -->|UdTokens::walk| T["UdTokens<br/>UdTokenIndex"]
//!     T -->|map_tokens| M["%mor items + %gra"]
//!     T -->|align| A["UdAlignment&lt;W&gt;"]
//!     A -->|word| W["aligned CHAT word<br/>MorItemIndex / SpanWordPosition"]
//!     W -->|representative| R["UD word<br/>UdWordId"]
//!     R -->|head| H["HeadTarget<br/>Root or CHAT word"]
//! ```

use std::collections::BTreeMap;
use std::marker::PhantomData;

use talkbank_model::alignment::MorItemIndex;

use crate::morphosyntax::{UdHead, UdId, UdSentence, UdWord, UdWordId, is_terminator_punct};

/// Where a word's head lies, in the CHAT-side space of the alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadTarget<W> {
    /// The word is the sentence root.
    Root,
    /// The head is (part of) this CHAT word.
    Word(W),
}

/// Position of a word within one dispatched secondary span (0-based).
///
/// A distinct space from [`MorItemIndex`]: position 0 of a span is
/// whatever host word the span starts at. Only an alignment mints one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpanWordPosition(usize);

impl SpanWordPosition {
    /// The raw position, for diagnostics.
    pub fn as_usize(self) -> usize {
        self.0
    }
}

mod private {
    /// Minting and reading a word-space index. Private, so only this
    /// module can produce a [`super::SpanWordPosition`] from a bare
    /// integer.
    pub trait Mint: Copy {
        fn mint(position: usize) -> Self;
        fn position(self) -> usize;
    }
}

/// A CHAT-side word space an alignment can be indexed by.
pub trait WordSpace: private::Mint + std::fmt::Debug + Eq {}

impl private::Mint for MorItemIndex {
    fn mint(position: usize) -> Self {
        MorItemIndex::new(position)
    }
    fn position(self) -> usize {
        self.as_usize()
    }
}
impl WordSpace for MorItemIndex {}

impl private::Mint for SpanWordPosition {
    fn mint(position: usize) -> Self {
        Self(position)
    }
    fn position(self) -> usize {
        self.0
    }
}
impl WordSpace for SpanWordPosition {}

/// The 0-based position of a top-level token in a walked sentence: a single
/// word, or a complete multi-word token, in order, terminator and empty nodes
/// skipped. Only [`UdTokens::walk`] mints one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UdTokenIndex(usize);

impl UdTokenIndex {
    /// The raw position, for diagnostics and for indexing what the mapper
    /// produced per token.
    pub fn as_usize(self) -> usize {
        self.0
    }
}

/// What one top-level token covers in the UD sentence.
#[derive(Debug, Clone, Copy)]
pub enum AlignedUd<'s> {
    /// One syntactic word.
    Word(&'s UdWord),
    /// A multi-word token (`it's` = `it` + `'s`): its range row and its
    /// syntactic components, in order.
    Mwt {
        /// The range row (`UdId::Range`).
        range: &'s UdWord,
        /// The component words (`UdId::Single`), ids `start..=end`.
        components: &'s [UdWord],
    },
}

impl<'s> AlignedUd<'s> {
    /// The syntactic words of the token, in order: the word itself, or the
    /// components of a multi-word token.
    pub fn words(&self) -> &'s [UdWord] {
        match *self {
            Self::Word(word) => std::slice::from_ref(word),
            Self::Mwt { components, .. } => components,
        }
    }
}

/// Where a token's representative attaches, in token positions: resolved at
/// the walk, so no later lookup can miss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenHead {
    Root,
    Token(UdTokenIndex),
}

/// One walked token, with its representative and that word's head resolved.
#[derive(Debug, Clone, Copy)]
struct WalkedToken<'s> {
    ud: AlignedUd<'s>,
    representative: &'s UdWord,
    representative_id: UdWordId,
    head: TokenHead,
    head_word: Option<&'s UdWord>,
}

/// A borrowed token admitted by [`UdTokens::walk`], paired with the
/// representative that the same walk selected for its external attachment.
/// Only the producer can construct this view; consumers cannot pair a token
/// with a representative from another token or sentence.
#[derive(Debug, Clone, Copy)]
pub struct WalkedUdToken<'a, 's> {
    token: &'a WalkedToken<'s>,
}

impl<'s> WalkedUdToken<'_, 's> {
    /// The admitted token and, for a multiword token, its components.
    pub fn ud(self) -> AlignedUd<'s> {
        self.token.ud
    }

    /// The admitted syntactic words covered by this token.
    pub fn words(self) -> &'s [UdWord] {
        self.token.ud.words()
    }

    /// The syntactic word selected by the producer for this token's attachment.
    pub fn representative(self) -> &'s UdWord {
        self.token.representative
    }
}

/// A token walked but not yet head-resolved (heads may point forward).
struct UnresolvedToken<'s> {
    ud: AlignedUd<'s>,
    representative: &'s UdWord,
    representative_id: UdWordId,
}

/// Why a UD sentence cannot be walked: it is structurally malformed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UdSentenceError {
    /// A multi-word token's range row is not followed by its components.
    #[error("multi-word token {start}-{end} is not followed by its components")]
    MalformedMwt {
        /// First id of the range.
        start: usize,
        /// Last id of the range.
        end: usize,
    },
    /// A syntactic word has id 0, which UD reserves for the root.
    #[error("a UD word has id 0")]
    ZeroId,
    /// Two syntactic words share an id.
    #[error("UD word id {id} appears twice")]
    DuplicateId {
        /// The repeated id.
        id: usize,
    },
    /// A word's head is neither the root nor a word of the walk (it may
    /// point at a terminator, which is not walked, or past the end).
    #[error("UD word {word} has head {head}, which is not a walked word")]
    HeadOutsideSentence {
        /// The dependent's id.
        word: usize,
        /// Its head id.
        head: usize,
    },
    /// No component of a multi-word token attaches outside it, so the
    /// token has no representative.
    #[error("no component of multi-word token {start}-{end} attaches outside it")]
    MwtWithoutExternalHead {
        /// First id of the range.
        start: usize,
        /// Last id of the range.
        end: usize,
    },
}

/// Why a UD sentence cannot be aligned to its CHAT words.
///
/// In the L2 path every variant sends the affected words to the `L2|xxx`
/// fallback, with a report naming the variant.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UdAlignmentError {
    /// The sentence itself is malformed.
    #[error(transparent)]
    Sentence(#[from] UdSentenceError),
    /// The sentence has a different number of top-level UD tokens than
    /// there are CHAT words.
    #[error("{ud_tokens} UD tokens for {chat_words} CHAT words")]
    WordCountMismatch {
        /// CHAT words the analysis was requested for.
        chat_words: usize,
        /// Top-level UD tokens found (words and complete multi-word tokens).
        ud_tokens: usize,
    },
}

/// A UD sentence walked once into its top-level tokens, validated (see the
/// module documentation). The `%mor` mapper and every alignment consume this
/// one walk.
#[derive(Debug)]
pub struct UdTokens<'s> {
    /// One token per top-level token, in order.
    tokens: Vec<WalkedToken<'s>>,
    /// Every syntactic word by UD id, with the token it is in.
    by_id: BTreeMap<UdWordId, (UdTokenIndex, &'s UdWord)>,
}

impl<'s> UdTokens<'s> {
    /// Walk `sentence`, validating every fact the lookups rely on.
    pub fn walk(sentence: &'s UdSentence) -> Result<Self, UdSentenceError> {
        let mut walked: Vec<UnresolvedToken<'s>> = Vec::new();
        let mut by_id: BTreeMap<UdWordId, (UdTokenIndex, &'s UdWord)> = BTreeMap::new();
        let mut rows = sentence.words.as_slice();

        while let Some((row, rest)) = rows.split_first() {
            rows = rest;
            let at = UdTokenIndex(walked.len());
            match row.id {
                UdId::Decimal(_) => {}
                UdId::Single(_) if is_terminator_punct(row) => {}
                UdId::Single(_) => {
                    let id = record(&mut by_id, row, at)?;
                    walked.push(UnresolvedToken {
                        ud: AlignedUd::Word(row),
                        representative: row,
                        representative_id: id,
                    });
                }
                UdId::Range(start, end) => {
                    let malformed = UdSentenceError::MalformedMwt { start, end };
                    let count = match end.checked_sub(start) {
                        Some(span) => span + 1,
                        None => return Err(malformed),
                    };
                    if rows.len() < count {
                        return Err(malformed);
                    }
                    let (components, rest) = rows.split_at(count);
                    rows = rest;
                    let mut representative = None;
                    for (expected, component) in (start..=end).zip(components) {
                        if component.id != UdId::Single(expected) {
                            return Err(malformed);
                        }
                        let id = record(&mut by_id, component, at)?;
                        let attaches_inside = component
                            .head
                            .word()
                            .is_some_and(|head| (start..=end).contains(&head.get()));
                        if representative.is_none() && !attaches_inside {
                            representative = Some((component, id));
                        }
                    }
                    let Some((representative, representative_id)) = representative else {
                        return Err(UdSentenceError::MwtWithoutExternalHead { start, end });
                    };
                    walked.push(UnresolvedToken {
                        ud: AlignedUd::Mwt {
                            range: row,
                            components,
                        },
                        representative,
                        representative_id,
                    });
                }
            }
        }

        // Every head, not only the representatives', must resolve: a
        // dependent read by `AlignedWord::dependents` is trusted too, and the
        // `%gra` builder resolves every component's head.
        for (id, (_, word)) in &by_id {
            if let UdHead::Word(head) = word.head
                && !by_id.contains_key(&head)
            {
                return Err(UdSentenceError::HeadOutsideSentence {
                    word: id.get(),
                    head: head.get(),
                });
            }
        }

        let tokens = walked
            .into_iter()
            .map(|token| {
                let (head, head_word) = match token.representative.head {
                    UdHead::Root => (TokenHead::Root, None),
                    UdHead::Word(head) => {
                        let (at, word) = by_id.get(&head).copied().ok_or(
                            UdSentenceError::HeadOutsideSentence {
                                word: token.representative_id.get(),
                                head: head.get(),
                            },
                        )?;
                        (TokenHead::Token(at), Some(word))
                    }
                };
                Ok(WalkedToken {
                    ud: token.ud,
                    representative: token.representative,
                    representative_id: token.representative_id,
                    head,
                    head_word,
                })
            })
            .collect::<Result<Vec<_>, UdSentenceError>>()?;

        Ok(Self { tokens, by_id })
    }

    /// How many top-level tokens the sentence has.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the sentence has no top-level token.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Every top-level token, in order.
    pub fn iter(&self) -> impl Iterator<Item = (UdTokenIndex, WalkedUdToken<'_, 's>)> + '_ {
        self.tokens
            .iter()
            .enumerate()
            .map(|(position, token)| (UdTokenIndex(position), WalkedUdToken { token }))
    }

    /// Where a syntactic word is, by its UD id: its token, and its place
    /// among the token's words (0 for a single word; a multi-word token's
    /// components count from its first).
    pub fn locate(&self, id: UdWordId) -> Option<(UdTokenIndex, usize)> {
        let (at, _) = self.by_id.get(&id)?;
        let token = self.tokens.get(at.0)?;
        let offset = token
            .ud
            .words()
            .iter()
            .position(|word| word.id == UdId::Single(id.get()))?;
        Some((*at, offset))
    }

    /// Pair the tokens with `chat_words` CHAT words, one each.
    pub fn align<W: WordSpace>(
        self,
        chat_words: usize,
    ) -> Result<UdAlignment<'s, W>, UdAlignmentError> {
        if self.tokens.len() != chat_words {
            return Err(UdAlignmentError::WordCountMismatch {
                chat_words,
                ud_tokens: self.tokens.len(),
            });
        }
        Ok(UdAlignment {
            tokens: self,
            space: PhantomData,
        })
    }
}

/// A UD sentence aligned to the CHAT words it was requested for: one
/// top-level token per CHAT word.
///
/// `W` is the CHAT-side space: [`MorItemIndex`] for a host utterance,
/// [`SpanWordPosition`] for a dispatched secondary span.
#[derive(Debug)]
pub struct UdAlignment<'s, W> {
    tokens: UdTokens<'s>,
    space: PhantomData<fn() -> W>,
}

impl<'s, W: WordSpace> UdAlignment<'s, W> {
    /// Walk `sentence` and align it to `chat_words` CHAT words.
    pub fn new(sentence: &'s UdSentence, chat_words: usize) -> Result<Self, UdAlignmentError> {
        UdTokens::walk(sentence)?.align(chat_words)
    }

    /// The walk this alignment is over, for the `%mor` mapper.
    pub fn tokens(&self) -> &UdTokens<'s> {
        &self.tokens
    }

    /// The aligned token for one CHAT word, or `None` past the end.
    pub fn word(&self, index: W) -> Option<AlignedWord<'_, 's, W>> {
        let position = UdTokenIndex(index.position());
        self.tokens.tokens.get(position.0).map(|token| AlignedWord {
            alignment: self,
            index,
            position,
            token,
        })
    }

    /// Every aligned CHAT word, in order.
    pub fn words(&self) -> impl Iterator<Item = AlignedWord<'_, 's, W>> {
        self.tokens
            .tokens
            .iter()
            .enumerate()
            .map(|(position, token)| AlignedWord {
                alignment: self,
                index: W::mint(position),
                position: UdTokenIndex(position),
                token,
            })
    }

    /// The CHAT word a token is aligned to: the same position, in `W`.
    fn word_at(token: UdTokenIndex) -> W {
        W::mint(token.0)
    }

    /// The UD word the model made the sentence root, with the CHAT word it
    /// is in. `None` only for a sentence with no root word, which the
    /// `%mor` mapper rejects before any caller asks.
    pub fn root(&self) -> Option<(W, &'s UdWord)> {
        self.tokens
            .by_id
            .values()
            .find(|(_, word)| word.head == UdHead::Root)
            .map(|(at, word)| (Self::word_at(*at), *word))
    }
}

/// Record one syntactic word (a `UdId::Single` row) under its id, refusing
/// a reserved or repeated id.
fn record<'s>(
    by_id: &mut BTreeMap<UdWordId, (UdTokenIndex, &'s UdWord)>,
    word: &'s UdWord,
    at: UdTokenIndex,
) -> Result<UdWordId, UdSentenceError> {
    let id = UdWordId::of_row(word).ok_or(UdSentenceError::ZeroId)?;
    match by_id.insert(id, (at, word)) {
        None => Ok(id),
        Some(_) => Err(UdSentenceError::DuplicateId { id: id.get() }),
    }
}

/// One CHAT word of an alignment.
#[derive(Debug, Clone, Copy)]
pub struct AlignedWord<'a, 's, W> {
    alignment: &'a UdAlignment<'s, W>,
    index: W,
    position: UdTokenIndex,
    token: &'a WalkedToken<'s>,
}

impl<'a, 's, W: WordSpace> AlignedWord<'a, 's, W> {
    /// This word's index in the alignment's CHAT-side space.
    pub fn index(&self) -> W {
        self.index
    }

    /// The top-level token this word is aligned to.
    pub fn token_index(&self) -> UdTokenIndex {
        self.position
    }

    /// The UD material this CHAT word covers.
    pub fn ud(&self) -> AlignedUd<'s> {
        self.token.ud
    }

    /// The UD word that speaks for this CHAT word in the dependency tree:
    /// the word itself, or, for a multi-word token, the first component
    /// whose head lies outside the token (`a` of `al`, both of whose
    /// components attach to the following noun).
    pub fn representative(&self) -> &'s UdWord {
        self.token.representative
    }

    /// The CHAT word the representative's head lies in.
    pub fn head(&self) -> HeadTarget<W> {
        match self.token.head {
            TokenHead::Root => HeadTarget::Root,
            TokenHead::Token(at) => HeadTarget::Word(UdAlignment::<W>::word_at(at)),
        }
    }

    /// The UD word the representative's head is, if not the root.
    pub fn head_word(&self) -> Option<&'s UdWord> {
        self.token.head_word
    }

    /// The UD id of the representative's head, if not the root.
    pub fn head_id(&self) -> Option<UdWordId> {
        self.token.representative.head.word()
    }

    /// The UD words whose head is the representative.
    pub fn dependents(&self) -> impl Iterator<Item = &'s UdWord> + '_ {
        let own = UdHead::Word(self.token.representative_id);
        self.alignment
            .tokens
            .by_id
            .values()
            .map(|(_, word)| *word)
            .filter(move |word| word.head == own)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphosyntax::l2::pipeline_tests::ud_sentence;

    /// `avui anem al cole .` as the Catalan model returns it (realigned,
    /// so no terminator row): `al` is a range row and two components.
    const CONTRACTION: &str = "1 avui avui ADV _ 2 advmod
         2 anem anar VERB _ 0 root
         3-4 al _ X _ 0 dep
         3 a a ADP _ 5 case
         4 el el DET _ 5 det
         5 cole cole NOUN _ 2 obl:arg";

    /// A word after a contraction is found by its CHAT index, and its head
    /// is reported as the CHAT word it lies in, not as a UD id.
    #[test]
    fn a_word_after_a_contraction_is_its_own_ud_word() {
        let sentence = ud_sentence(CONTRACTION);
        let alignment =
            UdAlignment::<MorItemIndex>::new(&sentence, 4).expect("the sentence aligns");

        let cole = alignment.word(MorItemIndex::new(3)).expect("CHAT word 3");
        assert_eq!(cole.representative().text, "cole");
        assert_eq!(cole.representative().deprel, "obl:arg");
        assert_eq!(cole.head(), HeadTarget::Word(MorItemIndex::new(1)));
        assert_eq!(cole.head_word().map(|w| w.text.as_str()), Some("anem"));
        let dependents: Vec<&str> = cole.dependents().map(|w| w.text.as_str()).collect();
        assert_eq!(dependents, ["a", "el"]);

        let anem = alignment.word(MorItemIndex::new(1)).expect("CHAT word 1");
        assert_eq!(anem.head(), HeadTarget::Root);
    }

    /// A multi-word token is represented by its first component whose head
    /// lies outside the token, and words attached to any of its components
    /// resolve to its CHAT word.
    #[test]
    fn a_contraction_is_represented_by_its_externally_attached_component() {
        // `dámelo`-shaped: the first component attaches inside the token.
        let sentence = ud_sentence(
            "1-2 xy _ X _ 0 dep
             1 x x PRON _ 2 obj
             2 y y VERB _ 0 root
             3 z z ADV _ 1 advmod",
        );
        let alignment =
            UdAlignment::<MorItemIndex>::new(&sentence, 2).expect("the sentence aligns");

        let token = alignment.word(MorItemIndex::new(0)).expect("CHAT word 0");
        assert_eq!(token.representative().text, "y");
        assert!(matches!(token.ud(), AlignedUd::Mwt { components, .. } if components.len() == 2));
        let z = alignment.word(MorItemIndex::new(1)).expect("CHAT word 1");
        assert_eq!(z.head(), HeadTarget::Word(MorItemIndex::new(0)));
    }

    /// A terminator row (left in place when Stanza owns tokenization) is not
    /// a CHAT word, and the sentence root is found by head, not position.
    #[test]
    fn a_terminator_row_is_not_a_word() {
        let sentence = ud_sentence(
            "1 wake wake VERB _ 0 root
             2 up up ADP _ 1 compound:prt
             3 . . PUNCT _ 1 punct",
        );
        let alignment =
            UdAlignment::<SpanWordPosition>::new(&sentence, 2).expect("the sentence aligns");

        assert_eq!(alignment.words().count(), 2);
        let (root_at, root) = alignment.root().expect("a root");
        assert_eq!(root_at.as_usize(), 0);
        assert_eq!(root.text, "wake");
    }

    /// Every malformation is a named error, never a guessed alignment.
    #[test]
    fn malformed_sentences_are_refused_by_name() {
        let refuse = |rows: &str, words: usize| {
            UdAlignment::<MorItemIndex>::new(&ud_sentence(rows), words)
                .expect_err("the sentence must not align")
        };
        assert_eq!(
            refuse(CONTRACTION, 6),
            UdAlignmentError::WordCountMismatch {
                chat_words: 6,
                ud_tokens: 4
            }
        );
        assert_eq!(
            refuse("1-2 xy _ X _ 0 dep\n 1 x x PRON _ 0 root", 1),
            UdAlignmentError::Sentence(UdSentenceError::MalformedMwt { start: 1, end: 2 })
        );
        assert_eq!(
            refuse("1 a a NOUN _ 0 root\n 1 b b NOUN _ 1 dep", 2),
            UdAlignmentError::Sentence(UdSentenceError::DuplicateId { id: 1 })
        );
        assert_eq!(
            refuse(
                "1 a a NOUN _ 0 root\n 2 b b NOUN _ 3 dep\n 3 . . PUNCT _ 1 punct",
                2
            ),
            UdAlignmentError::Sentence(UdSentenceError::HeadOutsideSentence { word: 2, head: 3 })
        );
        assert_eq!(
            refuse(
                "1-2 xy _ X _ 0 dep\n 1 x x PRON _ 2 obj\n 2 y y VERB _ 1 dep",
                1
            ),
            UdAlignmentError::Sentence(UdSentenceError::MwtWithoutExternalHead {
                start: 1,
                end: 2
            })
        );
    }
}
