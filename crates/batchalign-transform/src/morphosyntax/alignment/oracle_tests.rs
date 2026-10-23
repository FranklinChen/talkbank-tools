//! An oracle for [`UdTokens::walk`] and [`UdAlignment`]: a naive model of the
//! same facts, compared on generated sentences.
//!
//! The production walk indexes words by UD id in a `BTreeMap` and resolves
//! every head once; the model below does each lookup by a linear scan over
//! the tokens, with no table, so the two share no code. Sentences are
//! generated with random multi-word tokens, a random order of id blocks (ids
//! are contiguous inside a multi-word token, as UD requires, but tokens need
//! not be in id order), random heads (some pointing past the sentence or at
//! the terminator), and an optional terminator row the walk must skip.

use proptest::prelude::*;

use super::*;
use crate::morphosyntax::l2::pipeline_tests::ud_sentence;

/// One generated top-level token: its syntactic word ids (one, or a
/// multi-word token's contiguous range) and each word's head (0 is the
/// root).
#[derive(Debug, Clone)]
struct NaiveToken {
    ids: Vec<usize>,
    heads: Vec<usize>,
}

/// A generated sentence: its tokens in row order, and whether a terminator
/// row (id `words + 1`) follows them.
#[derive(Debug, Clone)]
struct Generated {
    tokens: Vec<NaiveToken>,
    terminator: bool,
}

impl Generated {
    fn word_count(&self) -> usize {
        self.tokens.iter().map(|token| token.ids.len()).sum()
    }

    /// The sentence as the rows the fixture parser reads.
    fn rows(&self) -> String {
        let mut rows = String::new();
        for token in &self.tokens {
            if let [start, .., end] = token.ids[..] {
                rows.push_str(&format!("{start}-{end} mwt{start} _ X _ 0 dep\n"));
            }
            for (id, head) in token.ids.iter().zip(&token.heads) {
                rows.push_str(&format!("{id} w{id} w{id} NOUN _ {head} dep\n"));
            }
        }
        if self.terminator {
            let id = self.word_count() + 1;
            rows.push_str(&format!("{id} . . PUNCT _ 0 punct\n"));
        }
        rows
    }
}

/// Generate a sentence: token sizes, a permutation of their id blocks, and
/// a head per word drawn from `0..=words + 1` (so some heads name the
/// terminator or nothing).
fn generated() -> impl Strategy<Value = Generated> {
    (prop::collection::vec(1usize..=3, 1..=6), any::<bool>())
        .prop_flat_map(|(sizes, terminator)| {
            let words: usize = sizes.iter().sum();
            let order = Just((0..sizes.len()).collect::<Vec<_>>()).prop_shuffle();
            let heads = prop::collection::vec(0..=words + 1, words);
            (Just(sizes), order, heads, Just(terminator))
        })
        .prop_map(|(sizes, order, heads, terminator)| {
            // Give each token, in the shuffled order, the next block of ids.
            let mut first_id = vec![0; sizes.len()];
            let mut next = 1;
            for &token in &order {
                first_id[token] = next;
                next += sizes[token];
            }
            let mut heads = heads.into_iter();
            let tokens = sizes
                .iter()
                .zip(first_id)
                .map(|(&size, first)| NaiveToken {
                    ids: (first..first + size).collect(),
                    heads: heads.by_ref().take(size).collect(),
                })
                .collect();
            Generated { tokens, terminator }
        })
}

/// What the model says one aligned token is.
#[derive(Debug, PartialEq, Eq)]
struct NaiveAligned {
    representative: usize,
    head: HeadTarget<usize>,
    dependents: Vec<usize>,
}

/// What the model says the whole sentence is.
#[derive(Debug, PartialEq, Eq)]
struct NaiveModel {
    tokens: Vec<NaiveAligned>,
    root: Option<(usize, usize)>,
}

/// Which token holds word `id`, by linear scan.
fn token_of(tokens: &[NaiveToken], id: usize) -> Option<usize> {
    tokens.iter().position(|token| token.ids.contains(&id))
}

/// The model: the same facts the walk establishes, each by a scan.
fn naive(sentence: &Generated) -> Result<NaiveModel, UdSentenceError> {
    let tokens = &sentence.tokens;
    // A multi-word token's representative: its first component whose head
    // is the root or lies outside the token. Checked token by token in row
    // order, as the walk does.
    let mut representatives = Vec::new();
    for token in tokens {
        // A single word speaks for itself, whatever its head.
        let representative = match token.ids[..] {
            [id] => Some(id),
            _ => token
                .ids
                .iter()
                .zip(&token.heads)
                .find(|(_, head)| **head == 0 || !token.ids.contains(head))
                .map(|(id, _)| *id),
        };
        match representative {
            Some(id) => representatives.push(id),
            None => {
                return Err(UdSentenceError::MwtWithoutExternalHead {
                    start: token.ids[0],
                    end: token.ids[token.ids.len() - 1],
                });
            }
        }
    }
    // Every head must name a walked word; the first offender in id order.
    let mut words: Vec<(usize, usize)> = tokens
        .iter()
        .flat_map(|token| token.ids.iter().copied().zip(token.heads.iter().copied()))
        .collect();
    words.sort_unstable();
    if let Some(&(word, head)) = words
        .iter()
        .find(|(_, head)| *head != 0 && token_of(tokens, *head).is_none())
    {
        return Err(UdSentenceError::HeadOutsideSentence { word, head });
    }
    // Every word and every head is now known to be in a token.
    let held = |id: usize| {
        token_of(tokens, id).unwrap_or_else(|| panic!("word {id} was checked to be in a token"))
    };
    let head_of = |id: usize| {
        words
            .iter()
            .find(|(word, _)| *word == id)
            .map(|(_, head)| *head)
    };
    let aligned = representatives
        .iter()
        .map(|&representative| {
            let head = match head_of(representative) {
                Some(0) | None => HeadTarget::Root,
                Some(head) => HeadTarget::Word(held(head)),
            };
            let dependents = words
                .iter()
                .filter(|(_, head)| *head == representative)
                .map(|(word, _)| *word)
                .collect();
            NaiveAligned {
                representative,
                head,
                dependents,
            }
        })
        .collect();
    let root = words
        .iter()
        .find(|(_, head)| *head == 0)
        .map(|(word, _)| (held(*word), *word));
    Ok(NaiveModel {
        tokens: aligned,
        root,
    })
}

/// The production walk and alignment, read into the model's terms.
fn production(sentence: &UdSentence, tokens: usize) -> Result<NaiveModel, UdSentenceError> {
    let alignment = match UdAlignment::<SpanWordPosition>::new(sentence, tokens) {
        Ok(alignment) => alignment,
        Err(UdAlignmentError::Sentence(error)) => return Err(error),
        Err(mismatch @ UdAlignmentError::WordCountMismatch { .. }) => {
            panic!("the walk found another number of tokens than generated: {mismatch}")
        }
    };
    let single = |word: &UdWord| match &word.id {
        UdId::Single(id) => *id,
        other => panic!("a representative is a syntactic word, not {other:?}"),
    };
    let tokens = alignment
        .words()
        .map(|word| NaiveAligned {
            representative: single(word.representative()),
            head: match word.head() {
                HeadTarget::Root => HeadTarget::Root,
                HeadTarget::Word(at) => HeadTarget::Word(at.as_usize()),
            },
            dependents: word.dependents().map(single).collect(),
        })
        .collect();
    let root = alignment
        .root()
        .map(|(at, word)| (at.as_usize(), single(word)));
    Ok(NaiveModel { tokens, root })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    /// The walk and the naive model agree on every generated sentence:
    /// the same refusal, or the same representative, head and dependents
    /// for every token, and the same root.
    #[test]
    fn the_walk_agrees_with_a_naive_model(sentence in generated()) {
        let ud = ud_sentence(&sentence.rows());
        prop_assert_eq!(
            production(&ud, sentence.tokens.len()),
            naive(&sentence),
            "rows:\n{}",
            sentence.rows()
        );
    }
}
