//! Sentence-level UD-to-CHAT mapping: the walked tokens of one UD sentence
//! become `%mor` items and their `%gra` relations.
//!
//! The mapper consumes [`UdTokens`], the one walk of a sentence that every
//! alignment also consumes (`crate::morphosyntax::alignment`), so which UD
//! words a `%mor` item covers and which CHAT word an alignment pairs them
//! with cannot drift apart.

use crate::morphosyntax::alignment::{AlignedUd, UdTokenIndex, UdTokens};
use crate::morphosyntax::mapping_helpers::map_ud_word_item;
use crate::morphosyntax::{
    ChunkHead, MappedItem, MappingContext, MappingError, MorProvenance, UdId, UdPunctable,
    UdSentence, UdWord, UdWordId, UniversalPos, assemble_mors, lang2,
    try_handle_italian_range_override, try_handle_italian_single_override, validate_generated_gra,
};
use std::collections::HashMap;
use std::ops::Range;
use talkbank_model::model::GrammaticalRelation;
use talkbank_model::model::dependent_tier::mor::Mor;

/// How a sentence's top-level tokens become `%mor` items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemLayout {
    /// One item per token: a multi-word token's components join as
    /// clitics (`verb|go~part|to`), so the item count is the CHAT word
    /// count. The language reconcilers (Italian) apply here.
    PerToken,
    /// One item per syntactic word: a multi-word token's components are
    /// items of their own, as the retokenizing path writes them.
    PerWord,
}

/// The `%mor` items and `%gra` relations of a mapped sentence, with which
/// items each top-level token produced.
#[derive(Debug)]
pub struct MappedTokens {
    mors: Vec<Mor>,
    gras: Vec<GrammaticalRelation>,
    items: TokenItems,
}

impl MappedTokens {
    /// The items and relations, for injection.
    pub fn into_parts(self) -> (Vec<Mor>, Vec<GrammaticalRelation>) {
        (self.mors, self.gras)
    }

    /// The items and relations, and which items each token produced.
    pub fn into_indexed(self) -> (Vec<Mor>, Vec<GrammaticalRelation>, TokenItems) {
        (self.mors, self.gras, self.items)
    }

    /// The items, in order.
    pub fn mors(&self) -> &[Mor] {
        &self.mors
    }

    /// Which items each token produced.
    pub fn items(&self) -> &TokenItems {
        &self.items
    }
}

/// Which `%mor` items each top-level token of a mapped walk produced.
#[derive(Debug)]
pub struct TokenItems {
    layout: ItemLayout,
    /// Per token, in walk order: the range of items it produced.
    ranges: Vec<Range<usize>>,
}

impl TokenItems {
    /// The range of item positions one token produced; `None` for a token
    /// of another walk that lies past this one's end.
    pub fn items_of(&self, token: UdTokenIndex) -> Option<Range<usize>> {
        self.ranges.get(token.as_usize()).cloned()
    }

    /// The item a syntactic word of `tokens` (the walk this was mapped from)
    /// is written in: its token's one item, or its own item when each word
    /// is one.
    pub fn item_of_word(&self, tokens: &UdTokens<'_>, id: UdWordId) -> Option<usize> {
        let (token, offset) = tokens.locate(id)?;
        let items = self.items_of(token)?;
        match self.layout {
            ItemLayout::PerToken => Some(items.start),
            ItemLayout::PerWord => Some(items.start + offset).filter(|item| items.contains(item)),
        }
    }
}

/// Items and provenance collected while walking a sentence, kept in step.
#[derive(Default)]
struct SentenceItems {
    mors: Vec<Mor>,
    provenance: Vec<MorProvenance>,
}

impl SentenceItems {
    fn push(&mut self, item: MappedItem) {
        let (mor, provenance) = item.into_parts();
        self.mors.push(mor);
        self.provenance.push(provenance);
    }

    fn len(&self) -> usize {
        self.mors.len()
    }
}

/// Map a walked sentence into `%mor` items in `layout`, and build its `%gra`.
pub fn map_tokens(
    tokens: &UdTokens<'_>,
    ctx: &MappingContext,
    layout: ItemLayout,
) -> Result<MappedTokens, MappingError> {
    let is_it = lang2(&ctx.lang) == "it";
    let mut items = SentenceItems::default();
    let mut token_items = Vec::with_capacity(tokens.len());
    for (_, token) in tokens.iter() {
        let start = items.len();
        match (layout, token.ud()) {
            (ItemLayout::PerToken, AlignedUd::Word(word)) => {
                let item = match is_it {
                    true => try_handle_italian_single_override(word, ctx)?,
                    false => None,
                };
                items.push(match item {
                    Some(item) => item,
                    None => map_ud_word_item(word, ctx)?,
                });
            }
            (ItemLayout::PerToken, AlignedUd::Mwt { components, .. }) => {
                let item = match is_it {
                    true => try_handle_italian_range_override(token, ctx)?,
                    false => None,
                };
                items.push(match item {
                    Some(item) => item,
                    None => assemble_mors(components, ctx)?,
                });
            }
            (ItemLayout::PerWord, token) => {
                for word in token.words() {
                    items.push(map_ud_word_item(word, ctx)?);
                }
            }
        }
        token_items.push(start..items.len());
    }
    let gras = build_gra_and_validate(&items)?;
    Ok(MappedTokens {
        mors: items.mors,
        gras,
        items: TokenItems {
            layout,
            ranges: token_items,
        },
    })
}

/// Map a UD sentence to MOR and GRA structures, one item per CHAT word (a
/// multi-word token's components joined as clitics), with the canonical
/// language-specific reconcilers.
pub fn map_ud_sentence(
    sentence: &UdSentence,
    ctx: &MappingContext,
) -> Result<(Vec<Mor>, Vec<GrammaticalRelation>), MappingError> {
    let tokens = UdTokens::walk(sentence)?;
    Ok(map_tokens(&tokens, ctx, ItemLayout::PerToken)?.into_parts())
}

/// Map a UD sentence to MOR and GRA structures with multi-word tokens
/// expanded into per-component items instead of merged clitics.
pub fn map_ud_sentence_expanded(
    sentence: &UdSentence,
    ctx: &MappingContext,
) -> Result<(Vec<Mor>, Vec<GrammaticalRelation>), MappingError> {
    let tokens = UdTokens::walk(sentence)?;
    Ok(map_tokens(&tokens, ctx, ItemLayout::PerWord)?.into_parts())
}

/// Language-neutral GRA builder + validator: one relation per chunk, then the
/// terminator's `PUNCT` relation on the root. Every item carries one
/// provenance entry per chunk by construction ([`MappedItem`]).
fn build_gra_and_validate(items: &SentenceItems) -> Result<Vec<GrammaticalRelation>, MappingError> {
    let provenance = &items.provenance;
    let total_chunks: usize = provenance.iter().map(|chunks| chunks.len()).sum();

    // Every UD word resolves to the 1-based chunk it was written in.
    let mut chunk_of: HashMap<UdWordId, usize> = HashMap::with_capacity(total_chunks);
    let mut chunk = 1usize;
    for chunks in provenance {
        for chunk_prov in chunks {
            for &id in chunk_prov.source_ud_ids() {
                chunk_of.insert(id, chunk);
            }
            chunk += 1;
        }
    }

    let mut gras: Vec<GrammaticalRelation> = Vec::with_capacity(total_chunks + 1);
    let mut root_chunk: Option<usize> = None;
    let mut main_chunk = 1usize;
    for chunks in provenance {
        for (offset, chunk_prov) in chunks.iter().enumerate() {
            let chunk = main_chunk + offset;
            let (head, relation) = match chunk_prov.head() {
                ChunkHead::Root => {
                    root_chunk = Some(chunk);
                    (0, "ROOT".into())
                }
                ChunkHead::FromUd(head) => (
                    *chunk_of
                        .get(head)
                        .ok_or_else(|| MappingError::InvalidHeadReference {
                            details: format!(
                                "chunk {chunk} (deprel={}) has UD head {head} not mapped to any chunk",
                                chunk_prov.deprel()
                            ),
                        })?,
                    chunk_prov.deprel().clone(),
                ),
                ChunkHead::OwningMorMain => (main_chunk, chunk_prov.deprel().clone()),
            };
            gras.push(GrammaticalRelation {
                index: chunk,
                head,
                relation,
            });
        }
        main_chunk += chunks.len();
    }

    let root_chunk = match (root_chunk, gras.is_empty()) {
        (Some(root), _) => root,
        // An empty sentence has no chunk to root; its terminator attaches to
        // the utterance root.
        (None, true) => 0,
        (None, false) => {
            return Err(MappingError::InvalidRoot {
                details: format!(
                    "no chunk with ChunkHead::Root in provenance (Stanza returned no root). GRA so far: {gras:?}"
                ),
            });
        }
    };

    gras.push(GrammaticalRelation {
        index: total_chunks + 1,
        head: root_chunk,
        relation: "PUNCT".into(),
    });

    validate_generated_gra(&gras)?;

    Ok(gras)
}

/// Return whether a UD word is a CHAT utterance terminator.
pub fn is_terminator_punct(ud: &UdWord) -> bool {
    if !matches!(ud.id, UdId::Single(_)) {
        return false;
    }
    let is_punct_pos = matches!(
        ud.upos,
        UdPunctable::Value(UniversalPos::Punct) | UdPunctable::Punct(_)
    );
    if !is_punct_pos {
        return false;
    }
    use talkbank_model::model::content::Terminator;
    Terminator::is_chat_terminator(ud.lemma.trim())
        || Terminator::is_chat_terminator(ud.text.trim())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphosyntax::UdWordAnalysis;
    use talkbank_model::model::LanguageCode;

    fn ud_word(
        id: usize,
        text: &str,
        lemma: &str,
        upos: UniversalPos,
        head: usize,
        deprel: &str,
    ) -> UdWord {
        UdWord::from(UdWordAnalysis {
            id: UdId::Single(id),
            text: text.to_string(),
            lemma: lemma.to_string(),
            upos: UdPunctable::Value(upos),
            xpos: None,
            feats: None,
            head,
            deprel: deprel.to_string(),
            deps: None,
            misc: None,
        })
    }

    #[test]
    fn map_ud_sentence_normalizes_compound_prt_to_chat_gra() {
        let sentence = UdSentence {
            words: vec![
                ud_word(1, "wake", "wake", UniversalPos::Verb, 0, "root"),
                ud_word(2, "up", "up", UniversalPos::Adp, 1, "compound:prt"),
                ud_word(3, ".", ".", UniversalPos::Punct, 1, "punct"),
            ],
        };
        let ctx = MappingContext {
            lang: LanguageCode::new("eng").expect("valid test language code"),
        };

        let (_mors, gras) = map_ud_sentence(&sentence, &ctx).expect("map UD sentence");

        let actual: Vec<String> = gras.iter().map(ToString::to_string).collect();
        assert_eq!(
            actual,
            vec![
                "1|0|ROOT".to_string(),
                "2|1|COMPOUND-PRT".to_string(),
                "3|1|PUNCT".to_string(),
            ]
        );
    }

    #[test]
    fn map_ud_sentence_rewrites_head_zero_dep_to_root() {
        let sentence = UdSentence {
            words: vec![
                ud_word(1, "uh", "uh", UniversalPos::Intj, 0, "dep"),
                ud_word(2, ".", ".", UniversalPos::Punct, 1, "punct"),
            ],
        };
        let ctx = MappingContext {
            lang: LanguageCode::new("eng").expect("valid test language code"),
        };

        let (_mors, gras) = map_ud_sentence(&sentence, &ctx).expect("map UD sentence");

        let actual: Vec<String> = gras.iter().map(ToString::to_string).collect();
        assert_eq!(
            actual,
            vec!["1|0|ROOT".to_string(), "2|1|PUNCT".to_string()]
        );
    }

    #[test]
    fn map_ud_sentence_rewrites_head_zero_discourse_to_root() {
        let sentence = UdSentence {
            words: vec![
                ud_word(1, "well", "well", UniversalPos::Intj, 0, "discourse"),
                ud_word(2, ".", ".", UniversalPos::Punct, 1, "punct"),
            ],
        };
        let ctx = MappingContext {
            lang: LanguageCode::new("eng").expect("valid test language code"),
        };

        let (_mors, gras) = map_ud_sentence(&sentence, &ctx).expect("map UD sentence");

        let actual: Vec<String> = gras.iter().map(ToString::to_string).collect();
        assert_eq!(
            actual,
            vec!["1|0|ROOT".to_string(), "2|1|PUNCT".to_string()]
        );
    }

    /// The mapper and the alignment read one walk, so a sentence the
    /// alignment refuses is refused by the mapper with the same error. The
    /// mapper used to walk on its own and map a range row with missing
    /// components as if it were a word, and let a repeated id overwrite the
    /// first in its head table.
    #[test]
    fn the_mapper_refuses_what_the_walk_refuses() {
        use crate::morphosyntax::alignment::UdSentenceError;
        use crate::morphosyntax::l2::pipeline_tests::ud_sentence;
        let ctx = MappingContext {
            lang: LanguageCode::new("eng").expect("valid test language code"),
        };
        let truncated = ud_sentence("1-2 xy _ X _ 0 dep\n 1 x x PRON _ 0 root");
        assert!(matches!(
            map_ud_sentence(&truncated, &ctx),
            Err(MappingError::Sentence(UdSentenceError::MalformedMwt {
                start: 1,
                end: 2
            }))
        ));
        let repeated = ud_sentence("1 a a NOUN _ 0 root\n 1 b b NOUN _ 1 dep");
        assert!(matches!(
            map_ud_sentence_expanded(&repeated, &ctx),
            Err(MappingError::Sentence(UdSentenceError::DuplicateId {
                id: 1
            }))
        ));
    }

    /// Each token's items are recorded: one per token merged, one per
    /// syntactic word expanded.
    #[test]
    fn each_token_knows_its_items() {
        use crate::morphosyntax::l2::pipeline_tests::ud_sentence;
        let ctx = MappingContext {
            lang: LanguageCode::new("cat").expect("valid test language code"),
        };
        let sentence = ud_sentence(
            "1 anem anar VERB _ 0 root
             2-3 al _ X _ 0 dep
             2 a a ADP _ 4 case
             3 el el DET _ 4 det
             4 cole cole NOUN _ 1 obl",
        );
        let tokens = UdTokens::walk(&sentence).expect("the sentence walks");
        let positions: Vec<UdTokenIndex> = tokens.iter().map(|(at, _)| at).collect();
        let merged = map_tokens(&tokens, &ctx, ItemLayout::PerToken).expect("maps");
        let expanded = map_tokens(&tokens, &ctx, ItemLayout::PerWord).expect("maps");
        let ranges = |mapped: &MappedTokens| -> Vec<Range<usize>> {
            positions
                .iter()
                .map(|at| mapped.items().items_of(*at).expect("a token of this walk"))
                .collect()
        };
        assert_eq!(ranges(&merged), [0..1, 1..2, 2..3]);
        assert_eq!(ranges(&expanded), [0..1, 1..3, 3..4]);
    }
}
