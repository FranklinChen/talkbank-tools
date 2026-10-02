//! Finite-verb-requirement rescue for English copula-progressive constructions.
//!
//! Stanza reads `the sink's overflowing .` with `'s` as a possessive, leaving
//! the clause without a finite verb. It has misparsed this in two shapes:
//!
//! - through 1.14, a noun phrase: `sink` is `nmod:poss` of the `-ing` word,
//!   which is a NOUN carrying the NOUN lemma (`washing`, `barking`);
//! - from 1.15 (UD 2.18), `sink` is the `nsubj` of the `-ing` word, which is a
//!   VERB with `VerbForm=Ger` and a verbal lemma, while `'s` is still the
//!   possessive `case` marker on `sink`.
//!
//! Both are rewritten to the same analysis: `'s` is the finite copula `be`, the
//! `-ing` word the present participle heading the clause. The participle must
//! carry a VERB lemma. In the gerund shape Stanza supplies one; in the noun
//! shape the worker asks Stanza's lemmatizer for the word's verb reading and
//! sends it in the UD MISC field as `VerbReadingLemma=...`
//! (`batchalign/inference/_english_verb_reading.py`). Without a verb lemma
//! there is no rescue: a VERB with a noun lemma (`verb|barking`) is a
//! fabricated analysis, which is what this rule existed to prevent.

use crate::morphosyntax::{
    CuratedFeats, DepRel, FeatName, UdHead, UdId, UdPunctable, UdSentence, UdWord, UdWordId,
    UniversalPos, ud_pair_value,
};
use verb_lemma::VerbLemma;

/// The UD MISC key under which the worker sends an English `-ing` noun's
/// lemma as a verb. Must match `VERB_READING_LEMMA_MISC_KEY` in
/// `batchalign/inference/_english_verb_reading.py`.
pub const VERB_READING_LEMMA_MISC_KEY: &str = "VerbReadingLemma";

/// The rescue's analysis of `'s`: the finite copula, present, third person
/// singular.
const FINITE_COPULA_PRES_3SG: CuratedFeats =
    CuratedFeats::new("Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin");

/// The rescue's analysis of the `-ing` word: a present participle.
const PRESENT_PARTICIPLE: CuratedFeats = CuratedFeats::new("Tense=Pres|VerbForm=Part");

/// English-specific rewrite, in place. Returns the input untouched when no
/// rescue applies.
pub fn rescue_english_copula_progressive(mut sentence: UdSentence) -> UdSentence {
    if let Some(plan) = detect_rescue(&sentence) {
        apply_rescue(&mut sentence, plan);
    }
    sentence
}

mod verb_lemma {
    /// The verb lemma the promoted `-ing` word will carry: a non-empty lemma
    /// from Stanza's own verbal analysis. Its field is private to this module,
    /// so [`VerbLemma::parse`] is the only way to make one and the rescue
    /// cannot promote a word to VERB with a missing or placeholder lemma.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(super) struct VerbLemma(String);

    impl VerbLemma {
        /// Stanza writes `_` for a lemma it could not produce; that and the
        /// empty string are not lemmas.
        pub(super) fn parse(raw: &str) -> Option<Self> {
            match raw {
                "" | "_" => None,
                lemma => Some(Self(lemma.to_string())),
            }
        }

        pub(super) fn into_string(self) -> String {
            self.0
        }
    }
}

/// An `-ing` word in one of the two misparse shapes, as one classification:
/// its id, and its verb lemma when the analysis supplies one.
struct IngHead {
    id: UdWordId,
    verb_lemma: Option<VerbLemma>,
}

#[derive(Debug, Clone)]
struct RescuePlan {
    part_id: UdWordId,
    possessor_id: UdWordId,
    verb_id: UdWordId,
    verb_lemma: VerbLemma,
    old_root_id: UdWordId,
}

fn detect_rescue(sentence: &UdSentence) -> Option<RescuePlan> {
    if sentence.words.iter().any(UdWord::has_finite_verb_form) {
        return None;
    }

    // Only sentences with a multi-word token (`sink's`) can carry the
    // possessive-particle misreading this rescue corrects.
    sentence.mwt_component_ranges().next()?;

    let part = sentence.words.iter().find(|w| is_possessive_part(w))?;
    let possessor_id = part.head.word()?;
    let possessor = find_word_by_single_id(sentence, possessor_id)?;
    let possessor_upos = match &possessor.upos {
        UdPunctable::Value(u) => *u,
        UdPunctable::Punct(_) => return None,
    };
    if !matches!(possessor_upos, UniversalPos::Noun | UniversalPos::Propn) {
        return None;
    }

    let mut ing_heads = sentence.words.iter().filter_map(ing_head);
    let IngHead {
        id: verb_id,
        verb_lemma,
    } = ing_heads.next()?;
    if ing_heads.next().is_some() {
        return None;
    }
    // The two misparse shapes (module docs): a possessor of the noun phrase,
    // or the subject of the gerund. Any other relation is not this defect.
    match possessor.dep_rel() {
        DepRel::NmodPoss => {}
        DepRel::NSubj if possessor.head.is(verb_id) => {}
        _ => return None,
    }
    // No verb lemma, no rescue: a VERB with a noun's lemma is the fabrication
    // this rule exists to prevent.
    let verb_lemma = verb_lemma?;

    let root = sentence
        .words
        .iter()
        .find(|w| w.head == UdHead::Root && w.dep_rel() == DepRel::Root)?;
    let old_root_id = single_id(root)?;
    let part_id = single_id(part)?;

    Some(RescuePlan {
        part_id,
        possessor_id,
        verb_id,
        verb_lemma,
        old_root_id,
    })
}

fn apply_rescue(sentence: &mut UdSentence, plan: RescuePlan) {
    let RescuePlan {
        part_id,
        possessor_id,
        verb_id,
        verb_lemma,
        old_root_id,
    } = plan;
    // Moved into the one word it belongs to, the first time that word is met.
    let mut verb_lemma = Some(verb_lemma.into_string());

    for word in &mut sentence.words {
        let Some(id_n) = single_id(word) else {
            continue;
        };

        if id_n == part_id {
            word.upos = UdPunctable::Value(UniversalPos::Aux);
            word.lemma = "be".to_string();
            word.xpos = Some("VBZ".to_string());
            word.apply_curated(FINITE_COPULA_PRES_3SG);
            word.deprel = DepRel::Aux.as_str().to_string();
            word.head = UdHead::Word(verb_id);
        } else if id_n == possessor_id {
            word.deprel = DepRel::NSubj.as_str().to_string();
            word.head = UdHead::Word(verb_id);
        } else if id_n == verb_id {
            word.upos = UdPunctable::Value(UniversalPos::Verb);
            if let Some(lemma) = verb_lemma.take() {
                word.lemma = lemma;
            }
            word.xpos = Some("VBG".to_string());
            word.apply_curated(PRESENT_PARTICIPLE);
            word.deprel = DepRel::Root.as_str().to_string();
            word.head = UdHead::Root;
        } else if id_n == old_root_id && old_root_id != verb_id {
            word.deprel = DepRel::Obj.as_str().to_string();
            word.head = UdHead::Word(verb_id);
        } else if word.head.is(old_root_id)
            && old_root_id != verb_id
            && matches!(
                word.dep_rel(),
                DepRel::Cc | DepRel::Punct | DepRel::Discourse | DepRel::Mark,
            )
        {
            word.head = UdHead::Word(verb_id);
        }
    }
}

/// The id of a syntactic word (a `UdId::Single` row); `None` for a range
/// row or an empty node.
fn single_id(word: &UdWord) -> Option<UdWordId> {
    match word.id {
        UdId::Single(_) => UdWordId::of_row(word),
        UdId::Range(..) | UdId::Decimal(_) => None,
    }
}

fn is_possessive_part(word: &UdWord) -> bool {
    matches!(&word.upos, UdPunctable::Value(UniversalPos::Part))
        && word.lemma == "'s"
        && word.dep_rel() == DepRel::Case
}

/// Classify a word as an `-ing` head of either misparse shape, once: a NOUN
/// (verb lemma from the worker's `VerbReadingLemma`), or a gerund VERB (verb
/// lemma from Stanza). Only single-id words can head the clause.
fn ing_head(word: &UdWord) -> Option<IngHead> {
    let id = single_id(word)?;
    if !ends_with_ing(&word.text) {
        return None;
    }
    let verb_lemma = match &word.upos {
        UdPunctable::Value(UniversalPos::Noun) => {
            ud_pair_value(word.misc.as_deref(), VERB_READING_LEMMA_MISC_KEY)
                .and_then(VerbLemma::parse)
        }
        UdPunctable::Value(UniversalPos::Verb)
            if word.features().value(FeatName::VerbForm) == Some("Ger") =>
        {
            VerbLemma::parse(&word.lemma)
        }
        _ => return None,
    };
    Some(IngHead { id, verb_lemma })
}

fn ends_with_ing(text: &str) -> bool {
    text.len() >= 4 && text.to_ascii_lowercase().ends_with("ing")
}

fn find_word_by_single_id(sentence: &UdSentence, id: UdWordId) -> Option<&UdWord> {
    sentence.words.iter().find(|w| single_id(w) == Some(id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphosyntax::UdWordAnalysis;

    fn word(
        id: UdId,
        text: &str,
        lemma: &str,
        upos: UniversalPos,
        feats: Option<&str>,
        head: usize,
        deprel: &str,
    ) -> UdWord {
        UdWord::from(UdWordAnalysis {
            id,
            text: text.to_string(),
            lemma: lemma.to_string(),
            upos: UdPunctable::Value(upos),
            xpos: None,
            feats: feats.map(|s| s.to_string()),
            head,
            deprel: deprel.to_string(),
            deps: None,
            misc: None,
        })
    }

    /// `w` as the worker sends an English `-ing` noun: with its verb lemma.
    fn with_verb_reading(mut w: UdWord, lemma: &str) -> UdWord {
        w.misc = Some(format!("{VERB_READING_LEMMA_MISC_KEY}={lemma}"));
        w
    }

    fn punct_word(id: UdId, text: &str, head: usize) -> UdWord {
        UdWord::from(UdWordAnalysis {
            id,
            text: text.to_string(),
            lemma: text.to_string(),
            upos: UdPunctable::Punct(text.to_string()),
            xpos: None,
            feats: None,
            head,
            deprel: "punct".to_string(),
            deps: None,
            misc: None,
        })
    }

    fn range_parent(start: usize, end: usize, text: &str) -> UdWord {
        UdWord::from(UdWordAnalysis {
            id: UdId::Range(start, end),
            text: text.to_string(),
            lemma: String::new(),
            upos: UdPunctable::Value(UniversalPos::X),
            xpos: None,
            feats: None,
            head: 0,
            deprel: String::new(),
            deps: None,
            misc: None,
        })
    }

    fn fixture_sink() -> UdSentence {
        UdSentence {
            words: vec![
                word(
                    UdId::Single(1),
                    "and",
                    "and",
                    UniversalPos::Cconj,
                    None,
                    5,
                    "cc",
                ),
                word(
                    UdId::Single(2),
                    "the",
                    "the",
                    UniversalPos::Det,
                    Some("Definite=Def|PronType=Art"),
                    3,
                    "det",
                ),
                range_parent(3, 4, "sink's"),
                word(
                    UdId::Single(3),
                    "sink",
                    "sink",
                    UniversalPos::Noun,
                    Some("Number=Sing"),
                    5,
                    "nmod:poss",
                ),
                word(
                    UdId::Single(4),
                    "'s",
                    "'s",
                    UniversalPos::Part,
                    None,
                    3,
                    "case",
                ),
                with_verb_reading(
                    word(
                        UdId::Single(5),
                        "overflowing",
                        "overflow",
                        UniversalPos::Noun,
                        Some("Number=Sing"),
                        0,
                        "root",
                    ),
                    "overflow",
                ),
                punct_word(UdId::Single(6), ".", 5),
            ],
        }
    }

    fn fixture_lady() -> UdSentence {
        UdSentence {
            words: vec![
                word(
                    UdId::Single(1),
                    "the",
                    "the",
                    UniversalPos::Det,
                    Some("Definite=Def|PronType=Art"),
                    2,
                    "det",
                ),
                range_parent(2, 3, "lady's"),
                word(
                    UdId::Single(2),
                    "lady",
                    "lady",
                    UniversalPos::Noun,
                    Some("Number=Sing"),
                    5,
                    "nmod:poss",
                ),
                word(
                    UdId::Single(3),
                    "'s",
                    "'s",
                    UniversalPos::Part,
                    None,
                    2,
                    "case",
                ),
                with_verb_reading(
                    word(
                        UdId::Single(4),
                        "washing",
                        "washing",
                        UniversalPos::Noun,
                        Some("Number=Sing"),
                        5,
                        "compound",
                    ),
                    "wash",
                ),
                word(
                    UdId::Single(5),
                    "dishes",
                    "dish",
                    UniversalPos::Noun,
                    Some("Number=Plur"),
                    0,
                    "root",
                ),
                punct_word(UdId::Single(6), ".", 5),
            ],
        }
    }

    fn find_by_id(s: &UdSentence, id: usize) -> &UdWord {
        s.words
            .iter()
            .find(|w| w.id == UdId::Single(id))
            .expect("expected Single id to exist")
    }

    fn assert_unchanged(sentence: UdSentence) {
        let out = rescue_english_copula_progressive(sentence.clone());
        assert_eq!(sentence, out, "expected rule to be a no-op");
    }

    #[test]
    fn sink_pattern_a_rewrite() {
        let out = rescue_english_copula_progressive(fixture_sink());
        let s = find_by_id(&out, 4);
        assert!(matches!(s.upos, UdPunctable::Value(UniversalPos::Aux)));
        assert_eq!(s.lemma, "be");
        assert_eq!(s.deprel, "aux");
        assert_eq!(s.head.conllu(), 5);
        assert!(s.has_finite_verb_form());
        // The rescue's own table wrote both words' features.
        for word in [s, find_by_id(&out, 5)] {
            assert!(
                word.features()
                    .iter()
                    .all(|(_, value)| value.source() == crate::morphosyntax::FeatSource::Curated)
            );
        }

        let v = find_by_id(&out, 5);
        assert!(matches!(v.upos, UdPunctable::Value(UniversalPos::Verb)));
        assert_eq!(v.deprel, "root");
        assert_eq!(v.head.conllu(), 0);
        assert_eq!(v.features().to_string(), "Tense=Pres|VerbForm=Part");

        let p = find_by_id(&out, 3);
        assert_eq!(p.deprel, "nsubj");
        assert_eq!(p.head.conllu(), 5);
    }

    #[test]
    fn lady_pattern_b_rewrite() {
        let out = rescue_english_copula_progressive(fixture_lady());
        let s = find_by_id(&out, 3);
        assert!(matches!(s.upos, UdPunctable::Value(UniversalPos::Aux)));
        assert_eq!(s.lemma, "be");
        assert_eq!(s.deprel, "aux");
        assert_eq!(s.head.conllu(), 4);

        let v = find_by_id(&out, 4);
        assert!(matches!(v.upos, UdPunctable::Value(UniversalPos::Verb)));
        // Stanza's NOUN lemma was `washing`; the verb carries the verb reading.
        assert_eq!(v.lemma, "wash");
        assert_eq!(v.deprel, "root");
        assert_eq!(v.head.conllu(), 0);
        assert_eq!(v.features().to_string(), "Tense=Pres|VerbForm=Part");

        let p = find_by_id(&out, 2);
        assert_eq!(p.deprel, "nsubj");
        assert_eq!(p.head.conllu(), 4);

        let old = find_by_id(&out, 5);
        assert_eq!(old.deprel, "obj");
        assert_eq!(old.head.conllu(), 4);

        let dot = find_by_id(&out, 6);
        assert_eq!(dot.head.conllu(), 4);
    }

    /// Stanza 1.15's shape: the subject is `nsubj` of a gerund VERB whose
    /// lemma is already verbal, and `'s` is still the possessive marker.
    fn fixture_sink_gerund() -> UdSentence {
        let mut s = fixture_sink();
        for w in &mut s.words {
            match w.id {
                UdId::Single(3) => w.deprel = "nsubj".to_string(),
                UdId::Single(5) => {
                    *w = word(
                        w.id.clone(),
                        &w.text,
                        &w.lemma,
                        UniversalPos::Verb,
                        Some("VerbForm=Ger"),
                        w.head.conllu(),
                        &w.deprel,
                    );
                }
                _ => {}
            }
        }
        s
    }

    #[test]
    fn gerund_shape_from_stanza_1_15_is_rescued() {
        let out = rescue_english_copula_progressive(fixture_sink_gerund());
        let s = find_by_id(&out, 4);
        assert!(matches!(s.upos, UdPunctable::Value(UniversalPos::Aux)));
        assert_eq!(s.lemma, "be");
        let v = find_by_id(&out, 5);
        assert!(matches!(v.upos, UdPunctable::Value(UniversalPos::Verb)));
        assert_eq!(v.lemma, "overflow");
        assert_eq!(v.features().to_string(), "Tense=Pres|VerbForm=Part");
        assert_eq!(find_by_id(&out, 3).deprel, "nsubj");
    }

    #[test]
    fn gerund_shape_whose_subject_heads_elsewhere_is_left_alone() {
        let mut s = fixture_sink_gerund();
        for w in &mut s.words {
            if w.id == UdId::Single(3) {
                w.head = UdHead::of_conllu(2);
            }
        }
        assert_unchanged(s);
    }

    #[test]
    fn noun_shape_without_a_verb_reading_is_left_alone() {
        let mut s = fixture_lady();
        for w in &mut s.words {
            w.misc = None;
        }
        assert_unchanged(s);
    }

    #[test]
    fn a_placeholder_lemma_is_not_a_verb_lemma() {
        assert_eq!(VerbLemma::parse("_"), None);
        assert_eq!(VerbLemma::parse(""), None);
        assert!(VerbLemma::parse("bark").is_some());
    }

    #[test]
    fn negative_copula_adj_no_ing() {
        let s = UdSentence {
            words: vec![
                word(
                    UdId::Single(1),
                    "the",
                    "the",
                    UniversalPos::Det,
                    None,
                    2,
                    "det",
                ),
                range_parent(2, 3, "boy's"),
                word(
                    UdId::Single(2),
                    "boy",
                    "boy",
                    UniversalPos::Noun,
                    Some("Number=Sing"),
                    4,
                    "nmod:poss",
                ),
                word(
                    UdId::Single(3),
                    "'s",
                    "'s",
                    UniversalPos::Part,
                    None,
                    2,
                    "case",
                ),
                word(
                    UdId::Single(4),
                    "tall",
                    "tall",
                    UniversalPos::Adj,
                    Some("Degree=Pos"),
                    0,
                    "root",
                ),
                punct_word(UdId::Single(5), ".", 4),
            ],
        };
        assert_unchanged(s);
    }

    #[test]
    fn negative_existential_there_no_ing() {
        let s = UdSentence {
            words: vec![
                range_parent(1, 2, "there's"),
                word(
                    UdId::Single(1),
                    "there",
                    "there",
                    UniversalPos::Pron,
                    None,
                    2,
                    "expl",
                ),
                word(
                    UdId::Single(2),
                    "'s",
                    "be",
                    UniversalPos::Verb,
                    Some("Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin"),
                    0,
                    "root",
                ),
                word(UdId::Single(3), "a", "a", UniversalPos::Det, None, 4, "det"),
                word(
                    UdId::Single(4),
                    "cat",
                    "cat",
                    UniversalPos::Noun,
                    Some("Number=Sing"),
                    2,
                    "nsubj",
                ),
                punct_word(UdId::Single(5), ".", 2),
            ],
        };
        assert_unchanged(s);
    }

    #[test]
    fn ends_with_ing_predicate_is_conservative() {
        assert!(ends_with_ing("going"));
        assert!(ends_with_ing("WASHING"));
        assert!(!ends_with_ing("ing"));
        assert!(!ends_with_ing("dog"));
        assert!(!ends_with_ing("sinking things"));
    }
}
