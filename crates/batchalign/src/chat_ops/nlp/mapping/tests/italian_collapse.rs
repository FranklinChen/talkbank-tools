//! Italian Stanza defect-6: morphologically-shaped Italian items (parla, arancione, piccolo, gomitolo, ...) collapse to the correct head POS+lemma when ambiguous; plus dammela merge-preservation.

#![allow(unused_imports, dead_code)]

use super::*;

use crate::chat_ops::nlp::mapping::validate_generated_gra;
use crate::chat_ops::nlp::mapping::*;
use crate::chat_ops::nlp::{UdId, UdPunctable, UdSentence, UdWord, UniversalPos};
use crate::chat_ops::nlp::{clean_lemma, map_ud_word};
use talkbank_model::model::GrammaticalRelation;
use talkbank_model::model::dependent_tier::mor::Mor;

#[test]
fn test_italian_defect6_parla_collapses_to_verb_parlare() {
    let sentence = UdSentence {
        words: vec![
            it_range(1, 2, "parla"),
            it_word(
                1,
                "par",
                "par",
                UniversalPos::Verb,
                0,
                "root",
                Some("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
            ),
            it_word(
                2,
                "la",
                "il",
                UniversalPos::Pron,
                1,
                "obj",
                Some("Number=Sing|Person=3|PronType=Prs"),
            ),
            it_word(
                3,
                "forte",
                "forte",
                UniversalPos::Adj,
                1,
                "advmod",
                Some("Number=Sing"),
            ),
        ],
    };
    let (mors, _gras) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
    let items = chat_strings(&mors);
    assert_eq!(
        items.len(),
        2,
        "Expected 2 mors (parla + forte), got {items:?}"
    );
    assert!(
        items[0].starts_with("v|parlare") || items[0].starts_with("verb|parlare"),
        "Expected verb|parlare, got {:?}",
        items[0]
    );
    assert!(
        !items[0].contains("~"),
        "Reconciled mor must not carry a clitic suffix, got {:?}",
        items[0]
    );
}

#[test]
fn test_italian_collapse_keeps_a_non_first_representatives_attachment() {
    let sentence = UdSentence {
        words: vec![
            it_range(1, 2, "parla"),
            it_word(
                1,
                "par",
                "par",
                UniversalPos::Verb,
                2,
                "aux",
                Some("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
            ),
            it_word(
                2,
                "la",
                "il",
                UniversalPos::Pron,
                0,
                "root",
                Some("Number=Sing|Person=3|PronType=Prs"),
            ),
            it_word(
                3,
                "forte",
                "forte",
                UniversalPos::Adj,
                2,
                "advmod",
                Some("Number=Sing"),
            ),
        ],
    };
    let (mors, gras) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
    assert!(chat_strings(&mors)[0].starts_with("verb|parlare"));
    assert_eq!(
        gras.iter()
            .map(|gra| (gra.index, gra.head, gra.relation.as_str()))
            .collect::<Vec<_>>(),
        [(1, 0, "ROOT"), (2, 1, "ADVMOD"), (3, 1, "PUNCT")],
    );
}

#[test]
fn test_italian_defect6_arancione_collapses_to_noun_arancione() {
    assert_defect6_2component_collapses(
        "arancione",
        ("arancio", "arancio"),
        ("ne", "ne"),
        ExpectedPos::Noun,
        "arancione",
    );
}

#[test]
fn test_italian_defect6_piccolo_collapses_to_adj_piccolo() {
    assert_defect6_2component_collapses(
        "piccolo",
        ("picco", "picco"),
        ("lo", "il"),
        ExpectedPos::Adj,
        "piccolo",
    );
}

#[test]
fn test_italian_defect6_gomitolo_collapses_to_noun_gomitolo() {
    assert_defect6_2component_collapses(
        "gomitolo",
        ("gomito", "gomito"),
        ("lo", "il"),
        ExpectedPos::Noun,
        "gomitolo",
    );
}

#[test]
fn test_italian_defect6_pallone_collapses_to_noun_pallone() {
    assert_defect6_2component_collapses(
        "pallone",
        ("pallo", "pallare"),
        ("ne", "ne"),
        ExpectedPos::Noun,
        "pallone",
    );
}

#[test]
fn test_italian_defect6_bastone_collapses_to_noun_bastone() {
    assert_defect6_2component_collapses(
        "bastone",
        ("basto", "bastare"),
        ("ne", "ne"),
        ExpectedPos::Noun,
        "bastone",
    );
}

#[test]
fn test_italian_defect6_cappello_collapses_to_noun_cappello() {
    assert_defect6_2component_collapses(
        "cappello",
        ("cappe", "cappere"),
        ("lo", "lo"),
        ExpectedPos::Noun,
        "cappello",
    );
}

#[test]
fn test_italian_defect6_seggiola_collapses_to_noun_seggiola() {
    assert_defect6_2component_collapses(
        "seggiola",
        ("seggio", "seggio"),
        ("la", "la"),
        ExpectedPos::Noun,
        "seggiola",
    );
}

#[test]
fn test_italian_defect6_piccola_collapses_to_adj_piccolo() {
    assert_defect6_2component_collapses(
        "piccola",
        ("picco", "picco"),
        ("la", "la"),
        ExpectedPos::Adj,
        "piccolo",
    );
}

#[test]
fn test_italian_defect6_trottola_collapses_to_noun_trottola() {
    assert_defect6_2component_collapses(
        "trottola",
        ("trotto", "trotto"),
        ("la", "la"),
        ExpectedPos::Noun,
        "trottola",
    );
}

#[test]
fn test_italian_defect6_cielo_collapses_to_noun_cielo() {
    assert_defect6_2component_collapses(
        "cielo",
        ("cie", "cie"),
        ("lo", "lo"),
        ExpectedPos::Noun,
        "cielo",
    );
}

#[test]
fn test_italian_defect6_normale_collapses_to_adj_normale() {
    assert_defect6_2component_collapses(
        "normale",
        ("norma", "norma"),
        ("le", "le"),
        ExpectedPos::Adj,
        "normale",
    );
}

#[test]
fn test_italian_defect6_cavallone_collapses_to_noun_cavallone() {
    assert_defect6_2component_collapses(
        "cavallone",
        ("cavallo", "cavallo"),
        ("ne", "ne"),
        ExpectedPos::Noun,
        "cavallone",
    );
}

#[test]
fn test_italian_defect6_coccole_collapses_to_noun_coccole() {
    assert_defect6_2component_collapses(
        "coccole",
        ("cocco", "cocco"),
        ("le", "le"),
        ExpectedPos::Noun,
        "coccole",
    );
}

#[test]
fn test_italian_defect6_bottone_collapses_to_noun_bottone() {
    assert_defect6_2component_collapses(
        "bottone",
        ("botto", "botto"),
        ("ne", "ne"),
        ExpectedPos::Noun,
        "bottone",
    );
}

#[test]
fn test_italian_defect6_difficile_collapses_to_adj_difficile() {
    assert_defect6_2component_collapses(
        "difficile",
        ("diffici", "diffire"),
        ("le", "le"),
        ExpectedPos::Adj,
        "difficile",
    );
}

#[test]
fn test_italian_defect6_divano_collapses_to_noun_divano() {
    assert_defect6_2component_collapses(
        "divano",
        ("diva", "diva"),
        ("no", "no"),
        ExpectedPos::Noun,
        "divano",
    );
}

#[test]
fn test_italian_dammela_stays_correctly_merged() {
    let sentence = UdSentence {
        words: vec![
            it_range(1, 3, "dammela"),
            it_word(
                1,
                "da",
                "dare",
                UniversalPos::Verb,
                0,
                "root",
                Some("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
            ),
            it_word(
                2,
                "mme",
                "me",
                UniversalPos::Pron,
                1,
                "iobj",
                Some("Number=Sing|Person=1|PronType=Prs"),
            ),
            it_word(
                3,
                "la",
                "il",
                UniversalPos::Pron,
                1,
                "obj",
                Some("Number=Sing|Person=3|PronType=Prs"),
            ),
        ],
    };
    let (mors, _) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
    let items = chat_strings(&mors);
    assert_eq!(items.len(), 1);
    assert!(
        items[0].contains("~"),
        "Genuine verb+clitic compound must stay merged with ~, got {:?}",
        items[0]
    );
    assert!(
        items[0].contains("dare") || items[0].contains("v|dar"),
        "Verb lemma must be `dare`, got {:?}",
        items[0]
    );
}

/// Stanza 1.15's analysis of `per favore dammela`: the compound split, but as
/// preposition `da` plus clitics hanging off `favore`. The curated imperative
/// replaces the components, and the verb keeps the split word's attachment.
#[test]
fn test_italian_dammela_split_as_preposition_takes_the_curated_imperative() {
    let sentence = UdSentence {
        words: vec![
            it_word(1, "per", "per", UniversalPos::Adp, 2, "case", None),
            it_word(
                2,
                "favore",
                "favore",
                UniversalPos::Noun,
                0,
                "root",
                Some("Gender=Masc|Number=Sing"),
            ),
            it_range(3, 5, "dammela"),
            it_word(3, "da", "da", UniversalPos::Adp, 4, "case", None),
            it_word(
                4,
                "me",
                "me",
                UniversalPos::Pron,
                2,
                "nmod",
                Some("Clitic=Yes|Person=1|PronType=Prs"),
            ),
            it_word(
                5,
                "la",
                "la",
                UniversalPos::Pron,
                2,
                "expl",
                Some("Clitic=Yes|Person=3|PronType=Prs"),
            ),
        ],
    };
    let (mors, _) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
    let items = chat_strings(&mors);
    assert_eq!(items.len(), 3, "{items:?}");
    assert!(
        items[2].starts_with("verb|dare-Fin-Imp-S2~"),
        "the split compound must be the imperative of `dare`, got {:?}",
        items[2]
    );
}

/// Stanza 1.14's analysis of an isolated `dammela`: split, with `da` as a verb
/// in the imperfect indicative. The curated imperative replaces those features.
#[test]
fn test_italian_dammela_split_with_indicative_features_takes_the_imperative() {
    let sentence = UdSentence {
        words: vec![
            it_range(1, 3, "dammela"),
            it_word(
                1,
                "da",
                "dare",
                UniversalPos::Verb,
                0,
                "root",
                Some("Mood=Ind|Number=Sing|Person=2|Tense=Imp|VerbForm=Fin"),
            ),
            it_word(
                2,
                "me",
                "me",
                UniversalPos::Pron,
                1,
                "expl",
                Some("Clitic=Yes|Person=1|PronType=Prs"),
            ),
            it_word(
                3,
                "la",
                "la",
                UniversalPos::Pron,
                1,
                "expl",
                Some("Clitic=Yes|Person=3|PronType=Prs"),
            ),
        ],
    };
    let (mors, _) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
    let items = chat_strings(&mors);
    assert_eq!(items.len(), 1, "{items:?}");
    assert!(
        items[0].starts_with("verb|dare-Fin-Imp-S2~"),
        "got {:?}",
        items[0]
    );
    assert!(
        !items[0].contains("Ind"),
        "no indicative left, got {:?}",
        items[0]
    );
}
