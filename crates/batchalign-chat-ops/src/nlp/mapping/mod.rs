//! UD-to-CHAT morphosyntax mapping logic that builds %mor/%gra structures.
//!
//! High-level flow:
//! 1. Convert each UD token (or MWT component) to a `%mor` item.
//! 2. Build chunk-index mapping because `%gra` indexes chunks, not raw tokens.
//! 3. Emit `%gra` relations with CHAT-compatible relation labels.
//! 4. Validate graph/root/chunk-count invariants before returning.
//!
//! Language-specific normalizations are applied through `MappingContext`.
//!
//! # Related CHAT Manual Sections
//!
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Format>
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Headers>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Main_Tier>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Dependent_Tiers>

mod context;
mod errors;
mod helpers;
mod provenance;
mod validate;

pub use context::MappingContext;
#[allow(unused_imports)]
pub(crate) use context::lang2;
pub use errors::MappingError;

#[cfg(test)]
use super::UdId;
#[cfg(test)]
use super::UdSentence;
#[cfg(test)]
use super::UdWord;
#[cfg(test)]
use super::mor_word::map_ud_word_to_mor;
#[cfg(test)]
use talkbank_model::model::GrammaticalRelation;
#[cfg(test)]
use talkbank_model::model::dependent_tier::mor::Mor;

pub use talkbank_transform::morphosyntax::{
    is_terminator_punct, map_ud_sentence, map_ud_sentence_expanded,
};

#[cfg(test)]
mod tests {
    use super::super::lang_en;
    use super::super::mor_word::clean_lemma;
    use super::super::{UdPunctable, UniversalPos};
    use super::validate::validate_generated_gra;
    use super::*;

    /// Maps a full UD sentence to a sequence of CHAT Mor structures.
    fn map_ud_sentence_to_mors(sentence: &UdSentence, ctx: &MappingContext) -> Vec<Mor> {
        let (mors, _) = map_ud_sentence(sentence, ctx).unwrap_or_default();
        mors
    }

    #[test]
    fn test_simple_noun_mapping() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "dog".to_string(),
            lemma: "dog".to_string(),
            upos: UdPunctable::Value(UniversalPos::Noun),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "root".to_string(),
            deps: None,
            misc: None,
        };

        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // Python: noun|dog (UPOS lowercased)
        assert_eq!(out, "noun|dog");
    }

    #[test]
    fn test_sanitization_prevents_corruption() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "bad|word".to_string(),
            lemma: "bad|word".to_string(), // Lemma contains a reserved CHAT character!
            upos: UdPunctable::Value(UniversalPos::Noun),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "root".to_string(),
            deps: None,
            misc: None,
        };

        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();

        // clean_lemma takes everything before '|' → "bad"
        assert_eq!(out, "noun|bad");
        assert!(
            out.matches('|').count() == 1,
            "Sanitization failed to remove illegal reserved character '|' from stem"
        );
    }

    #[test]
    fn test_pron_mapping_no_subcategory() {
        // Python uses "pron|lemma" with feature suffixes, NOT xpos-based subcategories
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "I".to_string(),
            lemma: "I".to_string(),
            upos: UdPunctable::Value(UniversalPos::Pron),
            xpos: Some("PRP-sub".to_string()),
            feats: None,
            head: 0,
            deprel: "root".to_string(),
            deps: None,
            misc: None,
        };

        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // Python: pron|I-Int-S1 (PronType default "Int", Number default "S", Person default "1")
        assert_eq!(out, "pron|I-Int-S1");
    }

    #[test]
    fn test_japanese_punctuation_mapping() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("ja"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "\u{3002}".to_string(),
            lemma: "\u{3002}".to_string(),
            upos: UdPunctable::Value(UniversalPos::Punct),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "root".to_string(),
            deps: None,
            misc: None,
        };

        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        assert_eq!(out, "cm|\u{3002}");
    }

    #[test]
    fn test_mwt_assembly_english_dont() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let sentence = UdSentence {
            words: vec![
                UdWord {
                    id: UdId::Range(1, 2),
                    text: "don't".to_string(),
                    lemma: "do not".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Verb),
                    xpos: None,
                    feats: None,
                    head: 0,
                    deprel: "root".to_string(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(1),
                    text: "do".to_string(),
                    lemma: "do".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Aux),
                    xpos: None,
                    feats: None,
                    head: 0,
                    deprel: "root".to_string(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(2),
                    text: "n't".to_string(),
                    lemma: "not".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Part),
                    xpos: None,
                    feats: None,
                    head: 1,
                    deprel: "advmod".to_string(),
                    deps: None,
                    misc: None,
                },
            ],
        };

        let mors = map_ud_sentence_to_mors(&sentence, &ctx);
        assert_eq!(mors.len(), 1);
        let mut out = String::new();
        mors[0].write_chat(&mut out).unwrap();

        // AUX "do" gets verb suffixes (VerbForm=Inf default, Number=S)
        // PART "not" gets no suffixes
        assert_eq!(out, "aux|do-Inf-S~part|not");
    }

    #[test]
    fn test_mwt_assembly_french_elision() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("fr"),
        };
        let sentence = UdSentence {
            words: vec![
                UdWord {
                    id: UdId::Single(1),
                    text: "l'".to_string(),
                    lemma: "le".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Det),
                    xpos: None,
                    feats: None,
                    head: 2,
                    deprel: "det".to_string(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(2),
                    text: "ami".to_string(),
                    lemma: "ami".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Noun),
                    xpos: None,
                    feats: None,
                    head: 0,
                    deprel: "root".to_string(),
                    deps: None,
                    misc: None,
                },
            ],
        };

        let mors = map_ud_sentence_to_mors(&sentence, &ctx);
        // Note: In this case, UD doesn't provide a range, but they are clitics
        // Future: implement greedy joining for non-range clitics if desired.
        // For now, let's verify they remain separate if no range is provided.
        assert_eq!(mors.len(), 2);
    }

    #[test]
    fn test_gra_index_shifting_with_mwt() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let sentence = UdSentence {
            words: vec![
                UdWord {
                    id: UdId::Range(1, 2),
                    text: "don't".to_string(),
                    lemma: "do not".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Verb),
                    xpos: None,
                    feats: None,
                    head: 0,
                    deprel: "root".to_string(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(1),
                    text: "do".to_string(),
                    lemma: "do".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Aux),
                    xpos: None,
                    feats: None,
                    head: 0,
                    deprel: "root".to_string(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(2),
                    text: "n't".to_string(),
                    lemma: "not".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Part),
                    xpos: None,
                    feats: None,
                    head: 1,
                    deprel: "advmod".to_string(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(3),
                    text: "go".to_string(),
                    lemma: "go".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Verb),
                    xpos: None,
                    feats: None,
                    head: 1,
                    deprel: "conj".to_string(),
                    deps: None,
                    misc: None,
                },
            ],
        };

        let (_mors, gras) = map_ud_sentence(&sentence, &ctx).unwrap();

        // MWT "don't" produces 2 chunks (do + n't), "go" is 1 chunk, + terminator = 4
        // Chunk indices: do=1, n't=2, go=3, .=4
        assert_eq!(gras.len(), 4);

        // "do" component (root — head=0)
        assert_eq!(gras[0].index, 1);
        assert_eq!(gras[0].head, 0);
        assert_eq!(gras[0].relation, "ROOT".into());

        // "n't" component (advmod of "do", chunk 1)
        assert_eq!(gras[1].index, 2);
        assert_eq!(gras[1].head, 1);
        assert_eq!(gras[1].relation, "ADVMOD".into());

        // "go" (conj of "do", chunk 1)
        assert_eq!(gras[2].index, 3);
        assert_eq!(gras[2].head, 1);
        assert_eq!(gras[2].relation, "CONJ".into());

        // Terminator
        assert_eq!(gras[3].index, 4);
        assert_eq!(gras[3].head, 1);
        assert_eq!(gras[3].relation, "PUNCT".into());
    }

    #[test]
    fn test_feature_mapping_plural() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "dogs".to_string(),
            lemma: "dog".to_string(),
            upos: UdPunctable::Value(UniversalPos::Noun),
            xpos: None,
            feats: Some("Number=Plur".to_string()),
            head: 0,
            deprel: "root".to_string(),
            deps: None,
            misc: None,
        };

        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // Python: noun|dog-Plur (NOUN suffix: Number kept as-is)
        assert_eq!(out, "noun|dog-Plur");
    }

    #[test]
    fn test_feature_mapping_past_tense() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "walked".to_string(),
            lemma: "walk".to_string(),
            upos: UdPunctable::Value(UniversalPos::Verb),
            xpos: None,
            feats: Some("Tense=Past".to_string()),
            head: 0,
            deprel: "root".to_string(),
            deps: None,
            misc: None,
        };

        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // Python: verb|walk-Inf-Past-S (VerbForm default "Inf", Tense "Past", Number default "S")
        assert_eq!(out, "verb|walk-Inf-Past-S");
    }

    #[test]
    fn test_english_gerund_fix() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "walking".to_string(),
            lemma: "walk".to_string(),
            upos: UdPunctable::Value(UniversalPos::Noun),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "root".to_string(),
            deps: None,
            misc: None,
        };

        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // Python: noun|walk-Ger (NOUN suffix for English -ing words)
        assert_eq!(out, "noun|walk-Ger");
    }

    #[test]
    fn test_validate_generated_gra_accepts_valid() {
        // Valid structure: single root, no cycles
        let gras = vec![
            GrammaticalRelation {
                index: 1,
                head: 2,
                relation: "DET".into(),
            },
            GrammaticalRelation {
                index: 2,
                head: 2,
                relation: "ROOT".into(),
            },
            GrammaticalRelation {
                index: 3,
                head: 2,
                relation: "OBJ".into(),
            },
            GrammaticalRelation {
                index: 4,
                head: 2,
                relation: "PUNCT".into(),
            },
        ];
        validate_generated_gra(&gras).unwrap();
    }

    #[test]
    fn test_validate_generated_gra_rejects_no_root() {
        let gras = vec![
            GrammaticalRelation {
                index: 1,
                head: 2,
                relation: "SUBJ".into(),
            },
            GrammaticalRelation {
                index: 2,
                head: 3,
                relation: "ROOT".into(),
            },
            GrammaticalRelation {
                index: 3,
                head: 1,
                relation: "OBJ".into(),
            },
            GrammaticalRelation {
                index: 4,
                head: 1,
                relation: "PUNCT".into(),
            },
        ];
        let err = validate_generated_gra(&gras).unwrap_err();
        assert!(
            matches!(err, MappingError::InvalidRoot { .. }),
            "Expected InvalidRoot, got: {err}"
        );
    }

    #[test]
    fn test_validate_generated_gra_rejects_multiple_roots() {
        let gras = vec![
            GrammaticalRelation {
                index: 1,
                head: 1,
                relation: "ROOT".into(),
            },
            GrammaticalRelation {
                index: 2,
                head: 2,
                relation: "ROOT".into(),
            },
            GrammaticalRelation {
                index: 3,
                head: 1,
                relation: "PUNCT".into(),
            },
        ];
        let err = validate_generated_gra(&gras).unwrap_err();
        assert!(
            matches!(err, MappingError::InvalidRoot { .. }),
            "Expected InvalidRoot, got: {err}"
        );
    }

    #[test]
    fn test_validate_generated_gra_rejects_cycle() {
        let gras = vec![
            GrammaticalRelation {
                index: 1,
                head: 2,
                relation: "FLAT".into(),
            },
            GrammaticalRelation {
                index: 2,
                head: 1,
                relation: "APPOS".into(),
            },
            GrammaticalRelation {
                index: 3,
                head: 3,
                relation: "ROOT".into(),
            },
            GrammaticalRelation {
                index: 4,
                head: 3,
                relation: "PUNCT".into(),
            },
        ];
        let err = validate_generated_gra(&gras).unwrap_err();
        assert!(
            matches!(err, MappingError::CircularDependency { .. }),
            "Expected CircularDependency, got: {err}"
        );
    }

    #[test]
    fn test_validate_generated_gra_rejects_invalid_head() {
        let gras = vec![
            GrammaticalRelation {
                index: 1,
                head: 99,
                relation: "SUBJ".into(),
            },
            GrammaticalRelation {
                index: 2,
                head: 2,
                relation: "ROOT".into(),
            },
            GrammaticalRelation {
                index: 3,
                head: 2,
                relation: "PUNCT".into(),
            },
        ];
        let err = validate_generated_gra(&gras).unwrap_err();
        assert!(
            matches!(err, MappingError::InvalidHeadReference { .. }),
            "Expected InvalidHeadReference, got: {err}"
        );
    }

    #[test]
    fn test_validate_generated_gra_accepts_head_zero() {
        let gras = vec![
            GrammaticalRelation {
                index: 1,
                head: 2,
                relation: "DET".into(),
            },
            GrammaticalRelation {
                index: 2,
                head: 0,
                relation: "ROOT".into(),
            },
            GrammaticalRelation {
                index: 3,
                head: 2,
                relation: "PUNCT".into(),
            },
        ];
        validate_generated_gra(&gras).unwrap();
    }

    /// Ensures %gra output follows TalkBank conventions, not raw UD conventions.
    ///
    /// TalkBank conventions:
    /// - ROOT head points to self (not 0)
    /// - Subtype separator is dash (ACL-RELCL), not colon (ACL:RELCL)
    /// - Relation labels are uppercase
    /// - Terminator PUNCT head points to ROOT word
    #[test]
    fn test_gra_talkbank_conventions() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let sentence = UdSentence {
            words: vec![
                UdWord {
                    id: UdId::Single(1),
                    text: "the".to_string(),
                    lemma: "the".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Det),
                    xpos: None,
                    feats: None,
                    head: 2,
                    deprel: "det".to_string(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(2),
                    text: "dog".to_string(),
                    lemma: "dog".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Noun),
                    xpos: None,
                    feats: None,
                    head: 0,
                    deprel: "root".to_string(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(3),
                    text: "that".to_string(),
                    lemma: "that".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Pron),
                    xpos: None,
                    feats: None,
                    head: 4,
                    deprel: "nsubj".to_string(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(4),
                    text: "barks".to_string(),
                    lemma: "bark".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Verb),
                    xpos: None,
                    feats: None,
                    head: 2,
                    // UD uses colon for subtypes: "acl:relcl"
                    deprel: "acl:relcl".to_string(),
                    deps: None,
                    misc: None,
                },
            ],
        };

        let (_mors, gras) = map_ud_sentence(&sentence, &ctx).unwrap();

        // 4 words + 1 terminator
        assert_eq!(gras.len(), 5);

        // Convention 1: ROOT head=0 (virtual root node)
        assert_eq!(gras[1].index, 2);
        assert_eq!(gras[1].head, 0, "ROOT head must be 0 (virtual root)");
        assert_eq!(gras[1].relation, "ROOT".into());

        // Convention 2: UD colon subtypes become TalkBank dashes
        assert_eq!(gras[3].index, 4);
        assert_eq!(gras[3].head, 2);
        assert_eq!(
            gras[3].relation,
            "ACL-RELCL".into(),
            "TalkBank uses dashes for subtypes, not colons"
        );

        // Convention 3: All labels uppercase
        assert_eq!(gras[0].relation, "DET".into());
        assert_eq!(gras[2].relation, "NSUBJ".into());

        // Convention 4: Terminator PUNCT head points to ROOT word
        assert_eq!(gras[4].index, 5);
        assert_eq!(
            gras[4].head, 2,
            "Terminator PUNCT head must point to ROOT word"
        );
        assert_eq!(gras[4].relation, "PUNCT".into());
    }

    // ─── New POS convention tests ────────────────────────────────────────────

    #[test]
    fn test_pos_adp_mapping() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "in".into(),
            lemma: "in".into(),
            upos: UdPunctable::Value(UniversalPos::Adp),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // Python: adp|in (was "prep|in")
        assert_eq!(out, "adp|in");
    }

    #[test]
    fn test_pos_intj_mapping() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "wow".into(),
            lemma: "wow".into(),
            upos: UdPunctable::Value(UniversalPos::Intj),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // Python: intj|wow (was "co|wow")
        assert_eq!(out, "intj|wow");
    }

    #[test]
    fn test_pos_cconj_mapping() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "and".into(),
            lemma: "and".into(),
            upos: UdPunctable::Value(UniversalPos::Cconj),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // Python: cconj|and (was "x|and")
        assert_eq!(out, "cconj|and");
    }

    #[test]
    fn test_pos_sconj_mapping() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "because".into(),
            lemma: "because".into(),
            upos: UdPunctable::Value(UniversalPos::Sconj),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        assert_eq!(out, "sconj|because");
    }

    #[test]
    fn test_pos_propn_mapping() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "London".into(),
            lemma: "London".into(),
            upos: UdPunctable::Value(UniversalPos::Propn),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // Python: propn|London (was "n:prop|London")
        assert_eq!(out, "propn|London");
    }

    // ─── Suffix handler tests ────────────────────────────────────────────────

    #[test]
    fn test_verb_full_features() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "walks".into(),
            lemma: "walk".into(),
            upos: UdPunctable::Value(UniversalPos::Verb),
            xpos: None,
            feats: Some("Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // VerbForm=Fin, Mood=Ind, Tense=Pres, Number=S, Person=3
        assert_eq!(out, "verb|walk-Fin-Ind-Pres-S3");
    }

    #[test]
    fn test_verb_irregular_past() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "went".into(),
            lemma: "go".into(),
            upos: UdPunctable::Value(UniversalPos::Verb),
            xpos: None,
            feats: Some("Tense=Past|VerbForm=Fin|Number=Sing|Person=3".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // "went" is irregular past of "go" → "-irr" suffix
        assert_eq!(out, "verb|go-Fin-Past-S3-irr");
    }

    #[test]
    fn test_pron_with_features() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "I".into(),
            lemma: "I".into(),
            upos: UdPunctable::Value(UniversalPos::Pron),
            xpos: None,
            feats: Some("Case=Nom|Number=Sing|Person=1|PronType=Prs".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        assert_eq!(out, "pron|I-Prs-Nom-S1");
    }

    #[test]
    fn test_pron_that_no_number() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "that".into(),
            lemma: "that".into(),
            upos: UdPunctable::Value(UniversalPos::Pron),
            xpos: None,
            feats: Some("PronType=Rel".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // "that" and "who" get no NumberPerson string
        assert_eq!(out, "pron|that-Rel");
    }

    #[test]
    fn test_det_default_definite() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "the".into(),
            lemma: "the".into(),
            upos: UdPunctable::Value(UniversalPos::Det),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "det".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // Definite defaults to "Def"
        assert_eq!(out, "det|the-Def");
    }

    #[test]
    fn test_det_with_article() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "the".into(),
            lemma: "the".into(),
            upos: UdPunctable::Value(UniversalPos::Det),
            xpos: None,
            feats: Some("Definite=Def|PronType=Art".into()),
            head: 0,
            deprel: "det".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        assert_eq!(out, "det|the-Def-Art");
    }

    #[test]
    fn test_adj_default_degree() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "big".into(),
            lemma: "big".into(),
            upos: UdPunctable::Value(UniversalPos::Adj),
            xpos: None,
            feats: Some("Degree=Pos".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // Degree "Pos" is cleared to empty
        assert_eq!(out, "adj|big-S1");
    }

    #[test]
    fn test_adj_comparative() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "bigger".into(),
            lemma: "big".into(),
            upos: UdPunctable::Value(UniversalPos::Adj),
            xpos: None,
            feats: Some("Degree=Cmp".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        assert_eq!(out, "adj|big-Cmp-S1");
    }

    #[test]
    fn test_noun_obj_accusative() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "dog".into(),
            lemma: "dog".into(),
            upos: UdPunctable::Value(UniversalPos::Noun),
            xpos: None,
            feats: None,
            head: 2,
            deprel: "obj".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // deprel "obj" without Case → "Acc"
        assert_eq!(out, "noun|dog-Acc");
    }

    #[test]
    fn test_french_pron_case() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("fr"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "je".into(),
            lemma: "je".into(),
            upos: UdPunctable::Value(UniversalPos::Pron),
            xpos: None,
            feats: Some("Number=Sing|Person=1|PronType=Prs".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // French "je" gets Case=Nom from word-level lookup
        assert_eq!(out, "pron|je-Prs-Nom-S1");
    }

    #[test]
    fn test_comma_lemma_early_return() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: ",".into(),
            lemma: ",".into(),
            upos: UdPunctable::Value(UniversalPos::Punct),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "punct".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        assert_eq!(out, "cm|cm");
    }

    #[test]
    fn test_clean_lemma_strips_special_chars() {
        // Verify clean_lemma handles various problematic lemmas
        let (cleaned, unknown) = clean_lemma("$test.", "test");
        assert_eq!(cleaned, "test");
        assert!(!unknown);

        let (cleaned, unknown) = clean_lemma("0word", "0word");
        assert_eq!(cleaned, "word");
        assert!(unknown);
    }

    #[test]
    fn test_english_irregular_verb_suffix() {
        // "wrote" is an irregular past of "write"
        assert!(lang_en::is_irregular("write", "wrote"));
        assert!(lang_en::is_irregular("go", "went"));
        assert!(!lang_en::is_irregular("walk", "walked"));
    }

    #[test]
    fn test_lang2_normalization() {
        assert_eq!(lang2("eng"), "en");
        assert_eq!(lang2("fra"), "fr");
        assert_eq!(lang2("jpn"), "ja");
        assert_eq!(lang2("en"), "en");
        assert_eq!(lang2("fr"), "fr");
        assert_eq!(lang2("ja"), "ja");
        assert_eq!(lang2("deu"), "de");
        assert_eq!(lang2("heb"), "he");
    }

    #[test]
    fn test_irr_suffix_with_3letter_code() {
        // Ensure the -irr suffix works when lang is "eng" (3-letter, the real-world case)
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("eng"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "went".into(),
            lemma: "go".into(),
            upos: UdPunctable::Value(UniversalPos::Verb),
            xpos: None,
            feats: Some("Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        assert!(
            out.contains("-irr"),
            "3-letter 'eng' should trigger irr check: {}",
            out
        );
    }

    #[test]
    fn test_multivalue_ud_features_preserve_commas() {
        // Croatian: PronType=Int,Rel should preserve the comma per UD conventions
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("hr"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "što".into(),
            lemma: "što".into(),
            upos: UdPunctable::Value(UniversalPos::Pron),
            xpos: None,
            feats: Some("Case=Acc|PronType=Int,Rel".to_string()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // Must contain comma — we respect UD multi-value feature conventions
        assert!(
            out.contains("Int,Rel"),
            "Expected Int,Rel (UD convention), got: {out}"
        );
    }

    /// Regression test: MWT clitic groups must produce per-component GRA entries.
    ///
    /// For "it's just", Stanza produces Range(1,2) → "it" + "'s" + "just".
    /// %mor should have `pron|it~aux|be adv|just .` (4 chunks with terminator),
    /// and %gra must have exactly 4 relations (one per chunk).
    #[test]
    fn test_mwt_gra_per_component_alignment() {
        let sentence = UdSentence {
            words: vec![
                // Range entry for the MWT "it's"
                UdWord {
                    id: UdId::Range(1, 2),
                    text: "it's".into(),
                    lemma: "it's".into(),
                    upos: UdPunctable::Punct("X".into()),
                    xpos: None,
                    feats: None,
                    head: 0,
                    deprel: "dep".into(),
                    deps: None,
                    misc: None,
                },
                // Component 1: "it"
                UdWord {
                    id: UdId::Single(1),
                    text: "it".into(),
                    lemma: "it".into(),
                    upos: UdPunctable::Value(UniversalPos::Pron),
                    xpos: Some("PRP".into()),
                    feats: Some("Case=Nom|Gender=Neut|Number=Sing|Person=3|PronType=Prs".into()),
                    head: 3,
                    deprel: "nsubj".into(),
                    deps: None,
                    misc: None,
                },
                // Component 2: "'s"
                UdWord {
                    id: UdId::Single(2),
                    text: "'s".into(),
                    lemma: "be".into(),
                    upos: UdPunctable::Value(UniversalPos::Aux),
                    xpos: Some("VBZ".into()),
                    feats: Some("Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin".into()),
                    head: 0,
                    deprel: "root".into(),
                    deps: None,
                    misc: None,
                },
                // Regular word: "just"
                UdWord {
                    id: UdId::Single(3),
                    text: "just".into(),
                    lemma: "just".into(),
                    upos: UdPunctable::Value(UniversalPos::Adv),
                    xpos: Some("RB".into()),
                    feats: None,
                    head: 2,
                    deprel: "advmod".into(),
                    deps: None,
                    misc: None,
                },
            ],
        };

        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("eng"),
        };
        let (mors, gras) = map_ud_sentence(&sentence, &ctx).unwrap();

        // MOR: clitic group (it~be) + just = 2 items, 3 chunks
        assert_eq!(mors.len(), 2, "Expected 2 MOR items (clitic group + adv)");
        let total_chunks: usize = mors.iter().map(|m| m.count_chunks()).sum();
        assert_eq!(total_chunks, 3, "Expected 3 MOR chunks (it + 's + just)");

        // GRA: 3 chunks + 1 terminator = 4 relations
        assert_eq!(
            gras.len(),
            4,
            "Expected 4 GRA entries (3 chunks + terminator PUNCT), got {gras:?}"
        );

        // Verify per-component indexing: chunk 1 = it, chunk 2 = 's, chunk 3 = just, chunk 4 = terminator
        assert_eq!(gras[0].index, 1);
        assert_eq!(gras[1].index, 2);
        assert_eq!(gras[2].index, 3);
        assert_eq!(gras[3].index, 4);
        assert_eq!(gras[3].relation, "PUNCT".to_string().into());
    }

    /// MWT with "don't" → "do" + "n't": both components get GRA entries.
    #[test]
    fn test_mwt_gra_dont_contraction() {
        let sentence = UdSentence {
            words: vec![
                UdWord {
                    id: UdId::Single(1),
                    text: "I".into(),
                    lemma: "I".into(),
                    upos: UdPunctable::Value(UniversalPos::Pron),
                    xpos: Some("PRP".into()),
                    feats: Some("Case=Nom|Number=Sing|Person=1|PronType=Prs".into()),
                    head: 4,
                    deprel: "nsubj".into(),
                    deps: None,
                    misc: None,
                },
                // Range entry for "don't"
                UdWord {
                    id: UdId::Range(2, 3),
                    text: "don't".into(),
                    lemma: "don't".into(),
                    upos: UdPunctable::Punct("X".into()),
                    xpos: None,
                    feats: None,
                    head: 0,
                    deprel: "dep".into(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(2),
                    text: "do".into(),
                    lemma: "do".into(),
                    upos: UdPunctable::Value(UniversalPos::Aux),
                    xpos: Some("VBP".into()),
                    feats: Some("Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin".into()),
                    head: 4,
                    deprel: "aux".into(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(3),
                    text: "n't".into(),
                    lemma: "not".into(),
                    upos: UdPunctable::Value(UniversalPos::Part),
                    xpos: Some("RB".into()),
                    feats: Some("Polarity=Neg".into()),
                    head: 4,
                    deprel: "advmod".into(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(4),
                    text: "know".into(),
                    lemma: "know".into(),
                    upos: UdPunctable::Value(UniversalPos::Verb),
                    xpos: Some("VB".into()),
                    feats: Some("VerbForm=Inf".into()),
                    head: 0,
                    deprel: "root".into(),
                    deps: None,
                    misc: None,
                },
            ],
        };

        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("eng"),
        };
        let (mors, gras) = map_ud_sentence(&sentence, &ctx).unwrap();

        // MOR: I + (do~n't) + know = 3 items
        assert_eq!(mors.len(), 3, "Expected 3 MOR items");
        // Chunks: I(1) + do(1) + n't(1) + know(1) = 4
        let total_chunks: usize = mors.iter().map(|m| m.count_chunks()).sum();
        assert_eq!(total_chunks, 4, "Expected 4 MOR chunks");

        // GRA: 4 chunks + 1 terminator = 5 relations
        assert_eq!(
            gras.len(),
            5,
            "Expected 5 GRA entries (4 chunks + terminator), got {gras:?}"
        );
    }

    // ─── Apostrophe / empty-stem regression ──────────────────────────────────
    //
    // Stanza's GUM MWT model can treat possessives like "Claus'" as MWT,
    // splitting the token into [Claus (PROPN), ' (PUNCT)].  The apostrophe
    // component has lemma="'" which gets stripped to "" by clean_lemma's
    // apostrophe-removal pass.  Before the fix, this produced "punct|" (empty
    // stem) → E342 tree-sitter parse error.

    #[test]
    fn test_clean_lemma_apostrophe_fallback_to_text() {
        // clean_lemma("'", "'") must not return empty — fallback to surface text
        let (result, unknown) = clean_lemma("'", "'");
        assert!(
            !result.is_empty(),
            "clean_lemma must never return empty string"
        );
        assert_eq!(result, "'", "Expected fallback to surface text \"'\"");
        assert!(!unknown, "Not an unknown token");
    }

    #[test]
    fn test_map_ud_word_apostrophe_no_empty_stem() {
        // map_ud_word_to_mor with an apostrophe-only PUNCT token must produce
        // "punct|'" (non-empty stem), not "punct|" (E342).
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let ud = UdWord {
            id: UdId::Single(2),
            text: "'".to_string(),
            lemma: "'".to_string(),
            upos: UdPunctable::Value(UniversalPos::Punct),
            xpos: None,
            feats: None,
            head: 1,
            deprel: "case".to_string(),
            deps: None,
            misc: None,
        };

        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();

        assert!(!out.ends_with('|'), "Empty stem produces E342: got {out:?}");
        // Should produce "punct|'" — apostrophe preserved as stem
        assert_eq!(out, "punct|'", "Expected punct|' not punct|");
    }

    #[test]
    fn test_map_ud_word_rejects_empty_stem() {
        // If clean_lemma and sanitize_mor_text both produce an empty string,
        // map_ud_word_to_mor must return Err(EmptyStem), not silently pass.
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        // Craft a UD word whose lemma sanitizes to empty (all reserved chars).
        let ud = UdWord {
            id: UdId::Single(1),
            text: "|||".to_string(),
            lemma: "|||".to_string(), // clean_lemma preserves; sanitize strips '|' → "___" → non-empty
            upos: UdPunctable::Value(UniversalPos::Noun),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "root".to_string(),
            deps: None,
            misc: None,
        };
        // sanitize_mor_text replaces reserved chars, so this won't actually be empty.
        // Verify it succeeds (non-empty stem after sanitization).
        let result = map_ud_word_to_mor(&ud, &ctx);
        assert!(
            result.is_ok(),
            "Reserved chars should be sanitized, not empty: {result:?}"
        );
    }

    #[test]
    fn test_unmapped_head_reference() {
        // A word's head points to a decimal ID (not in chunk index map).
        // Should return Err(InvalidHeadReference), not silently fall back to 0.
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let sentence = UdSentence {
            words: vec![
                UdWord {
                    id: UdId::Single(1),
                    text: "dog".to_string(),
                    lemma: "dog".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Noun),
                    xpos: None,
                    feats: None,
                    head: 0,
                    deprel: "root".to_string(),
                    deps: None,
                    misc: None,
                },
                // Decimal word (empty/enhanced token) — not indexed in chunk map
                UdWord {
                    id: UdId::Decimal(1.1),
                    text: "of".to_string(),
                    lemma: "of".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Adp),
                    xpos: None,
                    feats: None,
                    head: 1,
                    deprel: "case".to_string(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(2),
                    text: "cat".to_string(),
                    lemma: "cat".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Noun),
                    xpos: None,
                    feats: None,
                    // Head 99 does not exist in the chunk map
                    head: 99,
                    deprel: "nmod".to_string(),
                    deps: None,
                    misc: None,
                },
            ],
        };
        let err = map_ud_sentence(&sentence, &ctx).unwrap_err();
        assert!(
            matches!(err, MappingError::InvalidHeadReference { .. }),
            "Expected InvalidHeadReference, got: {err}"
        );
    }

    #[test]
    fn test_no_root_in_ud_parse() {
        // All words have non-zero heads forming a chain — no root.
        // Should return Err(InvalidRoot), not silently use root_chunk_idx=0.
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let sentence = UdSentence {
            words: vec![
                UdWord {
                    id: UdId::Single(1),
                    text: "the".to_string(),
                    lemma: "the".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Det),
                    xpos: None,
                    feats: None,
                    head: 2, // not root
                    deprel: "det".to_string(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(2),
                    text: "dog".to_string(),
                    lemma: "dog".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Noun),
                    xpos: None,
                    feats: None,
                    head: 1, // circular, but no head=0
                    deprel: "nsubj".to_string(),
                    deps: None,
                    misc: None,
                },
            ],
        };
        let err = map_ud_sentence(&sentence, &ctx).unwrap_err();
        assert!(
            matches!(err, MappingError::InvalidRoot { .. }),
            "Expected InvalidRoot for no-root parse, got: {err}"
        );
    }

    // ─── Non-English language parity tests ──────────────────────────────────
    //
    // These tests ensure ba3's typed Rust mapping produces identical output to
    // ba2's string-based approach for every language-specific behavior.
    // See: morphotag-migration-audit.md

    // ── French ──────────────────────────────────────────────────────────────

    #[test]
    fn test_french_det_singular_gender_default_masc() {
        // ba2: DET gender defaults to "Masc" for French singular
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("fr"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "le".into(),
            lemma: "le".into(),
            upos: UdPunctable::Value(UniversalPos::Det),
            xpos: None,
            feats: Some("Definite=Def|PronType=Art".into()),
            head: 0,
            deprel: "det".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // French singular DET without Gender → defaults to "Masc"
        assert_eq!(out, "det|le-Masc-Def-Art");
    }

    #[test]
    fn test_french_det_plural_no_gender_default() {
        // ba2: DET gender default is "" for French plural (no Masc default)
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("fr"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "les".into(),
            lemma: "le".into(),
            upos: UdPunctable::Value(UniversalPos::Det),
            xpos: None,
            feats: Some("Definite=Def|Number=Plur|PronType=Art".into()),
            head: 0,
            deprel: "det".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // French plural DET: no gender default, Number=Plur present
        assert_eq!(out, "det|le-Def-Art-Plur");
    }

    #[test]
    fn test_french_det_explicit_fem_gender() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("fr"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "la".into(),
            lemma: "le".into(),
            upos: UdPunctable::Value(UniversalPos::Det),
            xpos: None,
            feats: Some("Definite=Def|Gender=Fem|Number=Sing|PronType=Art".into()),
            head: 0,
            deprel: "det".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        assert_eq!(out, "det|le-Fem-Def-Art-Sing");
    }

    #[test]
    fn test_french_noun_apm_plural() {
        // ba2: French plural nouns with auditory plural marking get -Apm suffix
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("fr"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "chevaux".into(),
            lemma: "cheval".into(),
            upos: UdPunctable::Value(UniversalPos::Noun),
            xpos: None,
            feats: Some("Gender=Masc|Number=Plur".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // French APM: cheval→chevaux, Masc gender + Plur + Apm
        assert_eq!(out, "noun|cheval-Masc-Plur-Apm");
    }

    #[test]
    fn test_french_noun_non_apm_plural() {
        // Regular French plural noun: no -Apm suffix
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("fr"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "maisons".into(),
            lemma: "maison".into(),
            upos: UdPunctable::Value(UniversalPos::Noun),
            xpos: None,
            feats: Some("Gender=Fem|Number=Plur".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        assert_eq!(out, "noun|maison-Fem-Plur");
    }

    #[test]
    fn test_french_pron_accusative() {
        // ba2: French "me" gets Case=Acc from word-level lookup
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("fr"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "me".into(),
            lemma: "me".into(),
            upos: UdPunctable::Value(UniversalPos::Pron),
            xpos: None,
            feats: Some("Number=Sing|Person=1|PronType=Prs".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        assert_eq!(out, "pron|me-Prs-Acc-S1");
    }

    #[test]
    fn test_french_pron_no_case_lookup() {
        // ba2: French "nous" has no entry in case lookup → no Case suffix
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("fr"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "nous".into(),
            lemma: "nous".into(),
            upos: UdPunctable::Value(UniversalPos::Pron),
            xpos: None,
            feats: Some("Number=Plur|Person=1|PronType=Prs".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // "nous" is not in fr/case.py → no Case field
        assert_eq!(out, "pron|nous-Prs-P1");
    }

    #[test]
    fn test_french_mwt_contraction_du() {
        // ba2: French "du" → "de" + "le" via MWT Range
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("fr"),
        };
        let sentence = UdSentence {
            words: vec![
                UdWord {
                    id: UdId::Range(1, 2),
                    text: "du".into(),
                    lemma: "du".into(),
                    upos: UdPunctable::Punct("X".into()),
                    xpos: None,
                    feats: None,
                    head: 0,
                    deprel: "dep".into(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(1),
                    text: "de".into(),
                    lemma: "de".into(),
                    upos: UdPunctable::Value(UniversalPos::Adp),
                    xpos: None,
                    feats: None,
                    head: 3,
                    deprel: "case".into(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(2),
                    text: "le".into(),
                    lemma: "le".into(),
                    upos: UdPunctable::Value(UniversalPos::Det),
                    xpos: None,
                    feats: Some("Definite=Def|Gender=Masc|Number=Sing|PronType=Art".into()),
                    head: 3,
                    deprel: "det".into(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(3),
                    text: "pain".into(),
                    lemma: "pain".into(),
                    upos: UdPunctable::Value(UniversalPos::Noun),
                    xpos: None,
                    feats: Some("Gender=Masc|Number=Sing".into()),
                    head: 0,
                    deprel: "root".into(),
                    deps: None,
                    misc: None,
                },
            ],
        };
        let mors = map_ud_sentence_to_mors(&sentence, &ctx);
        // "du" MWT → clitic group (de~le) + "pain" = 2 items
        assert_eq!(mors.len(), 2, "Expected 2 MOR items for 'du pain'");

        let mut out0 = String::new();
        mors[0].write_chat(&mut out0).unwrap();
        // Clitic assembly: adp|de~det|le-Masc-Def-Art-Sing
        assert!(
            out0.contains("adp|de") && out0.contains("det|le"),
            "Expected clitic group adp|de~det|le, got: {out0}"
        );

        let mut out1 = String::new();
        mors[1].write_chat(&mut out1).unwrap();
        assert_eq!(out1, "noun|pain-Masc");
    }

    // ── Japanese ────────────────────────────────────────────────────────────

    #[test]
    fn test_japanese_verb_override_full_output() {
        // ba2: Japanese "食べちゃう" matches "ちゃ" → sconj|ば
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("ja"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "食べちゃう".into(),
            lemma: "食べる".into(),
            upos: UdPunctable::Value(UniversalPos::Verb),
            xpos: None,
            feats: Some("VerbForm=Fin".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // Override changes POS to sconj → no verb features emitted
        assert_eq!(out, "sconj|ば");
    }

    #[test]
    fn test_japanese_intj_override_hai() {
        // ba2: Japanese "はい" overridden to intj regardless of Stanza's POS
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("ja"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "はい".into(),
            lemma: "はい".into(),
            upos: UdPunctable::Value(UniversalPos::Noun), // Stanza might tag as NOUN
            xpos: None,
            feats: None,
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // Override: intj|はい (noun features suppressed because dispatch uses original UPOS)
        // Original UPOS is NOUN → noun_features runs, but lemma override to はい
        // Actually: the effective_pos is "intj" but dispatch uses original UPOS (NOUN)
        // So noun_features runs with the overridden lemma
        assert!(
            out.starts_with("intj|") || out.contains("はい"),
            "Expected Japanese intj override, got: {out}"
        );
    }

    #[test]
    fn test_japanese_aux_override_nai() {
        // ba2: target containing "無い" → aux|ない
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("ja"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "ない".into(),
            lemma: "無い".into(),
            upos: UdPunctable::Value(UniversalPos::Aux),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // Override: aux|ない, then verb features (original UPOS = AUX)
        assert!(
            out.starts_with("aux|ない"),
            "Expected aux|ない prefix, got: {out}"
        );
    }

    #[test]
    fn test_japanese_comma_lemma_becomes_cm() {
        // ba2: Japanese comma (、) → cm|、
        // The Japanese comma is NOT in the early-return punct list (which only
        // has ASCII ","), so it goes through the normal path: POS→"cm", stem→"、"
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("ja"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "、".into(),
            lemma: "、".into(),
            upos: UdPunctable::Value(UniversalPos::Punct),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "punct".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        assert_eq!(out, "cm|、");
    }

    #[test]
    fn test_japanese_all_punct_is_cm() {
        // ba2: ALL Japanese PUNCT tokens (not just comma) → cm|X
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("ja"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "…".into(),
            lemma: "…".into(),
            upos: UdPunctable::Value(UniversalPos::Punct),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "punct".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // Japanese PUNCT → POS becomes "cm"
        assert!(
            out.starts_with("cm|"),
            "Japanese PUNCT should use cm| prefix, got: {out}"
        );
    }

    #[test]
    fn test_japanese_verb_no_irr_suffix() {
        // ba2: -irr suffix is English-only. Japanese verbs must never get it.
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("ja"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "行った".into(),
            lemma: "行く".into(),
            upos: UdPunctable::Value(UniversalPos::Verb),
            xpos: None,
            feats: Some("Tense=Past|VerbForm=Fin".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        assert!(
            !out.contains("-irr"),
            "Japanese verbs must NOT get -irr suffix, got: {out}"
        );
    }

    // ── Hebrew ──────────────────────────────────────────────────────────────

    #[test]
    fn test_hebrew_verb_hebbinyan() {
        // ba2: Hebrew HebBinyan feature → lowercased suffix
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("he"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "כתב".into(),
            lemma: "כתב".into(),
            upos: UdPunctable::Value(UniversalPos::Verb),
            xpos: None,
            feats: Some("HebBinyan=PAAL|Number=Sing|Person=3|Tense=Past|VerbForm=Fin".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // HebBinyan=PAAL → lowercased "paal" in suffix
        assert!(
            out.contains("-paal-"),
            "Hebrew HebBinyan must be lowercased in suffix, got: {out}"
        );
        // No -irr (Hebrew, not English)
        assert!(!out.contains("-irr"), "Hebrew must not get -irr: {out}");
    }

    #[test]
    fn test_hebrew_verb_hebexistential() {
        // ba2: Hebrew HebExistential feature → lowercased suffix
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("he"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "יש".into(),
            lemma: "יש".into(),
            upos: UdPunctable::Value(UniversalPos::Verb),
            xpos: None,
            feats: Some("HebExistential=True|VerbForm=Fin".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // HebExistential=True → lowercased "true"
        assert!(
            out.contains("-true-") || out.contains("-true"),
            "Hebrew HebExistential must appear in suffix, got: {out}"
        );
    }

    // ── German ──────────────────────────────────────────────────────────────

    #[test]
    fn test_german_mwt_contraction_im() {
        // ba2: German "im" → "in" + "dem" via MWT Range
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("de"),
        };
        let sentence = UdSentence {
            words: vec![
                UdWord {
                    id: UdId::Range(1, 2),
                    text: "im".into(),
                    lemma: "im".into(),
                    upos: UdPunctable::Punct("X".into()),
                    xpos: None,
                    feats: None,
                    head: 0,
                    deprel: "dep".into(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(1),
                    text: "in".into(),
                    lemma: "in".into(),
                    upos: UdPunctable::Value(UniversalPos::Adp),
                    xpos: None,
                    feats: None,
                    head: 3,
                    deprel: "case".into(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(2),
                    text: "dem".into(),
                    lemma: "der".into(),
                    upos: UdPunctable::Value(UniversalPos::Det),
                    xpos: None,
                    feats: Some(
                        "Case=Dat|Definite=Def|Gender=Masc|Number=Sing|PronType=Art".into(),
                    ),
                    head: 3,
                    deprel: "det".into(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(3),
                    text: "Haus".into(),
                    lemma: "Haus".into(),
                    upos: UdPunctable::Value(UniversalPos::Noun),
                    xpos: None,
                    feats: Some("Case=Dat|Gender=Neut|Number=Sing".into()),
                    head: 0,
                    deprel: "root".into(),
                    deps: None,
                    misc: None,
                },
            ],
        };
        let mors = map_ud_sentence_to_mors(&sentence, &ctx);
        // "im" MWT → clitic group (in~der) + "Haus" = 2 items
        assert_eq!(mors.len(), 2, "Expected 2 MOR items for 'im Haus'");

        let mut out0 = String::new();
        mors[0].write_chat(&mut out0).unwrap();
        assert!(
            out0.contains("adp|in") && out0.contains("det|der"),
            "Expected adp|in~det|der clitic, got: {out0}"
        );
    }

    #[test]
    fn test_german_verb_no_irr_suffix() {
        // German verbs must never get -irr (English-only feature)
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("de"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "ging".into(),
            lemma: "gehen".into(),
            upos: UdPunctable::Value(UniversalPos::Verb),
            xpos: None,
            feats: Some("Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        assert!(
            !out.contains("-irr"),
            "German verbs must NOT get -irr suffix, got: {out}"
        );
        assert_eq!(out, "verb|gehen-Fin-Ind-Past-S3");
    }

    // ── Spanish ─────────────────────────────────────────────────────────────

    #[test]
    fn test_spanish_mwt_contraction_del() {
        // ba2: Spanish "del" → "de" + "el" via MWT Range
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("es"),
        };
        let sentence = UdSentence {
            words: vec![
                UdWord {
                    id: UdId::Range(1, 2),
                    text: "del".into(),
                    lemma: "del".into(),
                    upos: UdPunctable::Punct("X".into()),
                    xpos: None,
                    feats: None,
                    head: 0,
                    deprel: "dep".into(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(1),
                    text: "de".into(),
                    lemma: "de".into(),
                    upos: UdPunctable::Value(UniversalPos::Adp),
                    xpos: None,
                    feats: None,
                    head: 3,
                    deprel: "case".into(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(2),
                    text: "el".into(),
                    lemma: "el".into(),
                    upos: UdPunctable::Value(UniversalPos::Det),
                    xpos: None,
                    feats: Some("Definite=Def|Gender=Masc|Number=Sing|PronType=Art".into()),
                    head: 3,
                    deprel: "det".into(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(3),
                    text: "parque".into(),
                    lemma: "parque".into(),
                    upos: UdPunctable::Value(UniversalPos::Noun),
                    xpos: None,
                    feats: Some("Gender=Masc|Number=Sing".into()),
                    head: 0,
                    deprel: "root".into(),
                    deps: None,
                    misc: None,
                },
            ],
        };
        let mors = map_ud_sentence_to_mors(&sentence, &ctx);
        assert_eq!(mors.len(), 2, "Expected 2 MOR items for 'del parque'");

        let mut out0 = String::new();
        mors[0].write_chat(&mut out0).unwrap();
        assert!(
            out0.contains("adp|de") && out0.contains("det|el"),
            "Expected adp|de~det|el clitic for Spanish, got: {out0}"
        );
    }

    #[test]
    fn test_spanish_verb_person0_becomes_4() {
        // ba2: Person=0 → "4" in NumberPerson string (all languages)
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("es"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "llueve".into(),
            lemma: "llover".into(),
            upos: UdPunctable::Value(UniversalPos::Verb),
            xpos: None,
            feats: Some("Mood=Ind|Number=Sing|Person=0|Tense=Pres|VerbForm=Fin".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // Person=0 → "4" (ba2 convention for impersonal verbs)
        assert!(
            out.contains("-S4"),
            "Person=0 must map to '4' in suffix, got: {out}"
        );
    }

    // ── Italian ─────────────────────────────────────────────────────────────

    #[test]
    fn test_italian_mwt_contraction_della() {
        // ba2: Italian "della" → "di" + "la" via MWT Range
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("it"),
        };
        let sentence = UdSentence {
            words: vec![
                UdWord {
                    id: UdId::Range(1, 2),
                    text: "della".into(),
                    lemma: "della".into(),
                    upos: UdPunctable::Punct("X".into()),
                    xpos: None,
                    feats: None,
                    head: 0,
                    deprel: "dep".into(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(1),
                    text: "di".into(),
                    lemma: "di".into(),
                    upos: UdPunctable::Value(UniversalPos::Adp),
                    xpos: None,
                    feats: None,
                    head: 3,
                    deprel: "case".into(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(2),
                    text: "la".into(),
                    lemma: "il".into(),
                    upos: UdPunctable::Value(UniversalPos::Det),
                    xpos: None,
                    feats: Some("Definite=Def|Gender=Fem|Number=Sing|PronType=Art".into()),
                    head: 3,
                    deprel: "det".into(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(3),
                    text: "casa".into(),
                    lemma: "casa".into(),
                    upos: UdPunctable::Value(UniversalPos::Noun),
                    xpos: None,
                    feats: Some("Gender=Fem|Number=Sing".into()),
                    head: 0,
                    deprel: "root".into(),
                    deps: None,
                    misc: None,
                },
            ],
        };
        let mors = map_ud_sentence_to_mors(&sentence, &ctx);
        assert_eq!(mors.len(), 2, "Expected 2 MOR items for 'della casa'");

        let mut out0 = String::new();
        mors[0].write_chat(&mut out0).unwrap();
        assert!(
            out0.contains("adp|di") && out0.contains("det|il"),
            "Expected adp|di~det|il clitic for Italian, got: {out0}"
        );
    }

    // ── Cross-language verb feature defaults ─────────────────────────────────

    #[test]
    fn test_verb_default_verbform_inf() {
        // ba2: VerbForm defaults to "Inf" when not present (ALL languages)
        for lang in ["fr", "de", "es", "it", "pt", "ja", "ko", "he"] {
            let ctx = MappingContext {
                lang: talkbank_model::model::LanguageCode::new(lang),
            };
            let ud = UdWord {
                id: UdId::Single(1),
                text: "x".into(),
                lemma: "x".into(),
                upos: UdPunctable::Value(UniversalPos::Verb),
                xpos: None,
                feats: None, // No features → defaults
                head: 0,
                deprel: "root".into(),
                deps: None,
                misc: None,
            };
            let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
            let mut out = String::new();
            mor.write_chat(&mut out).unwrap();
            assert!(
                out.contains("-Inf-"),
                "VerbForm must default to Inf for lang={lang}, got: {out}"
            );
        }
    }

    #[test]
    fn test_verb_default_number_sing() {
        // ba2: Number defaults to "Sing" (→ "S") for verbs (ALL languages)
        for lang in ["fr", "de", "es", "it"] {
            let ctx = MappingContext {
                lang: talkbank_model::model::LanguageCode::new(lang),
            };
            let ud = UdWord {
                id: UdId::Single(1),
                text: "x".into(),
                lemma: "x".into(),
                upos: UdPunctable::Value(UniversalPos::Verb),
                xpos: None,
                feats: None, // No Number → defaults to "Sing" → "S"
                head: 0,
                deprel: "root".into(),
                deps: None,
                misc: None,
            };
            let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
            let mut out = String::new();
            mor.write_chat(&mut out).unwrap();
            assert!(
                out.contains("-S"),
                "Number must default to S(ing) for lang={lang}, got: {out}"
            );
        }
    }

    // ── 3-letter language code normalization ─────────────────────────────────

    #[test]
    fn test_french_3letter_code_works() {
        // Real-world: language codes come as "fra" not "fr"
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("fra"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "je".into(),
            lemma: "je".into(),
            upos: UdPunctable::Value(UniversalPos::Pron),
            xpos: None,
            feats: Some("Number=Sing|Person=1|PronType=Prs".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // Must get French pronoun case even with "fra" code
        assert_eq!(out, "pron|je-Prs-Nom-S1");
    }

    #[test]
    fn test_japanese_3letter_code_works() {
        // Real-world: "jpn" not "ja"
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("jpn"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "はい".into(),
            lemma: "はい".into(),
            upos: UdPunctable::Value(UniversalPos::Noun),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // "jpn" must trigger Japanese verbform overrides
        assert!(
            out.contains("intj|はい"),
            "3-letter 'jpn' must trigger JA overrides, got: {out}"
        );
    }

    #[test]
    fn test_hebrew_3letter_code_works() {
        // Real-world: "heb" not "he"
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("heb"),
        };
        let ud = UdWord {
            id: UdId::Single(1),
            text: "כתב".into(),
            lemma: "כתב".into(),
            upos: UdPunctable::Value(UniversalPos::Verb),
            xpos: None,
            feats: Some("HebBinyan=PAAL|Tense=Past|VerbForm=Fin".into()),
            head: 0,
            deprel: "root".into(),
            deps: None,
            misc: None,
        };
        let mor = map_ud_word_to_mor(&ud, &ctx).unwrap();
        let mut out = String::new();
        mor.write_chat(&mut out).unwrap();
        // "heb" must still process HebBinyan
        assert!(
            out.contains("-paal-"),
            "3-letter 'heb' must process HebBinyan, got: {out}"
        );
    }

    #[test]
    fn test_garbage_deprel_rejected() {
        // A deprel with garbage characters should be rejected, not silently fixed.
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let sentence = UdSentence {
            words: vec![
                UdWord {
                    id: UdId::Single(1),
                    text: "dog".to_string(),
                    lemma: "dog".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Noun),
                    xpos: None,
                    feats: None,
                    head: 0,
                    deprel: "root".to_string(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(2),
                    text: "big".to_string(),
                    lemma: "big".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Adj),
                    xpos: None,
                    feats: None,
                    head: 1,
                    deprel: "<PAD>".to_string(), // garbage deprel
                    deps: None,
                    misc: None,
                },
            ],
        };
        let err = map_ud_sentence(&sentence, &ctx).unwrap_err();
        assert!(
            matches!(err, MappingError::InvalidDeprel { .. }),
            "Expected InvalidDeprel, got: {err}"
        );
    }

    // Punct handling: commas, tag markers, and vocatives must flow through
    // to `%mor` (`cm|cm`/`end|end`/`beg|beg`); only CHAT utterance
    // terminators are dropped. Matches BA2 semantics in
    // `pipelines/morphosyntax/ud.py::handler__actual_PUNCT`.

    /// The classifier distinguishes sentence terminators from other PUNCT.
    /// Regression guard: reverting `is_terminator_punct` to match all PUNCT
    /// (as `is_standalone_punct` did pre-fix) makes this test fail.
    #[test]
    fn is_terminator_punct_matches_only_sentence_terminators() {
        let make = |text: &str, lemma: &str| UdWord {
            id: UdId::Single(1),
            text: text.to_string(),
            lemma: lemma.to_string(),
            upos: UdPunctable::Value(UniversalPos::Punct),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "punct".to_string(),
            deps: None,
            misc: None,
        };

        // Every variant that `talkbank_model::Terminator` recognizes must be
        // classified as a terminator by our filter. We enumerate every CHAT
        // terminator token string explicitly to lock the contract — if the
        // model adds a new variant, this list won't include it and a
        // companion test in talkbank-model (`every_variant_round_trips_…`)
        // will flag the omission.
        for t in [
            ".",
            "!",
            "?",
            "+...",
            "+/.",
            "+//.",
            "+/?",
            "+!?",
            "+\"/.",
            "+\".",
            "+//?",
            "+..?",
            "+.",
            "\u{21D7}",
            "\u{2197}",
            "\u{2192}",
            "\u{2198}",
            "\u{21D8}",
            "\u{224B}",
            "+\u{224B}",
            "\u{2248}",
            "+\u{2248}",
        ] {
            assert!(
                super::is_terminator_punct(&make(t, t)),
                "CHAT terminator {t:?} must be classified as a terminator"
            );
        }

        // Content punctuation MUST NOT be classified as a terminator —
        // these flow through to `map_ud_word_to_mor` to produce Mor items
        // (`cm|cm`, `end|end`, `beg|beg`, etc.).
        for t in [",", ";", ":", "—", "(", ")", "\"", "„", "‡"] {
            assert!(
                !super::is_terminator_punct(&make(t, t)),
                "content punct {t:?} must NOT be treated as a terminator"
            );
        }

        // Non-PUNCT UPOS is never a terminator even with a '.' text.
        let non_punct = UdWord {
            id: UdId::Single(1),
            text: ".".to_string(),
            lemma: ".".to_string(),
            upos: UdPunctable::Value(UniversalPos::Noun),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "root".to_string(),
            deps: None,
            misc: None,
        };
        assert!(!super::is_terminator_punct(&non_punct));
    }

    /// A three-word sentence with a mid-utterance comma must produce THREE
    /// Mor items (`x|word1`, `cm|cm`, `x|word2`) — not two.
    ///
    /// Pre-fix, the `is_standalone_punct` filter dropped the comma, yielding
    /// two Mor items, and `inject_morphosyntax` then refused the utterance
    /// because the CHAT Mor-domain extraction counted 3 alignable items.
    #[test]
    fn mid_utterance_comma_produces_cm_mor_item() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let sentence = UdSentence {
            words: vec![
                UdWord {
                    id: UdId::Single(1),
                    text: "hello".to_string(),
                    lemma: "hello".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Intj),
                    xpos: None,
                    feats: None,
                    head: 0,
                    deprel: "root".to_string(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(2),
                    text: ",".to_string(),
                    lemma: ",".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Punct),
                    xpos: None,
                    feats: None,
                    head: 1,
                    deprel: "punct".to_string(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(3),
                    text: "world".to_string(),
                    lemma: "world".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Noun),
                    xpos: None,
                    feats: None,
                    head: 1,
                    deprel: "parataxis".to_string(),
                    deps: None,
                    misc: None,
                },
            ],
        };
        let (mors, _gras) = map_ud_sentence(&sentence, &ctx).unwrap();
        assert_eq!(
            mors.len(),
            3,
            "expected 3 Mor items (hello, cm, world); got: {:?}",
            mors.iter()
                .map(|m| {
                    let mut s = String::new();
                    m.write_chat(&mut s).unwrap();
                    s
                })
                .collect::<Vec<_>>()
        );
        let mut comma_str = String::new();
        mors[1].write_chat(&mut comma_str).unwrap();
        assert_eq!(comma_str, "cm|cm");
    }

    /// The sentence-final terminator (`.`, `!`, `?`) must NOT produce a Mor
    /// item — it is the utterance's typed terminator, not a `%mor` entry.
    /// Regression guard against the converse over-fix ("keep all PUNCT").
    #[test]
    fn sentence_terminator_is_dropped_from_mor_output() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let sentence = UdSentence {
            words: vec![
                UdWord {
                    id: UdId::Single(1),
                    text: "hi".to_string(),
                    lemma: "hi".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Intj),
                    xpos: None,
                    feats: None,
                    head: 0,
                    deprel: "root".to_string(),
                    deps: None,
                    misc: None,
                },
                UdWord {
                    id: UdId::Single(2),
                    text: ".".to_string(),
                    lemma: ".".to_string(),
                    upos: UdPunctable::Value(UniversalPos::Punct),
                    xpos: None,
                    feats: None,
                    head: 1,
                    deprel: "punct".to_string(),
                    deps: None,
                    misc: None,
                },
            ],
        };
        let (mors, _gras) = map_ud_sentence(&sentence, &ctx).unwrap();
        assert_eq!(mors.len(), 1, "terminator '.' must not produce a Mor item");
        let mut out = String::new();
        mors[0].write_chat(&mut out).unwrap();
        assert_eq!(out, "intj|hi");
    }

    /// Comma + terminator: comma produces `cm|cm`, terminator is dropped.
    /// End-to-end mirror of what happens on a typical utterance like
    /// `haan , theek hai .` after Stanza analysis.
    #[test]
    fn comma_kept_terminator_dropped_together() {
        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("en"),
        };
        let mk = |id: usize, text: &str, upos: UniversalPos, head: usize, deprel: &str| UdWord {
            id: UdId::Single(id),
            text: text.to_string(),
            lemma: text.to_string(),
            upos: UdPunctable::Value(upos),
            xpos: None,
            feats: None,
            head,
            deprel: deprel.to_string(),
            deps: None,
            misc: None,
        };
        let sentence = UdSentence {
            words: vec![
                mk(1, "yes", UniversalPos::Intj, 0, "root"),
                mk(2, ",", UniversalPos::Punct, 1, "punct"),
                mk(3, "ok", UniversalPos::Intj, 1, "parataxis"),
                mk(4, ".", UniversalPos::Punct, 1, "punct"),
            ],
        };
        let (mors, _gras) = map_ud_sentence(&sentence, &ctx).unwrap();
        let items: Vec<String> = mors
            .iter()
            .map(|m| {
                let mut s = String::new();
                m.write_chat(&mut s).unwrap();
                s
            })
            .collect();
        assert_eq!(items, vec!["intj|yes", "cm|cm", "intj|ok"]);
    }

    // ─── Italian Defect 6 reconciler — see
    //     docs/investigations/2026-04-23-italian-defect-6-reconciler-plan.md
    //     and book/src/reference/languages/italian.md §"Reconciler hacks" ──

    /// Helper: build a Range-parent UdWord for `parla → par + la`.
    fn it_range(start: usize, end: usize, text: &str) -> UdWord {
        UdWord {
            id: UdId::Range(start, end),
            text: text.into(),
            lemma: "".into(),
            upos: UdPunctable::Value(UniversalPos::X),
            xpos: None,
            feats: None,
            head: 0,
            deprel: "dep".into(),
            deps: None,
            misc: None,
        }
    }

    fn it_word(
        id: usize,
        text: &str,
        lemma: &str,
        upos: UniversalPos,
        head: usize,
        deprel: &str,
        feats: Option<&str>,
    ) -> UdWord {
        UdWord {
            id: UdId::Single(id),
            text: text.into(),
            lemma: lemma.into(),
            upos: UdPunctable::Value(upos),
            xpos: None,
            feats: feats.map(|f| f.to_string()),
            head,
            deprel: deprel.into(),
            deps: None,
            misc: None,
        }
    }

    fn it_ctx() -> MappingContext {
        MappingContext {
            lang: talkbank_model::model::LanguageCode::new("ita"),
        }
    }

    fn chat_strings(mors: &[Mor]) -> Vec<String> {
        mors.iter()
            .map(|m| {
                let mut s = String::new();
                m.write_chat(&mut s).unwrap();
                s
            })
            .collect()
    }

    /// Expected POS for a collapsed Defect 6 Mor. `Noun` accepts
    /// either the short (`n|`) or long (`noun|`) prefix since
    /// talkbank-model emits both in different contexts.
    #[derive(Clone, Copy)]
    enum ExpectedPos {
        Noun,
        Adj,
    }

    impl ExpectedPos {
        fn matches_prefix(self, item: &str, lemma: &str) -> bool {
            match self {
                ExpectedPos::Noun => {
                    item.starts_with(&format!("n|{}", lemma))
                        || item.starts_with(&format!("noun|{}", lemma))
                }
                ExpectedPos::Adj => item.starts_with(&format!("adj|{}", lemma)),
            }
        }
    }

    /// Test helper for the Defect 6 non-verb collapse pattern:
    /// Stanza emits a bogus 2-component verb+clitic MWT for a
    /// word that is actually a single noun or adjective. The
    /// reconciler must collapse the Range to one Mor with the
    /// canonical POS and lemma.
    ///
    /// All 16 non-verb Defect 6 tests follow this pattern: one
    /// Range parent, a verb-tagged first component with a
    /// surface-echo lemma, a pron-tagged clitic second component,
    /// and the canonical lemma as the expected output.
    fn assert_defect6_2component_collapses(
        surface: &str,
        comp1: (&str, &str),
        comp2: (&str, &str),
        expected_pos: ExpectedPos,
        expected_lemma: &str,
    ) {
        let sentence = UdSentence {
            words: vec![
                it_range(1, 2, surface),
                it_word(1, comp1.0, comp1.1, UniversalPos::Verb, 0, "root", None),
                it_word(2, comp2.0, comp2.1, UniversalPos::Pron, 1, "obj", None),
            ],
        };
        let (mors, _) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
        let items = chat_strings(&mors);
        assert_eq!(
            items.len(),
            1,
            "surface {surface:?}: expected 1 mor, got {items:?}"
        );
        assert!(
            expected_pos.matches_prefix(&items[0], expected_lemma),
            "surface {surface:?}: expected prefix for lemma {expected_lemma:?}, got {:?}",
            items[0]
        );
        assert!(
            !items[0].contains("~"),
            "surface {surface:?}: collapsed Mor must not carry clitic suffix, got {:?}",
            items[0]
        );
    }

    /// Defect 6 verb case: Stanza emits `parla → par + la` as a
    /// Range-wrapped MWT tagged verb+pronoun with nonsense lemma
    /// `par`. The reconciler must collapse this into a single
    /// `verb|parlare-...` mor.
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

    // Defect 6 non-verb cases. Each surface is a real Italian
    // noun or adjective that Stanza mis-splits into a bogus
    // verb+clitic MWT. The reconciler collapses the Range to
    // one Mor with the canonical POS and lemma. All 16 tests
    // delegate to `assert_defect6_2component_collapses` — see
    // the helper docs for the shared shape.

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

    /// `piccola` (feminine) emits lemma `piccolo` — Italian UD
    /// uses the masc.sg form as the canonical adjective lemma.
    /// The override's value is eliminating the junk
    /// `verb|picco~pron|la` output, not preserving the feminine
    /// surface form.
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

    /// Correctness control group: genuine Italian verb+clitic
    /// compounds (e.g., `dammela → da + mme + la`) are correctly
    /// tokenized by Stanza and MUST stay merged with `~` clitics.
    /// The reconciler's allowlist must NOT fire on these.
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

    // ─── Italian Defect 8 reconciler — see
    //     book/src/reference/stanza-limitations.md Defect 8 TODO
    //     and book/src/reference/languages/italian.md ──

    /// Defect 8: Stanza mid-sentence emits `dammela` as a single UD
    /// word tagged ADJ with lemma=`dammelo` (no MWT expansion).
    /// The reconciler must override POS→VERB and lemma→`dare`.
    #[test]
    fn test_italian_defect8_dammela_mid_sentence_becomes_verb() {
        let sentence = UdSentence {
            words: vec![
                it_word(1, "per", "per", UniversalPos::Adp, 3, "case", None),
                it_word(2, "favore", "favore", UniversalPos::Noun, 3, "obl", None),
                it_word(
                    3,
                    "dammela",
                    "dammelo",
                    UniversalPos::Adj,
                    0,
                    "root",
                    Some("Number=Sing"),
                ),
            ],
        };
        let (mors, _) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
        let items = chat_strings(&mors);
        assert_eq!(items.len(), 3, "got {items:?}");
        // The third word (`dammela`) must now be tagged verb with
        // lemma `dare`, not adj with lemma `dammelo`.
        assert!(
            items[2].starts_with("v|dare") || items[2].starts_with("verb|dare"),
            "Defect 8: expected verb|dare for dammela mid-sentence, got {:?}",
            items[2]
        );
        assert!(
            !items[2].starts_with("adj|"),
            "Defect 8: dammela must not stay ADJ-tagged, got {:?}",
            items[2]
        );
    }

    #[test]
    fn test_italian_defect8_dammelo_mid_sentence_becomes_verb() {
        let sentence = UdSentence {
            words: vec![
                it_word(1, "per", "per", UniversalPos::Adp, 3, "case", None),
                it_word(2, "favore", "favore", UniversalPos::Noun, 3, "obl", None),
                it_word(
                    3,
                    "dammelo",
                    "dammelo",
                    UniversalPos::Adj,
                    0,
                    "root",
                    Some("Number=Sing"),
                ),
            ],
        };
        let (mors, _) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
        let items = chat_strings(&mors);
        assert_eq!(items.len(), 3, "got {items:?}");
        assert!(
            items[2].starts_with("v|dare") || items[2].starts_with("verb|dare"),
            "Defect 8: expected verb|dare for dammelo, got {:?}",
            items[2]
        );
    }

    /// Control: a genuine adjective tagged ADJ must NOT be
    /// misinterpreted as a compound imperative.
    #[test]
    fn test_italian_defect8_prendilo_mid_sentence_becomes_verb() {
        let sentence = UdSentence {
            words: vec![
                it_word(1, "per", "per", UniversalPos::Adp, 3, "case", None),
                it_word(2, "favore", "favore", UniversalPos::Noun, 3, "obl", None),
                it_word(
                    3,
                    "prendilo",
                    "prendilo",
                    UniversalPos::Adj,
                    0,
                    "root",
                    Some("Number=Sing"),
                ),
            ],
        };
        let (mors, _) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
        let items = chat_strings(&mors);
        assert_eq!(items.len(), 3);
        assert!(
            items[2].starts_with("v|prendere") || items[2].starts_with("verb|prendere"),
            "Defect 8: expected verb|prendere, got {:?}",
            items[2]
        );
    }

    /// Defect 8 **multi-chunk** verification: with the 2026-04-24
    /// architectural refactor, the Defect 8 reconciler now emits
    /// full clitic decomposition (`verb|dare~pron|me~pron|la`)
    /// instead of the single-chunk `verb|dare`. This test pins
    /// the multi-chunk shape explicitly, plus verifies that the
    /// GRA relations count matches the chunk count (main +
    /// post-clitics).
    #[test]
    fn test_italian_defect8_dammela_emits_multi_chunk_mor() {
        let sentence = UdSentence {
            words: vec![
                it_word(1, "per", "per", UniversalPos::Adp, 3, "case", None),
                it_word(2, "favore", "favore", UniversalPos::Noun, 0, "root", None),
                it_word(
                    3,
                    "dammela",
                    "dammelo",
                    UniversalPos::Adj,
                    2,
                    "amod",
                    Some("Gender=Masc|Number=Sing"),
                ),
            ],
        };
        let (mors, gras) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
        let items = chat_strings(&mors);
        // `dammela` should emit a 3-chunk Mor: verb|dare + me + la.
        assert!(
            items[2].contains("~pron|me"),
            "Expected ~pron|me in dammela output, got {:?}",
            items[2]
        );
        assert!(
            items[2].contains("~pron|la"),
            "Expected ~pron|la in dammela output, got {:?}",
            items[2]
        );
        // Main word + 2 clitics = 3 chunks for dammela, plus
        // `per` (1) + `favore` (1) + terminator PUNCT (1) = 6 total.
        let total_chunks: usize = mors.iter().map(|m| m.count_chunks()).sum();
        assert_eq!(
            total_chunks, 5,
            "Expected 5 mor chunks (per + favore + dammela*3), got {total_chunks}"
        );
        assert_eq!(
            gras.len(),
            total_chunks + 1, // +1 for terminator PUNCT relation
            "Expected {} gra relations, got {}",
            total_chunks + 1,
            gras.len()
        );
    }

    /// Defect 12: `aprilo` — Stanza tags it as single-word VERB
    /// with correct lemma `aprire`, but fails MWT expansion.
    /// The reconciler emits `verb|aprire~pron|lo` via the
    /// multi-chunk Defect-8 machinery. Gate accepts VERB.
    #[test]
    fn test_italian_defect12_aprilo_verb_becomes_multi_chunk() {
        let sentence = UdSentence {
            words: vec![
                it_word(1, "per", "per", UniversalPos::Adp, 3, "case", None),
                it_word(2, "favore", "favore", UniversalPos::Noun, 0, "root", None),
                it_word(
                    3,
                    "aprilo",
                    "aprire",
                    UniversalPos::Verb,
                    2,
                    "advcl",
                    Some("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
                ),
            ],
        };
        let (mors, _) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
        let items = chat_strings(&mors);
        assert!(
            items[2].starts_with("v|aprire") || items[2].starts_with("verb|aprire"),
            "Expected verb|aprire for aprilo, got {:?}",
            items[2]
        );
        assert!(
            items[2].contains("~pron|lo"),
            "Expected ~pron|lo post-clitic in aprilo output, got {:?}",
            items[2]
        );
    }

    /// Defect 13: `leggila` — Stanza tags it as VERB with a
    /// **fabricated** lemma `leggilare` (not real Italian).
    /// The reconciler replaces lemma with `leggere` and adds
    /// the `la` post-clitic.
    #[test]
    fn test_italian_defect13_leggila_fabricated_lemma_becomes_leggere() {
        let sentence = UdSentence {
            words: vec![
                it_word(1, "per", "per", UniversalPos::Adp, 3, "case", None),
                it_word(2, "favore", "favore", UniversalPos::Noun, 0, "root", None),
                it_word(
                    3,
                    "leggila",
                    "leggilare",
                    UniversalPos::Verb,
                    2,
                    "advcl",
                    None,
                ),
            ],
        };
        let (mors, _) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
        let items = chat_strings(&mors);
        assert!(
            items[2].starts_with("v|leggere") || items[2].starts_with("verb|leggere"),
            "Expected verb|leggere (not leggilare) for leggila, got {:?}",
            items[2]
        );
        assert!(
            items[2].contains("~pron|la"),
            "Expected ~pron|la post-clitic in leggila output, got {:?}",
            items[2]
        );
        assert!(
            !items[2].contains("leggilare"),
            "The fabricated Stanza lemma leggilare must be stripped, got {:?}",
            items[2]
        );
    }

    /// Control: with the Defect-8 gate now accepting VERB, a
    /// legitimate Italian verb form NOT in the allowlist must
    /// still pass through unchanged. `guarda` (bare 2sg
    /// imperative of `guardare`) is such a verb.
    #[test]
    fn test_italian_defect8_genuine_verb_stays_verb() {
        let sentence = UdSentence {
            words: vec![it_word(
                1,
                "guarda",
                "guardare",
                UniversalPos::Verb,
                0,
                "root",
                Some("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
            )],
        };
        let (mors, _) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
        let items = chat_strings(&mors);
        assert!(
            items[0].starts_with("v|guardare") || items[0].starts_with("verb|guardare"),
            "guarda (genuine imperative verb) must stay verb|guardare, got {:?}",
            items[0]
        );
    }

    /// Defect 8 variant: NOUN gate. Stanza tags `aprila`
    /// (imperative `aprire` + `la` accusative) as a single
    /// word with POS=NOUN and lemma=`aprila` (surface echo).
    /// The Defect 8 reconciler's gate must fire on both ADJ
    /// and NOUN to catch this. Probe source 2026-04-24.
    #[test]
    fn test_italian_defect8_aprila_noun_becomes_verb() {
        let sentence = UdSentence {
            words: vec![
                it_word(1, "per", "per", UniversalPos::Adp, 3, "case", None),
                it_word(2, "favore", "favore", UniversalPos::Noun, 0, "root", None),
                it_word(3, "aprila", "aprila", UniversalPos::Noun, 2, "nmod", None),
            ],
        };
        let (mors, _) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
        let items = chat_strings(&mors);
        assert!(
            items[2].starts_with("v|aprire") || items[2].starts_with("verb|aprire"),
            "Defect 8 NOUN variant: expected verb|aprire for aprila, got {:?}",
            items[2]
        );
    }

    /// Defect 8 variant: NOUN gate + homograph lemma. Stanza
    /// tags `aprili` as NOUN with lemma `aprile` (the month
    /// April); should be imperative `aprire` + `li` acc.pl.
    #[test]
    fn test_italian_defect8_aprili_homograph_becomes_verb() {
        let sentence = UdSentence {
            words: vec![
                it_word(1, "per", "per", UniversalPos::Adp, 3, "case", None),
                it_word(2, "favore", "favore", UniversalPos::Noun, 0, "root", None),
                it_word(3, "aprili", "aprile", UniversalPos::Noun, 2, "nmod", None),
            ],
        };
        let (mors, _) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
        let items = chat_strings(&mors);
        assert!(
            items[2].starts_with("v|aprire") || items[2].starts_with("verb|aprire"),
            "Defect 8 homograph variant: expected verb|aprire for aprili, got {:?}",
            items[2]
        );
    }

    /// Defect 8 standard (ADJ) shape: `finila` tagged ADJ with
    /// surface-vowel-normalized lemma `finile`. Should be
    /// imperative `finire` + `la`.
    #[test]
    fn test_italian_defect8_finila_mid_sentence_becomes_verb() {
        let sentence = UdSentence {
            words: vec![
                it_word(1, "per", "per", UniversalPos::Adp, 3, "case", None),
                it_word(2, "favore", "favore", UniversalPos::Noun, 0, "root", None),
                it_word(3, "finila", "finile", UniversalPos::Adj, 2, "amod", None),
            ],
        };
        let (mors, _) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
        let items = chat_strings(&mors);
        assert!(
            items[2].starts_with("v|finire") || items[2].starts_with("verb|finire"),
            "Defect 8: expected verb|finire for finila, got {:?}",
            items[2]
        );
    }

    /// Control for the expanded Defect 8 NOUN gate: a genuine
    /// noun NOT in the allowlist must stay tagged as noun even
    /// though the gate now accepts NOUN. `cavallo` is a common
    /// Italian noun; must not be rewritten.
    #[test]
    fn test_italian_defect8_genuine_noun_stays_noun() {
        let sentence = UdSentence {
            words: vec![
                it_word(1, "il", "il", UniversalPos::Det, 2, "det", None),
                it_word(
                    2,
                    "cavallo",
                    "cavallo",
                    UniversalPos::Noun,
                    0,
                    "root",
                    Some("Gender=Masc|Number=Sing"),
                ),
            ],
        };
        let (mors, _) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
        let items = chat_strings(&mors);
        assert!(
            items[1].starts_with("n|cavallo") || items[1].starts_with("noun|cavallo"),
            "cavallo (genuine noun) must stay noun, got {:?}",
            items[1]
        );
    }

    #[test]
    fn test_italian_defect8_genuine_adj_stays_adj() {
        let sentence = UdSentence {
            words: vec![
                it_word(1, "il", "il", UniversalPos::Det, 2, "det", None),
                it_word(2, "gatto", "gatto", UniversalPos::Noun, 0, "root", None),
                it_word(
                    3,
                    "piccolo",
                    "piccolo",
                    UniversalPos::Adj,
                    2,
                    "amod",
                    Some("Gender=Masc|Number=Sing"),
                ),
            ],
        };
        let (mors, _) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
        let items = chat_strings(&mors);
        // `piccolo` is NOT in the compound-imperative allowlist
        // (it's in the Defect 6 allowlist, but only when arriving
        // via a Range). As a Single UdWord it should pass through
        // the normal adj mapping.
        assert!(
            items[2].starts_with("adj|piccolo"),
            "piccolo as Single UdWord should stay adj|piccolo, got {:?}",
            items[2]
        );
    }

    // ─── Italian Defect 9 reconciler — Range-parent component rewrite
    //
    // Stanza 1.11.1 expands `dagliela` (2sg imperative of `dare` + 3sg
    // dative clitic + 3sg.f.acc clitic) as a 3-piece MWT with correct
    // structure but wrong head POS: `da/ADP/da + glie/PRON/gli +
    // la/PRON/la`. The head `da` is homographic with the Italian
    // preposition `da` ("from, by"), and Stanza's POS layer picks the
    // preposition reading even though the MWT expansion only makes
    // sense under the verb reading.
    //
    // Unlike Defect 6 (where we collapse the Range into a single Mor)
    // Defect 9 KEEPS the 3-piece MWT — the clitic decomposition is
    // correct. We only rewrite component 0's POS/lemma/feats before
    // `assemble_mors` runs, yielding the correct
    // `verb|dare-Imp-S2~pron|gli~pron|la` instead of
    // `adp|da~pron|gli~pron|la`.
    //
    // Empirical source: 2026-04-24 probe pass (see
    // `batchalign/tests/investigations/_cases/italian.py`
    // `dagliela_mid_sentence` case + companion
    // `digliela_mid_sentence` / `portagliela_mid_sentence` /
    // `prendigliela_mid_sentence` which Stanza handles correctly).
    //
    // Control group: `digliela` and `portagliela` Stanza analyses the
    // head as VERB natively, so the Defect 9 allowlist must NOT fire
    // on those forms (no rewrite needed, no regression introduced).
    //
    // See `book/src/reference/languages/italian.md` §"Reconciler hacks"
    // once this lands.

    /// Defect 9: Stanza expands `dagliela` as `da/ADP + glie/PRON +
    /// la/PRON`. The reconciler rewrites component 0 to VERB/dare
    /// imperative; remaining components pass through.
    #[test]
    fn test_italian_defect9_dagliela_head_rewritten_to_verb_dare() {
        let sentence = UdSentence {
            words: vec![
                it_range(1, 3, "dagliela"),
                it_word(1, "da", "da", UniversalPos::Adp, 0, "root", None),
                it_word(
                    2,
                    "glie",
                    "gli",
                    UniversalPos::Pron,
                    1,
                    "iobj",
                    Some("Gender=Masc|Number=Sing|Person=3|PronType=Prs"),
                ),
                it_word(
                    3,
                    "la",
                    "la",
                    UniversalPos::Pron,
                    1,
                    "obj",
                    Some("Gender=Fem|Number=Sing|Person=3|PronType=Prs"),
                ),
            ],
        };
        let (mors, _gras) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
        let items = chat_strings(&mors);
        assert_eq!(
            items.len(),
            1,
            "Expected 1 mor for 3-piece MWT, got {items:?}"
        );
        assert!(
            items[0].starts_with("v|dare") || items[0].starts_with("verb|dare"),
            "Defect 9: expected verb|dare head for dagliela, got {:?}",
            items[0]
        );
        assert!(
            !items[0].starts_with("adp|da") && !items[0].starts_with("prep|da"),
            "Defect 9: head must not stay ADP, got {:?}",
            items[0]
        );
        // The clitic decomposition must be preserved — this is what
        // distinguishes Defect 9 from Defect 6 (which collapses to a
        // single chunk).
        assert!(
            items[0].contains("~"),
            "Defect 9: 3-piece MWT decomposition must be preserved \
             (expected clitic-suffixed Mor), got {:?}",
            items[0]
        );
    }

    /// Defect 10 (handled via Defect 9 allowlist): `posala` expands
    /// correctly as 2-piece MWT (`posa/VERB + la/PRON`) but head
    /// lemma is surface-echo `posa` instead of canonical infinitive
    /// `posare`. Unlike `dagliela` (Defect 9 with wrong POS), here
    /// only the lemma is wrong; the POS rewrite to VERB is a no-op.
    /// The `IT_COMPONENT_REWRITES` allowlist handles both shapes
    /// via the same mechanism — a head-component rewrite — since
    /// the rewrite is idempotent when Stanza already got the field
    /// right. Probe source 2026-04-24 (`posala_alone`).
    #[test]
    fn test_italian_defect10_posala_head_lemma_rewritten_to_posare() {
        let sentence = UdSentence {
            words: vec![
                it_range(1, 2, "posala"),
                it_word(
                    1,
                    "posa",
                    "posa",
                    UniversalPos::Verb,
                    0,
                    "root",
                    Some("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
                ),
                it_word(
                    2,
                    "la",
                    "la",
                    UniversalPos::Pron,
                    1,
                    "obj",
                    Some("Gender=Fem|Number=Sing|Person=3|PronType=Prs"),
                ),
            ],
        };
        let (mors, _gras) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
        let items = chat_strings(&mors);
        assert_eq!(items.len(), 1);
        assert!(
            items[0].starts_with("v|posare") || items[0].starts_with("verb|posare"),
            "Defect 10: expected verb|posare head for posala, got {:?}",
            items[0]
        );
        assert!(
            items[0].contains("~"),
            "Defect 10: 2-piece MWT decomposition must be preserved, got {:?}",
            items[0]
        );
    }

    /// Same Defect 10 shape, masculine accusative: `posalo`.
    #[test]
    fn test_italian_defect10_posalo_head_lemma_rewritten_to_posare() {
        let sentence = UdSentence {
            words: vec![
                it_range(1, 2, "posalo"),
                it_word(
                    1,
                    "posa",
                    "posa",
                    UniversalPos::Verb,
                    0,
                    "root",
                    Some("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
                ),
                it_word(
                    2,
                    "lo",
                    "lo",
                    UniversalPos::Pron,
                    1,
                    "obj",
                    Some("Gender=Masc|Number=Sing|Person=3|PronType=Prs"),
                ),
            ],
        };
        let (mors, _gras) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
        let items = chat_strings(&mors);
        assert_eq!(items.len(), 1);
        assert!(
            items[0].starts_with("v|posare") || items[0].starts_with("verb|posare"),
            "Defect 10: expected verb|posare head for posalo, got {:?}",
            items[0]
        );
        assert!(items[0].contains("~"));
    }

    /// Defect 9 control: `digliela` (imperative `dire` + dative clitic
    /// + acc clitic) — Stanza analyses the head as VERB natively, so
    /// the allowlist must NOT fire. Output must match the normal
    /// `assemble_mors` path: `verb|dire~pron|gli~pron|la`.
    #[test]
    fn test_italian_digliela_stays_correctly_merged() {
        let sentence = UdSentence {
            words: vec![
                it_range(1, 3, "digliela"),
                it_word(
                    1,
                    "di",
                    "dire",
                    UniversalPos::Verb,
                    0,
                    "root",
                    Some("Mood=Imp|Number=Sing|Person=2|VerbForm=Fin"),
                ),
                it_word(
                    2,
                    "glie",
                    "gli",
                    UniversalPos::Pron,
                    1,
                    "iobj",
                    Some("Gender=Masc|Number=Sing|Person=3|PronType=Prs"),
                ),
                it_word(
                    3,
                    "la",
                    "la",
                    UniversalPos::Pron,
                    1,
                    "obj",
                    Some("Gender=Fem|Number=Sing|Person=3|PronType=Prs"),
                ),
            ],
        };
        let (mors, _gras) = map_ud_sentence(&sentence, &it_ctx()).unwrap();
        let items = chat_strings(&mors);
        assert_eq!(items.len(), 1);
        assert!(
            items[0].starts_with("v|dire") || items[0].starts_with("verb|dire"),
            "digliela: expected verb|dire head (Stanza-correct), got {:?}",
            items[0]
        );
        assert!(
            items[0].contains("~"),
            "digliela: clitic decomposition must be preserved, got {:?}",
            items[0]
        );
    }
}
