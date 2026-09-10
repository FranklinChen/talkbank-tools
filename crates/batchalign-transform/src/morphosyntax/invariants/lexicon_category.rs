//! Lexicon-licensed category constraint for English.
//!
//! Stanza's English model reads a sentence-final period as evidence for a
//! plural noun (`whoops .` becomes `whoop` + `Number=Plur`). The CHILDES MOR
//! lexicon licenses `whoops` only as a communicator, and MOR+POST would never
//! have produced that analysis. This rewrite applies the lexicon the way MOR
//! does: where the lexicon's verdict is unambiguous and Stanza's part of
//! speech contradicts it, the lexicon wins; everywhere else Stanza's analysis
//! is left alone. The dependency parse is kept, as the Cantonese POS override
//! keeps it: only the word's category, lemma and category-determined features
//! change.

use crate::morphosyntax::lexicon::{LexiconVerdict, LexiconVerdicts, NounNumber};
use crate::morphosyntax::{UdId, UdPunctable, UdSentence, UniversalPos};

/// CHAT main tiers are lowercase, so a capitalized word is the transcriber
/// marking a proper name (`Gin (.) look at Momma !`): evidence the lexicon
/// does not carry, and a reason to leave the word alone.
pub(super) fn is_transcriber_marked_name(text: &str) -> bool {
    text.chars().next().is_some_and(char::is_uppercase)
}

/// Rewrite, in place, every whole-word token whose lexicon verdict contradicts
/// Stanza's UPOS.
pub fn constrain_to_lexicon(mut sentence: UdSentence, verdicts: &LexiconVerdicts) -> UdSentence {
    // The lexicon speaks about whole CHAT words. A multi-word token's
    // components (`gonna` as `gon` + `na`, and `na` is a communicator) are
    // Stanza's sub-tokens and out of scope; collected once, before the
    // words are borrowed mutably.
    let mwt_components: Vec<_> = sentence.mwt_component_ranges().collect();

    for word in &mut sentence.words {
        // MWT range parents carry no analysis of their own, and punctuation
        // has no category to constrain.
        let UdId::Single(id) = word.id else {
            continue;
        };
        if mwt_components.iter().any(|r| r.contains(&id)) {
            continue;
        }
        let UdPunctable::Value(upos) = word.upos else {
            continue;
        };
        if is_transcriber_marked_name(&word.text) {
            continue;
        }
        let Some(verdict) = verdicts.verdict(&word.text) else {
            continue;
        };
        match verdict {
            // A communicator among other categories is genuinely ambiguous
            // to MOR; only the transcriber's evidence settles it, and that
            // rule ran before this one.
            LexiconVerdict::CommunicatorAmongOthers => continue,
            LexiconVerdict::Communicator => {
                if upos == UniversalPos::Intj {
                    continue;
                }
                tracing::debug!(text = %word.text, from = ?upos, "lexicon: communicator");
                word.upos = UdPunctable::Value(UniversalPos::Intj);
                word.lemma = word.text.to_lowercase();
                word.xpos = Some("UH".to_string());
                word.feats = None;
            }
            LexiconVerdict::Noun(noun) => {
                // A proper noun is a noun; case is evidence the lexicon does
                // not carry, so a name that coincides with a common noun
                // (`Rose`, `Daisy`) is left as Stanza read it.
                if matches!(upos, UniversalPos::Noun | UniversalPos::Propn) {
                    continue;
                }
                tracing::debug!(text = %word.text, from = ?upos, "lexicon: noun");
                word.upos = UdPunctable::Value(UniversalPos::Noun);
                word.lemma = noun.lemma.clone();
                let (xpos, feats) = match noun.number {
                    NounNumber::Singular => ("NN", "Number=Sing"),
                    NounNumber::Plural => ("NNS", "Number=Plur"),
                };
                word.xpos = Some(xpos.to_string());
                word.feats = Some(feats.to_string());
            }
        }
    }
    sentence
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphosyntax::UdWord;
    use crate::morphosyntax::evidence::UtteranceEvidence;
    use crate::morphosyntax::invariants::test_support::{mor_texts, punct, word};
    use crate::morphosyntax::lexicon::{NounVerdict, RawVerdicts, VerdictSource};

    fn test_source() -> VerdictSource {
        VerdictSource {
            repo: "test".into(),
            commit: "test".into(),
            path: "test".into(),
            generator: "test".into(),
        }
    }

    fn verdicts_of(communicator: &[&str], noun: Vec<NounVerdict>) -> LexiconVerdicts {
        LexiconVerdicts::try_from(RawVerdicts {
            source: test_source(),
            communicator: communicator.iter().map(|s| s.to_string()).collect(),
            communicator_among_others: vec![],
            noun,
        })
        .expect("test verdicts are well-formed")
    }

    fn verdicts() -> LexiconVerdicts {
        verdicts_of(
            &["whoops"],
            vec![
                NounVerdict {
                    form: "doggy".into(),
                    lemma: "doggy".into(),
                    number: NounNumber::Singular,
                },
                NounVerdict {
                    form: "children".into(),
                    lemma: "child".into(),
                    number: NounNumber::Plural,
                },
            ],
        )
    }

    fn with_xpos(mut word: UdWord, xpos: &str) -> UdWord {
        word.xpos = Some(xpos.to_string());
        word
    }

    fn range_parent(start: usize, end: usize, text: &str) -> UdWord {
        let mut w = UdWord::synthetic(text, "", UniversalPos::X, None, 0, "");
        w.id = UdId::Range(start, end);
        w
    }

    /// Exactly what Stanza 1.14.0 (`combined_charlm`) returns for `whoops .`
    /// when handed the terminator: a plural noun with lemma `whoop`.
    fn stanza_whoops() -> UdSentence {
        UdSentence {
            words: vec![
                with_xpos(
                    word(
                        1,
                        "whoops",
                        "whoop",
                        UniversalPos::Noun,
                        Some("Number=Plur"),
                        0,
                        "root",
                    ),
                    "NNS",
                ),
                punct(2, ".", 1),
            ],
        }
    }

    fn assert_untouched(sentence: UdSentence, verdicts: &LexiconVerdicts) {
        assert_eq!(constrain_to_lexicon(sentence.clone(), verdicts), sentence);
    }

    /// The seam that matters: raw Stanza analysis in, `%mor` text out, through
    /// the production invariant dispatcher and the production mapping.
    #[test]
    fn whoops_with_a_period_is_a_communicator_in_mor() {
        // The terminator is not a `%mor` item; the mapping emits it separately.
        assert_eq!(
            mor_texts(&stanza_whoops(), UtteranceEvidence::none(0)),
            vec!["intj|whoops"]
        );
    }

    /// A one-word `banana !` comes back from Stanza as an interjection; the
    /// lexicon knows `banana` only as a noun and nothing derives it.
    #[test]
    fn banana_read_as_an_interjection_is_a_noun_in_mor() {
        let sentence = UdSentence {
            words: vec![
                with_xpos(
                    word(1, "banana", "banana", UniversalPos::Intj, None, 0, "root"),
                    "UH",
                ),
                punct(2, "!", 1),
            ],
        };
        assert_eq!(
            mor_texts(&sentence, UtteranceEvidence::none(0)),
            vec!["noun|banana"]
        );
    }

    /// `doggy ?` is read as ADJ by Stanza 1.14.0 and is a noun-only ENTRY,
    /// but MOR could derive it as `dog` + adjectival `-y`, so the embedded
    /// verdicts carry nothing for it and the analysis stands. Pinned so the
    /// limit of this mechanism is visible, not discovered again.
    #[test]
    fn doggy_is_derivable_and_therefore_left_to_stanza() {
        let sentence = UdSentence {
            words: vec![
                with_xpos(
                    word(
                        1,
                        "doggy",
                        "doggy",
                        UniversalPos::Adj,
                        Some("Degree=Pos"),
                        0,
                        "root",
                    ),
                    "JJ",
                ),
                punct(2, "?", 1),
            ],
        };
        assert_eq!(
            mor_texts(&sentence, UtteranceEvidence::none(0)),
            vec!["adj|doggy-S1"]
        );
    }

    #[test]
    fn communicator_verdict_rewrites_category_lemma_and_features() {
        let out = constrain_to_lexicon(stanza_whoops(), &verdicts());
        let w = &out.words[0];
        assert_eq!(w.upos, UdPunctable::Value(UniversalPos::Intj));
        assert_eq!(w.lemma, "whoops");
        assert_eq!(w.xpos.as_deref(), Some("UH"));
        assert_eq!(w.feats, None);
        // The parse is kept.
        assert_eq!((w.head, w.deprel.as_str()), (0, "root"));
    }

    /// `gonna` arrives as a range parent plus the components `gon` and `na`.
    /// `na` is a communicator in the lexicon; the particle must survive.
    #[test]
    fn mwt_components_are_not_constrained() {
        let sentence = UdSentence {
            words: vec![
                range_parent(1, 2, "gonna"),
                word(
                    1,
                    "gon",
                    "go",
                    UniversalPos::Verb,
                    Some("Tense=Pres|VerbForm=Part"),
                    0,
                    "root",
                ),
                word(2, "na", "to", UniversalPos::Part, None, 3, "mark"),
                word(
                    3,
                    "go",
                    "go",
                    UniversalPos::Verb,
                    Some("VerbForm=Inf"),
                    1,
                    "xcomp",
                ),
            ],
        };
        assert_untouched(sentence, &verdicts_of(&["na"], vec![]));
    }

    /// CHAT main tiers are lowercase, so a capitalized word is a proper name
    /// by transcriber convention, whatever the lexicon says about the
    /// lowercase form (`Gin` the child, not `gin` the drink).
    #[test]
    fn capitalized_words_are_proper_names_and_untouched() {
        assert!(is_transcriber_marked_name("Gin"));
        assert!(!is_transcriber_marked_name("gin"));
        let sentence = UdSentence {
            words: vec![word(
                1,
                "Doggy",
                "doggy",
                UniversalPos::Adj,
                None,
                0,
                "root",
            )],
        };
        assert_untouched(sentence, &verdicts());
    }

    #[test]
    fn plural_noun_verdict_carries_the_stem_and_number() {
        let sentence = UdSentence {
            words: vec![word(
                1,
                "children",
                "children",
                UniversalPos::Adj,
                None,
                0,
                "root",
            )],
        };
        let out = constrain_to_lexicon(sentence, &verdicts());
        let w = &out.words[0];
        assert_eq!(w.upos, UdPunctable::Value(UniversalPos::Noun));
        assert_eq!(w.lemma, "child");
        assert_eq!(w.feats.as_deref(), Some("Number=Plur"));
        assert_eq!(w.xpos.as_deref(), Some("NNS"));
    }

    #[test]
    fn words_without_a_verdict_are_untouched() {
        // `whoop` is ambiguous in the lexicon and has no verdict; `Miffy` is
        // unknown to it. Both must come back byte-identical.
        let sentence = UdSentence {
            words: vec![
                word(
                    1,
                    "whoop",
                    "whoop",
                    UniversalPos::Noun,
                    Some("Number=Sing"),
                    0,
                    "root",
                ),
                word(2, "Miffy", "Miffy", UniversalPos::Propn, None, 1, "nsubj"),
                punct(3, ".", 1),
            ],
        };
        assert_untouched(sentence, &verdicts());
    }

    #[test]
    fn a_proper_noun_reading_of_a_lexicon_noun_is_kept() {
        let sentence = UdSentence {
            words: vec![word(
                1,
                "doggy",
                "doggy",
                UniversalPos::Propn,
                None,
                0,
                "root",
            )],
        };
        assert_untouched(sentence, &verdicts());
    }

    #[test]
    fn agreeing_analyses_are_not_rewritten() {
        let sentence = UdSentence {
            words: vec![with_xpos(
                word(1, "whoops", "whoops", UniversalPos::Intj, None, 0, "root"),
                "UH",
            )],
        };
        assert_untouched(sentence, &verdicts());
    }

    #[test]
    fn range_parents_and_punctuation_are_skipped() {
        let sentence = UdSentence {
            words: vec![range_parent(1, 2, "whoops"), punct(3, "whoops", 1)],
        };
        assert_untouched(sentence, &verdicts());
    }
}
