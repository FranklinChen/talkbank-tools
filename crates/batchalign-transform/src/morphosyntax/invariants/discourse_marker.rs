//! Tag questions and stand-alone responses: `... (.) okay ?`, `oh (.) okay .`
//!
//! A word the lexicon licenses as a communicator among other categories
//! (`okay` is an adjective too) is genuinely ambiguous to MOR, and Stanza,
//! shown the terminator, resolves it as the adjective in a tag question:
//! `put the lady on the chair (.) okay ?` came out as `adj|okay-S1`. What
//! settles it is not in the words Stanza saw but in the transcript: the
//! transcriber set the word off with a pause, and a word set off on both
//! sides (pause, separator or utterance edge) is a discourse element.
//!
//! Which readings the evidence can overturn is decided by whether the
//! reading can head an utterance on its own. An imperative can (`look !`,
//! `see ?`, `wait .` are complete clauses, and UD tags them VERB), so a verb
//! or auxiliary reading stands. An isolated adjective, adverb, determiner,
//! noun or pronoun cannot be a whole utterance without an elided predicate,
//! so when the lexicon has the communicator reading, that is the one:
//! `okay ?`, `right .`, `no (.) ...`, `boom !`, `oh (.) my .`. Measured on
//! the 100 Bates files against the isolation evidence, and against 716
//! files of CLAN MOR+POST output for the convention on `see` (verb, 124 of
//! 124) versus `okay` (communicator, 731 of 768).
//!
//! Pure communicators (`whoops`) are handled by the lexicon rule without
//! needing evidence.

use crate::morphosyntax::evidence::UtteranceEvidence;
use crate::morphosyntax::invariants::lexicon_category::is_transcriber_marked_name;
use crate::morphosyntax::lexicon::{LexiconVerdict, LexiconVerdicts};
use crate::morphosyntax::sentence_mapping::is_terminator_punct;
use crate::morphosyntax::{UdId, UdPunctable, UdSentence, UniversalPos};

/// Whether an isolated word with this Stanza category takes the lexicon's
/// communicator reading. Exhaustive so a new category is a decision, not a
/// fall-through.
fn evidence_overturns(upos: UniversalPos) -> bool {
    match upos {
        // A predicate or a name stands on its own.
        UniversalPos::Verb
        | UniversalPos::Aux
        | UniversalPos::Propn
        | UniversalPos::Intj
        | UniversalPos::Punct
        | UniversalPos::Num
        | UniversalPos::Sym => false,
        // A response token, a tag, a vocative, an onomatopoeia.
        UniversalPos::Adj
        | UniversalPos::Adv
        | UniversalPos::Det
        | UniversalPos::Noun
        | UniversalPos::Pron
        | UniversalPos::Adp
        | UniversalPos::Part
        | UniversalPos::Cconj
        | UniversalPos::Sconj
        | UniversalPos::X => true,
    }
}

/// Rewrite, in place, every isolated word that the lexicon licenses as a
/// communicator and Stanza read as a category that cannot stand alone.
pub fn mark_isolated_communicators(
    mut sentence: UdSentence,
    verdicts: &LexiconVerdicts,
    evidence: &UtteranceEvidence,
) -> UdSentence {
    let mwt_components: Vec<_> = sentence.mwt_component_ranges().collect();
    // Payload words correspond, in order, to what the mapper emits one MOR
    // item for: a range parent (standing for its components) or a single
    // token that is not the terminator. Separators the payload kept are
    // punctuation tokens and count; the terminator comes last and does not.
    let mut payload_index = 0usize;
    for word in &mut sentence.words {
        let is_top_level = match word.id {
            UdId::Single(id) => !mwt_components.iter().any(|r| r.contains(&id)),
            UdId::Range(_, _) => true,
            UdId::Decimal(_) => false,
        };
        if !is_top_level || is_terminator_punct(word) {
            continue;
        }
        let index = payload_index;
        payload_index += 1;
        let UdPunctable::Value(upos) = word.upos else {
            continue;
        };
        if matches!(word.id, UdId::Range(_, _)) || !evidence_overturns(upos) {
            continue;
        }
        if !evidence.isolation(index).is_some_and(|i| i.is_isolated()) {
            continue;
        }
        if !matches!(
            verdicts.verdict(&word.text),
            Some(LexiconVerdict::Communicator | LexiconVerdict::CommunicatorAmongOthers)
        ) {
            continue;
        }
        if is_transcriber_marked_name(&word.text) {
            continue;
        }
        tracing::debug!(text = %word.text, from = ?upos, "discourse marker: isolated communicator");
        word.upos = UdPunctable::Value(UniversalPos::Intj);
        word.lemma = word.text.to_lowercase();
        word.xpos = Some("UH".to_string());
        word.feats = None;
    }
    sentence
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphosyntax::UdWord;
    use crate::morphosyntax::evidence::WordIsolation;
    use crate::morphosyntax::invariants::test_support::{mor_texts, punct, word};
    use crate::morphosyntax::lexicon::{RawVerdicts, VerdictSource};

    fn verdicts() -> LexiconVerdicts {
        LexiconVerdicts::try_from(RawVerdicts {
            source: VerdictSource {
                repo: "test".into(),
                commit: "test".into(),
                path: "test".into(),
                generator: "test".into(),
            },
            communicator: vec![],
            communicator_among_others: vec!["okay".into(), "see".into(), "right".into()],
            noun: vec![],
        })
        .expect("test verdicts are well-formed")
    }

    /// Stanza 1.14.0 for `put the lady on the chair (.) okay ?`: `okay` is an
    /// adjective modifying `put`.
    fn stanza_tag_okay() -> UdSentence {
        UdSentence {
            words: vec![
                word(
                    1,
                    "put",
                    "put",
                    UniversalPos::Verb,
                    Some("Mood=Imp|VerbForm=Fin"),
                    0,
                    "root",
                ),
                word(
                    2,
                    "the",
                    "the",
                    UniversalPos::Det,
                    Some("Definite=Def|PronType=Art"),
                    3,
                    "det",
                ),
                word(
                    3,
                    "lady",
                    "lady",
                    UniversalPos::Noun,
                    Some("Number=Sing"),
                    1,
                    "obj",
                ),
                word(4, "on", "on", UniversalPos::Adp, None, 6, "case"),
                word(
                    5,
                    "the",
                    "the",
                    UniversalPos::Det,
                    Some("Definite=Def|PronType=Art"),
                    6,
                    "det",
                ),
                word(
                    6,
                    "chair",
                    "chair",
                    UniversalPos::Noun,
                    Some("Number=Sing"),
                    1,
                    "obl",
                ),
                word(
                    7,
                    "okay",
                    "okay",
                    UniversalPos::Adj,
                    Some("Degree=Pos"),
                    1,
                    "advmod",
                ),
                punct(8, "?", 1),
            ],
        }
    }

    fn evidence(isolated_last: bool, n: usize) -> UtteranceEvidence {
        let mut e = UtteranceEvidence::none(n);
        if isolated_last {
            e = UtteranceEvidence::from_isolation(
                (0..n)
                    .map(|i| WordIsolation {
                        break_before: i + 1 == n,
                        break_after: i + 1 == n || i + 2 == n,
                    })
                    .collect(),
            );
        }
        e
    }

    /// The production seam with the embedded lexicon: `okay` is a
    /// communicator among others there, and the pause makes it a tag.
    #[test]
    fn an_isolated_okay_tag_is_an_interjection_in_mor() {
        let mors = mor_texts(&stanza_tag_okay(), evidence(true, 7));
        assert_eq!(mors.last().map(String::as_str), Some("intj|okay"));
    }

    #[test]
    fn without_the_pause_the_adjective_reading_stands() {
        let mors = mor_texts(&stanza_tag_okay(), evidence(false, 7));
        assert_eq!(mors.last().map(String::as_str), Some("adj|okay-S1"));
    }

    #[test]
    fn the_rewrite_keeps_the_parse_and_changes_only_the_word() {
        let out = mark_isolated_communicators(stanza_tag_okay(), &verdicts(), &evidence(true, 7));
        let okay = &out.words[6];
        assert_eq!(okay.upos, UdPunctable::Value(UniversalPos::Intj));
        assert_eq!(okay.lemma, "okay");
        assert_eq!(okay.feats, None);
        assert_eq!((okay.head, okay.deprel.as_str()), (1, "advmod"));
        assert_eq!(out.words[0], stanza_tag_okay().words[0]);
    }

    /// `see ?` is a complete imperative clause: the verb reading stands even
    /// though the lexicon has `co|see`.
    #[test]
    fn an_isolated_imperative_keeps_its_verb_reading() {
        let sentence = UdSentence {
            words: vec![
                word(
                    1,
                    "see",
                    "see",
                    UniversalPos::Verb,
                    Some("Mood=Imp|VerbForm=Fin"),
                    0,
                    "root",
                ),
                punct(2, "?", 1),
            ],
        };
        let out = mark_isolated_communicators(sentence.clone(), &verdicts(), &evidence(true, 1));
        assert_eq!(out, sentence);
    }

    /// A comma the payload kept is a token before the tag; the payload index
    /// must count it or the evidence lands on the wrong word.
    #[test]
    fn a_kept_separator_counts_as_a_payload_word() {
        // `that's the cat , right ?` with Stanza's `'s` expanded.
        let mut parent = UdWord::synthetic("that's", "", UniversalPos::X, None, 0, "");
        parent.id = UdId::Range(1, 2);
        let sentence = UdSentence {
            words: vec![
                parent,
                word(
                    1,
                    "that",
                    "that",
                    UniversalPos::Pron,
                    Some("Number=Sing|PronType=Dem"),
                    4,
                    "nsubj",
                ),
                word(
                    2,
                    "'s",
                    "be",
                    UniversalPos::Aux,
                    Some("Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin"),
                    4,
                    "cop",
                ),
                word(
                    3,
                    "the",
                    "the",
                    UniversalPos::Det,
                    Some("Definite=Def|PronType=Art"),
                    4,
                    "det",
                ),
                word(
                    4,
                    "cat",
                    "cat",
                    UniversalPos::Noun,
                    Some("Number=Sing"),
                    0,
                    "root",
                ),
                punct(5, ",", 4),
                word(
                    6,
                    "right",
                    "right",
                    UniversalPos::Adj,
                    Some("Degree=Pos"),
                    4,
                    "advmod",
                ),
                punct(7, "?", 4),
            ],
        };
        // Payload: that's, the, cat, ",", right.
        let evidence = UtteranceEvidence::from_isolation(vec![
            WordIsolation {
                break_before: true,
                break_after: false,
            },
            WordIsolation::default(),
            WordIsolation {
                break_before: false,
                break_after: true,
            },
            WordIsolation {
                break_before: true,
                break_after: true,
            },
            WordIsolation {
                break_before: true,
                break_after: true,
            },
        ]);
        let out = mark_isolated_communicators(sentence, &verdicts(), &evidence);
        assert_eq!(out.words[6].upos, UdPunctable::Value(UniversalPos::Intj));
        assert_eq!(out.words[4].upos, UdPunctable::Value(UniversalPos::Noun));
    }

    #[test]
    fn a_word_the_lexicon_never_licenses_as_a_communicator_is_untouched() {
        let sentence = UdSentence {
            words: vec![
                word(
                    1,
                    "chair",
                    "chair",
                    UniversalPos::Noun,
                    Some("Number=Sing"),
                    0,
                    "root",
                ),
                punct(2, ".", 1),
            ],
        };
        let out = mark_isolated_communicators(sentence.clone(), &verdicts(), &evidence(true, 1));
        assert_eq!(out, sentence);
    }

    #[test]
    fn a_word_stanza_already_reads_as_an_interjection_is_untouched() {
        let sentence = UdSentence {
            words: vec![
                word(1, "okay", "okay", UniversalPos::Intj, None, 0, "root"),
                punct(2, ".", 1),
            ],
        };
        let out = mark_isolated_communicators(sentence.clone(), &verdicts(), &evidence(true, 1));
        assert_eq!(out, sentence);
    }
}
