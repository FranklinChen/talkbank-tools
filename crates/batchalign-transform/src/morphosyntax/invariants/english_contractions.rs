//! English CHAT contractions that Stanza's multi-word-token vocabulary lacks.
//!
//! Stanza's English MWT expander knows `gonna`, `wanna` and `gotta` from its
//! training data and returns them as a range parent plus components (`gon` +
//! `na`), which the mapping renders as `verb|go-Part-Pres-S~part|to`. CHAT
//! transcribers also write `hafta`, `hasta`, `hadta`, `oughta`, `useta`,
//! `sposta`, `gimme`, `lemme` and `dunno`, which the expander has never seen:
//! it leaves them whole and the tagger invents a category and a lemma
//! (`aux|hafta-Fin-Ind-Pres-S2`, or `verb|haft-Inf-S`). Hinting the expander
//! to split them would only produce a seq2seq guess.
//!
//! The expansion of each of these is fixed, so it is done here, at the
//! post-depparse stage, by synthesizing exactly the shape Stanza produces for
//! the expanded words: a `Range` parent followed by its components, every
//! later id and head renumbered, and the tree reshaped the way Stanza parses
//! `have to put`: the host is the clause head and the verb it governs is its
//! `xcomp`, with `to` marking that verb. When Stanza read the whole token as
//! a dependent of that verb (an auxiliary, an adverb, an interjection), the
//! host is raised into the verb's position; the verb's clause-level
//! dependents (subject, auxiliaries, negation, punctuation) move to the
//! host, its arguments stay, and an adjunct goes with whichever of the two
//! it precedes or follows, which is where Stanza puts it. The person and number Stanza read off the subject
//! are kept on the finite part; the table supplies what the token hid (the
//! lemma, the tense the form spells, and the particle or pronoun).
//!
//! `gonna`, `wanna` and `gotta` are in the table as well, so a Stanza build
//! that fails to expand them gets the same treatment; the rewrite never
//! touches a token Stanza already expanded.

use crate::morphosyntax::mor_word::parse_feats;
use crate::morphosyntax::{UdId, UdPunctable, UdSentence, UdWord, UniversalPos};

/// How one component of an expanded contraction attaches in the parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Attachment {
    /// The word that takes the token's place in the parse: its head and
    /// relation, or the governed verb's when the host is raised over it.
    Host,
    /// The infinitival `to`: `mark` on the verb the host governs.
    Marker,
    /// An object pronoun of the host (`gimme` → `me`): `obj`, or `iobj` when
    /// the host already has an object (`gimme that`).
    Object,
    /// A finite auxiliary of the host (`dunno` → `do`).
    Auxiliary,
    /// The negation particle of the host (`dunno` → `n't`).
    Negation,
}

/// What the host governs, which decides whether the token's tree is reshaped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Complement {
    /// A `to`-infinitive (`hafta go`): the verb becomes the host's `xcomp`
    /// and `to` marks it.
    Infinitive,
    /// A bare infinitive (`lemme see`): the verb becomes the host's `xcomp`.
    BareInfinitive,
    /// Nothing (`gimme`, `dunno`): the tree keeps the token's attachment.
    None,
}

/// One component of an expanded contraction.
#[derive(Debug, Clone, Copy)]
struct Part {
    text: &'static str,
    lemma: &'static str,
    upos: UniversalPos,
    /// UD features the form itself spells. The finite part (`VerbForm=Fin`)
    /// also carries the `Person` and `Number` Stanza read off the subject
    /// for the whole token.
    feats: &'static str,
    attach: Attachment,
}

const fn to() -> Part {
    Part {
        text: "to",
        lemma: "to",
        upos: UniversalPos::Part,
        feats: "",
        attach: Attachment::Marker,
    }
}

const fn me() -> Part {
    Part {
        text: "me",
        lemma: "I",
        upos: UniversalPos::Pron,
        feats: "Case=Acc|Number=Sing|Person=1|PronType=Prs",
        attach: Attachment::Object,
    }
}

const fn host(
    text: &'static str,
    lemma: &'static str,
    upos: UniversalPos,
    feats: &'static str,
) -> Part {
    Part {
        text,
        lemma,
        upos,
        feats,
        attach: Attachment::Host,
    }
}

const PRESENT: &str = "Mood=Ind|Tense=Pres|VerbForm=Fin";
const PRESENT_3SG: &str = "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin";
const PAST: &str = "Mood=Ind|Tense=Past|VerbForm=Fin";
const IMPERATIVE: &str = "Mood=Imp|VerbForm=Fin";

/// One contraction: its surface form (ASCII lowercase), its parts, and what
/// its host governs.
struct Contraction {
    surface: &'static str,
    parts: &'static [Part],
    complement: Complement,
}

const fn infinitive(surface: &'static str, parts: &'static [Part]) -> Contraction {
    Contraction {
        surface,
        parts,
        complement: Complement::Infinitive,
    }
}

/// The contractions and their expansions.
const CONTRACTIONS: &[Contraction] = &[
    infinitive(
        "hafta",
        &[host("have", "have", UniversalPos::Verb, PRESENT), to()],
    ),
    infinitive(
        "havta",
        &[host("have", "have", UniversalPos::Verb, PRESENT), to()],
    ),
    infinitive(
        "hasta",
        &[host("has", "have", UniversalPos::Verb, PRESENT_3SG), to()],
    ),
    infinitive(
        "hadta",
        &[host("had", "have", UniversalPos::Verb, PAST), to()],
    ),
    infinitive(
        "oughta",
        &[
            host("ought", "ought", UniversalPos::Aux, "VerbForm=Fin"),
            to(),
        ],
    ),
    infinitive(
        "useta",
        &[host("used", "use", UniversalPos::Verb, PAST), to()],
    ),
    infinitive(
        "sposta",
        &[
            host(
                "supposed",
                "suppose",
                UniversalPos::Verb,
                "Tense=Past|VerbForm=Part",
            ),
            to(),
        ],
    ),
    infinitive(
        "gonna",
        &[
            host(
                "going",
                "go",
                UniversalPos::Verb,
                "Tense=Pres|VerbForm=Part",
            ),
            to(),
        ],
    ),
    infinitive(
        "wanna",
        &[host("want", "want", UniversalPos::Verb, PRESENT), to()],
    ),
    infinitive(
        "gotta",
        &[host("got", "get", UniversalPos::Verb, PAST), to()],
    ),
    Contraction {
        surface: "gimme",
        parts: &[host("give", "give", UniversalPos::Verb, IMPERATIVE), me()],
        complement: Complement::None,
    },
    Contraction {
        surface: "lemme",
        parts: &[host("let", "let", UniversalPos::Verb, IMPERATIVE), me()],
        complement: Complement::BareInfinitive,
    },
    Contraction {
        surface: "dunno",
        parts: &[
            Part {
                text: "do",
                lemma: "do",
                upos: UniversalPos::Aux,
                feats: PRESENT,
                attach: Attachment::Auxiliary,
            },
            Part {
                text: "n't",
                lemma: "not",
                upos: UniversalPos::Part,
                feats: "Polarity=Neg",
                attach: Attachment::Negation,
            },
            host("know", "know", UniversalPos::Verb, "VerbForm=Inf"),
        ],
        complement: Complement::None,
    },
];

/// The table entry for a surface form, if any. The surfaces are ASCII, so
/// the comparison needs no allocation.
fn expansion_of(text: &str) -> Option<&'static Contraction> {
    CONTRACTIONS
        .iter()
        .find(|c| text.eq_ignore_ascii_case(c.surface))
}

/// Expand, in place, every whole token the table knows that Stanza left as a
/// single word. Tokens Stanza already expanded are ranges and are skipped.
pub fn expand_english_contractions(mut sentence: UdSentence) -> UdSentence {
    // Component membership is read once; an expansion inserts a new range
    // whose components are the just-built parts, which the loop steps over.
    let mut components: Vec<std::ops::RangeInclusive<usize>> =
        sentence.mwt_component_ranges().collect();
    let mut i = 0;
    while i < sentence.words.len() {
        let word = &sentence.words[i];
        let UdId::Single(id) = word.id else {
            i += 1;
            continue;
        };
        if components.iter().any(|r| r.contains(&id)) {
            i += 1;
            continue;
        }
        let Some(contraction) = expansion_of(&word.text) else {
            i += 1;
            continue;
        };
        let inserted = expand_at(&mut sentence, i, id, contraction);
        let shift = inserted - 2;
        for range in &mut components {
            if *range.start() > id {
                *range = range.start() + shift..=range.end() + shift;
            }
        }
        i += inserted;
    }
    sentence
}

/// Relations that belong to the clause, not to the verb's own phrase: when
/// the host is raised over the verb they follow the host wherever they
/// stand.
const CLAUSE_LEVEL: &[&str] = &[
    "nsubj",
    "nsubj:pass",
    "nsubj:outer",
    "csubj",
    "csubj:pass",
    "expl",
    "punct",
    "discourse",
    "vocative",
    "parataxis",
    "dislocated",
];

/// The verb's own arguments: they stay with the verb wherever they stand
/// (`what dyou hafta do ?` keeps `what` as the object of `do`).
const VERB_ARGUMENTS: &[&str] = &["obj", "iobj", "xcomp", "ccomp", "compound:prt"];

/// Whether a dependent of the governed verb follows the host when the host
/// is raised. Clause-level relations always do and the verb's arguments
/// never do; everything else (auxiliaries, copulas, negation, markers,
/// conjunctions, adjuncts) belongs to the predicate it precedes or follows,
/// which is where Stanza attaches it in the expanded sentence: `do` and
/// `n't` in `I don't hafta go` to the host, `be` in `hafta be good` and a
/// final `now` to the verb.
fn moves_to_host(word: &UdWord, precedes_host: bool) -> bool {
    if CLAUSE_LEVEL.contains(&word.deprel.as_str()) {
        return true;
    }
    if VERB_ARGUMENTS.contains(&word.deprel.as_str()) {
        return false;
    }
    precedes_host
}

/// Replace the single word at index `i` (UD id `id`) by a range parent and
/// the contraction's components; returns how many words now occupy that slot.
fn expand_at(sentence: &mut UdSentence, i: usize, id: usize, contraction: &Contraction) -> usize {
    let parts = contraction.parts;
    let shift = parts.len() - 1;
    // Every table entry has a host (`every_contraction_has_one_host` checks
    // the table); an entry without one is left unexpanded rather than
    // half-expanded.
    let Some(host_offset) = parts.iter().position(|p| p.attach == Attachment::Host) else {
        return 1;
    };
    let host_id = id + host_offset;
    let original = sentence.words[i].clone();
    let governed = governed_verb(sentence, &original, id, contraction.complement);
    let host_already_has_object = sentence
        .words
        .iter()
        .any(|w| w.head == id && w.deprel == "obj");

    renumber(sentence, id, shift, host_id);
    let shifted = |head: usize| if head > id { head + shift } else { head };
    let governed = governed.map(shifted);
    let attachment = match governed {
        // Raised when Stanza made the token a dependent of the very verb it
        // governs.
        Some(verb) if verb == shifted(original.head) => raise_host(sentence, verb, host_id),
        _ => None,
    }
    .unwrap_or_else(|| (shifted(original.head), original.deprel.clone()));

    let replacement = build_parts(
        parts,
        &original,
        id,
        host_id,
        attachment,
        governed,
        host_already_has_object,
    );
    let n = replacement.len();
    sentence.words.splice(i..=i, replacement);
    n
}

/// Move every id and head after the token down by `shift`; what depended on
/// the token now depends on the host.
fn renumber(sentence: &mut UdSentence, id: usize, shift: usize, host_id: usize) {
    for word in &mut sentence.words {
        word.id = match &word.id {
            UdId::Single(n) if *n > id => UdId::Single(n + shift),
            UdId::Range(s, e) if *s > id => UdId::Range(s + shift, e + shift),
            other => other.clone(),
        };
        if word.head == id {
            word.head = host_id;
        } else if word.head > id {
            word.head += shift;
        }
    }
}

/// Raise the host into the governed verb's place: the verb becomes the
/// host's `xcomp` and its clause-level dependents move to the host. Returns
/// the head and relation the host takes, or `None` when `verb` names no
/// word (malformed input: nothing is reshaped).
fn raise_host(sentence: &mut UdSentence, verb: usize, host_id: usize) -> Option<(usize, String)> {
    let verb_word = sentence
        .words
        .iter_mut()
        .find(|w| w.id == UdId::Single(verb))?;
    let taken = (verb_word.head, std::mem::take(&mut verb_word.deprel));
    verb_word.head = host_id;
    verb_word.deprel = "xcomp".to_string();
    for w in &mut sentence.words {
        let precedes_host = matches!(w.id, UdId::Single(n) if n < host_id);
        if w.head == verb && w.id != UdId::Single(verb) && moves_to_host(w, precedes_host) {
            w.head = host_id;
        }
    }
    Some(taken)
}

/// The range parent and the component words, ids from `id` upward.
fn build_parts(
    parts: &[Part],
    original: &UdWord,
    id: usize,
    host_id: usize,
    (host_head, host_deprel): (usize, String),
    governed: Option<usize>,
    host_already_has_object: bool,
) -> Vec<UdWord> {
    let shift = parts.len() - 1;
    let mut replacement = Vec::with_capacity(parts.len() + 1);
    replacement.push(UdWord {
        id: UdId::Range(id, id + shift),
        text: original.text.clone(),
        lemma: String::new(),
        upos: UdPunctable::Value(UniversalPos::X),
        xpos: None,
        feats: None,
        head: 0,
        deprel: String::new(),
        deps: None,
        misc: None,
    });
    for (offset, part) in parts.iter().enumerate() {
        let (head, deprel) = match part.attach {
            Attachment::Host => (host_head, host_deprel.clone()),
            Attachment::Marker => (governed.unwrap_or(host_id), "mark".to_string()),
            Attachment::Object => (
                host_id,
                if host_already_has_object {
                    "iobj"
                } else {
                    "obj"
                }
                .to_string(),
            ),
            Attachment::Auxiliary => (host_id, "aux".to_string()),
            Attachment::Negation => (host_id, "advmod".to_string()),
        };
        let feats = merge_agreement(part.feats, original.feats.as_deref());
        // The table's category, always: `gotta` renders as
        // `verb|get~part|to` from Stanza's own expansion, and `hafta` must
        // render the same way whether Stanza read the whole token as an
        // auxiliary, an adverb or an interjection.
        replacement.push(UdWord {
            id: UdId::Single(id + offset),
            text: part.text.to_string(),
            lemma: part.lemma.to_string(),
            upos: UdPunctable::Value(part.upos),
            xpos: None,
            feats: if feats.is_empty() { None } else { Some(feats) },
            head,
            deprel,
            deps: None,
            misc: None,
        });
    }
    replacement
}

/// The verb the host governs, by the token's original ids: the host's
/// `xcomp` child when Stanza already parsed the token as the clause head,
/// else the token's head when Stanza attached it to the verb. `None` when
/// the host governs nothing, or stands alone (`hafta .`).
fn governed_verb(
    sentence: &UdSentence,
    original: &UdWord,
    id: usize,
    complement: Complement,
) -> Option<usize> {
    match complement {
        Complement::None => return None,
        Complement::Infinitive | Complement::BareInfinitive => {}
    }
    let xcomp_child = sentence.words.iter().find_map(|w| match w.id {
        UdId::Single(n) if w.head == id && w.deprel == "xcomp" => Some(n),
        _ => None,
    });
    xcomp_child.or((original.head != 0).then_some(original.head))
}

/// The table's features for a part; for the finite part, plus the `Person`
/// and `Number` Stanza read off the subject for the whole token, when the
/// table does not fix them itself.
fn merge_agreement(table_feats: &str, original: Option<&str>) -> String {
    let mut feats = parse_feats(Some(table_feats));
    if feats.get("VerbForm").is_some_and(|v| v == "Fin") {
        for (k, v) in parse_feats(original) {
            if matches!(k.as_str(), "Person" | "Number") {
                feats.entry(k).or_insert(v);
            }
        }
    }
    let mut pairs: Vec<(String, String)> = feats.into_iter().collect();
    pairs.sort();
    pairs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("|")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphosyntax::evidence::UtteranceEvidence;
    use crate::morphosyntax::invariants::test_support::{english, mor_texts, punct, word};
    use crate::morphosyntax::{apply_grammatical_invariants, map_ud_sentence};
    use talkbank_model::WriteChat;

    /// Stanza 1.14.0 for `you hafta put that one in .`: `hafta` is a single
    /// AUX attached to `put`, with the subject's person and number.
    fn stanza_hafta() -> UdSentence {
        UdSentence {
            words: vec![
                word(
                    1,
                    "you",
                    "you",
                    UniversalPos::Pron,
                    Some("Case=Nom|Person=2|PronType=Prs"),
                    3,
                    "nsubj",
                ),
                word(
                    2,
                    "hafta",
                    "hafta",
                    UniversalPos::Aux,
                    Some("Mood=Ind|Number=Sing|Person=2|Tense=Pres|VerbForm=Fin"),
                    3,
                    "aux",
                ),
                word(
                    3,
                    "put",
                    "put",
                    UniversalPos::Verb,
                    Some("VerbForm=Inf"),
                    0,
                    "root",
                ),
                word(
                    4,
                    "that",
                    "that",
                    UniversalPos::Det,
                    Some("Number=Sing|PronType=Dem"),
                    5,
                    "det",
                ),
                word(
                    5,
                    "one",
                    "one",
                    UniversalPos::Noun,
                    Some("Number=Sing"),
                    3,
                    "obj",
                ),
                word(6, "in", "in", UniversalPos::Adp, None, 3, "compound:prt"),
                punct(7, ".", 3),
            ],
        }
    }

    /// The production seam: Stanza's whole-token analysis in, `%mor` out.
    #[test]
    fn hafta_expands_to_have_plus_to_in_mor() {
        let mors = mor_texts(&stanza_hafta(), UtteranceEvidence::none(7));
        assert_eq!(
            mors,
            vec![
                "pron|you-Prs-Nom-S2",
                "verb|have-Fin-Ind-Pres-S2~part|to",
                "verb|put-Inf-S",
                "det|that-Def-Dem-Sing",
                "noun|one-Acc",
                "adp|in",
            ]
        );
    }

    /// Stanza's shape for `you have to put that one in .`: `have` is the
    /// root, `put` its `xcomp`, `to` marks `put`; the subject and the
    /// terminator hang off `have`, the object and particle stay on `put`.
    #[test]
    fn expansion_raises_the_host_over_the_verb_it_governs() {
        let out = expand_english_contractions(stanza_hafta());
        let ids: Vec<UdId> = out.words.iter().map(|w| w.id.clone()).collect();
        assert_eq!(
            ids,
            vec![
                UdId::Single(1),
                UdId::Range(2, 3),
                UdId::Single(2),
                UdId::Single(3),
                UdId::Single(4),
                UdId::Single(5),
                UdId::Single(6),
                UdId::Single(7),
                UdId::Single(8),
            ]
        );
        let have = &out.words[2];
        assert_eq!((have.text.as_str(), have.lemma.as_str()), ("have", "have"));
        assert_eq!(have.upos, UdPunctable::Value(UniversalPos::Verb));
        assert_eq!((have.head, have.deprel.as_str()), (0, "root"));
        assert_eq!(
            have.feats.as_deref(),
            Some("Mood=Ind|Number=Sing|Person=2|Tense=Pres|VerbForm=Fin")
        );
        let to = &out.words[3];
        assert_eq!(
            (to.text.as_str(), to.head, to.deprel.as_str()),
            ("to", 4, "mark")
        );
        let heads: Vec<(&str, usize, &str)> = out.words[4..]
            .iter()
            .map(|w| (w.text.as_str(), w.head, w.deprel.as_str()))
            .collect();
        assert_eq!(
            heads,
            vec![
                ("put", 2, "xcomp"),
                ("that", 6, "det"),
                ("one", 4, "obj"),
                ("in", 4, "compound:prt"),
                (".", 2, "punct"),
            ]
        );
        // The subject moved to the clause head.
        assert_eq!(
            (out.words[0].head, out.words[0].deprel.as_str()),
            (2, "nsubj")
        );
    }

    /// The `%gra` the mapping builds from the raised tree is consistent with
    /// the `%mor`: `have` is a verb and the root.
    #[test]
    fn hafta_gra_has_the_host_as_root() {
        let ctx = english();
        let rewritten =
            apply_grammatical_invariants(&stanza_hafta(), &ctx, || UtteranceEvidence::none(7));
        let (_mors, gras) = map_ud_sentence(&rewritten, &ctx).expect("mapping succeeds");
        let rendered: Vec<String> = gras
            .iter()
            .map(|g| {
                let mut out = String::new();
                g.write_chat(&mut out).expect("gra renders");
                out
            })
            .collect();
        assert_eq!(
            rendered,
            vec![
                "1|2|NSUBJ",
                "2|0|ROOT",
                "3|4|MARK",
                "4|2|XCOMP",
                "5|6|DET",
                "6|4|OBJ",
                "7|4|COMPOUND-PRT",
                "8|2|PUNCT",
            ]
        );
    }

    #[test]
    fn a_misread_host_takes_the_table_category_and_the_form_spells_its_tense() {
        // Stanza read `hadta` as an interjection with no features.
        let sentence = UdSentence {
            words: vec![
                word(
                    1,
                    "she",
                    "she",
                    UniversalPos::Pron,
                    Some("Case=Nom|Number=Sing|Person=3|PronType=Prs"),
                    3,
                    "nsubj",
                ),
                word(2, "hadta", "hadta", UniversalPos::Intj, None, 3, "aux"),
                word(
                    3,
                    "go",
                    "go",
                    UniversalPos::Verb,
                    Some("VerbForm=Inf"),
                    0,
                    "root",
                ),
            ],
        };
        let out = expand_english_contractions(sentence);
        let had = &out.words[2];
        assert_eq!(had.upos, UdPunctable::Value(UniversalPos::Verb));
        assert_eq!(had.lemma, "have");
        assert_eq!(
            had.feats.as_deref(),
            Some("Mood=Ind|Tense=Past|VerbForm=Fin")
        );
    }

    #[test]
    fn a_host_read_as_a_modifier_is_raised_over_the_verb() {
        // `gonna call him ?` with Stanza's `gonna` left whole as an adverb.
        let sentence = UdSentence {
            words: vec![
                word(1, "gonna", "gonna", UniversalPos::Adv, None, 2, "advmod"),
                word(
                    2,
                    "call",
                    "call",
                    UniversalPos::Verb,
                    Some("VerbForm=Inf"),
                    0,
                    "root",
                ),
                word(
                    3,
                    "him",
                    "he",
                    UniversalPos::Pron,
                    Some("Case=Acc|Person=3|PronType=Prs"),
                    2,
                    "obj",
                ),
            ],
        };
        let out = expand_english_contractions(sentence);
        assert_eq!(out.words[2].text, "to");
        // `going` takes the root, `call` becomes its xcomp, `to` marks `call`.
        assert_eq!(
            (out.words[2].head, out.words[2].deprel.as_str()),
            (3, "mark")
        );
        assert_eq!(
            (out.words[1].head, out.words[1].deprel.as_str()),
            (0, "root")
        );
        assert_eq!(
            (out.words[3].head, out.words[3].deprel.as_str()),
            (1, "xcomp")
        );
        assert_eq!(
            (out.words[4].head, out.words[4].deprel.as_str()),
            (3, "obj")
        );
        assert_eq!(out.words[1].upos, UdPunctable::Value(UniversalPos::Verb));
        assert_eq!(out.words[1].lemma, "go");
    }

    /// A host Stanza already made the clause head keeps its place; `to`
    /// marks its existing xcomp child.
    #[test]
    fn a_host_that_is_already_the_head_keeps_its_xcomp_child() {
        let sentence = UdSentence {
            words: vec![
                word(1, "I", "I", UniversalPos::Pron, None, 2, "nsubj"),
                word(
                    2,
                    "hafta",
                    "hafta",
                    UniversalPos::Verb,
                    Some(PRESENT),
                    0,
                    "root",
                ),
                word(
                    3,
                    "go",
                    "go",
                    UniversalPos::Verb,
                    Some("VerbForm=Inf"),
                    2,
                    "xcomp",
                ),
            ],
        };
        let out = expand_english_contractions(sentence);
        assert_eq!(
            (out.words[2].head, out.words[2].deprel.as_str()),
            (0, "root")
        );
        assert_eq!(
            (out.words[3].head, out.words[3].deprel.as_str()),
            (4, "mark")
        );
        assert_eq!(
            (out.words[4].head, out.words[4].deprel.as_str()),
            (2, "xcomp")
        );
    }

    /// `gimme that`: `me` is the indirect object when the host already has
    /// an object, as Stanza parses `give me that`.
    #[test]
    fn gimme_with_an_object_makes_me_the_indirect_object() {
        let sentence = UdSentence {
            words: vec![
                word(
                    1,
                    "gimme",
                    "gimme",
                    UniversalPos::Verb,
                    Some(IMPERATIVE),
                    0,
                    "root",
                ),
                word(
                    2,
                    "that",
                    "that",
                    UniversalPos::Pron,
                    Some("Number=Sing|PronType=Dem"),
                    1,
                    "obj",
                ),
                punct(3, ".", 1),
            ],
        };
        let out = expand_english_contractions(sentence);
        assert_eq!(
            (
                out.words[2].text.as_str(),
                out.words[2].head,
                out.words[2].deprel.as_str()
            ),
            ("me", 1, "iobj")
        );
        assert_eq!(
            (out.words[3].head, out.words[3].deprel.as_str()),
            (1, "obj")
        );
    }

    /// `lemme see` with Stanza reading `lemme` as an interjection on `see`:
    /// `let` is raised, `see` is its bare-infinitive complement.
    #[test]
    fn lemme_is_raised_over_the_bare_infinitive() {
        let sentence = UdSentence {
            words: vec![
                word(
                    1,
                    "lemme",
                    "lemme",
                    UniversalPos::Intj,
                    None,
                    2,
                    "discourse",
                ),
                word(
                    2,
                    "see",
                    "see",
                    UniversalPos::Verb,
                    Some(IMPERATIVE),
                    0,
                    "root",
                ),
                punct(3, ".", 2),
            ],
        };
        let out = expand_english_contractions(sentence);
        let shape: Vec<(&str, usize, &str)> = out.words[1..]
            .iter()
            .map(|w| (w.text.as_str(), w.head, w.deprel.as_str()))
            .collect();
        assert_eq!(
            shape,
            vec![
                ("let", 0, "root"),
                ("me", 1, "obj"),
                ("see", 1, "xcomp"),
                (".", 1, "punct")
            ]
        );
    }

    #[test]
    fn three_part_dunno() {
        let sentence = UdSentence {
            words: vec![
                word(1, "I", "I", UniversalPos::Pron, None, 2, "nsubj"),
                word(
                    2,
                    "dunno",
                    "dunno",
                    UniversalPos::Verb,
                    Some("Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin"),
                    0,
                    "root",
                ),
                punct(3, ".", 2),
            ],
        };
        let out = expand_english_contractions(sentence);
        let texts: Vec<&str> = out.words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, vec!["I", "dunno", "do", "n't", "know", "."]);
        assert_eq!(out.words[1].id, UdId::Range(2, 4));
        // Stanza's shape for `I don't know .`: `know` is the root, `do` and
        // `n't` depend on it, and so do the subject and the terminator.
        let shape: Vec<(&str, usize, &str)> = out
            .words
            .iter()
            .filter(|w| matches!(w.id, UdId::Single(_)))
            .map(|w| (w.text.as_str(), w.head, w.deprel.as_str()))
            .collect();
        assert_eq!(
            shape,
            vec![
                ("I", 4, "nsubj"),
                ("do", 4, "aux"),
                ("n't", 4, "advmod"),
                ("know", 0, "root"),
                (".", 4, "punct"),
            ]
        );
        assert_eq!(
            out.words[2].feats.as_deref(),
            Some("Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin")
        );
        assert_eq!(out.words[5].id, UdId::Single(5));
    }

    #[test]
    fn tokens_stanza_already_expanded_are_untouched() {
        let mut parent = UdWord::synthetic("gonna", "", UniversalPos::X, None, 0, "");
        parent.id = UdId::Range(1, 2);
        let sentence = UdSentence {
            words: vec![
                parent,
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
        assert_eq!(expand_english_contractions(sentence.clone()), sentence);
    }

    /// `I don't hafta go there .` with Stanza's `hafta` as an adverb of
    /// `go`: the subject follows the host; the auxiliary and negation
    /// precede the host and follow it too; `there` follows the verb and
    /// stays with it.
    #[test]
    fn raising_moves_clause_level_dependents_and_leaves_the_verbs_adjunct() {
        let sentence = UdSentence {
            words: vec![
                word(
                    1,
                    "I",
                    "I",
                    UniversalPos::Pron,
                    Some("Case=Nom|Number=Sing|Person=1|PronType=Prs"),
                    5,
                    "nsubj",
                ),
                word(
                    2,
                    "do",
                    "do",
                    UniversalPos::Aux,
                    Some("Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin"),
                    5,
                    "aux",
                ),
                word(
                    3,
                    "n't",
                    "not",
                    UniversalPos::Part,
                    Some("Polarity=Neg"),
                    5,
                    "advmod",
                ),
                word(4, "hafta", "hafta", UniversalPos::Adv, None, 5, "advmod"),
                word(
                    5,
                    "go",
                    "go",
                    UniversalPos::Verb,
                    Some("VerbForm=Inf"),
                    0,
                    "root",
                ),
                word(
                    6,
                    "there",
                    "there",
                    UniversalPos::Adv,
                    Some("PronType=Dem"),
                    5,
                    "advmod",
                ),
                punct(7, ".", 5),
            ],
        };
        let out = expand_english_contractions(sentence);
        let shape: Vec<(&str, usize, &str)> = out
            .words
            .iter()
            .filter(|w| matches!(w.id, UdId::Single(_)))
            .map(|w| (w.text.as_str(), w.head, w.deprel.as_str()))
            .collect();
        assert_eq!(
            shape,
            vec![
                ("I", 4, "nsubj"),
                ("do", 4, "aux"),
                ("n't", 4, "advmod"),
                ("have", 0, "root"),
                ("to", 6, "mark"),
                ("go", 4, "xcomp"),
                ("there", 6, "advmod"),
                (".", 4, "punct"),
            ]
        );
    }

    /// `what dyou hafta do ?` keeps `what` as the object of `do`; a fronted
    /// `now` goes to the clause head, as Stanza attaches them in the
    /// expanded sentences.
    #[test]
    fn raising_keeps_the_verbs_arguments_and_moves_fronted_adjuncts() {
        let sentence = UdSentence {
            words: vec![
                word(1, "now", "now", UniversalPos::Adv, None, 5, "advmod"),
                word(
                    2,
                    "what",
                    "what",
                    UniversalPos::Pron,
                    Some("PronType=Int"),
                    5,
                    "obj",
                ),
                word(3, "you", "you", UniversalPos::Pron, None, 5, "nsubj"),
                word(
                    4,
                    "hafta",
                    "hafta",
                    UniversalPos::Aux,
                    Some(PRESENT),
                    5,
                    "aux",
                ),
                word(
                    5,
                    "do",
                    "do",
                    UniversalPos::Verb,
                    Some("VerbForm=Inf"),
                    0,
                    "root",
                ),
                punct(6, "?", 5),
            ],
        };
        let out = expand_english_contractions(sentence);
        let shape: Vec<(&str, usize, &str)> = out
            .words
            .iter()
            .filter(|w| matches!(w.id, UdId::Single(_)))
            .map(|w| (w.text.as_str(), w.head, w.deprel.as_str()))
            .collect();
        assert_eq!(
            shape,
            vec![
                ("now", 4, "advmod"),
                ("what", 6, "obj"),
                ("you", 4, "nsubj"),
                ("have", 0, "root"),
                ("to", 6, "mark"),
                ("do", 4, "xcomp"),
                ("?", 4, "punct"),
            ]
        );
    }

    /// `you hafta be good .`: the copula belongs to the predicate `good`,
    /// which is what the host governs.
    #[test]
    fn raising_leaves_a_copula_with_its_predicate() {
        let sentence = UdSentence {
            words: vec![
                word(1, "you", "you", UniversalPos::Pron, None, 4, "nsubj"),
                word(
                    2,
                    "hafta",
                    "hafta",
                    UniversalPos::Aux,
                    Some(PRESENT),
                    4,
                    "aux",
                ),
                word(
                    3,
                    "be",
                    "be",
                    UniversalPos::Aux,
                    Some("VerbForm=Inf"),
                    4,
                    "cop",
                ),
                word(
                    4,
                    "good",
                    "good",
                    UniversalPos::Adj,
                    Some("Degree=Pos"),
                    0,
                    "root",
                ),
                punct(5, ".", 4),
            ],
        };
        let out = expand_english_contractions(sentence);
        let shape: Vec<(&str, usize, &str)> = out
            .words
            .iter()
            .filter(|w| matches!(w.id, UdId::Single(_)))
            .map(|w| (w.text.as_str(), w.head, w.deprel.as_str()))
            .collect();
        assert_eq!(
            shape,
            vec![
                ("you", 2, "nsubj"),
                ("have", 0, "root"),
                ("to", 5, "mark"),
                ("be", 5, "cop"),
                ("good", 2, "xcomp"),
                (".", 2, "punct"),
            ]
        );
    }

    #[test]
    fn every_contraction_has_one_host() {
        for contraction in CONTRACTIONS {
            let hosts = contraction
                .parts
                .iter()
                .filter(|p| p.attach == Attachment::Host)
                .count();
            assert_eq!(hosts, 1, "{}", contraction.surface);
            let markers = contraction
                .parts
                .iter()
                .filter(|p| p.attach == Attachment::Marker)
                .count();
            assert_eq!(
                markers,
                usize::from(contraction.complement == Complement::Infinitive),
                "{}",
                contraction.surface
            );
        }
    }

    #[test]
    fn unknown_words_are_untouched() {
        let sentence = UdSentence {
            words: vec![word(
                1,
                "hat",
                "hat",
                UniversalPos::Noun,
                Some("Number=Sing"),
                0,
                "root",
            )],
        };
        assert_eq!(expand_english_contractions(sentence.clone()), sentence);
    }
}
