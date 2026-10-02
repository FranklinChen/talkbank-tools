//! End-to-end L2 tests on the UD analyses BA3's worker actually returns.
//!
//! Each fixture is the primary and secondary UD output of BA3's
//! morphosyntax worker entry point (`batch_infer_morphosyntax`, the call the
//! Rust side makes) on Stanza 1.15.0, transcribed row for row in a
//! CoNLL-U-like layout (`ID FORM LEMMA UPOS FEATS HEAD DEPREL`, `_` for an
//! absent value, `3-4` for a multi-word token range). Two worker modes
//! matter, and each fixture says which one it reproduces:
//!
//! - **Primary** utterances are realigned to the CHAT words, so the
//!   terminator the worker appended for Stanza is taken back off: no
//!   punctuation row.
//! - **Secondary** spans are dispatched with Stanza owning tokenization,
//!   so the terminator survives as a final `PUNCT` row. That row is why a
//!   secondary sentence has more UD words than the span has CHAT words.
//!
//! The tests drive the production order with no model loaded: collect
//! payloads, inject the primary result (which defers the `@s` positions
//! from the same analysis), plan the secondary spans, merge each span's
//! secondary analysis, splice, and render the `%mor` and `%gra` lines a
//! reader sees.

use talkbank_model::WriteChat;
use talkbank_model::model::{LanguageCode, Line};

use super::{merge_planned_secondary_span, plan_dispatch_spans, splice_l2_into_chat};
use crate::morphosyntax::{
    MultilingualPolicy, TokenizationMode, UdId, UdPunctable, UdResponse, UdSentence, UdWord,
    UdWordAnalysis, UniversalPos, collect_payloads, inject_results,
};

/// Parse fixture rows (`ID FORM LEMMA UPOS FEATS HEAD DEPREL`) into a
/// sentence. Rows are whitespace-separated; `_` marks an absent value.
pub(crate) fn ud_sentence(rows: &str) -> UdSentence {
    let words = rows
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            let [id, text, lemma, upos, feats, head, deprel] = fields[..] else {
                panic!("fixture row must have seven fields: {line:?}");
            };
            let id = match id.split_once('-') {
                Some((start, end)) => UdId::Range(
                    start.parse().expect("range start"),
                    end.parse().expect("range end"),
                ),
                None => UdId::Single(id.parse().expect("word id")),
            };
            let upos: UniversalPos =
                serde_json::from_value(serde_json::Value::String(upos.to_string()))
                    .expect("fixture UPOS must be a UD tag");
            let absent = |value: &str| (value != "_").then(|| value.to_string());
            UdWord::from(UdWordAnalysis {
                id,
                text: text.to_string(),
                lemma: absent(lemma).unwrap_or_default(),
                upos: UdPunctable::Value(upos),
                xpos: None,
                feats: absent(feats),
                head: head.parse().expect("head"),
                deprel: deprel.to_string(),
                deps: None,
                misc: None,
            })
        })
        .collect();
    UdSentence { words }
}

/// The `%mor` and `%gra` lines of the one utterance, after the L2 splice.
#[derive(Debug)]
struct L2Lines {
    mor: String,
    gra: String,
}

/// The `dependent|head` pairs of a `%gra` line, labels dropped.
fn heads(gra: &str) -> String {
    gra.trim_start_matches("%gra:\t")
        .split(' ')
        .map(|relation| relation.rsplit_once('|').map_or(relation, |(pair, _)| pair))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Run one utterance through the whole L2 path.
///
/// `languages` is the `@Languages` value (primary first); `primary` is the
/// primary analysis; `secondaries` is one analysis per planned span, in
/// transcript order.
fn run_l2(languages: &str, main_tier: &str, primary: &str, secondaries: &[&str]) -> L2Lines {
    run_l2_in(
        TokenizationMode::Preserve,
        languages,
        main_tier,
        primary,
        secondaries,
    )
}

/// [`run_l2`] in a chosen tokenization mode.
fn run_l2_in(
    mode: TokenizationMode,
    languages: &str,
    main_tier: &str,
    primary: &str,
    secondaries: &[&str],
) -> L2Lines {
    let codes: Vec<LanguageCode> = languages
        .split(',')
        .map(|code| LanguageCode::new(code.trim()).expect("fixture language code"))
        .collect();
    let primary_lang = codes[0].clone();
    let chat_text = format!(
        "@UTF8\n\
         @Begin\n\
         @Languages:\t{languages}\n\
         @Participants:\tPAR Participant\n\
         @ID:\t{primary_lang}|test|PAR|||||Participant|||\n\
         *PAR:\t{main_tier}\n\
         @End\n"
    );
    let parser = talkbank_parser::TreeSitterParser::new().expect("parser");
    let mut chat = parser.parse_chat_file(&chat_text).expect_built();

    let payloads = collect_payloads(&chat, &primary_lang, &codes, MultilingualPolicy::ProcessAll);
    let responses = vec![UdResponse {
        sentences: vec![ud_sentence(primary)],
    }];
    let injection = inject_results(
        &parser,
        &mut chat,
        payloads.batch_items,
        responses,
        &primary_lang,
        mode,
        &std::collections::BTreeMap::new(),
    )
    .expect("primary injection");
    assert!(
        injection.decisions.is_empty(),
        "the primary analysis must inject: {:?}",
        injection.decisions
    );
    assert!(
        injection.l2.unaligned().is_empty(),
        "the primary analysis must align: {:?}",
        injection.l2.unaligned()
    );
    let deferred = injection.l2.into_reported_positions();

    let plan = plan_dispatch_spans(deferred);
    assert_eq!(
        plan.spans.len(),
        secondaries.len(),
        "one secondary analysis per planned span"
    );
    let merged = plan
        .spans
        .into_iter()
        .zip(secondaries)
        .map(|(span, secondary)| {
            merge_planned_secondary_span(span, &ud_sentence(secondary)).expect("secondary merge")
        })
        .collect();
    let outcome = splice_l2_into_chat(&mut chat, merged);
    assert_eq!(outcome.fallback, 0, "every span splices: {outcome:?}");

    let utterance = chat
        .lines
        .iter()
        .find_map(|line| match line {
            Line::Utterance(utterance) => Some(utterance),
            _ => None,
        })
        .expect("fixture utterance");
    L2Lines {
        mor: utterance
            .mor_tier()
            .expect("%mor after splice")
            .to_chat_string(),
        gra: utterance
            .gra_tier()
            .expect("%gra after splice")
            .to_chat_string(),
    }
}

/// D1, D2: an `@s` word after a contraction takes its OWN primary relation.
///
/// The primary Catalan analysis splits `al` into `a` + `el`, so the UD word
/// list holds a range row and two components before `cole`. `cole` is
/// CHAT word 3 but UD word 5 and vector position 5; reading position 3
/// picked up `a`'s `case` relation. Stanza's own relation for `cole` is
/// `obl:arg` under `anem`.
#[test]
fn at_s_word_after_a_contraction_takes_its_own_primary_relation() {
    let lines = run_l2(
        "cat, spa",
        "avui anem al cole@s per jugar .",
        // BA3 worker, cat, realigned.
        "1 avui avui ADV _ 2 advmod
         2 anem anar VERB Mood=Ind|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin 0 root
         3-4 al _ X _ 0 dep
         3 a a ADP _ 5 case
         4 el el DET Definite=Def|Gender=Masc|Number=Sing|PronType=Art 5 det
         5 cole cole NOUN Gender=Masc|Number=Sing 2 obl:arg
         6 per per ADP _ 7 mark
         7 jugar jugar VERB VerbForm=Inf 2 advcl",
        // BA3 worker, spa, Stanza-owned tokenization (secondary dispatch).
        &["1 cole cole NOUN Gender=Masc|Number=Sing 0 root
           2 . . PUNCT PunctType=Peri 1 punct"],
    );
    assert_eq!(
        lines.gra,
        "%gra:\t1|2|ADVMOD 2|0|ROOT 3|5|CASE 4|5|DET 5|2|OBL-ARG 6|7|MARK 7|2|ADVCL 8|2|PUNCT",
        "`cole` must carry Stanza's own `obl:arg` under `anem`; %mor is {}",
        lines.mor
    );
}

/// D1, D2: a two-word span after a contraction is attached through the
/// word whose primary head lies OUTSIDE the span.
///
/// The primary attaches `bon` (UD word 5) to `cole` (UD word 6), which is
/// inside the span, and `cole` to `anem`. The planner compared the CHAT
/// index of each span word plus one with the head's UD id; after `al` the
/// two are off by one, so `bon`'s head looked external and `bon` (or,
/// through D1, the `a` of `al`) became the attachment source.
///
/// Only the heads are asserted here; the external label is the subject of
/// the span-label test below.
#[test]
fn span_after_a_contraction_attaches_through_its_external_word() {
    let lines = run_l2(
        "cat, spa",
        "avui anem al bon@s cole@s per jugar .",
        // BA3 worker, cat, realigned.
        //.
        "1 avui avui ADV _ 2 advmod
         2 anem anar VERB Mood=Ind|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin 0 root
         3-4 al _ X _ 0 dep
         3 a a ADP _ 6 case
         4 el el DET Definite=Def|Gender=Masc|Number=Sing|PronType=Art 6 det
         5 bon bo ADJ Gender=Masc|Number=Sing 6 amod
         6 cole cole NOUN Gender=Masc|Number=Sing 2 obl:arg
         7 per per ADP _ 8 mark
         8 jugar jugar VERB VerbForm=Inf 2 advcl",
        // BA3 worker, spa, Stanza-owned tokenization.
        //.
        &["1 bon bon PROPN _ 0 root
           2 cole cole NOUN Gender=Masc|Number=Sing 1 compound
           3 . . PUNCT PunctType=Peri 1 punct"],
    );
    assert_eq!(
        heads(&lines.gra),
        "1|2 2|0 3|5 4|5 5|2 6|5 7|8 8|2 9|2",
        "the span's root `bon` attaches to `anem`, the head of `cole`; %gra is {}",
        lines.gra
    );
}

/// The `%mor` items of a `%mor` line, terminator included.
fn mor_items(mor: &str) -> Vec<&str> {
    mor.trim_start_matches("%mor:\t").split(' ').collect()
}

/// The `%gra` relation of chunk `index` (1-based), label included.
fn relation(gra: &str, index: usize) -> &str {
    gra.trim_start_matches("%gra:\t")
        .split(' ')
        .nth(index - 1)
        .unwrap_or("")
}

/// The English secondary analysis of `wake up .` / `give up .`, Stanza
/// owning tokenization:
/// the verb is the root, the particle is ADP `compound:prt`, and the
/// terminator row stays.
fn phrasal_secondary(verb: &str) -> String {
    format!(
        "1 {verb} {verb} VERB Mood=Imp|VerbForm=Fin 0 root
         2 up up ADP _ 1 compound:prt
         3 . . PUNCT _ 1 punct"
    )
}

/// D6: phrasal-verb structure reaches the merge even though the secondary
/// sentence has a terminator row the span has no word for.
///
/// The merge passed the secondary sentence context only when the UD word
/// count equalled the mapped `%mor` count. With the terminator row they
/// differ (3 against 2), so the context was withheld without a report and
/// the particle could not be recognised.
#[test]
fn phrasal_verb_context_survives_the_terminator_row() {
    let wake = run_l2(
        "deu, eng",
        "ich möchte wake@s up@s jetzt .",
        // BA3 worker, deu, realigned.
        "1 ich ich PRON Case=Nom|Number=Sing|Person=1|PronType=Prs 2 nsubj
         2 möchte mögen VERB Mood=Ind|Number=Sing|Person=1|Tense=Past|VerbForm=Fin 0 root
         3 wake wake INTJ _ 2 obj
         4 up up INTJ _ 2 obj
         5 jetzt jetzt ADV _ 2 advmod",
        &[&phrasal_secondary("wake")],
    );
    let give = run_l2(
        "deu, eng",
        "die kinder give@s up@s immer .",
        // BA3 worker, deu, realigned.
        "1 die der DET Case=Nom|Definite=Def|Number=Plur|PronType=Art 2 det
         2 kinder kind NOUN Case=Nom|Gender=Neut|Number=Plur 0 root
         3 give give X Foreign=Yes 2 advmod
         4 up up X Foreign=Yes 2 advmod
         5 immer immer ADV _ 2 advmod",
        &[&phrasal_secondary("give")],
    );
    for (lines, verb) in [(&wake, "wake"), (&give, "give")] {
        let items = mor_items(&lines.mor);
        assert!(
            items[2].starts_with(&format!("verb|{verb}")) && items[3] == "part|up",
            "`{verb} up` must be verb + particle; %mor is {}",
            lines.mor
        );
        assert_eq!(
            relation(&lines.gra, 4),
            "4|3|COMPOUND-PRT",
            "the particle attaches to the verb; %gra is {}",
            lines.gra
        );
    }
}

/// D5: the resolved POS is one a model assigned.
///
/// The primary French model calls `ja` an `obj` of `dis`; the merge read
/// the `obj` constraint and picked its most likely category, NOUN, while
/// both models tagged the word INTJ.
#[test]
fn an_interjection_both_models_agree_on_stays_an_interjection() {
    let lines = run_l2(
        "fra, nld",
        "je dis ja@s:nld maintenant .",
        // BA3 worker, fra, realigned.
        "1 je moi PRON Case=Nom|Emph=No|Number=Sing|Person=1|PronType=Prs 2 nsubj
         2 dis dire VERB Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin 0 root
         3 ja ja INTJ _ 2 obj
         4 maintenant maintenant ADV _ 2 advmod",
        // BA3 worker, nld, Stanza-owned tokenization.
        //.
        &["1 ja ja INTJ _ 0 root
           2 . . PUNCT _ 1 punct"],
    );
    assert_eq!(mor_items(&lines.mor)[2], "intj|ja", "%mor is {}", lines.mor);
}

/// D5: the secondary model's verb stays a verb.
///
/// The primary German model marks the English words foreign (`X`,
/// `Foreign=Yes`) and guesses `obj` for `working`; the merge took NOUN from
/// that guess and kept the secondary's verbal features, giving the
/// incoherent `noun|work-Part-Pres`.
#[test]
fn a_secondary_verb_is_not_renamed_from_the_primary_guess() {
    let lines = run_l2(
        "deu, eng",
        "ich glaube it's@s:eng working@s:eng und don't@s:eng stop@s:eng .",
        // BA3 worker, deu, realigned.
        "1 ich ich PRON Case=Nom|Number=Sing|Person=1|PronType=Prs 2 nsubj
         2 glaube glauben VERB Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin 0 root
         3 it's it's X Foreign=Yes 4 nsubj
         4 working working X Foreign=Yes 2 obj
         5 und und CCONJ Foreign=Yes 7 cc
         6 don't don't X Foreign=Yes 7 advmod
         7 stop stop NOUN Foreign=Yes 4 conj",
        // BA3 worker, eng, Stanza-owned tokenization.
        //.
        &[
            "1-2 it's _ X _ 0 dep
             1 it it PRON Case=Nom|Gender=Neut|Number=Sing|Person=3|PronType=Prs 3 nsubj
             2 's be AUX Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin 3 aux
             3 working work VERB Tense=Pres|VerbForm=Part 0 root
             4 . . PUNCT _ 3 punct",
            "1-2 don't _ X _ 0 dep
             1 do do AUX Mood=Imp|VerbForm=Fin 3 aux
             2 n't not PART Polarity=Neg 3 advmod
             3 stop stop VERB VerbForm=Inf 0 root
             4 . . PUNCT _ 3 punct",
        ],
    );
    let items = mor_items(&lines.mor);
    assert_eq!(items[3], "verb|work-Part-Pres", "%mor is {}", lines.mor);
    assert!(
        items[6].starts_with("verb|stop"),
        "`stop` is the secondary's verb; %mor is {}",
        lines.mor
    );
}

/// D4: inside a multi-word span the secondary's relations stand, and the
/// span's external label comes from the attachment source alone.
///
/// The primary English parse has `los` `obl` under `talked` and `niños`
/// `flat` under `los`; the Spanish secondary has `los` `det` under the root
/// `niños`. The span attaches through `los` (its head is outside), so the
/// secondary root `niños` carries `obl` to `talked`. A per-word correction
/// of `niños`'s own `flat` to `nmod` used to overwrite the secondary root's
/// relation, giving `5|2|NMOD`.
///
/// `about` still points at `los` (`3|4|CASE`): host words that pointed into
/// the span collapse to its first chunk in chatter's splice (D3), which the
/// next chatter release replaces with redirects BA3 supplies.
#[test]
fn span_label_comes_from_its_attachment_source() {
    let lines = run_l2(
        "eng, spa",
        "we talked about los@s:spa niños@s:spa .",
        // BA3 worker, eng, realigned.
        //.
        "1 we we PRON Case=Nom|Number=Plur|Person=1|PronType=Prs 2 nsubj
         2 talked talk VERB Mood=Ind|Number=Plur|Person=1|Tense=Past|VerbForm=Fin 0 root
         3 about about ADP _ 4 case
         4 los Los PROPN Number=Sing 2 obl
         5 niños niño PROPN Number=Sing 4 flat",
        // BA3 worker, spa, Stanza-owned tokenization.
        //.
        &[
            "1 los el DET Definite=Def|Gender=Masc|Number=Plur|PronType=Art 2 det
           2 niños niño NOUN Gender=Masc|Number=Plur 0 root
           3 . . PUNCT PunctType=Peri 2 punct",
        ],
    );
    assert_eq!(
        (relation(&lines.gra, 4), relation(&lines.gra, 5)),
        ("4|5|DET", "5|2|OBL"),
        "`los` is the secondary's DET under `niños`; `niños` carries `los`'s \
         `obl` to `talked`; %gra is {}",
        lines.gra
    );
}

/// D4 with D2's fixture: the span root `bon` carries the relation of the
/// attachment source `cole` (`obl:arg`), corrected to `obl` because the
/// secondary tagged `bon` PROPN, which `obl:arg` does not admit. The
/// per-word correction of `bon`'s own `amod` gave `NMOD`.
#[test]
fn span_label_is_corrected_against_the_span_root() {
    let lines = run_l2(
        "cat, spa",
        "avui anem al bon@s cole@s per jugar .",
        "1 avui avui ADV _ 2 advmod
         2 anem anar VERB Mood=Ind|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin 0 root
         3-4 al _ X _ 0 dep
         3 a a ADP _ 6 case
         4 el el DET Definite=Def|Gender=Masc|Number=Sing|PronType=Art 6 det
         5 bon bo ADJ Gender=Masc|Number=Sing 6 amod
         6 cole cole NOUN Gender=Masc|Number=Sing 2 obl:arg
         7 per per ADP _ 8 mark
         8 jugar jugar VERB VerbForm=Inf 2 advcl",
        &["1 bon bon PROPN _ 0 root
           2 cole cole NOUN Gender=Masc|Number=Sing 1 compound
           3 . . PUNCT PunctType=Peri 1 punct"],
    );
    assert_eq!(
        lines.gra,
        "%gra:\t1|2|ADVMOD 2|0|ROOT 3|5|CASE 4|5|DET 5|2|OBL 6|5|COMPOUND 7|8|MARK 8|2|ADVCL 9|2|PUNCT",
        "%mor is {}",
        lines.mor
    );
}

/// A `compound:prt` attached to a noun is no phrasal verb: `time out` is a
/// compound noun, so `out` keeps the secondary's ADP; the relation is the
/// secondary's own `compound:prt`.
#[test]
fn a_noun_with_a_particle_relation_is_not_a_phrasal_verb() {
    let lines = run_l2(
        "deu, eng",
        "die zeit ist time@s out@s .",
        // BA3 worker, deu, realigned.
        "1 die der DET Case=Nom|Definite=Def|Gender=Fem|Number=Sing|PronType=Art 2 det
         2 zeit zeit NOUN Case=Nom|Gender=Fem|Number=Sing 5 nsubj
         3 ist sein AUX Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin 5 cop
         4 time timen X Foreign=Yes 5 advmod
         5 out outen X Foreign=Yes 0 root",
        // BA3 worker, eng, Stanza-owned tokenization.
        //.
        &["1 time time NOUN Number=Sing 0 root
           2 out out ADP _ 1 compound:prt
           3 . . PUNCT _ 1 punct"],
    );
    let items = mor_items(&lines.mor);
    assert_eq!(
        (items[3], items[4]),
        ("noun|time", "adp|out"),
        "%mor is {}",
        lines.mor
    );
    assert_eq!(
        relation(&lines.gra, 5),
        "5|4|COMPOUND-PRT",
        "%gra is {}",
        lines.gra
    );
}

/// With `--retokenize`, an `@s` word after a contraction is placed by the
/// item it was written as, not by its CHAT word index.
///
/// The main tier is rebuilt from the model's tokens, one `%mor` item per
/// model word, so `gonna` becomes two items (`gon`, `na`) and `camino` is
/// item 3 while it is CHAT word 2. Placing by CHAT word gave item 2, `see`:
/// the special-form relabel wrote `see` as `L2|xxx`, `camino` kept the
/// primary's analysis, and the L2 splice replaced `see`.
#[test]
fn retokenized_at_s_word_after_a_contraction_is_placed_by_its_own_item() {
    let lines = run_l2_in(
        TokenizationMode::StanzaRetokenize,
        "eng, spa",
        "gonna see camino@s:spa .",
        "1-2 gonna _ X _ 0 dep
         1 gon go VERB VerbForm=Part 3 aux
         2 na to PART _ 3 mark
         3 see see VERB VerbForm=Inf 0 root
         4 camino camino NOUN Number=Sing 3 obj
         5 . . PUNCT _ 3 punct",
        &["1 camino camino NOUN Gender=Masc|Number=Sing 0 root
           2 . . PUNCT PunctType=Peri 1 punct"],
    );
    assert_eq!(
        mor_items(&lines.mor),
        [
            "verb|go-Part",
            "part|to",
            "verb|see-Inf",
            "noun|camino-Masc",
            "."
        ]
    );
    // `camino` keeps the primary's attachment to `see` (chunk 3).
    assert_eq!(relation(&lines.gra, 4), "4|3|OBJ");
}
