//! Unit tests for the morphosyntax module's classifier, outcome
//! conversion, UD type round-trips, and small text-sanitization
//! helpers.

#![cfg(test)]

use super::*;
use talkbank_model::alignment::helpers::{MorAlignableWordCount, MorItemCount};
use talkbank_model::model::ChatFile;
use talkbank_model::model::{LanguageCode, SpeakerCode, Utterance};
use talkbank_parser::TreeSitterParser;

use crate::inject::{MisalignmentClass, MisalignmentDiagnostic};
use crate::parse::parse_lenient;

pub(crate) fn parse_chat(text: &str) -> ChatFile {
    let parser = TreeSitterParser::new().expect("parser init");
    parser.parse_chat_file(text).expect_built()
}

fn validate_morphosyntax(chat: &mut ChatFile) {
    use talkbank_model::ParseValidateOptions;
    let opts = ParseValidateOptions::default().with_alignment();
    if let Err(e) = talkbank_model::validate_chat_file_with_options(chat, &opts) {
        panic!("Morphosyntax validation failed: {:#?}", e);
    }
}

fn one_utterance(main_tier: &str) -> String {
    one_utterance_in("eng", main_tier)
}

/// One-utterance fixture with an explicit `@Languages` list, for the
/// code-switching cases where a second declared language is the point.
pub(crate) fn one_utterance_in(languages: &str, main_tier: &str) -> String {
    format!(
        "@UTF8\n\
         @Begin\n\
         @Languages:\t{languages}\n\
         @Participants:\tCHI Target_Child\n\
         @ID:\teng|test|CHI||female|||Target_Child|||\n\
         *CHI:\t{main_tier}\n\
         @End\n"
    )
}

pub(crate) fn first_utterance(chat: &ChatFile) -> &Utterance {
    for line in &chat.lines {
        if let talkbank_model::model::Line::Utterance(u) = line {
            return u;
        }
    }
    panic!("no utterance in chat file")
}

#[test]
fn classify_filler_only() {
    let mut chat = parse_chat(&one_utterance("&-hmm ."));
    assert_eq!(
        classify_not_applicable(first_utterance(&chat)),
        NotApplicableReason::FillerOnly,
    );
    validate_morphosyntax(&mut chat);
}

#[test]
fn classify_multiple_fillers() {
    let chat = parse_chat(&one_utterance("&-hmm &-hmm ."));
    assert_eq!(
        classify_not_applicable(first_utterance(&chat)),
        NotApplicableReason::FillerOnly,
    );
}

#[test]
fn classify_fragment_only() {
    let chat = parse_chat(&one_utterance("&+le ."));
    assert_eq!(
        classify_not_applicable(first_utterance(&chat)),
        NotApplicableReason::FragmentOnly,
    );
}

#[test]
fn classify_nonword_only() {
    let chat = parse_chat(&one_utterance("&~ach ."));
    assert_eq!(
        classify_not_applicable(first_utterance(&chat)),
        NotApplicableReason::NonwordOnly,
    );
}

#[test]
fn classify_untranscribed_only() {
    let chat = parse_chat(&one_utterance("xxx ."));
    assert_eq!(
        classify_not_applicable(first_utterance(&chat)),
        NotApplicableReason::UntranscribedOnly,
    );
}

#[test]
fn classify_mixed_nonlinguistic() {
    let chat = parse_chat(&one_utterance("&-hmm &+le ."));
    assert_eq!(
        classify_not_applicable(first_utterance(&chat)),
        NotApplicableReason::MixedNonLinguistic,
    );
}

#[test]
fn to_decision_record_aligned_is_none() {
    let outcome = MorOutcome {
        line_idx: 5,
        speaker: SpeakerCode::new("CHI"),
        kind: MorOutcomeKind::Aligned { n_words: 3 },
    };
    assert!(outcome.to_decision_record().is_none());
}

#[test]
fn to_decision_record_not_applicable_has_reason() {
    let outcome = MorOutcome {
        line_idx: 5,
        speaker: SpeakerCode::new("CHI"),
        kind: MorOutcomeKind::NotApplicable {
            reason: NotApplicableReason::FillerOnly,
        },
    };
    let d = outcome.to_decision_record().unwrap();
    assert!(matches!(
        d.strategy,
        crate::decisions::DecisionStrategy::Morphosyntax(
            crate::decisions::MorphosyntaxStrategy::NotApplicable
        )
    ));
    assert_eq!(d.reason, "reason=filler_only");
    assert!(!d.needs_review);
}

#[test]
fn to_decision_record_misalignment_has_diagnostic() {
    let outcome = MorOutcome {
        line_idx: 5,
        speaker: SpeakerCode::new("CHI"),
        kind: MorOutcomeKind::MisalignmentBug(MisalignmentDiagnostic {
            chat_words: vec!["hello".into(), "world".into()],
            stanza_tokens_after_mapping: vec!["hello".into()],
            expected: MorAlignableWordCount::new(2),
            actual: MorItemCount::new(1),
            suspected_class: MisalignmentClass::TerminatorFilterBug,
        }),
    };
    let d = outcome.to_decision_record().unwrap();
    assert!(matches!(
        d.strategy,
        crate::decisions::DecisionStrategy::Morphosyntax(
            crate::decisions::MorphosyntaxStrategy::MisalignmentBug
        )
    ));
    assert!(d.needs_review);
    assert!(d.reason.contains("class=terminator_filter_bug"));
    assert!(d.reason.contains("expected=2"));
    assert!(d.reason.contains("actual=1"));
}

#[test]
fn universal_pos_round_trips_to_chat_name_and_back() {
    for v in [
        UniversalPos::Adj,
        UniversalPos::Adp,
        UniversalPos::Adv,
        UniversalPos::Aux,
        UniversalPos::Cconj,
        UniversalPos::Det,
        UniversalPos::Intj,
        UniversalPos::Noun,
        UniversalPos::Num,
        UniversalPos::Part,
        UniversalPos::Pron,
        UniversalPos::Propn,
        UniversalPos::Punct,
        UniversalPos::Sconj,
        UniversalPos::Verb,
    ] {
        let name = v.to_chat_pos_name();
        assert_eq!(UniversalPos::from_pos_name(name), Some(v));
    }
    assert_eq!(UniversalPos::Sym.to_chat_pos_name(), "x");
    assert_eq!(UniversalPos::X.to_chat_pos_name(), "x");
    assert_eq!(UniversalPos::from_pos_name("x"), Some(UniversalPos::X));
    assert_eq!(UniversalPos::from_pos_name("sym"), Some(UniversalPos::X));
}

#[test]
fn universal_pos_accepts_case_insensitive_names() {
    assert_eq!(
        UniversalPos::from_pos_name("NOUN"),
        Some(UniversalPos::Noun)
    );
    assert_eq!(
        UniversalPos::from_pos_name("noun"),
        Some(UniversalPos::Noun)
    );
    assert_eq!(
        UniversalPos::from_pos_name("Noun"),
        Some(UniversalPos::Noun)
    );
    assert_eq!(UniversalPos::from_pos_name("notreal"), None);
}

#[test]
fn stanza_language_support_matches_expected_examples() {
    for code in ["eng", "spa", "fra", "deu", "zho", "jpn", "rus", "ara"] {
        assert!(is_stanza_supported(
            &LanguageCode::new(code).expect("valid test language code")
        ));
    }
    for code in ["que", "jam", "nan", "taq", "und", "xmm", "jav", "wuu"] {
        assert!(!is_stanza_supported(
            &LanguageCode::new(code).expect("valid test language code")
        ));
    }
    assert!(is_stanza_supported(
        &LanguageCode::new("yue").expect("valid test language code")
    ));
    assert!(is_stanza_supported(
        &LanguageCode::new("cmn").expect("valid test language code")
    ));
    for code in ["ben", "kan", "mal", "msa", "tgl", "ltz"] {
        assert!(!is_stanza_supported(
            &LanguageCode::new(code).expect("valid test language code")
        ));
    }
}

#[test]
fn dep_rel_roundtrips_known_variants() {
    for rel in [
        "root",
        "nsubj",
        "nsubj:pass",
        "obj",
        "aux",
        "aux:pass",
        "cop",
        "case",
        "nmod:poss",
        "det",
        "cc",
        "conj",
        "compound",
        "compound:prt",
        "amod",
        "advmod",
        "punct",
        "discourse",
        "mark",
        "expl",
    ] {
        assert_eq!(DepRel::parse(rel).as_str(), rel);
    }
}

#[test]
fn dep_rel_preserves_unknown_values() {
    let rel = DepRel::parse("orphan");
    assert_eq!(rel, DepRel::Other("orphan".to_string()));
    assert_eq!(rel.as_str(), "orphan");
}

#[test]
fn verb_form_roundtrips() {
    for value in ["Fin", "Part", "Ger", "Inf", "Sup", "Conv", "Vnoun"] {
        assert_eq!(VerbForm::parse(value).as_str(), value);
    }
}

#[test]
fn ud_pair_value_reads_one_key_among_several() {
    let misc = Some("SpaceAfter=No|VerbReadingLemma=bark|Note=a=b");
    assert_eq!(ud_pair_value(misc, "VerbReadingLemma"), Some("bark"));
    assert_eq!(ud_pair_value(misc, "Note"), Some("a=b"));
    assert_eq!(ud_pair_value(misc, "Verb"), None);
    assert_eq!(ud_pair_value(None, "SpaceAfter"), None);
}

#[test]
fn bogus_lemma_detection_matches_expected_cases() {
    assert!(is_bogus_lemma("hello", "."));
    assert!(is_bogus_lemma("world", ","));
    assert!(is_bogus_lemma("cat", "\u{2013}"));
    assert!(!is_bogus_lemma("hello", "hello"));
    assert!(!is_bogus_lemma("hello", ""));
    assert!(!is_bogus_lemma(".", "."));
    assert!(!is_bogus_lemma(",", "--"));
    assert!(!is_bogus_lemma("running", "run"));
    assert!(!is_bogus_lemma("cats", "cat"));
}

#[test]
fn validate_and_clean_fixes_pad_deprel_and_bogus_lemma() {
    let mut word = UdWord::from(UdWordAnalysis {
        id: UdId::Single(1),
        text: "hello".to_string(),
        lemma: ".".to_string(),
        upos: UdPunctable::Value(UniversalPos::Intj),
        xpos: None,
        feats: None,
        head: 0,
        deprel: "<pad>".to_string(),
        deps: None,
        misc: None,
    });

    validate_and_clean(&mut word);

    assert_eq!(word.lemma, "hello");
    assert_eq!(word.deprel, "dep");
}

#[test]
fn sanitize_mor_text_replaces_structural_separators() {
    assert_eq!(sanitize_mor_text("foo|bar"), "foo_bar");
    assert_eq!(sanitize_mor_text("a#b-c&d$e~f"), "a_b_c_d_e_f");
}

#[test]
fn sanitize_mor_text_strips_whitespace() {
    assert_eq!(sanitize_mor_text("ふ す"), "ふす");
    assert_eq!(sanitize_mor_text(" hello world "), "helloworld");
    assert_eq!(sanitize_mor_text("a\tb\nc"), "abc");
}

#[test]
fn sanitize_mor_text_handles_combined_issues() {
    assert_eq!(sanitize_mor_text("foo | bar"), "foo_bar");
    assert_eq!(sanitize_mor_text("ふ す#test"), "ふす_test");
}

#[test]
fn sanitize_mor_text_passthroughs_clean_text() {
    assert_eq!(sanitize_mor_text("hello"), "hello");
    assert_eq!(sanitize_mor_text("ふす"), "ふす");
}

/// CA-prosody terminators on the main tier must be substituted with
/// Period in the morphotag payload so synthesized `%mor` is valid CHAT.
#[test]
fn ca_arrow_terminator_must_normalize_to_period_in_morphotag_payload() {
    use crate::morphosyntax::payload::{collect_payloads, declared_languages};
    use crate::morphosyntax::types::MultilingualPolicy;
    let parser = TreeSitterParser::new().unwrap();
    let chat = "@UTF8\n\
                @Begin\n\
                @Languages:\teng\n\
                @Participants:\tPAR Participant\n\
                @ID:\teng|test|PAR|||||Participant|||\n\
                @Options:\tCA\n\
                *PAR:\tyes →\n\
                @End\n";
    let (chat_file, _) = parse_lenient(&parser, chat);

    let primary = LanguageCode::new("eng").expect("valid test language code");
    let langs = declared_languages(&chat_file, &primary);
    let items =
        collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll).batch_items;

    assert_eq!(
        items.len(),
        1,
        "expected one batch item for the single utterance"
    );
    let item = items[0].item();

    assert!(
        matches!(item.terminator, talkbank_model::Terminator::Period { .. }),
        "got: {:?}",
        item.terminator,
    );
}

/// `@s` in a `cat,spa` document with `primary_lang="eng"` (the job-level
/// fallback the dispatch layer fabricates when the job's language is
/// `WorkerLanguage::Unspecified`) must resolve to the file's secondary
/// declared language `spa`, not to the bogus `eng`.
///
/// Pre-2026-05-02 this resolved to `Single("eng")` because
/// `collect_payloads` computed `tier_language = utt_lang.or(Some(primary_lang))`,
/// skipping the `declared_languages.first()` step that `utterance_lang`
/// already used. Combined with the resolver's then-fabricated
/// `Single(tier_lang)` sentinel, every `@s` token in a batch-default-eng
/// run produced fake `Single("eng")` resolutions that routed L2
/// secondary dispatch through the wrong Stanza pipeline.
#[test]
fn collect_payloads_resolves_at_s_against_file_languages_not_batch_default() {
    use crate::parse::parse_lenient;

    let parser = TreeSitterParser::new().unwrap();
    let chat = include_str!("../../../../test-fixtures/cat_spa_dona_at_s.cha");
    let (chat_file, _) = parse_lenient(&parser, chat);

    let primary = LanguageCode::new("eng").expect("valid test language code"); // simulated batch default
    let langs = declared_languages(&chat_file, &primary);
    let items =
        collect_payloads(&chat_file, &primary, &langs, MultilingualPolicy::ProcessAll).batch_items;

    assert_eq!(items.len(), 1);
    let item = items[0].item();
    assert_eq!(
        item.lang.as_str(),
        "cat",
        "dispatch lang must come from file header"
    );

    let dona = item
        .words()
        .iter()
        .find(|w| w.text().as_ref() == "dona")
        .expect("payload must include the dona word");

    let WordRole::CodeSwitched(resolved) = dona.role() else {
        panic!("dona@s must produce a language resolution");
    };
    let resolved_langs: Vec<&str> = resolved.languages().iter().map(|c| c.as_str()).collect();
    assert_eq!(
        resolved_langs,
        vec!["spa"],
        "@s on cat-tier with declared [cat, spa] must resolve to spa, never eng",
    );
}

#[test]
fn lang2_normalizes_common_codes() {
    assert_eq!(lang2("eng"), "en");
    assert_eq!(lang2("fra"), "fr");
    assert_eq!(lang2("jpn"), "ja");
    assert_eq!(lang2("deu"), "de");
    assert_eq!(lang2("heb"), "he");
    assert_eq!(lang2("en"), "en");
}

/// Each batch word carries its own role, and the text the model receives
/// follows from the role: a special form sends the placeholder, a
/// code-switched word (even one with a form type) its own text.
#[test]
fn a_batch_word_carries_its_role_and_the_text_it_implies() {
    let chat = parse_chat(&one_utterance_in(
        "eng, spa",
        "the gumma@c and camino@s:spa go .",
    ));
    let eng = LanguageCode::new("eng").expect("valid language code");
    let spa = LanguageCode::new("spa").expect("valid language code");
    let payloads = payload::collect_payloads(
        &chat,
        &eng,
        &[eng.clone(), spa],
        types::MultilingualPolicy::ProcessAll,
    );
    let item = payloads.batch_items[0].item();
    let roles: Vec<(&str, &str)> = item
        .words()
        .iter()
        .map(|word| {
            let role = match word.role() {
                WordRole::Analysed => "analysed",
                WordRole::SpecialForm(_) => "special form",
                WordRole::CodeSwitched(_) => "code-switched",
            };
            (word.text().as_str(), role)
        })
        .collect();
    let placeholder = talkbank_model::ChatCleanedText::stanza_placeholder();
    assert_eq!(
        roles,
        [
            ("the", "analysed"),
            (placeholder.as_str(), "special form"),
            ("and", "analysed"),
            ("camino", "code-switched"),
            ("go", "analysed"),
        ]
    );
    // On the wire, each word's role is a (form type, language) pair.
    let wire = serde_json::to_value(item).expect("the item serializes");
    assert_eq!(
        wire["special_forms"],
        serde_json::json!([
            [null, null],
            ["c", null],
            [null, null],
            [null, "spa"],
            [null, null]
        ])
    );
}

/// Inject `primary` (fixture rows, see `l2::pipeline_tests::ud_sentence`)
/// as the analysis of one utterance in preserve mode; the `%mor` and `%gra`
/// lines written, and the injection's result.
fn inject_one(
    languages: &str,
    main_tier: &str,
    primary: &str,
) -> (String, String, InjectionResult) {
    use talkbank_model::WriteChat;
    let mut chat = parse_chat(&one_utterance_in(languages, main_tier));
    let codes: Vec<LanguageCode> = languages
        .split(',')
        .map(|code| LanguageCode::new(code.trim()).expect("fixture language code"))
        .collect();
    let payloads = payload::collect_payloads(
        &chat,
        &codes[0],
        &codes,
        types::MultilingualPolicy::ProcessAll,
    );
    let parser = TreeSitterParser::new().expect("parser");
    let injection = inject_results(
        &parser,
        &mut chat,
        payloads.batch_items,
        vec![UdResponse {
            sentences: vec![l2::pipeline_tests::ud_sentence(primary)],
        }],
        &codes[0],
        TokenizationMode::Preserve,
        &std::collections::BTreeMap::new(),
    )
    .expect("injection");
    let utt = first_utterance(&chat);
    let mor = utt
        .mor_tier()
        .map(|t| t.to_chat_string())
        .unwrap_or_default();
    let gra = utt
        .gra_tier()
        .map(|t| t.to_chat_string())
        .unwrap_or_default();
    (mor, gra, injection)
}

/// A special form after a contraction is relabelled at its own chunk.
///
/// `it's` is one `%mor` item of two chunks, so the special form is item 1
/// but chunk 3. The relabel used to pair items with relations by position,
/// so it rewrote chunk 2, the clitic `'s`, from `COP` to `DEP`.
#[test]
fn a_special_form_after_a_contraction_is_relabelled_at_its_own_chunk() {
    let (mor, gra, injection) = inject_one(
        "eng",
        "it's gumma@c .",
        "1-2 it's _ X _ 0 dep
         1 it it PRON Case=Nom|Number=Sing|Person=3|PronType=Prs 3 nsubj
         2 's be AUX Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin 3 cop
         3 xbxxx xbxxx NOUN Number=Sing 0 root",
    );
    assert!(injection.decisions.is_empty(), "{:?}", injection.decisions);
    assert!(mor.contains("~aux|be"), "{mor}");
    assert_eq!(gra, "%gra:\t1|3|NSUBJ 2|3|COP 3|0|ROOT 4|3|PUNCT");
}

/// The `@s` positions are read from the analysis the `%mor` mapping reads:
/// after the grammatical invariants rewrote it.
///
/// Stanza 1.14 reads `hafta` as one AUX under `put`, the root; the
/// contraction invariant expands it and raises `have` over `put`, which
/// becomes its `xcomp`. The `%gra` says so, and the deferred position of
/// `put@s` must too. Extraction used to read the raw analysis, so the
/// position called `put` the utterance root while the `%gra` hung it under
/// `have`.
#[test]
fn at_s_positions_read_the_analysis_the_mapping_reads() {
    let (_, gra, injection) = inject_one(
        "eng, spa",
        "you hafta put@s:spa .",
        "1 you you PRON Case=Nom|Person=2|PronType=Prs 3 nsubj
         2 hafta hafta AUX Mood=Ind|Number=Sing|Person=2|Tense=Pres|VerbForm=Fin 3 aux
         3 put put VERB VerbForm=Inf 0 root",
    );
    assert!(injection.decisions.is_empty(), "{:?}", injection.decisions);
    assert_eq!(
        gra,
        "%gra:\t1|2|NSUBJ 2|0|ROOT 3|4|MARK 4|2|XCOMP 5|2|PUNCT"
    );
    let position = injection.l2.positions().first().expect("put@s is deferred");
    assert_eq!(position.word_idx().as_usize(), 2);
    assert_eq!(position.primary().deprel().base(), "xcomp");
    assert_eq!(
        position.primary().head(),
        l2::HeadTarget::Word(talkbank_model::alignment::MorItemIndex::new(1))
    );
}
