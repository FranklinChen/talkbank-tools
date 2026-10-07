//! Valid-source regression deck for complete scoped contraction expansion.

use super::*;
use talkbank_model::WriteChat;

const CONTRACTION: &str = "1 I I PRON Case=Nom|Person=1 4 nsubj
    2-3 can't _ X _ 0 dep
    2 ca can AUX VerbForm=Fin 4 aux
    3 n't not PART _ 4 advmod
    4 go go VERB VerbForm=Inf 0 root
    5 . . PUNCT _ 4 punct";
const EXCLUDED: &str = "1 I I PRON Case=Nom|Person=1 2 nsubj
    2 go go VERB VerbForm=Inf 0 root
    3 . . PUNCT _ 2 punct";

fn inject_case(
    body: &str,
    rows: &str,
    mode: TokenizationMode,
) -> (ChatFile, Result<InjectionResult, InjectionError>) {
    inject_language_case(body, rows, mode, "eng")
}

fn inject_language_case(
    body: &str,
    rows: &str,
    mode: TokenizationMode,
    language: &str,
) -> (ChatFile, Result<InjectionResult, InjectionError>) {
    let parser = TreeSitterParser::new().expect("parser");
    let source = one_utterance(body).replace("eng", language);
    let mut chat = crate::parse_and_validate_with_parser(
        &parser,
        &source,
        talkbank_model::ParseValidateOptions::default(),
    )
    .expect("the algorithm fixture must be fully valid CHAT");
    let language = LanguageCode::new(language).expect("language");
    let payloads = collect_payloads(
        &chat,
        &language,
        std::slice::from_ref(&language),
        MultilingualPolicy::ProcessAll,
    );
    let result = inject_results(
        &parser,
        &mut chat,
        payloads.batch_items,
        vec![UdResponse {
            sentences: vec![l2::pipeline_tests::ud_sentence(rows)],
        }],
        &language,
        mode,
        &std::collections::BTreeMap::new(),
    );
    (chat, result)
}

#[test]
fn explicit_mwt_ownership_survives_spelling_changes_and_annotation_scopes() {
    // Controlled UD double, not a captured model response. The paired managed
    // run separately establishes that the model produces this expansion.
    let rows = "1-3 decírmelo _ X _ 0 dep\n1 decir decir VERB VerbForm=Inf 0 root\n2 me yo PRON Case=Dat|Person=1 1 obl:arg\n3 lo él PRON Case=Acc|Person=3 1 obj\n4 . . PUNCT _ 1 punct";
    let cases = [
        ("decírmelo .", "decir me lo ."),
        (
            "decírmelo [= infinitive] .",
            "<decir me lo> [= infinitive] .",
        ),
        (
            "<decírmelo> [= infinitive] .",
            "<decir me lo> [= infinitive] .",
        ),
        (
            "decirlo [: decírmelo] [= infinitive] .",
            "decirlo [: decir me lo] [= infinitive] .",
        ),
    ];
    let mut contracts = Vec::new();
    for (body, expected) in cases {
        let (mut chat, result) =
            inject_language_case(body, rows, TokenizationMode::StanzaRetokenize, "spa");
        result.expect("explicit MWT ownership admits every component");
        validate_morphosyntax(&mut chat);
        let utterance = first_utterance(&chat);
        let expected = parse_chat(&one_utterance(expected));
        assert_eq!(
            utterance.main.to_chat_string(),
            first_utterance(&expected).main.to_chat_string()
        );
        assert_eq!(utterance.mor_tier().expect("complete MOR").items().len(), 3);
        assert_eq!(
            utterance
                .gra_tier()
                .expect("complete GRA")
                .relations()
                .len(),
            4
        );
        crate::parse_and_validate_with_parser(
            &TreeSitterParser::new().expect("parser"),
            &chat.to_chat_string(),
            talkbank_model::ParseValidateOptions::default().with_alignment(),
        )
        .expect("serialized expansion must remain fully valid CHAT");
        contracts.push(utterance.main.to_chat_string());
    }
    insta::assert_debug_snapshot!(contracts, @r###"
    [
        "*CHI:\tdecir me lo .",
        "*CHI:\t<decir me lo> [= infinitive] .",
        "*CHI:\t<decir me lo> [= infinitive] .",
        "*CHI:\tdecirlo [: decir me lo] [= infinitive] .",
    ]
    "###);
}

#[test]
fn scoped_expansion_preserves_every_token_and_the_complete_annotation_scope() {
    let cases = [
        ("I can't [= cannot] go .", "I <ca n't> [= cannot] go ."),
        ("<I can't> [= refusal] go .", "<I ca n't> [= refusal] go ."),
        (
            "I <can't [= cannot] go> [= scope] .",
            "I <<ca n't> [= cannot] go> [= scope] .",
        ),
        ("“I can't [= cannot]” go .", "“I <ca n't> [= cannot]” go ."),
        ("I cannot [: can't] go .", "I cannot [: ca n't] go ."),
        (
            "I <cannot [: can't] go> [= scope] .",
            "I <cannot [: ca n't] go> [= scope] .",
        ),
        (
            "I cannot [: can't] [= corrected] go .",
            "I cannot [: ca n't] [= corrected] go .",
        ),
    ];
    let mut contracts = Vec::new();
    for (source, expected) in cases {
        let (mut chat, admitted) =
            inject_case(source, CONTRACTION, TokenizationMode::StanzaRetokenize);
        admitted.expect("the whole expansion must complete");
        validate_morphosyntax(&mut chat);
        let actual = first_utterance(&chat);
        let expected_chat = parse_chat(&one_utterance(expected));
        assert_eq!(
            actual.main.to_chat_string(),
            first_utterance(&expected_chat).main.to_chat_string(),
            "{source}"
        );
        assert_eq!(actual.mor_tier().expect("complete MOR").items().len(), 4);
        assert_eq!(
            actual.gra_tier().expect("complete GRA").relations().len(),
            5
        );
        // Serialization/grammar boundary: typed output must also be valid
        // when consumed by another parser, including nested group scopes.
        let parser = TreeSitterParser::new().expect("parser");
        crate::parse_and_validate_with_parser(
            &parser,
            &chat.to_chat_string(),
            talkbank_model::ParseValidateOptions::default().with_alignment(),
        )
        .expect("serialized output must be fully valid CHAT");
        contracts.push(actual.main.to_chat_string());
    }
    insta::assert_debug_snapshot!(contracts, @r###"
    [
        "*CHI:\tI <ca n't> [= cannot] go .",
        "*CHI:\t<I ca n't> [= refusal] go .",
        "*CHI:\tI <<ca n't> [= cannot] go> [= scope] .",
        "*CHI:\t“I <ca n't> [= cannot]” go .",
        "*CHI:\tI cannot [: ca n't] go .",
        "*CHI:\tI <cannot [: ca n't] go> [= scope] .",
        "*CHI:\tI cannot [: ca n't] [= corrected] go .",
    ]
    "###);
}

#[test]
fn preserve_mode_does_not_expand_scoped_words_or_replacement_targets() {
    for body in ["I can't [= cannot] go .", "I cannot [: can't] go ."] {
        let (mut chat, admitted) = inject_case(body, CONTRACTION, TokenizationMode::Preserve);
        admitted.expect("preserved contraction");
        validate_morphosyntax(&mut chat);
        let source = parse_chat(&one_utterance(body));
        assert_eq!(
            first_utterance(&chat).main.to_chat_string(),
            first_utterance(&source).main.to_chat_string()
        );
        assert_eq!(
            first_utterance(&chat)
                .mor_tier()
                .expect("MOR")
                .items()
                .len(),
            3
        );
    }
}

#[test]
fn excluded_scoped_words_and_replacements_consume_no_model_positions() {
    for body in ["I can't [e] go .", "I cannot [: can't] [e] go ."] {
        let (mut chat, admitted) = inject_case(body, EXCLUDED, TokenizationMode::StanzaRetokenize);
        admitted.expect("excluded source stays excluded");
        validate_morphosyntax(&mut chat);
        let source = parse_chat(&one_utterance(body));
        assert_eq!(
            first_utterance(&chat).main.to_chat_string(),
            first_utterance(&source).main.to_chat_string()
        );
        assert_eq!(
            first_utterance(&chat)
                .mor_tier()
                .expect("MOR")
                .items()
                .len(),
            2
        );
    }
}

#[test]
fn an_unsafe_expansion_refuses_without_mutating_the_source_or_claiming_completion() {
    for (body, rows) in [
        ("I can't$n [= cannot] go .", CONTRACTION),
        // Equal cardinality cannot disguise a length-guessed correspondence.
        (
            "I dog [= animal] go .",
            "1 I I PRON _ 3 nsubj\n2 cat cat NOUN _ 3 obj\n3 go go VERB _ 0 root\n4 . . PUNCT _ 3 punct",
        ),
        // Normalized concatenation matches, but a token would cross scope.
        (
            "I can [= ability] not go .",
            "1 I I PRON _ 3 nsubj\n2 cannot can AUX _ 3 aux\n3 go go VERB _ 0 root\n4 . . PUNCT _ 3 punct",
        ),
    ] {
        let source = parse_chat(&one_utterance(body));
        let (chat, refused) = inject_case(body, rows, TokenizationMode::StanzaRetokenize);
        assert!(
            refused.is_err(),
            "{body}: no completion for unsafe expansion"
        );
        let actual = first_utterance(&chat);
        assert_eq!(
            actual.main.to_chat_string(),
            first_utterance(&source).main.to_chat_string()
        );
        assert!(actual.mor_tier().is_none());
        assert!(actual.gra_tier().is_none());
    }
}
