//! Valid CHAT controls with adversarial model-protocol doubles, not corpus gold.

use super::*;
use talkbank_model::WriteChat;

const HELLO: &str = "1 hello hello INTJ _ 0 root
    2 . . PUNCT _ 1 punct";

fn response(count: usize) -> UdResponse {
    UdResponse {
        sentences: (0..count)
            .map(|_| l2::pipeline_tests::ud_sentence(HELLO))
            .collect(),
    }
}

fn admitted_chat(parser: &TreeSitterParser, source: &str) -> ChatFile {
    crate::parse_and_validate_with_parser(
        parser,
        source,
        talkbank_model::ParseValidateOptions::default().with_alignment(),
    )
    .expect("every algorithm input must be fully valid CHAT")
}

fn items(chat: &ChatFile, language: &LanguageCode) -> Vec<CollectedUtterance> {
    collect_payloads(
        chat,
        language,
        std::slice::from_ref(language),
        MultilingualPolicy::ProcessAll,
    )
    .batch_items
}

#[test]
fn response_admission_preserves_the_whole_single_sentence_or_explicit_absence() {
    assert!(AdmittedUdResponse::empty().sentence().is_none());
    assert!(
        AdmittedUdResponse::try_from(response(0))
            .expect("empty admitted")
            .sentence()
            .is_none()
    );
    let raw = response(1);
    let expected = raw.sentences[0].clone();
    let admitted = AdmittedUdResponse::try_from(raw).expect("one sentence admitted");
    assert_eq!(admitted.sentence(), Some(&expected));
    for count in [2, 3, 8] {
        let error = AdmittedUdResponse::try_from(response(count)).expect_err("no prefix selection");
        assert_eq!(error.actual, count);
    }
}

#[test]
fn response_admission_keeps_wire_deserialization_distinct_from_model_admission() {
    let raw: UdResponse = serde_json::from_str(r#"{"sentences":[{"words":[]},{"words":[]}]}"#)
        .expect("raw wire remains representable for diagnostics");
    assert_eq!(raw.sentences.len(), 2);
    assert_eq!(
        AdmittedUdResponse::try_from(raw)
            .expect_err("unusable analysis")
            .actual,
        2
    );
}

#[test]
fn response_admission_refuses_first_or_late_multiple_sentences_before_any_mutation() {
    let parser = TreeSitterParser::new().expect("parser");
    let language = LanguageCode::new("eng").expect("language");
    let source = one_utterance("hello .").replace("@End", "*CHI:\thello .\n@End");
    for mode in [
        TokenizationMode::Preserve,
        TokenizationMode::StanzaRetokenize,
    ] {
        for bad_index in [0, 1] {
            let mut chat = admitted_chat(&parser, &source);
            let before = chat.to_chat_string();
            let payloads = items(&chat, &language);
            assert_eq!(payloads.len(), 2);
            let responses = (0..2)
                .map(|index| response(if index == bad_index { 2 } else { 1 }))
                .collect();
            let error = inject_results(
                &parser,
                &mut chat,
                payloads,
                responses,
                &language,
                mode,
                &MwtDict::default(),
            )
            .expect_err("a matching first sentence cannot justify ignoring the rest");
            assert!(
                matches!(error, InjectionError::SentenceCount { index, error }
                if index == bad_index && error.actual == 2)
            );
            assert_eq!(
                chat.to_chat_string(),
                before,
                "no earlier utterance may be mutated"
            );
        }
    }
}

#[test]
fn response_admission_preserves_batch_count_precedence_and_refuses_before_mutation() {
    let parser = TreeSitterParser::new().expect("parser");
    let language = LanguageCode::new("eng").expect("language");
    for count in [0, 2] {
        let mut chat = admitted_chat(&parser, &one_utterance("hello ."));
        let before = chat.to_chat_string();
        let payloads = items(&chat, &language);
        let error = inject_results(
            &parser,
            &mut chat,
            payloads,
            (0..count).map(|_| response(2)).collect(),
            &language,
            TokenizationMode::Preserve,
            &MwtDict::default(),
        )
        .expect_err("a mismatched batch is refused independently of its sentences");
        assert!(matches!(error, InjectionError::ResponseCount(error)
            if error.expected == 1 && error.actual == count));
        assert_eq!(chat.to_chat_string(), before);
    }
}

#[test]
fn response_admission_empty_analysis_is_not_lexical_completion_but_special_forms_still_complete() {
    let parser = TreeSitterParser::new().expect("parser");
    let language = LanguageCode::new("eng").expect("language");
    for mode in [
        TokenizationMode::Preserve,
        TokenizationMode::StanzaRetokenize,
    ] {
        for (body, special_form) in [("hello .", false), ("boom@o .", true)] {
            let mut chat = admitted_chat(&parser, &one_utterance(body));
            let payloads = items(&chat, &language);
            let result = MatchedMorphosyntaxResponses::from_admitted(
                payloads,
                vec![AdmittedUdResponse::empty()],
            )
            .expect("paired empty analysis")
            .inject(&parser, &mut chat, mode, &MwtDict::default());
            if special_form {
                result.expect("typed onomatopoeia synthesis needs no model");
                assert!(chat.to_chat_string().contains("on|boom"));
                admitted_chat(&parser, &chat.to_chat_string());
            } else {
                assert!(matches!(result, Err(InjectionError::Incomplete(_))));
            }
        }
    }
}

#[test]
fn response_admission_single_sentence_preserves_normal_injection_in_both_modes() {
    let parser = TreeSitterParser::new().expect("parser");
    let language = LanguageCode::new("eng").expect("language");
    let mut contracts = Vec::new();
    for mode in [
        TokenizationMode::Preserve,
        TokenizationMode::StanzaRetokenize,
    ] {
        let mut chat = admitted_chat(&parser, &one_utterance("hello ."));
        let payloads = items(&chat, &language);
        inject_results(
            &parser,
            &mut chat,
            payloads,
            vec![response(1)],
            &language,
            mode,
            &MwtDict::default(),
        )
        .expect("one complete analysis remains supported");
        admitted_chat(&parser, &chat.to_chat_string());
        let utterance = first_utterance(&chat);
        assert!(utterance.mor_tier().is_some());
        assert!(utterance.gra_tier().is_some());
        contracts.push(utterance.main.to_chat_string());
    }
    insta::assert_debug_snapshot!(contracts, @r###"
    [
        "*CHI:\thello .",
        "*CHI:\thello .",
    ]
    "###);
}
