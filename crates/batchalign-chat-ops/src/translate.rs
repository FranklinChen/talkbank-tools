//! Compatibility re-exports for canonical translation helpers now defined in
//! `talkbank-transform`.

pub use talkbank_transform::translate::{
    TranslateBatchItem, TranslateResponse, TranslationStringsEntry, apply_translate_results,
    chat_punct_chars, collect_translate_payloads, extract_translation_strings, inject_translation,
    postprocess_translation, preprocess_for_translate,
};

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use talkbank_model::model::{ChatFile, Line, WriteChat};
    use talkbank_parser::TreeSitterParser;

    fn parse_chat(text: &str) -> ChatFile {
        let parser = TreeSitterParser::new().unwrap();
        parser.parse_chat_file(text).unwrap()
    }

    fn get_utterance_mut(chat: &mut ChatFile, idx: usize) -> &mut talkbank_model::model::Utterance {
        let mut utt_idx = 0;
        for line in &mut chat.lines {
            if let Line::Utterance(utt) = line {
                if utt_idx == idx {
                    return utt;
                }
                utt_idx += 1;
            }
        }
        panic!("Utterance {idx} not found");
    }

    #[test]
    fn test_collect_translate_payloads() {
        let chat_text = include_str!("../../../test-fixtures/eng_hello_i_eat_cookies_zero.cha");
        let chat = parse_chat(chat_text);
        let payloads = collect_translate_payloads(&chat);

        assert!(payloads.len() >= 2);
        assert_eq!(payloads[0].1.text, "hello");
        assert_eq!(payloads[1].1.text, "I eat cookies");
    }

    #[test]
    fn test_inject_translation() {
        let chat_text = include_str!("../../../test-fixtures/eng_hello_female.cha");
        let mut chat = parse_chat(chat_text);
        let utt = get_utterance_mut(&mut chat, 0);
        inject_translation(utt, "hola").unwrap();

        let output = chat.to_chat_string();
        assert!(output.contains("%xtra:\thola"), "Output: {output}");
    }

    #[test]
    fn test_inject_translation_replaces_existing() {
        let chat_text = include_str!("../../../test-fixtures/eng_hello_with_xtra.cha");
        let mut chat = parse_chat(chat_text);

        let output_before = chat.to_chat_string();
        assert!(
            output_before.contains("old translation"),
            "Before: {output_before}"
        );

        let utt = get_utterance_mut(&mut chat, 0);
        inject_translation(utt, "new translation").unwrap();

        let output = chat.to_chat_string();
        assert!(output.contains("new translation"), "After: {output}");
        assert!(
            !output.contains("old translation"),
            "Old should be gone: {output}"
        );
    }

    #[test]
    fn test_apply_translate_results() {
        let chat_text = include_str!("../../../test-fixtures/eng_hello_goodbye.cha");
        let mut chat = parse_chat(chat_text);

        let payloads = collect_translate_payloads(&chat);
        assert_eq!(payloads.len(), 2);
        let line_idx_0 = payloads[0].0;
        let line_idx_1 = payloads[1].0;

        let mut results = HashMap::new();
        results.insert(line_idx_0, "hola".to_string());
        results.insert(line_idx_1, "adiós".to_string());

        apply_translate_results(&mut chat, &results);

        let output = chat.to_chat_string();
        assert!(output.contains("%xtra:\thola"), "Output: {output}");
        assert!(output.contains("%xtra:\tadiós"), "Output: {output}");
    }

    #[test]
    fn test_extract_translation_strings() {
        let chat_text = include_str!("../../../test-fixtures/eng_hello_female.cha");
        let mut chat = parse_chat(chat_text);

        let payloads = collect_translate_payloads(&chat);
        let line_idx = payloads[0].0;
        let utt = get_utterance_mut(&mut chat, 0);
        inject_translation(utt, "hola").unwrap();

        let entries = extract_translation_strings(&chat, &[line_idx]);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].line_idx, line_idx);
        assert_eq!(entries[0].translation, "hola");
    }

    #[test]
    fn test_inject_empty_translation_is_noop() {
        let chat_text = include_str!("../../../test-fixtures/eng_hello_female.cha");
        let mut chat = parse_chat(chat_text);
        let output_before = chat.to_chat_string();

        let utt = get_utterance_mut(&mut chat, 0);
        inject_translation(utt, "").unwrap();

        let output_after = chat.to_chat_string();
        assert_eq!(output_before, output_after);
    }

    #[test]
    fn snapshot_translate_batch_item() {
        let item = TranslateBatchItem {
            text: "I eat cookies".into(),
        };
        insta::assert_json_snapshot!(item, @r#"
        {
          "text": "I eat cookies"
        }
        "#);
    }

    #[test]
    fn snapshot_translate_response() {
        let resp = TranslateResponse {
            translation: "Yo como galletas".into(),
        };
        insta::assert_json_snapshot!(resp, @r#"
        {
          "translation": "Yo como galletas"
        }
        "#);
    }

    #[test]
    fn test_preprocess_chinese() {
        let lang = talkbank_model::model::LanguageCode::new("zho");
        assert_eq!(preprocess_for_translate("你 好 。", &lang), "你好\u{3002}");
    }

    #[test]
    fn test_preprocess_cantonese() {
        let lang = talkbank_model::model::LanguageCode::new("yue");
        assert_eq!(preprocess_for_translate("你 好.", &lang), "你好\u{3002}");
    }

    #[test]
    fn test_preprocess_non_chinese() {
        let lang = talkbank_model::model::LanguageCode::new("eng");
        assert_eq!(
            preprocess_for_translate("hello world", &lang),
            "hello world"
        );
    }

    #[test]
    fn test_postprocess_basic() {
        let raw = "Hello\u{3002} World\u{2019}s";
        let punct = vec![".", "?"];
        let result = postprocess_translation(raw, &punct);
        assert_eq!(result, "Hello . World's");
    }

    #[test]
    fn test_postprocess_zero_width_space() {
        let raw = "hello\u{200b}world";
        let result = postprocess_translation(raw, &[]);
        assert_eq!(result, "helloworld");
    }
}
