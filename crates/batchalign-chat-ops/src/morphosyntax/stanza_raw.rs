//! Compatibility re-export for the canonical raw Stanza parser helpers now
//! defined in `talkbank-transform`.

pub use talkbank_transform::morphosyntax::{
    StanzaParseError, StanzaWordDiagnostic, diagnose_parse_failure, is_bogus_lemma,
    parse_raw_stanza_output, validate_and_clean,
};

#[cfg(test)]
mod tests {
    use crate::nlp::{MappingContext, map_ud_sentence};
    use serde_json::json;

    use super::parse_raw_stanza_output;

    #[test]
    fn cantonese_6_words_produces_6_mors() {
        let raw = vec![json!([
            {"id": 1, "text": "呢", "lemma": "呢", "upos": "PART", "head": 2, "deprel": "case"},
            {"id": 2, "text": "度", "lemma": "度", "upos": "NUM", "head": 5, "deprel": "nmod"},
            {"id": 3, "text": "食飯", "lemma": "食飯", "upos": "VERB", "head": 4, "deprel": "compound"},
            {"id": 4, "text": "啦", "lemma": "啦", "upos": "NOUN", "head": 5, "deprel": "nmod"},
            {"id": 5, "text": "飯", "lemma": "飯", "upos": "NOUN", "head": 0, "deprel": "root"},
            {"id": 6, "text": "啦", "lemma": "啦", "upos": "NOUN", "head": 5, "deprel": "discourse:sp"}
        ])];

        let resp = parse_raw_stanza_output(&raw).unwrap();
        assert_eq!(resp.sentences.len(), 1);
        assert_eq!(resp.sentences[0].words.len(), 6);

        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("yue"),
        };
        let (mors, gras) = map_ud_sentence(&resp.sentences[0], &ctx).unwrap();
        assert_eq!(mors.len(), 6);
        assert_eq!(gras.len(), 7);
    }
}
