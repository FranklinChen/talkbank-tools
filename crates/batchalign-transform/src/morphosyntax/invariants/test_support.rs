//! Shared helpers for the invariant modules' tests: a word as the analysis
//! returns it, with an explicit id, a punctuation token, and the production seam (dispatcher
//! plus mapping) rendered to `%mor` strings.

use crate::morphosyntax::evidence::UtteranceEvidence;
use crate::morphosyntax::{
    MappingContext, UdId, UdPunctable, UdSentence, UdWord, UdWordAnalysis, UniversalPos,
    apply_grammatical_invariants, map_ud_sentence,
};

/// A word with the given UD id, as the analysis returns it.
pub(crate) fn word(
    id: usize,
    text: &str,
    lemma: &str,
    upos: UniversalPos,
    feats: Option<&str>,
    head: usize,
    deprel: &str,
) -> UdWord {
    UdWord::from(UdWordAnalysis {
        id: UdId::Single(id),
        text: text.to_string(),
        lemma: lemma.to_string(),
        upos: UdPunctable::Value(upos),
        xpos: None,
        feats: feats.map(str::to_string),
        head,
        deprel: deprel.to_string(),
        deps: None,
        misc: None,
    })
}

/// A punctuation token with the given UD id.
pub(crate) fn punct(id: usize, text: &str, head: usize) -> UdWord {
    let mut w = word(id, text, text, UniversalPos::Punct, None, head, "punct");
    w.upos = UdPunctable::Punct(text.to_string());
    w
}

/// The English mapping context.
pub(crate) fn english() -> MappingContext {
    MappingContext {
        lang: talkbank_model::model::LanguageCode::new("eng").expect("valid language code"),
    }
}

/// Run the production invariant dispatcher and mapping for English and
/// render each `%mor` item.
pub(crate) fn mor_texts(sentence: &UdSentence, evidence: UtteranceEvidence) -> Vec<String> {
    let ctx = english();
    let rewritten = apply_grammatical_invariants(sentence, &ctx, || evidence);
    let (mors, _gras) = map_ud_sentence(&rewritten, &ctx).expect("mapping succeeds");
    mors.iter()
        .map(|m| {
            let mut out = String::new();
            m.write_chat(&mut out).expect("mor renders");
            out
        })
        .collect()
}
