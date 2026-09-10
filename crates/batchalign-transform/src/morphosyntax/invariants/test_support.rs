//! Shared helpers for the invariant modules' tests: a synthetic UD word with
//! an explicit id, a punctuation token, and the production seam (dispatcher
//! plus mapping) rendered to `%mor` strings.

use crate::morphosyntax::evidence::UtteranceEvidence;
use crate::morphosyntax::{
    MappingContext, UdId, UdPunctable, UdSentence, UdWord, UniversalPos,
    apply_grammatical_invariants, map_ud_sentence,
};

/// A word with the given UD id.
pub(crate) fn word(
    id: usize,
    text: &str,
    lemma: &str,
    upos: UniversalPos,
    feats: Option<&str>,
    head: usize,
    deprel: &str,
) -> UdWord {
    let mut w = UdWord::synthetic(text, lemma, upos, feats, head, deprel);
    w.id = UdId::Single(id);
    w
}

/// A punctuation token with the given UD id.
pub(crate) fn punct(id: usize, text: &str, head: usize) -> UdWord {
    let mut w = UdWord::synthetic(text, text, UniversalPos::Punct, None, head, "punct");
    w.id = UdId::Single(id);
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
