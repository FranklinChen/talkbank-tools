//! Grammatical-invariant rewrites over typed UD sentences.
//!
//! This module corrects specific classes of Stanza misanalysis: where the
//! entire UD output violates a universal grammatical constraint
//! (`finite_verb_main_clause`), and where Stanza assigns a word a category the
//! CHILDES lexicon does not license for it (`lexicon_category`).

mod discourse_marker;
mod english_contractions;
mod finite_verb_main_clause;
mod lexicon_category;
#[cfg(test)]
pub(crate) mod test_support;

use crate::morphosyntax::evidence::UtteranceEvidence;
use crate::morphosyntax::{MappingContext, UdSentence, lang2, lexicon};

pub use discourse_marker::mark_isolated_communicators;
pub use english_contractions::expand_english_contractions;
pub use finite_verb_main_clause::rescue_english_copula_progressive;
pub use lexicon_category::constrain_to_lexicon;

/// Apply all language-appropriate grammatical-invariant rewrites to a UD
/// sentence. `evidence` produces the transcriber's per-word evidence for the
/// utterance; it is called only by a chain that uses it.
pub fn apply_grammatical_invariants(
    sentence: &UdSentence,
    ctx: &MappingContext,
    evidence: impl FnOnce() -> UtteranceEvidence,
) -> UdSentence {
    match lang2(ctx.lang.as_str()) {
        "en" => {
            // One clone for the whole chain: every rewrite takes ownership
            // and works in place.
            let verdicts = lexicon::embedded_english();
            // Tokens first: a contraction Stanza left whole becomes the
            // range it should have been, so every later rule sees the parts.
            let expanded = english_contractions::expand_english_contractions(sentence.clone());
            // Then what the transcriber wrote and Stanza never saw.
            let marked =
                discourse_marker::mark_isolated_communicators(expanded, verdicts, &evidence());
            // Then the lexicon's word category.
            let constrained = lexicon_category::constrain_to_lexicon(marked, verdicts);
            // Then clause structure: the finite-verb rescue may promote an
            // `-ing` word the lexicon knows as a noun to the clause's verb,
            // and the clause-level invariant outranks the word-level ones.
            finite_verb_main_clause::rescue_english_copula_progressive(constrained)
        }
        _ => sentence.clone(),
    }
}
