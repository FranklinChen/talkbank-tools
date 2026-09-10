//! Grammatical-invariant rewrites over typed UD sentences.
//!
//! This module corrects specific classes of Stanza misanalysis: where the
//! entire UD output violates a universal grammatical constraint
//! (`finite_verb_main_clause`), and where Stanza assigns a word a category the
//! CHILDES lexicon does not license for it (`lexicon_category`).

mod finite_verb_main_clause;
mod lexicon_category;

use crate::morphosyntax::{MappingContext, UdSentence, lang2, lexicon};

pub use finite_verb_main_clause::rescue_english_copula_progressive;
pub use lexicon_category::constrain_to_lexicon;

/// Apply all language-appropriate grammatical-invariant rewrites to a UD
/// sentence.
pub fn apply_grammatical_invariants(sentence: &UdSentence, ctx: &MappingContext) -> UdSentence {
    match lang2(ctx.lang.as_str()) {
        "en" => {
            // Word category first, clause structure second: the finite-verb
            // rescue may promote an `-ing` word the lexicon knows as a noun
            // to the clause's verb, and the clause-level invariant outranks
            // the word-level one.
            // One clone for the whole chain: both rewrites take ownership and
            // work in place.
            finite_verb_main_clause::rescue_english_copula_progressive(
                lexicon_category::constrain_to_lexicon(
                    sentence.clone(),
                    lexicon::embedded_english(),
                ),
            )
        }
        _ => sentence.clone(),
    }
}
