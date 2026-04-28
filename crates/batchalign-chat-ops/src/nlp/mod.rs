//! NLP mapping/validation surface shared by language-specific morphosyntax adapters.
//!
//! This module is the typed boundary between external NLP engine output
//! (UD-like JSON from callbacks) and TalkBank-native `%mor`/`%gra` structures.

pub mod dep_rel;
pub mod invariants;
pub mod lang_en;
pub mod lang_fr;
pub mod lang_it;
pub mod lang_ja;
pub mod mapping;
mod mor_word;
mod types;
pub mod ud_feats;
pub mod validation;

pub use dep_rel::DepRel;
pub use invariants::apply_grammatical_invariants;
pub use mapping::{MappingContext, map_ud_sentence, map_ud_sentence_expanded};
pub use mor_word::map_ud_word_to_mor;
pub use types::{
    FaIndexedTiming, FaRawResponse, FaRawToken, UdId, UdPunctable, UdResponse, UdSentence, UdWord,
    UniversalPos,
};
pub use ud_feats::VerbForm;
pub use validation::sanitize_mor_text;
