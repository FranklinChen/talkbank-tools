//! Morphosyntax payload extraction and result injection (pure Rust).

mod inject;
pub mod l2;
pub mod outcome;
mod payloads;
pub mod pos_hints;
pub mod preprocess;
pub mod stanza_languages;
pub mod stanza_raw;
#[cfg(test)]
mod tests;

pub use inject::*;
pub use payloads::*;
pub use pos_hints::{HintOutcome, apply_pos_hints};
pub use talkbank_transform::morphosyntax::{
    AlignmentWarning, BatchItemWithPosition, MorphosyntaxBatchItem, MultilingualPolicy, MwtDict,
    TokenizationMode,
};
