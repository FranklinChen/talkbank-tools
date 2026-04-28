//! Retokenize an utterance's main tier to match Stanza's UD tokenization.
//!
//! When `retokenize=true`, Stanza uses its own tokenizer instead of Batchalign's
//! custom `tokenize_postprocessor`. This means Stanza may split or merge words
//! differently from the original CHAT transcript. For example:
//!   - Original: `["don't"]`  -> Stanza: `["do", "n't"]` (1:N split)
//!   - Original: `["gon", "na"]` -> Stanza: `["gonna"]` (N:1 merge)
//!
//! This module replaces the main tier's Word nodes with the Stanza tokenization
//! while preserving all non-word AST content (Groups, Separators, Events, etc.).
//!
//! # Related CHAT Manual Sections
//!
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Format>
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Headers>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Main_Tier>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Dependent_Tiers>

pub mod mapping;
mod parse_helpers;
#[cfg(test)]
mod tests;

pub use talkbank_transform::retokenize::retokenize_utterance;
