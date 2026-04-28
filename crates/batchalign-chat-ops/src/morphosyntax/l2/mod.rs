//! L2 code-switching morphotag support for `@s`-marked words.
//!
//! L2 dispatch is on by default (opt-out via `--no-l2-morphotag`). This
//! module provides the building blocks for routing @s words to secondary
//! language Stanza models and merging the results with the primary model's
//! structural analysis.
//!
//! ## Architecture
//!
//! The primary model (matrix language) provides cross-linguistically valid
//! structural information: dependency relations, head attachment, UPOS.
//! The secondary model (embedded language) provides language-specific lexical
//! information: lemma, morphological features, and validated POS.
//!
//! The merge algorithm combines both, with the secondary model's POS taking
//! priority (especially for NOUN and closed-class words, which the primary
//! model frequently misclassifies for foreign words).
//!
//! ## Modules
//!
//! - [`deprel`] — `UdDeprel` newtype, deprel→POS constraint mapping
//! - [`merge`] — POS resolution, structural merge algorithm
//! - [`extract`] — primary structural info extraction from UD responses
//! - [`spans`] — contiguous span grouping for secondary dispatch
//! - [`splice`] — splice merged morphology back into ChatFile

pub mod deprel;
pub mod extract;
pub mod merge;
pub mod spans;
pub mod splice;

#[cfg(test)]
mod tests;

// Re-export public types for convenience.
pub use deprel::{PosConstraint, UdDeprel, deprel_to_pos_constraint, refine_with_dependents};
pub use extract::{L2DeferredPosition, PrimaryStructuralInfo, extract_l2_deferred_positions};
pub use merge::{
    MergedL2Morphology, SecondaryUdContext, merge_primary_secondary,
    merge_primary_secondary_with_context, resolve_merged_pos, resolve_merged_pos_with_context,
};
pub use spans::{DispatchSpan, L2Span, group_deferred_into_dispatch_spans, group_l2_spans};
pub use splice::{SpliceOutcome, apply_l2_fallback, splice_l2_into_chat};
