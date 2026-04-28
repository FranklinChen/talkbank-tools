//! Compatibility re-exports for canonical L2 merge helpers now defined in
//! `talkbank-transform`.

pub use talkbank_transform::morphosyntax::l2::{
    MergedL2Morphology, SecondaryUdContext, merge_primary_secondary,
    merge_primary_secondary_with_context, resolve_merged_pos, resolve_merged_pos_with_context,
};
