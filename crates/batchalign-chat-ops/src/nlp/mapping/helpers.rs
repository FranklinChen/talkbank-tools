//! Compatibility re-exports for the canonical sentence-mapping helpers now
//! defined in `talkbank-transform`.

#[allow(unused_imports)]
pub use talkbank_transform::morphosyntax::{
    assemble_mors, normalize_deprel, provenance_for_ud_word,
};
