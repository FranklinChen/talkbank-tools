//! Compatibility re-exports for canonical morphosyntax injection helpers now
//! defined in `talkbank-transform`.

pub use talkbank_transform::morphosyntax::{
    InjectionResult, RetokenizationInfo, clear_morphosyntax, clear_morphosyntax_selective,
    inject_results, remove_empty_morphosyntax_placeholders, validate_mor_alignment,
};
