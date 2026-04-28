//! Compatibility re-exports for canonical L2 span helpers now defined in
//! `talkbank-transform`.

pub use talkbank_transform::morphosyntax::l2::{
    DispatchSpan, L2Span, group_deferred_into_dispatch_spans, group_l2_spans,
};
