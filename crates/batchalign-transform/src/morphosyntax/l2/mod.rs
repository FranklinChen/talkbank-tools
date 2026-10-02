//! L2 code-switching helpers for `@s`-marked words.

mod deprel;
mod extract;
mod merge;
#[cfg(test)]
pub(crate) mod pipeline_tests;
mod plan;
mod splice;

pub use crate::morphosyntax::alignment::{
    AlignedUd, AlignedWord, HeadTarget, SpanWordPosition, UdAlignment, UdAlignmentError,
    UdSentenceError, UdTokenIndex, UdTokens, WordSpace,
};
pub use deprel::{PosConstraint, UdDeprel, deprel_to_pos_constraint, infer_deprel_from_pos};
pub(crate) use extract::{ItemPlacement, RetokenizedItems};
pub use extract::{
    L2DeferredPosition, L2ExtractError, L2Extraction, PrimaryStructuralInfo, UnalignedL2Utterance,
};
pub use merge::{
    L2MergeError, MergedL2Span, ModelAssignedPos, PosSource, SecondaryUdContext,
    merge_planned_secondary_span,
};
pub use plan::{
    AsPlanned, ExternalRelation, L2Attachment, L2DispatchPlan, L2SpanPlan, RelationStage,
    plan_dispatch_spans,
};
pub use splice::{SpliceOutcome, splice_l2_into_chat};
