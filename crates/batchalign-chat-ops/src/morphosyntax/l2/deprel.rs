//! Compatibility re-exports for canonical L2 deprel helpers now defined in
//! `talkbank-transform`.

pub use talkbank_transform::morphosyntax::l2::{
    PosConstraint, UdDeprel, deprel_to_pos_constraint, infer_deprel_from_pos,
    refine_with_dependents,
};
