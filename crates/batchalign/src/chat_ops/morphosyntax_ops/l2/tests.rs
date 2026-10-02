//! Tests for the L2 code-switching morphotag module.

use super::*;

use crate::chat_ops::nlp::UniversalPos;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ud(s: &str) -> UdDeprel {
    UdDeprel::new(s)
}

// ---------------------------------------------------------------------------
// Deprel → POS constraint tests
// ---------------------------------------------------------------------------

#[test]
fn deprel_det_constrains_to_det() {
    assert_eq!(
        deprel_to_pos_constraint(&ud("det")),
        PosConstraint::Exact(UniversalPos::Det)
    );
}

#[test]
fn deprel_amod_constrains_to_adj() {
    assert_eq!(
        deprel_to_pos_constraint(&ud("amod")),
        PosConstraint::Exact(UniversalPos::Adj)
    );
}

#[test]
fn deprel_advmod_constrains_to_adv() {
    assert_eq!(
        deprel_to_pos_constraint(&ud("advmod")),
        PosConstraint::Exact(UniversalPos::Adv)
    );
}

#[test]
fn deprel_case_constrains_to_adp() {
    assert_eq!(
        deprel_to_pos_constraint(&ud("case")),
        PosConstraint::Exact(UniversalPos::Adp)
    );
}

#[test]
fn deprel_obj_constrains_to_noun_pron_propn() {
    assert_eq!(
        deprel_to_pos_constraint(&ud("obj")),
        PosConstraint::OneOf(vec![
            UniversalPos::Noun,
            UniversalPos::Pron,
            UniversalPos::Propn
        ])
    );
}

#[test]
fn deprel_nsubj_constrains_to_noun_pron_propn() {
    assert_eq!(
        deprel_to_pos_constraint(&ud("nsubj")),
        PosConstraint::OneOf(vec![
            UniversalPos::Noun,
            UniversalPos::Pron,
            UniversalPos::Propn
        ])
    );
}

#[test]
fn deprel_root_constrains_to_verb_noun_adj() {
    let c = deprel_to_pos_constraint(&ud("root"));
    assert!(c.contains(&UniversalPos::Verb));
    assert!(c.contains(&UniversalPos::Noun));
    assert!(c.contains(&UniversalPos::Adj));
    assert!(!c.contains(&UniversalPos::Adv));
}

#[test]
fn deprel_flat_is_unconstrained() {
    let c = deprel_to_pos_constraint(&ud("flat"));
    assert!(c.contains(&UniversalPos::Propn));
    assert!(c.contains(&UniversalPos::Noun));
    assert!(c.contains(&UniversalPos::Adj));
    assert!(c.contains(&UniversalPos::Adv));
    assert!(c.contains(&UniversalPos::Det));
    assert!(c.contains(&UniversalPos::Verb));
    assert!(c.contains(&UniversalPos::Pron));
}

#[test]
fn deprel_subtype_stripped() {
    assert_eq!(
        deprel_to_pos_constraint(&ud("obl:arg")),
        deprel_to_pos_constraint(&ud("obl"))
    );
}

#[test]
fn unknown_deprel_unconstrained() {
    assert_eq!(
        deprel_to_pos_constraint(&ud("xyzzy")),
        PosConstraint::Unconstrained
    );
}

#[test]
fn constraint_contains_exact() {
    let c = PosConstraint::Exact(UniversalPos::Adv);
    assert!(c.contains(&UniversalPos::Adv));
    assert!(!c.contains(&UniversalPos::Noun));
}

#[test]
fn constraint_contains_unconstrained() {
    let c = PosConstraint::Unconstrained;
    assert!(c.contains(&UniversalPos::Verb));
    assert!(c.contains(&UniversalPos::Adv));
}

// ---------------------------------------------------------------------------
// GRA deprel inference tests
// ---------------------------------------------------------------------------

#[test]
fn infer_advmod_when_adv_head_adj() {
    assert_eq!(
        infer_deprel_from_pos(UniversalPos::Adv, Some(UniversalPos::Adj), false),
        Some(UdDeprel::new("advmod"))
    );
}

#[test]
fn infer_advmod_when_adv_head_verb() {
    assert_eq!(
        infer_deprel_from_pos(UniversalPos::Adv, Some(UniversalPos::Verb), false),
        Some(UdDeprel::new("advmod"))
    );
}

#[test]
fn infer_amod_when_adj_head_noun() {
    assert_eq!(
        infer_deprel_from_pos(UniversalPos::Adj, Some(UniversalPos::Noun), false),
        Some(UdDeprel::new("amod"))
    );
}

#[test]
fn infer_det_when_det_head_noun() {
    assert_eq!(
        infer_deprel_from_pos(UniversalPos::Det, Some(UniversalPos::Noun), false),
        Some(UdDeprel::new("det"))
    );
}

#[test]
fn infer_obj_when_noun_head_verb_no_case() {
    assert_eq!(
        infer_deprel_from_pos(UniversalPos::Noun, Some(UniversalPos::Verb), false),
        Some(UdDeprel::new("obj"))
    );
}

#[test]
fn infer_obl_when_noun_head_verb_with_case() {
    assert_eq!(
        infer_deprel_from_pos(UniversalPos::Noun, Some(UniversalPos::Verb), true),
        Some(UdDeprel::new("obl"))
    );
}

#[test]
fn infer_nmod_when_noun_head_noun() {
    assert_eq!(
        infer_deprel_from_pos(UniversalPos::Noun, Some(UniversalPos::Noun), false),
        Some(UdDeprel::new("nmod"))
    );
}

#[test]
fn no_inference_for_uncommon_combination() {
    assert_eq!(
        infer_deprel_from_pos(UniversalPos::Verb, Some(UniversalPos::Noun), false),
        None
    );
}

#[test]
fn no_inference_when_head_unknown() {
    assert_eq!(infer_deprel_from_pos(UniversalPos::Adv, None, false), None);
}
