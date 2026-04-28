//! Compatibility re-export for the canonical decision-tier model now defined in
//! `talkbank-transform`.

pub use talkbank_transform::decisions::*;

impl From<&super::fa::repair::RepairDecision> for DecisionRecord {
    fn from(d: &super::fa::repair::RepairDecision) -> Self {
        Self {
            line_idx: d.line_idx,
            speaker: d.speaker.clone(),
            strategy: DecisionStrategy::Fa(d.strategy),
            reason: d.reason.clone(),
            needs_review: d.needs_review,
        }
    }
}
