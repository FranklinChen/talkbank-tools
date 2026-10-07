//! Interior-only evidence bounds a forced-alignment search, never a timing.
//!
//! An utterance whose first or last alignable word has no correspondence
//! common to every optimum cannot receive a hint: the unproved endpoint may
//! lie anywhere outside its proved interior words, and cropping to them
//! would cut real speech. It is still not unlocated. In a monotonic region
//! every one of its words was spoken after every proved word of the
//! utterances before it and before every proved word of the utterances after
//! it, and inside any retained non-overlap bullets around it. Those
//! neighbouring proved timings bound where all of its words, proved or not,
//! can be.
//!
//! That bound is published as an order-corridor search envelope: forced
//! alignment may search the utterance's words inside it, and the envelope
//! never becomes a bullet. It is distinct from an endpoint-proved timing in
//! type (`FaSearchEnvelope::OrderCorridor`, not `UtrTimingProposal`) and in
//! the review decision, which still records the incomplete boundary.
//!
//! Interleaved regions already derive their corridors from the joint
//! producer (`lexical::plan_interleaving`); this module covers the monotonic
//! dynamic-programming strategy only, whose order model makes the neighbours'
//! proved words a valid bound.

use super::UtrUtteranceInfo;
use super::evidence::{LocalRegionPlan, UtrUtteranceAlignmentEvidence};
use super::lexical::{AdmittedUtrWordMatch, UtrLexicalStream};
use super::search::FaSearchEnvelope;

/// Proved timings one utterance contributes as a neighbour: its admitted
/// correspondences (either strength), then its retained non-overlap bullet.
fn proved_matches(evidence: &UtrUtteranceAlignmentEvidence) -> Vec<&AdmittedUtrWordMatch> {
    match evidence {
        UtrUtteranceAlignmentEvidence::Matched {
            admitted_matches, ..
        } => admitted_matches.iter().collect(),
        UtrUtteranceAlignmentEvidence::InteriorOnly {
            admitted_matches, ..
        } => admitted_matches.iter().collect(),
        UtrUtteranceAlignmentEvidence::SelectedOnly { .. }
        | UtrUtteranceAlignmentEvidence::Refused { .. }
        | UtrUtteranceAlignmentEvidence::RetainedUnsearched { .. }
        | UtrUtteranceAlignmentEvidence::Unmatched { .. }
        | UtrUtteranceAlignmentEvidence::ExcludedMarkedOverlap { .. }
        | UtrUtteranceAlignmentEvidence::NoAlignableWords { .. } => Vec::new(),
    }
}

fn overlaps(info: &UtrUtteranceInfo) -> bool {
    info.has_lazy_overlap || info.has_ca_overlap
}

/// Give every untimed, unmarked interior-only utterance of a monotonic
/// region an order-corridor search envelope bounded by the neighbouring
/// proved timings, when those bounds contain its own proved words.
///
/// The bounds use each neighbour's outer edge (a preceding proved token's
/// start, a following one's end), as the interleaved corridor does, and a
/// retained bullet's inner edge (a preceding bullet's end, a following
/// bullet's start), as hint projection does. Both sides must be bounded. A
/// provider stream whose timing order disagrees with its token order grants
/// no corridor at all.
pub(super) fn bound_interior_only_searches(
    plan: &mut LocalRegionPlan<'_>,
    lexical: &UtrLexicalStream<'_>,
) {
    if lexical.ordered_search().is_none() {
        return;
    }
    let utterances = plan.utterances_mut();
    // The latest proved timing before each utterance, and the earliest after
    // it. `None` means no proved timing on that side: then there is no
    // bound, and no corridor (see below).
    let mut floors = Vec::with_capacity(utterances.len());
    let mut floor: Option<u64> = None;
    for planned in utterances.iter() {
        floors.push(floor);
        let mut raise = |bound: u64| {
            floor = Some(floor.map_or(bound, |current| current.max(bound)));
        };
        if !overlaps(planned.info)
            && let Some(timing) = planned.info.retained_timing
        {
            raise(timing.end_ms);
        }
        for matched in proved_matches(&planned.evidence) {
            raise(matched.token_timing().start_ms);
        }
    }
    let mut ceilings = vec![None::<u64>; utterances.len()];
    let mut ceiling: Option<u64> = None;
    for (ordinal, planned) in utterances.iter().enumerate().rev() {
        ceilings[ordinal] = ceiling;
        let mut lower = |bound: u64| {
            ceiling = Some(ceiling.map_or(bound, |current| current.min(bound)));
        };
        if !overlaps(planned.info)
            && let Some(timing) = planned.info.retained_timing
        {
            lower(timing.start_ms);
        }
        for matched in proved_matches(&planned.evidence) {
            lower(matched.token_timing().end_ms);
        }
    }
    for (ordinal, planned) in utterances.iter_mut().enumerate() {
        let info = planned.info;
        let UtrUtteranceAlignmentEvidence::InteriorOnly {
            admitted_matches, ..
        } = &planned.evidence
        else {
            continue;
        };
        if info.retained_timing.is_some() || overlaps(info) || planned.envelope.is_some() {
            continue;
        }
        let own_start = admitted_matches
            .iter()
            .map(|matched| matched.token_timing().start_ms)
            .min();
        let own_end = admitted_matches
            .iter()
            .map(|matched| matched.token_timing().end_ms)
            .max();
        let (Some(own_start), Some(own_end)) = (own_start, own_end) else {
            continue;
        };
        // Both sides must be bounded by a proved neighbour. An unbounded side
        // would make the corridor run to the recording's edge: an obligation
        // no wider window can discharge, where the unhinted path would still
        // try. Contradictory neighbours prove nothing about this utterance.
        // Either way it keeps forced alignment's ordinary unhinted path.
        let (Some(floor), Some(ceiling)) = (floors[ordinal], ceilings[ordinal]) else {
            continue;
        };
        if floor > own_start || ceiling < own_end {
            continue;
        }
        planned.envelope = Some(FaSearchEnvelope::order_corridor(
            info.words.clone(),
            floor,
            Some(ceiling),
        ));
    }
}
