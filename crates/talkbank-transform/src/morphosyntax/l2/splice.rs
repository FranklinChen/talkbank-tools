//! Splice merged L2 morphology back into a ChatFile.
//!
//! Overwrites `L2|xxx` MOR items with pre-mapped `Mor` items from the
//! structural merge algorithm, and optionally corrects GRA deprels.

use super::extract::L2DeferredPosition;
use super::merge::MergedL2Morphology;

/// Outcome of splicing L2 results into a `ChatFile`.
#[derive(Debug, Default)]
pub struct SpliceOutcome {
    /// Number of @s positions successfully spliced with real morphology.
    pub spliced: usize,
    /// Number of @s positions that fell back to L2|xxx (no secondary result).
    pub fallback: usize,
    /// Number of GRA deprels corrected.
    pub gra_upgraded: usize,
}

/// Overwrite `L2|xxx` MOR items with merged morphology.
///
/// Each `MergedL2Morphology` contains a fully-mapped `Mor` item (produced
/// by `map_ud_sentence` which handles MWT contractions, then POS-overridden
/// by the merge algorithm). The splice is a simple assignment.
///
/// This function must be called AFTER `inject_results` has set L2|xxx on
/// all @s positions.
pub fn splice_l2_into_chat(
    chat_file: &mut talkbank_model::model::ChatFile,
    deferred: &[L2DeferredPosition],
    merged_results: &[Option<MergedL2Morphology>],
) -> SpliceOutcome {
    use talkbank_model::model::DependentTier;
    use talkbank_model::model::Line;

    let mut outcome = SpliceOutcome::default();

    for (def, merged_opt) in deferred.iter().zip(merged_results.iter()) {
        let merged = match merged_opt {
            Some(m) => m,
            None => {
                outcome.fallback += 1;
                continue;
            }
        };

        let utt = match &mut chat_file.lines[def.line_idx] {
            Line::Utterance(u) => u,
            _ => {
                outcome.fallback += 1;
                continue;
            }
        };

        // Replace the MOR item with the pre-mapped Mor from the merge.
        let mor_tier = utt.dependent_tiers.iter_mut().find_map(|t| match t {
            DependentTier::Mor(m) => Some(m),
            _ => None,
        });

        if let Some(mor) = mor_tier {
            if let Some(mor_item) = mor.items.0.get_mut(def.word_idx) {
                *mor_item = merged.mor.clone();
                outcome.spliced += 1;
            } else {
                outcome.fallback += 1;
                continue;
            }
        } else {
            outcome.fallback += 1;
            continue;
        }

        // Correct GRA deprel if needed.
        if let Some(ref corrected) = merged.corrected_deprel {
            let gra_tier = utt.dependent_tiers.iter_mut().find_map(|t| match t {
                DependentTier::Gra(g) => Some(g),
                _ => None,
            });
            if let Some(gra) = gra_tier
                && let Some(gra_item) = gra.relations.0.get_mut(def.word_idx)
            {
                gra_item.relation = corrected.to_chat_gra();
                outcome.gra_upgraded += 1;
            }
        }
    }

    outcome
}

/// Apply L2|xxx fallback to deferred positions that have no merged result.
pub fn apply_l2_fallback(
    chat_file: &mut talkbank_model::model::ChatFile,
    deferred: &[L2DeferredPosition],
) {
    use talkbank_model::model::DependentTier;
    use talkbank_model::model::Line;
    use talkbank_model::model::dependent_tier::mor::{MorStem, PosCategory};

    for def in deferred {
        let utt = match &mut chat_file.lines[def.line_idx] {
            Line::Utterance(u) => u,
            _ => continue,
        };
        let mor_tier = utt.dependent_tiers.iter_mut().find_map(|t| match t {
            DependentTier::Mor(m) => Some(m),
            _ => None,
        });
        if let Some(mor) = mor_tier
            && let Some(mor_item) = mor.items.0.get_mut(def.word_idx)
        {
            mor_item.main.pos = PosCategory::new("L2");
            mor_item.main.lemma = MorStem::new("xxx");
            mor_item.main.features.clear();
        }
    }
}
