//! Structural merge: combine primary structural info with secondary lexical output.
//!
//! The core POS resolution algorithm with priority-ordered rules:
//! 1. Copula predicate check
//! 2. Constraint agreement
//! 3. Closed-class function word override
//! 4. Content noun (NOUN/PROPN) override
//! 5. Primary POS structural fallback
//! 6. Best-guess from constraint

use talkbank_model::model::LanguageCode;
use talkbank_model::model::dependent_tier::mor::{Mor, PosCategory};

use super::deprel::{
    UdDeprel, deprel_to_pos_constraint, infer_deprel_from_pos, refine_with_dependents,
};
use super::extract::PrimaryStructuralInfo;
use crate::morphosyntax::{UdId, UdSentence, UdWord, UniversalPos};

/// Sentence-level context from the secondary model for a single `@s`
/// word being merged.
///
/// The individual `Mor` passed to [`merge_primary_secondary`] carries
/// the secondary's UPOS and lemma for one word, but not the structural
/// evidence that identifies phrasal-verb constructions (the
/// `compound:prt` relation). Callers thread this context alongside the
/// `Mor` so [`resolve_merged_pos_with_context`] can detect phrasal-verb
/// heads and particles and override the POS priority chain accordingly.
///
/// `word_position` is the 0-based index into `sentence.words` of the
/// word whose POS is currently being resolved. It is NOT the 1-based
/// UD `id`; see [`current_ud_id`](Self::current_ud_id).
#[derive(Debug, Clone, Copy)]
pub struct SecondaryUdContext<'a> {
    /// The full UD sentence produced by the secondary model for the
    /// contiguous `@s` span.
    pub sentence: &'a UdSentence,
    /// Position of the current word within `sentence.words`.
    pub word_position: usize,
}

impl<'a> SecondaryUdContext<'a> {
    /// Returns the `UdWord` at `word_position`, or `None` if the
    /// position is out of range.
    pub fn current_word(&self) -> Option<&'a UdWord> {
        self.sentence.words.get(self.word_position)
    }

    /// 1-based UD id of the current word (for head/child matching).
    ///
    /// Returns `None` for Range or Decimal ids (MWT parent tokens /
    /// empty nodes), which never participate in phrasal-verb relations.
    pub fn current_ud_id(&self) -> Option<usize> {
        match self.current_word()?.id {
            UdId::Single(n) => Some(n),
            _ => None,
        }
    }

    /// Whether the current word is the head of a `compound:prt`
    /// relation (i.e., has a dependent whose deprel is `compound:prt`
    /// pointing back at this word).
    ///
    /// The `compound:prt` relation is Stanza's signal for a phrasal
    /// verb (`wake up`, `give up`, `figure out`). When it appears, the
    /// head is the verb and the dependent is the particle.
    pub fn is_phrasal_verb_head(&self) -> bool {
        let Some(my_id) = self.current_ud_id() else {
            return false;
        };
        self.sentence
            .words
            .iter()
            .any(|w| w.head == my_id && is_compound_prt(&w.deprel))
    }

    /// Whether the current word is itself a `compound:prt` particle.
    pub fn is_phrasal_verb_particle(&self) -> bool {
        self.current_word()
            .map(|w| is_compound_prt(&w.deprel))
            .unwrap_or(false)
    }
}

/// Deprel-base comparison against the UD phrasal-verb particle
/// relation. Matches `compound:prt` exactly; the only other
/// `compound:*` subtype we have seen in Stanza output is `compound:svc`
/// (serial verb construction), which has a different semantics and
/// must not trigger phrasal-verb promotion.
fn is_compound_prt(deprel: &str) -> bool {
    deprel == "compound:prt"
}

/// Result of merging primary structural info with secondary model output.
#[derive(Debug, Clone)]
pub struct MergedL2Morphology {
    /// Pre-mapped CHAT MOR item with resolved POS override applied.
    ///
    /// Produced by `map_ud_sentence` (which handles MWT Range tokens for
    /// contractions like `it's` → `pron|it~aux|be`) and then POS-overridden
    /// by the merge algorithm.
    pub mor: Mor,
    /// Corrected deprel when the primary model's deprel was unreliable.
    /// `None` means keep the primary deprel as-is.
    pub corrected_deprel: Option<UdDeprel>,
}

/// Resolve the merged POS from primary structural info and a secondary
/// UPOS, without sentence-level context. Thin wrapper over
/// [`resolve_merged_pos_with_context`]; see that function for the full
/// priority chain.
pub fn resolve_merged_pos(
    primary: &PrimaryStructuralInfo,
    secondary_upos: Option<UniversalPos>,
) -> UniversalPos {
    resolve_merged_pos_with_context(primary, secondary_upos, None)
}

/// Resolve the merged POS with optional secondary UD sentence context.
///
/// POS resolution priority for @s words:
///
/// 0. **Phrasal verb** (context required): a word with `compound:prt`
///    deprel is the particle → `Part`; a word whose UPOS is Verb and
///    which is the head of a `compound:prt` relation stays `Verb`
///    even if the primary constraint would reject it.
/// 1. **Copula predicate**: if `cop` dependent, reject VERB → NOUN/ADJ
/// 2. **Agreement**: secondary POS matches constraint → use it
/// 3. **Function word**: secondary is closed-class → trust it
/// 4. **Content noun**: secondary is NOUN/PROPN → trust it
/// 5. **Structural fallback**: primary POS matches constraint → use it
/// 6. **Best guess**: constraint's most likely POS
///
/// Priority 0 exists because the sentence-level evidence of a verb +
/// particle construction (`wake up`, `give up`, `figure out`) is more
/// reliable than either the primary's deprel constraint (which rejects
/// VERB when the primary parser tagged a foreign word as `advmod`) or
/// Priority 3's blind trust of any closed-class POS (which would lock
/// in `adp|up` instead of `part|up`).
pub fn resolve_merged_pos_with_context(
    primary: &PrimaryStructuralInfo,
    secondary_upos: Option<UniversalPos>,
    secondary_context: Option<&SecondaryUdContext<'_>>,
) -> UniversalPos {
    // Priority 0: phrasal-verb structural recognition. Runs first because
    // Stanza's compound:prt analysis is cross-linguistically reliable
    // for true verb + particle constructions.
    if let Some(ctx) = secondary_context {
        if ctx.is_phrasal_verb_particle() {
            return UniversalPos::Part;
        }
        if ctx.is_phrasal_verb_head() && secondary_upos == Some(UniversalPos::Verb) {
            return UniversalPos::Verb;
        }
    }

    let base_constraint = deprel_to_pos_constraint(&primary.deprel);
    let constraint = refine_with_dependents(&base_constraint, &primary.dependent_deprels);

    let has_copula = primary.dependent_deprels.iter().any(|d| d.base() == "cop");

    if let Some(sec_pos) = secondary_upos {
        // Priority 1: copula predicate — reject VERB
        if has_copula && sec_pos == UniversalPos::Verb {
            return if primary.upos == Some(UniversalPos::Noun)
                || primary.upos == Some(UniversalPos::Propn)
            {
                UniversalPos::Noun
            } else {
                UniversalPos::Adj
            };
        }

        // Priority 2: secondary agrees with structural constraint
        if constraint.contains(&sec_pos) {
            return sec_pos;
        }

        // Priority 3: closed-class function words are unambiguous
        if is_closed_class(sec_pos) {
            return sec_pos;
        }

        // Priority 4: NOUN/PROPN from the secondary model overrides wrong deprel
        if sec_pos == UniversalPos::Noun || sec_pos == UniversalPos::Propn {
            return sec_pos;
        }

        // Priority 5: primary POS within constraint
        if let Some(pri_pos) = primary.upos
            && constraint.contains(&pri_pos)
        {
            return pri_pos;
        }

        // Priority 6: best guess from constraint, or secondary as last resort
        constraint.most_likely().unwrap_or(sec_pos)
    } else if let Some(pri_pos) = primary.upos {
        if constraint.contains(&pri_pos) {
            pri_pos
        } else {
            constraint.most_likely().unwrap_or(pri_pos)
        }
    } else {
        constraint.most_likely().unwrap_or(UniversalPos::Noun)
    }
}

/// Whether a UPOS tag is a closed-class (function word) category.
fn is_closed_class(upos: UniversalPos) -> bool {
    matches!(
        upos,
        UniversalPos::Det
            | UniversalPos::Adp
            | UniversalPos::Sconj
            | UniversalPos::Cconj
            | UniversalPos::Aux
            | UniversalPos::Part
            | UniversalPos::Pron
    )
}

/// Merge primary structural info with a secondary `Mor` item, without
/// sentence-level context. Thin wrapper over
/// [`merge_primary_secondary_with_context`].
pub fn merge_primary_secondary(
    primary: &PrimaryStructuralInfo,
    secondary_mor: Mor,
    secondary_lang: &LanguageCode,
) -> MergedL2Morphology {
    merge_primary_secondary_with_context(primary, secondary_mor, secondary_lang, None)
}

/// Merge primary structural info with a secondary `Mor` item, with
/// optional secondary UD sentence context for phrasal-verb recognition.
///
/// The `Mor` is pre-mapped from the secondary model's UD response via
/// `map_ud_sentence` (which handles MWT Range tokens for contractions).
/// This function resolves POS via the priority algorithm, overrides the
/// POS in the `Mor`, and computes deprel correction.
///
/// When `secondary_context` is supplied and identifies the current word
/// as a phrasal-verb particle, the `corrected_deprel` is set to
/// `compound:prt` so the CHAT %gra tier reflects the verb-particle
/// structure.
pub fn merge_primary_secondary_with_context(
    primary: &PrimaryStructuralInfo,
    mut secondary_mor: Mor,
    secondary_lang: &LanguageCode,
    secondary_context: Option<&SecondaryUdContext<'_>>,
) -> MergedL2Morphology {
    let _ = secondary_lang; // reserved for future language-specific overrides

    // Extract the secondary model's UPOS from the Mor's POS category name.
    let secondary_upos = UniversalPos::from_pos_name(secondary_mor.main.pos.as_str());

    let resolved_pos = resolve_merged_pos_with_context(primary, secondary_upos, secondary_context);

    // Override POS in the Mor with the structurally-resolved POS.
    secondary_mor.main.pos = PosCategory::new(resolved_pos.to_chat_pos_name());

    // Phrasal-verb particles have a well-defined UD deprel that the
    // secondary sentence already reports — carry it through directly
    // so the CHAT %gra tier matches the verb-particle structure.
    if let Some(ctx) = secondary_context
        && ctx.is_phrasal_verb_particle()
    {
        return MergedL2Morphology {
            mor: secondary_mor,
            corrected_deprel: Some(UdDeprel::new("compound:prt")),
        };
    }

    // Determine if GRA deprel needs correction.
    let primary_constraint = deprel_to_pos_constraint(&primary.deprel);
    let needs_correction =
        primary.deprel.base() == "flat" || !primary_constraint.contains(&resolved_pos);

    let corrected_deprel = if needs_correction {
        infer_deprel_from_pos(
            resolved_pos,
            primary.head_upos,
            primary.has_case_dependent(),
        )
    } else {
        None
    };

    MergedL2Morphology {
        mor: secondary_mor,
        corrected_deprel,
    }
}
