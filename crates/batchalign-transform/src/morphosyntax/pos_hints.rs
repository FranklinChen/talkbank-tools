//! Transcriber-supplied `$POS` hint application and the Stanza
//! language-support gate.
//!
//! [`collect_pos_hints`] captures typed `$POS` evidence before retokenization,
//! and [`apply_pos_hint_evidence`] uses it to override Stanza's UPOS in the
//! corresponding `%mor` item when the two disagree. The bookkeeping is
//! summarized in [`HintOutcome`].
//! [`is_stanza_supported`] / [`supported_iso3_codes`] front the static
//! list of ISO-639-3 codes that Stanza actually has a pipeline for.

use std::collections::HashSet;
use std::sync::LazyLock;

use talkbank_model::alignment::helpers::visit_mor_positions;
use talkbank_model::alignment::{MainWordIndex, MorItemIndex};
use talkbank_model::model::{Line, Utterance};

use super::ud_types::UniversalPos;

/// Counts of hint-application outcomes across one `apply_pos_hints` run.
#[derive(Debug, Default, Clone, Copy)]
pub struct HintOutcome {
    /// Total `$POS`-annotated words encountered.
    pub hints_considered: usize,
    /// Hints where Stanza's POS already matched the transcriber's hint.
    pub hints_agreed: usize,
    /// Hints where Stanza's POS was replaced with the transcriber's hint.
    pub hints_overridden: usize,
    /// CLAN tags with no UD UPOS mapping.
    pub hints_unmapped: usize,
    /// Hints on utterances with no `%mor` tier to modify.
    pub hints_skipped_no_mor: usize,
}

/// Typed `$POS` observations captured before a main-tier rewrite.
#[derive(Debug, Default)]
pub struct PosHintEvidence {
    by_line: Vec<PosHintLineEvidence>,
}

#[derive(Debug)]
struct PosHintLineEvidence {
    line_idx: usize,
    utterance_ordinal: usize,
    hints: Vec<PosHint>,
}

/// A hint attached to a position admitted by Chatter's morphology traversal.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PosHint {
    source: MainWordIndex,
    clan_tag: String,
}

impl PosHint {
    fn target_item(&self, trace: Option<&super::RetokenizationInfo>) -> Option<MorItemIndex> {
        match trace {
            Some(trace) => trace
                .mapping
                .get(self.source.as_usize())
                .and_then(|items| items.first())
                .copied()
                .map(MorItemIndex::new),
            None => Some(MorItemIndex::new(self.source.as_usize())),
        }
    }
}

/// Capture `$POS` hints before any retokenization can rewrite main-tier words.
pub fn collect_pos_hints(chat_file: &talkbank_model::model::ChatFile) -> PosHintEvidence {
    let mut by_line = Vec::new();
    let mut utterance_ordinal = 0usize;
    for (line_idx, line) in chat_file.lines.iter().enumerate() {
        let Line::Utterance(utt) = line else {
            continue;
        };
        let hints = utterance_pos_hints(utt);
        if !hints.is_empty() {
            by_line.push(PosHintLineEvidence {
                line_idx,
                utterance_ordinal,
                hints,
            });
        }
        utterance_ordinal += 1;
    }
    PosHintEvidence { by_line }
}

/// One owner for the typed hint inputs to application and reuse compatibility.
pub(crate) fn utterance_pos_hints(utt: &Utterance) -> Vec<PosHint> {
    let mut hints = Vec::new();
    visit_mor_positions(&utt.main.content.content, &mut |position| {
        if let Some(word) = position.word()
            && let Some(pos) = &word.part_of_speech
        {
            hints.push(PosHint {
                source: position.index(),
                clan_tag: pos.to_string(),
            });
        }
    });
    hints
}

/// Apply previously captured hint evidence to injected `%mor` tiers.
pub fn apply_pos_hint_evidence(
    chat_file: &mut talkbank_model::model::ChatFile,
    evidence: PosHintEvidence,
    retokenizations: &[super::RetokenizationInfo],
) -> HintOutcome {
    use talkbank_model::model::dependent_tier::mor::{MorTier, clan_to_ud_upos};
    use talkbank_model::model::{DependentTier, Utterance};

    fn mor_tier_mut(utt: &mut Utterance) -> Option<&mut MorTier> {
        utt.dependent_tiers
            .iter_mut()
            .find_map(|t| match &mut t.tier {
                DependentTier::Mor(m) => Some(m),
                _ => None,
            })
    }

    enum HintResolution {
        Agreed,
        Overridden,
        Unmapped,
        NoMorItem,
    }

    fn resolve_hint(clan_tag: &str, mor: &mut MorTier, item_idx: MorItemIndex) -> HintResolution {
        let Some(upos_name) = clan_to_ud_upos(clan_tag) else {
            return HintResolution::Unmapped;
        };
        let Some(hinted) = UniversalPos::from_pos_name(upos_name) else {
            return HintResolution::Unmapped;
        };
        let Some(mor_item) = mor.items_mut().get_mut(item_idx.as_usize()) else {
            return HintResolution::NoMorItem;
        };

        let stanza = UniversalPos::from_pos_name(mor_item.main.pos.as_ref());
        if stanza == Some(hinted) {
            return HintResolution::Agreed;
        }
        mor_item.override_main_pos(hinted.to_chat_pos_name());
        HintResolution::Overridden
    }

    let mut outcome = HintOutcome::default();

    for line_evidence in evidence.by_line {
        let retokenization = retokenizations
            .iter()
            .find(|trace| trace.utterance_ordinal == line_evidence.utterance_ordinal);
        let Some(line) = chat_file
            .lines
            .as_mut_slice()
            .get_mut(line_evidence.line_idx)
        else {
            continue;
        };
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };
        let Some(mor) = mor_tier_mut(utt) else {
            outcome.hints_considered += line_evidence.hints.len();
            outcome.hints_skipped_no_mor += line_evidence.hints.len();
            continue;
        };

        for hint in line_evidence.hints {
            outcome.hints_considered += 1;
            let Some(target_idx) = hint.target_item(retokenization) else {
                outcome.hints_skipped_no_mor += 1;
                continue;
            };
            match resolve_hint(&hint.clan_tag, mor, target_idx) {
                HintResolution::Agreed => outcome.hints_agreed += 1,
                HintResolution::Overridden => outcome.hints_overridden += 1,
                HintResolution::Unmapped => outcome.hints_unmapped += 1,
                HintResolution::NoMorItem => outcome.hints_skipped_no_mor += 1,
            }
        }
    }

    outcome
}

/// Capture and immediately apply hints when the main tier will not be mutated
/// between the two operations.
pub fn apply_pos_hints(chat_file: &mut talkbank_model::model::ChatFile) -> HintOutcome {
    let evidence = collect_pos_hints(chat_file);
    apply_pos_hint_evidence(chat_file, evidence, &[])
}

/// ISO 639-3 codes that have a known Stanza pipeline.
static SUPPORTED_STANZA_CODES: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "eng", "spa", "fra", "deu", "ita", "por", "nld", "cat", "glg", "dan", "swe", "nor", "fin",
        "est", "lav", "lit", "isl", "pol", "ces", "ron", "hun", "bul", "hrv", "slk", "slv", "ukr",
        "rus", "ell", "cym", "gle", "gla", "eus", "mlt", "ara", "heb", "fas", "hin", "urd", "tur",
        "tam", "tel", "tha", "vie", "ind", "zho", "cmn", "yue", "jpn", "kor", "kat", "hye", "afr",
        "lat",
    ]
    .into_iter()
    .collect()
});

/// Check whether a language code is supported by the Stanza worker.
pub fn is_stanza_supported(lang: &talkbank_model::model::LanguageCode) -> bool {
    SUPPORTED_STANZA_CODES.contains(lang.as_ref())
}

/// Sorted list of ISO-639-3 codes the Rust gate considers Stanza-supported.
pub fn supported_iso3_codes() -> &'static [&'static str] {
    static SORTED: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
        let mut v: Vec<&'static str> = SUPPORTED_STANZA_CODES.iter().copied().collect();
        v.sort_unstable();
        v
    });
    &SORTED
}
