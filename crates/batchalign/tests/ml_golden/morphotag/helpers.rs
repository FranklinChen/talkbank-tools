//! Morphotag-only test helpers. The ones every golden command shares
//! (`parse_output`, `has_mor_tier`, `find_mor_line_for`) are in
//! `crate::ml_golden::golden::helpers`.

use batchalign::chat_ops::{ChatFile, DependentTier};

pub(super) fn repeated_chat(lang: &str, speaker: &str, stem: &str, utterances: usize) -> String {
    let mut chat = format!(
        "@UTF8\n@Begin\n@Languages:\t{lang}\n@Participants:\t{speaker} Participant\n@ID:\t{lang}|test|{speaker}|||||Participant|||\n"
    );
    for i in 0..utterances {
        chat.push_str(&format!("*{speaker}:\t{stem} number {i} today .\n"));
    }
    chat.push_str("@End\n");
    chat
}

pub(super) fn count_mor_lines(chat: &str) -> usize {
    chat.lines()
        .filter(|line| line.starts_with("%mor:"))
        .count()
}

pub(super) fn minimal_chat(lang: &str, speaker: &str, utterance: &str) -> String {
    format!(
        "@UTF8\n@Begin\n@Languages:\t{lang}\n@Participants:\t{speaker} Participant\n@ID:\t{lang}|test|{speaker}|||||Participant|||\n*{speaker}:\t{utterance} .\n@End\n"
    )
}

pub(super) fn count_ast_mor_tiers(file: &ChatFile) -> usize {
    file.lines
        .iter()
        .filter(|line| {
            if let batchalign::chat_ops::Line::Utterance(utt) = line {
                utt.dependent_tiers
                    .iter()
                    .any(|t| matches!(t.tier, DependentTier::Mor(_)))
            } else {
                false
            }
        })
        .count()
}

/// Drop every line holding one of our provenance stamps, under the current
/// `[fc-ba3 ` name or the legacy `[ba3 ` name, so outputs can be compared
/// without their wall-clock timestamps.
pub(super) fn strip_provenance_stamps(chat: &str) -> String {
    chat.lines()
        .filter(|line| !line.contains("[fc-ba3 ") && !line.contains("[ba3 "))
        .collect::<Vec<_>>()
        .join("\n")
}
