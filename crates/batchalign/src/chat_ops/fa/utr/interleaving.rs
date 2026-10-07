//! Read-only two-speaker lexical controls for the offline UTR evidence seam.
//!
//! These observations deliberately do not enter `UtrAlignmentPlan`, anchors,
//! grouping or CHAT output. The local relaxation examines adjacent different
//! speakers against the full retained ASR lexical stream; it does not reserve
//! words belonging to other turns. Production integration must establish those
//! source/timing bounds rather than treating this diagnostic as a global proof.

use batchalign_transform::dp_align::interleaving::{
    InterleavingRefusal, SpeakerChain, TwoSpeakerAlignment,
};
use serde::Serialize;
use talkbank_model::model::ChatFile;

use super::evidence::{UtrUtteranceOrdinal, UtrWordOrdinal};
use super::{
    AsrTimingToken, GlobalUtrParticipation, UtrMatchMode, UtrWordAddress, UtrWordMatch,
    collect_utr_utterance_info, lexical::UtrLexicalStream,
};

/// Adjacent different-speaker source pair, bound by the observation producer.
#[derive(Debug, Serialize)]
pub struct InterleavedUtrPair {
    first_utterance: UtrUtteranceOrdinal,
    second_utterance: UtrUtteranceOrdinal,
    observation: PairObservation,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum PairObservation {
    Complete {
        matched_words: usize,
        selected: Vec<UtrWordMatch>,
        words: Vec<WordObservation>,
    },
    CellBudgetExceeded,
}

#[derive(Debug, Serialize)]
struct WordObservation {
    word: UtrWordAddress,
    can_be_missing: bool,
    common: Option<UtrWordMatch>,
    candidates: Vec<UtrWordMatch>,
}

/// Observe a bounded partial-order alternative without inference or mutation.
///
/// All adjacent different-speaker pairs participating in the requested census
/// are included, not just pairs selected by a short-turn heuristic. Common
/// lexical matches here are local diagnostic facts, not acoustic speaker proof
/// and not producer-admitted UTR timing authority. Refused analysis is explicit.
pub fn observe(
    chat: &ChatFile,
    tokens: &[AsrTimingToken],
    mode: UtrMatchMode,
    participation: GlobalUtrParticipation,
) -> Vec<InterleavedUtrPair> {
    let census = collect_utr_utterance_info(chat);
    let lexical = UtrLexicalStream::from_tokens(tokens);
    let reference = lexical.texts();
    census.windows(2).enumerate().filter_map(|(index, pair)| {
        if pair[0].speaker == pair[1].speaker || pair.iter().any(|info| info.excluded_from(participation)) {
            return None;
        }
        let first_utterance = UtrUtteranceOrdinal(index);
        let second_utterance = UtrUtteranceOrdinal(index + 1);
        let observation = match TwoSpeakerAlignment::observe(
            &pair[0].words, &pair[1].words, &reference, mode.to_dp_match_mode(),
        ) {
            Err(InterleavingRefusal::CellBudgetExceeded) => PairObservation::CellBudgetExceeded,
            Ok(analysis) => {
                let address = |chain, word_index| UtrWordAddress {
                    utterance_index: match chain {
                        SpeakerChain::First => first_utterance,
                        SpeakerChain::Second => second_utterance,
                    },
                    word_index: UtrWordOrdinal(word_index),
                };
                let project = |matched: batchalign_transform::dp_align::interleaving::InterleavingMatch<'_>| {
                    lexical.matched_word(matched.reference_index(),
                        address(matched.chain(), matched.word_index()), matched.source_text())
                };
                let selected = analysis.selected().map(project).collect();
                let mut words = Vec::with_capacity(pair[0].words.len() + pair[1].words.len());
                for chain in [SpeakerChain::First, SpeakerChain::Second] {
                    words.extend(analysis.words(chain).enumerate().map(|(word_index, evidence)| WordObservation {
                        word: address(chain, word_index),
                        can_be_missing: evidence.can_be_missing(),
                        common: evidence.common().map(|common| project(common.matched())),
                        candidates: evidence.candidates().map(project).collect(),
                    }));
                }
                PairObservation::Complete { matched_words: analysis.matched_words(), selected, words }
            }
        };
        Some(InterleavedUtrPair { first_utterance, second_utterance, observation })
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use batchalign_transform::parse_source_with_parser;
    use talkbank_model::{
        NullErrorSink,
        model::{FileStem, TranscriptName},
    };
    use talkbank_parser::TreeSitterParser;

    fn admitted_pair(first: &str, second: &str, speaker: &str) -> ChatFile {
        let source = format!(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant, INV Investigator\n@ID:\teng|test|PAR|||||Participant|||\n@ID:\teng|test|INV|||||Investigator|||\n*PAR:\t{first} .\n*{speaker}:\t{second} .\n@End\n"
        );
        let parser = TreeSitterParser::new().expect("parser");
        parse_source_with_parser(&parser, &source)
            .admit(
                TranscriptName::Named(
                    FileStem::from_path(std::path::Path::new("input.cha")).expect("name"),
                ),
                &NullErrorSink,
            )
            .expect("valid source")
            .into_valid_file()
            .into_unchecked()
    }

    #[test]
    fn typed_source_projection_retains_early_and_late_candidates_without_timing_authority() {
        let chat = admitted_pair("who are paving the way for all of you", "all of", "INV");
        let before = serde_json::to_value(&chat).expect("source structure");
        let tokens = "who are paving the way for all all of of you all of"
            .split_whitespace()
            .enumerate()
            .map(|(i, text)| AsrTimingToken {
                text: text.into(),
                start_ms: i as u64 * 100,
                end_ms: i as u64 * 100 + 80,
            })
            .collect::<Vec<_>>();
        let observed = observe(
            &chat,
            &tokens,
            UtrMatchMode::Exact,
            GlobalUtrParticipation::AllUtterances,
        );
        assert_eq!(observed.len(), 1);
        let PairObservation::Complete {
            matched_words,
            words,
            ..
        } = &observed[0].observation
        else {
            panic!("bounded control");
        };
        assert_eq!(*matched_words, 11);
        assert_eq!(
            words[9]
                .candidates
                .iter()
                .map(|matched| matched.token.token_index.0)
                .collect::<Vec<_>>(),
            [6, 7, 11]
        );
        assert_eq!(
            words[10]
                .candidates
                .iter()
                .map(|matched| matched.token.token_index.0)
                .collect::<Vec<_>>(),
            [8, 9, 12]
        );
        assert!(
            words[9..]
                .iter()
                .all(|word| word.common.is_none() && !word.can_be_missing)
        );
        assert_eq!(
            serde_json::to_value(&chat).expect("source structure"),
            before,
            "observation must not mutate CHAT"
        );
        assert!(
            observe(
                &admitted_pair("one two", "yes", "PAR"),
                &tokens,
                UtrMatchMode::Exact,
                GlobalUtrParticipation::AllUtterances
            )
            .is_empty(),
            "same-speaker order is never relaxed"
        );
    }
}
