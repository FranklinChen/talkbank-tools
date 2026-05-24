//! DIST — Word distribution analysis across conversational turns.
//!
//! Reimplements CLAN's DIST command, which counts turns and tracks for each
//! word the first and last turn in which it appears. CLAN counts every
//! utterance as its own turn, regardless of whether the speaker changed.
//!
//! DIST is part of the FREQ family of commands and is useful for studying
//! when words first appear and how their usage is distributed across a
//! conversation.
//!
//! # CLAN Equivalence
//!
//! | CLAN command                     | Rust equivalent                                  |
//! |----------------------------------|--------------------------------------------------|
//! | `dist file.cha`                  | `chatter analyze dist file.cha`                  |
//! | `dist +t*CHI file.cha`           | `chatter analyze dist file.cha -s CHI`           |
//!
//! # Output
//!
//! Global word list (sorted alphabetically by display form) with:
//! - Occurrence count across all turns
//! - First turn number (1-based) in which the word occurs
//! - Last turn number (omitted in CLAN output if same as first)
//! - Total number of turns in the transcript
//!
//! # Differences from CLAN
//!
//! - Word identification uses AST-based `is_countable_word()` instead of
//!   CLAN's string-prefix matching (`word[0] == '&'`, etc.).
//! - Turn detection and word extraction operate on parsed AST content
//!   rather than raw text lines.
//! - Output supports text, JSON, and CSV formats (CLAN produces text only).
//! - Deterministic output ordering via sorted collections.

use std::collections::BTreeMap;

use serde::Serialize;
use talkbank_model::Utterance;

use crate::framework::word_filter::countable_words;
use crate::framework::{
    AnalysisCommand, CommandOutput, FileContext, NormalizedWord, TurnCount, WordCount,
    clan_display_form,
};

/// Configuration for the DIST command.
#[derive(Debug, Clone, Default)]
pub struct DistConfig {
    /// CLAN `+g`: count each word at most once per utterance/turn.
    /// Mainly affects the `total_count` column; `first_turn` /
    /// `last_turn` are unchanged.
    pub once_per_turn: bool,
    /// CLAN `+k`: case-sensitive word keying. Default (`false`)
    /// lowercases via `NormalizedWord::from_word`; when `true`,
    /// the key preserves original case so `Want`/`want`/`WANT`
    /// land in separate by-word entries.
    pub case_sensitive: bool,
}

/// Distribution stats for a single word.
#[derive(Debug, Clone, Serialize)]
pub struct DistWordEntry {
    /// The word (lowercased).
    pub word: String,
    /// CLAN display form (preserves `+` in compounds).
    pub display_form: String,
    /// Total occurrences across all turns.
    pub total_count: WordCount,
    /// First turn number (1-based) in which this word appears.
    pub first_turn: TurnCount,
    /// Last turn number (1-based) in which this word appears.
    pub last_turn: TurnCount,
    /// Average distance = (last_turn - first_turn) / total_count.
    /// Only present when total_count >= 2.
    pub average_distance: Option<f64>,
}

/// Typed output for the DIST command.
#[derive(Debug, Clone, Serialize)]
pub struct DistResult {
    /// Total number of turns (one per utterance).
    pub total_turns: TurnCount,
    /// Word entries sorted alphabetically by display form.
    pub words: Vec<DistWordEntry>,
}

impl CommandOutput for DistResult {
    /// Use CLAN-formatted output as the default text representation.
    fn render_text(&self) -> String {
        self.render_clan()
    }

    /// CLAN-compatible output matching legacy CLAN character-for-character.
    ///
    /// Format:
    /// ```text
    /// There were 4 turns.
    ///
    ///
    ///                  Occurrence   First    Last        Average
    /// Word                  Count   Occurs   Occurs      Distance
    /// -----------------------------------------------------------
    /// choo+choo's              1        4
    /// cookie                   1        1
    /// ```
    fn render_clan(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();

        writeln!(out, "There were {} turns.", self.total_turns).ok();
        writeln!(out).ok();
        writeln!(out).ok();

        // Header
        writeln!(
            out,
            "                 Occurrence   First    Last        Average "
        )
        .ok();
        writeln!(
            out,
            "Word                  Count   Occurs   Occurs      Distance"
        )
        .ok();
        writeln!(
            out,
            "-----------------------------------------------------------"
        )
        .ok();

        for entry in &self.words {
            // CLAN's dist uses an 11-char-wide Average Distance column
            // (`{:>11.4}` — observed against the legacy binary), so the
            // gap from the previous 5-wide column is 5 leading spaces +
            // 6-char float. The other columns stay 5-wide with 4-space
            // gaps.
            if let Some(avg_dist) = entry.average_distance {
                writeln!(
                    out,
                    "{:<20} {:>5}    {:>5}    {:>5}    {:>11.4}",
                    entry.display_form,
                    entry.total_count,
                    entry.first_turn,
                    entry.last_turn,
                    avg_dist,
                )
                .ok();
            } else {
                writeln!(
                    out,
                    "{:<20} {:>5}    {:>5}",
                    entry.display_form, entry.total_count, entry.first_turn,
                )
                .ok();
            }
        }

        out
    }
}

/// Per-word distribution data (internal accumulation).
#[derive(Debug, Default)]
struct WordDist {
    /// Total occurrences.
    total_count: WordCount,
    /// First turn (1-based) containing this word.
    first_turn: TurnCount,
    /// Last turn (1-based) containing this word.
    last_turn: TurnCount,
    /// CLAN display form.
    display_form: String,
}

/// Accumulated state for DIST across all files.
#[derive(Debug, Default)]
pub struct DistState {
    /// Per-word distribution data, keyed by normalized word.
    by_word: BTreeMap<NormalizedWord, WordDist>,
    /// Current turn number (incremented per utterance).
    current_turn: TurnCount,
}

/// DIST command implementation.
///
/// Tracks turns (one per utterance) and records per-word first/last turn.
#[derive(Debug, Clone, Default)]
pub struct DistCommand {
    /// User-facing configuration (e.g. CLAN `+g` once-per-turn).
    pub config: DistConfig,
}

impl DistCommand {
    /// Construct with explicit configuration.
    pub fn new(config: DistConfig) -> Self {
        Self { config }
    }
}

impl AnalysisCommand for DistCommand {
    type Config = DistConfig;
    type State = DistState;
    type Output = DistResult;

    /// Each utterance is a new turn. Update per-word first/last-turn metadata.
    fn process_utterance(
        &self,
        utterance: &Utterance,
        _file_context: &FileContext<'_>,
        state: &mut Self::State,
    ) {
        // CLAN counts every utterance as its own turn, regardless of speaker.
        state.current_turn += 1;

        let turn = state.current_turn;

        // `+g` (`once_per_turn`) collapses repeated occurrences of
        // the same word within one utterance to a single count.
        // `first_turn` / `last_turn` are unaffected because they
        // only ever update on first/most-recent encounter.
        let mut seen_this_turn: std::collections::HashSet<NormalizedWord> =
            std::collections::HashSet::new();
        let case_sensitive = self.config.case_sensitive;
        for word in countable_words(&utterance.main.content.content) {
            let key = NormalizedWord::from_word_cased(word, case_sensitive);
            let display = clan_display_form(word);

            let dist = state.by_word.entry(key.clone()).or_default();
            if !self.config.once_per_turn || seen_this_turn.insert(key) {
                dist.total_count += 1;
            }
            if dist.first_turn == 0 {
                dist.first_turn = turn;
                dist.display_form = display;
            }
            dist.last_turn = turn;
        }
    }

    /// Build sorted word rows and finalize total-turn count.
    fn finalize(&self, state: Self::State) -> DistResult {
        // Sort by display form alphabetically
        let mut entries: Vec<(NormalizedWord, WordDist)> = state.by_word.into_iter().collect();
        entries.sort_by(|a, b| a.1.display_form.cmp(&b.1.display_form));

        let words: Vec<DistWordEntry> = entries
            .into_iter()
            .map(|(key, dist)| {
                let average_distance = if dist.total_count >= 2 {
                    Some((dist.last_turn - dist.first_turn) as f64 / dist.total_count as f64)
                } else {
                    None
                };
                DistWordEntry {
                    word: key.as_str().to_owned(),
                    display_form: dist.display_form,
                    total_count: dist.total_count,
                    first_turn: dist.first_turn,
                    last_turn: dist.last_turn,
                    average_distance,
                }
            })
            .collect();

        DistResult {
            total_turns: state.current_turn,
            words,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use talkbank_model::Span;
    use talkbank_model::{MainTier, Terminator, UtteranceContent, Word};

    /// Build a minimal utterance with plain lexical tokens for tests.
    fn make_utterance(speaker: &str, words: &[&str]) -> Utterance {
        let content: Vec<UtteranceContent> = words
            .iter()
            .map(|w| UtteranceContent::Word(Box::new(Word::simple(*w))))
            .collect();
        let main = MainTier::new(speaker, content, Terminator::Period { span: Span::DUMMY });
        Utterance::new(main)
    }

    /// Build a stable `FileContext` fixture reused by command tests.
    fn file_ctx(chat_file: &talkbank_model::ChatFile) -> FileContext<'_> {
        FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file,
            filename: "test",
            line_map: None,
        }
    }

    /// Every utterance counts as its own turn (matching CLAN behavior).
    #[test]
    fn dist_turn_counting() {
        let command = DistCommand::default();
        let mut state = DistState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = file_ctx(&chat_file);

        // CHI → MOT → CHI = 3 utterances = 3 turns
        let u1 = make_utterance("CHI", &["hello"]);
        let u2 = make_utterance("MOT", &["hi"]);
        let u3 = make_utterance("CHI", &["bye"]);
        command.process_utterance(&u1, &ctx, &mut state);
        command.process_utterance(&u2, &ctx, &mut state);
        command.process_utterance(&u3, &ctx, &mut state);

        let result = command.finalize(state);
        assert_eq!(result.total_turns, 3);
    }

    /// Consecutive same-speaker utterances each count as a turn (CLAN behavior).
    #[test]
    fn dist_same_speaker_still_increments_turns() {
        let command = DistCommand::default();
        let mut state = DistState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = file_ctx(&chat_file);

        // CHI → CHI → CHI = 3 utterances = 3 turns
        let u1 = make_utterance("CHI", &["hello"]);
        let u2 = make_utterance("CHI", &["there"]);
        let u3 = make_utterance("CHI", &["bye"]);
        command.process_utterance(&u1, &ctx, &mut state);
        command.process_utterance(&u2, &ctx, &mut state);
        command.process_utterance(&u3, &ctx, &mut state);

        let result = command.finalize(state);
        assert_eq!(result.total_turns, 3);
    }

    /// `+g` (`once_per_turn`) deduplicates repeated words within
    /// a single turn — `hello hello bye` counts `hello` once, not
    /// twice. `first_turn` / `last_turn` are unchanged by the flag.
    #[test]
    fn dist_once_per_turn_collapses_repeats_in_one_turn() {
        let command = DistCommand::new(DistConfig {
            once_per_turn: true,
            case_sensitive: false,
        });
        let mut state = DistState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = file_ctx(&chat_file);

        // Turn 1 has "hello" twice + "bye"; Turn 2 has "hello".
        // Default: total_count(hello)=3, total_count(bye)=1.
        // +g:      total_count(hello)=2 (one per turn), bye=1.
        let u1 = make_utterance("CHI", &["hello", "hello", "bye"]);
        let u2 = make_utterance("MOT", &["hello"]);
        command.process_utterance(&u1, &ctx, &mut state);
        command.process_utterance(&u2, &ctx, &mut state);

        let result = command.finalize(state);
        let hello = result.words.iter().find(|w| w.word == "hello").unwrap();
        assert_eq!(hello.total_count, 2);
        assert_eq!(hello.first_turn, 1);
        assert_eq!(hello.last_turn, 2);
        let bye = result.words.iter().find(|w| w.word == "bye").unwrap();
        assert_eq!(bye.total_count, 1);
    }

    /// Default behaviour (without `+g`) still counts every occurrence.
    /// Companion to the once-per-turn test for an obvious diff.
    #[test]
    fn dist_default_counts_every_occurrence() {
        let command = DistCommand::default();
        let mut state = DistState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = file_ctx(&chat_file);

        let u1 = make_utterance("CHI", &["hello", "hello", "bye"]);
        let u2 = make_utterance("MOT", &["hello"]);
        command.process_utterance(&u1, &ctx, &mut state);
        command.process_utterance(&u2, &ctx, &mut state);

        let result = command.finalize(state);
        let hello = result.words.iter().find(|w| w.word == "hello").unwrap();
        assert_eq!(hello.total_count, 3);
    }

    /// Word entries should retain first and last turn positions across speakers.
    #[test]
    fn dist_word_first_last_turn() {
        let command = DistCommand::default();
        let mut state = DistState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = file_ctx(&chat_file);

        // Turn 1: CHI says "hello", Turn 2: MOT says "hello"
        let u1 = make_utterance("CHI", &["hello"]);
        let u2 = make_utterance("MOT", &["hello"]);
        command.process_utterance(&u1, &ctx, &mut state);
        command.process_utterance(&u2, &ctx, &mut state);

        let result = command.finalize(state);
        let hello = result.words.iter().find(|w| w.word == "hello").unwrap();
        assert_eq!(hello.first_turn, 1);
        assert_eq!(hello.last_turn, 2);
        assert_eq!(hello.total_count, 2);
    }

    /// Finalizing untouched state should produce zero turns and no words.
    #[test]
    fn dist_empty_state() {
        let command = DistCommand::default();
        let state = DistState::default();
        let result = command.finalize(state);
        assert!(result.words.is_empty());
        assert_eq!(result.total_turns, 0);
    }

    /// CLAN DIST `+k` / `--case-sensitive`: case variants land in
    /// separate by-word entries.
    #[test]
    fn dist_case_sensitive_splits_case_variants() {
        let command = DistCommand::new(DistConfig {
            case_sensitive: true,
            ..DistConfig::default()
        });
        let mut state = DistState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = file_ctx(&chat_file);

        let u = make_utterance("CHI", &["Want", "want", "WANT"]);
        command.process_utterance(&u, &ctx, &mut state);

        let result = command.finalize(state);
        // Three distinct keys.
        assert_eq!(result.words.len(), 3);
    }

    /// Default lowercases the key, collapsing case variants into
    /// one entry.
    #[test]
    fn dist_default_collapses_case_variants() {
        let command = DistCommand::default();
        let mut state = DistState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = file_ctx(&chat_file);

        let u = make_utterance("CHI", &["Want", "want", "WANT"]);
        command.process_utterance(&u, &ctx, &mut state);

        let result = command.finalize(state);
        assert_eq!(result.words.len(), 1);
    }
}
