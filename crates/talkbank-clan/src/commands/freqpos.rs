//! FREQPOS — Word frequency by position in utterance.
//!
//! Reimplements CLAN's FREQPOS command, which counts how often each word
//! appears in initial, final, other (middle), or one-word positions within
//! utterances. FREQPOS is part of the FREQ family of commands and is useful
//! for studying positional word preferences -- for example, whether a child
//! tends to place certain words at the beginning or end of utterances.
//!
//! Position classification rules:
//! - **Initial**: first word of a multi-word utterance
//! - **Final**: last word of a multi-word utterance
//! - **Other**: any middle word of a multi-word utterance (3+ words)
//! - **One-word**: the sole word in a single-word utterance
//!
//! # CLAN Equivalence
//!
//! | CLAN command                | Rust equivalent                           |
//! |-----------------------------|-------------------------------------------|
//! | `freqpos file.cha`          | `chatter analyze freqpos file.cha`        |
//! | `freqpos +t*CHI file.cha`   | `chatter analyze freqpos file.cha -s CHI` |
//!
//! # Output
//!
//! Global word list (sorted alphabetically by display form) with positional
//! breakdown (initial/final/other/one-word counts per word), followed by
//! aggregate position totals.
//!
//! # Differences from CLAN
//!
//! - Word identification uses AST-based `is_countable_word()` instead of
//!   CLAN's string-prefix matching (`word[0] == '&'`, etc.).
//! - Position classification operates on parsed AST word lists rather than
//!   raw text token splitting.
//! - Output supports text, JSON, and CSV formats (CLAN produces text only).
//! - Deterministic output ordering via sorted collections.

use std::collections::BTreeMap;

use serde::Serialize;
use talkbank_model::Utterance;

use crate::framework::word_filter::countable_words;
use crate::framework::{
    AnalysisCommand, CommandOutput, FileContext, NormalizedWord, clan_display_form,
};

/// Position classification mode for FREQPOS (CLAN `+d`).
///
/// Default `FirstLastOther` matches CLAN's default behaviour:
/// position 0 is "initial", position `len-1` is "final", all
/// middle positions are "other". The `FirstSecondOther` mode
/// (CLAN `+d`) reclassifies position 1 as "second" instead, so
/// nothing past position 1 carries a positional label.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub enum PositionClassification {
    /// CLAN default: position 0 → initial, position `len-1` →
    /// final, middle → other.
    #[default]
    FirstLastOther,
    /// CLAN `+d`: position 0 → initial, position 1 → second
    /// (formerly "final"), positions ≥ 2 → other.
    FirstSecondOther,
}

/// Configuration for the FREQPOS command.
#[derive(Debug, Clone, Default)]
pub struct FreqposConfig {
    /// CLAN `+d`: switch the classification mode. Default
    /// `FirstLastOther` matches the legacy CLAN behaviour.
    pub position_classification: PositionClassification,
    /// CLAN `+k`: case-sensitive word keying. Default (`false`)
    /// lowercases each word's `cleaned_text()` via the standard
    /// `NormalizedWord::from_word`; when `true`, the key preserves
    /// original case so `Want`/`want`/`WANT` become three distinct
    /// entries in the position-classification table.
    pub case_sensitive: bool,
}

/// Positional counts for a single word.
#[derive(Debug, Default, Clone)]
struct WordPositionCounts {
    /// Total occurrences
    total: u64,
    /// Occurrences as first word of a multi-word utterance
    initial: u64,
    /// Occurrences in the "second slot" of a multi-word utterance.
    /// Meaning depends on `position_classification`:
    /// `FirstLastOther` ⇒ last position (`i == len - 1`);
    /// `FirstSecondOther` (CLAN `+d`) ⇒ position 1.
    final_pos: u64,
    /// Occurrences in middle positions of a multi-word utterance
    other: u64,
    /// Occurrences as the sole word in a one-word utterance
    one_word: u64,
    /// CLAN display form (preserves `+` in compounds)
    display_form: String,
}

/// A single word position entry in the output.
#[derive(Debug, Clone, Serialize)]
pub struct FreqposEntry {
    /// The word (normalized).
    pub word: String,
    /// CLAN display form.
    pub display_form: String,
    /// Total occurrences.
    pub total: u64,
    /// Occurrences in initial position.
    pub initial: u64,
    /// Occurrences in the "second slot" — meaning depends on the
    /// `position_classification` mode this result was produced
    /// under. `FirstLastOther` (default): the LAST position of a
    /// multi-word utterance (`i == len - 1`). `FirstSecondOther`
    /// (CLAN `+d`): position 1 specifically (`i == 1`). Field name
    /// is `final_pos` for JSON-schema stability across modes;
    /// renderers consult the result's `position_classification`
    /// to label the column "final" vs "second" accordingly.
    pub final_pos: u64,
    /// Occurrences in other (middle) positions.
    pub other: u64,
    /// Occurrences as one-word utterance.
    pub one_word: u64,
}

/// Typed output for the FREQPOS command.
#[derive(Debug, Clone, Serialize)]
pub struct FreqposResult {
    /// Word entries sorted alphabetically by display form.
    pub entries: Vec<FreqposEntry>,
    /// Total words in initial position across all entries.
    pub total_initial: u64,
    /// Total words in other (middle) position.
    pub total_other: u64,
    /// Total words in final position. Under `FirstSecondOther` mode
    /// this counter holds the position-1 ("second") count instead.
    pub total_final: u64,
    /// Total one-word utterances.
    pub total_one_word: u64,
    /// Classification mode used to produce these counts; render
    /// uses this to label `total_final` as "final" or "second".
    pub position_classification: PositionClassification,
}

impl CommandOutput for FreqposResult {
    /// Use CLAN-aligned text as the default textual representation.
    fn render_text(&self) -> String {
        self.render_clan()
    }

    /// CLAN-compatible output matching legacy CLAN character-for-character.
    ///
    /// Format:
    /// ```text
    ///   1  cookie               initial =  0, final =  1, other =  0, one word =  0
    ///
    /// Number of words in an initial position =  3
    /// Number of words in an other position   =  6
    /// Number of words in a final position    =  3
    /// Number of one word utterences          =  1
    /// ```
    fn render_clan(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();

        // Find the max display form length for alignment.
        // CLAN's freqpos uses a 20-character word-display column.
        let max_display_len = self
            .entries
            .iter()
            .map(|e| e.display_form.len())
            .max()
            .unwrap_or(0)
            .max(20);

        // CLAN labels the position-1 column "final" by default; with
        // `+d` (`FirstSecondOther`), the same column reports a
        // different population (position 1 instead of position
        // `len-1`) and the label becomes "second".
        let second_label = match self.position_classification {
            PositionClassification::FirstLastOther => "final",
            PositionClassification::FirstSecondOther => "second",
        };
        let second_footer_label = match self.position_classification {
            PositionClassification::FirstLastOther => "Number of words in a final position    =",
            PositionClassification::FirstSecondOther => "Number of words in a second position   =",
        };

        for entry in &self.entries {
            writeln!(
                out,
                "{:>3}  {:<width$} initial = {:>2}, {} = {:>2}, other = {:>2}, one word = {:>2}",
                entry.total,
                entry.display_form,
                entry.initial,
                second_label,
                entry.final_pos,
                entry.other,
                entry.one_word,
                width = max_display_len,
            )
            .ok();
        }

        // Position summary footer
        writeln!(out).ok();
        writeln!(
            out,
            "Number of words in an initial position = {:>2}",
            self.total_initial
        )
        .ok();
        writeln!(
            out,
            "Number of words in an other position   = {:>2}",
            self.total_other
        )
        .ok();
        writeln!(out, "{} {:>2}", second_footer_label, self.total_final).ok();
        writeln!(
            out,
            "Number of one word utterences          = {:>2}",
            self.total_one_word
        )
        .ok();

        out
    }
}

/// Accumulated state for FREQPOS across all files.
#[derive(Debug, Default)]
pub struct FreqposState {
    /// Per-word position counts, keyed by normalized word.
    by_word: BTreeMap<NormalizedWord, WordPositionCounts>,
}

/// FREQPOS command implementation.
///
/// For each utterance, classifies each word by its position
/// (initial/final/other/one-word) and accumulates counts globally.
#[derive(Debug, Clone, Default)]
pub struct FreqposCommand {
    /// User-facing configuration.
    pub config: FreqposConfig,
}

impl FreqposCommand {
    /// Construct with explicit configuration.
    pub fn new(config: FreqposConfig) -> Self {
        Self { config }
    }
}

impl AnalysisCommand for FreqposCommand {
    type Config = FreqposConfig;
    type State = FreqposState;
    type Output = FreqposResult;

    /// Classify each lexical token by utterance position and accumulate counts.
    fn process_utterance(
        &self,
        utterance: &Utterance,
        _file_context: &FileContext<'_>,
        state: &mut Self::State,
    ) {
        let case_sensitive = self.config.case_sensitive;
        let words: Vec<(NormalizedWord, String)> = countable_words(&utterance.main.content.content)
            .map(|w| {
                (
                    NormalizedWord::from_word_cased(w, case_sensitive),
                    clan_display_form(w),
                )
            })
            .collect();

        let len = words.len();
        if len == 0 {
            return;
        }

        for (i, (key, display)) in words.iter().enumerate() {
            let entry = state.by_word.entry(key.clone()).or_default();
            if entry.display_form.is_empty() {
                entry.display_form.clone_from(display);
            }
            entry.total += 1;

            // Classification depends on `position_classification`.
            // `FirstSecondOther` reinterprets the "final" counter
            // as "second" — position 1 increments it, positions
            // past 1 go to "other".
            let mode = self.config.position_classification;
            if len == 1 {
                entry.one_word += 1;
            } else if i == 0 {
                entry.initial += 1;
            } else {
                let is_second_slot = match mode {
                    PositionClassification::FirstLastOther => i == len - 1,
                    PositionClassification::FirstSecondOther => i == 1,
                };
                if is_second_slot {
                    entry.final_pos += 1;
                } else {
                    entry.other += 1;
                }
            }
        }
    }

    /// Build sorted entries and compute global position totals.
    fn finalize(&self, state: Self::State) -> FreqposResult {
        let mut total_initial: u64 = 0;
        let mut total_other: u64 = 0;
        let mut total_final: u64 = 0;
        let mut total_one_word: u64 = 0;

        // Sort by display form alphabetically
        let mut entries_vec: Vec<(NormalizedWord, WordPositionCounts)> =
            state.by_word.into_iter().collect();
        entries_vec.sort_by(|a, b| a.1.display_form.cmp(&b.1.display_form));

        let entries: Vec<FreqposEntry> = entries_vec
            .into_iter()
            .map(|(key, counts)| {
                total_initial += counts.initial;
                total_other += counts.other;
                total_final += counts.final_pos;
                total_one_word += counts.one_word;

                FreqposEntry {
                    word: key.as_str().to_owned(),
                    display_form: counts.display_form,
                    total: counts.total,
                    initial: counts.initial,
                    final_pos: counts.final_pos,
                    other: counts.other,
                    one_word: counts.one_word,
                }
            })
            .collect();

        FreqposResult {
            entries,
            total_initial,
            total_other,
            total_final,
            total_one_word,
            position_classification: self.config.position_classification,
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

    /// `+d` (`FirstSecondOther`) reclassifies position 1 as the
    /// "second" slot and pushes positions ≥ 2 to "other". For
    /// `[I, want, a, cookie]`:
    ///   default: I=initial, cookie=final, want+a=other
    ///   +d:      I=initial, want=second, a+cookie=other
    #[test]
    fn freqpos_second_mode_reclassifies_position_one() {
        let command = FreqposCommand::new(FreqposConfig {
            position_classification: PositionClassification::FirstSecondOther,
            case_sensitive: false,
        });
        let mut state = FreqposState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = file_ctx(&chat_file);

        let u = make_utterance("CHI", &["I", "want", "a", "cookie"]);
        command.process_utterance(&u, &ctx, &mut state);

        let result = command.finalize(state);
        assert_eq!(result.total_initial, 1); // I
        assert_eq!(result.total_final, 1); // want (position 1; counter reused as "second")
        assert_eq!(result.total_other, 2); // a + cookie
        assert_eq!(result.total_one_word, 0);

        // Render uses "second" label, not "final".
        let clan = result.render_clan();
        assert!(
            clan.contains("Number of words in a second position"),
            "expected 'second' footer label, got:\n{clan}"
        );
        assert!(
            !clan.contains("Number of words in a final position"),
            "default 'final' label should NOT appear in +d mode"
        );
        assert!(clan.contains("second = "));
        assert!(!clan.contains("final = "));
    }

    /// Default mode (`FirstLastOther`) renders with "final" label.
    /// Companion to the +d test for an obvious diff.
    #[test]
    fn freqpos_default_mode_keeps_final_label() {
        let command = FreqposCommand::default();
        let mut state = FreqposState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = file_ctx(&chat_file);

        let u = make_utterance("CHI", &["I", "want", "a", "cookie"]);
        command.process_utterance(&u, &ctx, &mut state);

        let result = command.finalize(state);
        assert_eq!(result.total_initial, 1); // I
        assert_eq!(result.total_final, 1); // cookie
        assert_eq!(result.total_other, 2); // want + a

        let clan = result.render_clan();
        assert!(clan.contains("Number of words in a final position"));
        assert!(clan.contains("final = "));
    }

    /// Multi-word utterances should split counts across initial/other/final buckets.
    #[test]
    fn freqpos_position_tracking() {
        let command = FreqposCommand::default();
        let mut state = FreqposState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = file_ctx(&chat_file);

        // "I want cookie" → I=initial, want=other, cookie=final
        let u = make_utterance("CHI", &["I", "want", "cookie"]);
        command.process_utterance(&u, &ctx, &mut state);

        let result = command.finalize(state);
        assert_eq!(result.entries.len(), 3);
        assert_eq!(result.total_initial, 1);
        assert_eq!(result.total_other, 1);
        assert_eq!(result.total_final, 1);
        assert_eq!(result.total_one_word, 0);
    }

    /// Single-token utterances should increment only the one-word bucket.
    #[test]
    fn freqpos_one_word_utterance() {
        let command = FreqposCommand::default();
        let mut state = FreqposState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = file_ctx(&chat_file);

        let u = make_utterance("CHI", &["hello"]);
        command.process_utterance(&u, &ctx, &mut state);

        let result = command.finalize(state);
        assert_eq!(result.total_one_word, 1);
        assert_eq!(result.total_initial, 0);
    }

    /// Finalizing untouched state should produce empty entries and zero totals.
    #[test]
    fn freqpos_empty_state() {
        let command = FreqposCommand::default();
        let state = FreqposState::default();
        let result = command.finalize(state);
        assert!(result.entries.is_empty());
    }

    /// CLAN FREQPOS `+k` / `--case-sensitive`: word keying preserves
    /// original case. Without `+k`, the by-word entries collapse
    /// case variants under one normalized key.
    #[test]
    fn freqpos_case_sensitive_splits_case_variants() {
        let command = FreqposCommand::new(FreqposConfig {
            case_sensitive: true,
            ..FreqposConfig::default()
        });
        let mut state = FreqposState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = file_ctx(&chat_file);

        let u = make_utterance("CHI", &["Want", "want", "WANT"]);
        command.process_utterance(&u, &ctx, &mut state);

        let result = command.finalize(state);
        // Three distinct keys, one for each case variant.
        assert_eq!(result.entries.len(), 3);
    }

    /// Companion regression: default lowercases the key, collapsing
    /// the three case variants into one entry.
    #[test]
    fn freqpos_default_collapses_case_variants() {
        let command = FreqposCommand::default();
        let mut state = FreqposState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = file_ctx(&chat_file);

        let u = make_utterance("CHI", &["Want", "want", "WANT"]);
        command.process_utterance(&u, &ctx, &mut state);

        let result = command.finalize(state);
        assert_eq!(result.entries.len(), 1);
    }
}
