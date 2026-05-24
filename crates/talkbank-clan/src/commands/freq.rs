//! FREQ — Word frequency analysis.
//!
//! Reimplements CLAN's FREQ command, which counts word tokens and types
//! on the main tier and/or `%mor` tier, computing type-token ratio (TTR).
//! FREQ is the most commonly used CLAN command and serves as the foundation
//! for lexical diversity analysis in child language research.
//!
//! Word normalization uses [`NormalizedWord`], which lowercases and strips
//! compound markers (`+`) for grouping, while preserving the original
//! CLAN display form (with `+`) for output.
//!
//! # CLAN Manual
//!
//! See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html#_Toc220409093)
//! for the original FREQ command specification.
//!
//! # CLAN Equivalence
//!
//! | CLAN command              | Rust equivalent                        |
//! |---------------------------|----------------------------------------|
//! | `freq file.cha`           | `chatter analyze freq file.cha`        |
//! | `freq +t*CHI file.cha`    | `chatter analyze freq file.cha -s CHI` |
//!
//! # Output
//!
//! Per-speaker frequency tables with:
//! - Word frequency counts (sorted by count descending, then alphabetically)
//! - Total types (unique words) and tokens (total words)
//! - TTR (type-token ratio = types / tokens)
//!
//! # Differences from CLAN
//!
//! - Word identification uses AST-based `is_countable_word()` instead of
//!   CLAN's string-prefix matching (`word[0] == '&'`, etc.).
//! - Output supports text, JSON, and CSV formats (CLAN produces text only).
//! - Deterministic output ordering via sorted collections.

use std::collections::{BTreeSet, HashMap};
use std::fmt;

use indexmap::IndexMap;
use serde::Serialize;
use talkbank_model::{SpeakerCode, Utterance};

use crate::framework::word_filter::{CapitalizationFilter, countable_words};
use crate::framework::{
    AnalysisCommand, CommandOutput, FileContext, NormalizedWord, TypeCount, WordCount,
    clan_display_form_preserve_case,
};

/// Configuration for the FREQ command.
#[derive(Debug, Clone, Default)]
pub struct FreqConfig {
    /// Count morphemes from %mor tier instead of words from main tier
    pub use_mor: bool,
    /// CLAN's `+c` / `+c0` / `+c1`: restrict counting to words
    /// whose surface form matches a capitalization predicate.
    /// `Any` (default) counts every countable word.
    pub capitalization: CapitalizationFilter,
    /// CLAN `+o1`: sort frequency entries by the reversed
    /// character sequence of each word — groups words with
    /// shared suffixes. Default (`false`) sorts by frequency
    /// descending with alphabetical tiebreak.
    pub reverse_concordance: bool,
    /// CLAN `+d1`: emit only an alphabetized deduped word list,
    /// one word per line, with no banners, counts, or totals.
    /// Intended as fodder for `kwal +s@FILE`.
    pub word_list_only: bool,
    /// CLAN `+d4`: emit only per-speaker type/token/TTR summary,
    /// dropping per-word frequency entries.
    pub types_tokens_only: bool,
    /// CLAN `+k`: case-sensitive keying. Default (`false`) lowercases
    /// each word's `cleaned_text()` into the standard `NormalizedWord`
    /// before counting, collapsing case variants. When `true`, the key
    /// preserves original case, so `Want`/`want`/`WANT` become three
    /// distinct entries.
    pub case_sensitive: bool,
}

/// Per-speaker frequency data accumulated during processing.
#[derive(Debug, Default)]
struct SpeakerFreq {
    /// Normalized word → count mapping
    counts: HashMap<NormalizedWord, WordCount>,
    /// Normalized word → CLAN display form (preserves `+` in compounds)
    display_forms: HashMap<NormalizedWord, String>,
    /// Total tokens (sum of all counts)
    total_tokens: WordCount,
}

/// Accumulated state for FREQ across all files.
#[derive(Debug, Default)]
pub struct FreqState {
    /// Per-speaker frequency data, keyed by speaker code
    by_speaker: IndexMap<SpeakerCode, SpeakerFreq>,
}

/// Typed output from the FREQ command.
///
/// Contains per-speaker frequency tables with strongly-typed fields.
#[derive(Debug, Clone, Serialize)]
pub struct FreqResult {
    /// Per-speaker frequency data, in encounter order.
    pub speakers: Vec<FreqSpeakerResult>,
    /// CLAN `+d1`: render as an alphabetized deduped word list,
    /// one word per line, with no banners, counts, or totals.
    /// Default `false` preserves the standard per-speaker layout.
    /// Field is `#[serde(default, skip_serializing_if = ...)]` so
    /// the JSON schema for default-mode FREQ output is unchanged.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub word_list_only: bool,
    /// CLAN `+d4`: emit only per-speaker type/token/TTR summary;
    /// drop all per-word frequency entries. Same defaulting rules
    /// as `word_list_only` for serde compatibility.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub types_tokens_only: bool,
}

/// Frequency statistics for a single speaker.
#[derive(Debug, Clone, Serialize)]
pub struct FreqSpeakerResult {
    /// Speaker code (e.g., "CHI", "MOT")
    pub speaker: String,
    /// Word frequency entries, sorted by count descending then alphabetically.
    pub entries: Vec<FreqEntry>,
    /// Number of unique word types
    pub total_types: TypeCount,
    /// Total word tokens
    pub total_tokens: WordCount,
    /// Type-token ratio (types / tokens)
    pub ttr: f64,
}

/// A single word frequency entry.
#[derive(Debug, Clone, Serialize)]
pub struct FreqEntry {
    /// The word (lowercased, cleaned — `+` stripped from compounds)
    pub word: String,
    /// CLAN display form (lowercased but `+` preserved in compounds)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_form: Option<String>,
    /// Number of occurrences
    pub count: WordCount,
}

impl CommandOutput for FreqResult {
    /// Our clean text format with aligned table columns.
    fn render_text(&self) -> String {
        let mut out = String::new();
        for (i, s) in self.speakers.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            fmt::write(
                &mut out,
                format_args!(
                    "Speaker: {}\n\
                     \x20 Total types: {}\n\
                     \x20 Total tokens: {}\n\
                     \x20 TTR: {:.3}\n",
                    s.speaker, s.total_types, s.total_tokens, s.ttr
                ),
            )
            .ok();

            // Table with aligned columns
            if !s.entries.is_empty() {
                let count_width = s
                    .entries
                    .iter()
                    .map(|e| e.count.to_string().len())
                    .max()
                    .unwrap_or(5)
                    .max(5); // "Count" header
                let word_width = s
                    .entries
                    .iter()
                    .map(|e| e.word.len())
                    .max()
                    .unwrap_or(4)
                    .max(4); // "Word" header

                fmt::write(
                    &mut out,
                    format_args!(
                        "  {:<cw$}  {:<ww$}\n  {:-<cw$}  {:-<ww$}\n",
                        "Count",
                        "Word",
                        "",
                        "",
                        cw = count_width,
                        ww = word_width
                    ),
                )
                .ok();

                for entry in &s.entries {
                    fmt::write(
                        &mut out,
                        format_args!(
                            "  {:<cw$}  {:<ww$}\n",
                            entry.count,
                            entry.word,
                            cw = count_width,
                            ww = word_width
                        ),
                    )
                    .ok();
                }
            }
        }
        out
    }

    /// CLAN-compatible output matching legacy CLAN character-for-character.
    ///
    /// Format (from CLAN snapshot):
    /// ```text
    /// Speaker: *CHI:
    ///   1 cookie
    ///   1 more
    /// ------------------------------
    ///     3  Total number of different item types used
    ///     3  Total number of items (tokens)
    /// 1.000  Type/Token ratio
    ///     This TTR number was not calculated on the basis of %mor line forms.
    ///     If you want a TTR based on lemmas, run FREQ on the %mor line
    ///     with option: +sm;*,o%
    /// ```
    fn render_clan(&self) -> String {
        if self.word_list_only {
            // CLAN `+d1`: alphabetized deduped word list, one per
            // line — per the manual, fodder for `kwal +s@FILE`,
            // which wants one global vocabulary, not a per-speaker
            // partition. Banners, counts, separators, and TTR are
            // intentionally omitted.
            let mut words: BTreeSet<&str> = BTreeSet::new();
            for s in &self.speakers {
                for entry in &s.entries {
                    words.insert(entry.display_form.as_deref().unwrap_or(&entry.word));
                }
            }
            let mut out = String::new();
            for w in &words {
                out.push_str(w);
                out.push('\n');
            }
            return out;
        }
        let mut out = String::new();
        for (i, s) in self.speakers.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            fmt::write(&mut out, format_args!("Speaker: *{}:\n", s.speaker)).ok();

            // CLAN sorts alphabetically by display form
            let mut sorted_entries: Vec<&FreqEntry> = s.entries.iter().collect();
            sorted_entries.sort_by(|a, b| {
                let a_display = a.display_form.as_deref().unwrap_or(&a.word);
                let b_display = b.display_form.as_deref().unwrap_or(&b.word);
                a_display.cmp(b_display)
            });

            // Word list: " <count> <display_form>". CLAN's `+d4`
            // suppresses the per-word entries but keeps the
            // surrounding speaker banner, separator, totals, and
            // TTR note.
            if !self.types_tokens_only {
                for entry in &sorted_entries {
                    let display = entry.display_form.as_deref().unwrap_or(&entry.word);
                    fmt::write(&mut out, format_args!("{:>3} {}\n", entry.count, display)).ok();
                }
            }

            // Separator and summary
            fmt::write(
                &mut out,
                format_args!(
                    "------------------------------\n\
                     {:>5}  Total number of different item types used\n\
                     {:>5}  Total number of items (tokens)\n\
                     {:.3}  Type/Token ratio\n",
                    s.total_types, s.total_tokens, s.ttr
                ),
            )
            .ok();

            // TTR note (CLAN always includes this for main-tier freq)
            out.push_str(
                "    This TTR number was not calculated on the basis of %mor line forms.\n\
                 \x20   If you want a TTR based on lemmas, run FREQ on the %mor line\n\
                 \x20   with option: +sm;*,o%\n",
            );
        }
        // CLAN emits a trailing blank line after the last per-speaker block;
        // match that so a hex-level diff against the legacy freq output ends
        // cleanly.
        if !self.speakers.is_empty() {
            out.push('\n');
        }
        out
    }

    /// CSV rendering with header row.
    fn render_csv(&self) -> String {
        let mut out = String::new();
        for s in &self.speakers {
            out.push_str(&format!("Speaker,{}\n", s.speaker));
            // CLAN `+d3`: drop the per-word `Count,Word` header and
            // rows, keep the summary statistics. CLAN `+d2` (and
            // default csv) keeps them.
            if !self.types_tokens_only {
                out.push_str("Count,Word\n");
                for entry in &s.entries {
                    out.push_str(&format!("{},{}\n", entry.count, entry.word));
                }
            }
            out.push_str(&format!(
                "Total types,{}\nTotal tokens,{}
TTR,{:.3}\n",
                s.total_types, s.total_tokens, s.ttr
            ));
        }
        out
    }
}

/// FREQ command implementation.
///
/// Counts word frequencies on the main tier, producing per-speaker
/// frequency tables with TTR.
#[derive(Debug, Clone, Default)]
pub struct FreqCommand {
    config: FreqConfig,
}

impl FreqCommand {
    /// Create a FREQ command with the given configuration.
    pub fn new(config: FreqConfig) -> Self {
        Self { config }
    }
}

impl AnalysisCommand for FreqCommand {
    type Config = FreqConfig;
    type State = FreqState;
    type Output = FreqResult;

    /// Accumulate per-speaker token counts from main-tier words or `%mor` items.
    fn process_utterance(
        &self,
        utterance: &Utterance,
        _file_context: &FileContext<'_>,
        state: &mut Self::State,
    ) {
        // Arc<str> clone — cheap atomic ref-count increment, no allocation
        let speaker = utterance.main.speaker.clone();
        let speaker_freq = state
            .by_speaker
            .entry(speaker)
            .or_insert_with(SpeakerFreq::default);

        let cap_filter = self.config.capitalization;
        let case_sensitive = self.config.case_sensitive;
        if self.config.use_mor {
            // Count morphemes from %mor tier, using CHAT representation as key.
            // Each MorWord (main + post-clitics) counts as a separate frequency
            // item, matching CLAN's space-separated token counting on %mor.
            if let Some(mor_tier) = utterance.mor_tier() {
                for mor_item in mor_tier.items().iter() {
                    let mut raw = String::new();
                    let _ = mor_item.main.write_chat(&mut raw);
                    if cap_filter.includes(&raw) {
                        *speaker_freq
                            .counts
                            .entry(NormalizedWord::from_text_cased(&raw, case_sensitive))
                            .or_insert(0) += 1;
                        speaker_freq.total_tokens += 1;
                    }

                    // Post-clitics are separate frequency items in CLAN
                    for clitic in &mor_item.post_clitics {
                        let mut craw = String::new();
                        let _ = clitic.write_chat(&mut craw);
                        if cap_filter.includes(&craw) {
                            *speaker_freq
                                .counts
                                .entry(NormalizedWord::from_text_cased(&craw, case_sensitive))
                                .or_insert(0) += 1;
                            speaker_freq.total_tokens += 1;
                        }
                    }
                }
            }
        } else {
            // Count words from main tier using the shared countable_words() iterator
            for word in countable_words(&utterance.main.content.content) {
                if !cap_filter.includes(word.cleaned_text()) {
                    continue;
                }
                let key = NormalizedWord::from_word_cased(word, case_sensitive);
                speaker_freq
                    .display_forms
                    .entry(key.clone())
                    .or_insert_with(|| clan_display_form_preserve_case(word));
                *speaker_freq.counts.entry(key).or_insert(0) += 1;
                speaker_freq.total_tokens += 1;
            }
        }
    }

    /// Convert accumulated counts into sorted per-speaker frequency tables.
    fn finalize(&self, state: Self::State) -> FreqResult {
        let mut speakers = Vec::new();

        for (speaker, freq) in &state.by_speaker {
            // Default sort: count descending, then alphabetically.
            // `+o1` (`reverse_concordance`) swaps the tiebreak to
            // a reversed-character comparison so words with shared
            // suffixes cluster together.
            // Schwartzian transform when `+o1` is active: build
            // the reversed key once per word, sort by precomputed
            // key. Drops per-comparison char-reversal + 2 String
            // allocations to one allocation per word.
            let raw_entries: Vec<(&NormalizedWord, &u64)> = if self.config.reverse_concordance {
                let mut keyed: Vec<(String, &NormalizedWord, &u64)> = freq
                    .counts
                    .iter()
                    .map(|(w, c)| (w.as_str().chars().rev().collect::<String>(), w, c))
                    .collect();
                keyed.sort_by(|a, b| a.0.cmp(&b.0));
                keyed.into_iter().map(|(_, w, c)| (w, c)).collect()
            } else {
                let mut entries: Vec<(&NormalizedWord, &u64)> = freq.counts.iter().collect();
                entries.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
                entries
            };

            let total_types = freq.counts.len() as u64;
            let total_tokens = freq.total_tokens;
            let ttr = if total_tokens > 0 {
                total_types as f64 / total_tokens as f64
            } else {
                0.0
            };

            let entries: Vec<FreqEntry> = raw_entries
                .iter()
                .map(|(word, count)| {
                    let display = freq.display_forms.get(*word).cloned();
                    FreqEntry {
                        word: word.as_str().to_owned(),
                        display_form: display,
                        count: **count,
                    }
                })
                .collect();

            speakers.push(FreqSpeakerResult {
                speaker: speaker.as_str().to_owned(),
                entries,
                total_types,
                total_tokens,
                ttr,
            });
        }

        FreqResult {
            speakers,
            word_list_only: self.config.word_list_only,
            types_tokens_only: self.config.types_tokens_only,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use talkbank_model::Span;
    use talkbank_model::{MainTier, Terminator, UtteranceContent, Word};

    /// Build a minimal utterance with plain words for command tests.
    fn make_utterance(speaker: &str, words: &[&str]) -> Utterance {
        let content: Vec<UtteranceContent> = words
            .iter()
            .map(|w| UtteranceContent::Word(Box::new(Word::simple(*w))))
            .collect();
        let main = MainTier::new(speaker, content, Terminator::Period { span: Span::DUMMY });
        Utterance::new(main)
    }

    /// CLAN's `+c1` mode (`CapitalizationFilter::MidUpper`) drops
    /// words without an uppercase letter past position 0 — so
    /// `McDonald` and `iPhone` survive, plain `Cookie` (initial-
    /// only) does not.
    #[test]
    fn freq_mid_upper_filters_initial_only_words() {
        let command = FreqCommand {
            config: FreqConfig {
                use_mor: false,
                capitalization: CapitalizationFilter::MidUpper,
                reverse_concordance: false,
                word_list_only: false,
                types_tokens_only: false,
                case_sensitive: false,
            },
        };
        let mut state = FreqState::default();

        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let file_ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        // `McDonald` and `iPhone` pass; `I`, `Cookie`, `want`, `a`
        // all fail (either no uppercase at all, or only initial).
        let u = make_utterance(
            "CHI",
            &[
                "I", "want", "a", "Cookie", "from", "McDonald", "on", "iPhone",
            ],
        );
        command.process_utterance(&u, &file_ctx, &mut state);

        let result = command.finalize(state);
        assert_eq!(result.speakers.len(), 1);
        let chi = &result.speakers[0];
        assert_eq!(chi.total_tokens, 2);
        let words: Vec<&str> = chi.entries.iter().map(|e| e.word.as_str()).collect();
        assert!(words.contains(&"mcdonald"));
        assert!(words.contains(&"iphone"));
        assert!(!words.contains(&"cookie"));
        assert!(!words.contains(&"i"));
    }

    /// `+o1` (`reverse_concordance`) sorts entries by their
    /// reversed character sequence, grouping words by suffix.
    /// Input `cat`, `bat`, `dog`, `log`: by reverse-concordance
    /// the keys become `tac`, `tab`, `god`, `gol`; sorted →
    /// `god` (gol), `log` (gol), `bat` (tab), `cat` (tac).
    /// Words sharing a suffix cluster together.
    #[test]
    fn freq_reverse_concordance_groups_by_suffix() {
        let command = FreqCommand {
            config: FreqConfig {
                use_mor: false,
                capitalization: CapitalizationFilter::Any,
                reverse_concordance: true,
                word_list_only: false,
                types_tokens_only: false,
                case_sensitive: false,
            },
        };
        let mut state = FreqState::default();

        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let file_ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        let u = make_utterance("CHI", &["cat", "bat", "dog", "log"]);
        command.process_utterance(&u, &file_ctx, &mut state);

        let result = command.finalize(state);
        let words: Vec<&str> = result.speakers[0]
            .entries
            .iter()
            .map(|e| e.word.as_str())
            .collect();
        // Sorted by reversed string: god, gol, tab, tac
        //                  original: dog, log, bat, cat
        assert_eq!(words, vec!["dog", "log", "bat", "cat"]);
    }

    /// Default sort (frequency descending, alphabetical tiebreak)
    /// is unchanged when `reverse_concordance: false`. Companion
    /// to the +o1 test for an obvious diff on the same input.
    #[test]
    fn freq_default_sort_is_alphabetical_when_freqs_equal() {
        let command = FreqCommand::default();
        let mut state = FreqState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let file_ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        let u = make_utterance("CHI", &["cat", "bat", "dog", "log"]);
        command.process_utterance(&u, &file_ctx, &mut state);

        let result = command.finalize(state);
        let words: Vec<&str> = result.speakers[0]
            .entries
            .iter()
            .map(|e| e.word.as_str())
            .collect();
        // All freqs are 1, so alphabetical tiebreak applies.
        assert_eq!(words, vec!["bat", "cat", "dog", "log"]);
    }

    /// CLAN's `+c` / `+c0` mode drops words whose first character
    /// isn't uppercase. Two capitalized tokens in a mixed utterance
    /// should be counted once each; the lower-case tokens disappear
    /// from both the token total and the per-type table.
    #[test]
    fn freq_capitalized_only_filters_lowercase_words() {
        let command = FreqCommand {
            config: FreqConfig {
                use_mor: false,
                capitalization: CapitalizationFilter::InitialUpper,
                reverse_concordance: false,
                word_list_only: false,
                types_tokens_only: false,
                case_sensitive: false,
            },
        };
        let mut state = FreqState::default();

        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let file_ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        // "I" and "Cookie" pass the filter; "want", "a", "and"
        // do not (lowercase initial); "123" has no leading letter.
        let u = make_utterance("CHI", &["I", "want", "a", "Cookie", "and", "123"]);
        command.process_utterance(&u, &file_ctx, &mut state);

        let result = command.finalize(state);
        assert_eq!(result.speakers.len(), 1);
        let chi = &result.speakers[0];
        assert_eq!(chi.total_tokens, 2);
        assert_eq!(chi.total_types, 2);
        let words: Vec<&str> = chi.entries.iter().map(|e| e.word.as_str()).collect();
        assert!(words.contains(&"i"));
        assert!(words.contains(&"cookie"));
        assert!(!words.contains(&"want"));
        assert!(!words.contains(&"a"));
    }

    /// Counts should remain isolated per speaker key.
    #[test]
    fn freq_counts_words_per_speaker() {
        let command = FreqCommand::default();
        let mut state = FreqState::default();

        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let file_ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        let u1 = make_utterance("CHI", &["I", "want", "cookie"]);
        let u2 = make_utterance("CHI", &["I", "want", "more"]);
        let u3 = make_utterance("MOT", &["here", "you", "go"]);

        command.process_utterance(&u1, &file_ctx, &mut state);
        command.process_utterance(&u2, &file_ctx, &mut state);
        command.process_utterance(&u3, &file_ctx, &mut state);

        let result = command.finalize(state);
        assert_eq!(result.speakers.len(), 2);

        // CHI section
        let chi = &result.speakers[0];
        assert_eq!(chi.speaker, "CHI");
        assert_eq!(chi.total_tokens, 6);
        assert_eq!(chi.total_types, 4); // i, want, cookie, more

        // MOT section
        let mot = &result.speakers[1];
        assert_eq!(mot.speaker, "MOT");
        assert_eq!(mot.total_tokens, 3);
        assert_eq!(mot.total_types, 3);
    }

    /// TTR should be computed as `types / tokens`.
    #[test]
    fn freq_ttr_calculation() {
        let command = FreqCommand::default();
        let mut state = FreqState::default();

        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let file_ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        // All same word → TTR = 1/5 = 0.200
        let u = make_utterance("CHI", &["the", "the", "the", "the", "the"]);
        command.process_utterance(&u, &file_ctx, &mut state);

        let result = command.finalize(state);
        let chi = &result.speakers[0];
        assert!((chi.ttr - 0.2).abs() < 1e-10);
    }

    /// Text rendering should include speaker summary and token rows.
    #[test]
    fn freq_render_text_format() {
        let result = FreqResult {
            word_list_only: false,
            types_tokens_only: false,
            speakers: vec![FreqSpeakerResult {
                speaker: "CHI".to_owned(),
                entries: vec![
                    FreqEntry {
                        word: "want".to_owned(),
                        display_form: None,
                        count: 2,
                    },
                    FreqEntry {
                        word: "cookie".to_owned(),
                        display_form: None,
                        count: 1,
                    },
                ],
                total_types: 2,
                total_tokens: 3,
                ttr: 0.667,
            }],
        };

        let text = result.render_text();
        assert!(text.contains("Speaker: CHI"));
        assert!(text.contains("Total types: 2"));
        assert!(text.contains("want"));
        assert!(text.contains("cookie"));
    }

    /// CLAN rendering should expose legacy-style summary labels.
    #[test]
    fn freq_render_clan_format() {
        let result = FreqResult {
            word_list_only: false,
            types_tokens_only: false,
            speakers: vec![FreqSpeakerResult {
                speaker: "CHI".to_owned(),
                entries: vec![
                    FreqEntry {
                        word: "want".to_owned(),
                        display_form: None,
                        count: 2,
                    },
                    FreqEntry {
                        word: "cookie".to_owned(),
                        display_form: None,
                        count: 1,
                    },
                ],
                total_types: 2,
                total_tokens: 3,
                ttr: 0.667,
            }],
        };

        let clan = result.render_clan();
        assert!(clan.contains("Speaker: *CHI:"));
        assert!(clan.contains("2 want"));
        assert!(clan.contains("1 cookie"));
        assert!(clan.contains("Total number of different item types used"));
        assert!(clan.contains("Total number of items (tokens)"));
        assert!(clan.contains("Type/Token ratio"));
    }

    /// FREQ `+d1` / `--word-list-only`: emit one word per line, no
    /// frequencies, no per-speaker banners, no totals. Output is
    /// meant to be usable as input to `kwal +s@FILE`. Words are
    /// alphabetized and deduped across the result.
    ///
    /// CLAN manual §7.10.15 (+d1):
    /// > "Outputs each of the words found in the input data file(s)
    /// > one word per line with no further information about
    /// > frequency. Later this output could be used as a word list
    /// > file for kwal or combo programs."
    #[test]
    fn freq_word_list_only_strips_everything_but_words() {
        let result = FreqResult {
            word_list_only: true,
            types_tokens_only: false,
            speakers: vec![FreqSpeakerResult {
                speaker: "CHI".to_owned(),
                entries: vec![
                    FreqEntry {
                        word: "want".to_owned(),
                        display_form: None,
                        count: 2,
                    },
                    FreqEntry {
                        word: "cookie".to_owned(),
                        display_form: None,
                        count: 1,
                    },
                ],
                total_types: 2,
                total_tokens: 3,
                ttr: 0.667,
            }],
        };
        let clan = result.render_clan();
        let lines: Vec<&str> = clan.lines().filter(|l| !l.is_empty()).collect();
        // Alphabetized, one word per line, nothing else.
        assert_eq!(lines, vec!["cookie", "want"]);
        // No counts, banners, separators, or TTR matter for downstream
        // `kwal +s@FILE` consumption.
        assert!(
            !clan.contains("Speaker:"),
            "word-list-only must not emit Speaker banners"
        );
        assert!(
            !clan.contains("Total"),
            "word-list-only must not emit totals"
        );
        assert!(
            !clan.contains("Type/Token"),
            "word-list-only must not emit TTR"
        );
        assert!(
            !clan.contains("---"),
            "word-list-only must not emit separators"
        );
    }

    /// FREQ `+d4` / `--types-tokens-only`: emit only the
    /// per-speaker type/token/TTR summary, dropping all per-word
    /// frequency entries. The CLAN-format banner shape (Speaker
    /// header + separator + totals + TTR note) is preserved.
    ///
    /// CLAN manual §7.10.15 (+d4): "Allows you to output just the
    /// type-token information."
    #[test]
    fn freq_types_tokens_only_drops_per_word_entries() {
        let result = FreqResult {
            word_list_only: false,
            types_tokens_only: true,
            speakers: vec![FreqSpeakerResult {
                speaker: "CHI".to_owned(),
                entries: vec![
                    FreqEntry {
                        word: "want".to_owned(),
                        display_form: None,
                        count: 2,
                    },
                    FreqEntry {
                        word: "cookie".to_owned(),
                        display_form: None,
                        count: 1,
                    },
                ],
                total_types: 2,
                total_tokens: 3,
                ttr: 0.667,
            }],
        };
        let clan = result.render_clan();
        // Summary lines, speaker banner, separator, TTR all kept.
        assert!(clan.contains("Speaker: *CHI:"));
        assert!(clan.contains("Total number of different item types used"));
        assert!(clan.contains("Total number of items (tokens)"));
        assert!(clan.contains("Type/Token ratio"));
        assert!(clan.contains("------------------------------"));
        // Per-word entry lines (shape: ` <count> <word>`) are dropped.
        // Note: "want" also appears in the static TTR-note boilerplate
        // ("If you want a TTR based on lemmas"), so we cannot bare-
        // substring-check for the word — match the entry-line shape.
        assert!(
            !clan.contains("  2 want\n"),
            "types-tokens-only must not emit `<count> <word>` entry lines: {clan:?}"
        );
        assert!(
            !clan.contains("  1 cookie\n"),
            "types-tokens-only must not emit `<count> <word>` entry lines: {clan:?}"
        );
    }

    /// CSV companion to `+d4`: `+d3` is the same content in
    /// spreadsheet form. `render_csv` must honor `types_tokens_only`
    /// the same way `render_clan` does — keep the `Speaker,X` /
    /// `Total types,N` / `Total tokens,N` / `TTR,X` rows but drop
    /// the per-word `Count,Word` header and `<count>,<word>` rows.
    #[test]
    fn freq_types_tokens_only_csv_drops_per_word_rows() {
        let result = FreqResult {
            word_list_only: false,
            types_tokens_only: true,
            speakers: vec![FreqSpeakerResult {
                speaker: "CHI".to_owned(),
                entries: vec![
                    FreqEntry {
                        word: "want".to_owned(),
                        display_form: None,
                        count: 2,
                    },
                    FreqEntry {
                        word: "cookie".to_owned(),
                        display_form: None,
                        count: 1,
                    },
                ],
                total_types: 2,
                total_tokens: 3,
                ttr: 0.667,
            }],
        };
        let csv = result.render_csv();
        // Speaker row + summary rows kept.
        assert!(csv.contains("Speaker,CHI"));
        assert!(csv.contains("Total types,2"));
        assert!(csv.contains("Total tokens,3"));
        assert!(csv.contains("TTR,0.667"));
        // Per-word header and rows dropped.
        assert!(
            !csv.contains("Count,Word"),
            "types-tokens-only CSV must not emit Count,Word header: {csv:?}"
        );
        assert!(
            !csv.contains("2,want"),
            "types-tokens-only CSV must not emit per-word rows: {csv:?}"
        );
        assert!(
            !csv.contains("1,cookie"),
            "types-tokens-only CSV must not emit per-word rows: {csv:?}"
        );
    }

    /// CLAN FREQ `+k` / `--case-sensitive`: word frequency keying
    /// preserves case, so `Want`, `want`, `WANT` become three
    /// separate entries (each count 1) instead of one entry "want"
    /// with count 3. Per CLAN manual: `+k` "Match search strings in
    /// a case-sensitive way."
    #[test]
    fn freq_case_sensitive_preserves_case_in_keys() {
        use talkbank_model::ChatFile;
        let command = FreqCommand::new(FreqConfig {
            use_mor: false,
            capitalization: CapitalizationFilter::Any,
            reverse_concordance: false,
            word_list_only: false,
            types_tokens_only: false,
            case_sensitive: true,
        });
        let mut state = FreqState::default();
        let chat_file = ChatFile::new(vec![]);
        let file_ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        // Three case variants of "want" in a single utterance.
        let u = make_utterance("CHI", &["Want", "want", "WANT"]);
        command.process_utterance(&u, &file_ctx, &mut state);

        let result = command.finalize(state);
        let chi = result
            .speakers
            .iter()
            .find(|s| s.speaker == "CHI")
            .expect("CHI speaker should be present");
        // Three distinct keys, each appearing once. The exact map
        // key is internal; what matters is that the count is split
        // by case.
        assert_eq!(chi.total_tokens, 3, "all three tokens are counted");
        assert_eq!(
            chi.total_types, 3,
            "case-sensitive keying splits into 3 types"
        );
    }

    /// Companion regression to `freq_case_sensitive_preserves_case_in_keys`:
    /// with the default (`case_sensitive: false`), the three case
    /// variants collapse into one entry.
    #[test]
    fn freq_default_collapses_case_variants() {
        use talkbank_model::ChatFile;
        let command = FreqCommand::new(FreqConfig::default());
        let mut state = FreqState::default();
        let chat_file = ChatFile::new(vec![]);
        let file_ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        let u = make_utterance("CHI", &["Want", "want", "WANT"]);
        command.process_utterance(&u, &file_ctx, &mut state);

        let result = command.finalize(state);
        let chi = result
            .speakers
            .iter()
            .find(|s| s.speaker == "CHI")
            .expect("CHI speaker should be present");
        assert_eq!(chi.total_tokens, 3);
        assert_eq!(
            chi.total_types, 1,
            "case-insensitive default collapses to 1 type"
        );
    }

    /// Multi-speaker case for `+d1`: words from all speakers
    /// merge into one alphabetized deduped list.
    #[test]
    fn freq_word_list_only_dedupes_across_speakers() {
        let result = FreqResult {
            word_list_only: true,
            types_tokens_only: false,
            speakers: vec![
                FreqSpeakerResult {
                    speaker: "CHI".to_owned(),
                    entries: vec![
                        FreqEntry {
                            word: "want".to_owned(),
                            display_form: None,
                            count: 2,
                        },
                        FreqEntry {
                            word: "cookie".to_owned(),
                            display_form: None,
                            count: 1,
                        },
                    ],
                    total_types: 2,
                    total_tokens: 3,
                    ttr: 0.667,
                },
                FreqSpeakerResult {
                    speaker: "MOT".to_owned(),
                    entries: vec![
                        FreqEntry {
                            word: "want".to_owned(),
                            display_form: None,
                            count: 1,
                        },
                        FreqEntry {
                            word: "apple".to_owned(),
                            display_form: None,
                            count: 1,
                        },
                    ],
                    total_types: 2,
                    total_tokens: 2,
                    ttr: 1.0,
                },
            ],
        };
        let clan = result.render_clan();
        let lines: Vec<&str> = clan.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines, vec!["apple", "cookie", "want"]);
    }
}
