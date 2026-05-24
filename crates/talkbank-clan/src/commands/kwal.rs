//! KWAL — Keyword And Line (keyword-in-context search).
//!
//! Searches for utterances containing specified keywords and displays
//! matching lines with context. Keywords are matched as case-insensitive
//! exact words against countable words on the main tier. Wildcards (`*`)
//! are supported for partial matching (e.g., `cook*` matches `cookies`).
//!
//! # CLAN Equivalence
//!
//! | CLAN command                    | Rust equivalent                                  |
//! |---------------------------------|--------------------------------------------------|
//! | `kwal +s"want" file.cha`        | `chatter analyze kwal file.cha -k want`          |
//! | `kwal +s"want" +t*CHI file.cha` | `chatter analyze kwal file.cha -k want -s CHI`   |
//!
//! KWAL does not have a dedicated section in the CLAN manual; it is
//! described alongside other search commands.
//!
//! # Output
//!
//! Each matching utterance with:
//! - Speaker code
//! - Full utterance text
//! - File path (for multi-file searches)
//! - Match count summary per keyword
//!
//! # Differences from CLAN
//!
//! - Search operates on parsed AST word content rather than raw text lines.
//! - Word identification uses AST-based `is_countable_word()` instead of
//!   CLAN's string-prefix matching.
//! - Output supports text, JSON, and CSV formats (CLAN produces text only).

use indexmap::IndexMap;
use serde::Serialize;
use talkbank_model::{Utterance, WriteChat};

use crate::framework::word_filter::{countable_words, word_pattern_matches};
use crate::framework::{
    AnalysisCommand, AnalysisResult, CommandOutput, FileContext, NormalizedWord, OutputFormat,
    Section, TableRow,
};

/// Configuration for the KWAL command.
#[derive(Debug, Clone, Default)]
pub struct KwalConfig {
    /// Keywords to search for (case-insensitive exact match, `*` wildcards supported)
    pub keywords: Vec<crate::framework::KeywordPattern>,
    /// CLAN `+b`: match only utterances whose tier consists of
    /// exactly one countable word, and that word matches one of
    /// the configured keywords. Default `false` reverts to "match
    /// anywhere on the tier."
    pub strict_match: bool,
    /// CLAN `+k`: keyword matching is case-sensitive. Default
    /// `false` (CLAN default) lowercases both sides before
    /// comparison. When true, neither keyword nor word is folded.
    pub case_sensitive: bool,
    /// CLAN `+d` (no N): emit matching utterances as legal CHAT
    /// (drop the `---` separator and `*** File ... Keyword: X`
    /// location annotation).
    pub legal_chat: bool,
    /// CLAN `-wN` / `--context-before`: number of utterances
    /// immediately preceding each match to include as
    /// pre-context. Default `0` ⇒ no leading context.
    pub context_before: u32,
    /// CLAN `+wN` / `--context-after`: number of utterances
    /// immediately following each match to include as
    /// post-context. Default `0` ⇒ no trailing context.
    pub context_after: u32,
}

/// A single match found during KWAL processing.
#[derive(Debug, Clone, Serialize)]
pub struct KwalMatch {
    /// Speaker code.
    pub speaker: String,
    /// Full utterance text (CHAT format).
    pub utterance_text: String,
    /// Source filename.
    pub filename: String,
    /// Matched keyword that triggered this result.
    pub keyword: String,
    /// 1-based line number of this utterance in the source file.
    pub line_number: usize,
    /// CLAN `-wN` pre-context: up to `context_before` preceding
    /// utterance texts, oldest-first. Default empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pre_context: Vec<String>,
    /// CLAN `+wN` post-context: up to `context_after` following
    /// utterance texts, in stream order. Default empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub post_context: Vec<String>,
}

/// Typed output for the KWAL command.
#[derive(Debug, Clone, Serialize)]
pub struct KwalResult {
    /// All matching utterances in order encountered.
    pub matches: Vec<KwalMatch>,
    /// Per-keyword match counts.
    pub keyword_counts: IndexMap<String, u64>,
    /// CLAN `+d` (no N): emit the matching utterances as a legal
    /// CHAT fragment — drop the `---` separator and the `*** File
    /// ... Keyword: X` location annotation. Default `false`
    /// preserves the location-annotated layout.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub legal_chat: bool,
}

impl KwalResult {
    /// Convert typed KWAL matches into the shared section/table render model.
    fn to_analysis_result(&self) -> AnalysisResult {
        let mut result = AnalysisResult::new("kwal");

        if !self.matches.is_empty() {
            let rows: Vec<TableRow> = self
                .matches
                .iter()
                .map(|m| TableRow {
                    values: vec![
                        m.filename.clone(),
                        m.speaker.clone(),
                        m.utterance_text.clone(),
                    ],
                })
                .collect();

            let mut matches_section = Section::with_table(
                "Matches".to_owned(),
                vec![
                    "File".to_owned(),
                    "Speaker".to_owned(),
                    "Utterance".to_owned(),
                ],
                rows,
            );
            matches_section
                .fields
                .insert("Total matches".to_owned(), self.matches.len().to_string());
            result.add_section(matches_section);
        }

        if !self.keyword_counts.is_empty() {
            let mut fields = IndexMap::new();
            for (keyword, count) in &self.keyword_counts {
                fields.insert(format!("\"{keyword}\""), count.to_string());
            }
            result.add_section(Section::with_fields("Keyword counts".to_owned(), fields));
        }

        result
    }
}

impl CommandOutput for KwalResult {
    /// Render via the shared tabular text formatter.
    fn render_text(&self) -> String {
        self.to_analysis_result().render(OutputFormat::Text)
    }

    /// CLAN-compatible output matching legacy CLAN character-for-character.
    ///
    /// Format (from CLAN snapshot):
    /// ```text
    /// ----------------------------------------
    /// *** File "pipeout": line 10. Keyword: cookie
    /// *CHI:\tmore cookie . [+ IMP]
    /// ```
    fn render_clan(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();

        for m in &self.matches {
            // `+d` / `legal_chat`: drop the `---` separator and the
            // `*** File ... Keyword: X` location annotation; emit
            // just the matching utterance (plus any context lines)
            // as legal CHAT.
            if !self.legal_chat {
                writeln!(out, "----------------------------------------").ok();
                // CLAN uses "pipeout" as filename when reading from
                // stdin pipe, and 0-based line numbers (doesn't count
                // the @UTF8 BOM line).
                writeln!(
                    out,
                    "*** File \"pipeout\": line {}. Keyword: {} ",
                    m.line_number, m.keyword
                )
                .ok();
            }
            for line in &m.pre_context {
                writeln!(out, "{line}").ok();
            }
            writeln!(out, "{}", m.utterance_text).ok();
            for line in &m.post_context {
                writeln!(out, "{line}").ok();
            }
        }

        out
    }
}

/// Accumulated state for KWAL across all files.
#[derive(Debug, Default)]
pub struct KwalState {
    /// All matches found
    matches: Vec<KwalMatch>,
    /// Per-keyword match count
    keyword_counts: IndexMap<String, u64>,
    /// Ring buffer of recent utterance CHAT texts (capacity =
    /// `config.context_before`). Holds the most recent N
    /// non-matching utterances so a new match can snapshot them
    /// as `pre_context`. Empty when `context_before == 0`.
    recent: std::collections::VecDeque<String>,
    /// Matches still collecting post-context lines. Pair is
    /// `(match_index, remaining_after_lines)`. Each subsequent
    /// utterance appends to all entries and decrements; an entry
    /// is removed when its counter hits zero. Empty when
    /// `context_after == 0`.
    awaiting_after: Vec<(usize, u32)>,
}

/// KWAL command implementation.
///
/// For each utterance, extracts all countable words and checks whether
/// any match the configured keywords (case-insensitive). Matching
/// utterances are collected and displayed in the output.
#[derive(Debug, Clone, Default)]
pub struct KwalCommand {
    config: KwalConfig,
    /// Per-keyword string used for `word_pattern_matches`, folded
    /// once at construction time according to `config.case_sensitive`
    /// (lowercased when `false`, preserved when `true`). Hoisted out
    /// of the per-utterance hot path — `+s` keywords don't change
    /// mid-run, so this avoids `keyword.to_lowercase()` allocations
    /// on every utterance.
    keyword_match_forms: Vec<String>,
}

impl KwalCommand {
    /// Create a KWAL command with the given configuration.
    pub fn new(config: KwalConfig) -> Self {
        let keyword_match_forms = config
            .keywords
            .iter()
            .map(|k| {
                if config.case_sensitive {
                    k.as_str().to_owned()
                } else {
                    k.to_lowercase()
                }
            })
            .collect();
        Self {
            config,
            keyword_match_forms,
        }
    }
}

impl AnalysisCommand for KwalCommand {
    type Config = KwalConfig;
    type State = KwalState;
    type Output = KwalResult;

    /// Find keyword matches in one utterance and record match metadata.
    ///
    /// Context-window ordering invariant: post-context for *earlier*
    /// matches must drain BEFORE the current match is recorded (the
    /// current utterance shouldn't be its own post-context), and the
    /// pre-context ring update must happen AFTER (the current
    /// utterance shouldn't be its own pre-context either).
    fn process_utterance(
        &self,
        utterance: &Utterance,
        file_context: &FileContext<'_>,
        state: &mut Self::State,
    ) {
        if self.config.keywords.is_empty() {
            return;
        }

        // Detect match without the serialized text (cheap).
        let case_sensitive = self.config.case_sensitive;
        let words: Vec<String> = countable_words(&utterance.main.content.content)
            .map(|w| NormalizedWord::from_word_cased(w, case_sensitive).0)
            .collect();

        // `+b` doesn't early-return: even a strict-rejected utterance
        // still has to count as non-match for any open
        // `awaiting_after` and still has to feed the pre-context ring.
        let strict_rejects = self.config.strict_match && words.len() != 1;
        let mut matched = Vec::new();
        if !strict_rejects {
            for (keyword, kw_for_match) in
                self.config.keywords.iter().zip(&self.keyword_match_forms)
            {
                for word in &words {
                    if word_pattern_matches(word, kw_for_match) {
                        matched.push(keyword.clone());
                        break;
                    }
                }
            }
        }

        // Skip the allocating CHAT serialization in the common
        // zero-context, non-match path. Default-config callers (no
        // `+wN`/`-wN`) pay nothing extra per-utterance.
        let needs_text = !matched.is_empty()
            || !state.awaiting_after.is_empty()
            || self.config.context_before > 0;
        if !needs_text {
            return;
        }
        let utterance_text = utterance.main.to_chat_string();

        state.awaiting_after.retain_mut(|(match_idx, remaining)| {
            state.matches[*match_idx]
                .post_context
                .push(utterance_text.clone());
            *remaining -= 1;
            *remaining > 0
        });

        if !matched.is_empty() {
            for kw in &matched {
                *state.keyword_counts.entry(kw.to_string()).or_insert(0) += 1;
            }
            let line_number = file_context
                .line_map
                .map(|lm| lm.line_of(utterance.main.span.start))
                .unwrap_or(0);
            let pre_context: Vec<String> = state.recent.iter().cloned().collect();
            let match_idx = state.matches.len();
            state.matches.push(KwalMatch {
                speaker: utterance.main.speaker.as_str().to_owned(),
                utterance_text: utterance_text.clone(),
                filename: file_context.filename.to_owned(),
                keyword: matched[0].to_string(),
                line_number,
                pre_context,
                post_context: Vec::new(),
            });
            if self.config.context_after > 0 {
                state
                    .awaiting_after
                    .push((match_idx, self.config.context_after));
            }
        }

        let cap = self.config.context_before as usize;
        if cap > 0 {
            if state.recent.len() == cap {
                state.recent.pop_front();
            }
            state.recent.push_back(utterance_text);
        }
    }

    /// Move collected match rows and keyword counters into typed output.
    fn finalize(&self, state: Self::State) -> KwalResult {
        KwalResult {
            matches: state.matches,
            keyword_counts: state.keyword_counts,
            legal_chat: self.config.legal_chat,
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

    /// `+b` (`strict_match`) restricts matches to utterances
    /// whose entire tier consists of exactly one keyword.
    /// `["want"]` matches; `["I", "want", "cookie"]` does not,
    /// even though it contains "want".
    #[test]
    fn kwal_strict_match_only_solo_word_matches() {
        let command = KwalCommand::new(KwalConfig {
            keywords: vec![crate::framework::KeywordPattern::from("want")],
            strict_match: true,
            case_sensitive: false,
            legal_chat: false,
            context_before: 0,
            context_after: 0,
        });
        let mut state = KwalState::default();

        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let file_ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        let solo = make_utterance("CHI", &["want"]);
        let mixed = make_utterance("CHI", &["I", "want", "cookie"]);

        command.process_utterance(&solo, &file_ctx, &mut state);
        command.process_utterance(&mixed, &file_ctx, &mut state);

        let result = command.finalize(state);
        assert_eq!(result.matches.len(), 1);
        assert_eq!(result.matches[0].utterance_text, "*CHI:\twant .");
    }

    /// Default (no `+b`) still matches the keyword anywhere on the
    /// tier. Companion to the strict-match test for an obvious diff.
    #[test]
    fn kwal_default_matches_anywhere_on_tier() {
        let command = KwalCommand::new(KwalConfig {
            keywords: vec![crate::framework::KeywordPattern::from("want")],
            strict_match: false,
            case_sensitive: false,
            legal_chat: false,
            context_before: 0,
            context_after: 0,
        });
        let mut state = KwalState::default();

        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let file_ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        let solo = make_utterance("CHI", &["want"]);
        let mixed = make_utterance("CHI", &["I", "want", "cookie"]);

        command.process_utterance(&solo, &file_ctx, &mut state);
        command.process_utterance(&mixed, &file_ctx, &mut state);

        let result = command.finalize(state);
        assert_eq!(result.matches.len(), 2);
    }

    /// Matching keywords should produce one row per matching utterance.
    #[test]
    fn kwal_finds_keyword() {
        let command = KwalCommand::new(KwalConfig {
            keywords: vec![crate::framework::KeywordPattern::from("cookie")],
            ..KwalConfig::default()
        });
        let mut state = KwalState::default();

        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let file_ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        let u1 = make_utterance("CHI", &["I", "want", "cookie"]);
        let u2 = make_utterance("CHI", &["more", "milk"]);
        let u3 = make_utterance("MOT", &["have", "a", "cookie"]);

        command.process_utterance(&u1, &file_ctx, &mut state);
        command.process_utterance(&u2, &file_ctx, &mut state);
        command.process_utterance(&u3, &file_ctx, &mut state);

        let result = command.finalize(state);
        assert_eq!(result.matches.len(), 2);
        assert_eq!(result.keyword_counts["cookie"], 2);
    }

    /// Keyword matching should be case-insensitive.
    #[test]
    fn kwal_case_insensitive() {
        let command = KwalCommand::new(KwalConfig {
            keywords: vec![crate::framework::KeywordPattern::from("WANT")],
            ..KwalConfig::default()
        });
        let mut state = KwalState::default();

        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let file_ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        let u = make_utterance("CHI", &["I", "want", "cookie"]);
        command.process_utterance(&u, &file_ctx, &mut state);

        assert_eq!(state.matches.len(), 1);
    }

    /// `+k` / `--case-sensitive`: an uppercase keyword should NOT
    /// match a lowercase word, and vice versa. Pinned by contrast
    /// against `kwal_case_insensitive` above.
    #[test]
    fn kwal_case_sensitive_uppercase_keyword_misses_lowercase_word() {
        let command = KwalCommand::new(KwalConfig {
            keywords: vec![crate::framework::KeywordPattern::from("WANT")],
            case_sensitive: true,
            ..KwalConfig::default()
        });
        let mut state = KwalState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let file_ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        let u = make_utterance("CHI", &["I", "want", "cookie"]);
        command.process_utterance(&u, &file_ctx, &mut state);

        assert_eq!(state.matches.len(), 0);
    }

    /// CLAN KWAL `+d` (no N) / `--legal-chat`: render emits ONLY
    /// the matching utterance lines as legal CHAT, no `---`
    /// separators and no `*** File ... Keyword: X` decoration. The
    /// default render keeps the decoration.
    ///
    /// Per CLAN manual §7.16.7 (+d, no number): "Normally, kwal
    /// outputs the location of the tier where the match occurs.
    /// When the +d switch is turned on you can [output] in these
    /// formats: ... outputs legal CHAT format."
    #[test]
    fn kwal_legal_chat_format_drops_location_decoration() {
        use crate::framework::CommandOutput;
        let result = KwalResult {
            matches: vec![
                KwalMatch {
                    filename: "test".to_owned(),
                    speaker: "CHI".to_owned(),
                    utterance_text: "*CHI:\tI want a cookie .".to_owned(),
                    line_number: 6,
                    keyword: "want".to_owned(),
                    pre_context: Vec::new(),
                    post_context: Vec::new(),
                },
                KwalMatch {
                    filename: "test".to_owned(),
                    speaker: "MOT".to_owned(),
                    utterance_text: "*MOT:\tI Want milk .".to_owned(),
                    line_number: 7,
                    keyword: "want".to_owned(),
                    pre_context: Vec::new(),
                    post_context: Vec::new(),
                },
            ],
            keyword_counts: IndexMap::new(),
            legal_chat: true,
        };
        let clan = result.render_clan();
        // Both utterance bodies appear.
        assert!(clan.contains("*CHI:\tI want a cookie ."));
        assert!(clan.contains("*MOT:\tI Want milk ."));
        // No location decoration and no separators.
        assert!(
            !clan.contains("***"),
            "legal-chat format must not emit `*** File ...` decoration: {clan:?}"
        );
        assert!(
            !clan.contains("----------------------------------------"),
            "legal-chat format must not emit `---` separators: {clan:?}"
        );
    }

    /// Default render (legal_chat=false) keeps the location
    /// decoration — regression companion to the +d test above.
    #[test]
    fn kwal_default_render_keeps_location_decoration() {
        use crate::framework::CommandOutput;
        let result = KwalResult {
            matches: vec![KwalMatch {
                filename: "test".to_owned(),
                speaker: "CHI".to_owned(),
                utterance_text: "*CHI:\tI want a cookie .".to_owned(),
                line_number: 6,
                keyword: "want".to_owned(),
                pre_context: Vec::new(),
                post_context: Vec::new(),
            }],
            keyword_counts: IndexMap::new(),
            legal_chat: false,
        };
        let clan = result.render_clan();
        assert!(clan.contains("*** File \"pipeout\": line 6. Keyword: want"));
        assert!(clan.contains("----------------------------------------"));
        assert!(clan.contains("*CHI:\tI want a cookie ."));
    }

    /// CLAN KWAL `+wN` (`--context-after N`) emits the N
    /// utterances immediately following each match as post-context.
    /// Each `KwalMatch.post_context` is filled lazily as later
    /// utterances stream by `process_utterance`.
    #[test]
    fn kwal_context_after_captures_post_match_lines() {
        let command = KwalCommand::new(KwalConfig {
            keywords: vec![crate::framework::KeywordPattern::from("cookie")],
            context_after: 2,
            ..KwalConfig::default()
        });
        let mut state = KwalState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        // Stream: hello → milk → cookie(MATCH) → thanks → bye.
        // The cookie match should pull "thanks" and "bye" as
        // post-context (2 lines).
        command.process_utterance(&make_utterance("CHI", &["hello"]), &ctx, &mut state);
        command.process_utterance(&make_utterance("CHI", &["milk"]), &ctx, &mut state);
        command.process_utterance(&make_utterance("CHI", &["cookie"]), &ctx, &mut state);
        command.process_utterance(&make_utterance("CHI", &["thanks"]), &ctx, &mut state);
        command.process_utterance(&make_utterance("CHI", &["bye"]), &ctx, &mut state);

        assert_eq!(state.matches.len(), 1);
        let m = &state.matches[0];
        assert_eq!(m.post_context.len(), 2);
        assert!(m.post_context[0].contains("thanks"));
        assert!(m.post_context[1].contains("bye"));
    }

    /// CLAN KWAL `-wN` (`--context-before N`) emits the N
    /// utterances immediately preceding each match as pre-context.
    /// The `KwalState`'s sliding-window ring buffer holds the most
    /// recent N utterances; on a match they're snapshotted into
    /// `KwalMatch.pre_context`.
    #[test]
    fn kwal_context_before_captures_pre_match_lines() {
        let command = KwalCommand::new(KwalConfig {
            keywords: vec![crate::framework::KeywordPattern::from("cookie")],
            context_before: 2,
            ..KwalConfig::default()
        });
        let mut state = KwalState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        // Stream of 4: hello → world → milk → cookie(MATCH).
        // Pre-context window is 2, so the match should carry
        // "world" and "milk" but NOT "hello".
        command.process_utterance(&make_utterance("CHI", &["hello"]), &ctx, &mut state);
        command.process_utterance(&make_utterance("CHI", &["world"]), &ctx, &mut state);
        command.process_utterance(&make_utterance("CHI", &["milk"]), &ctx, &mut state);
        command.process_utterance(&make_utterance("CHI", &["cookie"]), &ctx, &mut state);

        assert_eq!(state.matches.len(), 1);
        let m = &state.matches[0];
        assert_eq!(m.pre_context.len(), 2);
        assert!(m.pre_context[0].contains("world"));
        assert!(m.pre_context[1].contains("milk"));
    }

    /// Default (no `+wN`/`-wN`) carries no pre- or post-context.
    /// Regression companion to the two tests above.
    #[test]
    fn kwal_default_no_context_window() {
        let command = KwalCommand::new(KwalConfig {
            keywords: vec![crate::framework::KeywordPattern::from("cookie")],
            ..KwalConfig::default()
        });
        let mut state = KwalState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        command.process_utterance(&make_utterance("CHI", &["hello"]), &ctx, &mut state);
        command.process_utterance(&make_utterance("CHI", &["cookie"]), &ctx, &mut state);
        command.process_utterance(&make_utterance("CHI", &["bye"]), &ctx, &mut state);

        assert_eq!(state.matches.len(), 1);
        assert!(state.matches[0].pre_context.is_empty());
        assert!(state.matches[0].post_context.is_empty());
    }

    /// `+k` companion: case must match exactly — uppercase keyword
    /// matches uppercase word; default lowercase keyword still
    /// matches lowercase word.
    #[test]
    fn kwal_case_sensitive_matches_when_case_aligned() {
        let command = KwalCommand::new(KwalConfig {
            keywords: vec![crate::framework::KeywordPattern::from("Want")],
            case_sensitive: true,
            ..KwalConfig::default()
        });
        let mut state = KwalState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let file_ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        let u = make_utterance("CHI", &["I", "Want", "cookie"]);
        command.process_utterance(&u, &file_ctx, &mut state);

        assert_eq!(state.matches.len(), 1);
    }

    /// Exact keyword should NOT match partial words (CLAN parity).
    #[test]
    fn kwal_exact_match_no_substring() {
        let command = KwalCommand::new(KwalConfig {
            keywords: vec![crate::framework::KeywordPattern::from("cook")],
            ..KwalConfig::default()
        });
        let mut state = KwalState::default();

        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let file_ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        // "cook" does NOT match "cookie" without wildcard
        let u = make_utterance("CHI", &["I", "want", "cookie"]);
        command.process_utterance(&u, &file_ctx, &mut state);

        assert_eq!(state.matches.len(), 0);
    }

    /// Wildcard `*` should enable partial matching (CLAN parity).
    #[test]
    fn kwal_wildcard_match() {
        let command = KwalCommand::new(KwalConfig {
            keywords: vec![crate::framework::KeywordPattern::from("cook*")],
            ..KwalConfig::default()
        });
        let mut state = KwalState::default();

        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let file_ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        // "cook*" matches "cookie" via wildcard
        let u = make_utterance("CHI", &["I", "want", "cookie"]);
        command.process_utterance(&u, &file_ctx, &mut state);

        assert_eq!(state.matches.len(), 1);
    }

    /// The `word_pattern_matches` function should handle wildcards correctly.
    #[test]
    fn keyword_matches_patterns() {
        use crate::framework::word_pattern_matches;

        // Exact match
        assert!(word_pattern_matches("cookie", "cookie"));
        assert!(!word_pattern_matches("cookies", "cookie"));

        // Prefix wildcard
        assert!(word_pattern_matches("cookie", "cook*"));
        assert!(word_pattern_matches("cookies", "cook*"));
        assert!(!word_pattern_matches("book", "cook*"));

        // Suffix wildcard
        assert!(word_pattern_matches("going", "*ing"));
        assert!(!word_pattern_matches("gong", "*ing"));

        // Contains wildcard
        assert!(word_pattern_matches("cookie", "*oki*"));
        assert!(!word_pattern_matches("cook", "*oki*"));

        // Star alone matches everything
        assert!(word_pattern_matches("anything", "*"));
    }

    /// Non-matching keywords should leave output collections empty.
    #[test]
    fn kwal_no_matches() {
        let command = KwalCommand::new(KwalConfig {
            keywords: vec![crate::framework::KeywordPattern::from("zebra")],
            ..KwalConfig::default()
        });
        let mut state = KwalState::default();

        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let file_ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        let u = make_utterance("CHI", &["I", "want", "cookie"]);
        command.process_utterance(&u, &file_ctx, &mut state);

        let result = command.finalize(state);
        assert!(result.matches.is_empty());
        assert!(result.keyword_counts.is_empty());
    }

    /// Empty keyword configuration should short-circuit to no matches.
    #[test]
    fn kwal_empty_keywords() {
        let command = KwalCommand::new(KwalConfig::default());
        let mut state = KwalState::default();

        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let file_ctx = FileContext {
            path: std::path::Path::new("test.cha"),
            chat_file: &chat_file,
            filename: "test",
            line_map: None,
        };

        let u = make_utterance("CHI", &["hello"]);
        command.process_utterance(&u, &file_ctx, &mut state);

        let result = command.finalize(state);
        assert!(result.matches.is_empty());
    }
}
