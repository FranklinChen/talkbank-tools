//! Shared filtering criteria for CLAN analysis commands.
//!
//! This module provides the Rust equivalent of CUTT's speaker selection (`+t`/`-t`),
//! word search (`+s`/`-s`), gem filtering (`+g`/`-g`), and utterance range
//! (`+z`). The [`AnalysisRunner`](super::AnalysisRunner) applies filters before
//! passing utterances to commands, so each command only sees relevant data --
//! exactly like CUTT's `checktier()` + `getwholeutter()`.
//!
//! See the [CLAN manual](https://talkbank.org/0info/manuals/CLAN.html) for the
//! original filter flag semantics.
//!
//! # Filter evaluation order
//!
//! 1. Utterance range (cheapest check, `+z`)
//! 2. Speaker inclusion/exclusion (`+t`/`-t`)
//! 3. Gem segment boundaries (`+g`/`-g`)
//! 4. Word pattern matching (`+s`/`-s`)

use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use talkbank_model::{Header, SpeakerCode, Utterance};
use thiserror::Error;

use std::borrow::Cow;

use super::domain_types::WordPattern;
use super::word_filter::{countable_words_in_utterance, word_pattern_matches};

/// Failure modes when loading a CLAN-style word-list file
/// (`+s@FILE` / `-s@FILE`).
#[derive(Debug, Error)]
pub enum LoadWordListError {
    /// The file could not be opened or read.
    #[error("could not read word-list file {path}: {source}")]
    Io {
        /// Path that failed to open.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

/// Read a CLAN-style `@FILE` list: one item per non-comment line,
/// with the conventions from OSX-CLAN's `cutt.cpp::rdexclf`:
///
/// * Leading UTF-8 BOM (`U+FEFF`) on the first line is stripped.
/// * Lines beginning with `# ` (hash + space) are skipped (human
///   comments).
/// * Lines beginning with `;%* ` are skipped (CLAN's annotation
///   prefix for grep-friendly notes).
/// * Blank or whitespace-only lines are skipped.
/// * Trailing whitespace (spaces, tabs) is stripped from each line.
///
/// Source order is preserved. Casing is preserved — callers that
/// want case-folding apply it downstream.
///
/// Shared between [`load_word_list_file`] (word patterns) and
/// [`load_search_expr_file`] (COMBO boolean expressions); the file
/// format is identical, only the per-line value type differs.
fn read_clan_list_file_lines(path: &Path) -> Result<Vec<String>, LoadWordListError> {
    let content = std::fs::read_to_string(path).map_err(|source| LoadWordListError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let body = content.strip_prefix('\u{feff}').unwrap_or(&content);
    Ok(body
        .lines()
        .map(|l| l.trim_end_matches([' ', '\t']))
        .filter(|l| !l.is_empty() && !l.starts_with("# ") && !l.starts_with(";%* "))
        .map(|l| l.to_owned())
        .collect())
}

/// Load a CLAN-style word-list file (`+s@FILE` / `-s@FILE` for
/// every command except COMBO and SCRIPT).
///
/// Each surviving line becomes one [`WordPattern`]. See
/// [`read_clan_list_file_lines`] for the file-format conventions.
pub fn load_word_list_file(path: &Path) -> Result<Vec<WordPattern>, LoadWordListError> {
    Ok(read_clan_list_file_lines(path)?
        .into_iter()
        .map(WordPattern::from)
        .collect())
}

/// Load a CLAN-style COMBO search-expression file (`+s@FILE` /
/// `-s@FILE` for COMBO only).
///
/// Each surviving line is returned verbatim — the caller parses
/// it into a `SearchExpr` (the boolean-expression AST defined in
/// `commands::combo`) before feeding the analysis runner. See
/// [`read_clan_list_file_lines`] for the file-format conventions.
pub fn load_search_expr_file(path: &Path) -> Result<Vec<String>, LoadWordListError> {
    read_clan_list_file_lines(path)
}

/// Shared filtering criteria applied before utterances reach a command.
///
/// Replaces CUTT's global filtering flags. The runner evaluates these
/// against each utterance and only passes matching utterances to the
/// command's `process_utterance`.
#[derive(Debug, Clone, Default)]
pub struct FilterConfig {
    /// Include/exclude speakers (CUTT: +t/-t @ID)
    pub speakers: SpeakerFilter,
    /// Include/exclude dependent tiers (CUTT: +t/-t %tier)
    pub tiers: TierFilter,
    /// Word/morpheme search patterns (CUTT: +s/-s)
    pub words: WordFilter,
    /// Gem segment filtering (CUTT: +g/-g)
    pub gems: GemFilter,
    /// Restrict to a 1-based utterance range within each file (CUTT: +z)
    /// inclusive — e.g., `25-125` processes utterances 25–125
    pub utterance_range: Option<UtteranceRange>,
    /// Filter by `@ID` header pattern (CUTT: `+t@ID="lang|*|CHI|*"`).
    ///
    /// When `Some`, the analysis runner uses it twice:
    ///  - **file prefilter:** skip any file whose `@ID` headers all fail
    ///    the match;
    ///  - **utterance filter:** drop utterances whose speaker's `@ID` row
    ///    fails the match.
    ///
    /// `FilterConfig::matches` does not consult this field directly; the
    /// runner is responsible for both passes because it owns the parsed
    /// `@ID` headers.
    pub id_filter: Option<super::IdFilter>,
    /// Filter by participant role (CLAN: `+t#ROLE`).
    ///
    /// `FilterConfig::matches` does not consult this field directly;
    /// the runner reads the speaker's `ParticipantRole` from the
    /// `@ID:` header map and drops utterances whose role is not in
    /// the include list. When `include` is empty, role filtering is
    /// inactive.
    pub roles: RoleFilter,
}

/// Inclusive 1-based utterance range within a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UtteranceRange {
    start: usize,
    end: usize,
}

/// Error returned when parsing an utterance range.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ParseUtteranceRangeError {
    /// The input was not in `start-end` form.
    #[error("invalid range format '{input}' — expected 'start-end' (e.g., '25-125')")]
    InvalidFormat {
        /// Original input string.
        input: String,
    },
    /// The bounds were syntactically valid but semantically invalid.
    #[error("invalid range '{input}' — start must be >= 1 and end >= start")]
    InvalidBounds {
        /// Original input string.
        input: String,
    },
}

impl UtteranceRange {
    /// Create a validated utterance range.
    pub fn new(start: usize, end: usize) -> Result<Self, ParseUtteranceRangeError> {
        if start == 0 || end < start {
            return Err(ParseUtteranceRangeError::InvalidBounds {
                input: format!("{start}-{end}"),
            });
        }

        Ok(Self { start, end })
    }

    /// Inclusive lower bound.
    pub const fn start(self) -> usize {
        self.start
    }

    /// Inclusive upper bound.
    pub const fn end(self) -> usize {
        self.end
    }

    /// Check whether a 1-based utterance index falls within the range.
    pub const fn contains(self, utterance_index: usize) -> bool {
        utterance_index >= self.start && utterance_index <= self.end
    }
}

impl fmt::Display for UtteranceRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.start, self.end)
    }
}

impl FromStr for UtteranceRange {
    type Err = ParseUtteranceRangeError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = input.splitn(2, '-').collect();
        if parts.len() != 2 {
            return Err(ParseUtteranceRangeError::InvalidFormat {
                input: input.to_owned(),
            });
        }

        let start =
            parts[0]
                .parse::<usize>()
                .map_err(|_| ParseUtteranceRangeError::InvalidFormat {
                    input: input.to_owned(),
                })?;
        let end =
            parts[1]
                .parse::<usize>()
                .map_err(|_| ParseUtteranceRangeError::InvalidFormat {
                    input: input.to_owned(),
                })?;

        Self::new(start, end).map_err(|_| ParseUtteranceRangeError::InvalidBounds {
            input: input.to_owned(),
        })
    }
}

/// Parse a clap-friendly utterance range argument.
pub fn parse_utterance_range(input: &str) -> Result<UtteranceRange, String> {
    input
        .parse::<UtteranceRange>()
        .map_err(|error| error.to_string())
}

/// Speaker inclusion/exclusion filter.
///
/// When `include` is non-empty, only those speakers are processed.
/// When `exclude` is non-empty, those speakers are skipped.
/// When both are empty, all speakers pass (default behavior).
#[derive(Debug, Clone, Default)]
pub struct SpeakerFilter {
    /// Speakers to include (empty = include all)
    pub include: Vec<SpeakerCode>,
    /// Speakers to exclude
    pub exclude: Vec<SpeakerCode>,
}

/// Participant-role inclusion filter (CLAN: `+t#ROLE`).
///
/// When `include` is non-empty, only utterances from speakers whose
/// `@ID:` role field matches one of the listed roles (case-
/// insensitive) are processed. When `include` is empty, role
/// filtering is inactive (every speaker passes).
///
/// Files with no `@ID:` headers cannot have role filtering applied
/// — the runner processes them unchanged, matching CLAN's behaviour
/// (no `@ID` data ⇒ no `+t#ROLE` match information).
#[derive(Debug, Clone, Default)]
pub struct RoleFilter {
    /// Role names to include. Stored as raw user-supplied strings;
    /// the matcher in `runner.rs` compares case-insensitively against
    /// the speaker's `ParticipantRole` from `@ID:`.
    pub include: Vec<String>,
}

/// Dependent tier inclusion/exclusion filter.
///
/// Controls which dependent tiers are visible to commands.
/// By default all tiers are visible.
#[derive(Debug, Clone, Default)]
pub struct TierFilter {
    /// Tier kinds to include (empty = include all)
    pub include: Vec<super::TierKind>,
    /// Tier kinds to exclude
    pub exclude: Vec<super::TierKind>,
}

/// Word/morpheme pattern filter (CUTT: +s/-s).
///
/// When `include` is non-empty, only utterances containing at least
/// one matching word are processed. When `exclude` is non-empty,
/// utterances containing any matching word are skipped.
///
/// `case_sensitive` (CLAN `+k`) defaults to `false` — patterns and
/// words are lower-cased before matching. When `true`, both sides
/// keep their original casing and an exact-case match is required.
#[derive(Debug, Clone, Default)]
pub struct WordFilter {
    /// Word patterns to include (empty = include all)
    pub include: Vec<super::WordPattern>,
    /// Word patterns to exclude
    pub exclude: Vec<super::WordPattern>,
    /// Case-sensitive matching (CLAN `+k`). Default `false` =
    /// case-insensitive (lower-case both sides).
    pub case_sensitive: bool,
}

/// Gem segment filter (CUTT: +g/-g).
///
/// When `include` is non-empty, only utterances within matching
/// @BG/@EG segments are processed. Gem labels use case-insensitive matching.
#[derive(Debug, Clone, Default)]
pub struct GemFilter {
    /// Gem labels to include (empty = include all)
    pub include: Vec<super::GemLabel>,
    /// Gem labels to exclude
    pub exclude: Vec<super::GemLabel>,
}

impl FilterConfig {
    /// Check whether an utterance passes all filter criteria.
    ///
    /// # Preconditions
    /// - `utterance` is a valid parsed utterance from a ChatFile
    /// - `active_gems` is the set of currently open @BG labels
    /// - `utterance_index` is the 1-based index of this utterance within its file
    ///
    /// # Returns
    /// `true` if the utterance should be passed to the command
    pub fn matches(
        &self,
        utterance: &Utterance,
        active_gems: &[String],
        utterance_index: usize,
    ) -> bool {
        // Check utterance range first (cheapest check)
        if let Some(range) = self.utterance_range
            && !range.contains(utterance_index)
        {
            return false;
        }

        self.speakers.matches(&utterance.main.speaker)
            && self.gems.matches(active_gems)
            && self.words.matches(utterance)
    }
}

impl SpeakerFilter {
    /// Check whether a speaker passes this filter.
    ///
    /// - If `include` is non-empty, speaker must be in the include list.
    /// - If `exclude` is non-empty, speaker must NOT be in the exclude list.
    /// - If both are empty, all speakers pass.
    pub fn matches(&self, speaker: &SpeakerCode) -> bool {
        if !self.include.is_empty() && !self.include.contains(speaker) {
            return false;
        }
        if self.exclude.contains(speaker) {
            return false;
        }
        true
    }
}

impl WordFilter {
    /// Check whether an utterance's words pass this filter.
    ///
    /// - If `include` is non-empty, the utterance must contain at least one
    ///   countable word matching any include pattern (case-insensitive exact match,
    ///   `*` wildcards supported).
    /// - If `exclude` is non-empty, the utterance must NOT contain any countable
    ///   word matching any exclude pattern.
    /// - If both are empty, all utterances pass.
    pub fn matches(&self, utterance: &Utterance) -> bool {
        if self.include.is_empty() && self.exclude.is_empty() {
            return true;
        }

        // Normalize both sides identically. Case-insensitive (default,
        // CLAN's behaviour without `+k`) lower-cases both pattern and
        // word text; case-sensitive keeps the original casing on both
        // sides. On the case-sensitive path we skip the per-pattern
        // `to_lowercase` allocation by borrowing the originals.
        let include_folded: Vec<Cow<'_, str>> = self
            .include
            .iter()
            .map(|p| fold_case(p, self.case_sensitive))
            .collect();
        let exclude_folded: Vec<Cow<'_, str>> = self
            .exclude
            .iter()
            .map(|p| fold_case(p, self.case_sensitive))
            .collect();

        // The cleaned word text needs to outlive `word_texts`, so we
        // collect the owned `cleaned_text().to_string()` first, then
        // borrow from that vector.
        let words_owned: Vec<String> = countable_words_in_utterance(utterance)
            .map(|w| w.cleaned_text().to_string())
            .collect();
        let word_texts: Vec<Cow<'_, str>> = words_owned
            .iter()
            .map(|s| fold_case(s.as_str(), self.case_sensitive))
            .collect();

        // If include patterns specified, at least one word must match
        if !include_folded.is_empty() {
            let has_match = word_texts.iter().any(|text| {
                include_folded
                    .iter()
                    .any(|pattern| word_pattern_matches(text, pattern))
            });
            if !has_match {
                return false;
            }
        }

        // If exclude patterns specified, no word may match
        if !exclude_folded.is_empty() {
            let has_excluded = word_texts.iter().any(|text| {
                exclude_folded
                    .iter()
                    .any(|pattern| word_pattern_matches(text, pattern))
            });
            if has_excluded {
                return false;
            }
        }

        true
    }
}

/// Apply CLAN's `+k` case-sensitivity rule to a pattern or word.
///
/// `case_sensitive = true` returns the input borrowed (matching CLAN's
/// behaviour with `+k`). `case_sensitive = false` returns a lower-
/// cased copy.
fn fold_case(s: &str, case_sensitive: bool) -> Cow<'_, str> {
    if case_sensitive {
        Cow::Borrowed(s)
    } else {
        Cow::Owned(s.to_lowercase())
    }
}

impl GemFilter {
    /// Check whether the current gem context passes this filter.
    ///
    /// - If `include` is non-empty, at least one active gem must match.
    /// - If `exclude` is non-empty, no active gem must match.
    /// - If both are empty, all contexts pass.
    pub fn matches(&self, active_gems: &[String]) -> bool {
        if !self.include.is_empty() {
            let has_match = active_gems.iter().any(|gem| {
                self.include
                    .iter()
                    .any(|pattern| gem.eq_ignore_ascii_case(pattern))
            });
            if !has_match {
                return false;
            }
        }
        if !self.exclude.is_empty() {
            let has_excluded = active_gems.iter().any(|gem| {
                self.exclude
                    .iter()
                    .any(|pattern| gem.eq_ignore_ascii_case(pattern))
            });
            if has_excluded {
                return false;
            }
        }
        true
    }
}

/// Track @BG/@EG gem boundaries across utterances.
///
/// Call `update` for each utterance's preceding headers to maintain
/// the set of currently active gem labels.
pub fn update_active_gems(headers: &[Header], active_gems: &mut Vec<String>) {
    for header in headers {
        match header {
            Header::BeginGem { label: Some(label) } => {
                active_gems.push(label.as_str().to_owned());
            }
            Header::EndGem { label: Some(label) } => {
                // Remove the most recent matching @BG
                if let Some(pos) = active_gems
                    .iter()
                    .rposition(|g| g.eq_ignore_ascii_case(label.as_str()))
                {
                    active_gems.remove(pos);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::domain_types::{GemLabel, WordPattern};
    use super::*;

    /// Empty speaker filters should allow any speaker code.
    #[test]
    fn speaker_filter_empty_matches_all() {
        let filter = SpeakerFilter::default();
        let speaker: SpeakerCode = "CHI".into();
        assert!(filter.matches(&speaker));
    }

    /// An include list should restrict matches to listed speakers only.
    #[test]
    fn speaker_filter_include_restricts() {
        let filter = SpeakerFilter {
            include: vec!["CHI".into(), "MOT".into()],
            exclude: vec![],
        };
        assert!(filter.matches(&"CHI".into()));
        assert!(filter.matches(&"MOT".into()));
        assert!(!filter.matches(&"FAT".into()));
    }

    /// Excluded speakers should fail even when include is empty.
    #[test]
    fn speaker_filter_exclude_removes() {
        let filter = SpeakerFilter {
            include: vec![],
            exclude: vec!["INV".into()],
        };
        assert!(filter.matches(&"CHI".into()));
        assert!(!filter.matches(&"INV".into()));
    }

    /// Empty gem filters should match whether or not a gem is active.
    #[test]
    fn gem_filter_empty_matches_all() {
        let filter = GemFilter::default();
        assert!(filter.matches(&[]));
        assert!(filter.matches(&["Story".to_owned()]));
    }

    /// Gem include filters require at least one active matching label.
    #[test]
    fn gem_filter_include_requires_match() {
        let filter = GemFilter {
            include: vec![GemLabel::from("Story")],
            exclude: vec![],
        };
        assert!(!filter.matches(&[]));
        assert!(filter.matches(&["Story".to_owned()]));
        assert!(filter.matches(&["story".to_owned()])); // case-insensitive
        assert!(!filter.matches(&["Narrative".to_owned()]));
    }

    /// Gem exclude filters should reject matching labels case-insensitively.
    #[test]
    fn gem_filter_exclude_blocks_match() {
        let filter = GemFilter {
            include: vec![],
            exclude: vec![GemLabel::from("Warmup")],
        };
        assert!(filter.matches(&[]));
        assert!(filter.matches(&["Story".to_owned()]));
        assert!(!filter.matches(&["Warmup".to_owned()]));
        assert!(!filter.matches(&["warmup".to_owned()]));
    }

    /// Empty word filters should not gate utterances.
    #[test]
    fn word_filter_empty_matches_all() {
        let filter = WordFilter::default();
        let utterance = make_test_utterance(&["hello", "world"]);
        assert!(filter.matches(&utterance));
    }

    /// Include patterns should require at least one lexical match.
    #[test]
    fn word_filter_include_requires_match() {
        let filter = WordFilter {
            include: vec![WordPattern::from("hello")],
            exclude: vec![],
            ..WordFilter::default()
        };
        let matching = make_test_utterance(&["hello", "world"]);
        assert!(filter.matches(&matching));

        let non_matching = make_test_utterance(&["goodbye", "world"]);
        assert!(!filter.matches(&non_matching));
    }

    /// Word include matching is case-insensitive.
    #[test]
    fn word_filter_include_case_insensitive() {
        let filter = WordFilter {
            include: vec![WordPattern::from("Hello")],
            exclude: vec![],
            ..WordFilter::default()
        };
        let utterance = make_test_utterance(&["hello", "world"]);
        assert!(filter.matches(&utterance));
    }

    /// CLAN's `+k` flag flips matching to case-sensitive — a
    /// lower-case word should NOT match a capitalised pattern.
    #[test]
    fn word_filter_case_sensitive_pattern_does_not_match_other_case() {
        let filter = WordFilter {
            include: vec![WordPattern::from("Hello")],
            exclude: vec![],
            case_sensitive: true,
        };
        let lower = make_test_utterance(&["hello", "world"]);
        assert!(!filter.matches(&lower));

        let mixed = make_test_utterance(&["Hello", "world"]);
        assert!(filter.matches(&mixed));
    }

    /// Word includes use exact match semantics (CLAN parity).
    #[test]
    fn word_filter_include_exact() {
        let filter = WordFilter {
            include: vec![WordPattern::from("hello")],
            exclude: vec![],
            ..WordFilter::default()
        };
        let utterance = make_test_utterance(&["hello", "world"]);
        assert!(filter.matches(&utterance));

        // Substring should NOT match
        let filter_sub = WordFilter {
            include: vec![WordPattern::from("ell")],
            exclude: vec![],
            ..WordFilter::default()
        };
        assert!(!filter_sub.matches(&utterance));
    }

    /// Wildcard `*` enables partial matching in word filters.
    #[test]
    fn word_filter_include_wildcard() {
        let filter = WordFilter {
            include: vec![WordPattern::from("hel*")],
            exclude: vec![],
            ..WordFilter::default()
        };
        let utterance = make_test_utterance(&["hello", "world"]);
        assert!(filter.matches(&utterance));
    }

    /// Exclude patterns block utterances containing any matching word.
    #[test]
    fn word_filter_exclude_blocks() {
        let filter = WordFilter {
            include: vec![],
            exclude: vec![WordPattern::from("world")],
            ..WordFilter::default()
        };
        let blocked = make_test_utterance(&["hello", "world"]);
        assert!(!filter.matches(&blocked));

        let allowed = make_test_utterance(&["hello", "there"]);
        assert!(filter.matches(&allowed));
    }

    /// Exclude matches win when both include and exclude patterns match.
    #[test]
    fn word_filter_include_and_exclude() {
        let filter = WordFilter {
            include: vec![WordPattern::from("hello")],
            exclude: vec![WordPattern::from("world")],
            ..WordFilter::default()
        };
        // Has include match but also has exclude match → blocked
        let blocked = make_test_utterance(&["hello", "world"]);
        assert!(!filter.matches(&blocked));

        // Has include match, no exclude match → pass
        let allowed = make_test_utterance(&["hello", "there"]);
        assert!(filter.matches(&allowed));
    }

    /// The utterance range filter uses inclusive 1-based bounds.
    #[test]
    fn utterance_range_filters() {
        let config = FilterConfig {
            utterance_range: Some(UtteranceRange::new(2, 4).expect("valid test range")),
            ..FilterConfig::default()
        };
        let utterance = make_test_utterance(&["hello"]);
        let gems: Vec<String> = vec![];

        assert!(!config.matches(&utterance, &gems, 1));
        assert!(config.matches(&utterance, &gems, 2));
        assert!(config.matches(&utterance, &gems, 3));
        assert!(config.matches(&utterance, &gems, 4));
        assert!(!config.matches(&utterance, &gems, 5));
    }

    /// Utterance ranges should parse from CLAN-style `start-end` strings.
    #[test]
    fn utterance_range_parses() {
        let range = "25-125"
            .parse::<UtteranceRange>()
            .expect("range should parse");
        assert_eq!(range.start(), 25);
        assert_eq!(range.end(), 125);
        assert_eq!(range.to_string(), "25-125");
    }

    /// Invalid utterance ranges should report whether syntax or bounds failed.
    #[test]
    fn utterance_range_rejects_invalid_input() {
        assert!(matches!(
            "oops".parse::<UtteranceRange>(),
            Err(ParseUtteranceRangeError::InvalidFormat { .. })
        ));
        assert!(matches!(
            "0-5".parse::<UtteranceRange>(),
            Err(ParseUtteranceRangeError::InvalidBounds { .. })
        ));
        assert!(matches!(
            "9-3".parse::<UtteranceRange>(),
            Err(ParseUtteranceRangeError::InvalidBounds { .. })
        ));
    }

    /// Reads one pattern per non-comment line; skips blanks,
    /// `# `-comments, and `;%* `-annotation lines (CLAN's
    /// `cutt.cpp::rdexclf` conventions).
    #[test]
    fn load_word_list_file_strips_comments_and_blanks() {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::with_suffix(".cut").expect("tmp file");
        file.write_all(
            "\u{feff}# leading comment\n\
             want\n\
             \n\
             cookie   \n\
             ;%* annotation marker — skipped\n\
             milk\t\n\
             # another comment\n\
             juice"
                .as_bytes(),
        )
        .expect("write tmp word-list");
        let patterns = super::load_word_list_file(file.path()).expect("load");
        let texts: Vec<&str> = patterns.iter().map(|p| p.as_str()).collect();
        assert_eq!(texts, vec!["want", "cookie", "milk", "juice"]);
    }

    /// Missing files surface as `LoadWordListError::Io` with the
    /// original path attached — the CLI maps this to a CLAN-style
    /// stderr message.
    #[test]
    fn load_word_list_file_missing_path_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bogus = dir.path().join("never.cut");
        let err = super::load_word_list_file(&bogus).expect_err("should fail");
        match err {
            super::LoadWordListError::Io { path, .. } => assert_eq!(path, bogus),
        }
    }

    /// COMBO's `+s@FILE` shares the file format but returns raw
    /// expression lines (parsed downstream by `SearchExpr::parse`),
    /// not `WordPattern`s.
    #[test]
    fn load_search_expr_file_keeps_expression_lines_intact() {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::with_suffix(".cut").expect("tmp file");
        file.write_all(
            "\u{feff}# search expressions\n\
             want+cookie\n\
             \n\
             milk,juice\n\
             ;%* annotated boolean — skipped\n\
             hello"
                .as_bytes(),
        )
        .expect("write tmp search-list");
        let exprs = super::load_search_expr_file(file.path()).expect("load");
        assert_eq!(exprs, vec!["want+cookie", "milk,juice", "hello"]);
    }

    /// Build a minimal Utterance with the given words for filter testing.
    fn make_test_utterance(words: &[&str]) -> talkbank_model::Utterance {
        use talkbank_model::Span;
        use talkbank_model::{MainTier, Terminator, UtteranceContent, Word};

        let content: Vec<UtteranceContent> = words
            .iter()
            .map(|w| UtteranceContent::Word(Box::new(Word::simple(*w))))
            .collect();
        let main = MainTier::new("CHI", content, Terminator::Period { span: Span::DUMMY });
        talkbank_model::Utterance::new(main)
    }
}
