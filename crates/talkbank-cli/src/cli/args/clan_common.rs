//! Shared clap argument groups for CLAN analysis commands.
//!
//! This module keeps the CLI-facing flag surface small and typed. In
//! particular, `--range` now parses directly into the library-owned
//! [`talkbank_clan::framework::UtteranceRange`] model so the CLI stops carrying
//! raw `start-end` strings past argument parsing.

use std::path::PathBuf;

use clap::{Args, ValueEnum};
use talkbank_clan::framework::{IdFilter, UtteranceRange, parse_id_filter, parse_utterance_range};

/// Shared filtering and output arguments for CLAN analysis commands.
#[derive(Args, Debug, Clone)]
pub struct CommonAnalysisArgs {
    /// Filter by speaker code(s) — only process these speakers (can be repeated)
    #[arg(short, long)]
    pub speaker: Vec<String>,

    /// Exclude speaker code(s) — skip these speakers (can be repeated)
    #[arg(short = 'X', long)]
    pub exclude_speaker: Vec<String>,

    /// Only process utterances within gem segments matching these labels (can be repeated)
    #[arg(short, long)]
    pub gem: Vec<String>,

    /// Skip utterances within gem segments matching these labels (can be repeated)
    #[arg(long)]
    pub exclude_gem: Vec<String>,

    /// Only process utterances containing these words — case-insensitive substring (can be repeated)
    #[arg(short = 'w', long)]
    pub include_word: Vec<String>,

    /// Skip utterances containing these words — case-insensitive substring (can be repeated)
    #[arg(short = 'W', long)]
    pub exclude_word: Vec<String>,

    /// Load `--include-word` patterns from one or more files.
    /// One pattern per line; blank lines, lines starting with
    /// `# `, and lines starting with `;%* ` are skipped. Maps
    /// CLAN's `+s@FILE` flag. Loaded patterns are appended to
    /// the (possibly empty) `--include-word` list. Repeatable.
    #[arg(long = "include-word-file", value_name = "PATH")]
    pub include_word_file: Vec<PathBuf>,

    /// Load `--exclude-word` patterns from one or more files
    /// (same file format as `--include-word-file`). Maps
    /// CLAN's `-s@FILE` flag. Repeatable.
    #[arg(long = "exclude-word-file", value_name = "PATH")]
    pub exclude_word_file: Vec<PathBuf>,

    /// Restrict to a 1-based utterance range within each file (e.g., "25-125")
    #[arg(long, value_parser = parse_utterance_range)]
    pub range: Option<UtteranceRange>,

    /// Filter by `@ID` header pattern, pipe-separated in @ID column order
    /// (`lang|corpus|speaker|age|sex|group|ses|role|education|custom`).
    ///
    /// Each field is `*` / empty (wildcard) or a literal match.
    /// Trailing wildcards may be omitted: `eng|*|CHI` ≡ `eng|*|CHI|`
    /// ≡ `eng|*|CHI|*`. A file is included only if at least one `@ID`
    /// matches; within matching files, utterances from non-matching
    /// speakers are dropped. Replaces legacy CLAN `+t@ID="…"`.
    #[arg(long, value_parser = parse_id_filter)]
    pub id_filter: Option<IdFilter>,

    /// Filter by participant role — only process utterances from
    /// speakers whose `@ID:` role field matches one of these names
    /// (can be repeated). Case-insensitive. Maps CLAN's `+t#ROLE`
    /// flag. Example: `--role Target_Child --role Mother`. Files
    /// with no `@ID:` headers are processed unchanged (no role
    /// information ⇒ no role filtering applied per-speaker).
    #[arg(long = "role")]
    pub role: Vec<String>,

    /// Output results per file instead of aggregated across all files
    #[arg(long)]
    pub per_file: bool,

    /// Include retraced words in counting (CLAN +r6 equivalent)
    #[arg(long)]
    pub include_retracings: bool,

    /// Match `+s`/`-s` / `--include-word`/`--exclude-word` patterns
    /// case-sensitively. CLAN's `+k` flag. Default is case-
    /// insensitive matching (both pattern and word are lower-cased
    /// before comparison) — matching legacy CLAN's behaviour
    /// without `+k`.
    #[arg(long = "case-sensitive")]
    pub case_sensitive: bool,

    /// Output format: clan (default — character-for-character match with legacy CLAN), text, json, or csv
    #[arg(short, long, value_enum, default_value_t = ClanOutputFormat::Clan)]
    pub format: ClanOutputFormat,
}

/// Output format for CLAN analysis commands.
///
/// `Clan` is the default — the TalkBank mandate is faithful
/// reproduction of CLAN's output, so researchers who have built
/// pipelines against CLAN output get byte-level compatibility by
/// default. `Text` is the opt-in for chatter's cleaner format.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ClanOutputFormat {
    /// CLAN-compatible output (character-for-character match with legacy CLAN)
    Clan,
    /// Human-readable text (chatter's cleaner format)
    Text,
    /// Structured JSON
    Json,
    /// CSV for spreadsheets
    Csv,
}
