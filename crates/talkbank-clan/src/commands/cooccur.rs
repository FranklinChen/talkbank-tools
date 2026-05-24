//! COOCCUR — Word co-occurrence (bigram) counting.
//!
//! Reimplements CLAN's COOCCUR command, which counts adjacent word pairs
//! (bigrams) across utterances. For each utterance, every pair of consecutive
//! countable words is recorded as a directed bigram. Pairs are directional:
//! ("put", "the") and ("the", "put") are counted separately.
//!
//! COOCCUR is part of the FREQ family of commands and is useful for studying
//! word collocations and sequential patterns in speech.
//!
//! # CLAN Equivalence
//!
//! | CLAN command                         | Rust equivalent                                       |
//! |--------------------------------------|-------------------------------------------------------|
//! | `cooccur file.cha`                   | `chatter analyze cooccur file.cha`                    |
//! | `cooccur +t*CHI file.cha`            | `chatter analyze cooccur file.cha -s CHI`             |
//!
//! # Output
//!
//! - Table of adjacent word pairs with co-occurrence counts
//! - Default sort: by frequency descending, then alphabetically
//! - CLAN output: sorted alphabetically by pair display form
//! - Summary: unique pair count, total pair instances, total utterances
//!
//! # Differences from CLAN
//!
//! - Word identification uses AST-based `is_countable_word()` instead of
//!   CLAN's string-prefix matching (`word[0] == '&'`, etc.).
//! - Bigram extraction operates on parsed AST content rather than raw text.
//! - Output supports text, JSON, and CSV formats (CLAN produces text only).
//! - Deterministic output ordering via sorted collections.

use std::collections::BTreeMap;

use serde::Serialize;
use smallvec::SmallVec;
use talkbank_model::Utterance;

/// Inline storage for the common bigram case. Avoids a heap
/// allocation per `WordCluster`/`ClusterData` when `cluster_size`
/// is 2 (the default and overwhelmingly common case).
type ClusterInline<T> = SmallVec<[T; 2]>;

use crate::framework::word_filter::countable_words;
use crate::framework::{
    AnalysisCommand, AnalysisResult, CommandOutput, FileContext, NormalizedWord, OutputFormat,
    Section, TableRow, UtteranceCount, clan_display_form,
};

/// Configuration for the COOCCUR command.
#[derive(Debug, Clone)]
pub struct CooccurConfig {
    /// CLAN `+d`: render output without the leading frequency-
    /// count column. Each row becomes `<words…>` instead of
    /// `count <words…>`.
    pub no_frequency_counts: bool,
    /// CLAN `+nN`: cluster size (number of adjacent words counted
    /// per row). Default `2` = bigrams; `3` = trigrams; etc. Values
    /// below `2` collapse to `2` at use-site so `.windows(N)` never
    /// panics on `0` and the bigram default is the floor.
    pub cluster_size: u8,
}

impl Default for CooccurConfig {
    fn default() -> Self {
        Self {
            no_frequency_counts: false,
            cluster_size: 2,
        }
    }
}

/// A co-occurring adjacent word cluster (N-gram) with its frequency count.
/// `words.len() == displays.len() == cluster_size`.
#[derive(Debug, Clone, Serialize)]
pub struct CooccurCluster {
    /// Words in the cluster, in utterance order. Lowercased/normalized.
    pub words: Vec<String>,
    /// CLAN display forms (preserve `+` in compounds), one per word.
    pub displays: Vec<String>,
    /// Number of times this cluster occurs.
    pub count: u64,
}

/// Typed output for the COOCCUR command.
#[derive(Debug, Clone, Serialize)]
pub struct CooccurResult {
    /// Word clusters sorted by co-occurrence count descending.
    pub clusters: Vec<CooccurCluster>,
    /// Number of unique clusters observed.
    pub unique_clusters: usize,
    /// Sum of all cluster counts.
    pub total_cluster_instances: u64,
    /// Total utterances examined.
    pub total_utterances: UtteranceCount,
    /// CLAN `+d`: whether the CLAN-format renderer should omit
    /// the leading frequency-count column.
    pub no_frequency_counts: bool,
}

impl CooccurResult {
    /// Convert typed co-occurrence data into the shared section/table render model.
    fn to_analysis_result(&self) -> AnalysisResult {
        let mut result = AnalysisResult::new("cooccur");
        if self.clusters.is_empty() {
            return result;
        }

        // Header columns: "Word 1", "Word 2", ..., "Word N", "Count".
        // Length comes from the first cluster (all clusters in a run
        // share the same `cluster_size` from `CooccurConfig`).
        let n = self.clusters[0].words.len();
        let mut headers: Vec<String> = (1..=n).map(|i| format!("Word {i}")).collect();
        headers.push("Count".to_owned());

        let rows: Vec<TableRow> = self
            .clusters
            .iter()
            .map(|c| {
                let mut values = c.words.clone();
                values.push(c.count.to_string());
                TableRow { values }
            })
            .collect();

        let mut section = Section::with_table("Co-occurrences".to_owned(), headers, rows);
        section.fields.insert(
            "Unique clusters".to_owned(),
            self.unique_clusters.to_string(),
        );
        section.fields.insert(
            "Total cluster instances".to_owned(),
            self.total_cluster_instances.to_string(),
        );
        section.fields.insert(
            "Total utterances".to_owned(),
            self.total_utterances.to_string(),
        );

        result.add_section(section);
        result
    }
}

impl CommandOutput for CooccurResult {
    /// Render via the shared tabular text formatter.
    fn render_text(&self) -> String {
        self.to_analysis_result().render(OutputFormat::Text)
    }

    /// CLAN-compatible output matching legacy CLAN character-for-character.
    ///
    /// Format:
    /// ```text
    ///   1  gonna put
    ///   1  more cookie
    ///   1  the choo+choo's
    /// ```
    ///
    /// With `+nN` (cluster_size > 2) each row carries N space-
    /// separated display words instead of 2.
    fn render_clan(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();

        // CLAN sorts alphabetically by display-form sequence.
        let mut sorted: Vec<&CooccurCluster> = self.clusters.iter().collect();
        sorted.sort_by(|a, b| a.displays.cmp(&b.displays));

        for cluster in &sorted {
            let words = cluster.displays.join(" ");
            if self.no_frequency_counts {
                // CLAN `+d`: word-only row, no count column.
                writeln!(out, "{words}").ok();
            } else {
                writeln!(out, "{:>3}  {words}", cluster.count).ok();
            }
        }

        out
    }
}

/// An ordered word cluster (N-gram) used as a map key for adjacent
/// word groups. Order is preserved — ("put", "the") and ("the", "put")
/// are distinct keys. `N` matches `CooccurConfig::cluster_size`
/// (default 2 = bigrams).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct WordCluster(ClusterInline<NormalizedWord>);

impl WordCluster {
    /// Test-only helper for constructing cluster keys from string literals.
    #[cfg(test)]
    fn new(words: &[&str]) -> Self {
        WordCluster(
            words
                .iter()
                .map(|w| NormalizedWord(w.to_string()))
                .collect(),
        )
    }
}

/// Per-cluster accumulated data: count and display forms.
#[derive(Debug, Clone)]
struct ClusterData {
    count: u64,
    displays: ClusterInline<String>,
}

/// Accumulated state for COOCCUR across all files.
#[derive(Debug, Default)]
pub struct CooccurState {
    /// Co-occurrence data for each adjacent cluster (merged counts + display forms).
    clusters: BTreeMap<WordCluster, ClusterData>,
    /// Total utterances examined.
    pub total_utterances: UtteranceCount,
}

/// COOCCUR command implementation.
///
/// For each utterance, extracts countable words and counts adjacent pairs
/// (bigrams), matching CLAN's behavior.
#[derive(Debug, Clone, Default)]
pub struct CooccurCommand {
    /// User-facing configuration.
    pub config: CooccurConfig,
}

impl CooccurCommand {
    /// Construct with explicit configuration.
    pub fn new(config: CooccurConfig) -> Self {
        Self { config }
    }
}

impl AnalysisCommand for CooccurCommand {
    type Config = CooccurConfig;
    type State = CooccurState;
    type Output = CooccurResult;

    /// Count adjacent N-grams from the current utterance. `N`
    /// comes from `CooccurConfig::cluster_size` (default 2 =
    /// bigrams; clamped to a floor of 2 to keep `.windows(N)`
    /// well-defined).
    fn process_utterance(
        &self,
        utterance: &Utterance,
        _file_context: &FileContext<'_>,
        state: &mut Self::State,
    ) {
        state.total_utterances += 1;

        let words: Vec<(NormalizedWord, String)> = countable_words(&utterance.main.content.content)
            .map(|w| (NormalizedWord::from_word(w), clan_display_form(w)))
            .collect();

        let cluster_size = (self.config.cluster_size as usize).max(2);
        if words.len() < cluster_size {
            return;
        }
        for window in words.windows(cluster_size) {
            let key = WordCluster(window.iter().map(|(k, _)| k.clone()).collect());
            state
                .clusters
                .entry(key)
                .and_modify(|data| data.count += 1)
                .or_insert_with(|| ClusterData {
                    count: 1,
                    displays: window.iter().map(|(_, d)| d.clone()).collect(),
                });
        }
    }

    /// Materialize sorted output rows and aggregate totals from map state.
    fn finalize(&self, state: Self::State) -> CooccurResult {
        if state.clusters.is_empty() {
            return CooccurResult {
                clusters: Vec::new(),
                unique_clusters: 0,
                total_cluster_instances: 0,
                total_utterances: state.total_utterances,
                no_frequency_counts: self.config.no_frequency_counts,
            };
        }

        let unique_clusters = state.clusters.len();
        let total_cluster_instances: u64 = state.clusters.values().map(|d| d.count).sum();

        // Sort clusters by frequency (descending), then alphabetically.
        let mut sorted: Vec<(WordCluster, ClusterData)> = state.clusters.into_iter().collect();
        sorted.sort_by(|a, b| b.1.count.cmp(&a.1.count).then_with(|| a.0.cmp(&b.0)));

        let clusters: Vec<CooccurCluster> = sorted
            .into_iter()
            .map(|(cluster, data)| CooccurCluster {
                words: cluster
                    .0
                    .into_iter()
                    .map(|nw| nw.as_str().to_owned())
                    .collect(),
                displays: data.displays.into_iter().collect(),
                count: data.count,
            })
            .collect();

        CooccurResult {
            clusters,
            unique_clusters,
            total_cluster_instances,
            total_utterances: state.total_utterances,
            no_frequency_counts: self.config.no_frequency_counts,
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

    /// Adjacent tokens should produce one ordered bigram per sliding window.
    /// `+d` (`no_frequency_counts`) strips the leading count
    /// column from the CLAN-format output. Each row becomes
    /// `display1 display2` instead of `  N  display1 display2`.
    #[test]
    fn cooccur_no_frequency_counts_strips_count_column() {
        let command = CooccurCommand::new(CooccurConfig {
            no_frequency_counts: true,
            ..CooccurConfig::default()
        });
        let mut state = CooccurState::default();
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
        let clan = result.render_clan();
        // Each non-empty row should be exactly two whitespace-
        // separated words, no leading count.
        for line in clan.lines().filter(|l| !l.is_empty()) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(
                parts.len(),
                2,
                "row should be exactly two tokens (word, word); got {parts:?} from line {line:?}"
            );
        }
        // And: no purely-digit token at the start of any row.
        assert!(
            !clan.lines().any(|l| {
                l.split_whitespace()
                    .next()
                    .is_some_and(|t| t.chars().all(|c| c.is_ascii_digit()))
            }),
            "+d output must not have a leading count column:\n{clan}"
        );
    }

    /// Default render keeps the count column. Companion test
    /// pinning the contrast against identical input.
    #[test]
    fn cooccur_default_keeps_count_column() {
        let command = CooccurCommand::default();
        let mut state = CooccurState::default();
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
        let clan = result.render_clan();
        // Default has a leading count column — every non-empty row
        // starts with a digit-only token.
        for line in clan.lines().filter(|l| !l.is_empty()) {
            let first = line.split_whitespace().next().unwrap();
            assert!(
                first.chars().all(|c| c.is_ascii_digit()),
                "default render should start with count column; got {line:?}"
            );
        }
    }

    #[test]
    fn cooccur_adjacent_pairs() {
        let command = CooccurCommand::default();
        let mut state = CooccurState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = file_ctx(&chat_file);

        // "I want cookie" → adjacent pairs: (i, want), (want, cookie) — utterance order
        let u = make_utterance("CHI", &["I", "want", "cookie"]);
        command.process_utterance(&u, &ctx, &mut state);

        // Should have 2 adjacent pairs in utterance order
        assert_eq!(state.clusters.len(), 2);
        assert_eq!(state.clusters[&WordCluster::new(&["i", "want"])].count, 1);
        assert_eq!(
            state.clusters[&WordCluster::new(&["want", "cookie"])].count,
            1
        );
    }

    /// Pair counts should accumulate across multiple utterances.
    #[test]
    fn cooccur_accumulates_across_utterances() {
        let command = CooccurCommand::default();
        let mut state = CooccurState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = file_ctx(&chat_file);

        let u1 = make_utterance("CHI", &["I", "want"]);
        let u2 = make_utterance("CHI", &["I", "want", "more"]);
        command.process_utterance(&u1, &ctx, &mut state);
        command.process_utterance(&u2, &ctx, &mut state);

        // (i, want) should have count 2
        assert_eq!(state.clusters[&WordCluster::new(&["i", "want"])].count, 2);
        assert_eq!(state.total_utterances, 2);
    }

    /// One-token utterances should not emit any pair entries.
    #[test]
    fn cooccur_single_word_no_pairs() {
        let command = CooccurCommand::default();
        let mut state = CooccurState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = file_ctx(&chat_file);

        let u = make_utterance("CHI", &["hello"]);
        command.process_utterance(&u, &ctx, &mut state);

        assert_eq!(state.clusters.len(), 0);
    }

    /// Finalizing untouched state should return an empty result set.
    #[test]
    fn cooccur_empty_state() {
        let command = CooccurCommand::default();
        let state = CooccurState::default();
        let result = command.finalize(state);
        assert!(result.clusters.is_empty());
    }

    /// Cluster keys are directional: `(a,b)` and `(b,a)` are distinct.
    #[test]
    fn word_cluster_preserves_utterance_order() {
        let p1 = WordCluster::new(&["want", "cookie"]);
        let p2 = WordCluster::new(&["cookie", "want"]);
        assert_ne!(p1, p2);
        assert_eq!(p1.0[0].as_str(), "want");
        assert_eq!(p1.0[1].as_str(), "cookie");
    }

    /// CLAN COOCCUR `+nN` / `--cluster-size N`: emit N-grams.
    /// With `+n3` on a 4-word utterance, we get 2 trigram windows.
    #[test]
    fn cooccur_cluster_size_three_emits_trigrams() {
        let command = CooccurCommand::new(CooccurConfig {
            cluster_size: 3,
            ..CooccurConfig::default()
        });
        let mut state = CooccurState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = file_ctx(&chat_file);

        let u = make_utterance("CHI", &["I", "want", "a", "cookie"]);
        command.process_utterance(&u, &ctx, &mut state);

        let result = command.finalize(state);
        assert_eq!(result.clusters.len(), 2);
        // Each cluster carries 3 words (the cluster_size).
        for c in &result.clusters {
            assert_eq!(c.words.len(), 3);
        }
        // Trigrams in utterance order.
        let trigrams: Vec<Vec<String>> = result.clusters.iter().map(|c| c.words.clone()).collect();
        assert!(trigrams.contains(&vec!["i".to_owned(), "want".to_owned(), "a".to_owned()]));
        assert!(trigrams.contains(&vec![
            "want".to_owned(),
            "a".to_owned(),
            "cookie".to_owned()
        ]));
    }

    /// `+nN` with N greater than the utterance length produces no
    /// clusters from that utterance.
    #[test]
    fn cooccur_cluster_size_larger_than_utterance_is_skipped() {
        let command = CooccurCommand::new(CooccurConfig {
            cluster_size: 5,
            ..CooccurConfig::default()
        });
        let mut state = CooccurState::default();
        let chat_file = talkbank_model::ChatFile::new(vec![]);
        let ctx = file_ctx(&chat_file);

        let u = make_utterance("CHI", &["I", "want", "a", "cookie"]);
        command.process_utterance(&u, &ctx, &mut state);

        let result = command.finalize(state);
        assert!(result.clusters.is_empty());
    }
}
